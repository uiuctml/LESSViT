"""Evaluate fine-tuned checkpoints (segmentation or multilabel) on the native
128x128 downstream benchmark tiles by quad-split aggregation: split each tile into
its four non-overlapping 64x64 quadrants (matching the 64x64 crop these checkpoints
were pretrained/fine-tuned at), run the model on all four, and aggregate back into
one prediction per tile -- averaged logits for multilabel, a stitched full 128x128
logit map for segmentation -- before computing the exact same metrics
finetune.py's own Trainer.evaluate() would report. Built for the
arm_a_less/arm_b_full/arm_c_additive attention-ablation checkpoints specifically
(see launch_finetune_arm_*.sh), but works for any LESSViT checkpoint fine-tuned at
crop_size 64.

Usage:
    python3 GeospatialFM/finetune/eval_quad_split.py \\
        --data_dir /datasets/disk3/geospatial \\
        --output_csv results/quad_split_eval.csv
"""
import argparse
import csv
import glob
import json
import logging
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EvalPrediction

from GeospatialFM.data_process.collate_func import modal_specific_collate_fn
from GeospatialFM.data_process.srf import build_srf_matrix, summarize_srf
from GeospatialFM.data_process.transforms import quad_crop_offsets, segmentation_quad_eval_transform_one_sample, classification_quad_eval_transform_one_sample
from GeospatialFM.datasets.enmap import ENMAP_DATASET, get_enmap_downstream_dataset, get_enmap_downstream_metadata
from GeospatialFM.datasets.enmap.enmap import SELECTED_CHANNEL_IDX_A, SELECTED_CHANNEL_IDX_B
from GeospatialFM.datasets.enmap.sensors import SENSOR_CONFIGS, compute_target_stats, load_sensor_config
from GeospatialFM.finetune.utils import get_metric
from GeospatialFM.models.downstream_models import LESSWithProjection, LESSWithUPerNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ARMS = ["arm_a_less", "arm_b_full", "arm_c_additive"]
DEFAULT_DATASETS = ["enmap_cdl", "enmap_corine", "enmap_eurocrops", "enmap_bdforet", "enmap_bnetd"]

# Same mapping launch_finetune_ablation_base.sh's FINETUNE_DATASETS array encodes.
DATASET_TASK_TYPE = {
    "enmap_cdl": "segmentation",
    "enmap_corine": "multilabel",
    "enmap_eurocrops": "segmentation",
    "enmap_bdforet": "segmentation",
    "enmap_bnetd": "segmentation",
}

# Same convention as eval_cross_sensor.py: gen_task ranges over native EnMAP channel-subset
# settings and SRF-resampled sensor configs, one shared axis (see finetune/args.py's --gen_task).
# Checkpoint *selection* (find_checkpoint) always stays anchored to native "id" val performance
# (that's what LR-sweep training/validation always used, per launch_finetune_arm_*.sh) --
# gen_tasks here only controls which spectral configs the *selected* checkpoint gets reported on.
NATIVE_GEN_TASKS = ["id", "ood_a", "ood_full", "ood_complement"]
SENSOR_GEN_TASKS = list(SENSOR_CONFIGS.keys())

# "desis"/"eo1h": real alternative-sensor test sets (desis_cdl/eo1_cdl) that share
# enmap_cdl's CDL label scheme -- evaluated the same way as ood_*/prisma_like/sentinel2_like:
# no separate fine-tune, just a different eval-time input source for the SAME enmap_cdl
# checkpoint. Internally uses gen_task=None (full native bands, no EnMAP-index-based
# subsetting -- that subsetting is specific to EnMAP's own 202-band grid and doesn't apply
# to a genuinely different sensor's own bands).
REAL_SENSOR_GEN_TASKS = {"desis": "desis_cdl", "eo1h": "eo1_cdl"}
REAL_SENSOR_ANCHOR_DATASET = "enmap_cdl"  # only dataset whose label scheme (CDL) matches

ALL_GEN_TASKS = NATIVE_GEN_TASKS + SENSOR_GEN_TASKS + list(REAL_SENSOR_GEN_TASKS.keys())

CSV_FIELDS = ["arm", "dataset", "gen_task", "n_bands", "task_type", "metric_name", "value", "n_samples", "reason"]


def n_bands_for_gen_task(gen_task):
    if gen_task == "id":
        return len(SELECTED_CHANNEL_IDX_B)
    elif gen_task == "ood_a":
        return len(SELECTED_CHANNEL_IDX_A)
    elif gen_task == "ood_full":
        return 202
    elif gen_task == "ood_complement":
        return 202 - len(SELECTED_CHANNEL_IDX_B)
    elif gen_task in SENSOR_CONFIGS:
        return len(SENSOR_CONFIGS[gen_task]["lam_tgt"])
    elif gen_task in REAL_SENSOR_GEN_TASKS:
        return len(ENMAP_DATASET[REAL_SENSOR_GEN_TASKS[gen_task]].metadata["s2c"]["channel_wv"])
    raise ValueError(f"Unknown gen_task: {gen_task}")


def order_gen_tasks(requested):
    """Run the enmap_identity/ood_full sanity-check pair first, if both are requested."""
    ordered = [g for g in ("enmap_identity", "ood_full") if g in requested]
    ordered += [g for g in requested if g not in ordered]
    return ordered


def build_sensor_transform_args(dataset_name, gen_task, sensor_stats_cache, pretrain_data_dir, n_patches, cache_dir):
    """Return (optical_mean, optical_std, optical_srf_matrix) for a sensor-config gen_task,
    caching the (sensor-level, dataset-independent) SRF matrix/stats across datasets. Ported
    from eval_cross_sensor.py verbatim -- same sensor-config machinery, dataset-generic."""
    if gen_task not in sensor_stats_cache:
        metadata = get_enmap_downstream_metadata(dataset_name)
        sensor = load_sensor_config(gen_task)
        W = build_srf_matrix(metadata["s2c"]["channel_wv"], sensor["lam_tgt"], sensor["fwhm_tgt"])
        summarize_srf(sensor["lam_tgt"], W, sensor["name"])
        mean, std = compute_target_stats(
            sensor["name"], W, metadata["s2c"]["mean"], lam_tgt=sensor["lam_tgt"],
            data_root=pretrain_data_dir, cache_dir=cache_dir, n_patches=n_patches,
        )
        sensor_stats_cache[gen_task] = (mean.tolist(), std.tolist(), W)
    return sensor_stats_cache[gen_task]


def _highest_step_checkpoint(run_dir):
    candidates = glob.glob(os.path.join(run_dir, "checkpoint-*"))
    if not candidates:
        return None

    def _step(path):
        try:
            return int(path.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(candidates, key=_step)


def find_checkpoint(results_dir, dataset_name, arm, task_type):
    """Locate the best-on-val fine-tuned checkpoint for (arm, dataset), across the
    --learning_rate sweep launch_finetune_arm_*.sh runs (Appendix D.1's grid, plain
    loop -- not Optuna): globs every results/models/<dataset>/<arm>_<dataset>_lr*/
    run dir, reads each one's val_results.json (written by
    finetune.py's trainer.save_metrics("val", ...)) for its eval_<metric_name>
    value, and picks the run with the highest value -- matching the paper's "picks
    the best on the validation set" protocol. Returns the winning run's
    highest-step checkpoint dir, or None if no run has both a val_results.json and
    at least one checkpoint.
    """
    _, metric_name = get_metric(task_type, num_classes=1, ignore_index=None)  # metric_name doesn't depend on num_classes/ignore_index
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "models", dataset_name, f"{arm}_{dataset_name}_lr*")))

    best_run, best_value = None, None
    for run_dir in run_dirs:
        val_results_path = os.path.join(run_dir, "val_results.json")
        if not os.path.exists(val_results_path):
            continue
        with open(val_results_path) as f:
            val_results = json.load(f)
        value = val_results.get(f"eval_{metric_name}")
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_run, best_value = run_dir, value

    if best_run is None:
        return None
    logger.info("Selected %s for arm=%s dataset=%s (eval_%s=%.4f among %d swept LR runs)",
                os.path.basename(best_run), arm, dataset_name, metric_name, best_value, len(run_dirs))
    return _highest_step_checkpoint(best_run)


def build_quad_eval_dataset(args, dataset_name, gen_task, sensor_stats_cache):
    task_type = DATASET_TASK_TYPE[dataset_name]

    if gen_task in REAL_SENSOR_GEN_TASKS:
        # desis/eo1h: swap in a genuinely different sensor's own real test set (and its own
        # normalization stats), still scored against enmap_cdl's checkpoint -- only meaningful
        # when dataset_name is the CDL-labeled anchor dataset (see REAL_SENSOR_GEN_TASKS comment).
        if dataset_name != REAL_SENSOR_ANCHOR_DATASET:
            raise ValueError(
                f"gen_task={gen_task!r} only applies to dataset_name={REAL_SENSOR_ANCHOR_DATASET!r} "
                f"(desis_cdl/eo1_cdl share its CDL label scheme; evaluating any other dataset's "
                f"classifier against them would compare incompatible label spaces)"
            )
        real_dataset_name = REAL_SENSOR_GEN_TASKS[gen_task]
        real_metadata = get_enmap_downstream_metadata(real_dataset_name)
        eval_transform = partial(
            segmentation_quad_eval_transform_one_sample, crop_size=args.crop_size,
            optical_mean=real_metadata["s2c"]["mean"], optical_std=real_metadata["s2c"]["std"],
        )
        ds_args = argparse.Namespace(data_dir=args.data_dir, dataset_name=real_dataset_name)
        # gen_task=None here (not the outer gen_task) -- desis_cdl/eo1_cdl's own gen_task-driven
        # subsetting assumes EnMAP's 202-band grid and doesn't apply to these real sensors' own
        # bands; None skips it entirely and returns the full native band set, unmodified.
        dataset = get_enmap_downstream_dataset(ds_args, eval_transform, eval_transform, gen_task=None)
        return dataset["test"], real_metadata, task_type

    metadata = get_enmap_downstream_metadata(dataset_name)
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    optical_srf_matrix = None

    if gen_task in SENSOR_CONFIGS:
        optical_mean, optical_std, optical_srf_matrix = build_sensor_transform_args(
            dataset_name, gen_task, sensor_stats_cache, args.pretrain_data_dir, args.n_patches, args.srf_cache_dir,
        )

    if task_type == "segmentation":
        eval_transform = partial(
            segmentation_quad_eval_transform_one_sample, crop_size=args.crop_size,
            optical_mean=optical_mean, optical_std=optical_std, optical_srf_matrix=optical_srf_matrix,
        )
    elif task_type == "multilabel":
        eval_transform = partial(
            classification_quad_eval_transform_one_sample, crop_size=args.crop_size,
            optical_mean=optical_mean, optical_std=optical_std, optical_srf_matrix=optical_srf_matrix,
        )
    else:
        raise NotImplementedError(f"eval_quad_split.py only supports segmentation/multilabel, got {task_type!r}")

    # train_transform is never exercised here (only dataset['test'] is used).
    ds_args = argparse.Namespace(data_dir=args.data_dir, dataset_name=dataset_name)
    dataset = get_enmap_downstream_dataset(ds_args, eval_transform, eval_transform, gen_task)
    return dataset["test"], metadata, task_type


@torch.no_grad()
def evaluate(model, dataset, task_type, batch_size, device, wv_min, wv_max, n_samples=None):
    """Reshape each batch's [B, C, 4, H, W] quad-split input to [B*4, C, H, W], run
    one forward pass, reshape back, and aggregate: mean logits over the quad axis
    for multilabel, stitched full 128x128 logit map (using the exact same
    quad_crop_offsets QuadCropAll split with) for segmentation. Returns
    (all_logits, all_labels) as numpy arrays, ready for an EvalPrediction.
    """
    if n_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(n_samples, len(dataset))))
    # normalize_wv=True unconditionally -- every ablation arm (less/full/additive) uses
    # SSRoPE, which assumes optical_channel_wv already sits in [0,1] (pos_chan_embed.py's
    # `coords_c = 2*optical_channel_wv - 1`); wv_min/wv_max must be this dataset's own
    # full native channel_wv range (not modal_specific_collate_fn's stale hardcoded
    # defaults, which don't match this dataset family), matching scripts/train.py's
    # normalization exactly or the model receives raw ~400-2450nm values instead of the
    # [0,1] range it was pretrained on -- silently wrong, not a crash.
    collate_fn = partial(modal_specific_collate_fn, modal="optical", normalize_wv=True, wv_min=wv_min, wv_max=wv_max)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        optical = batch["optical"].to(device)  # [B, C, 4, H, W]
        B, C, Q, H, W = optical.shape
        assert Q == 4, f"expected 4 quadrants, got {Q}"
        optical_flat = optical.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        optical_channel_wv = batch["optical_channel_wv"].to(device)
        labels = batch["labels"]

        outputs = model(
            optical=optical_flat, optical_channel_wv=optical_channel_wv,
            spatial_resolution=batch["spatial_resolution"],
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        logits = logits.reshape(B, Q, *logits.shape[1:])  # [B, 4, num_labels, (H, W)]

        if task_type == "multilabel":
            agg = logits.mean(dim=1)  # [B, num_labels]
        else:  # segmentation
            # crop_size taken from the actual per-quadrant output, not a config
            # attribute -- self-consistent with whatever ConvHead actually returned,
            # regardless of what --crop_size the CLI/config claims.
            num_labels, quad_size = logits.shape[2], logits.shape[-1]
            native_size = 2 * quad_size
            agg = torch.zeros(B, num_labels, native_size, native_size, device=logits.device, dtype=logits.dtype)
            for q, (top, left) in enumerate(quad_crop_offsets(quad_size)):
                agg[:, :, top:top + quad_size, left:left + quad_size] = logits[:, q]

        all_logits.append(agg.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def main(args):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    def emit(arm, dataset_name, gen_task, n_bands, task_type, metric_name, value, n_samples, reason=""):
        row = {
            "arm": arm, "dataset": dataset_name, "gen_task": gen_task, "n_bands": n_bands,
            "task_type": task_type, "metric_name": metric_name, "value": value,
            "n_samples": n_samples, "reason": reason,
        }
        writer.writerow(row)
        csv_file.flush()
        logger.info("%s", row)

    device = torch.device(args.device)
    sensor_stats_cache = {}
    gen_tasks = order_gen_tasks(args.gen_tasks)

    for arm in args.arms:
        for dataset_name in args.datasets:
            task_type = DATASET_TASK_TYPE[dataset_name]
            n_bands_by_task = {g: n_bands_for_gen_task(g) for g in gen_tasks}

            # Checkpoint selection is always anchored to native "id" val performance --
            # find_checkpoint reads each LR-swept run's val_results.json, which is always from
            # training/validating on native id (see launch_finetune_arm_*.sh), regardless of
            # which gen_tasks get evaluated below. Best-hparam selection and cross-spectral
            # reporting are deliberately decoupled.
            ckpt_dir = find_checkpoint(args.results_dir, dataset_name, arm, task_type)
            if ckpt_dir is None:
                logger.warning("No fine-tuned checkpoint found for arm=%s dataset=%s -- skipping all gen_tasks", arm, dataset_name)
                for gen_task in gen_tasks:
                    emit(arm, dataset_name, gen_task, n_bands_by_task[gen_task], task_type, "n/a", None, 0, "checkpoint_not_found")
                continue

            metadata = get_enmap_downstream_metadata(dataset_name)
            num_classes, ignore_index = metadata["num_classes"], metadata["ignore_index"]
            channel_wv = np.array(metadata["s2c"]["channel_wv"])
            wv_min, wv_max = channel_wv.min(), channel_wv.max()
            model_cls = LESSWithUPerNet if task_type == "segmentation" else LESSWithProjection

            model = None
            sanity_values = {}
            try:
                logger.info("Loading checkpoint for arm=%s dataset=%s from %s", arm, dataset_name, ckpt_dir)
                model = model_cls.from_pretrained(ckpt_dir, num_labels=num_classes).to(device)
            except Exception as e:
                logger.exception("Failed to load checkpoint for arm=%s dataset=%s from %s", arm, dataset_name, ckpt_dir)
                for gen_task in gen_tasks:
                    emit(arm, dataset_name, gen_task, n_bands_by_task[gen_task], task_type, "n/a", None, 0, f"checkpoint load error: {e}")
                continue

            try:
                for gen_task in gen_tasks:
                    n_bands = n_bands_by_task[gen_task]
                    try:
                        dataset, _, _ = build_quad_eval_dataset(args, dataset_name, gen_task, sensor_stats_cache)
                        n_samples = len(dataset) if args.n_samples is None else min(args.n_samples, len(dataset))
                        all_logits, all_labels = evaluate(
                            model, dataset, task_type, args.batch_size, device, wv_min, wv_max, n_samples=args.n_samples,
                        )
                        compute_metrics_fn, metric_name = get_metric(task_type, num_classes, ignore_index)
                        metrics = compute_metrics_fn(EvalPrediction(predictions=all_logits, label_ids=all_labels))
                        value = metrics[metric_name]
                    except Exception as e:
                        logger.exception("Runtime failure evaluating arm=%s dataset=%s gen_task=%s (n_bands=%d)",
                                          arm, dataset_name, gen_task, n_bands)
                        emit(arm, dataset_name, gen_task, n_bands, task_type, "n/a", None, 0, f"runtime error: {e}")
                        continue

                    emit(arm, dataset_name, gen_task, n_bands, task_type, metric_name, value, n_samples)

                    if gen_task in ("enmap_identity", "ood_full"):
                        sanity_values[gen_task] = value
                        if len(sanity_values) == 2:
                            diff = abs(sanity_values["enmap_identity"] - sanity_values["ood_full"])
                            if diff > args.sanity_tol:
                                csv_file.close()
                                raise RuntimeError(
                                    f"Sanity check failed for arm={arm} dataset={dataset_name}: "
                                    f"enmap_identity {metric_name}={sanity_values['enmap_identity']:.4f} vs "
                                    f"ood_full {metric_name}={sanity_values['ood_full']:.4f} (diff={diff:.4f} > "
                                    f"--sanity_tol={args.sanity_tol}). Stop and debug the SRF/normalization "
                                    "wiring before trusting other-sensor numbers."
                                )
                            logger.info(
                                "Sanity check passed for arm=%s dataset=%s: enmap_identity=%.4f, ood_full=%.4f (diff=%.4f)",
                                arm, dataset_name, sanity_values["enmap_identity"], sanity_values["ood_full"], diff,
                            )
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    csv_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Quad-split downstream eval for the 64x64-pretrained attention-ablation checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench downstream data root")
    parser.add_argument("--pretrain_data_dir", type=str, default=None,
                        help="EnMAP pretraining data root (SpectralEarthDataset), used to compute SRF std stats "
                             "when a sensor-config gen_task isn't already cached")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results dir (fine-tuned checkpoints live under <results_dir>/models/<dataset>/<arm>_<dataset>_lr*/)")
    parser.add_argument("--srf_cache_dir", type=str, default=os.path.join("results", "srf_stats"))
    parser.add_argument("--output_csv", type=str, default=os.path.join("results", "quad_split_eval.csv"))
    parser.add_argument("--arms", type=str, nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS, choices=list(DATASET_TASK_TYPE.keys()))
    parser.add_argument("--gen_tasks", type=str, nargs="+", default=ALL_GEN_TASKS, choices=ALL_GEN_TASKS,
                        help="Spectral configs to report the best-on-native-id checkpoint's performance on -- "
                             "checkpoint *selection* always uses native id val performance regardless of this list. "
                             "'desis'/'eo1h' evaluate enmap_cdl's checkpoint against desis_cdl's/eo1_cdl's own real "
                             "test data (same CDL labels, genuinely different sensor) -- only valid with --datasets "
                             "enmap_cdl, since that's the only checkpoint whose label space matches.")
    parser.add_argument("--sanity_tol", type=float, default=0.1, help="Max allowed |enmap_identity - ood_full| metric difference")
    parser.add_argument("--n_patches", type=int, default=2000, help="Patches to sample when computing SRF std stats")
    parser.add_argument("--crop_size", type=int, default=64, help="Quadrant size (native tile size is always 2x this)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=None, help="Limit eval to this many test samples (for a quick dry run); default: full test set")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
