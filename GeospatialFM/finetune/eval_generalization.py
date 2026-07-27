"""Evaluate the checkpoint selected by launch_finetune_sweep.sh's LR sweep (trained on native
--gen_task id, best-on-val LR picked from results/models/<dataset>/<model>_id_lr*/val_results.json)
across the full generalization grid -- no retraining, one checkpoint, repeated eval passes:

  * native EnMAP channel-subset settings: id, ood_a, ood_full, ood_complement (4)
  * SRF-resampled sensor configs: prisma_like, sentinel2_like (2)
  * real alternative-sensor test sets sharing enmap_cdl's CDL label scheme: desis, eo1h (2)

desis/eo1h only apply when --dataset_name is enmap_cdl (desis_cdl/eo1_cdl are the only real
sensor datasets with CDL labels). EO1CDLDataset/DESISCDLDataset each build their OWN raw-CDL
-code -> ordinal-index mapping from their own (different) `classes` list, e.g. EO1's ordinal
index 6 is CDL code 41 while enmap_cdl's ordinal index 6 is CDL code 45 -- left alone, the
labels handed to the metric wouldn't correspond to the same class as the logit channel the
checkpoint's head produces at that index. build_eval_dataset() below overrides each real
dataset's `.ordinal_map` with one built from enmap_cdl's OWN `classes` ordering (see
build_ordinal_map()), so ordinal index i always means "the i-th class in the anchor dataset's
`classes` list", matching what the checkpoint was actually fine-tuned to predict. Any CDL code
desis/eo1h's test masks contain that enmap_cdl's `classes` doesn't recognize becomes
ignore_index -- the checkpoint has no output for a class it was never trained on, so those
pixels can't be scored either way.

Writes a tidy CSV: model, dataset, gen_task, n_bands, task_type, metric_name, value, lr, reason.

Usage:
    python3 GeospatialFM/finetune/eval_generalization.py \\
        --data_dir /datasets/geospatial --pretrain_data_dir /datasets/geospatial/enmap/enmap \\
        --results_dir ./results --model_name lessvit --dataset_name enmap_cdl \\
        --task_type segmentation --output_csv results/generalization_eval.csv
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
from torchgeo.datasets.cdl import CDL
from transformers import EvalPrediction

from GeospatialFM.data_process.collate_func import modal_specific_collate_fn
from GeospatialFM.data_process.srf import build_srf_matrix, summarize_srf
from GeospatialFM.data_process.transforms import get_enmap_transform
from GeospatialFM.datasets.enmap import ENMAP_DATASET, get_enmap_downstream_dataset, get_enmap_downstream_metadata
from GeospatialFM.datasets.enmap.enmap import SELECTED_CHANNEL_IDX_A, SELECTED_CHANNEL_IDX_B
from GeospatialFM.datasets.enmap.sensors import SENSOR_CONFIGS, compute_target_stats, load_sensor_config
from GeospatialFM.finetune.utils import get_metric
from GeospatialFM.models.downstream_models import LESSWithProjection, LESSWithUPerNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NATIVE_GEN_TASKS = ["id", "ood_a", "ood_full", "ood_complement"]
# enmap_identity (an SRF-identity passthrough, used elsewhere purely as an id/ood_full sanity
# check) is deliberately excluded from the default sensor list -- prisma_like/sentinel2_like are
# the two genuine SRF-resampled sensors this pipeline reports on.
SENSOR_GEN_TASKS = [name for name in SENSOR_CONFIGS.keys() if name != "enmap_identity"]
REAL_SENSOR_GEN_TASKS = {"desis": "desis_cdl", "eo1h": "eo1_cdl"}
REAL_SENSOR_ANCHOR_DATASET = "enmap_cdl"

DEFAULT_GEN_TASKS = NATIVE_GEN_TASKS + SENSOR_GEN_TASKS + list(REAL_SENSOR_GEN_TASKS.keys())
ALL_GEN_TASKS = NATIVE_GEN_TASKS + list(SENSOR_CONFIGS.keys()) + list(REAL_SENSOR_GEN_TASKS.keys())

TASK_TYPE_MODEL_CLS = {
    "segmentation": LESSWithUPerNet,
    "classification": LESSWithProjection,
    "multilabel": LESSWithProjection,
}

CSV_FIELDS = ["model", "dataset", "gen_task", "n_bands", "task_type", "metric_name", "value", "lr", "reason"]


def n_bands_for_gen_task(gen_task, dataset_name):
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


def build_ordinal_map(classes, ignore_index):
    """Raw-CDL-code -> ordinal-index tensor, matching EnMAPCDLDataset/EO1CDLDataset/
    DESISCDLDataset's own __init__ convention (code 0 sorts last, everything else keeps
    `classes`' order; any code not in `classes` maps to ignore_index) -- but computed from a
    plain `classes` list with no `self` mutation, so it can build a DIFFERENT dataset's own
    label scheme onto a real-sensor dataset instance (see build_eval_dataset)."""
    ordinal_map = torch.zeros(max(CDL.cmap.keys()) + 1, dtype=torch.long) + ignore_index
    ordered = [c for c in classes if c != 0] + [0]
    for v, k in enumerate(ordered):
        ordinal_map[k] = v
    return ordinal_map


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


def find_checkpoint(results_dir, dataset_name, model_name, gen_task_train, task_type):
    """Pick the best-on-val LR among launch_finetune_sweep.sh's LR-swept runs
    (results/models/<dataset>/<model>_<gen_task_train>_lr*/val_results.json, written by
    finetune.py's trainer.save_metrics("val", ...)), matching the paper's search-on-val
    protocol. Returns (checkpoint_dir, lr, val_score), or (None, None, None) if no run has
    both a val_results.json and at least one checkpoint."""
    _, metric_name = get_metric(task_type, num_classes=1, ignore_index=None)  # metric_name doesn't depend on num_classes/ignore_index
    prefix = f"{model_name}_{gen_task_train}_lr"
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "models", dataset_name, f"{prefix}*")))

    best_run, best_lr, best_value = None, None, None
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
            best_lr = os.path.basename(run_dir)[len(prefix):]

    if best_run is None:
        return None, None, None
    ckpt_dir = _highest_step_checkpoint(best_run)
    if ckpt_dir is None:
        return None, None, None
    logger.info("Selected %s for model=%s dataset=%s (eval_%s=%.4f among %d swept LR runs)",
                os.path.basename(best_run), model_name, dataset_name, metric_name, best_value, len(run_dirs))
    return ckpt_dir, best_lr, best_value


def build_eval_dataset(args, dataset_name, gen_task, task_type, sensor_stats_cache):
    if gen_task in REAL_SENSOR_GEN_TASKS:
        # desis/eo1h: swap in a genuinely different sensor's own real test set (and its own
        # normalization stats), still scored against dataset_name's checkpoint -- only
        # meaningful when dataset_name is the CDL-labeled anchor dataset (see module docstring).
        if dataset_name != REAL_SENSOR_ANCHOR_DATASET:
            raise ValueError(
                f"gen_task={gen_task!r} only applies to dataset_name={REAL_SENSOR_ANCHOR_DATASET!r} "
                f"(desis_cdl/eo1_cdl share its CDL label scheme; evaluating any other dataset's "
                f"classifier against them would compare incompatible label spaces)"
            )
        real_dataset_name = REAL_SENSOR_GEN_TASKS[gen_task]
        real_metadata = get_enmap_downstream_metadata(real_dataset_name)
        _, eval_transform = get_enmap_transform(
            task_type, crop_size=args.crop_size, scale=None, random_rotation=False,
            optical_mean=real_metadata["s2c"]["mean"], optical_std=real_metadata["s2c"]["std"],
            dataset_name=real_dataset_name,
        )
        ds_args = argparse.Namespace(data_dir=args.data_dir, dataset_name=real_dataset_name)
        # gen_task=None here (not the outer gen_task) -- desis_cdl/eo1_cdl's own gen_task-driven
        # subsetting assumes EnMAP's 202-band grid and doesn't apply to these real sensors' own
        # bands; None skips it entirely and returns the full native band set, unmodified.
        dataset = get_enmap_downstream_dataset(ds_args, eval_transform, eval_transform, gen_task=None)

        # Realign labels to the ANCHOR (training) dataset's own ordinal scheme -- see module
        # docstring. Overriding .ordinal_map after construction (but before any __getitem__
        # call) is safe: __init__ has already run and __getitem__ reads self.ordinal_map fresh
        # every call, so this is the only place that needs to change.
        anchor_cls = ENMAP_DATASET[dataset_name]
        dataset["test"].ordinal_map = build_ordinal_map(anchor_cls.classes, anchor_cls.metadata["ignore_index"])
        return dataset["test"]

    metadata = get_enmap_downstream_metadata(dataset_name)
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    optical_srf_matrix = None

    if gen_task in SENSOR_CONFIGS:
        optical_mean, optical_std, optical_srf_matrix = build_sensor_transform_args(
            dataset_name, gen_task, sensor_stats_cache, args.pretrain_data_dir, args.n_patches, args.srf_cache_dir,
        )

    # train_transform is never exercised here (only dataset['test'] is used).
    _, eval_transform = get_enmap_transform(
        task_type, crop_size=args.crop_size, scale=None, random_rotation=False,
        optical_mean=optical_mean, optical_std=optical_std, dataset_name=dataset_name,
        optical_srf_matrix=optical_srf_matrix,
    )
    ds_args = argparse.Namespace(data_dir=args.data_dir, dataset_name=dataset_name)
    dataset = get_enmap_downstream_dataset(ds_args, eval_transform, eval_transform, gen_task)
    return dataset["test"]


@torch.no_grad()
def evaluate(model, dataset, task_type, num_classes, ignore_index, batch_size, device, wv_min, wv_max, use_rope):
    # normalize_wv must reflect whether *this specific checkpoint* uses RoPE
    # (RopePositionChannelEmbedding assumes optical_channel_wv already sits in [0,1] --
    # pos_chan_embed.py's `coords_c = 2*optical_channel_wv - 1`); wv_min/wv_max are the
    # TRAINING dataset's own full native channel_wv range, matching finetune.py's convention
    # (correct regardless of which gen_task/real sensor is actually flowing through here).
    collate_fn = partial(modal_specific_collate_fn, modal="optical", normalize_wv=use_rope, wv_min=wv_min, wv_max=wv_max)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        optical = batch["optical"].to(device)
        optical_channel_wv = batch["optical_channel_wv"].to(device)
        labels = batch["labels"]
        outputs = model(
            optical=optical, optical_channel_wv=optical_channel_wv,
            spatial_resolution=batch["spatial_resolution"],
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    compute_metrics_fn, metric_name = get_metric(task_type, num_classes, ignore_index)
    metrics = compute_metrics_fn(EvalPrediction(predictions=all_logits, label_ids=all_labels))
    return metrics[metric_name], metric_name


def main(args):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    def emit(gen_task, n_bands, metric_name, value, lr, reason=""):
        row = {
            "model": args.model_name, "dataset": args.dataset_name, "gen_task": gen_task,
            "n_bands": n_bands, "task_type": args.task_type, "metric_name": metric_name,
            "value": value, "lr": lr, "reason": reason,
        }
        writer.writerow(row)
        csv_file.flush()
        logger.info("%s", row)

    device = torch.device(args.device)
    sensor_stats_cache = {}
    gen_tasks = order_gen_tasks(args.gen_tasks)
    n_bands_by_task = {g: n_bands_for_gen_task(g, args.dataset_name) for g in gen_tasks}

    ckpt_dir, lr, val_score = find_checkpoint(args.results_dir, args.dataset_name, args.model_name, args.gen_task_train, args.task_type)
    if ckpt_dir is None:
        logger.warning("No fine-tuned checkpoint found for model=%s dataset=%s -- skipping all gen_tasks", args.model_name, args.dataset_name)
        for gen_task in gen_tasks:
            emit(gen_task, n_bands_by_task[gen_task], "n/a", None, None, "checkpoint_not_found")
        csv_file.close()
        return

    metadata = get_enmap_downstream_metadata(args.dataset_name)
    num_classes, ignore_index = metadata["num_classes"], metadata["ignore_index"]
    channel_wv = np.array(metadata["s2c"]["channel_wv"])
    wv_min, wv_max = channel_wv.min(), channel_wv.max()
    model_cls = TASK_TYPE_MODEL_CLS[args.task_type]

    try:
        logger.info("Loading checkpoint for model=%s dataset=%s from %s (lr=%s)", args.model_name, args.dataset_name, ckpt_dir, lr)
        model = model_cls.from_pretrained(ckpt_dir, num_labels=num_classes).to(device)
    except Exception as e:
        logger.exception("Failed to load checkpoint from %s", ckpt_dir)
        for gen_task in gen_tasks:
            emit(gen_task, n_bands_by_task[gen_task], "n/a", None, lr, f"checkpoint load error: {e}")
        csv_file.close()
        return

    use_rope = getattr(model.config, "use_rope_embed", False)
    sanity_values = {}
    try:
        for gen_task in gen_tasks:
            n_bands = n_bands_by_task[gen_task]
            try:
                dataset = build_eval_dataset(args, args.dataset_name, gen_task, args.task_type, sensor_stats_cache)
                value, metric_name = evaluate(
                    model, dataset, args.task_type, num_classes, ignore_index, args.batch_size, device,
                    wv_min, wv_max, use_rope,
                )
            except Exception as e:
                logger.exception("Runtime failure evaluating model=%s dataset=%s gen_task=%s (n_bands=%d)",
                                  args.model_name, args.dataset_name, gen_task, n_bands)
                emit(gen_task, n_bands, "n/a", None, lr, f"runtime error: {e}")
                continue

            emit(gen_task, n_bands, metric_name, value, lr)

            if gen_task in ("enmap_identity", "ood_full"):
                sanity_values[gen_task] = value
                if len(sanity_values) == 2:
                    diff = abs(sanity_values["enmap_identity"] - sanity_values["ood_full"])
                    if diff > args.sanity_tol:
                        raise RuntimeError(
                            f"Sanity check failed for model={args.model_name} dataset={args.dataset_name}: "
                            f"enmap_identity {metric_name}={sanity_values['enmap_identity']:.4f} vs "
                            f"ood_full {metric_name}={sanity_values['ood_full']:.4f} (diff={diff:.4f} > "
                            f"--sanity_tol={args.sanity_tol}). Stop and debug the SRF/normalization "
                            "wiring before trusting other-sensor numbers."
                        )
                    logger.info(
                        "Sanity check passed for model=%s dataset=%s: enmap_identity=%.4f, ood_full=%.4f (diff=%.4f)",
                        args.model_name, args.dataset_name, sanity_values["enmap_identity"], sanity_values["ood_full"], diff,
                    )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        csv_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Full generalization eval (native/SRF/real-sensor) for a launch_finetune_sweep.sh LR-swept checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench downstream data root")
    parser.add_argument("--pretrain_data_dir", type=str, default=None,
                        help="EnMAP pretraining data root (SpectralEarthDataset), used to compute SRF std stats "
                             "when a sensor-config gen_task isn't already cached")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results dir (checkpoints live under <results_dir>/models/<dataset>/<model>_<gen_task_train>_lr*/)")
    parser.add_argument("--srf_cache_dir", type=str, default=os.path.join("results", "srf_stats"))
    parser.add_argument("--output_csv", type=str, default=os.path.join("results", "generalization_eval.csv"))
    parser.add_argument("--model_name", type=str, required=True, choices=["lessvit", "specvit", "dinov3", "dofa", "spatsigma", "channelvit", "hyperfree"])
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(ENMAP_DATASET.keys()))
    parser.add_argument("--task_type", type=str, required=True, choices=list(TASK_TYPE_MODEL_CLS.keys()))
    parser.add_argument("--gen_task_train", type=str, default="id", help="--gen_task the LR sweep was trained/val-selected on (launch_finetune_sweep.sh's GEN_TASK)")
    parser.add_argument("--gen_tasks", type=str, nargs="+", default=DEFAULT_GEN_TASKS, choices=ALL_GEN_TASKS,
                        help="Spectral configs / real sensors to report the selected checkpoint's performance on. "
                             "'desis'/'eo1h' only valid with --dataset_name enmap_cdl.")
    parser.add_argument("--sanity_tol", type=float, default=0.1, help="Max allowed |enmap_identity - ood_full| metric difference (only checked if both are in --gen_tasks)")
    parser.add_argument("--n_patches", type=int, default=2000, help="Patches to sample when computing SRF std stats")
    parser.add_argument("--crop_size", type=int, default=128, help="Eval crop size -- must match what the checkpoint was fine-tuned at (see launch_finetune_sweep.sh's CROP_SIZE)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
