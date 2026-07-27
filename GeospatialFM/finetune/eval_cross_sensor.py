"""Evaluate fine-tuned checkpoints across {model} x {dataset} x {gen_task} settings, where
gen_task ranges over the native EnMAP channel-subset settings (id/ood_a/ood_full/ood_complement)
and the SRF-resampled sensor configs (enmap_identity/prisma_like/sentinel2_like) -- the same
single axis, see GeospatialFM/finetune/args.py's --gen_task. Writes a tidy CSV:
model, sensor, task, n_bands, metric_name, value.

Each (model, dataset) pair uses exactly one fine-tuned checkpoint (trained on the native
C120_VNIR+ subset, i.e. --gen_task id), evaluated repeatedly across gen_task settings -- no
retraining, matching the existing cross-spectral eval convention (see README.md).

Usage:
    python3 GeospatialFM/finetune/eval_cross_sensor.py \\
        --data_dir /datasets/disk3/geospatial \\
        --pretrain_data_dir /datasets/disk2/geospatial/enmap/enmap \\
        --output_csv results/cross_sensor_eval.csv
"""
import argparse
import csv
import glob
import logging
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from GeospatialFM.data_process.collate_func import modal_specific_collate_fn
from GeospatialFM.data_process.srf import build_srf_matrix, summarize_srf
from GeospatialFM.data_process.transforms import get_enmap_transform
from GeospatialFM.datasets.enmap import ENMAP_DATASET, get_enmap_downstream_dataset, get_enmap_downstream_metadata
from GeospatialFM.datasets.enmap.enmap import SELECTED_CHANNEL_IDX_A, SELECTED_CHANNEL_IDX_B
from GeospatialFM.datasets.enmap.sensors import SENSOR_CONFIGS, compute_target_stats, load_sensor_config
from GeospatialFM.models.downstream_models import LESSWithUPerNet, LESSWithUPerNetConfig
from GeospatialFM.models.registry import ENCODER_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NATIVE_GEN_TASKS = ["id", "ood_a", "ood_full", "ood_complement"]
SENSOR_GEN_TASKS = list(SENSOR_CONFIGS.keys())
ALL_GEN_TASKS = NATIVE_GEN_TASKS + SENSOR_GEN_TASKS

# Baselines whose channel count is baked into the model at construction time and can't accept an
# arbitrary band set (see wrappers/channelvit_wrapper.py, wrappers/dinov3_wrapper.py). spatsigma
# defaults to 202 but is zero-padded up to it regardless of actual input width
# (downstream_models.py), so it's compatible with any n_bands <= 202 and isn't listed here.
FIXED_CHANNEL_MODELS = {"channelvit": 202, "dinov3": 155}

DEFAULT_MODELS = list(ENCODER_CONFIGS.keys())
DEFAULT_DATASETS = ["enmap_cdl", "enmap_corine", "enmap_eurocrops", "enmap_bdforet", "enmap_bnetd"]

CSV_FIELDS = ["model", "sensor", "task", "n_bands", "metric_name", "value", "reason"]


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
    raise ValueError(f"Unknown gen_task: {gen_task}")


def order_gen_tasks(requested):
    """Run the enmap_identity/ood_full sanity-check pair first, if both are requested."""
    ordered = [g for g in ("enmap_identity", "ood_full") if g in requested]
    ordered += [g for g in requested if g not in ordered]
    return ordered


def build_sensor_transform_args(dataset_name, gen_task, sensor_stats_cache, pretrain_data_dir, n_patches, cache_dir):
    """Return (optical_mean, optical_std, optical_srf_matrix) for a sensor-config gen_task,
    caching the (sensor-level, dataset-independent) SRF matrix/stats across datasets."""
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


def find_checkpoint(results_dir, dataset_name, model_name):
    """Locate the single fine-tuned checkpoint for (model, dataset), trained with --gen_task id
    (see launch_finetune.sh's RUN_NAME=${MODEL_NAME}_${GEN_TASK} convention) -- the highest-step
    checkpoint-N under results/models/<dataset_name>/<model_name>_id/."""
    print(results_dir)
    print(dataset_name)
    print(model_name)
    pattern = os.path.join(results_dir, "models", dataset_name, f"{model_name}_id_lr3e-4", "checkpoint-*")
    print(pattern)
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _step(path):
        try:
            return int(path.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(candidates, key=_step)


def build_eval_dataset(args, dataset_name, gen_task, sensor_stats_cache):
    metadata = get_enmap_downstream_metadata(dataset_name)
    crop_size = 128
    optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
    optical_srf_matrix = None

    if gen_task in SENSOR_CONFIGS:
        optical_mean, optical_std, optical_srf_matrix = build_sensor_transform_args(
            dataset_name, gen_task, sensor_stats_cache, args.pretrain_data_dir, args.n_patches, args.srf_cache_dir,
        )

    # train_transform is never exercised here (only dataset['test'] is used), so it's fine to
    # pass the eval_transform in its place -- avoids constructing/threading a second transform
    # that would never run.
    _, eval_transform = get_enmap_transform(
        "segmentation", crop_size=crop_size, scale=None, random_rotation=False,
        optical_mean=optical_mean, optical_std=optical_std, dataset_name=dataset_name,
        optical_srf_matrix=optical_srf_matrix,
    )

    ds_args = argparse.Namespace(data_dir=args.data_dir, dataset_name=dataset_name)
    dataset = get_enmap_downstream_dataset(ds_args, eval_transform, eval_transform, gen_task)
    return dataset["test"], metadata


@torch.no_grad()
def evaluate(model, dataset, num_classes, ignore_index, batch_size, device, normalize_wv=False, wv_min=None, wv_max=None):
    # normalize_wv must reflect whether *this specific model* uses RoPE
    # (RopePositionChannelEmbedding assumes optical_channel_wv already sits in [0,1] --
    # pos_chan_embed.py's `coords_c = 2*optical_channel_wv - 1`), not a blanket setting --
    # this script sweeps multiple baseline architectures, most of which don't use RoPE at
    # all and may expect raw wavelengths or nothing. See main()'s
    # getattr(model.config, "use_rope_embed", False) check.
    collate_fn = partial(modal_specific_collate_fn, modal="optical", normalize_wv=normalize_wv, wv_min=wv_min, wv_max=wv_max)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    metric = MulticlassJaccardIndex(num_classes=num_classes, average="micro", ignore_index=ignore_index).to(device)

    model.eval()
    for batch in loader:
        optical = batch["optical"].to(device)
        optical_channel_wv = batch["optical_channel_wv"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            optical=optical, optical_channel_wv=optical_channel_wv,
            spatial_resolution=batch["spatial_resolution"],
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        preds = torch.argmax(logits, dim=1)
        metric.update(preds, labels)

    return metric.compute().item()


def main(args):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    def emit(model_name, gen_task, dataset_name, n_bands, metric_name, value, reason=""):
        row = {
            "model": model_name, "sensor": gen_task, "task": dataset_name,
            "n_bands": n_bands, "metric_name": metric_name, "value": value, "reason": reason,
        }
        writer.writerow(row)
        csv_file.flush()
        logger.info("%s", row)

    sensor_stats_cache = {}
    gen_tasks = order_gen_tasks(args.gen_tasks)
    device = torch.device(args.device)

    for model_name in args.models:
        for dataset_name in args.datasets:
            n_bands_by_task = {g: n_bands_for_gen_task(g) for g in gen_tasks}

            ckpt_dir = find_checkpoint(args.results_dir, dataset_name, model_name)
            if ckpt_dir is None:
                logger.warning("No checkpoint found for model=%s dataset=%s -- skipping all gen_tasks", model_name, dataset_name)
                for gen_task in gen_tasks:
                    emit(model_name, gen_task, dataset_name, n_bands_by_task[gen_task], "n/a", None, "checkpoint_not_found")
                continue

            metadata = get_enmap_downstream_metadata(dataset_name)
            num_classes, ignore_index = metadata["num_classes"], metadata["ignore_index"]
            channel_wv = np.array(metadata["s2c"]["channel_wv"])
            wv_min, wv_max = channel_wv.min(), channel_wv.max()

            model = None
            sanity_values = {}
            for gen_task in gen_tasks:
                n_bands = n_bands_by_task[gen_task]

                fixed_n_bands = FIXED_CHANNEL_MODELS.get(model_name)
                if fixed_n_bands is not None and fixed_n_bands != n_bands:
                    emit(model_name, gen_task, dataset_name, n_bands, "n/a", None,
                         f"fixed input channel count ({fixed_n_bands}) != n_bands ({n_bands})")
                    continue

                # A channel count that merely *differs* from what this checkpoint was fine-tuned
                # on is not treated as incompatible here -- e.g. SpecViT's spectral_adapter has
                # no weights tied to a specific band count and is expected to run zero-shot at a
                # different n_bands, same as the native ood_a/ood_full/ood_complement settings
                # already do for every model. Only a genuine runtime failure (e.g. a very small
                # n_bands collapsing an intermediate tensor to zero/negative length) gets caught
                # here and recorded -- never predicted in advance via a static table.
                try:
                    if model is None:
                        logger.info("Loading checkpoint for model=%s dataset=%s from %s", model_name, dataset_name, ckpt_dir)
                        model = LESSWithUPerNet.from_pretrained(
                            ckpt_dir,
                            num_labels=num_classes,
                        ).to(device)

                    dataset, _ = build_eval_dataset(args, dataset_name, gen_task, sensor_stats_cache)
                    use_rope = getattr(model.config, "use_rope_embed", False)
                    iou = evaluate(model, dataset, num_classes, ignore_index, args.batch_size, device,
                                   normalize_wv=use_rope, wv_min=wv_min, wv_max=wv_max)
                except Exception as e:
                    logger.exception(
                        "Runtime failure evaluating model=%s dataset=%s gen_task=%s (n_bands=%d)",
                        model_name, dataset_name, gen_task, n_bands,
                    )
                    emit(model_name, gen_task, dataset_name, n_bands, "n/a", None, f"runtime error: {e}")
                    continue

                emit(model_name, gen_task, dataset_name, n_bands, "IoU", iou)

                if gen_task in ("enmap_identity", "ood_full"):
                    sanity_values[gen_task] = iou
                    if len(sanity_values) == 2:
                        diff = abs(sanity_values["enmap_identity"] - sanity_values["ood_full"])
                        if diff > args.sanity_tol:
                            csv_file.close()
                            raise RuntimeError(
                                f"Sanity check failed for model={model_name} dataset={dataset_name}: "
                                f"enmap_identity IoU={sanity_values['enmap_identity']:.4f} vs "
                                f"ood_full IoU={sanity_values['ood_full']:.4f} (diff={diff:.4f} > "
                                f"--sanity_tol={args.sanity_tol}). Stop and debug the SRF/normalization "
                                "wiring before trusting other-sensor numbers."
                            )
                        logger.info(
                            "Sanity check passed for model=%s dataset=%s: enmap_identity=%.4f, ood_full=%.4f (diff=%.4f)",
                            model_name, dataset_name, sanity_values["enmap_identity"], sanity_values["ood_full"], diff,
                        )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-sensor SRF-resampling eval")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench downstream data root")
    parser.add_argument("--pretrain_data_dir", type=str, default=None,
                        help="EnMAP pretraining data root (SpectralEarthDataset), used to compute SRF std stats "
                             "when they aren't already cached")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results dir (checkpoints live under <results_dir>/models/)")
    parser.add_argument("--srf_cache_dir", type=str, default=os.path.join("results", "srf_stats"))
    parser.add_argument("--output_csv", type=str, default=os.path.join("results", "cross_sensor_eval.csv"))
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS, choices=list(ENCODER_CONFIGS.keys()))
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS, choices=list(ENMAP_DATASET.keys()))
    parser.add_argument("--gen_tasks", type=str, nargs="+", default=ALL_GEN_TASKS, choices=ALL_GEN_TASKS)
    parser.add_argument("--sanity_tol", type=float, default=0.1, help="Max allowed |enmap_identity - ood_full| mIoU difference")
    parser.add_argument("--n_patches", type=int, default=2000, help="Patches to sample when computing SRF std stats")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
