"""Linear-probing counterpart to eval_cross_sensor.py: evaluate a frozen-encoder linear probe,
trained once on --gen_task id (see launch_finetune_lp.sh, which trains finetune.py with --lp),
across the same {model} x {dataset} x {gen_task} settings -- no retraining, matching the ft
(full fine-tune) cross-sensor eval convention. Writes a tidy CSV with the same schema as
cross_sensor_eval.csv: model, sensor, task, n_bands, metric_name, value.

Unlike a fine-tuned checkpoint, finetune.py's --lp path trains and saves only the decoder (a
bare nn.Module, not a PreTrainedModel -- see model_init_template's `return model.decoder`), so
there's no saved encoder/config to reload via from_pretrained. This script instead rebuilds the
architecture from CLI args (get_task_model, same convention as finetune.py itself), reloads the
*original* pretrained encoder checkpoint (frozen, identical to what --lp training started from),
and loads only the trained decoder weights on top.

Usage:
    python3 GeospatialFM/finetune/eval_cross_sensor_lp.py \\
        --data_dir /datasets/disk3/geospatial \\
        --pretrain_data_dir /datasets/disk2/geospatial/enmap/enmap \\
        --output_csv results/cross_sensor_eval_lp.csv
"""
import argparse
import csv
import glob
import logging
import os

import torch

from GeospatialFM.finetune.args import parse_args as parse_finetune_args
from GeospatialFM.finetune.eval_cross_sensor import (
    ALL_GEN_TASKS,
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    FIXED_CHANNEL_MODELS,
    build_eval_dataset,
    evaluate,
    n_bands_for_gen_task,
    order_gen_tasks,
)
from GeospatialFM.finetune.utils import get_task_model
from GeospatialFM.datasets.enmap import get_enmap_downstream_metadata
from GeospatialFM.models.registry import ENCODER_CONFIGS

# SpatSigma bundles its task head inside the encoder (see finetune.py's compute_encoding), so
# there is no --lp checkpoint to find for it -- excluded from the default sweep, same models
# list as eval_cross_sensor.py's DEFAULT_MODELS otherwise.
LP_UNSUPPORTED_MODELS = {"spatsigma"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_FIELDS = ["model", "sensor", "task", "n_bands", "metric_name", "value", "reason"]

# Segmentation eval always runs at this crop size (see eval_cross_sensor.py's build_eval_dataset),
# independent of --crop_size, which is only meaningful for training.
EVAL_CROP_SIZE = 128


def find_lp_decoder_checkpoint(results_dir, dataset_name, model_name, lr):
    """Locate the single --lp checkpoint for (model, dataset), trained with --gen_task id and
    --learning_rate {lr} (see launch_finetune_lp.sh's RUN_NAME=${{MODEL_NAME}}_lp_${{GEN_TASK}}_lr${{LR}}
    convention) -- the highest-step checkpoint-N under results/models/<dataset_name>/<model_name>_lp_id_lr<lr>/."""
    pattern = os.path.join(results_dir, "models", dataset_name, f"{model_name}_lp_id_lr{lr}", "checkpoint-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _step(path):
        try:
            return int(path.rsplit("-", 1)[-1])
        except ValueError:
            return -1

    return max(candidates, key=_step)


def load_decoder_state_dict(ckpt_dir):
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    return torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")


def build_lp_model(model_name, dataset_name, data_dir, num_classes, pretrained_model_path, decoder_ckpt_dir, device):
    """Rebuild the (encoder + linear decoder) architecture exactly as finetune.py's --lp path
    does, reload the frozen pretrained encoder, then overwrite the decoder with the trained
    probe weights."""
    args = parse_finetune_args([
        "--dataset_name", dataset_name,
        "--task_type", "segmentation",
        "--data_dir", data_dir,
        "--model_name", model_name,
        "--run_name", "cross_sensor_lp_eval",
        "--lp",
    ])
    model = get_task_model(args, num_classes, EVAL_CROP_SIZE)
    model.load_pretrained_encoder(pretrained_model_path)
    model.decoder.load_state_dict(load_decoder_state_dict(decoder_ckpt_dir))
    model.eval()
    return model.to(device)


def main(args):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    def emit(model_name, gen_task, dataset_name, n_bands, metric_name, value, reason=""):
        row = {
            "model": f"{model_name}_lp", "sensor": gen_task, "task": dataset_name,
            "n_bands": n_bands, "metric_name": metric_name, "value": value, "reason": reason,
        }
        writer.writerow(row)
        csv_file.flush()
        logger.info("%s", row)

    sensor_stats_cache = {}
    gen_tasks = order_gen_tasks(args.gen_tasks)
    device = torch.device(args.device)

    for model_name in args.models:
        pretrained_model_path = os.path.join(args.pretrained_dir, model_name)

        for dataset_name in args.datasets:
            n_bands_by_task = {g: n_bands_for_gen_task(g) for g in gen_tasks}

            if model_name in LP_UNSUPPORTED_MODELS:
                logger.warning("--lp is not supported for model=%s -- skipping all gen_tasks", model_name)
                for gen_task in gen_tasks:
                    emit(model_name, gen_task, dataset_name, n_bands_by_task[gen_task], "n/a", None, "lp_not_supported")
                continue

            ckpt_dir = find_lp_decoder_checkpoint(args.results_dir, dataset_name, model_name, args.lr)
            if ckpt_dir is None:
                logger.warning("No --lp checkpoint found for model=%s dataset=%s -- skipping all gen_tasks", model_name, dataset_name)
                for gen_task in gen_tasks:
                    emit(model_name, gen_task, dataset_name, n_bands_by_task[gen_task], "n/a", None, "checkpoint_not_found")
                continue

            metadata = get_enmap_downstream_metadata(dataset_name)
            num_classes, ignore_index = metadata["num_classes"], metadata["ignore_index"]

            model = None
            sanity_values = {}
            for gen_task in gen_tasks:
                n_bands = n_bands_by_task[gen_task]

                fixed_n_bands = FIXED_CHANNEL_MODELS.get(model_name)
                if fixed_n_bands is not None and fixed_n_bands != n_bands:
                    emit(model_name, gen_task, dataset_name, n_bands, "n/a", None,
                         f"fixed input channel count ({fixed_n_bands}) != n_bands ({n_bands})")
                    continue

                try:
                    if model is None:
                        logger.info("Building --lp model=%s dataset=%s (encoder=%s, decoder=%s)",
                                    model_name, dataset_name, pretrained_model_path, ckpt_dir)
                        model = build_lp_model(
                            model_name, dataset_name, args.data_dir, num_classes,
                            pretrained_model_path, ckpt_dir, device,
                        )

                    dataset, _ = build_eval_dataset(args, dataset_name, gen_task, sensor_stats_cache)
                    iou = evaluate(model, dataset, num_classes, ignore_index, args.batch_size, device)
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
    parser = argparse.ArgumentParser(description="Cross-sensor SRF-resampling eval for the --lp (linear probing) baseline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench downstream data root")
    parser.add_argument("--pretrain_data_dir", type=str, default=None,
                        help="EnMAP pretraining data root (SpectralEarthDataset), used to compute SRF std stats "
                             "when they aren't already cached")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results dir (--lp checkpoints live under <results_dir>/models/)")
    parser.add_argument("--pretrained_dir", type=str, default=os.path.join("results", "models"),
                        help="Root dir containing each backbone's frozen pretrained checkpoint, i.e. <pretrained_dir>/<model_name>/ "
                             "(same convention as --pretrained_model_path in launch_finetune.sh)")
    parser.add_argument("--srf_cache_dir", type=str, default=os.path.join("results", "srf_stats"))
    parser.add_argument("--output_csv", type=str, default=os.path.join("results", "cross_sensor_eval_lp.csv"))
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS, choices=list(ENCODER_CONFIGS.keys()))
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--gen_tasks", type=str, nargs="+", default=ALL_GEN_TASKS, choices=ALL_GEN_TASKS)
    parser.add_argument("--lr", type=str, default="3e-4", help="Learning rate suffix used in the --lp run's name (see launch_finetune_lp.sh)")
    parser.add_argument("--sanity_tol", type=float, default=0.1, help="Max allowed |enmap_identity - ood_full| mIoU difference")
    parser.add_argument("--n_patches", type=int, default=2000, help="Patches to sample when computing SRF std stats")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
