"""Effective rank of frozen encoder representations, across {model} x {native gen_task} settings.

For the NeurIPS rebuttal question "what is the effective rank of the produced representations,
and how does it compare to other foundation models" -- computes three effective-rank measures
(RankMe, participation ratio, stable rank) on the pre-head pooled feature that each model's
downstream head actually consumes (the CLS token: `outputs[:, :, 0][:, 0, :]` for LESSViT,
`hidden_states[:, 0, :]` for every baseline wrapper -- see GeospatialFM/models/downstream_models.py's
LinearHead/LESSWithProjection.forward, which this script mirrors exactly, stopping one step before
the classifier).

Held-out set: the union of the val+test splits of the 5 downstream datasets (enmap_cdl, enmap_corine,
enmap_eurocrops, enmap_bdforet, enmap_bnetd) -- ~4853 samples, never touched by fine-tuning. The same
physical samples (dataset, split, sample_id) are reused across every model and every gen_task; only the
channel subsetting differs per gen_task (each EnMAP dataset class does this subsetting from the *same*
underlying raster inside __getitem__, so pooling by (dataset_name, split, local_idx) is sufficient to
guarantee sample identity across configs). The pool is written to --sample_index_file for reproducibility.

Centering: features are column-centered (subtract the per-dimension mean across the M held-out samples)
before every metric. No per-dimension standardization is applied. All three measures are then computed
from ONE shared SVD of the centered feature matrix: RankMe uses the singular values sigma directly;
participation ratio and stable rank use sigma^2 (proportional to the covariance eigenvalues, which is
scale-invariant for PR and exact for stable rank regardless of the M-1 vs M normalization convention).

Cross-model comparability: raw effective rank is bounded by the embedding dim D, which differs across
models/checkpoints (LESSViT's D is whatever --embed_dim was set to for the loaded checkpoint, not
necessarily 768 -- read D empirically from the extracted feature tensor, never assumed). Every measure
is therefore also reported normalized by D; the normalized columns are what should be used for the
cross-model comparison in the rebuttal, not the raw numbers.

Known limitations, recorded explicitly in the output rather than silently glossed over:
  - dinov3 is excluded by default: its patch embed is hardcoded to 155 input channels, which matches
    none of C120_VNIR+/C120_SWIR+/C82/C202 (120/120/82/202).
  - channelvit's `load_pretrained_weights` is a no-op in this repo (see wrappers/channelvit_wrapper.py)
    -- if you request it, the encoder runs with random init, not a trained checkpoint, and every row is
    tagged accordingly. It's also only valid at n_bands=202 (C202), same fixed-channel-count restriction
    as eval_cross_sensor.py's FIXED_CHANNEL_MODELS.
  - spatsigma exposes no pre-head pooled representation through the shared encoder interface --
    `forward_encoder` returns SSFusionFramework's fused output directly (already logits-shaped), and
    downstream_models.py special-cases it to skip CLS pooling entirely. Recorded as n/a.

Usage:
    python3 GeospatialFM/finetune/eval_effective_rank.py \\
        --data_dir /datasets/disk3/geospatial \\
        --pretrained_model_dir lessvit=results/models/LESSVIT_S_ablation_arm_a_less/checkpoint-20350 \\
        --embed_dim 384 --num_heads 6 --depth 12 --channel_embed_dims_per_head 4 --rank 1 --attn_type less \\
        --output_csv results/effective_rank.csv
"""
import argparse
import csv
import glob
import logging
import os
import random
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from GeospatialFM.data_process.collate_func import modal_specific_collate_fn
from GeospatialFM.data_process.transforms import get_enmap_transform
from GeospatialFM.datasets.enmap.enmap import SELECTED_CHANNEL_IDX_A, SELECTED_CHANNEL_IDX_B
from GeospatialFM.datasets.enmap.utils import ENMAP_DATASET
from GeospatialFM.models.downstream_models import LESSWithProjection, LESSWithProjectionConfig
from GeospatialFM.models.spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NATIVE_GEN_TASKS = ["id", "ood_a", "ood_full", "ood_complement"]
GEN_TASK_TO_PAPER_NAME = {
    "id": "C120_VNIR+",
    "ood_a": "C120_SWIR+",
    "ood_complement": "C82",
    "ood_full": "C202",
}

# Same restriction eval_cross_sensor.py encodes: these baselines' patch embed has a channel count
# baked in at construction time and can't ingest an arbitrary band set.
FIXED_CHANNEL_MODELS = {"channelvit": 202}

# ChannelViT emits one token per (channel, patch) pair, so its self-attention cost is
# quadratic in n_bands*num_patches -- at 202 channels (the only gen_task it ever actually
# runs, per FIXED_CHANNEL_MODELS) x 64 patches/128px-crop, --batch_size 32 tries to allocate
# ~239GiB. Every other wrapped encoder here tokenizes per-patch (channel count folds into a
# per-band embedding, not the token count), so this cap is ChannelViT-specific.
MODEL_BATCH_SIZE_CAP = {"channelvit": 2}

# Models with no accessible pre-head pooled representation through the shared encoder interface.
NO_PRE_HEAD_MODELS = {
    "spatsigma": "SSFusionFramework.forward returns fused logits directly (no CLS pooling path) "
                 "-- see downstream_models.py's SpatSigmaMixin special-case",
}

# dinov3 excluded by default: fixed at 155 input channels, which matches none of 120/82/202.
DEFAULT_MODELS = ["lessvit", "dofa", "specvit", "channelvit", "hyperfree", "spatsigma"]
DEFAULT_DATASETS = ["enmap_cdl", "enmap_corine", "enmap_eurocrops", "enmap_bdforet", "enmap_bnetd"]

DATASET_TASK_TYPE = {
    "enmap_cdl": "segmentation",
    "enmap_corine": "multilabel",
    "enmap_eurocrops": "segmentation",
    "enmap_bdforet": "segmentation",
    "enmap_bnetd": "segmentation",
}

ARCH_FIELDS = [
    "patch_size", "embed_dim", "channel_embed_dims_per_head", "depth", "num_heads", "mlp_ratio",
    "qkv_bias", "qk_norm", "drop_path_rate", "drop_path_uniform", "init_values", "attn_drop", "proj_drop",
    "num_experts", "use_moe", "topk", "rank", "attn_type", "fusion_init_scale", "use_rope_embed",
    "rope_embed_base", "channel_dropout",
]

CSV_FIELDS = [
    "model", "config", "gen_task", "D", "n_samples",
    "rankme", "rankme_norm", "PR", "PR_norm", "stable_rank", "stable_rank_norm",
    "reason",
]


def n_bands_for_gen_task(gen_task):
    if gen_task == "id":
        return len(SELECTED_CHANNEL_IDX_B)
    elif gen_task == "ood_a":
        return len(SELECTED_CHANNEL_IDX_A)
    elif gen_task == "ood_full":
        return 202
    elif gen_task == "ood_complement":
        return 202 - len(SELECTED_CHANNEL_IDX_B)
    raise ValueError(f"Unknown gen_task: {gen_task}")


def build_holdout_pool(data_dir, datasets, splits=("val", "test")):
    """Union of split-file entries (default val+test) across `datasets`. Every EnMAP dataset
    class reads its split file into `sample_collection` preserving line order (see e.g.
    enmap_cdl.py's read_split_file), so (dataset_name, split, local_idx) is a stable,
    gen_task-independent identity for a physical sample -- channel subsetting happens
    per-gen_task inside __getitem__ on the same underlying raster."""
    pool = []
    for dataset_name in datasets:
        for split in splits:
            split_file = os.path.join(data_dir, "splits", dataset_name, f"{split}.txt")
            with open(split_file, "r") as f:
                sample_ids = [line.strip() for line in f if line.strip()]
            for local_idx, sample_id in enumerate(sample_ids):
                pool.append((dataset_name, split, sample_id, local_idx))
    return pool


def save_sample_index_file(path, pool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("dataset_name\tsplit\tsample_id\n")
        for dataset_name, split, sample_id, _ in sorted(pool, key=lambda t: (t[0], t[1], t[2])):
            f.write(f"{dataset_name}\t{split}\t{sample_id}\n")


def parse_pretrained_model_dir(pairs):
    mapping = {}
    for pair in pairs or []:
        model_name, _, path = pair.partition("=")
        if not path:
            raise argparse.ArgumentTypeError(f"--pretrained_model_dir expects MODEL=PATH, got: {pair!r}")
        mapping[model_name] = path
    return mapping


def resolve_checkpoint_dir(model_name, args):
    override = args.pretrained_model_dir_map.get(model_name)
    if override is not None:
        return override
    return os.path.join(args.results_dir, "models", model_name)


def build_arch_namespace(args, model_name):
    ns = argparse.Namespace(**{f: getattr(args, f) for f in ARCH_FIELDS})
    ns.model_name = model_name
    ns.task_type = "classification"
    ns.return_dict = False
    ns.use_perception_field_mask = False
    ns.attention_radius = 640
    return ns


def build_frozen_model(model_name, arch_ns):
    config = LESSWithProjectionConfig(num_labels=2, **vars(arch_ns))
    return LESSWithProjection(config)


def load_finetuned_encoder(model, ckpt_dir):
    """Load just the `encoder.*` submodule out of a full downstream-task checkpoint (e.g. a
    LESSWithUPerNet/LESSWithProjection saved by finetune.py's Trainer -- HF save_pretrained's
    model.safetensors, keys prefixed "encoder."/"decoder." or "encoder."/"classifier."),
    ignoring the decoder/classifier head entirely. This is deliberately generic across every
    --model_name (unlike the frozen-checkpoint path's model.load_pretrained_encoder(), which for
    every non-lessvit backbone dispatches to that wrapper's own load_pretrained_weights() --
    logic written to parse each backbone's ORIGINAL released pretrained-checkpoint format, e.g.
    dofa_wrapper.py strips a 'mask_token' key and resizes pos_embed, specvit_wrapper.py filters
    for a 'vit.' prefix -- none of which matches a finetuned checkpoint's keys, which are exactly
    today's model's own state_dict names since finetune.py builds/saves with the same code this
    script imports)."""
    from safetensors import safe_open

    safetensors_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not safetensors_paths:
        raise FileNotFoundError(f"No .safetensors file found in {ckpt_dir}")

    model_state = model.state_dict()
    with safe_open(safetensors_paths[-1], framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith("encoder.") or key == "encoder.perception_field_mask":
                continue
            if key not in model_state:
                raise KeyError(
                    f"Checkpoint key {key!r} has no matching encoder parameter -- "
                    f"the eval --model_name/arch flags likely don't match what finetune.py used"
                )
            model_state[key].copy_(f.get_tensor(key))


@torch.no_grad()
def extract_pooled_features(model, batch, device):
    """Mirrors LESSWithProjection.forward (downstream_models.py) up to, but not including, the
    classifier -- i.e. exactly the pre-head pooled representation the LinearHead actually consumes."""
    optical = batch["optical"].to(device)
    optical_channel_wv = batch["optical_channel_wv"].to(device)
    spatial_resolution = batch["spatial_resolution"]
    wave_list = (optical_channel_wv.squeeze(dim=0) / 1000).cpu().tolist()

    if isinstance(model.encoder, SpatialSpectralLowRankViTEncoder):
        outputs = model.encoder(optical, None, optical_channel_wv, None, spatial_resolution)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        pooled = outputs[:, :, 0][:, 0, :]  # [B, C+1, D] -> channel-CLS x spatial-CLS -> [B, D]
    else:
        hidden_states = model.encoder.forward_encoder(
            optical, wave_list=wave_list, spatial_resolution=spatial_resolution
        )
        pooled = hidden_states[:, 0, :]  # CLS token, [B, D]

    return pooled.float().cpu()


def extract_feature_matrix(model, args, by_ds_split, gen_task, device, model_name=None):
    # normalize_wv must reflect whether *this specific model* uses RoPE
    # (RopePositionChannelEmbedding assumes optical_channel_wv already sits in [0,1] --
    # pos_chan_embed.py's `coords_c = 2*optical_channel_wv - 1`). Only SpatialSpectralLowRankViTEncoder
    # (LESSViT) uses this convention -- the else branch in extract_pooled_features feeds other
    # backbones a wholly different (/1000, micrometers-ish) convention via `wave_list`, unrelated
    # to this normalization and untouched here.
    use_rope = isinstance(model.encoder, SpatialSpectralLowRankViTEncoder) and getattr(model.config, "use_rope_embed", False)
    batch_size = min(args.batch_size, MODEL_BATCH_SIZE_CAP.get(model_name, args.batch_size))

    feats = []
    for (dataset_name, split), local_indices in by_ds_split.items():
        dataset_cls = ENMAP_DATASET[dataset_name]
        metadata = dataset_cls.metadata
        task_type = DATASET_TASK_TYPE[dataset_name]
        optical_mean, optical_std = metadata["s2c"]["mean"], metadata["s2c"]["std"]
        channel_wv = np.array(metadata["s2c"]["channel_wv"])
        wv_min, wv_max = channel_wv.min(), channel_wv.max()

        _, eval_transform = get_enmap_transform(
            task_type, crop_size=args.crop_size, scale=None, random_rotation=False,
            optical_mean=optical_mean, optical_std=optical_std, dataset_name=dataset_name,
        )
        dataset = dataset_cls(root=args.data_dir, split=split, transform=eval_transform, gen_task=gen_task)
        subset = Subset(dataset, local_indices)
        collate_fn = partial(modal_specific_collate_fn, modal="optical", normalize_wv=use_rope, wv_min=wv_min, wv_max=wv_max)
        loader = DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            num_workers=args.dataloader_num_workers, collate_fn=collate_fn,
        )
        for batch in loader:
            feats.append(extract_pooled_features(model, batch, device))

    X = torch.cat(feats, dim=0)
    return X.double().numpy()


def effective_rank_metrics(X):
    """X: [M, D] float64, NOT yet centered. Returns RankMe/PR/stable-rank (raw + /D) from one shared
    SVD of the column-centered matrix -- see module docstring for the centering/normalization choice."""
    Xc = X - X.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)

    eps = 1e-12
    p = (S + eps) / (S.sum() + eps * len(S))
    H = -(p * np.log(p)).sum()
    rankme = float(np.exp(H))

    lam = S ** 2
    lam_sum = float(lam.sum())
    pr = (lam_sum ** 2) / float((lam ** 2).sum())
    stable_rank = lam_sum / float(lam.max())

    D = X.shape[1]
    return {
        "rankme": rankme, "rankme_norm": rankme / D,
        "PR": pr, "PR_norm": pr / D,
        "stable_rank": stable_rank, "stable_rank_norm": stable_rank / D,
    }


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    pool = build_holdout_pool(args.data_dir, args.datasets, splits=args.splits)
    if args.n_samples is not None and args.n_samples < len(pool):
        pool = random.sample(pool, args.n_samples)
    save_sample_index_file(args.sample_index_file, pool)
    logger.info("Held-out pool: %d samples across %d datasets, splits=%s (saved to %s)",
                len(pool), len(args.datasets), args.splits, args.sample_index_file)

    by_ds_split = defaultdict(list)
    for dataset_name, split, _sample_id, local_idx in pool:
        by_ds_split[(dataset_name, split)].append(local_idx)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    rows = []

    def emit(model_name, gen_task, D=None, n_samples=None, reason="", **metrics):
        row = {
            "model": model_name, "config": GEN_TASK_TO_PAPER_NAME[gen_task], "gen_task": gen_task,
            "D": D, "n_samples": n_samples, "reason": reason,
        }
        for k in ("rankme", "rankme_norm", "PR", "PR_norm", "stable_rank", "stable_rank_norm"):
            row[k] = metrics.get(k)
        writer.writerow(row)
        csv_file.flush()
        rows.append(row)
        logger.info("%s", row)
        return row

    for model_name in args.models:
        if model_name in NO_PRE_HEAD_MODELS:
            for gen_task in args.gen_tasks:
                emit(model_name, gen_task, reason=NO_PRE_HEAD_MODELS[model_name])
            continue

        if model_name == "channelvit":
            logger.warning(
                "channelvit: load_pretrained_weights() is a no-op in this repo -- the encoder will "
                "run with RANDOM weights, not a trained checkpoint. Rows below are tagged 'random_init'."
            )

        ckpt_dir = resolve_checkpoint_dir(model_name, args)
        arch_ns = build_arch_namespace(args, model_name)
        model = build_frozen_model(model_name, arch_ns).to(device)

        loaded = model_name == "channelvit"  # its loader is a no-op regardless of ckpt_dir
        if not loaded:
            if ckpt_dir is None or not os.path.isdir(ckpt_dir):
                for gen_task in args.gen_tasks:
                    emit(model_name, gen_task, reason="checkpoint_not_found")
                continue
            try:
                if args.finetuned:
                    load_finetuned_encoder(model, ckpt_dir)
                else:
                    model.load_pretrained_encoder(ckpt_dir)
                loaded = True
            except Exception as e:
                logger.exception("Failed to load checkpoint for %s from %s", model_name, ckpt_dir)
                for gen_task in args.gen_tasks:
                    emit(model_name, gen_task, reason=f"checkpoint_load_failed: {e}")
                continue
        model.eval()

        for gen_task in args.gen_tasks:
            n_bands = n_bands_for_gen_task(gen_task)
            fixed_n_bands = FIXED_CHANNEL_MODELS.get(model_name)
            if fixed_n_bands is not None and fixed_n_bands != n_bands:
                emit(model_name, gen_task,
                     reason=f"fixed input channel count ({fixed_n_bands}) != n_bands ({n_bands})")
                continue
            try:
                X = extract_feature_matrix(model, args, by_ds_split, gen_task, device, model_name=model_name)
                metrics = effective_rank_metrics(X)
            except Exception as e:
                logger.exception("Runtime failure for model=%s gen_task=%s", model_name, gen_task)
                emit(model_name, gen_task, reason=f"runtime_error: {e}")
                continue
            reason = "" if model_name != "channelvit" else "random_init (see warning above)"
            emit(model_name, gen_task, D=X.shape[1], n_samples=X.shape[0], reason=reason, **metrics)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_file.close()
    print_markdown_table(rows)
    print_vnir_ood_comparison(rows)


def print_markdown_table(rows):
    print("\n" + "=" * 100)
    print("RankMe/D is the primary cross-model comparison column (raw numbers are NOT comparable")
    print("across models with different D -- see module docstring).")
    print("=" * 100)
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    for gen_task, config in GEN_TASK_TO_PAPER_NAME.items():
        if config not in by_config:
            continue
        print(f"\n### {config} (gen_task={gen_task})\n")
        print("| model | D | n | RankMe/D | RankMe | PR/D | PR | stable_rank/D | stable_rank |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in by_config[config]:
            if r["rankme"] is None:
                print(f"| {r['model']} | - | - | n/a | n/a | n/a | n/a | n/a | n/a ({r['reason']}) |")
            else:
                print(
                    f"| {r['model']} | {r['D']} | {r['n_samples']} | {r['rankme_norm']:.4f} | "
                    f"{r['rankme']:.2f} | {r['PR_norm']:.4f} | {r['PR']:.2f} | "
                    f"{r['stable_rank_norm']:.4f} | {r['stable_rank']:.2f} |"
                )


def print_vnir_ood_comparison(rows):
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["gen_task"]] = r

    print("\n" + "=" * 100)
    print("Per-model RankMe/D summary, C120_VNIR+ (training distribution) vs OOD configs:")
    print("=" * 100)
    for model_name, by_task in by_model.items():
        vnir = by_task.get("id")
        if vnir is None or vnir["rankme"] is None:
            print(f"{model_name}: n/a on C120_VNIR+ ({vnir['reason'] if vnir else 'not run'})")
            continue
        line = f"{model_name}: C120_VNIR+ RankMe/D={vnir['rankme_norm']:.4f}"
        for gen_task in ("ood_a", "ood_complement", "ood_full"):
            r = by_task.get(gen_task)
            if r is None or r["rankme"] is None:
                continue
            delta = r["rankme_norm"] - vnir["rankme_norm"]
            line += f", {GEN_TASK_TO_PAPER_NAME[gen_task]}={r['rankme_norm']:.4f} ({delta:+.4f})"
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Effective rank of frozen encoder representations")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the GFMBench downstream data root")
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS, choices=list(ENMAP_DATASET.keys()),
                         help="Downstream datasets whose --splits are pooled into the held-out set")
    parser.add_argument("--splits", type=str, nargs="+", default=["val", "test"], choices=["train", "val", "test"],
                         help="Which split(s) of --datasets to pool into the held-out set. Default val+test "
                              "(never touched by fine-tuning); pass --splits test to eval on the test set only "
                              "(e.g. when scoring a --finetuned checkpoint that already trained on train+val).")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                         choices=["lessvit", "specvit", "dinov3", "dofa", "spatsigma", "channelvit", "hyperfree"])
    parser.add_argument("--gen_tasks", type=str, nargs="+", default=NATIVE_GEN_TASKS, choices=NATIVE_GEN_TASKS)
    parser.add_argument("--pretrained_model_dir", type=str, nargs="+", default=[],
                         help="MODEL=PATH pairs, e.g. lessvit=results/models/foo/checkpoint-1000. "
                              "Any model not given here falls back to <results_dir>/models/<model_name>/.")
    parser.add_argument("--finetuned", action="store_true",
                         help="--pretrained_model_dir points at full downstream-task checkpoints (e.g. "
                              "finetune.py's results/models/<dataset>/<model>_<gen_task>_lr*/checkpoint-*) "
                              "rather than frozen pretrained backbones -- loads only the encoder.* "
                              "submodule out of each, ignoring the decoder/classifier head.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--crop_size", type=int, default=64,
                         help="Explicit eval-time center-crop size (deterministic, is_train=False). "
                              "Passed explicitly to route around metadata['size'] not being set on any "
                              "downstream dataset class -- see finetune.py's `metadata['size']` usage.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=None,
                         help="Optional cap: randomly subsample the pooled held-out set to this size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, default=os.path.join("results", "effective_rank.csv"))
    parser.add_argument("--sample_index_file", type=str,
                         default=os.path.join("results", "effective_rank_sample_indices.txt"))

    # LESSViT architecture (only used when 'lessvit' in --models) -- must match the checkpoint given
    # via --pretrained_model_dir lessvit=... or state_dict loading fails with a shape mismatch. Defaults
    # mirror GeospatialFM/finetune/args.py's Model arguments exactly.
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--channel_embed_dims_per_head", type=int, default=4)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--qkv_bias", type=bool, default=True)
    parser.add_argument("--qk_norm", type=bool, default=False)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_uniform", type=bool, default=False)
    parser.add_argument("--init_values", type=float, default=None)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--proj_drop", type=float, default=0.0)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--use_moe", action="store_true")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--attn_type", type=str, default="less", choices=["less", "full", "additive"])
    parser.add_argument("--fusion_init_scale", type=float, default=1.0)
    parser.add_argument("--use_rope_embed", action="store_true")
    parser.add_argument("--rope_embed_base", type=float, default=100.0)
    parser.add_argument("--channel_dropout", type=float, nargs="+", default=None)

    args = parser.parse_args()
    args.pretrained_model_dir_map = parse_pretrained_model_dir(args.pretrained_model_dir)
    return args


if __name__ == "__main__":
    main(parse_args())
