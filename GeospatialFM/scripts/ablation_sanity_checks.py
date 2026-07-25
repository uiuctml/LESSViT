"""
Sanity checks for the three-arm attention ablation (arm_a_less / arm_b_full /
arm_c_additive), to be run standalone before launching any full pretraining job.

Usage:
    python -m GeospatialFM.scripts.ablation_sanity_checks --data_dir /path/to/enmap
    python -m GeospatialFM.scripts.ablation_sanity_checks --skip_timing --skip_data_order

Implements the 5 checks from the ablation plan:
    1. Param count / FLOPs estimate per arm, side by side.
    2. Zero arm C's channel (P_c) projection -> output constant along channel axis.
    3. Timed forward+backward step per arm at HCS-pretrain channel count and full C=120.
    4. Arm A vs arm C post-fusion activation-std calibration at init.
    5. CLS/grid-index placement (arm B flatten/unflatten arithmetic) + data-order
       determinism across arms given the same seed.
"""
import argparse
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from GeospatialFM.models.spatial_spectral_low_rank_vit import SpatialSpectralLowRankViTConfig
from GeospatialFM.models.mae import SpatialSpectralMAEViT
from GeospatialFM.models.attention_ops import LESSAttention, AdditiveSpatialSpectralAttention

ARM_TYPES = ["less", "full", "additive"]

# ViT-S sizing, matching launch_train_ablation_base.sh, at the ablation's 64x64/patch16 resolution.
VIT_S_KWARGS = dict(
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    channel_embed_dims_per_head=4,
    decoder_embed_dim=512,
    decoder_depth=4,
    decoder_num_heads=16,
    decoder_channel_embed_dims_per_head=4,
    rank=1,
    use_rope_embed=True,
    mask_ratio=0.75,
    channel_mask_ratio=0.75,
    init_values=1.0,
    proj_drop=0.1,
    attn_drop=0.1,
    drop_path_rate=0.1,
)

CROP_SIZE = 64
HCS_C = 30       # representative HCS-subsampled pretraining channel count (range is ~24-36)
FULL_C = 120     # fine-tuning / inference channel count


def build_config(attn_type: str, fusion_init_scale: float = 1.0) -> SpatialSpectralLowRankViTConfig:
    return SpatialSpectralLowRankViTConfig(
        attn_type=attn_type,
        fusion_init_scale=fusion_init_scale,
        decoder_out_chans=1,
        **VIT_S_KWARGS,
    )


def make_dummy_batch(B: int, C: int, crop_size: int, device):
    optical = torch.randn(B, C, crop_size, crop_size, device=device)
    optical_channel_wv = torch.rand(1, C, device=device)  # normalized in [0,1], mirrors normalize_wv path
    return optical, optical_channel_wv


# --------------------------------------------------------------------------- #
# Check 1: param count / FLOPs
# --------------------------------------------------------------------------- #
def check_param_count_and_flops():
    print("\n=== Check 1: param count / FLOPs per arm ===")
    results = {}
    for attn_type in ARM_TYPES:
        cfg = build_config(attn_type)
        model = SpatialSpectralMAEViT(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        n_params_no_embed = sum(
            p.numel() for n, p in model.named_parameters()
            if "patch_embed" not in n
        )
        for C in (HCS_C, FULL_C):
            tokens = 1 * C * CROP_SIZE * CROP_SIZE  # B=1, matches estimate_tokens (optical.numel())
            flops = 6 * tokens * n_params_no_embed  # same formula as MAETrainer.floating_point_ops
            results[(attn_type, C)] = flops
        results[attn_type] = n_params
        print(f"  {attn_type:>9s}: {n_params:>12,d} params  "
              f"(FLOPs@C={HCS_C}: {results[(attn_type, HCS_C)]:.3e}, "
              f"FLOPs@C={FULL_C}: {results[(attn_type, FULL_C)]:.3e})")

    delta = results["additive"] - results["less"]
    pct = 100.0 * delta / results["less"]
    print(f"  arm C - arm A param delta: {delta:,d} ({pct:.1f}% of arm A)")
    if results["additive"] <= results["less"]:
        print("  !! WARNING: arm C is not larger than arm A -- investigate.")
    elif pct > 50:
        print(f"  NOTE: arm C is {pct:.0f}% larger than arm A, not just 'slightly' larger.\n"
              f"        This is an expected consequence of d_s=d_c=D_h (dropping the Kronecker's\n"
              f"        c_head_dim*s_head_dim factorization inflates LowDimPool's projections and\n"
              f"        the branch attentions themselves, not just the fusion step) -- flagging\n"
              f"        per the plan's request to report the delta, not silently downplay it.")
    return results


# --------------------------------------------------------------------------- #
# Check 2: zero arm C's channel projection -> constant along channel axis
# --------------------------------------------------------------------------- #
def check_additive_zero_ablation():
    print("\n=== Check 2: zero arm C's P_c -> output constant along channel axis ===")
    # This must test the attention module's own output in isolation, not a full
    # multi-block model: the residual stream (x = x + attn(norm1(x))) carries
    # forward channel-varying content from patch embeddings/earlier blocks
    # regardless of what one block's attention output looks like, so checking the
    # full model's final hidden state would not actually exercise this property.
    dim, num_heads, rank = 384, 6, VIT_S_KWARGS["rank"]
    channel_dim, spatial_dim = 24, 96  # arm A's ViT-S per-branch widths, reused unchanged by arm C
    B, C, HW = 2, HCS_C + 1, 17  # C+1/HW+1 CLS-augmented grid, matches block-level input
    attn = AdditiveSpatialSpectralAttention(dim=dim, channel_dim=channel_dim, spatial_dim=spatial_dim, num_heads=num_heads, rank=rank)
    attn.eval()
    with torch.no_grad():
        attn.P_c.weight.zero_()
        attn.P_c.bias.zero_()
        x = torch.randn(B, C, HW, dim)
        out = attn(x)  # B, C, HW, D

    max_dev = (out - out[:, :1, :, :]).abs().max().item()
    print(f"  max deviation across channel axis after zeroing P_c: {max_dev:.3e}")
    assert max_dev < 1e-4, "output is NOT constant along the channel axis after zeroing P_c"
    print("  PASS")


# --------------------------------------------------------------------------- #
# Check 3: timed forward+backward step, wall clock / peak mem / tokens-per-sec
# --------------------------------------------------------------------------- #
def check_timed_step(device: str, n_steps: int = 5, batch_size: int = 8):
    print(f"\n=== Check 3: timed step per arm (device={device}, n_steps={n_steps}, B={batch_size}) ===")
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, skipping.")
        return

    for C_label, C in (("HCS", HCS_C), ("full", FULL_C)):
        print(f"  -- C={C} ({C_label}) --")
        for attn_type in ARM_TYPES:
            cfg = build_config(attn_type)
            model = SpatialSpectralMAEViT(cfg).to(device)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            optical, wv = make_dummy_batch(batch_size, C, CROP_SIZE, device=device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                for _ in range(n_steps):
                    opt.zero_grad()
                    out = model(optical=optical, optical_channel_wv=wv, modal="optical")
                    loss = (out["optical_recon"] ** 2).mean()
                    loss.backward()
                    opt.step()
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                tokens_per_step = batch_size * C * (CROP_SIZE // cfg.patch_size) ** 2
                tokens_per_sec = tokens_per_step * n_steps / dt
                peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
                print(f"    {attn_type:>9s}: {dt:.2f}s total, {tokens_per_sec:.0f} tokens/s, "
                      f"peak_mem={peak_mem:.2f} GB")
            except torch.cuda.OutOfMemoryError as e:
                print(f"    {attn_type:>9s}: OOM at C={C}, batch_size={batch_size} -- "
                      f"reporting, not silently shrinking batch size. ({e})")
            finally:
                del model, opt
                if device == "cuda":
                    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Check 4: fusion_init_scale calibration (arm A vs arm C post-fusion std at init)
# --------------------------------------------------------------------------- #
def check_fusion_init_scale_calibration(seed: int = 0):
    print("\n=== Check 4: fusion_init_scale calibration (arm A vs arm C post-fusion std) ===")
    dim, num_heads, rank = 384, 6, VIT_S_KWARGS["rank"]
    channel_dim_a, spatial_dim_a = 24, 96  # arm A's default ViT-S per-branch widths
    B, C, HW = 4, HCS_C + 1, 17  # +1/HW+1 CLS-augmented grid sizes, matches block-level input

    def build_dummy_input(g):
        return torch.randn(B, C, HW, dim, generator=g)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x = build_dummy_input(g)

    torch.manual_seed(seed)
    arm_a = LESSAttention(dim=dim, channel_dim=channel_dim_a, spatial_dim=spatial_dim_a, num_heads=num_heads, rank=rank)
    arm_a.eval()
    with torch.no_grad():
        out_a = arm_a(x)
    std_a = out_a.std().item()
    print(f"  arm A post-fusion std (scale=1.0 baseline): {std_a:.4f}")

    def measure(scale):
        torch.manual_seed(seed)
        arm_c = AdditiveSpatialSpectralAttention(dim=dim, channel_dim=channel_dim_a, spatial_dim=spatial_dim_a, num_heads=num_heads, rank=rank, fusion_init_scale=scale)
        arm_c.eval()
        with torch.no_grad():
            out_c = arm_c(x)
        return out_c.std().item()

    # A single weight-and-bias rescale of P_c/P_s makes their own output scale
    # cleanly (P(x) -> scale*P(x) exactly), but the shared final `proj` layer's
    # own bias is not part of that rescale, so it adds a small scale-independent
    # offset to the measured std. Iterate the ratio a few times rather than
    # assuming one ratio is exact.
    scale = 1.0
    std_c = measure(scale)
    print(f"  arm C post-fusion std (scale=1.0, uncalibrated): {std_c:.4f}")
    for _ in range(8):
        rel_diff = abs(std_c - std_a) / std_a
        if rel_diff < 0.10:
            break
        scale *= std_a / std_c
        std_c = measure(scale)

    print(f"  converged fusion_init_scale = {scale:.4f}")
    print(f"  arm C post-fusion std (scale={scale:.4f}, calibrated): {std_c:.4f} "
          f"(rel. diff from arm A: {rel_diff:.1%})")
    assert rel_diff < 0.10, f"calibrated arm C std does not match arm A within 10% (got {rel_diff:.1%})"
    print(f"  PASS -- use --fusion_init_scale {scale:.4f} for arm C launches")
    return scale


# --------------------------------------------------------------------------- #
# Check 5a: CLS/grid-index placement -- arm B's flatten/unflatten arithmetic
# --------------------------------------------------------------------------- #
def check_cls_placement_arm_b():
    print("\n=== Check 5a: CLS/grid-index round-trip through arm B's flatten/unflatten ===")
    B, num_heads, C, HW, Dh = 1, 2, 5, 3, 4
    x = torch.zeros(B, num_heads, C, HW, Dh)
    marker = 999.0
    x[0, :, 0, 0, :] = marker  # global CLS cell, grid position (c=0, n=0)

    flat = x.reshape(B, num_heads, C * HW, Dh)
    assert torch.equal(flat[0, :, 0, :], torch.full((num_heads, Dh), marker)), \
        "grid position (0,0) did not land at flat index 0"

    unflat = flat.reshape(B, num_heads, C, HW, Dh)
    assert torch.equal(x, unflat), "flatten/unflatten round-trip is not exact"
    assert torch.equal(unflat[0, :, 0, 0, :], torch.full((num_heads, Dh), marker)), \
        "grid position (0,0) did not round-trip back to (0,0)"
    print("  PASS -- (c=0,n=0) <-> flat index 0, round-trips exactly")

    print("  Cross-arm output shape check (via full encoder forward):")
    for attn_type in ARM_TYPES:
        cfg = build_config(attn_type)
        cfg.depth = 2  # keep this cheap, only shape matters here
        model = SpatialSpectralMAEViT(cfg)
        model.eval()
        optical, wv = make_dummy_batch(2, HCS_C, CROP_SIZE, device="cpu")
        with torch.no_grad():
            x_out, cls_token, patch_tokens, _ = model.encoder(optical=optical, optical_channel_wv=wv, spatial_resolution=30)
        expected_C, expected_HW = HCS_C + 1, (CROP_SIZE // cfg.patch_size) ** 2 + 1
        assert x_out.shape == (2, expected_C, expected_HW, cfg.embed_dim), \
            f"{attn_type}: unexpected grid shape {x_out.shape}"
        assert cls_token.shape == (2, cfg.embed_dim)
        assert patch_tokens.shape == (2, HCS_C, expected_HW - 1, cfg.embed_dim)
        print(f"    {attn_type:>9s}: grid {tuple(x_out.shape)}, cls {tuple(cls_token.shape)} -- OK")


# --------------------------------------------------------------------------- #
# Check 5b: data order determinism across arms given the same seed
# --------------------------------------------------------------------------- #
def check_data_order_determinism(data_dir: str, seed: int = 42):
    print(f"\n=== Check 5b: data order determinism given seed={seed} ===")
    if not data_dir:
        print("  --data_dir not provided, skipping (this check only exercises the dataloader, "
              "not attn_type, so it's independent of which arm you're about to launch).")
        return
    from transformers import set_seed
    from GeospatialFM.datasets.enmap import get_enmap_metadata, SpectralEarthDataset, SELECTED_CHANNEL_IDX_B
    from GeospatialFM.data_process import pretrain_transform, unimodal_collate_fn
    from torch.utils.data import DataLoader

    metadata = get_enmap_metadata()
    optical_mean = [metadata["s2c"]["mean"][i] for i in SELECTED_CHANNEL_IDX_B]
    optical_std = [metadata["s2c"]["std"][i] for i in SELECTED_CHANNEL_IDX_B]
    channel_wv = np.array(metadata["s2c"]["channel_wv"])
    transform = partial(pretrain_transform, crop_size=CROP_SIZE, optical_mean=optical_mean,
                         optical_std=optical_std, radar_mean=None, radar_std=None)
    collate_fn = partial(unimodal_collate_fn, modal="optical", transform=transform, random_crop=False,
                          scale=None, crop_size=CROP_SIZE, normalize_wv=True,
                          wv_max=channel_wv.max(), wv_min=channel_wv.min())

    def first_batch():
        set_seed(seed)
        dataset = SpectralEarthDataset(root=data_dir, subset=120)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        return next(iter(loader))

    batch_1 = first_batch()
    batch_2 = first_batch()
    match = torch.equal(batch_1["optical"], batch_2["optical"])
    print(f"  first-batch optical tensors identical across two independent runs given seed={seed}: {match}")
    assert match, "data order is not deterministic given the same seed -- check dataloader worker seeding"
    print("  PASS (attn_type never touches the data pipeline, so this holds for all 3 arms identically)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="EnMAP data dir, for check 5b. Skipped if omitted.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_timing", action="store_true")
    parser.add_argument("--skip_data_order", action="store_true")
    parser.add_argument("--timing_steps", type=int, default=5)
    parser.add_argument("--timing_batch_size", type=int, default=8)
    args = parser.parse_args()

    check_param_count_and_flops()
    check_additive_zero_ablation()
    if not args.skip_timing:
        check_timed_step(args.device, n_steps=args.timing_steps, batch_size=args.timing_batch_size)
    check_fusion_init_scale_calibration()
    check_cls_placement_arm_b()
    if not args.skip_data_order:
        check_data_order_determinism(args.data_dir)

    print("\nAll requested sanity checks completed.")


if __name__ == "__main__":
    main()
