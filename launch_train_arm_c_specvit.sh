#!/usr/bin/env bash
# Arm C (redefined): SpecViT -- a genuinely different architecture family (spectral
# adapter + spatial-only ViT tokenization, GeospatialFM/models/SpecViT/), not another
# attention operator on the LowRank backbone -- with HCS-style random channel masking
# at input (GeospatialFM/models/SpecViT/mae.py's new --channel_dropout support: samples
# a fresh channel_mask_ratio each training step from [min,max], same values/semantics
# as the LowRank family's --channel_dropout, applied via SpecViT's existing zero-mask
# mechanism rather than a channel-count-shrinking gather -- see mae.py's
# SpecViTMAEConfig.channel_dropout docstring for why).
#
# This SUPERSEDES the old arm_c_additive as "arm C" going forward, but does not
# overwrite it -- results/models/LESSVIT_S_ablation_arm_c_additive/ and its
# fine-tune/eval results are left untouched for reference. Because this is a
# different architecture, not a different attention operator on the same backbone,
# none of arm_c_additive's pretrain/fine-tune/eval artifacts carry over -- this is a
# full from-scratch run.
#
# Every hyperparameter that's genuinely comparable across architectures (data, crop
# size, epochs, LR, batch size, seed, spatial mask ratio, channel_dropout range,
# regularization) is matched exactly to launch_train_ablation_base.sh's
# ABLATION_COMMON_ARGS. LowRank-only flags (--attn_type, --fusion_init_scale, --rank,
# --channel_embed_dims_per_head, --decoder_channel_embed_dims_per_head,
# --use_rope_embed, --rope_embed_base, --channel_mask_ratio -- inert here since
# --channel_dropout takes precedence during training, see mae.py) are dropped.
# SpecViT-specific sizing (embed_dim/depth/num_heads) matches SpecViTSmall's own
# preset, which happens to equal the ViT-S sizing arms A/B use (384/12/6) --
# reduced_channels/decoder_* use SpecViT's own established defaults rather than
# forcing LowRank's decoder shape onto a different architecture.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

RUN_NAME=LESSVIT_S_ablation_arm_c_specvit

SPECVIT_ARGS=(
    --model_name specvit
    --dataset_name enmap
    --data_dir /datasets/disk2/geospatial/enmap/enmap
    --enmap_subset 120
    --per_device_train_batch_size 256
    --gradient_accumulation_steps 1
    --num_train_epochs 50
    --learning_rate 1.5e-4
    --weight_decay 0.05
    --mask_ratio 0.75
    --warmup_ratio 0.05
    --report_to wandb
    --save_steps 0.1
    --save_total_limit 5
    --seed 42
    --mixed_precision bf16
    --dataloader_num_workers 16
    --dataloader_pin_memory
    --output_dir ./results/models
    --logging_dir ./results/logs
    --wandb_dir ./results/
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --proj_drop 0.1
    --attn_drop 0.1
    --drop_path_rate 0.1
    --init_values 1.0
    --loss_type mse
    --modal_mode optical
    --crop_size 64
    --channel_dropout 0.7 0.8
    --embed_dim 384
    --depth 12
    --num_heads 6
    --patch_size 16
    # Required, not optional: args.py's --in_channels defaults to None, and argparse
    # always populates it, so train.py's SpecViTMAEConfig(**vars(args), ...) would
    # otherwise pass in_channels=None explicitly, overriding SpecViTMAEConfig's own
    # sensible default (120) with None -- a real, never-before-hit bug (no launch
    # script has ever actually run --model_name specvit pretraining before this one).
    --in_channels 120
    --reduced_channels 128
    --decoder_embed_dim 512
    --decoder_depth 8
    --decoder_num_heads 16
    --run_name ${RUN_NAME}
)

mkdir -p "./results/models/${RUN_NAME}"
printf 'accelerate launch GeospatialFM/scripts/train.py %s\n' "${SPECVIT_ARGS[*]}" \
    > "./results/models/${RUN_NAME}/launch_command.sh"

accelerate launch GeospatialFM/scripts/train.py \
    "${SPECVIT_ARGS[@]}"
