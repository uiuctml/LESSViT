#!/usr/bin/env bash
# Arm C: identical channel/spatial branch attentions as arm A (same LowDimPool,
# same low-rank branch widths -- channel_dim/spatial_dim are NOT widened), fused
# by project-then-broadcast-add instead of outer product, projecting each
# branch's per-head output up to head_dim only at the fusion step
# (attention_ops.py: AdditiveSpatialSpectralAttention). This isolates the fusion
# mechanism specifically -- param delta over arm A is ~0.1%, not the ~3x an
# earlier (widened-branch) design produced.
# --fusion_init_scale below was calibrated by
# `python -m GeospatialFM.scripts.ablation_sanity_checks` (check 4): it rescales
# P_c/P_s's init so arm C's post-fusion activation std at step 0 matches arm A's
# within ~10%. Re-run that check if you change embed_dim/num_heads/rank and update
# this value -- it's architecture-specific, not something to leave stale.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./launch_train_ablation_base.sh

FUSION_INIT_SCALE=0.1323

RUN_NAME=LESSVIT_S_ablation_arm_c_additive
mkdir -p "./results/models/${RUN_NAME}"
printf 'accelerate launch GeospatialFM/scripts/train.py %s\n' \
    "${ABLATION_COMMON_ARGS[*]} --attn_type additive --fusion_init_scale ${FUSION_INIT_SCALE} --run_name ${RUN_NAME}" \
    > "./results/models/${RUN_NAME}/launch_command.sh"

accelerate launch GeospatialFM/scripts/train.py \
    "${ABLATION_COMMON_ARGS[@]}" \
    --attn_type additive \
    --fusion_init_scale "${FUSION_INIT_SCALE}" \
    --run_name "${RUN_NAME}"
