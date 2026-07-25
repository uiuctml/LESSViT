#!/usr/bin/env bash
# Arm B: full softmax attention over the flattened (channel, position) token grid
# (paper Appendix E, Algorithm 2), with joint RoPE (see attention_ops.py:
# FullSpatialSpectralAttention). Confirmed by sanity checks not to be memory-bound
# at N=16 (N*C <= 1920 even at full C=120 -- full attention over ~2000 tokens is
# cheap), so no batch-size reduction is needed to match the other arms.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./launch_train_ablation_base.sh

RUN_NAME=LESSVIT_S_ablation_arm_b_full
mkdir -p "./results/models/${RUN_NAME}"
printf 'accelerate launch GeospatialFM/scripts/train.py %s\n' "${ABLATION_COMMON_ARGS[*]} --attn_type full --run_name ${RUN_NAME}" \
    > "./results/models/${RUN_NAME}/launch_command.sh"

accelerate launch GeospatialFM/scripts/train.py \
    "${ABLATION_COMMON_ARGS[@]}" \
    --attn_type full \
    --run_name "${RUN_NAME}"
