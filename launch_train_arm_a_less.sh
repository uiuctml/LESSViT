#!/usr/bin/env bash
# Arm A: existing LESS (Kronecker) attention, re-run at the ablation's 64x64/patch16
# resolution. This is a re-run, not the existing 128px checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./launch_train_ablation_base.sh

RUN_NAME=LESSVIT_S_ablation_arm_a_less
mkdir -p "./results/models/${RUN_NAME}"
printf 'accelerate launch GeospatialFM/scripts/train.py %s\n' "${ABLATION_COMMON_ARGS[*]} --attn_type less --run_name ${RUN_NAME}" \
    > "./results/models/${RUN_NAME}/launch_command.sh"

accelerate launch GeospatialFM/scripts/train.py \
    "${ABLATION_COMMON_ARGS[@]}" \
    --attn_type less \
    --run_name "${RUN_NAME}"
