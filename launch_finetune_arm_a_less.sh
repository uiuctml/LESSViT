#!/usr/bin/env bash
# Fine-tune arm A (LESS attention / Kronecker fusion) on each of the 5 downstream
# benchmarks, sweeping --learning_rate over LEARNING_RATES per dataset (Appendix
# D.1's grid, plain loop -- not Optuna). Sources launch_finetune_ablation_base.sh
# for every flag that must be identical across arms/datasets/LRs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./launch_finetune_ablation_base.sh

ATTN_TYPE=less
PRETRAINED_MODEL_PATH=./results/models/LESSVIT_S_ablation_arm_a_less/checkpoint-20350

for pair in "${FINETUNE_DATASETS[@]}"; do
    DATASET_NAME="${pair%%:*}"
    TASK_TYPE="${pair##*:}"
    for LR in "${LEARNING_RATES[@]}"; do
        RUN_NAME="arm_a_less_${DATASET_NAME}_lr${LR}"
        OUT_DIR="./results/models/${DATASET_NAME}/${RUN_NAME}"
        mkdir -p "${OUT_DIR}"
        printf 'python3 GeospatialFM/finetune/finetune.py %s\n' \
            "${FINETUNE_COMMON_ARGS[*]} --dataset_name ${DATASET_NAME} --task_type ${TASK_TYPE} --attn_type ${ATTN_TYPE} --learning_rate ${LR} --pretrained_model_path ${PRETRAINED_MODEL_PATH} --run_name ${RUN_NAME}" \
            > "${OUT_DIR}/launch_command.sh"

        echo "=== Fine-tuning arm_a_less on ${DATASET_NAME} (${TASK_TYPE}), lr=${LR} ==="
        python3 GeospatialFM/finetune/finetune.py \
            "${FINETUNE_COMMON_ARGS[@]}" \
            --dataset_name "${DATASET_NAME}" \
            --task_type "${TASK_TYPE}" \
            --attn_type "${ATTN_TYPE}" \
            --learning_rate "${LR}" \
            --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
            --run_name "${RUN_NAME}"
    done
done
