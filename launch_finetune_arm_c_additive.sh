#!/usr/bin/env bash
# Fine-tune arm C (additive project-then-broadcast-add fusion) on each of the 5
# downstream benchmarks, sweeping --learning_rate over LEARNING_RATES per dataset.
# See launch_finetune_arm_a_less.sh for the shared structure/comments.
# --fusion_init_scale only affects random init and is irrelevant once
# load_pretrained_encoder restores the trained P_c/P_s weights, but it's passed
# anyway to keep the launched config self-documenting/consistent with pretraining.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./launch_finetune_ablation_base.sh

ATTN_TYPE=additive
FUSION_INIT_SCALE=0.1323
PRETRAINED_MODEL_PATH=./results/models/LESSVIT_S_ablation_arm_c_additive/checkpoint-20350

for pair in "${FINETUNE_DATASETS[@]}"; do
    DATASET_NAME="${pair%%:*}"
    TASK_TYPE="${pair##*:}"
    for LR in "${LEARNING_RATES[@]}"; do
        RUN_NAME="arm_c_additive_${DATASET_NAME}_lr${LR}"
        OUT_DIR="./results/models/${DATASET_NAME}/${RUN_NAME}"
        mkdir -p "${OUT_DIR}"
        printf 'python3 GeospatialFM/finetune/finetune.py %s\n' \
            "${FINETUNE_COMMON_ARGS[*]} --dataset_name ${DATASET_NAME} --task_type ${TASK_TYPE} --attn_type ${ATTN_TYPE} --learning_rate ${LR} --fusion_init_scale ${FUSION_INIT_SCALE} --pretrained_model_path ${PRETRAINED_MODEL_PATH} --run_name ${RUN_NAME}" \
            > "${OUT_DIR}/launch_command.sh"

        echo "=== Fine-tuning arm_c_additive on ${DATASET_NAME} (${TASK_TYPE}), lr=${LR} ==="
        python3 GeospatialFM/finetune/finetune.py \
            "${FINETUNE_COMMON_ARGS[@]}" \
            --dataset_name "${DATASET_NAME}" \
            --task_type "${TASK_TYPE}" \
            --attn_type "${ATTN_TYPE}" \
            --learning_rate "${LR}" \
            --fusion_init_scale "${FUSION_INIT_SCALE}" \
            --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
            --run_name "${RUN_NAME}"
    done
done
