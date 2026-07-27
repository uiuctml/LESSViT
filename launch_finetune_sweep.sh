# NOTE: /home/haozhesi/GeospatialFM is a stale, separate checkout that predates EnMAP support
# (its finetune/args.py has no --gen_task/--quad_tile_train) -- every other launch_*.sh script
# in this repo (launch_eval_cross_sensor.sh, launch_finetune_ablation_base.sh, ...) points at
# /home/haozhesi/LESSViT instead; this one must match or --gen_task/--dataset_name enmap_cdl
# fail to parse.
ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=1

# Backbone to finetune. One of: lessvit | specvit | dinov3 | dofa | spatsigma | channelvit | hyperfree
MODEL_NAME=lessvit

# Directory containing the pretrained checkpoint(s) for $MODEL_NAME. See launch_finetune.sh
# for the full explanation of this convention.
PRETRAINED_MODEL_PATH=/project/geospatial/baseline_models/lessvit/LESSVIT_S_b4_d4_r4

DATASET_NAME=enmap_cdl
TASK_TYPE=segmentation
GEN_TASK=id
DATA_DIR=/datasets/geospatial/
# Only needed to compute SRF std stats for the prisma_like/sentinel2_like generalization
# eval below (cached under SRF_CACHE_DIR afterward); irrelevant to training itself.
PRETRAIN_DATA_DIR=/datasets/geospatial/enmap/enmap

# No EnMAP-family dataset's metadata carries a "size" key, so finetune.py's
# `args.crop_size = metadata["size"] if args.crop_size is None else args.crop_size` KeyErrors
# unless --crop_size is passed explicitly. 128 is the native tile size shared by all 5 README
# downstream EnMAP benchmarks (see launch_finetune_ablation_base.sh's identical note, and
# eval_cross_sensor.py/eval_quad_split.py's hardcoded 128/64*2).
CROP_SIZE=128

# Appendix D.1: "a comprehensive search over a wide range of learning rates ... applied to
# all models and benchmark datasets ... we report the best results achieved across all
# learning rates", selected on the validation set. This is a plain grid search, run one full
# fine-tune per candidate LR -- not the same thing as the --use_optuna path in finetune.py,
# which does a TPE search over a different (wider) candidate set instead of this exact grid.
LEARNING_RATES=(8e-5 1e-4 3e-4 5e-4 8e-4)

# finetune.py never overrides Trainer.evaluate()'s default metric_key_prefix, so both
# val_results.json and test_results.json end up with "eval_"-prefixed keys regardless of
# split -- not "val_"/"test_"-prefixed as the filenames might suggest.
case ${TASK_TYPE} in
    classification) METRIC_KEY=eval_accuracy ;;
    multilabel) METRIC_KEY=eval_micro_f1 ;;
    segmentation) METRIC_KEY=eval_IoU ;;
    *) echo "Unknown TASK_TYPE ${TASK_TYPE}"; exit 1 ;;
esac

for LR in "${LEARNING_RATES[@]}"; do
    RUN_NAME=${MODEL_NAME}_${GEN_TASK}_lr${LR}

    python3 GeospatialFM/finetune/finetune.py \
        --dataset_name ${DATASET_NAME} \
        --task_type ${TASK_TYPE} \
        --data_dir ${DATA_DIR} \
        --gen_task ${GEN_TASK} \
        --model_name ${MODEL_NAME} \
        --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
        --run_name ${RUN_NAME} \
        --output_dir ./results/models \
        --logging_dir ./results/logs \
        --wandb_dir ./results/ \
        --report_to wandb \
        --crop_size ${CROP_SIZE} \
        --return_dict \
        --per_device_train_batch_size 128 \
        --gradient_accumulation_steps 2 \
        --per_device_eval_batch_size 128 \
        --num_train_epochs 10 \
        --learning_rate ${LR} \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type cosine \
        --random_rotation \
        --save_strategy epoch \
        --eval_strategy epoch \
        --save_total_limit 5 \
        --seed 42 \
        --mixed_precision bf16 \
        --dataloader_num_workers 16 \
        --dataloader_pin_memory
done

# Pick the LR with the best validation metric (written by finetune.py to each run's
# val_results.json) and report the corresponding test_results.json, matching the paper's
# search-on-val / report-on-test protocol.
DATASET_NAME=${DATASET_NAME} MODEL_NAME=${MODEL_NAME} GEN_TASK=${GEN_TASK} METRIC_KEY=${METRIC_KEY} \
LEARNING_RATES="${LEARNING_RATES[*]}" python3 - <<'PY'
import json
import os

dataset_name = os.environ["DATASET_NAME"]
model_name = os.environ["MODEL_NAME"]
gen_task = os.environ["GEN_TASK"]
metric_key = os.environ["METRIC_KEY"]
lrs = os.environ["LEARNING_RATES"].split()

best_lr, best_score = None, float("-inf")
for lr in lrs:
    run_name = f"{model_name}_{gen_task}_lr{lr}"
    val_path = os.path.join("results", "models", dataset_name, run_name, "val_results.json")
    with open(val_path) as f:
        score = json.load(f)[metric_key]
    print(f"lr={lr} {metric_key}={score:.4f}")
    if score > best_score:
        best_lr, best_score = lr, score

print(f"\nBest learning rate: {best_lr} ({metric_key}={best_score:.4f})")
test_path = os.path.join("results", "models", dataset_name, f"{model_name}_{gen_task}_lr{best_lr}", "test_results.json")
with open(test_path) as f:
    print(f"Test-set results ({test_path}):")
    print(json.dumps(json.load(f), indent=2))
PY

# Take that same best-on-val checkpoint and report it across the full generalization grid --
# no retraining, one eval pass per gen_task: native id/ood_a/ood_full/ood_complement (4),
# SRF-resampled prisma_like/sentinel2_like (2), and (only when DATASET_NAME=enmap_cdl) the real
# desis/eo1h alternative-sensor test sets (2). See eval_generalization.py's module docstring for
# the desis/eo1h class-ordinal caveat. The first sensor-config run computes SRF std stats from
# PRETRAIN_DATA_DIR (slow, one-time); later runs reuse SRF_CACHE_DIR.
SRF_CACHE_DIR=./results/srf_stats
GENERALIZATION_CSV=./results/generalization_eval.csv
N_PATCHES=2000
EVAL_BATCH_SIZE=32

python3 GeospatialFM/finetune/eval_generalization.py \
    --data_dir ${DATA_DIR} \
    --pretrain_data_dir ${PRETRAIN_DATA_DIR} \
    --results_dir ./results \
    --srf_cache_dir ${SRF_CACHE_DIR} \
    --output_csv ${GENERALIZATION_CSV} \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --task_type ${TASK_TYPE} \
    --gen_task_train ${GEN_TASK} \
    --crop_size ${CROP_SIZE} \
    --n_patches ${N_PATCHES} \
    --batch_size ${EVAL_BATCH_SIZE}
