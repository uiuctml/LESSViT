ROOT_DIR="/home/haozhesi/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# Backbone to finetune. One of: lessvit | specvit | dinov3 | dofa | spatsigma | channelvit | hyperfree
MODEL_NAME=hyperfree

# Directory containing the pretrained checkpoint(s) for $MODEL_NAME. This convention is
# shared by every backbone (native LESSViT included): each one's checkpoint(s) live under
# their own subdirectory of results/models, and every encoder's load_pretrained_weights
# globs for its file(s) inside the directory it's given. Leave empty to train from scratch.
PRETRAINED_MODEL_PATH=./results/models/${MODEL_NAME}

DATASET_NAME=enmap_cdl
TASK_TYPE=segmentation
GEN_TASK=id
DATA_DIR=/datasets/disk2/geospatial/enmap/enmap

RUN_NAME=${MODEL_NAME}_${GEN_TASK}

# Fine-tuning protocol below follows the paper's Appendix D.1 as closely as a single fixed
# run can: effective batch size 256 (128 * 2 grad-accum steps on 1 GPU -- raise
# per_device_train_batch_size and drop gradient_accumulation_steps if you have more GPUs or
# memory headroom), 10 epochs, AdamW (beta1=0.9, beta2=0.999, weight_decay=1e-2), cosine
# schedule with 20% warmup, bf16, and random flips + rotation (no resize). The paper sweeps
# --learning_rate over {8e-5, 1e-4, 3e-4, 5e-4, 8e-4} per dataset/model and picks the best on
# the validation set; LEARNING_RATE below is a single point in that sweep, not the sweep
# itself -- rerun with different values (or wire up --use_optuna) to reproduce the full search.
LEARNING_RATE=1e-4

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
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 10 \
    --learning_rate ${LEARNING_RATE} \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
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
