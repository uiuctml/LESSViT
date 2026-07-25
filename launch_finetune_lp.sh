ROOT_DIR="/home/haozhesi/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# Linear-probing baseline: freeze the pretrained encoder and train only a linear read-out head
# (see finetune.py's --lp path / GeospatialFM/models/conv_head.py's LinearConvHead) on top of
# it, following the same train-on-id / test-on-oods protocol as full fine-tuning (see
# launch_finetune.sh + launch_eval_cross_sensor.sh). Once this run finishes, evaluate cross-
# sensor generalization with launch_eval_cross_sensor_lp.sh.

# Backbone to linear-probe. One of: lessvit | specvit | dinov3 | dofa | channelvit | hyperfree
# (spatsigma is not supported -- it bundles its task head inside the encoder, see finetune.py's
# compute_encoding)
MODEL_NAME=hyperfree

# Directory containing the pretrained checkpoint(s) for $MODEL_NAME -- same convention as
# launch_finetune.sh, and reused verbatim (frozen, unmodified) by eval_cross_sensor_lp.py.
PRETRAINED_MODEL_PATH=./results/models/${MODEL_NAME}

DATASET_NAME=enmap_cdl
TASK_TYPE=segmentation
GEN_TASK=id
DATA_DIR=/datasets/disk2/geospatial/enmap/enmap

# eval_cross_sensor_lp.py's find_lp_decoder_checkpoint expects this exact
# ${MODEL_NAME}_lp_${GEN_TASK}_lr${LEARNING_RATE} run-name convention (mirroring
# launch_finetune_sweep.sh's naming, since eval_cross_sensor.py's FT counterpart also expects a
# specific lr-suffixed run name rather than launch_finetune.sh's plain one).
LEARNING_RATE=3e-4
RUN_NAME=${MODEL_NAME}_lp_${GEN_TASK}_lr${LEARNING_RATE}

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
    --lp \
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
