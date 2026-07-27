ROOT_DIR="/home/haozhesi/GeospatialFM"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ViT-S backbone
EMBED_DIM=384
DEPTH=12
NUM_HEADS=6

DECODER_DEPTH=4
EMBED_DIMS=4
RANK=1

accelerate launch GeospatialFM/scripts/train.py \
    --dataset_name enmap \
    --data_dir /datasets/disk2/geospatial/enmap/enmap \
    --enmap_subset 120 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 50 \
    --learning_rate 1.5e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --channel_mask_ratio 0.75 \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --save_steps 0.1 \
    --save_total_limit 5 \
    --seed 42 \
    --mixed_precision bf16 \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory \
    --output_dir ./results/models \
    --logging_dir ./results/logs \
    --wandb_dir ./results/ \
    --run_name LESSVIT_S_b${EMBED_DIMS}_d${DECODER_DEPTH}_r${RANK} \
    --lr_scheduler_type cosine \
    --embed_dim $EMBED_DIM \
    --depth $DEPTH \
    --num_heads $NUM_HEADS \
    --channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_channel_embed_dims_per_head $EMBED_DIMS \
    --decoder_depth $DECODER_DEPTH \
    --max_grad_norm 1.0 \
    --proj_drop 0.1 \
    --attn_drop 0.1 \
    --drop_path_rate 0.1 \
    --loss_type mse \
    --modal_mode optical \
    --scale 1 \
    --crop_size 128 \
    --init_values 1.0 \
    --rank $RANK \
    --use_rope_embed \
    --channel_dropout 0.7 0.8
