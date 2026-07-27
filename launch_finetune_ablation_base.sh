#!/usr/bin/env bash
# Shared base for fine-tuning the three attention-ablation checkpoints (arm_a_less /
# arm_b_full / arm_c_additive) on the downstream EnMAP benchmarks. Sourced by
# launch_finetune_arm_{a,b,c}_*.sh -- every flag that must be identical across arms
# lives here so a diff between the arm scripts shows only --attn_type (and arm C's
# --fusion_init_scale) and --pretrained_model_path.
#
# --crop_size 64 below is the actual fix for a real bug: finetune.py falls back to
# metadata["size"] when --crop_size is omitted, but no enmap-family dataset's
# metadata has a "size" key, so omitting this crashes with KeyError('size'). 64 also
# matches these checkpoints' pretrain resolution -- train-time random-crop-64 keeps
# the same augmentation/distribution as pretrain; the paired quad-split eval script
# (eval_quad_split.py) is what recovers full native-128 coverage at eval time.
#
# --return_dict is also required, not optional, despite defaulting to off in
# args.py: custom_loss_function (finetune/utils.py) does outputs.get("logits"),
# which crashes with AttributeError on the bare tensor the model returns otherwise.
# launch_finetune.sh doesn't pass it (a latent bug there too); sweep_finetune.py
# already knows to (its own --return_dict, sweep_finetune.py:147).
#
# --max_grad_norm 1.0 is likewise required with the transformers version installed
# here: args.py defaults it to None, but Trainer._run_epoch does
# `if self.args.max_grad_norm > 0:` unconditionally, which TypeErrors on None. None
# of launch_finetune.sh/launch_finetune_sweep.sh/sweep_finetune.py set this either
# (same latent bug); launch_train_ablation_base.sh (pretraining) already does.
#
# --quad_tile_train replaces RandomCropAll with systematic 4-quadrant tiling for
# training (QuadTiledDataset, GeospatialFM/data_process/quad_tiled_dataset.py):
# every native tile's 4 fixed 64x64 quadrants become 4 separate training samples,
# so every corner is covered every epoch (not just in expectation over many
# epochs). This is a real cost multiplier -- train set length (and so steps/epoch)
# is 4x what it'd be under RandomCropAll, and --num_train_epochs 10 is kept as-is
# rather than scaled down, so total gradient steps are 4x too. Combined with the
# 5-point LEARNING_RATES sweep below (3 arms x 5 datasets x 5 LRs = 75 runs), this
# is a substantially larger compute commitment than a single-LR/random-crop run --
# confirm GPU availability accordingly before launching launch_finetune_ablation.sh.
set -euo pipefail
ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=3

# Downstream benchmark tiles (as opposed to the pretraining SpectralEarth patches) --
# needs read access granted before this can run, same note as launch_eval_effective_rank.sh.
DATA_DIR=/datasets/disk3/geospatial

# The 5 README downstream datasets, same set eval_effective_rank.py/eval_cross_sensor.py
# already default to, each paired with its task_type.
FINETUNE_DATASETS=(
    "enmap_cdl:segmentation"
    "enmap_corine:multilabel"
    "enmap_eurocrops:segmentation"
    "enmap_bdforet:segmentation"
    "enmap_bnetd:segmentation"
)

# LESSViT architecture -- MUST match the pretrain checkpoints' sizing (see
# launch_train_ablation_base.sh), or load_pretrained_encoder's state_dict copy fails
# with a shape mismatch.
PATCH_SIZE=16
EMBED_DIM=384
DEPTH=12
NUM_HEADS=6
CHANNEL_EMBED_DIMS_PER_HEAD=4
RANK=1
INIT_VALUES=1.0
CROP_SIZE=64

# Fine-tuning protocol, following the paper's Appendix D.1: effective batch size
# 256, 10 epochs, AdamW (beta1=0.9, beta2=0.999, weight_decay=1e-2), cosine schedule
# with 20% warmup, bf16, random flips + rotation, no resize. --learning_rate is
# swept per Appendix D.1 (the paper's exact 5-value grid), NOT included in
# FINETUNE_COMMON_ARGS -- each arm script loops over LEARNING_RATES and appends
# --learning_rate itself, picking the best-on-val checkpoint per (arm,dataset) at
# eval time (see eval_quad_split.py's find_checkpoint). Explicit grid, not Optuna --
# finetune.py's --use_optuna/trainer.hyperparameter_search exists but isn't used
# here; this is a plain loop over fixed LR values instead.
LEARNING_RATES=(8e-5 1e-4 3e-4 5e-4 8e-4)

FINETUNE_COMMON_ARGS=(
    --data_dir "${DATA_DIR}"
    --model_name lessvit
    --patch_size ${PATCH_SIZE}
    --embed_dim ${EMBED_DIM}
    --depth ${DEPTH}
    --num_heads ${NUM_HEADS}
    --channel_embed_dims_per_head ${CHANNEL_EMBED_DIMS_PER_HEAD}
    --rank ${RANK}
    --init_values ${INIT_VALUES}
    --use_rope_embed
    --return_dict
    --crop_size ${CROP_SIZE}
    --quad_tile_train
    --gen_task id
    --output_dir ./results/models
    --logging_dir ./results/logs
    --wandb_dir ./results/
    --report_to wandb
    --per_device_train_batch_size 128
    --gradient_accumulation_steps 2
    --per_device_eval_batch_size 128
    --num_train_epochs 10
    --adam_beta1 0.9
    --adam_beta2 0.999
    --weight_decay 0.01
    --max_grad_norm 1.0
    --warmup_ratio 0.2
    --lr_scheduler_type cosine
    --random_rotation
    --save_strategy epoch
    --eval_strategy epoch
    --save_total_limit 2
    --seed 42
    --mixed_precision bf16
    --dataloader_num_workers 16
    --dataloader_pin_memory
)
