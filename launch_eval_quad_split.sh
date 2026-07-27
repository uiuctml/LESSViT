#!/usr/bin/env bash
# Quad-split downstream eval for the 64x64-pretrained attention-ablation checkpoints
# (arm_a_less/arm_b_full/arm_c_additive), after they've been fine-tuned via
# launch_finetune_ablation.sh (or the per-arm launch_finetune_arm_*.sh scripts).
# Splits each native 128x128 tile into its four 64x64 quadrants, evaluates all four,
# and aggregates (mean logits for multilabel, stitched full 128x128 logit map for
# segmentation) -- see eval_quad_split.py's module docstring for why.
#
# Checkpoint *selection* (which of the LR-swept runs to use) is always anchored to
# native "id" validation performance -- GEN_TASKS below only controls which spectral
# configs that selected checkpoint gets *reported* on, matching the intent: pick
# hparams on id, report results on all configs.
ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=0

# Downstream benchmark tiles (as opposed to splits/) -- needs read access granted
# before this can run, same note as launch_eval_effective_rank.sh.
DATA_DIR=/datasets/disk3/geospatial
# EnMAP pretraining data root -- only needed the first time a sensor-config gen_task
# (prisma_like/sentinel2_like) runs, to compute its SRF std stats (cached afterward
# under SRF_CACHE_DIR). Leave empty if only running native gen_tasks (id/ood_*).
PRETRAIN_DATA_DIR=/datasets/disk2/geospatial/enmap/enmap

# Space-separated lists; leave empty ("") to use eval_quad_split.py's defaults (all
# 3 arms / all 5 README downstream datasets / all 9 gen_tasks: id, ood_a, ood_full,
# ood_complement, enmap_identity, prisma_like, sentinel2_like, desis, eo1h). The last
# two (desis/eo1h) evaluate enmap_cdl's checkpoint against desis_cdl's/eo1_cdl's own
# real test data -- only meaningful when DATASETS includes enmap_cdl; requesting them
# against any other dataset raises a clear error (incompatible label spaces).
ARMS=""
DATASETS=""
GEN_TASKS=""

RESULTS_DIR=./results
SRF_CACHE_DIR=./results/srf_stats
OUTPUT_CSV=./results/quad_split_eval.csv

CROP_SIZE=64
BATCH_SIZE=32
N_SAMPLES=""

python3 GeospatialFM/finetune/eval_quad_split.py \
    --data_dir ${DATA_DIR} \
    --pretrain_data_dir ${PRETRAIN_DATA_DIR} \
    --results_dir ${RESULTS_DIR} \
    --srf_cache_dir ${SRF_CACHE_DIR} \
    --output_csv ${OUTPUT_CSV} \
    ${ARMS:+--arms ${ARMS}} \
    ${DATASETS:+--datasets ${DATASETS}} \
    ${GEN_TASKS:+--gen_tasks ${GEN_TASKS}} \
    --crop_size ${CROP_SIZE} \
    --batch_size ${BATCH_SIZE} \
    ${N_SAMPLES:+--n_samples ${N_SAMPLES}}
