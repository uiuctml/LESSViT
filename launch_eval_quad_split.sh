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
export CUDA_VISIBLE_DEVICES=1

# Downstream benchmark tiles (as opposed to splits/) -- needs read access granted
# before this can run, same note as launch_eval_effective_rank.sh.
DATA_DIR=/datasets/disk3/geospatial
# EnMAP pretraining data root -- only needed the first time a sensor-config gen_task
# (prisma_like/sentinel2_like) runs, to compute its SRF std stats (cached afterward
# under SRF_CACHE_DIR). Leave empty if only running native gen_tasks (id/ood_*).
PRETRAIN_DATA_DIR=/datasets/disk2/geospatial/enmap/enmap

# Space-separated lists; leave empty ("") to use eval_quad_split.py's defaults: all 3
# arms / all 5 README downstream datasets / DEFAULT_GEN_TASKS = id, ood_a, ood_full,
# ood_complement, prisma_like, sentinel2_like, eo1h_like, desis_like. The last two
# SRF-resample enmap_cdl's OWN imagery onto EO-1 Hyperion's/DESIS's real band centres
# (isolates sensor-spectral-response transfer from geography/crop-mix confounds).
# enmap_identity (sanity-check only) and the real desis_cdl/eo1_cdl datasets (desis/
# eo1h -- superseded by desis_like/eo1h_like, confounded with a geography/crop-mix
# shift) are opt-in only, not in the default -- pass them explicitly in GEN_TASKS if
# wanted. desis/eo1h/desis_like/eo1h_like are only meaningful when DATASETS includes
# enmap_cdl (the only checkpoint whose label space matches); desis_like/eo1h_like
# work identically for any dataset (they just change the input channels), but
# desis/eo1h raise a clear error against any dataset other than enmap_cdl.
ARMS=""
DATASETS=""
# Excludes desis_like and sentinel2_like per explicit request (skip DESIS/Sentinel
# generalization this run) -- keeps native configs + prisma_like + eo1h_like.
GEN_TASKS="id ood_a ood_full ood_complement prisma_like eo1h_like"

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
