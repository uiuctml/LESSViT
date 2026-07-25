ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# Linear-probing counterpart to launch_eval_cross_sensor.sh -- see that script + this one's
# eval_cross_sensor_lp.py for the full explanation. Each (model, dataset) here needs a --lp
# checkpoint already trained by launch_finetune_lp.sh with the matching LEARNING_RATE.

DATA_DIR=/datasets/disk3/geospatial
PRETRAIN_DATA_DIR=/datasets/disk2/geospatial/enmap/enmap

# Root dir containing each backbone's frozen pretrained checkpoint, i.e. <PRETRAINED_DIR>/<model>/
# (same convention as PRETRAINED_MODEL_PATH in launch_finetune_lp.sh).
PRETRAINED_DIR=./results/models

# Space-separated lists; leave empty ("") to use eval_cross_sensor_lp.py's defaults (all
# lp-supported models / the 5 README downstream datasets / all 7 gen_task settings).
MODELS="lessvit"
DATASETS="enmap_cdl"
GEN_TASKS=""

# Must match the LEARNING_RATE used in launch_finetune_lp.sh for the run(s) being evaluated.
LEARNING_RATE=3e-4

RESULTS_DIR=./results
SRF_CACHE_DIR=./results/srf_stats
OUTPUT_CSV=./results/cross_sensor_eval_lp.csv

# See launch_eval_cross_sensor.sh: sample size for compute_target_stats' empirical std, cached
# to SRF_CACHE_DIR after the first run per sensor config.
N_PATCHES=2000

BATCH_SIZE=32
SANITY_TOL=0.1

python3 GeospatialFM/finetune/eval_cross_sensor_lp.py \
    --data_dir ${DATA_DIR} \
    --pretrain_data_dir ${PRETRAIN_DATA_DIR} \
    --pretrained_dir ${PRETRAINED_DIR} \
    --results_dir ${RESULTS_DIR} \
    --srf_cache_dir ${SRF_CACHE_DIR} \
    --output_csv ${OUTPUT_CSV} \
    --lr ${LEARNING_RATE} \
    ${MODELS:+--models ${MODELS}} \
    ${DATASETS:+--datasets ${DATASETS}} \
    ${GEN_TASKS:+--gen_tasks ${GEN_TASKS}} \
    --n_patches ${N_PATCHES} \
    --batch_size ${BATCH_SIZE} \
    --sanity_tol ${SANITY_TOL}
