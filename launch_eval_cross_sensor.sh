ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# NOTE: /datasets/disk3/geospatial/enmap_* (the downstream benchmark images, as opposed to
# splits/) are currently owned by yuxuanwan:yuxuanwan with 770 perms -- this needs read access
# granted before either fine-tuning or this script can run. DATA_DIR below is the downstream
# benchmark root (has splits/<dataset>/*.txt + <dataset>/enmap/, <dataset>/<mask_root>/); it is
# NOT the same as the pretraining root (PRETRAIN_DATA_DIR), which only has raw .tif patches.
DATA_DIR=/datasets/disk3/geospatial
PRETRAIN_DATA_DIR=/datasets/disk2/geospatial/enmap/enmap

# Space-separated lists; leave empty ("") to use eval_cross_sensor.py's defaults (all 7 models /
# the 5 README downstream datasets / all 7 gen_task settings).
MODELS="lessvit"
DATASETS="enmap_cdl"
GEN_TASKS=""

RESULTS_DIR=./results
SRF_CACHE_DIR=./results/srf_stats
OUTPUT_CSV=./results/cross_sensor_eval.csv

# Sample size for compute_target_stats' empirical std (see GeospatialFM/datasets/enmap/sensors.py).
# The FIRST run for each sensor config (enmap_identity/prisma_like/sentinel2_like) constructs
# SpectralEarthDataset(root=PRETRAIN_DATA_DIR) to sample from -- on this filesystem, just listing
# its patch directories has been observed to take >30s, so expect the first run to be slow.
# Results are cached to SRF_CACHE_DIR afterward; later runs skip this cost. Lower N_PATCHES (e.g.
# 200) for a faster first pass before committing to the full default of 2000.
N_PATCHES=2000

BATCH_SIZE=32
SANITY_TOL=0.1

python3 GeospatialFM/finetune/eval_cross_sensor.py \
    --data_dir ${DATA_DIR} \
    --pretrain_data_dir ${PRETRAIN_DATA_DIR} \
    --results_dir ${RESULTS_DIR} \
    --srf_cache_dir ${SRF_CACHE_DIR} \
    --output_csv ${OUTPUT_CSV} \
    ${MODELS:+--models ${MODELS}} \
    ${DATASETS:+--datasets ${DATASETS}} \
    ${GEN_TASKS:+--gen_tasks ${GEN_TASKS}} \
    --n_patches ${N_PATCHES} \
    --batch_size ${BATCH_SIZE} \
    --sanity_tol ${SANITY_TOL}
