ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# Downstream benchmark images (enmap_cdl/, enmap_corine/, ...) live here. The split files
# (<DATA_DIR>/splits/<dataset>/{train,val,test}.txt) don't ship alongside them on this
# filesystem -- they're symlinked in from /datasets/yuxuanwan/splits (world-readable).
DATA_DIR=/datasets/geospatial

# Space-separated lists; leave empty ("") to use eval_effective_rank.py's defaults (all models
# except dinov3 / all 5 README downstream datasets / all 4 native gen_task settings).
# channelvit is excluded here: its load_pretrained_weights() is a no-op in this repo (see
# eval_effective_rank.py's known-limitations notes), so it only ever runs randomly initialized
# -- not a trained checkpoint worth comparing against the others.
MODELS="lessvit dofa specvit hyperfree spatsigma"
DATASETS=""
GEN_TASKS=""

RESULTS_DIR=./results
OUTPUT_CSV=./results/effective_rank.csv
SAMPLE_INDEX_FILE=./results/effective_rank_sample_indices.txt

# MODEL=PATH pairs (space-separated). Baseline checkpoints live under
# /project/geospatial/baseline_models/<name>/ -- each dir is itself the final checkpoint (the
# wrappers' load_pretrained_weights glob for *.pth/*.bin/*.safetensors at that top level;
# checkpoint-* subdirs underneath are just intermediate training snapshots and are ignored).
# channelvit has no entry: its load_pretrained_weights() is a no-op regardless of path (see
# eval_effective_rank.py's NO_PRE_HEAD/known-limitations notes), so it always runs random-init.
PRETRAINED_MODEL_DIR="lessvit=/project/geospatial/baseline_models/lessvit dofa=/project/geospatial/baseline_models/dofa specvit=/project/geospatial/baseline_models/specvit hyperfree=/project/geospatial/baseline_models/hyperfree spatsigma=/project/geospatial/baseline_models/spatsigma"

# Held-out set / eval knobs. Leave N_SAMPLES empty ("") to use the full pooled val+test set
# (~4853 samples across the 5 default datasets).
CROP_SIZE=128
BATCH_SIZE=32
N_SAMPLES=""
SEED=0

# LESSViT architecture -- MUST match whatever checkpoint PRETRAINED_MODEL_DIR points
# lessvit at, or state_dict loading fails with a shape mismatch. Defaults below are ViT-B,
# rank=1, crop_size=128 -- matching /project/geospatial/baseline_models/lessvit (run name
# LESSVIT_b2_d8_r1, see launch_train.sh's --run_name; channel_embed_dims_per_head=2 there is
# the "b2"). That checkpoint predates the "arms experiment" refactor (commit 8e09169), which
# renamed arm A's ("less") internal keys -- LESSWithTaskHead.load_pretrained_encoder()
# (downstream_models.py) transparently remaps the old key layout, so --attn_type less below
# still loads correctly.
EMBED_DIM=768
DEPTH=12
NUM_HEADS=12
CHANNEL_EMBED_DIMS_PER_HEAD=2
RANK=1
ATTN_TYPE=less
INIT_VALUES=1.0

python3 GeospatialFM/finetune/eval_effective_rank.py \
    --data_dir ${DATA_DIR} \
    --results_dir ${RESULTS_DIR} \
    --output_csv ${OUTPUT_CSV} \
    --sample_index_file ${SAMPLE_INDEX_FILE} \
    ${MODELS:+--models ${MODELS}} \
    ${DATASETS:+--datasets ${DATASETS}} \
    ${GEN_TASKS:+--gen_tasks ${GEN_TASKS}} \
    ${PRETRAINED_MODEL_DIR:+--pretrained_model_dir ${PRETRAINED_MODEL_DIR}} \
    --crop_size ${CROP_SIZE} \
    --batch_size ${BATCH_SIZE} \
    ${N_SAMPLES:+--n_samples ${N_SAMPLES}} \
    --seed ${SEED} \
    --embed_dim ${EMBED_DIM} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --channel_embed_dims_per_head ${CHANNEL_EMBED_DIMS_PER_HEAD} \
    --rank ${RANK} \
    --attn_type ${ATTN_TYPE} \
    --init_values ${INIT_VALUES} \
    --use_rope_embed
