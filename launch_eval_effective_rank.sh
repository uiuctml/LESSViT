ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# /datasets/disk3/geospatial/enmap_* (the downstream benchmark images, as opposed to splits/)
# needs read access granted before this can run -- see launch_eval_cross_sensor.sh's note.
DATA_DIR=/datasets/disk3/geospatial

# Space-separated lists; leave empty ("") to use eval_effective_rank.py's defaults (all models
# except dinov3 / all 5 README downstream datasets / all 4 native gen_task settings).
MODELS=""
DATASETS=""
GEN_TASKS=""

RESULTS_DIR=./results
OUTPUT_CSV=./results/effective_rank.csv
SAMPLE_INDEX_FILE=./results/effective_rank_sample_indices.txt

# MODEL=PATH pairs (space-separated), e.g. "lessvit=results/models/foo/checkpoint-1000
# dofa=results/models/dofa". Any model not listed here falls back to
# ${RESULTS_DIR}/models/<model_name>/ and is recorded as n/a (checkpoint_not_found) if that
# doesn't exist -- fill these in as baseline checkpoints become available.
PRETRAINED_MODEL_DIR="lessvit=results/models/LESSVIT_S_ablation_arm_a_less/checkpoint-20350"

# Held-out set / eval knobs. Leave N_SAMPLES empty ("") to use the full pooled val+test set
# (~4853 samples across the 5 default datasets).
CROP_SIZE=64
BATCH_SIZE=32
N_SAMPLES=""
SEED=0

# LESSViT architecture -- MUST match whatever checkpoint PRETRAINED_MODEL_DIR points
# lessvit at, or state_dict loading fails with a shape mismatch. Defaults below match the
# ViT-S checkpoint above (see launch_train_vits.sh); override if you point lessvit elsewhere.
EMBED_DIM=384
DEPTH=12
NUM_HEADS=6
CHANNEL_EMBED_DIMS_PER_HEAD=4
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
