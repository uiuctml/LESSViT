ROOT_DIR="/home/haozhesi/LESSViT"
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
export CUDA_VISIBLE_DEVICES=0

# Effective rank of *finetuned* encoders (as opposed to launch_eval_effective_rank.sh's frozen
# pretrained backbones): loads each backbone's results/models/enmap_cdl/<model>_id_lr*/
# checkpoint saved by launch_finetune_sweep.sh, keeping only the encoder.* submodule and
# discarding the segmentation decoder -- see eval_effective_rank.py's load_finetuned_encoder().
DATA_DIR=/datasets/geospatial

# Only the 4 backbones with a completed enmap_cdl finetune under results/models/enmap_cdl/ as of
# this writing (checkpoint-50, each run's own best_model_checkpoint per trainer_state.json).
# spatsigma/channelvit have no finetuned checkpoint here (and would be n/a / random-init anyway,
# per launch_eval_effective_rank.sh's identical notes).
MODELS="lessvit dofa specvit hyperfree"

# Single dataset: these checkpoints were finetuned specifically on enmap_cdl (segmentation),
# not across all 5 README datasets like the frozen-backbone eval.
DATASETS="enmap_cdl"

# All 4 native gen_task spectral configs (C120_VNIR+/C120_SWIR+/C202/C82) -- leave empty to use
# eval_effective_rank.py's default (NATIVE_GEN_TASKS, i.e. all 4).
GEN_TASKS=""

# Test split only -- these checkpoints were selected on val (launch_finetune_sweep.sh's
# best-on-val protocol), so val is no longer a clean held-out set for them; unlike the frozen
# backbones (never trained on this data at all, hence val+test there).
SPLITS="test"

RESULTS_DIR=./results
OUTPUT_CSV=./results/effective_rank_finetuned.csv
SAMPLE_INDEX_FILE=./results/effective_rank_finetuned_sample_indices.txt

# MODEL=PATH pairs (space-separated): each backbone's best (highest-val) enmap_cdl checkpoint.
PRETRAINED_MODEL_DIR="lessvit=results/models/enmap_cdl/lessvit_id_lr5e-4/checkpoint-50 dofa=results/models/enmap_cdl/dofa_id_lr5e-4/checkpoint-50 specvit=results/models/enmap_cdl/specvit_id_lr5e-4/checkpoint-50 hyperfree=results/models/enmap_cdl/hyperfree_id_lr5e-4/checkpoint-50"

# Load only encoder.* out of each full downstream (encoder+decoder) checkpoint above, instead
# of routing through each wrapper's load_pretrained_weights() (written for each backbone's
# ORIGINAL released pretrained-checkpoint format, not a finetuned one) -- see
# eval_effective_rank.py's load_finetuned_encoder() docstring.
FINETUNED=true

# Held-out set / eval knobs. Leave N_SAMPLES empty ("") to use the full test split
# (240 samples for enmap_cdl).
CROP_SIZE=128
BATCH_SIZE=32
N_SAMPLES=""
SEED=0

# LESSViT architecture -- MUST match launch_finetune_sweep.sh's overrides for MODEL_NAME=lessvit
# (same ViT-B/rank=1/crop=128 checkpoint as launch_eval_effective_rank.sh; see that script's
# identical note).
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
    ${SPLITS:+--splits ${SPLITS}} \
    ${PRETRAINED_MODEL_DIR:+--pretrained_model_dir ${PRETRAINED_MODEL_DIR}} \
    ${FINETUNED:+--finetuned} \
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
