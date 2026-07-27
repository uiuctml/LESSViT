#!/usr/bin/env bash
# Orchestrator for fine-tuning the three attention-ablation checkpoints across the 5
# downstream benchmarks. Runs all three arms sequentially, each on a single GPU (no
# accelerate/DDP -- matches launch_finetune.sh's existing single-GPU convention),
# each looping internally over its own 5 datasets (see launch_finetune_arm_*.sh).
#
# Scale check before running this: 3 arms x 5 datasets = 15 fine-tuning runs, each
# ~10 epochs -- a real, multi-hour-plus compute commitment on a shared machine, on
# top of the pretraining already done. Don't launch this without confirming GPU
# availability first (same caution as launch_train_ablation.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Launch mode: sequential arms, single GPU each, 5 datasets per arm ==="

for arm in a_less b_full c_additive; do
    script="./launch_finetune_arm_${arm}.sh"
    echo ""
    echo "=== Starting arm: ${arm} ($(date)) ==="
    start_ts=$(date +%s)
    bash "${script}"
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "=== Finished arm: ${arm} in ${elapsed}s wall clock (all 5 datasets) ==="
done

echo ""
echo "=== All three arms fine-tuned on all 5 datasets. Checkpoints under ./results/models/<dataset>/arm_*_<dataset>/ ==="
