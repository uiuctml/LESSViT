#!/usr/bin/env bash
# Orchestrator for the three-arm attention ablation. Runs all three arms
# sequentially, each using all 4 GPUs via accelerate/DDP -- the same pattern as
# the existing launch_train_vits.sh. Sequential-by-default because arm B's
# per-GPU memory headroom was unknown until profiled; the ablation's own sanity
# checks (ablation_sanity_checks.py, check 3) found arm B is in fact cheap at
# N=16 (peak ~6GB even at full C=120), so one-arm-per-GPU parallel execution with
# adjusted grad-accum is a valid follow-up once you've re-confirmed that holds at
# your actual batch size -- not switched to automatically here, since batch size
# must stay identical across arms for the comparison to be valid and that
# tradeoff is worth a deliberate choice, not a default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Launch mode: sequential, all 4 GPUs per arm (see header comment for why) ==="

for arm in a_less b_full c_additive; do
    script="./launch_train_arm_${arm}.sh"
    echo ""
    echo "=== Starting arm: ${arm} ($(date)) ==="
    start_ts=$(date +%s)
    bash "${script}"
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    gpu_hours=$(python3 -c "print(f'{${elapsed} * 4 / 3600:.2f}')")
    echo "=== Finished arm: ${arm} in ${elapsed}s wall clock (~${gpu_hours} GPU-hours across 4 GPUs) ==="
done

echo ""
echo "=== All three arms complete. Checkpoints under ./results/models/LESSVIT_S_ablation_arm_*/ ==="
