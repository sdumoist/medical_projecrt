#!/bin/bash
# Run CoPAS ablation experiments (A3: original, A4: cbasl).
# Usage: bash scripts/run_copas_ablation.sh [--dry-run]

set -e

PROJECT_ROOT="/mnt/cfs_algo_bj/models/experiments/lirunze/code/project"
PYTHON="/root/miniforge3/envs/srre/bin/python"
cd "$PROJECT_ROOT"

DRY_RUN=0
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    echo "[DRY RUN] Commands will be printed but not executed."
fi

run_exp() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "============================================"
    echo "  $name"
    echo "============================================"
    echo "CMD: $cmd"
    if [ "$DRY_RUN" -eq 0 ]; then
        eval "$cmd"
    fi
}

# A3: CoPAS original
run_exp "A3: CoPAS original" \
    "PYTHONPATH=. $PYTHON copas/train.py --loss_type original --batch_size 12 --prefix CoPAS_orig"

# A4: CoPAS cbasl
run_exp "A4: CoPAS cbasl" \
    "PYTHONPATH=. $PYTHON copas/train.py --loss_type cbasl --batch_size 12 --prefix CoPAS_cbasl"

echo ""
echo "============================================"
echo "  CoPAS ablation complete."
echo "  Run scripts/summarize_phase1_results.py for comparison."
echo "============================================"
