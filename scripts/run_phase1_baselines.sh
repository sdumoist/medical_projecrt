#!/bin/bash
# Run all Phase 1 baseline experiments (A1-A6).
# Usage: bash scripts/run_phase1_baselines.sh [--dry-run]
#
# Experiments:
#   A1: G1 DenseNet binary
#   A2: G2 ResNet binary
#   A5: G3 Swin binary
#   L1: G2L ResNet+localizer binary
#   L2: G3L Swin+localizer binary
#
# CoPAS experiments (A3/A4) use scripts/run_copas_ablation.sh

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

# A1: G1 DenseNet binary
run_exp "A1: G1 DenseNet binary" \
    "PYTHONPATH=. $PYTHON train.py --config configs/g1_densenet_binary.yaml"

# A2: G2 ResNet binary
run_exp "A2: G2 ResNet binary" \
    "PYTHONPATH=. $PYTHON train.py --config configs/g2_resnet_binary.yaml"

# A5: G3 Swin Transformer binary
run_exp "A5: G3 Swin binary" \
    "PYTHONPATH=. $PYTHON train.py --config configs/g3_swin_binary.yaml"

# L1: G2L ResNet + localizer binary
run_exp "L1: G2L ResNet+localizer binary" \
    "PYTHONPATH=. $PYTHON train.py --config configs/g2l_resnet_binary.yaml"

# L2: G3L Swin + localizer binary
run_exp "L2: G3L Swin+localizer binary" \
    "PYTHONPATH=. $PYTHON train.py --config configs/g3l_swin_binary.yaml"

echo ""
echo "============================================"
echo "  All baseline experiments complete."
echo "  Run scripts/run_copas_ablation.sh for A3/A4."
echo "  Run scripts/summarize_phase1_results.py for comparison."
echo "============================================"
