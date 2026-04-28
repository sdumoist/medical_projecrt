#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIGS=(
  "configs/mricv_medicalnet_resnet18_binary.yaml"
  "configs/mricv_medicalnet_resnet34_binary.yaml"
  "configs/mricv_medicalnet_resnet50_binary.yaml"
  "configs/mricv_videoswin_tiny_binary.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  exp_name="$(${PYTHON_BIN} - <<PY
import yaml
with open('${cfg}') as f:
    print(yaml.safe_load(f)['output']['exp_name'])
PY
)"
  out_dir="outputs_clean/experiments/${exp_name}"
  mkdir -p "${out_dir}"
  echo "[MRI-CV] running ${cfg} -> ${out_dir}"
  "${PYTHON_BIN}" train.py --config "${cfg}" --output "${out_dir}" 2>&1 | tee "${out_dir}/train.log"
done

"${PYTHON_BIN}" scripts/summarize_phase1_results.py \
  --exp_dirs \
  outputs_clean/experiments/mricv_medicalnet_resnet18_binary \
  outputs_clean/experiments/mricv_medicalnet_resnet34_binary \
  outputs_clean/experiments/mricv_medicalnet_resnet50_binary \
  outputs_clean/experiments/mricv_videoswin_tiny_binary \
  --output outputs_clean/mricv_pretrain_screen_summary.json
