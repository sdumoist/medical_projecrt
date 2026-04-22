# Shoulder MRI Multi-Label Classification — Project Overview

## Task
7 shoulder diseases binary classification from 5 MRI sequences.

### Diseases
| ID | Disease | Positive Rate | pos_weight (neg/pos) |
|----|---------|--------------|----------------------|
| 0  | SST     | 83.2%        | 0.20                 |
| 1  | IST     | 26.9%        | 2.72                 |
| 2  | SSC     | 19.1%        | 4.23                 |
| 3  | LHBT    | 23.2%        | 3.31                 |
| 4  | IGHL    | 27.3%        | 2.66                 |
| 5  | RIPI    | 36.2%        | 1.76                 |
| 6  | GHOA    | 27.5%        | 2.63                 |

### MRI Sequences (5 inputs)
1. sagittal_PD
2. coronal_PD
3. axial_PD
4. sagittal_T1WI
5. coronal_T2WI

### Dataset
- Total valid exams: 7847 (after label 2→0 fix: 7847, some labels shifted)
- Train/Val split: ~6279/1568 (stratified by positive ratio bins, seed=42)
- Cache: preprocessed .pt files at `outputs/cache_cls/`, each `[5, 20, 448, 448]`

### Label Policy
- `1` → positive (disease present)
- `0` → negative (disease absent)
- `2` → uncertain → **mapped to 0** (binary training policy)
- `-1` → postop/unmappable → **case dropped**

## Repository Structure
```
project/
├── configs/               # YAML configs for G1/G2 experiments
├── models/                # G1/G2 model code (DenseNet3D, ResNet3D, CoPAS-style fusion)
├── copas/                 # CoPAS reproduction (standalone module)
│   ├── resnet3d.py        # Original CoPAS ResNet3D backbone
│   ├── model.py           # CoPAS_Shoulder model + losses
│   ├── dataloader.py      # Shoulder-adapted data loading
│   └── train.py           # Self-contained training script
├── train.py               # G1/G2 training entry point (config-driven)
├── outputs/
│   ├── cache_cls/         # 7847 preprocessed .pt files
│   ├── experiments/       # G1/G2 experiment results
│   ├── experiments_copas/ # CoPAS experiment results
│   └── workflow/          # This documentation
```

## Hardware
- 8x NVIDIA A800-SXM4-80GB
- CFS shared storage (I/O contention with parallel training)
