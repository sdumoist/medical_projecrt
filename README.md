# Shoulder MRI Multi-Sequence Experiments

Language: [中文](./README.zh-CN.md) | [English](./README.en.md)

A research framework for multi-sequence shoulder MRI experiments, including disease classification, coarse localization priors, token export, and preparation for downstream multimodal reasoning.

## What this repository does

This project focuses on five-sequence shoulder MRI and supports:

- multi-sequence visual front-end training
- binary and ternary label strategies
- CoPAS-style primary and auxiliary sequence fusion
- optional nnUNet coarse-mask-guided localizer
- medical token export for downstream Qwen-based SFT and RL

## Input MRI sequences

- axial_PD
- coronal_PD
- sagittal_PD
- coronal_T2WI
- sagittal_T1WI

## Target diseases

- SST
- IST
- SSC
- LHBT
- IGHL
- RIPI
- GHOA

## Main experiment tracks

- G1: DenseNet + CoPAS-style fusion
- G2: ResNet(MedicalNet) + CoPAS-style fusion
- G1-L / G2-L: localizer-guided variants with coarse nnUNet masks

## Project structure

```text
project/
├── configs/
├── data/
├── models/
├── utils/
├── scripts/
│   ├── build_cls_cache.py
│   └── build_loc_cache.py
├── train.py
├── infer.py
└── outputs/
```

## Typical workflow

### 1. Build metadata

```bash
python data/build_index.py
```

### 2. Train a model

```bash
python train.py --config configs/g1_densenet_binary.yaml
```

### 3. Run inference

```bash
python infer.py --config configs/g1_densenet_binary.yaml
```

## Notes

* This repository is for research use only.
* Private patient data should not be committed.
* Paths and checkpoints should be configured locally.

## Documentation

* [中文说明](./README.zh-CN.md)
* [English documentation](./README.en.md)
