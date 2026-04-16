# Shoulder MRI Multi-Sequence Experiments

[中文](./README.zh-CN.md)

## Overview

This repository is a research-oriented framework for **multi-sequence shoulder MRI** experiments.

Its goal is not limited to final disease classification. Instead, it aims to build a staged pipeline:

**multi-sequence MRI → visual front-end → coarse localization prior → medical tokens → LLM reasoning → structured findings / impression / report**

At the current stage, this repository mainly focuses on:

- multi-sequence visual front-end training
- binary and ternary label strategies
- CoPAS-style primary and auxiliary sequence fusion
- nnUNet coarse-mask-guided key-slice / ROI modeling
- medical token export
- preparation for downstream Qwen-based SFT and RL

This repository mainly covers the **visual front-end and experiment framework**, rather than the final report generation system.

---

## Input MRI Sequences

Each exam is expected to contain five standard sequences:

- axial_PD
- coronal_PD
- sagittal_PD
- coronal_T2WI
- sagittal_T1WI

---

## Target Disease Categories

- **SST**: supraspinatus tendon injury
- **IST**: infraspinatus tendon injury
- **SSC**: subscapularis tendon injury
- **LHBT**: long head of biceps tendon injury
- **IGHL**: inferior glenohumeral ligament / axillary pouch abnormality
- **RIPI**: rotator interval pathology
- **GHOA**: glenohumeral osteoarthritis

---

## Disease-specific Anchor Sequence Prior

The current disease-specific sequence prior is:

| Disease | Abbreviation | Anchor sequence |
|---|---|---|
| Rotator interval pathology | RIPI | sagittal_PD |
| IGHL / axillary pouch abnormality | IGHL | coronal_PD |
| Supraspinatus tendon injury | SST | coronal_PD |
| Glenohumeral degeneration | GHOA | coronal_PD |
| Infraspinatus tendon injury | IST | axial_PD |
| Subscapularis tendon injury | SSC | axial_PD |
| Long head biceps tendon injury | LHBT | coronal_PD |

These priors are used for:

- primary and auxiliary sequence design in fusion
- anchor-sequence nnUNet coarse masks
- downstream structured reasoning supervision

---

## Label Schema

### Raw label values

The structured JSON labels may contain four raw values:

- `1`: explicit positive
- `0`: negative
- `2`: uncertain
- `-1`: postoperative unmappable

### Label status

The JSON also stores finer-grained label status:

- `explicit_positive`
- `explicit_negative`
- `implicit_negative`
- `uncertain`
- `postop_unmappable`

---

## Supported Training Modes

### Binary mode

Mapping rule:

- `1 -> 1`
- `0 -> 0`
- `2 -> 0`
- `-1 -> mask`

### Ternary mode

Mapping rule:

- `0 -> negative`
- `2 -> uncertain`
- `1 -> positive`
- `-1 -> mask`

### Design principle

- `metadata_master.csv` stores only raw labels and raw status
- binary / ternary mapping is applied at runtime by `label_mapper.py`
- training keeps raw labels, train labels, and train masks separately

---

## Data Sources

The repository is designed around three data sources:

### 1. Five-sequence shoulder MRI
One folder per exam, containing all five standard sequences.

### 2. Structured JSON labels
One JSON file per exam, including labels, label_status, evidence_text, negative_evidence, structured_findings, and structured_impression.

### 3. nnUNet coarse masks
Used as weak localization priors rather than final segmentation ground truth in the main pipeline.

Typical uses include:

- key-slice extraction
- ROI bbox extraction
- local branch guidance
- lesion-focused token construction

---

## What metadata_master.csv Does

`metadata_master.csv` is the unified case-level index table of the project.

It links together:

- exam_id
- image paths
- json_path
- raw labels
- label status
- source summary
- nnUNet mask paths
- split information

This avoids repeated directory scanning and makes the pipeline easier to reproduce and debug.

---

## Project Structure

```text
project/
├── configs/
│   ├── g1_densenet_binary.yaml
│   ├── g1_densenet_ternary.yaml
│   ├── g2_resnet_binary.yaml
│   ├── g2_resnet_ternary.yaml
│   ├── g1l_densenet_binary.yaml
│   ├── g1l_densenet_ternary.yaml
│   ├── g2l_resnet_binary.yaml
│   └── g2l_resnet_ternary.yaml
├── data/
│   ├── build_index.py
│   ├── json_parser.py
│   ├── label_mapper.py
│   ├── mask_index.py
│   └── shoulder_dataset.py
├── models/
│   ├── encoders.py
│   ├── fusion_copas.py
│   ├── localizer_branch.py
│   ├── heads.py
│   └── multiseq_model.py
├── utils/
│   ├── io.py
│   ├── losses.py
│   ├── metrics.py
│   ├── export_tokens.py
│   ├── vis.py
│   └── seed.py
├── scripts/
│   ├── build_cls_cache.py
│   └── build_loc_cache.py
├── train.py
├── infer.py
└── outputs/
```

### Directory summary

#### `configs/`

Experiment configs for:

* G1 / G2
* binary / ternary
* localizer-guided and non-localizer runs

#### `data/`

Responsible for:

* metadata construction
* JSON parsing
* label mapping
* mask indexing
* dataset construction

#### `models/`

Responsible for:

* 3D encoders
* CoPAS-style fusion
* classification heads
* localizer branch
* full multi-sequence model wrapper

#### `utils/`

Responsible for:

* NIfTI and mask I/O
* loss functions
* metrics
* token export
* visualization
* random seed utilities

#### `scripts/`

Responsible for:

* classification cache building (build_cls_cache.py)
* localizer cache building (build_loc_cache.py)

#### `outputs/`

Stores:

* metadata
* experiment outputs
* model checkpoints
* exported tokens

---

## Current Experiment Tracks

### G1

**DenseNet + CoPAS-style fusion**

Used as the stable CNN baseline for multi-sequence classification.

### G2

**ResNet(MedicalNet) + CoPAS-style fusion**

Used as the stronger practical baseline with 3D medical pretraining support.

### G1-L / G2-L

Localizer-guided variants using coarse nnUNet mask priors.

---

## Training Workflow

### 1. Build metadata

```bash
python data/build_index.py
```

### 2. Train model

For example, run G1 in binary mode:

```bash
python train.py --config configs/g1_densenet_binary.yaml
```

### 3. Run inference and evaluation

```bash
python infer.py --config configs/g1_densenet_binary.yaml
```

### 4. Export tokens

After training, export:

* sequence-level features
* fused features
* sequence weights
* optional local features
* optional key slices
* optional ROI boxes

These exported tokens are intended for downstream Qwen SFT and RL.

---

## Why Token Export Matters

The visual front-end is not treated as a plain classifier only.

A key design goal is to make it output reusable **medical visual tokens** for downstream structured reasoning.

Typical exported outputs may include:

* `seq_feats`
* `fused_feat`
* `seq_weights`
* optional `local_feats`
* optional `key_slices`
* optional `roi_boxes`

---

## Why Coarse Masks Are Included

Many shoulder diseases are not broad global abnormalities. They are localized structural problems.

Therefore, purely global features may be insufficient.

The nnUNet coarse masks are introduced not as perfect segmentation targets, but as localization priors that help identify:

* which slices are most relevant
* which local regions deserve attention
* which local features should form lesion-aware tokens

This helps move the system beyond black-box classification toward an interpretable diagnostic chain.

---

## Long-term Goal

The long-term objective of this project is to build a full staged shoulder MRI intelligence pipeline:

**multi-sequence MRI → visual front-end → coarse localization prior → medical tokens → Qwen reasoning → structured findings / impression / report**

The current repository mainly covers the visual front-end and tokenization stages.

---

## Planned Extensions

Future directions include:

* stricter multi-label data splitting
* localizer-guided fusion experiments
* Triad encoder support
* Decipher-MR encoder support
* finer key-slice and ROI token export
* Qwen-based structured diagnostic chain SFT
* RL with verifiable intermediate rewards

---

## Usage Notes

This repository is for research use only.

Please note:

* it does not provide medical advice
* private patient data should not be committed
* all paths and checkpoints should be configured locally
* make sure any released data or weights are compliant with local rules and privacy requirements

---

## Contact

For questions, issues, or collaboration, please open a GitHub issue or contact the repository maintainer.
