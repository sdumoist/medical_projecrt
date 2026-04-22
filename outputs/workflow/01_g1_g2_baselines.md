# Phase 1: G1 DenseNet & G2 ResNet Baselines

## G1: DenseNet-121 (2D baseline)

### Architecture
- 5 independent DenseNet3D-121 encoders (one per MRI sequence)
- 3 PD branches (sag, cor, axi) with CoPlaneAttention + CrossModalAttention
- FinalHead: concatenate branch features → Linear → 7 logits
- This is the project's custom CoPAS-style fusion with DenseNet backbone

### Experiments

#### g1_densenet_binary (DenseNet-121, bs=2)
- Config: `configs/g1_densenet_binary.yaml`
- encoder=densenet121, bs=2, lr=1e-4, max_epochs=50, patience=10
- **Result: 34 epochs, best F1=0.287 at epoch 33**
- Diagnosis: severe underfitting — 6/7 diseases F1 near zero
  - SST: F1=0.911 (only disease learned, 83% positive rate)
  - SSC: F1=0.000, LHBT: F1=0.055, IST: F1=0.046
  - Train/val loss nearly identical (~0.96) → not overfitting, just not learning

#### g1_densenet201_binary (DenseNet-201, bs=4)
- Config: `configs/g1_densenet201_binary.yaml`
- encoder=densenet201 (feature_dim=1920 vs 1024), bs=4, lr=1e-4
- **Result: 30 epochs** (started as ablation comparison)
- Originally tried bs=8, OOM on single 80GB GPU

---

## G2: ResNet-50 (2D baseline)

### Architecture
- Same CoPAS-style fusion as G1 but with ResNet3D-50 backbone
- Note: ResNet-50 layer3/layer4 modified to stride=2 (from original stride=1 dilation)
  to prevent OOM (dilated convolutions keep spatial dims at 56×56)

### Experiments

#### g2_resnet_binary (ResNet-50, bs=2)
- Config: `configs/g2_resnet_binary.yaml`
- encoder=resnet50, bs=2, lr=1e-4, max_epochs=50, patience=10
- **Result: 41 epochs (early stop at 31), best F1=0.516**
- Much better than G1 on rare diseases:
  - SST: F1=0.905, IST: F1=0.417, SSC: F1=0.366
  - IGHL: F1=0.569, RIPI: F1=0.628, GHOA: F1=0.414
- Diagnosis: clear overfitting
  - Train loss: 0.694, Val loss: 1.049 (val/train = 1.51x)
  - Val AUC peaked at epoch 17 (0.73), then declined
  - Train F1: 0.638 vs Val F1: 0.516 (gap 0.12)

---

## G1 vs G2 Comparison (best epoch)

| Metric | G1 DenseNet-121 | G2 ResNet-50 |
|--------|-----------------|--------------|
| Val Macro F1 | 0.287 | **0.516** |
| Val Macro AUC | 0.708 | **0.713** |
| Overfitting | None (underfitting) | Moderate |
| SSC F1 | 0.000 | 0.366 |
| LHBT F1 | 0.055 | 0.313 |
| IGHL F1 | 0.273 | 0.569 |

### Key Takeaways
1. DenseNet-121 capacity insufficient for this task (not a model size issue — architectural)
2. ResNet-50 learns but overfits, especially on rare diseases
3. Both use the project's custom CoPAS-style fusion — same architecture, different backbone
4. Neither approaches the original CoPAS paper's methodology (different ResNet3D, different attention)
