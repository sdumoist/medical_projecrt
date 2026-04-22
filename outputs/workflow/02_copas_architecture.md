# Phase 2: CoPAS Original Reproduction

## Motivation
Reproduce the exact CoPAS architecture from Nature Communications 2024 paper,
adapted for shoulder 7-disease classification (original was knee 12-disease).

## Architecture: CoPAS_Shoulder

### Backbone: ResNet3D-18 (BasicBlock)
- 5 independent encoders, NOT shared weights
- layer3: stride=1, dilation=2 (preserves spatial dims)
- layer4: stride=1, dilation=4
- Output: 512-dim features
- Total model: ~170M parameters (59M unique, rest from 5 encoder copies)

### Three PD Branches (sag / cor / axi)
Each branch:
1. **Main encoding**: ResNet3D-18 on PD volume → feature map [B, Z, 512]
2. **Co-Plane Attention**:
   - Transpose other PD volumes to match this view's plane
   - Encode transposed volumes with SAME encoder (no_grad)
   - Q=main, K/V=co-plane features → attention → 512-d vector
   - Purpose: cross-plane anatomical context
3. **Cross-Modal Attention** (sag+cor only):
   - Auxiliary encoder for T2WI (sag branch) or T1WI (cor branch)
   - Gate: concat(PD, aux) → Linear → ReLU → softmax → element-wise multiply
   - Purpose: fuse complementary contrast information
4. **Branch Classifier**: Dropout(0.05) → Linear(512, 7)

### Correlation Mining Fusion
- 3 branch sigmoid outputs → 3D outer product [B, 7, 7, 7]
- Conv3d(1, 7, kernel=(7,7,7)) → learned disease co-occurrence
- Multiply with union probability (sag × cor × axi)
- Output: final_pred [B, 7]

### Loss Design

#### Original loss (--loss_type original)
- Branch loss: weighted BCE per-disease (pos_weight = neg/pos ratio)
- Final loss: Binary Focal Loss (gamma=2, alpha from pos_weight)
- Total: α × (branch_sag + branch_cor + branch_axi) + final_focal
- α = 0.1

#### CB-ASL loss (--loss_type cbasl)
- Branch loss: same weighted BCE (unchanged)
- Final loss: ClassBalancedASL + 0.2 × SoftF1Loss
  - CB weight: w_c = (1-β)/(1-β^n_c), β=0.9999, normalized, clamped ≤5.0
  - Asymmetric: γ_pos=1.0, γ_neg=4.0 (suppress easy negatives)
  - Negative clip=0.05 (ignore very confident negatives)
  - SoftF1: differentiable macro F1 as auxiliary loss (weight=0.2)
- Total: α × (branch_sag + branch_cor + branch_axi) + cb_asl + 0.2 × soft_f1

## Code Changes from Original CoPAS

| Change | Reason |
|--------|--------|
| Hardcoded `12` → `self.class_num` (7) | Shoulder has 7 diseases, not 12 |
| `focal_loss_torch` removed | Multi-class library, wrong for multi-label binary |
| Self-implemented binary focal loss | Pure PyTorch, no dependency |
| Added ClassBalancedASL + SoftF1Loss | Alternative loss for long-tail multi-label |
| `--loss_type` CLI flag | Switch between original and cbasl |
| Device placement → CPU init | Needed for DataParallel multi-GPU |
| Forward accepts stacked tensor | DataParallel compatibility |
| Label `2 → 0` mapping in dataloader | Sync with main project's binary policy |
| Per-disease optimal threshold search | Fixed 0.5 threshold misses true performance |

## Files
```
copas/
├── __init__.py         # Module marker
├── resnet3d.py         # Exact copy of original CoPAS ResNet3D
├── model.py            # CoPAS_Shoulder + all losses
├── dataloader.py       # Shoulder-adapted data loading
└── train.py            # Self-contained training entry point
```
