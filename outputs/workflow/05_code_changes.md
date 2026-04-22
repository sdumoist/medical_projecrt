# Code Changes Log

All changes are in `project/copas/` module (CoPAS reproduction).
The original CoPAS-main directory was NOT modified.

## copas/model.py

### 1. Removed external focal_loss dependency
```diff
- from focal_loss.focal_loss import FocalLoss as FL
```

### 2. Hardcoded 12 → self.class_num
All instances of hardcoded `12` (original CoPAS knee diseases) replaced with `self.class_num` (7):
- `mining_conv = nn.Conv3d(1, n_cls, (n_cls, n_cls, n_cls))`
- `multi_view_classifier = nn.Linear(n_cls, n_cls)`
- Loss loop: `range(self.class_num)`
- pos_weights tensor size

### 3. Self-implemented Binary Focal Loss
```python
def Focal_Loss_with_logits(pred, label, pos_weight=None, gamma=2, reduction='mean'):
    ce = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    p = torch.sigmoid(pred)
    p_t = p * label + (1 - p) * (1 - label)
    focal_weight = (1 - p_t) ** gamma
    if pos_weight is not None:
        alpha = pos_weight / (1 + pos_weight)
        alpha_t = label * alpha + (1 - label) * (1 - alpha)
        loss = alpha_t * focal_weight * ce
    else:
        loss = focal_weight * ce
    ...
```

### 4. Added ClassBalancedASL + SoftF1Loss
- `ClassBalancedASL`: effective-number weighting + asymmetric gamma + negative clip
- `SoftF1Loss`: differentiable macro F1
- Initialized in `__init__` when `loss_type='cbasl'`

### 5. Refactored criterion()
- Branch losses: always weighted BCE (unchanged)
- Final head: switches on `self.loss_type`
  - `'original'`: per-disease binary focal loss
  - `'cbasl'`: CB-ASL + 0.2 × SoftF1 on full [B, 7]

### 6. Device placement for DataParallel
- `self.device_list = ["cpu"] * 3` (all init on CPU)
- External `.cuda()` + `nn.DataParallel` handles multi-GPU
- Forward accepts stacked tensor `[B, 5, C, Z, H, W]`
- `criterion` uses `device=label.device` instead of `.cuda()`

---

## copas/dataloader.py

### 1. Label 2 → 0 mapping
```python
# binary policy: uncertain (2) -> negative (0)
if val == 2:
    val = 0
```

---

## copas/train.py

### 1. Multi-GPU DataParallel support
```python
if n_gpus > 1:
    model = nn.DataParallel(model)
raw_model = model.module if isinstance(model, nn.DataParallel) else model
```

### 2. CLI arguments added
- `--loss_type`: 'original' | 'cbasl'
- `--gpu`, `--prefix`, etc.

### 3. Per-disease optimal threshold
```python
def find_optimal_threshold(trues, preds):
    # Sweep 0.05-0.95, maximize F1
    ...
```
Reports both `f1` (threshold=0.5) and `opt_f1` (best threshold) per disease.

### 4. Robust evaluate_prediction
- `label.astype(int)` + `np.where(label==2, 0, label)` safety guard
- Per-disease try/except for sklearn edge cases
- Macro computed from per-disease aggregation (avoids sklearn 2D issues)

### 5. Experiment naming
- Format: `{prefix}_d{depth}_bs{batch_size}_{loss_type}_{date}_{time}`
- Logs include loss_type in config info

---

## configs/g1_densenet201_binary.yaml (NEW)
- Copy of g1_densenet_binary.yaml with `encoder: "densenet201"`
- batch_size=4, patience=15
