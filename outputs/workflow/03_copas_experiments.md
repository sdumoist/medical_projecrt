# Phase 3: CoPAS Experiments & Results

## Experiment Timeline

### Round 1: bs=120, lr=3e-4 (2026-04-21)
**Problem: lr too high, models stuck in "predict all positive" mode**

#### CoPAS_orig_d18_bs120_original_0421_151502
- GPUs: 1,3,4 (DataParallel)
- Config: bs=120, lr=3e-4, loss_type=original
- Ran 42 epochs
- Val Macro F1 barely moved: 0.487 → 0.495
- Val Macro AUC: 0.518 → 0.707 (ranking learned, but threshold broken)
- All diseases: recall=1.0, precision=class_prevalence → predicting all positive
- Train/Val loss ratio: ~9x (massive overfitting)

#### CoPAS_cbasl_d18_bs120_cbasl_0421_151524
- GPUs: 5,6,7 (DataParallel)
- Config: bs=120, lr=3e-4, loss_type=cbasl
- Ran 36 epochs (early stopped)
- Val Macro F1: 0.4874 constant (never moved at all)
- Val Macro AUC: 0.572 → 0.725 (better ranking than orig)
- CB-ASL's negative clip actually made it HARDER to escape "all positive"

#### Round 1 Comparison

| Metric | orig (bs=120) | cbasl (bs=120) |
|--------|---------------|----------------|
| Val Macro F1 | 0.495 | 0.487 |
| Val Macro AUC | **0.707** | **0.725** |
| Behavior | Tiny F1 movement | Completely stuck |
| Best AUC epoch | 15 | 14 |

**Diagnosis**: lr=3e-4 too aggressive. Sqrt scaling rule (5e-5 × √60 ≈ 3.9e-4) didn't work for this model. 53 gradient updates/epoch insufficient. Both models predict all positive.

---

### Round 2: bs=12, lr=5e-5 (2026-04-22)
**Fix: return to original CoPAS hyperparameters with multi-GPU speedup**

#### CoPAS_orig_d18_bs12_original_0422_035416
- GPUs: 1,3,4
- Config: bs=12, lr=5e-5, loss_type=original
- 524 gradient updates/epoch (10x more than Round 1)
- **Status: running** (17 epochs as of last check)

#### CoPAS_cbasl_d18_bs12_cbasl_0422_035451
- GPUs: 5,6,7
- Config: bs=12, lr=5e-5, loss_type=cbasl
- **Status: running** (17 epochs as of last check)

#### CoPAS_orig_d18_bs12_original_0422_061202
- Restarted run (0 epochs logged at last check)

---

## Bugs Encountered & Fixed

### 1. `focal_loss_torch` ModuleNotFoundError
- **Cause**: external library not installed on cluster
- **Fix**: `pip install focal_loss_torch`
- **Better fix**: replaced with self-implemented binary focal loss (no dependency)

### 2. `focal_loss_torch` CUDA device-side assert
- **Cause**: library expects integer class labels [B] for multi-class, got [B,1] float for binary
- **Fix**: self-implemented binary focal loss handles [B,1] float labels natively

### 3. DataParallel `.criterion()` AttributeError
- **Cause**: `nn.DataParallel` only wraps `forward()`, not custom methods
- **Fix**: pass `raw_model = model.module if DataParallel else model`, use `raw_model.criterion()`

### 4. sklearn ValueError in `f1_score`
- **Cause 1**: `label` was float (from `.tolist()`), sklearn type detection confused
- **Cause 2**: `f1_score(2D_label, 2D_pred, average='macro')` incompatible in some sklearn versions
- **Fix**: `label.astype(int)`, compute macro from per-disease aggregation instead

### 5. Label `2` (uncertain) treated as class index
- **Cause**: `copas/dataloader.py` read `raw_label_*` directly, only excluded `-1`
- **Impact**: BCE/focal loss received label=2 as target (should be 0 or 1)
  - Training with wrong supervision
  - Evaluation crash (sklearn sees 3 classes in "binary" labels)
- **Fix**: `if val == 2: val = 0` in dataloader + safety guard in evaluate_prediction

### 6. "Predict all positive" with bs=120
- **Cause**: lr=3e-4 too high for this architecture, 53 steps/epoch too few
- **Evidence**: all diseases recall=1.0, precision=class_prevalence, F1 constant
- **Fix**: reduce to bs=12, lr=5e-5 (original CoPAS values)

## Key Learnings

1. **Large batch ≠ better**: For medical imaging with small datasets (6280 samples),
   more gradient updates per epoch matters more than GPU utilization
2. **lr scaling rules are approximate**: sqrt scaling failed for this multi-branch attention model
3. **Fixed threshold (0.5) can hide learning**: AUC improved while F1 was stuck —
   added per-disease optimal threshold search to evaluation
4. **Multi-GPU correctly done**: DataParallel for speed, not for larger batch
5. **Label mapping must be consistent**: CoPAS line wasn't synced with main project's 2→0 policy
