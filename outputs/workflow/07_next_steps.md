# Next Steps & Open Questions

## Immediate (waiting for Round 2 results)

1. **Wait for CoPAS bs=12 experiments to complete**
   - orig (GPUs 1,3,4) and cbasl (GPUs 5,6,7)
   - Check if models escape "all positive" with lr=5e-5
   - Compare macro_opt_f1 (optimal threshold) between orig and cbasl

2. **Analyze per-disease optimal thresholds**
   - Do thresholds stabilize across epochs?
   - Are they consistent between train and val?
   - Can we use val thresholds for final inference?

## Short-term

3. **Compare all baselines side by side**
   - G1 DenseNet-121 vs G1 DenseNet-201 vs G2 ResNet-50 vs CoPAS orig vs CoPAS cbasl
   - Per-disease F1 with optimal thresholds
   - AUC comparison

4. **Address overfitting if present**
   - Increase dropout (currently 0.05)
   - Add weight decay to Adam
   - Try lr warmup (currently jumps straight to 5e-5)
   - Data augmentation tuning

## Medium-term

5. **Ablation study (for paper)**
   - CoPAS full vs no Co-Plane Attention (`--no_co_att`)
   - CoPAS full vs no Cross-Modal Attention (`--no_cross_modal`)
   - CoPAS full vs no Correlation Mining (`--no_corr_mining`)
   - Single branch vs multi-branch
   - ResNet3D-18 vs ResNet3D-50 backbone

6. **Loss function study**
   - original focal vs CB-ASL vs plain BCE vs weighted BCE
   - SoftF1 weight ablation (0.1, 0.2, 0.5)
   - Per-disease vs global threshold optimization

## Open Questions

- **ClassDistr needs update**: `Config.ClassDistr` still uses pre-label-fix counts.
  After mapping 2→0, some diseases have fewer positives. Should re-count.
- **Stratified split consistency**: with label 2→0, the split might change slightly.
  Current code uses seed=42, but class ratios shifted. Probably negligible.
- **Data augmentation**: current augmentation matches original CoPAS (random crop,
  rotation, contrast, shear). Is it sufficient for shoulder MRI?
- **Warmup**: original CoPAS didn't use warmup, but medical imaging often benefits from it.
- **Mixed precision**: `--half` flag exists but not tested. Could speed up training.
