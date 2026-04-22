# All Experiments Summary Table

## Completed Experiments

| Experiment | Model | Backbone | bs | lr | GPUs | Epochs | Val F1 | Val AUC | Status |
|-----------|-------|----------|----|----|------|--------|--------|---------|--------|
| g1_densenet_binary | G1 CoPAS-fusion | DenseNet3D-121 | 2 | 1e-4 | 1 | 34 | 0.287 | 0.708 | Done (underfitting) |
| g2_resnet_binary | G2 CoPAS-fusion | ResNet3D-50 | 2 | 1e-4 | 1 | 41 | 0.516 | 0.713 | Done (overfitting) |
| g1_densenet201_binary | G1 CoPAS-fusion | DenseNet3D-201 | 4 | 1e-4 | 1 | 30 | TBD | TBD | Done |
| CoPAS_orig bs120 | CoPAS original | ResNet3D-18 | 120 | 3e-4 | 3 | 42 | 0.495 | 0.707 | Done (all-positive, lr too high) |
| CoPAS_cbasl bs120 | CoPAS + CB-ASL | ResNet3D-18 | 120 | 3e-4 | 3 | 36 | 0.487 | 0.725 | Done (all-positive, lr too high) |

## Running Experiments (2026-04-22)

| Experiment | Model | Backbone | bs | lr | GPUs | Loss | Status |
|-----------|-------|----------|----|----|------|------|--------|
| CoPAS_orig bs12 | CoPAS original | ResNet3D-18 | 12 | 5e-5 | 1,3,4 | focal+BCE | Running |
| CoPAS_cbasl bs12 | CoPAS + CB-ASL | ResNet3D-18 | 12 | 5e-5 | 5,6,7 | CB-ASL+SoftF1+BCE | Running |
| g1_densenet201 | G1 CoPAS-fusion | DenseNet3D-201 | 4 | 1e-4 | 0 | BCE | Running/Done |

## Planned / Future

| Experiment | Purpose |
|-----------|---------|
| CoPAS bs12 results analysis | Compare orig vs cbasl with correct lr |
| Best model inference | Per-disease optimal thresholds on test set |
| CoPAS + warmup | Try lr warmup to stabilize early training |
| G2 with regularization | Dropout/weight decay to reduce overfitting |

## Experiment Directory Map

```
outputs/
├── experiments/
│   ├── g1_densenet_binary/          # DenseNet-121 bs=2
│   ├── g1_densenet_binary_bs16/     # DenseNet-121 bs=16 (no jsonl)
│   ├── g1_densenet201_binary/       # DenseNet-201 bs=4
│   ├── g2_resnet_binary/            # ResNet-50 bs=2
│   ├── g2_resnet_binary_bs8/        # ResNet-50 bs=8 (OOM, no jsonl)
│   └── g2l_resnet_binary/           # ResNet-50 + localizer (no jsonl)
│
├── experiments_copas/
│   ├── CoPAS_orig_d18_bs120_original_0421_151502/   # Round 1: lr too high
│   ├── CoPAS_cbasl_d18_bs120_cbasl_0421_151524/     # Round 1: lr too high
│   ├── CoPAS_orig_d18_bs12_original_0422_035416/    # Round 2: correct lr ← ACTIVE
│   ├── CoPAS_cbasl_d18_bs12_cbasl_0422_035451/      # Round 2: correct lr ← ACTIVE
│   └── CoPAS_orig_d18_bs12_original_0422_061202/    # Restarted run
```
