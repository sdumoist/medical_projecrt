# Training Commands Reference

## CoPAS Experiments

### Recommended (Round 2 settings)
```bash
cd /mnt/cfs_algo_bj/models/experiments/lirunze/code/project

# CoPAS original loss (3 GPUs)
CUDA_VISIBLE_DEVICES=1,3,4 PYTHONPATH=. python copas/train.py \
    --batch_size 12 --num_workers 8 --loss_type original --prefix CoPAS_orig --lr 5e-5

# CoPAS CB-ASL loss (3 GPUs)
CUDA_VISIBLE_DEVICES=5,6,7 PYTHONPATH=. python copas/train.py \
    --batch_size 12 --num_workers 8 --loss_type cbasl --prefix CoPAS_cbasl --lr 5e-5
```

### NOT recommended (Round 1 — lr too high)
```bash
# DO NOT USE: bs=120 + lr=3e-4 causes "predict all positive"
--batch_size 120 --lr 3e-4
```

### Debug mode
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python copas/train.py --debug
```

---

## G1/G2 Experiments

### G1 DenseNet-201 (ablation baseline)
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train.py \
    --config configs/g1_densenet201_binary.yaml
```

### G1 DenseNet-121 (original)
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train.py \
    --config configs/g1_densenet_binary.yaml
```

### G2 ResNet-50
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train.py \
    --config configs/g2_resnet_binary.yaml
```

---

## GPU Assignment Convention

| GPU | Typical Use |
|-----|-------------|
| 0   | G1/G2 baselines (single GPU) |
| 1,3,4 | CoPAS experiment A (DataParallel) |
| 5,6,7 | CoPAS experiment B (DataParallel) |
| 2   | Spare / preprocessing |

---

## Git Workflow
```bash
cd /mnt/cfs_algo_bj/models/experiments/lirunze/code/project
git add <files>
git commit -m "description"
git push
```

---

## Key Hyperparameters

| Parameter | Original CoPAS | Our Setting | Notes |
|-----------|---------------|-------------|-------|
| backbone | ResNet3D-18 | ResNet3D-18 | Same |
| batch_size | 2 | 12 | 3 GPUs × 4/GPU |
| lr | 5e-5 | 5e-5 | Same |
| optimizer | Adam | Adam | Same |
| scheduler | ExponentialLR γ=0.9 | Same, step every 10 epochs | Same |
| epochs | 100 | 100 | Same |
| patience | 30 | 30 | Early stop on val loss |
| α (branch weight) | 0.1 | 0.1 | Same |
| focal γ | 2 | 2 | Same |
| diseases | 12 (knee) | 7 (shoulder) | Adapted |
