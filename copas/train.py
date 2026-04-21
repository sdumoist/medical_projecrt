"""
CoPAS Shoulder training script — self-contained entry point.

Reproduces the original CoPAS (Nature Communications 2024) training pipeline
adapted for shoulder MRI with 7 diseases and 5 sequences.

Usage:
    cd /mnt/cfs_algo_bj/models/experiments/lirunze/code/project
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python copas/train.py --batch_size 4
    # Multi-GPU (6 GPUs, total bs=24)
    CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 PYTHONPATH=. python copas/train.py --batch_size 24
    # Debug
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python copas/train.py --debug
"""

import os
import sys
import time
import csv
import json
import shutil
import logging
import argparse
import pickle

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, average_precision_score,
                             hamming_loss, balanced_accuracy_score, confusion_matrix)

from copas.model import CoPAS_Shoulder
from copas.dataloader import (load_metadata, create_split,
                               ShoulderCoPASDataset, DISEASES)

# ============================================================
# Config
# ============================================================
class Config:
    """Training configuration (matches original CoPAS defaults)."""

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_csv = os.path.join(project_root, "outputs/metadata/metadata_master.csv")
    cache_root = os.path.join(project_root, "outputs/cache_cls")
    exp_root = os.path.join(project_root, "outputs/experiments_copas")

    # Task
    DiseaseList = DISEASES
    ClassNum = len(DISEASES)  # 7

    # Data
    INPUT_DIM = 224
    SliceNum = 20
    Keep_slice = True

    # Class distribution: [total, SST, IST, SSC, LHBT, IGHL, RIPI, GHOA]
    ClassDistr = [7847, 6528, 2108, 1500, 1822, 2143, 2840, 2159]

    # Model
    backbone = "ResNet3D"
    model_depth = 18
    pretrain = False
    pretrain_path = ""
    emb_dim = 512
    emb_num = 28
    alpha = 0.1        # branch loss weight

    no_co_att = False
    no_cross_modal = False
    no_corr_mining = False
    separate_final = False
    active_class = [1] * 7
    active_branch = [1, 1, 1]

    # Training
    lr = 5e-5
    epochs = 100
    batch_size = 2
    num_workers = 4
    patience = 30
    val_ratio = 0.2
    seed = 42
    half = False
    data_balance = False
    debug = False
    show_patch_sample = False
    write_metrix = True

    def __init__(self):
        self.cal_class_weight()
        self.parse_args()

    def cal_class_weight(self):
        """pos_weight for BCE: (total - pos) / pos."""
        self.pos_weights = []
        total = self.ClassDistr[0]
        for pos_num in self.ClassDistr[1:]:
            self.pos_weights.append((total - pos_num) / pos_num if pos_num > 0 else 1.0)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="CoPAS Shoulder Training")
        parser.add_argument('--epochs', type=int, default=self.epochs)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--lr', type=float, default=self.lr)
        parser.add_argument('--patience', type=int, default=self.patience)
        parser.add_argument('--num_workers', type=int, default=self.num_workers)
        parser.add_argument('--val_ratio', type=float, default=self.val_ratio)
        parser.add_argument('--seed', type=int, default=self.seed)
        parser.add_argument('--half', action='store_true', default=self.half)
        parser.add_argument('--debug', action='store_true', default=self.debug)
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--prefix', type=str, default='CoPAS')
        parser.add_argument('--loss_type', type=str, default='original',
                            choices=['original', 'cbasl'],
                            help='original = focal+BCE, cbasl = CB-ASL+SoftF1')

        args, _ = parser.parse_known_args()
        for k, v in vars(args).items():
            setattr(self, k, v)

        if self.debug:
            self.epochs = 2
            self.num_workers = 0
            self.prefix = "Debug"


# ============================================================
# Metrics
# ============================================================
def evaluate_prediction(pred_list, label_list, disease_list):
    """Compute per-disease AUC, ACC, F1 and overall metrics."""
    prediction = np.array(pred_list)
    label = np.array(label_list)
    results = {}
    for i, task in enumerate(disease_list):
        preds = prediction[:, i]
        trues = label[:, i].astype(int)
        binary = (preds >= 0.5).astype(int)
        try:
            auc = roc_auc_score(trues, preds)
        except:
            auc = -1
        try:
            acc = accuracy_score(trues, binary)
            f1 = f1_score(trues, binary, zero_division=0, pos_label=1)
            prec = precision_score(trues, binary, zero_division=0, pos_label=1)
            rec = recall_score(trues, binary, zero_division=0, pos_label=1)
        except ValueError:
            acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
        results[task] = {'auc': auc, 'acc': acc, 'f1': f1,
                         'precision': prec, 'recall': rec}

    # Overall
    binary_all = (prediction >= 0.5).astype(int)
    results['macro_f1'] = f1_score(label, binary_all, average='macro', zero_division=0)
    results['macro_auc'] = np.mean([v['auc'] for v in results.values()
                                     if isinstance(v, dict) and v.get('auc', -1) > 0])
    return results


# ============================================================
# Training
# ============================================================
def train_epoch(model, raw_model, loader, optimizer, scaler, cfg):
    model.train()
    losses = []
    pred_list, label_list = [], []
    final_active = True

    tbar = tqdm(loader, desc="Train", leave=False)
    for i, (images, label, _) in enumerate(tbar):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            images = [img.cuda() for img in images]
            label = label.cuda()

        # Stack for DataParallel: [B, 5, 1, Z, H, W]
        images_stacked = torch.stack(images, dim=1)

        if cfg.half:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images_stacked)
                loss, loss_val = raw_model.criterion(logits, label, act_task=-1, final=final_active)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images_stacked)
            loss, loss_val = raw_model.criterion(logits, label, act_task=-1, final=final_active)
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits[0]).detach().cpu().tolist()
        pred_list.extend(probs)
        label_list.extend(label.cpu().tolist())
        losses.append(loss_val)
        tbar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        if cfg.debug and i >= 2:
            break

    results = evaluate_prediction(pred_list, label_list, cfg.DiseaseList)
    return np.mean(losses), results


def evaluate_epoch(model, raw_model, loader, cfg):
    model.eval()
    losses = []
    pred_list, label_list = [], []

    tbar = tqdm(loader, desc="Val", leave=False)
    for i, (images, label, _) in enumerate(tbar):
        if torch.cuda.is_available():
            images = [img.cuda() for img in images]
            label = label.cuda()

        # Stack for DataParallel: [B, 5, 1, Z, H, W]
        images_stacked = torch.stack(images, dim=1)

        logits = model(images_stacked)
        _, loss_val = raw_model.criterion(logits, label)

        probs = torch.sigmoid(logits[0]).detach().cpu().tolist()
        pred_list.extend(probs)
        label_list.extend(label.cpu().tolist())
        losses.append(loss_val)

        if cfg.debug and i >= 2:
            break

    results = evaluate_prediction(pred_list, label_list, cfg.DiseaseList)
    return np.mean(losses), results


def save_metrics_csv(csv_path, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_metrics_jsonl(jsonl_path, record):
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config()

    # GPU: controlled by CUDA_VISIBLE_DEVICES env var (don't override here)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Experiment folder
    exp_name = f"{cfg.prefix}_d{cfg.model_depth}_bs{cfg.batch_size}_{cfg.loss_type}_{time.strftime('%m%d_%H%M%S')}"
    exp_dir = os.path.join(cfg.exp_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(f"Experiment: {exp_name}")
    logging.info(f"Config: epochs={cfg.epochs}, bs={cfg.batch_size}, lr={cfg.lr}, "
                 f"depth={cfg.model_depth}, patience={cfg.patience}, gpus={n_gpus}, "
                 f"loss_type={cfg.loss_type}")

    # Data
    logging.info("Loading metadata...")
    valid_ids, label_lookup = load_metadata(cfg.metadata_csv, cfg.cache_root)
    train_ids, val_ids = create_split(valid_ids, label_lookup,
                                       val_ratio=cfg.val_ratio, seed=cfg.seed)
    logging.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    train_ds = ShoulderCoPASDataset(train_ids, label_lookup, cfg.cache_root,
                                     input_dim=cfg.INPUT_DIM, slice_num=cfg.SliceNum,
                                     transform=True)
    val_ds = ShoulderCoPASDataset(val_ids, label_lookup, cfg.cache_root,
                                   input_dim=cfg.INPUT_DIM, slice_num=cfg.SliceNum,
                                   transform=False)
    logging.info(f"Train DS: {len(train_ds)}, Val DS: {len(val_ds)}")

    # Model
    logging.info("Creating CoPAS model...")
    model = CoPAS_Shoulder(cfg)
    if torch.cuda.is_available():
        model = model.cuda()
        if n_gpus > 1:
            model = nn.DataParallel(model)
            logging.info(f"Using DataParallel on {n_gpus} GPUs")

    # Unwrapped model reference for criterion / saving
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=False)
    scaler = GradScaler()

    # Training loop
    best_val_auc = 0.0
    best_val_loss = float('inf')
    no_improve = 0
    csv_path = os.path.join(exp_dir, "metrics_epoch.csv")
    jsonl_path = os.path.join(exp_dir, "metrics_epoch.jsonl")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_ds.balance_cls(-1)  # no class balancing (act_task=-1)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers, shuffle=False)

        train_loss, train_metrics = train_epoch(model, raw_model, train_loader, optimizer, scaler, cfg)

        if epoch % 10 == 0 and epoch > 0:
            scheduler.step()

        with torch.no_grad():
            val_loss, val_metrics = evaluate_epoch(model, raw_model, val_loader, cfg)

        elapsed = time.time() - t0

        # Log
        val_macro_f1 = val_metrics.get('macro_f1', 0)
        val_macro_auc = val_metrics.get('macro_auc', 0)
        is_best = val_macro_auc > best_val_auc

        logging.info(
            f"Epoch {epoch+1}/{cfg.epochs} ({elapsed:.0f}s) lr={lr:.6f}\n"
            f"  Train Loss: {train_loss:.4f} | macro_f1={train_metrics.get('macro_f1',0):.4f}\n"
            f"  Val   Loss: {val_loss:.4f} | macro_f1={val_macro_f1:.4f} | macro_auc={val_macro_auc:.4f}\n"
            f"  Per-disease F1/AUC:")
        for d in cfg.DiseaseList:
            dm = val_metrics.get(d, {})
            logging.info(f"    {d}: F1={dm.get('f1',0):.4f}  AUC={dm.get('auc',0):.4f}  "
                         f"Prec={dm.get('precision',0):.4f}  Rec={dm.get('recall',0):.4f}")

        # Save metrics
        row = {
            'epoch': epoch + 1, 'lr': lr,
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_macro_f1': val_macro_f1, 'val_macro_auc': val_macro_auc,
            'best_val_auc': max(best_val_auc, val_macro_auc),
            'is_best': int(is_best),
        }
        for d in cfg.DiseaseList:
            dm = val_metrics.get(d, {})
            row[f'{d}_f1'] = dm.get('f1', 0)
            row[f'{d}_auc'] = dm.get('auc', 0)
            row[f'{d}_recall'] = dm.get('recall', 0)
            row[f'{d}_precision'] = dm.get('precision', 0)
        save_metrics_csv(csv_path, row)
        save_metrics_jsonl(jsonl_path, {
            'epoch': epoch + 1, 'lr': lr,
            'train_loss': float(train_loss), 'val_loss': float(val_loss),
            'train_metrics': {k: v for k, v in train_metrics.items() if isinstance(v, dict)},
            'val_metrics': {k: v for k, v in val_metrics.items() if isinstance(v, dict)},
            'val_macro_f1': val_macro_f1, 'val_macro_auc': val_macro_auc,
        })

        # Save models (use raw_model to avoid DataParallel "module." prefix)
        torch.save(raw_model.state_dict(), os.path.join(exp_dir, "last_model.pt"))
        if is_best:
            best_val_auc = val_macro_auc
            torch.save(raw_model.state_dict(), os.path.join(exp_dir, "best_model.pt"))
            raw_model.save_or_load_encoder_para(path=exp_dir)
            logging.info(f"  ** New best AUC: {best_val_auc:.4f} **")

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    logging.info(f"Training complete! Best val AUC: {best_val_auc:.4f}")
    logging.info(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
