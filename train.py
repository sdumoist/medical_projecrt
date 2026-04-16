"""
Main training script for shoulder MRI 3D classification.
Adapted for ShoulderCoPASModel with 3 PD branches + final head.
"""
import os
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, DISEASES
from utils.io import list_exam_ids, load_json_label
from data.label_mapper import LabelMapper
from data.shoulder_dataset import ShoulderDataset3D
from data.json_parser import JSONParser
from models import create_model
from utils.losses import MaskedBCEWithLogitsLoss
from utils.metrics import compute_per_disease_metrics


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_exam_list(config):
    data_root = config['data']['data_root']
    json_root = config['data']['json_root']

    all_exams = list_exam_ids()
    print("DEBUG: list_exam_ids returned %d exams from DATA_ROOT" % len(all_exams))

    valid_exams = []
    for eid in all_exams:
        complete = True
        for seq in config['data']['sequences']:
            path = os.path.join(data_root, eid, "%s.nii.gz" % seq)
            if not os.path.exists(path):
                complete = False
                break

        json_path = os.path.join(json_root, "%s.json" % eid)
        if complete and os.path.exists(json_path):
            try:
                data = load_json_label(eid)
                if 'labels' in data:
                    valid_exams.append(eid)
            except:
                pass

    print("Found %d/%d valid cases" % (len(valid_exams), len(all_exams)))
    return valid_exams


def compute_total_loss(output, labels, mask, criterion, branch_alpha):
    """Compute multi-branch loss.

    Loss = alpha * (sag_loss + cor_loss + axi_loss) + final_loss

    Args:
        output: dict from ShoulderCoPASModel.forward()
        labels: [B, num_diseases]
        mask:   [B, num_diseases]
        criterion: MaskedBCEWithLogitsLoss
        branch_alpha: weight for branch losses

    Returns:
        total_loss, loss_dict
    """
    final_loss = criterion(output['final_logits'], labels, mask)
    sag_loss = criterion(output['sag_logits'], labels, mask)
    cor_loss = criterion(output['cor_logits'], labels, mask)
    axi_loss = criterion(output['axi_logits'], labels, mask)

    branch_loss = sag_loss + cor_loss + axi_loss
    total_loss = branch_alpha * branch_loss + final_loss

    loss_dict = {
        'total': total_loss.item(),
        'final': final_loss.item(),
        'sag': sag_loss.item(),
        'cor': cor_loss.item(),
        'axi': axi_loss.item(),
    }
    return total_loss, loss_dict


def train_epoch(model, loader, optimizer, criterion, device, branch_alpha, mode="train"):
    """Train or validate for one epoch."""
    model.train() if mode == "train" else model.eval()

    total_losses = {'total': 0, 'final': 0, 'sag': 0, 'cor': 0, 'axi': 0}
    all_preds = []
    all_labels = []
    all_masks = []
    num_batches = 0

    try:
        from tqdm import tqdm
        iter_ = lambda x: tqdm(x, desc=mode, leave=False)
    except ImportError:
        iter_ = lambda x: x

    ctx = torch.no_grad() if mode != "train" else torch.enable_grad()
    with ctx:
        for batch in iter_(loader):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            if mode == "train":
                optimizer.zero_grad()

            output = model(images)

            loss, loss_dict = compute_total_loss(
                output, labels, mask, criterion, branch_alpha)

            if mode == "train":
                loss.backward()
                optimizer.step()

            for k in total_losses:
                total_losses[k] += loss_dict[k]
            num_batches += 1

            preds = (torch.sigmoid(output['final_logits']) > 0.5).long()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    metrics = compute_per_disease_metrics(all_labels, all_preds, DISEASES, binary=True)

    for k in total_losses:
        metrics['loss_' + k] = total_losses[k] / max(num_batches, 1)
    metrics['loss'] = metrics['loss_total']

    return metrics


def train(config, output_dir):
    """Main training function."""
    import csv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    set_seed(config.get("seed", 42))

    exam_ids = build_exam_list(config)
    print("Total valid cases: %d" % len(exam_ids))

    # Build raw_labels_lookup from metadata
    metadata_path = os.path.join(config['output']['output_dir'], "../metadata/metadata_master.csv")
    raw_labels_lookup = {}
    if os.path.exists(metadata_path):
        print("Loading raw_labels from metadata...")
        with open(metadata_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row['exam_id']
                raw_labels = {}
                for d in DISEASES:
                    raw_labels[d] = int(row['label_' + d])
                raw_labels_lookup[eid] = raw_labels
        print("Loaded %d raw_labels" % len(raw_labels_lookup))

    label_mapper = LabelMapper(mode="binary")

    # Create split
    if raw_labels_lookup:
        from data.label_mapper import create_train_val_split as split_func
        train_ids, val_ids = split_func(exam_ids, raw_labels_lookup,
                                         task_mode="binary", val_ratio=0.2)
    else:
        np.random.seed(42)
        np.random.shuffle(exam_ids)
        n_val = int(len(exam_ids) * 0.2)
        train_ids, val_ids = exam_ids[n_val:], exam_ids[:n_val]

    print("Train: %d, Val: %d" % (len(train_ids), len(val_ids)))

    crop_size = tuple(config['model']['crop_size'])

    train_dataset = ShoulderDataset3D(
        exam_ids=train_ids,
        data_root=config['data']['data_root'],
        json_root=config['data']['json_root'],
        sequences=config['data']['sequences'],
        label_mapper=label_mapper,
        raw_labels_lookup=raw_labels_lookup,
        crop_size=crop_size,
        mode="train"
    )

    val_dataset = ShoulderDataset3D(
        exam_ids=val_ids,
        data_root=config['data']['data_root'],
        json_root=config['data']['json_root'],
        sequences=config['data']['sequences'],
        label_mapper=label_mapper,
        raw_labels_lookup=raw_labels_lookup,
        crop_size=crop_size,
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Create model
    branch_alpha = config['training'].get('branch_alpha', 0.3)
    model = create_model(
        encoder=config['model']['encoder'],
        num_diseases=len(config['data']['diseases']),
        pretrained=config['model']['pretrained'],
        dropout=config['training'].get('dropout', 0.3),
        num_heads=config['model'].get('num_heads', 4),
        branch_alpha=branch_alpha,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: %d" % n_params)

    criterion = MaskedBCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['max_epochs']
    )

    # Training loop
    best_f1 = 0
    patience_counter = 0
    epochs = config['training']['max_epochs']

    for epoch in range(epochs):
        print("[Epoch %d/%d] lr=%.6f" % (epoch + 1, epochs, optimizer.param_groups[0]['lr']))

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, branch_alpha, "train")

        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_metrics = train_epoch(
                model, val_loader, optimizer, criterion, device, branch_alpha, "val")

            print("Epoch %d/%d" % (epoch + 1, epochs))
            print("  Train Loss: %.4f (final=%.4f sag=%.4f cor=%.4f axi=%.4f)" % (
                train_metrics['loss'], train_metrics['loss_final'],
                train_metrics['loss_sag'], train_metrics['loss_cor'],
                train_metrics['loss_axi']))
            print("  Val   Loss: %.4f (final=%.4f sag=%.4f cor=%.4f axi=%.4f)" % (
                val_metrics['loss'], val_metrics['loss_final'],
                val_metrics['loss_sag'], val_metrics['loss_cor'],
                val_metrics['loss_axi']))

            print("  Per-disease F1:")
            for disease in DISEASES:
                f1 = val_metrics[disease].get('f1', 0)
                print("    %s: %.4f" % (disease, f1))

            avg_f1 = np.mean([val_metrics[d].get('f1', 0) for d in DISEASES])
            print("  Avg F1: %.4f (best: %.4f)" % (avg_f1, best_f1))

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                save_path = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'branch_alpha': branch_alpha,
                }, save_path)
                print("  Saved best model: %s" % save_path)
                patience_counter = 0
            else:
                patience_counter += 1

            patience = config['training'].get('patience', 10)
            if patience_counter >= patience:
                print("Early stopping at epoch %d" % (epoch + 1))
                break

        scheduler.step()

    print("Training complete! Best avg F1: %.4f" % best_f1)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output:
        output_dir = args.output
    else:
        exp_name = config['output']['exp_name']
        output_dir = os.path.join(config['output']['output_dir'], exp_name)

    os.makedirs(output_dir, exist_ok=True)
    print("Output directory: %s" % output_dir)

    model = train(config, output_dir)


if __name__ == "__main__":
    main()
