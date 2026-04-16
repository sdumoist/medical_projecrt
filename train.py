"""
Main training script for shoulder MRI 3D classification.
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
from utils.losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss
from utils.metrics import compute_per_disease_metrics


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_exam_list(config):
    """Build list of valid exam IDs."""
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
                print("DEBUG: Missing image: %s" % path)
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


def train_epoch(model, loader, optimizer, criterion, device, mode="train"):
    """Train or validate for one epoch."""
    model.train() if mode == "train" else model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_masks = []

    try:
        from tqdm import tqdm
        iter_ = lambda x: tqdm(x, desc=mode, leave=False)
    except ImportError:
        iter_ = lambda x: x

    for batch in iter_(loader):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        if mode == "train":
            optimizer.zero_grad()

        logits = model(images)

        if logits.dim() == 3:
            loss = criterion(logits.view(-1, 3), labels.view(-1), mask.view(-1))
        else:
            loss = criterion(logits, labels, mask)

        if mode == "train":
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if logits.dim() == 3:
            preds = logits.argmax(dim=2)
        else:
            preds = (torch.sigmoid(logits) > 0.5).long()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    metrics = compute_per_disease_metrics(
        all_labels, all_preds, DISEASES, binary=(logits.dim() != 3)
    )

    avg_loss = total_loss / len(loader)
    metrics["loss"] = avg_loss

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
        # Handle UTF-8 BOM
        with open(metadata_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row['exam_id']
                # Use label_ field (exists in CSV)
                raw_labels = {}
                for d in DISEASES:
                    raw_labels[d] = int(row['label_' + d])
                raw_labels_lookup[eid] = raw_labels
        print("Loaded %d raw_labels" % len(raw_labels_lookup))

    label_mapper = LabelMapper(
        mode="binary" if config['training']['num_classes'] == 2 else "ternary"
    )

    # Create split using raw_labels_lookup
    if raw_labels_lookup:
        from data.label_mapper import create_train_val_split as split_func
        task_mode = "binary" if config['training']['num_classes'] == 2 else "ternary"
        train_ids, val_ids = split_func(exam_ids, raw_labels_lookup, task_mode=task_mode, val_ratio=0.2)
    else:
        # Fallback: random split
        np.random.seed(42)
        np.random.shuffle(exam_ids)
        n_val = int(len(exam_ids) * 0.2)
        train_ids, val_ids = exam_ids[n_val:], exam_ids[:n_val]

    print("Train: %d, Val: %d" % (len(train_ids), len(val_ids)))

    # Create datasets
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
    model = create_model(
        encoder=config['model']['encoder'],
        num_classes=config['training']['num_classes'],
        num_diseases=len(config['data']['diseases']),
        pretrained=config['model']['pretrained'],
        hidden_dim=256,
        dropout=0.3,
        fusion=config['model'].get('fusion', 'copas')
    )
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: %d" % n_params)

    # Loss
    if config['training']['loss'] == 'bce':
        criterion = MaskedBCEWithLogitsLoss()
    else:
        criterion = MaskedCrossEntropyLoss()

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
        # Print progress
        print("[Epoch %d/%d] lr=%.6f" % (epoch + 1, epochs, optimizer.param_groups[0]['lr']))

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, "train")

        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_metrics = train_epoch(model, val_loader, optimizer, criterion, device, "val")

            print("Epoch %d/%d" % (epoch + 1, epochs))
            print("  Train Loss: %.4f" % train_metrics['loss'])
            print("  Val Loss: %.4f" % val_metrics['loss'])

            print("  Per-disease F1:")
            for disease in DISEASES:
                f1 = val_metrics[disease].get('f1', 0)
                print("    %s: %.4f" % (disease, f1))

            avg_f1 = np.mean([val_metrics[d].get('f1', 0) for d in DISEASES])
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                save_path = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
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

    print("Training complete!")
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