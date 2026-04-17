"""
Main training script for shoulder MRI 3D classification.
Adapted for ShoulderCoPASModel with 3 PD branches + final head.

Supports:
  - cache_cls / cache_loc / raw nii data sources
  - optional localizer loss for G1L / G2L
"""
import os
import csv
import yaml
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, DISEASES
from utils.io import list_exam_ids, load_json_label
from data.label_mapper import LabelMapper, create_train_val_split
from data.shoulder_dataset import ShoulderCacheDataset, ShoulderDataset3D
from models import create_model
from utils.losses import MaskedBCEWithLogitsLoss
from utils.metrics import compute_per_disease_metrics


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_labels(config):
    """Load raw_labels_lookup from metadata_master.csv."""
    metadata_path = os.path.join(
        config['output']['output_dir'], "../metadata/metadata_master.csv")
    raw_labels_lookup = {}
    if os.path.exists(metadata_path):
        print("Loading raw_labels from %s ..." % metadata_path)
        with open(metadata_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row['exam_id']
                raw_labels = {}
                for d in DISEASES:
                    raw_labels[d] = int(row['raw_label_' + d])
                raw_labels_lookup[eid] = raw_labels
        print("Loaded %d raw_labels" % len(raw_labels_lookup))
    else:
        warnings.warn("metadata_master.csv not found at %s" % metadata_path)
    return raw_labels_lookup


def build_exam_list(config):
    """Build list of valid exam IDs (for raw mode or split generation)."""
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


def create_dataloaders(config, raw_labels_lookup, project_root):
    """Create train and val DataLoaders based on config."""
    cache_mode = config['data'].get('cache_mode', 'none')
    use_cache = config['data'].get('use_cache', False)
    cache_root = config['data'].get('cache_root', None)

    label_mapper = LabelMapper(mode="binary")
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # Resolve cache_root relative to project_root
    if cache_root and not os.path.isabs(cache_root):
        cache_root = os.path.join(project_root, cache_root)

    if use_cache and cache_mode in ("cls", "loc"):
        # Check cache exists, fallback if not
        index_name = "cache_cls_index.csv" if cache_mode == "cls" else "cache_loc_index.csv"
        index_path = os.path.join(cache_root, index_name) if cache_root else None
        if index_path is None or not os.path.exists(index_path):
            warnings.warn(
                "Cache index not found at %s, falling back to raw nii mode" % index_path)
            use_cache = False

    if use_cache and cache_mode in ("cls", "loc"):
        print("Using cache mode: %s from %s" % (cache_mode, cache_root))

        # Get all exam IDs from raw_labels_lookup for splitting
        all_exam_ids = list(raw_labels_lookup.keys())
        train_ids, val_ids = create_train_val_split(
            all_exam_ids, raw_labels_lookup, task_mode="binary", val_ratio=0.2)
        train_id_set = set(train_ids)
        val_id_set = set(val_ids)

        print("Split: train=%d, val=%d" % (len(train_ids), len(val_ids)))

        train_dataset = ShoulderCacheDataset(
            cache_root=cache_root,
            exam_ids=train_id_set,
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            cache_mode=cache_mode,
            project_root=project_root,
        )
        val_dataset = ShoulderCacheDataset(
            cache_root=cache_root,
            exam_ids=val_id_set,
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            cache_mode=cache_mode,
            project_root=project_root,
        )
    else:
        # Raw NIfTI mode
        print("Using raw NIfTI mode")
        exam_ids = build_exam_list(config)
        crop_size = tuple(config['model'].get('crop_size', [20, 448, 448]))

        if raw_labels_lookup:
            train_ids, val_ids = create_train_val_split(
                exam_ids, raw_labels_lookup, task_mode="binary", val_ratio=0.2)
        else:
            np.random.seed(42)
            np.random.shuffle(exam_ids)
            n_val = int(len(exam_ids) * 0.2)
            train_ids, val_ids = exam_ids[n_val:], exam_ids[:n_val]

        print("Split: train=%d, val=%d" % (len(train_ids), len(val_ids)))

        train_dataset = ShoulderDataset3D(
            exam_ids=train_ids,
            data_root=config['data']['data_root'],
            json_root=config['data']['json_root'],
            sequences=config['data']['sequences'],
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            target_shape=crop_size,
            mode="train",
        )
        val_dataset = ShoulderDataset3D(
            exam_ids=val_ids,
            data_root=config['data']['data_root'],
            json_root=config['data']['json_root'],
            sequences=config['data']['sequences'],
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            target_shape=crop_size,
            mode="val",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def compute_total_loss(output, labels, mask, criterion, branch_alpha,
                       localizer_alpha=0.0, batch=None, use_localizer=False):
    """Compute multi-branch loss with optional localizer loss.

    Loss = final_loss
         + branch_alpha * (sag_loss + cor_loss + axi_loss)
         + localizer_alpha * localizer_loss
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

    # Localizer loss (key-slice prediction)
    if use_localizer and localizer_alpha > 0 and 'slice_logits' in output:
        key_slices = batch.get("key_slices", None)  # [B, 7] in input Z-space
        if key_slices is not None:
            slice_logits = output['slice_logits']  # [B, 7, D']
            B, num_diseases, D = slice_logits.shape
            key_slices = key_slices.to(slice_logits.device).float()

            # Rescale key_slices from input Z-space to feature D'-space
            input_Z = batch["image"].shape[3]  # [B, 5, 1, Z, H, W]
            key_slices_scaled = (key_slices * D / input_Z).long().clamp(0, D - 1)

            # Only supervise where key_slice >= 0 and label mask > 0
            orig_ks = batch.get("key_slices").to(slice_logits.device)
            valid = (orig_ks >= 0) & (mask > 0)

            if valid.any() and D > 1:
                target = key_slices_scaled[valid]  # [N]
                pred = slice_logits[valid]  # [N, D]
                loc_loss = nn.functional.cross_entropy(pred, target)
                total_loss = total_loss + localizer_alpha * loc_loss
                loss_dict['localizer'] = loc_loss.item()
            else:
                loss_dict['localizer'] = 0.0
        else:
            loss_dict['localizer'] = 0.0

    loss_dict['total'] = total_loss.item()
    return total_loss, loss_dict


def train_epoch(model, loader, optimizer, criterion, device, branch_alpha,
                mode="train", localizer_alpha=0.0, use_localizer=False):
    """Train or validate for one epoch."""
    model.train() if mode == "train" else model.eval()

    loss_keys = ['total', 'final', 'sag', 'cor', 'axi']
    if use_localizer:
        loss_keys.append('localizer')
    total_losses = {k: 0.0 for k in loss_keys}
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
            mask_labels = batch["mask"].to(device)

            if mode == "train":
                optimizer.zero_grad()

            # Build model kwargs for localizer
            model_kwargs = {}
            if use_localizer:
                if "key_slices" in batch:
                    model_kwargs["key_slices"] = batch["key_slices"].to(device)
                if "localizer_mask" in batch:
                    model_kwargs["localizer_mask"] = batch["localizer_mask"].to(device)

            output = model(images, **model_kwargs)

            loss, loss_dict = compute_total_loss(
                output, labels, mask_labels, criterion, branch_alpha,
                localizer_alpha=localizer_alpha,
                batch=batch,
                use_localizer=use_localizer,
            )

            if mode == "train":
                loss.backward()
                optimizer.step()

            for k in total_losses:
                total_losses[k] += loss_dict.get(k, 0.0)
            num_batches += 1

            preds = (torch.sigmoid(output['final_logits']) > 0.5).long()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_masks.append(mask_labels.cpu().numpy())

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    set_seed(config.get("seed", 42))

    # Project root (for resolving relative paths)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load raw labels
    raw_labels_lookup = load_raw_labels(config)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(config, raw_labels_lookup, project_root)

    # Localizer settings
    use_localizer = config['model'].get('use_localizer', False)
    localizer_alpha = config['training'].get('localizer_alpha', 0.0)
    if use_localizer:
        print("Localizer enabled: alpha=%.3f" % localizer_alpha)

    # Create model
    branch_alpha = config['training'].get('branch_alpha', 0.3)
    model = create_model(
        encoder=config['model']['encoder'],
        num_diseases=len(config['data']['diseases']),
        pretrained=config['model']['pretrained'],
        dropout=config['training'].get('dropout', 0.3),
        num_heads=config['model'].get('num_heads', 4),
        branch_alpha=branch_alpha,
        use_localizer=use_localizer,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: %d (%.1fM)" % (n_params, n_params / 1e6))

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
            model, train_loader, optimizer, criterion, device, branch_alpha,
            "train", localizer_alpha=localizer_alpha, use_localizer=use_localizer)

        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_metrics = train_epoch(
                model, val_loader, optimizer, criterion, device, branch_alpha,
                "val", localizer_alpha=localizer_alpha, use_localizer=use_localizer)

            print("Epoch %d/%d" % (epoch + 1, epochs))
            loss_str = "  Train Loss: %.4f (final=%.4f sag=%.4f cor=%.4f axi=%.4f" % (
                train_metrics['loss'], train_metrics['loss_final'],
                train_metrics['loss_sag'], train_metrics['loss_cor'],
                train_metrics['loss_axi'])
            if use_localizer:
                loss_str += " loc=%.4f" % train_metrics.get('loss_localizer', 0)
            loss_str += ")"
            print(loss_str)

            loss_str = "  Val   Loss: %.4f (final=%.4f sag=%.4f cor=%.4f axi=%.4f" % (
                val_metrics['loss'], val_metrics['loss_final'],
                val_metrics['loss_sag'], val_metrics['loss_cor'],
                val_metrics['loss_axi'])
            if use_localizer:
                loss_str += " loc=%.4f" % val_metrics.get('loss_localizer', 0)
            loss_str += ")"
            print(loss_str)

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
                    'config': config,
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
