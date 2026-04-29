"""
Main training script for shoulder MRI 3D classification.
Adapted for ShoulderCoPASModel with 3 PD branches + final head.

Supports:
  - cache_cls / cache_loc / raw nii data sources
  - optional localizer loss for G1L / G2L
  - binary (num_classes=2) and ternary (num_classes=3) classification
  - DDP multi-GPU training (torchrun)
  - AMP mixed precision (bf16/fp16)
"""
import os
import csv
import json
import yaml
import datetime
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import set_seed, DISEASES
from utils.io import list_exam_ids, load_json_label
from data.label_mapper import LabelMapper, create_train_val_split
from data.shoulder_dataset import ShoulderCacheDataset, ShoulderDataset3D
from models import create_model
from utils.losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss
from utils.metrics import compute_per_disease_metrics


# ── DDP helpers ──────────────────────────────────────────────
def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (local_rank, is_ddp)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, True
    return 0, False


def is_main_process():
    """True on rank 0 or non-DDP."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def summarize_macro_metrics(metrics_dict):
    """Compute macro-averaged metrics across 7 diseases.

    Returns dict with avg_acc, avg_f1, avg_auc, avg_recall, avg_precision,
    avg_opt_f1, avg_opt_precision, avg_opt_recall.
    """
    keys = ['accuracy', 'f1', 'auc', 'recall', 'precision',
            'opt_f1', 'opt_precision', 'opt_recall']
    summary = {}
    for k in keys:
        vals = []
        for d in DISEASES:
            if d in metrics_dict and k in metrics_dict[d]:
                vals.append(metrics_dict[d][k])
        summary['avg_' + k] = float(np.mean(vals)) if vals else 0.0
    return summary


def save_epoch_metrics_csv(csv_path, row_dict):
    """Append one row to metrics CSV. Creates header on first call."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def save_epoch_metrics_jsonl(jsonl_path, epoch_record):
    """Append one JSON line to metrics JSONL file."""
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(epoch_record, ensure_ascii=False) + '\n')


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


def create_dataloaders(config, raw_labels_lookup, project_root, is_ddp=False):
    """Create train and val DataLoaders based on config."""
    cache_mode = config['data'].get('cache_mode', 'none')
    use_cache = config['data'].get('use_cache', False)
    cache_root = config['data'].get('cache_root', None)

    # Determine task mode from num_classes (default binary)
    num_classes = config['training'].get('num_classes', 2)
    task_mode = "ternary" if num_classes == 3 else "binary"

    label_mapper = LabelMapper(mode=task_mode)
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
            all_exam_ids, raw_labels_lookup, task_mode=task_mode, val_ratio=0.2)
        train_id_set = set(train_ids)
        val_id_set = set(val_ids)

        print("Split: train=%d, val=%d" % (len(train_ids), len(val_ids)))

        load_dense_masks = config['data'].get('load_dense_masks', False)
        train_dataset = ShoulderCacheDataset(
            cache_root=cache_root,
            exam_ids=train_id_set,
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            cache_mode=cache_mode,
            project_root=project_root,
            load_dense_masks=load_dense_masks,
        )
        val_dataset = ShoulderCacheDataset(
            cache_root=cache_root,
            exam_ids=val_id_set,
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            cache_mode=cache_mode,
            project_root=project_root,
            load_dense_masks=load_dense_masks,
        )
    else:
        # Raw NIfTI mode
        print("Using raw NIfTI mode")
        exam_ids = build_exam_list(config)
        crop_size = tuple(config['model'].get('crop_size', [20, 448, 448]))

        if raw_labels_lookup:
            train_ids, val_ids = create_train_val_split(
                exam_ids, raw_labels_lookup, task_mode=task_mode, val_ratio=0.2)
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

    # DDP: only train_loader uses DistributedSampler.
    # val_loader always uses the full val_dataset on every rank (rank 0 runs it,
    # others skip via barrier), so no DistributedSampler for val.
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None

    pw = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,           # no DistributedSampler: full set on rank 0
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,        # must be False for val (no sample loss)
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
    )

    return train_loader, val_loader


def compute_total_loss(output, labels, mask, criterion, branch_alpha,
                       localizer_alpha=0.0, batch=None, use_localizer=False,
                       num_classes=2):
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

    # Localizer loss (disease-specific key-slice prediction)
    # slice_logits: [B, 7, D'] from DiseaseSpecificSliceHeads
    # Each disease's logits come from its anchor branch's slice features
    if use_localizer and localizer_alpha > 0 and 'slice_logits' in output:
        key_slices = batch.get("key_slices", None)  # [B, 7] in input Z-space
        if key_slices is not None:
            slice_logits = output['slice_logits']  # [B, 7, D']
            B, num_diseases, D = slice_logits.shape

            orig_ks = key_slices.to(slice_logits.device)          # [B, 7], long
            key_slices_f = orig_ks.float()

            # Rescale key_slices from input Z-space to feature D'-space
            input_Z = batch["image"].shape[3]  # [B, 5, 1, Z, H, W]
            key_slices_scaled = (key_slices_f * D / input_Z).long().clamp(0, D - 1)

            # Clean v1: localizer validity depends only on key_slice existence
            valid = (orig_ks >= 0)

            if valid.any() and D > 1:
                pred = slice_logits[valid]              # [N, D]
                target = key_slices_scaled[valid]       # [N]
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
                mode="train", localizer_alpha=0.0, use_localizer=False,
                num_classes=2, scaler=None, amp_dtype=torch.bfloat16):
    """Train or validate for one epoch."""
    model.train() if mode == "train" else model.eval()

    is_binary = (num_classes == 2)

    loss_keys = ['total', 'final', 'sag', 'cor', 'axi']
    if use_localizer:
        loss_keys.append('localizer')
    total_losses = {k: 0.0 for k in loss_keys}
    all_preds = []
    all_probs = []
    all_labels = []
    all_masks = []
    all_pred_slices = []
    all_gt_slices = []
    all_slice_valid = []
    num_batches = 0

    try:
        from tqdm import tqdm
        iter_ = lambda x: tqdm(x, desc=mode, leave=False)
    except ImportError:
        iter_ = lambda x: x

    ctx = torch.no_grad() if mode != "train" else torch.enable_grad()
    use_amp = (scaler is not None) or (amp_dtype != torch.float32)

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

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                output = model(images, **model_kwargs)

                loss, loss_dict = compute_total_loss(
                    output, labels, mask_labels, criterion, branch_alpha,
                    localizer_alpha=localizer_alpha,
                    batch=batch,
                    use_localizer=use_localizer,
                    num_classes=num_classes,
                )

            # One-time debug for localizer
            if num_batches == 0 and use_localizer and mode == "train":
                ks = batch.get("key_slices")
                print("[DEBUG localizer] batch0:",
                      "has_ks=%s" % (ks is not None),
                      "ks_min/max=%d/%d" % (ks.min().item(), ks.max().item()) if ks is not None else "",
                      "has_slice_logits=%s" % ("slice_logits" in output),
                      "loc_loss=%.4f" % loss_dict.get("localizer", -1),
                      flush=True)

            if mode == "train":
                if scaler is not None:
                    # fp16: use GradScaler to prevent underflow
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # bf16 / no-amp: plain backward (autocast still active above)
                    loss.backward()
                    optimizer.step()

            for k in total_losses:
                total_losses[k] += loss_dict.get(k, 0.0)
            num_batches += 1

            # --- Predictions: binary vs ternary ---
            if is_binary:
                # Binary: sigmoid > 0.5
                preds = (torch.sigmoid(output['final_logits']) > 0.5).long()
                probs = torch.sigmoid(output['final_logits'])
            else:
                # Ternary: argmax over num_classes dim
                # logits shape: [B, num_diseases, num_classes]
                probs = torch.softmax(output['final_logits'], dim=2)
                preds = probs.argmax(dim=2)  # [B, num_diseases]

            all_preds.append(preds.float().cpu().numpy())
            all_probs.append(probs.detach().float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
            all_masks.append(mask_labels.float().cpu().numpy())

            # Collect localizer slice predictions for metrics
            # NOTE: pred_slice_idx is in D' (feature) space.
            #       gt key_slices are in Z (input) space.
            #       We rescale gt to D' space here to match, same as in loss.
            if use_localizer and 'slice_logits' in output:
                slice_logits = output['slice_logits']  # [B, 7, D']
                D_prime = slice_logits.shape[2]
                pred_slice_idx = slice_logits.argmax(dim=2).float().cpu().numpy()  # [B, 7]
                all_pred_slices.append(pred_slice_idx)
                if "key_slices" in batch:
                    gt_ks = batch["key_slices"].numpy()  # [B, 7] in Z space
                    input_Z = batch["image"].shape[3]     # Z dimension of input
                    # Rescale gt from Z space to D' space (same as loss)
                    gt_ks_scaled = np.where(
                        gt_ks >= 0,
                        np.clip((gt_ks * D_prime / input_Z).astype(int), 0, D_prime - 1),
                        gt_ks  # keep -1 for invalid
                    )
                    all_gt_slices.append(gt_ks_scaled)
                    all_slice_valid.append((gt_ks >= 0).astype(np.float32))

    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # For binary: y_prob_all is [N, 7] sigmoid probabilities
    # For ternary: y_prob_all is [N, 7, 3] softmax probabilities
    metrics = compute_per_disease_metrics(
        all_labels, all_preds, DISEASES, binary=is_binary,
        y_prob_all=all_probs,
        mask_all=all_masks)

    for k in total_losses:
        metrics['loss_' + k] = total_losses[k] / max(num_batches, 1)
    metrics['loss'] = metrics['loss_total']

    # Compute key-slice metrics if localizer is active
    if use_localizer and all_pred_slices and all_gt_slices:
        from utils.metrics import compute_key_slice_metrics
        pred_slices = np.concatenate(all_pred_slices, axis=0)
        gt_slices = np.concatenate(all_gt_slices, axis=0)
        slice_valid = np.concatenate(all_slice_valid, axis=0)
        ks_metrics = compute_key_slice_metrics(pred_slices, gt_slices, slice_valid, DISEASES)
        metrics['key_slice'] = ks_metrics

    return metrics


def train(config, output_dir):
    """Main training function."""
    local_rank, is_ddp = setup_ddp()
    device = torch.device("cuda", local_rank) if is_ddp else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print("Using device: %s (DDP=%s, world_size=%s)" % (
            device, is_ddp,
            dist.get_world_size() if is_ddp else 1))

    set_seed(config.get("seed", 42))

    # Project root (for resolving relative paths)
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Load raw labels
    raw_labels_lookup = load_raw_labels(config)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(config, raw_labels_lookup, project_root, is_ddp=is_ddp)

    # Localizer settings
    use_localizer = config['model'].get('use_localizer', False)
    localizer_alpha = config['training'].get('localizer_alpha', 0.0)
    if use_localizer and is_main_process():
        print("Localizer enabled: alpha=%.3f" % localizer_alpha)

    # Task mode
    num_classes = config['training'].get('num_classes', 2)
    is_binary = (num_classes == 2)
    task_mode = "ternary" if num_classes == 3 else "binary"
    if is_main_process():
        print("Task mode: %s (num_classes=%d)" % (task_mode, num_classes))

    # Create model
    branch_alpha = config['training'].get('branch_alpha', 0.3)
    model = create_model(
        encoder=config['model']['encoder'],
        num_diseases=len(config['data']['diseases']),
        pretrained=config['model']['pretrained'],
        pretrained_path=config['model'].get('pretrained_path'),
        dropout=config['training'].get('dropout', 0.3),
        num_heads=config['model'].get('num_heads', 4),
        branch_alpha=branch_alpha,
        use_localizer=use_localizer,
        num_classes=num_classes,
    )
    model = model.to(device)

    # Multi-GPU support
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        if is_main_process():
            print("Using DDP with %d GPUs" % dist.get_world_size())
    elif torch.cuda.device_count() > 1:
        print("Using %d GPUs with DataParallel" % torch.cuda.device_count())
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print("Model parameters: %d (%.1fM)" % (n_params, n_params / 1e6))

    # Loss function
    if is_binary:
        # pos_weight = neg/pos per disease, matches project overview table
        # SST: 0.20, IST: 2.72, SSC: 4.23, LHBT: 3.31, IGHL: 2.66, RIPI: 1.76, GHOA: 2.63
        _pw_values = config['training'].get(
            'pos_weights',
            [0.20, 2.72, 4.23, 3.31, 2.66, 1.76, 2.63]
        )
        pos_weight = torch.tensor(_pw_values, dtype=torch.float32)
        criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        criterion = MaskedCrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['max_epochs']
    )

    # AMP scaler: only for fp16; bf16 uses autocast without GradScaler
    use_amp = config['training'].get('use_amp', True)
    amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    # GradScaler is only needed for fp16; bf16 does not require loss scaling
    scaler = torch.amp.GradScaler('cuda') if (use_amp and amp_dtype == torch.float16) else None
    if is_main_process() and use_amp:
        print("AMP enabled (%s%s)" % (amp_dtype_str,
              ", GradScaler" if scaler is not None else ", no GradScaler"))

    # Training loop
    best_metric = 0
    patience_counter = 0
    epochs = config['training']['max_epochs']
    csv_path = os.path.join(output_dir, "metrics_epoch.csv")
    jsonl_path = os.path.join(output_dir, "metrics_epoch.jsonl")

    for epoch in range(epochs):
        # DDP: set epoch for sampler shuffling
        if is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        cur_lr = optimizer.param_groups[0]['lr']
        if is_main_process():
            print("[Epoch %d/%d] lr=%.6f" % (epoch + 1, epochs, cur_lr))

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, branch_alpha,
            "train", localizer_alpha=localizer_alpha, use_localizer=use_localizer,
            num_classes=num_classes, scaler=scaler, amp_dtype=amp_dtype)

        if (epoch + 1) % config['validation']['val_interval'] == 0:
            # ── Validation: only rank 0 runs full val set ──────────────────
            # Non-rank-0 processes wait at a barrier while rank 0 validates.
            # This guarantees metrics come from the complete val set.
            if is_ddp and not is_main_process():
                dist.barrier()  # wait for rank 0 to finish val
            else:
                # Use raw model (not DDP-wrapped) to avoid triggering all_reduce during val
                raw_model = model.module if is_ddp else model
                val_metrics = train_epoch(
                    raw_model, val_loader, optimizer, criterion, device, branch_alpha,
                    "val", localizer_alpha=localizer_alpha, use_localizer=use_localizer,
                    num_classes=num_classes, scaler=None, amp_dtype=amp_dtype)
                if is_ddp:
                    dist.barrier()  # signal other ranks that val is done

            # ── After barrier: only rank 0 computes metrics and decides stop ─
            if not is_main_process():
                # Receive early-stop decision broadcast from rank 0
                stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    break
                scheduler.step()
                continue

            # --- Macro averages (rank 0 only) ---
            train_macro = summarize_macro_metrics(train_metrics)
            val_macro = summarize_macro_metrics(val_metrics)

            avg_f1 = val_macro['avg_f1']
            cur_metric = val_macro['avg_opt_f1']   # early-stop / best-model on opt_f1
            is_best = cur_metric > best_metric

            # --- Print losses (rank 0 only) ---
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

            # --- Print per-disease metrics ---
            print("  Per-disease metrics (val):")
            print("    %-6s  %6s  %6s  %6s  %6s  %6s  %6s  %6s" % (
                "", "F1", "AUC", "Recall", "Prec", "Acc", "OptF1", "OptThr"))
            for disease in DISEASES:
                dm = val_metrics.get(disease, {})
                print("    %-6s  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %5.2f" % (
                    disease,
                    dm.get('f1', 0),
                    dm.get('auc', 0),
                    dm.get('recall', 0),
                    dm.get('precision', 0),
                    dm.get('accuracy', 0),
                    dm.get('opt_f1', 0),
                    dm.get('opt_thr', 0.5)))

            print("  Avg F1: %.4f  AUC: %.4f  OptF1: %.4f  Recall: %.4f  Prec: %.4f  (best F1: %.4f)" % (
                val_macro['avg_f1'], val_macro['avg_auc'], val_macro['avg_opt_f1'],
                val_macro['avg_recall'], val_macro['avg_precision'], best_metric))

            # Print key-slice metrics if available
            if 'key_slice' in val_metrics:
                ks = val_metrics['key_slice']
                print("  Key-slice: top1=%.4f  pm1=%.4f" % (
                    ks.get('macro_ks_top1', 0), ks.get('macro_ks_pm1', 0)))

            # --- Save CSV row ---
            csv_row = {
                'epoch': epoch + 1,
                'lr': cur_lr,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_avg_acc': train_macro['avg_accuracy'],
                'train_avg_f1': train_macro['avg_f1'],
                'train_avg_auc': train_macro['avg_auc'],
                'train_avg_recall': train_macro['avg_recall'],
                'train_avg_precision': train_macro['avg_precision'],
                'train_avg_opt_f1': train_macro['avg_opt_f1'],
                'val_avg_acc': val_macro['avg_accuracy'],
                'val_avg_f1': val_macro['avg_f1'],
                'val_avg_auc': val_macro['avg_auc'],
                'val_avg_recall': val_macro['avg_recall'],
                'val_avg_precision': val_macro['avg_precision'],
                'val_avg_opt_f1': val_macro['avg_opt_f1'],
                'best_avg_f1': max(best_metric, cur_metric),
                'is_best': int(is_best),
            }
            # Per-disease val F1/AUC/opt in CSV
            for d in DISEASES:
                dm = val_metrics.get(d, {})
                csv_row['%s_f1' % d] = dm.get('f1', 0)
                csv_row['%s_auc' % d] = dm.get('auc', 0)
                csv_row['%s_recall' % d] = dm.get('recall', 0)
                csv_row['%s_precision' % d] = dm.get('precision', 0)
                csv_row['%s_opt_f1' % d] = dm.get('opt_f1', 0)
                csv_row['%s_opt_thr' % d] = dm.get('opt_thr', 0.5)
            # Key-slice metrics in CSV
            if 'key_slice' in val_metrics:
                ks = val_metrics['key_slice']
                csv_row['macro_ks_top1'] = ks.get('macro_ks_top1', 0)
                csv_row['macro_ks_pm1'] = ks.get('macro_ks_pm1', 0)
            save_epoch_metrics_csv(csv_path, csv_row)

            # --- Save JSONL record ---
            jsonl_record = {
                'epoch': epoch + 1,
                'lr': cur_lr,
                'train_loss': {k: train_metrics.get('loss_' + k, 0)
                               for k in ['total', 'final', 'sag', 'cor', 'axi']},
                'val_loss': {k: val_metrics.get('loss_' + k, 0)
                             for k in ['total', 'final', 'sag', 'cor', 'axi']},
                'train_macro': train_macro,
                'val_macro': val_macro,
                'train_per_disease': {d: train_metrics.get(d, {})
                                      for d in DISEASES},
                'val_per_disease': {d: val_metrics.get(d, {})
                                    for d in DISEASES},
                'best_avg_f1': max(best_metric, cur_metric),
                'is_best': is_best,
            }
            if 'key_slice' in val_metrics:
                jsonl_record['val_key_slice'] = val_metrics['key_slice']
            if 'key_slice' in train_metrics:
                jsonl_record['train_key_slice'] = train_metrics['key_slice']
            save_epoch_metrics_jsonl(jsonl_path, jsonl_record)

            # --- Save last_model.pt (every epoch) ---
            raw_model = model.module if hasattr(model, 'module') else model
            last_path = os.path.join(output_dir, "last_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'branch_alpha': branch_alpha,
                'config': config,
            }, last_path)

            # --- Save best_model.pt ---
            if is_best:
                best_metric = cur_metric
                save_path = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'branch_alpha': branch_alpha,
                    'config': config,
                }, save_path)
                print("  Saved best model: %s" % save_path)

                # --- Save best_thresholds.json ---
                thresholds = {}
                for d in DISEASES:
                    dm = val_metrics.get(d, {})
                    thresholds[d] = {
                        'opt_thr': dm.get('opt_thr', 0.5),
                        'opt_f1': dm.get('opt_f1', 0),
                        'opt_precision': dm.get('opt_precision', 0),
                        'opt_recall': dm.get('opt_recall', 0),
                        'auc': dm.get('auc', 0),
                    }
                thresholds['_meta'] = {
                    'epoch': epoch + 1,
                    'val_macro_auc': val_macro['avg_auc'],
                    'val_macro_f1': val_macro['avg_f1'],
                    'val_macro_opt_f1': val_macro['avg_opt_f1'],
                }
                thr_path = os.path.join(output_dir, "best_thresholds.json")
                with open(thr_path, 'w') as f:
                    json.dump(thresholds, f, indent=2)
                print("  Saved best thresholds: %s" % thr_path)
                patience_counter = 0
            else:
                patience_counter += 1

            patience = config['training'].get('patience', 10)
            should_stop = (patience_counter >= patience)
            if should_stop:
                print("Early stopping at epoch %d" % (epoch + 1))

            # Broadcast early-stop decision to all DDP ranks
            if is_ddp:
                stop_tensor = torch.tensor([1 if should_stop else 0],
                                           dtype=torch.int32, device=device)
                dist.broadcast(stop_tensor, src=0)

            if should_stop:
                break

        scheduler.step()

    if is_main_process():
        print("Training complete! Best avg opt_f1: %.4f" % best_metric)
    cleanup_ddp()
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config['output']['output_dir'], "%s_%s" % (exp_name, timestamp))

    os.makedirs(output_dir, exist_ok=True)
    print("Output directory: %s" % output_dir)

    # Save resolved config
    config_path = os.path.join(output_dir, "config_resolved.yaml")
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    model = train(config, output_dir)


if __name__ == "__main__":
    main()
