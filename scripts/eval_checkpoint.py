"""
Standalone single-GPU checkpoint evaluator.

Usage:
    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python scripts/eval_checkpoint.py \
        --config configs/g3_grounded_swin_binary_clean.yaml \
        --checkpoint outputs_clean/experiments/g3_grounded_swin_binary_clean/best_model.pt \
        --output outputs_clean/experiments/g3_grounded_swin_binary_clean/eval_best.json

Does NOT use DDP. Runs the full val set on a single GPU.
"""
import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, DISEASES
from utils.metrics import compute_per_disease_metrics
from data.label_mapper import LabelMapper, create_train_val_split
from data.shoulder_dataset import ShoulderCacheDataset
from models import create_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_val_loader(config, raw_labels_lookup, project_root):
    cache_mode = config['data'].get('cache_mode', 'cls')
    cache_root = config['data'].get('cache_root', None)
    if cache_root and not os.path.isabs(cache_root):
        cache_root = os.path.join(project_root, cache_root)

    num_classes = config['training'].get('num_classes', 2)
    task_mode = "ternary" if num_classes == 3 else "binary"
    label_mapper = LabelMapper(mode=task_mode)
    batch_size = config['training']['batch_size']
    num_workers = config['training'].get('num_workers', 0)

    all_exam_ids = list(raw_labels_lookup.keys())
    _, val_ids = create_train_val_split(
        all_exam_ids, raw_labels_lookup, task_mode=task_mode, val_ratio=0.2)

    load_dense_masks = config['data'].get('load_dense_masks', False)
    val_dataset = ShoulderCacheDataset(
        cache_root=cache_root,
        exam_ids=set(val_ids),
        label_mapper=label_mapper,
        raw_labels_lookup=raw_labels_lookup,
        cache_mode=cache_mode,
        project_root=project_root,
        load_dense_masks=load_dense_masks,
    )
    print("Val dataset: %d samples" % len(val_dataset))

    pw = num_workers > 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=pw,
        prefetch_factor=2 if pw else None,
    )
    return val_loader, num_classes


def evaluate(model, loader, device, use_localizer, num_classes, amp_dtype):
    model.eval()
    is_binary = (num_classes == 2)

    all_preds, all_probs, all_labels, all_masks = [], [], [], []
    all_pred_slices, all_gt_slices, all_slice_valid = [], [], []

    use_amp = amp_dtype != torch.float32

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            mask_labels = batch["mask"].to(device)

            model_kwargs = {}
            if use_localizer and "key_slices" in batch:
                model_kwargs["key_slices"] = batch["key_slices"].to(device)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                output = model(images, **model_kwargs)

            if is_binary:
                probs = torch.sigmoid(output['final_logits'])
                preds = (probs > 0.5).long()
            else:
                probs = torch.softmax(output['final_logits'], dim=2)
                preds = probs.argmax(dim=2)

            all_preds.append(preds.float().cpu().numpy())
            all_probs.append(probs.detach().float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
            all_masks.append(mask_labels.float().cpu().numpy())

            if use_localizer and 'slice_logits' in output:
                sl = output['slice_logits']
                D_prime = sl.shape[2]
                pred_si = sl.argmax(dim=2).float().cpu().numpy()
                all_pred_slices.append(pred_si)
                if "key_slices" in batch:
                    gt_ks = batch["key_slices"].numpy()
                    input_Z = batch["image"].shape[3]
                    gt_scaled = np.where(
                        gt_ks >= 0,
                        np.clip((gt_ks * D_prime / input_Z).astype(int), 0, D_prime - 1),
                        gt_ks,
                    )
                    all_gt_slices.append(gt_scaled)
                    all_slice_valid.append((gt_ks >= 0).astype(np.float32))

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks  = np.concatenate(all_masks,  axis=0)

    metrics = compute_per_disease_metrics(
        all_labels, all_preds, DISEASES,
        binary=is_binary, y_prob_all=all_probs, mask_all=all_masks)

    if use_localizer and all_pred_slices:
        from utils.metrics import compute_key_slice_metrics
        ks = compute_key_slice_metrics(
            np.concatenate(all_pred_slices, axis=0),
            np.concatenate(all_gt_slices,   axis=0),
            np.concatenate(all_slice_valid,  axis=0),
            DISEASES)
        metrics['key_slice'] = ks

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     "-c", required=True)
    parser.add_argument("--checkpoint", "-k", required=True)
    parser.add_argument("--output",     "-o", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Load raw labels
    import csv
    import warnings
    metadata_path = os.path.join(
        config['output']['output_dir'], "../metadata/metadata_master.csv")
    if not os.path.isabs(metadata_path):
        metadata_path = os.path.join(project_root, metadata_path)
    raw_labels_lookup = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                eid = row['exam_id']
                raw_labels_lookup[eid] = {d: int(row['raw_label_' + d]) for d in DISEASES}
        print("Loaded %d raw labels" % len(raw_labels_lookup))
    else:
        warnings.warn("metadata_master.csv not found: %s" % metadata_path)

    val_loader, num_classes = build_val_loader(config, raw_labels_lookup, project_root)
    is_binary = (num_classes == 2)
    use_localizer = config['model'].get('use_localizer', False)

    # Build model
    model = create_model(
        encoder=config['model']['encoder'],
        num_diseases=len(config['data']['diseases']),
        pretrained=False,   # weights loaded from checkpoint below
        dropout=config['training'].get('dropout', 0.3),
        num_heads=config['model'].get('num_heads', 4),
        branch_alpha=config['training'].get('branch_alpha', 0.3),
        use_localizer=use_localizer,
        num_classes=num_classes,
    )
    model = model.to(device)

    # Load checkpoint
    print("Loading checkpoint: %s" % args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    ckpt_epoch = ckpt.get('epoch', '?')
    print("Checkpoint epoch: %s" % ckpt_epoch)

    # AMP dtype
    amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16

    # Evaluate
    print("Running evaluation on %d val batches..." % len(val_loader))
    metrics = evaluate(model, val_loader, device, use_localizer, num_classes, amp_dtype)

    # Summarize
    from train import summarize_macro_metrics
    macro = summarize_macro_metrics(metrics)

    result = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt_epoch,
        "macro_auc":    round(macro.get('avg_auc', 0),     4),
        "macro_f1":     round(macro.get('avg_f1', 0),      4),
        "macro_opt_f1": round(macro.get('avg_opt_f1', 0),  4),
        "macro_recall": round(macro.get('avg_recall', 0),  4),
        "per_disease":  {d: metrics.get(d, {}) for d in DISEASES},
    }
    if 'key_slice' in metrics:
        ks = metrics['key_slice']
        result["key_slice_top1"] = round(ks.get('macro_ks_top1', 0), 4)
        result["key_slice_pm1"]  = round(ks.get('macro_ks_pm1',  0), 4)

    # Print summary
    print("\n=== Eval Results ===")
    print("  macro AUC    : %.4f" % result['macro_auc'])
    print("  macro F1     : %.4f" % result['macro_f1'])
    print("  macro OptF1  : %.4f" % result['macro_opt_f1'])
    if 'key_slice_top1' in result:
        print("  KS top1      : %.4f" % result['key_slice_top1'])
        print("  KS ±1        : %.4f" % result['key_slice_pm1'])
    print("\n  Per-disease:")
    print("    %-6s  %6s  %6s  %6s" % ("", "AUC", "F1", "OptF1"))
    for d in DISEASES:
        dm = result['per_disease'].get(d, {})
        print("    %-6s  %6.4f  %6.4f  %6.4f" % (
            d, dm.get('auc', 0), dm.get('f1', 0), dm.get('opt_f1', 0)))

    # Save JSON
    out_path = args.output
    if out_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_path = os.path.join(ckpt_dir, "eval_%s.json" % ckpt_name)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nSaved to: %s" % out_path)


if __name__ == "__main__":
    main()
