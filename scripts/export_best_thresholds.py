#!/usr/bin/env python3
"""
Export best per-disease thresholds from an experiment directory.

Reads best_thresholds.json if available, otherwise reconstructs from
metrics_epoch.csv (last best epoch row).

Usage:
    python scripts/export_best_thresholds.py --exp_dir outputs/experiments/g2_resnet_binary
    python scripts/export_best_thresholds.py --exp_dir outputs/experiments_copas/CoPAS_orig_... --output thresholds.json
"""
import os
import sys
import csv
import json
import argparse

# ---- project imports -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.constants import DISEASES


def safe_float(val, default=0.0):
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def export_from_json(json_path):
    """Load thresholds directly from best_thresholds.json."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def export_from_csv(csv_path):
    """Reconstruct thresholds from the best epoch in metrics_epoch.csv."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    # Find best row
    best_rows = [r for r in rows if r.get('is_best', '0') == '1']
    row = best_rows[-1] if best_rows else rows[-1]

    thresholds = {}
    for d in DISEASES:
        thresholds[d] = {
            'opt_thr': safe_float(row.get('%s_opt_thr' % d), 0.5),
            'opt_f1': safe_float(row.get('%s_opt_f1' % d)),
            'auc': safe_float(row.get('%s_auc' % d)),
        }

    # Macro metrics for _meta
    macro_auc = 0.0
    macro_f1 = 0.0
    macro_opt_f1 = 0.0
    for prefix in ['val_avg_', 'val_macro_']:
        if prefix + 'auc' in row:
            macro_auc = safe_float(row[prefix + 'auc'])
        if prefix + 'f1' in row:
            macro_f1 = safe_float(row[prefix + 'f1'])
        if prefix + 'opt_f1' in row:
            macro_opt_f1 = safe_float(row[prefix + 'opt_f1'])

    thresholds['_meta'] = {
        'epoch': int(safe_float(row.get('epoch', 0))),
        'val_macro_auc': macro_auc,
        'val_macro_f1': macro_f1,
        'val_macro_opt_f1': macro_opt_f1,
        'source': 'reconstructed_from_csv',
    }
    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Export best thresholds from experiment")
    parser.add_argument("--exp_dir", required=True, help="Experiment directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <exp_dir>/best_thresholds.json)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing best_thresholds.json")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    json_path = os.path.join(exp_dir, "best_thresholds.json")
    csv_path = os.path.join(exp_dir, "metrics_epoch.csv")

    # If best_thresholds.json already exists and --force not set, just display it
    if os.path.exists(json_path) and not args.force:
        thresholds = export_from_json(json_path)
        print("Loaded existing best_thresholds.json from %s" % json_path)
    elif os.path.exists(csv_path):
        thresholds = export_from_csv(csv_path)
        if thresholds is None:
            print("ERROR: metrics_epoch.csv is empty")
            return
        print("Reconstructed thresholds from %s" % csv_path)
    else:
        print("ERROR: No best_thresholds.json or metrics_epoch.csv found in %s" % exp_dir)
        return

    # Print summary
    print("\n%-8s  %8s  %8s  %8s" % ("Disease", "Opt Thr", "Opt F1", "AUC"))
    print("-" * 40)
    for d in DISEASES:
        if d in thresholds:
            td = thresholds[d]
            print("%-8s  %8.4f  %8.4f  %8.4f" % (
                d, td.get('opt_thr', 0.5), td.get('opt_f1', 0), td.get('auc', 0)))

    if '_meta' in thresholds:
        meta = thresholds['_meta']
        print("\nMeta: epoch=%s  macro_auc=%.4f  macro_f1=%.4f  macro_opt_f1=%.4f" % (
            meta.get('epoch', '?'),
            meta.get('val_macro_auc', 0),
            meta.get('val_macro_f1', 0),
            meta.get('val_macro_opt_f1', 0)))

    # Save output
    output_path = args.output or json_path
    with open(output_path, 'w') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print("\nSaved to: %s" % output_path)


if __name__ == "__main__":
    main()
