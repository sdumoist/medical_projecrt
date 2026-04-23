#!/usr/bin/env python3
"""
Summarize Phase 1 experiment results into a unified comparison table.

Scans experiment directories for metrics_epoch.csv / best_thresholds.json,
extracts best-epoch metrics, and outputs a consolidated Markdown table.

Usage:
    python scripts/summarize_phase1_results.py [--exp_dirs DIR1 DIR2 ...]
    python scripts/summarize_phase1_results.py --auto   # auto-discover all experiments
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path

# ---- project imports -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.constants import DISEASES

# Experiment roots to scan in --auto mode
DEFAULT_ROOTS = [
    "outputs/experiments",
    "outputs/experiments_copas",
]


def find_experiment_dirs(roots, project_root):
    """Discover all experiment directories containing metrics_epoch.csv."""
    exp_dirs = []
    for root in roots:
        abs_root = root if os.path.isabs(root) else os.path.join(project_root, root)
        if not os.path.isdir(abs_root):
            continue
        for name in sorted(os.listdir(abs_root)):
            d = os.path.join(abs_root, name)
            if os.path.isfile(os.path.join(d, "metrics_epoch.csv")):
                exp_dirs.append(d)
    return exp_dirs


def load_best_row_from_csv(csv_path):
    """Load the best-epoch row from metrics_epoch.csv.

    Heuristic: row with is_best==1, or if multiple, the last one.
    Falls back to last row if no is_best column.
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    # Find best row
    best_rows = [r for r in rows if r.get('is_best', '0') == '1']
    if best_rows:
        return best_rows[-1]  # last best (highest epoch that was best)
    return rows[-1]  # fallback to last epoch


def load_thresholds(exp_dir):
    """Load best_thresholds.json if present."""
    path = os.path.join(exp_dir, "best_thresholds.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def safe_float(val, default=0.0):
    """Safely convert to float."""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def extract_summary(exp_dir):
    """Extract summary metrics from one experiment directory."""
    csv_path = os.path.join(exp_dir, "metrics_epoch.csv")
    if not os.path.exists(csv_path):
        return None

    row = load_best_row_from_csv(csv_path)
    if row is None:
        return None

    thresholds = load_thresholds(exp_dir)
    exp_name = os.path.basename(exp_dir)

    summary = {
        'exp_name': exp_name,
        'exp_dir': exp_dir,
        'best_epoch': int(safe_float(row.get('epoch', 0))),
    }

    # -- Macro metrics --
    # Try different column naming conventions
    for prefix in ['val_avg_', 'val_macro_']:
        for key in ['f1', 'auc', 'opt_f1']:
            col = prefix + key
            if col in row:
                summary['macro_' + key] = safe_float(row[col])
    # Fallback: compute from per-disease if macro not in CSV
    if 'macro_f1' not in summary:
        f1_vals = [safe_float(row.get('%s_f1' % d)) for d in DISEASES
                   if '%s_f1' % d in row]
        if f1_vals:
            summary['macro_f1'] = sum(f1_vals) / len(f1_vals)
    if 'macro_auc' not in summary:
        auc_vals = [safe_float(row.get('%s_auc' % d)) for d in DISEASES
                    if '%s_auc' % d in row]
        if auc_vals:
            summary['macro_auc'] = sum(auc_vals) / len(auc_vals)
    if 'macro_opt_f1' not in summary:
        opt_vals = [safe_float(row.get('%s_opt_f1' % d)) for d in DISEASES
                    if '%s_opt_f1' % d in row]
        if opt_vals:
            summary['macro_opt_f1'] = sum(opt_vals) / len(opt_vals)

    # -- Per-disease metrics --
    for d in DISEASES:
        summary['%s_f1' % d] = safe_float(row.get('%s_f1' % d))
        summary['%s_auc' % d] = safe_float(row.get('%s_auc' % d))
        summary['%s_recall' % d] = safe_float(row.get('%s_recall' % d))
        summary['%s_precision' % d] = safe_float(row.get('%s_precision' % d))
        summary['%s_opt_f1' % d] = safe_float(row.get('%s_opt_f1' % d))
        summary['%s_opt_thr' % d] = safe_float(row.get('%s_opt_thr' % d), 0.5)

    # -- Thresholds from best_thresholds.json (override CSV if available) --
    if thresholds:
        for d in DISEASES:
            if d in thresholds:
                td = thresholds[d]
                summary['%s_opt_thr' % d] = td.get('opt_thr', summary.get('%s_opt_thr' % d, 0.5))
                summary['%s_opt_f1' % d] = td.get('opt_f1', summary.get('%s_opt_f1' % d, 0))
        if '_meta' in thresholds:
            meta = thresholds['_meta']
            summary['macro_auc'] = meta.get('val_macro_auc', summary.get('macro_auc', 0))
            summary['macro_opt_f1'] = meta.get('val_macro_opt_f1', summary.get('macro_opt_f1', 0))

    # -- Key-slice metrics --
    summary['ks_top1'] = safe_float(row.get('macro_ks_top1'))
    summary['ks_pm1'] = safe_float(row.get('macro_ks_pm1'))

    return summary


def print_main_table(summaries):
    """Print main comparison table in Markdown."""
    print("\n## Phase 1 Main Results\n")
    header = "| Experiment | Epoch | Macro AUC | Macro F1 | Macro Opt-F1 | KS Top1 | KS ±1 |"
    sep = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for s in summaries:
        ks_top1 = "%.4f" % s['ks_top1'] if s.get('ks_top1', 0) > 0 else "-"
        ks_pm1 = "%.4f" % s['ks_pm1'] if s.get('ks_pm1', 0) > 0 else "-"
        print("| %s | %d | %.4f | %.4f | %.4f | %s | %s |" % (
            s['exp_name'], s['best_epoch'],
            s.get('macro_auc', 0), s.get('macro_f1', 0),
            s.get('macro_opt_f1', 0), ks_top1, ks_pm1))


def print_per_disease_table(summaries):
    """Print per-disease detail table in Markdown."""
    print("\n## Per-Disease AUC\n")
    header = "| Experiment | " + " | ".join(DISEASES) + " |"
    sep = "|---|" + "|".join(["---"] * len(DISEASES)) + "|"
    print(header)
    print(sep)
    for s in summaries:
        vals = " | ".join("%.4f" % s.get('%s_auc' % d, 0) for d in DISEASES)
        print("| %s | %s |" % (s['exp_name'], vals))

    print("\n## Per-Disease Opt-F1\n")
    print(header.replace("AUC", "Opt-F1"))
    print(sep)
    for s in summaries:
        vals = " | ".join("%.4f" % s.get('%s_opt_f1' % d, 0) for d in DISEASES)
        print("| %s | %s |" % (s['exp_name'], vals))

    print("\n## Per-Disease Optimal Threshold\n")
    print(header.replace("AUC", "Opt-Thr"))
    print(sep)
    for s in summaries:
        vals = " | ".join("%.2f" % s.get('%s_opt_thr' % d, 0.5) for d in DISEASES)
        print("| %s | %s |" % (s['exp_name'], vals))


def save_summary_json(summaries, output_path):
    """Save summaries as JSON for downstream use."""
    with open(output_path, 'w') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print("\nSaved summary JSON: %s" % output_path)


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 1 experiment results")
    parser.add_argument("--exp_dirs", nargs="+", default=None,
                        help="Explicit experiment directories to summarize")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover experiments in default output roots")
    parser.add_argument("--output", "-o", default=None,
                        help="Save summary JSON to this path")
    parser.add_argument("--project_root", default=None,
                        help="Project root for resolving relative paths")
    args = parser.parse_args()

    project_root = args.project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.exp_dirs:
        exp_dirs = args.exp_dirs
    elif args.auto:
        exp_dirs = find_experiment_dirs(DEFAULT_ROOTS, project_root)
    else:
        exp_dirs = find_experiment_dirs(DEFAULT_ROOTS, project_root)

    if not exp_dirs:
        print("No experiment directories found.")
        return

    print("Found %d experiment(s):" % len(exp_dirs))
    for d in exp_dirs:
        print("  %s" % d)

    summaries = []
    for d in exp_dirs:
        s = extract_summary(d)
        if s:
            summaries.append(s)

    if not summaries:
        print("No valid experiment results found.")
        return

    # Sort by macro AUC descending
    summaries.sort(key=lambda x: x.get('macro_auc', 0), reverse=True)

    print_main_table(summaries)
    print_per_disease_table(summaries)

    if args.output:
        save_summary_json(summaries, args.output)
    else:
        default_out = os.path.join(project_root, "outputs", "phase1_summary.json")
        os.makedirs(os.path.dirname(default_out), exist_ok=True)
        save_summary_json(summaries, default_out)


if __name__ == "__main__":
    main()
