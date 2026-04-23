#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_sft_jsonl.py
--------------------
Inspect SFT JSONL files: sample counts, field completeness, label distribution,
quality bucket distribution, output token count estimates.

Usage:
    PYTHONPATH=. python scripts/inspect_sft_jsonl.py \
        --input_dir outputs/sft_data \
        --output_json outputs/sft_data/inspection_report.json
"""
from __future__ import print_function

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import DISEASES


def inspect_jsonl(jsonl_path):
    """Inspect a single JSONL file and return statistics."""
    stats = {
        "file": os.path.basename(jsonl_path),
        "total_samples": 0,
        "bucket_distribution": defaultdict(int),
        "label_distribution": {d: defaultdict(int) for d in DISEASES},
        "field_completeness": defaultdict(int),
        "output_token_lengths": [],
        "task_type": None,
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            stats["total_samples"] += 1

            if stats["task_type"] is None:
                stats["task_type"] = sample.get("task_type", "unknown")

            # Bucket
            bucket = sample.get("quality_bucket", "?")
            stats["bucket_distribution"][bucket] += 1

            # Output token length estimate (chars / 1.5 for Chinese)
            output_str = sample.get("output", "")
            token_est = len(output_str) / 1.5
            stats["output_token_lengths"].append(token_est)

            # Parse output for field-level analysis
            try:
                output = json.loads(output_str)
            except json.JSONDecodeError:
                stats["field_completeness"]["parse_fail"] += 1
                continue

            stats["field_completeness"]["parse_ok"] += 1

            task_type = sample.get("task_type", "")

            if task_type == "label_binary":
                labels_dict = output.get("labels", {})
                for d in DISEASES:
                    entry = labels_dict.get(d, {})
                    if isinstance(entry, dict):
                        lab = entry.get("label", -99)
                    else:
                        lab = entry
                    if lab == 1:
                        stats["label_distribution"][d]["positive"] += 1
                    elif lab == 0:
                        stats["label_distribution"][d]["negative"] += 1
                    elif lab == 2:
                        stats["label_distribution"][d]["uncertain"] += 1
                    elif lab == -1:
                        stats["label_distribution"][d]["masked"] += 1

                if "labels" in output:
                    stats["field_completeness"]["has_labels"] += 1

            elif task_type == "diagnosis_chain":
                if "labels" in output:
                    stats["field_completeness"]["has_labels"] += 1
                if "evidence" in output:
                    stats["field_completeness"]["has_evidence"] += 1
                    # Count non-empty evidence
                    ev = output["evidence"]
                    n_pos_ev = sum(1 for d in DISEASES
                                   if ev.get(d, {}).get("positive"))
                    n_neg_ev = sum(1 for d in DISEASES
                                   if ev.get(d, {}).get("negative"))
                    stats["field_completeness"]["evidence_pos_nonempty"] += n_pos_ev
                    stats["field_completeness"]["evidence_neg_nonempty"] += n_neg_ev
                if "anchor_sequence" in output:
                    stats["field_completeness"]["has_anchor_sequence"] += 1
                if "key_slice" in output:
                    stats["field_completeness"]["has_key_slice"] += 1
                    ks = output["key_slice"]
                    n_ks = sum(1 for d in DISEASES
                               if ks.get(d) is not None)
                    stats["field_completeness"]["key_slice_nonempty"] += n_ks
                if "roi_box" in output:
                    stats["field_completeness"]["has_roi_box"] += 1
                    rb = output["roi_box"]
                    n_rb = sum(1 for d in DISEASES
                               if rb.get(d) is not None)
                    stats["field_completeness"]["roi_box_nonempty"] += n_rb

                # Also count labels for label distribution
                labels_dict = output.get("labels", {})
                for d in DISEASES:
                    lab = labels_dict.get(d, -99)
                    if lab == 1:
                        stats["label_distribution"][d]["positive"] += 1
                    elif lab == 0:
                        stats["label_distribution"][d]["negative"] += 1
                    elif lab == 2:
                        stats["label_distribution"][d]["uncertain"] += 1
                    elif lab == -1:
                        stats["label_distribution"][d]["masked"] += 1

            elif task_type == "structured_findings":
                findings = output.get("structured_findings", [])
                if findings:
                    stats["field_completeness"]["has_findings"] += 1
                    stats["field_completeness"]["findings_total_sentences"] += len(findings)

            elif task_type == "structured_impression":
                impression = output.get("structured_impression", [])
                if impression:
                    stats["field_completeness"]["has_impression"] += 1
                    stats["field_completeness"]["impression_total_sentences"] += len(impression)

    return stats


def print_report(all_stats):
    """Print human-readable inspection report."""
    print("\n" + "=" * 70)
    print("SFT JSONL Inspection Report")
    print("=" * 70)

    for stats in all_stats:
        n = stats["total_samples"]
        print("\n--- %s (%s) ---" % (stats["file"], stats["task_type"]))
        print("Total samples: %d" % n)

        # Buckets
        print("Quality buckets:")
        for b in sorted(stats["bucket_distribution"].keys()):
            cnt = stats["bucket_distribution"][b]
            print("  %s: %d (%.1f%%)" % (b, cnt, 100.0 * cnt / max(n, 1)))

        # Output token lengths
        lengths = stats["output_token_lengths"]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            print("Output token estimates: avg=%.0f, min=%.0f, max=%.0f" % (
                avg_len, min_len, max_len))

        # Field completeness
        fc = dict(stats["field_completeness"])
        if n > 0:
            print("Field completeness:")
            for key in sorted(fc.keys()):
                val = fc[key]
                if key.endswith("_nonempty") or key.startswith("has_"):
                    print("  %s: %d (%.1f%%)" % (key, val, 100.0 * val / n))
                elif key == "parse_ok":
                    print("  JSON parse success: %d/%d (%.1f%%)" % (
                        val, n, 100.0 * val / n))
                elif key == "parse_fail":
                    print("  JSON parse failures: %d" % val)
                else:
                    print("  %s: %d" % (key, val))

        # Label distribution (if applicable)
        task = stats["task_type"]
        if task in ("label_binary", "diagnosis_chain"):
            print("Label distribution:")
            print("  %-6s %8s %8s %8s %8s" % (
                "", "pos", "neg", "unc", "masked"))
            for d in DISEASES:
                ld = stats["label_distribution"][d]
                print("  %-6s %8d %8d %8d %8d" % (
                    d, ld.get("positive", 0), ld.get("negative", 0),
                    ld.get("uncertain", 0), ld.get("masked", 0)))


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SFT JSONL files")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing JSONL files")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Specific JSONL files to inspect (default: all sft_*.jsonl)")
    parser.add_argument("--output_json", default=None,
                        help="Optional path to save JSON report")
    args = parser.parse_args()

    if args.files:
        jsonl_files = [os.path.join(args.input_dir, f) for f in args.files]
    else:
        jsonl_files = sorted(glob.glob(
            os.path.join(args.input_dir, "sft_*.jsonl")))

    if not jsonl_files:
        print("No JSONL files found in %s" % args.input_dir)
        return

    all_stats = []
    for jf in jsonl_files:
        if not os.path.exists(jf):
            print("WARNING: %s not found, skipping" % jf)
            continue
        print("Inspecting %s ..." % jf)
        stats = inspect_jsonl(jf)
        all_stats.append(stats)

    print_report(all_stats)

    if args.output_json:
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable = []
        for s in all_stats:
            sc = dict(s)
            sc["bucket_distribution"] = dict(sc["bucket_distribution"])
            sc["field_completeness"] = dict(sc["field_completeness"])
            sc["label_distribution"] = {
                d: dict(v) for d, v in sc["label_distribution"].items()
            }
            # Remove raw token lengths for compactness
            lengths = sc.pop("output_token_lengths")
            if lengths:
                sc["output_token_stats"] = {
                    "avg": round(sum(lengths) / len(lengths), 1),
                    "min": round(min(lengths), 1),
                    "max": round(max(lengths), 1),
                }
            serializable.append(sc)

        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print("\nJSON report saved to: %s" % args.output_json)


if __name__ == "__main__":
    main()
