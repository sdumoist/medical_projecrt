#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_clean_data.py
-------------------
One-shot script to create a "clean" parallel data directory for grounded SFT.

Clean rules (strict):
    1. quality_flag != "low"
    2. postoperative != 1
    3. exclude_from_main_training != 1
    4. All 7 diseases raw_label in {0, 1} (no uncertain, no missing)
    5. Must have both cache_cls and cache_loc entries with success=1

Outputs (no large .pt files are copied, only index/metadata filtered):
    outputs_clean/
    ├── metadata/
    │   └── metadata_master.csv          # filtered rows only
    ├── cache_cls/
    │   └── cache_cls_index.csv          # filtered, paths point to original cache_cls/
    ├── cache_loc/
    │   └── cache_loc_index.csv          # filtered, paths point to original cache_loc/
    ├── sft_data/
    │   ├── sft_label_binary.jsonl
    │   ├── sft_diagnosis_chain.jsonl
    │   └── split/
    │       ├── sft_label_binary_train.jsonl
    │       ├── sft_label_binary_val.jsonl
    │       ├── sft_label_binary_test.jsonl
    │       ├── sft_diagnosis_chain_train.jsonl
    │       ├── sft_diagnosis_chain_val.jsonl
    │       ├── sft_diagnosis_chain_test.jsonl
    │       └── split_meta.json
    └── stats/
        └── clean_summary.json

Usage:
    PYTHONPATH=. python scripts/build_clean_data.py
    PYTHONPATH=. python scripts/build_clean_data.py --output_root outputs_clean
"""
from __future__ import print_function

import os
import sys
import csv
import json
import random
import argparse
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.constants import DISEASES, DISEASE_ANCHOR_SEQ, NUM_DISEASES
from sft.prompts import build_prompt_plain
from data.label_mapper import create_train_val_split


# ── Clean filter ────────────────────────────────────────────────────────

def is_clean(row):
    """Apply strict clean rules to a metadata row.

    Returns (True, None) if clean, (False, reason) otherwise.
    """
    if row.get("quality_flag", "") == "low":
        return False, "quality_low"
    if int(row.get("postoperative", "0")) == 1:
        return False, "postoperative"
    if int(row.get("exclude_from_main_training", "0")) == 1:
        return False, "exclude_flag"

    for d in DISEASES:
        raw = int(row.get("raw_label_%s" % d, "-1"))
        if raw not in (0, 1):
            return False, "not_all_binary"

    return True, None


# ── SFT output builders (label_binary + diagnosis_chain only) ───────

def build_label_binary_output(row):
    """Build label_binary JSON output from metadata row."""
    result = {"labels": {}}
    for d in DISEASES:
        label = int(row["raw_label_%s" % d])
        status = row.get("status_%s" % d, "unknown")
        result["labels"][d] = {"label": label, "status": status}
    return json.dumps(result, ensure_ascii=False)


def build_diagnosis_chain_output(row, case_json, loc_row):
    """Build diagnosis_chain JSON output from metadata + case_json + loc_row."""
    result = {
        "labels": {},
        "evidence": {},
        "anchor_sequence": {},
        "key_slice": {},
        "structured_findings": case_json.get("structured_findings", []),
        "structured_impression": case_json.get("structured_impression", []),
    }

    evidence_text = case_json.get("evidence_text", {})
    negative_evidence = case_json.get("negative_evidence", {})

    for d in DISEASES:
        result["labels"][d] = int(row["raw_label_%s" % d])
        result["evidence"][d] = {
            "positive": evidence_text.get(d, []),
            "negative": negative_evidence.get(d, []),
        }
        result["anchor_sequence"][d] = DISEASE_ANCHOR_SEQ.get(d, "unknown")

        # key_slice from loc_row CSV columns
        ks_col = "%s_key_slice" % d
        ks_val = int(loc_row.get(ks_col, "-1")) if loc_row else -1
        result["key_slice"][d] = ks_val if ks_val >= 0 else None

    return json.dumps(result, ensure_ascii=False)


def build_sample(exam_id, task_type, output_str, row, case_json):
    """Build a JSONL sample dict."""
    qf = row.get("quality_flag", "unknown")
    return {
        "exam_id": exam_id,
        "task_type": task_type,
        "quality_bucket": "A" if qf == "high" else "B",
        "instruction": build_prompt_plain(task_type),
        "output": output_str,
        "train_policy": {
            "use_for_label": True,
            "use_for_chain": True,
            "use_for_findings": False,
            "use_for_impression": False,
        },
        "metadata": {
            "quality_flag": qf,
            "laterality": row.get("laterality", "unknown"),
            "sex": case_json.get("sex", "unknown"),
            "age": case_json.get("age", "unknown"),
            "num_valid_labels": 7,
            "has_evidence": any(
                len(case_json.get("evidence_text", {}).get(d, [])) > 0
                for d in DISEASES),
            "has_key_slice": True,
            "has_roi": False,
        },
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build clean parallel data directory for grounded SFT")
    parser.add_argument("--output_root", default="outputs_clean",
                        help="Root of clean output directory")
    parser.add_argument("--metadata_csv",
                        default="outputs/metadata/metadata_master.csv")
    parser.add_argument("--cache_cls_index",
                        default="outputs/cache_cls/cache_cls_index.csv")
    parser.add_argument("--cache_loc_index",
                        default="outputs/cache_loc/cache_loc_index.csv")
    parser.add_argument("--json_root",
                        default="/mnt/cfs_algo_bj/models/experiments/lirunze"
                        "/code/shouder/final_output/to_extract/case_json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve paths relative to project root
    def resolve(p):
        if not os.path.isabs(p):
            return os.path.join(PROJECT_ROOT, p)
        return p

    metadata_csv = resolve(args.metadata_csv)
    cls_index_path = resolve(args.cache_cls_index)
    loc_index_path = resolve(args.cache_loc_index)
    json_root = args.json_root
    output_root = resolve(args.output_root)

    # ── Step 1: Read source data ──

    # Metadata
    meta_rows = {}
    meta_header = None
    with open(metadata_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        meta_header = reader.fieldnames
        for row in reader:
            meta_rows[row["exam_id"]] = row
    print("Metadata: %d rows" % len(meta_rows))

    # cache_cls index
    cls_rows = {}
    cls_header = None
    with open(cls_index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cls_header = reader.fieldnames
        for row in reader:
            if row.get("success", "1") == "1":
                cls_rows[row["exam_id"]] = row
    print("cache_cls: %d success rows" % len(cls_rows))

    # cache_loc index
    loc_rows = {}
    loc_header = None
    with open(loc_index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        loc_header = reader.fieldnames
        for row in reader:
            if row.get("success", "1") == "1":
                loc_rows[row["exam_id"]] = row
    print("cache_loc: %d success rows" % len(loc_rows))

    # ── Step 2: Apply clean filter ──

    excluded = Counter()
    clean_ids = []

    for eid, row in sorted(meta_rows.items()):
        ok, reason = is_clean(row)
        if not ok:
            excluded[reason] += 1
            continue
        if eid not in cls_rows:
            excluded["no_cache_cls"] += 1
            continue
        if eid not in loc_rows:
            excluded["no_cache_loc"] += 1
            continue
        clean_ids.append(eid)

    print("\n=== Clean Filter ===")
    print("Total: %d" % len(meta_rows))
    for k, v in sorted(excluded.items()):
        print("  excluded %-25s %d" % (k, v))
    print("Clean: %d (%.1f%%)" % (
        len(clean_ids), 100 * len(clean_ids) / len(meta_rows)))

    # ── Step 3: Create output directories ──

    dirs = [
        os.path.join(output_root, "metadata"),
        os.path.join(output_root, "cache_cls"),
        os.path.join(output_root, "cache_loc"),
        os.path.join(output_root, "sft_data"),
        os.path.join(output_root, "sft_data", "split"),
        os.path.join(output_root, "stats"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # ── Step 4: Write filtered metadata ──

    clean_set = set(clean_ids)
    meta_out = os.path.join(output_root, "metadata", "metadata_master.csv")
    with open(meta_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=meta_header)
        writer.writeheader()
        for eid in clean_ids:
            writer.writerow(meta_rows[eid])
    print("\nWrote %s (%d rows)" % (meta_out, len(clean_ids)))

    # ── Step 5: Write filtered cache index (point to original .pt) ──

    cls_out = os.path.join(output_root, "cache_cls", "cache_cls_index.csv")
    with open(cls_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cls_header)
        writer.writeheader()
        for eid in clean_ids:
            writer.writerow(cls_rows[eid])
    print("Wrote %s (%d rows)" % (cls_out, len(clean_ids)))

    loc_out = os.path.join(output_root, "cache_loc", "cache_loc_index.csv")
    with open(loc_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=loc_header)
        writer.writeheader()
        for eid in clean_ids:
            writer.writerow(loc_rows[eid])
    print("Wrote %s (%d rows)" % (loc_out, len(clean_ids)))

    # ── Step 6: Build SFT JSONL (label_binary + diagnosis_chain) ──

    label_binary_path = os.path.join(output_root, "sft_data", "sft_label_binary.jsonl")
    diag_chain_path = os.path.join(output_root, "sft_data", "sft_diagnosis_chain.jsonl")

    label_dist = {d: Counter() for d in DISEASES}
    n_lb = 0
    n_dc = 0
    n_has_evidence = 0
    n_has_ks = 0

    f_lb = open(label_binary_path, "w", encoding="utf-8")
    f_dc = open(diag_chain_path, "w", encoding="utf-8")

    for eid in clean_ids:
        row = meta_rows[eid]
        loc_row = loc_rows.get(eid, {})

        # Load case JSON
        jpath = os.path.join(json_root, "%s.json" % eid)
        if not os.path.exists(jpath):
            continue
        with open(jpath, "r", encoding="utf-8") as f:
            case_json = json.load(f)

        # Label distribution
        for d in DISEASES:
            label_dist[d][int(row["raw_label_%s" % d])] += 1

        # label_binary
        lb_output = build_label_binary_output(row)
        lb_sample = build_sample(eid, "label_binary", lb_output, row, case_json)
        f_lb.write(json.dumps(lb_sample, ensure_ascii=False) + "\n")
        n_lb += 1

        # diagnosis_chain (requires at least some evidence)
        has_ev = any(
            len(case_json.get("evidence_text", {}).get(d, [])) > 0
            for d in DISEASES)
        if has_ev:
            n_has_evidence += 1

        has_ks = any(
            int(loc_row.get("%s_key_slice" % d, "-1")) >= 0
            for d in DISEASES)
        if has_ks:
            n_has_ks += 1

        dc_output = build_diagnosis_chain_output(row, case_json, loc_row)
        dc_sample = build_sample(eid, "diagnosis_chain", dc_output, row, case_json)
        dc_sample["train_policy"]["use_for_chain"] = has_ev
        f_dc.write(json.dumps(dc_sample, ensure_ascii=False) + "\n")
        n_dc += 1

    f_lb.close()
    f_dc.close()

    print("\nSFT JSONL:")
    print("  label_binary: %d" % n_lb)
    print("  diagnosis_chain: %d" % n_dc)

    # ── Step 7: Train/val/test split ──

    raw_labels_lookup = {}
    for eid in clean_ids:
        row = meta_rows[eid]
        raw_labels_lookup[eid] = {
            d: int(row["raw_label_%s" % d]) for d in DISEASES}

    train_ids, val_ids = create_train_val_split(
        clean_ids, raw_labels_lookup, task_mode="binary",
        val_ratio=0.2, seed=42)

    # Further split val into SFT val + test (50/50)
    random.seed(args.seed)
    val_list = sorted(val_ids)
    random.shuffle(val_list)
    n_val = len(val_list) // 2
    sft_val = set(val_list[:n_val])
    sft_test = set(val_list[n_val:])
    sft_train = set(train_ids)

    print("\nSplit: train=%d, val=%d, test=%d" % (
        len(sft_train), len(sft_val), len(sft_test)))

    # Split each JSONL
    split_dir = os.path.join(output_root, "sft_data", "split")
    for src_name, src_path in [
        ("sft_label_binary", label_binary_path),
        ("sft_diagnosis_chain", diag_chain_path),
    ]:
        out_files = {}
        counts = {"train": 0, "val": 0, "test": 0}
        for sp in ("train", "val", "test"):
            out_files[sp] = open(
                os.path.join(split_dir, "%s_%s.jsonl" % (src_name, sp)),
                "w", encoding="utf-8")

        with open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                eid = sample["exam_id"]
                if eid in sft_train:
                    out_files["train"].write(line + "\n")
                    counts["train"] += 1
                elif eid in sft_val:
                    out_files["val"].write(line + "\n")
                    counts["val"] += 1
                elif eid in sft_test:
                    out_files["test"].write(line + "\n")
                    counts["test"] += 1

        for fp in out_files.values():
            fp.close()
        print("  %s: train=%d, val=%d, test=%d" % (
            src_name, counts["train"], counts["val"], counts["test"]))

    # Save split_meta.json
    split_meta = {
        "seed": args.seed,
        "cv_split_seed": 42,
        "clean_filter": "strict_v1",
        "train_count": len(sft_train),
        "val_count": len(sft_val),
        "test_count": len(sft_test),
        "train_ids": sorted(sft_train),
        "val_ids": sorted(sft_val),
        "test_ids": sorted(sft_test),
    }
    split_meta_path = os.path.join(split_dir, "split_meta.json")
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)

    # ── Step 8: Write clean_summary.json ──

    summary = {
        "clean_filter": {
            "rules": [
                "quality_flag != low",
                "postoperative != 1",
                "exclude_from_main_training != 1",
                "all 7 diseases raw_label in {0, 1}",
                "cache_cls and cache_loc both available",
            ],
            "total_scanned": len(meta_rows),
            "excluded": dict(excluded),
            "clean_count": len(clean_ids),
            "clean_rate": round(len(clean_ids) / len(meta_rows), 4),
        },
        "split": {
            "train": len(sft_train),
            "val": len(sft_val),
            "test": len(sft_test),
        },
        "sft_tasks": {
            "label_binary": n_lb,
            "diagnosis_chain": n_dc,
        },
        "label_distribution": {
            d: {"positive": label_dist[d].get(1, 0),
                "negative": label_dist[d].get(0, 0)}
            for d in DISEASES
        },
        "field_completeness": {
            "evidence_nonempty": n_has_evidence,
            "key_slice_available": n_has_ks,
            "evidence_rate": round(n_has_evidence / max(n_dc, 1), 4),
            "key_slice_rate": round(n_has_ks / max(n_dc, 1), 4),
        },
    }

    summary_path = os.path.join(output_root, "stats", "clean_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary written to %s ===" % summary_path)
    print("Clean: %d / %d (%.1f%%)" % (
        len(clean_ids), len(meta_rows),
        100 * len(clean_ids) / len(meta_rows)))
    print("Label distribution:")
    for d in DISEASES:
        pos = label_dist[d].get(1, 0)
        neg = label_dist[d].get(0, 0)
        t = pos + neg
        print("  %-6s pos=%d neg=%d (pos_rate=%.1f%%)" % (
            d, pos, neg, 100 * pos / t if t > 0 else 0))
    print("\nDone!")


if __name__ == "__main__":
    main()
