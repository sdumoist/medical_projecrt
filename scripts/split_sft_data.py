#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
split_sft_data.py
-----------------
Split SFT JSONL files into train/val/test at exam_id level.

Uses the same stratified split as the CV training pipeline to prevent
data leakage. CV val_ids are further split 50/50 into SFT val + test.

Usage:
    PYTHONPATH=. python scripts/split_sft_data.py \
        --input_dir outputs/sft_data \
        --output_dir outputs/sft_data/split \
        --metadata_csv outputs/metadata/metadata_master.csv
"""
from __future__ import print_function

import os
import sys
import csv
import json
import glob
import random
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import DISEASES
from data.label_mapper import create_train_val_split


def load_raw_labels_from_csv(metadata_csv):
    """Load raw_labels_lookup from metadata_master.csv."""
    raw_labels_lookup = {}
    with open(metadata_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["exam_id"]
            raw_labels = {}
            for d in DISEASES:
                raw_labels[d] = int(row["raw_label_" + d])
            raw_labels_lookup[eid] = raw_labels
    return raw_labels_lookup


def split_sft_data(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    metadata_csv = args.metadata_csv
    seed = args.seed

    os.makedirs(output_dir, exist_ok=True)

    # Load raw labels for stratified split
    raw_labels_lookup = load_raw_labels_from_csv(metadata_csv)
    all_exam_ids = sorted(raw_labels_lookup.keys())
    print("Loaded %d exam IDs from metadata" % len(all_exam_ids))

    # Use the same CV split (seed=42, 80/20)
    train_ids, val_ids = create_train_val_split(
        all_exam_ids, raw_labels_lookup, task_mode="binary", val_ratio=0.2, seed=42)

    # Further split val_ids into SFT val + test (50/50)
    random.seed(seed)
    val_ids_shuffled = list(val_ids)
    random.shuffle(val_ids_shuffled)
    n_val = len(val_ids_shuffled) // 2
    sft_val_ids = set(val_ids_shuffled[:n_val])
    sft_test_ids = set(val_ids_shuffled[n_val:])
    sft_train_ids = set(train_ids)

    print("Split: train=%d, val=%d, test=%d" % (
        len(sft_train_ids), len(sft_val_ids), len(sft_test_ids)))

    # Find all JSONL files
    jsonl_files = sorted(glob.glob(os.path.join(input_dir, "sft_*.jsonl")))
    if not jsonl_files:
        print("No sft_*.jsonl files found in %s" % input_dir)
        return

    split_counts = {}

    for jf in jsonl_files:
        basename = os.path.basename(jf).replace(".jsonl", "")
        counts = {"train": 0, "val": 0, "test": 0, "unmatched": 0}

        # Open output files
        out_files = {}
        for split in ("train", "val", "test"):
            path = os.path.join(output_dir, "%s_%s.jsonl" % (basename, split))
            out_files[split] = open(path, "w", encoding="utf-8")

        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                eid = sample.get("exam_id", "")

                if eid in sft_train_ids:
                    out_files["train"].write(line + "\n")
                    counts["train"] += 1
                elif eid in sft_val_ids:
                    out_files["val"].write(line + "\n")
                    counts["val"] += 1
                elif eid in sft_test_ids:
                    out_files["test"].write(line + "\n")
                    counts["test"] += 1
                else:
                    # exam_id not in metadata (should be rare)
                    counts["unmatched"] += 1

        for fp in out_files.values():
            fp.close()

        split_counts[basename] = counts
        print("  %s: train=%d, val=%d, test=%d, unmatched=%d" % (
            basename, counts["train"], counts["val"],
            counts["test"], counts["unmatched"]))

    # Save split metadata
    split_meta = {
        "seed": seed,
        "cv_split_seed": 42,
        "train_count": len(sft_train_ids),
        "val_count": len(sft_val_ids),
        "test_count": len(sft_test_ids),
        "train_ids": sorted(sft_train_ids),
        "val_ids": sorted(sft_val_ids),
        "test_ids": sorted(sft_test_ids),
        "per_file_counts": split_counts,
    }

    meta_path = os.path.join(output_dir, "split_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)
    print("\nSplit metadata saved to: %s" % meta_path)


def main():
    parser = argparse.ArgumentParser(
        description="Split SFT JSONL into train/val/test")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing sft_*.jsonl files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for split files")
    parser.add_argument("--metadata_csv", required=True,
                        help="Path to metadata_master.csv")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for val/test split")
    args = parser.parse_args()
    split_sft_data(args)


if __name__ == "__main__":
    main()
