#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_branch_features.py
--------------------------
Pre-extract MRI-CV branch features for faster SFT Stage 1 training.

Instead of loading full MRI volumes and running the frozen MRI-CV model
every training step, this script pre-computes and saves the branch features
as compact .npz files (~7KB per exam vs ~50MB for cache .pt).

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/extract_branch_features.py \
        --config configs/sft_stage1_frozen.yaml \
        --output_dir outputs/branch_features \
        --batch_size 8
"""
from __future__ import print_function

import os
import sys
import csv
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from models import create_model
from data.shoulder_dataset import ShoulderCacheDataset
from data.label_mapper import LabelMapper


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract(args):
    config = load_config(args.config)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Load MRI-CV model
    mri_cv_cfg = config["mri_cv"]
    model = create_model(
        encoder=mri_cv_cfg["encoder"],
        num_diseases=7,
        pretrained=False,
        use_localizer=mri_cv_cfg.get("use_localizer", False),
        num_classes=2,
    )

    ckpt_path = mri_cv_cfg.get("checkpoint_path")
    if ckpt_path:
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("Loaded: %s" % ckpt_path)

    model = model.to(device)
    model.eval()

    feat_dim = model.encoders["axial_PD"].num_features
    print("Feature dim: %d" % feat_dim)

    # Load metadata for exam list
    data_cfg = config["data"]
    cache_root = data_cfg.get("cache_root", "outputs/cache_cls")
    if not os.path.isabs(cache_root):
        cache_root = os.path.join(PROJECT_ROOT, cache_root)

    cache_index = data_cfg.get("cache_index")
    if cache_index and not os.path.isabs(cache_index):
        cache_index = os.path.join(PROJECT_ROOT, cache_index)

    # Read exam IDs from cache index
    exam_ids = []
    with open(cache_index, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exam_ids.append(row["exam_id"])

    print("Total exams: %d" % len(exam_ids))

    # Simple dataset that just loads images
    label_mapper = LabelMapper(mode="binary")
    # Need a raw_labels_lookup - load from metadata
    metadata_csv = os.path.join(PROJECT_ROOT, "outputs/metadata/metadata_master.csv")
    raw_labels_lookup = {}
    if os.path.exists(metadata_csv):
        with open(metadata_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row["exam_id"]
                from utils.io import DISEASES
                raw_labels = {d: int(row["raw_label_" + d]) for d in DISEASES}
                raw_labels_lookup[eid] = raw_labels

    dataset = ShoulderCacheDataset(
        cache_root=cache_root,
        exam_ids=set(exam_ids),
        label_mapper=label_mapper,
        raw_labels_lookup=raw_labels_lookup,
        cache_mode="cls",
        project_root=PROJECT_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Extract features
    count = 0
    index_rows = []

    try:
        from tqdm import tqdm
        iter_ = tqdm(loader, desc="Extracting")
    except ImportError:
        iter_ = loader

    with torch.no_grad():
        for batch in iter_:
            images = batch["image"].to(device)
            batch_eids = batch["exam_id"]

            cv_out = model(images)

            sag = cv_out["sag_feat"].cpu().numpy()
            cor = cv_out["cor_feat"].cpu().numpy()
            axi = cv_out["axi_feat"].cpu().numpy()

            for i, eid in enumerate(batch_eids):
                out_path = os.path.join(output_dir, "%s.npz" % eid)
                np.savez_compressed(
                    out_path,
                    sag_feat=sag[i],
                    cor_feat=cor[i],
                    axi_feat=axi[i],
                )
                index_rows.append({"exam_id": eid, "feature_path": out_path})
                count += 1

    # Write index
    index_path = os.path.join(output_dir, "branch_features_index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["exam_id", "feature_path"])
        writer.writeheader()
        writer.writerows(index_rows)

    print("\nExtracted %d exam features to %s" % (count, output_dir))
    print("Index: %s" % index_path)
    print("Feature shape: [3, %d] (sag/cor/axi)" % feat_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract MRI-CV branch features")
    parser.add_argument("--config", "-c", required=True,
                        help="SFT config YAML (for MRI-CV model spec)")
    parser.add_argument("--output_dir", "-o", default="outputs/branch_features",
                        help="Output directory for .npz files")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
