#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_axes.py
--------------
Minimal verification: confirm that normalize_axes converts NIfTI (H,W,Z)
to project-standard (Z,H,W) for both images and masks.

Usage (in docker with srre env):
    python scripts/verify_axes.py \
        --metadata_csv outputs/metadata/metadata_master.csv \
        --limit 3
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import load_nifti, normalize_axes, SEQUENCE_TYPES, DISEASES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()

    df = pd.read_csv(args.metadata_csv)
    df = df[df["has_all_images"] == 1].head(args.limit)

    seqs = list(SEQUENCE_TYPES)
    all_ok = True

    for _, row in df.iterrows():
        eid = row["exam_id"]
        print("=== %s ===" % eid)

        for seq in seqs:
            path = row["image_%s" % seq]
            raw, _ = load_nifti(path)
            raw = np.squeeze(raw)
            zhw = normalize_axes(raw)

            # raw should be (H, W, Z) where H,W >> Z
            # zhw should be (Z, H, W) where Z << H,W
            raw_z = raw.shape[2]  # last dim = slices
            zhw_z = zhw.shape[0]  # first dim = slices

            ok = (zhw_z == raw_z) and (zhw_z < zhw.shape[1]) and (zhw_z < zhw.shape[2])
            status = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False

            print("  %-15s  raw=(%3d,%3d,%2d)  ->  zhw=(%2d,%3d,%3d)  [%s]" % (
                seq,
                raw.shape[0], raw.shape[1], raw.shape[2],
                zhw.shape[0], zhw.shape[1], zhw.shape[2],
                status))

        # Check one mask if available
        for disease in DISEASES:
            col = "mask_path_%s" % disease
            if col not in row.index:
                continue
            mpath = row[col]
            if pd.isna(mpath) or not str(mpath).strip():
                continue
            mpath = str(mpath).strip()
            if not os.path.exists(mpath):
                continue

            raw_m, _ = load_nifti(mpath)
            raw_m = np.squeeze(raw_m)
            zhw_m = normalize_axes(raw_m)

            ok = (zhw_m.shape[0] == raw_m.shape[2])
            status = "OK" if ok else "FAIL"
            if not ok:
                all_ok = False

            fg = (zhw_m > 0).sum()
            print("  mask %-5s      raw=(%3d,%3d,%2d)  ->  zhw=(%2d,%3d,%3d)  fg=%d  [%s]" % (
                disease,
                raw_m.shape[0], raw_m.shape[1], raw_m.shape[2],
                zhw_m.shape[0], zhw_m.shape[1], zhw_m.shape[2],
                fg, status))
            break  # just check one mask per case

    print()
    if all_ok:
        print("ALL CHECKS PASSED: axis 0 is Z (slices), axes 1,2 are H,W.")
    else:
        print("SOME CHECKS FAILED! Review output above.")


if __name__ == "__main__":
    main()
