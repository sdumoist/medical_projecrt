#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_cls_cache.py
------------------
Build deterministic classification-only cache (.pt) from five-sequence
shoulder MRI NIfTI files.  Serves G1 / G2 binary / ternary training.

Axis convention: all outputs are [Z, H, W]  (slices, rows, cols).
NIfTI on-disk order (H, W, Z) is transposed via utils.io.normalize_axes.

Usage:
    python scripts/build_cls_cache.py \
        --metadata_csv outputs/metadata/metadata_master.csv \
        --output_dir   outputs/cache_cls \
        --target_shape 20 448 448
"""
from __future__ import print_function

import os
import sys
import csv
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom as ndizoom
from tqdm import tqdm

# ---- project imports -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import load_nifti, normalize_axes
from utils.seed import set_seed

# ---- constants -----------------------------------------------------------
SEQUENCE_ORDER = [
    "axial_PD",
    "coronal_PD",
    "coronal_T2WI",
    "sagittal_PD",
    "sagittal_T1WI",
]

PREPROCESS_VERSION = "v2"  # v2: normalize_axes + resize H/W

logger = logging.getLogger("build_cls_cache")

# =========================================================================
# 1. CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build classification cache")
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--split",        type=str, default="all",
                   help="train / val / all")
    p.add_argument("--limit",        type=int, default=None,
                   help="Max cases to process (debug)")
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--target_shape", type=int, nargs=3,
                   default=[20, 448, 448],
                   help="Z H W target shape")
    p.add_argument("--clip_percentile_low",  type=float, default=1.0)
    p.add_argument("--clip_percentile_high", type=float, default=99.0)
    p.add_argument("--normalize_mode", type=str, default="zscore",
                   choices=["zscore", "minmax", "robust_zscore"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =========================================================================
# 2. Metadata
# =========================================================================

REQUIRED_COLUMNS = [
    "exam_id",
    "image_axial_PD", "image_coronal_PD", "image_coronal_T2WI",
    "image_sagittal_PD", "image_sagittal_T1WI",
    "has_all_images", "has_json",
    "exclude_from_main_training", "quality_flag", "split",
]


def load_metadata(metadata_csv: str) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("metadata_master.csv missing columns: %s" % missing)
    return df


def filter_metadata_for_cls_cache(
    df: pd.DataFrame,
    split: str = "all",
    limit: Optional[int] = None,
    allow_quality: tuple = ("high", "medium"),
) -> pd.DataFrame:
    mask = (
        (df["has_all_images"] == 1)
        & (df["has_json"] == 1)
        & (df["exclude_from_main_training"] == 0)
        & (df["quality_flag"].isin(allow_quality))
    )
    if split != "all":
        mask = mask & (df["split"] == split)
    out = df[mask].reset_index(drop=True)
    if limit is not None:
        out = out.head(limit)
    return out


# =========================================================================
# 3. Path helpers
# =========================================================================

def get_sequence_paths_from_row(row: pd.Series) -> Dict[str, str]:
    return {seq: row["image_%s" % seq] for seq in SEQUENCE_ORDER}


# =========================================================================
# 4. Image loading  (with axis normalization)
# =========================================================================

def load_case_images(seq_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Load five sequences and normalize to [Z, H, W]."""
    images = {}
    for seq, path in seq_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError("Missing sequence %s: %s" % (seq, path))
        data, _ = load_nifti(path)
        data = normalize_axes(data)  # (H,W,Z) -> (Z,H,W)
        images[seq] = data.astype(np.float32)
    return images


# =========================================================================
# 5. Spatial transform:  Z pad/crop  +  H/W resize
# =========================================================================

def pad_or_crop_z(volume: np.ndarray, target_z: int) -> np.ndarray:
    """Centre pad/crop along Z (axis 0) only."""
    Z = volume.shape[0]
    if Z == target_z:
        return volume
    elif Z > target_z:
        start = (Z - target_z) // 2
        return volume[start: start + target_z]
    else:
        out = np.zeros((target_z,) + volume.shape[1:], dtype=volume.dtype)
        start = (target_z - Z) // 2
        out[start: start + Z] = volume
        return out


def resize_hw(volume: np.ndarray, target_h: int, target_w: int,
              order: int = 1) -> np.ndarray:
    """Resize H and W dimensions using scipy zoom.

    Args:
        order: interpolation order. 1=linear (for images), 0=nearest (for masks).
    """
    Z, H, W = volume.shape
    if H == target_h and W == target_w:
        return volume
    zoom_factors = (1.0, target_h / H, target_w / W)
    return ndizoom(volume, zoom_factors, order=order).astype(volume.dtype)


# =========================================================================
# 6. Intensity preprocessing
# =========================================================================

def clip_intensity(
    volume: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0
) -> np.ndarray:
    lo = np.percentile(volume, low_pct)
    hi = np.percentile(volume, high_pct)
    return np.clip(volume, lo, hi)


def normalize_volume(volume: np.ndarray, mode: str = "zscore") -> np.ndarray:
    v = volume.astype(np.float32)
    if mode == "zscore":
        mu, std = v.mean(), v.std()
        if std < 1e-8:
            return v - mu
        return (v - mu) / std
    elif mode == "minmax":
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-8:
            return v - vmin
        return (v - vmin) / (vmax - vmin)
    elif mode == "robust_zscore":
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad < 1e-8:
            return v - med
        return (v - med) / (mad * 1.4826)
    else:
        raise ValueError("Unknown normalize_mode: %s" % mode)


# =========================================================================
# 7. Case-level preprocessing
# =========================================================================

def preprocess_case_images(
    image_dict: Dict[str, np.ndarray],
    sequence_order: List[str],
    target_shape: tuple,
    clip_percentile_low: float,
    clip_percentile_high: float,
    normalize_mode: str,
) -> Tuple[np.ndarray, Dict]:
    """Preprocess five [Z,H,W] sequences -> stack to [5, TZ, TH, TW]."""
    TZ, TH, TW = target_shape
    volumes = []
    for seq in sequence_order:
        vol = image_dict[seq]              # [Z, H, W]
        vol = clip_intensity(vol, clip_percentile_low, clip_percentile_high)
        vol = pad_or_crop_z(vol, TZ)       # Z -> TZ
        vol = resize_hw(vol, TH, TW, order=1)  # H,W -> TH,TW
        vol = normalize_volume(vol, normalize_mode)
        volumes.append(vol)

    image_tensor = np.stack(volumes, axis=0)  # [5, TZ, TH, TW]

    spatial_meta = {
        "target_shape": list(target_shape),
        "axis_order": "Z_H_W",
        "preprocess_version": PREPROCESS_VERSION,
    }
    return image_tensor, spatial_meta


# =========================================================================
# 8. Cache record & verification
# =========================================================================

def build_cache_record(
    exam_id: str,
    image_tensor: np.ndarray,
    sequence_order: List[str],
    spatial_meta: Dict,
) -> Dict:
    return {
        "exam_id": exam_id,
        "image": torch.from_numpy(image_tensor).float(),
        "sequence_order": list(sequence_order),
        "spatial_meta": spatial_meta,
    }


def verify_cls_cache_record(record: Dict, target_shape: tuple) -> None:
    img = record["image"]
    TZ, TH, TW = target_shape
    assert img.shape == (5, TZ, TH, TW), \
        "image shape %s != expected (5, %d, %d, %d)" % (img.shape, TZ, TH, TW)
    assert img.dtype == torch.float32, \
        "image dtype %s != float32" % img.dtype
    seq = record["sequence_order"]
    assert len(seq) == 5 and seq == SEQUENCE_ORDER, \
        "sequence_order mismatch: %s" % seq


def save_cls_cache_record(record: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(record, save_path)


# =========================================================================
# 9. Index & failure logging
# =========================================================================

def build_cache_index_row(
    exam_id: str,
    cache_path: str,
    target_shape: tuple,
    success: int = 1,
) -> Dict:
    return {
        "exam_id": exam_id,
        "cache_path": cache_path,
        "cache_exists": int(os.path.exists(cache_path)) if cache_path else 0,
        "shape_Z": target_shape[0],
        "shape_H": target_shape[1],
        "shape_W": target_shape[2],
        "sequence_order": ";".join(SEQUENCE_ORDER),
        "target_shape": "%d,%d,%d" % target_shape,
        "cache_format": "pt",
        "preprocess_version": PREPROCESS_VERSION,
        "success": success,
    }


def append_failed_case(
    failed_log_path: str,
    exam_id: str,
    error_type: str,
    error_message: str,
) -> None:
    write_header = not os.path.exists(failed_log_path)
    with open(failed_log_path, "a") as f:
        w = csv.DictWriter(f, fieldnames=["exam_id", "error_type", "error_message"])
        if write_header:
            w.writeheader()
        w.writerow({
            "exam_id": exam_id,
            "error_type": error_type,
            "error_message": error_message,
        })


# =========================================================================
# 10. Main driver
# =========================================================================

def build_cls_cache(
    metadata_csv: str,
    output_dir: str,
    split: str = "all",
    limit: Optional[int] = None,
    overwrite: bool = False,
    target_shape: tuple = (20, 448, 448),
    clip_percentile_low: float = 1.0,
    clip_percentile_high: float = 99.0,
    normalize_mode: str = "zscore",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "cache_cls_index.csv")
    failed_path = os.path.join(output_dir, "failed_cases.csv")

    df = load_metadata(metadata_csv)
    df = filter_metadata_for_cls_cache(df, split=split, limit=limit)
    total = len(df)
    logger.info("Cases to cache: %d  (split=%s)", total, split)

    index_rows: List[Dict] = []
    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df.iterrows(), total=total, desc="cls-cache"):
        exam_id = row["exam_id"]
        save_path = os.path.join(output_dir, "%s.pt" % exam_id)

        if not overwrite and os.path.exists(save_path):
            index_rows.append(build_cache_index_row(
                exam_id, save_path, target_shape, success=1))
            success_count += 1
            continue

        try:
            seq_paths = get_sequence_paths_from_row(row)
            images = load_case_images(seq_paths)
            tensor, meta = preprocess_case_images(
                images, SEQUENCE_ORDER, target_shape,
                clip_percentile_low, clip_percentile_high, normalize_mode,
            )
            record = build_cache_record(exam_id, tensor, SEQUENCE_ORDER, meta)
            verify_cls_cache_record(record, target_shape)
            save_cls_cache_record(record, save_path)
            index_rows.append(build_cache_index_row(
                exam_id, save_path, target_shape, success=1))
            success_count += 1

        except Exception as e:
            fail_count += 1
            err_type = type(e).__name__
            err_msg = str(e)[:300]
            logger.warning("[FAIL] %s : %s: %s", exam_id, err_type, err_msg)
            append_failed_case(failed_path, exam_id, err_type, err_msg)
            index_rows.append(build_cache_index_row(
                exam_id, "", target_shape, success=0))

    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    logger.info("Done. success=%d  fail=%d  index -> %s",
                success_count, fail_count, index_path)


# =========================================================================
# 11. CLI entry
# =========================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    logger.info("=== build_cls_cache (v2: Z_H_W + resize) ===")
    logger.info("metadata_csv : %s", args.metadata_csv)
    logger.info("output_dir   : %s", args.output_dir)
    logger.info("target_shape : %s  (Z, H, W)", args.target_shape)
    logger.info("normalize    : %s", args.normalize_mode)
    logger.info("split        : %s", args.split)

    build_cls_cache(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        overwrite=args.overwrite,
        target_shape=tuple(args.target_shape),
        clip_percentile_low=args.clip_percentile_low,
        clip_percentile_high=args.clip_percentile_high,
        normalize_mode=args.normalize_mode,
    )


if __name__ == "__main__":
    main()
