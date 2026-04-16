#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_cls_cache.py
------------------
Build deterministic classification-only cache (.pt) from five-sequence
shoulder MRI NIfTI files.  Serves G1 / G2 binary / ternary training.

Usage:
    python scripts/build_cls_cache.py \
        --metadata_csv outputs/metadata/metadata_master.csv \
        --output_dir   outputs/cache_cls \
        --target_shape 32 96 96
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
from tqdm import tqdm

# ---- project imports (add project root to path) -------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import load_nifti
from utils.seed import set_seed

# ---- constants (hardcoded, do NOT depend on external implicit order) -----
SEQUENCE_ORDER = [
    "axial_PD",
    "coronal_PD",
    "coronal_T2WI",
    "sagittal_PD",
    "sagittal_T1WI",
]

PREPROCESS_VERSION = "v1"

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
                   default=[32, 96, 96],
                   help="D H W for centre-crop / pad")
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
# 4. Image loading
# =========================================================================

def load_case_images(seq_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
    images = {}
    for seq, path in seq_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError("Missing sequence %s: %s" % (seq, path))
        data, _ = load_nifti(path)
        images[seq] = data.astype(np.float32)
    return images


# =========================================================================
# 5. Spatial transform
# =========================================================================

def compute_center_crop_params(
    shape: tuple, target_shape: tuple
) -> Dict[str, int]:
    D, H, W = shape
    TD, TH, TW = target_shape
    d_start = max(0, (D - TD) // 2)
    h_start = max(0, (H - TH) // 2)
    w_start = max(0, (W - TW) // 2)
    return dict(
        d_start=d_start, h_start=h_start, w_start=w_start,
        td=TD, th=TH, tw=TW,
    )


def apply_crop_or_pad(
    volume: np.ndarray,
    crop_params: Dict[str, int],
    target_shape: tuple,
) -> np.ndarray:
    D, H, W = volume.shape
    TD, TH, TW = target_shape
    ds = crop_params["d_start"]
    hs = crop_params["h_start"]
    ws = crop_params["w_start"]

    cropped = volume[
        ds: ds + min(D, TD),
        hs: hs + min(H, TH),
        ws: ws + min(W, TW),
    ]

    cd, ch, cw = cropped.shape
    if cd == TD and ch == TH and cw == TW:
        return cropped

    out = np.zeros((TD, TH, TW), dtype=volume.dtype)
    pd_ = (TD - cd) // 2
    ph_ = (TH - ch) // 2
    pw_ = (TW - cw) // 2
    out[pd_: pd_ + cd, ph_: ph_ + ch, pw_: pw_ + cw] = cropped
    return out


# =========================================================================
# 6. Intensity preprocessing
# =========================================================================

def clip_intensity(
    volume: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
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
    volumes = []
    for seq in sequence_order:
        vol = image_dict[seq]
        vol = clip_intensity(vol, clip_percentile_low, clip_percentile_high)
        crop_p = compute_center_crop_params(vol.shape, target_shape)
        vol = apply_crop_or_pad(vol, crop_p, target_shape)
        vol = normalize_volume(vol, normalize_mode)
        volumes.append(vol)

    image_tensor = np.stack(volumes, axis=0)  # [5, D, H, W]

    spatial_meta = {
        "target_shape": list(target_shape),
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
        "image": torch.from_numpy(image_tensor).float(),  # [5,D,H,W]
        "sequence_order": list(sequence_order),
        "spatial_meta": spatial_meta,
    }


def verify_cls_cache_record(record: Dict, target_shape: tuple) -> None:
    """Minimal sanity check after building a cache record."""
    img = record["image"]
    TD, TH, TW = target_shape
    assert img.shape == (5, TD, TH, TW), \
        "image shape %s != expected (5, %d, %d, %d)" % (img.shape, TD, TH, TW)
    assert img.dtype == torch.float32, \
        "image dtype %s != float32" % img.dtype
    seq = record["sequence_order"]
    assert len(seq) == 5, "sequence_order length %d != 5" % len(seq)
    assert seq == SEQUENCE_ORDER, \
        "sequence_order mismatch: %s" % seq


def save_cls_cache_record(
    record: Dict, save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(record, save_path)


# =========================================================================
# 9. Index & failure logging
# =========================================================================

def build_cache_index_row(
    exam_id: str,
    cache_path: str,
    target_shape: tuple,
    cache_format: str,
    success: int = 1,
) -> Dict:
    return {
        "exam_id": exam_id,
        "cache_path": cache_path,
        "cache_exists": int(os.path.exists(cache_path)) if cache_path else 0,
        "shape_D": target_shape[0],
        "shape_H": target_shape[1],
        "shape_W": target_shape[2],
        "sequence_order": ";".join(SEQUENCE_ORDER),
        "target_shape": "%d,%d,%d" % target_shape,
        "cache_format": cache_format,
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
    target_shape: tuple = (32, 96, 96),
    clip_percentile_low: float = 1.0,
    clip_percentile_high: float = 99.0,
    normalize_mode: str = "zscore",
) -> None:
    cache_format = "pt"
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
                exam_id, save_path, target_shape, cache_format, success=1))
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
                exam_id, save_path, target_shape, cache_format, success=1))
            success_count += 1

        except Exception as e:
            fail_count += 1
            err_type = type(e).__name__
            err_msg = str(e)[:300]
            logger.warning("[FAIL] %s : %s: %s", exam_id, err_type, err_msg)
            append_failed_case(failed_path, exam_id, err_type, err_msg)
            index_rows.append(build_cache_index_row(
                exam_id, "", target_shape, cache_format, success=0))

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

    logger.info("=== build_cls_cache ===")
    logger.info("metadata_csv : %s", args.metadata_csv)
    logger.info("output_dir   : %s", args.output_dir)
    logger.info("target_shape : %s", args.target_shape)
    logger.info("normalize    : %s", args.normalize_mode)
    logger.info("split        : %s", args.split)
    logger.info("sequence_order: %s", SEQUENCE_ORDER)

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
