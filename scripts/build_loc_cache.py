#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_loc_cache.py
------------------
Build localizer cache (.pt) with **image-mask aligned spatial transforms**.
For each disease, the mask shares the SAME spatial transform as its anchor
sequence image.  Serves G1-L / G2-L, key-slice, ROI, and lesion-token
workflows.

Usage:
    python scripts/build_loc_cache.py \
        --metadata_csv outputs/metadata/metadata_master.csv \
        --output_dir   outputs/cache_loc \
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

# ---- project imports -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import load_nifti
from utils.seed import set_seed

# ---- constants (hardcoded, no external implicit order) -------------------
SEQUENCE_ORDER = [
    "axial_PD",
    "coronal_PD",
    "coronal_T2WI",
    "sagittal_PD",
    "sagittal_T1WI",
]

DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]

DISEASE_ANCHOR_SEQ = {
    "SST":  "coronal_PD",
    "IST":  "axial_PD",
    "SSC":  "axial_PD",
    "LHBT": "coronal_PD",
    "IGHL": "coronal_PD",
    "RIPI": "sagittal_PD",
    "GHOA": "coronal_PD",
}

PREPROCESS_VERSION = "v1"

logger = logging.getLogger("build_loc_cache")

# =========================================================================
# 1. CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build localizer cache")
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--split",        type=str, default="all")
    p.add_argument("--limit",        type=int, default=None)
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--target_shape", type=int, nargs=3, default=[32, 96, 96])
    p.add_argument("--clip_percentile_low",  type=float, default=1.0)
    p.add_argument("--clip_percentile_high", type=float, default=99.0)
    p.add_argument("--normalize_mode", type=str, default="zscore",
                   choices=["zscore", "minmax", "robust_zscore"])
    p.add_argument("--diseases", type=str, nargs="+", default=None,
                   help="Subset of diseases (default: all 7)")
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


def filter_metadata_for_loc_cache(
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
    # require at least one mask
    if "has_any_mask" in df.columns:
        mask = mask & (df["has_any_mask"] == 1)
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


def get_mask_paths_from_row(
    row: pd.Series, diseases: List[str]
) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {}
    for d in diseases:
        col = "mask_path_%s" % d
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                paths[d] = str(val).strip()
            else:
                paths[d] = None
        else:
            paths[d] = None
    return paths


# =========================================================================
# 4. Image / mask loading
# =========================================================================

def load_case_images(seq_paths: Dict[str, str]) -> Dict[str, np.ndarray]:
    images = {}
    for seq, path in seq_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError("Missing sequence %s: %s" % (seq, path))
        data, _ = load_nifti(path)
        images[seq] = data.astype(np.float32)
    return images


def load_case_masks(
    mask_paths: Dict[str, Optional[str]]
) -> Dict[str, Optional[np.ndarray]]:
    masks: Dict[str, Optional[np.ndarray]] = {}
    for disease, path in mask_paths.items():
        if path is None or not os.path.exists(path):
            masks[disease] = None
        else:
            data, _ = load_nifti(path)
            masks[disease] = data.astype(np.int16)
    return masks


# =========================================================================
# 5. Deterministic spatial transform
# =========================================================================

def compute_spatial_transform_params_from_volume(
    volume: np.ndarray,
    target_shape: tuple,
) -> Dict:
    """Compute centre-crop parameters from a single reference volume.
    These parameters must be shared by image AND mask for alignment.
    """
    D, H, W = volume.shape
    TD, TH, TW = target_shape
    return {
        "crop_start": [
            max(0, (D - TD) // 2),
            max(0, (H - TH) // 2),
            max(0, (W - TW) // 2),
        ],
        "crop_size": [min(D, TD), min(H, TH), min(W, TW)],
        "target_shape": list(target_shape),
    }


def _apply_transform(
    vol: np.ndarray,
    params: Dict,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Apply crop + pad using externally-provided params (no recomputation)."""
    d0, h0, w0 = params["crop_start"]
    cd, ch, cw = params["crop_size"]
    TD, TH, TW = params["target_shape"]

    cropped = vol[d0: d0 + cd, h0: h0 + ch, w0: w0 + cw]

    if cropped.shape == (TD, TH, TW):
        return cropped

    out = np.full((TD, TH, TW), pad_value, dtype=vol.dtype)
    pd_ = (TD - cropped.shape[0]) // 2
    ph_ = (TH - cropped.shape[1]) // 2
    pw_ = (TW - cropped.shape[2]) // 2
    out[pd_: pd_ + cropped.shape[0],
        ph_: ph_ + cropped.shape[1],
        pw_: pw_ + cropped.shape[2]] = cropped
    return out


def apply_spatial_transform_to_image(
    volume: np.ndarray, transform_params: Dict
) -> np.ndarray:
    return _apply_transform(volume, transform_params, pad_value=0.0)


def apply_spatial_transform_to_mask(
    mask: np.ndarray, transform_params: Dict
) -> np.ndarray:
    """Mask uses the SAME params as its anchor image. Pad with 0 (background)."""
    return _apply_transform(mask, transform_params, pad_value=0)


# =========================================================================
# 6. Intensity preprocessing (image only, NOT mask)
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
# 7. Case-level preprocessing  (two layers)
# =========================================================================

def preprocess_case_for_localizer(
    image_dict: Dict[str, np.ndarray],
    mask_dict: Dict[str, Optional[np.ndarray]],
    sequence_order: List[str],
    diseases: List[str],
    target_shape: tuple,
    clip_percentile_low: float,
    clip_percentile_high: float,
    normalize_mode: str,
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Layer 1: Build global 5-sequence image tensor [5, D, H, W].
             Each sequence independently centre-cropped to target_shape.
    Layer 2: For each disease, compute anchor-space transform from the
             RAW anchor image, then apply that SAME transform to both
             the anchor image and the disease mask.

    Returns:
        image_tensor:  np.ndarray [5, D, H, W]
        processed_masks: {disease: np.ndarray [D,H,W] or None}
        spatial_meta: {
            "global_image": {...},
            "per_disease": {disease: {...}, ...}
        }
    """
    # ---- Layer 1: global image tensor ------------------------------------
    processed_images = []
    global_per_seq_params = {}
    for seq in sequence_order:
        raw_vol = image_dict[seq]
        vol = clip_intensity(raw_vol, clip_percentile_low, clip_percentile_high)
        seq_params = compute_spatial_transform_params_from_volume(
            vol, target_shape)
        vol = apply_spatial_transform_to_image(vol, seq_params)
        vol = normalize_volume(vol, normalize_mode)
        processed_images.append(vol)
        global_per_seq_params[seq] = seq_params

    image_tensor = np.stack(processed_images, axis=0)  # [5, D, H, W]

    # ---- Layer 2: per-disease anchor-space mask alignment ----------------
    processed_masks: Dict[str, Optional[np.ndarray]] = {}
    per_disease_meta: Dict[str, Dict] = {}

    for disease in diseases:
        anchor_seq = DISEASE_ANCHOR_SEQ[disease]
        raw_anchor = image_dict[anchor_seq]  # original raw volume

        # compute transform from RAW anchor (before clip/norm)
        anchor_params = compute_spatial_transform_params_from_volume(
            raw_anchor, target_shape)
        per_disease_meta[disease] = {
            "anchor_seq": anchor_seq,
            "transform_params": anchor_params,
        }

        m = mask_dict.get(disease)
        if m is None:
            processed_masks[disease] = None
        else:
            # apply the SAME anchor_params to the mask
            processed_masks[disease] = apply_spatial_transform_to_mask(
                m, anchor_params)

    spatial_meta = {
        "global_image": {
            "target_shape": list(target_shape),
            "preprocess_version": PREPROCESS_VERSION,
        },
        "per_disease": per_disease_meta,
    }
    return image_tensor, processed_masks, spatial_meta


# =========================================================================
# 8. Key-slice & bbox extraction (from aligned masks)
# =========================================================================

def extract_key_slice_from_mask(mask: np.ndarray) -> int:
    if mask is None:
        return -1
    fg = (mask > 0)
    if not fg.any():
        return -1
    sums = fg.sum(axis=(1, 2))
    return int(np.argmax(sums))


def extract_bbox_from_mask(
    mask: np.ndarray, margin: int = 2
) -> Optional[List[int]]:
    if mask is None:
        return None
    fg = np.where(mask > 0)
    if len(fg[0]) == 0:
        return None
    D, H, W = mask.shape
    z1 = max(0, int(fg[0].min()) - margin)
    z2 = min(D - 1, int(fg[0].max()) + margin)
    y1 = max(0, int(fg[1].min()) - margin)
    y2 = min(H - 1, int(fg[1].max()) + margin)
    x1 = max(0, int(fg[2].min()) - margin)
    x2 = min(W - 1, int(fg[2].max()) + margin)
    return [z1, z2, y1, y2, x1, x2]


def extract_localizer_targets(
    mask_dict: Dict[str, Optional[np.ndarray]]
) -> Tuple[Dict[str, int], Dict[str, Optional[List[int]]]]:
    key_slices: Dict[str, int] = {}
    roi_boxes: Dict[str, Optional[List[int]]] = {}
    for disease, m in mask_dict.items():
        if m is None:
            key_slices[disease] = -1
            roi_boxes[disease] = None
        else:
            key_slices[disease] = extract_key_slice_from_mask(m)
            roi_boxes[disease] = extract_bbox_from_mask(m)
    return key_slices, roi_boxes


# =========================================================================
# 9. Cache record
# =========================================================================

def build_loc_cache_record(
    exam_id: str,
    image_tensor: np.ndarray,
    processed_masks: Dict[str, Optional[np.ndarray]],
    key_slices: Dict[str, int],
    roi_boxes: Dict[str, Optional[List[int]]],
    sequence_order: List[str],
    spatial_meta: Dict,
) -> Dict:
    mask_tensors = {}
    for d, m in processed_masks.items():
        if m is None:
            mask_tensors[d] = None
        else:
            mask_tensors[d] = torch.from_numpy(m.astype(np.int64))

    return {
        "exam_id": exam_id,
        "image": torch.from_numpy(image_tensor).float(),
        "mask": mask_tensors,
        "key_slices": key_slices,
        "roi_boxes": roi_boxes,
        "sequence_order": list(sequence_order),
        "spatial_meta": spatial_meta,
    }


def save_loc_cache_record(record: Dict, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(record, save_path)


# =========================================================================
# 10. Index & failure logging
# =========================================================================

def build_loc_cache_index_row(
    exam_id: str,
    cache_path: str,
    key_slices: Dict[str, int],
    roi_boxes: Dict[str, Optional[List[int]]],
    processed_masks: Dict[str, Optional[np.ndarray]],
    diseases: List[str],
) -> Dict:
    row: Dict = {
        "exam_id": exam_id,
        "cache_path": cache_path,
    }
    available = 0
    for d in diseases:
        m = processed_masks.get(d)
        has = int(m is not None and (m > 0).any())
        row["%s_mask_available" % d] = has
        row["%s_key_slice" % d] = key_slices.get(d, -1)
        available += has
    row["num_available_masks"] = available
    row["has_any_mask"] = int(available > 0)
    row["success"] = 1
    return row


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
# 11. Main driver
# =========================================================================

def build_loc_cache(
    metadata_csv: str,
    output_dir: str,
    split: str = "all",
    limit: Optional[int] = None,
    overwrite: bool = False,
    target_shape: tuple = (32, 96, 96),
    clip_percentile_low: float = 1.0,
    clip_percentile_high: float = 99.0,
    normalize_mode: str = "zscore",
    diseases: Optional[List[str]] = None,
) -> None:
    if diseases is None:
        diseases = list(DISEASES)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "cache_loc_index.csv")
    failed_path = os.path.join(output_dir, "failed_cases.csv")

    df = load_metadata(metadata_csv)
    df = filter_metadata_for_loc_cache(df, split=split, limit=limit)
    total = len(df)
    logger.info("Cases to cache: %d  (split=%s)", total, split)

    index_rows: List[Dict] = []
    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df.iterrows(), total=total, desc="loc-cache"):
        exam_id = row["exam_id"]
        save_path = os.path.join(output_dir, "%s.pt" % exam_id)

        if not overwrite and os.path.exists(save_path):
            placeholder_masks = {d: None for d in diseases}
            placeholder_ks = {d: -1 for d in diseases}
            index_rows.append(build_loc_cache_index_row(
                exam_id, save_path, placeholder_ks, {},
                placeholder_masks, diseases))
            success_count += 1
            continue

        try:
            # load raw data
            seq_paths = get_sequence_paths_from_row(row)
            images = load_case_images(seq_paths)
            mask_paths = get_mask_paths_from_row(row, diseases)
            masks = load_case_masks(mask_paths)

            # preprocess with anchor-space alignment
            img_tensor, proc_masks, spatial_meta = preprocess_case_for_localizer(
                images, masks, SEQUENCE_ORDER, diseases, target_shape,
                clip_percentile_low, clip_percentile_high, normalize_mode,
            )

            # extract key-slices / bbox from ALIGNED masks
            key_slices, roi_boxes = extract_localizer_targets(proc_masks)

            # save
            record = build_loc_cache_record(
                exam_id, img_tensor, proc_masks,
                key_slices, roi_boxes, SEQUENCE_ORDER, spatial_meta,
            )
            save_loc_cache_record(record, save_path)

            index_rows.append(build_loc_cache_index_row(
                exam_id, save_path, key_slices, roi_boxes,
                proc_masks, diseases))
            success_count += 1

        except Exception as e:
            fail_count += 1
            err_type = type(e).__name__
            err_msg = str(e)[:300]
            logger.warning("[FAIL] %s : %s: %s", exam_id, err_type, err_msg)
            append_failed_case(failed_path, exam_id, err_type, err_msg)

    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    logger.info("Done. success=%d  fail=%d  index -> %s",
                success_count, fail_count, index_path)


# =========================================================================
# 12. CLI entry
# =========================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    diseases = args.diseases if args.diseases else list(DISEASES)

    logger.info("=== build_loc_cache ===")
    logger.info("metadata_csv : %s", args.metadata_csv)
    logger.info("output_dir   : %s", args.output_dir)
    logger.info("target_shape : %s", args.target_shape)
    logger.info("diseases     : %s", diseases)
    logger.info("sequence_order: %s", SEQUENCE_ORDER)

    build_loc_cache(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        overwrite=args.overwrite,
        target_shape=tuple(args.target_shape),
        clip_percentile_low=args.clip_percentile_low,
        clip_percentile_high=args.clip_percentile_high,
        normalize_mode=args.normalize_mode,
        diseases=diseases,
    )


if __name__ == "__main__":
    main()
