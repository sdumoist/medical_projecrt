"""
Build 2D ROI grounding targets from cache_loc masks and key_slices.

For each (exam_id, disease):
  1. Load cache_loc .pt file
  2. On the key_slice layer, find the disease mask
  3. If key_slice layer mask is empty, fall back to the layer with max mask area
  4. Take the largest connected component
  5. Compute bounding box, normalize to [x1, y1, x2, y2] in [0, 1]

Output:
  outputs_clean/grounding/grounding_targets.json  -- per (exam_id, disease) dict
  outputs_clean/grounding/grounding_targets.csv   -- flat CSV for inspection

Usage:
    PYTHONPATH=. python scripts/build_grounding_targets.py \
        --cache_root outputs_clean/cache_loc \
        --output_dir outputs_clean/grounding \
        [--max_samples N]
"""
import os
import sys
import json
import csv
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.constants import DISEASES


def load_cache_pt(cache_path):
    """Load .pt cache file and return (image, mask_dict, key_slices_dict)."""
    import torch
    record = torch.load(cache_path, map_location="cpu", weights_only=False)

    # mask: dict {disease: [Z, H, W] tensor} or [7, Z, H, W]
    mask_data = record.get("mask", None)

    # key_slices: dict {disease: int} or list[7] or tensor[7]
    ks_data = record.get("key_slices", None)

    return record.get("image"), mask_data, ks_data


def parse_key_slices(ks_data):
    """Parse key_slices to dict {disease: int}. Returns -1 for missing."""
    import torch
    if ks_data is None:
        return {d: -1 for d in DISEASES}
    if isinstance(ks_data, dict):
        return {d: int(ks_data.get(d, -1)) for d in DISEASES}
    if isinstance(ks_data, (list, tuple)):
        return {d: int(ks_data[i]) if i < len(ks_data) else -1
                for i, d in enumerate(DISEASES)}
    if isinstance(ks_data, torch.Tensor):
        ks_arr = ks_data.numpy()
        return {d: int(ks_arr[i]) if i < len(ks_arr) else -1
                for i, d in enumerate(DISEASES)}
    return {d: -1 for d in DISEASES}


def parse_mask(mask_data, disease_idx):
    """Parse mask_data for one disease. Returns [Z, H, W] numpy uint8."""
    import torch
    if mask_data is None:
        return None

    if isinstance(mask_data, dict):
        # key could be disease name or index
        disease_name = DISEASES[disease_idx]
        m = mask_data.get(disease_name, mask_data.get(disease_idx, None))
        if m is None:
            return None
        if isinstance(m, torch.Tensor):
            return m.numpy().astype(np.uint8)
        return np.array(m, dtype=np.uint8)

    if isinstance(mask_data, torch.Tensor):
        # [7, Z, H, W]
        if mask_data.ndim == 4 and mask_data.shape[0] >= len(DISEASES):
            return mask_data[disease_idx].numpy().astype(np.uint8)
        return None

    if isinstance(mask_data, np.ndarray):
        if mask_data.ndim == 4:
            return mask_data[disease_idx].astype(np.uint8)
        return None

    return None


def find_bbox_2d(mask_2d):
    """
    Find bounding box of non-zero region in a 2D mask [H, W].
    Returns (y1, x1, y2, x2) in pixel coords, or None if mask is empty.
    """
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return int(rmin), int(cmin), int(rmax), int(cmax)


def largest_connected_component_2d(mask_2d):
    """Return mask of only the largest connected component in a 2D binary mask."""
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask_2d)
        if n == 0:
            return mask_2d
        # Find largest component
        sizes = ndimage.sum(mask_2d, labeled, range(1, n + 1))
        largest = np.argmax(sizes) + 1
        return (labeled == largest).astype(np.uint8)
    except ImportError:
        # scipy not available, return as-is
        return mask_2d


def compute_roi_from_mask(mask_3d, key_slice_z):
    """
    Given a [Z, H, W] mask and the key slice index (Z-space),
    compute a normalized 2D ROI box.

    Returns dict with:
        has_target: bool
        source_slice: int (which Z slice was used)
        used_fallback_slice: bool
        box_2d: [x1, y1, x2, y2] normalized to [0, 1] or None
        box_area: float (proportion of image area)
    """
    Z, H, W = mask_3d.shape
    result = {
        "has_target": False,
        "source_slice": int(key_slice_z),
        "used_fallback_slice": False,
        "box_2d": None,
        "box_area": 0.0,
    }

    # 1. Try key_slice layer first
    use_z = None
    if 0 <= key_slice_z < Z:
        slice_mask = mask_3d[key_slice_z]
        if slice_mask.any():
            use_z = key_slice_z

    # 2. Fallback: find Z with max mask area
    if use_z is None:
        area_per_z = mask_3d.sum(axis=(1, 2))
        if area_per_z.max() > 0:
            use_z = int(np.argmax(area_per_z))
            result["used_fallback_slice"] = True
            result["source_slice"] = use_z

    if use_z is None:
        return result

    slice_mask = mask_3d[use_z]

    # 3. Largest connected component
    slice_mask = largest_connected_component_2d(slice_mask)

    # 4. Bounding box
    bbox = find_bbox_2d(slice_mask)
    if bbox is None:
        return result

    y1, x1, y2, x2 = bbox
    # Normalize to [0, 1]
    box_2d = [
        round(x1 / W, 4),
        round(y1 / H, 4),
        round(x2 / W, 4),
        round(y2 / H, 4),
    ]
    area = ((x2 - x1) * (y2 - y1)) / (H * W)

    result["has_target"] = True
    result["box_2d"] = box_2d
    result["box_area"] = round(float(area), 6)
    return result


def process_exam(cache_path, exam_id, key_slices_dict):
    """Process one exam_id and return per-disease ROI info."""
    _, mask_data, ks_from_cache = load_cache_pt(cache_path)

    # Prefer externally passed key_slices; fall back to cache
    if key_slices_dict is not None:
        ks_dict = key_slices_dict
    else:
        ks_dict = parse_key_slices(ks_from_cache)

    exam_result = {"exam_id": exam_id}

    for i, disease in enumerate(DISEASES):
        mask_3d = parse_mask(mask_data, i)
        key_slice_z = ks_dict.get(disease, -1)

        if mask_3d is None or mask_3d.sum() == 0:
            exam_result[disease] = {
                "has_target": False,
                "key_slice": int(key_slice_z),
                "source_slice": int(key_slice_z),
                "used_fallback_slice": False,
                "box_2d": None,
                "box_area": 0.0,
            }
            continue

        roi = compute_roi_from_mask(mask_3d, key_slice_z)
        roi["key_slice"] = int(key_slice_z)
        exam_result[disease] = roi

    return exam_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default="outputs_clean/cache_loc")
    parser.add_argument("--output_dir", default="outputs_clean/grounding")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    # Resolve paths relative to project root
    cache_root = args.cache_root if os.path.isabs(args.cache_root) \
        else os.path.join(PROJECT_ROOT, args.cache_root)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(PROJECT_ROOT, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Read index
    index_path = os.path.join(cache_root, "cache_loc_index.csv")
    if not os.path.exists(index_path):
        print("ERROR: cache_loc_index.csv not found at %s" % index_path)
        sys.exit(1)

    records = []
    with open(index_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if args.max_samples > 0:
        records = records[:args.max_samples]
        print("Processing %d samples (limited by --max_samples)" % len(records))
    else:
        print("Processing %d samples" % len(records))

    all_results = []
    stats = {d: {"total": 0, "has_box": 0, "fallback": 0} for d in DISEASES}

    for idx, row in enumerate(records):
        exam_id = row["exam_id"]
        cache_path = row.get("cache_path", "")
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(PROJECT_ROOT, cache_path)

        if not os.path.exists(cache_path):
            print("[%d/%d] SKIP %s — file not found: %s" % (
                idx + 1, len(records), exam_id, cache_path))
            continue

        try:
            result = process_exam(cache_path, exam_id, None)
            all_results.append(result)

            for disease in DISEASES:
                d_info = result.get(disease, {})
                stats[disease]["total"] += 1
                if d_info.get("has_target"):
                    stats[disease]["has_box"] += 1
                if d_info.get("used_fallback_slice"):
                    stats[disease]["fallback"] += 1

        except Exception as e:
            print("[%d/%d] ERROR %s: %s" % (idx + 1, len(records), exam_id, e))
            continue

        if (idx + 1) % 500 == 0:
            print("[%d/%d] processed..." % (idx + 1, len(records)))

    # Save JSON
    json_path = os.path.join(output_dir, "grounding_targets.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nSaved JSON: %s (%d exams)" % (json_path, len(all_results)))

    # Save CSV (flat)
    csv_path = os.path.join(output_dir, "grounding_targets.csv")
    fieldnames = ["exam_id", "disease", "has_target", "key_slice",
                  "source_slice", "used_fallback_slice",
                  "box_x1", "box_y1", "box_x2", "box_y2", "box_area"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exam in all_results:
            eid = exam["exam_id"]
            for disease in DISEASES:
                d_info = exam.get(disease, {})
                box = d_info.get("box_2d")
                writer.writerow({
                    "exam_id": eid,
                    "disease": disease,
                    "has_target": int(d_info.get("has_target", False)),
                    "key_slice": d_info.get("key_slice", -1),
                    "source_slice": d_info.get("source_slice", -1),
                    "used_fallback_slice": int(d_info.get("used_fallback_slice", False)),
                    "box_x1": box[0] if box else "",
                    "box_y1": box[1] if box else "",
                    "box_x2": box[2] if box else "",
                    "box_y2": box[3] if box else "",
                    "box_area": d_info.get("box_area", 0),
                })
    print("Saved CSV: %s" % csv_path)

    # Print stats
    print("\n=== Stats ===")
    print("%-8s  %6s  %8s  %8s" % ("Disease", "Total", "HasBox", "Fallback"))
    for d in DISEASES:
        s = stats[d]
        print("%-8s  %6d  %8d  %8d" % (
            d, s["total"], s["has_box"], s["fallback"]))


if __name__ == "__main__":
    main()
