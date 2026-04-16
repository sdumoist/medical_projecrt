"""
Rebuild metadata_master.csv with full fields, filling missing masks
from both best_pred and no_mask_pred directories.
"""
import os
import sys
import csv
import json

# === Paths ===
DATA_ROOT = "/mnt/cfs_algo_bj/models/experiments/lirunze/data/Shoulder/RightData"
JSON_ROOT = "/mnt/cfs_algo_bj/models/experiments/lirunze/code/shouder/final_output/to_extract/case_json"
NNUNET_ROOT = "/mnt/cfs_algo_bj/models/experiments/lirunze/code/nnUNet/output"
OUTPUT_PATH = "/mnt/cfs_algo_bj/models/experiments/lirunze/code/project/outputs/metadata/metadata_master.csv"

SEQUENCE_TYPES = ["axial_PD", "coronal_PD", "coronal_T2WI", "sagittal_PD", "sagittal_T1WI"]
DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]

# Disease -> mask folders (priority order: best_pred first, then no_mask_pred as fallback)
DISEASE_TO_MASK_FOLDERS = {
    "SST":  ["Dataset001_jianjia_best_pred",      "Dataset001_jianjia_no_mask_pred"],
    "IST":  ["Dataset002_jianxiu_best_pred",      "Dataset002_jianxiu_no_mask_pred"],
    "SSC":  ["Dataset005_gangxia_best_pred",      "Dataset005_gangxia_no_mask_pred"],
    "LHBT": ["Dataset006_gangshang_best_pred",    "Dataset006_gangshang_no_mask_pred"],
    "IGHL": ["Dataset008_jianguanjie_best_pred",  "Dataset008_jianguanjie_no_mask_pred"],
    "RIPI": ["Dataset009_gongretou_best_pred",    "Dataset009_gongretou_no_mask_pred"],
    "GHOA": ["Dataset010_yenang_2_best_pred",     "Dataset010_yenang_2_no_mask_pred"],
}


def get_mask_path(exam_id, disease):
    """Get mask path: try best_pred first, then no_mask_pred."""
    folders = DISEASE_TO_MASK_FOLDERS.get(disease, [])
    for folder in folders:
        path = os.path.join(NNUNET_ROOT, folder, "%s.nii.gz" % exam_id)
        if os.path.exists(path):
            return path
    return ""


def main():
    # Pre-build mask file sets for fast lookup (avoid repeated os.path.exists on CFS)
    print("Building mask file index...")
    mask_index = {}  # (disease, exam_id) -> path
    for disease, folders in DISEASE_TO_MASK_FOLDERS.items():
        for folder in folders:
            folder_path = os.path.join(NNUNET_ROOT, folder)
            if not os.path.isdir(folder_path):
                print("  WARNING: %s not found, skipping" % folder_path)
                continue
            for fname in os.listdir(folder_path):
                if fname.endswith(".nii.gz"):
                    eid = fname.replace(".nii.gz", "")
                    key = (disease, eid)
                    if key not in mask_index:  # best_pred has priority
                        mask_index[key] = os.path.join(folder_path, fname)
            print("  Indexed %s" % folder)
    print("Mask index built: %d entries" % len(mask_index))

    # List all exam IDs
    exam_ids = sorted(os.listdir(DATA_ROOT))
    exam_ids = [e for e in exam_ids if os.path.isdir(os.path.join(DATA_ROOT, e))]
    print("Found %d exams in RightData" % len(exam_ids))

    rows = []
    skipped = 0
    for i, eid in enumerate(exam_ids):
        if (i + 1) % 500 == 0:
            print("Processing %d/%d..." % (i + 1, len(exam_ids)))

        row = {"exam_id": eid}

        # === Images ===
        has_all = True
        for seq in SEQUENCE_TYPES:
            img_path = os.path.join(DATA_ROOT, eid, "%s.nii.gz" % seq)
            if os.path.exists(img_path):
                row["image_" + seq] = img_path
            else:
                row["image_" + seq] = ""
                has_all = False
        row["has_all_images"] = 1 if has_all else 0

        # === JSON ===
        json_path = os.path.join(JSON_ROOT, "%s.json" % eid)
        has_json = os.path.exists(json_path)
        row["json_path"] = json_path if has_json else ""
        row["has_json"] = 1 if has_json else 0

        # Skip if missing images or json (same filter as build_index.py)
        if not has_all or not has_json:
            skipped += 1
            continue

        # === Parse JSON ===
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print("  Failed to load json for %s: %s" % (eid, str(e)))
            data = {}

        # Metadata
        row["laterality"] = data.get("laterality", "")
        row["postoperative"] = data.get("postoperative", 0)
        row["exclude_from_main_training"] = data.get("exclude_from_main_training", 0)
        qf = data.get("quality_flag", "")
        row["quality_flag"] = qf if qf in ["high", "medium", "low"] else "medium"

        # Raw labels
        labels = data.get("labels", {})
        for disease in DISEASES:
            row["raw_label_" + disease] = labels.get(disease, -1)

        # Status
        status = data.get("label_status", {})
        for disease in DISEASES:
            row["status_" + disease] = status.get(disease, "")

        # Source summary
        source_summary = data.get("source_summary", {})
        for disease in DISEASES:
            row["source_" + disease] = source_summary.get(disease, "")

        # has_any_valid_label
        has_any_valid_label = any(labels.get(d, -1) in [0, 1, 2] for d in DISEASES)
        row["has_any_valid_label"] = 1 if has_any_valid_label else 0

        # === Masks (from pre-built index) ===
        has_any_mask = 0
        for disease in DISEASES:
            mask_path = mask_index.get((disease, eid), "")
            row["mask_path_" + disease] = mask_path
            if mask_path:
                has_any_mask = 1
        row["has_any_mask"] = has_any_mask

        # Split (reserved)
        row["split"] = ""

        rows.append(row)

    print("Valid cases: %d, Skipped (incomplete): %d" % (len(rows), skipped))

    # === Write CSV ===
    fieldnames = ["exam_id"]
    fieldnames += ["image_" + seq for seq in SEQUENCE_TYPES]
    fieldnames += ["has_all_images"]
    fieldnames += ["json_path", "has_json"]
    fieldnames += ["laterality", "postoperative", "exclude_from_main_training", "quality_flag"]
    fieldnames += ["raw_label_" + d for d in DISEASES]
    fieldnames += ["status_" + d for d in DISEASES]
    fieldnames += ["source_" + d for d in DISEASES]
    fieldnames += ["has_any_valid_label"]
    fieldnames += ["mask_path_" + d for d in DISEASES]
    fieldnames += ["has_any_mask"]
    fieldnames += ["split"]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Backup old file
    if os.path.exists(OUTPUT_PATH):
        backup = OUTPUT_PATH + ".bak"
        import shutil
        shutil.copy2(OUTPUT_PATH, backup)
        print("Backed up old CSV to %s" % backup)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved to %s" % OUTPUT_PATH)
    print("Total fields: %d" % len(fieldnames))

    # Stats
    mask_stats = {}
    for disease in DISEASES:
        count = sum(1 for r in rows if r.get("mask_path_" + disease, ""))
        mask_stats[disease] = count
    print("\nMask coverage:")
    for d, c in mask_stats.items():
        print("  %s: %d / %d (%.1f%%)" % (d, c, len(rows), 100.0 * c / len(rows) if rows else 0))

    total_with_mask = sum(1 for r in rows if r.get("has_any_mask", 0))
    print("  Any mask: %d / %d (%.1f%%)" % (total_with_mask, len(rows), 100.0 * total_with_mask / len(rows) if rows else 0))


if __name__ == "__main__":
    main()
