"""
Build metadata index from all data sources.
Creates a master CSV with images, json labels, nnUNet masks, and train split.
"""
from __future__ import print_function
import os
import csv
import argparse

from utils.io import (
    DATA_ROOT, JSON_ROOT, NNUNET_ROOT,
    SEQUENCE_TYPES, list_exam_ids,
    get_image_path, get_json_path, load_json_label
)


DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]

# 直接映射: 病种 -> nnUNet mask 文件夹
DISEASE_TO_MASK_FOLDER = {
    "SST": "Dataset001_jianjia_best_pred",
    "IST": "Dataset002_jianxiu_best_pred",
    "SSC": "Dataset005_gangxia_best_pred",
    "LHBT": "Dataset006_gangshang_best_pred",
    "IGHL": "Dataset008_jianguanjie_best_pred",
    "RIPI": "Dataset009_gongretou_best_pred",
    "GHOA": "Dataset010_yenang_2_best_pred",
}


def get_mask_path(exam_id, disease):
    """Get nnUNet mask path for a disease."""
    mask_folder = DISEASE_TO_MASK_FOLDER.get(disease, "")
    if not mask_folder:
        return ""
    mask_path = os.path.join(NNUNET_ROOT, mask_folder, "%s.nii.gz" % exam_id)
    return mask_path if os.path.exists(mask_path) else ""


def generate_metadata_csv(output_path="outputs/metadata/metadata_master.csv", include_masks=True):
    """Generate master metadata CSV."""

    exam_ids = list_exam_ids()
    print("Found %d exams in RightData" % len(exam_ids))

    # 1. 扫描完整病例
    valid_exams = []
    for eid in exam_ids:
        # 检查五序列是否齐全
        has_all_images = all(os.path.exists(get_image_path(eid, seq)) for seq in SEQUENCE_TYPES)
        has_json = os.path.exists(get_json_path(eid))
        if has_all_images and has_json:
            valid_exams.append(eid)

    print("Complete cases (images + JSON): %d" % len(valid_exams))

    rows = []
    for i, eid in enumerate(valid_exams):
        print("Processing %d/%d: %s" % (i + 1, len(valid_exams), eid))

        row = {"exam_id": eid}

        # ============ 图像路径 ============
        for seq in SEQUENCE_TYPES:
            img_path = get_image_path(eid, seq)
            row["image_" + seq] = img_path if os.path.exists(img_path) else ""
        row["has_all_images"] = 1

        # ============ JSON 路径 ============
        json_path = get_json_path(eid)
        row["json_path"] = json_path
        row["has_json"] = 1

        # ============ 解析 JSON ============
        try:
            data = load_json_label(eid)
        except Exception as e:
            print("  Failed to load json: %s" % str(e))
            data = {}

        # Metadata
        row["laterality"] = data.get("laterality", "")
        row["postoperative"] = data.get("postoperative", 0)
        row["exclude_from_main_training"] = data.get("exclude_from_main_training", 0)

        # quality_flag: 只允许 high/medium/low
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

        # source_summary (对 SFT 数据构造很有用)
        source_summary = data.get("source_summary", {})
        for disease in DISEASES:
            row["source_" + disease] = source_summary.get(disease, "")

        # 是否有有效标签
        has_any_valid_label = any(
            labels.get(d, -1) in [0, 1, 2] for d in DISEASES
        )
        row["has_any_valid_label"] = 1 if has_any_valid_label else 0

        # ============ Mask 路径 ============
        if include_masks:
            has_any_mask = 0
            for disease in DISEASES:
                mask_path = get_mask_path(eid, disease)
                row["mask_path_" + disease] = mask_path
                if mask_path:
                    has_any_mask = 1
            row["has_any_mask"] = has_any_mask

        # ============ split (预留) ============
        row["split"] = ""

        rows.append(row)

    # ============ 输出 CSV ============
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["exam_id"]
    # 图像
    fieldnames += ["image_" + seq for seq in SEQUENCE_TYPES]
    fieldnames += ["has_all_images"]
    # JSON
    fieldnames += ["json_path", "has_json"]
    # Metadata
    fieldnames += ["laterality", "postoperative", "exclude_from_main_training", "quality_flag"]
    # Raw labels
    fieldnames += ["raw_label_" + d for d in DISEASES]
    # Status
    fieldnames += ["status_" + d for d in DISEASES]
    # Source summary
    fieldnames += ["source_" + d for d in DISEASES]
    # Valid label flag
    fieldnames += ["has_any_valid_label"]
    # Masks
    if include_masks:
        fieldnames += ["mask_path_" + d for d in DISEASES]
        fieldnames += ["has_any_mask"]
    # Split (预留)
    fieldnames += ["split"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Metadata saved to %s" % output_path)
    print("Total fields: %d" % len(fieldnames))
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="outputs/metadata/metadata_master.csv")
    parser.add_argument("--no-mask", action="store_true", help="Skip mask paths")
    args = parser.parse_args()

    generate_metadata_csv(args.output, include_masks=not args.no_mask)


if __name__ == "__main__":
    main()