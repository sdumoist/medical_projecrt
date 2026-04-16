"""
Mask indexer for nnUNet masks.
Extracts key-slices and ROIs from nnUNet predictions.
"""
from __future__ import print_function
import os
import csv
import argparse

from utils.io import (
    NNUNET_ROOT, DISEASES, DISEASE_ANCHOR_SEQ,
    load_nifti, get_key_slice, get_bbox
)


DATASET_TO_MASK_FOLDER = {
    1: "Dataset001_jianjia_best_pred",
    2: "Dataset002_jianxiu_best_pred",
    5: "Dataset005_gangxia_best_pred",
    6: "Dataset006_gangshang_best_pred",
    8: "Dataset008_jianguanjie_best_pred",
    9: "Dataset009_gongretou_best_pred",
    10: "Dataset010_yenang_2_best_pred",
}


def find_mask_path(exam_id, dataset_id):
    """Find nnUNet mask path for a case."""
    folder = DATASET_TO_MASK_FOLDER.get(dataset_id, "")
    if not folder:
        return ""

    mask_path = os.path.join(NNUNET_ROOT, folder, "%s.nii.gz" % exam_id)
    if os.path.exists(mask_path):
        return mask_path
    return ""


def extract_key_slice(mask_path, axis=0):
    """Extract key slice from mask."""
    if not mask_path or not os.path.exists(mask_path):
        return None

    try:
        data, _ = load_nifti(mask_path)
        if data is None:
            return None
        return get_key_slice(data, axis)
    except:
        return None


def extract_bbox(mask_path):
    """Extract bounding box from mask."""
    if not mask_path or not os.path.exists(mask_path):
        return None

    try:
        data, _ = load_nifti(mask_path)
        if data is None:
            return None
        return get_bbox(data)
    except:
        return None


def generate_mask_index_csv(exam_ids, output_path="outputs/metadata/mask_index.csv"):
    """Generate mask index CSV with key slices and bboxes."""

    rows = []
    for i, eid in enumerate(exam_ids):
        print("Processing %d/%d: %s" % (i + 1, len(exam_ids), eid))

        row = {"exam_id": eid}

        # For each disease, try to find mask and extract info
        for disease in DISEASES:
            dataset_id = None
            for ds_id, dis in {
                1: "SST", 2: "IST", 5: "SSC", 6: "LHBT",
                8: "IGHL", 9: "RIPI", 10: "GHOA"
            }.items():
                if dis == disease:
                    dataset_id = ds_id
                    break

            if dataset_id is None:
                row["mask_path_%s" % disease] = ""
                row["key_slice_%s" % disease] = ""
                row["bbox_%s" % disease] = ""
                continue

            mask_path = find_mask_path(eid, dataset_id)
            row["mask_path_%s" % disease] = mask_path

            if mask_path and os.path.exists(mask_path):
                try:
                    data, _ = load_nifti(mask_path)
                    key_slice = get_key_slice(data, axis=0)
                    row["key_slice_%s" % disease] = key_slice if key_slice is not None else ""

                    bbox = get_bbox(data)
                    if bbox:
                        row["bbox_%s" % disease] = ",".join(map(str, bbox))
                    else:
                        row["bbox_%s" % disease] = ""
                except Exception as e:
                    print("  Error: %s" % e)
                    row["key_slice_%s" % disease] = ""
                    row["bbox_%s" % disease] = ""
            else:
                row["mask_path_%s" % disease] = ""
                row["key_slice_%s" % disease] = ""
                row["bbox_%s" % disease] = ""

        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["exam_id"]
    for disease in DISEASES:
        fieldnames += ["mask_path_%s" % disease, "key_slice_%s" % disease, "bbox_%s" % disease]

    with open(output_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Mask index saved to %s" % output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="outputs/metadata/mask_index.csv")
    args = parser.parse_args()

    from utils.io import list_exam_ids
    exam_ids = list_exam_ids()
    generate_mask_index_csv(exam_ids, args.output)


if __name__ == "__main__":
    main()