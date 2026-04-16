"""
IO utilities for NIfTI images and path handling.
"""
import os
import json
import nibabel as nib
import numpy as np


# Data paths
DATA_ROOT = os.environ.get("SHOULDER_DATA_ROOT", "/mnt/cfs_algo_bj/models/experiments/lirunze/data/Shoulder/RightData")
JSON_ROOT = os.environ.get("SHOULDER_JSON_ROOT", "/mnt/cfs_algo_bj/models/experiments/lirunze/code/shouder/final_output/to_extract/case_json")
NNUNET_ROOT = os.environ.get("SHOULDER_NNUNET_ROOT", "/mnt/cfs_algo_bj/models/experiments/lirunze/code/nnUNet/output")

# Sequence types
SEQUENCE_TYPES = ["axial_PD", "coronal_PD", "coronal_T2WI", "sagittal_PD", "sagittal_T1WI"]

# Diseases
DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]

# Disease to anchor sequence (best single sequence for each disease)
DISEASE_ANCHOR_SEQ = {
    "SST": "coronal_PD",   # 冈上肌腱
    "IST": "axial_PD",     # 冈下肌腱
    "SSC": "axial_PD",     # 肩胛下肌腱
    "LHBT": "coronal_PD",   # 肱二头肌腱
    "IGHL": "coronal_PD",  # 盂唇
    "RIPI": "sagittal_PD", # 肩袖间隙
    "GHOA": "coronal_PD",  # 盂肱关节
}


def get_image_path(exam_id, sequence):
    """Get path to MRI sequence file."""
    return os.path.join(DATA_ROOT, exam_id, "%s.nii.gz" % sequence)


def get_json_path(exam_id):
    """Get path to JSON label file."""
    return os.path.join(JSON_ROOT, "%s.json" % exam_id)


def load_json_label(exam_id):
    """Load JSON label file for an exam."""
    path = get_json_path(exam_id)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_exam_ids():
    """List all available exam IDs from RightData."""
    if not os.path.exists(DATA_ROOT):
        return []
    return sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])


def check_case_complete(exam_id):
    """Check if a case has all 5 sequences."""
    for seq in SEQUENCE_TYPES:
        path = get_image_path(exam_id, seq)
        if not os.path.exists(path):
            return False
    return True


def load_nifti(path):
    """Load NIfTI file and return raw data and affine.

    NOTE: The returned array keeps the on-disk axis order (typically
    H, W, Z for shoulder MRI).  Call ``normalize_axes`` to convert
    to the project-standard [Z, H, W] layout.
    """
    img = nib.load(path)
    return img.get_fdata(), img.affine


def normalize_axes(volume):
    """Convert NIfTI volume from on-disk (H, W, Z) to project-standard (Z, H, W).

    Shoulder MRI NIfTI files are stored as (H, W, Z) where:
        axis 0 = H  (rows,    ~320-768)
        axis 1 = W  (columns, ~320-768)
        axis 2 = Z  (slices,  ~18-24)

    This function transposes to (Z, H, W) so that:
        axis 0 = Z  (slices)
        axis 1 = H  (rows)
        axis 2 = W  (columns)

    Works for both images (float) and masks (int).
    Handles 4-D volumes by squeezing trailing dimensions first.
    """
    vol = np.squeeze(volume)
    if vol.ndim != 3:
        raise ValueError(
            "Expected 3D volume after squeeze, got shape %s" % (vol.shape,))
    # Transpose (H, W, Z) -> (Z, H, W)
    return np.transpose(vol, (2, 0, 1))


def load_nifti_normalized(path):
    """Load NIfTI and immediately normalize to [Z, H, W].

    Returns (volume_zhw, affine).
    """
    data, affine = load_nifti(path)
    return normalize_axes(data), affine


def save_nifti(data, affine, path):
    """Save data as NIfTI file."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


def load_mask(exam_id, dataset_id):
    """Load nnUNet mask for a case, normalized to [Z, H, W].

    Returns (mask_zhw, affine) or (None, None).
    """
    mask_folder = "Dataset%03d_*_best_pred" % dataset_id
    mask_dir = os.path.join(NNUNET_ROOT, mask_folder)
    if not os.path.exists(mask_dir):
        return None, None

    # Find actual folder
    for f in os.listdir(NNUNET_ROOT):
        if f.startswith("Dataset%03d" % dataset_id) and "best_pred" in f:
            mask_dir = os.path.join(NNUNET_ROOT, f)
            break

    mask_path = os.path.join(mask_dir, "%s.nii.gz" % exam_id)
    if os.path.exists(mask_path):
        data, affine = load_nifti(mask_path)
        return normalize_axes(data), affine
    return None, None


def get_key_slice(mask_data, axis=0):
    """Extract key slice (slice with most foreground) from mask.

    Expects input in [Z, H, W] layout. Default axis=0 means
    searching along the Z (slice) dimension.
    """
    if mask_data is None:
        return None

    if axis == 0:
        sums = mask_data.sum(axis=(1, 2))
    elif axis == 1:
        sums = mask_data.sum(axis=(0, 2))
    else:
        sums = mask_data.sum(axis=(0, 1))

    key_idx = np.argmax(sums)
    return int(key_idx)


def get_bbox(mask_data, margin=5):
    """Get bounding box from mask with margin.

    Expects input in [Z, H, W] layout.
    Returns (z_min, h_min, w_min, z_max, h_max, w_max).
    """
    if mask_data is None:
        return None

    mask = (mask_data > 0).astype(np.uint8)

    # Find non-zero voxels
    indices = np.where(mask > 0)
    if len(indices[0]) == 0:
        return None

    d_min, d_max = indices[0].min(), indices[0].max()
    h_min, h_max = indices[1].min(), indices[1].max()
    w_min, w_max = indices[2].min(), indices[2].max()

    # Add margin
    d_min = max(0, d_min - margin)
    h_min = max(0, h_min - margin)
    w_min = max(0, w_min - margin)
    d_max = min(mask.shape[0] - 1, d_max + margin)
    h_max = min(mask.shape[1] - 1, h_max + margin)
    w_max = min(mask.shape[2] - 1, w_max + margin)

    return (d_min, h_min, w_min, d_max, h_max, w_max)


def crop_roi(data, bbox):
    """Crop 3D volume to ROI bounding box."""
    if bbox is None:
        return data

    d_min, h_min, w_min, d_max, h_max, w_max = bbox
    return data[d_min:d_max+1, h_min:h_max+1, w_min:w_max+1]