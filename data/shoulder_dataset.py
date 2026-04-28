"""
Shoulder MRI Dataset.
Loads 5-sequence MRI data with 3D volumes.

Supports three data source modes:
  - "cls":   read pre-built .pt from build_cls_cache.py
  - "loc":   read pre-built .pt from build_loc_cache.py  (includes masks)
  - "none":  read raw .nii.gz files on-the-fly  (fallback / debug)

All modes output images in [5, 1, Z, H, W] with project-standard Z-first axis.
"""
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom as ndizoom

from data.label_mapper import LabelMapper
from utils.io import SEQUENCE_TYPES, load_nifti, normalize_axes
from utils.constants import SEQUENCE_ORDER


# =========================================================================
# Cache-based dataset (cache_cls / cache_loc)
# =========================================================================

class ShoulderCacheDataset(Dataset):
    """Dataset that reads pre-built .pt cache files.

    Works with both cache_cls and cache_loc formats.
    cache_cls record: {exam_id, image[5,Z,H,W], sequence_order, spatial_meta}
    cache_loc record: {exam_id, image[5,Z,H,W], mask{...}, key_slices, roi_boxes, ...}

    For cache_loc, use load_dense_masks=False (default) to skip building
    localizer_mask and roi_boxes tensors.  This saves memory and I/O when
    only key_slices supervision is needed (Step 1 grounded training).
    """

    def __init__(
        self,
        cache_root,
        exam_ids=None,
        label_mapper=None,
        raw_labels_lookup=None,
        cache_mode="cls",
        project_root=None,
        min_z=32,
        load_dense_masks=False,
    ):
        """
        Args:
            cache_root: directory containing .pt files and index CSV
            exam_ids: if provided, only use these exam IDs (for train/val split)
            label_mapper: LabelMapper instance
            raw_labels_lookup: {exam_id: {disease: raw_label, ...}}
            cache_mode: "cls" or "loc"
            project_root: root dir to resolve relative cache_path; defaults to cwd
            min_z: minimum Z dimension; pads with zeros if cache Z < min_z
                   (DenseNet needs Z>=32 due to 5 pooling stages)
            load_dense_masks: if False (default), cache_loc only returns
                              image/labels/mask/key_slices — skips building
                              roi_boxes and localizer_mask tensors.
                              Set True only when dense mask supervision is needed.
        """
        self.cache_mode = cache_mode
        self.min_z = min_z
        self.load_dense_masks = load_dense_masks
        self.label_mapper = label_mapper
        self.raw_labels_lookup = raw_labels_lookup or {}
        self.project_root = project_root or os.getcwd()

        # Read index CSV
        index_name = "cache_cls_index.csv" if cache_mode == "cls" else "cache_loc_index.csv"
        index_path = os.path.join(cache_root, index_name)
        if not os.path.exists(index_path):
            raise FileNotFoundError("Cache index not found: %s" % index_path)

        import csv
        self.exam_ids = []
        self.cache_paths = []
        with open(index_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("success", 0)) != 1:
                    continue
                eid = row["exam_id"]
                if exam_ids is not None and eid not in exam_ids:
                    continue
                # Resolve cache path (may be relative)
                cp = row["cache_path"]
                if not os.path.isabs(cp):
                    cp = os.path.join(self.project_root, cp)
                self.exam_ids.append(eid)
                self.cache_paths.append(cp)

        # Convert exam_ids filter to set for quick note
        if exam_ids is not None:
            requested = len(set(exam_ids))
        else:
            requested = "all"
        print("ShoulderCacheDataset(cache_%s): %d cases (requested: %s)"
              % (cache_mode, len(self.exam_ids), requested))

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        cache_path = self.cache_paths[idx]
        record = torch.load(cache_path, map_location="cpu", weights_only=False)

        # image: [5, Z, H, W] -> pad Z if needed -> [5, 1, Z', H, W]
        images = record["image"]  # FloatTensor [5, Z, H, W]
        Z = images.shape[1]
        if Z < self.min_z:
            pad_total = self.min_z - Z
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            # F.pad expects (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
            # images is [5, Z, H, W], pad dim=1 (Z): need (0,0, 0,0, pad_before, pad_after)
            images = torch.nn.functional.pad(images, (0, 0, 0, 0, pad_before, pad_after))
        images = images.unsqueeze(1)  # [5, 1, Z', H, W]

        # labels
        labels = torch.zeros(7, dtype=torch.int64)
        mask = torch.zeros(7, dtype=torch.float32)

        if self.label_mapper is not None and exam_id in self.raw_labels_lookup:
            raw_labels = self.raw_labels_lookup[exam_id]
            train_labels, train_masks = self.label_mapper.map_labels(raw_labels)
            labels = torch.from_numpy(np.array(train_labels, dtype=np.int64))
            mask = torch.from_numpy(np.array(train_masks, dtype=np.float32))

        out = {
            "exam_id": exam_id,
            "image": images,    # [5, 1, Z, H, W]
            "labels": labels,   # [7]
            "mask": mask,       # [7]  (label mask, not segmentation mask)
        }

        # For cache_loc, also pass localizer targets
        if self.cache_mode == "loc":
            key_slices_raw = record.get("key_slices", {})

            from data.label_mapper import DISEASES

            # Z-padding offset for key_slices
            orig_Z = record["image"].shape[1]
            z_offset = (self.min_z - orig_Z) // 2 if orig_Z < self.min_z else 0

            # key_slices: dict -> tensor [7], -1 for missing, shifted by z_offset
            ks = torch.full((7,), -1, dtype=torch.long)
            for i, d in enumerate(DISEASES):
                v = key_slices_raw.get(d, -1)
                if v is not None and v >= 0:
                    ks[i] = v + z_offset
            out["key_slices"] = ks

            if self.load_dense_masks:
                # roi_boxes and localizer_mask: only loaded when explicitly requested
                seg_masks = record.get("mask", {})
                roi_boxes_raw = record.get("roi_boxes", {})

                # roi_boxes: dict -> tensor [7, 6], zeros for missing
                rb = torch.zeros(7, 6, dtype=torch.float32)
                for i, d in enumerate(DISEASES):
                    box = roi_boxes_raw.get(d, None)
                    if box is not None and len(box) == 6:
                        rb[i] = torch.tensor(box, dtype=torch.float32)
                out["roi_boxes"] = rb

                # seg_masks: dict -> tensor [7, Z', H, W] (padded same as image)
                sample_img = record["image"]  # [5, Z, H, W]
                Z_orig, H, W = sample_img.shape[1], sample_img.shape[2], sample_img.shape[3]
                Z_out = max(Z_orig, self.min_z)
                sm = torch.zeros(7, Z_out, H, W, dtype=torch.int64)
                for i, d in enumerate(DISEASES):
                    m = seg_masks.get(d, None)
                    if m is not None and isinstance(m, torch.Tensor):
                        sm[i, z_offset:z_offset + Z_orig] = m
                out["localizer_mask"] = sm

        return out


# =========================================================================
# Raw NIfTI dataset (fallback / debug)
# =========================================================================

class ShoulderDataset3D(Dataset):
    """3D Dataset that reads raw .nii.gz on the fly.

    All volumes are normalized to [Z, H, W] via normalize_axes.
    """

    def __init__(
        self,
        exam_ids,
        data_root,
        json_root,
        sequences=None,
        label_mapper=None,
        raw_labels_lookup=None,
        target_shape=(20, 448, 448),
        mode="train"
    ):
        self.exam_ids = exam_ids
        self.data_root = data_root
        self.json_root = json_root
        self.sequences = sequences or SEQUENCE_ORDER
        self.label_mapper = label_mapper
        self.raw_labels_lookup = raw_labels_lookup or {}
        self.target_shape = target_shape
        self.mode = mode

        # Build index
        self.valid_indices = []
        for i, eid in enumerate(exam_ids):
            json_path = os.path.join(json_root, "%s.json" % eid)
            if not os.path.exists(json_path):
                continue

            all_exist = True
            for seq in self.sequences:
                img_path = os.path.join(data_root, eid, "%s.nii.gz" % seq)
                if not os.path.exists(img_path):
                    all_exist = False
                    break

            if all_exist:
                self.valid_indices.append(i)

        print("ShoulderDataset3D(raw): %d/%d valid cases" % (
            len(self.valid_indices), len(exam_ids)))

    def __len__(self):
        return len(self.valid_indices)

    def _load_sequence(self, exam_id, sequence):
        """Load a single sequence and normalize to [Z, H, W]."""
        img_path = os.path.join(self.data_root, exam_id, "%s.nii.gz" % sequence)
        data, _ = load_nifti(img_path)
        data = normalize_axes(data)  # (H,W,Z) -> (Z,H,W)
        return data

    def _preprocess(self, data):
        """Pad/crop Z, resize H/W to target_shape. Input: [Z, H, W]."""
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError("Expected 3D, got shape %s" % (data.shape,))

        TZ, TH, TW = self.target_shape
        Z, H, W = data.shape

        # Z: centre pad/crop
        if Z > TZ:
            start = (Z - TZ) // 2
            data = data[start: start + TZ]
        elif Z < TZ:
            out = np.zeros((TZ, H, W), dtype=data.dtype)
            start = (TZ - Z) // 2
            out[start: start + Z] = data
            data = out

        # H, W: resize
        _, cH, cW = data.shape
        if cH != TH or cW != TW:
            zoom_factors = (1.0, TH / cH, TW / cW)
            data = ndizoom(data, zoom_factors, order=1).astype(data.dtype)

        return data

    def _normalize(self, data):
        """Z-score normalize."""
        mean = data.mean()
        std = data.std()
        if std > 0:
            data = (data - mean) / std
        else:
            data = data - mean
        return data

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        exam_id = self.exam_ids[real_idx]

        # Load all sequences [5, Z, H, W]
        images = []
        for seq in self.sequences:
            data = self._load_sequence(exam_id, seq)
            data = self._preprocess(data)
            data = self._normalize(data)
            images.append(data)

        images = np.stack(images, axis=0).astype(np.float32)

        # Load labels
        labels = np.zeros(7, dtype=np.int64)
        mask = np.zeros(7, dtype=np.float32)

        if self.label_mapper is not None and exam_id in self.raw_labels_lookup:
            raw_labels = self.raw_labels_lookup[exam_id]
            train_labels, train_masks = self.label_mapper.map_labels(raw_labels)
            labels = np.array(train_labels, dtype=np.int64)
            mask = np.array(train_masks, dtype=np.float32)

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        mask = torch.from_numpy(mask)

        # [5, Z, H, W] -> [5, 1, Z, H, W]
        images = images[:, None]

        return {
            "exam_id": exam_id,
            "image": images,    # [5, 1, Z, H, W]
            "labels": labels,   # [7]
            "mask": mask,       # [7]
        }
