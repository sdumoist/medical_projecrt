"""
Shoulder MRI Dataset.
Loads 5-sequence MRI data with 3D volumes.

Supports three data source modes:
  - "cache_cls":  read pre-built .pt from build_cls_cache.py
  - "cache_loc":  read pre-built .pt from build_loc_cache.py  (includes masks)
  - "raw":        read raw .nii.gz files on-the-fly  (fallback / debug)

All modes output images in [5, 1, Z, H, W] with project-standard Z-first axis.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom as ndizoom

from data.label_mapper import LabelMapper
from utils.io import SEQUENCE_TYPES, load_nifti, normalize_axes


# =========================================================================
# Shared constants
# =========================================================================
SEQUENCE_ORDER = [
    "axial_PD",
    "coronal_PD",
    "coronal_T2WI",
    "sagittal_PD",
    "sagittal_T1WI",
]


# =========================================================================
# Cache-based dataset (cache_cls / cache_loc)
# =========================================================================

class ShoulderCacheDataset(Dataset):
    """Dataset that reads pre-built .pt cache files.

    Works with both cache_cls and cache_loc formats.
    cache_cls record: {exam_id, image[5,Z,H,W], sequence_order, spatial_meta}
    cache_loc record: {exam_id, image[5,Z,H,W], mask{...}, key_slices, ...}
    """

    def __init__(
        self,
        cache_index_csv,
        label_mapper=None,
        raw_labels_lookup=None,
        mode="cache_cls",
    ):
        """
        Args:
            cache_index_csv: path to cache_cls_index.csv or cache_loc_index.csv
            label_mapper: LabelMapper instance
            raw_labels_lookup: {exam_id: {disease: raw_label, ...}}
            mode: "cache_cls" or "cache_loc"
        """
        import pandas as pd
        df = pd.read_csv(cache_index_csv)
        df = df[df["success"] == 1].reset_index(drop=True)

        self.cache_paths = df["cache_path"].tolist()
        self.exam_ids = df["exam_id"].tolist()
        self.label_mapper = label_mapper
        self.raw_labels_lookup = raw_labels_lookup or {}
        self.mode = mode

        print("ShoulderCacheDataset(%s): %d cases" % (mode, len(self.exam_ids)))

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, idx):
        exam_id = self.exam_ids[idx]
        cache_path = self.cache_paths[idx]
        record = torch.load(cache_path, map_location="cpu")

        # image: [5, Z, H, W] -> [5, 1, Z, H, W]
        images = record["image"]  # FloatTensor [5, Z, H, W]
        images = images.unsqueeze(1)  # [5, 1, Z, H, W]

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

        # For cache_loc, also pass segmentation masks and localizer targets
        if self.mode == "cache_loc" and "mask" in record:
            out["seg_masks"] = record["mask"]       # {disease: tensor or None}
            out["key_slices"] = record.get("key_slices", {})
            out["roi_boxes"] = record.get("roi_boxes", {})

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


# =========================================================================
# Factory function
# =========================================================================

def create_dataset(
    # --- common ---
    label_mode="binary",
    raw_labels_lookup=None,
    batch_size=2,
    num_workers=0,
    shuffle_train=True,
    # --- cache mode ---
    source="cache_cls",
    cache_index_csv=None,
    # --- raw mode ---
    exam_ids=None,
    data_root=None,
    json_root=None,
    target_shape=(20, 448, 448),
    mode_split="train",
):
    """Create a DataLoader for shoulder dataset.

    Args:
        source: "cache_cls", "cache_loc", or "raw"
        cache_index_csv: required for cache modes
        exam_ids, data_root, json_root: required for raw mode
    """
    label_mapper = LabelMapper(mode=label_mode)

    if source in ("cache_cls", "cache_loc"):
        if cache_index_csv is None:
            raise ValueError("cache_index_csv required for source=%s" % source)
        dataset = ShoulderCacheDataset(
            cache_index_csv=cache_index_csv,
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            mode=source,
        )
    elif source == "raw":
        if exam_ids is None or data_root is None or json_root is None:
            raise ValueError("exam_ids, data_root, json_root required for source=raw")
        dataset = ShoulderDataset3D(
            exam_ids=exam_ids,
            data_root=data_root,
            json_root=json_root,
            label_mapper=label_mapper,
            raw_labels_lookup=raw_labels_lookup,
            target_shape=target_shape,
            mode=mode_split,
        )
    else:
        raise ValueError("Unknown source: %s (use cache_cls/cache_loc/raw)" % source)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode_split == "train" and shuffle_train),
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
