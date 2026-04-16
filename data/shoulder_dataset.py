"""
Shoulder MRI Dataset.
Loads 5-sequence MRI data with 3D volumes.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage

from data.json_parser import JSONParser
from data.label_mapper import LabelMapper
from utils.io import SEQUENCE_TYPES, load_nifti, get_image_path


class ShoulderDataset3D(Dataset):
    """3D Dataset for shoulder MRI multi-sequence classification."""

    def __init__(
        self,
        exam_ids,
        data_root,
        json_root,
        sequences=SEQUENCE_TYPES,
        label_mapper=None,
        raw_labels_lookup=None,
        crop_size=(32, 64, 64),
        input_spacing=(1.0, 1.0, 1.0),
        mode="train"
    ):
        self.exam_ids = exam_ids
        self.data_root = data_root
        self.json_root = json_root
        self.sequences = sequences
        self.label_mapper = label_mapper
        self.raw_labels_lookup = raw_labels_lookup or {}
        self.crop_size = crop_size
        self.input_spacing = input_spacing
        self.mode = mode

        # Build index
        self.valid_indices = []
        for i, eid in enumerate(exam_ids):
            json_path = os.path.join(json_root, "%s.json" % eid)
            if not os.path.exists(json_path):
                continue

            all_exist = True
            for seq in sequences:
                img_path = os.path.join(data_root, eid, "%s.nii.gz" % seq)
                if not os.path.exists(img_path):
                    all_exist = False
                    break

            if all_exist:
                self.valid_indices.append(i)

        print("Loaded %d/%d valid cases" % (len(self.valid_indices), len(exam_ids)))

    def __len__(self):
        return len(self.valid_indices)

    def _load_sequence(self, exam_id, sequence):
        """Load a single sequence."""
        img_path = os.path.join(self.data_root, exam_id, "%s.nii.gz" % sequence)
        data, affine = load_nifti(img_path)
        return data

    def _resample_crop(self, data):
        """Resample and crop 3D volume."""
        # Squeeze extra dimensions (e.g. 4D NIfTI with trailing dim)
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError("Expected 3D volume after squeeze, got shape %s" % (data.shape,))
        d, h, w = data.shape
        td, th, tw = self.crop_size

        # Simple center crop (can be improved with actual resampling)
        d_start = max(0, (d - td) // 2)
        h_start = max(0, (h - th) // 2)
        w_start = max(0, (w - tw) // 2)

        data = data[
            d_start:d_start + td,
            h_start:h_start + th,
            w_start:w_start + tw
        ]

        # Pad if needed
        if data.shape != self.crop_size:
            padded = np.zeros(self.crop_size, dtype=data.dtype)
            cd, ch, cw = data.shape
            pd_start = (td - cd) // 2
            ph_start = (th - ch) // 2
            pw_start = (tw - cw) // 2
            padded[pd_start:pd_start + cd, ph_start:ph_start + ch, pw_start:pw_start + cw] = data
            data = padded

        return data

    def _normalize(self, data):
        """Normalize intensity."""
        mean = data.mean()
        std = data.std()
        if std > 0:
            data = (data - mean) / std
        else:
            data = data - mean
        return data

    def __getitem__(self, idx):
        """Get a sample."""
        real_idx = self.valid_indices[idx]
        exam_id = self.exam_ids[real_idx]

        # Load all sequences [5, D, H, W]
        images = []
        for seq in self.sequences:
            data = self._load_sequence(exam_id, seq)
            data = self._resample_crop(data)
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

        # Model expects [B, num_seq, C, D, H, W] after DataLoader
        # Dataset returns [num_seq, 1, D, H, W], DataLoader adds batch dim
        images = images[:, None]  # [5, 1, D, H, W]

        return {
            "exam_id": exam_id,
            "image": images,  # [5, 1, D, H, W] -> DataLoader makes [B, 5, 1, D, H, W]
            "labels": labels,  # [7]
            "mask": mask,  # [7]
        }


def create_dataset(
    exam_ids,
    data_root,
    json_root,
    sequences=SEQUENCE_TYPES,
    mode="binary",
    batch_size=2,
    num_workers=0,
    crop_size=(32, 64, 64),
    mode_split="train"
):
    """Create a DataLoader for shoulder dataset."""
    label_mapper = LabelMapper(mode=mode)

    dataset = ShoulderDataset3D(
        exam_ids=exam_ids,
        data_root=data_root,
        json_root=json_root,
        sequences=sequences,
        label_mapper=label_mapper,
        crop_size=crop_size,
        mode=mode_split
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode_split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )

    return loader