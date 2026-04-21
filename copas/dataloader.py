"""
Shoulder MRI dataloader for CoPAS reproduction.

Reads from cache_cls .pt files, returns 5 volumes as list.
Labels from metadata_master.csv (raw_label_* columns).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as TF

DISEASES = ['SST', 'IST', 'SSC', 'LHBT', 'IGHL', 'RIPI', 'GHOA']


def load_metadata(metadata_csv, cache_root):
    """Load metadata, filter valid exams, return (valid_ids, label_lookup)."""
    df = pd.read_csv(metadata_csv)
    df = df[df['has_all_images'] == 1]
    df = df[df['exclude_from_main_training'] == 0]

    # Fast filter by cache index
    index_csv = os.path.join(cache_root, "cache_cls_index.csv")
    if os.path.exists(index_csv):
        idx_df = pd.read_csv(index_csv)
        cached_set = set(idx_df[idx_df['cache_exists'] == 1]['exam_id'].astype(str))
        df = df[df['exam_id'].astype(str).isin(cached_set)]
        print(f"Filtered to {len(df)} exams with cache files")

    label_lookup = {}
    valid_ids = []
    for _, row in df.iterrows():
        eid = str(row['exam_id'])
        labels = []
        valid = True
        for d in DISEASES:
            val = row.get('raw_label_' + d, -1)
            if pd.isna(val) or val == -1:
                valid = False
                break
            labels.append(int(val))
        if valid:
            label_lookup[eid] = labels
            valid_ids.append(eid)

    return valid_ids, label_lookup


def create_split(valid_ids, label_lookup, val_ratio=0.2, seed=42):
    """Stratified train/val split based on positive label ratio."""
    rng = random.Random(seed)
    pos_ratios = np.array([
        sum(label_lookup[eid]) / len(DISEASES) for eid in valid_ids
    ])
    bins = np.digitize(pos_ratios, [0, 0.15, 0.3, 0.5, 1.0])

    train_ids, val_ids = [], []
    for b in range(6):
        indices = np.where(bins == b)[0].tolist()
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio)) if indices else 0
        val_ids.extend([valid_ids[i] for i in indices[:n_val]])
        train_ids.extend([valid_ids[i] for i in indices[n_val:]])

    return train_ids, val_ids


class ShoulderCoPASDataset(data.Dataset):
    """Dataset for CoPAS model: loads cache_cls .pt, returns 5 vols as list.

    CoPAS expects: [sag_PD, cor_PD, axi_PD, sag_T1WI, cor_T2WI]
    Cache stores:   [axial_PD, coronal_PD, coronal_T2WI, sagittal_PD, sagittal_T1WI]
    """

    COPAS_ORDER = ['sagittal_PD', 'coronal_PD', 'axial_PD',
                    'sagittal_T1WI', 'coronal_T2WI']

    def __init__(self, exam_ids, label_lookup, cache_root,
                 input_dim=224, slice_num=20, transform=False):
        self.label_lookup = label_lookup
        self.cache_root = cache_root
        self.input_dim = input_dim
        self.slice_num = slice_num
        self.transform = transform

        # Filter by available cache
        index_csv = os.path.join(cache_root, "cache_cls_index.csv")
        if os.path.exists(index_csv):
            idx_df = pd.read_csv(index_csv)
            cached_set = set(idx_df[idx_df['cache_exists'] == 1]['exam_id'].astype(str))
        else:
            cached_set = set(f.replace('.pt', '')
                             for f in os.listdir(cache_root) if f.endswith('.pt'))

        self.exam_ids = [eid for eid in exam_ids if eid in cached_set]
        missing = len(exam_ids) - len(self.exam_ids)
        if missing > 0:
            print(f"WARNING: {missing} exams missing cache, skipped")

        self.org_len = len(self.exam_ids)
        self.aug_indx_map = list(range(self.org_len))

    def balance_cls(self, cls_indx):
        """Oversample minority class (same as original CoPAS)."""
        if cls_indx == -1:
            self.aug_indx_map = list(range(self.org_len))
            return
        pos_list, neg_list = [], []
        for i, eid in enumerate(self.exam_ids):
            if self.label_lookup[eid][cls_indx] == 1:
                pos_list.append(i)
            else:
                neg_list.append(i)
        self.aug_indx_map = list(range(self.org_len))
        if len(pos_list) > len(neg_list):
            self.aug_indx_map.extend(
                np.random.choice(neg_list, len(pos_list) - len(neg_list)).tolist())
        elif len(neg_list) > len(pos_list):
            self.aug_indx_map.extend(
                np.random.choice(pos_list, len(neg_list) - len(pos_list)).tolist())

    def __len__(self):
        return len(self.aug_indx_map)

    def _reshape_vol(self, vol):
        """[Z, H, W] -> [1, Z, R, R]"""
        vol = vol.unsqueeze(0).unsqueeze(0)  # [1,1,Z,H,W]
        vol = F.interpolate(vol, size=(self.slice_num, self.input_dim, self.input_dim),
                            mode='trilinear', align_corners=False)
        return vol[0]  # [1,Z,R,R]

    def _apply_transform(self, vol, seed):
        """Augmentation matching original CoPAS."""
        s, h, w = vol.shape
        ct, cl = int(seed[0] * 20), int(seed[1] * 20)
        ch = int(h - ct - seed[2] * 10)
        cw = int(w - cl - seed[2] * 10)

        if seed[3] > 0.5:
            vol = TF.resized_crop(vol, ct, cl, ch, cw, [h, w], antialias=True)
        if seed[4] > 0.5:
            vol = TF.rotate(vol, angle=(seed[5] - 0.5) * 20, fill=0)
        if seed[6] > 0.5:
            vol = TF.adjust_contrast(
                vol.unsqueeze(1), contrast_factor=seed[7] * 0.5 + 0.75).squeeze(1)
            vol = TF.adjust_brightness(
                vol.unsqueeze(1), brightness_factor=seed[8] * 0.5 + 0.75).squeeze(1)
        if seed[9] > 0.5:
            vol = TF.affine(vol, angle=0, translate=[0, 0],
                            shear=(seed[10] - 0.5) * 10, scale=1, fill=0)
        return vol

    def __getitem__(self, aug_idx):
        idx = self.aug_indx_map[aug_idx]
        exam_id = self.exam_ids[idx]

        # Load cache
        cache = torch.load(os.path.join(self.cache_root, f"{exam_id}.pt"),
                           map_location='cpu', weights_only=False)
        image = cache['image']  # [5, Z, H, W]
        cache_seq = cache.get('sequence_order',
                              ['axial_PD', 'coronal_PD', 'coronal_T2WI',
                               'sagittal_PD', 'sagittal_T1WI'])
        seq_to_idx = {s: i for i, s in enumerate(cache_seq)}

        if self.transform:
            seed = torch.rand(20).tolist()

        vols = []
        for seq_name in self.COPAS_ORDER:
            vol = image[seq_to_idx.get(seq_name, 0)]  # [Z, H, W]
            if self.transform:
                vol = self._apply_transform(vol, seed)
            vol = self._reshape_vol(vol)  # [1, Z, R, R]
            vols.append(vol)

        label = torch.FloatTensor(self.label_lookup[exam_id])
        return vols, label, exam_id
