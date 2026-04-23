"""
Label mapper for converting raw labels to training labels.
Pure mapping layer - does NOT read files.

Design principle:
- label_mapper.py is a PURE mapping layer
- raw_labels should come from metadata or dataset via input parameter
- No file I/O, no JSONParser dependency in main flow
"""
from __future__ import print_function
import numpy as np
from typing import Dict, List, Tuple

from utils.constants import DISEASES


# ============================================================
# Core mapping functions (main interfaces)
# ============================================================

def map_single_label(
    raw_label: int,
    task_mode: str,
    label_status: str = None
) -> Tuple[int, int]:
    """Map a single raw label to training label and mask.

    Args:
        raw_label: raw label from metadata (0, 1, 2, or -1)
        task_mode: "binary" or "ternary"
        label_status: (reserved for future use) not used in v1

    Returns:
        train_label: label for training
        train_mask: 1 for valid, 0 for masked (no label)
    """
    if task_mode == "binary":
        # Binary: 1->1, 0->0, 2->0, -1->mask
        if raw_label == 1:
            return 1, 1
        elif raw_label == 0:
            return 0, 1
        elif raw_label == 2:
            return 0, 1
        elif raw_label == -1:
            return 0, 0
        else:
            return 0, 0

    elif task_mode == "ternary":
        # Ternary: 0->0(neg), 2->1(unc), 1->2(pos), -1->mask
        if raw_label == 0:
            return 0, 1
        elif raw_label == 2:
            return 1, 1
        elif raw_label == 1:
            return 2, 1
        elif raw_label == -1:
            return 0, 0
        else:
            return 0, 0

    else:
        raise ValueError("Unknown task_mode: %s" % task_mode)


def map_case_labels(
    raw_labels: Dict[str, int],
    task_mode: str,
    disease_order: List[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map all disease labels for one case.

    Args:
        raw_labels: {"SST": 1, "IST": 0, ...} from metadata row
        task_mode: "binary" or "ternary"
        disease_order: order of diseases

    Returns:
        train_labels: [num_diseases]
        train_masks: [num_diseases]
        raw_labels_arr: [num_diseases] (preserved for analysis)
    """
    if disease_order is None:
        disease_order = DISEASES

    n = len(disease_order)
    train_labels = np.zeros(n, dtype=np.int64)
    train_masks = np.zeros(n, dtype=np.float32)
    raw_labels_arr = np.zeros(n, dtype=np.int64)

    for i, disease in enumerate(disease_order):
        raw = raw_labels.get(disease, -1)
        raw_labels_arr[i] = raw
        train_label, train_mask = map_single_label(raw, task_mode)
        train_labels[i] = train_label
        train_masks[i] = train_mask

    return train_labels, train_masks, raw_labels_arr


# ============================================================
# LabelMapper class (pure mapping, no I/O)
# ============================================================

class LabelMapper:
    """Pure mapping layer for raw labels. No file I/O."""

    def __init__(
        self,
        mode: str = "binary",
        diseases: List[str] = None
    ):
        """Initialize mapper.

        Args:
            mode: "binary" or "ternary"
            diseases: list of disease abbreviations
        """
        self.mode = mode
        self.diseases = diseases if diseases is not None else DISEASES

        if mode == "binary":
            self.num_classes = 2
        elif mode == "ternary":
            self.num_classes = 3
        else:
            raise ValueError("Unknown mode: %s" % mode)

    def map_labels(
        self,
        raw_labels: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map raw labels to training labels.

        MAIN entry point for training.

        Args:
            raw_labels: {"SST": 1, "IST": 0, ...} from metadata row

        Returns:
            train_labels: [num_diseases]
            train_masks: [num_diseases]
        """
        train_labels, train_masks, _ = map_case_labels(
            raw_labels, self.mode, self.diseases
        )
        return train_labels, train_masks


# ============================================================
# Utility functions (metadata-based, no direct JSON reading)
# ============================================================

def create_train_val_split(
    exam_ids: List[str],
    raw_labels_lookup: Dict[str, Dict[str, int]],
    task_mode: str = "binary",
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Create train/val split by positive ratio binning.

    NOTE: Simplified stratification based on positive label ratio.
    NOT true multi-label stratification.
    For production, use sklearn's StratifiedKFold with multi-label.

    Args:
        exam_ids: list of exam IDs
        raw_labels_lookup: dict mapping exam_id -> {"SST": 1, "IST": 0, ...}
        task_mode: "binary" or "ternary"
        val_ratio: validation set ratio
        seed: random seed

    Returns:
        train_ids, val_ids
    """
    import random
    random.seed(seed)

    # Compute positive ratio from pre-computed lookup
    positive_counts = []
    for eid in exam_ids:
        raw_labels = raw_labels_lookup.get(eid, {})
        pos = sum(1 for d in DISEASES if raw_labels.get(d, -1) == 1)
        pos_ratio = pos / len(DISEASES)
        positive_counts.append(pos_ratio)

    positive_counts = np.array(positive_counts)

    # Bin by positive ratio
    bins = np.digitize(positive_counts, [0, 0.1, 0.3, 0.5, 1.0])

    train_ids = []
    val_ids = []

    for b in range(5):
        indices = np.where(bins == b)[0].tolist()
        random.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_ids.extend([exam_ids[i] for i in train_indices])
        val_ids.extend([exam_ids[i] for i in val_indices])

    return train_ids, val_ids


def get_label_counts_from_metadata(
    exam_ids: List[str],
    raw_labels_lookup: Dict[str, Dict[str, int]],
    task_mode: str = "binary"
) -> Dict[str, Dict]:
    """Get counts of each label class per disease.

    Args:
        exam_ids: list of exam IDs
        raw_labels_lookup: dict mapping exam_id -> {"SST": 1, "IST": 0, ...}
        task_mode: "binary" or "ternary"

    Returns:
        dict of counts per disease
    """
    counts = {d: {"positive": 0, "negative": 0, "uncertain": 0, "mask": 0}
             for d in DISEASES}

    for eid in exam_ids:
        raw_labels = raw_labels_lookup.get(eid, {})
        if not raw_labels:
            continue

        for disease in DISEASES:
            raw = raw_labels.get(disease, -1)
            train_label, train_mask = map_single_label(raw, task_mode)

            if train_mask == 0:
                counts[disease]["mask"] += 1
            elif task_mode == "binary":
                if train_label == 1:
                    counts[disease]["positive"] += 1
                else:
                    counts[disease]["negative"] += 1
            else:
                if train_label == 2:
                    counts[disease]["positive"] += 1
                elif train_label == 1:
                    counts[disease]["uncertain"] += 1
                else:
                    counts[disease]["negative"] += 1

    return counts