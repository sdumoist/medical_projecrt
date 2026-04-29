"""
GRPO Dataset: wraps SFTDataset for rollout-based training.

Each item provides:
    - images, input_ids, attention_mask (same as SFT)
    - output_str: reference output string (for reward computation)
    - task_type: for selecting reward function

The dataset does NOT include 'labels' for CE loss — GRPO uses reward signals only.
"""
from __future__ import print_function

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sft.dataset import SFTDataset, sft_collate_fn  # noqa: F401 (re-exported)


class GRPODataset(SFTDataset):
    """GRPO dataset: same as SFTDataset but returns output_str for reward.

    SFTDataset already returns 'output_str' in addition to 'labels'.
    This subclass just makes the intent explicit and overrides __getitem__
    to ensure output_str is always present.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Ensure output_str is present (SFTDataset stores it)
        if "output_str" not in item:
            raise KeyError(
                "SFTDataset must return 'output_str' for GRPO. "
                "Check SFTDataset.__getitem__ implementation.")
        return item


def grpo_collate_fn(batch):
    """Collate for GRPO: same as sft_collate_fn but preserves output_str list."""
    collated = sft_collate_fn(batch)
    # output_str is a list of strings (not tensorizable), preserve as list
    if "output_str" not in collated:
        collated["output_str"] = [b["output_str"] for b in batch]
    if "task_type" not in collated:
        collated["task_type"] = [b.get("task_type", "label_binary") for b in batch]
    return collated
