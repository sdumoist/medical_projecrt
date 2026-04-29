"""
GRPO Dataset: wraps SFTDataset for rollout-based training.

Each item provides:
    - image, input_ids, attention_mask (same as SFT)
    - output_text: reference output string (for reward computation)
    - task_type: for selecting reward function

The dataset does NOT include 'labels' for CE loss — GRPO uses reward signals only.
"""
from __future__ import print_function

import os
import sys
from functools import partial

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sft.dataset import SFTDataset, sft_collate_fn  # noqa: F401 (re-exported)


class GRPODataset(SFTDataset):
    """GRPO dataset: same as SFTDataset but asserts output_text is always present.

    SFTDataset already returns 'output_text' in addition to 'labels'.
    This subclass just makes the intent explicit.
    """

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Ensure output_text is present (SFTDataset stores it)
        if "output_text" not in item:
            raise KeyError(
                "SFTDataset must return 'output_text' for GRPO. "
                "Check SFTDataset.__getitem__ implementation.")
        return item


def grpo_collate_fn(batch, pad_token_id=0, num_visual_tokens=10):
    """Collate for GRPO: delegates to sft_collate_fn.

    The collated batch includes 'output_texts' and 'task_types' from
    sft_collate_fn, which are used by the GRPO reward and rollout code.
    """
    return sft_collate_fn(batch,
                          pad_token_id=pad_token_id,
                          num_visual_tokens=num_visual_tokens)


def make_grpo_collate(pad_token_id, num_visual_tokens):
    """Factory to create a grpo_collate_fn with bound pad/visual params."""
    return partial(grpo_collate_fn,
                   pad_token_id=pad_token_id,
                   num_visual_tokens=num_visual_tokens)
