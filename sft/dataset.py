"""
SFT Dataset: loads JSONL samples + MRI cache files for multi-task SFT training.

Supports 4 task types: label_binary, diagnosis_chain, structured_findings,
structured_impression.

Each sample returns:
    - image: [5, 1, Z, H, W] MRI tensor (from cache .pt)
    - input_ids: [L] tokenized instruction
    - labels: [L] target token IDs (-100 for instruction positions)
    - attention_mask: [L]
    - task_type: str
    - exam_id: str
    - key_slices: [7] int tensor (from cache_loc .pt, -1 = unavailable)
    - disease_mask: [7] float tensor (1.0 = valid label, 0.0 = masked)
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

from sft.prompts import get_task_messages
from utils.constants import DISEASES, NUM_DISEASES


class SFTDataset(Dataset):
    """SFT dataset that loads JSONL samples and corresponding MRI images.

    Args:
        jsonl_paths: list of JSONL file paths to load
        cache_root: root directory for MRI cache .pt files
        cache_index_path: path to cache index CSV (cls or loc)
        tokenizer: HuggingFace tokenizer
        max_length: max sequence length for tokenization
        num_visual_tokens: number of visual token positions to reserve
        target_z: target Z dimension for images (pad/crop)
    """

    def __init__(
        self,
        jsonl_paths,
        cache_root,
        cache_index_path,
        tokenizer,
        max_length=2048,
        num_visual_tokens=3,
        target_z=20,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_visual_tokens = num_visual_tokens
        self.target_z = target_z

        # Load all samples from JSONL files
        self.samples = []
        for path in jsonl_paths:
            if not os.path.exists(path):
                print("WARNING: JSONL not found: %s" % path)
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

        print("Loaded %d SFT samples from %d files" % (
            len(self.samples), len(jsonl_paths)))

        # Build exam_id -> cache_path lookup
        self.cache_lookup = self._build_cache_lookup(
            cache_root, cache_index_path)

    def _build_cache_lookup(self, cache_root, index_path):
        """Build exam_id -> cache file path mapping from index CSV."""
        import csv
        lookup = {}

        if not index_path or not os.path.exists(index_path):
            # Fallback: scan cache_root for .pt files
            if cache_root and os.path.isdir(cache_root):
                for f in os.listdir(cache_root):
                    if f.endswith(".pt"):
                        eid = f.replace(".pt", "")
                        lookup[eid] = os.path.join(cache_root, f)
            return lookup

        with open(index_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row["exam_id"]
                cache_path = row.get("cache_path", "")
                if not os.path.isabs(cache_path):
                    # Resolve relative to project root
                    project_root = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), ".."))
                    cache_path = os.path.join(project_root, cache_path)
                lookup[eid] = cache_path

        return lookup

    def __len__(self):
        return len(self.samples)

    def _load_cache(self, exam_id):
        """Load MRI image, key_slices, and disease_mask from cache .pt file.

        Returns:
            image: [5, 1, Z, H, W] float tensor, Z padded/cropped to target_z
            key_slices: [7] long tensor (-1 = unavailable)
            disease_mask: [7] float tensor (1.0 = valid label, 0.0 = masked)
        """
        default_ks = torch.full((NUM_DISEASES,), -1, dtype=torch.long)
        default_mask = torch.zeros(NUM_DISEASES, dtype=torch.float)

        cache_path = self.cache_lookup.get(exam_id)
        if not cache_path or not os.path.exists(cache_path):
            return (torch.zeros(5, 1, self.target_z, 448, 448),
                    default_ks, default_mask)

        record = torch.load(cache_path, map_location="cpu", weights_only=False)
        image = record["image"]  # [5, Z, H, W]

        # Pad or crop Z to target_z
        Z = image.shape[1]
        if Z < self.target_z:
            pad = torch.zeros(5, self.target_z - Z, image.shape[2], image.shape[3])
            image = torch.cat([image, pad], dim=1)
        elif Z > self.target_z:
            image = image[:, :self.target_z]

        # Add channel dim: [5, Z, H, W] -> [5, 1, Z, H, W]
        image = image.unsqueeze(1).float()

        # Extract key_slices [7] from cache_loc record
        key_slices = default_ks.clone()
        if "key_slices" in record:
            raw_ks = record["key_slices"]
            if isinstance(raw_ks, torch.Tensor):
                # Tensor shape [7] or dict-like
                key_slices = raw_ks.long()
            elif isinstance(raw_ks, dict):
                for i, d in enumerate(DISEASES):
                    v = raw_ks.get(d)
                    if v is not None and v >= 0:
                        key_slices[i] = int(v)
            elif isinstance(raw_ks, (list, np.ndarray)):
                for i, v in enumerate(raw_ks):
                    if v is not None and v >= 0:
                        key_slices[i] = int(v)

        # Extract disease_mask [7] (label validity mask)
        disease_mask = default_mask.clone()
        if "mask" in record:
            raw_mask = record["mask"]
            if isinstance(raw_mask, torch.Tensor):
                disease_mask = raw_mask.float()
            elif isinstance(raw_mask, (list, np.ndarray)):
                disease_mask = torch.tensor(raw_mask, dtype=torch.float)
        elif "labels" in record:
            # Infer mask from labels: valid if label in {0, 1, 2}
            raw_labels = record["labels"]
            if isinstance(raw_labels, torch.Tensor):
                disease_mask = ((raw_labels >= 0) & (raw_labels <= 2)).float()

        return image, key_slices, disease_mask

    def _tokenize_sample(self, sample):
        """Tokenize instruction + output, create labels.

        Returns dict with input_ids, attention_mask, labels.
        """
        task_type = sample["task_type"]
        output_text = sample["output"]

        # Build prompt using tokenizer's chat template
        messages = get_task_messages(task_type)

        # Tokenize instruction part (for computing label mask)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        # Full text = instruction + output + eos
        full_text = prompt_text + output_text + self.tokenizer.eos_token

        # Tokenize full text
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length - self.num_visual_tokens,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Tokenize instruction part only to find where output starts
        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_length - self.num_visual_tokens,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_encoded["input_ids"])

        # Labels: -100 for instruction tokens, actual IDs for output tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Truncate labels to match input_ids length
        labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": prompt_len,
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        exam_id = sample["exam_id"]
        task_type = sample["task_type"]

        # Load image + grounding supervision signals
        image, key_slices, disease_mask = self._load_cache(exam_id)

        # Tokenize
        tok = self._tokenize_sample(sample)

        return {
            "image": image,
            "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(tok["labels"], dtype=torch.long),
            "task_type": task_type,
            "exam_id": exam_id,
            "prompt_len": tok["prompt_len"],
            "output_text": sample["output"],
            "key_slices": key_slices,
            "disease_mask": disease_mask,
        }


def sft_collate_fn(batch, pad_token_id=0, num_visual_tokens=3):
    """Custom collate function for SFT batches.

    Pads input_ids, attention_mask, labels to max length in batch.
    Prepends num_visual_tokens placeholder positions.
    Stacks images.
    """
    # Stack images (all should be same shape after padding in __getitem__)
    images = torch.stack([b["image"] for b in batch])

    # Pad sequences
    max_len = max(len(b["input_ids"]) for b in batch)

    input_ids_padded = []
    attention_mask_padded = []
    labels_padded = []

    for b in batch:
        seq_len = len(b["input_ids"])
        pad_len = max_len - seq_len

        # Pad input_ids with pad_token_id
        input_ids_padded.append(
            torch.cat([b["input_ids"],
                        torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        # Pad attention_mask with 0
        attention_mask_padded.append(
            torch.cat([b["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long)]))
        # Pad labels with -100
        labels_padded.append(
            torch.cat([b["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long)]))

    # Prepend visual token placeholders
    B = len(batch)
    vis_ids = torch.zeros(B, num_visual_tokens, dtype=torch.long)
    vis_mask = torch.ones(B, num_visual_tokens, dtype=torch.long)
    vis_labels = torch.full((B, num_visual_tokens), -100, dtype=torch.long)

    input_ids = torch.cat([vis_ids, torch.stack(input_ids_padded)], dim=1)
    attention_mask = torch.cat([vis_mask, torch.stack(attention_mask_padded)], dim=1)
    labels = torch.cat([vis_labels, torch.stack(labels_padded)], dim=1)

    return {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "task_types": [b["task_type"] for b in batch],
        "exam_ids": [b["exam_id"] for b in batch],
        "prompt_lens": [b["prompt_len"] for b in batch],
        "output_texts": [b["output_text"] for b in batch],
        "key_slices": torch.stack([b["key_slices"] for b in batch]),   # [B, 7]
        "disease_mask": torch.stack([b["disease_mask"] for b in batch]),  # [B, 7]
    }
