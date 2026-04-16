"""
Export tokens from trained model for downstream tasks (Qwen SFT/RL).
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
import argparse

from models import create_model
from data.shoulder_dataset import ShoulderDataset
from data.label_mapper import LabelMapper
from utils.io import list_exam_ids, load_json_label


def export_tokens(
    model_path: str,
    exam_ids: List[str],
    data_root: str,
    json_root: str,
    sequences: List[str],
    output_dir: str,
    encoder: str = "resnet18",
    num_classes: int = 2,
    crop_size: tuple = (96, 96, 64),
    input_spacing: tuple = (1.0, 1.0, 1.0),
    batch_size: int = 1,
    device: str = "cuda"
):
    """Export tokens from trained model."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)

    model = create_model(
        encoder=encoder,
        num_classes=num_classes,
        num_diseases=7,
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create dataset
    label_mapper = LabelMapper(mode="binary" if num_classes == 2 else "ternary")

    dataset = ShoulderDataset(
        exam_ids=exam_ids,
        data_root=data_root,
        json_root=json_root,
        sequences=sequences,
        label_mapper=label_mapper,
        crop_size=crop_size,
        input_spacing=input_spacing,
        mode="val"
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Export tokens
    os.makedirs(output_dir, exist_ok=True)

    all_tokens = []
    all_labels = []
    all_exam_ids = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"]
        exam_ids_batch = batch["exam_id"]

        # Get token
        with torch.no_grad():
            token = model.get_token(images)  # [B, hidden_dim]

        all_tokens.append(token.cpu().numpy())
        all_labels.append(labels.numpy())
        all_exam_ids.extend(exam_ids_batch)

    tokens = np.concatenate(all_tokens, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Save tokens
    token_path = os.path.join(output_dir, "tokens.npz")
    np.savez(token_path, tokens=tokens, labels=labels, exam_ids=all_exam_ids)
    print(f"Tokens saved to {token_path}")

    # Save metadata
    metadata = {
        "model_path": model_path,
        "num_samples": len(all_exam_ids),
        "token_dim": tokens.shape[1],
        "encoder": encoder,
        "num_classes": num_classes,
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return tokens, labels, all_exam_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, help="Model checkpoint")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--data-root", required="/mnt/cfs_algo_bj/models/experiments/lirunze/data/Shoulder")
    parser.add_argument("--json-root", required="/mnt/cfs_algo_bj/models/experiments/lirunze/code/shouder/final_output/to_extract/case_json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Get all exam IDs
    exam_ids = list_exam_ids()

    export_tokens(
        model_path=args.model,
        exam_ids=exam_ids,
        data_root=args.data_root,
        json_root=args.json_root,
        sequences=["axial_PD", "coronal_PD", "coronal_T2WI", "sagittal_PD", "sagittal_T1WI"],
        output_dir=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()