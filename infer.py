"""
Inference script for shoulder MRI classification.
"""
from __future__ import print_function
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, DISEASES
from data.shoulder_dataset import ShoulderDataset3D
from data.label_mapper import LabelMapper
from models import create_model
from utils.metrics import compute_per_disease_metrics


def load_model(checkpoint_path, config, device):
    """Load trained model."""
    model = create_model(
        encoder=config['model']['encoder'],
        num_classes=config['training']['num_classes'],
        num_diseases=len(config['data']['diseases']),
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def predict(model, loader, device):
    """Run inference."""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_exam_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"]
            exam_ids = batch["exam_id"]

            logits = model(images)

            if logits.dim() == 3:
                probs = torch.softmax(logits, dim=2)
                preds = logits.argmax(dim=2)
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_exam_ids.extend(exam_ids)

    all_preds = np.concatenate(all_preds, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_probs, all_labels, all_exam_ids


def evaluate(preds, probs, labels, metrics):
    """Evaluate predictions."""
    results = {}

    # Per-disease metrics
    for i, disease in enumerate(DISEASES):
        y_true = labels[:, i]
        y_pred = preds[:, i]

        # Skip if no valid labels
        mask = y_true >= 0
        if mask.sum() == 0:
            continue

        results[disease] = {
            "accuracy": (y_true[mask] == y_pred[mask]).mean(),
        }

    return results


def run_inference(config_path, checkpoint_path, exam_ids, output_path=None):
    """Run inference on test set."""

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    # Load model
    model = load_model(checkpoint_path, config, device)
    print("Loaded model from %s" % checkpoint_path)

    # Create dataset
    label_mapper = LabelMapper(
        mode="binary" if config['training']['num_classes'] == 2 else "ternary"
    )

    crop_size = tuple(config['model']['crop_size'])

    dataset = ShoulderDataset3D(
        exam_ids=exam_ids,
        data_root=config['data']['data_root'],
        json_root=config['data']['json_root'],
        sequences=config['data']['sequences'],
        label_mapper=label_mapper,
        crop_size=crop_size,
        mode="val"
    )

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Predict
    preds, probs, labels, exam_ids_out = predict(model, loader, device)

    # Evaluate
    results = evaluate preds, probs, labels, DISEASES

    print("\nResults:")
    for disease, metrics in results.items():
        print("  %s: Accuracy=%.4f" % (disease, metrics['accuracy']))

    # Save predictions
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output = {
            "exam_ids": exam_ids_out,
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
            "labels": labels.tolist(),
            "results": results
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print("\nResults saved to %s" % output_path)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", "-m", required=True)
    parser.add_argument("--output", "-o", default="outputs/experiments/predictions.json")
    parser.add_argument("--exam-ids", "-e", nargs="+", default=None)
    args = parser.parse_args()

    # Get exam IDs
    if args.exam_ids:
        exam_ids = args.exam_ids
    else:
        from utils.io import list_exam_ids
        exam_ids = list_exam_ids()[:10]  # Default: first 10

    run_inference(args.config, args.checkpoint, exam_ids, args.output)


if __name__ == "__main__":
    main()