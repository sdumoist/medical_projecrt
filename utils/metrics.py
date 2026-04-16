"""
Metrics for evaluation.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from typing import Dict, List, Tuple
import torch


def compute_metrics_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict:
    """Compute binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except:
            pass

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0

    return metrics


def compute_metrics_ternary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict:
    """Compute multi-class classification metrics (3 classes: negative, uncertain, positive)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_neg": precision_score(y_true == 0, y_pred == 0, zero_division=0),
        "precision_unc": precision_score(y_true == 2, y_pred == 2, zero_division=0),
        "precision_pos": precision_score(y_true == 1, y_pred == 1, zero_division=0),
        "recall_neg": recall_score(y_true == 0, y_pred == 0, zero_division=0),
        "recall_unc": recall_score(y_true == 2, y_pred == 2, zero_division=0),
        "recall_pos": recall_score(y_true == 1, y_pred == 1, zero_division=0),
    }

    if y_prob is not None and y_prob.ndim == 2:
        try:
            metrics["auc_macro"] = roc_auc_score(y_prob, y_true, multi_class='ovr', average='macro')
        except:
            pass

    return metrics


def compute_per_disease_metrics(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    diseases: List[str],
    binary: bool = True
) -> Dict:
    """Compute metrics per disease."""
    results = {}
    for i, disease in enumerate(diseases):
        y_true = y_true_all[:, i]
        y_pred = y_pred_all[:, i]
        if binary:
            results[disease] = compute_metrics_binary(y_true, y_pred)
        else:
            results[disease] = compute_metrics_ternary(y_true, y_pred)
    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate metrics across multiple samples."""
    if not metrics_list:
        return {}

    # Take mean of all scalar metrics
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())

    aggregated = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values and isinstance(values[0], (int, float)):
            aggregated[key] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    return aggregated