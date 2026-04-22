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


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find optimal threshold that maximizes F1 for a single disease.

    Args:
        y_true: [N] binary ground-truth labels
        y_prob: [N] predicted probabilities
        metric: optimization target ('f1')

    Returns:
        best_thr: optimal threshold
        best_f1: F1 at optimal threshold
        best_precision: precision at optimal threshold
        best_recall: recall at optimal threshold
    """
    best_thr, best_f1 = 0.5, 0.0
    best_precision, best_recall = 0.0, 0.0
    for thr in np.arange(0.05, 0.96, 0.01):
        binary = (y_prob >= thr).astype(int)
        try:
            val = f1_score(y_true, binary, zero_division=0, pos_label=1)
        except ValueError:
            val = 0.0
        if val > best_f1:
            best_f1 = val
            best_thr = float(thr)
            best_precision = precision_score(y_true, binary, zero_division=0)
            best_recall = recall_score(y_true, binary, zero_division=0)
    return best_thr, best_f1, best_precision, best_recall


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

        # Optimal threshold search
        opt_thr, opt_f1, opt_prec, opt_rec = find_optimal_threshold(y_true, y_prob)
        metrics["opt_thr"] = opt_thr
        metrics["opt_f1"] = opt_f1
        metrics["opt_precision"] = opt_prec
        metrics["opt_recall"] = opt_rec

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
    binary: bool = True,
    y_prob_all: np.ndarray = None,
    mask_all: np.ndarray = None,
) -> Dict:
    """Compute metrics per disease.

    Args:
        y_true_all: [N, 7] ground-truth labels
        y_pred_all: [N, 7] hard predictions
        diseases: list of disease names
        binary: whether to use binary metrics
        y_prob_all: [N, 7] sigmoid probabilities (for AUC)
        mask_all: [N, 7] label masks (1=valid, 0=ignore)
    """
    results = {}
    for i, disease in enumerate(diseases):
        y_true = y_true_all[:, i]
        y_pred = y_pred_all[:, i]
        y_prob = y_prob_all[:, i] if y_prob_all is not None else None

        # Apply mask: only evaluate where label is valid
        if mask_all is not None:
            valid = mask_all[:, i] > 0
            y_true = y_true[valid]
            y_pred = y_pred[valid]
            if y_prob is not None:
                y_prob = y_prob[valid]

        if len(y_true) == 0:
            results[disease] = {
                "accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
            continue

        if binary:
            results[disease] = compute_metrics_binary(y_true, y_pred, y_prob)
        else:
            results[disease] = compute_metrics_ternary(y_true, y_pred, y_prob)
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


def compute_key_slice_metrics(
    pred_slices: np.ndarray,
    gt_slices: np.ndarray,
    valid_mask: np.ndarray,
    diseases: List[str],
) -> Dict:
    """Compute key-slice prediction accuracy per disease.

    Args:
        pred_slices: [N, 7] predicted key-slice indices (argmax of slice_logits)
        gt_slices: [N, 7] ground-truth key-slice indices
        valid_mask: [N, 7] mask for valid key-slice annotations (>= 0)
        diseases: list of disease names

    Returns:
        dict with per-disease and macro key-slice metrics:
            {disease}_ks_top1: exact match accuracy
            {disease}_ks_pm1: +/- 1 slice accuracy
            macro_ks_top1, macro_ks_pm1
    """
    results = {}
    top1_list = []
    pm1_list = []

    for i, disease in enumerate(diseases):
        valid = valid_mask[:, i] > 0
        if valid.sum() == 0:
            results[disease + "_ks_top1"] = 0.0
            results[disease + "_ks_pm1"] = 0.0
            continue
        pred = pred_slices[valid, i]
        gt = gt_slices[valid, i]
        top1 = float((pred == gt).mean())
        pm1 = float((np.abs(pred - gt) <= 1).mean())
        results[disease + "_ks_top1"] = top1
        results[disease + "_ks_pm1"] = pm1
        top1_list.append(top1)
        pm1_list.append(pm1)

    results["macro_ks_top1"] = float(np.mean(top1_list)) if top1_list else 0.0
    results["macro_ks_pm1"] = float(np.mean(pm1_list)) if pm1_list else 0.0
    return results