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
    """Compute multi-class classification metrics (3 classes: 0=neg, 1=unc, 2=pos).

    Label mapping (from label_mapper.py ternary mode):
        raw 0 -> train 0 (negative)
        raw 2 -> train 1 (uncertain)
        raw 1 -> train 2 (positive)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_neg": precision_score(y_true == 0, y_pred == 0, zero_division=0),
        "precision_unc": precision_score(y_true == 1, y_pred == 1, zero_division=0),
        "precision_pos": precision_score(y_true == 2, y_pred == 2, zero_division=0),
        "recall_neg": recall_score(y_true == 0, y_pred == 0, zero_division=0),
        "recall_unc": recall_score(y_true == 1, y_pred == 1, zero_division=0),
        "recall_pos": recall_score(y_true == 2, y_pred == 2, zero_division=0),
        "f1_neg": f1_score(y_true == 0, y_pred == 0, zero_division=0),
        "f1_unc": f1_score(y_true == 1, y_pred == 1, zero_division=0),
        "f1_pos": f1_score(y_true == 2, y_pred == 2, zero_division=0),
    }
    # Macro F1 across 3 classes
    metrics["f1"] = float(np.mean([metrics["f1_neg"], metrics["f1_unc"], metrics["f1_pos"]]))

    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 3:
        try:
            metrics["auc_macro"] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro')
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
        y_prob_all: binary: [N, 7] sigmoid probs; ternary: [N, 7, 3] softmax probs
        mask_all: [N, 7] label masks (1=valid, 0=ignore)
    """
    results = {}
    for i, disease in enumerate(diseases):
        y_true = y_true_all[:, i]
        y_pred = y_pred_all[:, i]
        if y_prob_all is not None:
            if binary:
                y_prob = y_prob_all[:, i]       # [N]
            else:
                y_prob = y_prob_all[:, i, :]    # [N, 3]
        else:
            y_prob = None

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

# =========================================================================
# ROI box metrics
# =========================================================================

def compute_box_iou_2d(pred_boxes, gt_boxes):
    """Compute IoU between predicted and ground-truth 2D boxes.

    Args:
        pred_boxes: [N, 4] numpy array, (x1, y1, x2, y2) in [0, 1]
        gt_boxes:   [N, 4] numpy array, (x1, y1, x2, y2) in [0, 1]

    Returns:
        iou: [N] float array
    """
    # Intersection
    ix1 = np.maximum(pred_boxes[:, 0], gt_boxes[:, 0])
    iy1 = np.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    ix2 = np.minimum(pred_boxes[:, 2], gt_boxes[:, 2])
    iy2 = np.minimum(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = np.maximum(ix2 - ix1, 0.0)
    inter_h = np.maximum(iy2 - iy1, 0.0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = np.maximum(pred_boxes[:, 2] - pred_boxes[:, 0], 0) * \
                np.maximum(pred_boxes[:, 3] - pred_boxes[:, 1], 0)
    gt_area   = np.maximum(gt_boxes[:, 2]   - gt_boxes[:, 0],   0) * \
                np.maximum(gt_boxes[:, 3]   - gt_boxes[:, 1],   0)
    union_area = pred_area + gt_area - inter_area + 1e-8

    return inter_area / union_area


def compute_box_metrics(pred_boxes_all, gt_boxes_all, valid_mask_all, diseases):
    """Compute per-disease and macro box IoU / L1 metrics.

    Args:
        pred_boxes_all: [N, 7, 4] numpy array of predicted boxes
        gt_boxes_all:   [N, 7, 4] numpy array of ground-truth boxes
        valid_mask_all: [N, 7]    numpy array, 1.0 where gt box exists
        diseases:       list of 7 disease names

    Returns:
        dict with per-disease and macro metrics:
            {disease}_box_iou, {disease}_box_l1
            {disease}_box_iou_at_03, {disease}_box_iou_at_05
            macro_box_iou, macro_box_l1
            macro_box_iou_at_03, macro_box_iou_at_05
    """
    results = {}
    iou_list = []
    l1_list = []

    for i, disease in enumerate(diseases):
        valid = valid_mask_all[:, i] > 0
        if valid.sum() == 0:
            results[disease + "_box_iou"] = 0.0
            results[disease + "_box_l1"] = 0.0
            results[disease + "_box_iou_at_03"] = 0.0
            results[disease + "_box_iou_at_05"] = 0.0
            continue

        pred = pred_boxes_all[valid, i]    # [M, 4]
        gt   = gt_boxes_all[valid, i]      # [M, 4]

        iou  = compute_box_iou_2d(pred, gt)    # [M]
        l1   = np.abs(pred - gt).mean(axis=-1)  # [M], mean over 4 coords

        mean_iou = float(iou.mean())
        mean_l1  = float(l1.mean())

        results[disease + "_box_iou"] = mean_iou
        results[disease + "_box_l1"]  = mean_l1
        results[disease + "_box_iou_at_03"] = float((iou >= 0.3).mean())
        results[disease + "_box_iou_at_05"] = float((iou >= 0.5).mean())

        iou_list.append(mean_iou)
        l1_list.append(mean_l1)

    results["macro_box_iou"]       = float(np.mean(iou_list)) if iou_list else 0.0
    results["macro_box_l1"]        = float(np.mean(l1_list))  if l1_list  else 0.0
    results["macro_box_iou_at_03"] = float(np.mean(
        [results[d + "_box_iou_at_03"] for d in diseases])) if iou_list else 0.0
    results["macro_box_iou_at_05"] = float(np.mean(
        [results[d + "_box_iou_at_05"] for d in diseases])) if iou_list else 0.0
    return results


# ── 2D Mask metrics ───────────────────────────────────────────────────────

def compute_mask_dice(pred_mask, gt_mask, threshold=0.5, smooth=1e-6):
    """Compute Dice coefficient for a single 2D binary mask pair.

    Args:
        pred_mask: ndarray [H, W] predicted probability or binary mask
        gt_mask:   ndarray [H, W] ground-truth binary mask
        threshold: binarization threshold for pred_mask
        smooth:    smoothing factor

    Returns:
        dice: float in [0, 1]
    """
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt_bin   = gt_mask.astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    return float((2.0 * intersection + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth))


def compute_mask_iou_2d(pred_mask, gt_mask, threshold=0.5, smooth=1e-6):
    """Compute IoU for a single 2D binary mask pair.

    Args:
        pred_mask: ndarray [H, W]
        gt_mask:   ndarray [H, W]
        threshold: binarization threshold
        smooth:    smoothing factor

    Returns:
        iou: float in [0, 1]
    """
    pred_bin = (pred_mask >= threshold).astype(np.float32)
    gt_bin   = gt_mask.astype(np.float32)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_mask_metrics(pred_masks_all, gt_masks_all, valid_mask_all, diseases,
                         threshold=0.5):
    """Compute per-disease and macro mask Dice and IoU.

    Args:
        pred_masks_all: ndarray [N, 7, H, W] predicted masks (probabilities or binary)
        gt_masks_all:   ndarray [N, 7, H, W] ground-truth binary masks
        valid_mask_all: ndarray [N, 7] float, 1.0 if this disease has a gt mask
        diseases:       list of 7 disease names
        threshold:      binarization threshold for predictions

    Returns:
        dict with per-disease and macro: dice, iou, iou_at_05
    """
    results = {}
    dice_list = []
    iou_list  = []

    for i, disease in enumerate(diseases):
        valid = valid_mask_all[:, i] > 0
        if valid.sum() == 0:
            results[disease + "_mask_dice"]      = 0.0
            results[disease + "_mask_iou"]       = 0.0
            results[disease + "_mask_iou_at_05"] = 0.0
            continue

        per_dice = []
        per_iou  = []
        for n in range(len(pred_masks_all)):
            if not valid[n]:
                continue
            d = compute_mask_dice(pred_masks_all[n, i], gt_masks_all[n, i], threshold)
            u = compute_mask_iou_2d(pred_masks_all[n, i], gt_masks_all[n, i], threshold)
            per_dice.append(d)
            per_iou.append(u)

        mean_dice = float(np.mean(per_dice))
        mean_iou  = float(np.mean(per_iou))
        results[disease + "_mask_dice"]      = mean_dice
        results[disease + "_mask_iou"]       = mean_iou
        results[disease + "_mask_iou_at_05"] = float(np.mean(
            [v >= 0.5 for v in per_iou]))

        dice_list.append(mean_dice)
        iou_list.append(mean_iou)

    results["macro_mask_dice"]      = float(np.mean(dice_list)) if dice_list else 0.0
    results["macro_mask_iou"]       = float(np.mean(iou_list))  if iou_list  else 0.0
    results["macro_mask_iou_at_05"] = float(np.mean(
        [results[d + "_mask_iou_at_05"] for d in diseases])) if iou_list else 0.0
    return results
