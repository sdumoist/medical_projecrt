"""
Evaluation utilities for SFT outputs.

Handles JSON parsing from LLM-generated text, fuzzy string matching,
and per-task metric computation.
"""
import json
import re
from collections import defaultdict

from utils.constants import DISEASES, DISEASE_ANCHOR_SEQ


# ── JSON Parsing ─────────────────────────────────────────────────────────

def try_parse_json(text):
    """Try to parse JSON from LLM-generated text.

    Handles common issues: extra text before/after JSON, trailing commas.

    Returns (parsed_dict, success_bool).
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        # Remove trailing commas before } or ]
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass

    return {}, False


# ── Fuzzy String Matching ────────────────────────────────────────────────

def levenshtein_ratio(s1, s2):
    """Compute Levenshtein similarity ratio between two strings.

    Returns float in [0, 1] where 1 is exact match.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    # Quick check for exact match
    if s1 == s2:
        return 1.0

    # Use dynamic programming for edit distance
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    distance = prev[len2]
    max_len = max(len1, len2)
    return 1.0 - distance / max_len


def fuzzy_sentence_match(pred_sentences, gt_sentences, threshold=0.7):
    """Compute fuzzy sentence-level precision, recall, F1.

    Each gt sentence is matched to the best pred sentence by Levenshtein ratio.
    A match counts if ratio >= threshold.

    Returns dict with precision, recall, f1.
    """
    if not gt_sentences and not pred_sentences:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_sentences:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred_sentences:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    # For each gt, find best matching pred
    gt_matched = 0
    for gt in gt_sentences:
        best_ratio = max(levenshtein_ratio(gt, p) for p in pred_sentences)
        if best_ratio >= threshold:
            gt_matched += 1

    # For each pred, find best matching gt
    pred_matched = 0
    for p in pred_sentences:
        best_ratio = max(levenshtein_ratio(p, gt) for gt in gt_sentences)
        if best_ratio >= threshold:
            pred_matched += 1

    precision = pred_matched / len(pred_sentences) if pred_sentences else 0.0
    recall = gt_matched / len(gt_sentences) if gt_sentences else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1}


# ── Per-Task Metrics ─────────────────────────────────────────────────────

def eval_label_binary(pred_output, gt_output):
    """Evaluate label_binary task output.

    Returns dict with per-disease and macro metrics.
    """
    pred_labels = pred_output.get("labels", {})
    gt_labels = gt_output.get("labels", {})

    per_disease = {}
    tp_total, fp_total, fn_total, correct_total, total = 0, 0, 0, 0, 0

    for d in DISEASES:
        gt_entry = gt_labels.get(d, {})
        pred_entry = pred_labels.get(d, {})

        gt_lab = gt_entry.get("label", -1) if isinstance(gt_entry, dict) else gt_entry
        pred_lab = pred_entry.get("label", -1) if isinstance(pred_entry, dict) else pred_entry

        if gt_lab == -1:
            continue  # Skip masked

        total += 1
        correct = int(pred_lab == gt_lab)
        correct_total += correct

        # Binary F1 (positive = 1)
        tp = int(pred_lab == 1 and gt_lab == 1)
        fp = int(pred_lab == 1 and gt_lab != 1)
        fn = int(pred_lab != 1 and gt_lab == 1)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        per_disease[d] = {
            "correct": correct,
            "gt": gt_lab,
            "pred": pred_lab,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Macro F1
    disease_f1s = []
    for d, dm in per_disease.items():
        p = dm["tp"] / (dm["tp"] + dm["fp"]) if (dm["tp"] + dm["fp"]) > 0 else 0
        r = dm["tp"] / (dm["tp"] + dm["fn"]) if (dm["tp"] + dm["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        dm["f1"] = f1
        disease_f1s.append(f1)

    accuracy = correct_total / total if total > 0 else 0
    macro_f1 = sum(disease_f1s) / len(disease_f1s) if disease_f1s else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_disease": per_disease,
        "total_diseases_evaluated": total,
    }


def eval_diagnosis_chain(pred_output, gt_output):
    """Evaluate diagnosis_chain task output.

    Returns dict with label metrics, evidence hit rate, anchor accuracy,
    key_slice hit rate, field completeness.
    """
    result = {}

    # Label accuracy (reuse label_binary logic)
    pred_labels = {d: pred_output.get("labels", {}).get(d, -1) for d in DISEASES}
    gt_labels = {d: gt_output.get("labels", {}).get(d, -1) for d in DISEASES}

    correct = 0
    total = 0
    for d in DISEASES:
        if gt_labels[d] == -1:
            continue
        total += 1
        if pred_labels[d] == gt_labels[d]:
            correct += 1
    result["label_accuracy"] = correct / total if total > 0 else 0

    # Evidence hit rate (fuzzy match)
    pred_ev = pred_output.get("evidence", {})
    gt_ev = gt_output.get("evidence", {})
    ev_hits = 0
    ev_total = 0
    for d in DISEASES:
        gt_pos = gt_ev.get(d, {}).get("positive", [])
        pred_pos = pred_ev.get(d, {}).get("positive", [])
        if gt_pos:
            ev_total += len(gt_pos)
            for gt_sent in gt_pos:
                if pred_pos:
                    best = max(levenshtein_ratio(gt_sent, p) for p in pred_pos)
                    if best >= 0.7:
                        ev_hits += 1
    result["evidence_hit_rate"] = ev_hits / ev_total if ev_total > 0 else 0

    # Anchor sequence accuracy
    pred_anch = pred_output.get("anchor_sequence", {})
    anch_correct = 0
    anch_total = 0
    for d in DISEASES:
        gt_a = DISEASE_ANCHOR_SEQ.get(d, "")
        pred_a = pred_anch.get(d, "")
        if gt_a:
            anch_total += 1
            if pred_a == gt_a:
                anch_correct += 1
    result["anchor_seq_accuracy"] = anch_correct / anch_total if anch_total > 0 else 0

    # Key-slice hit rate (exact and ±1)
    pred_ks = pred_output.get("key_slice", {})
    gt_ks = gt_output.get("key_slice", {})
    ks_exact = 0
    ks_pm1 = 0
    ks_total = 0
    for d in DISEASES:
        gt_v = gt_ks.get(d)
        pred_v = pred_ks.get(d)
        if gt_v is not None and isinstance(gt_v, (int, float)):
            ks_total += 1
            if pred_v is not None and isinstance(pred_v, (int, float)):
                if int(pred_v) == int(gt_v):
                    ks_exact += 1
                if abs(int(pred_v) - int(gt_v)) <= 1:
                    ks_pm1 += 1
    result["key_slice_exact"] = ks_exact / ks_total if ks_total > 0 else 0
    result["key_slice_pm1"] = ks_pm1 / ks_total if ks_total > 0 else 0

    # Core chain completeness: labels, evidence, anchor_sequence, key_slice
    core_fields = ["labels", "evidence", "anchor_sequence", "key_slice"]
    core_present = sum(
        1 for f in core_fields if f in pred_output and pred_output[f])
    result["core_chain_completeness"] = core_present / len(core_fields)

    # Report completeness: structured_findings, structured_impression
    report_fields = ["structured_findings", "structured_impression"]
    report_present = sum(
        1 for f in report_fields if f in pred_output and pred_output[f])
    result["report_completeness"] = report_present / len(report_fields)

    # If findings/impression are present, compute fuzzy metrics
    pred_findings = pred_output.get("structured_findings", [])
    gt_findings = gt_output.get("structured_findings", [])
    if gt_findings:
        fm = fuzzy_sentence_match(pred_findings, gt_findings)
        result["findings_f1"] = fm["f1"]

    pred_impression = pred_output.get("structured_impression", [])
    gt_impression = gt_output.get("structured_impression", [])
    if gt_impression:
        im = fuzzy_sentence_match(pred_impression, gt_impression)
        result["impression_f1"] = im["f1"]

    return result


def eval_structured_findings(pred_output, gt_output):
    """Evaluate structured_findings task output."""
    pred = pred_output.get("structured_findings", [])
    gt = gt_output.get("structured_findings", [])
    match_metrics = fuzzy_sentence_match(pred, gt)

    exact = 1 if pred == gt else 0

    return {
        "exact_match": exact,
        "sentence_precision": match_metrics["precision"],
        "sentence_recall": match_metrics["recall"],
        "sentence_f1": match_metrics["f1"],
        "pred_count": len(pred),
        "gt_count": len(gt),
    }


def eval_structured_impression(pred_output, gt_output):
    """Evaluate structured_impression task output."""
    pred = pred_output.get("structured_impression", [])
    gt = gt_output.get("structured_impression", [])
    match_metrics = fuzzy_sentence_match(pred, gt)

    exact = 1 if pred == gt else 0

    return {
        "exact_match": exact,
        "sentence_precision": match_metrics["precision"],
        "sentence_recall": match_metrics["recall"],
        "sentence_f1": match_metrics["f1"],
        "pred_count": len(pred),
        "gt_count": len(gt),
    }


TASK_EVALUATORS = {
    "label_binary": eval_label_binary,
    "diagnosis_chain": eval_diagnosis_chain,
    "structured_findings": eval_structured_findings,
    "structured_impression": eval_structured_impression,
}


def evaluate_sample(task_type, pred_text, gt_text):
    """Evaluate a single SFT sample.

    Args:
        task_type: one of TASK_EVALUATORS keys
        pred_text: generated text from LLM
        gt_text: ground truth output string

    Returns:
        dict with metrics + parse_success flag
    """
    pred_output, pred_ok = try_parse_json(pred_text)
    gt_output, gt_ok = try_parse_json(gt_text)

    result = {
        "parse_success": int(pred_ok),
        "gt_parse_success": int(gt_ok),
    }

    if not pred_ok or not gt_ok:
        return result

    evaluator = TASK_EVALUATORS.get(task_type)
    if evaluator:
        task_metrics = evaluator(pred_output, gt_output)
        result.update(task_metrics)

    return result


def aggregate_metrics(results_list):
    """Aggregate metrics from multiple samples.

    Args:
        results_list: list of dicts from evaluate_sample

    Returns:
        dict with averaged metrics
    """
    if not results_list:
        return {}

    agg = defaultdict(list)
    for r in results_list:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                agg[k].append(v)

    return {k: sum(v) / len(v) for k, v in agg.items()}
