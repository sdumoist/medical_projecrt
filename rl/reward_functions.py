"""
Reward functions for GRPO training on shoulder MRI structured diagnosis.

Each reward function takes a model generation (string) and reference data,
returns a scalar reward in [0, 1].

Reward hierarchy (all tasks):
    R_total = R_format * (R_content if R_format > 0 else 0)

Where:
    R_format  = 1.0 if output is valid JSON with required keys, else 0.0
    R_content = task-specific content reward

Task-specific content rewards:
    label_binary:     macro F1 over 7 diseases (binary, masked diseases excluded)
    diagnosis_chain:  0.4*label_F1 + 0.3*evidence_hit + 0.2*grounding_acc + 0.1*field_complete
    structured_findings/impression: sentence-level fuzzy F1
"""
from __future__ import print_function

import json
import re


# ── Utilities ─────────────────────────────────────────────────────────────

def safe_parse_json(text):
    """Try to parse JSON from model output. Returns dict or None."""
    text = text.strip()
    # Try to extract JSON block if wrapped in markdown
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        # Try to find first { ... } block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def binary_f1_masked(pred_labels, ref_labels, diseases):
    """Compute macro F1 over diseases, ignoring masked ones (label=-1).

    Args:
        pred_labels: dict {disease: int}
        ref_labels:  dict {disease: int or dict with 'label'}
        diseases:    list of disease names

    Returns:
        macro_f1: float in [0, 1]
    """
    def extract_label(v):
        if isinstance(v, dict):
            return v.get("label", -1)
        return int(v) if v is not None else -1

    tp = tn = fp = fn = 0
    disease_f1s = []
    for d in diseases:
        p = extract_label(pred_labels.get(d, -1))
        r = extract_label(ref_labels.get(d, -1))
        if r == -1:
            continue  # masked
        # binarize: treat 1 as positive, 0/2 as negative
        p_bin = 1 if p == 1 else 0
        r_bin = 1 if r == 1 else 0
        if r_bin == 1 and p_bin == 1:
            tp += 1
        elif r_bin == 0 and p_bin == 0:
            tn += 1
        elif r_bin == 0 and p_bin == 1:
            fp += 1
        else:
            fn += 1

    # Macro F1 as average of per-class F1 (positive and negative)
    total = tp + tn + fp + fn
    if total == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) \
        if (precision_neg + recall_neg) > 0 else 0.0

    return (f1_pos + f1_neg) / 2.0


def sentence_fuzzy_hit(pred_sentences, ref_sentences, threshold=0.5):
    """Compute sentence-level fuzzy F1 using token overlap.

    Args:
        pred_sentences: list of strings
        ref_sentences:  list of strings
        threshold:      minimum overlap ratio to count as a match

    Returns:
        f1: float in [0, 1]
    """
    if not ref_sentences:
        return 1.0 if not pred_sentences else 0.0
    if not pred_sentences:
        return 0.0

    def token_overlap(a, b):
        ta = set(a.split())
        tb = set(b.split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta), len(tb))

    # Precision: fraction of predictions that match some ref
    precision = sum(
        1 for p in pred_sentences
        if any(token_overlap(p, r) >= threshold for r in ref_sentences)
    ) / len(pred_sentences)

    # Recall: fraction of refs matched by some prediction
    recall = sum(
        1 for r in ref_sentences
        if any(token_overlap(p, r) >= threshold for p in pred_sentences)
    ) / len(ref_sentences)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]


# ── Reward functions ───────────────────────────────────────────────────────

def reward_label_binary(generation, reference):
    """Reward for label_binary task.

    Args:
        generation: model output string
        reference:  dict with 'labels' key from JSONL 'output' field

    Returns:
        reward: float in [0, 1]
    """
    parsed = safe_parse_json(generation)
    if parsed is None or "labels" not in parsed:
        return 0.0  # format failure

    pred_labels = parsed["labels"]
    ref_labels = reference.get("labels", {})

    return binary_f1_masked(pred_labels, ref_labels, DISEASES)


def reward_diagnosis_chain(generation, reference):
    """Reward for diagnosis_chain task.

    Composite: 0.4*label_F1 + 0.3*evidence_hit + 0.2*grounding_acc + 0.1*field_complete

    Args:
        generation: model output string
        reference:  dict with labels, evidence, visual_grounding keys

    Returns:
        reward: float in [0, 1]
    """
    parsed = safe_parse_json(generation)
    if parsed is None:
        return 0.0

    # Format check: must have at least labels
    if "labels" not in parsed:
        return 0.0

    # 1. Label F1 (weight 0.4)
    pred_labels = parsed.get("labels", {})
    ref_labels = reference.get("labels", {})
    r_label = binary_f1_masked(pred_labels, ref_labels, DISEASES)

    # 2. Evidence hit rate (weight 0.3)
    # For each disease with reference evidence, check if prediction has non-empty evidence
    r_evidence = 0.0
    ref_ev = reference.get("evidence", {})
    n_ref_ev = 0
    n_pred_ev = 0
    for d in DISEASES:
        ref_pos = ref_ev.get(d, {}).get("positive", [])
        if ref_pos:
            n_ref_ev += 1
            pred_pos = parsed.get("evidence", {}).get(d, {}).get("positive", [])
            if pred_pos:
                n_pred_ev += 1
    r_evidence = n_pred_ev / n_ref_ev if n_ref_ev > 0 else 0.5  # no ref = neutral

    # 3. Grounding accuracy (weight 0.2)
    # Check key_slice presence + rough roi_box validity
    r_grounding = 0.0
    ref_vg = reference.get("visual_grounding", {})
    pred_vg = parsed.get("visual_grounding", {})
    n_ks_ref = sum(1 for d in DISEASES
                   if ref_vg.get(d, {}).get("key_slice") is not None)
    n_ks_pred = sum(1 for d in DISEASES
                    if pred_vg.get(d, {}).get("key_slice") is not None)
    if n_ks_ref > 0:
        r_grounding = min(n_ks_pred, n_ks_ref) / n_ks_ref
    else:
        r_grounding = 0.5  # no ref grounding = neutral

    # 4. Field completeness (weight 0.1)
    required_keys = ["labels", "evidence", "anchor_sequence", "visual_grounding",
                     "structured_findings", "structured_impression"]
    r_fields = sum(1 for k in required_keys if k in parsed) / len(required_keys)

    reward = 0.4 * r_label + 0.3 * r_evidence + 0.2 * r_grounding + 0.1 * r_fields
    return reward


def reward_structured_findings(generation, reference):
    """Reward for structured_findings task.

    Args:
        generation: model output string
        reference:  dict with 'structured_findings' list

    Returns:
        reward: float in [0, 1]
    """
    parsed = safe_parse_json(generation)
    if parsed is None or "structured_findings" not in parsed:
        return 0.0

    pred_sents = parsed["structured_findings"]
    ref_sents = reference.get("structured_findings", [])
    if not isinstance(pred_sents, list):
        return 0.0

    return sentence_fuzzy_hit(pred_sents, ref_sents)


def reward_structured_impression(generation, reference):
    """Reward for structured_impression task.

    Args:
        generation: model output string
        reference:  dict with 'structured_impression' list

    Returns:
        reward: float in [0, 1]
    """
    parsed = safe_parse_json(generation)
    if parsed is None or "structured_impression" not in parsed:
        return 0.0

    pred_sents = parsed["structured_impression"]
    ref_sents = reference.get("structured_impression", [])
    if not isinstance(pred_sents, list):
        return 0.0

    return sentence_fuzzy_hit(pred_sents, ref_sents)


REWARD_FUNCTIONS = {
    "label_binary": reward_label_binary,
    "diagnosis_chain": reward_diagnosis_chain,
    "structured_findings": reward_structured_findings,
    "structured_impression": reward_structured_impression,
}


def compute_reward(task_type, generation, reference_output_str):
    """Compute reward for a single generation.

    Args:
        task_type: one of REWARD_FUNCTIONS keys
        generation: model output string (raw text after the prompt)
        reference_output_str: reference output string from JSONL (will be parsed)

    Returns:
        reward: float in [0, 1]
    """
    try:
        reference = json.loads(reference_output_str)
    except Exception:
        reference = {}

    fn = REWARD_FUNCTIONS.get(task_type)
    if fn is None:
        raise ValueError("Unknown task_type for reward: %s" % task_type)

    try:
        return fn(generation, reference)
    except Exception as e:
        print("WARNING: reward computation failed: %s" % e)
        return 0.0
