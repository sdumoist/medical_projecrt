#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_sft_jsonl.py
------------------
Generate 4 types of SFT JSONL from case JSONs + cache_loc data.

Usage:
    PYTHONPATH=. python scripts/build_sft_jsonl.py \
        --json_root /path/to/case_json \
        --output_dir outputs/sft_data \
        --task_types label_binary diagnosis_chain structured_findings structured_impression

Data sources:
    1. case_json/*.json  -> labels, evidence, findings, impression, metadata
    2. cache_loc_index.csv -> exam_id to cache file mapping
    3. cache_loc/<exam_id>.pt -> key_slices, roi_boxes (for diagnosis_chain)
"""
from __future__ import print_function

import os
import sys
import csv
import json
import glob
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.io import DISEASES, DISEASE_ANCHOR_SEQ
from sft.prompts import build_prompt_plain, TASK_TYPES


# ── Quality bucketing ────────────────────────────────────────────────────

def compute_quality_bucket(case_data):
    """Assign quality bucket A/B/C based on quality_flag and exclude."""
    qf = case_data.get("quality_flag", "unknown")
    exc = case_data.get("exclude_from_main_training", 0)
    if qf == "high" and exc == 0:
        return "A"
    if qf == "medium" and exc == 0:
        return "B"
    return "C"


def count_valid_labels(labels):
    """Count number of diseases with label in {0, 1, 2}."""
    return sum(1 for d in DISEASES if labels.get(d, -1) in (0, 1, 2))


def has_any_evidence(evidence_text):
    """Check if at least one disease has non-empty evidence."""
    for d in DISEASES:
        if evidence_text.get(d) and len(evidence_text[d]) > 0:
            return True
    return False


# ── Per-task eligibility ─────────────────────────────────────────────────

def check_eligibility(case_data, task_type):
    """Check if a case is eligible for a given task type.

    Returns True if eligible.
    """
    labels = case_data.get("labels", {})
    n_valid = count_valid_labels(labels)

    if task_type == "label_binary":
        return n_valid >= 5

    if task_type == "diagnosis_chain":
        evidence = case_data.get("evidence_text", {})
        return n_valid >= 5 and has_any_evidence(evidence)

    if task_type == "structured_findings":
        findings = case_data.get("structured_findings", [])
        qf = case_data.get("quality_flag", "unknown")
        return len(findings) > 0 and qf != "low"

    if task_type == "structured_impression":
        impression = case_data.get("structured_impression", [])
        qf = case_data.get("quality_flag", "unknown")
        return len(impression) > 0 and qf != "low"

    return False


def compute_train_policy(case_data):
    """Compute per-task eligibility flags."""
    return {
        "use_for_label": check_eligibility(case_data, "label_binary"),
        "use_for_chain": check_eligibility(case_data, "diagnosis_chain"),
        "use_for_findings": check_eligibility(case_data, "structured_findings"),
        "use_for_impression": check_eligibility(case_data, "structured_impression"),
    }


# ── Output builders ──────────────────────────────────────────────────────

def map_label_to_binary(raw_label):
    """Map raw label to binary: 1=positive, 0=negative/uncertain, -1=masked.

    Mapping:
        1  -> 1 (positive)
        0  -> 0 (negative)
        2  -> 0 (uncertain, mapped to negative for binary task)
        -1 -> -1 (masked / unavailable)
    """
    if raw_label == 1:
        return 1
    if raw_label in (0, 2):
        return 0
    return -1


def build_label_binary_output(case_data):
    """Build output string for label_binary task.

    Labels are strictly binary (1/0/-1). raw_label=2 (uncertain) is
    mapped to 0 with status preserved as "uncertain".
    """
    labels = case_data.get("labels", {})
    label_status = case_data.get("label_status", {})

    result = {"labels": {}}
    for d in DISEASES:
        raw_label = labels.get(d, -1)
        binary_label = map_label_to_binary(raw_label)
        status = label_status.get(d, "unknown")
        result["labels"][d] = {"label": binary_label, "status": status}

    return json.dumps(result, ensure_ascii=False)


def build_diagnosis_chain_output(case_data, loc_data=None):
    """Build output string for diagnosis_chain task.

    Args:
        case_data: dict from case JSON
        loc_data: dict with 'key_slices' and 'roi_boxes' from cache_loc .pt
    """
    labels = case_data.get("labels", {})
    evidence_text = case_data.get("evidence_text", {})
    negative_evidence = case_data.get("negative_evidence", {})

    result = {
        "labels": {},
        "evidence": {},
        "anchor_sequence": {},
        "key_slice": {},
        "roi_box": {},
    }

    for d in DISEASES:
        raw_label = labels.get(d, -1)
        result["labels"][d] = map_label_to_binary(raw_label)

        result["evidence"][d] = {
            "positive": evidence_text.get(d, []),
            "negative": negative_evidence.get(d, []),
        }

        result["anchor_sequence"][d] = DISEASE_ANCHOR_SEQ.get(d, "unknown")

        # key_slice and roi_box from cache_loc
        if loc_data:
            ks = loc_data.get("key_slices", {})
            rb = loc_data.get("roi_boxes", {})
            ks_val = ks.get(d)
            rb_val = rb.get(d)
            result["key_slice"][d] = ks_val if ks_val is not None and ks_val >= 0 else None
            result["roi_box"][d] = rb_val if rb_val else None
        else:
            result["key_slice"][d] = None
            result["roi_box"][d] = None

    return json.dumps(result, ensure_ascii=False)


def build_structured_findings_output(case_data):
    """Build output string for structured_findings task."""
    findings = case_data.get("structured_findings", [])
    result = {"structured_findings": findings}
    return json.dumps(result, ensure_ascii=False)


def build_structured_impression_output(case_data):
    """Build output string for structured_impression task."""
    impression = case_data.get("structured_impression", [])
    result = {"structured_impression": impression}
    return json.dumps(result, ensure_ascii=False)


OUTPUT_BUILDERS = {
    "label_binary": lambda cd, ld: build_label_binary_output(cd),
    "diagnosis_chain": lambda cd, ld: build_diagnosis_chain_output(cd, ld),
    "structured_findings": lambda cd, ld: build_structured_findings_output(cd),
    "structured_impression": lambda cd, ld: build_structured_impression_output(cd),
}


# ── Build metadata ───────────────────────────────────────────────────────

def build_metadata(case_data, loc_data=None):
    """Build metadata dict for a sample."""
    labels = case_data.get("labels", {})
    evidence = case_data.get("evidence_text", {})

    has_ks = False
    has_roi = False
    if loc_data:
        ks = loc_data.get("key_slices", {})
        rb = loc_data.get("roi_boxes", {})
        has_ks = any(v is not None and v >= 0 for v in ks.values())
        has_roi = any(v is not None and len(v) > 0 for v in rb.values())

    return {
        "quality_flag": case_data.get("quality_flag", "unknown"),
        "laterality": case_data.get("laterality", "unknown"),
        "sex": case_data.get("sex", "unknown"),
        "age": case_data.get("age", "unknown"),
        "num_valid_labels": count_valid_labels(labels),
        "has_evidence": has_any_evidence(evidence),
        "has_key_slice": has_ks,
        "has_roi": has_roi,
        "source_summary": case_data.get("source_summary", {}),
    }


# ── Load cache_loc index ────────────────────────────────────────────────

def load_cache_loc_index(index_path, project_root=None):
    """Load cache_loc_index.csv into a dict: exam_id -> row dict.

    The CSV has columns: exam_id, cache_path, {disease}_mask_available,
    {disease}_key_slice, ..., num_available_masks, has_any_mask, success.
    """
    if not index_path or not os.path.exists(index_path):
        return {}

    lookup = {}
    with open(index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["exam_id"]
            lookup[eid] = row
    return lookup


def load_loc_data_for_exam(exam_id, cache_loc_index, cache_loc_root, project_root):
    """Load key_slices and roi_boxes for an exam from cache_loc .pt file.

    Returns dict with 'key_slices' and 'roi_boxes', or None if unavailable.
    """
    row = cache_loc_index.get(exam_id)
    if not row:
        return None
    if row.get("success", "1") != "1":
        return None

    cache_path = row.get("cache_path", "")
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(project_root, cache_path)

    if not os.path.exists(cache_path):
        # Try cache_loc_root fallback
        alt_path = os.path.join(cache_loc_root, "%s.pt" % exam_id)
        if os.path.exists(alt_path):
            cache_path = alt_path
        else:
            return None

    try:
        import torch
        pt = torch.load(cache_path, map_location="cpu", weights_only=False)
        return {
            "key_slices": pt.get("key_slices", {}),
            "roi_boxes": pt.get("roi_boxes", {}),
        }
    except Exception as e:
        print("WARNING: Failed to load %s: %s" % (cache_path, e))
        return None


# ── Main ─────────────────────────────────────────────────────────────────

def build_sft_jsonl(args):
    json_root = args.json_root
    output_dir = args.output_dir
    task_types = args.task_types
    cache_loc_root = args.cache_loc_root
    cache_loc_index_path = args.cache_loc_index

    project_root = PROJECT_ROOT
    os.makedirs(output_dir, exist_ok=True)

    # Load cache_loc index
    cache_loc_index = load_cache_loc_index(cache_loc_index_path, project_root)
    print("Loaded cache_loc index: %d entries" % len(cache_loc_index))

    # Scan all case JSONs
    json_files = sorted(glob.glob(os.path.join(json_root, "*.json")))
    print("Found %d case JSONs in %s" % (len(json_files), json_root))

    # Open output files
    writers = {}
    for tt in task_types:
        path = os.path.join(output_dir, "sft_%s.jsonl" % tt)
        writers[tt] = open(path, "w", encoding="utf-8")

    # Statistics
    stats = {
        "total_scanned": 0,
        "excluded_low": 0,
        "excluded_postop": 0,
        "excluded_no_labels": 0,
        "per_task": {tt: {"count": 0, "bucket_A": 0, "bucket_B": 0, "bucket_C": 0}
                     for tt in task_types},
        "label_distribution": {d: {"positive": 0, "negative": 0, "uncertain": 0, "masked": 0}
                               for d in DISEASES},
        "field_completeness": {
            "evidence_nonempty": 0,
            "key_slice_available": 0,
            "roi_available": 0,
            "findings_nonempty": 0,
            "impression_nonempty": 0,
            "total_eligible": 0,
        },
    }

    need_loc = "diagnosis_chain" in task_types

    for jf in json_files:
        stats["total_scanned"] += 1

        with open(jf, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        exam_id = case_data.get("exam_id", os.path.basename(jf).replace(".json", ""))

        # Global exclusions
        if case_data.get("quality_flag") == "low":
            stats["excluded_low"] += 1
            continue
        if case_data.get("postoperative", 0) == 1:
            stats["excluded_postop"] += 1
            continue
        if case_data.get("exclude_from_main_training", 0) == 1:
            # Don't hard-exclude, but mark in bucket
            pass

        labels = case_data.get("labels", {})
        if count_valid_labels(labels) == 0:
            stats["excluded_no_labels"] += 1
            continue

        # Quality bucket
        bucket = compute_quality_bucket(case_data)

        # Train policy
        train_policy = compute_train_policy(case_data)

        # Load loc data if needed
        loc_data = None
        if need_loc and exam_id in cache_loc_index:
            loc_data = load_loc_data_for_exam(
                exam_id, cache_loc_index, cache_loc_root or "", project_root)

        # Metadata
        meta = build_metadata(case_data, loc_data)

        # Update field completeness
        stats["field_completeness"]["total_eligible"] += 1
        if has_any_evidence(case_data.get("evidence_text", {})):
            stats["field_completeness"]["evidence_nonempty"] += 1
        if loc_data and any(v is not None and v >= 0
                            for v in loc_data.get("key_slices", {}).values()):
            stats["field_completeness"]["key_slice_available"] += 1
        if loc_data and any(v is not None and len(v) > 0
                            for v in loc_data.get("roi_boxes", {}).values()):
            stats["field_completeness"]["roi_available"] += 1
        if len(case_data.get("structured_findings", [])) > 0:
            stats["field_completeness"]["findings_nonempty"] += 1
        if len(case_data.get("structured_impression", [])) > 0:
            stats["field_completeness"]["impression_nonempty"] += 1

        # Label distribution (from eligible cases only)
        for d in DISEASES:
            raw = labels.get(d, -1)
            if raw == 1:
                stats["label_distribution"][d]["positive"] += 1
            elif raw == 0:
                stats["label_distribution"][d]["negative"] += 1
            elif raw == 2:
                stats["label_distribution"][d]["uncertain"] += 1
            else:
                stats["label_distribution"][d]["masked"] += 1

        # Generate samples per task type
        for tt in task_types:
            # Check task-specific eligibility
            policy_key = {
                "label_binary": "use_for_label",
                "diagnosis_chain": "use_for_chain",
                "structured_findings": "use_for_findings",
                "structured_impression": "use_for_impression",
            }[tt]

            if not train_policy[policy_key]:
                continue

            # Build output
            output_str = OUTPUT_BUILDERS[tt](case_data, loc_data)

            # Build instruction (plain text for storage)
            instruction = build_prompt_plain(tt)

            sample = {
                "exam_id": exam_id,
                "task_type": tt,
                "quality_bucket": bucket,
                "instruction": instruction,
                "output": output_str,
                "train_policy": train_policy,
                "metadata": meta,
            }

            writers[tt].write(json.dumps(sample, ensure_ascii=False) + "\n")

            stats["per_task"][tt]["count"] += 1
            stats["per_task"][tt]["bucket_" + bucket] += 1

    # Close files
    for w in writers.values():
        w.close()

    # Compute rates
    total_eligible = stats["field_completeness"]["total_eligible"]
    if total_eligible > 0:
        fc = stats["field_completeness"]
        fc["evidence_nonempty_rate"] = round(fc["evidence_nonempty"] / total_eligible, 4)
        fc["key_slice_available_rate"] = round(fc["key_slice_available"] / total_eligible, 4)
        fc["roi_available_rate"] = round(fc["roi_available"] / total_eligible, 4)
        fc["findings_nonempty_rate"] = round(fc["findings_nonempty"] / total_eligible, 4)
        fc["impression_nonempty_rate"] = round(fc["impression_nonempty"] / total_eligible, 4)

    # Write summary
    summary_path = os.path.join(output_dir, "sft_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n=== SFT JSONL Build Summary ===")
    print("Total scanned: %d" % stats["total_scanned"])
    print("Excluded (low quality): %d" % stats["excluded_low"])
    print("Excluded (postoperative): %d" % stats["excluded_postop"])
    print("Excluded (no valid labels): %d" % stats["excluded_no_labels"])
    print("Eligible: %d" % total_eligible)
    print()
    for tt in task_types:
        ts = stats["per_task"][tt]
        print("  %s: %d samples (A=%d, B=%d, C=%d)" % (
            tt, ts["count"], ts["bucket_A"], ts["bucket_B"], ts["bucket_C"]))
    print()
    print("Summary written to: %s" % summary_path)
    for tt in task_types:
        print("  %s -> %s" % (tt, os.path.join(output_dir, "sft_%s.jsonl" % tt)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT JSONL from case JSONs + cache_loc")

    parser.add_argument("--json_root", required=True,
                        help="Root directory containing per-case JSON files")
    parser.add_argument("--output_dir", default="outputs/sft_data",
                        help="Output directory for JSONL files")
    parser.add_argument("--task_types", nargs="+", default=TASK_TYPES,
                        choices=TASK_TYPES,
                        help="Which task types to generate")
    parser.add_argument("--cache_loc_root", default=None,
                        help="Root directory for cache_loc .pt files")
    parser.add_argument("--cache_loc_index", default=None,
                        help="Path to cache_loc_index.csv")

    args = parser.parse_args()
    build_sft_jsonl(args)


if __name__ == "__main__":
    main()
