#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
score_diagnosis_chain.py
------------------------
Offline scoring tool for diagnosis_chain model outputs.

Reads a JSONL file with model predictions (one per line, with 'exam_id',
'task_type', 'prediction' fields) alongside reference JSONL, computes
per-exam and aggregate reward scores.

Usage:
    PYTHONPATH=. python scripts/score_diagnosis_chain.py \
        --pred_jsonl outputs/eval_predictions.jsonl \
        --ref_jsonl outputs_clean/sft_data/split/sft_diagnosis_chain_val.jsonl \
        --output outputs/eval_scores.json
"""
from __future__ import print_function

import os
import sys
import json
import argparse
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.reward_functions import compute_reward, REWARD_FUNCTIONS, DISEASES


def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Score diagnosis_chain (or other task) model outputs")
    parser.add_argument("--pred_jsonl", required=True,
                        help="JSONL with model predictions. Fields: exam_id, task_type, prediction")
    parser.add_argument("--ref_jsonl", required=True,
                        help="Reference JSONL (same format as SFT training data)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path for scores (default: stdout)")
    parser.add_argument("--task_type", default=None,
                        help="Override task_type for all predictions (default: use field in JSONL)")
    args = parser.parse_args()

    # Load references: build lookup by exam_id
    ref_records = load_jsonl(args.ref_jsonl)
    ref_lookup = {}
    for rec in ref_records:
        eid = rec.get("exam_id")
        tt = rec.get("task_type", "diagnosis_chain")
        ref_lookup[(eid, tt)] = rec.get("output", "{}")

    print("Loaded %d reference records" % len(ref_lookup))

    # Load predictions
    pred_records = load_jsonl(args.pred_jsonl)
    print("Loaded %d prediction records" % len(pred_records))

    # Score
    scores_per_task = defaultdict(list)
    per_exam = []

    for rec in pred_records:
        eid = rec.get("exam_id")
        tt = args.task_type or rec.get("task_type", "diagnosis_chain")
        prediction = rec.get("prediction", "")

        ref_str = ref_lookup.get((eid, tt))
        if ref_str is None:
            print("WARNING: No reference found for exam_id=%s task_type=%s" % (eid, tt))
            continue

        reward = compute_reward(tt, prediction, ref_str)
        scores_per_task[tt].append(reward)
        per_exam.append({
            "exam_id": eid,
            "task_type": tt,
            "reward": reward,
        })

    # Aggregate
    aggregate = {}
    for tt, rewards in scores_per_task.items():
        aggregate[tt] = {
            "count": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }

    all_rewards = [r["reward"] for r in per_exam]
    aggregate["overall"] = {
        "count": len(all_rewards),
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
    }

    result = {
        "aggregate": aggregate,
        "per_exam": per_exam,
    }

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Scores written to %s" % args.output)
    else:
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
