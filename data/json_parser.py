"""
JSON parser for shoulder MRI labels.
Reads new format JSON with labels, label_status, evidence_text, etc.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from utils.io import load_json_label, get_json_path


class JSONParser:
    """Parser for new format shoulder MRI JSON labels."""

    def __init__(self, exam_id: str):
        self.exam_id = exam_id
        self.data = None

    def load(self) -> Dict[str, Any]:
        """Load JSON data."""
        self.data = load_json_label(self.exam_id)
        return self.data

    def get_labels(self) -> Dict[str, int]:
        """Get raw labels (0, 1, 2, -1)."""
        if self.data is None:
            self.load()
        return self.data.get("labels", {})

    def get_label_status(self) -> Dict[str, str]:
        """Get label status (explicit_positive, explicit_negative, implicit_negative, etc)."""
        if self.data is None:
            self.load()
        return self.data.get("label_status", {})

    def get_evidence_text(self) -> Dict[str, List[str]]:
        """Get evidence text for positive labels."""
        if self.data is None:
            self.load()
        return self.data.get("evidence_text", {})

    def get_negative_evidence(self) -> Dict[str, List[str]]:
        """Get negative evidence for negative labels."""
        if self.data is None:
            self.load()
        return self.data.get("negative_evidence", {})

    def get_structured_findings(self) -> List[str]:
        """Get structured findings list."""
        if self.data is None:
            self.load()
        return self.data.get("structured_findings", [])

    def get_structured_impression(self) -> List[str]:
        """Get structured impression list."""
        if self.data is None:
            self.load()
        return self.data.get("structured_impression", [])

    def get_quality_flag(self) -> str:
        """Get quality flag (high, medium, low)."""
        if self.data is None:
            self.load()
        return self.data.get("quality_flag", "unknown")

    def get_exclude_flag(self) -> bool:
        """Get exclude from training flag."""
        if self.data is None:
            self.load()
        return bool(self.data.get("exclude_from_main_training", 0))

    def get_laterality(self) -> str:
        """Get laterality (left, right)."""
        if self.data is None:
            self.load()
        return self.data.get("laterality", "unknown")

    def get_sex(self) -> str:
        """Get patient sex."""
        if self.data is None:
            self.load()
        return self.data.get("sex", "unknown")

    def get_age(self) -> str:
        """Get patient age."""
        if self.data is None:
            self.load()
        return self.data.get("age", "unknown")

    def get_raw_findings(self) -> str:
        """Get raw findings text."""
        if self.data is None:
            self.load()
        return self.data.get("raw_findings", "")

    def get_raw_impression(self) -> str:
        """Get raw impression text."""
        if self.data is None:
            self.load()
        return self.data.get("raw_impression", "")

    def get_source_summary(self) -> Dict[str, str]:
        """Get source summary (findings, impression, both, none)."""
        if self.data is None:
            self.load()
        return self.data.get("source_summary", {})

    def has_valid_labels(self) -> bool:
        """Check if case has valid labels for training."""
        labels = self.get_labels()
        # Valid if at least one disease has label 0, 1, or 2
        return any(v in [0, 1, 2] for v in labels.values())

    def get_valid_diseases(self) -> List[str]:
        """Get list of diseases with valid labels (not -1)."""
        labels = self.get_labels()
        return [d for d, v in labels.items() if v != -1]


def load_exam_label(exam_id: str) -> Dict[str, Any]:
    """Convenience function to load exam label."""
    parser = JSONParser(exam_id)
    return parser.load()


def get_label_summary(exam_ids: List[str]) -> Dict[str, Dict]:
    """Get summary of labels for multiple exams."""
    summary = {}
    for eid in exam_ids:
        parser = JSONParser(eid)
        summary[eid] = {
            "labels": parser.get_labels(),
            "status": parser.get_label_status(),
            "quality": parser.get_quality_flag(),
            "exclude": parser.get_exclude_flag()
        }
    return summary