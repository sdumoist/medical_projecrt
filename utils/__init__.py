"""Project utilities."""
from .seed import set_seed
from .constants import (
    DISEASES, DISEASE_ANCHOR_SEQ, NUM_DISEASES,
    SEQUENCE_ORDER, DISEASE_BRANCH_MAP, DISEASE_TO_BRANCH,
)
# Backward compat alias
SEQUENCE_TYPES = SEQUENCE_ORDER
from .io import (
    load_nifti, save_nifti, load_json_label,
    list_exam_ids, check_case_complete,
    get_image_path, get_json_path,
    DATA_ROOT, JSON_ROOT, NNUNET_ROOT,
)
from .losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss, FocalLoss
from .metrics import (
    compute_metrics_binary, compute_metrics_ternary,
    compute_per_disease_metrics, compute_confusion_matrix, aggregate_metrics
)

__all__ = [
    "set_seed",
    "load_nifti", "save_nifti", "load_json_label",
    "list_exam_ids", "check_case_complete",
    "get_image_path", "get_json_path",
    "DATA_ROOT", "JSON_ROOT", "NNUNET_ROOT",
    "SEQUENCE_TYPES", "SEQUENCE_ORDER",
    "DISEASES", "DISEASE_ANCHOR_SEQ", "NUM_DISEASES",
    "DISEASE_BRANCH_MAP", "DISEASE_TO_BRANCH",
    "MaskedBCEWithLogitsLoss", "MaskedCrossEntropyLoss", "FocalLoss",
    "compute_metrics_binary", "compute_metrics_ternary",
    "compute_per_disease_metrics", "compute_confusion_matrix", "aggregate_metrics"
]