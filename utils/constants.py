"""
Shared constants for the shoulder MRI project.

This module is the single source of truth for disease names, sequence types,
anchor mappings, and branch routing. All other modules should import from here.
"""

# Canonical disease ordering (0-indexed)
DISEASES = ["SST", "IST", "SSC", "LHBT", "IGHL", "RIPI", "GHOA"]
NUM_DISEASES = len(DISEASES)

# MRI sequence ordering (must match config data.sequences)
SEQUENCE_ORDER = [
    "axial_PD",
    "coronal_PD",
    "coronal_T2WI",
    "sagittal_PD",
    "sagittal_T1WI",
]

# Disease -> best anchor MRI sequence for that disease
DISEASE_ANCHOR_SEQ = {
    "SST":  "coronal_PD",    # 冈上肌腱
    "IST":  "axial_PD",      # 冈下肌腱
    "SSC":  "axial_PD",      # 肩胛下肌腱
    "LHBT": "coronal_PD",    # 肱二头肌长头腱
    "IGHL": "coronal_PD",    # 盂肱下韧带/腋囊
    "RIPI": "sagittal_PD",   # 肩袖间隙
    "GHOA": "coronal_PD",    # 盂肱关节退行性变
}

# Branch SliceHead routing: branch_name -> list of disease indices
# Each branch's SliceHead only predicts key_slice for its assigned diseases
DISEASE_BRANCH_MAP = {
    "cor": [0, 3, 4, 6],   # SST, LHBT, IGHL, GHOA -> coronal_PD
    "axi": [1, 2],          # IST, SSC -> axial_PD
    "sag": [5],             # RIPI -> sagittal_PD
}

# Reverse mapping: disease_index -> branch_name
DISEASE_TO_BRANCH = {}
for _branch, _indices in DISEASE_BRANCH_MAP.items():
    for _idx in _indices:
        DISEASE_TO_BRANCH[_idx] = _branch
