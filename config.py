"""
Central configuration for concordant user group definitions.

This module defines the patient groups used across all comparison analyses.
Each group is derived from permutation testing results with specific criteria.
Scripts import from here and accept a --group CLI argument to select which
group definition to use.
"""

from pathlib import Path

# All 22 patients in the AAMOS-00 dataset
ALL_PATIENTS = [
    113, 190, 217, 278, 294, 328, 343, 398, 447, 454,
    473, 514, 562, 625, 701, 702, 748, 764, 808, 867, 917, 939
]

# 15 patients with sufficient data variation for concordance analysis
ASSESSED_PATIENTS = [
    113, 190, 294, 328, 343, 398, 447, 454, 473, 514, 625, 701, 702, 917, 939
]

# 9 assessed patients who provided end-of-study feedback
FEEDBACK_PATIENTS = [113, 190, 294, 343, 473, 514, 701, 702, 939]

GROUPS = {
    "concordant": {
        "patients": [294, 473, 702],
        "label": "Concordant",
        "description": "Mean-concordant users (primary analysis)",
        # Derived from: results/permutation_by_measure/mean/permutation_test_results.csv
        # Criteria: mean Fisher's Z >= 0.5 AND Bonferroni-significant (p < 0.0033)
        # These 3 patients met BOTH mean and median criteria.
        # 294: mean Z=0.520, p<0.0001
        # 473: mean Z=0.525, p<0.0001
        # 702: mean Z=0.779, p<0.0001
    },
    "median_concordant": {
        "patients": [294, 473, 702, 917],
        "label": "Median-Concordant",
        "description": "Median-concordant users (sensitivity analysis)",
        # Derived from: results/permutation_by_measure/median/permutation_test_results.csv
        # Criteria: median Fisher's Z >= 0.5 AND Bonferroni-significant (p < 0.0033)
        # Patient 917 met median criteria (Z=1.07, p=0.0025) but NOT mean criteria
        # (mean Z=1.00, p=0.008 > 0.0033 Bonferroni threshold).
        # 294: median Z=0.555, p<0.0001
        # 473: median Z=0.626, p<0.0001
        # 702: median Z=0.825, p<0.0001
        # 917: median Z=1.073, p=0.0025
    },
}

RESULTS_BASE = Path("./results")

# Concordance threshold used across analyses
CONCORDANCE_Z_THRESHOLD = 0.45  # Rounds to 0.5 at 1dp; corresponds to rho ~ 0.46


def get_group_config(group_name: str) -> dict:
    """Get group configuration by name, with validation."""
    if group_name not in GROUPS:
        valid = ", ".join(GROUPS.keys())
        raise ValueError(f"Unknown group '{group_name}'. Valid groups: {valid}")
    return GROUPS[group_name]


def get_remaining_patients(group_name: str) -> list:
    """Get the remaining assessed patients (those NOT in the concordant group)."""
    concordant = set(get_group_config(group_name)["patients"])
    return [p for p in ASSESSED_PATIENTS if p not in concordant]


def get_output_dir(group_name: str, analysis_type: str) -> Path:
    """Get the output directory for a given group and analysis type."""
    output_dir = RESULTS_BASE / group_name / analysis_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
