"""Shared core for the AAMOS-00 concordance analysis.

This package is the single implementation of the questionnaire/inhaler
window join, the categorisation methods, the zero-usage filter and the
132-configuration multiverse. It replaces three previously divergent copies
(data_loader.py, per_patient_correlation_analysis.py and the Cloud Run
job-worker). Behaviour is pinned to the pre-refactor copies by
tests/test_join_equivalence.py, which runs the frozen originals in
tests/oracles/ side by side with this package.
"""

from .categorization import (
    CategorizationMethod,
    categorize_inhaler_usage,
    filter_zero_usage,
)
from .configs import generate_param_combinations
from .join import (
    REFERENCE_DATE,
    add_timestamps,
    join_questionnaire_with_inhaler,
    join_multi_patient,
)
from .permutation import run_single_permutation, summarize_correlations

__all__ = [
    "CategorizationMethod",
    "categorize_inhaler_usage",
    "filter_zero_usage",
    "generate_param_combinations",
    "REFERENCE_DATE",
    "add_timestamps",
    "join_questionnaire_with_inhaler",
    "join_multi_patient",
    "run_single_permutation",
    "summarize_correlations",
]
