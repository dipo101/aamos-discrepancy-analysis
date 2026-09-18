"""Shared core for the AAMOS-00 concordance analysis.

This package is the single implementation of the questionnaire/inhaler
window join, the categorisation methods, the zero-usage filter and the
132-configuration multiverse. It replaces three previously divergent copies
(data_loader.py, per_patient_correlation_analysis.py and the Cloud Run
job-worker). Behaviour is pinned to the pre-refactor copies by
tests/test_join_equivalence.py, which runs the frozen originals in
tests/oracles/ side by side with this package.
"""

from .batch import PER_CONFIG_COLUMNS, null_from_per_config, run_batch
from .categorization import (
    CategorizationMethod,
    categorize_inhaler_usage,
    filter_zero_usage,
)
from .configs import dedupe_combinations, generate_param_combinations
from .pipeline import build_world, observed_per_config_table, world_frames
from .spans import Span, apply_span, patient_spans
from .engine import observed_per_config, run_patient_exact, run_patient_sampled
from .data import (
    DataIntegrityError,
    DataNotFoundError,
    RawData,
    find_data_dir,
    find_raw_file,
    load_raw,
    resolve_paths,
    verify_manifest,
    write_manifest,
)
from .join import (
    REFERENCE_DATE,
    add_timestamps,
    join_questionnaire_with_inhaler,
    join_multi_patient,
)
from .permutation import CORRELATION_TYPES, evaluate_permutation, run_single_permutation, summarize_correlations
from .provenance import build_record, read_sidecar, write_sidecar
from .summary import PRIMARY, V1_SPEC, SummarySpec

__all__ = [
    "build_world",
    "observed_per_config_table",
    "world_frames",
    "Span",
    "apply_span",
    "patient_spans",
    "observed_per_config",
    "run_patient_exact",
    "run_patient_sampled",
    "PER_CONFIG_COLUMNS",
    "null_from_per_config",
    "run_batch",
    "CategorizationMethod",
    "categorize_inhaler_usage",
    "filter_zero_usage",
    "DataIntegrityError",
    "DataNotFoundError",
    "RawData",
    "find_data_dir",
    "find_raw_file",
    "load_raw",
    "resolve_paths",
    "verify_manifest",
    "write_manifest",
    "PRIMARY",
    "V1_SPEC",
    "SummarySpec",
    "build_record",
    "read_sidecar",
    "write_sidecar",
    "dedupe_combinations",
    "generate_param_combinations",
    "REFERENCE_DATE",
    "add_timestamps",
    "join_questionnaire_with_inhaler",
    "join_multi_patient",
    "CORRELATION_TYPES",
    "evaluate_permutation",
    "run_single_permutation",
    "summarize_correlations",
]
