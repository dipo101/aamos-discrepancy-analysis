"""Real-data regression against the frozen v1 results that the manuscript reports.

These tests rerun the refactored code on the real timestamped AAMOS-00 data
and compare with the artifacts committed under ``results/v1/``:

* ``per_patient_analysis/per_patient_correlation_results.csv`` is the
  per-patient, per-configuration Fisher Z table (the observed multiverse).
  The observed mean and median Z used by the permutation tests are the mean
  and median of the *finite* values in this table, which
  ``test_frozen_observed_statistics_are_derived_from_per_config_table``
  checks without needing the raw data.
* ``permutation_aggregate_extended/median_concordant/all_permutations.parquet``
  holds the per-permutation null summaries produced by the Cloud Run
  job-worker (seed 42 + permutation index). A handful of permutations are
  recomputed locally and compared.

The tests are skipped when the raw data is absent (see conftest.py).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import generate_param_combinations, run_single_permutation, summarize_correlations

KEY = ["user_key", "timestamp_window", "use_daily_max_windows", "use_calendar_days",
       "filter_out_zero_usage_entries", "categorization_method"]

# Patients chosen for the permutation check: the smallest (917, ~7 days), a
# mid-size discordant one (454) and one concordant user (294). Each
# permutation recomputes all 132 configurations, so the count is kept small.
PERMUTATION_CASES = [(917, [0, 1, 2, 3, 4]), (454, [0, 1, 2]), (294, [0, 1])]


# --------------------------------------------------------------------------
# Provenance of the frozen artifacts (no raw data needed)
# --------------------------------------------------------------------------

def test_frozen_observed_statistics_are_derived_from_per_config_table(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    finite = per_config[np.isfinite(per_config["spearman_z"])]
    derived = finite.groupby("user_key")["spearman_z"].agg(["mean", "median"])

    for measure, column in [("mean", "observed_mean_z"), ("median", "observed_median_z")]:
        frozen = pd.read_csv(frozen_results_dir / "permutation_by_measure" / measure / "permutation_test_results.csv")
        frozen = frozen.set_index("patient_id")[column]
        diff = (frozen - derived.loc[frozen.index, measure]).abs().max()
        assert diff < 1e-12, f"{measure}: max |diff| = {diff}"

    summary = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_summary_statistics.csv").set_index("user_key")
    assert (summary["mean_spearman_z"] - derived.loc[summary.index, "mean"]).abs().max() < 1e-12
    assert (summary["median_spearman_z"] - derived.loc[summary.index, "median"]).abs().max() < 1e-12


def test_group_config_matches_centred_rule_on_v1_null():
    from config import GROUPS
    from tests.test_summary import V1_CONCORDANT_MEAN, V1_CONCORDANT_MEDIAN
    assert sorted(GROUPS["concordant"]["patients"]) == V1_CONCORDANT_MEAN
    assert sorted(GROUPS["median_concordant"]["patients"]) == V1_CONCORDANT_MEDIAN


# --------------------------------------------------------------------------
# Observed multiverse: rerun the per-patient script and diff the Z table
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recomputed_per_patient(data_dir):
    from per_patient_correlation_analysis import (
        CategorizationMethod,
        PerPatientCorrelationAnalysis,
        PerPatientCorrelationConfig,
    )

    config = PerPatientCorrelationConfig(
        min_duration_threshold=0,
        use_end_date=True,
        compare_inhaler_dates=False,
        timestamp_windows=[12, 24, 36, 48],
        use_daily_max_windows=[False, True],
        use_calendar_days=[False, True],
        filter_out_zero_usage_entries=[False, True],
        categorization_methods=[
            CategorizationMethod.ONE_HOT,
            CategorizationMethod.LOWER_BOUND,
            CategorizationMethod.MIDPOINT,
            CategorizationMethod.UPPER_BOUND,
            CategorizationMethod.MIDPOINT_WITH_INHALER,
            CategorizationMethod.UPPER_BOUND_WITH_INHALER,
        ],
        log_level=logging.WARNING,
        min_samples_per_patient=5,
    )
    return PerPatientCorrelationAnalysis(config, data_dir=data_dir).run_analysis()


def test_per_config_z_table_matches_frozen(recomputed_per_patient, frozen_results_dir):
    frozen = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    ours = recomputed_per_patient

    frozen_keys = set(map(tuple, frozen[KEY].itertuples(index=False)))
    our_keys = set(map(tuple, ours[KEY].itertuples(index=False)))
    assert our_keys == frozen_keys, (
        f"row sets differ: {len(our_keys - frozen_keys)} extra, {len(frozen_keys - our_keys)} missing"
    )

    merged = frozen.merge(ours, on=KEY, suffixes=("_frozen", "_ours"))
    assert len(merged) == len(frozen)
    for col in ["spearman_corr", "spearman_z", "pearson_corr", "pearson_z", "sample_size"]:
        a, b = merged[f"{col}_frozen"].to_numpy(float), merged[f"{col}_ours"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
        finite = np.isfinite(a) & np.isfinite(b)
        assert (both_nan | both_inf | finite).all(), f"{col}: finiteness differs"
        max_diff = np.abs(a[finite] - b[finite]).max() if finite.any() else 0.0
        assert max_diff < 1e-9, f"{col}: max |diff| = {max_diff}"


def test_per_patient_summary_matches_frozen(recomputed_per_patient, frozen_results_dir):
    finite = recomputed_per_patient[np.isfinite(recomputed_per_patient["spearman_z"])]
    derived = finite.groupby("user_key")["spearman_z"].agg(["mean", "median"])
    frozen = pd.read_csv(frozen_results_dir / "permutation_by_measure" / "mean" / "permutation_test_results.csv").set_index("patient_id")
    assert (frozen["observed_mean_z"] - derived.loc[frozen.index, "mean"]).abs().max() < 1e-9
    frozen_med = pd.read_csv(frozen_results_dir / "permutation_by_measure" / "median" / "permutation_test_results.csv").set_index("patient_id")
    assert (frozen_med["observed_median_z"] - derived.loc[frozen_med.index, "median"]).abs().max() < 1e-9


# --------------------------------------------------------------------------
# Permutation null: recompute a few permutations and compare with the parquet
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_frames(data_dir):
    questionnaire = pd.read_csv(data_dir / "anonym_aamos00_dailyquestionnaire_dt.csv")
    inhaler = pd.read_csv(data_dir / "anonym_aamos00_smartinhaler_dt.csv")
    return questionnaire, inhaler


@pytest.fixture(scope="module")
def frozen_null(frozen_results_dir):
    path = frozen_results_dir / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet"
    return pd.read_parquet(path).set_index(["patient_id", "permutation_idx"])


@pytest.mark.parametrize("patient_id,perm_indices", PERMUTATION_CASES)
def test_permutation_null_matches_frozen(raw_frames, frozen_null, patient_id, perm_indices):
    questionnaire, inhaler = raw_frames
    q = questionnaire[questionnaire["user_key"] == patient_id].copy()
    inh = inhaler[inhaler["user_key"] == patient_id].copy()
    combos = generate_param_combinations()

    for perm_idx in perm_indices:
        correlations, _ = run_single_permutation(patient_id, perm_idx, q, inh, combos, 42, "spearman")
        ours = summarize_correlations(correlations)
        frozen = frozen_null.loc[(patient_id, perm_idx)]
        assert ours["n_valid_configs"] == frozen["n_valid_configs"], (patient_id, perm_idx)
        for ours_key, frozen_key in [
            ("mean_fisher_z", "null_mean_z"), ("median_fisher_z", "null_median_z"),
            ("min_fisher_z", "null_min_z"), ("max_fisher_z", "null_max_z"),
            ("q25_fisher_z", "null_q25_z"), ("q75_fisher_z", "null_q75_z"),
            ("std_fisher_z", "null_std_z"),
        ]:
            assert abs(ours[ours_key] - frozen[frozen_key]) < 1e-9, (patient_id, perm_idx, ours_key)


@pytest.mark.parametrize("patient_id,perm_indices", [(917, [0, 1, 2, 3, 4]), (454, [0, 1, 2])])
def test_batch_per_config_resummarised_matches_frozen(raw_frames, frozen_null, patient_id, perm_indices):
    """Item 7: the stored per-config vectors, re-summarised, reproduce the v1 null summaries."""
    from aamos_concordance import null_from_per_config, run_batch
    questionnaire, inhaler = raw_frames
    q = questionnaire[questionnaire["user_key"] == patient_id].copy()
    inh = inhaler[inhaler["user_key"] == patient_id].copy()
    lo, hi = min(perm_indices), max(perm_indices) + 1
    records, per_config, _ = run_batch(patient_id, q, inh, generate_param_combinations(), lo, hi, 42, "spearman")
    null = null_from_per_config(per_config).set_index("permutation_idx")
    for perm_idx in perm_indices:
        frozen = frozen_null.loc[(patient_id, perm_idx)]
        row = null.loc[perm_idx]
        assert row["n_valid_configs"] == frozen["n_valid_configs"]
        for col in ["null_mean_z", "null_median_z", "null_min_z", "null_max_z", "null_q25_z", "null_q75_z", "null_std_z"]:
            assert abs(row[col] - frozen[col]) < 1e-9, (patient_id, perm_idx, col)
    # and the JSON-style records carry a Pearson summary alongside
    assert all("pearson" in r and "mean_fisher_z" in r["pearson"] for r in records)
