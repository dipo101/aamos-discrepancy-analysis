"""Item 7: batch records and per-configuration null storage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import (
    PER_CONFIG_COLUMNS,
    evaluate_permutation,
    generate_param_combinations,
    null_from_per_config,
    run_batch,
    summarize_correlations,
)
from tests.synthetic import make_dataset


def _nan_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)



@pytest.fixture(scope="module")
def batch_result():
    q, inh = make_dataset(2)
    combos = generate_param_combinations()
    uk = 104
    qq, ii = q[q["user_key"] == uk], inh[inh["user_key"] == uk]
    records, per_config, datasets = run_batch(uk, qq, ii, combos, 5, 9, 42, "spearman")
    return uk, qq, ii, combos, records, per_config, datasets


def test_batch_records_match_kernel_summaries(batch_result):
    uk, qq, ii, combos, records, per_config, _ = batch_result
    assert [r["permutation_idx"] for r in records] == [5, 6, 7, 8]
    for r in records:
        z, _ = evaluate_permutation(uk, r["permutation_idx"], qq, ii, combos, 42)
        expected_s = summarize_correlations(z["spearman"])
        expected_p = summarize_correlations(z["pearson"])
        for k, v in expected_s.items():
            assert r[k] == v, k
        assert r["pearson"] == expected_p


def test_per_config_table_shape_dtypes_and_alignment(batch_result):
    uk, qq, ii, combos, records, per_config, _ = batch_result
    assert list(per_config.columns) == PER_CONFIG_COLUMNS
    assert len(per_config) == 4 * len(combos)
    assert per_config["spearman_z"].dtype == np.float64 and per_config["pearson_z"].dtype == np.float64
    assert (per_config["patient_id"] == uk).all()
    for perm_idx in (5, 8):
        z, _ = evaluate_permutation(uk, perm_idx, qq, ii, combos, 42)
        rows = per_config[per_config["permutation_idx"] == perm_idx].sort_values("config_idx")
        assert list(rows["config_idx"]) == list(range(len(combos)))
        assert _nan_equal(rows["spearman_z"], z["spearman"])
        assert _nan_equal(rows["pearson_z"], z["pearson"])


def test_null_from_per_config_reproduces_batch_summaries(batch_result):
    _, _, _, _, records, per_config, _ = batch_result
    null = null_from_per_config(per_config, correlation_type="spearman").set_index("permutation_idx")
    for r in records:
        row = null.loc[r["permutation_idx"]]
        assert row["n_valid_configs"] == r["n_valid_configs"]
        for ours, theirs in [("null_mean_z", "mean_fisher_z"), ("null_median_z", "median_fisher_z"),
                             ("null_min_z", "min_fisher_z"), ("null_max_z", "max_fisher_z"),
                             ("null_q25_z", "q25_fisher_z"), ("null_q75_z", "q75_fisher_z"), ("null_std_z", "std_fisher_z")]:
            assert abs(row[ours] - r[theirs]) < 1e-12, ours
    null_p = null_from_per_config(per_config, correlation_type="pearson").set_index("permutation_idx")
    for r in records:
        assert abs(null_p.loc[r["permutation_idx"], "null_mean_z"] - r["pearson"]["mean_fisher_z"]) < 1e-12


def test_null_from_per_config_subset_and_all_invalid_rows():
    per_config = pd.DataFrame({
        "patient_id": [1] * 6, "permutation_idx": [0, 0, 0, 1, 1, 1], "config_idx": [0, 1, 2] * 2,
        "spearman_z": [0.1, 0.3, np.nan, np.nan, np.nan, np.nan], "pearson_z": [0.2, 0.2, 0.2, 0.5, 0.5, 0.5],
    })
    out = null_from_per_config(per_config).set_index("permutation_idx")
    assert out.loc[0, "n_valid_configs"] == 2 and abs(out.loc[0, "null_mean_z"] - 0.2) < 1e-12
    assert out.loc[1, "n_valid_configs"] == 0 and np.isnan(out.loc[1, "null_mean_z"])
    sub = null_from_per_config(per_config, config_indices=[1]).set_index("permutation_idx")
    assert sub.loc[0, "null_mean_z"] == 0.3
    with pytest.raises(ValueError):
        null_from_per_config(per_config, correlation_type="kendall")


def test_run_batch_empty_range():
    q, inh = make_dataset(0)
    records, per_config, datasets = run_batch(100, q[q.user_key == 100], inh[inh.user_key == 100], generate_param_combinations(), 3, 3, 42)
    assert records == [] and len(per_config) == 0 and list(per_config.columns) == PER_CONFIG_COLUMNS
