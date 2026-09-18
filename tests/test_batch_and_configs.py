"""Items 6, 7 and 21: dual-correlation kernel, per-config null storage, effective configurations."""

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
    run_single_permutation,
    summarize_correlations,
)
from aamos_concordance.configs import (
    N_SPEARMAN_EFFECTIVE,
    SPEARMAN_METHOD_CLASS,
    config_equivalence_classes,
    effective_config_indices,
    spearman_equivalence_key,
)
from aamos_concordance.diagnostics import attach_config_idx, effective_configurations, equivalence_class_check
from tests.synthetic import make_dataset


def _nan_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


# --------------------------------------------------------------------------
# Item 6: kernel computes both correlation types
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_permutation_matches_one_type_interface_for_both_types(seed):
    q, inh = make_dataset(seed)
    combos = generate_param_combinations()
    for uk in q["user_key"].unique():
        qq, ii = q[q["user_key"] == uk], inh[inh["user_key"] == uk]
        both, _ = evaluate_permutation(uk, 3, qq, ii, combos, 42)
        assert set(both) == {"spearman", "pearson"}
        assert len(both["spearman"]) == len(both["pearson"]) == len(combos)
        for t in ("spearman", "pearson"):
            one, _ = run_single_permutation(uk, 3, qq, ii, combos, 42, t)
            assert _nan_equal(both[t], one), (uk, t)


def test_both_types_invalid_in_the_same_places_except_the_unit_circle_rule():
    q, inh = make_dataset(1)
    combos = generate_param_combinations()
    qq, ii = q[q["user_key"] == 104], inh[inh["user_key"] == 104]
    both, _ = evaluate_permutation(104, 0, qq, ii, combos, 42)
    s, p = np.asarray(both["spearman"]), np.asarray(both["pearson"])
    # Structural invalidity (too few rows, constant column) hits both; only |r|==1 can differ.
    assert (np.isnan(s) & ~np.isnan(p)).sum() + (~np.isnan(s) & np.isnan(p)).sum() <= 3


def test_run_single_permutation_rejects_unknown_type():
    q, inh = make_dataset(0)
    with pytest.raises(ValueError):
        run_single_permutation(100, 0, q, inh, generate_param_combinations()[:2], 42, "kendall")


# --------------------------------------------------------------------------
# Item 7: batch records and per-config storage
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Item 21: effective configurations
# --------------------------------------------------------------------------

def test_equivalence_classes_partition_the_132():
    classes = config_equivalence_classes("spearman")
    members = sorted(i for m in classes.values() for i in m)
    assert members == list(range(132))
    assert len(classes) == N_SPEARMAN_EFFECTIVE == 60
    assert len(effective_config_indices("spearman")) == 60
    assert effective_config_indices("pearson") == list(range(132))
    sizes = sorted(len(m) for m in classes.values())
    # 3 method classes of sizes 4/1/1 x (10 windows, one of which merges two) x 2 filters
    assert sum(sizes) == 132 and max(sizes) == 8


def test_spearman_key_merges_exactly_the_intended_axes():
    combos = generate_param_combinations()
    by_key = {}
    for c in combos:
        by_key.setdefault(spearman_equivalence_key(c), []).append(c)
    for key, cs in by_key.items():
        assert len({c["filter_out_zero_usage"] for c in cs}) == 1
        assert len({SPEARMAN_METHOD_CLASS[c["categorization_method"]] for c in cs}) == 1
        assert len({c["use_calendar_days"] for c in cs}) == 1


def test_equivalence_classes_are_exact_duplicates_on_the_v1_table(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    check = equivalence_class_check(per_config, "spearman")
    assert len(check) == 60
    assert check["max_within_class_spread"].max() < 1e-12


def test_attach_config_idx_is_a_bijection_on_the_v1_table(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    df = attach_config_idx(per_config)
    assert df.groupby("user_key")["config_idx"].nunique().max() == 132
    assert df["config_idx"].min() == 0 and df["config_idx"].max() == 131


def test_effective_configurations_pins_the_finding(frozen_results_dir):
    from tests.test_summary import V1_ASSESSED
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    t = effective_configurations(per_config, V1_ASSESSED).set_index("patient_id")
    assert list(t.index) == V1_ASSESSED
    assert (t["spearman_n_valid_effective"] <= 60).all()
    assert t["spearman_n_distinct_z"].median() == 25
    # The published concordant set under mean-of-132 ...
    assert sorted(t.index[t["spearman_mean_all_ge_threshold"]]) == [294, 473, 702, 917]
    # ... loses 294 and 473 when each structural class is counted once.
    assert sorted(t.index[t["spearman_mean_effective_ge_threshold"]]) == [702, 917]
    assert t.loc[294, "spearman_mean_effective"] == pytest.approx(0.459, abs=5e-4)
    assert t.loc[473, "spearman_mean_effective"] == pytest.approx(0.432, abs=5e-4)
    assert t.loc[473, "spearman_median_effective"] == pytest.approx(0.485, abs=5e-4)
    # Under Pearson nothing collapses.
    assert (t["pearson_n_valid_effective"] == t["pearson_n_valid_all"]).all()
