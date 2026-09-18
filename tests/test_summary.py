"""World summary: p-value formula, reproduction of the frozen v1 tables, derived sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance.summary import (
    V1_SPEC,
    SummarySpec,
    assessed_patients,
    build_world_summary,
    concordant_set,
    derive_concordant_sets,
    observed_statistics,
    permutation_p_values,
    to_v1_measure_table,
)

V1_MEAN = V1_SPEC
V1_MEDIAN = SummarySpec("all", "spearman", "median")

# The sets the manuscript reports (v1 rule: |null| >= |observed|). Recorded here
# as documentation; the code no longer implements that rule.
V1_PUBLISHED_MEAN = [294, 473, 702]
V1_PUBLISHED_MEDIAN = [294, 473, 702, 917]
# The sets under the centred rule applied to the v1 null (all/spearman). These are
# the oracle for the derived config.py groups on the committed baseline world.
V1_CONCORDANT_MEAN = [294, 473, 702, 917]
V1_CONCORDANT_MEDIAN = [190, 294, 473, 702, 917]
V1_ASSESSED = [113, 190, 294, 328, 343, 398, 447, 454, 473, 514, 625, 701, 702, 917, 939]


# --------------------------------------------------------------------------
# Formula checks on tiny hand-computable inputs
# --------------------------------------------------------------------------

def test_observed_statistics_excludes_non_finite():
    df = pd.DataFrame({"user_key": [1] * 5 + [2] * 3,
                       "spearman_z": [0.1, 0.3, np.inf, np.nan, 0.2, -np.inf, 0.5, 0.7]})
    out = observed_statistics(df).set_index("patient_id")
    assert out.loc[1, "n_valid"] == 3 and abs(out.loc[1, "mean"] - 0.2) < 1e-12 and out.loc[1, "median"] == 0.2
    assert out.loc[2, "n_valid"] == 2 and out.loc[2, "mean"] == 0.6


def test_permutation_p_value_is_two_tailed_about_the_null_centre():
    observed = pd.Series({1: 0.5, 2: -0.5, 3: np.nan, 4: 0.9})
    null = pd.DataFrame({
        "patient_id": [1] * 4 + [2] * 4 + [4] * 2,
        "null_mean_z": [0.1, 0.6, -0.7, 0.5, 0.0, 0.1, 0.2, 0.3, np.nan, np.nan],
        "n_valid_configs": [132] * 10,
    })
    out = permutation_p_values(observed, null, null_column="null_mean_z").set_index("patient_id")
    # patient 1: centre 0.125; distances 0.025, 0.475, 0.825, 0.375; observed distance 0.375
    #   -> 0.475, 0.825, 0.375 (tie) = 3 extreme; p = (3+1)/(4+1)
    assert out.loc[1, "p_value"] == pytest.approx(4 / 5)
    assert out.loc[1, "n_ties_at_observed"] == 1
    assert out.loc[1, "null_mean"] == pytest.approx(0.125)
    # patient 2: centre 0.15; distances 0.15, 0.05, 0.05, 0.15; observed distance 0.65 -> none; p = 1/5
    assert out.loc[2, "p_value"] == pytest.approx(1 / 5)
    assert out.loc[2, "observed_distance_sd"] < 0
    # patient 3 (NaN observed) and 4 (all-NaN null) are excluded, so Bonferroni uses n = 2
    assert set(out.index) == {1, 2}
    assert (out["n_patients_bonferroni"] == 2).all()
    assert out.loc[2, "p_bonferroni"] == pytest.approx(min(1.0, 2 / 5))
    assert out.loc[1, "p_bonferroni"] == 1.0


def test_centred_rule_is_invariant_to_a_null_shift():
    """A constant shift of null and observed together must not change p; the old |.| rule would."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.1, 500)
    obs = 0.35
    for shift in (0.0, -0.5, -1.2):
        null = pd.DataFrame({"patient_id": 1, "null_mean_z": base + shift, "n_valid_configs": 132})
        out = permutation_p_values(pd.Series({1: obs + shift}), null, null_column="null_mean_z")
        if shift == 0.0:
            p0 = out["p_value"].iloc[0]
        assert out["p_value"].iloc[0] == pytest.approx(p0)


def test_concordant_set_requires_threshold_and_bonferroni():
    summary = pd.DataFrame({
        "patient_id": [1, 2, 3, 4], "config_set": ["all"] * 4, "correlation_type": ["spearman"] * 4, "measure": ["mean"] * 4,
        "observed_z": [0.6, 0.6, 0.4, 0.9],
        "significant_bonferroni": [True, False, True, True],
    })
    assert concordant_set(summary, V1_MEAN) == [1, 4]
    assert concordant_set(summary, "all/spearman/mean", threshold=0.8) == [4]
    assert derive_concordant_sets(summary) == {"all/spearman/mean": [1, 4]}


# --------------------------------------------------------------------------
# Reproduction of the frozen v1 tables from the frozen v1 inputs
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def v1_summary(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    null = pd.read_parquet(frozen_results_dir / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet")
    with pytest.warns(UserWarning, match="No null distribution for summary specs"):
        return build_world_summary(per_config, null)


@pytest.mark.parametrize("measure", ["mean", "median"])
def test_summary_reproduces_v1_permutation_tables(v1_summary, frozen_results_dir, measure):
    ours = to_v1_measure_table(v1_summary, measure).set_index("patient_id").sort_index()
    frozen = pd.read_csv(frozen_results_dir / "permutation_by_measure" / measure / "permutation_test_results.csv")
    frozen = frozen.set_index("patient_id").sort_index()
    assert list(ours.columns) == list(frozen.columns)
    assert list(ours.index) == list(frozen.index)

    # Everything except the p-value columns and the flags derived from them is v1's exactly.
    p_cols = {"p_value", "p_bonferroni", "significant_uncorrected", "significant_bonferroni"}
    for col in ours.columns:
        if col in p_cols:
            continue
        a, b = ours[col], frozen[col]
        diff = (a.astype(float) - b.astype(float)).abs()
        assert diff.max() < 1e-9, (col, diff.max())
    # The p-values follow the centred rule, checked independently against the raw null.
    s = v1_summary[(v1_summary["measure"] == measure) & (v1_summary["config_set"] == "all")].set_index("patient_id").sort_index()
    null = pd.read_parquet(frozen_results_dir / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet")
    for pid, row in s.iterrows():
        nv = null[null.patient_id == pid][f"null_{measure}_z"].dropna().to_numpy()
        c = nv.mean()
        expected = (np.sum(np.abs(nv - c) >= abs(row["observed_z"] - c) - 1e-12) + 1) / (len(nv) + 1)  # ties count
        assert row["p_value"] == pytest.approx(expected, abs=1e-12), pid


def test_v1_summary_derives_the_published_sets(v1_summary):
    assert derive_concordant_sets(v1_summary) == {"all/spearman/mean": V1_CONCORDANT_MEAN, "all/spearman/median": V1_CONCORDANT_MEDIAN}
    assert assessed_patients(v1_summary) == V1_ASSESSED
    # Only the v1 specs can be summarised from the v1 null; nothing is fabricated for the others.
    assert set(v1_summary["config_set"]) == {"all"} and set(v1_summary["correlation_type"]) == {"spearman"}


def test_v1_summary_reports_ties_for_small_n_patients(v1_summary):
    ties = v1_summary[(v1_summary["measure"] == "mean") & (v1_summary["config_set"] == "all")].set_index("patient_id")["n_ties_at_observed"]
    # 454 has only 840 distinct permutations; some reproduce the observed arrangement exactly.
    assert ties.loc[454] > 0
    assert ties.loc[294] == 0


def test_published_v1_sets_are_recoverable_from_the_frozen_tables(frozen_results_dir):
    """Documents the published sets; they came from the old |null| >= |observed| rule."""
    for measure, published in (("mean", V1_PUBLISHED_MEAN), ("median", V1_PUBLISHED_MEDIAN)):
        f = pd.read_csv(frozen_results_dir / "permutation_by_measure" / measure / "permutation_test_results.csv")
        got = sorted(f[(f[f"observed_{measure}_z"] >= 0.5) & f["significant_bonferroni"]]["patient_id"])
        assert got == published
