"""Resampling risk and sequential stopping (item 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aamos_concordance.resampling import DEFAULT_EPSILON, resampling_risk, sequential_stopping
from aamos_concordance.summary import permutation_p_values


def test_resampling_risk_matches_beta_posterior():
    # 3 exceedances in 1000 draws, alpha = 0.01: p_hat = 0.004 < alpha, risk = P(p > alpha)
    r = resampling_risk(3, 1000, 0.01)
    assert r == pytest.approx(1 - stats.beta.cdf(0.01, 3.5, 997.5))
    # 30 exceedances in 1000: p_hat = 0.031 > alpha, risk = P(p < alpha)
    r2 = resampling_risk(30, 1000, 0.01)
    assert r2 == pytest.approx(stats.beta.cdf(0.01, 30.5, 970.5))
    assert 0 <= r < 0.5 and 0 <= r2 < 0.5


def test_resampling_risk_shrinks_with_sample_size_and_distance():
    assert resampling_risk(0, 10000, 0.0033) < resampling_risk(0, 500, 0.0033)
    assert resampling_risk(10, 1000, 0.0033) > resampling_risk(100, 1000, 0.0033)  # further from alpha
    assert np.isnan(resampling_risk(0, 0, 0.05))


def test_resampling_risk_is_large_when_p_hat_sits_on_alpha():
    n = 1000
    s = int(round(0.05 * (n + 1))) - 1
    assert resampling_risk(s, n, 0.05) > 0.3


def test_sequential_stops_early_on_a_clear_decision():
    exceed = np.zeros(10000, dtype=bool)                 # p_hat -> 1/(n+1): clearly significant
    r = sequential_stopping(exceed, alpha=0.05 / 15)
    assert r.stopped and r.decision_at_stop is True and r.matches_full_sample is True
    assert 500 <= r.n_at_stop < 10000
    exceed = np.ones(10000, dtype=bool)                  # p_hat = 1: clearly not
    r = sequential_stopping(exceed, alpha=0.05 / 15)
    assert r.stopped and r.decision_at_stop is False and r.n_at_stop == 500


def test_sequential_does_not_stop_when_p_sits_on_alpha():
    rng = np.random.default_rng(0)
    alpha = 0.05
    exceed = rng.random(10000) < alpha
    r = sequential_stopping(exceed, alpha=alpha)
    assert not r.stopped and r.n_at_stop == 10000 and r.decision_at_stop is None


def test_sequential_spending_bounds_the_risk():
    """Over many null sequences with true p at the boundary, wrong stops are rare (<= epsilon-ish)."""
    rng = np.random.default_rng(1)
    alpha, true_p = 0.01, 0.006  # truly significant, but close
    wrong = 0
    trials = 400
    for _ in range(trials):
        exceed = rng.random(10000) < true_p
        r = sequential_stopping(exceed, alpha=alpha, epsilon=DEFAULT_EPSILON)
        if r.stopped and r.decision_at_stop is False:
            wrong += 1
    assert wrong / trials <= 0.01


def test_sequential_empty_and_short_sequences():
    r = sequential_stopping([], alpha=0.05)
    assert not r.stopped and r.n_at_stop == 0
    r = sequential_stopping(np.zeros(100, dtype=bool), alpha=0.05)  # shorter than min_n
    assert not r.stopped and r.n_at_stop == 100


# --------------------------------------------------------------------------
# Integration with permutation_p_values
# --------------------------------------------------------------------------

def _null(pid, values):
    return pd.DataFrame({"patient_id": pid, "permutation_idx": range(len(values)), "null_mean_z": values, "n_valid_configs": 60})


def test_p_values_report_risk_and_stopping_for_sampled_patients_only():
    rng = np.random.default_rng(2)
    null = pd.concat([_null(1, rng.normal(0, 0.1, 10000)), _null(2, rng.normal(0, 0.1, 840))], ignore_index=True)
    observed = pd.Series({1: 0.9, 2: 0.9})
    out = permutation_p_values(observed, null, null_column="null_mean_z", exact_patients=[2]).set_index("patient_id")
    assert out.loc[2, "null_is_exact"] and out.loc[2, "resampling_risk"] == 0.0 and out.loc[2, "n_perm_sequential"] == 840
    assert not out.loc[2, "sequential_stopped"]
    assert not out.loc[1, "null_is_exact"] and 0 <= out.loc[1, "resampling_risk"] < 1e-6
    assert out.loc[1, "sequential_stopped"] and out.loc[1, "n_perm_sequential"] < 10000 and out.loc[1, "sequential_matches_full"]


def test_p_values_sort_null_by_permutation_index_before_the_sequential_path():
    rng = np.random.default_rng(3)
    vals = rng.normal(0, 0.1, 10000)
    null = _null(1, vals).sample(frac=1, random_state=0)  # shuffled row order
    out = permutation_p_values(pd.Series({1: 0.9}), null, null_column="null_mean_z").set_index("patient_id")
    ref = permutation_p_values(pd.Series({1: 0.9}), _null(1, vals), null_column="null_mean_z").set_index("patient_id")
    assert out.loc[1, "n_perm_sequential"] == ref.loc[1, "n_perm_sequential"]
    assert out.loc[1, "p_value"] == ref.loc[1, "p_value"]


# --------------------------------------------------------------------------
# Committed worlds
# --------------------------------------------------------------------------

def test_committed_baseline_carries_risk_columns():
    from aamos_concordance.worlds import BASELINE, world_dir
    from aamos_concordance.summary import PRIMARY, select
    s = select(pd.read_csv(world_dir(BASELINE) / "summary.csv"), PRIMARY).set_index("patient_id")
    assert set(s.index[s.null_is_exact]) == {328, 398, 454, 917}
    assert (s.loc[s.null_is_exact, "resampling_risk"] == 0).all()
    sampled = s[~s.null_is_exact]
    assert (sampled["resampling_risk"] < 1e-6).all()          # every sampled decision is far from alpha_bf
    assert sampled["sequential_stopped"].all() and sampled["sequential_matches_full"].all()
    assert (sampled["n_perm_sequential"] <= 2400).all()


def test_committed_worlds_risk_profile():
    """Across all 48 worlds and specs: every sequential stop agrees with the full sample; in every
    A and C world the primary spec's decisions carry negligible risk and all settle sequentially;
    the borderline decisions (risk > 1e-3) are B worlds or non-primary specs."""
    from aamos_concordance.sensitivity import load_world_summaries
    from aamos_concordance.summary import PRIMARY, select
    n_risky = 0
    for wid, s in load_world_summaries().items():
        sampled = s[~s["null_is_exact"].astype(bool)]
        stopped = sampled[sampled["sequential_stopped"].astype(bool)]
        assert stopped["sequential_matches_full"].astype(bool).all(), wid
        risky = sampled[sampled["resampling_risk"] > 1e-3]
        n_risky += len(risky)
        if "case=B" not in wid:
            prim = select(sampled, PRIMARY)
            assert (prim["resampling_risk"] < 1e-6).all(), wid
            assert prim["sequential_stopped"].astype(bool).all(), wid
            assert select(risky, PRIMARY).empty, wid
    assert n_risky > 0  # the column is doing work: some non-primary / case-B decisions are borderline
