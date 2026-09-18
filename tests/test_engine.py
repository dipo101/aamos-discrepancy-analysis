"""Vectorised engine: equivalence to the kernel, to the observed table, and to the v1 null."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import generate_param_combinations, null_from_per_config, run_batch
from aamos_concordance.engine import (
    PERFECT_CORRELATION_TOLERANCE,
    distinct_permutations,
    n_distinct_permutations,
    observed_per_config,
    precompute_patient,
    run_patient,
    run_patient_exact,
    run_patient_sampled,
    sampled_permutations,
)
from tests.synthetic import make_dataset

LEAK = 5.0  # |Z| above this is a near-perfect correlation the kernel let through (r = 1 - ulp)


# --------------------------------------------------------------------------
# Permutation matrices
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,seed", [(7, 42), (25, 42), (173, 1000)])
def test_sampled_permutations_reproduce_pandas_sample(n, seed):
    s = pd.Series(np.arange(n) * 1.5)
    P = sampled_permutations(n, seed, [0, 1, 7, 9999])
    for row, k in zip(P, [0, 1, 7, 9999]):
        expected = s.sample(frac=1, random_state=seed + k).values
        assert np.array_equal(s.values[row], expected)


def test_distinct_permutations_enumerates_the_multiset():
    v = np.array([0, 0, 1, 2, 2, 2.0])
    P = distinct_permutations(v)
    assert n_distinct_permutations(v) == 60 and P.shape == (60, 6)
    assert np.array_equal(P[0], np.arange(6))
    arrangements = {tuple(v[r]) for r in P}
    assert len(arrangements) == 60
    for r in P:
        assert sorted(r) == list(range(6))
    with pytest.raises(ValueError, match="exceeds the limit"):
        distinct_permutations(np.arange(12.0), limit=1000)


def test_distinct_permutations_handles_nan_as_its_own_value():
    v = np.array([0, np.nan, 1.0])
    P = distinct_permutations(v)
    assert P.shape == (6, 3) and np.array_equal(P[0], np.arange(3))


# --------------------------------------------------------------------------
# Engine vs kernel on synthetic data
# --------------------------------------------------------------------------

def _compare(ref: pd.DataFrame, eng: pd.DataFrame):
    m = ref.merge(eng, on=["patient_id", "permutation_idx", "config_idx"], suffixes=("_k", "_e"))
    assert len(m) == len(ref) == len(eng)
    for t in ("spearman_z", "pearson_z"):
        a, b = m[f"{t}_k"].to_numpy(), m[f"{t}_e"].to_numpy()
        leak = np.abs(a) > LEAK  # kernel let a perfect correlation through; engine records NaN by design
        assert (np.isnan(b)[leak]).all(), f"{t}: engine must treat |r|~1 as invalid"
        same_nan = np.isnan(a) == np.isnan(b)
        assert same_nan[~leak].all(), f"{t}: NaN pattern differs off the perfect-correlation boundary"
        both = ~np.isnan(a) & ~np.isnan(b)
        if both.any():
            assert np.abs(a[both] - b[both]).max() < 1e-12, t
    return m


@pytest.mark.parametrize("seed", range(6))
def test_engine_matches_kernel_on_synthetic(seed):
    combos = generate_param_combinations()
    q, inh = make_dataset(seed)
    for pid in q["user_key"].unique():
        qq, ii = q[q.user_key == pid], inh[inh.user_key == pid]
        _, ref, _ = run_batch(pid, qq, ii, combos, 0, 6, 42)
        eng = run_patient_sampled(pid, qq, ii, 0, 6, 42)
        _compare(ref, eng)


def test_engine_chunking_and_config_subset_are_transparent():
    q, inh = make_dataset(2)
    qq, ii = q[q.user_key == 104], inh[inh.user_key == 104]
    a = run_patient_sampled(104, qq, ii, 0, 9, 42, chunk=4)
    b = run_patient_sampled(104, qq, ii, 0, 9, 42, chunk=100)
    pd.testing.assert_frame_equal(a, b)
    tables = precompute_patient(104, qq, ii)
    P = sampled_permutations(tables.n_rows, 42, [3, 4])
    sub = run_patient(tables, P, [3, 4], config_indices=[0, 5, 131])
    assert sorted(sub["config_idx"].unique()) == [0, 5, 131] and sorted(sub["permutation_idx"].unique()) == [3, 4]
    full = b[b.permutation_idx.isin([3, 4]) & b.config_idx.isin([0, 5, 131])].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub.sort_values(["permutation_idx", "config_idx"]).reset_index(drop=True), full)


def test_exact_row_zero_is_the_observed_evaluation():
    q, inh = make_dataset(3)
    pid = 105
    qq, ii = q[q.user_key == pid], inh[inh.user_key == pid]
    small = qq.head(6)
    exact = run_patient_exact(pid, small, ii)
    obs = observed_per_config(pid, small, ii)
    row0 = exact[exact.permutation_idx == 0].sort_values("config_idx").reset_index(drop=True)
    o = obs.sort_values("config_idx").reset_index(drop=True)
    for t in ("spearman_z", "pearson_z"):
        assert np.array_equal(row0[t].to_numpy(), o[t].to_numpy(), equal_nan=True)
    assert exact.permutation_idx.nunique() == n_distinct_permutations(small["daily_relief_inhaler"].to_numpy(float))


# --------------------------------------------------------------------------
# Real data: observed table and v1 null
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_frames(data_dir):
    return (pd.read_csv(data_dir / "anonym_aamos00_dailyquestionnaire_dt.csv"),
            pd.read_csv(data_dir / "anonym_aamos00_smartinhaler_dt.csv"))


def test_engine_observed_matches_v1_per_config_table(raw_frames, frozen_results_dir):
    from aamos_concordance.configs import attach_config_idx
    q, inh = raw_frames
    v1 = attach_config_idx(pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv"))
    n_checked = 0
    for pid, g in v1.groupby("user_key"):
        qq, ii = q[q.user_key == pid].drop_duplicates(), inh[inh.user_key == pid].drop_duplicates()
        obs = observed_per_config(pid, qq, ii).set_index("config_idx")
        g = g.set_index("config_idx")
        for t in ("spearman_z", "pearson_z"):
            a, b = g[t].reindex(obs.index).to_numpy(float), obs[t].to_numpy()
            # v1 kept the row (below the min_samples cut it is absent) ; +-inf in v1 is a perfect correlation
            present = ~np.isnan(a) | np.isinf(a)
            inf = np.isinf(a)
            assert np.isnan(b[inf]).all()
            fin = present & ~inf
            if fin.any():
                assert np.abs(a[fin] - b[fin]).max() < 1e-9, (pid, t)
            n_checked += int(fin.sum())
    assert n_checked > 2000


@pytest.mark.parametrize("patient_id", [917, 454, 398])
def test_engine_full_null_matches_v1_on_unaffected_permutations(raw_frames, frozen_results_dir, patient_id):
    """10,000 permutations, compared with the frozen null wherever v1 has no leaked |Z|>5."""
    q, inh = raw_frames
    qq, ii = q[q.user_key == patient_id], inh[inh.user_key == patient_id]
    pc = run_patient_sampled(patient_id, qq, ii, 0, 10000, 42)
    null = null_from_per_config(pc).set_index("permutation_idx")
    ref = pd.read_parquet(frozen_results_dir / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet")
    ref = ref[ref.patient_id == patient_id].set_index("permutation_idx")
    null = null.reindex(ref.index)
    clean = ((ref.null_max_z.abs() < LEAK) & (ref.null_min_z.abs() < LEAK)).to_numpy()
    assert clean.sum() > 5000
    for col in ("null_mean_z", "null_median_z", "null_min_z", "null_max_z", "null_q25_z", "null_q75_z", "null_std_z"):
        assert np.abs(null[col].to_numpy() - ref[col].to_numpy())[clean].max() < 1e-9, col
    nv_e, nv_r = null.n_valid_configs.to_numpy(), ref.n_valid_configs.to_numpy()
    assert (nv_e[clean] == nv_r[clean]).all()
    # where v1 leaked, the engine has strictly fewer valid configs (the leaked ones are NaN)
    assert (nv_e[~clean] < nv_r[~clean]).all()


def test_v1_null_contains_leaked_perfect_correlations(frozen_results_dir):
    """Documents the v1 defect the engine corrects: r = 1 - ulp passed the kernel's -1 < r < 1 check."""
    ref = pd.read_parquet(frozen_results_dir / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet")
    leaked = ref[(ref.null_max_z.abs() > LEAK) | (ref.null_min_z.abs() > LEAK)]
    assert len(leaked) == 7850
    assert leaked.groupby("patient_id").size().to_dict() == {328: 2156, 398: 4734, 454: 956, 917: 4}
    assert ref.null_max_z.max() > 18 and ref.null_min_z.min() < -18


def test_exact_enumeration_on_patient_454(raw_frames):
    q, inh = raw_frames
    qq, ii = q[q.user_key == 454], inh[inh.user_key == 454]
    assert n_distinct_permutations(qq["daily_relief_inhaler"].to_numpy(float)) == 840
    exact = run_patient_exact(454, qq, ii)
    assert exact.permutation_idx.nunique() == 840
    obs = observed_per_config(454, qq, ii).set_index("config_idx")["spearman_z"]
    row0 = exact[exact.permutation_idx == 0].set_index("config_idx")["spearman_z"]
    assert np.array_equal(row0.reindex(obs.index).to_numpy(), obs.to_numpy(), equal_nan=True)
