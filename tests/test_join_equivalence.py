"""Equivalence of aamos_concordance against the three frozen pre-refactor copies.

Every test runs the shared implementation and a verbatim oracle on the same
synthetic input and asserts identical output. NaN is treated as equal to NaN
(pandas' ``assert_frame_equal`` semantics) because several code paths
legitimately produce NaN and the point is that they produce it in the same
places.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from aamos_concordance import (
    CategorizationMethod,
    categorize_inhaler_usage,
    filter_zero_usage,
    generate_param_combinations,
    join_multi_patient,
    join_questionnaire_with_inhaler,
    run_single_permutation,
    summarize_correlations,
)
from tests.oracles import legacy_implementations as legacy
from tests.synthetic import ALL_192, WINDOW_CONFIGS, make_dataset

SEEDS = list(range(8))


def _per_patient_oracle():
    return legacy.PerPatientOracle()


def _data_loader_oracle(config, inhaler, questionnaire):
    return legacy.DataLoaderOracle(config, inhaler, questionnaire)


# --------------------------------------------------------------------------
# Window join
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("window,daily_max,calendar", WINDOW_CONFIGS)
def test_single_patient_join_matches_per_patient_and_jobworker(seed, window, daily_max, calendar):
    questionnaire, inhaler = make_dataset(seed)
    for user_key in questionnaire["user_key"].unique():
        q = questionnaire[questionnaire["user_key"] == user_key]
        inh = inhaler[inhaler["user_key"] == user_key]

        ours = join_questionnaire_with_inhaler(q, inh, window, daily_max, calendar)
        # Oracles mutate their inputs, so hand each a fresh copy.
        pp = _per_patient_oracle()._join_questionnaire_with_inhaler(q.copy(), inh.copy(), window, daily_max, calendar)
        jw = legacy.jobworker_join_questionnaire_with_inhaler(q.copy(), inh.copy(), user_key, window, daily_max, calendar)

        assert_frame_equal(ours, pp)
        assert_frame_equal(ours, jw)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("window,daily_max,calendar", WINDOW_CONFIGS)
def test_multi_patient_join_matches_data_loader(seed, window, daily_max, calendar):
    questionnaire, inhaler = make_dataset(seed)
    cfg = legacy._Cfg(timestamp_window_hours=window, use_daily_max_windows=daily_max, use_calendar_days=calendar)
    oracle = _data_loader_oracle(cfg, inhaler.copy(), questionnaire.copy())

    expected = oracle._join_questionnaire_with_inhaler_data()
    ours = join_multi_patient(questionnaire, inhaler, window, daily_max, calendar)
    assert_frame_equal(ours, expected)


def test_join_does_not_mutate_inputs():
    questionnaire, inhaler = make_dataset(0)
    q_before, i_before = questionnaire.copy(), inhaler.copy()
    join_multi_patient(questionnaire, inhaler, 24, False, False)
    for user_key in questionnaire["user_key"].unique():
        q = questionnaire[questionnaire["user_key"] == user_key]
        inh = inhaler[inhaler["user_key"] == user_key]
        join_questionnaire_with_inhaler(q, inh, 12, True, False)
    assert_frame_equal(questionnaire, q_before)
    assert_frame_equal(inhaler, i_before)


def test_empty_inhaler_shortcut_reproduces_both_historical_shapes():
    questionnaire, _ = make_dataset(0)
    q = questionnaire[questionnaire["user_key"] == 100]
    empty = pd.DataFrame(columns=["user_key", "date", "time", "name"])

    with_shortcut = join_questionnaire_with_inhaler(q, empty, 24, False, False)
    assert "timestamp" not in with_shortcut.columns
    assert (with_shortcut["inhaler_usage"] == 0).all()
    assert_frame_equal(with_shortcut, _per_patient_oracle()._join_questionnaire_with_inhaler(q.copy(), empty.copy(), 24, False, False))

    without = join_questionnaire_with_inhaler(q, empty, 24, False, False, empty_inhaler_shortcut=False)
    assert "timestamp" in without.columns
    assert (without["inhaler_usage"] == 0).all()


# --------------------------------------------------------------------------
# Categorisation
# --------------------------------------------------------------------------

def _joined_frames(seed):
    """A spread of joined frames, single- and multi-patient, across window configs."""
    questionnaire, inhaler = make_dataset(seed)
    frames = []
    for window, daily_max, calendar in WINDOW_CONFIGS[::3]:
        frames.append(join_multi_patient(questionnaire, inhaler, window, daily_max, calendar))
        for user_key in questionnaire["user_key"].unique():
            q = questionnaire[questionnaire["user_key"] == user_key]
            inh = inhaler[inhaler["user_key"] == user_key]
            frames.append(join_questionnaire_with_inhaler(q, inh, window, daily_max, calendar))
    return frames


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("method", list(CategorizationMethod))
def test_categorize_matches_data_loader_and_jobworker(seed, method):
    for df in _joined_frames(seed):
        ours = categorize_inhaler_usage(df, method)  # fallback=None reproduces these two callers
        cfg = legacy._Cfg(categorization_method=legacy.CategorizationMethod(method.value))
        dl = _data_loader_oracle(cfg, None, None)._categorize_inhaler_usage(df)
        assert_frame_equal(ours, dl)
        jw = legacy.jobworker_categorize_inhaler_usage(df, method.value, False)
        assert_frame_equal(ours, jw)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("method", list(CategorizationMethod))
def test_categorize_matches_per_patient_with_fallback_12(seed, method):
    for df in _joined_frames(seed):
        ours = categorize_inhaler_usage(df, method, top_category_fallback=12)
        pp = _per_patient_oracle()._categorize_inhaler_usage(df, legacy.CategorizationMethod(method.value))
        assert_frame_equal(ours, pp)


def test_top_category_fallback_documents_the_historical_disagreement():
    """When no usage >= 12 exists, data_loader/job-worker gave NaN and per-patient gave 12."""
    df = pd.DataFrame({"inhaler_usage": [0, 3, 5, 15], "daily_relief_inhaler": [13.0, 0, 1, 2]})
    df_low = df.assign(inhaler_usage=[0, 3, 5, 6])
    assert math.isnan(categorize_inhaler_usage(df_low, "midpoint_with_inhaler")["daily_relief_inhaler"].iloc[0])
    assert categorize_inhaler_usage(df_low, "midpoint_with_inhaler", top_category_fallback=12)["daily_relief_inhaler"].iloc[0] == 12
    # With data at or above 12 both agree.
    a = categorize_inhaler_usage(df, "midpoint_with_inhaler")
    b = categorize_inhaler_usage(df, "midpoint_with_inhaler", top_category_fallback=12)
    assert_frame_equal(a, b)


@pytest.mark.parametrize("method", list(CategorizationMethod))
def test_categorize_accepts_enum_and_string(method):
    df = pd.DataFrame({"inhaler_usage": [0, 1, 4, 13, 30], "daily_relief_inhaler": [0.0, 2, 8, 12, 15]})
    assert_frame_equal(categorize_inhaler_usage(df, method), categorize_inhaler_usage(df, method.value))


def test_categorize_rejects_unknown_method():
    df = pd.DataFrame({"inhaler_usage": [0], "daily_relief_inhaler": [0.0]})
    with pytest.raises(ValueError):
        categorize_inhaler_usage(df, "no_such_method")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("method", list(CategorizationMethod))
def test_zero_filter_matches_jobworker(seed, method):
    for df in _joined_frames(seed):
        ours = filter_zero_usage(categorize_inhaler_usage(df, method))
        jw = legacy.jobworker_categorize_inhaler_usage(df, method.value, True)
        if jw is None:
            assert len(ours) == 0
        else:
            assert_frame_equal(ours, jw)


# --------------------------------------------------------------------------
# Configuration enumeration
# --------------------------------------------------------------------------

def test_param_combinations_match_jobworker_exactly():
    assert generate_param_combinations() == legacy.jobworker_generate_param_combinations()


def test_param_combinations_match_per_patient_as_a_set():
    cfg = legacy._Cfg(
        timestamp_windows=(12, 24, 36, 48), use_daily_max_windows=(False, True), use_calendar_days=(False, True),
        filter_out_zero_usage_entries=(False, True), categorization_methods=tuple(legacy.CategorizationMethod),
    )
    pp = {(w, dm, cal, fz, cm.value) for (w, dm, cal, fz, cm) in legacy.perpatient_generate_param_combinations(cfg)}
    ours = {(c["timestamp_window"], c["use_daily_max_windows"], c["use_calendar_days"], c["filter_out_zero_usage"], c["categorization_method"])
            for c in generate_param_combinations()}
    assert ours == pp
    assert len(ours) == 132


def test_param_combinations_are_the_192_raw_combinations_deduplicated():
    from aamos_concordance.configs import effective_key
    raw_keys = {effective_key(*c) for c in ALL_192}
    ours_keys = [effective_key(c["timestamp_window"], c["use_daily_max_windows"], c["use_calendar_days"],
                               c["filter_out_zero_usage"], c["categorization_method"]) for c in generate_param_combinations()]
    assert len(ours_keys) == len(set(ours_keys)) == len(raw_keys) == 132


# --------------------------------------------------------------------------
# Permutation kernel and summary
# --------------------------------------------------------------------------

def _nan_equal(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("seed", SEEDS[:4])
@pytest.mark.parametrize("perm_idx", [0, 1, 7])
def test_run_single_permutation_matches_jobworker(seed, perm_idx):
    questionnaire, inhaler = make_dataset(seed)
    combos = generate_param_combinations()
    for user_key in questionnaire["user_key"].unique():
        q = questionnaire[questionnaire["user_key"] == user_key]
        inh = inhaler[inhaler["user_key"] == user_key]
        ours, _ = run_single_permutation(user_key, perm_idx, q, inh, combos, 42, "spearman")
        theirs, _ = legacy.jobworker_run_single_permutation(user_key, perm_idx, q.copy(), inh.copy(), combos, 42, "spearman")
        assert _nan_equal(ours, theirs), f"patient {user_key} perm {perm_idx}"


def test_run_single_permutation_pearson_matches_jobworker():
    questionnaire, inhaler = make_dataset(3)
    combos = generate_param_combinations()
    user_key = 104
    q = questionnaire[questionnaire["user_key"] == user_key]
    inh = inhaler[inhaler["user_key"] == user_key]
    ours, _ = run_single_permutation(user_key, 2, q, inh, combos, 42, "pearson")
    theirs, _ = legacy.jobworker_run_single_permutation(user_key, 2, q.copy(), inh.copy(), combos, 42, "pearson")
    assert _nan_equal(ours, theirs)


def test_run_single_permutation_is_deterministic_and_does_not_mutate():
    questionnaire, inhaler = make_dataset(1)
    combos = generate_param_combinations()
    q = questionnaire[questionnaire["user_key"] == 104]
    inh = inhaler[inhaler["user_key"] == 104]
    q_before = q.copy()
    a, _ = run_single_permutation(104, 5, q, inh, combos, 42)
    b, _ = run_single_permutation(104, 5, q, inh, combos, 42)
    c, _ = run_single_permutation(104, 6, q, inh, combos, 42)
    assert _nan_equal(a, b)
    assert not _nan_equal(a, c)
    assert_frame_equal(q, q_before)


def test_collect_datasets_labels_rows():
    questionnaire, inhaler = make_dataset(1)
    combos = generate_param_combinations()[:3]
    q = questionnaire[questionnaire["user_key"] == 104]
    inh = inhaler[inhaler["user_key"] == 104]
    _, datasets = run_single_permutation(104, 0, q, inh, combos, 42, collect_datasets=True)
    assert datasets, "expected at least one valid configuration"
    for d in datasets:
        assert set(["permutation_idx", "config_idx", "patient_id", "categorization_method"]).issubset(d.columns)
        assert (d["patient_id"] == 104).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_summarize_matches_jobworker_bookkeeping(seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=132)
    z[rng.random(132) < 0.1] = np.nan
    if seed == 0:
        z[:] = np.nan  # all-invalid permutation
    if seed == 1:
        z[5] = 40.0    # guaranteed outlier
    ours = summarize_correlations(list(z))
    theirs = legacy.jobworker_summarize(list(z))
    theirs.pop("permutation_idx")
    assert ours.keys() == theirs.keys()
    for k in ours:
        assert ours[k] == theirs[k], k
