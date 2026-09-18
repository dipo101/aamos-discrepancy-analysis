"""Device-side absence cases B and C (item 9) and the zero-filter coupling (item 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance.configs import effective_config_indices, generate_param_combinations
from aamos_concordance.engine import (
    apply_absence_case,
    device_rate_per_hour,
    imputation_seed,
    observed_per_config,
    precompute_patient,
    run_patient_sampled,
    window_hours,
)
from aamos_concordance.join import join_questionnaire_with_inhaler
from aamos_concordance.summary import SummarySpec
from aamos_concordance.worlds import WorldSpec
from tests.synthetic import make_dataset

COMBOS = generate_param_combinations()
WINDOW_KEYS = sorted({(c["timestamp_window"], c["use_daily_max_windows"], c["use_calendar_days"]) for c in COMBOS})


def _patient(seed=3, pid=104):
    q, inh = make_dataset(seed)
    return q[q.user_key == pid], inh[inh.user_key == pid]


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def test_window_hours():
    assert window_hours((12, False, False)) == 12 and window_hours((48, False, False)) == 48
    assert window_hours((12, True, False)) == 24 and window_hours((48, False, True)) == 24


def test_device_rate_per_hour():
    inh = pd.DataFrame({"user_key": 1, "date": [2, 2, 5], "time": "08:00:00", "name": "V"})
    assert device_rate_per_hour(inh) == pytest.approx(3 / (24 * 4))
    assert device_rate_per_hour(inh.iloc[0:0]) == 0.0


def test_imputation_seed_is_deterministic_and_distinct():
    assert imputation_seed(294, 3, 5) == imputation_seed(294, 3, 5)
    assert len({imputation_seed(294, k, 5) for k in range(10)}) == 10
    assert len({imputation_seed(294, 3, w) for w in range(11)}) == 11
    assert imputation_seed(294, 3, 5) != imputation_seed(473, 3, 5)


def test_apply_absence_case_A_is_identity():
    q, inh = _patient()
    merged = join_questionnaire_with_inhaler(q, inh, 24, False, False)
    out, keep = apply_absence_case(merged, WorldSpec("union", "A"), patient_id=104, window_key=(24, False, False), window_index=0, rate_per_hour=1.0)
    assert out is merged and keep.all()


def test_apply_absence_case_C_masks_exactly_the_empty_windows():
    q, inh = _patient()
    merged = join_questionnaire_with_inhaler(q, inh, 24, False, False)
    out, keep = apply_absence_case(merged, WorldSpec("union", "C"), patient_id=104, window_key=(24, False, False), window_index=0, rate_per_hour=1.0)
    assert out is merged
    assert np.array_equal(keep, (merged["inhaler_usage"] > 0).to_numpy())


def test_apply_absence_case_B_imputes_only_empty_windows_reproducibly():
    q, inh = _patient()
    merged = join_questionnaire_with_inhaler(q, inh, 12, False, False)
    empty = (merged["inhaler_usage"] == 0).to_numpy()
    assert empty.any()
    w3 = WorldSpec("union", "B", 3)
    out1, keep = apply_absence_case(merged, w3, patient_id=104, window_key=(12, False, False), window_index=0, rate_per_hour=0.5)
    out2, _ = apply_absence_case(merged, w3, patient_id=104, window_key=(12, False, False), window_index=0, rate_per_hour=0.5)
    assert keep.all()
    assert merged["inhaler_usage"].to_numpy()[empty].sum() == 0  # input untouched
    a, b = out1["inhaler_usage"].to_numpy(), out2["inhaler_usage"].to_numpy()
    assert np.array_equal(a, b)
    assert np.array_equal(a[~empty], merged["inhaler_usage"].to_numpy()[~empty])
    assert (a[empty] >= 0).all() and a[empty].dtype.kind in "iu"
    out4, _ = apply_absence_case(merged, WorldSpec("union", "B", 4), patient_id=104, window_key=(12, False, False), window_index=0, rate_per_hour=0.5)
    assert not np.array_equal(a, out4["inhaler_usage"].to_numpy())


def test_case_B_draws_have_the_intended_mean():
    """Over many imputations the mean imputed count is rate * window_hours."""
    merged = pd.DataFrame({"inhaler_usage": np.zeros(200, dtype=int), "daily_relief_inhaler": 1.0})
    rate = 0.25
    means = []
    for k in range(40):
        out, _ = apply_absence_case(merged, WorldSpec("union", "B", k), patient_id=7, window_key=(48, False, False), window_index=2, rate_per_hour=rate)
        means.append(out["inhaler_usage"].mean())
    assert np.mean(means) == pytest.approx(rate * 48, rel=0.05)
    out, _ = apply_absence_case(merged, WorldSpec("union", "B", 0), patient_id=7, window_key=(12, False, True), window_index=2, rate_per_hour=rate)
    assert out["inhaler_usage"].mean() == pytest.approx(rate * 24, rel=0.15)


# --------------------------------------------------------------------------
# Item 10: the zero filter is inert under B and C
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case,k", [("B", 0), ("C", None)])
def test_filter_configs_duplicate_unfiltered_under_B_and_C(case, k):
    q, inh = _patient()
    w = WorldSpec("union", case, k)
    tables = precompute_patient(104, q, inh, COMBOS, world=w)
    for idx, cfg in enumerate(COMBOS):
        assert tables.per_config[idx][2] is False
    obs = observed_per_config(104, q, inh, world=w).set_index("config_idx")
    cfgs = pd.DataFrame(COMBOS)
    for _, grp in cfgs.groupby(["timestamp_window", "use_daily_max_windows", "use_calendar_days", "categorization_method"]):
        vals = obs.loc[grp.index, "spearman_z"].to_numpy()
        assert np.allclose(vals, vals[0], equal_nan=True)


def test_filter_is_active_under_A():
    q, inh = _patient()
    tables = precompute_patient(104, q, inh, COMBOS, world=WorldSpec("union", "A"))
    flags = {tables.per_config[i][2] for i in range(len(COMBOS))}
    assert flags == {True, False}


def test_effective_sets_collapse_the_filter_axis_under_B_and_C():
    assert len(effective_config_indices("spearman", "A")) == 60
    assert len(effective_config_indices("spearman", "B")) == 30 == len(effective_config_indices("spearman", "C"))
    assert len(effective_config_indices("pearson", "A")) == 132
    assert len(effective_config_indices("pearson", "B")) == 66
    assert len(SummarySpec("effective", "spearman", "mean").config_indices_in("C")) == 30
    assert SummarySpec("effective", "spearman", "mean").config_indices == effective_config_indices("spearman", "A")


# --------------------------------------------------------------------------
# End-to-end behaviour of the cases
# --------------------------------------------------------------------------

def test_case_C_equals_case_A_on_windows_with_no_empty_rows():
    """Where every window has device records, C changes nothing (and the unfiltered A configs match)."""
    q, inh = _patient(seed=1, pid=101)  # heavy device user in seed 1
    obs_a = observed_per_config(101, q, inh, world=WorldSpec("union", "A")).set_index("config_idx")
    obs_c = observed_per_config(101, q, inh, world=WorldSpec("union", "C")).set_index("config_idx")
    for idx, cfg in enumerate(COMBOS):
        merged = join_questionnaire_with_inhaler(q, inh, cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"])
        if (merged["inhaler_usage"] == 0).any() or cfg["filter_out_zero_usage"]:
            continue
        assert np.isclose(obs_a.loc[idx, "spearman_z"], obs_c.loc[idx, "spearman_z"], equal_nan=True)


def test_case_C_row_counts_and_null_shape():
    q, inh = _patient()
    tables = precompute_patient(104, q, inh, COMBOS, world=WorldSpec("union", "C"))
    for idx, cfg in enumerate(COMBOS):
        merged = join_questionnaire_with_inhaler(q, inh, cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"])
        assert tables.observed_row_count(idx) == int((merged["inhaler_usage"] > 0).sum())
    pc = run_patient_sampled(104, q, inh, 0, 5, 42, world=WorldSpec("union", "C"))
    assert len(pc) == 5 * len(COMBOS)


def test_case_B_null_differs_across_imputations_but_is_reproducible():
    q, inh = _patient()
    a = run_patient_sampled(104, q, inh, 0, 4, 42, world=WorldSpec("union", "B", 0))
    a2 = run_patient_sampled(104, q, inh, 0, 4, 42, world=WorldSpec("union", "B", 0))
    b = run_patient_sampled(104, q, inh, 0, 4, 42, world=WorldSpec("union", "B", 1))
    pd.testing.assert_frame_equal(a, a2)
    assert not np.allclose(a["spearman_z"], b["spearman_z"], equal_nan=True)
