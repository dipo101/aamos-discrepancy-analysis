"""Items 13-15: summary specs, per-spec summaries from a per-config null, threshold sweep.

The v1 null only supports all/spearman. To exercise the other specs end to
end, a small synthetic world is built here with ``run_batch`` so that a real
per-config null exists, and every spec is summarised from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import generate_param_combinations, run_batch
from aamos_concordance.configs import N_SPEARMAN_EFFECTIVE, attach_config_idx, config_indices_for
from aamos_concordance.summary import (
    ALL_SPECS,
    PRIMARY,
    V1_SPEC,
    SummarySpec,
    available_specs,
    build_world_summary,
    concordant_set,
    derive_concordant_sets,
    null_sources_for,
    observed_statistics,
    observed_table,
    select,
    threshold_sweep,
    to_v1_measure_table,
)
from tests.synthetic import make_dataset

# --------------------------------------------------------------------------
# SummarySpec
# --------------------------------------------------------------------------

def test_spec_keys_and_parse_round_trip():
    for sp in ALL_SPECS:
        assert SummarySpec.parse(sp.key) == sp
        assert sp.null_key == f"{sp.config_set}/{sp.correlation_type}"
    assert PRIMARY == SummarySpec("effective", "spearman", "mean")
    assert V1_SPEC == SummarySpec("all", "spearman", "mean")
    assert len(ALL_SPECS) == 8


@pytest.mark.parametrize("bad", [("nope", "spearman", "mean"), ("all", "kendall", "mean"), ("all", "spearman", "mode")])
def test_spec_rejects_unknown_values(bad):
    with pytest.raises(ValueError):
        SummarySpec(*bad)


def test_spec_config_indices():
    assert len(PRIMARY.config_indices) == N_SPEARMAN_EFFECTIVE
    assert V1_SPEC.config_indices == list(range(132))
    assert len(SummarySpec("effective", "pearson", "mean").config_indices) == 132
    with pytest.raises(ValueError):
        config_indices_for("nope", "spearman")


# --------------------------------------------------------------------------
# A synthetic world with a per-config null
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_world():
    """Observed per-config table + per-config null for 4 synthetic patients, 12 permutations each."""
    from aamos_concordance.categorization import categorize_inhaler_usage, filter_zero_usage
    from aamos_concordance.join import join_questionnaire_with_inhaler
    from scipy import stats

    q, inh = make_dataset(5)
    combos = generate_param_combinations()
    patients = [101, 102, 104, 105]

    obs_rows = []
    nulls = []
    for pid in patients:
        qq, ii = q[q.user_key == pid], inh[inh.user_key == pid]
        for cfg in combos:
            merged = join_questionnaire_with_inhaler(qq, ii, cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"])
            cat = categorize_inhaler_usage(merged, cfg["categorization_method"])
            if cfg["filter_out_zero_usage"]:
                cat = filter_zero_usage(cat)
            row = {"user_key": pid, "timestamp_window": cfg["timestamp_window"], "use_daily_max_windows": cfg["use_daily_max_windows"],
                   "use_calendar_days": cfg["use_calendar_days"], "filter_out_zero_usage_entries": cfg["filter_out_zero_usage"],
                   "categorization_method": cfg["categorization_method"], "sample_size": len(cat)}
            ok = len(cat) >= 3 and cat["inhaler_usage"].nunique() > 1 and cat["daily_relief_inhaler"].nunique() > 1
            for t, fn in (("spearman", stats.spearmanr), ("pearson", stats.pearsonr)):
                r = fn(cat["inhaler_usage"], cat["daily_relief_inhaler"])[0] if ok else np.nan
                row[f"{t}_corr"] = r
                row[f"{t}_z"] = np.arctanh(r) if np.isfinite(r) and -1 < r < 1 else np.nan
            obs_rows.append(row)
        _, per_config, _ = run_batch(pid, qq, ii, combos, 0, 12, 42, "spearman")
        nulls.append(per_config)
    return pd.DataFrame(obs_rows), pd.concat(nulls, ignore_index=True), patients


def test_null_sources_cover_every_spec_with_per_config_null(synthetic_world):
    observed, per_config_null, _ = synthetic_world
    sources = null_sources_for(per_config_null=per_config_null)
    assert set(sources) == {"all/spearman", "all/pearson", "effective/spearman", "effective/pearson"}
    # legacy-only gives just the v1 key
    assert set(null_sources_for(legacy_null=sources["all/spearman"])) == {"all/spearman"}


def test_summary_has_every_spec_and_primary_differs_from_v1_where_expected(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    assert {sp.key for sp in available_specs(summary)} == {sp.key for sp in ALL_SPECS}
    for sp in ALL_SPECS:
        s = select(summary, sp)
        assert sorted(s["patient_id"]) == patients
        assert (s["n_configs_in_set"] == len(sp.config_indices)).all()
        assert (s["n_valid_configs_observed"] <= len(sp.config_indices)).all()
    # effective/spearman observed mean is the mean over the 60 representatives only
    eff = select(summary, PRIMARY).set_index("patient_id")
    idx = attach_config_idx(observed)
    for pid in patients:
        sub = idx[(idx.user_key == pid) & idx.config_idx.isin(PRIMARY.config_indices)]
        assert eff.loc[pid, "observed_z"] == pytest.approx(sub.loc[np.isfinite(sub.spearman_z), "spearman_z"].mean(), abs=1e-12)
    # and under Pearson 'effective' == 'all' (nothing collapses)
    a = select(summary, SummarySpec("all", "pearson", "mean")).set_index("patient_id")
    e = select(summary, SummarySpec("effective", "pearson", "mean")).set_index("patient_id")
    assert np.allclose(a["observed_z"], e["observed_z"]) and np.allclose(a["p_value"], e["p_value"])


def test_spearman_effective_p_values_use_the_effective_null(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    sources = null_sources_for(per_config_null=per_config_null)
    summary = build_world_summary(observed, null_sources=sources)
    eff = select(summary, PRIMARY).set_index("patient_id")
    null_eff = sources["effective/spearman"]
    for pid in patients:
        z = eff.loc[pid, "observed_z"]
        nv = null_eff[null_eff.patient_id == pid]["null_mean_z"].dropna().to_numpy()
        expected = (np.sum(np.abs(nv) >= abs(z)) + 1) / (len(nv) + 1)
        assert eff.loc[pid, "p_value"] == pytest.approx(expected)
        assert eff.loc[pid, "n_permutations"] == len(nv)


def test_summary_skips_specs_without_null_and_never_fabricates(synthetic_world):
    observed, per_config_null, _ = synthetic_world
    legacy = null_sources_for(per_config_null=per_config_null)["all/spearman"]
    with pytest.warns(UserWarning, match="No null distribution"):
        summary = build_world_summary(observed, legacy)
    assert {sp.key for sp in available_specs(summary)} == {"all/spearman/mean", "all/spearman/median"}
    assert select(summary, PRIMARY).empty
    with pytest.raises(ValueError, match="no spec has a null"):
        build_world_summary(observed, null_sources={})


def test_observed_table_covers_all_specs_without_a_null(synthetic_world):
    observed, _, patients = synthetic_world
    t = observed_table(observed)
    assert len(t) == len(ALL_SPECS) * len(patients)
    assert "p_value" not in t.columns
    prim = t[(t.config_set == "effective") & (t.correlation_type == "spearman") & (t.measure == "median")].set_index("patient_id")
    direct = observed_statistics(observed, "spearman_z", PRIMARY.config_indices).set_index("patient_id")
    assert np.allclose(prim["observed_z"], direct.loc[prim.index, "median"], equal_nan=True)


def test_threshold_sweep_is_monotone_and_matches_concordant_set(synthetic_world):
    observed, per_config_null, _ = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    sweep = threshold_sweep(summary, thresholds=(0.0, 0.3, 0.6, 5.0))
    assert len(sweep) == len(ALL_SPECS) * 4
    for sp in ALL_SPECS:
        rows = sweep[(sweep.config_set == sp.config_set) & (sweep.correlation_type == sp.correlation_type) & (sweep.measure == sp.measure)]
        rows = rows.sort_values("threshold")
        assert list(rows["n_concordant"]) == sorted(rows["n_concordant"], reverse=True)
        assert rows.iloc[-1]["n_concordant"] == 0
        members = [int(x) for x in str(rows.iloc[0]["concordant"]).split()] if rows.iloc[0]["n_concordant"] else []
        assert members == concordant_set(summary, sp, 0.0)
    sets = derive_concordant_sets(summary)
    assert set(sets) == {sp.key for sp in ALL_SPECS}


def test_to_v1_measure_table_defaults_to_v1_spec(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    t = to_v1_measure_table(summary, "median")
    assert "observed_median_z" in t.columns and len(t) == len(patients)
    t2 = to_v1_measure_table(summary, "median", spec=SummarySpec("effective", "spearman", "median"))
    assert not np.allclose(t.set_index("patient_id")["observed_median_z"], t2.set_index("patient_id").loc[t["patient_id"], "observed_median_z"])
