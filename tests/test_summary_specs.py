"""Items 13-15: summary specs, per-spec summaries from a per-config null, threshold sweep.

The v1 null only supports all/spearman. To exercise the other specs end to
end, a small synthetic world is built here with the engine so that a real
per-config null (with per-permutation row counts) exists, and every spec is
summarised from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import generate_param_combinations
from aamos_concordance import summary as summary_module
from aamos_concordance.engine import run_patient_sampled
from aamos_concordance.configs import N_SPEARMAN_EFFECTIVE, attach_config_idx, config_indices_for
from aamos_concordance.summary import (
    ALL_SPECS,
    BASE_SPECS,
    MIN_ROWS_GRID,
    SENSITIVITY_SPECS,
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
    for sp in BASE_SPECS:  # default options leave the v1-era keys unchanged
        assert sp.key == f"{sp.config_set}/{sp.correlation_type}/{sp.measure}"
        assert sp.null_key == f"{sp.config_set}/{sp.correlation_type}"
    assert PRIMARY == SummarySpec("effective", "spearman", "mean") and PRIMARY.min_rows == 3 and PRIMARY.exclude == "none"
    assert V1_SPEC == SummarySpec("all", "spearman", "mean")
    assert len(BASE_SPECS) == 12 and len(ALL_SPECS) == 16
    assert [sp.key for sp in SENSITIVITY_SPECS] == [
        "effective/spearman/mean/min_rows=5", "effective/spearman/mean/min_rows=20",
        "effective/spearman/mean/min_rows=37", "effective/spearman/mean/exclude=fostair"]
    assert SummarySpec.parse("effective/spearman/mean/min_rows=20").null_key == "effective/spearman/min_rows=20"
    assert SummarySpec.parse("effective/spearman/mean/exclude=fostair").null_key == "effective/spearman"
    assert SummarySpec.parse("all/pearson/median/min_rows=5/exclude=fostair") == SummarySpec("all", "pearson", "median", 5, "fostair")


@pytest.mark.parametrize("bad", [("nope", "spearman", "mean"), ("all", "kendall", "mean"), ("all", "spearman", "mode"),
                                 ("all", "spearman", "mean", 2), ("all", "spearman", "mean", 3, "everyone")])
def test_spec_rejects_unknown_values(bad):
    with pytest.raises(ValueError):
        SummarySpec(*bad)


@pytest.mark.parametrize("key", ["all/spearman", "all/spearman/mean/min_rows", "all/spearman/mean/foo=1",
                                 "all/spearman/mean/min_rows=5/min_rows=20"])
def test_spec_parse_rejects_malformed_keys(key):
    with pytest.raises(ValueError):
        SummarySpec.parse(key)


def test_spec_config_indices():
    assert len(SummarySpec("effective_lookahead").config_indices) == N_SPEARMAN_EFFECTIVE
    assert len(PRIMARY.config_indices) == N_SPEARMAN_EFFECTIVE * 8 // 10  # look-ahead windows excluded
    assert V1_SPEC.config_indices == list(range(132))
    assert len(SummarySpec("effective", "pearson", "mean").config_indices) == 96
    assert len(SummarySpec("effective_lookahead", "pearson", "mean").config_indices) == 120
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
        nulls.append(run_patient_sampled(pid, qq, ii, 0, 12, 42, param_combinations=combos))
    return pd.DataFrame(obs_rows), pd.concat(nulls, ignore_index=True), patients


def test_null_sources_cover_every_spec_with_per_config_null(synthetic_world):
    observed, per_config_null, _ = synthetic_world
    sources = null_sources_for(per_config_null=per_config_null)
    assert set(sources) == {f"{c}/{t}" for c in ("all", "effective", "effective_lookahead") for t in ("spearman", "pearson")} | {
        f"effective/spearman/min_rows={n}" for n in MIN_ROWS_GRID if n != 3}
    # legacy-only gives just the v1 key
    assert set(null_sources_for(legacy_null=sources["all/spearman"])) == {"all/spearman"}
    # a per-config null without row counts cannot serve a minimum above 3
    assert set(null_sources_for(per_config_null=per_config_null.drop(columns="n_rows"))) == {
        f"{c}/{t}" for c in ("all", "effective", "effective_lookahead") for t in ("spearman", "pearson")}


def test_summary_has_every_spec_and_primary_differs_from_v1_where_expected(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    # the synthetic patients are too short for the larger minimums; those specs have no one to assess
    reachable = {sp.key for sp in ALL_SPECS if sp.min_rows <= observed["sample_size"].max()}
    assert {sp.key for sp in available_specs(summary)} == reachable and len(reachable) >= len(BASE_SPECS)
    for sp in ALL_SPECS:
        s = select(summary, sp)
        if sp.min_rows == 3:
            assert sorted(s["patient_id"]) == patients
        assert (s["n_configs_in_set"] == len(sp.config_indices)).all()
        assert (s["n_valid_configs_observed"] <= len(sp.config_indices)).all()
    # effective/spearman observed mean is the mean over the effective representatives only
    eff = select(summary, PRIMARY).set_index("patient_id")
    idx = attach_config_idx(observed)
    for pid in patients:
        sub = idx[(idx.user_key == pid) & idx.config_idx.isin(PRIMARY.config_indices)]
        assert eff.loc[pid, "observed_z"] == pytest.approx(sub.loc[np.isfinite(sub.spearman_z), "spearman_z"].mean(), abs=1e-12)
    # effective/pearson drops only the chunk-24 duplicates of rolling-24
    pe = SummarySpec("effective", "pearson", "mean")
    e = select(summary, pe).set_index("patient_id")
    for pid in patients:
        sub = idx[(idx.user_key == pid) & idx.config_idx.isin(pe.config_indices)]
        assert e.loc[pid, "observed_z"] == pytest.approx(sub.loc[np.isfinite(sub.pearson_z), "pearson_z"].mean(), abs=1e-12)


def test_spearman_effective_p_values_use_the_effective_null(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    sources = null_sources_for(per_config_null=per_config_null)
    summary = build_world_summary(observed, null_sources=sources)
    eff = select(summary, PRIMARY).set_index("patient_id")
    null_eff = sources["effective/spearman"]
    for pid in patients:
        z = eff.loc[pid, "observed_z"]
        nv = null_eff[null_eff.patient_id == pid]["null_mean_z"].dropna().to_numpy()
        c = nv.mean()
        expected = (np.sum(np.abs(nv - c) >= abs(z - c) - 1e-12) + 1) / (len(nv) + 1)
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
    assert len(t) == len(ALL_SPECS) * len(patients)  # a patient with no config at the minimum is a NaN row, not missing
    assert "p_value" not in t.columns
    prim = select(t, SummarySpec("effective", "spearman", "median")).set_index("patient_id")
    direct = observed_statistics(observed, "spearman_z", PRIMARY.config_indices).set_index("patient_id")
    assert np.allclose(prim["observed_z"], direct.loc[prim.index, "median"], equal_nan=True)


def test_threshold_sweep_is_monotone_and_matches_concordant_set(synthetic_world):
    observed, per_config_null, _ = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    sweep = threshold_sweep(summary, thresholds=(0.0, 0.3, 0.6, 5.0))
    specs = available_specs(summary)
    assert len(sweep) == len(specs) * 4
    for sp in specs:
        rows = select(sweep, sp).sort_values("threshold")
        assert list(rows["n_concordant"]) == sorted(rows["n_concordant"], reverse=True)
        assert rows.iloc[-1]["n_concordant"] == 0
        members = [int(x) for x in str(rows.iloc[0]["concordant"]).split()] if rows.iloc[0]["n_concordant"] else []
        assert members == concordant_set(summary, sp, 0.0)
    sets = derive_concordant_sets(summary)
    assert set(sets) == {sp.key for sp in specs}


def test_to_v1_measure_table_defaults_to_v1_spec(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    t = to_v1_measure_table(summary, "median")
    assert "observed_median_z" in t.columns and len(t) == len(patients)
    t2 = to_v1_measure_table(summary, "median", spec=SummarySpec("effective", "spearman", "median"))
    assert not np.allclose(t.set_index("patient_id")["observed_median_z"], t2.set_index("patient_id").loc[t["patient_id"], "observed_median_z"])


# --------------------------------------------------------------------------
# Minimum rows and patient exclusion
# --------------------------------------------------------------------------

def test_engine_n_rows_is_the_observed_sample_size_and_varies_under_the_filter(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    idx = attach_config_idx(observed).set_index(["user_key", "config_idx"])["sample_size"]
    first = per_config_null[per_config_null.permutation_idx == 0].set_index(["patient_id", "config_idx"])["n_rows"]
    combos = generate_param_combinations()
    unfiltered = [i for i, c in enumerate(combos) if not c["filter_out_zero_usage"]]
    # without the filter every permutation correlates every row, which is the observed sample size
    for pid in patients:
        for ci in unfiltered:
            assert first.loc[(pid, ci)] == idx.loc[(pid, ci)]
    filtered = per_config_null[per_config_null.config_idx.isin([i for i in range(132) if i not in unfiltered])]
    assert filtered.groupby(["patient_id", "config_idx"])["n_rows"].nunique().max() > 1


def test_min_rows_invalidates_configs_below_the_minimum_in_observed_and_null(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    cut = int(np.median(observed["sample_size"]))
    obs = observed_statistics(observed, "spearman_z", min_rows=cut).set_index("patient_id")
    for pid in patients:
        g = observed[(observed.user_key == pid) & (observed.sample_size >= cut)]
        assert obs.loc[pid, "n_valid"] == np.isfinite(g.spearman_z).sum()
    from aamos_concordance.batch import null_from_per_config
    null = null_from_per_config(per_config_null, min_rows=cut)
    kept = per_config_null[(per_config_null.n_rows >= cut) & np.isfinite(per_config_null.spearman_z)]
    expected = kept.groupby(["patient_id", "permutation_idx"]).size()
    got = null.set_index(["patient_id", "permutation_idx"])["n_valid_configs"]
    assert (got.reindex(expected.index) == expected).all() and (got.drop(expected.index) == 0).all()
    with pytest.raises(ValueError, match="n_rows"):
        null_from_per_config(per_config_null.drop(columns="n_rows"), min_rows=cut)


def test_a_patient_with_no_config_at_the_minimum_leaves_the_bonferroni_count(synthetic_world):
    observed, per_config_null, patients = synthetic_world
    smallest = observed.groupby("user_key")["sample_size"].max().idxmin()
    cut = int(observed[observed.user_key == smallest]["sample_size"].max()) + 1
    spec = SummarySpec("all", "spearman", "mean", min_rows=cut)
    obs = observed_statistics(observed, "spearman_z", min_rows=cut).set_index("patient_id")
    assert np.isnan(obs.loc[smallest, "mean"])
    s = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null, specs=[spec]), specs=[spec])
    assert smallest not in set(s.patient_id)
    assert (s["n_patients_bonferroni"] == s["patient_id"].nunique()).all()


def test_exclusion_removes_patients_and_shrinks_bonferroni(synthetic_world, monkeypatch):
    observed, per_config_null, patients = synthetic_world
    monkeypatch.setitem(summary_module.PATIENT_EXCLUSIONS, "fostair", (patients[0],))
    sources = null_sources_for(per_config_null=per_config_null)
    base = select(build_world_summary(observed, null_sources=sources, specs=[PRIMARY]), PRIMARY).set_index("patient_id")
    spec = SummarySpec("effective", "spearman", "mean", exclude="fostair")
    excl = select(build_world_summary(observed, null_sources=sources, specs=[spec]), spec).set_index("patient_id")
    assert sorted(excl.index) == patients[1:]
    assert (excl["n_patients_bonferroni"] == len(patients) - 1).all() and (base["n_patients_bonferroni"] == len(patients)).all()
    # observed statistics and uncorrected p-values of the others are untouched
    assert np.allclose(excl["observed_z"], base.loc[excl.index, "observed_z"])
    assert np.allclose(excl["p_value"], base.loc[excl.index, "p_value"])
    assert np.allclose(excl["p_bonferroni"], np.minimum(1, excl["p_value"] * (len(patients) - 1)))


def test_select_reads_summaries_written_before_the_options_existed(synthetic_world):
    observed, per_config_null, _ = synthetic_world
    summary = build_world_summary(observed, null_sources=null_sources_for(per_config_null=per_config_null))
    old = summary[(summary.min_rows == 3) & (summary.exclude == "none")].drop(columns=["min_rows", "exclude"])
    assert {sp.key for sp in available_specs(old)} == {sp.key for sp in BASE_SPECS}
    assert select(old, PRIMARY)["observed_z"].tolist() == select(summary, PRIMARY)["observed_z"].tolist()
    assert select(old, SENSITIVITY_SPECS[0]).empty
