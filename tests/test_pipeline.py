"""World pipeline (item 8): span applied before the join, observed table, null, build_world end to end."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import worlds
from aamos_concordance.configs import attach_config_idx
from aamos_concordance.data import RawData
from aamos_concordance.engine import precompute_patient
from aamos_concordance.pipeline import DEFAULT_PATIENTS, build_world, generate_world_null, observed_per_config_table, world_frames
from aamos_concordance.spans import patient_spans
from aamos_concordance.summary import PRIMARY, available_specs, ALL_SPECS
from aamos_concordance.worlds import BASELINE, WorldSpec
from tests.synthetic import make_dataset


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(worlds, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(worlds, "V2_DIR", tmp_path / "v2")
    monkeypatch.setattr(worlds, "REPO_ROOT", tmp_path)
    import aamos_concordance.pipeline as pl
    monkeypatch.setattr(pl, "REPO_ROOT", tmp_path)
    return tmp_path


def test_world_frames_apply_the_span():
    q, inh = make_dataset(4)
    for pid in q.user_key.unique():
        spans = patient_spans(q[q.user_key == pid], inh[inh.user_key == pid])
        for span in ("Q", "D", "union", "intersection"):
            qq, ii, sp = world_frames(WorldSpec(span, "A"), q, inh, pid)
            assert sp == spans[span]
            if sp is not None:
                assert qq["date"].between(sp.start, sp.end).all() and ii["date"].between(sp.start, sp.end).all()
            else:
                assert len(qq) == 0 and len(ii) == 0


def test_union_is_no_trimming_and_Q_keeps_all_questionnaire_rows():
    q, inh = make_dataset(1)
    for pid in q.user_key.unique():
        qq_u, ii_u, _ = world_frames(BASELINE, q, inh, pid)
        assert len(qq_u) == (q.user_key == pid).sum() and len(ii_u) == (inh.user_key == pid).sum()
        qq_q, ii_q, _ = world_frames(WorldSpec("Q", "A"), q, inh, pid)
        assert len(qq_q) == len(qq_u) and len(ii_q) <= len(ii_u)


def test_observed_table_layout_and_sample_sizes():
    q, inh = make_dataset(2)
    pats = sorted(q.user_key.unique())
    obs = observed_per_config_table(BASELINE, q, inh, pats)
    assert set(obs.columns) >= {"user_key", "timestamp_window", "use_daily_max_windows", "use_calendar_days",
                                "filter_out_zero_usage_entries", "categorization_method", "spearman_z", "pearson_z", "sample_size", "config_idx"}
    assert obs.groupby("user_key").size().eq(132).all()
    # attach_config_idx round-trips the five-parameter key to the same config_idx
    assert (attach_config_idx(obs.drop(columns="config_idx"))["config_idx"].to_numpy() == obs["config_idx"].to_numpy()).all()
    # sample_size: unfiltered configs see every row; filtered ones no more
    for pid in pats:
        n = int((q.user_key == pid).sum())
        g = obs[obs.user_key == pid]
        assert (g[~g.filter_out_zero_usage_entries]["sample_size"] == n).all()
        assert (g[g.filter_out_zero_usage_entries]["sample_size"] <= n).all()


def test_observed_table_drops_patients_with_unformable_span():
    q, inh = make_dataset(0)  # patient 100 has no device records
    obs = observed_per_config_table(WorldSpec("D", "A"), q, inh, [100, 101])
    assert set(obs.user_key) == {101}
    null, modes = generate_world_null(WorldSpec("intersection", "A"), q, inh, [100, 101], n_perm=3, exact_max=0)
    assert modes[100]["mode"] == "excluded" and set(null.patient_id) == {101}


def test_build_world_end_to_end_on_synthetic_data(sandbox):
    q, inh = make_dataset(5)
    pats = [101, 102, 104, 105]
    raw = RawData(patient_info=pd.DataFrame({"user_key": pats}), questionnaire=q, inhaler=inh, data_dir=sandbox, hashes={}, verified=False)
    w = WorldSpec("D", "A")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sets = build_world(w, raw, pats, n_perm=8, exact_max=50, script="test")
    d = worlds.world_dir(w)
    for name in ("per_config_z.csv", "null.parquet", "null_per_config.parquet", "summary.csv", "observed.csv", "threshold_sweep.csv", "config.json"):
        assert (d / name).exists(), name
    summary = pd.read_csv(d / "summary.csv")
    # synthetic patients are too short for the largest minimum-rows sensitivity
    max_rows = pd.read_csv(d / "per_config_z.csv")["sample_size"].max()
    reachable = {sp.key for sp in ALL_SPECS if sp.min_rows <= max_rows}
    assert {sp.key for sp in available_specs(summary)} == reachable
    assert set(sets) == reachable
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["span"] == "D" and cfg["absence_case"] == "A" and cfg["null_engine"] == "vectorised"
    assert set(int(k) for k in cfg["null_modes"]) == set(pats)
    assert worlds.list_worlds() == [w.world_id]
    assert worlds.load_concordant_sets()[w.world_id] == sets
    # observed statistic for the primary spec derives from the per-config table written
    per_config = pd.read_csv(d / "per_config_z.csv")
    from aamos_concordance.summary import observed_statistics, select
    obs = observed_statistics(per_config, "spearman_z", PRIMARY.config_indices).set_index("patient_id")
    prim = select(summary, PRIMARY).set_index("patient_id")
    assert np.allclose(prim["observed_z"], obs.loc[prim.index, "mean"], equal_nan=True)


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------

def test_baseline_union_observed_table_equals_v1(v1_definitions, data_dir, frozen_results_dir):
    """The union span trims nothing, so the engine's observed table for the baseline is v1's table."""
    q = pd.read_csv(data_dir / "anonym_aamos00_dailyquestionnaire_dt.csv").drop_duplicates()
    inh = pd.read_csv(data_dir / "anonym_aamos00_smartinhaler_dt.csv").drop_duplicates()
    ours = observed_per_config_table(BASELINE, q, inh, DEFAULT_PATIENTS)
    v1 = attach_config_idx(pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv"))
    m = v1.merge(ours, on=["user_key", "config_idx"], suffixes=("_v1", "_e"))
    assert len(m) == len(v1[v1.user_key.isin(DEFAULT_PATIENTS)])
    assert (m.sample_size_v1 == m.sample_size_e).all()
    for t in ("spearman_z", "pearson_z"):
        a, b = m[f"{t}_v1"].to_numpy(float), m[f"{t}_e"].to_numpy()
        inf = np.isinf(a)
        assert np.isnan(b[inf]).all()
        fin = np.isfinite(a) & np.isfinite(b)
        assert np.abs(a[fin] - b[fin]).max() < 1e-9
        assert (np.isnan(a) == np.isnan(b))[~inf].all()


def test_Q_span_differs_from_union_only_at_edge_windows(data_dir):
    """Q removes device records outside the questionnaire period, which the lookback windows of the
    first questionnaire days (and calendar-day offsets) can reach; same-day calendar windows cannot."""
    q = pd.read_csv(data_dir / "anonym_aamos00_dailyquestionnaire_dt.csv").drop_duplicates()
    inh = pd.read_csv(data_dir / "anonym_aamos00_smartinhaler_dt.csv").drop_duplicates()
    u = observed_per_config_table(BASELINE, q, inh, DEFAULT_PATIENTS)
    qq = observed_per_config_table(WorldSpec("Q", "A"), q, inh, DEFAULT_PATIENTS)
    m = u.merge(qq, on=["user_key", "config_idx"], suffixes=("_u", "_q"))
    same_day = m[m.use_calendar_days_u & (m.timestamp_window_u == 12)]
    assert np.allclose(same_day.spearman_z_u, same_day.spearman_z_q, equal_nan=True)
    differs = m[~np.isclose(m.spearman_z_u, m.spearman_z_q, equal_nan=True)]
    assert set(differs.user_key) <= {447, 625, 917, 939}
