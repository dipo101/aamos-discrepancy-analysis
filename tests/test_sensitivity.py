"""Sensitivity grid (item 16), case B stability (item 17), rerun list (item 18)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aamos_concordance import worlds
from aamos_concordance.sensitivity import (
    case_b_stability,
    grid,
    load_world_summaries,
    spec_stability,
    threshold_curve,
    world_stability,
    worlds_to_rerun,
)
from aamos_concordance.summary import PRIMARY, SummarySpec
from aamos_concordance.worlds import BASELINE


def _summary(rows):
    """rows: (patient_id, observed_z, significant) for the primary spec only."""
    df = pd.DataFrame(rows, columns=["patient_id", "observed_z", "significant_bonferroni"])
    df["config_set"], df["correlation_type"], df["measure"] = "effective", "spearman", "mean"
    return df


@pytest.fixture
def toy_worlds():
    base = _summary([(1, 0.9, True), (2, 0.6, True), (3, 0.3, True), (4, 0.7, False)])
    return {
        "span=union__case=A": base,                                                 # {1, 2}
        "span=Q__case=A": _summary([(1, 0.9, True), (2, 0.4, True), (3, 0.3, True), (4, 0.7, False)]),  # {1}: 2 leaves
        "span=union__case=C": _summary([(1, 0.9, True), (2, 0.6, True), (3, 0.55, True), (4, 0.7, True)]),  # {1,2,3,4}: 3,4 join
        "span=union__case=B__k=00": _summary([(1, 0.9, True), (2, 0.52, True), (3, 0.3, True)]),   # {1, 2}
        "span=union__case=B__k=01": _summary([(1, 0.9, True), (2, 0.48, True), (3, 0.3, True)]),   # {1}
        "span=union__case=B__k=02": _summary([(1, 0.9, True), (2, 0.51, False), (3, 0.3, True)]),  # {1}
    }


def test_grid_exact_set_comparison(toy_worlds):
    g = grid(toy_worlds, specs=[PRIMARY], thresholds=[0.5])
    g = g.set_index("world_id")
    assert g.loc["span=union__case=A", "changed"] == False and g.loc["span=union__case=A", "concordant"] == "1 2"
    assert g.loc["span=Q__case=A", "changed"] == True and g.loc["span=Q__case=A", "leavers"] == "2" and g.loc["span=Q__case=A", "joiners"] == ""
    assert g.loc["span=union__case=C", "joiners"] == "3 4" and g.loc["span=union__case=C", "leavers"] == ""
    assert g.loc["span=union__case=B__k=00", "changed"] == False
    assert g.loc["span=union__case=B__k=01", "leavers"] == "2"
    assert set(g.columns) >= {"span", "absence_case", "imputation", "n_concordant", "n_assessed"}


def test_grid_thresholds_and_specs(toy_worlds):
    g = grid(toy_worlds, specs=[PRIMARY], thresholds=[0.1, 0.5, 0.8])
    assert len(g) == 6 * 3
    base = g[(g.world_id == "span=union__case=A")].set_index("threshold")
    assert base.loc[0.1, "concordant"] == "1 2 3" and base.loc[0.8, "concordant"] == "1"
    # a spec absent from the summaries yields no rows rather than an error
    assert grid(toy_worlds, specs=[SummarySpec("all", "pearson", "median")], thresholds=[0.5]).empty
    with pytest.raises(ValueError):
        grid({"span=Q__case=A": toy_worlds["span=Q__case=A"]})


def test_case_b_stability_counts(toy_worlds):
    cb = case_b_stability(toy_worlds).set_index("patient_id")
    assert (cb["n_imputations"] == 3).all()
    assert cb.loc[1, "n_concordant"] == 3 and cb.loc[1, "n_significant"] == 3
    assert cb.loc[2, "n_significant"] == 2 and cb.loc[2, "n_above_threshold"] == 2 and cb.loc[2, "n_concordant"] == 1
    assert cb.loc[3, "n_concordant"] == 0 and cb.loc[3, "n_significant"] == 3
    assert cb.loc[2, "mean_observed_z"] == pytest.approx((0.52 + 0.48 + 0.51) / 3)
    assert case_b_stability({"span=union__case=A": toy_worlds["span=union__case=A"]}).empty


def test_world_stability_collapses_b_to_always_concordant(toy_worlds):
    ws = world_stability(toy_worlds).set_index("world_id")
    assert list(ws.index) == ["span=union__case=A", "span=union__case=B", "span=union__case=C", "span=Q__case=A"]
    b = ws.loc["span=union__case=B"]
    assert b["concordant"] == "1" and b["changed"] == True and b["leavers"] == "2"
    assert b["imputations"] == "1:3/3; 2:1/3"
    assert ws.loc["span=union__case=C", "joiners"] == "3 4"


def test_threshold_curve_is_monotone(toy_worlds):
    tc = threshold_curve(toy_worlds["span=union__case=A"], thresholds=(0.0, 0.5, 1.0), specs=[PRIMARY])
    assert list(tc["n_concordant"]) == [3, 2, 0]


def test_worlds_to_rerun_is_exact_set_inequality(toy_worlds):
    r = worlds_to_rerun(toy_worlds)
    assert r["rerun"] == ["span=Q__case=A", "span=union__case=C"]
    assert r["empty"] == []
    # B: always-concordant set is {1}, differs from baseline {1, 2} and is non-empty -> flagged
    assert r["case_b_flagged"] == ["span=union__case=B"]
    empty_world = toy_worlds | {"span=D__case=C": _summary([(1, 0.2, True), (2, 0.1, True)])}
    assert worlds_to_rerun(empty_world)["empty"] == ["span=D__case=C"]


def test_load_world_summaries_reads_built_worlds(tmp_path, monkeypatch):
    monkeypatch.setattr(worlds, "V2_DIR", tmp_path / "v2")
    for wid in ("span=union__case=A", "span=D__case=A"):
        d = worlds.world_dir(wid, create=True)
        _summary([(1, 0.9, True)]).to_csv(d / "summary.csv", index=False)
        worlds.register_world(wid)
    worlds.register_world("span=Q__case=A")  # registered but no summary yet
    loaded = load_world_summaries()
    assert set(loaded) == {"span=union__case=A", "span=D__case=A"}


# --------------------------------------------------------------------------
# Committed grid over the real worlds
# --------------------------------------------------------------------------

def test_committed_grid_matches_recomputation():
    sens = worlds.V2_DIR / "sensitivity"
    committed = pd.read_csv(sens / "grid.csv", keep_default_na=False)
    fresh = grid(load_world_summaries())
    fresh["imputation"] = fresh["imputation"].astype(str)
    fresh["threshold"] = fresh["threshold"].astype(float)
    committed["threshold"] = committed["threshold"].astype(float)
    committed["changed"] = committed["changed"].astype(str) == "True"
    fresh["changed"] = fresh["changed"].astype(bool)
    pd.testing.assert_frame_equal(committed, fresh, check_dtype=False)
    listing = json.loads((sens / "worlds_to_rerun.json").read_text())
    fresh = worlds_to_rerun(load_world_summaries())
    assert {k: listing[k] for k in fresh} == fresh
    assert BASELINE.world_id not in listing["rerun"]
    # every A world except the baseline and D (under the artefact reading) loses 917; every C world gives 294, 473, 702
    assert listing["rerun"] == [
        "span=D__case=A__dup=real", "span=D__case=A__dup=resolution", "span=D__case=C",
        "span=D__case=C__dup=real", "span=D__case=C__dup=resolution", "span=Q__case=A",
        "span=Q__case=A__dup=real", "span=Q__case=A__dup=resolution", "span=Q__case=C",
        "span=Q__case=C__dup=real", "span=Q__case=C__dup=resolution", "span=intersection__case=A",
        "span=intersection__case=A__dup=real", "span=intersection__case=A__dup=resolution", "span=intersection__case=C",
        "span=intersection__case=C__dup=real", "span=intersection__case=C__dup=resolution", "span=union__case=A__dup=real",
        "span=union__case=A__dup=resolution", "span=union__case=C", "span=union__case=C__dup=real",
        "span=union__case=C__dup=resolution",
    ]
    assert listing["case_b_flagged"] == []


def test_committed_world_stability_headline():
    ws = pd.read_csv(worlds.V2_DIR / "sensitivity" / "world_stability.csv", keep_default_na=False).set_index("world_id")
    assert len(ws) == 36  # 3 duplicate readings x 4 spans x (A, B collapsed, C)
    assert ws.loc["span=union__case=A", "concordant"] == "190 702 917"
    assert ws.loc["span=D__case=A", "changed"] == False
    for dup in ("", "__dup=real", "__dup=resolution"):
        for span in ("union", "Q", "D", "intersection"):
            assert ws.loc[f"span={span}__case=C{dup}", "concordant"] == "294 473 702"
            assert ws.loc[f"span={span}__case=B{dup}", "n_concordant"] == 0
            if (span, dup) not in (("union", ""), ("D", "")):
                assert ws.loc[f"span={span}__case=A{dup}", "concordant"] == "190 702"  # 917 leaves
    assert ws.loc["span=union__case=B", "imputations"] == "398:5/10; 702:5/10"
    cb = pd.read_csv(worlds.V2_DIR / "sensitivity" / "case_b_stability.csv")
    core = cb[cb.patient_id.isin([294, 473, 702])]
    assert (core["n_significant"] == 10).all()  # B moves the threshold, not significance


def test_committed_spec_stability_headline():
    ss = pd.read_csv(worlds.V2_DIR / "sensitivity" / "spec_stability.csv", keep_default_na=False)
    base = ss[ss.world_id == "span=union__case=A"].set_index("spec")["concordant"].to_dict()
    assert base == {
        "effective/spearman/mean": "190 702 917",
        "effective_lookahead/spearman/mean": "702 917",
        "effective/spearman/mean/min_rows=5": "190 702 917",
        "effective/spearman/mean/min_rows=20": "190 398 702",
        "effective/spearman/mean/min_rows=37": "190 702",
        "effective/spearman/mean/exclude=fostair": "190 702",
        "all/spearman/mean": "294 473 702 917",
    }


def test_spec_stability_compares_each_option_with_the_primary():
    def rows(spec, members_z):
        df = pd.DataFrame(members_z, columns=["patient_id", "observed_z", "significant_bonferroni"])
        for c in ("config_set", "correlation_type", "measure", "min_rows", "exclude"):
            df[c] = getattr(spec, c)
        return df
    excl = SummarySpec("effective", "spearman", "mean", exclude="fostair")
    mr20 = SummarySpec("effective", "spearman", "mean", min_rows=20)
    summary = pd.concat([
        rows(PRIMARY, [(1, 0.9, True), (2, 0.6, True), (917, 0.99, True)]),
        rows(excl, [(1, 0.9, True), (2, 0.6, True)]),
        rows(mr20, [(1, 0.9, True), (2, 0.4, True), (3, 0.7, True)]),
    ], ignore_index=True)
    out = spec_stability({"span=union__case=A": summary, "span=union__case=B__k=00": summary}).set_index("spec")
    assert set(out["world_id"]) == {"span=union__case=A"}  # B is reported through k-of-n stability
    assert list(out.index) == [PRIMARY.key, mr20.key, excl.key]  # specs absent from the summary are skipped
    assert out.loc[PRIMARY.key, "changed"] == False
    assert out.loc[excl.key, "leavers"] == "917" and out.loc[excl.key, "n_assessed"] == 2
    assert out.loc[mr20.key, "joiners"] == "3" and out.loc[mr20.key, "leavers"] == "2 917"


def test_case_b_stability_and_world_stability_separate_duplicate_readings(toy_worlds):
    ws = dict(toy_worlds)
    ws["span=union__case=B__k=00__dup=real"] = _summary([(1, 0.9, True), (2, 0.9, True), (3, 0.9, True)])
    ws["span=union__case=B__k=01__dup=real"] = _summary([(1, 0.9, True), (2, 0.9, True), (3, 0.9, True)])
    ws["span=union__case=C__dup=real"] = _summary([(1, 0.9, True)])
    cb = case_b_stability(ws)
    assert set(cb["duplicates"]) == {"artefact", "real"}
    real = cb[cb.duplicates == "real"].set_index("patient_id")
    assert real.loc[3, "n_concordant"] == 2 and real.loc[3, "n_imputations"] == 2
    st = world_stability(ws).set_index("world_id")
    assert st.loc["span=union__case=B__dup=real", "concordant"] == "1 2 3"
    assert st.loc["span=union__case=B", "concordant"] == "1"
    assert st.loc["span=union__case=C__dup=real", "duplicates"] == "real"
    assert worlds_to_rerun(ws)["case_b_flagged"] == ["span=union__case=B", "span=union__case=B__dup=real"]
