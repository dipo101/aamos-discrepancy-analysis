"""Sensitivity grid over built worlds (items 16, 17, 18).

Everything here is post-processing of the per-world ``summary.csv`` files:
no permutation is rerun.

* :func:`grid`: one row per (world, summary spec, threshold) with the
  concordant set, its size, and the **exact set comparison** with the
  baseline world under the same spec and threshold: ``changed``, ``joiners``
  and ``leavers``. Set stability means set equality in both directions
  (KT16: "does anyone else join the group?"), never survival of a fixed
  trio.
* :func:`world_stability`: the item 16 marginal, the set under every world at
  the primary spec and threshold, with case B worlds collapsed to the
  per-imputation stability of item 17.
* :func:`case_b_stability`: item 17, per patient and span, in how many of
  the imputation worlds the patient is Bonferroni-significant, above the
  Z threshold, and both (concordant). Reported separately because under B
  significance is stable and the threshold is what moves.
* :func:`threshold_curve`: the other item 16 marginal, set size against the
  threshold cut for the baseline world and each spec.
* :func:`worlds_to_rerun`: item 18, the worlds whose primary set differs
  from the baseline's, which are the only ones whose downstream comparisons
  (temporal, Bland-Altman, feedback) need rerunning. Case B worlds are not
  listed individually: B is reported as k-of-n stability (item 17), and a
  span's B result is its *always-concordant* set; that set is compared with
  the baseline and, when it differs and is non-empty, the span is flagged
  for a decision rather than a mechanical rerun of ten imputations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .summary import ALL_SPECS, DEFAULT_THRESHOLD, DEFAULT_THRESHOLDS, PRIMARY, SummarySpec, concordant_set, select
from .worlds import BASELINE, SUMMARY_CSV, WorldSpec, list_worlds, world_dir


def load_world_summaries(world_ids: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """``{world_id: summary}`` for every built world that has a summary.csv."""
    out = {}
    for wid in (world_ids if world_ids is not None else list_worlds()):
        p = world_dir(wid) / SUMMARY_CSV
        if p.exists():
            out[wid] = pd.read_csv(p)
    return out


def _fmt(members: Sequence[int]) -> str:
    return " ".join(str(m) for m in members)


def grid(
    summaries: Dict[str, pd.DataFrame],
    *,
    specs: Iterable[SummarySpec] = ALL_SPECS,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    baseline: str = BASELINE.world_id,
) -> pd.DataFrame:
    """The full sensitivity grid with exact-set comparison against the baseline world."""
    if baseline not in summaries:
        raise ValueError(f"baseline world {baseline!r} has no summary among the loaded worlds")
    rows: List[Dict] = []
    for wid, summary in summaries.items():
        w = WorldSpec.parse(wid)
        for spec in specs:
            if select(summary, spec).empty:
                continue
            for t in thresholds:
                members = concordant_set(summary, spec, t)
                ref = concordant_set(summaries[baseline], spec, t)
                joiners = sorted(set(members) - set(ref))
                leavers = sorted(set(ref) - set(members))
                rows.append({
                    "world_id": wid, "span": w.span, "absence_case": w.absence_case,
                    "imputation": "" if w.imputation is None else w.imputation,
                    "config_set": spec.config_set, "correlation_type": spec.correlation_type, "measure": spec.measure,
                    "threshold": t,
                    "n_concordant": len(members), "concordant": _fmt(members),
                    "n_assessed": int(select(summary, spec)["patient_id"].nunique()),
                    "changed": members != ref,
                    "joiners": _fmt(joiners), "leavers": _fmt(leavers),
                })
    cols = ["world_id", "span", "absence_case", "imputation", "config_set", "correlation_type", "measure", "threshold",
            "n_concordant", "concordant", "n_assessed", "changed", "joiners", "leavers"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["span", "absence_case", "imputation", "config_set", "correlation_type", "measure", "threshold"], kind="stable").reset_index(drop=True)


def case_b_stability(
    summaries: Dict[str, pd.DataFrame],
    *,
    spec: SummarySpec = PRIMARY,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Item 17: per (span, patient), counts over the imputation worlds of significant / above-threshold / concordant."""
    rows: List[Dict] = []
    by_span: Dict[str, List[str]] = {}
    for wid in summaries:
        w = WorldSpec.parse(wid)
        if w.absence_case == "B":
            by_span.setdefault(w.span, []).append(wid)
    for span, wids in sorted(by_span.items()):
        frames = []
        for wid in wids:
            s = select(summaries[wid], spec)[["patient_id", "observed_z", "significant_bonferroni"]].copy()
            s["k"] = WorldSpec.parse(wid).imputation
            frames.append(s)
        allk = pd.concat(frames, ignore_index=True)
        allk["above_threshold"] = allk["observed_z"] >= threshold
        allk["concordant"] = allk["above_threshold"] & allk["significant_bonferroni"]
        g = allk.groupby("patient_id")
        stats = pd.DataFrame({
            "n_imputations": g.size(),
            "n_significant": g["significant_bonferroni"].sum().astype(int),
            "n_above_threshold": g["above_threshold"].sum().astype(int),
            "n_concordant": g["concordant"].sum().astype(int),
            "mean_observed_z": g["observed_z"].mean(),
            "min_observed_z": g["observed_z"].min(),
            "max_observed_z": g["observed_z"].max(),
        }).reset_index()
        stats.insert(0, "span", span)
        rows.append(stats)
    if not rows:
        return pd.DataFrame(columns=["span", "patient_id", "n_imputations", "n_significant", "n_above_threshold", "n_concordant",
                                     "mean_observed_z", "min_observed_z", "max_observed_z"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["span", "n_concordant", "n_significant", "patient_id"], ascending=[True, False, False, True]).reset_index(drop=True)


def world_stability(
    summaries: Dict[str, pd.DataFrame],
    *,
    spec: SummarySpec = PRIMARY,
    threshold: float = DEFAULT_THRESHOLD,
    baseline: str = BASELINE.world_id,
) -> pd.DataFrame:
    """Item 16 marginal: the set under every world at one spec and threshold; B collapsed to k-of-n."""
    g = grid(summaries, specs=[spec], thresholds=[threshold], baseline=baseline)
    non_b = g[g.absence_case != "B"][["world_id", "span", "absence_case", "n_concordant", "concordant", "n_assessed", "changed", "joiners", "leavers"]].copy()
    non_b["imputations"] = ""
    b = case_b_stability(summaries, spec=spec, threshold=threshold)
    b_rows: List[Dict] = []
    ref = _fmt(concordant_set(summaries[baseline], spec, threshold))
    for span, bs in b.groupby("span"):
        n = int(bs["n_imputations"].max())
        always = bs[bs.n_concordant == n]["patient_id"].tolist()
        ever = bs[bs.n_concordant > 0]["patient_id"].tolist()
        b_rows.append({
            "world_id": f"span={span}__case=B", "span": span, "absence_case": "B",
            "n_concordant": len(always), "concordant": _fmt(sorted(always)),
            "n_assessed": int(bs["patient_id"].nunique()),
            "changed": _fmt(sorted(always)) != ref,
            "joiners": _fmt(sorted(set(always) - set(map(int, ref.split())))) if ref else _fmt(sorted(always)),
            "leavers": _fmt(sorted(set(map(int, ref.split())) - set(always))) if ref else "",
            "imputations": "; ".join(f"{int(r.patient_id)}:{int(r.n_concordant)}/{n}" for r in bs[bs.n_concordant > 0].itertuples()),
        })
    out = pd.concat([non_b, pd.DataFrame(b_rows)], ignore_index=True)
    order = {"union": 0, "Q": 1, "D": 2, "intersection": 3}
    out["_o"] = out["span"].map(order)
    return out.sort_values(["_o", "absence_case"]).drop(columns="_o").reset_index(drop=True)


def threshold_curve(summary: pd.DataFrame, thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                    specs: Iterable[SummarySpec] = ALL_SPECS) -> pd.DataFrame:
    """Set size against threshold for one world (the baseline), per spec."""
    rows = []
    for spec in specs:
        if select(summary, spec).empty:
            continue
        for t in thresholds:
            members = concordant_set(summary, spec, t)
            rows.append({"config_set": spec.config_set, "correlation_type": spec.correlation_type, "measure": spec.measure,
                         "threshold": t, "n_concordant": len(members), "concordant": _fmt(members)})
    return pd.DataFrame(rows)


def worlds_to_rerun(
    summaries: Dict[str, pd.DataFrame],
    *,
    spec: SummarySpec = PRIMARY,
    threshold: float = DEFAULT_THRESHOLD,
    baseline: str = BASELINE.world_id,
) -> Dict[str, List[str]]:
    """Item 18: exact-set comparison with the baseline at the primary spec and threshold.

    Returns ``{"rerun": [...], "empty": [...], "case_b_flagged": [...]}``:

    * ``rerun``: non-B worlds whose set differs and is non-empty, so the
      downstream comparisons are meaningful and need regenerating;
    * ``empty``: non-B worlds whose set differs by being empty (nothing to
      compare downstream);
    * ``case_b_flagged``: spans whose always-concordant set under B differs
      from the baseline and is non-empty (a decision, not a mechanical rerun).
    """
    ws = world_stability(summaries, spec=spec, threshold=threshold, baseline=baseline)
    changed = ws[ws.changed]
    non_b = changed[changed.absence_case != "B"]
    b = changed[changed.absence_case == "B"]
    return {
        "rerun": sorted(non_b[non_b.n_concordant > 0]["world_id"].tolist()),
        "empty": sorted(non_b[non_b.n_concordant == 0]["world_id"].tolist()),
        "case_b_flagged": sorted(b[b.n_concordant > 0]["span"].tolist()),
    }
