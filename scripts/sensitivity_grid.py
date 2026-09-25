#!/usr/bin/env python
"""Items 16-18: the sensitivity grid over every built world, its marginals, and the rerun list.

Writes to results/v2/sensitivity/:
  grid.csv                 world x spec x threshold: concordant set, size, changed vs baseline, joiners, leavers
  world_stability.csv      item 16 marginal: the set per world at the primary spec and threshold
                           (case B collapsed to "patient:k/n" per span)
  case_b_stability.csv     item 17: per span x patient, counts over imputations of significant /
                           above threshold / concordant
  spec_stability.csv       one-at-a-time analytical sensitivities vs the primary spec, per world
  threshold_curve.csv      item 16 marginal: set size vs threshold for the baseline, per spec
  worlds_to_rerun.json     item 18: worlds whose primary set differs from the baseline (exact set equality)

Usage:  python scripts/sensitivity_grid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import write_sidecar  # noqa: E402
from aamos_concordance.sensitivity import (  # noqa: E402
    case_b_stability, grid, load_world_summaries, spec_stability, threshold_curve, world_stability, worlds_to_rerun,
)
from aamos_concordance.summary import PRIMARY  # noqa: E402
from aamos_concordance.worlds import BASELINE, V2_DIR  # noqa: E402


def main(argv=None) -> int:
    out = V2_DIR / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    summaries = load_world_summaries()
    extra = {"script": "sensitivity_grid.py", "worlds": sorted(summaries), "primary_spec": PRIMARY.key, "baseline": BASELINE.world_id}

    g = grid(summaries)
    g.to_csv(out / "grid.csv", index=False)
    write_sidecar(out / "grid.csv", extra=extra)

    ws = world_stability(summaries)
    ws.to_csv(out / "world_stability.csv", index=False)
    write_sidecar(out / "world_stability.csv", extra=extra)

    cb = case_b_stability(summaries)
    cb.to_csv(out / "case_b_stability.csv", index=False)
    write_sidecar(out / "case_b_stability.csv", extra=extra)

    ss = spec_stability(summaries)
    ss.to_csv(out / "spec_stability.csv", index=False)
    write_sidecar(out / "spec_stability.csv", extra=extra)

    tc = threshold_curve(summaries[BASELINE.world_id])
    tc.to_csv(out / "threshold_curve.csv", index=False)
    write_sidecar(out / "threshold_curve.csv", extra=extra)

    rerun = worlds_to_rerun(summaries)
    (out / "worlds_to_rerun.json").write_text(json.dumps({"baseline": BASELINE.world_id, "spec": PRIMARY.key,
                                                            "threshold": 0.5, **rerun}, indent=2) + "\n")

    print(f"{len(summaries)} worlds, {len(g)} grid cells\n")
    print("World stability at the primary spec (effective/spearman/mean, t = 0.5):")
    print(ws[["world_id", "n_concordant", "concordant", "changed", "joiners", "leavers", "imputations"]].to_string(index=False))
    print("\nOne-at-a-time sensitivities vs the primary spec (baseline world):")
    base = ss[ss.world_id == BASELINE.world_id]
    print(base[["spec", "n_assessed", "n_concordant", "concordant", "joiners", "leavers"]].to_string(index=False))
    print("\nCase B stability (patients concordant in at least one imputation):")
    print(cb[cb.n_concordant > 0][["span", "duplicates", "patient_id", "n_imputations", "n_significant", "n_above_threshold", "n_concordant", "mean_observed_z"]].round(3).to_string(index=False))
    print(f"\nScope guard: rerun downstream for {rerun['rerun']}; "
          f"differ but empty: {rerun['empty']}; case B families flagged: {rerun['case_b_flagged']}")
    print(f"\nWrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
