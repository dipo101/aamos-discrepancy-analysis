#!/usr/bin/env python
"""Items 19 and 20: cell counts per span and the questionnaire non-response check.

Writes to results/v2/diagnostics/:
  cell_counts_<window>.csv      per patient x span: KT14 2x2 (device zero/positive x self-report
                                zero/positive), device days with no questionnaire, span bounds,
                                entries trimmed by the span
  nonresponse_check.csv         per patient: device puffs on response vs non-response days
                                (Mann-Whitney), clustering of non-response (runs test)
  nonresponse_summary.json      pooled counts across patients

Usage:  python scripts/missingness_diagnostics.py [--window calendar_same_day|rolling_24h]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import load_raw, write_sidecar  # noqa: E402
from aamos_concordance.missingness import (  # noqa: E402
    CALENDAR_SAME_DAY, ROLLING_24H, cell_counts, nonresponse_check, pooled_nonresponse_summary,
)
from aamos_concordance.worlds import diagnostics_dir  # noqa: E402

ASSESSED = [113, 190, 294, 328, 343, 398, 447, 454, 473, 514, 625, 701, 702, 917, 939]
WINDOWS = {"calendar_same_day": CALENDAR_SAME_DAY, "rolling_24h": ROLLING_24H}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", choices=list(WINDOWS), default="calendar_same_day")
    parser.add_argument("--patients", nargs="*", type=int, default=ASSESSED)
    args = parser.parse_args(argv)

    raw = load_raw(drop_duplicates=True)
    out = diagnostics_dir()

    cells = cell_counts(raw.questionnaire, raw.inhaler, args.patients, window=WINDOWS[args.window])
    cells_path = out / f"cell_counts_{args.window}.csv"
    cells.to_csv(cells_path, index=False)
    write_sidecar(cells_path, config={"window": args.window, "patients": args.patients}, data=raw.provenance(),
                  extra={"script": "missingness_diagnostics.py"})

    check = nonresponse_check(raw.questionnaire, raw.inhaler, args.patients)
    check_path = out / "nonresponse_check.csv"
    check.to_csv(check_path, index=False)
    write_sidecar(check_path, config={"patients": args.patients}, data=raw.provenance(), extra={"script": "missingness_diagnostics.py"})

    pooled = pooled_nonresponse_summary(check)
    (out / "nonresponse_summary.json").write_text(json.dumps(pooled, indent=2) + "\n")

    pd_opts = {"index": False}
    print(f"Cell counts ({args.window}), span=Q:")
    q = cells[cells.span == "Q"][["patient_id", "n_rows", "rec_pos_q_pos", "rec_pos_q_zero", "rec_zero_q_pos", "rec_zero_q_zero",
                                  "device_days_without_questionnaire", "device_puffs_on_days_without_questionnaire"]]
    print(q.to_string(**pd_opts))
    print("\nRows per span (how much each span changes the data):")
    print(cells.pivot(index="patient_id", columns="span", values="n_rows")[["Q", "D", "union", "intersection"]].to_string())
    print("\nNon-response check:")
    print(check[["patient_id", "span_days", "response_days", "nonresponse_days", "mean_puffs_response_days",
                 "mean_puffs_nonresponse_days", "mannwhitney_p", "n_runs", "expected_runs", "runs_p"]].round(3).to_string(**pd_opts))
    print("\nPooled:", json.dumps(pooled))
    print(f"\nWrote {cells_path}\n      {check_path}\n      {out / 'nonresponse_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
