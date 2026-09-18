#!/usr/bin/env python
"""Item 19: cell counts per span.

Writes to results/v2/diagnostics/:
  cell_counts_<window>.csv      per patient x span: KT14 2x2 (device zero/positive x self-report
                                zero/positive), device days with no questionnaire, span bounds,
                                entries trimmed by the span

Usage:  python scripts/missingness_diagnostics.py [--window calendar_same_day|rolling_24h]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import load_raw, write_sidecar  # noqa: E402
from aamos_concordance.missingness import CALENDAR_SAME_DAY, ROLLING_24H, cell_counts  # noqa: E402
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

    pd_opts = {"index": False}
    print(f"Cell counts ({args.window}), span=Q:")
    q = cells[cells.span == "Q"][["patient_id", "n_rows", "rec_pos_q_pos", "rec_pos_q_zero", "rec_zero_q_pos", "rec_zero_q_zero",
                                  "device_days_without_questionnaire", "device_puffs_on_days_without_questionnaire"]]
    print(q.to_string(**pd_opts))
    print("\nRows per span (how much each span changes the data):")
    print(cells.pivot(index="patient_id", columns="span", values="n_rows")[["Q", "D", "union", "intersection"]].to_string())
    print(f"\nWrote {cells_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
