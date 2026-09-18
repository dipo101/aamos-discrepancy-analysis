#!/usr/bin/env python
"""Re-derive summary.csv (and everything downstream of it) for built worlds from their stored
observed table and per-config null. No permutation is regenerated.

    python scripts/resummarize_worlds.py                 # every world in results/v2/worlds.csv
    python scripts/resummarize_worlds.py --world span=union__case=A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import load_raw  # noqa: E402
from aamos_concordance.pipeline import resummarize_world  # noqa: E402
from aamos_concordance.worlds import NULL_PER_CONFIG_PARQUET, list_worlds, world_dir  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", nargs="*", default=None)
    args = parser.parse_args(argv)
    raw = load_raw()
    for wid in (args.world or list_worlds()):
        if not (world_dir(wid) / NULL_PER_CONFIG_PARQUET).exists():
            print(f"{wid}: no per-config null on disk, skipped")
            continue
        sets = resummarize_world(wid, raw, script="resummarize_worlds.py")
        print(f"{wid}: primary {sets.get('effective/spearman/mean')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
