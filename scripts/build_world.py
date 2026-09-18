#!/usr/bin/env python
"""Build a world end to end, locally: observed multiverse, null, summary, sets.

    python scripts/build_world.py --world span=D__case=A
    python scripts/build_world.py --world span=union__case=A --patients 917 454 --n-perm 2000

Writes results/v2/worlds/<world>/ (per_config_z.csv, null.parquet and
null_per_config.parquet [git-ignored], summary.csv, observed.csv,
threshold_sweep.csv, config.json) and updates results/v2/worlds.csv and
results/v2/groups/concordant_sets.json. See aamos_concordance.pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import load_raw  # noqa: E402
from aamos_concordance.pipeline import DEFAULT_PATIENTS, build_world  # noqa: E402
from aamos_concordance.worlds import BASELINE, as_world, world_dir  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default=str(BASELINE))
    parser.add_argument("--patients", nargs="*", type=int, default=DEFAULT_PATIENTS)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-max", type=int, default=20000)
    args = parser.parse_args(argv)
    world = as_world(args.world)

    t0 = time.time()
    raw = load_raw()
    sets = build_world(world, raw, args.patients, n_perm=args.n_perm, seed=args.seed, exact_max=args.exact_max,
                       script="build_world.py", progress=lambda m: print(m, flush=True))
    print(f"\nBuilt {world_dir(world)} in {time.time() - t0:.1f}s")
    for k, v in sets.items():
        print(f"  {k:28s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
