#!/usr/bin/env python
"""Build a world end to end, locally: observed multiverse, null, summary, sets.

    python scripts/build_world.py --world span=D__case=A
    python scripts/build_world.py --world span=union__case=A --patients 917 454 --n-perm 2000
    python scripts/build_world.py --all --skip-current          # every world of the grid, resumable
    python scripts/build_world.py --all --skip-current --shard 0/4   # one of 4 parallel processes

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
from aamos_concordance.pipeline import DEFAULT_PATIENTS, build_world, is_current_build  # noqa: E402
from aamos_concordance.worlds import BASELINE, all_world_specs, as_world, world_dir  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", nargs="*", default=None, help=f"world id(s) (default: {BASELINE})")
    parser.add_argument("--all", action="store_true", help="every world of the grid (span x case x imputation x duplicates)")
    parser.add_argument("--skip-current", action="store_true", help="skip worlds already built with the current build version")
    parser.add_argument("--shard", default=None, help="i/n: build every n-th world starting at i (parallel processes)")
    parser.add_argument("--patients", nargs="*", type=int, default=DEFAULT_PATIENTS)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-max", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.all and args.world:
        parser.error("--all and --world are exclusive")
    worlds = all_world_specs() if args.all else [as_world(w) for w in (args.world or [str(BASELINE)])]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        worlds = worlds[i::n]
    if args.skip_current:
        worlds = [w for w in worlds if not is_current_build(w)]

    raw = load_raw()
    for i, world in enumerate(worlds, 1):
        t0 = time.time()
        print(f"[{i}/{len(worlds)}] {world}", flush=True)
        sets = build_world(world, raw, args.patients, n_perm=args.n_perm, seed=args.seed, exact_max=args.exact_max,
                           script="build_world.py", progress=lambda m: print(f"  {m}", flush=True))
        print(f"Built {world_dir(world)} in {time.time() - t0:.1f}s")
        for k, v in sets.items():
            print(f"  {k:40s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
