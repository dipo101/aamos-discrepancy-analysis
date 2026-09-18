#!/usr/bin/env python
"""Generate a world's null distribution locally with the vectorised engine.

Writes into results/v2/worlds/<world>/:

  null_per_config.parquet   patient x permutation x config x {spearman_z, pearson_z}  (git-ignored)
  null.parquet              per-permutation summaries over all 132 Spearman configs (v1 layout)

and nothing else: summarising into summary.csv / concordant sets is the
aggregator's job (permutation_cloudrun/aggregator/aggregate.py --world ...
--local) so that one code path produces those files whether the null came
from Cloud Run or from here.

Permutation source per patient:
  * exact enumeration of every distinct arrangement of the self-report when
    that count is <= --exact-max (default 20,000), otherwise
  * --n-perm sampled permutations with the kernel's shuffle (seed + index),
    so sampled results are comparable with the published null.

Only the baseline world is supported until span/absence-case handling lands;
any other world id is refused.

Usage:
  python scripts/run_null_local.py                       # baseline, 15 assessed patients
  python scripts/run_null_local.py --patients 917 454    # subset
  python scripts/run_null_local.py --n-perm 2000 --exact-max 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import load_raw, null_from_per_config, write_sidecar  # noqa: E402
from aamos_concordance.engine import n_distinct_permutations, run_patient_exact, run_patient_sampled  # noqa: E402
from aamos_concordance.provenance import git_state  # noqa: E402
from aamos_concordance.worlds import (  # noqa: E402
    BASELINE, NULL_PARQUET, NULL_PER_CONFIG_PARQUET, as_world, register_world, world_dir, write_world_config,
)

DEFAULT_PATIENTS = [113, 190, 294, 328, 343, 398, 447, 454, 473, 514, 625, 701, 702, 917, 939]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default=str(BASELINE))
    parser.add_argument("--patients", nargs="*", type=int, default=DEFAULT_PATIENTS)
    parser.add_argument("--n-perm", type=int, default=10000, help="sampled permutations per patient (default 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exact-max", type=int, default=20000,
                        help="enumerate exactly when the number of distinct arrangements is <= this (0 disables)")
    args = parser.parse_args(argv)
    world = as_world(args.world)
    if world != BASELINE:
        raise SystemExit(f"world {world}: span/absence-case handling is not implemented yet; only the baseline can be generated.")

    raw = load_raw()
    q, inh = raw.questionnaire, raw.inhaler
    wdir = world_dir(world, create=True)

    parts, modes = [], {}
    t_all = time.time()
    for pid in args.patients:
        qq, ii = q[q.user_key == pid], inh[inh.user_key == pid]
        if len(qq) == 0:
            print(f"{pid}: no questionnaire rows, skipped")
            continue
        n_exact = n_distinct_permutations(qq["daily_relief_inhaler"].to_numpy(float))
        t0 = time.time()
        if args.exact_max and n_exact <= args.exact_max:
            pc = run_patient_exact(pid, qq, ii, limit=args.exact_max)
            modes[pid] = {"mode": "exact", "n_permutations": int(n_exact)}
        else:
            pc = run_patient_sampled(pid, qq, ii, 0, args.n_perm, args.seed)
            modes[pid] = {"mode": "sampled", "n_permutations": args.n_perm, "seed": args.seed}
        parts.append(pc)
        print(f"{pid}: {len(qq)} rows, {modes[pid]['mode']} {modes[pid]['n_permutations']} permutations in {time.time() - t0:.1f}s")

    per_config = pd.concat(parts, ignore_index=True)
    pc_path = wdir / NULL_PER_CONFIG_PARQUET
    per_config.to_parquet(pc_path, index=False)
    config = {"world": str(world), "n_perm": args.n_perm, "seed": args.seed, "exact_max": args.exact_max, "per_patient": modes}
    write_sidecar(pc_path, config=config, data=raw.provenance(), extra={"script": "run_null_local.py", "engine": "vectorised"})

    null = null_from_per_config(per_config, correlation_type="spearman")
    null_path = wdir / NULL_PARQUET
    null.to_parquet(null_path, index=False)
    write_sidecar(null_path, config=config, data=raw.provenance(), extra={"script": "run_null_local.py", "engine": "vectorised"})

    write_world_config(world, {"null_source": str(null_path.relative_to(REPO_ROOT)), "null_engine": "vectorised",
                               "null_modes": modes})
    register_world(world, git_commit=git_state()["git_commit"], data_hashes=raw.hashes, note="run_null_local.py")
    print(f"\nWrote {pc_path} ({len(per_config)} rows) and {null_path} in {time.time() - t_all:.1f}s total")
    print("Next: python permutation_cloudrun/aggregator/aggregate.py --world", world, "--local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
