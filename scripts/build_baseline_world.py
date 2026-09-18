#!/usr/bin/env python
"""Build the baseline world (span=Q__case=A) from the frozen v1 artifacts.

The published analysis did not have the v2 layout. This script creates
``results/v2/worlds/span=Q__case=A/`` from ``results/v1/`` so that:

* the per-config Z table is the v1 table verbatim;
* the null distribution is the v1 parquet (referenced, not copied: it is
  9 MB and already committed under v1);
* ``summary.csv`` is rebuilt from those two with ``aamos_concordance.summary``;
* ``results/v2/groups/concordant_sets.json`` is generated from the summary.

``tests/test_baseline_world.py`` checks that the generated sets equal the
sets the manuscript reports, and that the summary reproduces v1's
``permutation_test_results.csv`` tables.

Usage:  python scripts/build_baseline_world.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import write_sidecar  # noqa: E402
from aamos_concordance.data import DEFAULT_DATA_DIR, MANIFEST_NAME, read_manifest  # noqa: E402
from aamos_concordance.provenance import git_state  # noqa: E402
from aamos_concordance.summary import build_world_summary, derive_concordant_sets  # noqa: E402
from aamos_concordance.worlds import (  # noqa: E402
    BASELINE, PER_CONFIG_Z, SUMMARY_CSV, V1_DIR, register_world, world_dir,
    write_concordant_sets, write_world_config,
)

V1_PER_CONFIG = V1_DIR / "per_patient_analysis" / "per_patient_correlation_results.csv"
V1_NULL = V1_DIR / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet"


def main() -> int:
    wdir = world_dir(BASELINE, create=True)

    per_config_path = wdir / PER_CONFIG_Z
    shutil.copyfile(V1_PER_CONFIG, per_config_path)
    write_sidecar(per_config_path, extra={"script": "build_baseline_world.py", "source": str(V1_PER_CONFIG.relative_to(REPO_ROOT))})

    per_config = pd.read_csv(per_config_path)
    null = pd.read_parquet(V1_NULL)
    summary = build_world_summary(per_config, null)
    summary_path = wdir / SUMMARY_CSV
    summary.to_csv(summary_path, index=False)
    write_sidecar(summary_path, extra={"script": "build_baseline_world.py",
                                       "inputs": {"per_config_z": str(per_config_path.relative_to(REPO_ROOT)),
                                                  "null": str(V1_NULL.relative_to(REPO_ROOT))}})

    write_world_config(BASELINE, {
        "null_source": str(V1_NULL.relative_to(REPO_ROOT)),
        "built_by": "scripts/build_baseline_world.py",
        "note": "Baseline world reconstructed from the frozen v1 artifacts; identical to the published analysis.",
    })

    sets = derive_concordant_sets(summary)
    write_concordant_sets(BASELINE, sets)

    manifest = DEFAULT_DATA_DIR / MANIFEST_NAME
    hashes = read_manifest(manifest) if manifest.exists() else {}
    register_world(BASELINE, git_commit=git_state()["git_commit"], data_hashes=hashes, note="from v1 artifacts")

    print(f"Built {wdir}")
    print(f"  concordant sets: {sets}")
    print(f"  assessed patients: {sorted(summary['patient_id'].unique().tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
