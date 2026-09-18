#!/usr/bin/env python
"""Item 21: how many of the 132 configurations are distinct analyses, and does it matter?

Reads a world's observed per-config Z table and writes to results/v2/diagnostics/:

  effective_configurations_<world>.csv     per patient: distinct Z values, mean/median under
                                           all 132 vs the structurally effective set (60 under
                                           Spearman, 132 under Pearson), threshold flags
  config_equivalence_classes_<world>.csv   each structural class, its members, and the largest
                                           within-class spread of Z on this data (0 = confirmed duplicate)

Usage:  python scripts/effective_configurations.py [--world span=union__case=A]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aamos_concordance import write_sidecar  # noqa: E402
from aamos_concordance.configs import N_SPEARMAN_EFFECTIVE  # noqa: E402
from aamos_concordance.diagnostics import effective_configurations, equivalence_class_check  # noqa: E402
from aamos_concordance.summary import assessed_patients  # noqa: E402
from aamos_concordance.worlds import BASELINE, PER_CONFIG_Z, SUMMARY_CSV, as_world, diagnostics_dir, world_dir  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default=str(BASELINE))
    args = parser.parse_args(argv)
    world = as_world(args.world)

    wdir = world_dir(world)
    per_config = pd.read_csv(wdir / PER_CONFIG_Z)
    patients = assessed_patients(pd.read_csv(wdir / SUMMARY_CSV))
    out_dir = diagnostics_dir()

    classes = pd.concat([
        equivalence_class_check(per_config, "spearman").assign(correlation_type="spearman"),
    ], ignore_index=True)
    classes_path = out_dir / f"config_equivalence_classes_{world}.csv"
    classes.to_csv(classes_path, index=False)
    write_sidecar(classes_path, config={"world": str(world)}, extra={"script": "effective_configurations.py"})

    table = effective_configurations(per_config, patients)
    table_path = out_dir / f"effective_configurations_{world}.csv"
    table.to_csv(table_path, index=False)
    write_sidecar(table_path, config={"world": str(world)}, extra={"script": "effective_configurations.py"})

    print(f"World {world}: {len(patients)} assessed patients")
    print(f"Structural classes under Spearman: {len(classes)} (= {N_SPEARMAN_EFFECTIVE} effective configurations)")
    print(f"Largest within-class spread of Spearman Z on this data: {classes['max_within_class_spread'].max():.2e}")
    cols = ["patient_id", "spearman_n_valid_all", "spearman_n_distinct_z", "spearman_mean_all", "spearman_mean_effective",
            "spearman_median_all", "spearman_median_effective", "spearman_mean_all_ge_threshold", "spearman_mean_effective_ge_threshold"]
    print(table[cols].round(3).to_string(index=False))
    print(f"\nWrote {classes_path}\n      {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
