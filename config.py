"""
Central configuration for concordant user group definitions.

The concordant sets are **derived, not typed**. They are read from
``results/v2/groups/concordant_sets.json``, which is generated from each
world's ``summary.csv`` (see ``aamos_concordance.summary``). Scripts import
``GROUPS`` from here and accept ``--group`` and ``--world`` CLI arguments;
``set_active_world()`` re-points ``GROUPS`` and ``ASSESSED_PATIENTS`` at the
chosen world in place so existing references keep working.

Group names:
    concordant          mean Fisher's Z >= 0.5 AND Bonferroni-significant (primary analysis)
    median_concordant   median Fisher's Z >= 0.5 AND Bonferroni-significant (sensitivity analysis)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from aamos_concordance.summary import assessed_patients as _assessed_from_summary
from aamos_concordance.worlds import (
    BASELINE,
    SUMMARY_CSV,
    WorldSpec,
    as_world,
    comparisons_dir,
    load_concordant_sets,
    world_dir,
)

# All 22 patients in the AAMOS-00 dataset
ALL_PATIENTS = [
    113, 190, 217, 278, 294, 328, 343, 398, 447, 454,
    473, 514, 562, 625, 701, 702, 748, 764, 808, 867, 917, 939
]

# Assessed patients who provided end-of-study feedback (from the feedback CSV, not derivable here)
FEEDBACK_PATIENTS = [113, 190, 294, 343, 473, 514, 701, 702, 939]

# Concordance threshold used across analyses
CONCORDANCE_Z_THRESHOLD = 0.45  # Rounds to 0.5 at 1dp; corresponds to rho ~ 0.46

_GROUP_META = {
    "concordant": {
        "measure": "mean",
        "label": "Concordant",
        "description": "Mean-concordant users (primary analysis)",
    },
    "median_concordant": {
        "measure": "median",
        "label": "Median-Concordant",
        "description": "Median-concordant users (sensitivity analysis)",
    },
}

ACTIVE_WORLD: WorldSpec = BASELINE
GROUPS: dict = {}
# Patients with a valid observed statistic and a null distribution in the active world (15 in v1)
ASSESSED_PATIENTS: list = []


def _load_world_groups(world: WorldSpec) -> tuple[dict, list]:
    sets = load_concordant_sets().get(world.world_id)
    if sets is None:
        raise RuntimeError(
            f"No concordant sets recorded for world {world}. Build the world first "
            f"(scripts/build_baseline_world.py for the baseline, or aggregate.py --world {world})."
        )
    groups = {}
    for name, meta in _GROUP_META.items():
        groups[name] = {
            "patients": list(sets[meta["measure"]]),
            "label": meta["label"],
            "description": meta["description"],
            "measure": meta["measure"],
            "world": world.world_id,
        }
    summary_path = world_dir(world) / SUMMARY_CSV
    assessed = _assessed_from_summary(pd.read_csv(summary_path)) if summary_path.exists() else []
    return groups, assessed


def set_active_world(world: "WorldSpec | str | None" = None) -> WorldSpec:
    """Point GROUPS / ASSESSED_PATIENTS at ``world`` (in place) and return it."""
    global ACTIVE_WORLD
    w = as_world(world)
    groups, assessed = _load_world_groups(w)
    GROUPS.clear()
    GROUPS.update(groups)
    ASSESSED_PATIENTS[:] = assessed
    ACTIVE_WORLD = w
    return w


set_active_world(BASELINE)


def get_group_config(group_name: str) -> dict:
    """Get group configuration by name, with validation."""
    if group_name not in GROUPS:
        valid = ", ".join(GROUPS.keys())
        raise ValueError(f"Unknown group '{group_name}'. Valid groups: {valid}")
    return GROUPS[group_name]


def get_remaining_patients(group_name: str) -> list:
    """Get the remaining assessed patients (those NOT in the concordant group)."""
    concordant = set(get_group_config(group_name)["patients"])
    return [p for p in ASSESSED_PATIENTS if p not in concordant]


def get_output_dir(group_name: str, analysis_type: str, world: "WorldSpec | str | None" = None) -> Path:
    """Output directory for a group/analysis: results/v2/comparisons/<world>/<group>/<analysis>/."""
    return comparisons_dir(world if world is not None else ACTIVE_WORLD, group_name, analysis_type)


def add_world_argument(parser) -> None:
    """Add the standard ``--world`` option to an argparse parser."""
    parser.add_argument(
        "--world", default=str(BASELINE),
        help="world id whose concordant sets and output folder to use (default: baseline span=Q__case=A)",
    )


def record_run_provenance(output_dir: Path, *, group_name: str, script: str, config=None, data=None) -> Path:
    """Write ``<output_dir>/RUN.provenance.json`` describing this run.

    ``data`` should be the loader's ``data_provenance`` dict when the script
    read the raw CSVs through a loader; when omitted the raw files are
    located and hashed here so the record still names them.
    """
    from aamos_concordance import find_data_dir, verify_manifest, write_sidecar
    from aamos_concordance.data import DataNotFoundError

    if data is None:
        try:
            d = find_data_dir()
            data = {"data_dir": str(d), "sha256": verify_manifest(d), "manifest_verified": True}
        except DataNotFoundError:
            data = None
    extra = {
        "script": script,
        "world": ACTIVE_WORLD.world_id,
        "group": group_name,
        "group_patients": get_group_config(group_name)["patients"],
        "assessed_patients": list(ASSESSED_PATIENTS),
    }
    return write_sidecar(output_dir / "RUN", config=config, data=data, extra=extra)
