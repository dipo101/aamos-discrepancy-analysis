"""
Central configuration for concordant user group definitions.

The concordant sets are **derived, not typed**. They are read from
``results/v2/groups/concordant_sets.json``, which is generated from each
world's ``summary.csv`` (see ``aamos_concordance.summary``) and keyed by
summary spec (``config_set/correlation_type/measure``). Scripts import
``GROUPS`` from here and accept ``--group``, ``--world`` and ``--summary``
CLI arguments; ``set_active_world()`` re-points ``GROUPS`` and
``ASSESSED_PATIENTS`` in place so existing references keep working.

Group names (for the active spec's configuration set and correlation type):
    concordant          mean Fisher's Z >= 0.5 AND Bonferroni-significant (primary analysis)
    median_concordant   median Fisher's Z >= 0.5 AND Bonferroni-significant (sensitivity analysis)

The primary spec is ``aamos_concordance.summary.PRIMARY`` (effective/spearman).
If a world has no null for it yet (the v1-derived baseline), loading falls
back to the v1 spec (all/spearman) with a warning, and ``ACTIVE_SPEC`` says so.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Optional

import pandas as pd

import warnings

from aamos_concordance.summary import PRIMARY, V1_SPEC, SummarySpec, select
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
ACTIVE_SPEC: SummarySpec = PRIMARY  # config_set/correlation_type actually in use; measure varies per group
GROUPS: dict = {}
# Patients with a valid observed statistic and a null distribution in the active world (15 in v1)
ASSESSED_PATIENTS: list = []


def _resolve_spec(sets: dict, world: WorldSpec, spec: Optional[SummarySpec]) -> SummarySpec:
    """The spec whose sets will be used, falling back from PRIMARY to V1_SPEC with a warning."""
    candidates = [spec] if spec is not None else [PRIMARY, V1_SPEC]
    for cand in candidates:
        if all(replace(cand, measure=m["measure"]).key in sets for m in _GROUP_META.values()):
            if spec is None and cand != PRIMARY:
                warnings.warn(
                    f"World {world}: no null distribution for the primary summary spec {PRIMARY.null_key}; "
                    f"using {cand.null_key} (the v1 definition). Rerun the world's permutations with the "
                    "per-config worker to enable the primary spec.",
                    stacklevel=3,
                )
            return cand
    available = sorted(sets)
    raise RuntimeError(
        f"World {world} has no concordant sets for {[c.key for c in candidates]}. Available: {available}. "
        f"Build the world first (scripts/build_world.py --world {world})."
    )


def _load_world_groups(world: WorldSpec, spec: Optional[SummarySpec]) -> tuple[dict, list, SummarySpec]:
    sets = load_concordant_sets().get(world.world_id)
    if sets is None:
        raise RuntimeError(
            f"No concordant sets recorded for world {world}. Build the world first "
            f"(scripts/build_world.py --world {world})."
        )
    used = _resolve_spec(sets, world, spec)
    groups = {}
    for name, meta in _GROUP_META.items():
        key = replace(used, measure=meta["measure"]).key
        groups[name] = {
            "patients": list(sets[key]),
            "label": meta["label"],
            "description": meta["description"],
            "measure": meta["measure"],
            "summary_spec": key,
            "world": world.world_id,
        }
    summary_path = world_dir(world) / SUMMARY_CSV
    assessed = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        in_spec = select(summary, used)  # an exclusion spec assesses fewer patients
        assessed = _assessed_from_summary(in_spec if not in_spec.empty else summary)
    return groups, assessed, used


def set_active_world(world: "WorldSpec | str | None" = None, summary: "SummarySpec | str | None" = None) -> WorldSpec:
    """Point GROUPS / ASSESSED_PATIENTS at ``world`` under ``summary`` (in place) and return the world.

    ``summary`` is a SummarySpec or ``"config_set/correlation_type[/option=value...]"``
    (measure is ignored; each group uses its own). ``None`` means the primary spec, with a
    warned fallback to the v1 spec when the primary has no null in that world.
    """
    global ACTIVE_WORLD, ACTIVE_SPEC
    w = as_world(world)
    spec = None
    if summary is not None:
        if isinstance(summary, SummarySpec):
            spec = summary
        else:
            parts = summary.split("/")
            spec = SummarySpec.parse("/".join([*parts[:2], "mean", *parts[2:]]))
    groups, assessed, used = _load_world_groups(w, spec)
    GROUPS.clear()
    GROUPS.update(groups)
    ASSESSED_PATIENTS[:] = assessed
    ACTIVE_WORLD = w
    ACTIVE_SPEC = used
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
    """Add the standard ``--world`` and ``--summary`` options to an argparse parser."""
    parser.add_argument(
        "--world", default=str(BASELINE),
        help="world id whose concordant sets and output folder to use (default: baseline span=union__case=A)",
    )
    parser.add_argument(
        "--summary", default=None,
        help="summary spec 'config_set/correlation_type' for the concordant sets (default: primary "
             f"{PRIMARY.null_key}, falling back to {V1_SPEC.null_key} with a warning if the world has no null for it)",
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
        "summary_spec": ACTIVE_SPEC.null_key,
        "group": group_name,
        "group_patients": get_group_config(group_name)["patients"],
        "assessed_patients": list(ASSESSED_PATIENTS),
    }
    return write_sidecar(output_dir / "RUN", config=config, data=data, extra=extra)
