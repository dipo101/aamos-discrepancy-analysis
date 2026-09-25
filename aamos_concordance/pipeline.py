"""Building a world: from raw frames to per-config observed table, null and summary.

A :class:`~aamos_concordance.worlds.WorldSpec` is applied to the raw frames
in three places:

0. **Duplicates**: :func:`world_frames` first applies the world's reading of
   exact duplicate device rows (:mod:`aamos_concordance.duplicates`) and
   drops exact duplicate questionnaire rows. The observed table and the null
   both go through :func:`world_frames`, so they always see the same rows.
1. **Span** (item 8): :func:`world_frames` then trims both raw frames of a patient
   to the world's span. A patient whose span cannot be
   formed (no device records under ``D``, non-overlapping periods under
   ``intersection``) has no rows in that world and drops out of it.
2. **Absence case** (items 9 and 10): applied inside the engine's
   precomputation, after the window join and before categorisation. See
   :mod:`aamos_concordance.engine`.

:func:`build_world` runs the whole thing for a world: observed per-config
table (engine, unshuffled), null (engine, exact enumeration for small n,
sampled otherwise), summary under every spec, threshold sweep, concordant
sets, index and config records. It is what ``scripts/build_world.py`` calls
and what the tests exercise on synthetic data.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dataclasses import asdict

from . import definitions, engine
from .batch import null_from_per_config
from .configs import OBSERVED_CONFIG_KEY, generate_param_combinations
from .data import RawData
from .duplicates import apply_duplicate_reading
from .provenance import git_state, write_sidecar
from .spans import Span, apply_span
from .summary import (
    build_world_summary,
    derive_concordant_sets,
    null_sources_for,
    observed_table,
    threshold_sweep,
)
from .worlds import (
    REPO_ROOT,
    NULL_PARQUET,
    NULL_PER_CONFIG_PARQUET,
    OBSERVED_CSV,
    PER_CONFIG_Z,
    SUMMARY_CSV,
    THRESHOLD_SWEEP_CSV,
    WorldSpec,
    as_world,
    read_world_config,
    register_world,
    world_dir,
    write_concordant_sets,
    write_world_config,
)

DEFAULT_PATIENTS = [113, 190, 294, 328, 343, 398, 447, 454, 473, 514, 625, 701, 702, 917, 939]
# Bumped when a change to the definitions or the null layout invalidates built worlds.
# 2: v2 definitions, n_rows in the per-config null, duplicate reading applied to observed and null alike.
BUILD_VERSION = 2


def world_frames(
    world: "WorldSpec | str",
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    patient_id: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Span]]:
    """One patient's raw frames under the world's duplicate reading and span (both frames trimmed)."""
    w = as_world(world)
    q = questionnaire_df[questionnaire_df.user_key == patient_id].drop_duplicates()
    inh = apply_duplicate_reading(inhaler_df[inhaler_df.user_key == patient_id], w.duplicates)
    return apply_span(q, inh, w.span)


def observed_per_config_table(
    world: "WorldSpec | str",
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    patients: Sequence[int],
) -> pd.DataFrame:
    """Observed multiverse for a world in the per_config_z.csv layout (one row per patient x config).

    Columns: ``user_key``, the five configuration parameters (with
    ``filter_out_zero_usage_entries`` as the filter column name, as the v1
    table has), ``spearman_z``, ``pearson_z``, ``sample_size``,
    ``config_idx``. Z is NaN for a configuration that is invalid (fewer than
    three rows after the filter, a constant column, or a perfect
    correlation).
    """
    w = as_world(world)
    combos = generate_param_combinations()
    cfg_frame = pd.DataFrame(combos).rename(columns={"filter_out_zero_usage": "filter_out_zero_usage_entries"})
    cfg_frame["config_idx"] = np.arange(len(combos))
    parts = []
    for pid in patients:
        q, inh, sp = world_frames(w, questionnaire_df, inhaler_df, pid)
        if sp is None or len(q) == 0:
            continue
        tables = engine.precompute_patient(pid, q, inh, combos, world=w)
        P = np.arange(tables.n_rows, dtype=np.intp)[None, :]
        obs = engine.run_patient(tables, P, [-1]).drop(columns="permutation_idx")
        obs["sample_size"] = [tables.observed_row_count(ci) for ci in obs["config_idx"]]
        obs = obs.rename(columns={"patient_id": "user_key"}).merge(cfg_frame, on="config_idx")
        parts.append(obs)
    if not parts:
        return pd.DataFrame(columns=["user_key", *OBSERVED_CONFIG_KEY, "spearman_z", "pearson_z", "sample_size", "config_idx"])
    out = pd.concat(parts, ignore_index=True)
    return out[["user_key", *OBSERVED_CONFIG_KEY, "spearman_z", "pearson_z", "sample_size", "config_idx"]]


def generate_world_null(
    world: "WorldSpec | str",
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    patients: Sequence[int],
    *,
    n_perm: int = 10000,
    seed: int = 42,
    exact_max: int = 20000,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """Per-config null for every patient with rows in the world. Returns (table, per-patient modes)."""
    w = as_world(world)
    combos = generate_param_combinations()
    parts, modes = [], {}
    for pid in patients:
        q, inh, sp = world_frames(w, questionnaire_df, inhaler_df, pid)
        if sp is None or len(q) == 0:
            modes[pid] = {"mode": "excluded", "reason": "span not formable" if sp is None else "no rows"}
            if progress:
                progress(f"{pid}: excluded ({modes[pid]['reason']})")
            continue
        t0 = time.time()
        n_exact = engine.n_distinct_permutations(q["daily_relief_inhaler"].to_numpy(float))
        if exact_max and n_exact <= exact_max:
            pc = engine.run_patient_exact(pid, q, inh, param_combinations=combos, limit=exact_max, world=w)
            modes[pid] = {"mode": "exact", "n_permutations": int(n_exact)}
        else:
            pc = engine.run_patient_sampled(pid, q, inh, 0, n_perm, seed, param_combinations=combos, world=w)
            modes[pid] = {"mode": "sampled", "n_permutations": n_perm, "seed": seed}
        modes[pid]["n_rows"] = int(len(q))
        parts.append(pc)
        if progress:
            progress(f"{pid}: {len(q)} rows, {modes[pid]['mode']} {modes[pid]['n_permutations']} permutations in {time.time() - t0:.1f}s")
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=engine.PER_CONFIG_COLUMNS)
    return table, modes


def build_world(
    world: "WorldSpec | str",
    raw: RawData,
    patients: Sequence[int] = DEFAULT_PATIENTS,
    *,
    n_perm: int = 10000,
    seed: int = 42,
    exact_max: int = 20000,
    script: str = "build_world.py",
    progress: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Build every artifact of a world into results/v2/worlds/<id>/ and return the concordant sets."""
    w = as_world(world)
    wdir = world_dir(w, create=True)
    q_all, i_all = raw.questionnaire, raw.inhaler  # duplicates are handled per world in world_frames
    config = {"world": w.to_dict(), "n_perm": n_perm, "seed": seed, "exact_max": exact_max, "patients": list(patients)}
    prov = raw.provenance()

    observed = observed_per_config_table(w, q_all, i_all, patients)
    obs_path = wdir / PER_CONFIG_Z
    observed.to_csv(obs_path, index=False)
    write_sidecar(obs_path, config=config, data=prov, extra={"script": script, "engine": "vectorised"})

    per_config_null, modes = generate_world_null(w, q_all, i_all, patients, n_perm=n_perm, seed=seed,
                                                 exact_max=exact_max, progress=progress)
    config["per_patient"] = modes
    pc_path = wdir / NULL_PER_CONFIG_PARQUET
    per_config_null.to_parquet(pc_path, index=False)
    write_sidecar(pc_path, config=config, data=prov, extra={"script": script, "engine": "vectorised"})
    sets = summarize_world(w, observed, per_config_null, modes, config, prov, script=script, data_hashes=raw.hashes)
    write_world_config(w, {"build_version": BUILD_VERSION, "definitions": asdict(definitions.ACTIVE)})
    return sets


def is_current_build(world: "WorldSpec | str") -> bool:
    """Whether a world was built with the current definitions and null layout (resummarising keeps the stamp)."""
    return read_world_config(world).get("build_version") == BUILD_VERSION


def summarize_world(world, observed, per_config_null, modes, config, prov, *, script, data_hashes=None,
                    built_by: Optional[str] = None) -> Dict:
    """Write summary.csv, observed.csv, threshold_sweep.csv, sets, config and index for a world."""
    w = as_world(world)
    wdir = world_dir(w, create=True)
    null_path = wdir / NULL_PARQUET
    legacy = null_from_per_config(per_config_null, correlation_type="spearman")
    legacy.to_parquet(null_path, index=False)
    write_sidecar(null_path, config=config, data=prov, extra={"script": script, "engine": "vectorised"})
    exact_patients = [int(pid) for pid, m in modes.items() if m.get("mode") == "exact"]
    summary = build_world_summary(
        observed, null_sources=null_sources_for(legacy_null=legacy, per_config_null=per_config_null, absence_case=w.absence_case),
        absence_case=w.absence_case, exact_patients=exact_patients)
    summary_path = wdir / SUMMARY_CSV
    summary.to_csv(summary_path, index=False)
    write_sidecar(summary_path, config=config, data=prov, extra={"script": script})
    observed_table(observed, absence_case=w.absence_case).to_csv(wdir / OBSERVED_CSV, index=False)
    write_sidecar(wdir / OBSERVED_CSV, config=config, data=prov, extra={"script": script})
    threshold_sweep(summary).to_csv(wdir / THRESHOLD_SWEEP_CSV, index=False)
    write_sidecar(wdir / THRESHOLD_SWEEP_CSV, config=config, data=prov, extra={"script": script})

    sets = derive_concordant_sets(summary)
    write_concordant_sets(w, sets)
    write_world_config(w, {"null_source": str(null_path.relative_to(REPO_ROOT)), "null_engine": "vectorised",
                           "null_modes": modes, "built_by": built_by or script})
    register_world(w, git_commit=git_state()["git_commit"], data_hashes=data_hashes or {}, note=script)
    return sets


def resummarize_world(world, raw: RawData, *, script: str = "resummarize") -> Dict:
    """Re-derive every summary artifact of an already-built world from its stored observed table and per-config null."""
    import json
    w = as_world(world)
    wdir = world_dir(w)
    observed = pd.read_csv(wdir / PER_CONFIG_Z)
    per_config_null = pd.read_parquet(wdir / NULL_PER_CONFIG_PARQUET)
    cfg = json.loads((wdir / "config.json").read_text())
    modes = {int(k): v for k, v in cfg.get("null_modes", {}).items()}
    config = {"world": w.to_dict(), "resummarized_from": cfg.get("built_by"), "per_patient": modes}
    return summarize_world(w, observed, per_config_null, modes, config, raw.provenance(), script=script,
                           data_hashes=raw.hashes, built_by=cfg.get("built_by"))
