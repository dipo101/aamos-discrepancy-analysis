"""From a world's per-config Z table and null parquet to its summary and concordant set.

This is the post-processing step of the pipeline, factored out of the
Cloud Run aggregator so that it runs identically for every world and can be
tested against the frozen v1 tables (``tests/test_summary.py``).

A summary is computed under a :class:`SummarySpec`: which configuration
set (``all`` = 132, ``effective`` = the structurally distinct ones, 60 for
Spearman and 132 for Pearson), which correlation type, and which statistic
(mean or median). :data:`PRIMARY` is ``effective / spearman / mean``; the
v1 publication used ``all / spearman / mean`` (:data:`V1_SPEC`). A world's
summary table carries every spec whose null distribution is available, so
the choice of primary is a one-line constant, not a rerun. The null for a
spec other than ``all/spearman`` requires the per-configuration null table
(``null_per_config.parquet``); the v1 null only has ``all/spearman``
summaries.

Definitions, unchanged from v1 apart from the configuration set:

* The **observed statistic** for a patient is the mean (or median) of the
  *finite* Fisher Z values across the spec's configurations. Non-finite Z
  (a rho of exactly ±1, or an undefined correlation) is excluded.
* The **permutation p-value** is two-tailed with the +1 correction,
  ``(#{|null| >= |observed|} + 1) / (n_perm + 1)``.
* **Bonferroni** multiplies by the number of patients with a valid observed
  statistic and a non-empty null, capped at 1.
* ``n_ties_at_observed`` counts null values within 1e-12 of |observed|.
  These arise when a permutation reproduces the observed arrangement (few
  distinct permutations for small-n patients). They are counted as extreme,
  but whether a tie registers as equal depends on the last bit of the
  observed statistic, so a p-value with ties > 0 is uncertain by
  ``n_ties / (n_perm + 1)``. The v2 plan's exact enumeration for small-n
  patients removes this.
* A patient is **concordant** when the observed statistic is at least the
  threshold (0.5 in v1) *and* the Bonferroni-corrected p-value is below 0.05.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .configs import CONFIG_SETS, attach_config_idx, config_indices_for
from .permutation import CORRELATION_TYPES
from .worlds import MEASURES

DEFAULT_THRESHOLD = 0.5
ALPHA = 0.05
DEFAULT_THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class SummarySpec:
    """Which configurations, which correlation, which statistic."""

    config_set: str = "effective"
    correlation_type: str = "spearman"
    measure: str = "mean"

    def __post_init__(self):
        if self.config_set not in CONFIG_SETS:
            raise ValueError(f"config_set must be one of {CONFIG_SETS}, got {self.config_set!r}")
        if self.correlation_type not in CORRELATION_TYPES:
            raise ValueError(f"correlation_type must be one of {CORRELATION_TYPES}, got {self.correlation_type!r}")
        if self.measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}, got {self.measure!r}")

    @property
    def null_key(self) -> str:
        """Identifies the null distribution a spec needs (statistic is applied per permutation)."""
        return f"{self.config_set}/{self.correlation_type}"

    @property
    def key(self) -> str:
        return f"{self.config_set}/{self.correlation_type}/{self.measure}"

    @classmethod
    def parse(cls, key: str) -> "SummarySpec":
        parts = key.split("/")
        if len(parts) != 3:
            raise ValueError(f"not a summary spec: {key!r} (expected config_set/correlation_type/measure)")
        return cls(*parts)

    @property
    def config_indices(self) -> List[int]:
        return config_indices_for(self.config_set, self.correlation_type)

    def __str__(self) -> str:
        return self.key


PRIMARY = SummarySpec("effective", "spearman", "mean")
V1_SPEC = SummarySpec("all", "spearman", "mean")
ALL_SPECS = [SummarySpec(c, t, m) for c in CONFIG_SETS for t in CORRELATION_TYPES for m in MEASURES]
# Null values this close to |observed| are reported as ties: whether they count
# as ">= observed" depends on the last bits of the observed statistic.
TIE_TOLERANCE = 1e-12


def observed_statistics(
    per_config: pd.DataFrame,
    z_column: str = "spearman_z",
    config_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Per-patient mean and median of finite Z, plus the count of finite configs.

    ``config_indices`` restricts to a configuration set (``None`` = all).

    Computed per patient with ``Series.mean()`` / ``Series.median()`` on the
    finite subset, exactly as v1's ``create_summary_statistics`` did. The
    summation order matters: a groupby aggregate can differ in the last ulp,
    and when a null distribution contains permutations that reproduce the
    observed arrangement exactly (small-n patients with few distinct
    permutations) the ``|null| >= |observed|`` tie count, and hence the
    p-value, flips on that ulp.
    """
    if config_indices is not None:
        per_config = attach_config_idx(per_config)
        per_config = per_config[per_config["config_idx"].isin(list(config_indices))]
    rows = []
    for pid, g in per_config.groupby("user_key", sort=True):
        valid = g.loc[np.isfinite(g[z_column]), z_column]
        rows.append({
            "patient_id": pid,
            "mean": valid.mean() if len(valid) > 0 else np.nan,
            "median": valid.median() if len(valid) > 0 else np.nan,
            "n_valid": int(len(valid)),
        })
    return pd.DataFrame(rows, columns=["patient_id", "mean", "median", "n_valid"])


def permutation_p_values(
    observed: pd.Series,
    null: pd.DataFrame,
    *,
    null_column: str,
    n_patients: Optional[int] = None,
) -> pd.DataFrame:
    """Two-tailed permutation p-values for ``observed`` (indexed by patient id).

    ``null`` needs ``patient_id`` and ``null_column`` (and optionally
    ``n_valid_configs``). ``n_patients`` defaults to the number of patients
    that have both a finite observed value and a non-empty null, which is
    what the v1 aggregator used for Bonferroni.
    """
    rows: List[Dict] = []
    grouped = {pid: g for pid, g in null.groupby("patient_id")}
    eligible = [pid for pid, z in observed.items() if pd.notna(z) and pid in grouped and grouped[pid][null_column].notna().any()]
    n_pat = n_patients if n_patients is not None else len(eligible)

    for pid in eligible:
        z = float(observed[pid])
        g = grouped[pid]
        null_values = g[null_column].dropna().to_numpy(float)
        n_perms = len(null_values)
        abs_null = np.abs(null_values)
        n_extreme = int(np.sum(abs_null >= abs(z)))
        n_ties = int(np.sum(np.abs(abs_null - abs(z)) <= TIE_TOLERANCE))
        p = (n_extreme + 1) / (n_perms + 1)
        p_bonf = min(p * n_pat, 1.0)
        rows.append({
            "patient_id": pid,
            "observed_z": z,
            "n_permutations": n_perms,
            "n_ties_at_observed": n_ties,
            "avg_valid_configs": float(g["n_valid_configs"].mean()) if "n_valid_configs" in g else np.nan,
            "p_value": p,
            "p_bonferroni": p_bonf,
            "significant_uncorrected": p < ALPHA,
            "significant_bonferroni": p_bonf < ALPHA,
            "null_mean": float(np.mean(null_values)),
            "null_std": float(np.std(null_values)),
            "null_q025": float(np.percentile(null_values, 2.5)),
            "null_q975": float(np.percentile(null_values, 97.5)),
            "n_patients_bonferroni": n_pat,
        })
    return pd.DataFrame(rows)


def null_sources_for(
    legacy_null: Optional[pd.DataFrame] = None,
    per_config_null: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Map each available ``config_set/correlation_type`` null key to its per-permutation null table.

    ``legacy_null`` is the v1-style summary parquet (``null_mean_z`` etc. over
    all 132 Spearman configurations) and provides only ``all/spearman``.
    ``per_config_null`` is the per-configuration table and provides every
    key. When both are given the per-config table wins for ``all/spearman``.
    """
    from .batch import null_from_per_config  # local import: batch imports permutation, not summary

    sources: Dict[str, pd.DataFrame] = {}
    if legacy_null is not None:
        sources[V1_SPEC.null_key] = legacy_null
    if per_config_null is not None:
        for c in CONFIG_SETS:
            for t in CORRELATION_TYPES:
                sources[f"{c}/{t}"] = null_from_per_config(
                    per_config_null, correlation_type=t, config_indices=config_indices_for(c, t))
    return sources


def build_world_summary(
    per_config: pd.DataFrame,
    null: Optional[pd.DataFrame] = None,
    *,
    null_sources: Optional[Dict[str, pd.DataFrame]] = None,
    specs: Iterable[SummarySpec] = ALL_SPECS,
    measures: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Long summary table with one row per (patient, spec) for every spec whose null is available.

    ``null`` is the legacy all/spearman null (kept for the v1 call signature);
    ``null_sources`` is the general form from :func:`null_sources_for`.
    Specs with no null source are skipped with a warning, never fabricated.
    """
    if null_sources is None:
        null_sources = null_sources_for(legacy_null=null)
    if measures is not None:  # legacy keyword: restrict statistics
        specs = [sp for sp in specs if sp.measure in set(measures)]

    parts = []
    skipped = []
    for spec in specs:
        if spec.null_key not in null_sources:
            skipped.append(spec.key)
            continue
        obs = observed_statistics(per_config, f"{spec.correlation_type}_z", spec.config_indices).set_index("patient_id")
        table = permutation_p_values(obs[spec.measure], null_sources[spec.null_key], null_column=f"null_{spec.measure}_z")
        if table.empty:
            continue
        table.insert(1, "config_set", spec.config_set)
        table.insert(2, "correlation_type", spec.correlation_type)
        table.insert(3, "measure", spec.measure)
        table["n_valid_configs_observed"] = table["patient_id"].map(obs["n_valid"]).astype(int)
        table["n_configs_in_set"] = len(spec.config_indices)
        parts.append(table)
    if skipped:
        warnings.warn(f"No null distribution for summary specs {skipped}; they are omitted.", stacklevel=2)
    if not parts:
        raise ValueError("No summary could be built: no spec has a null distribution.")
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["config_set", "correlation_type", "measure", "p_value", "patient_id"], kind="stable").reset_index(drop=True)


def observed_table(per_config: pd.DataFrame, specs: Iterable[SummarySpec] = ALL_SPECS) -> pd.DataFrame:
    """Observed statistics under every spec, with no p-values.

    Needs only the observed per-config table, so it is available for every
    world immediately, including specs whose null has not been generated.
    """
    parts = []
    for spec in specs:
        obs = observed_statistics(per_config, f"{spec.correlation_type}_z", spec.config_indices)
        obs = obs.rename(columns={spec.measure: "observed_z"})[["patient_id", "observed_z", "n_valid"]]
        obs.insert(1, "config_set", spec.config_set)
        obs.insert(2, "correlation_type", spec.correlation_type)
        obs.insert(3, "measure", spec.measure)
        obs["n_configs_in_set"] = len(spec.config_indices)
        parts.append(obs.rename(columns={"n_valid": "n_valid_configs_observed"}))
    return pd.concat(parts, ignore_index=True)


def available_specs(summary: pd.DataFrame) -> List[SummarySpec]:
    keys = summary[["config_set", "correlation_type", "measure"]].drop_duplicates()
    return [SummarySpec(*row) for row in keys.itertuples(index=False)]


def select(summary: pd.DataFrame, spec: SummarySpec) -> pd.DataFrame:
    """Rows of ``summary`` for one spec, or an empty frame if that spec is absent."""
    m = (summary["config_set"] == spec.config_set) & (summary["correlation_type"] == spec.correlation_type) & (summary["measure"] == spec.measure)
    return summary[m]


def concordant_set(summary: pd.DataFrame, spec: "SummarySpec | str", threshold: float = DEFAULT_THRESHOLD) -> List[int]:
    """Patients with observed Z >= threshold AND Bonferroni-significant under ``spec``."""
    sp = spec if isinstance(spec, SummarySpec) else SummarySpec.parse(spec)
    s = select(summary, sp)
    hit = s[(s["observed_z"] >= threshold) & s["significant_bonferroni"]]
    return sorted(int(p) for p in hit["patient_id"])


def derive_concordant_sets(summary: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, List[int]]:
    """``{spec.key: [patients]}`` for every spec present in ``summary``."""
    return {sp.key: concordant_set(summary, sp, threshold) for sp in available_specs(summary)}


def threshold_sweep(
    summary: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    specs: Optional[Iterable[SummarySpec]] = None,
) -> pd.DataFrame:
    """Concordant set under every (spec, threshold): one row each with the set and its size.

    Re-thresholding is free: significance does not depend on the threshold,
    only the Z cut does.
    """
    rows = []
    for sp in (specs if specs is not None else available_specs(summary)):
        for t in thresholds:
            members = concordant_set(summary, sp, t)
            rows.append({"config_set": sp.config_set, "correlation_type": sp.correlation_type, "measure": sp.measure,
                         "threshold": t, "n_concordant": len(members), "concordant": " ".join(map(str, members))})
    return pd.DataFrame(rows)


def assessed_patients(summary: pd.DataFrame) -> List[int]:
    """Patients with a valid observed statistic and a null distribution (the v1 'assessed' 15)."""
    return sorted(int(p) for p in summary["patient_id"].unique())


def to_v1_measure_table(summary: pd.DataFrame, measure: str, spec: Optional[SummarySpec] = None) -> pd.DataFrame:
    """Reshape one measure into the column layout of v1's ``permutation_test_results.csv``.

    ``spec`` defaults to the v1 spec's configuration set and correlation type
    (all/spearman) with ``measure`` substituted.
    """
    sp = spec if spec is not None else SummarySpec(V1_SPEC.config_set, V1_SPEC.correlation_type, measure)
    s = select(summary, sp).copy()
    measure = sp.measure
    s = s.rename(columns={
        "observed_z": f"observed_{measure}_z",
        "null_mean": f"null_{measure}_mean", "null_std": f"null_{measure}_std",
        "null_q025": f"null_{measure}_q025", "null_q975": f"null_{measure}_q975",
    })
    cols = ["patient_id", f"observed_{measure}_z", "n_permutations", "avg_valid_configs", "p_value", "p_bonferroni",
            "significant_uncorrected", "significant_bonferroni",
            f"null_{measure}_mean", f"null_{measure}_std", f"null_{measure}_q025", f"null_{measure}_q975"]
    return s[cols].sort_values(["p_value", "patient_id"], kind="stable").reset_index(drop=True)
