"""From a world's per-config Z table and null parquet to its summary and concordant set.

This is the post-processing step of the pipeline, factored out of the
Cloud Run aggregator so that it runs identically for every world and can be
tested against the frozen v1 tables (``tests/test_summary.py``).

Definitions, unchanged from v1:

* The **observed statistic** for a patient is the mean (or median) of the
  *finite* Fisher Z values across the 132 configurations. Non-finite Z
  (a Spearman rho of exactly ±1, or an undefined correlation) is excluded.
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

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .worlds import MEASURES

DEFAULT_THRESHOLD = 0.5
ALPHA = 0.05
# Null values this close to |observed| are reported as ties: whether they count
# as ">= observed" depends on the last bits of the observed statistic.
TIE_TOLERANCE = 1e-12


def observed_statistics(per_config: pd.DataFrame, z_column: str = "spearman_z") -> pd.DataFrame:
    """Per-patient mean and median of finite Z, plus the count of finite configs.

    Computed per patient with ``Series.mean()`` / ``Series.median()`` on the
    finite subset, exactly as v1's ``create_summary_statistics`` did. The
    summation order matters: a groupby aggregate can differ in the last ulp,
    and when a null distribution contains permutations that reproduce the
    observed arrangement exactly (small-n patients with few distinct
    permutations) the ``|null| >= |observed|`` tie count, and hence the
    p-value, flips on that ulp.
    """
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


def build_world_summary(
    per_config: pd.DataFrame,
    null: pd.DataFrame,
    *,
    measures: Iterable[str] = MEASURES,
) -> pd.DataFrame:
    """Long summary table with one row per (patient, measure)."""
    obs = observed_statistics(per_config).set_index("patient_id")
    parts = []
    for measure in measures:
        table = permutation_p_values(obs[measure], null, null_column=f"null_{measure}_z")
        table.insert(1, "measure", measure)
        table["n_valid_configs_observed"] = table["patient_id"].map(obs["n_valid"]).astype(int)
        parts.append(table)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["measure", "p_value", "patient_id"], kind="stable").reset_index(drop=True)


def concordant_set(summary: pd.DataFrame, measure: str, threshold: float = DEFAULT_THRESHOLD) -> List[int]:
    s = summary[summary["measure"] == measure]
    hit = s[(s["observed_z"] >= threshold) & s["significant_bonferroni"]]
    return sorted(int(p) for p in hit["patient_id"])


def derive_concordant_sets(summary: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, List[int]]:
    return {m: concordant_set(summary, m, threshold) for m in summary["measure"].unique()}


def assessed_patients(summary: pd.DataFrame) -> List[int]:
    """Patients with a valid observed statistic and a null distribution (the v1 'assessed' 15)."""
    return sorted(int(p) for p in summary["patient_id"].unique())


def to_v1_measure_table(summary: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Reshape one measure into the column layout of v1's ``permutation_test_results.csv``."""
    s = summary[summary["measure"] == measure].copy()
    s = s.rename(columns={
        "observed_z": f"observed_{measure}_z",
        "null_mean": f"null_{measure}_mean", "null_std": f"null_{measure}_std",
        "null_q025": f"null_{measure}_q025", "null_q975": f"null_{measure}_q975",
    })
    cols = ["patient_id", f"observed_{measure}_z", "n_permutations", "avg_valid_configs", "p_value", "p_bonferroni",
            "significant_uncorrected", "significant_bonferroni",
            f"null_{measure}_mean", f"null_{measure}_std", f"null_{measure}_q025", f"null_{measure}_q975"]
    return s[cols].sort_values(["p_value", "patient_id"], kind="stable").reset_index(drop=True)
