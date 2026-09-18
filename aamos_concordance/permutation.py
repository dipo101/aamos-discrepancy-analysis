"""Single-permutation multiverse evaluation and per-permutation summaries.

This is the compute kernel of the permutation test, lifted from the Cloud
Run job-worker so that it can be run and tested locally. The job-worker's
``main.py`` is now a thin wrapper that handles GCS I/O and environment
variables around these two functions.

For one patient and one permutation index the kernel:

1. shuffles the ``daily_relief_inhaler`` column with
   ``Series.sample(frac=1, random_state=random_seed + perm_idx)``;
2. for each of the 132 configurations, joins, categorises, applies the zero
   filter, and computes **both** the Spearman and the Pearson correlation on
   the same categorised frame;
3. Fisher-Z-transforms each valid correlation.

A correlation is recorded as NaN when the joined frame has fewer than three
rows, when either column is constant, or when that coefficient is not
strictly inside (-1, 1); the two types are validated independently.
``evaluate_permutation`` returns both vectors; ``run_single_permutation``
keeps the historical one-type interface. ``summarize_correlations`` reduces
a vector of per-configuration Z values to the summary statistics the
aggregator expects, including the boxplot fences used in the figures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .categorization import DEFAULT_TOP_CATEGORY_FALLBACK, categorize_inhaler_usage, filter_zero_usage
from .join import join_questionnaire_with_inhaler

MIN_ROWS_FOR_CORRELATION = 3
CORRELATION_TYPES = ("spearman", "pearson")


def _fisher_z(corr: float) -> float:
    """Fisher Z of a coefficient, or NaN if it is not strictly inside (-1, 1)."""
    if np.isfinite(corr) and -1 < corr < 1:
        return float(np.arctanh(corr))
    return np.nan


def _both_correlations(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    spearman, _ = stats.spearmanr(x, y)
    pearson, _ = stats.pearsonr(x, y)
    return {"spearman": spearman, "pearson": pearson}


def evaluate_permutation(
    patient_id,
    perm_idx: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    param_combinations: Sequence[Dict],
    random_seed: int,
    collect_datasets: bool = False,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
) -> Tuple[Dict[str, List[float]], List[pd.DataFrame]]:
    """Evaluate every configuration on one shuffled copy of the questionnaire.

    ``questionnaire_df`` and ``inhaler_df`` must already be filtered to
    ``patient_id``. Returns ``({"spearman": z_per_config, "pearson":
    z_per_config}, datasets)``; ``datasets`` is empty unless
    ``collect_datasets`` is set. ``patient_id`` only labels collected
    datasets. Each vector has exactly ``len(param_combinations)`` entries.
    """
    perm_seed = random_seed + perm_idx

    shuffled = questionnaire_df.copy()
    shuffled["daily_relief_inhaler"] = (
        shuffled["daily_relief_inhaler"].sample(frac=1, random_state=perm_seed).values
    )

    z: Dict[str, List[float]] = {t: [] for t in CORRELATION_TYPES}
    datasets: List[pd.DataFrame] = []

    def record_invalid():
        for t in CORRELATION_TYPES:
            z[t].append(np.nan)

    for config_idx, params in enumerate(param_combinations):
        try:
            merged = join_questionnaire_with_inhaler(
                shuffled,
                inhaler_df,
                params["timestamp_window"],
                params["use_daily_max_windows"],
                params["use_calendar_days"],
            )
            if merged is None or len(merged) < MIN_ROWS_FOR_CORRELATION:
                record_invalid()
                continue

            categorized = categorize_inhaler_usage(
                merged, params["categorization_method"],
                top_category_fallback=top_category_fallback,
            )
            if params["filter_out_zero_usage"]:
                categorized = filter_zero_usage(categorized)
            if len(categorized) < MIN_ROWS_FOR_CORRELATION:
                record_invalid()
                continue

            if categorized["inhaler_usage"].nunique() == 1 or categorized["daily_relief_inhaler"].nunique() == 1:
                record_invalid()
                continue

            if collect_datasets:
                records = categorized.copy()
                records["permutation_idx"] = perm_idx
                records["config_idx"] = config_idx
                for k, v in params.items():
                    records[k] = v
                records["patient_id"] = patient_id
                datasets.append(records)

            corrs = _both_correlations(categorized["inhaler_usage"], categorized["daily_relief_inhaler"])
            for t in CORRELATION_TYPES:
                z[t].append(_fisher_z(corrs[t]))
        except Exception:  # noqa: BLE001 - a failed configuration is recorded as NaN, as before
            # Keep the vectors aligned even if the failure happened mid-record.
            for t in CORRELATION_TYPES:
                if len(z[t]) <= config_idx:
                    z[t].append(np.nan)

    return z, datasets


def run_single_permutation(
    patient_id,
    perm_idx: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    param_combinations: Sequence[Dict],
    random_seed: int,
    correlation_type: str = "spearman",
    collect_datasets: bool = False,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
) -> Tuple[List[float], List[pd.DataFrame]]:
    """One-type interface kept for the historical callers and the oracle tests.

    Identical to :func:`evaluate_permutation` restricted to ``correlation_type``.
    """
    if correlation_type not in CORRELATION_TYPES:
        raise ValueError(f"correlation_type must be one of {CORRELATION_TYPES}, got {correlation_type!r}")
    z, datasets = evaluate_permutation(
        patient_id, perm_idx, questionnaire_df, inhaler_df, param_combinations, random_seed,
        collect_datasets=collect_datasets, top_category_fallback=top_category_fallback,
    )
    return z[correlation_type], datasets


def summarize_correlations(correlations: Sequence[float]) -> Dict:
    """Reduce per-configuration Fisher Z values to the aggregator's summary record.

    Mirrors the job-worker's per-permutation bookkeeping exactly, including
    the 1.5 IQR boxplot fences and the cap of twenty stored outliers.
    """
    valid = [c for c in correlations if c is not None and np.isfinite(c)]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "n_valid_configs": 0,
            "median_fisher_z": None, "mean_fisher_z": None,
            "min_fisher_z": None, "max_fisher_z": None,
            "q25_fisher_z": None, "q75_fisher_z": None, "std_fisher_z": None,
            "iqr_fisher_z": None, "lower_fence": None, "upper_fence": None,
            "lower_whisker": None, "upper_whisker": None,
            "n_outliers": 0, "outliers": [],
        }

    median_z = float(np.median(valid))
    mean_z = float(np.mean(valid))
    min_z = float(np.min(valid))
    max_z = float(np.max(valid))
    q25_z = float(np.percentile(valid, 25))
    q75_z = float(np.percentile(valid, 75))
    std_z = float(np.std(valid))

    iqr = q75_z - q25_z
    lower_fence = q25_z - 1.5 * iqr
    upper_fence = q75_z + 1.5 * iqr
    non_outliers = [c for c in valid if lower_fence <= c <= upper_fence]
    if non_outliers:
        lower_whisker, upper_whisker = float(min(non_outliers)), float(max(non_outliers))
    else:
        lower_whisker, upper_whisker = min_z, max_z
    outliers = [float(c) for c in valid if c < lower_fence or c > upper_fence]

    return {
        "n_valid_configs": n_valid,
        "median_fisher_z": median_z, "mean_fisher_z": mean_z,
        "min_fisher_z": min_z, "max_fisher_z": max_z,
        "q25_fisher_z": q25_z, "q75_fisher_z": q75_z, "std_fisher_z": std_z,
        "iqr_fisher_z": float(iqr), "lower_fence": float(lower_fence), "upper_fence": float(upper_fence),
        "lower_whisker": lower_whisker, "upper_whisker": upper_whisker,
        "n_outliers": len(outliers),
        "outliers": outliers if len(outliers) <= 20 else outliers[:20],
    }
