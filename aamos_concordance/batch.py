"""A batch of permutations for one patient, independent of any I/O.

The Cloud Run job-worker calls :func:`run_batch` and uploads what it
returns. Keeping the loop here means the whole batch can be run and checked
locally on real or synthetic data.

For each permutation index in ``[perm_start, perm_end)`` the batch records:

* the per-permutation **summary** of the primary correlation type (the
  historical JSON record: mean, median, quartiles, boxplot fences), plus the
  same summary for the other type under the key ``"pearson"`` /
  ``"spearman"``;
* the **per-configuration Fisher Z vectors** for both types, as one long
  table ``(patient_id, permutation_idx, config_idx, spearman_z, pearson_z)``.
  Z is kept in float64: the permutation p-value compares ``|null| >=
  |observed|`` and, for small-n patients, exact ties occur, so rounding the
  stored null would change tie counts. The cost is ~320 MB uncompressed for
  15 patients x 10,000 permutations, which lives in the bucket, not in git.

The per-configuration table is the new, load-bearing artifact. With it the
null distribution can be re-summarised after the fact under any subset of
configurations, any summary statistic and either correlation type, exactly
as the observed per-config table can. Without it (the v1 situation) only the
mean and median over all 132 configurations are ever recoverable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .permutation import CORRELATION_TYPES, evaluate_permutation, summarize_correlations

PER_CONFIG_COLUMNS = ["patient_id", "permutation_idx", "config_idx", "spearman_z", "pearson_z"]


def run_batch(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    param_combinations: Sequence[Dict],
    perm_start: int,
    perm_end: int,
    random_seed: int,
    primary_type: str = "spearman",
    *,
    collect_datasets: bool = False,
    progress=None,
) -> Tuple[List[Dict], pd.DataFrame, List[pd.DataFrame]]:
    """Run permutations ``perm_start .. perm_end-1`` for one patient.

    Returns ``(records, per_config, datasets)``:

    * ``records`` is a list of per-permutation dicts in the job-worker's JSON
      layout: ``permutation_idx``, the summary of ``primary_type`` at top
      level, and the other type's summary nested under its name;
    * ``per_config`` is a DataFrame with :data:`PER_CONFIG_COLUMNS`, one row
      per (permutation, configuration), Z stored as float64;
    * ``datasets`` is the list of categorised frames if ``collect_datasets``.

    ``progress`` is an optional callable ``(perm_idx, record)`` for logging.
    """
    if primary_type not in CORRELATION_TYPES:
        raise ValueError(f"primary_type must be one of {CORRELATION_TYPES}, got {primary_type!r}")
    other_type = [t for t in CORRELATION_TYPES if t != primary_type][0]
    n_configs = len(param_combinations)

    records: List[Dict] = []
    blocks: List[np.ndarray] = []
    datasets: List[pd.DataFrame] = []

    for perm_idx in range(perm_start, perm_end):
        z, ds = evaluate_permutation(
            patient_id, perm_idx, questionnaire_df, inhaler_df, param_combinations, random_seed,
            collect_datasets=collect_datasets,
        )
        if collect_datasets and ds:
            datasets.extend(ds)

        record = {"permutation_idx": perm_idx, **summarize_correlations(z[primary_type])}
        record[other_type] = summarize_correlations(z[other_type])
        records.append(record)

        block = np.empty((n_configs, 5), dtype=np.float64)
        block[:, 0] = patient_id
        block[:, 1] = perm_idx
        block[:, 2] = np.arange(n_configs)
        block[:, 3] = z["spearman"]
        block[:, 4] = z["pearson"]
        blocks.append(block)

        if progress is not None:
            progress(perm_idx, record)

    if blocks:
        arr = np.vstack(blocks)
        per_config = pd.DataFrame({
            "patient_id": arr[:, 0].astype(np.int64),
            "permutation_idx": arr[:, 1].astype(np.int64),
            "config_idx": arr[:, 2].astype(np.int32),
            "spearman_z": arr[:, 3],
            "pearson_z": arr[:, 4],
        })
    else:
        per_config = pd.DataFrame({c: pd.Series(dtype=t) for c, t in
                                   zip(PER_CONFIG_COLUMNS, ["int64", "int64", "int32", "float64", "float64"])})
    return records, per_config, datasets


def null_from_per_config(
    per_config: pd.DataFrame,
    *,
    correlation_type: str = "spearman",
    config_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Re-summarise a per-config null table into per-permutation statistics.

    Produces the columns the aggregator's null table has always had
    (``null_mean_z``, ``null_median_z``, quartiles, min, max, std,
    ``n_valid_configs``), computed over the configurations in
    ``config_indices`` (all of them when ``None``) for ``correlation_type``.
    Summaries use the same finite-only rule as :func:`summarize_correlations`.
    """
    if correlation_type not in CORRELATION_TYPES:
        raise ValueError(f"correlation_type must be one of {CORRELATION_TYPES}, got {correlation_type!r}")
    col = f"{correlation_type}_z"
    df = per_config
    if config_indices is not None:
        df = df[df["config_idx"].isin(list(config_indices))]
    df = df[["patient_id", "permutation_idx", col]]
    valid = df[np.isfinite(df[col])]

    g = valid.groupby(["patient_id", "permutation_idx"])[col]
    out = pd.DataFrame({
        "n_valid_configs": g.size(),
        "null_mean_z": g.mean(),
        "null_median_z": g.median(),
        "null_min_z": g.min(),
        "null_max_z": g.max(),
        "null_q25_z": g.quantile(0.25),
        "null_q75_z": g.quantile(0.75),
        "null_std_z": g.std(ddof=0),
    })
    # Permutations where every configuration was invalid still get a row (n_valid_configs = 0).
    all_pairs = df[["patient_id", "permutation_idx"]].drop_duplicates().set_index(["patient_id", "permutation_idx"]).index
    out = out.reindex(all_pairs)
    out["n_valid_configs"] = out["n_valid_configs"].fillna(0).astype(int)
    return out.reset_index().sort_values(["patient_id", "permutation_idx"]).reset_index(drop=True)
