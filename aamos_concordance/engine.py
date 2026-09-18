"""Vectorised permutation engine.

The permutation test shuffles only the self-report column. The window join,
which is the expensive step in the per-permutation kernel, depends only on
the questionnaire timestamps and the inhaler records, so it is identical in
every permutation. This module exploits that:

1. **Precompute per patient** the device count for every questionnaire row
   under each of the distinct windows (eleven behind the 132 configurations),
   using the very same join function as the kernel. Categorise the device
   column and the *unshuffled* self-report column once per (window, method);
   categorisation is elementwise, so a shuffled self-report categorises to
   the shuffled categorised vector.
2. **Evaluate all permutations of a configuration at once.** With a
   permutation index matrix ``P`` of shape ``(K, n)``, the shuffled
   categorised self-report is ``s[P]``; the zero filter is a per-row mask;
   ranks are taken row-wise with NaNs for masked entries; Spearman and
   Pearson are computed as row-wise Pearson coefficients on ranks and raw
   values respectively.

Output is the same per-configuration table that :func:`batch.run_batch`
produces, so the aggregator, ``null_from_per_config`` and the summary code
consume it unchanged. Equivalence to the kernel is checked by
``tests/test_engine.py`` on synthetic data and against the v1 null on the
real data.

Permutation sources:

* :func:`sampled_permutations` reproduces the kernel's shuffle exactly
  (``Series.sample(frac=1, random_state=seed + k)`` draws
  ``RandomState(seed + k).choice(n, n, replace=False)``), so results are
  comparable with the published null one permutation index at a time.
* :func:`distinct_permutations` enumerates every distinct arrangement of the
  self-report multiset, for exact tests on small-n patients (the plan's
  exact-enumeration case). Patient 454 has 840; 917 has a few hundred.

Numerical note: the coefficients are computed with the same centring and
normalisation scipy uses, but summation order differs, so values agree with
the kernel to about 1e-12 rather than bit for bit, and a perfect correlation
(|r| within 1e-9 of 1) is treated as invalid in both, as the kernel's
``-1 < r < 1`` rule intends. The residual last-ulp differences matter only
for exact ties between a null value and the observed statistic, which exact
enumeration is designed to make irrelevant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .categorization import DEFAULT_TOP_CATEGORY_FALLBACK, build_categorize_fn
from .configs import CATEGORIZATION_METHODS, generate_param_combinations
from .join import join_questionnaire_with_inhaler
from .permutation import CORRELATION_TYPES, MIN_ROWS_FOR_CORRELATION

PER_CONFIG_COLUMNS = ["patient_id", "permutation_idx", "config_idx", "spearman_z", "pearson_z"]


# --------------------------------------------------------------------------
# Permutation matrices
# --------------------------------------------------------------------------

def sampled_permutations(n: int, random_seed: int, perm_indices: Sequence[int]) -> np.ndarray:
    """Index matrix ``(len(perm_indices), n)`` reproducing the kernel's shuffle for each index."""
    out = np.empty((len(perm_indices), n), dtype=np.intp)
    for row, k in enumerate(perm_indices):
        out[row] = np.random.RandomState(random_seed + k).choice(n, size=n, replace=False)
    return out


def n_distinct_permutations(values: Sequence) -> int:
    """Number of distinct arrangements of a multiset."""
    _, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
    n = int(counts.sum())
    out = factorial(n)
    for c in counts:
        out //= factorial(int(c))
    return out


def _multiset_permutations(counts: List[int]):
    """Yield every distinct arrangement of a multiset given as per-value counts (values are 0..k-1)."""
    n = sum(counts)
    counts = list(counts)
    seq = [0] * n

    def rec(pos):
        if pos == n:
            yield tuple(seq)
            return
        for v in range(len(counts)):
            if counts[v]:
                counts[v] -= 1
                seq[pos] = v
                yield from rec(pos + 1)
                counts[v] += 1

    yield from rec(0)


def distinct_permutations(values: Sequence, limit: Optional[int] = None) -> np.ndarray:
    """Every distinct arrangement of ``values`` as an index matrix into the original positions.

    Row 0 is the identity. Arrangements are generated directly from the
    multiset (never by deduplicating all n! permutations). Raises if the
    count exceeds ``limit`` (default 100,000).
    """
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    total = n_distinct_permutations(vals)
    cap = 100_000 if limit is None else limit
    if total > cap:
        raise ValueError(f"{total} distinct permutations exceeds the limit of {cap}; use sampled permutations")

    # Distinct values -> code 0..k-1 (NaN gets its own code); positions grouped by code.
    codes = np.full(n, -1, dtype=int)
    uniq = np.unique(vals[~np.isnan(vals)])
    for i, u in enumerate(uniq):
        codes[vals == u] = i
    if np.isnan(vals).any():
        codes[np.isnan(vals)] = len(uniq)
    k = codes.max() + 1
    positions = [np.flatnonzero(codes == c) for c in range(k)]
    counts = [len(p) for p in positions]

    rows = np.empty((total, n), dtype=np.intp)
    for r, arrangement in enumerate(_multiset_permutations(counts)):
        used = [0] * k
        for j, c in enumerate(arrangement):
            rows[r, j] = positions[c][used[c]]
            used[c] += 1
    # Make the observed arrangement row 0: find the row equal to the identity of codes.
    ident = tuple(codes.tolist())
    for r, arrangement in enumerate(_multiset_permutations(counts)):
        if arrangement == ident:
            if r != 0:
                rows[[0, r]] = rows[[r, 0]]
            break
    # rows[0] now places each position's own value at that position, but possibly a *different*
    # position with the same value; remap so that row 0 is literally arange(n) and all rows stay valid.
    inverse = np.empty(n, dtype=np.intp)
    inverse[rows[0]] = np.arange(n)
    return rows[:, inverse]


# --------------------------------------------------------------------------
# Precomputation
# --------------------------------------------------------------------------

@dataclass
class PatientTables:
    """Everything the engine needs for one patient, computed once."""

    patient_id: int
    n_rows: int
    # config_idx -> (device categorised (n,), self-report categorised (n,), filter flag)
    per_config: Dict[int, Tuple[np.ndarray, np.ndarray, bool]]


def precompute_patient(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    param_combinations: Optional[Sequence[Dict]] = None,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
) -> PatientTables:
    """Join once per distinct window and categorise once per (window, method).

    ``questionnaire_df`` / ``inhaler_df`` must already be filtered to the
    patient. Row order of ``questionnaire_df`` is the row order of every
    vector, so permutation index matrices refer to it.
    """
    combos = list(param_combinations) if param_combinations is not None else generate_param_combinations()
    n = len(questionnaire_df)
    joined_by_window: Dict[Tuple, pd.DataFrame] = {}
    cat_by_window_method: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
    per_config: Dict[int, Tuple[np.ndarray, np.ndarray, bool]] = {}

    for idx, cfg in enumerate(combos):
        wkey = (cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"])
        if wkey not in joined_by_window:
            joined_by_window[wkey] = join_questionnaire_with_inhaler(
                questionnaire_df, inhaler_df, *wkey)
        merged = joined_by_window[wkey]
        mkey = (wkey, cfg["categorization_method"])
        if mkey not in cat_by_window_method:
            fn = build_categorize_fn(merged, cfg["categorization_method"], top_category_fallback=top_category_fallback)
            d = merged["inhaler_usage"].apply(fn).to_numpy(dtype=float)
            s = merged["daily_relief_inhaler"].apply(fn).to_numpy(dtype=float)
            cat_by_window_method[mkey] = (d, s)
        d, s = cat_by_window_method[mkey]
        per_config[idx] = (d, s, bool(cfg["filter_out_zero_usage"]))
    return PatientTables(patient_id=patient_id, n_rows=n, per_config=per_config)


# --------------------------------------------------------------------------
# Row-wise statistics on a (K, n) matrix with NaN for excluded entries
# --------------------------------------------------------------------------

def _rowwise_pearson(X: np.ndarray, Y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pearson r per row over ``valid`` entries, scipy-style centring and scaling. NaN where undefined."""
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xz = np.where(valid, X, 0.0)
        Yz = np.where(valid, Y, 0.0)
        xmean = Xz.sum(axis=1) / cnt
        ymean = Yz.sum(axis=1) / cnt
        xm = np.where(valid, X - xmean[:, None], 0.0)
        ym = np.where(valid, Y - ymean[:, None], 0.0)
        xmax = np.abs(xm).max(axis=1)
        ymax = np.abs(ym).max(axis=1)
        normx = xmax * np.sqrt(((xm / xmax[:, None]) ** 2).sum(axis=1))
        normy = ymax * np.sqrt(((ym / ymax[:, None]) ** 2).sum(axis=1))
        r = ((xm / normx[:, None]) * (ym / normy[:, None])).sum(axis=1)
    r = np.clip(r, -1.0, 1.0)
    r[~np.isfinite(r)] = np.nan
    return r


# |r| this close to 1 is a perfect correlation up to rounding. The kernel gets
# exactly +-1 from scipy in that case and records NaN (Fisher Z undefined);
# a different summation order can land a few ulps short and give Z ~ 18.
PERFECT_CORRELATION_TOLERANCE = 1e-9


def _fisher_z(r: np.ndarray) -> np.ndarray:
    out = np.full(r.shape, np.nan)
    ok = np.isfinite(r) & (np.abs(r) < 1 - PERFECT_CORRELATION_TOLERANCE)
    out[ok] = np.arctanh(r[ok])
    return out


def evaluate_config(d: np.ndarray, s: np.ndarray, use_filter: bool, P: np.ndarray) -> Dict[str, np.ndarray]:
    """Fisher Z of both correlation types for every permutation row of ``P``.

    ``d`` is the categorised device vector, ``s`` the categorised
    (unshuffled) self-report vector, ``P`` a ``(K, n)`` index matrix.
    Applies the kernel's validity rules: at least three rows after the
    filter, and neither column constant.
    """
    K, n = P.shape
    S = s[P]                       # shuffled categorised self-report, (K, n)
    D = np.broadcast_to(d, (K, n))
    if use_filter:
        valid = (D > 0) | (S > 0)
    else:
        valid = np.ones((K, n), dtype=bool)
    # NaN in either column: the kernel's nunique() counts NaN as a value but
    # scipy propagates NaN into the coefficient -> NaN Z. Treat NaN entries as
    # present for the row-count and constancy checks, then let the statistic be NaN.
    cnt = valid.sum(axis=1)
    Dn = np.where(valid, D, np.nan)
    Sn = np.where(valid, S, np.nan)
    with np.errstate(invalid="ignore"):
        d_const = np.nanmax(Dn, axis=1) == np.nanmin(Dn, axis=1)
        s_const = np.nanmax(Sn, axis=1) == np.nanmin(Sn, axis=1)
    has_nan = (valid & np.isnan(D)).any(axis=1) | (valid & np.isnan(S)).any(axis=1)
    # nunique() == 1 in the kernel means exactly one distinct value; a NaN alongside
    # one value gives nunique()==1 too (pandas drops NaN), which the kernel treats as invalid.
    invalid = (cnt < MIN_ROWS_FOR_CORRELATION) | d_const | s_const

    out: Dict[str, np.ndarray] = {}
    # Pearson on raw categorised values
    r_p = _rowwise_pearson(Dn, Sn, valid)
    # Spearman: Pearson on average ranks over the valid entries
    Rd = rankdata(Dn, method="average", axis=1, nan_policy="omit")
    Rs = rankdata(Sn, method="average", axis=1, nan_policy="omit")
    r_s = _rowwise_pearson(Rd, Rs, valid)
    for name, r in (("spearman", r_s), ("pearson", r_p)):
        z = _fisher_z(r)
        z[invalid | has_nan] = np.nan
        out[name] = z
    return out


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def run_patient(
    tables: PatientTables,
    P: np.ndarray,
    perm_indices: Sequence[int],
    config_indices: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Per-config table for the permutations in ``P`` (labelled by ``perm_indices``)."""
    cfg_ids = list(config_indices) if config_indices is not None else sorted(tables.per_config)
    K = P.shape[0]
    n_cfg = len(cfg_ids)
    spearman = np.empty((n_cfg, K))
    pearson = np.empty((n_cfg, K))
    for j, ci in enumerate(cfg_ids):
        d, s, use_filter = tables.per_config[ci]
        z = evaluate_config(d, s, use_filter, P)
        spearman[j] = z["spearman"]
        pearson[j] = z["pearson"]
    perm_col = np.repeat(np.asarray(perm_indices, dtype=np.int64), n_cfg)
    cfg_col = np.tile(np.asarray(cfg_ids, dtype=np.int32), K)
    return pd.DataFrame({
        "patient_id": np.full(K * n_cfg, tables.patient_id, dtype=np.int64),
        "permutation_idx": perm_col,
        "config_idx": cfg_col,
        "spearman_z": spearman.T.reshape(-1),
        "pearson_z": pearson.T.reshape(-1),
    })


def run_patient_sampled(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    perm_start: int,
    perm_end: int,
    random_seed: int,
    *,
    param_combinations: Optional[Sequence[Dict]] = None,
    chunk: int = 2000,
) -> pd.DataFrame:
    """Drop-in for the per-config output of ``batch.run_batch`` for one patient."""
    tables = precompute_patient(patient_id, questionnaire_df, inhaler_df, param_combinations)
    parts = []
    idx = list(range(perm_start, perm_end))
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        P = sampled_permutations(tables.n_rows, random_seed, sub)
        parts.append(run_patient(tables, P, sub))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(PER_CONFIG_COLUMNS, ["int64", "int64", "int32", "float64", "float64"])})
    return pd.concat(parts, ignore_index=True).sort_values(["permutation_idx", "config_idx"], kind="stable").reset_index(drop=True)


def run_patient_exact(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    *,
    param_combinations: Optional[Sequence[Dict]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Per-config table over every distinct arrangement of the self-report (row 0 = observed)."""
    tables = precompute_patient(patient_id, questionnaire_df, inhaler_df, param_combinations)
    P = distinct_permutations(questionnaire_df["daily_relief_inhaler"].to_numpy(dtype=float), limit=limit)
    return run_patient(tables, P, list(range(P.shape[0])))


def observed_per_config(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    *,
    param_combinations: Optional[Sequence[Dict]] = None,
) -> pd.DataFrame:
    """The unshuffled evaluation (permutation_idx = -1): the observed multiverse for one patient."""
    tables = precompute_patient(patient_id, questionnaire_df, inhaler_df, param_combinations)
    P = np.arange(tables.n_rows, dtype=np.intp)[None, :]
    return run_patient(tables, P, [-1])
