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

Device-side absence cases (v2 items 9 and 10), applied per window after the
join and before categorisation, selected by the world's ``absence_case``:

* ``A`` naive zero: a window with no device records has count 0 (v1).
* ``B`` Poisson imputation: every such window gets a draw from
  ``Poisson(rate * window_hours)`` where ``rate`` is the patient's device
  records per hour over the device-active period of the (span-trimmed)
  data. Draws are seeded by (patient, imputation index, window), so a world
  ``k`` is one coherent set of draws across all configurations; all twelve
  configurations sharing a window share its draws. Windows with a zero
  self-report are imputed too (KT8).
* ``C`` complete case: every such window is dropped from that
  configuration (a permutation-independent row mask; under a permutation
  the surviving rows receive self-reports from anywhere, exactly as the
  zero filter's survivors do).

The zero-usage filter is an active axis only under ``A``. Under ``B`` the
imputed cells are what the case is about, and under ``C`` no ``rec = 0``
row survives, so for both the filter is forced off and the
``filter_out_zero_usage=True`` configurations duplicate their ``False``
counterparts (``configs.effective_config_indices`` collapses them).

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

from .categorization import DEFAULT_TOP_CATEGORY_FALLBACK, build_categorize_fn, categorize_columns
from .configs import CATEGORIZATION_METHODS, _window_key, generate_param_combinations
from .join import join_questionnaire_with_inhaler
from .permutation import CORRELATION_TYPES, MIN_ROWS_FOR_CORRELATION
from .worlds import BASELINE, WorldSpec, as_world

# ``n_rows`` is the number of rows each correlation used (after the zero filter
# and case C), so minimum-row sensitivities can be applied at summary time.
PER_CONFIG_COLUMNS = ["patient_id", "permutation_idx", "config_idx", "spearman_z", "pearson_z", "n_rows"]
_PER_CONFIG_DTYPES = ["int64", "int64", "int32", "float64", "float64", "int16"]


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
    # config_idx -> (device categorised (n,), self-report categorised (n,), filter flag, keep mask (n,))
    per_config: Dict[int, Tuple[np.ndarray, np.ndarray, bool, np.ndarray]]
    world: WorldSpec = BASELINE

    def observed_row_count(self, config_idx: int) -> int:
        """Rows the unshuffled evaluation of a configuration correlates (after the zero filter if on).

        Counts rows as v1's ``sample_size`` did: every joined row the case
        keeps, minus those the zero filter removes. A NaN categorised value
        (the hybrid upper bound with a single count >= 12 has an undefined
        SD) still counts as a row; it makes the correlation NaN, not the row
        absent.
        """
        d, s, use_filter, keep = self.per_config[config_idx]
        keep = keep.copy()
        if use_filter:
            keep &= (d > 0) | (s > 0)
        return int(keep.sum())


def precompute_patient(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    param_combinations: Optional[Sequence[Dict]] = None,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
    world: "WorldSpec | str | None" = None,
) -> PatientTables:
    """Join once per distinct window and categorise once per (window, method).

    ``questionnaire_df`` / ``inhaler_df`` must already be filtered to the
    patient and trimmed to the world's span (see ``pipeline.world_frames``).
    Row order of ``questionnaire_df`` is the row order of every vector, so
    permutation index matrices refer to it. ``world`` selects the device-side
    absence case applied after each join; only case A (the v1 behaviour) is
    implemented so far.
    """
    w = as_world(world)
    combos = list(param_combinations) if param_combinations is not None else generate_param_combinations()
    n = len(questionnaire_df)
    rate = device_rate_per_hour(inhaler_df) if w.absence_case == "B" else 0.0
    joined_by_window: Dict[Tuple, pd.DataFrame] = {}
    keep_by_window: Dict[Tuple, np.ndarray] = {}
    cat_by_window_method: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
    per_config: Dict[int, Tuple[np.ndarray, np.ndarray, bool, np.ndarray]] = {}

    for idx, cfg in enumerate(combos):
        wkey = (cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"])
        if wkey not in joined_by_window:
            merged = join_questionnaire_with_inhaler(questionnaire_df, inhaler_df, *wkey)
            merged, keep = apply_absence_case(merged, w, patient_id=patient_id, window_key=wkey,
                                              rate_per_hour=rate)
            joined_by_window[wkey] = merged
            keep_by_window[wkey] = keep
        merged = joined_by_window[wkey]
        mkey = (wkey, cfg["categorization_method"])
        if mkey not in cat_by_window_method:
            dev, rep = categorize_columns(merged, cfg["categorization_method"], top_category_fallback=top_category_fallback)
            d = dev.to_numpy(dtype=float)
            s = rep.to_numpy(dtype=float)
            cat_by_window_method[mkey] = (d, s)
        d, s = cat_by_window_method[mkey]
        use_filter = bool(cfg["filter_out_zero_usage"]) and w.absence_case == "A"   # item 10
        per_config[idx] = (d, s, use_filter, keep_by_window[wkey])
    return PatientTables(patient_id=patient_id, n_rows=n, per_config=per_config, world=w)


def window_hours(window_key: Tuple[int, bool, bool]) -> float:
    """Duration in hours of a window configuration (rolling = its length; chunk and calendar day = 24)."""
    timestamp_window, use_daily_max, use_calendar = window_key
    if use_calendar or use_daily_max:
        return 24.0
    return float(timestamp_window)


def device_rate_per_hour(inhaler_df: pd.DataFrame) -> float:
    """Device records per hour over the device-active period (first to last record, inclusive days)."""
    if len(inhaler_df) == 0:
        return 0.0
    days = int(inhaler_df["date"].max() - inhaler_df["date"].min()) + 1
    return float(len(inhaler_df)) / (24.0 * days)


_WINDOW_KIND_CODE = {"rolling": 0, "fixed_chunk": 1, "calendar": 2}


def imputation_seed(patient_id: int, imputation: int, window_key: Tuple[int, bool, bool]) -> int:
    """Deterministic seed for the Poisson draws of one (patient, imputation world, window).

    Seeded on the window's canonical interval (``configs._window_key``), so
    windows covering the same interval (the chunk starting 24 h back and the
    rolling 24-hour window) get the same draws and stay equivalent under B.
    """
    kind, value = _window_key(*window_key)
    return int(np.random.SeedSequence([int(patient_id), int(imputation), _WINDOW_KIND_CODE[kind], int(value), 2026]).generate_state(1)[0])


def apply_absence_case(
    merged: pd.DataFrame,
    world: WorldSpec,
    *,
    patient_id: int,
    window_key: Tuple[int, bool, bool],
    rate_per_hour: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Treat windows with no device records per the world's case. Returns (frame, keep mask)."""
    n = len(merged)
    keep = np.ones(n, dtype=bool)
    empty = (merged["inhaler_usage"] == 0).to_numpy()
    if world.absence_case == "A" or not empty.any():
        return merged, keep
    if world.absence_case == "C":
        return merged, ~empty
    if world.absence_case == "B":
        rng = np.random.default_rng(imputation_seed(patient_id, world.imputation, window_key))
        lam = rate_per_hour * window_hours(window_key)
        draws = rng.poisson(lam, size=int(empty.sum()))
        out = merged.copy()
        counts = out["inhaler_usage"].to_numpy().copy()
        counts[empty] = draws
        out["inhaler_usage"] = counts
        return out, keep
    raise ValueError(f"unknown absence case {world.absence_case!r}")


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


def evaluate_config(d: np.ndarray, s: np.ndarray, use_filter: bool, P: np.ndarray,
                    keep: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Fisher Z of both correlation types for every permutation row of ``P``.

    ``d`` is the categorised device vector, ``s`` the categorised
    (unshuffled) self-report vector, ``P`` a ``(K, n)`` index matrix,
    ``keep`` an optional permutation-independent row mask (case C).
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
    if keep is not None:
        valid = valid & keep[None, :]
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
    out["n_rows"] = cnt
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
    n_rows = np.empty((n_cfg, K), dtype=np.int16)
    for j, ci in enumerate(cfg_ids):
        d, s, use_filter, keep = tables.per_config[ci]
        z = evaluate_config(d, s, use_filter, P, keep)
        spearman[j] = z["spearman"]
        pearson[j] = z["pearson"]
        n_rows[j] = z["n_rows"]
    perm_col = np.repeat(np.asarray(perm_indices, dtype=np.int64), n_cfg)
    cfg_col = np.tile(np.asarray(cfg_ids, dtype=np.int32), K)
    return pd.DataFrame({
        "patient_id": np.full(K * n_cfg, tables.patient_id, dtype=np.int64),
        "permutation_idx": perm_col,
        "config_idx": cfg_col,
        "spearman_z": spearman.T.reshape(-1),
        "pearson_z": pearson.T.reshape(-1),
        "n_rows": n_rows.T.reshape(-1),
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
    world: "WorldSpec | str | None" = None,
) -> pd.DataFrame:
    """Drop-in for the per-config output of ``batch.run_batch`` for one patient."""
    tables = precompute_patient(patient_id, questionnaire_df, inhaler_df, param_combinations, world=world)
    parts = []
    idx = list(range(perm_start, perm_end))
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        P = sampled_permutations(tables.n_rows, random_seed, sub)
        parts.append(run_patient(tables, P, sub))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(PER_CONFIG_COLUMNS, _PER_CONFIG_DTYPES)})
    return pd.concat(parts, ignore_index=True).sort_values(["permutation_idx", "config_idx"], kind="stable").reset_index(drop=True)


def run_patient_exact(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    *,
    param_combinations: Optional[Sequence[Dict]] = None,
    limit: Optional[int] = None,
    world: "WorldSpec | str | None" = None,
) -> pd.DataFrame:
    """Per-config table over every distinct arrangement of the self-report (row 0 = observed)."""
    tables = precompute_patient(patient_id, questionnaire_df, inhaler_df, param_combinations, world=world)
    P = distinct_permutations(questionnaire_df["daily_relief_inhaler"].to_numpy(dtype=float), limit=limit)
    return run_patient(tables, P, list(range(P.shape[0])))


def observed_per_config(
    patient_id: int,
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    *,
    param_combinations: Optional[Sequence[Dict]] = None,
    world: "WorldSpec | str | None" = None,
) -> pd.DataFrame:
    """The unshuffled evaluation (permutation_idx = -1): the observed multiverse for one patient."""
    tables = precompute_patient(patient_id, questionnaire_df, inhaler_df, param_combinations, world=world)
    P = np.arange(tables.n_rows, dtype=np.intp)[None, :]
    return run_patient(tables, P, [-1])
