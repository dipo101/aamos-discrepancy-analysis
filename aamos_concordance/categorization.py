"""Categorisation of inhaler-usage values and the zero-usage filter.

The self-reported ``daily_relief_inhaler`` column is a categorical range and
the device-derived ``inhaler_usage`` column is a count. Each categorisation
method maps *both* columns through the same step function so that they are
on a common scale before correlation.

The step functions below are deliberately written in the same shape as the
pre-refactor code (``x is None or x == 0`` first, then closed integer
ranges, then a catch-all). That shape has two quirks that are preserved on
purpose because the published results depend on them:

* a NaN input falls through every range test and lands in the top
  category (except for ``ONE_HOT``, where it maps to 0);
* a value strictly between 0 and 1 also lands in the top category.

``top_category_fallback`` is what the two data-driven methods return for the
top category when the input frame has no device usage at or above 12 (so
the data-driven value would be the mean or max of an empty set). The
pre-refactor copies disagreed: ``data_loader.py`` and the Cloud Run
job-worker let it be NaN (an unguarded edge case, which then invalidated the
whole configuration), while ``per_patient_correlation_analysis.py`` used 12.
The default is now 12 everywhere, i.e. the plain midpoint/upper-bound value.
The fallback matters when a self-report of "12 or more" (code 12) meets a
frame with no device count of 12 or more. Pass ``None`` only to reproduce
the old NaN behaviour (the oracle tests do).

Self-reports are band codes, not counts: {0, 1, 3, 5, 9, 12} stand for
"none", "1 to 2", "3 to 4", "5 to 8", "9 to 12" and "12 or more".
:func:`categorize_columns` maps code 12 to the top category (see
:mod:`aamos_concordance.definitions`).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from . import definitions


class CategorizationMethod(Enum):
    ONE_HOT = "one_hot"
    LOWER_BOUND = "lower_bound"
    MIDPOINT = "midpoint"
    MIDPOINT_WITH_INHALER = "midpoint_with_inhaler"
    UPPER_BOUND = "upper_bound"
    UPPER_BOUND_WITH_INHALER = "upper_bound_with_inhaler"


MethodLike = Union[CategorizationMethod, str]


def _as_method(method: MethodLike) -> CategorizationMethod:
    if isinstance(method, CategorizationMethod):
        return method
    try:
        return CategorizationMethod(method)
    except ValueError as exc:
        raise ValueError(f"Invalid categorization method: {method!r}") from exc


def _step_fn(v_1_2, v_3_4, v_5_8, v_9_12, v_top) -> Callable:
    """Build the five-range step function shared by every non-binary method."""

    def categorize_fn(x):
        if x is None or x == 0:
            return 0
        elif x >= 1 and x <= 2:
            return v_1_2
        elif x >= 3 and x <= 4:
            return v_3_4
        elif x >= 5 and x <= 8:
            return v_5_8
        elif x >= 9 and x <= 12:
            return v_9_12
        else:
            return v_top

    return categorize_fn


def _one_hot(x):
    return 1 if x is not None and x >= 1 else 0


def _top_value_midpoint_with_inhaler(df: pd.DataFrame, fallback: Optional[float]) -> float:
    above = df[df["inhaler_usage"] >= 12]["inhaler_usage"]
    if len(above) == 0 and fallback is not None:
        return fallback
    return above.mean()  # NaN when empty and no fallback


def _top_value_upper_bound_with_inhaler(df: pd.DataFrame, fallback: Optional[float]) -> float:
    above = df[df["inhaler_usage"] > 12]["inhaler_usage"]
    if len(above) > 0 and above.max() > 24:
        at_least_12 = df[df["inhaler_usage"] >= 12]["inhaler_usage"]
        return 3 * at_least_12.std() + at_least_12.median()
    if len(above) == 0:
        return fallback if fallback is not None else np.nan
    return above.max()


DEFAULT_TOP_CATEGORY_FALLBACK = 12


def build_categorize_fn(
    df: pd.DataFrame,
    method: MethodLike,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
) -> Callable:
    """Return the scalar step function for ``method``.

    ``df`` is only consulted by the two ``*_WITH_INHALER`` methods, which
    derive their top-category value from the observed ``inhaler_usage``
    distribution of the frame they are applied to.
    """
    m = _as_method(method)
    if m is CategorizationMethod.ONE_HOT:
        return _one_hot
    if m is CategorizationMethod.LOWER_BOUND:
        return _step_fn(1, 3, 5, 9, 12)
    if m is CategorizationMethod.MIDPOINT:
        return _step_fn(1.5, 3.5, 6.5, 10.5, 12)
    if m is CategorizationMethod.UPPER_BOUND:
        return _step_fn(2, 4, 8, 12, 12)
    if m is CategorizationMethod.MIDPOINT_WITH_INHALER:
        top = _top_value_midpoint_with_inhaler(df, top_category_fallback)
        return _step_fn(1.5, 3.5, 6.5, 10.5, top)
    if m is CategorizationMethod.UPPER_BOUND_WITH_INHALER:
        top = _top_value_upper_bound_with_inhaler(df, top_category_fallback)
        return _step_fn(2, 4, 8, 12, top)
    raise ValueError(f"Invalid categorization method: {method!r}")  # pragma: no cover


def categorize_columns(
    df: pd.DataFrame,
    method: MethodLike,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
) -> "tuple[pd.Series, pd.Series]":
    """Categorised (device, self-report) columns.

    The device column holds counts and goes through the step function. The
    self-report column holds questionnaire codes; the code meaning "12 or
    more" (``definitions.ACTIVE.self_report_top_code``) is a band label and
    goes to the top category, i.e. whatever the step function returns for
    an unbounded count. Under ``definitions.V1`` it is range-checked like a
    count, which puts it in the 9-to-12 band.
    """
    fn = build_categorize_fn(df, method, top_category_fallback=top_category_fallback)
    device = df["inhaler_usage"].apply(fn)
    report = df["daily_relief_inhaler"].apply(fn)
    top_code = definitions.ACTIVE.self_report_top_code
    if top_code is not None:
        is_top = df["daily_relief_inhaler"] == top_code
        if is_top.any():
            report = report.astype(float)
            report[is_top] = fn(float("inf"))
    return device, report


def categorize_inhaler_usage(
    df: pd.DataFrame,
    method: MethodLike,
    *,
    top_category_fallback: Optional[float] = DEFAULT_TOP_CATEGORY_FALLBACK,
) -> pd.DataFrame:
    """Return a copy of ``df`` with both usage columns categorised (see :func:`categorize_columns`)."""
    out = df.copy()
    out["inhaler_usage"], out["daily_relief_inhaler"] = categorize_columns(
        df, method, top_category_fallback=top_category_fallback)
    return out


def filter_zero_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where both the device count and the self-report are zero.

    Applied *after* categorisation in every historical code path.
    """
    return df[(df["inhaler_usage"] > 0) | (df["daily_relief_inhaler"] > 0)]
