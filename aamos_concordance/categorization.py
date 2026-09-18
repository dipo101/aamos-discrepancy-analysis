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

``top_category_fallback`` controls what the two data-driven methods return
for the top category when the input frame has no usage at or above 12. The
pre-refactor copies disagreed on this: ``data_loader.py`` and the Cloud Run
job-worker returned NaN, while ``per_patient_correlation_analysis.py``
returned 12. Callers pass the value that reproduces their own historical
behaviour; unifying it is a separate, deliberate decision.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd


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


def build_categorize_fn(
    df: pd.DataFrame,
    method: MethodLike,
    *,
    top_category_fallback: Optional[float] = None,
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


def categorize_inhaler_usage(
    df: pd.DataFrame,
    method: MethodLike,
    *,
    top_category_fallback: Optional[float] = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with both usage columns mapped through ``method``."""
    fn = build_categorize_fn(df, method, top_category_fallback=top_category_fallback)
    out = df.copy()
    out["inhaler_usage"] = out["inhaler_usage"].apply(fn)
    out["daily_relief_inhaler"] = out["daily_relief_inhaler"].apply(fn)
    return out


def filter_zero_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where both the device count and the self-report are zero.

    Applied *after* categorisation in every historical code path.
    """
    return df[(df["inhaler_usage"] > 0) | (df["daily_relief_inhaler"] > 0)]
