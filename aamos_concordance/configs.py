"""The 132-configuration multiverse.

Five axes are varied:

* ``timestamp_window`` in {12, 24, 36, 48} hours
* ``use_daily_max_windows`` (fixed 24-hour chunk vs rolling window)
* ``use_calendar_days`` (calendar-day boundaries instead of timestamps)
* ``filter_out_zero_usage`` (drop rows where both signals are zero)
* ``categorization_method`` (six methods)

The raw product is 192 combinations. Sixty are redundant: under calendar
days ``use_daily_max_windows`` has no effect, and the 24- and 36-hour
windows both map to a one-day offset. De-duplication keeps the first
occurrence in nested-loop order, so the surviving calendar-day entries carry
``use_daily_max_windows=False`` and ``timestamp_window=24`` for the one-day
offset.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

TIMESTAMP_WINDOWS: Tuple[int, ...] = (12, 24, 36, 48)
USE_DAILY_MAX_OPTIONS: Tuple[bool, ...] = (False, True)
USE_CALENDAR_OPTIONS: Tuple[bool, ...] = (False, True)
FILTER_ZERO_OPTIONS: Tuple[bool, ...] = (False, True)
CATEGORIZATION_METHODS: Tuple[str, ...] = (
    "one_hot",
    "lower_bound",
    "midpoint",
    "midpoint_with_inhaler",
    "upper_bound",
    "upper_bound_with_inhaler",
)

N_RAW_COMBINATIONS = 192
N_UNIQUE_COMBINATIONS = 132


def effective_key(window: int, daily_max: bool, calendar: bool, filter_zeros: bool, cat_method: str):
    """Canonical identity of a configuration, collapsing the redundant ones."""
    if calendar:
        return ("calendar", window // 24, filter_zeros, cat_method)
    mode = "fixed_chunk" if daily_max else "rolling"
    return (mode, window, filter_zeros, cat_method)


def generate_param_combinations() -> List[Dict]:
    """Return the 132 unique configurations as dicts, in canonical order.

    The order is the nested-loop order of the job-worker that produced the
    published permutation results. Downstream code that indexes by position
    (``config_idx``) relies on it.
    """
    combinations: List[Dict] = []
    seen = set()
    for window in TIMESTAMP_WINDOWS:
        for daily_max in USE_DAILY_MAX_OPTIONS:
            for calendar in USE_CALENDAR_OPTIONS:
                for filter_zeros in FILTER_ZERO_OPTIONS:
                    for cat_method in CATEGORIZATION_METHODS:
                        key = effective_key(window, daily_max, calendar, filter_zeros, cat_method)
                        if key in seen:
                            continue
                        seen.add(key)
                        combinations.append({
                            "timestamp_window": window,
                            "use_daily_max_windows": daily_max,
                            "use_calendar_days": calendar,
                            "filter_out_zero_usage": filter_zeros,
                            "categorization_method": cat_method,
                        })
    assert len(combinations) == N_UNIQUE_COMBINATIONS, len(combinations)
    return combinations
