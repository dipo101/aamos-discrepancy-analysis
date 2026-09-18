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

from typing import Dict, Iterable, List, Tuple

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


def dedupe_combinations(
    timestamp_windows: Iterable[int] = TIMESTAMP_WINDOWS,
    use_daily_max_options: Iterable[bool] = USE_DAILY_MAX_OPTIONS,
    use_calendar_options: Iterable[bool] = USE_CALENDAR_OPTIONS,
    filter_zero_options: Iterable[bool] = FILTER_ZERO_OPTIONS,
    categorization_methods: Iterable = CATEGORIZATION_METHODS,
) -> List[Tuple]:
    """Nested-loop product of the five axes with redundant entries removed.

    Returns ``(window, daily_max, calendar, filter_zeros, cat_method)`` tuples
    in loop order, keeping the first occurrence of each effective
    configuration. ``cat_method`` is passed through unchanged, so callers
    may supply strings or enum members.
    """
    combinations: List[Tuple] = []
    seen = set()
    for window in timestamp_windows:
        for daily_max in use_daily_max_options:
            for calendar in use_calendar_options:
                for filter_zeros in filter_zero_options:
                    for cat_method in categorization_methods:
                        key = effective_key(window, daily_max, calendar, filter_zeros, cat_method)
                        if key in seen:
                            continue
                        seen.add(key)
                        combinations.append((window, daily_max, calendar, filter_zeros, cat_method))
    return combinations


def generate_param_combinations() -> List[Dict]:
    """Return the 132 unique configurations as dicts, in canonical order.

    The order is the nested-loop order of the job-worker that produced the
    published permutation results. Downstream code that indexes by position
    (``config_idx``) relies on it.
    """
    combinations = [
        {
            "timestamp_window": window,
            "use_daily_max_windows": daily_max,
            "use_calendar_days": calendar,
            "filter_out_zero_usage": filter_zeros,
            "categorization_method": cat_method,
        }
        for (window, daily_max, calendar, filter_zeros, cat_method) in dedupe_combinations()
    ]
    assert len(combinations) == N_UNIQUE_COMBINATIONS, len(combinations)
    return combinations


# --------------------------------------------------------------------------
# Effective configurations under a rank-based statistic
# --------------------------------------------------------------------------
#
# Spearman's rho depends only on ranks. Every categorisation method maps the
# self-report codes {0, 1, 3, 5, 9, 12} monotonically, so the self-report
# ranks are the same under every method; categorisation can only change a
# Spearman result through the *device* column, by which counts it merges into
# the same bin. Four methods bin device counts into the same six groups with
# strictly increasing values and are therefore rank-identical; upper_bound
# merges the 9-12 and >12 groups; one_hot merges everything above zero.
# Separately, the fixed 24-hour chunk starting 24 h before the questionnaire
# is the rolling 24-hour window except for a record at the questionnaire's
# exact second. The v1 table confirms both collapses (tests/test_configs.py).
#
# Under Pearson the categorisation values matter, so nothing collapses.

SPEARMAN_METHOD_CLASS = {
    "lower_bound": "six_bins_monotone",
    "midpoint": "six_bins_monotone",
    "midpoint_with_inhaler": "six_bins_monotone",
    "upper_bound_with_inhaler": "six_bins_monotone",
    "upper_bound": "top_two_bins_merged",
    "one_hot": "binary",
}


def _window_key(window: int, daily_max: bool, calendar: bool):
    if calendar:
        return ("calendar", window // 24)
    if daily_max and window == 24:
        return ("rolling", 24)  # chunk starting 24 h back == rolling 24 h (bar the exact-second case)
    return ("fixed_chunk" if daily_max else "rolling", window)


def spearman_equivalence_key(cfg: Dict):
    """Configurations sharing this key give the same Spearman result on any data."""
    return (
        _window_key(cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"]),
        cfg["filter_out_zero_usage"],
        SPEARMAN_METHOD_CLASS[cfg["categorization_method"]],
    )


def config_equivalence_classes(correlation_type: str = "spearman") -> Dict[Tuple, List[int]]:
    """Map each equivalence class to the indices (into generate_param_combinations()) it contains."""
    combos = generate_param_combinations()
    classes: Dict[Tuple, List[int]] = {}
    for idx, cfg in enumerate(combos):
        if correlation_type == "spearman":
            key = spearman_equivalence_key(cfg)
        elif correlation_type == "pearson":
            key = (idx,)
        else:
            raise ValueError(f"unknown correlation_type {correlation_type!r}")
        classes.setdefault(key, []).append(idx)
    return classes


def effective_config_indices(correlation_type: str = "spearman") -> List[int]:
    """Indices of one representative per equivalence class, in canonical order.

    60 for Spearman (10 windows x 2 zero-filter x 3 method classes), 132 for Pearson.
    """
    return sorted(members[0] for members in config_equivalence_classes(correlation_type).values())


N_SPEARMAN_EFFECTIVE = 60
