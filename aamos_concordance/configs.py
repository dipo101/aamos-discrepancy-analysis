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

from . import definitions

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
# Spearman's rho depends only on ranks, so categorisation can change a
# Spearman result only by which values it merges into the same bin.
# lower_bound, midpoint and midpoint_with_inhaler map both columns into six
# groups with strictly increasing values (none, 1-2, 3-4, 5-8, 9-12, top) and
# are rank-identical. upper_bound merges the 9-12 and top groups; one_hot
# merges everything above zero. upper_bound_with_inhaler keeps the top group
# separate only when the frame has a device count above 12; otherwise its top
# value falls back to 12 and it merges like upper_bound. Under v1's
# definitions no self-report reached the top group, so that fallback never
# mattered and upper_bound_with_inhaler was rank-identical to the first
# three; with code 12 ("12 or more") in the top group (definitions.CURRENT) it
# is its own class. Separately, the fixed 24-hour chunk starting 24 h before
# the questionnaire covers the same interval as the rolling 24-hour window:
# identical by construction now that chunk ends are closed, and identical on
# AAMOS-00 even under v1's open end (no record falls on it).
#
# Under Pearson the categorisation values matter, so the six methods stay
# distinct; only the window duplicate (and, under B/C, the filter) collapses.

SPEARMAN_METHOD_CLASS = {
    "lower_bound": "six_bins_monotone",
    "midpoint": "six_bins_monotone",
    "midpoint_with_inhaler": "six_bins_monotone",
    "upper_bound_with_inhaler": "six_bins_or_top_merged",
    "upper_bound": "top_two_bins_merged",
    "one_hot": "binary",
}
SPEARMAN_METHOD_CLASS_V1 = {**SPEARMAN_METHOD_CLASS, "upper_bound_with_inhaler": "six_bins_monotone"}


def spearman_method_class(method: str) -> str:
    """The Spearman class of a categorisation method under the active definitions."""
    v1 = definitions.ACTIVE.self_report_top_code is None
    return (SPEARMAN_METHOD_CLASS_V1 if v1 else SPEARMAN_METHOD_CLASS)[method]


def _window_key(window: int, daily_max: bool, calendar: bool):
    if calendar:
        return ("calendar", window // 24)
    if daily_max and window == 24:
        return ("rolling", 24)  # chunk starting 24 h back == rolling 24 h (both ends closed)
    return ("fixed_chunk" if daily_max else "rolling", window)


def spearman_equivalence_key(cfg: Dict, absence_case: str = "A"):
    """Configurations sharing this key give the same Spearman result on any data.

    Under absence cases B and C the zero filter is forced off, so the filter
    flag does not distinguish configurations.
    """
    return (
        _window_key(cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"]),
        cfg["filter_out_zero_usage"] if absence_case == "A" else None,
        spearman_method_class(cfg["categorization_method"]),
    )


def pearson_equivalence_key(cfg: Dict, idx: int, absence_case: str = "A"):
    """Under Pearson the categorisation values matter, so the six methods stay distinct.

    What still collapses is independent of the correlation type: the window
    duplicate (identical joined counts) and, under B and C, the inert filter.
    """
    return (
        _window_key(cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"]),
        cfg["filter_out_zero_usage"] if absence_case == "A" else None,
        cfg["categorization_method"],
    )


def config_equivalence_classes(correlation_type: str = "spearman", absence_case: str = "A") -> Dict[Tuple, List[int]]:
    """Map each equivalence class to the indices (into generate_param_combinations()) it contains."""
    combos = generate_param_combinations()
    classes: Dict[Tuple, List[int]] = {}
    for idx, cfg in enumerate(combos):
        if correlation_type == "spearman":
            key = spearman_equivalence_key(cfg, absence_case)
        elif correlation_type == "pearson":
            key = pearson_equivalence_key(cfg, idx, absence_case)
        else:
            raise ValueError(f"unknown correlation_type {correlation_type!r}")
        classes.setdefault(key, []).append(idx)
    return classes


def effective_config_indices(correlation_type: str = "spearman", absence_case: str = "A") -> List[int]:
    """Indices of one representative per equivalence class, in canonical order.

    Case A: 80 for Spearman (10 windows x 2 zero-filter x 4 method classes; 60 with
    v1's 3 classes), 120 for Pearson (10 x 2 x 6). Cases B/C (filter inert): 40
    (v1: 30) for Spearman, 60 for Pearson.
    """
    return sorted(members[0] for members in config_equivalence_classes(correlation_type, absence_case).values())


N_SPEARMAN_EFFECTIVE = 80
N_SPEARMAN_EFFECTIVE_V1 = 60

# Windows that reach past the questionnaire answer: the fixed chunk starting
# 12 h back runs to 12 h after it, and calendar day 0 is the whole day of the
# answer. The question asks about "the past 24 hours", so the primary
# configuration set excludes them; ``effective_lookahead`` keeps them as a
# sensitivity (v1 included them).
LOOKAHEAD_WINDOWS = frozenset({("fixed_chunk", 12), ("calendar", 0)})


def is_lookahead(cfg: Dict) -> bool:
    return _window_key(cfg["timestamp_window"], cfg["use_daily_max_windows"], cfg["use_calendar_days"]) in LOOKAHEAD_WINDOWS


# all: the 132 (v1). effective: one representative per structural class,
# look-ahead windows excluded (primary). effective_lookahead: the same with
# the look-ahead windows kept.
CONFIG_SETS = ("all", "effective", "effective_lookahead")

OBSERVED_CONFIG_KEY = ["timestamp_window", "use_daily_max_windows", "use_calendar_days",
                       "filter_out_zero_usage_entries", "categorization_method"]


def config_indices_for(config_set: str, correlation_type: str, absence_case: str = "A") -> List[int]:
    """Indices (into generate_param_combinations()) making up a named configuration set."""
    if config_set == "all":
        return list(range(N_UNIQUE_COMBINATIONS))
    if config_set == "effective_lookahead":
        return effective_config_indices(correlation_type, absence_case)
    if config_set == "effective":
        combos = generate_param_combinations()
        return [i for i in effective_config_indices(correlation_type, absence_case) if not is_lookahead(combos[i])]
    raise ValueError(f"config_set must be one of {CONFIG_SETS}, got {config_set!r}")


def attach_config_idx(per_config):
    """Add ``config_idx`` (position in generate_param_combinations()) to an observed per-config table.

    The observed table (per_patient_correlation_analysis.py) keys configurations
    by their five parameters with ``filter_out_zero_usage_entries`` as the
    filter column name; the null table keys them by ``config_idx``. This is the
    bridge between the two.
    """
    combos = generate_param_combinations()
    lookup = {
        (c["timestamp_window"], c["use_daily_max_windows"], c["use_calendar_days"],
         c["filter_out_zero_usage"], c["categorization_method"]): i
        for i, c in enumerate(combos)
    }
    keys = list(zip(*(per_config[k] for k in OBSERVED_CONFIG_KEY)))
    out = per_config.copy()
    out["config_idx"] = [lookup[k] for k in keys]
    return out
