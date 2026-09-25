"""Window join between daily questionnaire entries and smart-inhaler records.

For every questionnaire row the join counts the inhaler records that fall in
a time window anchored on that row. The three window families are:

* rolling (``use_calendar_days=False, use_daily_max_windows=False``):
  records with ``t - window <= timestamp <= t`` for questionnaire time ``t``;
* fixed 24-hour chunk (``use_daily_max_windows=True``):
  records with ``t - window <= timestamp <= t - window + 24h`` (v1 used
  ``<`` at the end; see :mod:`aamos_concordance.definitions`);
* calendar day (``use_calendar_days=True``):
  records whose ``date`` equals ``row.date - window // 24``.

The join is one-directional: rows come only from the questionnaire. An
inhaler record with no questionnaire window covering it contributes to
nothing. A questionnaire row with no records in its window gets a count of
zero. Both facts are load-bearing for the analysis and are documented in the
manuscript.

Timestamps are built exactly as the pre-refactor code built them: ``date``
is a day offset added to an arbitrary reference date, and ``time`` is parsed
with ``pd.to_datetime`` and combined with it.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import definitions

REFERENCE_DATE = pd.Timestamp("2000-01-01")


def _combine_date_time(row) -> pd.Timestamp:
    base_date = REFERENCE_DATE + pd.Timedelta(days=row["date"])
    time_obj = pd.to_datetime(row["time"]).time()
    return pd.Timestamp.combine(base_date.date(), time_obj)


def add_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``timestamp`` column built from ``date`` and ``time``."""
    out = df.copy()
    if len(out) == 0:
        out["timestamp"] = pd.Series(dtype="datetime64[ns]")
        return out
    out["timestamp"] = out.apply(_combine_date_time, axis=1)
    return out


def _window_mask(row, inhaler_df: pd.DataFrame, timestamp_window: int,
                 use_daily_max_windows: bool, use_calendar_days: bool):
    if use_calendar_days:
        days_back = timestamp_window // 24
        target_day = row["date"] - days_back
        return inhaler_df["date"] == target_day

    window_hours = pd.Timedelta(hours=timestamp_window)
    if use_daily_max_windows:
        window_start = row["timestamp"] - window_hours
        window_end = window_start + pd.Timedelta(hours=24)
        if definitions.ACTIVE.closed_chunk_end:
            return (inhaler_df["timestamp"] >= window_start) & (inhaler_df["timestamp"] <= window_end)
        return (inhaler_df["timestamp"] >= window_start) & (inhaler_df["timestamp"] < window_end)

    return (inhaler_df["timestamp"] <= row["timestamp"]) & (
        inhaler_df["timestamp"] >= row["timestamp"] - window_hours
    )


def join_questionnaire_with_inhaler(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    timestamp_window: int,
    use_daily_max_windows: bool,
    use_calendar_days: bool,
    *,
    empty_inhaler_shortcut: bool = True,
) -> pd.DataFrame:
    """Join one patient's questionnaire rows with that patient's inhaler records.

    Both frames must already be filtered to a single patient. Returns a copy
    of ``questionnaire_df`` with an ``inhaler_usage`` column holding the
    record count in each row's window. Inputs are not modified.

    ``empty_inhaler_shortcut`` reproduces a historical difference between
    callers. When true and ``inhaler_df`` is empty, the function returns the
    questionnaire with ``inhaler_usage = 0`` and *no* ``timestamp`` column,
    which is what the per-patient script and the job-worker did. When false
    the general path runs, which produces the same zeros but also adds the
    ``timestamp`` column, as ``data_loader.py`` did.
    """
    if empty_inhaler_shortcut and len(inhaler_df) == 0:
        out = questionnaire_df.copy()
        out["inhaler_usage"] = 0
        return out

    inhaler_ts = add_timestamps(inhaler_df)
    out = add_timestamps(questionnaire_df)

    def aggregate_window(row):
        mask = _window_mask(row, inhaler_ts, timestamp_window, use_daily_max_windows, use_calendar_days)
        return len(inhaler_ts[mask])

    out["inhaler_usage"] = out.apply(aggregate_window, axis=1)
    return out


def join_multi_patient(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    timestamp_window: int,
    use_daily_max_windows: bool,
    use_calendar_days: bool,
) -> pd.DataFrame:
    """Join frames containing several patients, matching on ``user_key``.

    Equivalent to running :func:`join_questionnaire_with_inhaler` per patient
    (without the empty-inhaler shortcut) and reassembling the result in the
    original row order of ``questionnaire_df``. This is what the original
    ``data_loader.py`` computed with a per-row user mask.
    """
    if len(questionnaire_df) == 0:
        out = add_timestamps(questionnaire_df)
        out["inhaler_usage"] = pd.Series(dtype="int64")
        return out

    # Work by row position rather than by index label so that a non-unique
    # index is handled the same way the row-wise original did.
    pieces = []
    for user_key, rows in questionnaire_df.groupby("user_key", sort=False).indices.items():
        q_part = questionnaire_df.iloc[rows]
        i_part = inhaler_df[inhaler_df["user_key"] == user_key]
        joined = join_questionnaire_with_inhaler(
            q_part, i_part, timestamp_window, use_daily_max_windows, use_calendar_days,
            empty_inhaler_shortcut=False,
        )
        joined["_pos"] = rows
        pieces.append(joined)
    return pd.concat(pieces).sort_values("_pos", kind="stable").drop(columns="_pos")
