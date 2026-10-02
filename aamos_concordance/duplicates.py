"""Duplicate device rows: seven readings, applied identically to the observed data and the null.

About 17 % of the assessed patients' smart-inhaler rows (490 of 2,863) repeat
another row exactly (same patient, day, time and medication name). Most of
them (316) sit on a timestamp ending in ``:00``, in records that switch to
minute resolution; the rest repeat a timestamp to the second. Whether a
repeat is a second puff or an artefact of syncing cannot be told from the
data, so the analysis carries several readings as a world axis:

* ``artefact`` (primary): every exact duplicate is an artefact and is
  dropped. This is what v1's observed analysis did; v1's Cloud Run null kept
  them, so v1 compared a deduplicated observed statistic with a null built
  from duplicated data. v2 applies the reading to both.
* ``real``: every row is a puff.
* ``resolution``: a repeat at minute resolution (``:00`` seconds) is a
  distinct puff within the same minute that the device could not separate;
  a repeat to the second is an artefact. A genuine second-resolution
  timestamp ends in ``:00`` one time in sixty, so a few artefacts are kept.
* ``anchor30``, ``chain30``, ``anchor60``, ``chain60``: rows close in time
  are one dosing event, not separate puffs. Per patient and medication, in
  time order, a row is kept only if it is more than 30 (or 60) seconds after
  the last *kept* row (``anchor``), or after the *previous* row (``chain``,
  so a run of close rows collapses to its first however long it lasts).
  Exact repeats (0 s apart) are always dropped. These count dosing events,
  a lower bound on puffs: they merge genuine quick double puffs too (939's
  morning and evening pairs, 294's 1-5 s pairs). Rows stored at minute
  resolution are 0 or at least 60 s apart, so only the 60 s windows can merge
  stamps in adjacent minutes.

The questionnaire has no exact duplicate rows; it is deduplicated regardless.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# reading -> (rule, window in seconds) for the time-window readings
WINDOW_READINGS: Dict[str, Tuple[str, int]] = {
    "anchor30": ("anchor", 30), "chain30": ("chain", 30),
    "anchor60": ("anchor", 60), "chain60": ("chain", 60),
}
DUPLICATE_READINGS = ("artefact", "real", "resolution", *WINDOW_READINGS)
DEFAULT_DUPLICATES = "artefact"


def _at_minute_resolution(df: pd.DataFrame) -> pd.Series:
    return df["time"].astype(str).str.endswith(":00")


def _seconds(df: pd.DataFrame) -> np.ndarray:
    """Absolute time of each row in seconds (day offset plus time of day)."""
    tod = pd.to_timedelta(df["time"].astype(str)).dt.total_seconds().to_numpy()
    return df["date"].to_numpy(dtype=float) * 86400.0 + tod


def _window_keep(seconds: np.ndarray, rule: str, window: float) -> np.ndarray:
    """Keep mask for time-ordered rows under the anchored or chained window rule."""
    keep = np.zeros(len(seconds), dtype=bool)
    if len(seconds) == 0:
        return keep
    if rule == "chain":
        keep[0] = True
        keep[1:] = np.diff(seconds) > window
        return keep
    last = None
    for j, t in enumerate(seconds):
        if last is None or t - last > window:
            keep[j] = True
            last = t
    return keep


def _apply_window(inhaler_df: pd.DataFrame, rule: str, window: float) -> pd.DataFrame:
    keep = pd.Series(False, index=inhaler_df.index)
    secs = pd.Series(_seconds(inhaler_df), index=inhaler_df.index) if len(inhaler_df) else pd.Series(dtype=float)
    for _, g in inhaler_df.groupby(["user_key", "name"], sort=False):
        order = secs.loc[g.index].sort_values(kind="stable")
        keep.loc[order.index] = _window_keep(order.to_numpy(), rule, window)
    return inhaler_df[keep.to_numpy()]


def apply_duplicate_reading(inhaler_df: pd.DataFrame, reading: str) -> pd.DataFrame:
    """Device rows under a duplicate reading, in their original order (the first of each group is kept)."""
    if reading not in DUPLICATE_READINGS:
        raise ValueError(f"duplicates must be one of {DUPLICATE_READINGS}, got {reading!r}")
    if reading == "real":
        return inhaler_df
    if reading in WINDOW_READINGS:
        return _apply_window(inhaler_df, *WINDOW_READINGS[reading])
    repeat = inhaler_df.duplicated(keep="first")
    if reading == "artefact":
        return inhaler_df[~repeat]
    return inhaler_df[~repeat | _at_minute_resolution(inhaler_df)]


def duplicate_counts(inhaler_df: pd.DataFrame) -> pd.DataFrame:
    """Per patient: rows, exact repeats, and how many of those are at minute resolution."""
    repeat = inhaler_df.duplicated(keep="first")
    minute = _at_minute_resolution(inhaler_df)
    g = inhaler_df.assign(_r=repeat, _rm=repeat & minute, _rs=repeat & ~minute).groupby("user_key")
    out = pd.DataFrame({
        "n_rows": g.size(),
        "n_repeats": g["_r"].sum().astype(int),
        "n_repeats_minute_resolution": g["_rm"].sum().astype(int),
        "n_repeats_second_resolution": g["_rs"].sum().astype(int),
    })
    for reading in DUPLICATE_READINGS:
        out[f"n_rows_{reading}"] = apply_duplicate_reading(inhaler_df, reading).groupby("user_key").size()
    return out.reset_index().rename(columns={"user_key": "patient_id"})
