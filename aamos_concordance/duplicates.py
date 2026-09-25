"""Exact duplicate device rows: three readings, applied identically to the observed data and the null.

About 17 % of the assessed patients' smart-inhaler rows (490 of 2,863) repeat
another row exactly (same patient, day, time and medication name). Most of
them (316) sit on a timestamp ending in ``:00``, in records that switch to
minute resolution; the rest repeat a timestamp to the second. Whether a
repeat is a second puff or an artefact of syncing cannot be told from the
data, so the analysis carries three readings as a world axis:

* ``artefact`` (primary): every exact duplicate is an artefact and is
  dropped. This is what v1's observed analysis did; v1's Cloud Run null kept
  them, so v1 compared a deduplicated observed statistic with a null built
  from duplicated data. v2 applies the reading to both.
* ``real``: every row is a puff.
* ``resolution``: a repeat at minute resolution (``:00`` seconds) is a
  distinct puff within the same minute that the device could not separate;
  a repeat to the second is an artefact. A genuine second-resolution
  timestamp ends in ``:00`` one time in sixty, so a few artefacts are kept.

The questionnaire has no exact duplicate rows; it is deduplicated regardless.
"""

from __future__ import annotations

import pandas as pd

DUPLICATE_READINGS = ("artefact", "real", "resolution")
DEFAULT_DUPLICATES = "artefact"


def _at_minute_resolution(df: pd.DataFrame) -> pd.Series:
    return df["time"].astype(str).str.endswith(":00")


def apply_duplicate_reading(inhaler_df: pd.DataFrame, reading: str) -> pd.DataFrame:
    """Device rows under a duplicate reading (the first of each repeated group is always kept)."""
    if reading not in DUPLICATE_READINGS:
        raise ValueError(f"duplicates must be one of {DUPLICATE_READINGS}, got {reading!r}")
    if reading == "real":
        return inhaler_df
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
