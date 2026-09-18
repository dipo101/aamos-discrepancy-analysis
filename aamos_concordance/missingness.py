"""Missingness diagnostics (v2 item 19: cell counts per span).

Item 19, :func:`cell_counts`: for each patient and each span, the join's
rows classified by device count (zero / positive) and self-report code
(zero / positive), i.e. Kevin Tsang's KT14 2x2, plus the count that cannot
enter a correlation at all: days inside the span with device records but no
questionnaire. Also the span boundaries and how many raw entries the span
trimmed. Counts depend on the window configuration; the default is the
calendar-day, same-day window, which is the most literal "did the device
record anything on the day of the questionnaire".

Descriptive only: nothing here imputes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .join import join_questionnaire_with_inhaler
from .spans import apply_span, patient_spans
from .worlds import SPANS

# (timestamp_window, use_daily_max_windows, use_calendar_days)
CALENDAR_SAME_DAY = (12, False, True)
ROLLING_24H = (24, False, False)


def cell_counts(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    patients: List[int],
    *,
    window=CALENDAR_SAME_DAY,
    spans=SPANS,
) -> pd.DataFrame:
    """One row per (patient, span): KT14 cells, device-only days, span bounds, rows trimmed."""
    rows: List[Dict] = []
    for pid in patients:
        q_all = questionnaire_df[questionnaire_df.user_key == pid]
        i_all = inhaler_df[inhaler_df.user_key == pid]
        all_spans = patient_spans(q_all, i_all)
        for span in spans:
            sp = all_spans[span]
            row: Dict = {"patient_id": pid, "span": span,
                         "span_start": sp.start if sp else np.nan, "span_end": sp.end if sp else np.nan,
                         "span_days": sp.n_days if sp else 0}
            if sp is None:
                row.update({"n_rows": 0, "q_rows_trimmed": len(q_all), "device_records_trimmed": len(i_all)})
                rows.append(row)
                continue
            q, inh, _ = apply_span(q_all, i_all, span)
            joined = join_questionnaire_with_inhaler(q, inh, *window)
            rec_pos = joined["inhaler_usage"] > 0
            q_pos = joined["daily_relief_inhaler"] > 0
            q_dates = set(q["date"].unique())
            device_days = set(inh["date"].unique())
            row.update({
                "n_rows": int(len(joined)),
                "rec_pos_q_pos": int((rec_pos & q_pos).sum()),
                "rec_pos_q_zero": int((rec_pos & ~q_pos).sum()),
                "rec_zero_q_pos": int((~rec_pos & q_pos).sum()),
                "rec_zero_q_zero": int((~rec_pos & ~q_pos).sum()),
                "device_days_without_questionnaire": len(device_days - q_dates),
                "device_puffs_on_days_without_questionnaire": int(inh[~inh["date"].isin(q_dates)].shape[0]),
                "questionnaire_days_without_device": len(q_dates - device_days),
                "q_rows_trimmed": int(len(q_all) - len(q)),
                "device_records_trimmed": int(len(i_all) - len(inh)),
            })
            rows.append(row)
    return pd.DataFrame(rows)
