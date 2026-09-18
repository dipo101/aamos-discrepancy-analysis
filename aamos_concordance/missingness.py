"""Missingness diagnostics (v2 items 19 and 20).

Item 19, :func:`cell_counts`: for each patient and each span, the join's
rows classified by device count (zero / positive) and self-report code
(zero / positive), i.e. Kevin Tsang's KT14 2x2, plus the count that cannot
enter a correlation at all: days inside the span with device records but no
questionnaire. Also the span boundaries and how many raw entries the span
trimmed. Counts depend on the window configuration; the default is the
calendar-day, same-day window, which is the most literal "did the device
record anything on the day of the questionnaire".

Item 20, :func:`nonresponse_check`: is questionnaire non-response related
to device use? Within each patient's Q span, every calendar day is a
response day or a non-response day; device puffs per day (from the raw
records) are compared between the two with a Mann-Whitney U test, and the
fraction of days with any device use is reported for each. Temporal
clustering of non-response is measured with the Wald-Wolfowitz runs test on
the response/non-response sequence (fewer runs than expected means
non-response comes in blocks). Descriptive only: nothing here imputes.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

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


def _runs_test(seq: np.ndarray) -> Dict[str, float]:
    """Wald-Wolfowitz runs test on a binary sequence; z < 0 means fewer runs than expected (clustering)."""
    seq = np.asarray(seq).astype(bool)
    n1, n0 = int(seq.sum()), int((~seq).sum())
    n = n1 + n0
    if n1 == 0 or n0 == 0 or n < 3:
        return {"n_runs": float(1 if n else 0), "expected_runs": np.nan, "runs_z": np.nan, "runs_p": np.nan}
    runs = 1 + int(np.sum(seq[1:] != seq[:-1]))
    mu = 1 + 2 * n1 * n0 / n
    var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n * n * (n - 1))
    z = (runs - mu) / sqrt(var) if var > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {"n_runs": float(runs), "expected_runs": mu, "runs_z": z, "runs_p": p}


def nonresponse_check(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    patients: List[int],
) -> pd.DataFrame:
    """One row per patient: device use on response vs non-response days, and clustering of non-response."""
    rows: List[Dict] = []
    for pid in patients:
        q = questionnaire_df[questionnaire_df.user_key == pid]
        inh = inhaler_df[inhaler_df.user_key == pid]
        if len(q) == 0:
            continue
        sp = patient_spans(q, inh)["Q"]
        days = np.arange(sp.start, sp.end + 1)
        responded = np.isin(days, q["date"].unique())
        puffs_by_day = inh.groupby("date").size()
        puffs = np.array([int(puffs_by_day.get(d, 0)) for d in days], dtype=float)

        resp, nonresp = puffs[responded], puffs[~responded]
        if len(nonresp) and len(resp):
            try:
                u_p = float(stats.mannwhitneyu(resp, nonresp, alternative="two-sided").pvalue)
            except ValueError:
                u_p = np.nan
        else:
            u_p = np.nan
        rows.append({
            "patient_id": pid,
            "span_days": int(len(days)),
            "response_days": int(responded.sum()),
            "nonresponse_days": int((~responded).sum()),
            "nonresponse_rate": float((~responded).mean()),
            "mean_puffs_response_days": float(resp.mean()) if len(resp) else np.nan,
            "mean_puffs_nonresponse_days": float(nonresp.mean()) if len(nonresp) else np.nan,
            "frac_days_with_device_use_response": float((resp > 0).mean()) if len(resp) else np.nan,
            "frac_days_with_device_use_nonresponse": float((nonresp > 0).mean()) if len(nonresp) else np.nan,
            "mannwhitney_p": u_p,
            **_runs_test(responded),
        })
    return pd.DataFrame(rows)


def pooled_nonresponse_summary(check: pd.DataFrame) -> Dict[str, float]:
    """Across patients: how many show more device use on non-response days, and how many cluster."""
    has_both = check.dropna(subset=["mean_puffs_response_days", "mean_puffs_nonresponse_days"])
    more_on_nonresp = has_both["mean_puffs_nonresponse_days"] > has_both["mean_puffs_response_days"]
    return {
        "n_patients": int(len(check)),
        "n_with_nonresponse_days": int((check["nonresponse_days"] > 0).sum()),
        "median_nonresponse_rate": float(check["nonresponse_rate"].median()),
        "n_more_device_use_on_nonresponse_days": int(more_on_nonresp.sum()),
        "n_mannwhitney_p_below_0_05": int((check["mannwhitney_p"] < 0.05).sum()),
        "n_clustered_runs_p_below_0_05": int(((check["runs_p"] < 0.05) & (check["runs_z"] < 0)).sum()),
    }
