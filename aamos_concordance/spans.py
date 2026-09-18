"""Per-patient timespans: which raw entries are believed.

The v2 plan varies, per patient, the interval of days inside which raw
entries are used at all. Entries outside it are removed from *both* raw
frames before the window join (span as "trusted data", not as a filter on
windows). The four spans are derived from the records themselves, as Kevin
Tsang's KT13 comment describes ("first questionnaire data per patient ...
first smart inhaler record per patient"):

* ``Q``             first to last questionnaire entry
* ``D``             first to last device record
* ``union``         min of the two starts to max of the two ends
* ``intersection``  max of the two starts to min of the two ends

``Q`` is the v1 behaviour: every questionnaire row lies inside it, and no
device record is ever a join *row*, so trimming device records outside Q
only affects the counts of windows at the edges. The patient-info file also
carries start/end dates per device; those describe the period the device
was assigned, which for some patients extends well beyond the last record
(454: assigned to day 38, last record on day 1), so they are reported
alongside but not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from .worlds import SPANS


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def n_days(self) -> int:
        return max(0, self.end - self.start + 1)

    def contains(self, dates: pd.Series) -> pd.Series:
        return (dates >= self.start) & (dates <= self.end)


def patient_spans(questionnaire_df: pd.DataFrame, inhaler_df: pd.DataFrame) -> Dict[str, Optional[Span]]:
    """The four spans for one patient's raw frames (already filtered to the patient).

    A span is ``None`` when it cannot be formed: ``D`` (and hence ``union``
    is just ``Q``, and ``intersection`` is ``None``) for a patient with no
    device records; everything for a patient with no questionnaire rows.
    ``intersection`` is also ``None`` when the two periods do not overlap.
    """
    q = Span(int(questionnaire_df["date"].min()), int(questionnaire_df["date"].max())) if len(questionnaire_df) else None
    d = Span(int(inhaler_df["date"].min()), int(inhaler_df["date"].max())) if len(inhaler_df) else None
    if q is None:
        return {"Q": None, "D": d, "union": d, "intersection": None}
    if d is None:
        return {"Q": q, "D": None, "union": q, "intersection": None}
    union = Span(min(q.start, d.start), max(q.end, d.end))
    lo, hi = max(q.start, d.start), min(q.end, d.end)
    inter = Span(lo, hi) if lo <= hi else None
    return {"Q": q, "D": d, "union": union, "intersection": inter}


def apply_span(
    questionnaire_df: pd.DataFrame,
    inhaler_df: pd.DataFrame,
    span: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Span]]:
    """Trim both raw frames of one patient to the named span.

    Returns ``(questionnaire, inhaler, span_used)``. When the span cannot be
    formed for this patient (see :func:`patient_spans`) both frames come
    back empty and ``span_used`` is ``None``, so the patient drops out of
    that world rather than being silently analysed under a different span.
    """
    if span not in SPANS:
        raise ValueError(f"span must be one of {SPANS}, got {span!r}")
    sp = patient_spans(questionnaire_df, inhaler_df)[span]
    if sp is None:
        return questionnaire_df.iloc[0:0], inhaler_df.iloc[0:0], None
    return (
        questionnaire_df[sp.contains(questionnaire_df["date"])],
        inhaler_df[sp.contains(inhaler_df["date"])],
        sp,
    )
