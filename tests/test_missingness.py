"""Spans (item 8 groundwork) and the cell-count diagnostic (item 19)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance.missingness import CALENDAR_SAME_DAY, cell_counts
from aamos_concordance.spans import Span, apply_span, patient_spans
from tests.synthetic import make_dataset


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------

def _frames(q_dates, d_dates, pid=1):
    q = pd.DataFrame({"user_key": pid, "date": q_dates, "time": "10:00:00", "daily_relief_inhaler": 1.0})
    inh = pd.DataFrame({"user_key": pid, "date": d_dates, "time": "09:00:00", "name": "V"})
    return q, inh


def test_patient_spans_four_variants():
    q, inh = _frames([3, 5, 9], [1, 2, 6])
    s = patient_spans(q, inh)
    assert s == {"Q": Span(3, 9), "D": Span(1, 6), "union": Span(1, 9), "intersection": Span(3, 6)}
    assert Span(3, 9).n_days == 7


def test_patient_spans_degenerate_cases():
    q, inh = _frames([3, 5], [10, 12])
    s = patient_spans(q, inh)
    assert s["intersection"] is None and s["union"] == Span(3, 12)
    q, inh = _frames([3, 5], [])
    s = patient_spans(q, inh)
    assert s == {"Q": Span(3, 5), "D": None, "union": Span(3, 5), "intersection": None}
    q, inh = _frames([], [1, 2])
    s = patient_spans(q, inh)
    assert s["Q"] is None and s["union"] == Span(1, 2) and s["intersection"] is None


def test_apply_span_trims_both_frames_and_drops_patient_when_unformable():
    q, inh = _frames([3, 5, 9], [1, 2, 6, 11])
    qq, ii, sp = apply_span(q, inh, "D")
    assert sp == Span(1, 11) and list(qq.date) == [3, 5, 9] and list(ii.date) == [1, 2, 6, 11]
    qq, ii, sp = apply_span(q, inh, "intersection")
    assert sp == Span(3, 9) and list(qq.date) == [3, 5, 9] and list(ii.date) == [6]
    qq, ii, sp = apply_span(q, inh, "Q")
    assert sp == Span(3, 9) and len(qq) == 3 and list(ii.date) == [6]
    qq, ii, sp = apply_span(q, inh, "union")
    assert len(qq) == 3 and len(ii) == 4
    q2, inh2 = _frames([3, 5], [10, 12])
    qq, ii, sp = apply_span(q2, inh2, "intersection")
    assert sp is None and len(qq) == 0 and len(ii) == 0
    with pytest.raises(ValueError):
        apply_span(q, inh, "nope")


def test_span_Q_keeps_every_questionnaire_row_on_synthetic_data():
    q, inh = make_dataset(1)
    for pid in q.user_key.unique():
        qq, ii = q[q.user_key == pid], inh[inh.user_key == pid]
        trimmed_q, _, _ = apply_span(qq, ii, "Q")
        assert len(trimmed_q) == len(qq)


# --------------------------------------------------------------------------
# Cell counts
# --------------------------------------------------------------------------

def test_cell_counts_hand_computed():
    # days 1..5 questionnaire (codes 0,2,0,3,1); device on days 1,2,2,4,7
    q = pd.DataFrame({"user_key": 1, "date": [1, 2, 3, 4, 5], "time": "12:00:00", "daily_relief_inhaler": [0, 2, 0, 3, 1]})
    inh = pd.DataFrame({"user_key": 1, "date": [1, 2, 2, 4, 7], "time": "08:00:00", "name": "V"})
    c = cell_counts(q, inh, [1], window=CALENDAR_SAME_DAY).set_index("span")
    qrow = c.loc["Q"]
    # same-day counts: d1=1,q0 ; d2=2,q2 ; d3=0,q0 ; d4=1,q3 ; d5=0,q1
    assert qrow["rec_pos_q_zero"] == 1 and qrow["rec_pos_q_pos"] == 2 and qrow["rec_zero_q_zero"] == 1 and qrow["rec_zero_q_pos"] == 1
    assert qrow["n_rows"] == 5 and qrow["span_start"] == 1 and qrow["span_end"] == 5
    assert qrow["device_days_without_questionnaire"] == 0  # day 7 is outside span Q
    assert qrow["device_records_trimmed"] == 1
    urow = c.loc["union"]
    assert urow["span_end"] == 7 and urow["device_days_without_questionnaire"] == 1 and urow["device_puffs_on_days_without_questionnaire"] == 1
    drow = c.loc["D"]
    assert drow["span_start"] == 1 and drow["span_end"] == 7 and drow["n_rows"] == 5
    assert c.loc["intersection"]["n_rows"] == 5


def test_cell_counts_cells_sum_to_rows_on_synthetic_data():
    q, inh = make_dataset(2)
    pats = sorted(q.user_key.unique())
    c = cell_counts(q, inh, pats)
    c = c[c.n_rows > 0]
    total = c[["rec_pos_q_pos", "rec_pos_q_zero", "rec_zero_q_pos", "rec_zero_q_zero"]].sum(axis=1)
    assert (total == c["n_rows"]).all()
    assert set(c["span"]) <= {"Q", "D", "union", "intersection"}
    # Q keeps all questionnaire rows; union has at least as many device records as any other span
    qn = c[c.span == "Q"].set_index("patient_id")["n_rows"]
    un = c[c.span == "union"].set_index("patient_id")
    assert (un["n_rows"] == qn.loc[un.index]).all()


# --------------------------------------------------------------------------
# Real data: pin the committed table
# --------------------------------------------------------------------------


def test_committed_cell_counts_match_recomputation(data_dir):
    from aamos_concordance import load_raw
    from aamos_concordance.worlds import diagnostics_dir
    raw = load_raw(drop_duplicates=True)
    committed = pd.read_csv(diagnostics_dir(create=False) / "cell_counts_calendar_same_day.csv")
    pats = sorted(committed.patient_id.unique())
    fresh = cell_counts(raw.questionnaire, raw.inhaler, pats)
    pd.testing.assert_frame_equal(committed.sort_values(["patient_id", "span"]).reset_index(drop=True),
                                  fresh.sort_values(["patient_id", "span"]).reset_index(drop=True), check_dtype=False)
    # the finding: device-only days are large for 473 and 917 under span Q
    qq = committed[committed.span == "Q"].set_index("patient_id")
    assert qq.loc[473, "device_days_without_questionnaire"] == 40 and qq.loc[917, "device_days_without_questionnaire"] == 20
