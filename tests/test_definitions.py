"""The v2 measurement definitions (closed chunk end, self-report code 12) and the v1 switch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance import definitions
from aamos_concordance.categorization import CategorizationMethod, categorize_columns
from aamos_concordance.definitions import CURRENT, V1, use_definitions
from aamos_concordance.join import join_questionnaire_with_inhaler

# Band codes of daily_relief_inhaler: none, 1-2, 3-4, 5-8, 9-12, "12 or more"
CODES = [0, 1, 3, 5, 9, 12]


def _frame(device=(0, 2, 4, 8, 12, 30)):
    return pd.DataFrame({"inhaler_usage": list(device), "daily_relief_inhaler": CODES})


def test_use_definitions_restores_on_exit_and_on_error():
    assert definitions.ACTIVE == CURRENT
    with use_definitions(V1):
        assert definitions.ACTIVE == V1
    with pytest.raises(RuntimeError):
        with use_definitions(V1):
            raise RuntimeError
    assert definitions.ACTIVE == CURRENT


@pytest.mark.parametrize("method,expected", [
    (CategorizationMethod.LOWER_BOUND, [0, 1, 3, 5, 9, 12]),
    (CategorizationMethod.MIDPOINT, [0, 1.5, 3.5, 6.5, 10.5, 12]),
    (CategorizationMethod.UPPER_BOUND, [0, 2, 4, 8, 12, 12]),
    (CategorizationMethod.ONE_HOT, [0, 1, 1, 1, 1, 1]),
    # hybrids: the top category is data-driven from the device counts >= 12 (here 12 and 30)
    (CategorizationMethod.MIDPOINT_WITH_INHALER, [0, 1.5, 3.5, 6.5, 10.5, 21.0]),
    (CategorizationMethod.UPPER_BOUND_WITH_INHALER, [0, 2, 4, 8, 12, 3 * np.std([12, 30], ddof=1) + 21.0]),
])
def test_code_12_is_the_top_band(method, expected):
    _, report = categorize_columns(_frame(), method)
    assert np.allclose(report.to_numpy(float), expected)


def test_code_12_uses_the_fallback_when_no_device_count_reaches_12():
    _, report = categorize_columns(_frame(device=(0, 1, 2, 3, 4, 5)), CategorizationMethod.MIDPOINT_WITH_INHALER)
    assert report.iloc[-1] == 12


def test_v1_puts_code_12_in_the_9_to_12_band():
    with use_definitions(V1):
        _, report = categorize_columns(_frame(), CategorizationMethod.MIDPOINT)
    assert report.iloc[-1] == 10.5


def test_device_counts_are_unaffected():
    f = _frame()
    for defs in (CURRENT, V1):
        with use_definitions(defs):
            device, _ = categorize_columns(f, CategorizationMethod.MIDPOINT)
        assert device.tolist() == [0, 1.5, 3.5, 6.5, 10.5, 12]


def _records_at(offsets_h, anchor="2000-01-11 09:00:00"):
    ts = pd.Timestamp(anchor) + pd.to_timedelta(offsets_h, unit="h")
    return pd.DataFrame({"user_key": 1, "date": (ts.normalize() - pd.Timestamp("2000-01-01")).days,
                         "time": ts.strftime("%H:%M:%S"), "name": "V"})


def test_fixed_chunk_end_is_closed():
    q = pd.DataFrame({"user_key": [1], "date": [10], "time": ["09:00:00"], "daily_relief_inhaler": [1]})
    # chunk starting 48 h back ends exactly 24 h back; one record on each end, one just past the end
    inh = _records_at([-48, -24, -23.99])
    with use_definitions(V1):
        v1 = join_questionnaire_with_inhaler(q, inh, 48, True, False)["inhaler_usage"].iloc[0]
    v2 = join_questionnaire_with_inhaler(q, inh, 48, True, False)["inhaler_usage"].iloc[0]
    assert (v1, v2) == (1, 2)


def test_chunk_24_is_the_rolling_24_hour_window():
    q = pd.DataFrame({"user_key": [1], "date": [10], "time": ["09:00:00"], "daily_relief_inhaler": [1]})
    inh = _records_at([-24, -12, 0, 0.01])
    rolling = join_questionnaire_with_inhaler(q, inh, 24, False, False)["inhaler_usage"].iloc[0]
    chunk = join_questionnaire_with_inhaler(q, inh, 24, True, False)["inhaler_usage"].iloc[0]
    assert rolling == chunk == 3
    with use_definitions(V1):  # v1's open end missed a record at the questionnaire time
        assert join_questionnaire_with_inhaler(q, inh, 24, True, False)["inhaler_usage"].iloc[0] == 2
