"""Exact duplicate device rows: the three readings, and the same treatment in observed data and null."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aamos_concordance.configs import generate_param_combinations
from aamos_concordance.duplicates import DUPLICATE_READINGS, apply_duplicate_reading, duplicate_counts
from aamos_concordance.pipeline import generate_world_null, observed_per_config_table, world_frames
from aamos_concordance.worlds import WorldSpec
from tests.synthetic import make_dataset


def _device():
    return pd.DataFrame({
        "user_key": [1, 1, 1, 1, 1, 1, 2],
        "date": [3, 3, 3, 3, 4, 4, 3],
        "time": ["08:15:00", "08:15:00", "08:15:00", "09:10:07", "09:10:07", "09:10:07", "08:15:00"],
        "name": "V",
    })


def test_readings():
    d = _device()
    assert apply_duplicate_reading(d, "real") is d
    art = apply_duplicate_reading(d, "artefact")
    assert art.index.tolist() == [0, 3, 4, 6]  # first of each repeated group kept
    # minute-resolution repeats are puffs; second-resolution repeats are artefacts
    assert apply_duplicate_reading(d, "resolution").index.tolist() == [0, 1, 2, 3, 4, 6]
    with pytest.raises(ValueError):
        apply_duplicate_reading(d, "maybe")


def _times(*hms, pid=1, day=3):
    return pd.DataFrame({"user_key": pid, "date": day, "time": list(hms), "name": "V"})


def test_window_readings_anchor_vs_chain():
    # gaps: 0, 20, 20, 45, 70 s
    d = _times("08:00:00", "08:00:00", "08:00:20", "08:00:40", "08:01:25", "08:02:35")
    kept = lambda r: apply_duplicate_reading(d, r)["time"].tolist()
    # anchored 30 s: keep 08:00:00; 08:00:40 is 40 s after it -> kept; 08:01:25 is 45 s after -> kept
    assert kept("anchor30") == ["08:00:00", "08:00:40", "08:01:25", "08:02:35"]
    # chained 30 s: 0/20/20 s gaps chain into one event; 45 s and 70 s gaps start new ones
    assert kept("chain30") == ["08:00:00", "08:01:25", "08:02:35"]
    # anchored 60 s: 08:01:25 is 85 s after 08:00:00 -> kept; 08:02:35 is 70 s after that -> kept
    assert kept("anchor60") == ["08:00:00", "08:01:25", "08:02:35"]
    # chained 60 s: every gap up to 45 s chains; the 70 s gap starts a new event
    assert kept("chain60") == ["08:00:00", "08:02:35"]


def test_window_boundary_is_strictly_greater_and_crosses_midnight():
    d = pd.concat([_times("23:59:45", day=3), _times("00:00:15", day=4), _times("00:00:46", day=4)], ignore_index=True)
    # 30 s apart exactly -> same event; 31 s -> new event, across the day boundary too
    assert apply_duplicate_reading(d, "chain30")["time"].tolist() == ["23:59:45", "00:00:46"]


def test_window_readings_are_per_patient_and_medication_and_keep_row_order():
    d = pd.concat([_times("08:00:10", pid=2), _times("08:00:00", pid=1), _times("08:00:05", pid=1),
                   _times("08:00:05", pid=1).assign(name="F")], ignore_index=True)
    out = apply_duplicate_reading(d, "anchor30")
    assert out.index.tolist() == [0, 1, 3]  # 1's V puff at :05 merges; other patient and medication untouched
    assert out.index.is_monotonic_increasing


def test_window_readings_subsume_exact_repeats():
    d = _device()
    for r in ("anchor30", "chain30", "anchor60", "chain60"):
        out = apply_duplicate_reading(d, r)
        assert not out.duplicated().any() and len(out) <= len(apply_duplicate_reading(d, "artefact"))


def test_duplicate_counts():
    c = duplicate_counts(_device()).set_index("patient_id")
    assert c.loc[1].to_dict() == {"n_rows": 6, "n_repeats": 3, "n_repeats_minute_resolution": 2, "n_repeats_second_resolution": 1,
                                  "n_rows_artefact": 3, "n_rows_real": 6, "n_rows_resolution": 5,
                                  "n_rows_anchor30": 3, "n_rows_chain30": 3, "n_rows_anchor60": 3, "n_rows_chain60": 3}
    assert c.loc[2, "n_repeats"] == 0


@pytest.fixture(scope="module")
def duplicated_patient():
    q, inh = make_dataset(2)
    pid = int(q.user_key.value_counts().idxmin())  # the smallest, so the null is an exact enumeration
    q, inh = q[q.user_key == pid], inh[inh.user_key == pid]
    extra = inh.iloc[::3].copy()                    # repeat every third device row
    return pid, q, pd.concat([inh, extra], ignore_index=True)


def test_world_frames_apply_the_reading(duplicated_patient):
    pid, q, inh = duplicated_patient
    sizes = {r: len(world_frames(WorldSpec("union", "A", duplicates=r), q, inh, pid)[1]) for r in DUPLICATE_READINGS}
    assert sizes["real"] == len(inh) and sizes["artefact"] == len(inh.drop_duplicates())
    assert sizes["artefact"] <= sizes["resolution"] <= sizes["real"]


def test_observed_and_null_see_the_same_rows(duplicated_patient):
    """The identity permutation of the null reproduces the observed table under every reading (v1 mixed readings)."""
    pid, q, inh = duplicated_patient
    combos = generate_param_combinations()
    obs_by_reading = {}
    for r in ("artefact", "real"):
        w = WorldSpec("union", "A", duplicates=r)
        obs = observed_per_config_table(w, q, inh, [pid]).set_index("config_idx")
        null, modes = generate_world_null(w, q, inh, [pid], exact_max=10**9)
        assert modes[pid]["mode"] == "exact"
        ident = null[null.permutation_idx == 0].set_index("config_idx")
        assert np.allclose(ident["spearman_z"], obs.loc[ident.index, "spearman_z"], equal_nan=True)
        assert (ident["n_rows"] == obs.loc[ident.index, "sample_size"]).all()
        obs_by_reading[r] = obs
    assert not np.allclose(obs_by_reading["real"]["spearman_z"], obs_by_reading["artefact"]["spearman_z"], equal_nan=True)
