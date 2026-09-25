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


def test_duplicate_counts():
    c = duplicate_counts(_device()).set_index("patient_id")
    assert c.loc[1].to_dict() == {"n_rows": 6, "n_repeats": 3, "n_repeats_minute_resolution": 2, "n_repeats_second_resolution": 1,
                                  "n_rows_artefact": 3, "n_rows_real": 6, "n_rows_resolution": 5}
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
