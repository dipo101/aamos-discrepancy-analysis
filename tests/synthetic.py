"""Synthetic questionnaire and inhaler frames for equivalence testing.

The generator deliberately covers the edge cases the categorisation and join
code is sensitive to:

* patients with no inhaler records at all;
* a patient with a day of more than 24 recorded puffs, which switches the
  data-driven upper bound onto its 3*SD branch;
* self-report values in every category bin, plus NaN and a value strictly
  between 0 and 1, both of which fall through to the top category;
* exact-duplicate rows in both frames (the real loaders drop_duplicates);
* time strings in the two formats pandas accepts from the raw files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RELIEF_VALUES = [0, 0, 0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 15, 20]
TIME_FORMATS = ("{:02d}:{:02d}:{:02d}", "{:02d}:{:02d}")


def _time_str(rng, fmt_idx=None):
    h, m, s = int(rng.integers(0, 24)), int(rng.integers(0, 60)), int(rng.integers(0, 60))
    fmt = TIME_FORMATS[fmt_idx if fmt_idx is not None else 0]
    return fmt.format(h, m, s) if fmt.count("{") == 3 else fmt.format(h, m)


def make_patient(rng, user_key, *, n_days, q_frac, inhaler_rate, heavy_day=None,
                 with_nan=False, with_fraction=False, time_fmt=0):
    days = np.arange(n_days)
    q_days = days[rng.random(n_days) < q_frac]
    q = pd.DataFrame({
        "user_key": user_key,
        "date": q_days,
        "time": [_time_str(rng, time_fmt) for _ in q_days],
        "daily_relief_inhaler": rng.choice(RELIEF_VALUES, size=len(q_days)).astype(float),
        "daily_day_symp": rng.choice([True, False], size=len(q_days)),
    })
    if with_nan and len(q):
        q.loc[q.index[0], "daily_relief_inhaler"] = np.nan
    if with_fraction and len(q) > 1:
        q.loc[q.index[1], "daily_relief_inhaler"] = 0.5

    counts = rng.poisson(inhaler_rate, size=n_days)
    if heavy_day is not None and n_days > heavy_day:
        counts[heavy_day] = 30
    rows = []
    for d, c in zip(days, counts):
        for _ in range(int(c)):
            rows.append({"user_key": user_key, "date": int(d), "time": _time_str(rng, time_fmt), "name": "VENTOLIN"})
    inh = pd.DataFrame(rows, columns=["user_key", "date", "time", "name"])
    return q, inh


def make_dataset(seed: int, n_patients: int = 6):
    rng = np.random.default_rng(seed)
    qs, inhs = [], []
    for i in range(n_patients):
        key = 100 + i
        kwargs = dict(n_days=int(rng.integers(5, 40)), q_frac=float(rng.uniform(0.4, 1.0)),
                      inhaler_rate=float(rng.uniform(0.0, 4.0)))
        if i == 0:
            kwargs["inhaler_rate"] = 0.0          # no device records at all
        if i == 1:
            kwargs["heavy_day"] = 2               # > 24 puffs on one day
            kwargs["inhaler_rate"] = 6.0
        if i == 2:
            kwargs.update(with_nan=True, with_fraction=True)
        if i == 3:
            kwargs["time_fmt"] = 1
        q, inh = make_patient(rng, key, **kwargs)
        qs.append(q)
        inhs.append(inh)
    questionnaire = pd.concat(qs, ignore_index=True)
    inhaler = pd.concat(inhs, ignore_index=True)
    # Duplicate a few rows, as the raw exports do, then drop them as the loaders do.
    if len(questionnaire):
        questionnaire = pd.concat([questionnaire, questionnaire.iloc[:2]], ignore_index=True)
    if len(inhaler):
        inhaler = pd.concat([inhaler, inhaler.iloc[:3]], ignore_index=True)
    questionnaire = questionnaire.drop_duplicates()
    inhaler = inhaler.drop_duplicates()
    return questionnaire, inhaler


ALL_192 = [
    (w, dm, cal, fz, cm)
    for w in (12, 24, 36, 48)
    for dm in (False, True)
    for cal in (False, True)
    for fz in (False, True)
    for cm in ("one_hot", "lower_bound", "midpoint", "midpoint_with_inhaler", "upper_bound", "upper_bound_with_inhaler")
]

WINDOW_CONFIGS = sorted({(w, dm, cal) for (w, dm, cal, _, _) in ALL_192})
