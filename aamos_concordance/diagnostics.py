"""Diagnostics over a world's observed per-config table.

``effective_configurations`` (v2 item 21) quantifies how many of the 132
configurations are actually distinct analyses for each patient under each
correlation type, and what the summary statistics look like when each
structural equivalence class is counted once instead of once per member.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .configs import config_equivalence_classes, effective_config_indices, generate_param_combinations
from .summary import DEFAULT_THRESHOLD

CONFIG_KEY = ["timestamp_window", "use_daily_max_windows", "use_calendar_days",
              "filter_out_zero_usage_entries", "categorization_method"]


def attach_config_idx(per_config: pd.DataFrame) -> pd.DataFrame:
    """Add ``config_idx`` (position in generate_param_combinations()) to an observed per-config table."""
    combos = generate_param_combinations()
    lookup = {
        (c["timestamp_window"], c["use_daily_max_windows"], c["use_calendar_days"],
         c["filter_out_zero_usage"], c["categorization_method"]): i
        for i, c in enumerate(combos)
    }
    keys = list(zip(*(per_config[k] for k in CONFIG_KEY)))
    out = per_config.copy()
    out["config_idx"] = [lookup[k] for k in keys]
    return out


def equivalence_class_check(per_config: pd.DataFrame, correlation_type: str = "spearman") -> pd.DataFrame:
    """For each structural equivalence class, the largest within-class spread of Z across patients.

    A spread of zero (up to floating point) confirms the class is a genuine
    duplicate on this data. Non-finite Z values are ignored.
    """
    col = f"{correlation_type}_z"
    df = attach_config_idx(per_config)
    classes = config_equivalence_classes(correlation_type)
    rows: List[Dict] = []
    for key, members in classes.items():
        sub = df[df["config_idx"].isin(members) & np.isfinite(df[col])]
        spread = sub.groupby("user_key")[col].agg(lambda s: s.max() - s.min())
        rows.append({
            "class": repr(key), "n_members": len(members), "member_config_idx": " ".join(map(str, members)),
            "max_within_class_spread": float(spread.max()) if len(spread) else np.nan,
            "n_patient_cells": int(len(sub)),
        })
    return pd.DataFrame(rows).sort_values(["n_members", "class"], ascending=[False, True]).reset_index(drop=True)


def effective_configurations(
    per_config: pd.DataFrame,
    patients: List[int],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Per patient: distinct Z values and summary statistics under all vs effective configurations."""
    df = attach_config_idx(per_config)
    df = df[df["user_key"].isin(patients)]
    eff = {t: set(effective_config_indices(t)) for t in ("spearman", "pearson")}
    rows: List[Dict] = []
    for pid, g in df.groupby("user_key", sort=True):
        row: Dict = {"patient_id": int(pid)}
        for t in ("spearman", "pearson"):
            col = f"{t}_z"
            valid_all = g.loc[np.isfinite(g[col]), col]
            valid_eff = g.loc[np.isfinite(g[col]) & g["config_idx"].isin(eff[t]), col]
            row.update({
                f"{t}_n_valid_all": int(len(valid_all)),
                f"{t}_n_distinct_z": int(valid_all.round(12).nunique()),
                f"{t}_n_valid_effective": int(len(valid_eff)),
                f"{t}_mean_all": valid_all.mean() if len(valid_all) else np.nan,
                f"{t}_mean_effective": valid_eff.mean() if len(valid_eff) else np.nan,
                f"{t}_median_all": valid_all.median() if len(valid_all) else np.nan,
                f"{t}_median_effective": valid_eff.median() if len(valid_eff) else np.nan,
            })
        for stat in ("mean", "median"):
            for scope in ("all", "effective"):
                row[f"spearman_{stat}_{scope}_ge_threshold"] = bool(row[f"spearman_{stat}_{scope}"] >= threshold)
        rows.append(row)
    return pd.DataFrame(rows)
