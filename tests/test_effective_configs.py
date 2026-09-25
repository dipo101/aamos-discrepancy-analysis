"""Item 21: effective configurations under Spearman, and the diagnostic that quantifies them."""

from __future__ import annotations

import pandas as pd
import pytest

from aamos_concordance import generate_param_combinations
from aamos_concordance.configs import (
    N_SPEARMAN_EFFECTIVE,
    N_SPEARMAN_EFFECTIVE_V1,
    spearman_method_class,
    config_equivalence_classes,
    effective_config_indices,
    spearman_equivalence_key,
)
from aamos_concordance.diagnostics import attach_config_idx, effective_configurations, equivalence_class_check



def test_equivalence_classes_partition_the_132():
    classes = config_equivalence_classes("spearman")
    members = sorted(i for m in classes.values() for i in m)
    assert members == list(range(132))
    assert len(classes) == N_SPEARMAN_EFFECTIVE == 80
    assert len(effective_config_indices("spearman")) == 80
    # Pearson keeps all six methods; only chunk-24 (identical to rolling-24) is dropped
    pearson = effective_config_indices("pearson")
    assert len(pearson) == len(config_equivalence_classes("pearson")) == 120
    dropped = [generate_param_combinations()[i] for i in sorted(set(range(132)) - set(pearson))]
    assert all(c["timestamp_window"] == 24 and c["use_daily_max_windows"] for c in dropped)
    sizes = sorted(len(m) for m in classes.values())
    # 4 method classes of sizes 3/1/1/1 x (10 windows, one of which merges two) x 2 filters
    assert sum(sizes) == 132 and max(sizes) == 6


def test_v1_definitions_have_three_spearman_method_classes(v1_definitions):
    """Under v1 no self-report reached the top band, so the upper-bound hybrid was rank-identical to lower bound."""
    classes = config_equivalence_classes("spearman")
    assert len(classes) == N_SPEARMAN_EFFECTIVE_V1 == 60
    assert max(len(m) for m in classes.values()) == 8
    assert spearman_method_class("upper_bound_with_inhaler") == spearman_method_class("lower_bound")


def test_upper_bound_hybrid_is_its_own_class_with_code_12():
    assert spearman_method_class("upper_bound_with_inhaler") not in {
        spearman_method_class(m) for m in ("lower_bound", "upper_bound", "one_hot")}
    assert spearman_method_class("midpoint_with_inhaler") == spearman_method_class("lower_bound")


def test_spearman_key_merges_exactly_the_intended_axes():
    combos = generate_param_combinations()
    by_key = {}
    for c in combos:
        by_key.setdefault(spearman_equivalence_key(c), []).append(c)
    for key, cs in by_key.items():
        assert len({c["filter_out_zero_usage"] for c in cs}) == 1
        assert len({spearman_method_class(c["categorization_method"]) for c in cs}) == 1
        assert len({c["use_calendar_days"] for c in cs}) == 1


def test_equivalence_classes_are_exact_duplicates_on_the_v1_table(v1_definitions, frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    check = equivalence_class_check(per_config, "spearman")
    assert len(check) == 60
    assert check["max_within_class_spread"].max() < 1e-12


def test_attach_config_idx_is_a_bijection_on_the_v1_table(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    df = attach_config_idx(per_config)
    assert df.groupby("user_key")["config_idx"].nunique().max() == 132
    assert df["config_idx"].min() == 0 and df["config_idx"].max() == 131


def test_effective_configurations_pins_the_finding(v1_definitions, frozen_results_dir):
    from tests.test_summary import V1_ASSESSED
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    t = effective_configurations(per_config, V1_ASSESSED).set_index("patient_id")
    assert list(t.index) == V1_ASSESSED
    assert (t["spearman_n_valid_effective"] <= 60).all()
    assert t["spearman_n_distinct_z"].median() == 25
    # The published concordant set under mean-of-132 ...
    assert sorted(t.index[t["spearman_mean_all_ge_threshold"]]) == [294, 473, 702, 917]
    # ... loses 294 and 473 when each structural class is counted once.
    assert sorted(t.index[t["spearman_mean_effective_ge_threshold"]]) == [702, 917]
    assert t.loc[294, "spearman_mean_effective"] == pytest.approx(0.459, abs=5e-4)
    assert t.loc[473, "spearman_mean_effective"] == pytest.approx(0.432, abs=5e-4)
    assert t.loc[473, "spearman_median_effective"] == pytest.approx(0.485, abs=5e-4)
    # Under Pearson only the 12 chunk-24 duplicates are dropped.
    assert (t["pearson_n_valid_all"] - t["pearson_n_valid_effective"]).between(0, 12).all()
