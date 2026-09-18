"""Item 21: effective configurations under Spearman, and the diagnostic that quantifies them."""

from __future__ import annotations

import pandas as pd
import pytest

from aamos_concordance import generate_param_combinations
from aamos_concordance.configs import (
    N_SPEARMAN_EFFECTIVE,
    SPEARMAN_METHOD_CLASS,
    config_equivalence_classes,
    effective_config_indices,
    spearman_equivalence_key,
)
from aamos_concordance.diagnostics import attach_config_idx, effective_configurations, equivalence_class_check



def test_equivalence_classes_partition_the_132():
    classes = config_equivalence_classes("spearman")
    members = sorted(i for m in classes.values() for i in m)
    assert members == list(range(132))
    assert len(classes) == N_SPEARMAN_EFFECTIVE == 60
    assert len(effective_config_indices("spearman")) == 60
    assert effective_config_indices("pearson") == list(range(132))
    sizes = sorted(len(m) for m in classes.values())
    # 3 method classes of sizes 4/1/1 x (10 windows, one of which merges two) x 2 filters
    assert sum(sizes) == 132 and max(sizes) == 8


def test_spearman_key_merges_exactly_the_intended_axes():
    combos = generate_param_combinations()
    by_key = {}
    for c in combos:
        by_key.setdefault(spearman_equivalence_key(c), []).append(c)
    for key, cs in by_key.items():
        assert len({c["filter_out_zero_usage"] for c in cs}) == 1
        assert len({SPEARMAN_METHOD_CLASS[c["categorization_method"]] for c in cs}) == 1
        assert len({c["use_calendar_days"] for c in cs}) == 1


def test_equivalence_classes_are_exact_duplicates_on_the_v1_table(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    check = equivalence_class_check(per_config, "spearman")
    assert len(check) == 60
    assert check["max_within_class_spread"].max() < 1e-12


def test_attach_config_idx_is_a_bijection_on_the_v1_table(frozen_results_dir):
    per_config = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    df = attach_config_idx(per_config)
    assert df.groupby("user_key")["config_idx"].nunique().max() == 132
    assert df["config_idx"].min() == 0 and df["config_idx"].max() == 131


def test_effective_configurations_pins_the_finding(frozen_results_dir):
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
    # Under Pearson nothing collapses.
    assert (t["pearson_n_valid_effective"] == t["pearson_n_valid_all"]).all()
