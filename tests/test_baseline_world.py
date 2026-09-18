"""The committed baseline world (span=Q__case=A) is the published v1 analysis.

Item 4 of the v2 plan: before any other world is trusted, the baseline must
reproduce v1. These tests check the committed artifacts under
``results/v2/worlds/span=Q__case=A/`` against ``results/v1/`` and against the
literal sets the manuscript reports, and that ``config.py`` derives its
groups from them rather than from typed lists.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from aamos_concordance.summary import build_world_summary, derive_concordant_sets
from aamos_concordance.worlds import (
    BASELINE,
    PER_CONFIG_Z,
    SUMMARY_CSV,
    concordant_sets_path,
    list_worlds,
    read_world_config,
    world_dir,
    world_null_path,
)
from tests.test_summary import V1_ASSESSED, V1_CONCORDANT_MEAN, V1_CONCORDANT_MEDIAN


@pytest.fixture(scope="module")
def baseline_dir():
    d = world_dir(BASELINE)
    assert d.is_dir(), "baseline world not built; run scripts/build_baseline_world.py"
    return d


def test_baseline_per_config_table_is_v1_verbatim(baseline_dir, frozen_results_dir):
    ours = pd.read_csv(baseline_dir / PER_CONFIG_Z)
    v1 = pd.read_csv(frozen_results_dir / "per_patient_analysis" / "per_patient_correlation_results.csv")
    assert_frame_equal(ours, v1)


def test_baseline_null_is_the_v1_parquet(baseline_dir, frozen_results_dir):
    assert world_null_path(BASELINE) == frozen_results_dir / "permutation_aggregate_extended" / "median_concordant" / "all_permutations.parquet"
    cfg = read_world_config(BASELINE)
    assert cfg["world_id"] == "span=Q__case=A" and cfg["span"] == "Q" and cfg["absence_case"] == "A"


def test_baseline_summary_is_reproducible_from_its_inputs(baseline_dir):
    committed = pd.read_csv(baseline_dir / SUMMARY_CSV)
    rebuilt = build_world_summary(pd.read_csv(baseline_dir / PER_CONFIG_Z), pd.read_parquet(world_null_path(BASELINE)))
    assert_frame_equal(committed, rebuilt, check_dtype=False)


def test_baseline_summary_yields_the_published_sets(baseline_dir):
    summary = pd.read_csv(baseline_dir / SUMMARY_CSV)
    assert derive_concordant_sets(summary) == {"mean": V1_CONCORDANT_MEAN, "median": V1_CONCORDANT_MEDIAN}
    assert sorted(summary["patient_id"].unique()) == V1_ASSESSED


def test_generated_concordant_sets_file_matches_summary(baseline_dir):
    sets = json.loads(concordant_sets_path().read_text())
    assert sets["span=Q__case=A"] == {"mean": V1_CONCORDANT_MEAN, "median": V1_CONCORDANT_MEDIAN}
    assert "span=Q__case=A" in list_worlds()


def test_config_groups_are_derived_not_typed():
    import config
    assert config.ACTIVE_WORLD == BASELINE
    assert config.GROUPS["concordant"]["patients"] == V1_CONCORDANT_MEAN
    assert config.GROUPS["median_concordant"]["patients"] == V1_CONCORDANT_MEDIAN
    assert config.GROUPS["concordant"]["world"] == "span=Q__case=A"
    assert config.ASSESSED_PATIENTS == V1_ASSESSED
    assert config.get_remaining_patients("concordant") == [p for p in V1_ASSESSED if p not in V1_CONCORDANT_MEAN]


def test_set_active_world_rejects_unbuilt_world():
    import config
    with pytest.raises(RuntimeError, match="No concordant sets recorded"):
        config.set_active_world("span=D__case=C")
    assert config.ACTIVE_WORLD == BASELINE  # unchanged after the failure


def test_output_dirs_are_per_world(tmp_path, monkeypatch):
    import config
    from aamos_concordance import worlds
    monkeypatch.setattr(worlds, "V2_DIR", tmp_path)
    assert config.get_output_dir("concordant", "bland_altman") == tmp_path / "comparisons" / "span=Q__case=A" / "concordant" / "bland_altman"
