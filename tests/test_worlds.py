"""World identity, v2 layout helpers, index and concordant-set files."""

from __future__ import annotations

import json

import pytest

from aamos_concordance import worlds
from aamos_concordance.worlds import (
    BASELINE,
    WorldSpec,
    as_world,
    comparisons_dir,
    gcs_results_prefix,
    list_worlds,
    load_concordant_sets,
    register_world,
    world_dir,
    world_null_path,
    write_concordant_sets,
    write_world_config,
)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(worlds, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(worlds, "V1_DIR", tmp_path / "v1")
    monkeypatch.setattr(worlds, "V2_DIR", tmp_path / "v2")
    monkeypatch.setattr(worlds, "REPO_ROOT", tmp_path)
    return tmp_path


def test_world_ids_round_trip():
    for w in [WorldSpec(), WorldSpec("D", "C"), WorldSpec("union", "B", 3), WorldSpec("intersection", "B", 99)]:
        assert WorldSpec.parse(w.world_id) == w
        assert str(w) == w.world_id
    assert WorldSpec("union", "B", 3).world_id == "span=union__case=B__k=03"
    assert BASELINE.world_id == "span=union__case=A" and BASELINE.span == "union"


@pytest.mark.parametrize("bad", ["Q", "span=Q", "span=Q__case=D", "span=X__case=A", "span=Q__case=B", "span=union__case=A__k=01", "span=Q__case=B__k=1"])
def test_invalid_world_ids_rejected(bad):
    with pytest.raises(ValueError):
        WorldSpec.parse(bad)


def test_invalid_specs_rejected():
    with pytest.raises(ValueError):
        WorldSpec("Q", "B")
    with pytest.raises(ValueError):
        WorldSpec("Q", "A", 1)
    with pytest.raises(ValueError):
        WorldSpec("Q", "B", 100)


def test_as_world_accepts_none_spec_and_string():
    assert as_world(None) == BASELINE
    assert as_world("span=D__case=C") == WorldSpec("D", "C")
    assert as_world(WorldSpec("D", "C")) == WorldSpec("D", "C")


def test_paths_are_keyed_by_world_id(sandbox):
    assert world_dir("span=D__case=C") == sandbox / "v2" / "worlds" / "span=D__case=C"
    assert not world_dir("span=D__case=C").exists()
    assert world_dir("span=D__case=C", create=True).is_dir()
    assert comparisons_dir(BASELINE, "concordant", "bland_altman") == sandbox / "v2" / "comparisons" / "span=union__case=A" / "concordant" / "bland_altman"
    assert comparisons_dir(BASELINE, "concordant", "bland_altman").is_dir()


def test_gcs_prefix_keeps_baseline_flat():
    assert gcs_results_prefix(BASELINE) == "results/"
    assert gcs_results_prefix("span=D__case=A") == "results/span=D__case=A/"


def test_register_world_is_idempotent_and_sorted(sandbox):
    register_world("span=D__case=A", git_commit="abc", note="first")
    register_world(BASELINE, git_commit="abc")
    register_world("span=D__case=A", git_commit="def", note="second")
    assert list_worlds() == ["span=D__case=A", "span=union__case=A"]
    rows = worlds.worlds_index_path().read_text().splitlines()
    assert len(rows) == 3  # header + 2
    assert "def" in rows[1] and "second" in rows[1]


def test_concordant_sets_merge_and_sort(sandbox):
    write_concordant_sets("span=D__case=A", {"mean": [702, 294], "median": []})
    write_concordant_sets(BASELINE, {"mean": [294, 473, 702], "median": [917, 294, 473, 702]})
    sets = load_concordant_sets()
    assert list(sets) == ["span=D__case=A", "span=union__case=A"]
    assert sets["span=D__case=A"]["mean"] == [294, 702]
    assert sets["span=union__case=A"]["median"] == [294, 473, 702, 917]
    # Rewriting one world leaves the other intact
    write_concordant_sets(BASELINE, {"mean": [294], "median": [294]})
    assert load_concordant_sets()["span=D__case=A"]["mean"] == [294, 702]


def test_world_null_path_falls_back_to_config_source(sandbox):
    with pytest.raises(FileNotFoundError):
        world_null_path(BASELINE)
    src = sandbox / "v1" / "some" / "null.parquet"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"")
    write_world_config(BASELINE, {"null_source": "v1/some/null.parquet"})
    assert world_null_path(BASELINE) == src
    local = world_dir(BASELINE) / "null.parquet"
    local.write_bytes(b"")
    assert world_null_path(BASELINE) == local


def test_world_config_round_trip_and_merge(sandbox):
    p = write_world_config("span=union__case=B__k=07", {"built_by": "test"})
    rec = json.loads(p.read_text())
    assert rec == {"world_id": "span=union__case=B__k=07", "span": "union", "absence_case": "B", "imputation": 7, "built_by": "test"}
    write_world_config("span=union__case=B__k=07", {"null_source": "x.parquet"})
    rec = json.loads(p.read_text())
    assert rec["built_by"] == "test" and rec["null_source"] == "x.parquet"  # merged, not replaced
