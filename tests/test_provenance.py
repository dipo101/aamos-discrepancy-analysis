"""Provenance sidecars: content, serialisation and the config.py helper."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from aamos_concordance import build_record, read_sidecar, write_sidecar
from aamos_concordance.provenance import git_state, sidecar_path, to_jsonable


class Colour(Enum):
    RED = "red"


@dataclasses.dataclass
class Cfg:
    window: int = 24
    method: Colour = Colour.RED
    path: Path = Path("/tmp/x")
    nested: dict = dataclasses.field(default_factory=lambda: {"a": (1, 2), "b": np.float64(0.5)})


def test_to_jsonable_handles_dataclass_enum_path_numpy():
    out = to_jsonable(Cfg())
    assert out == {"window": 24, "method": "red", "path": "/tmp/x", "nested": {"a": [1, 2], "b": 0.5}}
    json.dumps(out)


def test_git_state_reports_commit_from_working_tree():
    state = git_state()
    assert state["git_source"] == "working-tree"
    assert len(state["git_commit"]) == 40
    assert isinstance(state["git_dirty"], bool)


def test_sidecar_path_sits_next_to_output(tmp_path):
    assert sidecar_path(tmp_path / "results.csv") == tmp_path / "results.csv.provenance.json"


def test_write_and_read_sidecar(tmp_path):
    out = tmp_path / "sub" / "results.csv"
    data = {"data_dir": "/data", "sha256": {"a.csv": "0" * 64}, "manifest_verified": True}
    path = write_sidecar(out, config=Cfg(), data=data, extra={"script": "x.py"})
    assert path == sidecar_path(out) and path.exists()
    rec = read_sidecar(out)
    assert rec["config"]["method"] == "red"
    assert rec["data"] == data
    assert rec["script"] == "x.py"
    assert rec["git_commit"] == git_state()["git_commit"]
    assert rec["written_at"].endswith("+00:00")


def test_build_record_without_config_or_data():
    rec = build_record()
    assert rec["config"] is None and rec["data"] is None
    assert "git_commit" in rec


def test_record_run_provenance_writes_group_and_script(tmp_path, monkeypatch):
    import config as cfg_mod
    path = cfg_mod.record_run_provenance(tmp_path, group_name="concordant", script="t.py",
                                         config={"k": 1}, data={"sha256": {}})
    rec = json.loads(path.read_text())
    assert path.name == "RUN.provenance.json"
    assert rec["group"] == "concordant"
    assert rec["group_patients"] == cfg_mod.GROUPS["concordant"]["patients"]
    assert rec["script"] == "t.py" and rec["config"] == {"k": 1}


def test_record_run_provenance_tolerates_missing_data(tmp_path, monkeypatch):
    import config as cfg_mod
    from aamos_concordance import data as data_mod
    monkeypatch.delenv("AAMOS_DATA_DIR", raising=False)
    monkeypatch.setattr(data_mod, "DEFAULT_DATA_DIR", tmp_path / "none")
    monkeypatch.setattr(data_mod, "REPO_ROOT", tmp_path / "none")
    path = cfg_mod.record_run_provenance(tmp_path, group_name="median_concordant", script="t.py")
    assert json.loads(path.read_text())["data"] is None
