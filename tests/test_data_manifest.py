"""Manifest verification, raw-data lookup and loader wiring."""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

from aamos_concordance import data as data_mod
from aamos_concordance.data import (
    RAW_FILES,
    DataIntegrityError,
    DataNotFoundError,
    find_data_dir,
    find_raw_file,
    load_raw,
    read_manifest,
    sha256_of,
    verify_manifest,
    write_manifest,
)
from tests.synthetic import make_dataset


@pytest.fixture
def raw_dir(tmp_path):
    """A directory holding synthetic versions of the three raw files."""
    q, inh = make_dataset(0)
    d = tmp_path / "raw"
    d.mkdir()
    pd.DataFrame({"user_key": [100, 101], "daily_start_date": [0, 0], "daily_end_date": [30, 30],
                  "inhaler_start_date": [0, 0], "inhaler_end_date": [30, 30]}).to_csv(d / RAW_FILES[0], index=False)
    q.to_csv(d / RAW_FILES[1], index=False)
    inh.to_csv(d / RAW_FILES[2], index=False)
    return d


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Point the module's defaults away from the real repo so tests are hermetic."""
    monkeypatch.delenv("AAMOS_DATA_DIR", raising=False)
    monkeypatch.setattr(data_mod, "DEFAULT_DATA_DIR", tmp_path / "default")
    monkeypatch.setattr(data_mod, "REPO_ROOT", tmp_path / "repo")
    return tmp_path


# --------------------------------------------------------------------------
# Manifest round trip
# --------------------------------------------------------------------------

def test_write_then_verify_round_trip(raw_dir, tmp_path):
    manifest = write_manifest(raw_dir, tmp_path / "MANIFEST.sha256")
    entries = read_manifest(manifest)
    assert set(entries) == set(RAW_FILES)
    for name in RAW_FILES:
        assert entries[name] == sha256_of(raw_dir / name)
    hashes = verify_manifest(raw_dir, manifest)
    assert hashes == entries


def test_manifest_is_shasum_compatible(raw_dir, tmp_path):
    manifest = write_manifest(raw_dir, tmp_path / "MANIFEST.sha256")
    body = [l for l in manifest.read_text().splitlines() if l and not l.startswith("#")]
    for line in body:
        digest, sep, name = line.partition("  ")
        assert sep == "  " and len(digest) == 64 and name in RAW_FILES


def test_modified_file_is_detected(raw_dir, tmp_path):
    manifest = write_manifest(raw_dir, tmp_path / "MANIFEST.sha256")
    target = raw_dir / RAW_FILES[2]
    target.write_text(target.read_text() + "999,5,12:00:00,VENTOLIN\n")
    with pytest.raises(DataIntegrityError) as exc:
        verify_manifest(raw_dir, manifest)
    assert RAW_FILES[2] in str(exc.value)


def test_file_missing_from_manifest_is_detected(raw_dir, tmp_path):
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{sha256_of(raw_dir / RAW_FILES[0])}  {RAW_FILES[0]}\n")
    with pytest.raises(DataIntegrityError) as exc:
        verify_manifest(raw_dir, manifest)
    assert "not listed" in str(exc.value)


def test_missing_raw_file_raises_not_found(raw_dir, tmp_path):
    (raw_dir / RAW_FILES[1]).unlink()
    with pytest.raises(DataNotFoundError):
        verify_manifest(raw_dir, tmp_path / "MANIFEST.sha256")


def test_absent_manifest_warns_but_returns_hashes(raw_dir, tmp_path):
    with pytest.warns(UserWarning, match="NOT being verified"):
        hashes = verify_manifest(raw_dir, tmp_path / "nope.sha256")
    assert set(hashes) == set(RAW_FILES)


# --------------------------------------------------------------------------
# Lookup order
# --------------------------------------------------------------------------

def test_find_data_dir_prefers_env_then_default_then_root(isolated, raw_dir, monkeypatch):
    with pytest.raises(DataNotFoundError):
        find_data_dir()
    root = isolated / "repo"
    root.mkdir()
    for f in RAW_FILES:
        (root / f).write_bytes((raw_dir / f).read_bytes())
    assert find_data_dir() == root

    default = isolated / "default"
    default.mkdir()
    for f in RAW_FILES:
        (default / f).write_bytes((raw_dir / f).read_bytes())
    assert find_data_dir() == default

    monkeypatch.setenv("AAMOS_DATA_DIR", str(raw_dir))
    assert find_data_dir() == raw_dir
    assert find_data_dir(explicit=default) == default


def test_find_raw_file_locates_optional_inputs(isolated, raw_dir, monkeypatch):
    (raw_dir / "aamos00-end-final-freetext.csv").write_text("user_key,text\n100,ok\n")
    monkeypatch.setenv("AAMOS_DATA_DIR", str(raw_dir))
    assert find_raw_file("aamos00-end-final-freetext.csv") == raw_dir / "aamos00-end-final-freetext.csv"
    with pytest.raises(DataNotFoundError):
        find_raw_file("does-not-exist.csv")


# --------------------------------------------------------------------------
# load_raw and the two loaders
# --------------------------------------------------------------------------

def test_load_raw_verifies_and_reports_provenance(isolated, raw_dir, monkeypatch):
    default = isolated / "default"
    default.mkdir()
    for f in RAW_FILES:
        (default / f).write_bytes((raw_dir / f).read_bytes())
    write_manifest(default, default / "MANIFEST.sha256")

    raw = load_raw()
    assert raw.verified is True
    assert raw.data_dir == default
    assert set(raw.hashes) == set(RAW_FILES)
    prov = raw.provenance()
    assert prov["manifest_verified"] is True and prov["sha256"] == raw.hashes
    assert len(raw.questionnaire) > 0 and len(raw.inhaler) > 0

    (default / RAW_FILES[1]).write_text("user_key,date,time,daily_relief_inhaler\n")
    with pytest.raises(DataIntegrityError):
        load_raw()


def test_load_raw_drop_duplicates_matches_loader_behaviour(isolated, raw_dir, monkeypatch):
    monkeypatch.setenv("AAMOS_DATA_DIR", str(raw_dir))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = load_raw(drop_duplicates=False)
        b = load_raw(drop_duplicates=True)
    assert len(b.inhaler) == len(a.inhaler.drop_duplicates())


def test_asthma_data_loader_uses_manifest_verified_lookup(isolated, raw_dir, monkeypatch):
    from data_loader import AsthmaDataLoader, DataLoaderConfig
    monkeypatch.setenv("AAMOS_DATA_DIR", str(raw_dir))
    write_manifest(raw_dir, isolated / "default" / "MANIFEST.sha256")
    loader = AsthmaDataLoader(DataLoaderConfig())
    assert loader.data_provenance["manifest_verified"] is True
    assert loader.data_provenance["data_dir"] == str(raw_dir)

    (raw_dir / RAW_FILES[2]).write_text("user_key,date,time,name\n")
    with pytest.raises(DataIntegrityError):
        AsthmaDataLoader(DataLoaderConfig())


def test_asthma_data_loader_explicit_paths_are_hashed_not_verified(raw_dir):
    from data_loader import AsthmaDataLoader, DataLoaderConfig
    loader = AsthmaDataLoader(
        DataLoaderConfig(),
        patient_info_path=raw_dir / RAW_FILES[0],
        daily_questionnaire_path=raw_dir / RAW_FILES[1],
        inhaler_data_path=raw_dir / RAW_FILES[2],
    )
    assert loader.data_provenance["manifest_verified"] is False
    assert loader.data_provenance["sha256"][RAW_FILES[1]] == sha256_of(raw_dir / RAW_FILES[1])
    with pytest.raises(ValueError):
        AsthmaDataLoader(DataLoaderConfig(), patient_info_path=raw_dir / RAW_FILES[0])


def test_per_patient_loader_uses_manifest_verified_lookup(isolated, raw_dir, monkeypatch):
    from per_patient_correlation_analysis import PerPatientCorrelationConfig, PerPatientDataLoader
    monkeypatch.setenv("AAMOS_DATA_DIR", str(raw_dir))
    write_manifest(raw_dir, isolated / "default" / "MANIFEST.sha256")
    loader = PerPatientDataLoader(PerPatientCorrelationConfig())
    assert loader.data_provenance["manifest_verified"] is True
    assert loader.get_eligible_patients() == [100, 101]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_write_and_verify(isolated, raw_dir, capsys):
    (isolated / "default").mkdir()
    assert data_mod._main(["--data-dir", str(raw_dir), "--write-manifest"]) == 0
    out = capsys.readouterr().out
    assert "Wrote" in out and RAW_FILES[2] in out
    assert (isolated / "default" / "MANIFEST.sha256").exists()
    assert data_mod._main(["--data-dir", str(raw_dir), "--verify"]) == 0
    assert "OK: 3 files" in capsys.readouterr().out
