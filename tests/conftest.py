"""Shared fixtures.

Real-data tests need the three timestamped AAMOS-00 CSVs, which are not in
the repository. They are looked for in ``$AAMOS_DATA_DIR``, then in
``data/raw/`` at the repository root, then in the repository root itself
(where the analysis scripts historically expected them). If none of those
holds all three files the real-data tests are skipped with a message saying
so, rather than failing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_FILES = (
    "anonym_aamos00_patient_info.csv",
    "anonym_aamos00_dailyquestionnaire_dt.csv",
    "anonym_aamos00_smartinhaler_dt.csv",
)


def find_data_dir() -> Path | None:
    candidates = []
    if os.environ.get("AAMOS_DATA_DIR"):
        candidates.append(Path(os.environ["AAMOS_DATA_DIR"]))
    candidates += [REPO_ROOT / "data" / "raw", REPO_ROOT]
    for d in candidates:
        if all((d / f).exists() for f in RAW_FILES):
            return d
    return None


@pytest.fixture(scope="session")
def data_dir() -> Path:
    d = find_data_dir()
    if d is None:
        pytest.skip(
            "Timestamped AAMOS-00 data not found. Set AAMOS_DATA_DIR or place "
            f"{', '.join(RAW_FILES)} in data/raw/."
        )
    return d


@pytest.fixture(scope="session")
def frozen_results_dir() -> Path:
    return REPO_ROOT / "results"
