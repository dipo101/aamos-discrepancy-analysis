"""Shared fixtures.

Real-data tests need the three timestamped AAMOS-00 CSVs, which are not in
the repository. They are located with the same lookup the loaders use
(``$AAMOS_DATA_DIR``, then ``data/raw/``, then the repository root). If none
of those holds all three files the real-data tests are skipped with a
message saying so, rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aamos_concordance.data import RAW_FILES, DataNotFoundError, find_data_dir

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_dir() -> Path:
    try:
        return find_data_dir()
    except DataNotFoundError:
        pytest.skip(
            "Timestamped AAMOS-00 data not found. Set AAMOS_DATA_DIR or place "
            f"{', '.join(RAW_FILES)} in data/raw/."
        )


@pytest.fixture(scope="session")
def frozen_results_dir() -> Path:
    return REPO_ROOT / "results"
