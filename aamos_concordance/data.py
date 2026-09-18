"""Locating, hashing and loading the raw AAMOS-00 files.

The raw data is never written by any code in this repository. The three
timestamped CSVs live outside version control (they are sensitive) in one of:

1. ``$AAMOS_DATA_DIR``
2. ``data/raw/`` at the repository root  (recommended)
3. the repository root itself             (where the scripts used to look)

``data/raw/MANIFEST.sha256`` *is* committed. It records the SHA-256 of each
raw file, and every loader verifies the files against it before reading.
A mismatch raises :class:`DataIntegrityError` rather than silently
producing results from a different dataset. If the manifest is absent the
loaders warn instead, so the very first run (before the manifest has been
generated with ``python -m aamos_concordance --write-manifest``) still
works.

Loaders return a :class:`RawData` bundle that carries the frames together
with the hashes that were verified, so downstream provenance records can
name exactly which bytes produced a result.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_NAME = "MANIFEST.sha256"

PATIENT_INFO_FILE = "anonym_aamos00_patient_info.csv"
QUESTIONNAIRE_FILE = "anonym_aamos00_dailyquestionnaire_dt.csv"
INHALER_FILE = "anonym_aamos00_smartinhaler_dt.csv"
FEEDBACK_FILE = "aamos00-end-final-freetext.csv"

RAW_FILES: Tuple[str, ...] = (PATIENT_INFO_FILE, QUESTIONNAIRE_FILE, INHALER_FILE)
OPTIONAL_FILES: Tuple[str, ...] = (FEEDBACK_FILE,)


class DataIntegrityError(RuntimeError):
    """Raised when a raw file's hash does not match the committed manifest."""


class DataNotFoundError(FileNotFoundError):
    """Raised when the raw files cannot be located in any known directory."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_data_dir(explicit: Optional[os.PathLike] = None) -> Path:
    """Return the first directory that holds all three required raw files."""
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("AAMOS_DATA_DIR"):
        candidates.append(Path(os.environ["AAMOS_DATA_DIR"]))
    candidates += [DEFAULT_DATA_DIR, REPO_ROOT]
    for d in candidates:
        if all((d / f).exists() for f in RAW_FILES):
            return d
    searched = ", ".join(str(c) for c in candidates)
    raise DataNotFoundError(
        f"Raw AAMOS-00 files {RAW_FILES} not found. Searched: {searched}. "
        "Place them in data/raw/ or set AAMOS_DATA_DIR."
    )


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def read_manifest(manifest_path: Path) -> Dict[str, str]:
    """Parse a ``sha256sum``-style manifest into ``{filename: hexdigest}``."""
    entries: Dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        name = name.lstrip("*")
        entries[name] = digest.lower()
    return entries


def write_manifest(data_dir: Path, manifest_path: Optional[Path] = None) -> Path:
    """Hash every raw file present in ``data_dir`` and write the manifest."""
    manifest_path = manifest_path or (DEFAULT_DATA_DIR / MANIFEST_NAME)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SHA-256 of the raw AAMOS-00 inputs. Verified by aamos_concordance.data on every load.",
        "# Regenerate with: python -m aamos_concordance --write-manifest",
        "# Format is compatible with `shasum -a 256 -c MANIFEST.sha256` run inside data/raw/.",
    ]
    for name in RAW_FILES + OPTIONAL_FILES:
        p = data_dir / name
        if p.exists():
            lines.append(f"{sha256_of(p)}  {name}")
    manifest_path.write_text("\n".join(lines) + "\n")
    return manifest_path


def verify_manifest(
    data_dir: Path,
    manifest_path: Optional[Path] = None,
    *,
    required: Tuple[str, ...] = RAW_FILES,
) -> Dict[str, str]:
    """Check ``required`` files in ``data_dir`` against the manifest.

    Returns ``{filename: hexdigest}`` for the verified files. Raises
    :class:`DataIntegrityError` on any mismatch or on a required file that
    the manifest does not list. If the manifest file itself does not exist,
    a warning is issued and the actual hashes are returned unverified.
    """
    manifest_path = manifest_path or (DEFAULT_DATA_DIR / MANIFEST_NAME)
    actual = {name: sha256_of(data_dir / name) for name in required if (data_dir / name).exists()}
    missing = [name for name in required if name not in actual]
    if missing:
        raise DataNotFoundError(f"Missing raw files in {data_dir}: {missing}")

    if not manifest_path.exists():
        warnings.warn(
            f"No data manifest at {manifest_path}; raw files are NOT being verified. "
            "Generate one with: python -m aamos_concordance --write-manifest",
            stacklevel=2,
        )
        return actual

    expected = read_manifest(manifest_path)
    problems = []
    for name, digest in actual.items():
        if name not in expected:
            problems.append(f"{name}: not listed in manifest")
        elif expected[name] != digest:
            problems.append(f"{name}: sha256 {digest[:12]}… != manifest {expected[name][:12]}…")
    if problems:
        raise DataIntegrityError(
            f"Raw data in {data_dir} does not match {manifest_path}:\n  " + "\n  ".join(problems)
            + "\nIf the data was intentionally replaced, regenerate the manifest and commit it."
        )
    logger.debug("Verified %d raw files against %s", len(actual), manifest_path)
    return actual


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class RawData:
    """The three raw frames plus the provenance of the bytes they came from."""

    patient_info: pd.DataFrame
    questionnaire: pd.DataFrame
    inhaler: pd.DataFrame
    data_dir: Path
    hashes: Dict[str, str] = field(default_factory=dict)
    verified: bool = False

    def provenance(self) -> Dict:
        return {"data_dir": str(self.data_dir), "sha256": dict(self.hashes), "manifest_verified": self.verified}


def load_raw(
    data_dir: Optional[os.PathLike] = None,
    *,
    verify: bool = True,
    drop_duplicates: bool = False,
) -> RawData:
    """Load the three raw CSVs, verifying them against the manifest first.

    ``drop_duplicates`` applies ``DataFrame.drop_duplicates()`` to each frame,
    which is what both historical loaders did immediately after reading.
    """
    d = find_data_dir(data_dir)
    manifest = DEFAULT_DATA_DIR / MANIFEST_NAME
    if verify:
        hashes = verify_manifest(d, manifest)
        verified = manifest.exists()
    else:
        hashes = {name: sha256_of(d / name) for name in RAW_FILES}
        verified = False

    frames = [pd.read_csv(d / name) for name in RAW_FILES]
    if drop_duplicates:
        frames = [f.drop_duplicates() for f in frames]
    return RawData(*frames, data_dir=d, hashes=hashes, verified=verified)


def find_raw_file(name: str, data_dir: Optional[os.PathLike] = None) -> Path:
    """Locate one raw file by name using the same search order as :func:`find_data_dir`.

    Used for optional inputs (e.g. the end-of-study feedback CSV) that may be
    present without the three core files.
    """
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    if os.environ.get("AAMOS_DATA_DIR"):
        candidates.append(Path(os.environ["AAMOS_DATA_DIR"]))
    candidates += [DEFAULT_DATA_DIR, REPO_ROOT, Path.cwd()]
    for d in candidates:
        if (d / name).exists():
            return d / name
    raise DataNotFoundError(f"{name} not found in any of: {', '.join(str(c) for c in candidates)}")


def resolve_paths(data_dir: Optional[os.PathLike] = None) -> Dict[str, Path]:
    """Absolute paths of the raw files, for callers that read CSVs themselves."""
    d = find_data_dir(data_dir)
    return {name: d / name for name in RAW_FILES + OPTIONAL_FILES}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage the raw-data manifest.")
    parser.add_argument("--data-dir", type=Path, default=None, help="folder holding the raw CSVs")
    parser.add_argument("--write-manifest", action="store_true", help="hash the raw files and write data/raw/MANIFEST.sha256")
    parser.add_argument("--verify", action="store_true", help="verify the raw files against the manifest")
    args = parser.parse_args(argv)

    d = find_data_dir(args.data_dir)
    if args.write_manifest:
        path = write_manifest(d)
        print(f"Wrote {path}")
        print(path.read_text())
    if args.verify or not args.write_manifest:
        hashes = verify_manifest(d)
        for name, digest in hashes.items():
            print(f"{digest}  {name}")
        print(f"OK: {len(hashes)} files in {d} match the manifest")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
