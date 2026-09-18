"""Provenance sidecars for analysis outputs.

Every result file the pipeline writes should be traceable to the exact
inputs and code that produced it. :func:`write_sidecar` writes a small JSON
file next to an output (``<output>.provenance.json``) recording:

* when it was written;
* the git commit of the code (and whether the tree was dirty), or the
  build-time commit stamped into the package by ``deploy_job.sh`` when
  running inside the Cloud Run container;
* the SHA-256 of each raw data file that was read;
* the full configuration used, as a JSON-serialisable dict.

:func:`build_record` returns the same dict without writing it, for callers
that embed provenance inside their own output (the job-worker's result JSON).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_SUFFIX = ".provenance.json"


def _run_git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> Dict[str, Any]:
    """Current commit and dirtiness, falling back to the build stamp or 'unknown'."""
    commit = _run_git("rev-parse", "HEAD")
    if commit:
        dirty = bool(_run_git("status", "--porcelain", "--untracked-files=no"))
        return {"git_commit": commit, "git_dirty": dirty, "git_source": "working-tree"}
    try:
        from . import _build_info  # type: ignore  # written by deploy_job.sh, git-ignored
        return {"git_commit": _build_info.GIT_COMMIT, "git_dirty": _build_info.GIT_DIRTY, "git_source": "build-stamp"}
    except ImportError:
        return {"git_commit": os.environ.get("GIT_COMMIT", "unknown"), "git_dirty": None, "git_source": "unknown"}


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, paths and numpy scalars for JSON."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def build_record(
    *,
    config: Any = None,
    data: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the provenance dict. ``data`` is normally ``RawData.provenance()``."""
    record: Dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
        "data": to_jsonable(data) if data is not None else None,
        "config": to_jsonable(config) if config is not None else None,
    }
    if extra:
        record.update(to_jsonable(extra))
    return record


def sidecar_path(output_path: os.PathLike) -> Path:
    p = Path(output_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def write_sidecar(
    output_path: os.PathLike,
    *,
    config: Any = None,
    data: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``<output>.provenance.json`` and return its path."""
    record = build_record(config=config, data=data, extra=extra)
    path = sidecar_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def read_sidecar(output_path: os.PathLike) -> Dict[str, Any]:
    return json.loads(sidecar_path(output_path).read_text())
