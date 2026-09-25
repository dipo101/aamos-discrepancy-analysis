"""World identity and the v2 results layout.

A *world* is one combination of the outer-loop choices that the v2
sensitivity analysis varies around the fixed 132-configuration multiverse:

* ``span``          which raw entries are believed: ``Q`` (first to last
                    questionnaire), ``D`` (first to last device record),
                    ``union`` or ``intersection`` of the two;
* ``absence_case``  how a questionnaire window with no device records is
                    treated: ``A`` naive zero (the v1 behaviour), ``B``
                    Poisson imputation, ``C`` dropped;
* ``imputation``    the draw index for case ``B`` (``None`` otherwise);
* ``duplicates``    how exact duplicate device rows are read: ``artefact``
                    (dropped; primary, omitted from the world id), ``real``
                    or ``resolution`` (see :mod:`aamos_concordance.duplicates`).

The baseline world ``span=union__case=A`` is the published v1 analysis and must
reproduce it exactly (``tests/test_regression_v1.py``). v1 trimmed nothing,
and every record lies inside the union of the two periods by definition, so
``union`` is the no-trimming span; ``Q`` removes device records before the
first and after the last questionnaire, which changes the edge windows of
the calendar-day lookbacks.

Layout under ``results/``::

    v1/                                   frozen pre-v2 results; only the v1 figure scripts
                                          write there, regenerating figures from frozen tables
    v2/
      worlds.csv                          index of every world that has been built
      worlds/<world_id>/
        config.json                       the WorldSpec plus how the world was built
        per_config_z.csv                  patient x 132 configs x Fisher Z (observed)
        null.parquet                      patient x permutation x null summaries
                                          (or config.json:null_source pointing elsewhere)
        null_per_config.parquet           patient x permutation x config x Z, both correlation
                                          types; lets the null be re-summarised under any
                                          config subset / statistic / type (absent for v1)
        summary.csv                       long table: patient x (config_set, type, measure), observed Z, p-values
        threshold_sweep.csv               concordant set under every spec x threshold
        observed.csv                      observed Z under every spec (no p-values; always available)
      groups/concordant_sets.json         {world_id: {"config_set/type/measure": [patients]}}; generated
      comparisons/<world_id>/<group>/<analysis>/   downstream outputs
      diagnostics/                        cell counts, missingness checks
"""

from __future__ import annotations

import csv
import fcntl
import json
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .duplicates import DEFAULT_DUPLICATES, DUPLICATE_READINGS

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
V1_DIR = RESULTS_ROOT / "v1"
V2_DIR = RESULTS_ROOT / "v2"

SPANS = ("Q", "D", "union", "intersection")
ABSENCE_CASES = ("A", "B", "C")
MEASURES = ("mean", "median")

PER_CONFIG_Z = "per_config_z.csv"
NULL_PARQUET = "null.parquet"
NULL_PER_CONFIG_PARQUET = "null_per_config.parquet"  # patient x permutation x config x {spearman_z, pearson_z}
SUMMARY_CSV = "summary.csv"
THRESHOLD_SWEEP_CSV = "threshold_sweep.csv"
OBSERVED_CSV = "observed.csv"  # observed statistics under every spec, no p-values
CONFIG_JSON = "config.json"

_WORLD_ID_RE = re.compile(r"^span=(?P<span>[A-Za-z]+)__case=(?P<case>[ABC])(?:__k=(?P<k>\d{2}))?(?:__dup=(?P<dup>[a-z]+))?$")


@dataclass(frozen=True)
class WorldSpec:
    span: str = "union"
    absence_case: str = "A"
    imputation: Optional[int] = None
    duplicates: str = DEFAULT_DUPLICATES

    def __post_init__(self):
        if self.span not in SPANS:
            raise ValueError(f"span must be one of {SPANS}, got {self.span!r}")
        if self.absence_case not in ABSENCE_CASES:
            raise ValueError(f"absence_case must be one of {ABSENCE_CASES}, got {self.absence_case!r}")
        if (self.absence_case == "B") != (self.imputation is not None):
            raise ValueError("imputation index is required for case B and forbidden otherwise")
        if self.imputation is not None and not (0 <= self.imputation < 100):
            raise ValueError("imputation index must be in [0, 100)")
        if self.duplicates not in DUPLICATE_READINGS:
            raise ValueError(f"duplicates must be one of {DUPLICATE_READINGS}, got {self.duplicates!r}")

    @property
    def world_id(self) -> str:
        wid = f"span={self.span}__case={self.absence_case}"
        if self.imputation is not None:
            wid += f"__k={self.imputation:02d}"
        if self.duplicates != DEFAULT_DUPLICATES:
            wid += f"__dup={self.duplicates}"
        return wid

    @classmethod
    def parse(cls, world_id: str) -> "WorldSpec":
        m = _WORLD_ID_RE.match(world_id)
        if not m:
            raise ValueError(f"not a world id: {world_id!r} (expected e.g. span=union__case=A, span=D__case=B__k=03 "
                             "or span=Q__case=C__dup=real)")
        k, dup = m.group("k"), m.group("dup")
        if dup == DEFAULT_DUPLICATES:
            raise ValueError(f"not a canonical world id: {world_id!r} (the default duplicates reading is omitted)")
        return cls(span=m.group("span"), absence_case=m.group("case"), imputation=int(k) if k is not None else None,
                   duplicates=dup or DEFAULT_DUPLICATES)

    def to_dict(self) -> Dict:
        return {"world_id": self.world_id, **asdict(self)}

    def __str__(self) -> str:
        return self.world_id


BASELINE = WorldSpec()
N_IMPUTATIONS = 10


def all_world_specs(n_imputations: int = N_IMPUTATIONS, duplicates: Iterable[str] = DUPLICATE_READINGS) -> List[WorldSpec]:
    """Every world of the grid: span x case (x imputation for B) x duplicate reading, baseline first."""
    out = []
    for dup in duplicates:
        for span in ("union", "Q", "D", "intersection"):
            for case in ABSENCE_CASES:
                ks = range(n_imputations) if case == "B" else [None]
                out.extend(WorldSpec(span, case, k, dup) for k in ks)
    return out


def as_world(world: "WorldSpec | str | None") -> WorldSpec:
    if world is None:
        return BASELINE
    if isinstance(world, WorldSpec):
        return world
    return WorldSpec.parse(world)


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def world_dir(world: "WorldSpec | str | None" = None, *, create: bool = False) -> Path:
    d = V2_DIR / "worlds" / as_world(world).world_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def comparisons_dir(world: "WorldSpec | str | None", group: str, analysis: str, *, create: bool = True) -> Path:
    d = V2_DIR / "comparisons" / as_world(world).world_id / group / analysis
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def diagnostics_dir(*, create: bool = True) -> Path:
    d = V2_DIR / "diagnostics"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def concordant_sets_path() -> Path:
    return V2_DIR / "groups" / "concordant_sets.json"


def worlds_index_path() -> Path:
    return V2_DIR / "worlds.csv"


def read_world_config(world: "WorldSpec | str | None" = None) -> Dict:
    p = world_dir(world) / CONFIG_JSON
    return json.loads(p.read_text()) if p.exists() else {}


def write_world_config(world: "WorldSpec | str | None", extra: Optional[Dict] = None) -> Path:
    """Merge ``extra`` into the world's config.json (the spec fields are always rewritten)."""
    w = as_world(world)
    d = world_dir(w, create=True)
    record = {**read_world_config(w), **(extra or {}), **w.to_dict()}
    (d / CONFIG_JSON).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return d / CONFIG_JSON


def world_null_path(world: "WorldSpec | str | None" = None) -> Path:
    """The null parquet for a world: local file, else ``config.json:null_source``."""
    d = world_dir(world)
    local = d / NULL_PARQUET
    if local.exists():
        return local
    source = read_world_config(world).get("null_source")
    if source:
        p = Path(source)
        return p if p.is_absolute() else REPO_ROOT / p
    raise FileNotFoundError(f"No null distribution for world {as_world(world)}: neither {local} nor config.json:null_source")


def gcs_results_prefix(world: "WorldSpec | str | None" = None) -> str:
    """Where a world's per-batch JSON results live in the bucket.

    The baseline keeps the historical flat ``results/`` prefix so the
    published batches remain addressable; every other world gets its own
    sub-prefix.
    """
    w = as_world(world)
    return "results/" if w == BASELINE else f"results/{w.world_id}/"


# --------------------------------------------------------------------------
# Index and concordant sets
# --------------------------------------------------------------------------

INDEX_COLUMNS = ["world_id", "span", "absence_case", "imputation", "duplicates", "built_at", "git_commit", "data_sha256_questionnaire", "data_sha256_inhaler", "note"]


def register_world(world: "WorldSpec | str | None", *, git_commit: str = "", data_hashes: Optional[Dict[str, str]] = None, note: str = "") -> Path:
    """Insert or replace this world's row in ``results/v2/worlds.csv``."""
    w = as_world(world)
    path = worlds_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _shared_file_lock():
        return _register_world(w, path, git_commit, data_hashes, note)


def _register_world(w: WorldSpec, path: Path, git_commit: str, data_hashes: Optional[Dict[str, str]], note: str) -> Path:
    rows: List[Dict] = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["world_id"] != w.world_id]
    hashes = data_hashes or {}
    rows.append({
        "world_id": w.world_id, "span": w.span, "absence_case": w.absence_case,
        "imputation": "" if w.imputation is None else w.imputation,
        "duplicates": w.duplicates,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "data_sha256_questionnaire": hashes.get("anonym_aamos00_dailyquestionnaire_dt.csv", ""),
        "data_sha256_inhaler": hashes.get("anonym_aamos00_smartinhaler_dt.csv", ""),
        "note": note,
    })
    rows.sort(key=lambda r: r["world_id"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def list_worlds() -> List[str]:
    path = worlds_index_path()
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r["world_id"] for r in csv.DictReader(f)]


def load_concordant_sets() -> Dict[str, Dict[str, List[int]]]:
    p = concordant_sets_path()
    return json.loads(p.read_text()) if p.exists() else {}


def write_concordant_sets(world: "WorldSpec | str | None", sets: Dict[str, Iterable[int]]) -> Path:
    """Merge ``{summary spec key: [patients]}`` for this world into the generated JSON."""
    w = as_world(world)
    p = concordant_sets_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _shared_file_lock():
        all_sets = load_concordant_sets()
        all_sets[w.world_id] = {m: sorted(int(p) for p in ps) for m, ps in sets.items()}
        ordered = {k: all_sets[k] for k in sorted(all_sets)}
        p.write_text(json.dumps(ordered, indent=2) + "\n")
    return p


@contextmanager
def _shared_file_lock() -> Iterator[None]:
    """Serialise read-modify-write of the shared index files across processes (parallel world builds)."""
    V2_DIR.mkdir(parents=True, exist_ok=True)
    with open(V2_DIR / ".index.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
