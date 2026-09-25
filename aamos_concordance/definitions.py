"""Measurement definitions that differ between v1 and v2.

Two definitions were corrected after v1. Both are read at call time from
:data:`ACTIVE`, so the v1 behaviour can be reproduced exactly (tests pin the
refactor to the verbatim v1 code with ``use_definitions(V1)``):

* ``closed_chunk_end``: a fixed 24-hour chunk includes a record at exactly
  its end, like the rolling window (closed at both ends) and the calendar
  window ("until 23:59:59"). v1 used ``<``. With the end closed, the chunk
  starting 24 h back is the same interval as the rolling 24-hour window by
  construction. No device record falls exactly on a chunk end in AAMOS-00,
  so this changes no result on this dataset.
* ``self_report_top_code``: the questionnaire's top answer "12 or more" is
  stored as the number 12 (data dictionary: "0 = None, 1 = 1 to 2, 3 = 3 to
  4, 5 = 5 to 8, 9 = 9 to 12, 12 = 12 or more"). It is a band label, so it
  maps to the top category (Table 2: 12 for the plain methods, M or N for
  the data-driven ones). v1's range check put it in the 9-to-12 band,
  merging two distinct answers. Affects 4 answers (patients 113 and 473).
  It also makes the upper-bound hybrid its own Spearman class (80
  effective configurations instead of 60; see :mod:`aamos_concordance.configs`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class Definitions:
    closed_chunk_end: bool = True
    self_report_top_code: Optional[float] = 12


CURRENT = Definitions()
V1 = Definitions(closed_chunk_end=False, self_report_top_code=None)

ACTIVE: Definitions = CURRENT


@contextmanager
def use_definitions(defs: Definitions) -> Iterator[Definitions]:
    """Temporarily switch the active definitions (used to reproduce v1)."""
    global ACTIVE
    previous = ACTIVE
    ACTIVE = defs
    try:
        yield defs
    finally:
        ACTIVE = previous
