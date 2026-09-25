"""Look-ahead windows: excluded from the primary configuration set, kept in effective_lookahead."""

from __future__ import annotations

from aamos_concordance.configs import (
    LOOKAHEAD_WINDOWS,
    config_indices_for,
    effective_config_indices,
    generate_param_combinations,
    is_lookahead,
)
from aamos_concordance.summary import PRIMARY, SummarySpec


def test_lookahead_windows_are_chunk_12_and_calendar_day_0():
    combos = generate_param_combinations()
    flagged = [c for c in combos if is_lookahead(c)]
    assert len(flagged) == 24  # 2 windows x 2 filter x 6 methods
    kinds = {(c["use_daily_max_windows"], c["use_calendar_days"], c["timestamp_window"]) for c in flagged}
    assert kinds == {(True, False, 12), (False, True, 12)}
    assert LOOKAHEAD_WINDOWS == {("fixed_chunk", 12), ("calendar", 0)}


def test_effective_is_effective_lookahead_without_the_lookahead_windows():
    combos = generate_param_combinations()
    for t in ("spearman", "pearson"):
        for case in "ABC":
            with_la = config_indices_for("effective_lookahead", t, case)
            primary = config_indices_for("effective", t, case)
            assert with_la == effective_config_indices(t, case)
            assert primary == [i for i in with_la if not is_lookahead(combos[i])]
            assert len(primary) == len(with_la) * 8 // 10  # 8 of the 10 distinct windows


def test_primary_spec_uses_the_set_without_lookahead():
    assert PRIMARY.config_set == "effective"
    assert set(PRIMARY.config_indices) < set(SummarySpec("effective_lookahead").config_indices)
    assert not any(is_lookahead(generate_param_combinations()[i]) for i in PRIMARY.config_indices)
