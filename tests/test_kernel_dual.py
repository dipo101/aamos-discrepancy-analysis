"""Item 6: the kernel computes both correlation types per configuration."""

from __future__ import annotations

import numpy as np
import pytest

from aamos_concordance import evaluate_permutation, generate_param_combinations, run_single_permutation
from tests.synthetic import make_dataset


def _nan_equal(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)



@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_permutation_matches_one_type_interface_for_both_types(seed):
    q, inh = make_dataset(seed)
    combos = generate_param_combinations()
    for uk in q["user_key"].unique():
        qq, ii = q[q["user_key"] == uk], inh[inh["user_key"] == uk]
        both, _ = evaluate_permutation(uk, 3, qq, ii, combos, 42)
        assert set(both) == {"spearman", "pearson"}
        assert len(both["spearman"]) == len(both["pearson"]) == len(combos)
        for t in ("spearman", "pearson"):
            one, _ = run_single_permutation(uk, 3, qq, ii, combos, 42, t)
            assert _nan_equal(both[t], one), (uk, t)


def test_both_types_invalid_in_the_same_places_except_the_unit_circle_rule():
    q, inh = make_dataset(1)
    combos = generate_param_combinations()
    qq, ii = q[q["user_key"] == 104], inh[inh["user_key"] == 104]
    both, _ = evaluate_permutation(104, 0, qq, ii, combos, 42)
    s, p = np.asarray(both["spearman"]), np.asarray(both["pearson"])
    # Structural invalidity (too few rows, constant column) hits both; only |r|==1 can differ.
    assert (np.isnan(s) & ~np.isnan(p)).sum() + (~np.isnan(s) & np.isnan(p)).sum() <= 3


def test_run_single_permutation_rejects_unknown_type():
    q, inh = make_dataset(0)
    with pytest.raises(ValueError):
        run_single_permutation(100, 0, q, inh, generate_param_combinations()[:2], 42, "kendall")
