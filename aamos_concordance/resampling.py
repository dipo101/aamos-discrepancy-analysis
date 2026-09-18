"""Resampling risk of a Monte Carlo permutation p-value (v2 item 11).

A sampled permutation test estimates the exact p-value ``p`` with
``p_hat = (S + 1) / (n + 1)`` from ``S`` exceedances in ``n`` draws. What
matters for classification is not ``p_hat`` itself but whether it lands on
the same side of the decision threshold ``alpha`` as ``p`` would. The
*resampling risk* is the probability that it does not.

Two quantities are reported for every sampled patient (exact-enumeration
patients have no resampling risk):

* :func:`resampling_risk`: the posterior probability, under a Jeffreys
  Beta(1/2, 1/2) prior on ``p``, that the true ``p`` lies on the other side
  of ``alpha`` from ``p_hat``. This is the risk *of the p-value actually
  reported*, at the full sample size.
* :func:`sequential_stopping`: the point at which a sequential rule with a
  uniformly bounded resampling risk would have stopped generating
  permutations, applied post hoc to the draws in the order they were
  generated. The rule follows the design of Gandy (2009, JASA 104:1504-1511):
  check after each batch whether a confidence interval for ``p`` excludes
  ``alpha``, spending the overall risk budget ``epsilon`` across checks so
  that the total probability of a wrong stop is at most ``epsilon``. The
  interval is the exact Clopper-Pearson interval at level ``epsilon_n``,
  with ``epsilon_n = epsilon * n / (n + kappa)`` (Gandy's spending
  sequence). The reported stopping size shows how many permutations the
  decision actually needed; the full sample is still what the p-value uses,
  so the two views are a superset, not a replacement.

The exceedance indicators come from the centred two-tailed rule
(``|null - centre| >= |observed - centre|``) with the centre estimated from
the full sample; the sequential path therefore asks "at what n would the
decision have been settled", not "what would a strictly online rule have
seen", which is the honest post-hoc reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats

DEFAULT_EPSILON = 1e-3
DEFAULT_BATCH = 100
DEFAULT_MIN_N = 500
GANDY_KAPPA = 1000.0


def resampling_risk(n_exceed: int, n_perm: int, alpha: float) -> float:
    """P(true p on the other side of alpha from p_hat), Jeffreys posterior on p."""
    if n_perm <= 0:
        return np.nan
    a, b = n_exceed + 0.5, n_perm - n_exceed + 0.5
    p_hat = (n_exceed + 1) / (n_perm + 1)
    below = stats.beta.cdf(alpha, a, b)  # P(p < alpha)
    return float(1.0 - below) if p_hat < alpha else float(below)


@dataclass(frozen=True)
class SequentialResult:
    stopped: bool           # the rule reached a decision before the sample ran out
    n_at_stop: int          # permutations consumed when it stopped (== n_perm if it did not stop)
    decision_at_stop: Optional[bool]   # True = p < alpha at the stop; None if it did not stop
    matches_full_sample: Optional[bool]  # decision at stop == decision with the full sample
    epsilon: float


def sequential_stopping(
    exceed: Sequence[bool],
    alpha: float,
    *,
    epsilon: float = DEFAULT_EPSILON,
    batch: int = DEFAULT_BATCH,
    min_n: int = DEFAULT_MIN_N,
    kappa: float = GANDY_KAPPA,
) -> SequentialResult:
    """Post-hoc sequential stopping on an ordered exceedance sequence."""
    x = np.asarray(exceed, dtype=bool)
    n_total = len(x)
    if n_total == 0:
        return SequentialResult(False, 0, None, None, epsilon)
    cum = np.cumsum(x)
    full_decision = bool((cum[-1] + 1) / (n_total + 1) < alpha)
    n = min_n
    while n <= n_total:
        s = int(cum[n - 1])
        eps_n = epsilon * n / (n + kappa)
        lo = stats.beta.ppf(eps_n / 2, s, n - s + 1) if s > 0 else 0.0
        hi = stats.beta.ppf(1 - eps_n / 2, s + 1, n - s) if s < n else 1.0
        if hi < alpha:
            return SequentialResult(True, n, True, full_decision is True, epsilon)
        if lo > alpha:
            return SequentialResult(True, n, False, full_decision is False, epsilon)
        n += batch
    return SequentialResult(False, n_total, None, None, epsilon)
