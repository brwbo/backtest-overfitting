"""Permutation test: could the best strategy's profit have come from luck?

Null hypothesis: the order of daily returns carries no information a
strategy can exploit. Shuffling the returns keeps every day's move but
destroys any pattern in their order. If the strategy has real skill, its
result on the real order should beat almost all shuffled orders.

The search is re-run on every shuffle and the BEST result kept. Testing
only the winner, without repeating the search, would ignore that we tried
many strategies and picked the luckiest, which is the mistake this whole
project is about.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PermutationResult:
    observed: float
    null: np.ndarray

    @property
    def p_value(self) -> float:
        # +1 on both sides so the p-value can never be exactly zero.
        return float((np.sum(self.null >= self.observed) + 1) / (len(self.null) + 1))


def permutation_test(returns, statistic, n_permutations: int = 500, seed: int = 0) -> PermutationResult:
    """`statistic(returns) -> float` is computed on the real and shuffled series."""
    rng = np.random.default_rng(seed)
    observed = statistic(returns)
    null = np.array([statistic(rng.permutation(returns)) for _ in range(n_permutations)])
    return PermutationResult(observed, null)


def volatility_matched_test(
    returns, positions, statistic, pool_multiple: int = 5, n_draws: int = 500, seed: int = 0
) -> PermutationResult:
    """Is a long-or-flat strategy picking its flat days well, or just dodging volatile ones?

    Shuffling returns assumes days are interchangeable, but big moves cluster
    together, so a strategy that steps aside after big moves looks skilful
    under a shuffle without predicting direction at all. This null keeps the
    real return order and instead sits out the same number of days, chosen at
    random from the days that follow the largest absolute moves.
    """
    rng = np.random.default_rng(seed)
    flat = int(np.sum(positions == 0))
    observed = statistic(returns * positions)
    if flat == 0:
        return PermutationResult(observed, np.full(n_draws, observed))
    prior_move = np.abs(np.concatenate(([0.0], returns[:-1])))
    pool = np.argsort(-prior_move)[: min(len(returns), flat * pool_multiple)]
    null = []
    for _ in range(n_draws):
        pos = np.ones_like(returns)
        pos[rng.choice(pool, size=flat, replace=False)] = 0.0
        null.append(statistic(returns * pos))
    return PermutationResult(observed, np.array(null))
