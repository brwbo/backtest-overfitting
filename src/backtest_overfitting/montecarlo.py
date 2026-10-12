"""Monte Carlo: how bad can it get, under normal vs fat-tailed returns?

Simulates many price paths with the same average return and the same
volatility, once drawing daily returns from a normal distribution and once
from a Student-t with 3 degrees of freedom (fat tails: big moves happen far
more often). The strategy is run on every path and the worst cases compared.
"""

from dataclasses import dataclass

import numpy as np

from .backtest import crossover_positions, max_drawdown, prices_from_returns, strategy_returns

T_DOF = 3


def simulate_returns(mu: float, sigma: float, n_days: int, n_paths: int, fat_tails: bool, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if fat_tails:
        # A Student-t with dof v has variance v/(v-2); rescale to unit variance.
        z = rng.standard_t(T_DOF, size=(n_paths, n_days)) / np.sqrt(T_DOF / (T_DOF - 2))
    else:
        z = rng.standard_normal(size=(n_paths, n_days))
    # Floor at -99% so a single draw can never take a price below zero.
    return np.maximum(mu + sigma * z, -0.99)


@dataclass
class MonteCarloResult:
    drawdowns: np.ndarray
    worst_days: np.ndarray

    def percentile_drawdown(self, q: float = 95) -> float:
        return float(np.percentile(self.drawdowns, q))


def run(mu, sigma, params, n_days=2520, n_paths=2000, fat_tails=False, seed=0) -> MonteCarloResult:
    fast, slow = params
    paths = simulate_returns(mu, sigma, n_days, n_paths, fat_tails, seed)
    dds, worst = [], []
    for r in paths:
        strat = strategy_returns(r, crossover_positions(prices_from_returns(r), fast, slow))
        dds.append(max_drawdown(strat))
        worst.append(r.min())
    return MonteCarloResult(np.array(dds), np.array(worst))
