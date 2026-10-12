"""Parameter search, and what happens to the winner on data it never saw.

This is hyperparameter tuning with a train/test split: search every
fast/slow window pair on the in-sample years, pick the best, then score it
on the out-of-sample years.
"""

from dataclasses import dataclass

import numpy as np

from .backtest import crossover_positions, prices_from_returns, sharpe, strategy_returns

FAST_WINDOWS = tuple(range(5, 55, 5))
SLOW_WINDOWS = tuple(range(20, 260, 10))


def parameter_grid(fast=FAST_WINDOWS, slow=SLOW_WINDOWS) -> list[tuple[int, int]]:
    return [(f, s) for f in fast for s in slow if f < s]


def grid_sharpes(returns: np.ndarray, grid: list[tuple[int, int]]) -> np.ndarray:
    """Sharpe of every (fast, slow) pair on one return series."""
    prices = prices_from_returns(returns)
    return np.array(
        [sharpe(strategy_returns(returns, crossover_positions(prices, f, s))) for f, s in grid]
    )


@dataclass
class OverfitResult:
    grid: list[tuple[int, int]]
    in_sample: np.ndarray
    out_of_sample: np.ndarray
    best_index: int

    @property
    def best_params(self) -> tuple[int, int]:
        return self.grid[self.best_index]

    @property
    def best_in_sample(self) -> float:
        return float(self.in_sample[self.best_index])

    @property
    def best_out_of_sample(self) -> float:
        return float(self.out_of_sample[self.best_index])

    @property
    def rank_out_of_sample(self) -> int:
        """Where the in-sample winner ranks out of sample (1 = best)."""
        order = np.argsort(-self.out_of_sample)
        return int(np.where(order == self.best_index)[0][0]) + 1


def split(returns: np.ndarray, in_sample_fraction: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    cut = int(len(returns) * in_sample_fraction)
    return returns[:cut], returns[cut:]


def search(returns: np.ndarray, in_sample_fraction: float = 0.7) -> OverfitResult:
    """Pick the best pair in sample, then score every pair out of sample.

    The out-of-sample backtest is run on the full history and only the
    out-of-sample days are scored, so the moving averages are warmed up
    with real prior data rather than starting cold.
    """
    grid = parameter_grid()
    ins, _ = split(returns, in_sample_fraction)
    cut = len(ins)
    prices = prices_from_returns(returns)
    in_s = grid_sharpes(ins, grid)
    out_s = np.array(
        [
            sharpe(strategy_returns(returns, crossover_positions(prices, f, s))[cut:])
            for f, s in grid
        ]
    )
    return OverfitResult(grid, in_s, out_s, int(np.argmax(in_s)))
