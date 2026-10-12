"""Core backtest maths: moving averages, positions, returns, Sharpe.

Every strategy here is long-or-flat. The position held on day t is decided
from information available at the close of day t-1, so there is no
look-ahead: the strategy never trades on a price it could not have seen.
"""

import numpy as np

TRADING_DAYS = 252


def prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    """Rebuild a price path from simple daily returns."""
    return start * np.cumprod(1.0 + returns)


def moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of the last `window` prices. NaN until enough history."""
    out = np.full(prices.shape, np.nan)
    if window > len(prices):
        return out
    csum = np.cumsum(np.insert(prices, 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def crossover_positions(prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """1 when the fast average is above the slow one, else 0.

    The signal computed at the close of day t is the position for day t+1,
    hence the shift by one.
    """
    fast_ma = moving_average(prices, fast)
    slow_ma = moving_average(prices, slow)
    signal = np.where(np.isnan(slow_ma), 0.0, (fast_ma > slow_ma).astype(float))
    return np.concatenate(([0.0], signal[:-1]))


def strategy_returns(returns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Daily returns earned by holding `positions` (already lagged)."""
    return returns * positions


def sharpe(daily_returns: np.ndarray) -> float:
    """Annualised Sharpe ratio, risk-free rate taken as zero."""
    sd = np.std(daily_returns, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(np.mean(daily_returns) / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(daily_returns: np.ndarray) -> float:
    """Largest peak-to-trough fall of the equity curve, as a positive fraction."""
    equity = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def annual_return(daily_returns: np.ndarray) -> float:
    """Compound annual growth rate."""
    total = np.prod(1.0 + daily_returns)
    years = len(daily_returns) / TRADING_DAYS
    return float(total ** (1.0 / years) - 1.0)
