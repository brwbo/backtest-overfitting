"""Run every check on one ticker and write results.json plus three charts.

    uv run skill-or-luck SPY
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import montecarlo, overfit
from .backtest import sharpe
from .data import load_returns
from .ml import LAGS, fit_and_position
from .permutation import permutation_test, volatility_matched_test

OUT = Path(__file__).resolve().parents[2] / "results"


def run(ticker: str, start: str, end: str, n_perm: int, n_paths: int) -> dict:
    dates, returns = load_returns(ticker, start, end)
    ins, oos = overfit.split(returns)
    cut_date = dates[len(ins)]

    # 1-2. Parameter search: in-sample winner vs out of sample.
    search = overfit.search(returns)

    # 3. Permutation test on the whole search, not just the winner.
    grid = search.grid
    perm = permutation_test(ins, lambda r: float(np.max(overfit.grid_sharpes(r, grid))), n_perm)

    # ML strategy: logistic regression, same train/test split.
    _, ml_accuracy, ins_pos, oos_pos = fit_and_position(ins, oos)
    ml_in, ml_out = ins[LAGS:] * ins_pos, oos * oos_pos
    # Out-of-sample permutation: keep the trained model's positions fixed and
    # shuffle the test returns. Real skill means the timing beats random order.
    ml_perm = permutation_test(oos, lambda r: sharpe(r * oos_pos), n_perm, seed=1)
    ml_vol = volatility_matched_test(oos, oos_pos, sharpe, n_draws=n_perm, seed=2)

    # 4. Monte Carlo with the in-sample winner, normal vs fat tails.
    mu, sigma = float(np.mean(returns)), float(np.std(returns, ddof=1))
    mc_normal = montecarlo.run(mu, sigma, search.best_params, n_paths=n_paths, fat_tails=False)
    mc_fat = montecarlo.run(mu, sigma, search.best_params, n_paths=n_paths, fat_tails=True)

    res = {
        "ticker": ticker,
        "period": [str(dates[0].date()), str(dates[-1].date())],
        "split_date": str(cut_date.date()),
        "strategies_tried": len(grid),
        "buy_and_hold_sharpe": {"in_sample": sharpe(ins), "out_of_sample": sharpe(oos)},
        "crossover": {
            "best_params": list(search.best_params),
            "best_in_sample_sharpe": search.best_in_sample,
            "its_out_of_sample_sharpe": search.best_out_of_sample,
            "its_out_of_sample_rank": search.rank_out_of_sample,
            "median_out_of_sample_sharpe": float(np.median(search.out_of_sample)),
            "permutation_p_value": perm.p_value,
            "permutations": n_perm,
        },
        "logistic_regression": {
            "in_sample_sharpe": sharpe(ml_in),
            "out_of_sample_sharpe": sharpe(ml_out),
            "in_sample_accuracy": ml_accuracy,
            "time_in_market_out_of_sample": float(np.mean(oos_pos)),
            "out_of_sample_permutation_p_value": ml_perm.p_value,
            "out_of_sample_volatility_matched_p_value": ml_vol.p_value,
        },
        "monte_carlo": {
            "paths": n_paths,
            "drawdown_95th_percentile_normal": mc_normal.percentile_drawdown(95),
            "drawdown_95th_percentile_fat_tails": mc_fat.percentile_drawdown(95),
            "worst_single_day_median_normal": float(np.median(mc_normal.worst_days)),
            "worst_single_day_median_fat_tails": float(np.median(mc_fat.worst_days)),
        },
    }
    charts(OUT / ticker, ticker, search, perm, mc_normal, mc_fat)
    return res


def charts(out: Path, ticker, search, perm, mc_normal, mc_fat) -> None:
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(search.in_sample, search.out_of_sample, s=12, alpha=0.6, label="every strategy tried")
    ax.scatter([search.best_in_sample], [search.best_out_of_sample], s=80, c="red", label="in-sample winner")
    ax.axhline(0, lw=0.8, c="grey")
    ax.set_xlabel("Sharpe on the years used to pick it (in sample)")
    ax.set_ylabel("Sharpe on years it never saw (out of sample)")
    ax.set_title(f"{ticker}: does the best backtest stay the best?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "overfitting.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm.null, bins=40, alpha=0.7, label="best Sharpe on shuffled returns")
    ax.axvline(perm.observed, c="red", lw=2, label=f"real data (p = {perm.p_value:.2f})")
    ax.set_xlabel("Best in-sample Sharpe after searching every strategy")
    ax.set_title(f"{ticker}: is the winner better than luck?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "permutation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(mc_normal.drawdowns.max(), mc_fat.drawdowns.max()), 50)
    ax.hist(mc_normal.drawdowns, bins=bins, alpha=0.6, label="normal returns")
    ax.hist(mc_fat.drawdowns, bins=bins, alpha=0.6, label="fat-tailed returns (Student-t, 3 dof)")
    ax.set_xlabel("Maximum drawdown of the strategy over 10 simulated years")
    ax.set_title("Same average and volatility, different tails")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "monte_carlo.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Is a trading strategy's profit skill or luck?")
    p.add_argument("ticker", nargs="?", default="SPY")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--permutations", type=int, default=300)
    p.add_argument("--paths", type=int, default=1000)
    a = p.parse_args()
    res = run(a.ticker, a.start, a.end, a.permutations, a.paths)
    (OUT / a.ticker / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
