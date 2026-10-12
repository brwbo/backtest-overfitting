import numpy as np
import pytest

from skill_or_luck import montecarlo, overfit
from skill_or_luck.backtest import (
    crossover_positions,
    max_drawdown,
    moving_average,
    prices_from_returns,
    sharpe,
)
from skill_or_luck.ml import LogisticRegression, features, fit_and_position
from skill_or_luck.permutation import permutation_test


def test_moving_average_matches_naive():
    p = np.arange(1.0, 11.0)
    ma = moving_average(p, 3)
    assert np.isnan(ma[:2]).all()
    assert ma[2:] == pytest.approx([np.mean(p[i - 2 : i + 1]) for i in range(2, 10)])


def test_positions_use_only_yesterdays_information():
    # Changing today's price must not change today's position.
    rng = np.random.default_rng(0)
    prices = prices_from_returns(rng.normal(0, 0.01, 300))
    before = crossover_positions(prices, 5, 20)
    bumped = prices.copy()
    bumped[-1] *= 2
    after = crossover_positions(bumped, 5, 20)
    assert before[-1] == after[-1]


def test_sharpe_of_flat_returns_is_zero():
    assert sharpe(np.zeros(100)) == 0.0


def test_max_drawdown():
    # Up 10%, then down 50%: equity 1.1 -> 0.55, a 50% drawdown.
    assert max_drawdown(np.array([0.1, -0.5])) == pytest.approx(0.5)


def test_grid_only_has_fast_below_slow():
    assert all(f < s for f, s in overfit.parameter_grid())


def test_permutation_p_value_bounds():
    rng = np.random.default_rng(1)
    res = permutation_test(rng.normal(0, 1, 50), np.mean, n_permutations=99)
    assert 0 < res.p_value <= 1


def test_permutation_detects_a_real_pattern():
    # Returns that alternate sign carry a pattern in their ORDER; a statistic
    # that rewards that pattern should beat almost every shuffle.
    r = np.tile([0.01, -0.01], 200)
    flips = lambda x: float(np.mean(np.sign(x[1:]) != np.sign(x[:-1])))
    assert permutation_test(r, flips, n_permutations=199).p_value < 0.01


def test_search_on_pure_noise_is_not_significant():
    # On a random walk any "edge" is luck, so the search must not look skilful.
    rng = np.random.default_rng(2)
    r = rng.normal(0.0003, 0.01, 1500)
    grid = overfit.parameter_grid(fast=(5, 10, 20), slow=(50, 100))
    res = permutation_test(r, lambda x: float(np.max(overfit.grid_sharpes(x, grid))), n_permutations=60)
    assert res.p_value > 0.05


def test_logistic_regression_learns_a_separable_rule():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    model = LogisticRegression().fit(X, y)
    assert np.mean((model.predict_proba(X) > 0.5) == y) > 0.95


def test_features_align_with_labels():
    r = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0])
    X, y = features(r, lags=2)
    assert X[0].tolist() == [-2.0, 1.0]  # most recent first
    assert y.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]


def test_ml_positions_have_the_right_lengths():
    rng = np.random.default_rng(4)
    train, test = rng.normal(0, 0.01, 400), rng.normal(0, 0.01, 100)
    _, acc, in_pos, out_pos = fit_and_position(train, test)
    assert len(in_pos) == 395 and len(out_pos) == 100 and 0 <= acc <= 1


def test_fat_tails_have_same_volatility_but_worse_extremes():
    normal = montecarlo.simulate_returns(0, 0.01, 2000, 200, fat_tails=False)
    fat = montecarlo.simulate_returns(0, 0.01, 2000, 200, fat_tails=True)
    assert np.std(fat) == pytest.approx(np.std(normal), rel=0.1)
    assert fat.min() < normal.min()


def test_volatility_matched_test_keeps_flat_day_count():
    from skill_or_luck.permutation import volatility_matched_test

    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 500)
    pos = np.ones(500)
    pos[:10] = 0
    res = volatility_matched_test(r, pos, sharpe, n_draws=50)
    assert len(res.null) == 50 and 0 < res.p_value <= 1
