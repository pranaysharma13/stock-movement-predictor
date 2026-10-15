"""
Tests for src/evaluate.py, for both tasks: walk-forward never sees the
future, the backtest arithmetic is right, and the whole pipeline finds a
planted pattern but finds nothing in pure noise.
"""


import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.data import make_planted_pattern, make_random_walk, make_volatility_clusters
from src.evaluate import (accuracy_table, backtest, backtest_table, baseline_predictions,
                          first_test_year, get_models, luck_check, performance,
                          vol_baseline_predictions, walk_forward)
from src.features import ALL_FEATURES, FEATURES, VOL_FEATURES, build_dataset, build_vol_dataset


@pytest.fixture(scope="module")
def noise():
    return build_dataset(make_random_walk(n_days=3000, seed=3))


@pytest.fixture(scope="module")
def pattern():
    return build_dataset(make_planted_pattern(n_days=3000, seed=3))


# ---------------------------- walk-forward ------------------------------------

class RecordingModel(ClassifierMixin, BaseEstimator):
    """A stand-in model that remembers which dates it was trained on.

    walk_forward copies the model for every year, so the record is kept on
    the class rather than on one copy.
    """

    seen = []

    def fit(self, X, y):
        RecordingModel.seen.append((X.index.min(), X.index.max()))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def test_walk_forward_never_trains_on_the_year_it_predicts(noise):
    RecordingModel.seen = []
    preds = walk_forward(RecordingModel(), noise, first_year=2010)

    test_years = sorted(preds.index.year.unique())
    assert len(RecordingModel.seen) == len(test_years)
    for (_, last_training_day), year in zip(RecordingModel.seen, test_years):
        assert last_training_day < pd.Timestamp(year, 1, 1)


def test_walk_forward_predicts_every_test_day_exactly_once(noise):
    preds = walk_forward(RecordingModel(), noise, first_year=2010)
    expected = noise.index[noise.index.year >= 2010]
    assert preds.index.equals(expected)


def test_walk_forward_runs_with_its_defaults(noise):
    """The simplest possible call must work: no first year, default features.

    This is the call that broke when walk_forward's first line was left at
    an old version; a test like this catches that the moment the file is saved.
    """

    preds = walk_forward(get_models()["Logistic regression"], noise)
    assert preds["prob_up"].between(0, 1).all()
    assert set(preds["pred"].unique()) <= {0, 1}
    assert preds.index[0].year == first_test_year(noise)


def test_walk_forward_accepts_a_feature_list(noise):
    preds = walk_forward(get_models()["Logistic regression"], noise, 2012, features=ALL_FEATURES)
    assert preds.index[0].year == 2012


def test_gradient_boosting_runs(noise):
    short = noise[noise.index.year <= 2012]
    preds = walk_forward(get_models()["Gradient boosting"], short, 2012)
    assert len(preds) == (short.index.year == 2012).sum()


def test_first_test_year_has_enough_history_before_it(noise):
    year = first_test_year(noise, min_train_days=1000)
    assert (noise.index.year < year).sum() >= 1000
    assert (noise.index.year < year - 1).sum() < 1000


def test_first_test_year_refuses_too_little_data(noise):
    with pytest.raises(ValueError, match="Not enough history"):
        first_test_year(noise.iloc[:500])


# ---------------------------- baselines and accuracy ------------------------------------

def test_baselines(noise):
    base = baseline_predictions(noise)
    assert (base["Always up"] == 1).all()
    assert base["Repeat last move"].equals((noise["ret_1d"] > 0).astype(int))
    assert base["Coin flip"].isin([0, 1]).all()


def test_accuracy_table_on_a_tiny_example():
    days = pd.bdate_range("2020-01-01", periods=4)
    target = pd.Series([1, 0, 1, 1], index=days)
    table = accuracy_table(
        {"model": pd.Series([1, 1, 1, 0], index=days), "Always up": pd.Series(1, index=days)},
        target,
    )
    assert table.loc["model", "accuracy"] == pytest.approx(0.50)
    assert table.loc["Always up", "accuracy"] == pytest.approx(0.75)
    assert table.loc["model", "predicted up"] == pytest.approx(0.75)
    assert not table.loc["Always up", "beats always-up?"]


# ---------------------------- backtest ------------------------------------

def test_backtest_arithmetic_by_hand():
    """Hold, sell, buy back, hold. Cost 0.1% on each of the three trades."""

    days = pd.bdate_range("2020-01-01", periods=4)
    next_ret = pd.Series([0.10, -0.05, 0.02, 0.01], index=days)
    position = pd.Series([1, 0, 1, 1], index=days)

    daily = backtest(position, next_ret, cost=0.001)

    expected = [0.10 - 0.001,   # buy on day 1, earn +10%
                0.00 - 0.001,   # sell: sit out the -5% day
                0.02 - 0.001,   # buy back, earn +2%
                0.01]           # keep holding: no trade, no cost
    np.testing.assert_allclose(daily.values, expected)


def test_performance_by_hand():
    days = pd.bdate_range("2020-01-01", periods=3)
    stats = performance(pd.Series([0.10, -0.50, 0.20], index=days))
    # $1 -> 1.10 -> 0.55 -> 0.66
    assert stats["total return"] == pytest.approx(0.66 - 1)
    assert stats["max drawdown"] == pytest.approx(0.55 / 1.10 - 1)


def test_luck_check_spots_a_perfect_strategy(noise):
    """Sitting out exactly the down days can't be matched by random cash days."""

    test = noise[noise.index.year >= 2010]
    perfect = (test["next_ret"] > 0).astype(int)
    result = luck_check(perfect, test["next_ret"], n_trials=200)
    assert result["share of random >="] == 0
    assert result["cash days"] == (perfect == 0).sum()


# ---------------------------- the pipeline checks ------------------------------------

def _check(data):
    start = first_test_year(data)
    test = data[data.index.year >= start]
    preds = {
        "Logistic regression": walk_forward(get_models()["Logistic regression"], data, start)["pred"],
        "Always up": pd.Series(1, index=test.index),
    }
    return accuracy_table(preds, test["target"])


def test_pipeline_finds_the_planted_pattern(pattern):
    table = _check(pattern)
    assert table.loc["Logistic regression", "accuracy"] > 0.56
    assert table.loc["Logistic regression", "beats always-up?"]


def test_pipeline_finds_nothing_in_noise(noise):
    table = _check(noise)
    model = table.loc["Logistic regression", "accuracy"]
    always_up = table.loc["Always up", "accuracy"]
    assert not table.loc["Logistic regression", "beats always-up?"]
    assert abs(model - always_up) < 0.04


# ---------------------------- Part 2: move size ------------------------------------


@pytest.fixture(scope="module")
def calm_and_wild():
    return build_vol_dataset(make_volatility_clusters(n_days=3000, seed=3))


@pytest.fixture(scope="module")
def constant_vol():
    return build_vol_dataset(make_random_walk(n_days=3000, seed=3))


def test_vol_baselines(calm_and_wild):
    base = vol_baseline_predictions(calm_and_wild)
    assert (base["Always small"] == 0).all() and (base["Always big"] == 1).all()
    assert base["Same as today"].equals((calm_and_wild["abs_ret_rel"] > 1).astype(int))


def test_accuracy_table_against_a_named_baseline():
    days = pd.bdate_range("2020-01-01", periods=4)
    target = pd.Series([1, 0, 1, 0], index=days)
    table = accuracy_table(
        {"model": pd.Series([1, 0, 1, 0], index=days), "Same as today": pd.Series([1, 1, 1, 1], index=days)},
        target, baseline="Same as today",
    )
    assert "beats same as today?" in table.columns
    assert "beats always-up?" not in table.columns
    assert table.loc["Same as today", "accuracy"] == pytest.approx(0.5)


def test_backtest_with_half_positions_by_hand():
    """Full, half, half, full. Moving between 1 and 0.5 trades half the money."""

    days = pd.bdate_range("2020-01-01", periods=4)
    next_ret = pd.Series([0.10, -0.04, 0.02, 0.01], index=days)
    position = pd.Series([1, 0.5, 0.5, 1], index=days)
    daily = backtest(position, next_ret, cost=0.001)
    expected = [0.10 - 0.001,           # buy the full position
                -0.02 - 0.0005,         # sell half: half the -4% day
                0.01,                   # stay at half: half the +2% day, no trade
                0.01 - 0.0005]          # buy the other half back
    np.testing.assert_allclose(daily.values, expected)


def test_backtest_table_counts_every_position_change():
    days = pd.bdate_range("2020-01-01", periods=5)
    next_ret = pd.Series(0.0, index=days)
    stats, _ = backtest_table({"x": pd.Series([1, 0.5, 0.5, 1, 0.5], index=days)}, next_ret)
    assert stats.loc["x", "trades"] == 4          # initial buy + three changes
    assert stats.loc["x", "days in market"] == pytest.approx(0.7)


def test_luck_check_with_half_positions_and_sharpe(calm_and_wild):
    test = calm_and_wild[calm_and_wild.index.year >= 2010]
    position = 1 - 0.5 * (test["abs_ret_rel"] > 1).astype(int)
    result = luck_check(position, test["next_ret"], n_trials=50, metric="Sharpe ratio")
    assert "Sharpe ratio" in result
    assert result["cash days"] == int((position < 1).sum())
    assert 0 <= result["share of random >="] <= 1


def _vol_check(data):
    start = first_test_year(data)
    test = data[data.index.year >= start]
    preds = {"Logistic regression": walk_forward(get_models()["Logistic regression"], data, start,
                                                 features=VOL_FEATURES)["pred"]}
    preds.update(vol_baseline_predictions(test).to_dict("series"))
    return accuracy_table(preds, test["target"], baseline="Same as today")


def test_pipeline_finds_volatility_clusters(calm_and_wild):
    table = _vol_check(calm_and_wild)
    assert table.loc["Logistic regression", "beats same as today?"]
    assert table.loc["Logistic regression", "accuracy"] > 0.55


def test_pipeline_finds_nothing_when_volatility_is_constant(constant_vol):
    table = _vol_check(constant_vol)
    assert not table.loc["Logistic regression", "beats same as today?"]
    assert abs(table.loc["Logistic regression", "accuracy"] - 0.5) < 0.04
