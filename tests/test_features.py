"""
Tests for src/features.py: no peeking at the future, and correct labels,
for both tasks (direction, and big vs. small moves).
"""


import numpy as np
import pandas as pd
import pytest

from src.data import make_random_walk
from src.features import (ALL_FEATURES, FEATURES, TYPICAL_WINDOW, VOL_FEATURES,
                          build_dataset, build_vol_dataset, make_features, make_vol_features)


@pytest.fixture
def prices():
    return make_random_walk(n_days=400, seed=1)


def test_no_look_ahead(prices):
    """Changing prices after day k must not change any feature on or before day k.

    If a feature secretly used tomorrow's data, scrambling the future would
    change today's value and this test would fail.
    """

    k = 250
    original = make_features(prices)

    scrambled = prices.copy()
    rng = np.random.default_rng(99)
    future = scrambled.index[k + 1:]
    scrambled.loc[future, "Close"] *= rng.uniform(0.5, 1.5, len(future))
    scrambled.loc[future, "Volume"] *= rng.uniform(0.5, 1.5, len(future))
    changed = make_features(scrambled)

    pd.testing.assert_frame_equal(original.iloc[: k + 1], changed.iloc[: k + 1])
    # Sanity check: the scramble really did change later features.
    assert not original.iloc[k + 1:].equals(changed.iloc[k + 1:])


def test_label_is_tomorrows_direction(prices):
    """target = 1 exactly when the next day's close is higher than today's."""

    data = build_dataset(prices)
    close = prices["Close"]
    tomorrow_higher = (close.shift(-1) > close).astype(int)
    pd.testing.assert_series_equal(
        data["target"], tomorrow_higher.loc[data.index], check_names=False
    )
    expected_ret = close.shift(-1) / close - 1
    np.testing.assert_allclose(data["next_ret"], expected_ret.loc[data.index])


def test_dataset_has_no_gaps_and_no_last_day(prices):
    """No missing values, and the last day (which has no tomorrow) is dropped."""

    data = build_dataset(prices)
    assert not data.isna().any().any()
    assert data.index[-1] < prices.index[-1]
    # The first ~50 days are warm-up for the 50-day average.
    assert data.index[0] >= prices.index[49]


def test_feature_values_on_a_simple_series():
    """On prices rising exactly 1% a day, the features have known values."""
    
    days = pd.bdate_range("2020-01-01", periods=80, name="Date")
    close = 100 * 1.01 ** np.arange(80)
    prices = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000.0},
        index=days,
    )
    f = make_features(prices).iloc[-1]
    assert f["ret_1d"] == pytest.approx(0.01)
    assert f["ret_5d"] == pytest.approx(1.01 ** 5 - 1)
    assert f["ret_20d"] == pytest.approx(1.01 ** 20 - 1)
    assert f["vol_20d"] == pytest.approx(0, abs=1e-12)        # perfectly steady
    assert f["volume_ratio"] == pytest.approx(0, abs=1e-12)   # volume never changes
    assert f["dist_ma50"] > 0                                 # rising price sits above its average


def test_feature_lists_match_what_is_calculated(prices):
    assert list(make_features(prices).columns) == ALL_FEATURES
    assert set(FEATURES) <= set(ALL_FEATURES)
    assert len(FEATURES) == 5


# ---------------------------- part 2: move size ------------------------------------

@pytest.fixture
def long_prices():
    return make_random_walk(n_days=700, seed=2)


def test_vol_features_no_look_ahead(long_prices):
    """Scrambling prices after day k must not change any move-size feature on or before day k."""

    k = 500
    original = make_vol_features(long_prices)
    scrambled = long_prices.copy()
    rng = np.random.default_rng(7)
    future = scrambled.index[k + 1:]
    for col in ["Close", "High", "Low", "Volume"]:
        scrambled.loc[future, col] *= rng.uniform(0.5, 1.5, len(future))
    changed = make_vol_features(scrambled)
    pd.testing.assert_frame_equal(original.iloc[: k + 1], changed.iloc[: k + 1])
    assert not original.iloc[k + 1:].equals(changed.iloc[k + 1:])


def test_typical_move_uses_only_the_past_year(long_prices):
    """typical_move on day t is the median move size over the 250 days ending on day t."""

    feats = make_vol_features(long_prices)
    moves = long_prices["Close"].pct_change().abs()
    t = 400
    expected = moves.iloc[t - TYPICAL_WINDOW + 1: t + 1].median()
    assert feats["typical_move"].iloc[t] == pytest.approx(expected)


def test_big_day_label(long_prices):
    """target = 1 exactly when tomorrow's move, up or down, beats today's typical move."""

    data = build_vol_dataset(long_prices)
    close = long_prices["Close"]
    tomorrow_move = (close.shift(-1) / close - 1).abs().loc[data.index]
    expected = (tomorrow_move > data["typical_move"]).astype(int)
    pd.testing.assert_series_equal(data["target"], expected, check_names=False)
    assert not data.isna().any().any()
    assert data.index[0] >= long_prices.index[TYPICAL_WINDOW]


def test_vol_feature_values_on_a_simple_series():
    """Prices rising exactly 1% a day, no intraday swing: every move equals the typical move."""

    days = pd.bdate_range("2020-01-01", periods=300, name="Date")
    close = 100 * 1.01 ** np.arange(300)
    prices = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1_000_000.0},
        index=days,
    )
    f = make_vol_features(prices).iloc[-1]
    assert f["typical_move"] == pytest.approx(0.01)
    assert f["abs_ret_rel"] == pytest.approx(1)
    assert f["vol_5d_rel"] == pytest.approx(0, abs=1e-9)
    assert f["range_rel"] == pytest.approx(0)


def test_vol_feature_list_matches(long_prices):
    cols = make_vol_features(long_prices).columns
    assert set(VOL_FEATURES) <= set(cols)
    assert "typical_move" in cols


def test_wide_range_is_the_recent_swing_rule(long_prices):
    """wide_range on day t: today's high-to-low swing beats its median over the 250 days ending on day t."""

    feats = make_vol_features(long_prices)
    day_range = (long_prices["High"] - long_prices["Low"]) / long_prices["Close"]
    for t in [300, 450, 650]:
        usual = day_range.iloc[t - TYPICAL_WINDOW + 1: t + 1].median()
        assert feats["wide_range"].iloc[t] == float(day_range.iloc[t] > usual)
    assert feats["wide_range"].iloc[: TYPICAL_WINDOW - 1].isna().all()   # not enough history yet
    data = build_vol_dataset(long_prices)
    assert set(data["wide_range"].unique()) <= {0, 1}
    assert "wide_range" not in VOL_FEATURES       # a baseline, not a model feature


# ---------------------------- phase 4.2 ------------------------------------

def test_prices_without_volume_get_a_clear_error(long_prices):
    """Indexes like ^VIX report zero volume every day; both tasks should say so plainly."""
    
    no_volume = long_prices.assign(Volume=0.0)
    for build in (build_dataset, build_vol_dataset):
        with pytest.raises(ValueError, match="no trading volume"):
            build(no_volume)


def test_too_little_history_gets_a_clear_error(long_prices):
    with pytest.raises(ValueError, match="No usable rows"):
        build_vol_dataset(long_prices.iloc[:200])