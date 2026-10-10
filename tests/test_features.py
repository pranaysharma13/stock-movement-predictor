"""
Tests for src/features.py: no peeking at the future, and correct labels.
"""


import numpy as np
import pandas as pd
import pytest

from src.data import make_random_walk
from src.features import ALL_FEATURES, FEATURES, build_dataset, make_features


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