"""
Tests for src/data.py: the unfinished-day fix, the saved-file path, the
download path (with a fake yfinance, so no internet is needed), and the
fake prices used by the pipeline checks.
"""


import os
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import src.data as data_module
from src.data import (NEW_YORK, PRICE_COLUMNS, drop_unfinished_day, load_prices,
                      make_planted_pattern, make_random_walk, make_volatility_clusters)


def _prices_ending(last_day, n=5):
    days = pd.bdate_range(end=last_day, periods=n, name="Date")
    return pd.DataFrame({c: np.arange(n, dtype=float) + 1 for c in PRICE_COLUMNS}, index=days)


def _new_york(when):
    return datetime.fromisoformat(when).replace(tzinfo=NEW_YORK)



# ---------------------------- unfinished day ------------------------------------

def test_drops_today_when_downloaded_during_trading_hours():
    prices = _prices_ending("2026-10-09")
    result = drop_unfinished_day(prices, _new_york("2026-10-09 12:10"))
    assert result.index[-1] == pd.Timestamp("2026-10-08")
    assert len(result) == len(prices) - 1


@pytest.mark.parametrize("when", ["2026-10-09 17:00", "2026-10-10 09:00"])
def test_keeps_today_once_the_market_has_closed(when):
    prices = _prices_ending("2026-10-09")
    assert drop_unfinished_day(prices, _new_york(when)).index[-1] == pd.Timestamp("2026-10-09")


def test_keeps_yesterdays_close_during_today_trading():
    """Downloaded mid-morning, before today's row exists: nothing to drop."""

    prices = _prices_ending("2026-10-08")
    assert drop_unfinished_day(prices, _new_york("2026-10-09 10:00")).equals(prices)


def test_empty_prices_stay_empty():
    empty = pd.DataFrame(columns=PRICE_COLUMNS, index=pd.DatetimeIndex([], name="Date"))
    assert drop_unfinished_day(empty, _new_york("2026-10-09 12:00")).empty


# ---------------------------- saved file and download ------------------------------------

def test_saved_file_from_trading_hours_loses_its_last_day(tmp_path, monkeypatch):
    """A CSV saved at 12:10 pm on its last date is treated as unfinished."""

    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    _prices_ending("2026-10-09").to_csv(tmp_path / "SPY_from_2005-01-01.csv")
    saved = _new_york("2026-10-09 12:10").timestamp()
    os.utime(tmp_path / "SPY_from_2005-01-01.csv", (saved, saved))

    prices = load_prices("SPY", "2005-01-01")
    assert prices.index[-1] == pd.Timestamp("2026-10-08")


def test_download_path_with_fake_yfinance(tmp_path, monkeypatch):
    """Recent yfinance returns two-level columns and time-zone-aware dates; both are handled."""

    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    fake = make_random_walk(n_days=30, start="2010-01-04")
    raw = fake.copy()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["SPY"]], names=["Price", "Ticker"])
    raw.index = raw.index.tz_localize("America/New_York")
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *args, **kwargs: raw
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

    prices = load_prices("SPY", "2010-01-01", refresh=True)

    assert list(prices.columns) == PRICE_COLUMNS
    assert prices.index.tz is None
    assert prices.index[0] == pd.Timestamp("2010-01-04")   # local trading date, not shifted
    assert (tmp_path / "SPY_from_2010-01-01.csv").exists()


def test_download_that_returns_nothing_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *args, **kwargs: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    with pytest.raises(RuntimeError, match="No data came back"):
        load_prices("NOTATICKER", "2010-01-01", refresh=True)


# ---------------------------- fake prices ------------------------------------

@pytest.mark.parametrize("make", [make_random_walk, make_planted_pattern, make_volatility_clusters])
def test_fake_prices_look_like_real_prices(make):
    prices = make(n_days=500)
    assert list(prices.columns) == PRICE_COLUMNS
    assert len(prices) == 500
    assert (prices["Close"] > 0).all()
    assert (prices["High"] >= prices[["Open", "Close"]].max(axis=1)).all()
    assert (prices["Low"] <= prices[["Open", "Close"]].min(axis=1)).all()


def test_planted_pattern_reverses_about_60_percent_of_the_time():
    close = make_planted_pattern(n_days=5500)["Close"]
    up = (close.pct_change() > 0).iloc[1:]
    reversed_ = (up != up.shift(1)).iloc[1:]
    assert reversed_.mean() == pytest.approx(0.60, abs=0.02)
    assert up.mean() == pytest.approx(0.50, abs=0.03)


def test_random_walk_has_no_reversal_pattern():
    close = make_random_walk(n_days=5500)["Close"]
    up = (close.pct_change() > 0).iloc[1:]
    reversed_ = (up != up.shift(1)).iloc[1:]
    assert reversed_.mean() == pytest.approx(0.50, abs=0.03)


def _move_size_persistence(prices):
    """Correlation between today's move size and tomorrow's."""

    moves = prices["Close"].pct_change().abs().dropna()
    return moves.corr(moves.shift(1))


def test_volatility_clusters_have_wild_and_calm_spells():
    """Big moves follow big moves in the clustered prices..."""

    assert _move_size_persistence(make_volatility_clusters(n_days=5500)) > 0.1


def test_random_walk_has_no_volatility_clusters():
    """...but not in the random walk, where every day is equally wild."""

    assert abs(_move_size_persistence(make_random_walk(n_days=5500))) < 0.05


def test_volatility_clusters_keep_direction_random():
    """Only move size is planted: up days stay about as common as down days."""

    up = make_volatility_clusters(n_days=5500)["Close"].pct_change().dropna() > 0
    assert up.mean() == pytest.approx(0.5, abs=0.03)
    assert (up != up.shift(1)).iloc[1:].mean() == pytest.approx(0.5, abs=0.03)
