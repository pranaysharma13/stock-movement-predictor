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
from src.data import (NEW_YORK, PRICE_COLUMNS, TRADING_DAYS, cash_returns_on, drop_unfinished_day,
                      load_cash_returns, load_prices, make_planted_pattern, make_random_walk,
                      make_volatility_clusters, repair_prices)


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


@pytest.mark.parametrize("when", ["2026-10-09 18:30", "2026-10-10 09:00"])
def test_keeps_today_once_the_market_has_closed(when):
    prices = _prices_ending("2026-10-09")
    assert drop_unfinished_day(prices, _new_york(when)).index[-1] == pd.Timestamp("2026-10-09")


def test_drops_today_shortly_after_the_close():
    """At 4:36 pm Yahoo can still be adjusting the day (a review found Open above High then)."""

    prices = _prices_ending("2026-10-09")
    assert drop_unfinished_day(prices, _new_york("2026-10-09 16:36")).index[-1] == pd.Timestamp("2026-10-08")


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


# ---------------------------- phase 4.2: repairs, file names, cash ------------------------------------

def test_repair_prices_makes_high_and_low_contain_open_and_close():
    days = pd.bdate_range("2026-10-05", periods=3, name="Date")
    prices = pd.DataFrame({"Open": [10.0, 773.24, 10.0], "High": [11.0, 773.23, 11.0],
                           "Low": [9.0, 770.0, 9.5], "Close": [10.5, 772.0, 9.4],
                           "Volume": [1.0, 1.0, 1.0]}, index=days)
    fixed = repair_prices(prices)
    assert fixed.loc[days[1], "High"] == 773.24          # Open was above High
    assert fixed.loc[days[2], "Low"] == 9.4              # Close was below Low
    assert fixed["Close"].equals(prices["Close"])        # returns are never touched
    assert fixed.attrs["repaired days"] == ["2026-10-06", "2026-10-07"]
    assert repair_prices(fixed).attrs["repaired days"] == []


def _fake_yfinance(monkeypatch, frame):
    raw = frame.copy()
    fake_yf = types.ModuleType("yfinance")
    fake_yf.download = lambda *args, **kwargs: raw
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)


def test_tickers_with_symbols_get_safe_file_names(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    _fake_yfinance(monkeypatch, make_random_walk(n_days=20, start="2010-01-04"))
    load_prices("^GSPC", "2010-01-01", refresh=True)
    assert (tmp_path / "_GSPC_from_2010-01-01.csv").exists()
    assert len(load_prices("^GSPC", "2010-01-01")) == 20    # and the saved copy is found again


def test_cash_returns_carry_the_latest_rate_forward():
    rates = pd.Series([0.0002, 0.0001], index=pd.to_datetime(["2020-01-03", "2020-01-08"]))
    days = pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    cash = cash_returns_on(rates, days)
    # before the first quote: the first quote; then the latest known rate
    assert list(cash) == [0.0002, 0.0002, 0.0002, 0.0001, 0.0001]


def test_load_cash_returns_turns_a_yearly_percent_into_a_daily_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    days = pd.bdate_range("2010-01-04", periods=10, name="Date")
    quotes = pd.DataFrame({c: 5.04 for c in ["Open", "High", "Low", "Close"]}, index=days)
    quotes["Volume"] = 0.0
    _fake_yfinance(monkeypatch, quotes)
    cash = load_cash_returns(days, "2010-01-01", refresh=True)
    assert cash.iloc[0] == pytest.approx(0.0504 / TRADING_DAYS)
    assert (tmp_path / "_IRX_from_2010-01-01.csv").exists()


def test_saved_file_is_repaired_when_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    prices = _prices_ending("2013-01-25")
    prices.loc[pd.Timestamp("2013-01-22"), "Open"] = 99.0       # above that day's High
    prices.to_csv(tmp_path / "SPY_from_2005-01-01.csv")
    loaded = load_prices("SPY", "2005-01-01")
    assert loaded.attrs["repaired days"] == ["2013-01-22"]
    assert loaded.loc["2013-01-22", "High"] == 99.0
