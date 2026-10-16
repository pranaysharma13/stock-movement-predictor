"""
Getting price data.

load_prices() downloads daily prices from Yahoo Finance (through the yfinance
library) the first time it runs, and saves a CSV copy in the data/ folder.
Later runs load that CSV instead, so your results stay the same until you
choose to refresh.

drop_unfinished_day() removes today's row when the data was downloaded before
6 pm New York time, because that day's prices may not be final yet.

repair_prices() fixes rows where High and Low don't contain Open and Close,
which occasionally happens in Yahoo's data.

load_cash_returns() gives the daily interest earned on cash (13-week US
Treasury bills), used by the backtests for money that is out of the market.

Fake prices, used as checks on the whole pipeline:
- make_random_walk(): no pattern at all, and every day equally wild. A correct
  pipeline should find nothing in it, for direction or for move size. Clearly
  beating the baselines means future information is leaking in somewhere.
- make_planted_pattern(): a direction pattern built in on purpose. A working
  pipeline should find it and clearly beat "always up".
- make_volatility_clusters(): wild and calm spells built in on purpose. A
  working pipeline should predict big vs. small moves on it.
"""


from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import re

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

NEW_YORK = ZoneInfo("America/New_York")
# The US market closes at 4:00 pm New York time. Yahoo can still be adjusting
# the day's prices half an hour later, so a day only counts as final from 6 pm.
MARKET_SETTLED = time(18, 0)

# The 13-week US Treasury bill yield, in percent a year: the standard
# "risk-free" rate for money held in cash.
CASH_TICKER = "^IRX"
TRADING_DAYS = 252


def safe_name(ticker):
    """A ticker as a file name: letters, digits, dot, dash and underscore only ("^GSPC" -> "_GSPC")."""

    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def load_prices(ticker="SPY", start="2005-01-01", refresh=False):
    """Daily prices for one ticker, adjusted for splits and dividends.

    ticker:  any Yahoo Finance symbol, e.g. "SPY", "AAPL", "^GSPC"
    start:   first date to download, "YYYY-MM-DD"
    refresh: True re-downloads (for example to pick up the latest days)

    Returns a DataFrame indexed by date with Open, High, Low, Close, Volume.
    prices.attrs["repaired days"] lists the days repair_prices() had to fix.
    """

    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{safe_name(ticker)}_from_{start}.csv"

    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col="Date", parse_dates=True)
        # The file's save time tells us whether its last day was still in progress.
        saved_at = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
        return repair_prices(drop_unfinished_day(prices, saved_at))

    # Imported here so the rest of the project still works without yfinance.
    import yfinance as yf

    # auto_adjust=True folds dividends and splits into the prices, so a
    # day-to-day change in Close is the real return an investor got.
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data came back for {ticker!r}. "
            "Check the ticker symbol and your internet connection."
        )

    # Recent yfinance versions use two-level column names like ("Close", "SPY").
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw[PRICE_COLUMNS].dropna()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None).normalize()  # keep the local trading date
    prices.index.name = "Date"

    prices = drop_unfinished_day(prices, datetime.now(timezone.utc))
    prices.to_csv(cache)
    return repair_prices(prices)


def drop_unfinished_day(prices, downloaded_at):
    """Remove the last row if that trading day was still in progress.

    prices:        daily prices indexed by date
    downloaded_at: when the data was fetched (a timezone-aware datetime)

    If the last row is dated today (in New York) and the data was fetched
    before 6 pm New York time, its prices may still change, so the row is
    dropped.
    """

    if prices.empty:
        return prices
    fetched = downloaded_at.astimezone(NEW_YORK)
    last_day = prices.index[-1].date()
    if last_day >= fetched.date() and fetched.time() < MARKET_SETTLED:
        return prices.iloc[:-1]
    return prices


def repair_prices(prices):
    """Make every row's High and Low contain its Open and Close.

    Yahoo occasionally has rows where, say, Open is above High. High is set
    to the largest of the four prices and Low to the smallest; Close, which
    every return is built from, is never changed. The repaired days are
    listed in prices.attrs["repaired days"].
    """

    prices = prices.copy()
    ohlc = prices[["Open", "High", "Low", "Close"]]
    bad = (ohlc.max(axis=1) > prices["High"]) | (ohlc.min(axis=1) < prices["Low"])
    prices["High"] = ohlc.max(axis=1)
    prices["Low"] = ohlc.min(axis=1)
    prices.attrs["repaired days"] = [str(d.date()) for d in prices.index[bad]]
    return prices


def load_cash_returns(index, start="2005-01-01", refresh=False):
    """Interest earned by money held in cash, per trading day, for the days in index.

    Uses the 13-week US Treasury bill yield (^IRX on Yahoo Finance), quoted in
    percent a year, spread evenly over 252 trading days. The rate known at
    today's close is the one earned until tomorrow's close. Days before the
    first quote use the first quote.
    """

    rates = load_prices(CASH_TICKER, start, refresh)["Close"] / 100 / TRADING_DAYS
    return cash_returns_on(rates, index)


def cash_returns_on(daily_rates, index):
    """daily_rates lined up with index: the latest rate known on each day."""
    combined = daily_rates.reindex(daily_rates.index.union(index)).ffill().bfill()
    return combined.reindex(index).rename("cash_ret")


# ---------------------------- fake prices ------------------------------------

def _prices_from_returns(returns, start, daily_vol, rng):
    """Build Open/High/Low/Close/Volume rows around a series of daily returns.

    daily_vol sets how far High and Low stretch beyond Open and Close: one
    number for every day, or one per day so wild days also swing more within
    the day, as they do in real markets.
    """

    n_days = len(returns)
    dates = pd.bdate_range(start=start, periods=n_days, name="Date")
    close = 100 * np.cumprod(1 + returns)
    open_ = close / (1 + returns)  # yesterday's close
    spread = np.abs(rng.normal(0, daily_vol / 2, n_days))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(mean=18, sigma=0.3, size=n_days).round()
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_random_walk(n_days=5500, start="2005-01-03", daily_drift=0.0003,
                     daily_vol=0.012, seed=0):
    """Fake daily prices where every day's return is pure chance.

    The small positive drift mimics the stock market's long-run rise, so up
    days are slightly more common than down days, just like in real data.
    Volume is random too, and unrelated to the returns.
    """

    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_drift, daily_vol, n_days)
    return _prices_from_returns(returns, start, daily_vol, rng)


def make_planted_pattern(n_days=5500, start="2005-01-03", reversal_prob=0.6,
                         daily_vol=0.012, seed=0):
    """Fake daily prices with a known pattern built in: moves tend to reverse.

    After an up day, the next day is a down day with probability
    reversal_prob (60% by default); after a down day, the next day is an up
    day with the same probability. Up and down days are equally common
    overall, so "always up" scores about 50%, while a model that learns
    "bet against yesterday's move" should score about 60%.
    """

    rng = np.random.default_rng(seed)
    up = np.empty(n_days, dtype=bool)
    up[0] = rng.random() < 0.5
    for t in range(1, n_days):
        reverse = rng.random() < reversal_prob
        up[t] = (not up[t - 1]) if reverse else up[t - 1]
    sizes = np.abs(rng.normal(0, daily_vol, n_days))
    returns = np.where(up, sizes, -sizes)
    return _prices_from_returns(returns, start, daily_vol, rng)


def make_volatility_clusters(n_days=5500, start="2005-01-03", daily_vol=0.012,
                             reaction=0.10, persistence=0.88, seed=0):
    """Fake daily prices with wild and calm spells built in on purpose.

    Each day's volatility depends on yesterday's: a big move today makes
    tomorrow wilder (reaction), and wild periods fade only slowly
    (persistence). This is the GARCH(1,1) model, a standard description of
    real markets; 0.10 and 0.88 are typical values for a stock index.
    Direction stays pure chance, so only move size is predictable.
    daily_vol is the long-run average volatility.
    """

    rng = np.random.default_rng(seed)
    base = daily_vol ** 2 * (1 - reaction - persistence)
    variance = daily_vol ** 2
    returns = np.empty(n_days)
    vols = np.empty(n_days)
    for t in range(n_days):
        vols[t] = np.sqrt(variance)
        returns[t] = rng.normal(0, vols[t])
        variance = base + reaction * returns[t] ** 2 + persistence * variance
    return _prices_from_returns(returns, start, vols, rng)
