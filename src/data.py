"""
Getting price data.

load_prices() downloads daily prices from Yahoo Finance (through the yfinance
library) the first time it runs, and saves a CSV copy in the data/ folder.
Later runs load that CSV instead, so your results stay the same until you
choose to refresh.

drop_unfinished_day() removes today's row when the data was downloaded while
the US market was still open, because that "close" is not final yet.

Two kinds of fake prices, used as checks on the whole pipeline:
-   make_random_walk(): no pattern at all. A correct pipeline should score about
  the same as "always up" on it. Clearly beating it means future information
  is leaking in somewhere.
-   make_planted_pattern(): a known pattern built in on purpose. A working
    pipeline should find it and clearly beat "always up". This proves the models
    can detect a real pattern when one exists.
"""


from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
 
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

NEW_YORK = ZoneInfo("America/New_York")
# The US market closes at 4:00 pm New York time; allow 30 minutes for final prices.
MARKET_SETTLED = time(16, 30)


def load_prices(ticker="SPY", start="2005-01-01", refresh=False):
    """Daily prices for one ticker, adjusted for splits and dividends.

    ticker:  any Yahoo Finance symbol, e.g. "SPY", "AAPL", "^GSPC"
    start:   first date to download, "YYYY-MM-DD"
    refresh: True re-downloads (for example to pick up the latest days)

    Returns a DataFrame indexed by date with Open, High, Low, Close, Volume.
    """
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{ticker}_from_{start}.csv"

    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col="Date", parse_dates=True)
        # The file's save time tells us whether its last day was still in progress.
        saved_at = datetime.fromtimestamp(cache.stat().st_mtime, tz=timezone.utc)
        return drop_unfinished_day(prices, saved_at)

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
    return prices


def drop_unfinished_day(prices, downloaded_at):
    """Remove the last row if that trading day was still in progress.
 
    prices:        daily prices indexed by date
    downloaded_at: when the data was fetched (a timezone-aware datetime)
 
    If the last row is dated today (in New York) and the data was fetched
    before 4:30 pm New York time, its "close" is just the price at that
    moment, so the row is dropped.
    """

    if prices.empty:
        return prices
    fetched = downloaded_at.astimezone(NEW_YORK)
    last_day = prices.index[-1].date()
    if last_day >= fetched.date() and fetched.time() < MARKET_SETTLED:
        return prices.iloc[:-1]
    return prices


# ---------------------------- fake prices ------------------------------------

def _prices_from_returns(returns, start, daily_vol, rng):
    """Build Open/High/Low/Close/Volume rows around a series of daily returns."""

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