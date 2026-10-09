"""
Getting price data.

load_prices() downloads daily prices from Yahoo Finance (through the yfinance
library) the first time it runs, and saves a CSV copy in the data/ folder.
Later runs load that CSV instead, so your results stay the same until you
choose to refresh.

make_random_walk() creates fake prices with no pattern in them at all. It is a
sanity check: a correct pipeline should score about the same as "always up"
on it. If a model clearly "beats" a random walk, the code is leaking future
information somewhere.
"""



from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]



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
        return pd.read_csv(cache, index_col="Date", parse_dates=True)

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

    prices.to_csv(cache)
    return prices



def make_random_walk(n_days=5500, start="2005-01-03", daily_drift=0.0003,
                     daily_vol=0.012, seed=0):
    """Fake daily prices where every day's return is pure chance.

    The small positive drift mimics the stock market's long-run rise, so up
    days are slightly more common than down days, just like in real data.
    Volume is random too, and unrelated to the returns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="Date")

    returns = rng.normal(daily_drift, daily_vol, n_days)
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