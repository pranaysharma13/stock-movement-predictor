"""
Turning prices into model inputs, for two prediction tasks.

Part 1, direction: will tomorrow close higher than today?
    make_features(), build_dataset(), FEATURES
Part 2, move size: will tomorrow's move (up or down) be bigger than usual?
    make_vol_features(), build_vol_dataset(), VOL_FEATURES

The rule that matters most: every feature for day t may only use information
available at the close of day t. The label is the one thing allowed to look
forward, to day t+1's close.
"""


import numpy as np
import pandas as pd


# Every feature make_features() calculates.
ALL_FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "dist_ma10", "dist_ma50",
    "vol_20d",
    "volume_ratio",
]


# The ones the models use: one per idea, so no two measure nearly the same thing.
# Left out because they overlap with a kept feature (correlation on SPY 2005-2026):
#   dist_ma10 ~ ret_5d (0.91), ret_20d ~ dist_ma50 (0.86), ret_10d ~ both windows (0.67-0.82)
FEATURES = [
    "ret_1d",        # yesterday's move
    "ret_5d",        # the last week
    "dist_ma50",     # the longer-term trend
    "vol_20d",       # risk
    "volume_ratio",  # activity
]



def make_features(prices):
    """One row per day, one column per feature (see ALL_FEATURES)."""

    close = prices["Close"]
    volume = prices["Volume"]
    daily_ret = close / close.shift(1) - 1

    f = pd.DataFrame(index=prices.index)

    # Momentum: how much has the price moved over the last n days?
    f["ret_1d"] = daily_ret
    for n in (5, 10, 20):
        f[f"ret_{n}d"] = close / close.shift(n) - 1

    # Trend: how far is today's price above or below its recent average?
    # (+0.02 means 2% above the average.)
    for n in (10, 50):
        f[f"dist_ma{n}"] = close / close.rolling(n).mean() - 1

    # Risk: how jumpy have daily returns been over the last month?
    f["vol_20d"] = daily_ret.rolling(20).std()

    # Activity: was today's volume unusually high or low vs. the last month?
    f["volume_ratio"] = volume / volume.rolling(20).mean() - 1

    # Zero volume in old data can create infinities; treat them as missing.
    return f.replace([np.inf, -np.inf], np.nan)


def build_dataset(prices):
    """Features plus what the model has to predict.

    Adds two columns to the features:
      next_ret: return from today's close to tomorrow's close (for the backtest)
      target:   1 if next_ret > 0 (an up day tomorrow), otherwise 0

    Rows with missing values are dropped: the first ~50 days, while the
    moving averages warm up, and the last day, which has no tomorrow yet.
    """

    data = make_features(prices)
    close = prices["Close"]
    data["next_ret"] = close.shift(-1) / close - 1
    data = data.dropna()
    data["target"] = (data["next_ret"] > 0).astype(int)
    return data


# ---------------------------- Part 2: move size ------------------------------------

# How many past days define this stock's "typical" daily move (about a year).
TYPICAL_WINDOW = 250

# Every feature is measured relative to the stock's typical move, like the
# label, so "big" means the same thing for a calm fund and a wild stock.
VOL_FEATURES = [
    "abs_ret_rel",   # today's move size
    "vol_5d_rel",    # how jumpy the last week was
    "vol_20d_rel",   # how jumpy the last month was
    "range_rel",     # how far the price swung within today (high to low)
    "volume_ratio",  # today's volume vs. its 20-day average
]


def make_vol_features(prices):
    """Move-size features, plus two extra columns that aren't model features:

      typical_move: the median size of a daily move over the past
                    TYPICAL_WINDOW days, up to and including today
      wide_range:   1 if today's high-to-low swing is wider than its median
                    over the same days, else 0 (the "Recent swing" baseline)
    """

    close = prices["Close"]
    daily_ret = close / close.shift(1) - 1
    typical = daily_ret.abs().rolling(TYPICAL_WINDOW).median()

    f = pd.DataFrame(index=prices.index)
    f["typical_move"] = typical
    f["abs_ret_rel"] = daily_ret.abs() / typical
    f["vol_5d_rel"] = daily_ret.rolling(5).std() / typical
    f["vol_20d_rel"] = daily_ret.rolling(20).std() / typical
    day_range = (prices["High"] - prices["Low"]) / close
    f["range_rel"] = day_range / typical
    usual_range = day_range.rolling(TYPICAL_WINDOW).median()
    f["wide_range"] = (day_range > usual_range).astype(float).where(usual_range.notna())
    f["volume_ratio"] = prices["Volume"] / prices["Volume"].rolling(20).mean() - 1
    return f.replace([np.inf, -np.inf], np.nan)


def build_vol_dataset(prices):
    """Move-size features plus what the model has to predict.

    Adds:
      next_ret: return from today's close to tomorrow's close (for the backtest)
      target:   1 if tomorrow's move, up or down, is bigger than today's
                typical_move (a "big" day), otherwise 0

    The threshold uses only past days, so it's known at today's close. The
    first ~250 days are dropped while the typical move warms up.
    """

    data = make_vol_features(prices)
    close = prices["Close"]
    data["next_ret"] = close.shift(-1) / close - 1
    data = data.dropna()
    data["wide_range"] = data["wide_range"].astype(int)
    data["target"] = (data["next_ret"].abs() > data["typical_move"]).astype(int)
    return data
