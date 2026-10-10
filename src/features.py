"""
Turning prices into model inputs.

The rule that matters most: every feature for day t may only use information
available at the close of day t. The label is the one thing allowed to look
forward: did the price rise from day t's close to day t+1's close?
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