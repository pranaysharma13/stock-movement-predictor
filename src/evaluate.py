"""
Testing models honestly.

get_models()           the two models we compare
walk_forward()         train only on the past, predict the next year, step forward
baseline_predictions() simple rules any model has to beat
accuracy_table()       accuracy with a 95% range, so you can tell an edge from noise
accuracy_by_year()     how stable is the accuracy from year to year?
backtest_table()       turn predictions into a trading rule and compare with buy-and-hold
"""



import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

TRADING_DAYS = 252


# ---------------------------- models ------------------------------------


def get_models():
    """Fresh, untrained models, keyed by display name."""
    
    return {
        # Scaling puts every feature on the same footing before the
        # weighted sum, so the learned weights are comparable.
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        # Shallow trees and a small learning rate keep it from memorising noise.
        "Gradient boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            early_stopping=False, random_state=0,
        ),
    }



def walk_forward(model, data, first_test_year=2010):
    """Predict each year with a model trained only on the years before it.

    For first_test_year=2010: train on 2005-2009, predict 2010; then train on
    2005-2010, predict 2011; and so on up to the latest data.

    Returns a DataFrame indexed by date with:
      prob_up: the model's probability that tomorrow is an up day
      pred:    1 if prob_up > 0.5, else 0
    """

    years = sorted(y for y in data.index.year.unique() if y >= first_test_year)
    if not years:
        raise ValueError(f"No data from {first_test_year} onward to test on.")

    chunks = []
    for year in years:
        train = data[data.index.year < year]
        test = data[data.index.year == year]
        fitted = clone(model).fit(train[FEATURES], train["target"])
        prob_up = fitted.predict_proba(test[FEATURES])[:, 1]
        chunks.append(pd.DataFrame({"prob_up": prob_up}, index=test.index))

    preds = pd.concat(chunks)
    preds["pred"] = (preds["prob_up"] > 0.5).astype(int)
    return preds


# ---------------------------- baselines ------------------------------------


def baseline_predictions(data, seed=0):
    """Three simple rules, as 0/1 predictions for every row of data."""

    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            # Pure chance: about 50% accuracy.
            "Coin flip": rng.integers(0, 2, len(data)),
            # Markets rise slightly more often than they fall, so this
            # already beats a coin flip. It's the bar that matters.
            "Always up": 1,
            # Tomorrow moves the same way as today did.
            "Repeat last move": (data["ret_1d"] > 0).astype(int),
        },
        index=data.index,
    )


# ---------------------------- accuracy ------------------------------------


def accuracy_table(predictions, target):
    """Compare several sets of 0/1 predictions against the true labels.

    predictions: dict of {name: Series of 0/1}
    target:      Series of true 0/1 labels for the same days

    Columns:
      accuracy                 share of days predicted correctly
      range low / range high   95% range: accuracies inside it are
                               consistent with luck at this sample size
      predicted up             share of days the method said "up"
      p-value vs coin flip     chance of scoring this well by flipping coins
      beats always-up?         True only if even the low end of the range
                               is above the "always up" accuracy
    """

    always_up = target.mean()
    rows = {}
    for name, pred in predictions.items():
        pred = pd.Series(pred).reindex(target.index)
        correct = pred == target
        n, k = len(correct), int(correct.sum())
        acc = k / n
        half_width = 1.96 * np.sqrt(acc * (1 - acc) / n)
        rows[name] = {
            "accuracy": acc,
            "range low": acc - half_width,
            "range high": acc + half_width,
            "predicted up": pred.mean(),
            "p-value vs coin flip": binomtest(k, n, 0.5, alternative="greater").pvalue,
            "beats always-up?": bool(acc - half_width > always_up),
        }
    return pd.DataFrame(rows).T



def accuracy_by_year(predictions, target):
    """Accuracy of each method in each calendar year."""

    hits = pd.DataFrame(
        {name: pd.Series(p).reindex(target.index) == target
         for name, p in predictions.items()}
    )
    return hits.groupby(hits.index.year).mean()


# ---------------------------- backtest ------------------------------------


def backtest(position, next_ret, cost=0.001):
    """Daily returns of a simple long-or-cash strategy.

    position: 1 = hold the stock into tomorrow, 0 = sit in cash.
              Decided at today's close; earns tomorrow's return (next_ret).
    cost:     charged on every buy or sell (0.001 = 0.1% of the trade).

    Simplification: assumes you can trade at exactly the closing price.
    """

    position = pd.Series(position).reindex(next_ret.index).fillna(0)
    trades = position.diff().abs().fillna(position.iloc[0])
    return position * next_ret - cost * trades



def performance(daily_returns):
    """Summary statistics for a series of daily strategy returns."""

    equity = (1 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS
    std = daily_returns.std()
    return {
        "total return": equity.iloc[-1] - 1,
        "annual return": equity.iloc[-1] ** (1 / years) - 1,
        "annual volatility": std * np.sqrt(TRADING_DAYS),
        "Sharpe ratio": daily_returns.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "max drawdown": (equity / equity.cummax() - 1).min(),
    }



def backtest_table(positions, next_ret, cost=0.001):
    """Backtest several strategies at once.

    positions: dict of {name: Series of 0/1 positions}

    Returns (stats table, DataFrame of daily returns per strategy).
    """

    stats, returns = {}, {}
    for name, pos in positions.items():
        pos = pd.Series(pos).reindex(next_ret.index).fillna(0)
        r = backtest(pos, next_ret, cost)
        row = performance(r)
        row["days in market"] = pos.mean()
        row["trades"] = int(pos.diff().abs().fillna(pos.iloc[0]).sum())
        stats[name] = row
        returns[name] = r
    table = pd.DataFrame(stats).T
    table["trades"] = table["trades"].astype(int)
    return table, pd.DataFrame(returns)