"""
Testing models honestly.

get_models()           the two models we compare
first_test_year()      the first year with enough history before it to train on
walk_forward()         train only on the past, predict the next year, step forward
baseline_predictions() simple rules a direction model has to beat
vol_baseline_predictions()  simple rules a move-size model has to beat
accuracy_table()       accuracy with a 95% range, so you can tell an edge from noise
paired_p_value()       is one set of predictions really better than another on the same days?
accuracy_by_year()     how stable is the accuracy from year to year?
backtest_table()       turn predictions into a trading rule and compare with buy-and-hold
hold_for_week()        decide once a week instead of every day
luck_check()           could the backtest result come from picking random days (or weeks)?

Everything here works for both tasks: "target" is whatever the dataset
predicts (an up day, or a big day).
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


def first_test_year(data, min_train_days=1000):
    """The first calendar year with at least min_train_days of data before it.

    1,000 trading days is about four years. For SPY data starting in 2005
    this gives 2010; for a stock listed in 2015 it gives 2019 or 2020.
    """

    for year in sorted(data.index.year.unique()):
        if (data.index.year < year).sum() >= min_train_days:
            return int(year)
    raise ValueError(
        f"Not enough history: need at least {min_train_days} trading days "
        f"before a test year, but the data has {len(data)} days in total."
    )


def walk_forward(model, data, first_year=None, features=FEATURES):
    """Predict each year with a model trained only on the years before it.

    For first_year=2010: train on 2005-2009, predict 2010; then train on
    2005-2010, predict 2011; and so on up to the latest data.
    first_year=None picks it automatically with first_test_year().
    features: which feature columns the model sees (default: FEATURES).

    Returns a DataFrame indexed by date with:
      prob_up: the model's probability that target is 1 (an up day, or for
               the move-size task, a big day)
      pred:    1 if prob_up > 0.5, else 0
    """

    if first_year is None:
        first_year = first_test_year(data)
    features = list(features)
    years = sorted(y for y in data.index.year.unique() if y >= first_year)
    if not years:
        raise ValueError(f"No data from {first_year} onward to test on.")

    chunks = []
    for year in years:
        train = data[data.index.year < year]
        test = data[data.index.year == year]
        fitted = clone(model).fit(train[features], train["target"])
        prob_up = fitted.predict_proba(test[features])[:, 1]
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


def vol_baseline_predictions(data):
    """Simple rules for the move-size task, as 0/1 predictions (1 = big day).

    data needs abs_ret_rel (today's move relative to the typical move) and
    wide_range (from make_vol_features).
    """

    return pd.DataFrame(
        {
            # The majority rule's two versions; whichever is right more often
            # is the bar a model must clear just to be better than guessing.
            "Always small": 0,
            "Always big": 1,
            # Volatility clustering in one line: if today was big, tomorrow will be too.
            "Same as today": (data["abs_ret_rel"] > 1).astype(int),
            # A sharper one-line rule: if today's high-to-low swing was wider
            # than usual for the past year, predict a big day. No model, no
            # fitting. This is the real bar for the move-size task.
            "Recent swing": data["wide_range"].astype(int),
        },
        index=data.index,
    )



# ---------------------------- accuracy ------------------------------------

def accuracy_table(predictions, target, baseline=None):
    """Compare several sets of 0/1 predictions against the true labels.

    predictions: dict of {name: Series of 0/1}
    target:      Series of true 0/1 labels for the same days
    baseline:    name of the prediction every method must beat. None means
                 "always predict 1" (always up, for direction).

    Columns:
      accuracy                 share of days predicted correctly
      range low / range high   95% range: accuracies inside it are
                               consistent with luck at this sample size
      predicted up             share of days the method predicted 1 ("up",
                               or for move size, "big")
      p-value vs coin flip     chance of scoring this well by flipping coins
      beats always-up?         True only if even the low end of the range
                               is above the "always up" accuracy

    With a named baseline, two columns replace "beats always-up?":
      p-value vs <baseline>    from paired_p_value(): the baseline is itself
                               a prediction on the same days, so only the
                               days where the two disagree count
      beats <baseline>?        True if that p-value is below 0.05
    """

    if baseline is None:
        bar, bar_column = target.mean(), "beats always-up?"
    else:
        base_pred = pd.Series(predictions[baseline]).reindex(target.index)
        bar_column = f"beats {baseline.lower()}?"
    rows = {}
    for name, pred in predictions.items():
        pred = pd.Series(pred).reindex(target.index)
        correct = pred == target
        n, k = len(correct), int(correct.sum())
        acc = k / n
        half_width = 1.96 * np.sqrt(acc * (1 - acc) / n)
        row = {
            "accuracy": acc,
            "range low": acc - half_width,
            "range high": acc + half_width,
            "predicted up": pred.mean(),
            "p-value vs coin flip": binomtest(k, n, 0.5, alternative="greater").pvalue,
        }
        if baseline is None:
            row[bar_column] = bool(acc - half_width > bar)
        else:
            p = paired_p_value(pred, base_pred, target)
            row[f"p-value vs {baseline.lower()}"] = p
            row[bar_column] = bool(p < 0.05)
        rows[name] = row
    return pd.DataFrame(rows).T


def paired_p_value(pred, baseline_pred, target):
    """Is pred really more accurate than baseline_pred on the same days?

    Days where both are right, or both wrong, say nothing about which is
    better. Only the days where they disagree count: if the two were equally
    good, each would win about half of those. Returns the chance of pred
    winning at least as many of them as it did if they were equally good
    (a one-sided sign test, also called McNemar's exact test). 1.0 if they
    never disagree.

    This is stricter and fairer than comparing pred's accuracy with a fixed
    number, because a strong baseline agrees with the model on most days.
    """

    pred_right = pd.Series(pred).reindex(target.index) == target
    base_right = pd.Series(baseline_pred).reindex(target.index) == target
    wins = int((pred_right & ~base_right).sum())
    losses = int((~pred_right & base_right).sum())
    if wins + losses == 0:
        return 1.0
    return float(binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue)


def accuracy_by_year(predictions, target):
    """Accuracy of each method in each calendar year."""
    
    hits = pd.DataFrame(
        {name: pd.Series(p).reindex(target.index) == target
         for name, p in predictions.items()}
    )
    return hits.groupby(hits.index.year).mean()


# ---------------------------- weekly decisions ------------------------------------

def _week_ends(index):
    """True on the last trading day of each calendar week (usually Friday)."""

    days = index.to_series()
    week = index.to_period("W")
    return (days.groupby(week).transform("max") == days).to_numpy()


def hold_for_week(signal):
    """Decide once a week instead of every day.

    The value on the last trading day of each week (usually Friday's close) is
    kept until the next week's last day. Days before the first week end keep
    the first day's value. Trading weekly means far fewer trades, so lower
    costs, at the price of reacting more slowly.
    """

    signal = pd.Series(signal)
    decided = signal.where(_week_ends(signal.index))
    decided.iloc[0] = signal.iloc[0]
    return decided.ffill().astype(signal.dtype)


def week_blocks(index):
    """A label per day for the week a weekly decision is held: each block runs
    from one week's last trading day to the day before the next one."""

    return pd.Series(np.cumsum(_week_ends(index)), index=index)


# ---------------------------- backtest ------------------------------------

def backtest(position, next_ret, cost=0.001):
    """Daily returns of a simple long-or-cash strategy.

    position: share of money in the stock for tomorrow: 1 = fully invested,
              0 = all cash, 0.5 = half and half. Decided at today's close;
              earns that share of tomorrow's return (next_ret).
    cost:     charged on the amount bought or sold (0.001 = 0.1% of the
              trade), so moving from 1 to 0.5 costs half a full trade.

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

    positions: dict of {name: Series of positions} (see backtest())

    Returns (stats table, DataFrame of daily returns per strategy). "days in
    market" is the average position (0.75 = three quarters invested on
    average); "trades" counts every change of position.
    """

    stats, returns = {}, {}
    for name, pos in positions.items():
        pos = pd.Series(pos).reindex(next_ret.index).fillna(0)
        r = backtest(pos, next_ret, cost)
        row = performance(r)
        row["days in market"] = pos.mean()
        row["trades"] = int((pos.diff().fillna(pos.iloc[0]) != 0).sum())
        stats[name] = row
        returns[name] = r
    table = pd.DataFrame(stats).T
    table["trades"] = table["trades"].astype(int)
    return table, pd.DataFrame(returns)


def luck_check(position, next_ret, cost=0.001, n_trials=2000, seed=0,
               metric="annual return", by_week=False):
    """Could a strategy's result come from picking random days?

    Keeps the same number of days below a full position as the strategy (cash
    days, or half-position days), but picks those days at random, many times
    over. The strategy pays its trading cost; the random versions trade for
    free. Random days are scattered, so charging them costs would make them
    trade far more than the strategy and look worse than they are. Letting
    them trade free is generous to luck, which makes this a strict test.

    metric: which number from performance() to compare, e.g. "annual return"
            (direction) or "Sharpe ratio" (move size, where the aim is less
            risk per unit of return rather than more return).
    by_week: for a strategy that decides once a week (hold_for_week()).
            The random versions then reshuffle whole weeks rather than single
            days, so they are reduced for the same number of weeks, like for
            like. The position must be constant within each week block.

    Returns:
      <metric>:            the strategy's own value
      random median:       the typical value of the random versions
      share of random >=:  how often a random version did at least as well
                           (around 0.05 or less = hard to explain by luck)
      cash days:           how many days the strategy held less than a full position
    """

    position = pd.Series(position).reindex(next_ret.index).fillna(0)
    reduced = position < 1
    n_cash = int(reduced.sum())
    reduced_values = position[reduced].to_numpy()
    actual = performance(backtest(position, next_ret, cost))[metric]
    rng = np.random.default_rng(seed)
    if by_week:
        blocks = week_blocks(next_ret.index)
        if (position.groupby(blocks).nunique() > 1).any():
            raise ValueError("by_week=True needs a position that only changes at week ends "
                             "(use hold_for_week()).")
        block_values = position.groupby(blocks).first()
        day_block = blocks.to_numpy()
    random_values = []
    for _ in range(n_trials):
        if by_week:
            shuffled = rng.permutation(block_values.to_numpy())
            random_pos = pd.Series(shuffled[np.searchsorted(block_values.index, day_block)],
                                   index=next_ret.index)
        else:
            random_pos = pd.Series(1.0, index=next_ret.index)
            random_pos.iloc[rng.choice(len(next_ret), n_cash, replace=False)] = reduced_values
        random_values.append(performance(backtest(random_pos, next_ret, cost=0))[metric])
    random_values = np.array(random_values)
    return {
        metric: actual,
        "random median": float(np.median(random_values)),
        "share of random >=": float((random_values >= actual).mean()),
        "cash days": n_cash,
    }
