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

    The last training day is left out of each training set. Its label is
    tomorrow's move, and tomorrow is the first day of the test year, so
    keeping it would let one price from the test year into training.
    first_year=None picks it automatically with first_test_year().
    features: which feature columns the model sees (default: FEATURES).

    Returns a DataFrame indexed by date with:
      prob: the model's probability that target is 1 (an up day, or for the
            move-size task, a big day)
      pred: 1 if prob > 0.5, else 0
    """

    if first_year is None:
        first_year = first_test_year(data)
    features = list(features)
    years = sorted(y for y in data.index.year.unique() if y >= first_year)
    if not years:
        raise ValueError(f"No data from {first_year} onward to test on.")

    chunks = []
    for year in years:
        train = data[data.index.year < year].iloc[:-1]   # drop the day whose label is in the test year
        test = data[data.index.year == year]
        fitted = clone(model).fit(train[features], train["target"])
        prob = fitted.predict_proba(test[features])[:, 1]
        chunks.append(pd.DataFrame({"prob": prob}, index=test.index))

    preds = pd.concat(chunks)
    preds["pred"] = (preds["prob"] > 0.5).astype(int)
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
            # Volatility clustering in one line: if today was big, tomorrow
            # will be too.
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

    predictions: dict of {name: Series of 0/1}, one prediction for every day
                 of target (a missing prediction raises an error)
    target:      Series of true 0/1 labels for the same days
    baseline:    name of the prediction every method must beat. None means
                 "always predict 1" ("always up", for direction).

    Columns:
      accuracy                 share of days predicted correctly
      range low / range high   95% range: accuracies inside it are
                               consistent with luck at this sample size
      predicted up             share of days the method predicted 1 ("up",
                               or for move size, "big")
      p-value vs coin flip     chance of scoring this well by flipping coins
      p-value vs <baseline>    from paired_p_value(): the baseline is itself
                               a prediction for the same days, so only the
                               days where the two disagree count
      beats <baseline>?        True if that p-value is at most 0.05
    """

    if baseline is None:
        base_pred, base_name = pd.Series(1, index=target.index), "always up"
    else:
        base_pred, base_name = pd.Series(predictions[baseline]).reindex(target.index), baseline.lower()
    rows = {}
    for name, pred in predictions.items():
        pred = pd.Series(pred).reindex(target.index)
        if pred.isna().any():
            raise ValueError(f"{name}: no prediction for {int(pred.isna().sum())} of the "
                             f"{len(pred)} days; every day needs one.")
        correct = pred == target
        n, k = len(correct), int(correct.sum())
        acc = k / n
        half_width = 1.96 * np.sqrt(acc * (1 - acc) / n)
        p = paired_p_value(pred, base_pred, target)
        rows[name] = {
            "accuracy": acc,
            "range low": acc - half_width,
            "range high": acc + half_width,
            "predicted up": pred.mean(),
            "p-value vs coin flip": binomtest(k, n, 0.5, alternative="greater").pvalue,
            f"p-value vs {base_name}": p,
            f"beats {base_name}?": bool(p <= 0.05),
        }
    return pd.DataFrame(rows).T


def paired_counts(pred, baseline_pred, target):
    """On the days pred and baseline_pred disagree: how often each one is right.

    Returns (wins, losses): days only pred is right, days only the baseline is.
    """

    pred_right = pd.Series(pred).reindex(target.index) == target
    base_right = pd.Series(baseline_pred).reindex(target.index) == target
    return int((pred_right & ~base_right).sum()), int((~pred_right & base_right).sum())


def edge_half_width(wins, losses, n):
    """Half-width of the 95% range of an accuracy difference, from paired counts.

    The difference (wins - losses) / n is an average of +1 (pred right, the
    baseline wrong), -1 (the reverse) and 0 (both right or both wrong) over
    n days. Days where both agree add no uncertainty, so this range is much
    narrower than either method's own accuracy range when they mostly agree.
    """

    if n == 0:
        return float("nan")
    mean = (wins - losses) / n
    variance = (wins + losses) / n - mean ** 2
    return float(1.96 * np.sqrt(max(variance, 0) / n))


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

    wins, losses = paired_counts(pred, baseline_pred, target)
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
    """True on the last trading day of each calendar week (usually Friday).

    The very last day in the data only counts if it is a Friday: on a
    Wednesday you can't yet know whether it will be that week's last day.
    """

    days = index.to_series()
    week = index.to_period("W")
    ends = (days.groupby(week).transform("max") == days).to_numpy().copy()
    if len(ends) and index[-1].weekday() != 4:
        ends[-1] = False
    return ends


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

def _cash(cash_ret, index):
    """Interest on cash for each day of index; None means cash earns nothing."""

    if cash_ret is None:
        return pd.Series(0.0, index=index)
    return pd.Series(cash_ret).reindex(index).fillna(0.0)


def backtest(position, next_ret, cost=0.001, cash_ret=None):
    """Daily returns of a simple long-or-cash strategy.

    position: share of money in the stock for tomorrow: 1 = fully invested,
              0 = all cash, 0.5 = half and half. Decided at today's close;
              earns that share of tomorrow's return (next_ret).
    cost:     charged on the amount bought or sold (0.001 = 0.1% of the
              trade), so moving from 1 to 0.5 costs half a full trade.
    cash_ret: interest earned until tomorrow by money not in the stock, per
              day (from load_cash_returns()); None means cash earns nothing.

    Simplification: assumes you can trade at exactly the closing price.
    """

    position = pd.Series(position).reindex(next_ret.index).fillna(0)
    trades = position.diff().abs().fillna(position.iloc[0])
    cash = _cash(cash_ret, next_ret.index)
    return position * next_ret + (1 - position) * cash - cost * trades


def performance(daily_returns, cash_ret=None):
    """Summary statistics for a series of daily strategy returns.

    The Sharpe ratio uses returns above what cash would have earned
    (cash_ret, None = nothing): reward for the risk taken, not for the
    interest any saver gets.
    """

    equity = (1 + daily_returns).cumprod()
    years = len(daily_returns) / TRADING_DAYS
    std = daily_returns.std()
    excess = daily_returns - _cash(cash_ret, daily_returns.index)
    excess_std = excess.std()
    return {
        "total return": equity.iloc[-1] - 1,
        "annual return": equity.iloc[-1] ** (1 / years) - 1,
        "annual volatility": std * np.sqrt(TRADING_DAYS),
        "Sharpe ratio": excess.mean() / excess_std * np.sqrt(TRADING_DAYS) if excess_std > 0 else np.nan,
        "max drawdown": (equity / equity.cummax() - 1).min(),
    }


def backtest_table(positions, next_ret, cost=0.001, cash_ret=None):
    """Backtest several strategies at once.

    positions: dict of {name: Series of positions} (see backtest())

    Returns (stats table, DataFrame of daily returns per strategy). "days in
    market" is the average position (0.75 = three quarters invested on
    average); "trades" counts every change of position.
    """

    stats, returns = {}, {}
    for name, pos in positions.items():
        pos = pd.Series(pos).reindex(next_ret.index).fillna(0)
        r = backtest(pos, next_ret, cost, cash_ret)
        row = performance(r, cash_ret)
        row["days in market"] = pos.mean()
        row["trades"] = int((pos.diff().fillna(pos.iloc[0]) != 0).sum())
        stats[name] = row
        returns[name] = r
    table = pd.DataFrame(stats).T
    table["trades"] = table["trades"].astype(int)
    return table, pd.DataFrame(returns)


def luck_check(position, next_ret, cost=0.001, n_trials=2000, seed=0,
               metric="annual return", by_week=False, cash_ret=None):
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
      share of random >=:  how often a random version did at least as well,
                           counting the strategy itself as one of the
                           versions: (count + 1) / (n_trials + 1). So it is
                           never exactly 0: 2,000 trials can't prove that
                           something is impossible by luck. Around 0.05 or
                           less = hard to explain by luck.
      cash days:           how many days the strategy held less than a full position
    """

    position = pd.Series(position).reindex(next_ret.index).fillna(0)
    reduced = position < 1
    n_cash = int(reduced.sum())
    reduced_values = position[reduced].to_numpy()
    actual = performance(backtest(position, next_ret, cost, cash_ret), cash_ret)[metric]
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
        random_values.append(performance(backtest(random_pos, next_ret, cost=0, cash_ret=cash_ret),
                                         cash_ret)[metric])
    random_values = np.array(random_values)
    return {
        metric: actual,
        "random median": float(np.median(random_values)),
        "share of random >=": float(((random_values >= actual).sum() + 1) / (n_trials + 1)),
        "cash days": n_cash,
    }
