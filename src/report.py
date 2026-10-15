"""
Summaries for each stock, and the comparison across stocks.

safe_name()          a ticker turned into a safe file name ("^GSPC" -> "_GSPC")
summarize()          the headline numbers of one stock's direction results
summarize_vol()      the same for the move-size task (stored under "volatility")
save_summary()       write them to reports/summaries/<ticker>.json
load_summaries()     read them back for the comparison
comparison_table()   one row per stock and model, plus the lucky-winners check

The lucky-winners problem: test enough stocks and models, and a few will
"win" by chance alone. At the usual 5% level, about 1 comparison in 20 passes
even when nothing real is going on. So the comparison also applies a stricter
bar (the Bonferroni correction): divide 5% by the number of comparisons.
The two tasks answer different questions, so each is corrected on its own.
"""


import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .evaluate import backtest, luck_check, performance

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = ROOT / "reports" / "summaries"


def safe_name(ticker):
    """A ticker as a file name: letters, digits, dot, dash and underscore only."""

    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def summarize(ticker, test, model_preds, cost=0.001, luck=None, luck_trials=2000):
    """Headline numbers for one stock.

    ticker:      the stock's symbol
    test:        the test-period rows of the dataset (needs target and next_ret)
    model_preds: {model name: DataFrame with a "pred" column}, from walk_forward()
    cost:        trading cost per buy or sell, as in the backtest
    luck:        {model name: luck_check() result}, if already calculated;
                 missing ones are calculated here with luck_trials trials

    Returns a plain dict, ready to save as JSON.
    """

    target, next_ret = test["target"], test["next_ret"]
    always_up = float(target.mean())
    hold = performance(backtest(pd.Series(1, index=test.index), next_ret, cost))
    luck = dict(luck or {})

    models = {}
    for name, p in model_preds.items():
        pred = p["pred"].reindex(test.index)
        correct = pred == target
        n, k = len(correct), int(correct.sum())
        perf = performance(backtest(pred, next_ret, cost))
        cash_days = int((pred == 0).sum())
        if cash_days == 0:
            luck_share = None          # never in cash: nothing to compare with random cash days
        else:
            if name not in luck:
                luck[name] = luck_check(pred, next_ret, cost=cost, n_trials=luck_trials)
            luck_share = float(luck[name]["share of random >="])
        models[name] = {
            "accuracy": k / n,
            "predicted up": float(pred.mean()),
            # Chance of an accuracy this high if the model were no better than "always up"
            "p-value vs always up": float(binomtest(k, n, always_up, alternative="greater").pvalue),
            "annual return": float(perf["annual return"]),
            "Sharpe ratio": float(perf["Sharpe ratio"]),
            "max drawdown": float(perf["max drawdown"]),
            "cash days": cash_days,
            "share of random >=": luck_share,
        }

    return {
        "ticker": ticker,
        "test start": str(test.index[0].date()),
        "test end": str(test.index[-1].date()),
        "test days": int(len(test)),
        "cost": cost,
        "always up accuracy": always_up,
        "buy and hold": {
            "annual return": float(hold["annual return"]),
            "Sharpe ratio": float(hold["Sharpe ratio"]),
            "max drawdown": float(hold["max drawdown"]),
        },
        "models": models,
    }


def half_on_big(pred):
    """The move-size strategy: a full position on predicted calm days, half on predicted big days."""

    return 1 - 0.5 * pd.Series(pred)


def summarize_vol(test, model_preds, baselines, cost=0.001, luck=None, luck_trials=2000):
    """Headline numbers for the move-size task of one stock.

    test:        test-period rows of the move-size dataset (target, next_ret)
    model_preds: {model name: DataFrame with a "pred" column}, from walk_forward()
    baselines:   output of vol_baseline_predictions() for the test days
    luck:        {model name: luck_check(..., metric="Sharpe ratio") result},
                 if already calculated

    Each model is judged twice: accuracy against "same as today", and the
    half-on-big strategy's Sharpe ratio against random half-position days.
    """

    target, next_ret = test["target"], test["next_ret"]
    same_as_today = float((baselines["Same as today"] == target).mean())
    luck = dict(luck or {})

    def strategy(pred):
        perf = performance(backtest(half_on_big(pred), next_ret, cost))
        return {k: float(perf[k]) for k in ("annual return", "Sharpe ratio", "max drawdown")}

    models = {}
    for name, p in model_preds.items():
        pred = p["pred"].reindex(test.index)
        correct = pred == target
        n, k = len(correct), int(correct.sum())
        if name not in luck:
            luck[name] = luck_check(half_on_big(pred), next_ret, cost=cost,
                                    n_trials=luck_trials, metric="Sharpe ratio")
        models[name] = {
            "accuracy": k / n,
            "predicted big": float(pred.mean()),
            # Chance of an accuracy this high if the model were no better than "same as today"
            "p-value vs same as today": float(binomtest(k, n, same_as_today, alternative="greater").pvalue),
            "strategy": strategy(pred),
            "share of random >=": float(luck[name]["share of random >="]),
        }

    hold = performance(backtest(pd.Series(1, index=test.index), next_ret, cost))
    return {
        "test start": str(test.index[0].date()),
        "test end": str(test.index[-1].date()),
        "test days": int(len(test)),
        "share of big days": float(target.mean()),
        "majority accuracy": float(max(target.mean(), 1 - target.mean())),
        "same as today accuracy": same_as_today,
        "buy and hold": {k: float(hold[k]) for k in ("annual return", "Sharpe ratio", "max drawdown")},
        "same as today strategy": strategy(baselines["Same as today"]),
        "models": models,
    }


def save_summary(summary, folder=SUMMARY_DIR):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe_name(summary['ticker'])}.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def load_summaries(tickers, folder=SUMMARY_DIR):
    """Summaries for the given tickers, in that order. Missing ones raise an error."""

    folder = Path(folder)
    summaries = []
    for t in tickers:
        path = folder / f"{safe_name(t)}.json"
        if not path.exists():
            raise FileNotFoundError(f"No summary for {t}: run `python run.py {t}` first.")
        summaries.append(json.loads(path.read_text()))
    return summaries


def _direction_rows(summaries):
    rows = []
    for s in summaries:
        for model, m in s["models"].items():
            luck = m["share of random >="]
            rows.append({
                "stock": s["ticker"],
                "model": model,
                "accuracy": m["accuracy"],
                "always up": s["always up accuracy"],
                "edge": m["accuracy"] - s["always up accuracy"],
                "p-value vs always up": m["p-value vs always up"],
                "annual return": m["annual return"],
                "buy and hold": s["buy and hold"]["annual return"],
                "share of random >=": np.nan if luck is None else luck,
            })
    return rows, "p-value vs always up"


def _volatility_rows(summaries):
    rows = []
    for s in summaries:
        v = s.get("volatility")
        if v is None:
            raise ValueError(
                f"The summary for {s['ticker']} has no move-size results; "
                f"it was made before phase 4. Run `python run.py {s['ticker']}` again."
            )
        for model, m in v["models"].items():
            rows.append({
                "stock": s["ticker"],
                "model": model,
                "accuracy": m["accuracy"],
                "same as today": v["same as today accuracy"],
                "majority": v["majority accuracy"],
                "edge": m["accuracy"] - v["same as today accuracy"],
                "p-value vs same as today": m["p-value vs same as today"],
                "Sharpe (half on big)": m["strategy"]["Sharpe ratio"],
                "Sharpe (buy and hold)": v["buy and hold"]["Sharpe ratio"],
                "max drawdown (half on big)": m["strategy"]["max drawdown"],
                "max drawdown (buy and hold)": v["buy and hold"]["max drawdown"],
                "share of random >=": m["share of random >="],
            })
    return rows, "p-value vs same as today"


def comparison_table(summaries, alpha=0.05, task="direction"):
    """One row per stock and model, with plain and corrected pass/fail columns.

    task: "direction" (up vs. down, judged against "always up" and, in the
          backtest, on annual return) or "volatility" (big vs. small moves,
          judged against "same as today" and, in the half-on-big backtest,
          on Sharpe ratio). Each task gets its own correction.

    Returns (table, info):
      table: accuracy vs the task's baseline, backtest results, and four
             yes/no columns: does it pass at alpha, and at alpha divided by
             the number of comparisons?
      info:  number of comparisons, the corrected threshold, and how many
             passes you would expect by luck alone at alpha
    """

    if task == "direction":
        rows, p_column = _direction_rows(summaries)
    elif task == "volatility":
        rows, p_column = _volatility_rows(summaries)
    else:
        raise ValueError(f"task must be 'direction' or 'volatility', not {task!r}")
    table = pd.DataFrame(rows).set_index(["stock", "model"])
    n = len(table)
    corrected = alpha / n if n else alpha

    p = table[p_column]
    luck = table["share of random >="]
    table["accuracy edge (5%)?"] = p < alpha
    table["accuracy edge (corrected)?"] = p < corrected
    table["beats luck (5%)?"] = luck <= alpha
    table["beats luck (corrected)?"] = luck <= corrected

    info = {
        "comparisons": n,
        "alpha": alpha,
        "corrected threshold": corrected,
        "expected chance passes": n * alpha,
        "accuracy passes (5%)": int(table["accuracy edge (5%)?"].sum()),
        "accuracy passes (corrected)": int(table["accuracy edge (corrected)?"].sum()),
        "luck passes (5%)": int(table["beats luck (5%)?"].sum()),
        "luck passes (corrected)": int(table["beats luck (corrected)?"].sum()),
    }
    return table, info
