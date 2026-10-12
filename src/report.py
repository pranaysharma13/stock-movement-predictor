"""
Summaries for each stock, and the comparison across stocks.

safe_name()          a ticker turned into a safe file name ("^GSPC" -> "_GSPC")
summarize()          the headline numbers of one stock's report
save_summary()       write them to reports/summaries/<ticker>.json
load_summaries()     read them back for the comparison
comparison_table()   one row per stock and model, plus the lucky-winners check

The lucky-winners problem: test enough stocks and models, and a few will
"win" by chance alone. At the usual 5% level, about 1 comparison in 20 passes
even when nothing real is going on. So the comparison also applies a stricter
bar (the Bonferroni correction): divide 5% by the number of comparisons.
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


def comparison_table(summaries, alpha=0.05):
    """One row per stock and model, with plain and corrected pass/fail columns.

    Returns (table, info):
      table: accuracy vs "always up", backtest vs buy-and-hold, and four
             yes/no columns: does it pass at alpha, and at alpha divided by
             the number of comparisons?
      info:  number of comparisons, the corrected threshold, and how many
             passes you would expect by luck alone at alpha
    """
    
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
    table = pd.DataFrame(rows).set_index(["stock", "model"])
    n = len(table)
    corrected = alpha / n if n else alpha

    p = table["p-value vs always up"]
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