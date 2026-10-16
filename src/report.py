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
bar, the Holm correction (see holm()). The two tasks answer different
questions, so each is corrected on its own.

Accuracy is always compared with a baseline using the paired test
(paired_p_value): "always up" in Part 1, the "Recent swing" rule in Part 2.
"""


import json
from pathlib import Path

import numpy as np
import pandas as pd


from .data import safe_name  # noqa: F401  (kept here too: reports are named with it)
from .evaluate import (backtest, edge_half_width, hold_for_week, luck_check, paired_counts,
                       paired_p_value, performance)

ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = ROOT / "reports" / "summaries"


def _cash_of(test):
    """The test rows' interest on cash, if the dataset carries it (column cash_ret)."""

    return test["cash_ret"] if "cash_ret" in test else None


def _paired(pred, baseline_pred, target, name):
    wins, losses = paired_counts(pred, baseline_pred, target)
    return {f"p-value vs {name}": paired_p_value(pred, baseline_pred, target),
            f"wins vs {name}": wins, f"losses vs {name}": losses}


def summarize(ticker, test, model_preds, cost=0.001, luck=None, luck_trials=2000):
    """Headline numbers for one stock.

    ticker:      the stock's symbol
    test:        the test-period rows of the dataset (needs target and
                 next_ret; cash_ret too if cash earns interest)
    model_preds: {model name: DataFrame with a "pred" column}, from walk_forward()
    cost:        trading cost per buy or sell, as in the backtest
    luck:        {model name: luck_check() result}, if already calculated;
                 missing ones are calculated here with luck_trials trials

    Returns a plain dict, ready to save as JSON.
    """

    target, next_ret, cash = test["target"], test["next_ret"], _cash_of(test)
    always_up = float(target.mean())
    ones = pd.Series(1, index=test.index)
    hold = performance(backtest(ones, next_ret, cost, cash), cash)
    luck = dict(luck or {})

    models = {}
    for name, p in model_preds.items():
        pred = p["pred"].reindex(test.index)
        correct = pred == target
        n, k = len(correct), int(correct.sum())
        perf = performance(backtest(pred, next_ret, cost, cash), cash)
        cash_days = int((pred == 0).sum())
        if cash_days == 0:
            luck_share = None          # never in cash: nothing to compare with random cash days
        else:
            if name not in luck:
                luck[name] = luck_check(pred, next_ret, cost=cost, n_trials=luck_trials, cash_ret=cash)
            luck_share = float(luck[name]["share of random >="])
        models[name] = {
            "accuracy": k / n,
            "predicted up": float(pred.mean()),
            # Paired test: only days where the model says "down" can differ from "always up"
            **_paired(pred, ones, target, "always up"),
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
        "average cash interest": float(cash.mean() * 252) if cash is not None else 0.0,
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
    luck:        luck_check(..., metric="Sharpe ratio") results, if already
                 calculated: {model name: daily result,
                 f"{model name}, weekly": weekly result (by_week=True)}

    Each model is judged on accuracy against both rules (paired test; "Recent
    swing" is the main bar), and its half-on-big strategy, decided daily and
    weekly, on Sharpe ratio against random half-position days or weeks.
    """

    target, next_ret, cash = test["target"], test["next_ret"], _cash_of(test)
    luck = dict(luck or {})

    def strategy(position):
        perf = performance(backtest(position, next_ret, cost, cash), cash)
        return {k: float(perf[k]) for k in ("annual return", "Sharpe ratio", "max drawdown")}

    def both_strategies(pred):
        return {"strategy": strategy(half_on_big(pred)),
                "strategy (weekly)": strategy(half_on_big(hold_for_week(pred)))}

    models = {}
    for name, p in model_preds.items():
        pred = p["pred"].reindex(test.index)
        daily, weekly = half_on_big(pred), half_on_big(hold_for_week(pred))
        if name not in luck:
            luck[name] = luck_check(daily, next_ret, cost=cost, n_trials=luck_trials,
                                    metric="Sharpe ratio", cash_ret=cash)
        if f"{name}, weekly" not in luck:
            luck[f"{name}, weekly"] = luck_check(weekly, next_ret, cost=cost, n_trials=luck_trials,
                                                 metric="Sharpe ratio", by_week=True, cash_ret=cash)
        models[name] = {
            "accuracy": float((pred == target).mean()),
            "predicted big": float(pred.mean()),
            # Paired tests: the chance of winning this many of the days where
            # the model and the rule disagree, if the model were no better
            **_paired(pred, baselines["Recent swing"], target, "recent swing"),
            **_paired(pred, baselines["Same as today"], target, "same as today"),
            **both_strategies(pred),
            "share of random >=": float(luck[name]["share of random >="]),
            "share of random >= (weekly)": float(luck[f"{name}, weekly"]["share of random >="]),
        }

    hold = performance(backtest(pd.Series(1, index=test.index), next_ret, cost, cash), cash)
    return {
        "test start": str(test.index[0].date()),
        "test end": str(test.index[-1].date()),
        "test days": int(len(test)),
        "share of big days": float(target.mean()),
        "majority accuracy": float(max(target.mean(), 1 - target.mean())),
        "same as today accuracy": float((baselines["Same as today"] == target).mean()),
        "recent swing accuracy": float((baselines["Recent swing"] == target).mean()),
        "buy and hold": {k: float(hold[k]) for k in ("annual return", "Sharpe ratio", "max drawdown")},
        "same as today strategy": strategy(half_on_big(baselines["Same as today"])),
        "recent swing strategy": both_strategies(baselines["Recent swing"]),
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
            if "wins vs always up" not in m:
                raise ValueError(
                    f"The summary for {s['ticker']} was made before phase 4.2 (it has no paired "
                    f"counts). Run `python run.py {s['ticker']}` again."
                )
            luck = m["share of random >="]
            n = s["test days"]
            rows.append({
                "stock": s["ticker"],
                "model": model,
                "accuracy": m["accuracy"],
                "always up": s["always up accuracy"],
                "edge": m["accuracy"] - s["always up accuracy"],
                "edge ±": edge_half_width(m["wins vs always up"], m["losses vs always up"], n),
                "p-value vs always up": m["p-value vs always up"],
                "annual return": m["annual return"],
                "buy and hold": s["buy and hold"]["annual return"],
                "share of random >=": np.nan if luck is None else luck,
            })
    return rows, "p-value vs always up", {"": "share of random >="}


def _volatility_rows(summaries):
    rows = []
    for s in summaries:
        v = s.get("volatility")
        if (v is None or "recent swing accuracy" not in v
                or any("wins vs recent swing" not in m for m in v["models"].values())):
            raise ValueError(
                f"The summary for {s['ticker']} has no move-size results, or was made "
                f"before phase 4.2. Run `python run.py {s['ticker']}` again."
            )
        n = v["test days"]
        for model, m in v["models"].items():
            rows.append({
                "stock": s["ticker"],
                "model": model,
                "accuracy": m["accuracy"],
                "recent swing": v["recent swing accuracy"],
                "same as today": v["same as today accuracy"],
                "edge": m["accuracy"] - v["recent swing accuracy"],
                "edge ±": edge_half_width(m["wins vs recent swing"], m["losses vs recent swing"], n),
                "p-value vs recent swing": m["p-value vs recent swing"],
                "edge over same as today": m["accuracy"] - v["same as today accuracy"],
                "edge over same as today ±": edge_half_width(m["wins vs same as today"],
                                                             m["losses vs same as today"], n),
                "p-value vs same as today": m["p-value vs same as today"],
                "Sharpe (daily)": m["strategy"]["Sharpe ratio"],
                "Sharpe (weekly)": m["strategy (weekly)"]["Sharpe ratio"],
                "Sharpe (buy and hold)": v["buy and hold"]["Sharpe ratio"],
                "max drawdown (weekly)": m["strategy (weekly)"]["max drawdown"],
                "max drawdown (buy and hold)": v["buy and hold"]["max drawdown"],
                "share of random >= (daily)": m["share of random >="],
                "share of random >= (weekly)": m["share of random >= (weekly)"],
            })
    return rows, "p-value vs recent swing", {", daily": "share of random >= (daily)",
                                              ", weekly": "share of random >= (weekly)"}


def holm(values, alpha=0.05):
    """Which results pass after the Holm correction for testing many things.

    values: p-values (or luck shares); missing values count as tests that fail.

    Sort the m values from smallest to largest. The smallest must be at most
    alpha / m (the Bonferroni bar), the next at most alpha / (m - 1), and so
    on; the first one that misses its bar stops the process, and it and every
    larger value fail. Holm keeps the same guarantee as Bonferroni (the chance
    of even one lucky pass stays at most alpha) and never passes fewer results.
    """

    values = np.asarray(values, dtype=float)
    m = len(values)
    passed = np.zeros(m, dtype=bool)
    order = np.argsort(np.where(np.isnan(values), np.inf, values), kind="stable")
    for rank, i in enumerate(order):
        if np.isnan(values[i]) or values[i] > alpha / (m - rank):
            break
        passed[i] = True
    return passed


def comparison_table(summaries, alpha=0.05, task="direction"):
    """One row per stock and model, with plain and corrected pass/fail columns.

    task: "direction" (up vs. down, judged against "always up" and, in the
          backtest, on annual return) or "volatility" (big vs. small moves,
          judged against the "Recent swing" rule and, in the half-on-big
          backtests, daily and weekly, on Sharpe ratio). Each task gets its
          own correction.

    Returns (table, info):
      table: accuracy vs the task's baseline, with "edge ±", the 95% range of
             the edge from the paired counts; backtest results; and yes/no
             columns: does it pass at alpha, and after the Holm correction?
             The volatility task has luck columns for both the daily and the
             weekly strategy ("beats luck, daily (5%)?" and so on).
      info:  number of comparisons, the strictest corrected bar (alpha / number
             of comparisons, what the smallest value must clear), how many
             passes you would expect by luck alone at alpha, and pass counts

    Each luck column is one comparison per row, so the volatility task's two
    luck columns form one family of twice the number of rows. For the
    volatility task, "beats same as today (corrected)?" is shown for context.
    """

    if task == "direction":
        rows, p_column, luck_columns = _direction_rows(summaries)
    elif task == "volatility":
        rows, p_column, luck_columns = _volatility_rows(summaries)
    else:
        raise ValueError(f"task must be 'direction' or 'volatility', not {task!r}")
    table = pd.DataFrame(rows).set_index(["stock", "model"])
    n = len(table)
    n_luck = n * len(luck_columns)

    p = table[p_column]
    table["accuracy edge (5%)?"] = p <= alpha
    table["accuracy edge (corrected)?"] = holm(p, alpha)
    if task == "volatility":
        table["beats same as today (corrected)?"] = holm(table["p-value vs same as today"], alpha)
    # The luck columns are corrected together, as one family
    luck_values = np.concatenate([table[c].to_numpy(dtype=float) for c in luck_columns.values()])
    luck_passed = holm(luck_values, alpha)
    for k, (suffix, column) in enumerate(luck_columns.items()):
        table[f"beats luck{suffix} (5%)?"] = table[column] <= alpha
        table[f"beats luck{suffix} (corrected)?"] = luck_passed[k * n:(k + 1) * n]

    flags = [f"beats luck{x}" for x in luck_columns]
    info = {
        "method": "Holm",
        "comparisons": n,
        "alpha": alpha,
        "corrected threshold": alpha / n if n else alpha,
        "expected chance passes": n * alpha,
        "accuracy passes (5%)": int(table["accuracy edge (5%)?"].sum()),
        "accuracy passes (corrected)": int(table["accuracy edge (corrected)?"].sum()),
        "luck comparisons": n_luck,
        "luck corrected threshold": alpha / n_luck if n_luck else alpha,
        "expected chance luck passes": n_luck * alpha,
        "luck passes (5%)": int(sum(table[f"{f} (5%)?"].sum() for f in flags)),
        "luck passes (corrected)": int(sum(table[f"{f} (corrected)?"].sum() for f in flags)),
    }
    if task == "volatility":
        info["same as today passes (corrected)"] = int(table["beats same as today (corrected)?"].sum())
    return table, info
