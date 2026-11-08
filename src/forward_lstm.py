"""
Phase 6.1: the LSTM in the live forward test, for move size only.

It runs beside logistic regression's forward test but keeps its own log,
forward_test/lstm_predictions.csv, and its own scorecard. Logistic
regression's code and record don't change, and the LSTM is added only after
it beat the Recent swing rule on three stocks it had never seen (phase 5).

Training. On the first evening of each month (or when no saved model
exists), the LSTM is trained on every day whose label is already known, the
same way as each year of the walk-forward test (src/lstm.py): the last
VALIDATION_DAYS labelled days, about a year, pick the number of epochs, then
every labelled day is used for that many epochs; 3 seeds, averaged. The
trained networks and their scaler are saved to forward_test/lstm_models/ and
committed, so every prediction can be traced to the exact model that made it.
On the other evenings, the saved networks are only loaded and asked.

Like logistic regression's log, the prediction columns are written once and
never changed; each save is checked against the file's text on disk; a
prediction is refused once its day has opened; and an outcome is recorded
only for a plausible next trading day (see src/forward.py).
"""


import io

import numpy as np
import pandas as pd

from . import lstm
from .evaluate import edge_half_width, paired_p_value
from .features import build_vol_dataset, make_vol_features
from .forward import TICKERS, _p, _pct, next_trading_day

# About a year of labelled days held back to choose the number of epochs
VALIDATION_DAYS = 252

PREDICTION_COLUMNS = [
    "made_on",         # the close the prediction is made from (New York date)
    "ticker",
    "close",
    "typical_move",    # median absolute daily move over the past 250 days, as of made_on
    "trained_through", # the last labelled day the model trained on
    "model_month",     # the month the model was trained in, e.g. "2026-11"
    "size_prob",       # the seeds' average probability of a big move on the next trading day
    "size_pred",       # 1 = big, 0 = small
    "seeds_agree",     # 1 if every seed is on the same side of 0.5
    "recent_swing",    # what the Recent swing rule says (1 = big)
    "code_version",
    "recorded_at",
]
OUTCOME_COLUMNS = [
    "outcome_date",    # the next trading day
    "next_ret",        # close-to-close return from made_on to outcome_date
    "big",             # 1 if |next_ret| > typical_move
]
COLUMNS = PREDICTION_COLUMNS + OUTCOME_COLUMNS
TEXT_COLUMNS = ["made_on", "ticker", "trained_through", "model_month", "code_version", "recorded_at",
                "outcome_date"]
FLOAT_COLUMNS = ["close", "typical_move", "size_prob", "next_ret", "big"]


# ---------------------------- training ------------------------------------

def live_split(n_days, validation_days=VALIDATION_DAYS):
    """Positions of the validation-fit and validation days among n_days labelled days.

    Validation: the last validation_days. Validation-fit: the days before,
    without the last one, whose label is the first validation day's move.
    """

    if n_days <= validation_days + 1:
        raise ValueError(f"Need more than {validation_days + 1} labelled days to train, got {n_days}.")
    val_idx = np.arange(n_days - validation_days, n_days)
    fit_idx = np.arange(0, n_days - validation_days - 1)
    return fit_idx, val_idx


def train_live(prices, seeds=lstm.SEEDS, window=lstm.WINDOW, epochs=lstm.EPOCHS, hidden=lstm.HIDDEN,
               dropout=0.2, lr=1e-3, batch=128, validation_days=VALIDATION_DAYS):
    """Train the move-size LSTM on every labelled day in prices.

    The last row of prices has no label yet (its next close is unknown), so
    it isn't trained on: it's the day the next prediction is made from.
    Returns a model dict for save_model() and predict_lstm_today().
    """

    lstm.require_torch()
    data = build_vol_dataset(prices)
    inputs = lstm.sequence_inputs(prices)
    X_all, days = lstm.make_sequences(inputs, data.index, window)
    y_all = data.loc[days, "target"].to_numpy()
    fit_idx, val_idx = live_split(len(days), validation_days)
    fit_scaler = lstm.fit_scaler(inputs, days[fit_idx[-1]])
    scaler = lstm.fit_scaler(inputs, days[-1])
    X_fit, X_val = lstm.scale(X_all[fit_idx], fit_scaler), lstm.scale(X_all[val_idx], fit_scaler)
    X_train = lstm.scale(X_all, scaler)
    states, best_epochs = [], []
    for seed in seeds:
        # 1. find how many epochs help, holding back the last year
        _, best = lstm._train(X_fit, y_all[fit_idx], seed, epochs, hidden, dropout, lr, batch,
                              X_val, y_all[val_idx])
        # 2. train on every labelled day for that many epochs
        net, _ = lstm._train(X_train, y_all, seed, best, hidden, dropout, lr, batch)
        states.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        best_epochs.append(int(best))
    return {"states": states, "seeds": [int(s) for s in seeds], "best epochs": best_epochs,
            "mean": [float(x) for x in scaler[0]], "std": [float(x) for x in scaler[1]],
            "trained through": str(days[-1].date()), "month": str(prices.index[-1].strftime("%Y-%m")),
            "window": int(window), "hidden": int(hidden), "dropout": float(dropout),
            "inputs": list(lstm.INPUTS)}


def save_model(model, path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def load_model(path):
    """A saved model, or None if there isn't one."""

    import torch
    if not path.exists():
        return None
    return torch.load(path, weights_only=True)


def needs_training(model, made_on):
    """True if there's no model yet, or it was trained in an earlier month than made_on's."""

    return model is None or model["month"] != pd.Timestamp(made_on).strftime("%Y-%m")


def _networks(model):
    nets = []
    for state in model["states"]:
        net = lstm._network(len(model["inputs"]), model["hidden"], model["dropout"])
        net.load_state_dict(state)
        nets.append(net)
    return nets


def predict_lstm_today(prices, ticker, model, code_version="unknown", recorded_at=None):
    """The LSTM's move-size prediction from the last close in prices."""

    made_on = prices.index[-1]
    if pd.Timestamp(model["trained through"]) >= made_on:
        raise ValueError(f"{ticker}: the model was trained on {model['trained through']}, "
                         f"not before {made_on.date()}.")
    inputs = lstm.sequence_inputs(prices)
    X, kept = lstm.make_sequences(inputs, prices.index[-1:], model["window"])
    if len(kept) == 0:
        raise ValueError(f"{ticker}: the last {model['window']} days are incomplete; no prediction.")
    X = lstm.scale(X, (np.array(model["mean"], dtype=np.float32), np.array(model["std"], dtype=np.float32)))
    probs = [float(lstm._probs(net, X)[0]) for net in _networks(model)]
    today = make_vol_features(prices).iloc[-1]
    prob = float(np.mean(probs))
    return {
        "made_on": str(made_on.date()),
        "ticker": ticker,
        "close": float(prices["Close"].iloc[-1]),
        "typical_move": float(today["typical_move"]),
        "trained_through": model["trained through"],
        "model_month": model["month"],
        "size_prob": prob,
        "size_pred": int(prob > 0.5),
        "seeds_agree": int(len({p > 0.5 for p in probs}) == 1),
        "recent_swing": int(today["wide_range"]),
        "code_version": code_version,
        "recorded_at": recorded_at or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


# ---------------------------- the log ------------------------------------

def _typed(log):
    log = log.reindex(columns=COLUMNS)
    for c in TEXT_COLUMNS:
        log[c] = log[c].astype(object)
    for c in FLOAT_COLUMNS:
        log[c] = log[c].astype(float)
    return log


def empty_log():
    return _typed(pd.DataFrame(columns=COLUMNS))


def load_log(path):
    if not path.exists():
        return empty_log()
    # round_trip: every number comes back exactly as it was written (see issue 43)
    return _typed(pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, float_precision="round_trip"))


def log_text(log):
    buffer = io.StringIO()
    _typed(log).to_csv(buffer, index=False)
    return buffer.getvalue()


def check_saved_text(before_text, after_text):
    """Raise unless after_text keeps every old row of before_text, down to the last digit:
    prediction columns identical, outcomes only filled in where empty."""

    if not before_text or not before_text.strip():
        return

    def cells(text):
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    old, new = cells(before_text), cells(after_text)
    if list(old.columns) != list(new.columns):
        raise RuntimeError("The LSTM log's columns changed.")
    if len(new) < len(old):
        raise RuntimeError("Prediction rows were removed from the LSTM log.")
    kept = new.iloc[:len(old)].reset_index(drop=True)
    if not old[PREDICTION_COLUMNS].equals(kept[PREDICTION_COLUMNS]):
        raise RuntimeError("Existing LSTM predictions were changed; the log must only grow.")
    had = old["outcome_date"] != ""
    if not old.loc[had, OUTCOME_COLUMNS].equals(kept.loc[had, OUTCOME_COLUMNS]):
        raise RuntimeError("Recorded LSTM outcomes were changed.")


def save_log(log, path, before_text=None):
    text = log_text(log)
    check_saved_text(before_text, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def add_predictions(log, rows):
    """Append rows whose (made_on, ticker) isn't in the log yet. Returns (log, number added)."""

    have = set(zip(log["made_on"].astype(str), log["ticker"].astype(str)))
    new = [r for r in rows if (r["made_on"], r["ticker"]) not in have]
    if not new:
        return log, 0
    added = _typed(pd.DataFrame(new))
    return _typed(pd.concat([log, added], ignore_index=True) if len(log) else added), len(new)


def fill_outcomes(log, prices_by_ticker):
    """Fill in what happened, exactly as logistic regression's log does (src/forward.py)."""

    log = _typed(log.copy())
    for i, row in log.iterrows():
        if pd.notna(row["outcome_date"]) and str(row["outcome_date"]) != "":
            continue
        prices = prices_by_ticker.get(row["ticker"])
        made_on = pd.Timestamp(row["made_on"])
        if prices is None or made_on not in prices.index:
            continue
        next_day = next_trading_day(prices, made_on)
        if next_day is None:
            continue
        next_ret = prices.loc[next_day, "Close"] / prices.loc[made_on, "Close"] - 1
        log.loc[i, "outcome_date"] = str(next_day.date())
        log.loc[i, "next_ret"] = float(next_ret)
        log.loc[i, "big"] = int(abs(next_ret) > float(row["typical_move"]))
    return log


# ---------------------------- the scorecard ------------------------------------

def _scored(log):
    done = log[log["outcome_date"].notna() & (log["outcome_date"].astype(str) != "")].copy()
    for c in ["size_pred", "recent_swing", "big"]:
        done[c] = done[c].astype(int)
    return done


def score_rows(log, logreg_log=None):
    """Per stock, then all four together: the LSTM against the Recent swing rule
    and, on the days both were scored, against logistic regression's prediction."""

    done = _scored(log)
    if logreg_log is not None and len(logreg_log):
        lr = logreg_log[["made_on", "ticker", "size_pred"]].rename(columns={"size_pred": "logreg_pred"})
        lr = lr.assign(made_on=lr["made_on"].astype(str), ticker=lr["ticker"].astype(str))
        done = done.merge(lr, on=["made_on", "ticker"], how="left")
    else:
        done["logreg_pred"] = np.nan
    rows = []
    groups = [(t, done[done["ticker"] == t]) for t in sorted(done["ticker"].unique(), key=_order)]
    if len(groups) > 1:
        groups.append(("All four" if len(groups) == 4 else "All", done))
    for name, g in groups:
        target, lstm_pred, rule = g["big"], g["size_pred"], g["recent_swing"]
        wins, losses = int(((lstm_pred == target) & (rule != target)).sum()), int(((lstm_pred != target) & (rule == target)).sum())
        both = g[g["logreg_pred"].notna()]
        row = {"stock": name, "days scored": len(g),
               "LSTM": (lstm_pred == target).mean(), "Recent swing": (rule == target).mean(),
               "edge": (lstm_pred == target).mean() - (rule == target).mean(),
               "edge ±": edge_half_width(wins, losses, len(g)),
               "p-value vs Recent swing": paired_p_value(lstm_pred, rule, target),
               "days with both models": len(both)}
        if len(both):
            lr_pred = both["logreg_pred"].astype(int)
            row.update({"LSTM on those days": (both["size_pred"] == both["big"]).mean(),
                        "logistic regression": (lr_pred == both["big"]).mean(),
                        "p-value vs logistic regression": paired_p_value(both["size_pred"], lr_pred, both["big"])})
        rows.append(row)
    return pd.DataFrame(rows)


def _order(ticker):
    return TICKERS.index(ticker) if ticker in TICKERS else len(TICKERS)


def scorecard(log, logreg_log=None, models=None):
    """The LSTM forward test's results so far, as Markdown. It depends only on the
    logs and the saved models, so a run that adds nothing changes nothing.

    models: optional {ticker: model dict} to list which model made the latest predictions.
    """

    lines = ["# LSTM forward test scorecard (move size)", ""]
    if log.empty:
        return "\n".join(lines + ["No predictions yet.", ""])
    first, last = log["made_on"].min(), log["made_on"].max()
    outcomes = log["outcome_date"].dropna().astype(str)
    outcomes = outcomes[outcomes != ""]
    latest = f"; latest outcome: {outcomes.max()}" if len(outcomes) else ""
    pending = int(len(log) - len(outcomes))
    lines += [f"*Up to the close of {last}{latest}. Predictions are written before the day they predict; "
              f"see `lstm_predictions.csv` and its git history.*", "",
              f"Predictions made from {first} to {last}: {len(outcomes)} scored, {pending} still waiting.", ""]
    if models:
        lines += ["| Stock | Model trained in | Trained through | Epochs chosen (each seed) |", "| --- | --- | --- | --- |"]
        for t in sorted(models, key=_order):
            m = models[t]
            lines.append(f"| {t} | {m['month']} | {m['trained through']} | {', '.join(map(str, m['best epochs']))} |")
        lines.append("")
    if not len(outcomes):
        return "\n".join(lines + ["No outcomes yet.", ""])
    table = score_rows(log, logreg_log)
    lines += ["## Against the Recent swing rule", "",
              "| Stock | Days scored | LSTM | Recent swing | Edge | p-value |", "| --- | --- | --- | --- | --- | --- |"]
    for _, r in table.iterrows():
        lines.append(f"| {r['stock']} | {r['days scored']} | {_pct(r['LSTM'])} | {_pct(r['Recent swing'])} | "
                     f"{r['edge'] * 100:+.1f} ± {r['edge ±'] * 100:.1f} points | {_p(r['p-value vs Recent swing'])} |")
    if "logistic regression" in table:
        lines += ["", "## Against logistic regression, on the same days", "",
                  "| Stock | Days | LSTM | Logistic regression | p-value |", "| --- | --- | --- | --- | --- |"]
        for _, r in table[table["days with both models"] > 0].iterrows():
            lines.append(f"| {r['stock']} | {r['days with both models']} | {_pct(r['LSTM on those days'])} | "
                         f"{_pct(r['logistic regression'])} | {_p(r['p-value vs logistic regression'])} |")
    single = table[~table["stock"].isin(["All four", "All"])]
    n = int(single["days scored"].max())
    lines += ["", "## How much to read into this", "",
              f"Each stock has at most {n} scored days, so each accuracy is uncertain by about "
              f"±{100 / max(n, 1) ** 0.5:.0f} points. The backtest edge over Recent swing was 1.5 to 3.7 points; "
              "confirming an edge that size takes roughly 1,400 to 3,800 scored days per stock. The p-values "
              "are paired tests, one-sided and not corrected for the four stocks; the ranges are two-sided.", ""]
    return "\n".join(lines)
