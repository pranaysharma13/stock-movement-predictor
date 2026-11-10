"""
Phase 6: the live forward test.

Every weekday evening (see .github/workflows/forward-test.yml), forward_test.py:

  1. downloads the latest prices,
  2. fills in what actually happened for earlier predictions (fill_outcomes),
  3. trains logistic regression on all history up to today and writes down
     tomorrow's predictions (predict_today, add_predictions),
  4. rewrites the scorecard (scorecard).

The log, forward_test/predictions.csv, has one row per stock per day. Its
prediction columns are written once and never changed afterwards; only the
empty outcome columns are filled in later. Because each run is committed to
git, the commit history proves every prediction existed before its outcome.
Every save is checked against the file's text as it was on disk
(check_saved_text), down to the last digit, so an old row can't change even
by accident.

A prediction is only recorded if the day it predicts hasn't started yet
(recorded_too_late), and an outcome only if it is a plausible next trading day
(next_trading_day): a gap of at most MAX_GAP_DAYS calendar days, with trading volume.

Unlike the backtests, nothing here can be tuned after the fact: these are
predictions about days that hadn't happened yet.
"""
import io


import numpy as np
import pandas as pd

from .evaluate import edge_half_width, get_models, paired_p_value
from .features import FEATURES, VOL_FEATURES, build_dataset, build_vol_dataset, make_features, make_vol_features


TICKERS = ["SPY", "NVDA", "AAPL", "GOOGL"]
MODEL = "Logistic regression"


# Written once, when the prediction is made, and never changed
PREDICTION_COLUMNS = [
    "made_on",          # the last close the prediction could use (New York date)
    "ticker",
    "close",            # that day's adjusted close, for reference
    "typical_move",     # that day's bar for a "big" move
    "trained_through",  # the last day whose label the models trained on
    "dir_prob",         # logistic regression: probability tomorrow closes higher
    "dir_pred",         # 1 = up, 0 = down
    "size_prob",        # logistic regression: probability tomorrow's move is big
    "size_pred",        # 1 = big, 0 = small
    "recent_swing",     # the Recent swing rule: 1 = big
    "same_as_today",    # the "same as today" rule: 1 = big
    "code_version",     # the git commit of the code that made the prediction
    "recorded_at",      # when the prediction was written (UTC)
]


# Filled in on a later run, once the next trading day has closed
OUTCOME_COLUMNS = [
    "outcome_date",     # the next trading day
    "next_ret",         # its return, from that run's freshly downloaded prices
    "up",               # 1 if it closed higher
    "big",              # 1 if its move, up or down, beat typical_move
]
COLUMNS = PREDICTION_COLUMNS + OUTCOME_COLUMNS
TEXT_COLUMNS = ["made_on", "ticker", "trained_through", "code_version", "recorded_at", "outcome_date"]
# Always written with a decimal point, even when no value in them is missing,
# so the same value is always the same text
FLOAT_COLUMNS = ["close", "typical_move", "dir_prob", "size_prob", "next_ret", "up", "big"]

# A next trading day more than this many calendar days later is suspicious
# (a missing day in the download); the longest normal gap is 4 days, e.g. a
# Thursday before Good Friday to the Monday after
MAX_GAP_DAYS = 5
# When the regular session opens, New York time
SESSION_OPENS = "09:30"


# ---------------------------- predictions ------------------------------------

def predict_today(prices, ticker, code_version="unknown", recorded_at=None):
    """Tomorrow's predictions for one stock, using only prices up to the last row.

    The models train on every day whose label is already known: up to the
    day before the last row, whose next close is the last row. The last row
    itself has no label yet; it's the day the prediction is made from.
    """

    direction = build_dataset(prices)
    size = build_vol_dataset(prices)
    today_dir = make_features(prices).iloc[-1]
    today_size = make_vol_features(prices).iloc[-1]
    made_on = prices.index[-1]
    if today_dir[FEATURES].isna().any() or today_size[VOL_FEATURES].isna().any():
        raise ValueError(f"{ticker}: today's features are incomplete; not enough history.")

    dir_model = get_models()[MODEL].fit(direction[FEATURES], direction["target"])
    size_model = get_models()[MODEL].fit(size[VOL_FEATURES], size["target"])
    dir_prob = float(dir_model.predict_proba(today_dir[FEATURES].to_frame().T)[0, 1])
    size_prob = float(size_model.predict_proba(today_size[VOL_FEATURES].to_frame().T)[0, 1])

    return {
        "made_on": str(made_on.date()),
        "ticker": ticker,
        "close": float(prices["Close"].iloc[-1]),
        "typical_move": float(today_size["typical_move"]),
        "trained_through": str(min(direction.index[-1], size.index[-1]).date()),
        "dir_prob": dir_prob,
        "dir_pred": int(dir_prob > 0.5),
        "size_prob": size_prob,
        "size_pred": int(size_prob > 0.5),
        "recent_swing": int(today_size["wide_range"]),
        "same_as_today": int(today_size["abs_ret_rel"] > 1),
        "code_version": code_version,
        "recorded_at": recorded_at or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def _typed(log):
    """Text columns stay text, even while every value in them is still empty, and
    number columns that can be empty stay floats, so they are always saved the same way."""

    log = log.reindex(columns=COLUMNS)
    for c in TEXT_COLUMNS:
        log[c] = log[c].astype(object)
    for c in FLOAT_COLUMNS:
        log[c] = log[c].astype(float)
    return log


def empty_log():
    return _typed(pd.DataFrame(columns=COLUMNS))


def load_log(path):
    """The prediction log, or an empty one if it doesn't exist yet."""

    if not path.exists():
        return empty_log()
    # round_trip: read every number back exactly as it was written. pandas' default
    # parser can be off in the 17th digit, and saving that back would rewrite old rows.
    log = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS}, float_precision="round_trip")
    return _typed(log)


def log_text(log):
    """The log as the CSV text save_log writes."""

    buffer = io.StringIO()
    _typed(log).to_csv(buffer, index=False)
    return buffer.getvalue()


def save_log(log, path, before_text=None):
    """Write the log. With before_text (the file as it was on disk when the run
    started), first check that no old row's text changed (check_saved_text)."""

    text = log_text(log)
    if before_text is not None:
        check_saved_text(before_text, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def add_predictions(log, rows):
    """Append new prediction rows, skipping any (made_on, ticker) already in the log.

    A second run on the same day, or a run on a market holiday (when the last
    close is still yesterday's), adds nothing. Existing rows are never touched.
    """

    have = set(zip(log["made_on"].astype(str), log["ticker"].astype(str)))
    new = [r for r in rows if (r["made_on"], r["ticker"]) not in have]
    if not new:
        return log.copy(), 0
    added = _typed(pd.DataFrame(new))
    out = added if log.empty else pd.concat([_typed(log), added], ignore_index=True)
    return _typed(out), len(new)


# ---------------------------- outcomes ------------------------------------
def next_trading_day(prices, made_on):
    """The first later day in prices, if it is a plausible next trading day; else None.

    None if it hasn't closed yet, if it comes more than MAX_GAP_DAYS calendar
    days later (a day is probably missing from the download), or if it had
    no trading volume (a stale row). Those outcomes are left empty and tried
    again on the next run.
    """

    later = prices.index[prices.index > made_on]
    if len(later) == 0:
        return None
    next_day = later[0]
    if (next_day - made_on).days > MAX_GAP_DAYS:
        return None
    if "Volume" in prices and not prices.loc[next_day, "Volume"] > 0:
        return None
    return next_day

def fill_outcomes(log, prices_by_ticker):
    """Fill in what happened, for predictions whose next trading day has closed.

    prices_by_ticker: {ticker: prices freshly downloaded on this run}. The
    return is taken from that one download for both days, so the two closes
    are adjusted for dividends and splits in the same way. Rows that already
    have an outcome, or whose stock wasn't downloaded, are left alone, and so
    are rows whose next day doesn't look right (see next_trading_day).
    """

    log = _typed(log.copy())
    for i, row in log.iterrows():
        if pd.notna(row["outcome_date"]) and str(row["outcome_date"]) != "":
            continue
        prices = prices_by_ticker.get(row["ticker"])
        if prices is None:
            continue
        made_on = pd.Timestamp(row["made_on"])
        if made_on not in prices.index:
            continue
        next_day = next_trading_day(prices, made_on)
        if next_day is None:
            continue   # tomorrow hasn't closed yet, or the download looks wrong
        next_ret = prices.loc[next_day, "Close"] / prices.loc[made_on, "Close"] - 1
        log.loc[i, "outcome_date"] = str(next_day.date())
        log.loc[i, "next_ret"] = float(next_ret)
        log.loc[i, "up"] = int(next_ret > 0)
        log.loc[i, "big"] = int(abs(next_ret) > float(row["typical_move"]))
    return log
def recorded_too_late(made_on, now):
    """True if a prediction from made_on's close would be recorded after the next
    session had opened: the day it predicts would already be under way.

    The next session is taken as the next weekday at 9:30 am New York time. On a
    weekday holiday that is earlier than the real next session, so a run on a
    holiday morning is refused too; that only errs on the side of caution.
    """

    next_session = pd.Timestamp(made_on) + pd.offsets.BDay(1)
    opens = pd.Timestamp(f"{next_session.date()} {SESSION_OPENS}", tz="America/New_York")
    return pd.Timestamp(now) >= opens


def check_saved_text(before_text, after_text):
    """Raise unless after_text keeps every old row of before_text, down to the last digit.

    Compares the CSV text itself, not numbers read back from it, so an old value
    can't change even by a rounding error. Old rows must come first and in
    order; their prediction columns must be identical; an outcome may be filled
    in where it was empty, but once there it can't change.
    """

    if not before_text.strip():
        return

    def cells(text):
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    old, new = cells(before_text), cells(after_text)
    if list(old.columns) != list(new.columns):
        raise RuntimeError("The log's columns changed.")
    if len(new) < len(old):
        raise RuntimeError("Prediction rows were removed from the log.")
    kept = new.iloc[:len(old)].reset_index(drop=True)
    if not old[PREDICTION_COLUMNS].equals(kept[PREDICTION_COLUMNS]):
        raise RuntimeError("Existing predictions were changed; the log must only grow.")
    had = old["outcome_date"] != ""
    if not old.loc[had, OUTCOME_COLUMNS].equals(kept.loc[had, OUTCOME_COLUMNS]):
        raise RuntimeError("Recorded outcomes were changed.")


def check_predictions_unchanged(before, after):
    """Raise if any prediction already in the log was changed or removed.

    A safety net run before every save: the forward test is only worth
    anything if past predictions can't be edited after the fact.
    """

    if before.empty:
        return
    if len(after) < len(before):
        raise RuntimeError("Prediction rows were removed from the log.")
    old = before[PREDICTION_COLUMNS].astype(str).reset_index(drop=True)
    new = after[PREDICTION_COLUMNS].iloc[:len(before)].astype(str).reset_index(drop=True)
    if not old.equals(new):
        raise RuntimeError("Existing predictions were changed; the log must only grow.")
    # an outcome, once recorded, stays
    had = before["outcome_date"].notna() & (before["outcome_date"].astype(str) != "")
    old_out = before.loc[had, OUTCOME_COLUMNS].astype(str).reset_index(drop=True)
    new_out = after.iloc[:len(before)].loc[had.to_numpy(), OUTCOME_COLUMNS].astype(str).reset_index(drop=True)
    if not old_out.equals(new_out):
        raise RuntimeError("Recorded outcomes were changed.")


# ---------------------------- scorecard ------------------------------------

def _scored(log):
    done = log[log["outcome_date"].notna() & (log["outcome_date"].astype(str) != "")].copy()
    for c in ["dir_pred", "size_pred", "recent_swing", "same_as_today", "up", "big"]:
        done[c] = done[c].astype(int)
    return done


def score_rows(log):
    """Accuracy so far, per stock and for all stocks together.

    Direction: logistic regression against "always up". Move size: logistic
    regression against the Recent swing rule (the main bar) and "same as
    today". p-values are the same paired tests as in the reports; "edge ±" is
    the 95% range of the edge.
    """

    done = _scored(log)
    rows = []
    groups = [(t, done[done["ticker"] == t]) for t in sorted(done["ticker"].unique())]
    if len(groups) > 1:
        groups.append(("All four" if len(groups) == 4 else "All", done))
    for name, g in groups:
        n = len(g)
        if n == 0:
            continue
        up, big = g["up"], g["big"]
        ones = pd.Series(1, index=g.index)
        dir_right = g["dir_pred"] == up
        size_right = g["size_pred"] == big
        rule_right = g["recent_swing"] == big
        dir_w, dir_l = int((dir_right & ~(ones == up)).sum()), int((~dir_right & (ones == up)).sum())
        size_w, size_l = int((size_right & ~rule_right).sum()), int((~size_right & rule_right).sum())
        rows.append({
            "stock": name,
            "days scored": n,
            "direction: model": dir_right.mean(),
            "direction: always up": (up == 1).mean(),
            "direction: edge": (dir_w - dir_l) / n,
            "direction: edge ±": edge_half_width(dir_w, dir_l, n),
            "direction: p-value vs always up": paired_p_value(g["dir_pred"], ones, up),
            "move size: model": size_right.mean(),
            "move size: recent swing": rule_right.mean(),
            "move size: same as today": (g["same_as_today"] == big).mean(),
            "move size: edge": (size_w - size_l) / n,
            "move size: edge ±": edge_half_width(size_w, size_l, n),
            "move size: p-value vs recent swing": paired_p_value(g["size_pred"], g["recent_swing"], big),
        })
    return pd.DataFrame(rows)


def _pct(x):
    return f"{x:.1%}"
def _p(p):
    """A p-value as the reports show it: small ones as "<0.001" rather than "0.00"."""

    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _edge(edge, half):
    return f"{edge * 100:+.1f} ± {half * 100:.1f}"


def scorecard(log):
    """The forward test's results so far, as Markdown.

    It depends only on the log, so a run that adds nothing changes nothing.
    """

    lines = ["# Forward test scorecard", ""]
    if log.empty:
        return "\n".join(lines + ["No predictions yet.", ""])
    first, last = log["made_on"].min(), log["made_on"].max()
    outcomes = log["outcome_date"].dropna().astype(str)
    outcomes = outcomes[outcomes != ""]
    latest = f"; latest outcome: {outcomes.max()}" if len(outcomes) else ""
    lines += [f"*Up to the close of {last}{latest}. Predictions are written before the day they predict; "
              f"see `predictions.csv` and its git history.*", ""]
    pending = log[log["outcome_date"].isna() | (log["outcome_date"].astype(str) == "")]
    lines += [f"Predictions made from {first} to {last}: {len(log)} in all, "
              f"{len(pending)} still waiting for their day to close.", ""]
    table = score_rows(log)
    if table.empty:
        return "\n".join(lines + ["No outcomes yet: the first ones arrive after the next trading day closes.", ""])

    lines += ["## Direction: logistic regression vs \"always up\"", "",
              "| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |",
              "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['stock']} | {r['days scored']} | {_pct(r['direction: model'])} | "
                     f"{_pct(r['direction: always up'])} | {_edge(r['direction: edge'], r['direction: edge ±'])} | "
                     f"{_p(r['direction: p-value vs always up'])} |")
    lines += ["", "## Move size: logistic regression vs the Recent swing rule", "",
              "| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |",
              "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['stock']} | {r['days scored']} | {_pct(r['move size: model'])} | "
                     f"{_pct(r['move size: recent swing'])} | {_pct(r['move size: same as today'])} | "
                     f"{_edge(r['move size: edge'], r['move size: edge ±'])} | "
                     f"{_p(r['move size: p-value vs recent swing'])} |")
    single = table[~table["stock"].isin(["All four", "All"])]          # each stock, not the pooled row
    n = int(single["days scored"].max())
    lines += ["", "## How much to read into this", "",
              f"Each stock has at most {n} scored day{'' if n == 1 else 's'}. An accuracy over n days is uncertain by about "
              f"±{100 / np.sqrt(max(n, 1)):.0f} points, so early numbers swing a lot. In the backtests, "
              "logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. "
              "The two agree on about 4 days in 5, so only the other days carry evidence, and telling "
              "an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 "
              "years with all four stocks together (the \"All four\" row), though the stocks move "
              "together, so their days aren't fully independent. A year is enough to catch a large "
              "edge, or to rule one out. The p-values are paired tests, as in the reports, and are "
              "not corrected for the four stocks and two tasks. They are one-sided (is the model "
              "better?), while the 95% ranges are two-sided, so a range can just touch zero while "
              "the p-value is below 0.05.", ""]
    return "\n".join(lines)
