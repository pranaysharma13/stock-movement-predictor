"""
Tests for phase 6, the live forward test: predictions use only what was known
at the close, outcomes are filled correctly, the log only ever grows, and a
daily run works end to end (with a fake download instead of Yahoo).
"""


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import forward_test
from src.data import make_random_walk
from src.evaluate import edge_half_width, paired_p_value
from src.features import make_vol_features
from src.forward import (COLUMNS, OUTCOME_COLUMNS, PREDICTION_COLUMNS, add_predictions,
                         check_predictions_unchanged, check_saved_text, empty_log, fill_outcomes,
                         load_log, log_text, next_trading_day, predict_today, recorded_too_late,
                         save_log, score_rows, scorecard)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def prices():
    return make_random_walk(n_days=1500, seed=11)


# ---------------------------- predictions ------------------------------------

def test_prediction_uses_only_the_last_close_and_earlier(prices, monkeypatch):
    """Both models are fitted on days up to the day before the prediction (whose
    label is the prediction day's close) and predict from the prediction day's row."""

    import src.forward
    real = src.forward.get_models
    fitted, asked = [], []

    class Recording:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            fitted.append(X.index)
            self.model.fit(X, y)
            return self

        def predict_proba(self, X):
            asked.append(X.index)
            return self.model.predict_proba(X)

    monkeypatch.setattr(src.forward, "get_models", lambda: {k: Recording(m) for k, m in real().items()})
    row = predict_today(prices, "X", recorded_at="t")
    assert len(fitted) == 2 and len(asked) == 2                 # direction and move size
    for index in fitted:
        assert index[-1] == prices.index[-2]
    for index in asked:
        assert list(index) == [prices.index[-1]]
    assert row["made_on"] == str(prices.index[-1].date())


def test_models_train_only_on_days_whose_outcome_is_known(prices):
    """The last training day is the day before the prediction: its label is the prediction day's close."""

    row = predict_today(prices, "X")
    assert row["trained_through"] == str(prices.index[-2].date())
    assert row["made_on"] == str(prices.index[-1].date())


def test_rules_match_the_report_definitions(prices):
    row = predict_today(prices, "X")
    today = make_vol_features(prices).iloc[-1]
    assert row["recent_swing"] == int(today["wide_range"])
    assert row["same_as_today"] == int(today["abs_ret_rel"] > 1)
    assert row["typical_move"] == pytest.approx(today["typical_move"])
    assert row["dir_pred"] == int(row["dir_prob"] > 0.5)
    assert row["size_pred"] == int(row["size_prob"] > 0.5)
    assert set(row) == set(PREDICTION_COLUMNS)


# ---------------------------- the log only grows ------------------------------------

def _row(made_on, ticker="X", **kw):
    base = {c: np.nan for c in PREDICTION_COLUMNS}
    base.update({"made_on": made_on, "ticker": ticker, "close": 100.0, "typical_move": 0.01,
                 "dir_prob": 0.6, "dir_pred": 1, "size_prob": 0.4, "size_pred": 0,
                 "recent_swing": 1, "same_as_today": 0, "code_version": "abc", "recorded_at": "t",
                 "trained_through": "2020-01-01"})
    base.update(kw)
    return base


def test_add_predictions_skips_days_already_recorded():
    log, added = add_predictions(empty_log(), [_row("2026-11-05"), _row("2026-11-05", "Y")])
    assert added == 2
    # a second run on the same evening, or on a holiday (last close still the 5th), adds nothing
    again, added = add_predictions(log, [_row("2026-11-05", dir_prob=0.1, dir_pred=0)])
    assert added == 0
    assert again["dir_prob"].tolist() == [0.6, 0.6]


def test_safety_check_refuses_edited_or_removed_predictions():
    log, _ = add_predictions(empty_log(), [_row("2026-11-05"), _row("2026-11-06")])
    edited = log.copy()
    edited.loc[0, "dir_pred"] = 0
    with pytest.raises(RuntimeError, match="changed"):
        check_predictions_unchanged(log, edited)
    with pytest.raises(RuntimeError, match="removed"):
        check_predictions_unchanged(log, log.iloc[:1])
    grown, _ = add_predictions(log, [_row("2026-11-09")])
    check_predictions_unchanged(log, grown)          # adding rows is fine


def test_safety_check_refuses_rewritten_outcomes():
    log, _ = add_predictions(empty_log(), [_row("2026-11-05")])
    log.loc[0, OUTCOME_COLUMNS] = ["2026-11-06", 0.01, 1, 0]
    rewritten = log.copy()
    rewritten.loc[0, "up"] = 0
    with pytest.raises(RuntimeError, match="outcomes"):
        check_predictions_unchanged(log, rewritten)
# Values from the live log that pandas' default CSV parser didn't read back exactly
FULL_DIGITS = dict(typical_move=0.004943305862969205, dir_prob=0.49568276435249065,
                   close=677.5800170898438, size_prob=0.3071196413405543)


def test_log_round_trip(tmp_path):
    """Full-precision numbers come back exactly, and saving again gives the same text."""

    log, _ = add_predictions(empty_log(), [_row("2026-11-05", **FULL_DIGITS)])
    log.loc[0, OUTCOME_COLUMNS] = ["2026-11-06", 0.012345678901234567, 1, 0]
    path = tmp_path / "predictions.csv"
    first = save_log(log, path)
    back = load_log(path)
    assert list(back.columns) == COLUMNS
    for column, value in FULL_DIGITS.items():
        assert back.loc[0, column] == value                     # exactly, not approximately
    check_predictions_unchanged(log, back)
    assert save_log(back, path, before_text=first) == first     # a second save rewrites nothing
    assert load_log(tmp_path / "missing.csv").empty
def test_saved_text_check_catches_a_change_in_the_last_digit():
    log, _ = add_predictions(empty_log(), [_row("2026-11-05", **FULL_DIGITS)])
    before = log_text(log)
    check_saved_text(before, before)
    grown, _ = add_predictions(log, [_row("2026-11-06")])
    check_saved_text(before, log_text(grown))                  # new rows are fine
    with pytest.raises(RuntimeError, match="predictions were changed"):
        check_saved_text(before, before.replace("0.004943305862969205", "0.0049433058629692"))
    with pytest.raises(RuntimeError, match="removed"):
        check_saved_text(log_text(grown), before)
    filled = log.copy()
    filled.loc[0, OUTCOME_COLUMNS] = ["2026-11-06", 0.01, 1, 0]
    check_saved_text(before, log_text(filled))                 # an empty outcome may be filled in
    changed = filled.copy()
    changed.loc[0, "next_ret"] = 0.010000000000000002
    with pytest.raises(RuntimeError, match="outcomes"):
        check_saved_text(log_text(filled), log_text(changed))


def test_outcome_columns_are_saved_the_same_way_once_every_row_is_filled():
    """Whole-number outcomes stay "1.0", not "1", even when no row is still waiting."""

    log, _ = add_predictions(empty_log(), [_row("2026-11-05"), _row("2026-11-06")])
    log.loc[0, OUTCOME_COLUMNS] = ["2026-11-06", 0.01, 1, 0]
    half = log_text(log)
    log.loc[1, OUTCOME_COLUMNS] = ["2026-11-09", -0.02, 0, 1]
    check_saved_text(half, log_text(log))
    assert ",1.0,0.0" in log_text(log)
    log["up"], log["big"] = [1, 0], [0, 1]                       # even as whole-number columns
    check_saved_text(half, log_text(log))


# ---------------------------- outcomes ------------------------------------

def _prices_on(days, closes):
    idx = pd.DatetimeIndex(pd.to_datetime(days), name="Date")
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}, index=idx)


def test_outcome_is_the_next_trading_day_even_after_a_holiday():
    """Predicted from Wed 25 Nov 2026; Thursday is Thanksgiving, so the outcome is Friday."""

    log, _ = add_predictions(empty_log(), [_row("2026-11-25", typical_move=0.01)])
    fresh = _prices_on(["2026-11-24", "2026-11-25", "2026-11-27", "2026-11-30"], [99, 100, 98.5, 103])
    filled = fill_outcomes(log, {"X": fresh})
    assert filled.loc[0, "outcome_date"] == "2026-11-27"
    assert filled.loc[0, "next_ret"] == pytest.approx(-0.015)
    assert filled.loc[0, "up"] == 0
    assert filled.loc[0, "big"] == 1                  # a 1.5% move beats the 1% typical move


def test_outcome_waits_until_the_next_day_has_closed():
    log, _ = add_predictions(empty_log(), [_row("2026-11-06")])
    fresh = _prices_on(["2026-11-05", "2026-11-06"], [99, 100])
    assert pd.isna(fill_outcomes(log, {"X": fresh}).loc[0, "outcome_date"])


def test_outcome_uses_one_download_for_both_closes():
    """The stored close may have been adjusted differently since; the fresh download is used for both days."""

    log, _ = add_predictions(empty_log(), [_row("2026-11-05", close=50.0)])
    fresh = _prices_on(["2026-11-05", "2026-11-06"], [100, 101])
    assert fill_outcomes(log, {"X": fresh}).loc[0, "next_ret"] == pytest.approx(0.01)


def test_recorded_outcomes_are_never_overwritten():
    log, _ = add_predictions(empty_log(), [_row("2026-11-05")])
    log.loc[0, OUTCOME_COLUMNS] = ["2026-11-06", 0.02, 1, 1]
    fresh = _prices_on(["2026-11-05", "2026-11-06"], [100, 90])
    assert fill_outcomes(log, {"X": fresh}).loc[0, "next_ret"] == 0.02


# ---------------------------- scorecard ------------------------------------

def _scored_log():
    rows = []
    days = pd.bdate_range("2026-01-05", periods=6)
    #           dir_pred, up, size_pred, recent_swing, same_as_today, big
    data = [(1, 1, 1, 1, 0, 1), (0, 0, 1, 0, 0, 1), (0, 1, 0, 0, 1, 0),
            (1, 0, 0, 1, 1, 0), (1, 1, 1, 1, 0, 0), (0, 0, 0, 1, 0, 1)]
    for d, (dp, up, sp, rs, sat, big) in zip(days, data):
        r = _row(str(d.date()), dir_pred=dp, size_pred=sp, recent_swing=rs, same_as_today=sat)
        r.update({"outcome_date": "x", "next_ret": 0.0, "up": up, "big": big})
        rows.append(r)
    return pd.DataFrame(rows).reindex(columns=COLUMNS)


def test_score_rows_by_hand():
    log = _scored_log()
    row = score_rows(log).iloc[0]
    up, big = log["up"].astype(int), log["big"].astype(int)
    assert row["days scored"] == 6
    assert row["direction: model"] == pytest.approx(4 / 6)
    assert row["direction: always up"] == pytest.approx(3 / 6)
    # "down" calls on days 2, 3, 6: right on 2 and 6, wrong on 3 -> 2 wins, 1 loss
    assert row["direction: edge"] == pytest.approx((2 - 1) / 6)
    assert row["direction: edge ±"] == pytest.approx(edge_half_width(2, 1, 6))
    assert row["direction: p-value vs always up"] == pytest.approx(
        paired_p_value(log["dir_pred"].astype(int), pd.Series(1, index=log.index), up))
    assert row["move size: model"] == pytest.approx(4 / 6)
    assert row["move size: recent swing"] == pytest.approx(3 / 6)
    assert row["move size: same as today"] == pytest.approx(1 / 6)
    # against Recent swing: 2 wins, 1 loss (p = 0.5); against same as today it would be 4 and 1
    assert row["move size: edge"] == pytest.approx((2 - 1) / 6)
    assert row["move size: p-value vs recent swing"] == pytest.approx(
        paired_p_value(log["size_pred"].astype(int), log["recent_swing"].astype(int), big))


def test_scorecard_before_any_outcomes():
    assert "No predictions yet" in scorecard(empty_log())
    log, _ = add_predictions(empty_log(), [_row("2026-11-06")])
    text = scorecard(log)
    assert "No outcomes yet" in text and "1 still waiting" in text
    assert "Up to the close of 2026-11-06" in text


def test_scorecard_with_outcomes_has_both_tasks_and_a_warning():
    text = scorecard(_scored_log())
    assert "## Direction" in text and "## Move size" in text
    assert "How much to read into this" in text
    assert "two-sided" in text


def test_scorecard_counts_days_per_stock_not_all_four_together():
    """With 20 scored days for each of 4 stocks: 20 days and about ±22 points, not 80 and ±11."""

    rows = []
    for ticker in ["AAA", "BBB", "CCC", "DDD"]:
        for k, d in enumerate(pd.bdate_range("2026-01-05", periods=20)):
            r = _row(str(d.date()), ticker, dir_pred=k % 2, size_pred=k % 3 == 0)
            r.update({"outcome_date": "x", "next_ret": 0.0, "up": k % 2, "big": k % 2})
            rows.append(r)
    text = scorecard(pd.DataFrame(rows).reindex(columns=COLUMNS))
    assert "Each stock has at most 20 scored days" in text
    assert "about ±22 points" in text
    assert "| All four | 80 |" in text


def test_scorecard_shows_small_p_values():
    log = _scored_log()
    log = pd.concat([log] * 30, ignore_index=True)              # the same 6 days, 180 times over
    log["dir_pred"] = log["up"]                                   # a perfect direction model
    text = scorecard(log)
    assert "<0.001" in text and "| 0.00 |" not in text


def test_scorecard_only_changes_when_the_log_does():
    assert scorecard(_scored_log()) == scorecard(_scored_log())  # no clock time in it


# ---------------------------- a daily run, end to end ------------------------------------

def _fake_download(monkeypatch, series, fail=()):
    """forward_test.download, answering from fake price series instead of Yahoo."""

    def fake(ticker, attempts=3, wait=30):
        if ticker in fail:
            raise RuntimeError(f"Yahoo refused {ticker}")
        return series[ticker]
    monkeypatch.setattr(forward_test, "download", fake)
def _evening(day):
    """23:30 UTC on that day: when the scheduled job runs."""

    return pd.Timestamp(f"{day.date()} 23:30", tz="UTC")


def test_two_evenings_in_a_row(tmp_path, monkeypatch, capsys):
    a = make_random_walk(n_days=1500, seed=1)
    b = make_random_walk(n_days=1500, seed=2)
    path = tmp_path / "predictions.csv"
    # evening 1: data up to day 1400
    _fake_download(monkeypatch, {"AAA": a.iloc[:1401], "BBB": b.iloc[:1401]})
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)], now=_evening(a.index[1400])) == 0
    text1 = path.read_text()
    log1 = load_log(path)
    assert len(log1) == 2 and log1["outcome_date"].isna().all()
    # evening 2: one more day; yesterday's outcomes filled, today's predictions added
    _fake_download(monkeypatch, {"AAA": a.iloc[:1402], "BBB": b.iloc[:1402]})
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)], now=_evening(a.index[1401])) == 0
    text2 = path.read_text()
    log2 = load_log(path)
    assert len(log2) == 4
    check_predictions_unchanged(log1, log2)
    check_saved_text(text1, text2)                               # the first evening's rows, character for character
    first = log2.iloc[0]
    assert first["outcome_date"] == str(a.index[1401].date())
    assert first["next_ret"] == pytest.approx(a["Close"].iloc[1401] / a["Close"].iloc[1400] - 1)
    card = (tmp_path / "scorecard.md").read_text()
    assert card.startswith("# Forward test scorecard")
    # evening 3 is a holiday: no new close, so nothing new is added and no file changes
    holiday = _evening(a.index[1401] + pd.offsets.BDay(1))
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)], now=holiday) == 0
    assert path.read_text() == text2
    assert (tmp_path / "scorecard.md").read_text() == card
    out = capsys.readouterr().out
    assert "none new" in out and "not recorded" not in out


def test_a_run_checks_the_file_on_disk_not_what_it_read(tmp_path, monkeypatch):
    """Review issue 43: if reading the log ever changed a digit, the run must refuse to
    save, because it compares against the file's own text."""

    import src.forward
    a = make_random_walk(n_days=1500, seed=1)
    _fake_download(monkeypatch, {"AAA": a.iloc[:1401]})
    assert forward_test.main(["AAA", "--folder", str(tmp_path)], now=_evening(a.index[1400])) == 0
    before = (tmp_path / "predictions.csv").read_text()
    real_load = src.forward.load_log

    def misread(path):
        log = real_load(path)
        log.loc[0, "dir_prob"] = np.nextafter(log.loc[0, "dir_prob"], 1)  # off in the last digit
        return log

    monkeypatch.setattr(src.forward, "load_log", misread)
    _fake_download(monkeypatch, {"AAA": a.iloc[:1402]})
    with pytest.raises(RuntimeError, match="predictions were changed"):
        forward_test.main(["AAA", "--folder", str(tmp_path)], now=_evening(a.index[1401]))
    assert (tmp_path / "predictions.csv").read_text() == before         # nothing was written


def test_a_late_run_records_no_prediction(tmp_path, monkeypatch, capsys):
    """If the evening run failed and is re-run by hand during the next session, the
    prediction from the previous close is refused: its day has already started."""

    a = make_random_walk(n_days=1500, seed=1)
    _fake_download(monkeypatch, {"AAA": a.iloc[:1401]})
    next_day = a.index[1400] + pd.offsets.BDay(1)
    late = pd.Timestamp(f"{next_day.date()} 10:15", tz="America/New_York")
    assert forward_test.main(["AAA", "--folder", str(tmp_path)], now=late) == 0
    assert load_log(tmp_path / "predictions.csv").empty
    assert "not recorded" in capsys.readouterr().out


def test_recorded_too_late_by_hand():
    friday = pd.Timestamp("2026-11-06")
    assert not recorded_too_late(friday, pd.Timestamp("2026-11-06 23:30", tz="UTC"))
    assert not recorded_too_late(friday, pd.Timestamp("2026-11-08 12:00", tz="UTC"))     # the weekend
    assert not recorded_too_late(friday, pd.Timestamp("2026-11-09 09:29", tz="America/New_York"))
    assert recorded_too_late(friday, pd.Timestamp("2026-11-09 09:30", tz="America/New_York"))


def test_a_failed_download_doesnt_stop_the_others(tmp_path, monkeypatch, capsys):
    a = make_random_walk(n_days=1500, seed=1)
    _fake_download(monkeypatch, {"AAA": a}, fail={"BBB"})
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)], now=_evening(a.index[-1])) == 1
    log = load_log(tmp_path / "predictions.csv")
    assert log["ticker"].tolist() == ["AAA"]
    assert "BBB: FAILED" in capsys.readouterr().out


def test_download_tries_again(monkeypatch):
    calls = []

    def flaky(ticker, start, refresh, save=True):
        calls.append((ticker, refresh, save))
        if len(calls) < 3:
            raise RuntimeError("rate limited")
        return "prices"

    import src.data
    monkeypatch.setattr(src.data, "load_prices", flaky)
    assert forward_test.download("SPY", attempts=3, wait=0) == "prices"
    assert calls == [("SPY", True, False)] * 3       # fresh prices, never saved over data/


def test_downloading_without_saving_leaves_data_alone(tmp_path, monkeypatch):
    import sys
    import types
    import src.data
    monkeypatch.setattr(src.data, "DATA_DIR", tmp_path)
    days = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame({c: 1.0 for c in ["Open", "High", "Low", "Close", "Volume"]}, index=days)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: frame))
    src.data.load_prices("SPY", refresh=True, save=False)
    assert list(tmp_path.iterdir()) == []
    src.data.load_prices("SPY", refresh=True)
    assert [f.name for f in tmp_path.iterdir()] == ["SPY_from_2005-01-01.csv"]


def test_workflow_runs_every_weekday_evening_and_can_commit():
    text = (ROOT / ".github" / "workflows" / "forward-test.yml").read_text()
    assert 'cron: "30 23 * * 1-5"' in text
    assert "workflow_dispatch" in text
    assert "contents: write" in text
    assert "python forward_test.py" in text
    assert "git add forward_test/" in text
    assert "git pull --rebase && git push" in text           # never lose an evening to a rejected push
    assert "for attempt in 1 2 3" in text
    assert "date -u" not in text                             # named after the close, not the UTC date


def _prices_with(days, closes, volumes=None):
    p = _prices_on(days, closes)
    if volumes is not None:
        p["Volume"] = volumes
    return p


def test_next_trading_day_refuses_gaps_and_stale_rows():
    made_on = pd.Timestamp("2026-04-02")                          # the Thursday before Good Friday
    ok = _prices_with(["2026-04-02", "2026-04-06"], [100, 101])  # Monday: 4 days, normal
    assert next_trading_day(ok, made_on) == pd.Timestamp("2026-04-06")
    gap = _prices_with(["2026-04-02", "2026-04-09"], [100, 101])  # a week later: a day is missing
    assert next_trading_day(gap, made_on) is None
    stale = _prices_with(["2026-04-02", "2026-04-03"], [100, 100], volumes=[1.0, 0.0])
    assert next_trading_day(stale, made_on) is None
    assert next_trading_day(_prices_with(["2026-04-02"], [100]), made_on) is None
    log, _ = add_predictions(empty_log(), [_row("2026-04-02")])
    assert fill_outcomes(log, {"X": gap})["outcome_date"].isna().all()   # left for a later run


# ---------------------------- the models must not change mid-test ------------------------------------

@pytest.mark.parametrize("make, seed, dir_prob, size_prob", [
    ("random walk", 21, 0.5147846805683128, 0.4394661391504726),
    ("clusters", 22, 0.5132554644863383, 0.457840335752881),
])
def test_forward_test_predictions_are_unchanged(make, seed, dir_prob, size_prob):
    """Recorded on 6 Nov 2026, when the forward test started.

    The forward test is only honest if the models making its predictions stay
    the same throughout. If a later change to the features, the models or the
    forward-test code alters these numbers, this fails: decide deliberately,
    and log it, before changing anything the forward test depends on.
    """

    from src.data import make_volatility_clusters
    prices = make_random_walk(n_days=1500, seed=seed) if make == "random walk" else \
        make_volatility_clusters(n_days=1500, seed=seed)
    row = predict_today(prices, "X", code_version="v", recorded_at="t")
    assert row["dir_prob"] == pytest.approx(dir_prob, abs=1e-6)
    assert row["size_prob"] == pytest.approx(size_prob, abs=1e-6)
    assert row["trained_through"] == "2010-09-30"



def test_forward_test_doesnt_depend_on_the_lstm():
    """The LSTM is optional and separate: the forward test's code must not import it,
    and the models it uses are still exactly the two original ones."""
    
    from src.evaluate import get_models
    for name in ["forward.py", "evaluate.py", "features.py", "data.py"]:
        assert "lstm" not in (ROOT / "src" / name).read_text().lower()
    assert "lstm" not in (ROOT / "forward_test.py").read_text().lower()
    assert list(get_models()) == ["Logistic regression", "Gradient boosting"]