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
                         check_predictions_unchanged, empty_log, fill_outcomes, load_log,
                         predict_today, save_log, score_rows, scorecard)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def prices():
    return make_random_walk(n_days=1500, seed=11)


# ---------------------------- predictions ------------------------------------

def test_prediction_uses_only_the_last_close_and_earlier(prices):
    """A prediction made from day k must be the same whatever happens after day k."""

    k = 1200
    made = predict_today(prices.iloc[:k + 1], "X", recorded_at="t")
    future = prices.copy()
    rng = np.random.default_rng(5)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        future.iloc[k + 1:, future.columns.get_loc(col)] *= rng.uniform(0.5, 1.5, len(future) - k - 1)
    again = predict_today(future.iloc[:k + 1], "X", recorded_at="t")
    assert made == again
    assert made["made_on"] == str(prices.index[k].date())


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


def test_log_round_trip(tmp_path):
    log, _ = add_predictions(empty_log(), [_row("2026-11-05")])
    path = tmp_path / "predictions.csv"
    save_log(log, path)
    back = load_log(path)
    assert list(back.columns) == COLUMNS
    check_predictions_unchanged(log, back)            # nothing changes on the way through a CSV
    assert load_log(tmp_path / "missing.csv").empty


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
    assert "No predictions yet" in scorecard(empty_log(), updated_at="now")
    log, _ = add_predictions(empty_log(), [_row("2026-11-06")])
    text = scorecard(log, updated_at="now")
    assert "No outcomes yet" in text and "1 still waiting" in text


def test_scorecard_with_outcomes_has_both_tasks_and_a_warning():
    text = scorecard(_scored_log(), updated_at="now")
    assert "## Direction" in text and "## Move size" in text
    assert "How much to read into this" in text


# ---------------------------- a daily run, end to end ------------------------------------

def _fake_download(monkeypatch, series, fail=()):
    """forward_test.download, answering from fake price series instead of Yahoo."""

    def fake(ticker, attempts=3, wait=30):
        if ticker in fail:
            raise RuntimeError(f"Yahoo refused {ticker}")
        return series[ticker]
    monkeypatch.setattr(forward_test, "download", fake)


def test_two_evenings_in_a_row(tmp_path, monkeypatch, capsys):
    a = make_random_walk(n_days=1500, seed=1)
    b = make_random_walk(n_days=1500, seed=2)
    # evening 1: data up to day 1400
    _fake_download(monkeypatch, {"AAA": a.iloc[:1401], "BBB": b.iloc[:1401]})
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)]) == 0
    log1 = load_log(tmp_path / "predictions.csv")
    assert len(log1) == 2 and log1["outcome_date"].isna().all()
    # evening 2: one more day; yesterday's outcomes filled, today's predictions added
    _fake_download(monkeypatch, {"AAA": a.iloc[:1402], "BBB": b.iloc[:1402]})
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)]) == 0
    log2 = load_log(tmp_path / "predictions.csv")
    assert len(log2) == 4
    check_predictions_unchanged(log1, log2)
    first = log2.iloc[0]
    assert first["outcome_date"] == str(a.index[1401].date())
    assert first["next_ret"] == pytest.approx(a["Close"].iloc[1401] / a["Close"].iloc[1400] - 1)
    assert (tmp_path / "scorecard.md").read_text().startswith("# Forward test scorecard")
    # evening 3 is a holiday: no new close, so nothing new is added
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)]) == 0
    assert len(load_log(tmp_path / "predictions.csv")) == 4
    assert "none new" in capsys.readouterr().out


def test_a_failed_download_doesnt_stop_the_others(tmp_path, monkeypatch, capsys):
    a = make_random_walk(n_days=1500, seed=1)
    _fake_download(monkeypatch, {"AAA": a}, fail={"BBB"})
    assert forward_test.main(["AAA", "BBB", "--folder", str(tmp_path)]) == 1
    log = load_log(tmp_path / "predictions.csv")
    assert log["ticker"].tolist() == ["AAA"]
    assert "BBB: FAILED" in capsys.readouterr().out


def test_download_tries_again(monkeypatch):
    calls = []

    def flaky(ticker, start, refresh):
        calls.append(ticker)
        if len(calls) < 3:
            raise RuntimeError("rate limited")
        return "prices"

    import src.data
    monkeypatch.setattr(src.data, "load_prices", flaky)
    assert forward_test.download("SPY", attempts=3, wait=0) == "prices"
    assert len(calls) == 3


def test_workflow_runs_every_weekday_evening_and_can_commit():
    text = (ROOT / ".github" / "workflows" / "forward-test.yml").read_text()
    assert 'cron: "30 23 * * 1-5"' in text
    assert "workflow_dispatch" in text
    assert "contents: write" in text
    assert "python forward_test.py" in text
    assert "git add forward_test/" in text


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