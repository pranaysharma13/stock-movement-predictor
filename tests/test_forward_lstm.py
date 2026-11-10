"""
Tests for phase 6.1, the LSTM in the live forward test: it trains only on days
whose label is known, saves and reloads its models exactly, retrains once a
month, keeps its log append-only down to the last digit, and scores itself
against Recent swing and logistic regression. Tests that train a network are
skipped without PyTorch; the networks are tiny, so they stay fast.
"""


import numpy as np
import pandas as pd
import pytest

import forward_test
import forward_test_lstm
from src import forward_lstm as fl
from src.data import make_volatility_clusters
from src.evaluate import edge_half_width, paired_p_value
from src.lstm import _probs, fit_scaler, make_sequences, scale, sequence_inputs, torch_available

needs_torch = pytest.mark.skipif(not torch_available(), reason="needs PyTorch (pip install -r requirements-lstm.txt)")

FAST = dict(window=10, epochs=3, hidden=4, seeds=(0, 1), batch=256, validation_days=60)


@pytest.fixture(scope="module")
def prices():
    return make_volatility_clusters(n_days=1500, seed=7)


def _evening(day):
    """23:17 UTC on that day: when the scheduled job runs."""

    return pd.Timestamp(f"{day.date()} 23:17", tz="UTC")


def _fake_download(monkeypatch, series):
    monkeypatch.setattr(forward_test, "download", lambda ticker, attempts=3, wait=30: series[ticker])


# ---------------------------- training ------------------------------------

def test_live_split_by_hand():
    fit_idx, val_idx = fl.live_split(10, validation_days=3)
    assert list(val_idx) == [7, 8, 9]                   # the last 3 labelled days
    assert list(fit_idx) == [0, 1, 2, 3, 4, 5]          # day 6 left out: its label is day 7's move
    with pytest.raises(ValueError, match="Need more than"):
        fl.live_split(4, validation_days=3)


@needs_torch
def test_training_uses_only_labelled_days(prices):
    """Trained up to the day before the prediction day, whose label (the prediction
    day's move) is known; the prediction day's own inputs are never trained on."""

    upto = prices.iloc[:1201]
    model = fl.train_live(upto, **FAST)
    assert model["trained through"] == str(prices.index[1199].date())
    assert model["month"] == prices.index[1200].strftime("%Y-%m")
    mean, std = fit_scaler(sequence_inputs(upto), prices.index[1199])
    np.testing.assert_allclose(model["mean"], mean, rtol=1e-6)
    np.testing.assert_allclose(model["std"], std, rtol=1e-6)
    assert len(model["states"]) == 2 and len(model["best epochs"]) == 2
    # change the prediction day's high, low and volume: the trained networks don't change
    changed = upto.copy()
    changed.iloc[-1, changed.columns.get_loc("High")] *= 1.3
    changed.iloc[-1, changed.columns.get_loc("Low")] *= 0.7
    changed.iloc[-1, changed.columns.get_loc("Volume")] *= 5
    again = fl.train_live(changed, **FAST)
    for a, b in zip(model["states"], again["states"]):
        for k in a:
            assert (a[k] == b[k]).all()


@needs_torch
def test_a_saved_model_predicts_exactly_like_the_trained_one(prices, tmp_path):
    upto = prices.iloc[:1201]
    model = fl.train_live(upto, **FAST)
    path = tmp_path / "AAA.pt"
    fl.save_model(model, path)
    loaded = fl.load_model(path)
    assert fl.load_model(tmp_path / "missing.pt") is None
    a = fl.predict_lstm_today(upto, "AAA", model, recorded_at="t")
    b = fl.predict_lstm_today(upto, "AAA", loaded, recorded_at="t")
    assert a == b
    assert a["made_on"] == str(prices.index[1200].date())
    assert a["trained_through"] == str(prices.index[1199].date())
    assert a["size_pred"] == int(a["size_prob"] > 0.5)
    X, _ = make_sequences(sequence_inputs(upto), upto.index[-1:], FAST["window"])
    X = scale(X, (np.array(model["mean"], dtype=np.float32), np.array(model["std"], dtype=np.float32)))
    each = [float(_probs(net, X)[0]) for net in fl._networks(model)]
    assert a["size_prob"] == pytest.approx(np.mean(each))       # the seeds' average
    assert a["seeds_agree"] in (0, 1)
    # a model trained on the prediction day itself (or later) is refused
    with pytest.raises(ValueError, match="trained on"):
        fl.predict_lstm_today(prices.iloc[:1200], "AAA", model)


def test_retraining_happens_once_a_month():
    model = {"month": "2026-11"}
    assert fl.needs_training(None, pd.Timestamp("2026-11-09"))
    assert not fl.needs_training(model, pd.Timestamp("2026-11-30"))
    assert fl.needs_training(model, pd.Timestamp("2026-12-01"))


# ---------------------------- evenings in a row ------------------------------------

@needs_torch
def test_evenings_in_a_row_with_a_new_month(prices, tmp_path, monkeypatch, capsys):
    folder = tmp_path
    log_path, model_path = folder / "lstm_predictions.csv", folder / "lstm_models" / "AAA.pt"
    days = prices.index
    first = 1200
    new_month = next(i for i in range(first + 2, len(days)) if days[i].month != days[first].month)

    # evening 1: no model yet, so it trains one
    _fake_download(monkeypatch, {"AAA": prices.iloc[:first + 1]})
    assert forward_test_lstm.main(["AAA", "--folder", str(folder)], now=_evening(days[first]), settings=FAST) == 0
    model_bytes = model_path.read_bytes()
    text1 = log_path.read_text()
    log1 = fl.load_log(log_path)
    assert len(log1) == 1 and log1.loc[0, "model_month"] == days[first].strftime("%Y-%m")
    assert "training this month's LSTM" in capsys.readouterr().out

    # evening 2, same month: the saved model is used, yesterday's outcome is filled in
    _fake_download(monkeypatch, {"AAA": prices.iloc[:first + 2]})
    assert forward_test_lstm.main(["AAA", "--folder", str(folder)], now=_evening(days[first + 1]), settings=FAST) == 0
    assert model_path.read_bytes() == model_bytes
    assert "training" not in capsys.readouterr().out
    text2 = log_path.read_text()
    fl.check_saved_text(text1, text2)                       # the first row, character for character
    log2 = fl.load_log(log_path)
    ret = prices["Close"].iloc[first + 1] / prices["Close"].iloc[first] - 1
    assert log2.loc[0, "outcome_date"] == str(days[first + 1].date())
    assert log2.loc[0, "next_ret"] == pytest.approx(ret)
    assert log2.loc[0, "big"] == int(abs(ret) > log2.loc[0, "typical_move"])
    assert "No outcomes yet" not in (folder / "lstm_scorecard.md").read_text()

    # the same evening again: nothing new, no file changes
    card = (folder / "lstm_scorecard.md").read_text()
    assert forward_test_lstm.main(["AAA", "--folder", str(folder)], now=_evening(days[first + 1]), settings=FAST) == 0
    assert log_path.read_text() == text2 and (folder / "lstm_scorecard.md").read_text() == card
    assert "none new" in capsys.readouterr().out

    # the first evening of a new month: retrained on everything up to that close
    _fake_download(monkeypatch, {"AAA": prices.iloc[:new_month + 1]})
    assert forward_test_lstm.main(["AAA", "--folder", str(folder)], now=_evening(days[new_month]), settings=FAST) == 0
    model = fl.load_model(model_path)
    assert model["month"] == days[new_month].strftime("%Y-%m")
    assert model["trained through"] == str(days[new_month - 1].date())
    assert fl.load_log(log_path)["model_month"].iloc[-1] == model["month"]


@needs_torch
def test_a_run_checks_the_file_on_disk_not_what_it_read(prices, tmp_path, monkeypatch):
    """If reading the log ever changed a digit, the run refuses to save (issue 43)."""

    _fake_download(monkeypatch, {"AAA": prices.iloc[:1201]})
    assert forward_test_lstm.main(["AAA", "--folder", str(tmp_path)], now=_evening(prices.index[1200]), settings=FAST) == 0
    before = (tmp_path / "lstm_predictions.csv").read_text()
    real_load = fl.load_log

    def misread(path):
        log = real_load(path)
        log.loc[0, "size_prob"] = np.nextafter(log.loc[0, "size_prob"], 1)
        return log

    monkeypatch.setattr(fl, "load_log", misread)
    _fake_download(monkeypatch, {"AAA": prices.iloc[:1202]})
    with pytest.raises(RuntimeError, match="predictions were changed"):
        forward_test_lstm.main(["AAA", "--folder", str(tmp_path)], now=_evening(prices.index[1201]), settings=FAST)
    assert (tmp_path / "lstm_predictions.csv").read_text() == before


@needs_torch
def test_a_late_run_records_nothing_and_trains_nothing(prices, tmp_path, monkeypatch, capsys):
    _fake_download(monkeypatch, {"AAA": prices.iloc[:1201]})
    next_day = prices.index[1200] + pd.offsets.BDay(1)
    late = pd.Timestamp(f"{next_day.date()} 10:15", tz="America/New_York")
    assert forward_test_lstm.main(["AAA", "--folder", str(tmp_path)], now=late, settings=FAST) == 0
    assert fl.load_log(tmp_path / "lstm_predictions.csv").empty
    assert not (tmp_path / "lstm_models" / "AAA.pt").exists()
    assert "not recorded" in capsys.readouterr().out


# ---------------------------- the log ------------------------------------

def _row(made_on, ticker="AAA", **kw):
    r = {"made_on": made_on, "ticker": ticker, "close": 100.0, "typical_move": 0.004943305862969205,
         "trained_through": "2026-11-05", "model_month": "2026-11", "size_prob": 0.49568276435249065,
         "size_pred": 0, "seeds_agree": 1, "recent_swing": 1, "code_version": "abc1234", "recorded_at": "t"}
    r.update(kw)
    return r


def test_log_round_trip_and_text_check(tmp_path):
    log, added = fl.add_predictions(fl.empty_log(), [_row("2026-11-06")])
    assert added == 1
    path = tmp_path / "lstm_predictions.csv"
    first = fl.save_log(log, path)
    back = fl.load_log(path)
    assert back.loc[0, "size_prob"] == 0.49568276435249065       # exactly
    assert fl.save_log(back, path, before_text=first) == first
    assert fl.add_predictions(back, [_row("2026-11-06")])[1] == 0  # no duplicates
    with pytest.raises(RuntimeError, match="predictions were changed"):
        fl.check_saved_text(first, first.replace("0.49568276435249065", "0.4956827643524906"))
    filled = back.copy()
    filled.loc[0, fl.OUTCOME_COLUMNS] = ["2026-11-09", 0.01, 1]
    fl.check_saved_text(first, fl.log_text(filled))             # an outcome may be filled in
    changed = filled.copy()
    changed.loc[0, "big"] = 0
    with pytest.raises(RuntimeError, match="outcomes were changed"):
        fl.check_saved_text(fl.log_text(filled), fl.log_text(changed))


def test_a_filled_outcome_is_never_refilled():
    """Yahoo can revise old prices (a dividend adjustment); a recorded outcome stays as it was."""

    log, _ = fl.add_predictions(fl.empty_log(), [_row("2026-11-05")])
    days = pd.to_datetime(["2026-11-05", "2026-11-06"])
    first = pd.DataFrame({"Close": [100.0, 101.0], "Volume": 1.0}, index=days)
    revised = pd.DataFrame({"Close": [100.0, 99.0], "Volume": 1.0}, index=days)
    filled = fl.fill_outcomes(log, {"AAA": first})
    assert filled.loc[0, "next_ret"] == pytest.approx(0.01)
    again = fl.fill_outcomes(filled, {"AAA": revised})
    assert again.loc[0, "next_ret"] == pytest.approx(0.01) and again.loc[0, "big"] == 1


# ---------------------------- the scorecard ------------------------------------

def _scored_logs():
    """20 scored days for each of 4 stocks, with logistic regression's log beside it."""

    rows, lr_rows = [], []
    for ticker in ["SPY", "NVDA", "AAPL", "GOOGL"]:
        for k, d in enumerate(pd.bdate_range("2026-01-05", periods=20)):
            big = k % 2
            r = _row(str(d.date()), ticker, size_pred=big if k % 5 else 1 - big, recent_swing=k % 3 == 0)
            r.update({"outcome_date": "x", "next_ret": 0.0, "big": big})
            rows.append(r)
            lr_rows.append({"made_on": str(d.date()), "ticker": ticker, "size_pred": int(k % 4 == 0)})
    return pd.DataFrame(rows).reindex(columns=fl.COLUMNS), pd.DataFrame(lr_rows)


def test_score_rows_by_hand():
    log, lr = _scored_logs()
    table = fl.score_rows(log, lr).set_index("stock")
    spy = log[log["ticker"] == "SPY"]
    target, pred, rule = spy["big"].astype(int), spy["size_pred"].astype(int), spy["recent_swing"].astype(int)
    assert table.loc["SPY", "LSTM"] == pytest.approx((pred == target).mean())                # 16 of 20
    assert table.loc["SPY", "Recent swing"] == pytest.approx((rule == target).mean())
    wins, losses = int(((pred == target) & (rule != target)).sum()), int(((pred != target) & (rule == target)).sum())
    assert table.loc["SPY", "edge ±"] == pytest.approx(edge_half_width(wins, losses, 20))
    assert table.loc["SPY", "p-value vs Recent swing"] == pytest.approx(paired_p_value(pred, rule, target))
    lr_pred = lr[lr["ticker"] == "SPY"]["size_pred"].to_numpy()
    assert table.loc["SPY", "logistic regression"] == pytest.approx((lr_pred == target.to_numpy()).mean())
    assert table.loc["All four", "days scored"] == 80
    assert list(table.index) == ["SPY", "NVDA", "AAPL", "GOOGL", "All four"]


def test_scorecard_reads_per_stock_and_only_changes_with_the_logs():
    log, lr = _scored_logs()
    models = {"SPY": {"month": "2026-11", "trained through": "2026-11-05", "best epochs": [4, 6, 5]}}
    text = fl.scorecard(log, lr, models)
    assert "Each stock has at most 20 scored days" in text and "about ±22 points" in text
    assert "## Against logistic regression, on the same days" in text
    assert "| SPY | 2026-11 | 2026-11-05 | 4, 6, 5 |" in text
    assert text == fl.scorecard(log, lr, models)              # no clock time in it
    assert "No predictions yet" in fl.scorecard(fl.empty_log())
    waiting, _ = fl.add_predictions(fl.empty_log(), [_row("2026-11-06")])
    assert "No outcomes yet" in fl.scorecard(waiting)


# ---------------------------- the schedule ------------------------------------

def test_workflow_runs_the_lstm_after_logistic_regression_and_apart_from_it():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "forward-test.yml").read_text()
    lr_step, lstm_step = text.index("python forward_test.py"), text.index("python forward_test_lstm.py")
    assert lr_step < lstm_step
    for step in ["name: Install PyTorch for the LSTM", "name: Run the LSTM forward test"]:
        start = text.index(step)
        block = text[start:text.index("- name:", start)]
        assert "continue-on-error: true" in block              # a failing LSTM can't stop the save
    assert "steps.torch.outcome == 'success'" in text
    assert "download.pytorch.org/whl/cpu" in text and "requirements-lstm.txt" in text
    save = text.index("name: Save the results")
    assert save > lstm_step and "if: always()" in text[save:save + 200]


def test_logistic_regressions_forward_test_still_ignores_the_lstm():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    assert "lstm" not in (root / "forward_test.py").read_text().lower()
    assert "lstm" not in (root / "src" / "forward.py").read_text().lower()


def test_scorecard_says_day_not_days_for_one():
    log, _ = fl.add_predictions(fl.empty_log(), [_row("2026-11-06")])
    log.loc[0, fl.OUTCOME_COLUMNS] = ["2026-11-09", 0.02, 1]
    text = fl.scorecard(log)
    assert "at most 1 scored day," in text and "1 scored days" not in text

