"""
Tests for phase 3: per-stock summaries, the comparison with its
lucky-winners correction, run.py's options, and the notebook templates.
"""


import json
from pathlib import Path

import nbformat
import pandas as pd
import pytest

import run
from src.data import make_random_walk
from src.evaluate import get_models, walk_forward
from src.features import build_dataset
from src.evaluate import paired_p_value, vol_baseline_predictions
from src.features import VOL_FEATURES, build_vol_dataset
from src.report import (comparison_table, half_on_big, holm, load_summaries, lstm_comparison,
                        lstm_vs_logreg, safe_name, save_summary, summarize, summarize_vol)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def noise_summary():
    data = build_dataset(make_random_walk(n_days=2000, seed=5))
    test = data[data.index.year >= 2010]
    preds = {name: walk_forward(m, data, 2010) for name, m in get_models().items()}
    return summarize("FAKE", test, preds, cost=0.001, luck_trials=100)


def _summary(ticker, p_values, luck_shares):
    """A hand-made summary: one model per p-value."""

    return {
        "ticker": ticker, "test start": "2010-01-04", "test end": "2020-12-31", "test days": 2770,
        "cost": 0.001, "always up accuracy": 0.53,
        "buy and hold": {"annual return": 0.10, "Sharpe ratio": 0.6, "max drawdown": -0.3},
        "models": {
            f"model {i}": {"accuracy": 0.54, "predicted up": 0.9, "p-value vs always up": p,
                           "wins vs always up": 100, "losses vs always up": 72,
                           "annual return": 0.09, "Sharpe ratio": 0.5, "max drawdown": -0.3,
                           "cash days": 100, "share of random >=": luck}
            for i, (p, luck) in enumerate(zip(p_values, luck_shares))
        },
    }


# ---------------------------- summaries ------------------------------------

def test_summary_has_the_headline_numbers(noise_summary):
    s = noise_summary
    assert s["ticker"] == "FAKE"
    assert s["test start"].startswith("2010")
    assert set(s["models"]) == {"Logistic regression", "Gradient boosting"}
    for m in s["models"].values():
        assert 0 <= m["accuracy"] <= 1
        assert 0 <= m["p-value vs always up"] <= 1
        assert m["share of random >="] is None or 0 <= m["share of random >="] <= 1
    json.dumps(s)  # must be saveable as JSON


def test_summary_accuracy_matches_a_direct_count():
    data = build_dataset(make_random_walk(n_days=1800, seed=6))
    test = data[data.index.year >= 2010]
    preds = {"Always up": pd.DataFrame({"pred": 1}, index=test.index)}
    s = summarize("X", test, preds, luck_trials=10)
    assert s["models"]["Always up"]["accuracy"] == pytest.approx(test["target"].mean())
    assert s["models"]["Always up"]["share of random >="] is None   # never in cash


def test_save_and_load_round_trip(tmp_path, noise_summary):
    path = save_summary(noise_summary, folder=tmp_path)
    assert path.name == "FAKE.json"
    assert load_summaries(["FAKE"], folder=tmp_path) == [json.loads(json.dumps(noise_summary))]


def test_missing_summary_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="python run.py NVDA"):
        load_summaries(["NVDA"], folder=tmp_path)


def test_safe_name():
    assert safe_name("NVDA") == "NVDA"
    assert safe_name("BRK-B") == "BRK-B"
    assert safe_name("^GSPC") == "_GSPC"


# ---------------------------- comparison ------------------------------------

def test_corrected_bar_is_alpha_divided_by_comparisons():
    summaries = [_summary("A", [0.01, 0.30], [0.03, 0.5]), _summary("B", [0.004, 0.9], [0.001, None])]
    table, info = comparison_table(summaries)
    assert info["comparisons"] == 4
    assert info["corrected threshold"] == pytest.approx(0.05 / 4)
    assert info["expected chance passes"] == pytest.approx(0.2)
    # p = 0.01 and 0.004 are below both 0.05 and the corrected 0.0125; 0.30 and 0.9 pass neither
    assert table.loc[("A", "model 0"), "accuracy edge (5%)?"]
    assert table.loc[("A", "model 0"), "accuracy edge (corrected)?"]
    assert not table.loc[("A", "model 1"), "accuracy edge (5%)?"]
    assert info["accuracy passes (5%)"] == 2
    assert info["accuracy passes (corrected)"] == 2
    # luck: the model that never left the market (None) has no test and isn't counted,
    # so 3 tests. Holm's bars 0.0167, 0.025, 0.05: 0.001 passes, 0.03 passes at 5% only
    assert info["luck comparisons"] == 3
    assert table.loc[("A", "model 0"), "beats luck (5%)?"]
    assert not table.loc[("A", "model 0"), "beats luck (corrected)?"]
    assert table.loc[("B", "model 0"), "beats luck (corrected)?"]
    assert not table.loc[("B", "model 1"), "beats luck (5%)?"]


def test_a_borderline_win_fails_once_many_things_are_tested():
    """p = 0.03 passes as a single test, but not among 8 comparisons."""

    summaries = [_summary(t, [0.03, 0.5], [0.5, 0.5]) for t in ["A", "B", "C", "D"]]
    table, info = comparison_table(summaries)
    assert info["comparisons"] == 8
    assert info["accuracy passes (5%)"] == 4
    assert info["accuracy passes (corrected)"] == 0


def test_edge_is_accuracy_minus_always_up():
    table, _ = comparison_table([_summary("A", [0.5], [0.5])])
    assert table.loc[("A", "model 0"), "edge"] == pytest.approx(0.54 - 0.53)


# ---------------------------- run.py ------------------------------------

def test_run_options_defaults():
    args = run.parse_args(["spy", "nvda"])
    assert args.tickers == ["SPY", "NVDA"]
    assert args.start == "2005-01-01"
    assert args.first_test_year is None
    assert args.cost == 0.001
    assert not args.refresh and not args.no_comparison


def test_run_options_all_set():
    args = run.parse_args(["AAPL", "aapl", "GOOGL", "--refresh", "--cost", "0.002",
                           "--start", "2010-01-01", "--first-test-year", "2015", "--no-comparison"])
    assert args.tickers == ["AAPL", "GOOGL"]   # duplicates removed, order kept
    assert args.refresh and args.no_comparison
    assert args.cost == 0.002 and args.start == "2010-01-01" and args.first_test_year == 2015


def test_run_rejects_negative_cost():
    with pytest.raises(SystemExit):
        run.parse_args(["SPY", "--cost", "-0.1"])


def test_run_passes_the_interest_setting(monkeypatch):
    seen = []
    monkeypatch.setattr(run, "run_notebook", lambda template, output, parameters: seen.append(parameters))
    run.main(["SPY", "--no-comparison"])
    run.main(["SPY", "--no-comparison", "--no-interest"])
    assert [p["CASH_INTEREST"] for p in seen] == [True, False]


def test_run_keeps_going_when_one_stock_fails(monkeypatch, capsys):
    """A bad ticker is reported, the others still run, and the exit code says something failed."""

    ran = []

    def fake_run_notebook(template, output, parameters):
        if "TICKER" in parameters and parameters["TICKER"] == "BAD":
            raise RuntimeError("No data came back for 'BAD'.")
        ran.append(parameters)

    monkeypatch.setattr(run, "run_notebook", fake_run_notebook)
    code = run.main(["SPY", "BAD", "NVDA"])
    out = capsys.readouterr().out
    assert code == 1
    assert "BAD: FAILED" in out
    assert [p.get("TICKER") for p in ran] == ["SPY", "NVDA", None]
    assert ran[-1] == {"TICKERS": ["SPY", "NVDA"]}            # comparison of the ones that worked


# ---------------------------- templates ------------------------------------

@pytest.mark.parametrize("name, names", [
    ("template.ipynb", ["TICKER", "START", "FIRST_TEST_YEAR", "COST", "CASH_INTEREST", "RUN_LSTM"]),
    ("comparison_template.ipynb", ["TICKERS"]),
])
def test_template_has_a_parameters_cell(name, names):
    nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    tagged = [c for c in nb.cells if "parameters" in c.metadata.get("tags", [])]
    assert len(tagged) == 1, "papermill needs exactly one cell tagged 'parameters'"
    namespace = {}
    exec(tagged[0].source, namespace)
    assert all(n in namespace for n in names)


def test_templates_are_saved_without_outputs():
    for name in ["template.ipynb", "comparison_template.ipynb"]:
        nb = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


# ---------------------------- Part 2: move size ------------------------------------

def _vol_block(p_values, luck_shares, weekly_luck_shares=None):
    weekly_luck_shares = weekly_luck_shares or [0.5] * len(p_values)
    strat = {"annual return": 0.08, "Sharpe ratio": 0.7, "max drawdown": -0.2}
    return {
        "test start": "2010-01-04", "test end": "2020-12-31", "test days": 2770,
        "share of big days": 0.5, "majority accuracy": 0.51, "same as today accuracy": 0.53,
        "recent swing accuracy": 0.59,
        "buy and hold": {"annual return": 0.10, "Sharpe ratio": 0.8, "max drawdown": -0.3},
        "same as today strategy": {"annual return": 0.05, "Sharpe ratio": 0.4, "max drawdown": -0.25},
        "recent swing strategy": {"strategy": strat, "strategy (weekly)": strat},
        "models": {
            f"model {i}": {"accuracy": 0.60, "predicted big": 0.4,
                           "p-value vs recent swing": p, "p-value vs same as today": 0.0001,
                           "wins vs recent swing": 300, "losses vs recent swing": 272,
                           "wins vs same as today": 500, "losses vs same as today": 306,
                           "strategy": strat, "strategy (weekly)": strat,
                           "share of random >=": luck, "share of random >= (weekly)": wluck}
            for i, (p, luck, wluck) in enumerate(zip(p_values, luck_shares, weekly_luck_shares))
        },
    }


def test_vol_summary_has_the_headline_numbers():
    data = build_vol_dataset(make_random_walk(n_days=2200, seed=8))
    test = data[data.index.year >= 2010]
    preds = {name: walk_forward(m, data, 2010, features=VOL_FEATURES) for name, m in get_models().items()}
    baselines = vol_baseline_predictions(test)
    v = summarize_vol(test, preds, baselines, luck_trials=20)
    # each p-value is the paired test against the rule it names
    for name, m in v["models"].items():
        for rule in ["Recent swing", "Same as today"]:
            expected = paired_p_value(preds[name]["pred"], baselines[rule], test["target"])
            assert m[f"p-value vs {rule.lower()}"] == pytest.approx(expected)
    assert v["recent swing accuracy"] == pytest.approx((baselines["Recent swing"] == test["target"]).mean())
    assert set(v["models"]) == {"Logistic regression", "Gradient boosting"}
    assert 0 <= v["same as today accuracy"] <= 1
    assert 0 <= v["recent swing accuracy"] <= 1
    assert v["majority accuracy"] >= 0.5
    assert set(v["recent swing strategy"]) == {"strategy", "strategy (weekly)"}
    for m in v["models"].values():
        for key in ["strategy", "strategy (weekly)"]:
            assert set(m[key]) == {"annual return", "Sharpe ratio", "max drawdown"}
        for key in ["p-value vs recent swing", "p-value vs same as today",
                    "share of random >=", "share of random >= (weekly)"]:
            assert 0 <= m[key] <= 1
    json.dumps(v)


def test_half_on_big():
    assert list(half_on_big(pd.Series([0, 1, 1, 0]))) == [1.0, 0.5, 0.5, 1.0]


def test_volatility_comparison_is_corrected_on_its_own():
    s = _summary("A", [0.5, 0.5], [0.5, 0.5])
    s["volatility"] = _vol_block([0.001, 0.02], [0.001, 0.9], [0.02, 0.011])
    table, info = comparison_table([s], task="volatility")
    # accuracy: 2 comparisons (one per model) against Recent swing
    assert info["comparisons"] == 2
    assert info["corrected threshold"] == pytest.approx(0.025)
    assert table.loc[("A", "model 0"), "accuracy edge (corrected)?"]
    assert table.loc[("A", "model 1"), "accuracy edge (corrected)?"]      # 0.02 < 0.025
    assert table.loc[("A", "model 0"), "edge"] == pytest.approx(0.60 - 0.59)
    assert table.loc[("A", "model 0"), "edge over same as today"] == pytest.approx(0.60 - 0.53)
    assert info["same as today passes (corrected)"] == 2
    # luck: daily and weekly for each model = 4 comparisons, bar 0.05 / 4
    assert info["luck comparisons"] == 4
    assert info["luck corrected threshold"] == pytest.approx(0.0125)
    # Holm over the four luck shares 0.001, 0.011, 0.02, 0.9: bars 0.0125, 0.0167, 0.025, 0.05.
    # The first three pass; Bonferroni (0.0125 for all) would have passed only the first.
    assert table.loc[("A", "model 0"), "beats luck, daily (corrected)?"]      # 0.001
    assert table.loc[("A", "model 1"), "beats luck, weekly (corrected)?"]     # 0.011
    assert table.loc[("A", "model 0"), "beats luck, weekly (corrected)?"]     # 0.02
    assert not table.loc[("A", "model 1"), "beats luck, daily (corrected)?"]  # 0.9
    assert info["luck passes (5%)"] == 3
    assert info["luck passes (corrected)"] == 3


def test_direction_comparison_counts_one_luck_test_per_row():
    table, info = comparison_table([_summary("A", [0.5, 0.5], [0.02, 0.5])])
    assert info["luck comparisons"] == 2
    assert info["luck corrected threshold"] == pytest.approx(0.025)
    assert table.loc[("A", "model 0"), "beats luck (corrected)?"]


def test_summaries_from_before_recent_swing_say_what_to_do():
    s = _summary("OLD", [0.5], [0.5])
    s["volatility"] = _vol_block([0.5], [0.5])
    del s["volatility"]["recent swing accuracy"]
    with pytest.raises(ValueError, match="python run.py OLD"):
        comparison_table([s], task="volatility")


def test_old_summaries_without_move_size_say_what_to_do():
    with pytest.raises(ValueError, match="python run.py OLD"):
        comparison_table([_summary("OLD", [0.5], [0.5])], task="volatility")


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="task must be"):
        comparison_table([_summary("A", [0.5], [0.5])], task="price")



# ---------------------------- phase 4.2 ------------------------------------

def test_holm_by_hand():
    # Bonferroni's bar for 4 tests is 0.0125: only 0.01 passes. Holm's bars are
    # 0.0125, 0.0167, 0.025, 0.05 in turn: 0.01 and 0.015 pass, 0.2 stops it.
    assert list(holm([0.2, 0.015, 0.3, 0.01])) == [False, True, False, True]
    # missing values count as tests and never pass
    assert list(holm([float("nan"), 0.001])) == [False, True]


def test_holm_stops_at_the_first_miss():
    # sorted 0.03, 0.04: the first bar is 0.025, so nothing passes
    assert list(holm([0.04, 0.03], alpha=0.05)) == [False, False]


def test_summary_p_value_is_the_paired_test_against_always_up():
    data = build_dataset(make_random_walk(n_days=1800, seed=6))
    test = data[data.index.year >= 2010]
    pred = (test["ret_1d"] > 0).astype(int)
    s = summarize("X", test, {"m": pd.DataFrame({"pred": pred}, index=test.index)}, luck_trials=10)
    expected = paired_p_value(pred, pd.Series(1, index=test.index), test["target"])
    assert s["models"]["m"]["p-value vs always up"] == pytest.approx(expected)


def test_summary_paired_counts_match_the_accuracies(noise_summary):
    for m in noise_summary["models"].values():
        n = noise_summary["test days"]
        assert (m["wins vs always up"] - m["losses vs always up"]) / n == pytest.approx(
            m["accuracy"] - noise_summary["always up accuracy"])


def test_summary_uses_interest_on_cash():
    data = build_dataset(make_random_walk(n_days=1800, seed=6))
    test = data[data.index.year >= 2010]
    preds = {"Half the time": pd.DataFrame({"pred": (test["ret_1d"] > 0).astype(int)}, index=test.index)}
    plain = summarize("X", test, preds, luck_trials=10)
    with_cash = summarize("X", test.assign(cash_ret=0.0002), preds, luck_trials=10)
    assert with_cash["average cash interest"] == pytest.approx(0.0002 * 252)
    assert with_cash["models"]["Half the time"]["annual return"] > plain["models"]["Half the time"]["annual return"]
    assert with_cash["buy and hold"]["Sharpe ratio"] < plain["buy and hold"]["Sharpe ratio"]


def test_comparison_ranges_come_from_the_paired_counts():
    table, _ = comparison_table([_summary("A", [0.5], [0.5])])
    from src.evaluate import edge_half_width
    assert table.loc[("A", "model 0"), "edge ±"] == pytest.approx(edge_half_width(100, 72, 2770))


def test_summaries_from_before_phase_4_2_say_what_to_do():
    s = _summary("OLD", [0.5], [0.5])
    del s["models"]["model 0"]["wins vs always up"]
    with pytest.raises(ValueError, match="python run.py OLD"):
        comparison_table([s])


def _fake_papermill(monkeypatch, fail):
    """A stand-in for papermill that writes its output file, then fails if asked."""
    import sys
    import types

    def execute_notebook(template, output, **kwargs):
        Path(output).write_text("partly run" if fail else "new report")
        if fail:
            raise RuntimeError("cell 12 failed\nKeyError: 'Close'")

    fake = types.ModuleType("papermill")
    fake.execute_notebook = execute_notebook
    monkeypatch.setitem(sys.modules, "papermill", fake)


def test_a_failed_run_keeps_the_last_good_report(tmp_path, monkeypatch):
    output = tmp_path / "SPY.ipynb"
    output.write_text("last good report")
    _fake_papermill(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        run.run_notebook(Path("template.ipynb"), output, {})
    assert output.read_text() == "last good report"
    assert run.partial_path(output).read_text() == "partly run"


def test_a_successful_run_replaces_the_report(tmp_path, monkeypatch):
    output = tmp_path / "SPY.ipynb"
    output.write_text("old report")
    _fake_papermill(monkeypatch, fail=False)
    run.run_notebook(Path("template.ipynb"), output, {})
    assert output.read_text() == "new report"
    assert not run.partial_path(output).exists()


def test_a_comparison_error_without_a_message_is_reported(monkeypatch, capsys):
    def fake_run_notebook(template, output, parameters):
        if "TICKERS" in parameters:
            raise RuntimeError("")

    monkeypatch.setattr(run, "run_notebook", fake_run_notebook)
    assert run.main(["SPY", "NVDA"]) == 1
    assert "Comparison: FAILED - RuntimeError" in capsys.readouterr().out


def test_comparison_only_uses_saved_summaries(tmp_path, monkeypatch, capsys, noise_summary):
    import src.report as report_module
    monkeypatch.setattr(report_module, "SUMMARY_DIR", tmp_path)
    save_summary({**noise_summary, "ticker": "SPY"}, folder=tmp_path)
    ran = []
    monkeypatch.setattr(run, "run_notebook", lambda template, output, parameters: ran.append(parameters))
    code = run.main(["SPY", "NVDA", "--comparison-only"])
    assert ran == [{"TICKERS": ["SPY"]}]                 # no stock re-run; NVDA has no summary
    assert code == 1
    assert "NVDA: no saved summary" in capsys.readouterr().out


def test_template_tests_both_parts_over_the_same_years():
    source = "\n".join(c.source for c in nbformat.read(ROOT / "notebooks" / "template.ipynb", as_version=4).cells)
    assert "vol_first_year = FIRST_TEST_YEAR" in source
    assert "first_test_year(vol_data)" not in source



# ---------------------------- phase 5: the LSTM (no PyTorch needed here) ----------------------------

def _fake_lstm_preds():
    days = pd.bdate_range("2020-01-01", periods=6)
    target = pd.Series([1, 0, 1, 1, 0, 0], index=days)
    lstm = pd.DataFrame({"prob": [0.6, 0.4, 0.6, 0.4, 0.4, 0.6],
                         "prob seed 0": [0.7, 0.3, 0.6, 0.4, 0.45, 0.6],
                         "prob seed 1": [0.5, 0.5, 0.6, 0.4, 0.35, 0.6]}, index=days)
    lstm["pred"] = (lstm["prob"] > 0.5).astype(int)          # 1 0 1 0 0 1
    logreg = pd.DataFrame({"pred": [1, 1, 0, 0, 0, 0]}, index=days)
    return {"LSTM": lstm, "Logistic regression": logreg}, target


def test_lstm_vs_logreg_by_hand():
    preds, target = _fake_lstm_preds()
    r = lstm_vs_logreg(preds, target)
    # they disagree on days 2, 3, 6: the LSTM is right on 2 and 3, logistic regression on 6
    assert r["days they disagree"] == 3
    assert (r["LSTM right"], r["logistic regression right"]) == (2, 1)
    assert r["LSTM accuracy"] == pytest.approx(4 / 6)
    assert r["logistic regression accuracy"] == pytest.approx(3 / 6)
    assert r["p-value"] == pytest.approx(0.5)               # 2 or more wins of 3 by coin flips
    # seed 0 says 0.5 on day 2 (class 0) and seed 1 says 0.5 on day 1: they disagree on day 1 only
    assert r["seeds"] == 2 and r["days all seeds agree"] == pytest.approx(5 / 6)


def test_summary_adds_the_lstm_comparison_only_when_the_lstm_ran():
    data = build_dataset(make_random_walk(n_days=1800, seed=6))
    test = data[data.index.year >= 2010]
    lr = walk_forward(get_models()["Logistic regression"], data, 2010)
    fake_lstm = lr.assign(**{"prob seed 0": lr["prob"]})
    without = summarize("X", test, {"Logistic regression": lr}, luck_trials=5)
    with_lstm = summarize("X", test, {"Logistic regression": lr, "LSTM": fake_lstm}, luck_trials=5)
    assert "LSTM vs logistic regression" not in without
    assert with_lstm["LSTM vs logistic regression"]["days they disagree"] == 0
    assert "LSTM" in with_lstm["models"]


def test_lstm_comparison_table():
    assert lstm_comparison([_summary("A", [0.5], [0.5])]) == (None, None)
    s = _summary("A", [0.5], [0.5])
    s["LSTM vs logistic regression"] = {"LSTM accuracy": 0.55, "logistic regression accuracy": 0.54,
                                        "days they disagree": 300, "LSTM right": 160, "logistic regression right": 140,
                                        "p-value": 0.01, "days all seeds agree": 0.9}
    s["volatility"] = _vol_block([0.5], [0.5])
    s["volatility"]["LSTM vs logistic regression"] = {**s["LSTM vs logistic regression"], "p-value": 0.04}
    table, info = lstm_comparison([s])
    assert info["comparisons"] == 2
    # Holm: 0.01 must clear 0.025, then 0.04 must clear 0.05: both pass
    assert table["LSTM better (corrected)?"].all()
    assert table.loc[("A", "direction"), "edge"] == pytest.approx(0.01)
def test_lstm_comparison_is_also_corrected_within_each_task():
    """Review issue 45: both tasks as one family (the plan from phase 5), and within each task."""

    rows = {"direction": [0.20, 0.30], "move size": [0.0074, 0.0196]}
    summaries = []
    for k, ticker in enumerate(["A", "B"]):
        s = _summary(ticker, [0.5], [0.5])
        s["volatility"] = _vol_block([0.5], [0.5])
        s["volatility"]["test days"] = 2000                       # move size has fewer test days
        for task, block in [("direction", s), ("move size", s["volatility"])]:
            block["LSTM vs logistic regression"] = {
                "LSTM accuracy": 0.6, "logistic regression accuracy": 0.58, "days they disagree": 300,
                "LSTM right": 170, "logistic regression right": 130, "p-value": rows[task][k],
                "days all seeds agree": 0.9}
        summaries.append(s)
    table, info = lstm_comparison(summaries)
    # one family of 4: 0.0074 must clear 0.0125 (it does), then 0.0196 must clear 0.0167 (it doesn't)
    assert info["passes (corrected)"] == 1
    # within move size (2 tests): 0.0074 clears 0.025 and 0.0196 clears 0.05
    assert table.loc[("A", "move size"), "LSTM better (corrected within its task)?"]
    assert table.loc[("B", "move size"), "LSTM better (corrected within its task)?"]
    assert not table.loc[("A", "direction"), "LSTM better (corrected within its task)?"]
    assert info["passes (corrected within its task)"] == 2
    assert info["comparisons per task"] == {"direction": 2, "move size": 2}
    # each task's edge range uses that task's own number of test days
    from src.evaluate import edge_half_width
    assert table.loc[("A", "move size"), "edge ±"] == pytest.approx(edge_half_width(170, 130, 2000))
    assert table.loc[("A", "direction"), "edge ±"] == pytest.approx(edge_half_width(170, 130, 2770))


def test_run_lstm_option(monkeypatch, capsys):
    import src.lstm
    seen = []
    monkeypatch.setattr(run, "run_notebook", lambda template, output, parameters: seen.append(parameters))
    monkeypatch.setattr(src.lstm, "torch_available", lambda: True)
    run.main(["SPY", "--no-comparison", "--lstm"])
    run.main(["SPY", "--no-comparison"])
    assert [p["RUN_LSTM"] for p in seen] == [True, False]
    # without PyTorch, --lstm stops at once with a clear message
    monkeypatch.setattr(src.lstm, "torch_available", lambda: False)
    assert run.main(["SPY", "--lstm"]) == 1
    assert "pip install -r requirements-lstm.txt" in capsys.readouterr().out
    assert len(seen) == 2


def test_lstm_says_how_to_install_pytorch(monkeypatch):
    import src.lstm
    monkeypatch.setattr(src.lstm, "torch_available", lambda: False)
    with pytest.raises(ImportError, match="requirements-lstm.txt"):
        src.lstm.require_torch()


def test_template_draws_the_lstm_training_charts():
    source = "\n".join(c.source for c in nbformat.read(ROOT / "notebooks" / "template.ipynb", as_version=4).cells)
    for needed in ["curves=lstm_curves, record=lstm_record", "curves=vol_lstm_curves, record=vol_lstm_record",
                   "draw_network(ax)", "plot_learning_curves(lstm_curves", "plot_learning_curves(vol_lstm_curves",
                   "plot_epochs_chosen(vol_lstm_record", 'plot_confidence(vol_model_preds["LSTM"]["prob"]']:
        assert needed in source, needed
def test_run_warns_when_the_comparison_leaves_out_saved_summaries(tmp_path, monkeypatch, capsys):
    """Review issue 68: one stock run alone leaves the comparison stale; two of four rebuild it smaller."""

    import src.report as report_module
    monkeypatch.setattr(report_module, "SUMMARY_DIR", tmp_path)
    for ticker in ["SPY", "NVDA", "AAPL", "GOOGL"]:
        (tmp_path / f"{ticker}.json").write_text("{}")
    monkeypatch.setattr(run, "run_notebook", lambda template, output, parameters: None)
    run.main(["SPY"])
    out = capsys.readouterr().out
    assert "was not rebuilt" in out and "python run.py AAPL GOOGL NVDA SPY --comparison-only" in out
    run.main(["SPY", "NVDA"])
    out = capsys.readouterr().out
    assert "covers SPY, NVDA only" in out and "AAPL, GOOGL" in out
    run.main(["SPY", "NVDA", "AAPL", "GOOGL"])
    assert "Note:" not in capsys.readouterr().out