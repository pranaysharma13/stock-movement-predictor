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
from src.evaluate import vol_baseline_predictions
from src.features import VOL_FEATURES, build_vol_dataset
from src.report import (comparison_table, half_on_big, load_summaries, safe_name, save_summary,
                        summarize, summarize_vol)

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
    json.dumps(s)                       # must be saveable as JSON


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
    summaries = [_summary("A", [0.01, 0.30], [0.02, 0.5]), _summary("B", [0.004, 0.9], [0.001, None])]
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
    # luck 0.02 passes at 5% only; 0.001 passes both; None never passes
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
    ("template.ipynb", ["TICKER", "START", "FIRST_TEST_YEAR", "COST"]),
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

def _vol_block(p_values, luck_shares):
    return {
        "test start": "2010-01-04", "test end": "2020-12-31", "test days": 2770,
        "share of big days": 0.5, "majority accuracy": 0.51, "same as today accuracy": 0.53,
        "buy and hold": {"annual return": 0.10, "Sharpe ratio": 0.8, "max drawdown": -0.3},
        "same as today strategy": {"annual return": 0.05, "Sharpe ratio": 0.4, "max drawdown": -0.25},
        "models": {
            f"model {i}": {"accuracy": 0.60, "predicted big": 0.4, "p-value vs same as today": p,
                           "strategy": {"annual return": 0.08, "Sharpe ratio": 0.7, "max drawdown": -0.2},
                           "share of random >=": luck}
            for i, (p, luck) in enumerate(zip(p_values, luck_shares))
        },
    }


def test_vol_summary_has_the_headline_numbers():
    data = build_vol_dataset(make_random_walk(n_days=2200, seed=8))
    test = data[data.index.year >= 2010]
    preds = {name: walk_forward(m, data, 2010, features=VOL_FEATURES) for name, m in get_models().items()}
    v = summarize_vol(test, preds, vol_baseline_predictions(test), luck_trials=20)
    assert set(v["models"]) == {"Logistic regression", "Gradient boosting"}
    assert 0 <= v["same as today accuracy"] <= 1
    assert v["majority accuracy"] >= 0.5
    for m in v["models"].values():
        assert set(m["strategy"]) == {"annual return", "Sharpe ratio", "max drawdown"}
        assert 0 <= m["p-value vs same as today"] <= 1
    json.dumps(v)


def test_half_on_big():
    assert list(half_on_big(pd.Series([0, 1, 1, 0]))) == [1.0, 0.5, 0.5, 1.0]


def test_volatility_comparison_is_corrected_on_its_own():
    s = _summary("A", [0.5, 0.5], [0.5, 0.5])
    s["volatility"] = _vol_block([0.001, 0.02], [0.001, 0.9])
    table, info = comparison_table([s], task="volatility")
    assert info["comparisons"] == 2
    assert info["corrected threshold"] == pytest.approx(0.025)
    assert table.loc[("A", "model 0"), "accuracy edge (corrected)?"]
    assert table.loc[("A", "model 1"), "accuracy edge (corrected)?"]      # 0.02 < 0.025
    assert table.loc[("A", "model 0"), "edge"] == pytest.approx(0.60 - 0.53)
    assert info["luck passes (corrected)"] == 1


def test_old_summaries_without_move_size_say_what_to_do():
    with pytest.raises(ValueError, match="python run.py OLD"):
        comparison_table([_summary("OLD", [0.5], [0.5])], task="volatility")


def test_unknown_task_is_refused():
    with pytest.raises(ValueError, match="task must be"):
        comparison_table([_summary("A", [0.5], [0.5])], task="price")
