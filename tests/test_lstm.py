"""
Tests for phase 5, the optional LSTM: its inputs never use the future, its
walk-forward never trains on the year it predicts, and it finds a planted
pattern but nothing in pure noise.

Skipped automatically if PyTorch isn't installed (pip install -r requirements-lstm.txt).
The networks here are tiny and train briefly, so the tests stay fast.
"""


import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data import make_planted_pattern, make_random_walk  # noqa: E402
from src.evaluate import first_test_year, paired_p_value  # noqa: E402
from src.features import build_dataset, build_vol_dataset  # noqa: E402
from src.lstm import (INPUTS, fit_scaler, lstm_walk_forward, make_sequences,  # noqa: E402
                      scale, sequence_inputs)

FAST = dict(window=10, epochs=4, hidden=8, batch=256)


@pytest.fixture(scope="module")
def noise_prices():
    return make_random_walk(n_days=1600, seed=4)


# ---------------------------- inputs ------------------------------------

def test_inputs_never_use_the_future(noise_prices):
    k = 1000
    original = sequence_inputs(noise_prices)
    scrambled = noise_prices.copy()
    rng = np.random.default_rng(3)
    for col in ["Close", "High", "Low", "Volume"]:
        scrambled.iloc[k + 1:, scrambled.columns.get_loc(col)] *= rng.uniform(0.5, 1.5, len(scrambled) - k - 1)
    changed = sequence_inputs(scrambled)
    pd.testing.assert_frame_equal(original.iloc[:k + 1], changed.iloc[:k + 1])
    assert not original.iloc[k + 1:].equals(changed.iloc[k + 1:])
    assert list(original.columns) == INPUTS


def test_each_sequence_is_the_window_ending_on_its_day():
    days = pd.bdate_range("2020-01-01", periods=8)
    inputs = pd.DataFrame(np.arange(24, dtype=float).reshape(8, 3), index=days, columns=INPUTS)
    inputs.iloc[1, 0] = np.nan                                   # day 2 is incomplete
    X, kept = make_sequences(inputs, days, window=3)
    # windows touching day 2 (ending on days 2, 3, 4) are skipped; days 1 and 2 have no full window anyway
    assert list(kept) == list(days[4:])
    np.testing.assert_array_equal(X[0], inputs.iloc[2:5].to_numpy())   # ends on day 5, oldest first


def test_scaling_uses_only_the_training_days(noise_prices):
    inputs = sequence_inputs(noise_prices).dropna()
    cut = inputs.index[500]
    later_changed = inputs.copy()
    later_changed.iloc[501:] *= 10
    for a, b in zip(fit_scaler(inputs, cut), fit_scaler(later_changed, cut)):
        np.testing.assert_allclose(a, b)
    X = np.full((1, 2, 3), 1e6, dtype=np.float32)
    assert scale(X, fit_scaler(inputs, cut)).max() <= 8.0      # extreme days are clipped


# ---------------------------- walk-forward ------------------------------------

def test_walk_forward_never_trains_on_the_year_it_predicts(noise_prices):
    data = build_dataset(noise_prices)
    record = []
    preds = lstm_walk_forward(noise_prices, data, 2010, seeds=(0,), record=record, **FAST)
    assert [r[0] for r in record] == sorted(set(preds.index.year))
    for year, last_train, last_fit, _ in record:
        before = data.index[data.index.year < year]
        assert last_train == before[-2]                          # the day whose label is in the test year is left out
        before_val = data.index[data.index.year < year - 1]
        assert last_fit == before_val[-2]                         # and likewise for the validation year


def test_walk_forward_predicts_every_test_day_once(noise_prices):
    data = build_vol_dataset(noise_prices)
    start = first_test_year(data)
    preds = lstm_walk_forward(noise_prices, data, start, seeds=(0, 1), **FAST)
    assert preds.index.equals(data.index[data.index.year >= start])
    assert preds["prob"].between(0, 1).all()
    np.testing.assert_allclose(preds["prob"], preds[["prob seed 0", "prob seed 1"]].mean(axis=1), rtol=1e-6)
    assert (preds["pred"] == (preds["prob"] > 0.5).astype(int)).all()


def test_same_seed_gives_the_same_predictions(noise_prices):
    data = build_dataset(noise_prices)
    a = lstm_walk_forward(noise_prices, data, 2010, seeds=(7,), **FAST)
    b = lstm_walk_forward(noise_prices, data, 2010, seeds=(7,), **FAST)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------- the pipeline checks ------------------------------------

def _vs_always_up(prices):
    data = build_dataset(prices)
    start = first_test_year(data)
    preds = lstm_walk_forward(prices, data, start, seeds=(0,), window=10, epochs=15, hidden=16, batch=64)
    target = data.loc[preds.index, "target"]
    always_up = pd.Series(1, index=target.index)
    return (preds["pred"] == target).mean(), (always_up == target).mean(), paired_p_value(preds["pred"], always_up, target)


def test_lstm_finds_the_planted_pattern():
    acc, always_up, p = _vs_always_up(make_planted_pattern(n_days=2600, seed=3))
    assert acc > 0.55 and p <= 0.05


def test_lstm_finds_nothing_in_noise():
    acc, always_up, p = _vs_always_up(make_random_walk(n_days=2600, seed=3))
    assert abs(acc - always_up) < 0.04 and p > 0.05


def test_early_stopping_uses_the_validation_year_not_the_test_year(noise_prices, monkeypatch):
    """Each year: one run to pick the number of epochs, validated on the last
    training year, then one run on all training years. The test year is never used."""
    
    import src.lstm as lstm
    calls = []
    real_train = lstm._train

    def recording_train(X, y, seed, epochs, hidden, dropout, lr, batch, X_val=None, y_val=None, patience=3,
                        **recording):
        calls.append((len(X), None if X_val is None else len(X_val)))
        return real_train(X, y, seed, epochs, hidden, dropout, lr, batch, X_val, y_val, patience, **recording)

    monkeypatch.setattr(lstm, "_train", recording_train)
    data = build_dataset(noise_prices)
    inputs = sequence_inputs(noise_prices)
    _, days = make_sequences(inputs, data.index, FAST["window"])
    lstm_walk_forward(noise_prices, data, 2010, seeds=(0,), **FAST)
    years = sorted(set(days.year[days.year >= 2010]))
    assert len(calls) == 2 * len(years)
    for year, (pick, final) in zip(years, zip(calls[::2], calls[1::2])):
        n_val = int((days.year == year - 1).sum())
        n_before_val = int((days.year < year - 1).sum())
        # fit on years before the validation year (minus its last day, whose label is in the
        # validation year); validate on that year (minus its last day, whose label is in the test year)
        assert pick == (n_before_val - 1, n_val - 1)
        assert final == (n_before_val + n_val - 1, None)    # then all training years, no validation


# ---------------------------- phase 5.1: recording training ------------------------------------

def test_recording_the_training_changes_no_prediction(noise_prices):
    data = build_dataset(noise_prices)
    plain = lstm_walk_forward(noise_prices, data, 2010, seeds=(0, 1), **FAST)
    curves, record = [], []
    watched = lstm_walk_forward(noise_prices, data, 2010, seeds=(0, 1), curves=curves, record=record, **FAST)
    pd.testing.assert_frame_equal(plain, watched, check_exact=True)
    assert len(curves) == 2 * len(record)


def test_predictions_are_unchanged_since_phase_5(noise_prices):
    """Locks the LSTM's predictions on fixed fake prices to those of the phase 5 code
    (6 Nov 2026), before training was recorded for the charts. On the same computer
    they match exactly; the tolerance only allows for different computers doing
    float arithmetic slightly differently. If a deliberate change to the LSTM moves
    them, update the numbers and say why in the project log."""

    data = build_dataset(noise_prices)
    preds = lstm_walk_forward(noise_prices, data, 2010, seeds=(0, 1), **FAST)
    assert len(preds) == 295
    assert float(preds["prob"].astype("float64").sum()) == pytest.approx(159.77240937948227, abs=1e-3)
    assert preds["prob"].iloc[[0, 100, -1]].tolist() == pytest.approx(
        [0.5558313727378845, 0.5456246137619019, 0.5222107172012329], abs=1e-4)


def test_recorded_curves_match_what_training_did(noise_prices):
    from make_lstm_pictures import early_stopping_epoch
    data = build_dataset(noise_prices)
    curves, record = [], []
    lstm_walk_forward(noise_prices, data, 2010, seeds=(0, 1), curves=curves, record=record, **FAST)
    for year, _, _, best_epochs in record:
        runs = [c for c in curves if c["year"] == year]
        assert [c["seed"] for c in runs] == [0, 1]
        assert [c["best epoch"] for c in runs] == best_epochs
    for c in curves:
        val = [h["validation loss"] for h in c["validation run"]]
        assert c["best epoch"] == early_stopping_epoch(val)
        assert len(val) == min(FAST["epochs"], c["best epoch"] + 3)      # stopped after 3 epochs without progress
        assert len(c["final run"]) == c["best epoch"]                     # retrained for exactly that many
        assert all(h["validation loss"] is None for h in c["final run"])  # the final run sees no validation year
        losses = [h["train loss"] for h in c["validation run"] + c["final run"]]
        assert all(np.isfinite(losses)) and min(losses) > 0


def test_on_epoch_sees_the_start_and_every_epoch():
    from src.lstm import _train
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5, 3)).astype(np.float32)
    y = rng.integers(0, 2, 64)
    seen = []
    _train(X, y, 0, 3, 4, 0.2, 1e-3, 32, on_epoch=lambda net, epoch: seen.append(epoch))
    assert seen == [0, 1, 2, 3]


# ---------------------------- phase 5.1: the picture script ------------------------------------

def test_early_stopping_epoch_by_hand():
    from make_lstm_pictures import early_stopping_epoch
    assert early_stopping_epoch([0.70, 0.69, 0.695, 0.70, 0.71]) == 2
    assert early_stopping_epoch([0.70, 0.71, 0.72, 0.73, 0.60]) == 1     # stopped at epoch 4, before the late drop
    assert early_stopping_epoch([0.70, 0.71, 0.72, 0.60]) == 4           # two bad epochs are tolerated, three aren't


def test_animation_shows_the_real_training_run():
    """The animated network is the validation run lstm_walk_forward makes for that year
    and seed: the same losses, epoch for epoch, until early stopping ends it."""

    from make_lstm_pictures import train_one
    prices = make_random_walk(n_days=1600, seed=4)
    data = build_dataset(prices)
    curves = []
    lstm_walk_forward(prices, data, 2010, seeds=(0,), curves=curves, **FAST)
    real = curves[-1]                                                     # the last test year
    run = train_one(prices, task="direction", year=real["year"], seed=0, window=FAST["window"],
                    epochs=FAST["epochs"], hidden=FAST["hidden"], batch=FAST["batch"])
    assert len(run["history"]) == FAST["epochs"]                          # every epoch, no early stop
    assert len(run["weights"]) == FAST["epochs"] + 1                      # plus the starting weights
    for shown, actual in zip(run["history"], real["validation run"]):
        assert shown == pytest.approx(actual)
    assert run["chosen"] == real["best epoch"]
    assert run["validation year"] == real["year"] - 1


def test_picture_script_writes_both_pictures(tmp_path):
    from PIL import Image
    import make_lstm_pictures
    prices = make_random_walk(n_days=1600, seed=4)
    assert make_lstm_pictures.main(["FAKE", "--folder", str(tmp_path), "--task", "direction"], prices=prices,
                                   window=10, epochs=3, hidden=4, batch=256) == 0
    assert (tmp_path / "lstm_network.png").stat().st_size > 0
    gif = Image.open(tmp_path / "lstm_training.gif")
    assert gif.n_frames == 3 + 1                                          # the starting weights, then 3 epochs
    gif.seek(gif.n_frames - 1)
    assert gif.info["duration"] == 3000                                   # the last epoch is held
def test_latest_full_year_skips_a_part_year():
    """Review issue 51: data to 5 Nov 2026 must give 2025, not the unfinished 2026."""

    from make_lstm_pictures import latest_full_year
    assert latest_full_year(pd.bdate_range("2024-01-01", "2026-11-05")) == 2025
    assert latest_full_year(pd.bdate_range("2024-01-01", "2025-12-31")) == 2025
    assert latest_full_year(pd.bdate_range("2024-01-01", "2026-01-02")) == 2025


def test_picture_script_doesnt_download(tmp_path, monkeypatch, capsys):
    """Without saved prices it says what to run instead of downloading them."""

    import make_lstm_pictures
    import src.data
    monkeypatch.setattr(src.data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(src.data, "load_prices", lambda *a, **k: pytest.fail("it tried to download"))
    assert make_lstm_pictures.main(["ZZZ", "--folder", str(tmp_path)]) == 1
    assert "python run.py ZZZ" in capsys.readouterr().out