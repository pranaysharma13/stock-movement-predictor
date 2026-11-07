"""
Tests for phase 5.1's pictures of the LSTM (src/lstm_plots.py). They need
matplotlib only, not PyTorch: the charts draw numbers the training recorded.
"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from src.lstm_plots import (curves_table, draw_network, plot_confidence, plot_epochs_chosen,  # noqa: E402
                            plot_learning_curves)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def fake_curves():
    """Two years, two seeds, validation losses that fall then rise."""

    curves = []
    for year in (2015, 2016):
        for seed in (0, 1):
            val = [0.69, 0.68, 0.675, 0.68, 0.69, 0.70][: 4 + seed]
            curves.append({"year": year, "seed": seed, "best epoch": 3,
                           "validation run": [{"epoch": e + 1, "train loss": 0.7 - 0.01 * e, "validation loss": v}
                                              for e, v in enumerate(val)],
                           "final run": [{"epoch": e + 1, "train loss": 0.7 - 0.01 * e, "validation loss": None}
                                         for e in range(3)]})
    return curves


def test_network_draws_each_node_once():
    ax = draw_network(inputs=["a", "b", "c"], hidden=32, shown=8)
    assert sum(isinstance(p, Circle) for p in ax.patches) == 3 + 8 + 1
    assert len(ax.lines) == 8 * 3 + 8                                     # every input to every unit, every unit out
    labels = " ".join(t.get_text() for t in ax.texts)
    assert "32 units, 8 shown" in labels and "60 days" in labels


def test_network_lines_show_the_weights():
    weights = {"in": np.array([[1.0, -0.5], [0.0, 2.0]]), "out": np.array([-2.0, 0.5])}
    ax = draw_network(inputs=["a", "b"], hidden=2, weights=weights)
    widths = [line.get_linewidth() for line in ax.lines]
    colors = [line.get_color() for line in ax.lines]
    # order: unit 0's two inputs, unit 0 out, unit 1's two inputs, unit 1 out
    assert widths == pytest.approx([0.2 + 3.0 * w for w in (0.5, 0.25, 1.0, 0.0, 1.0, 0.25)])
    assert colors[0] == "#2a78d6" and colors[1] == "#eb6834"              # positive blue, negative orange
    fixed = draw_network(plt.subplots()[1], inputs=["a", "b"], hidden=2, weights=weights, scale=4.0)
    assert max(line.get_linewidth() for line in fixed.lines) == pytest.approx(0.2 + 3.0 * 0.5)


def test_curves_table_has_a_row_per_epoch():
    table = curves_table(fake_curves())
    assert len(table) == 2 * (4 + 5)
    assert set(table.columns) == {"year", "seed", "best epoch", "epoch", "loss"}
    assert table["loss"].iloc[2] == pytest.approx(0.675)
    final = curves_table(fake_curves(), run="final run", column="train loss")
    assert len(final) == 4 * 3


def test_training_charts_draw():
    fig = plot_learning_curves(fake_curves(), "test")
    left, right = fig.axes
    assert "validation" in left.get_title() and "training" in right.get_title()
    record = [(2015, None, None, [3, 3]), (2016, None, None, [2, 4])]
    fig = plot_epochs_chosen(record, "test", max_epochs=20)
    bars = [p.get_height() for p in fig.axes[0].patches]
    assert bars == pytest.approx([3, 3])
    rng = np.random.default_rng(0)
    fig = plot_confidence(rng.uniform(0.4, 0.6, 500), rng.uniform(0.3, 0.7, 500), "test")
    assert fig.axes[0].get_xlim()[0] >= 0 and fig.axes[0].get_xlim()[1] <= 1
def test_the_average_keeps_stopped_runs_at_their_last_value():
    """Review issue 59: two runs, one stopping after 2 epochs. Epoch 3's average uses
    both runs, the stopped one at its last value, not only the run still going."""

    curves = [{"year": 2015, "seed": s, "best epoch": 1, "final run": [],
               "validation run": [{"epoch": e + 1, "train loss": 0.7, "validation loss": v} for e, v in enumerate(vals)]}
              for s, vals in [(0, [0.70, 0.72]), (1, [0.70, 0.66, 0.64])]]
    fig = plot_learning_curves(curves, "test")
    average = [line for line in fig.axes[0].lines if line.get_label().startswith("average")][0]
    # changes since epoch 1: run 0 = 0, +0.02 (kept), run 1 = 0, -0.04, -0.06
    assert list(average.get_xdata()) == [1, 2, 3]
    assert list(average.get_ydata()) == pytest.approx([0.0, -0.01, -0.02])


def test_the_full_network_keeps_its_captions_apart():
    """Review issue 67: with all 32 units drawn, the columns are wider apart."""

    weights = {"in": np.ones((32, 3)), "out": np.ones(32)}
    ax = draw_network(weights=weights)
    captions = {t.get_text(): t.get_position()[0] for t in ax.texts}
    assert captions["LSTM layer (32 units)"] - captions["Inputs (3 per day)"] >= 1.4
    probability = [t for t in ax.texts if t.get_text() == "Probability"][0]
    assert probability.get_position()[0] > captions["Output"]            # beside the output node, off the lines
