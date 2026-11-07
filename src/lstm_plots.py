"""
Phase 5.1: pictures of the LSTM. A diagram of the network, and charts of how
its training went. Nothing here trains a model or changes a prediction: the
charts only draw numbers the training already recorded (see `curves` and
`record` in lstm.lstm_walk_forward).

Needs matplotlib only, so the charts can be tested without PyTorch.
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator

from .lstm import EPOCHS, HIDDEN, INPUTS, WINDOW

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a"
INPUT_NAMES = {"ret_rel": "Return", "range_rel": "High-low swing", "volume_ratio": "Volume"}


# ---------------------------- the network ------------------------------------

def network_positions(n_inputs, n_hidden):
    """(x, y) of each node: inputs, hidden units and the output, top to bottom."""

    def column(n):
        return np.linspace(0.5, -0.5, n) if n > 1 else np.zeros(1)
    spread = min(1.5, 0.12 * n_hidden)                    # a few units sit closer together
    return {"inputs": [(0.0, y * 0.45) for y in column(n_inputs)],
            "hidden": [(1.0, y * spread) for y in column(n_hidden)],
            "output": [(2.0, 0.0)]}


def draw_network(ax=None, inputs=INPUTS, hidden=HIDDEN, window=WINDOW, shown=8, weights=None, scale=None):
    """Draw the network: inputs, the LSTM layer with its loop over the days, and the output.

    shown:   how many of the hidden units to draw (all of them if weights are given)
    weights: optional {"in": array (hidden, inputs), "out": array (hidden,)}; each
             line is then coloured by its weight's sign (blue +, orange -) and drawn
             thicker the larger the weight. Without weights every line is thin grey.
    scale:   the weight drawn at full thickness (default: the largest one given);
             the animation passes one scale for every frame
    Returns the axes.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.6))
    if weights is not None:
        shown = hidden
    shown = min(shown, hidden)
    pos = network_positions(len(inputs), shown)
    radius = 0.045 if shown <= 8 else 0.018

    def edge(a, b, w=None, scale=1.0):
        if w is None:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GREY, linewidth=0.6, alpha=0.5, zorder=1)
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE if w > 0 else ORANGE,
                    linewidth=0.2 + 3.0 * abs(w) / scale, alpha=0.75, zorder=1)

    if weights is not None and scale is None:
        scale = max(np.abs(weights["in"]).max(), np.abs(weights["out"]).max(), 1e-12)
    for j, h in enumerate(pos["hidden"]):
        for i, x in enumerate(pos["inputs"]):
            edge(x, h, None if weights is None else weights["in"][j, i], scale)
        edge(h, pos["output"][0], None if weights is None else weights["out"][j], scale)

    for (x, y) in pos["inputs"]:
        ax.add_patch(Circle((x, y), radius, color=AQUA, zorder=3))
    for (x, y) in pos["hidden"]:
        ax.add_patch(Circle((x, y), radius, color=BLUE, zorder=3))
    ax.add_patch(Circle(pos["output"][0], radius, color=ORANGE, zorder=3))

    for name, (x, y) in zip(inputs, pos["inputs"]):
        ax.text(x - 0.08, y, INPUT_NAMES.get(name, name), ha="right", va="center", fontsize=10)
    ax.text(2.0, -0.12, "Probability", ha="center", va="top", fontsize=10)

    top = max(y for _, y in pos["hidden"]) + radius
    ax.add_patch(FancyArrowPatch((1.07, top + 0.02), (0.93, top + 0.02), connectionstyle="arc3,rad=1.6",
                                 arrowstyle="-|>", mutation_scale=12, color=BLUE, linewidth=1.3))
    ax.text(1.0, top + 0.2, f"memory carried from one day to the next, for {window} days",
            ha="center", va="bottom", fontsize=9.5, color=BLUE)

    bottom = min(y for _, y in pos["hidden"]) - radius - 0.12
    unit_note = f"{hidden} units" + (f", {shown} shown" if shown < hidden else "")
    for x, label in [(0.0, f"Inputs ({len(inputs)} per day)"), (1.0, f"LSTM layer ({unit_note})"),
                     (2.0, "Output")]:
        ax.text(x, bottom, label, ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.set_xlim(-0.75, 2.4)
    ax.set_ylim(bottom - 0.15, top + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def network_weights(net):
    """The weights drawn in the animation, from a trained network (see _network in lstm.py).

    in:  for each hidden unit and input, the weight with which that input writes
         new information into the unit's memory (PyTorch's "g" gate, the third
         of its four gates); the other three gates decide how much is kept
    out: the weight of each unit in the final probability
    """

    w_ih = net.lstm.weight_ih_l0.detach().numpy()
    hidden = w_ih.shape[0] // 4
    return {"in": w_ih[2 * hidden:3 * hidden].copy(),
            "out": net.out.weight.detach().numpy()[0].copy()}


# ---------------------------- training charts ------------------------------------

def curves_table(curves, run="validation run", column="validation loss"):
    """One row per (year, seed, epoch) with the loss, from lstm_walk_forward's curves."""

    rows = [{"year": c["year"], "seed": c["seed"], "best epoch": c["best epoch"],
             "epoch": h["epoch"], "loss": h[column]}
            for c in curves for h in c[run]]
    return pd.DataFrame(rows, columns=["year", "seed", "best epoch", "epoch", "loss"])


def plot_learning_curves(curves, title):
    """Learning curves for every test year and seed.

    Left: validation loss in the run that picks the number of epochs; right:
    training loss in the same runs. Each faint line is one year and seed; the
    bold line is the average over the runs still going at that epoch, and each
    dot marks the epoch early stopping chose. Losses are shown as the change
    since epoch 1, so years with different starting losses can share a chart.
    """

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, column, name in [(left, "validation loss", "validation loss (the year held back)"),
                             (right, "train loss", "training loss")]:
        table = curves_table(curves, column=column)
        table["change"] = table["loss"] - table.groupby(["year", "seed"])["loss"].transform("first")
        for _, run in table.groupby(["year", "seed"]):
            ax.plot(run["epoch"], run["change"], color=GREY, linewidth=0.7, alpha=0.3)
            if column == "validation loss":
                chosen = run[run["epoch"] == run["best epoch"]]
                ax.plot(chosen["epoch"], chosen["change"], "o", color=ORANGE, markersize=3.5, alpha=0.7)
        runs = table.groupby(["year", "seed"]).ngroups
        by_epoch = table.groupby("epoch")["change"]
        average = by_epoch.mean()[by_epoch.size() >= max(3, runs // 4)]
        ax.plot(average.index, average.values, color=BLUE, linewidth=2.2, label="average over runs")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    left.set_ylabel("change in loss since epoch 1\n(lower is better)")
    left.plot([], [], "o", color=ORANGE, markersize=4, label="epoch chosen")
    left.legend(loc="upper right", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_epochs_chosen(record, title, max_epochs=EPOCHS):
    """How many epochs early stopping chose each test year: a bar for the average
    over seeds, a dot per seed. Always 1 or 2 means there was little to learn."""

    fig, ax = plt.subplots(figsize=(11, 3.5))
    years = [r[0] for r in record]
    chosen = [r[3] for r in record]
    ax.bar(years, [np.mean(c) for c in chosen], color=BLUE, alpha=0.5, label="average over seeds")
    for year, c in zip(years, chosen):
        ax.plot([year] * len(c), c, "o", color=BLUE, markersize=4)
    ax.plot([], [], "o", color=BLUE, markersize=4, label="each seed")
    ax.set_ylim(0, max_epochs + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(years)
    ax.set_ylabel("epochs chosen")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confidence(lstm_prob, logreg_prob, title, what="up"):
    """Histograms of the LSTM's and logistic regression's probabilities on the test days.
    A narrow spike near 0.5 means the model rarely leans either way."""

    fig, ax = plt.subplots(figsize=(11, 3.5))
    low = min(np.min(lstm_prob), np.min(logreg_prob), 0.45)
    high = max(np.max(lstm_prob), np.max(logreg_prob), 0.55)
    bins = np.linspace(max(0, low - 0.01), min(1, high + 0.01), 61)     # fine bins over the range used
    ax.hist(logreg_prob, bins=bins, histtype="stepfilled", color=ORANGE, alpha=0.35,
            label="Logistic regression")
    ax.hist(lstm_prob, bins=bins, histtype="step", color=BLUE, linewidth=1.8, label="LSTM")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(f"predicted probability of {what}")
    ax.set_ylabel("test days")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
