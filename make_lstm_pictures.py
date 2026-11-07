"""
Pictures of the LSTM for the README (phase 5.1): a diagram of the network, and
an animation of one network training.

    python make_lstm_pictures.py                 SPY, move size, the latest complete year
    python make_lstm_pictures.py NVDA --year 2020 --task direction

Writes docs/lstm_network.png and docs/lstm_training.gif. Needs PyTorch.

The animation trains a single network: one stock, one test year, one seed. It
is the first of the two runs lstm_walk_forward makes for that year: trained on
the years before the validation year and checked on the validation year (the
year before the test year). Here it runs for every epoch instead of stopping
early, so you can see what early stopping avoids; the epoch it would have
chosen is marked. Lines are coloured by the sign of their weight and drawn
thicker the larger it is.

This script only makes pictures. The reports and the forward test never run it,
and it can't change any prediction.
"""


import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draw the LSTM and animate one network training.")
    parser.add_argument("ticker", nargs="?", default="SPY", help="stock symbol (default SPY)")
    parser.add_argument("--task", choices=["size", "direction"], default="size",
                        help="move size (default) or direction")
    parser.add_argument("--year", type=int, help="test year (default: the latest complete year)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", default="2005-01-01", help="first date of the saved prices")
    parser.add_argument("--folder", type=Path, default=DOCS, help="where the pictures go")
    return parser.parse_args(argv)


def early_stopping_epoch(val_losses, patience=3):
    """The epoch early stopping would have chosen, with the same rule as lstm._train."""

    best_loss, best_epoch, stale = float("inf"), 1, 0
    for epoch, loss in enumerate(val_losses, start=1):
        if loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_epoch


def latest_full_year(days):
    """The latest complete year: the last date's year only if the data reaches its
    last week (24 December or later), otherwise the year before."""

    last = days[-1]
    return int(last.year if last >= pd.Timestamp(year=last.year, month=12, day=24) else last.year - 1)


def train_one(prices, task="size", year=None, seed=0, window=None, epochs=None, hidden=None,
              dropout=0.2, lr=1e-3, batch=128):
    """Train one network as in lstm_walk_forward's validation run, recording every epoch.

    Returns a dict with the year, the per-epoch history, the weights at each epoch
    (epoch 0 = before training) and the epoch early stopping would have chosen.
    """

    from src import lstm
    from src.features import build_dataset, build_vol_dataset
    from src.lstm_plots import network_weights

    lstm.require_torch()
    window = window or lstm.WINDOW
    epochs = epochs or lstm.EPOCHS
    hidden = hidden or lstm.HIDDEN
    data = build_vol_dataset(prices) if task == "size" else build_dataset(prices)
    inputs = lstm.sequence_inputs(prices)
    X_all, days = lstm.make_sequences(inputs, data.index, window)
    y_all = data.loc[days, "target"].to_numpy()
    year = year or latest_full_year(days)
    _, _, fit_idx, val_idx = lstm.year_split(days, year)
    if len(fit_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Not enough data before {year} to train and validate.")
    scaler = lstm.fit_scaler(inputs, days[fit_idx[-1]])
    X_fit, X_val = lstm.scale(X_all[fit_idx], scaler), lstm.scale(X_all[val_idx], scaler)

    history, weights = [], []
    lstm._train(X_fit, y_all[fit_idx], seed, epochs, hidden, dropout, lr, batch,
                X_val, y_all[val_idx], patience=epochs + 1,   # run every epoch
                history=history, on_epoch=lambda net, epoch: weights.append(network_weights(net)))
    return {"year": year, "task": task, "history": history, "weights": weights,
            "chosen": early_stopping_epoch([h["validation loss"] for h in history]),
            "window": window, "hidden": hidden,
            "fit years": (int(days[fit_idx[0]].year), int(days[fit_idx[-1]].year)),
            "validation year": int(days[val_idx[0]].year)}


def animate(run, ticker, path, fps=2):
    """Save the training as a GIF: the network's weights on the left, the losses on the right."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    from PIL import Image
    from src.lstm_plots import BLUE, GREY, ORANGE, draw_network

    history, weights = run["history"], run["weights"]
    epochs = [h["epoch"] for h in history]
    val = [h["validation loss"] for h in history]
    train = [h["train loss"] for h in history]
    scale = max(max(abs(w["in"]).max(), abs(w["out"]).max()) for w in weights)
    what = "move size" if run["task"] == "size" else "direction"

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1]})

    def frame(k):
        left.clear()
        right.clear()
        # one scale for every frame, so growing weights look thicker
        draw_network(left, window=run["window"], hidden=run["hidden"], weights=weights[k], scale=scale)
        left.set_title(f"Epoch {k} of {len(weights) - 1}" + (" (starting weights)" if k == 0 else ""),
                       fontsize=12)
        right.plot(epochs, val, color=BLUE, alpha=0.2)
        right.plot(epochs, train, color=GREY, alpha=0.2)
        right.plot(epochs[:k], val[:k], color=BLUE, marker="o", markersize=4,
                   label=f"validation loss ({run['validation year']}, held back)")
        right.plot(epochs[:k], train[:k], color=GREY, marker="o", markersize=4,
                   label="training loss (average batch loss, dropout on)")
        if k >= run["chosen"]:
            right.axvline(run["chosen"], color=ORANGE, linestyle="--", linewidth=1.2,
                          label=f"early stopping would stop here (epoch {run['chosen']})")
        right.set_xlabel("epoch")
        right.set_ylabel("loss (lower is better)")
        right.set_xlim(0.5, len(history) + 0.5)
        right.xaxis.set_major_locator(MaxNLocator(integer=True))
        right.legend(loc="upper right", fontsize=8.5)
        right.set_title(f"Trained on {run['fit years'][0]}–{run['fit years'][1]}", fontsize=12)
        fig.suptitle(f"{ticker}: the LSTM learning {what}, one seed. Blue lines are positive weights, "
                     f"orange negative; thicker is larger", fontsize=11)

    images = []
    for k in range(len(weights)):
        frame(k)
        fig.canvas.draw()
        rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        images.append(rgb.convert("P", palette=Image.ADAPTIVE))
    plt.close(fig)
    durations = [int(1000 / fps)] * len(images)
    durations[-1] = 3000                                  # hold the last epoch before the GIF loops
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)


def main(argv=None, prices=None, **train_settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.lstm_plots import draw_network

    args = parse_args(argv)
    ticker = args.ticker.upper()
    if prices is None:
        from src.data import cache_path, load_prices
        if not cache_path(ticker, args.start).exists():
            print(f"No saved prices for {ticker}. Make them first with: python run.py {ticker}")
            return 1
        prices = load_prices(ticker, args.start)        # the saved prices; --refresh in run.py updates them
    args.folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    draw_network(ax)
    fig.savefig(args.folder / "lstm_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    run = train_one(prices, task=args.task, year=args.year, seed=args.seed, **train_settings)
    animate(run, ticker, args.folder / "lstm_training.gif")
    print(f"Saved {args.folder / 'lstm_network.png'} and {args.folder / 'lstm_training.gif'} "
          f"({ticker}, {'move size' if args.task == 'size' else 'direction'}, test year {run['year']}, "
          f"seed {args.seed}; early stopping would have chosen epoch {run['chosen']})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
