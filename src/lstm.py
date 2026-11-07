"""
Phase 5: an LSTM, a neural network that reads the last 60 trading days in order.

Optional: it needs PyTorch (pip install -r requirements-lstm.txt) and only
runs when asked (python run.py NVDA --lstm). Only the report template
imports this file, and only then, so the other models and the live forward
test are unaffected by it.

What it sees. For each of the last WINDOW days, three numbers:
    ret_rel       that day's return (signed) divided by the typical move
    range_rel     that day's high-to-low swing divided by the typical move
    volume_ratio  that day's volume relative to its 20-day average
The typical move is the median daily move over the past 250 days, as in the
move-size task. These are close to raw prices: the network has to find
patterns itself, rather than being handed our summary features.

How it's tested. Exactly like the other models: walk-forward by year, the last
training day left out (its label is in the test year), and inputs scaled with
statistics from the training years only. Training stops early when the loss on
a validation year (the last training year) stops improving; the model is then
retrained on all training years for the number of epochs that worked best.
Each year is trained with several random seeds and their probabilities are
averaged, and the per-seed predictions are kept to show how much they disagree.
"""


import numpy as np
import pandas as pd

from .features import TYPICAL_WINDOW

WINDOW = 60
INPUTS = ["ret_rel", "range_rel", "volume_ratio"]
SEEDS = (0, 1, 2)


def torch_available():
    """True if PyTorch is installed (checked without loading it, which is slow)."""

    import importlib.util
    return importlib.util.find_spec("torch") is not None


def require_torch():
    if not torch_available():
        raise ImportError("The LSTM needs PyTorch. Install it with: pip install -r requirements-lstm.txt")


# ---------------------------- inputs ------------------------------------

def sequence_inputs(prices):
    """The three daily inputs, each using only data up to that day's close."""

    close = prices["Close"]
    ret = close / close.shift(1) - 1
    typical = ret.abs().rolling(TYPICAL_WINDOW).median()
    volume = prices["Volume"]
    return pd.DataFrame({
        "ret_rel": ret / typical,
        "range_rel": (prices["High"] - prices["Low"]) / close / typical,
        "volume_ratio": volume / volume.rolling(20).mean() - 1,
    }, index=prices.index)


def make_sequences(inputs, dates, window=WINDOW):
    """For each date, the window of inputs ending on that date (oldest first).

    Returns (X, kept): X has shape (days, window, inputs), kept the dates that
    had a full window with no missing values.
    """

    values = inputs.to_numpy(dtype=np.float32)
    position = inputs.index.get_indexer(dates)
    out, kept = [], []
    for date, i in zip(dates, position):
        if i < window - 1:
            continue
        seq = values[i - window + 1: i + 1]
        if np.isnan(seq).any():
            continue
        out.append(seq)
        kept.append(date)
    X = np.stack(out) if out else np.zeros((0, window, values.shape[1]), dtype=np.float32)
    return X, pd.DatetimeIndex(kept)


def fit_scaler(inputs, last_day):
    """Mean and spread of each input over days up to last_day only."""

    past = inputs.loc[:last_day]
    return past.mean().to_numpy(np.float32), past.std().replace(0, 1).to_numpy(np.float32)


def scale(X, scaler, clip=8.0):
    """Standardize, and clip extreme days (a crash) so one row can't dominate training."""

    mean, std = scaler
    return np.clip((X - mean) / std, -clip, clip).astype(np.float32)


# ---------------------------- the network ------------------------------------

def _network(n_inputs, hidden, dropout):
    import torch
    from torch import nn

    class SequenceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.out(self.drop(h[-1])).squeeze(-1)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    return SequenceClassifier()


def _train(X, y, seed, epochs, hidden, dropout, lr, batch, X_val=None, y_val=None, patience=3):
    """Train one network. With validation data, stop early and return the best epoch count."""

    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = _network(X.shape[2], hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    best_loss, best_epoch, stale = np.inf, 1, 0
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(Xt))
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            opt.step()
        if X_val is not None:
            # validation loss changes smoothly; validation accuracy jumps about from one epoch to the next
            p = np.clip(_probs(net, X_val), 1e-6, 1 - 1e-6)
            loss = float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)))
            if loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = loss, epoch, 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return net, best_epoch


def _probs(net, X):
    import torch
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(X))).numpy()


# ---------------------------- walk-forward ------------------------------------

def lstm_walk_forward(prices, data, first_year, seeds=SEEDS, window=WINDOW, epochs=20,
                      hidden=32, dropout=0.2, lr=1e-3, batch=128, record=None):
    """Walk-forward test of the LSTM, the same way as walk_forward() in evaluate.py.

    prices: the stock's prices (the sequences are built from them)
    data:   a dataset from build_dataset() or build_vol_dataset(): its index
            gives the days and its target column the labels
    record: optional list; each year appends (year, last training day,
            last validation-fit day, best epoch count), for checking

    Returns a DataFrame indexed like the test days of data, with:
      prob   the average of the seeds' probabilities that target is 1
      pred   1 if prob > 0.5, else 0
      prob seed <s>, one column per seed
    """

    require_torch()
    inputs = sequence_inputs(prices)
    X_all, days = make_sequences(inputs, data.index, window)
    y_all = data.loc[days, "target"].to_numpy()
    years = sorted(set(data.index.year[data.index.year >= first_year]))
    chunks = []
    for year in years:
        before = days.year < year
        # leave out the last training day: its label is the first test day's move
        train_idx = np.flatnonzero(before)[:-1]
        test_idx = np.flatnonzero(days.year == year)
        if len(test_idx) == 0:
            continue
        val_mask = days[train_idx].year == year - 1
        fit_idx, val_idx = train_idx[~val_mask][:-1], train_idx[val_mask]
        last_train, last_fit = days[train_idx[-1]], days[fit_idx[-1]]
        # Scaling for the final fit uses the training years only; for the
        # validation step, only the years before the validation year.
        scaler, fit_scaler_ = fit_scaler(inputs, last_train), fit_scaler(inputs, last_fit)
        X_train, X_test = scale(X_all[train_idx], scaler), scale(X_all[test_idx], scaler)
        X_fit, X_val = scale(X_all[fit_idx], fit_scaler_), scale(X_all[val_idx], fit_scaler_)
        probs, best_epochs = [], []
        for seed in seeds:
            # 1. find how many epochs help, using the last training year as validation
            _, best = _train(X_fit, y_all[fit_idx], seed, epochs, hidden, dropout, lr, batch,
                             X_val, y_all[val_idx])
            # 2. retrain on all training years for that many epochs
            net, _ = _train(X_train, y_all[train_idx], seed, best, hidden, dropout, lr, batch)
            probs.append(_probs(net, X_test))
            best_epochs.append(best)
        if record is not None:
            record.append((year, last_train, last_fit, best_epochs))
        chunk = pd.DataFrame({f"prob seed {s}": p for s, p in zip(seeds, probs)}, index=days[test_idx])
        chunk.insert(0, "prob", np.mean(probs, axis=0))
        chunks.append(chunk)
    out = pd.concat(chunks)
    out.insert(1, "pred", (out["prob"] > 0.5).astype(int))
    missing = data.index[data.index.year >= first_year].difference(out.index)
    if len(missing):
        raise ValueError(f"The LSTM has no full {window}-day window for {len(missing)} test days.")
    return out.reindex(data.index[data.index.year >= first_year])
