"""
The LSTM's live forward test, for move size (phase 6.1): one run per weekday
evening, right after logistic regression's (see .github/workflows/forward-test.yml).

    python forward_test_lstm.py                  all four stocks
    python forward_test_lstm.py SPY --folder /tmp/forward-try

Each run downloads fresh prices, fills in yesterday's outcomes and writes
down tomorrow's predictions in forward_test/lstm_predictions.csv, then
rewrites forward_test/lstm_scorecard.md. On the first evening of a month it
first retrains each stock's LSTM and saves it in forward_test/lstm_models/;
on other evenings it uses the saved one. Needs PyTorch.

It is separate from forward_test.py on purpose: if PyTorch can't be
installed or this run fails, logistic regression's forward test is unaffected.
"""


import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "forward_test"


def parse_args(argv=None):
    from src.forward import TICKERS
    parser = argparse.ArgumentParser(description="Run one day of the LSTM's live forward test.")
    parser.add_argument("tickers", nargs="*", default=TICKERS, help="stock symbols (default: the four)")
    parser.add_argument("--folder", type=Path, default=FOLDER, help="where the logs, scorecards and models live")
    args = parser.parse_args(argv)
    args.tickers = list(dict.fromkeys(t.upper() for t in args.tickers))
    return args


def main(argv=None, wait=30, now=None, settings=None):
    """now: the time the run happens (for tests). settings: training settings
    passed to train_live() (for tests; the real run uses the walk-forward test's)."""

    import pandas as pd

    from forward_test import code_version, download
    from src import forward_lstm as fl
    from src.forward import load_log as load_logreg_log
    from src.forward import recorded_too_late
    from src.lstm import require_torch

    require_torch()
    args = parse_args(argv)
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    log_path = args.folder / "lstm_predictions.csv"
    models_folder = args.folder / "lstm_models"
    before_text = log_path.read_text() if log_path.exists() else None
    log = fl.load_log(log_path)

    prices, failed = {}, []
    for ticker in args.tickers:
        try:
            prices[ticker] = download(ticker, wait=wait)
        except Exception as error:
            failed.append(ticker)
            print(f"{ticker}: FAILED to download - {error}")

    filled_before = int(log["outcome_date"].notna().sum())
    log = fl.fill_outcomes(log, prices)
    filled = int(log["outcome_date"].notna().sum()) - filled_before

    version, rows, already, models = code_version(), [], 0, {}
    recorded_at = now.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    have = set(zip(log["made_on"].astype(str), log["ticker"].astype(str)))
    for ticker in args.tickers:
        path = models_folder / f"{ticker}.pt"
        model = fl.load_model(path)
        if ticker in prices:
            p = prices[ticker]
            made_on = p.index[-1]
            if (str(made_on.date()), ticker) in have:
                already += 1
            elif recorded_too_late(made_on, now):
                print(f"{ticker}: not recorded - a prediction from the close of {made_on.date()} "
                      f"would be made after the next session had opened")
            else:
                try:
                    if fl.needs_training(model, made_on):
                        print(f"{ticker}: training this month's LSTM on prices up to {made_on.date()}...")
                        model = fl.train_live(p, **(settings or {}))
                        fl.save_model(model, path)
                    rows.append(fl.predict_lstm_today(p, ticker, model, code_version=version,
                                                      recorded_at=recorded_at))
                except Exception as error:
                    failed.append(ticker)
                    print(f"{ticker}: FAILED to predict - {error}")
        if model is not None:
            models[ticker] = model
    log, added = fl.add_predictions(log, rows)

    fl.save_log(log, log_path, before_text=before_text)       # checks every old row's text
    card = args.folder / "lstm_scorecard.md"
    text = fl.scorecard(log, load_logreg_log(args.folder / "predictions.csv"), models)
    if not card.exists() or card.read_text() != text:
        card.write_text(text)

    for r in rows:
        print(f"{r['ticker']} from {r['made_on']}: LSTM says move size {'big' if r['size_pred'] else 'small'} "
              f"({r['size_prob']:.0%}, seeds {'agree' if r['seeds_agree'] else 'disagree'}), "
              f"Recent swing says {'big' if r['recent_swing'] else 'small'}")
    print(f"LSTM outcomes filled: {filled}. New predictions: {added}"
          + (" (none new: already recorded for this close)" if already and not added else "") + ".")
    if failed:
        print(f"Problems: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
