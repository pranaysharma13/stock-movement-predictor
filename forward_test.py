"""
The live forward test: one run per weekday evening (see .github/workflows/forward-test.yml).

    python forward_test.py                  all four stocks
    python forward_test.py SPY NVDA         just these

Each run downloads fresh prices, fills in yesterday's outcomes, writes down
tomorrow's predictions and rewrites the scorecard, in forward_test/. If one
stock fails (for example Yahoo refuses a download), the others still run, and
the exit code is 1 so the GitHub run shows as failed.

You can run it on your Mac too, but the official record is the one the
scheduled GitHub job commits: run it locally only to try it out, with
--folder pointing somewhere else, and don't commit the results. It downloads
its own prices and never touches the saved prices in data/ that the reports use.
"""


import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "forward_test"
START = "2005-01-01"


def code_version():
    """The git commit of the code making the predictions."""

    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def download(ticker, attempts=3, wait=30):
    """Fresh prices for one stock, trying again if Yahoo refuses. Not saved to data/."""

    from src.data import load_prices
    for attempt in range(1, attempts + 1):
        try:
            return load_prices(ticker, START, refresh=True, save=False)
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"{ticker}: download failed ({error}); trying again in {wait}s")
            time.sleep(wait)
def pd_now():
    import pandas as pd
    return pd.Timestamp.now(tz="UTC")


def parse_args(argv=None):
    from src.forward import TICKERS
    parser = argparse.ArgumentParser(description="Run one day of the live forward test.")
    parser.add_argument("tickers", nargs="*", default=TICKERS, help="stock symbols (default: the four)")
    parser.add_argument("--folder", type=Path, default=FOLDER, help="where the log and scorecard live")
    args = parser.parse_args(argv)
    args.tickers = list(dict.fromkeys(t.upper() for t in args.tickers))
    return args


def main(argv=None, wait=30, now=None):
    """now: the time the run happens (for tests); predictions are refused if the
    day they predict has already started (see forward.recorded_too_late)."""
    from src.forward import (add_predictions, check_predictions_unchanged, fill_outcomes,
                             load_log, predict_today, recorded_too_late, save_log, scorecard)
    args = parse_args(argv)
    now = pd_now() if now is None else now
    log_path = args.folder / "predictions.csv"
    before_text = log_path.read_text() if log_path.exists() else None
    before = load_log(log_path)

    prices, failed = {}, []
    for ticker in args.tickers:
        try:
            prices[ticker] = download(ticker, wait=wait)
        except Exception as error:
            failed.append(ticker)
            print(f"{ticker}: FAILED to download - {error}")

    log = fill_outcomes(before, prices)
    filled = int(log["outcome_date"].notna().sum() - before["outcome_date"].notna().sum())

    version, rows, already = code_version(), [], 0
    recorded_at = now.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    have = set(zip(log["made_on"].astype(str), log["ticker"].astype(str)))
    for ticker, p in prices.items():
        if (str(p.index[-1].date()), ticker) in have:
            already += 1           # a holiday or a second run: this close is already recorded
            continue
        if recorded_too_late(p.index[-1], now):
            print(f"{ticker}: not recorded - a prediction from the close of {p.index[-1].date()} "
                  f"would be made after the next session had opened")
            continue
        try:
            rows.append(predict_today(p, ticker, code_version=version, recorded_at=recorded_at))
        except Exception as error:
            failed.append(ticker)
            print(f"{ticker}: FAILED to predict - {error}")
    log, added = add_predictions(log, rows)

    check_predictions_unchanged(before, log)
    save_log(log, log_path, before_text=before_text)      # also checks every old row's text
    card = args.folder / "scorecard.md"
    text = scorecard(log)
    if not card.exists() or card.read_text() != text:
        card.write_text(text)

    for r in rows:
        print(f"{r['ticker']} from {r['made_on']}: direction {'up' if r['dir_pred'] else 'down'} "
              f"({r['dir_prob']:.0%}), move size {'big' if r['size_pred'] else 'small'} ({r['size_prob']:.0%}), "
              f"Recent swing says {'big' if r['recent_swing'] else 'small'}")
    print(f"Outcomes filled: {filled}. New predictions: {added}"
          + (" (none new: already recorded for this close)" if already and not added else "") + ".")
    if failed:
        print(f"Problems: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
    