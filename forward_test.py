"""
The live forward test: one run per weekday evening (see .github/workflows/forward-test.yml).

    python forward_test.py                  all four stocks
    python forward_test.py SPY NVDA         just these

Each run downloads fresh prices, fills in yesterday's outcomes, writes down
tomorrow's predictions and rewrites the scorecard, in forward_test/. If one
stock fails (for example Yahoo refuses a download), the others still run, and
the exit code is 1 so the GitHub run shows as failed.

You can run it on your Mac too, but the official record is the one the
scheduled GitHub job commits: run it locally only to try it out, and don't
commit the results.
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
    """Fresh prices for one stock, trying again if Yahoo refuses."""

    from src.data import load_prices
    for attempt in range(1, attempts + 1):
        try:
            return load_prices(ticker, START, refresh=True)
        except Exception as error:
            if attempt == attempts:
                raise
            print(f"{ticker}: download failed ({error}); trying again in {wait}s")
            time.sleep(wait)


def parse_args(argv=None):
    from src.forward import TICKERS
    parser = argparse.ArgumentParser(description="Run one day of the live forward test.")
    parser.add_argument("tickers", nargs="*", default=TICKERS, help="stock symbols (default: the four)")
    parser.add_argument("--folder", type=Path, default=FOLDER, help="where the log and scorecard live")
    args = parser.parse_args(argv)
    args.tickers = list(dict.fromkeys(t.upper() for t in args.tickers))
    return args


def main(argv=None, wait=30):
    from src.forward import (add_predictions, check_predictions_unchanged, fill_outcomes,
                             load_log, predict_today, save_log, scorecard)
    args = parse_args(argv)
    log_path = args.folder / "predictions.csv"
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

    version, rows = code_version(), []
    for ticker, p in prices.items():
        try:
            rows.append(predict_today(p, ticker, code_version=version))
        except Exception as error:
            failed.append(ticker)
            print(f"{ticker}: FAILED to predict - {error}")
    log, added = add_predictions(log, rows)

    check_predictions_unchanged(before, log)
    save_log(log, log_path)
    (args.folder / "scorecard.md").write_text(scorecard(log))

    for r in rows:
        print(f"{r['ticker']} from {r['made_on']}: direction {'up' if r['dir_pred'] else 'down'} "
              f"({r['dir_prob']:.0%}), move size {'big' if r['size_pred'] else 'small'} ({r['size_prob']:.0%}), "
              f"Recent swing says {'big' if r['recent_swing'] else 'small'}")
    print(f"Outcomes filled: {filled}. New predictions: {added}"
          + (" (none new: already recorded for this close)" if rows and not added else "") + ".")
    if failed:
        print(f"Problems: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
    