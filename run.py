"""
Run the full analysis for one or more stocks.

    python run.py NVDA                     one stock  -> reports/NVDA.ipynb
    python run.py SPY NVDA AAPL GOOGL      several    -> one report each, plus reports/comparison.ipynb
    python run.py AAPL --refresh           re-download the latest prices first
    python run.py SPY NVDA --comparison-only   rebuild only the comparison, from saved summaries

Each report is notebooks/template.ipynb with the settings filled in and every
cell run, saved with its tables and charts. Run `python run.py --help` for all options.
"""


import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "notebooks" / "template.ipynb"
COMPARISON_TEMPLATE = ROOT / "notebooks" / "comparison_template.ipynb"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a report notebook for each stock, and a comparison when there are several."
    )
    parser.add_argument("tickers", nargs="+", help="Yahoo Finance symbols, e.g. SPY NVDA AAPL GOOGL")
    parser.add_argument("--start", default="2005-01-01", help="first date of data (default: 2005-01-01)")
    parser.add_argument("--first-test-year", type=int, default=None,
                        help="first year to predict (default: automatic, about 4 years after the data starts)")
    parser.add_argument("--cost", type=float, default=0.001,
                        help="cost per buy or sell in the backtest (default: 0.001 = 0.1%%)")
    parser.add_argument("--refresh", action="store_true", help="re-download prices before running")
    parser.add_argument("--no-comparison", action="store_true", help="skip the comparison report")
    parser.add_argument("--comparison-only", action="store_true",
                        help="don't re-run the stocks; rebuild the comparison from their saved summaries")
    parser.add_argument("--no-interest", action="store_true",
                        help="cash earns nothing in the backtests (default: Treasury-bill interest)")
    args = parser.parse_args(argv)

    # Upper case, no duplicates, original order kept
    args.tickers = list(dict.fromkeys(t.strip().upper() for t in args.tickers if t.strip()))
    if not args.tickers:
        parser.error("give at least one ticker")
    if args.cost < 0:
        parser.error("--cost can't be negative")
    if args.comparison_only and args.no_comparison:
        parser.error("--comparison-only and --no-comparison can't be used together")
    return args


def partial_path(output):
    """Where a report is written while it runs: reports/SPY.partial.ipynb."""

    return output.with_name(f"{output.stem}.partial.ipynb")


def run_notebook(template, output, parameters):
    """Fill in a template's settings cell, run every cell, and save the result.

    The notebook is written to a .partial.ipynb file first and only replaces
    output once every cell has run, so a failed run never overwrites the last
    good report. After a failure, the partial file shows where it stopped.
    """

    import papermill as pm

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(output)
    pm.execute_notebook(
        str(template), str(partial),
        parameters=parameters,
        cwd=str(ROOT),
        progress_bar=False,
    )
    os.replace(partial, output)


def last_line(error):
    """The most useful line of an error message, or the error's type if it has no message."""

    lines = str(error).strip().splitlines()
    return lines[-1] if lines else type(error).__name__


def main(argv=None):
    args = parse_args(argv)
    from src.data import CASH_TICKER, load_prices
    from src.report import SUMMARY_DIR, safe_name

    done, failed = [], []
    if args.comparison_only:
        for ticker in args.tickers:
            if (SUMMARY_DIR / f"{safe_name(ticker)}.json").exists():
                done.append(ticker)
            else:
                failed.append(ticker)
                print(f"{ticker}: no saved summary; run `python run.py {ticker}` first.")
    else:
        if args.refresh and not args.no_interest:
            try:
                load_prices(CASH_TICKER, args.start, refresh=True)
            except Exception as error:
                print(f"Interest rates ({CASH_TICKER}): refresh FAILED - {last_line(error)}")
        for ticker in args.tickers:
            started = time.time()
            output = REPORTS / f"{safe_name(ticker)}.ipynb"
            print(f"{ticker}: running...", flush=True)
            try:
                if args.refresh:
                    load_prices(ticker, args.start, refresh=True)
                run_notebook(TEMPLATE, output, {
                    "TICKER": ticker,
                    "START": args.start,
                    "FIRST_TEST_YEAR": args.first_test_year,
                    "COST": args.cost,
                    "CASH_INTEREST": not args.no_interest,
                })
            except Exception as error:  # keep going with the other stocks
                failed.append(ticker)
                print(f"{ticker}: FAILED - {last_line(error)}")
                partial = partial_path(output)
                if partial.exists():
                    print(f"   The partly run report is in {partial.relative_to(ROOT)}; "
                          f"the red cell shows where it stopped.")
                if output.exists():
                    print(f"   The last good report, {output.relative_to(ROOT)}, is unchanged.")
                continue
            done.append(ticker)
            print(f"{ticker}: saved {output.relative_to(ROOT)} ({time.time() - started:.0f}s)")

    enough = len(done) >= (1 if args.comparison_only else 2)
    if enough and not args.no_comparison:
        output = REPORTS / "comparison.ipynb"
        print(f"Comparison of {', '.join(done)}: running...", flush=True)
        try:
            run_notebook(COMPARISON_TEMPLATE, output, {"TICKERS": done})
            print(f"Comparison: saved {output.relative_to(ROOT)}")
        except Exception as error:
            failed.append("comparison")
            print(f"Comparison: FAILED - {last_line(error)}")

    if failed:
        print(f"\nDone with problems: {', '.join(failed)} failed.")
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
