# Stock Direction Predictor

Can a machine learning model predict tomorrow's stock move, better than simple rules can? Tested on SPY (the S&P 500), NVDA, AAPL and GOOGL, and runnable for any stock with one command.

The project asks two questions about tomorrow, never the exact price:

1. **Direction:** will the stock close higher? Judged against "always predict up."
2. **Move size:** will the move, up or down, be bigger than usual? Judged against "Recent swing", a one-line rule: predict a big day if today's high-to-low swing was wider than usual for the past year.

Both are tested honestly:

- **Walk-forward testing:** every prediction comes from a model trained only on earlier years.
- **Real baselines:** models must beat "always predict up," not just a coin flip; for move size, they must beat the strongest simple rule we found.
- **Paired test against rules:** a model and a rule predict the same days, so only the days they disagree count.
- **Noise check:** every accuracy comes with a 95% range.
- **Backtest with trading costs,** compared against buy-and-hold.
- **Luck check:** any backtest edge is compared with sitting out the same number of random days.
- **Pipeline checks both ways:** the same code must find nothing in pure-chance prices (no leaked future information) and must find a pattern planted on purpose (it can detect a real signal).
- **Lucky-winners correction:** when several stocks and models are compared, results must clear a stricter bar, because some will "win" by chance alone.
- **Does it help?** Move-size predictions are turned into a strategy that halves the position on predicted big days, deciding daily or once a week, judged on risk (Sharpe ratio, worst drop) after trading costs.

For the full story (decisions, how to read every output, results, issues found and fixed, and lessons learned), see the [Project Log](PROJECT_LOG.md).

## Results

*Fill in after running the reports: headline accuracy vs. "always up", backtest vs. buy-and-hold, and your one-paragraph conclusion.* The reports are in [`reports/`](reports/); start with [`reports/comparison.ipynb`](reports/comparison.ipynb).

## Project structure

```
Stock_Movement_Predictor/
├── run.py                       one command to create the reports
├── src/
│   ├── data.py                  download and cache prices; fake prices for the pipeline checks
│   ├── features.py              features and labels for both tasks
│   ├── evaluate.py              models, walk-forward testing, baselines, accuracy, backtest, luck check
│   └── report.py                per-stock summaries and the cross-stock comparison
├── notebooks/
│   ├── template.ipynb           the analysis for one stock, both parts (run.py fills in the ticker)
│   └── comparison_template.ipynb  the side-by-side comparison
├── reports/                     created by run.py
│   ├── SPY.ipynb, NVDA.ipynb, ...   one finished report per stock
│   ├── comparison.ipynb
│   └── summaries/               each stock's headline numbers (used by the comparison)
├── tests/                       automated checks, run with `pytest`
├── PROJECT_LOG.md               decisions, results, issues and lessons, kept up to date
├── pytest.ini                   test settings
└── requirements.txt
```

The logic lives in `src/` as small, plain functions. The notebooks import them and tell the story.

## How to run

You need Python 3.10 or newer.

### First time only

```bash
cd Stock_Movement_Predictor

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest                           # optional: check everything works (a few seconds)
```

### Creating reports

```bash
python run.py SPY                       # one stock    -> reports/SPY.ipynb
python run.py SPY NVDA AAPL GOOGL       # several      -> one report each + reports/comparison.ipynb
python run.py AAPL --refresh            # re-download the latest prices first
python run.py --help                    # all options
```

Each report takes about 45–50 seconds. The first run of a stock downloads its prices from 2005 to today and saves them in `data/`; later runs use that saved copy, so results stay the same until you `--refresh`. If a ticker fails (a typo, no internet), the others still run and the message says which one failed.

Open a finished report in VS Code or with `jupyter notebook reports/NVDA.ipynb`. It already contains every table and chart; there's nothing to run.

| Option | Default | Meaning |
|---|---|---|
| `--start` | `2005-01-01` | First date of data |
| `--first-test-year` | automatic | First year predicted; automatic picks the first year with about 4 years of history before it (2010 for these stocks) |
| `--cost` | `0.001` | Cost per buy or sell in the backtest (0.1%) |
| `--refresh` | off | Re-download prices before running |
| `--no-comparison` | off | Skip the comparison when running several stocks |

### Every time you come back

```bash
cd Stock_Movement_Predictor      # use the full path if you're not in the parent folder,
                                 # e.g. cd ~/Documents/Stock_Movement_Predictor
source .venv/bin/activate        # your prompt should now start with (.venv)
```

Then `python run.py ...` as above. Run `deactivate` when you're done, or just close the terminal.

### Changing the analysis itself

Edit `notebooks/template.ipynb` (or the code in `src/`), then run `python run.py` again to regenerate the reports. Edits made directly in a file in `reports/` are overwritten the next time that stock is run. Keep the template saved without outputs, so it stays small and clean.

If you download while the US market is still open, today's row is left out automatically, because its price isn't final until the close (4 pm New York time).

## Running the tests

With the virtual environment active, from the project folder:

```bash
pytest
```

It takes a few seconds and needs no internet: every test runs on small fake price series. A green line of dots and "passed" means everything works; a failure names the test and shows what went wrong. Run it after any change to `src/`, `run.py` or the templates.

| Test file | What it checks |
|---|---|
| `test_features.py` | No feature uses future prices, for either task; labels match the next day's real move; the "typical move" and the Recent swing rule use only the past year; known values on simple price series |
| `test_data.py` | Today's unfinished row is dropped; saved files and the download path work (with a fake yfinance); fake prices look right; the planted pattern really reverses about 60% of the time; clustered volatility really clusters and the random walk doesn't |
| `test_evaluate.py` | Walk-forward never trains on the year it predicts; default calls work; backtest arithmetic matches hand calculations, including half positions; the paired test matches a hand calculation; weekly positions only change at week ends (holiday weeks included); the luck check works on returns, on Sharpe ratio and on whole weeks; both pipeline checks pass for both tasks |
| `test_report.py` | Summaries for both tasks are correct and saveable, with each p-value against the rule it names; the lucky-winners correction counts properly, is applied to each task separately, and counts daily and weekly strategies; `run.py` reads its options and keeps going when one stock fails; both templates have their settings cell and are saved without outputs |

The tests were checked by planting realistic bugs (features that peek at tomorrow, training on the test year, removing the unfinished-day fix, charging half positions as full trades, a baseline that cheats, a paired test that counts agreement days, weekly decisions on the wrong day): each one makes at least one test fail.

## Features

### Part 1: direction

The models use five features, one per idea:

| Feature | Meaning |
|---|---|
| `ret_1d` | Return over the last day |
| `ret_5d` | Return over the last 5 days |
| `dist_ma50` | Distance of price from its 50-day average |
| `vol_20d` | Standard deviation of daily returns over 20 days |
| `volume_ratio` | Today's volume vs. its 20-day average |

Three more are calculated but left out because they nearly duplicate a kept one: `ret_10d`, `ret_20d` and `dist_ma10`. Each report shows the overlap and checks that adding them back doesn't help.

### Part 2: move size

A day is **big** when its move, up or down, is larger than the stock's typical move: the median daily move over the previous 250 trading days. Every feature is measured relative to that typical move, so "big" means the same for a calm fund and a wild stock.

| Feature | Meaning |
|---|---|
| `abs_ret_rel` | Today's move size |
| `vol_5d_rel` | How jumpy the last week was |
| `vol_20d_rel` | How jumpy the last month was |
| `range_rel` | How far the price swung within today, from low to high |
| `volume_ratio` | Today's volume vs. its 20-day average |

The baselines need no model: "same as today" (big tomorrow if today's move was bigger than usual) and "Recent swing" (big tomorrow if today's high-to-low swing was wider than its median over the past 250 days). Recent swing is the bar.

## Limitations

- The backtest assumes trades happen exactly at the closing price.
- Costs are a flat percentage; real costs vary.
- Daily data only.
- The daily half-on-big strategy switches between full and half positions often, so its result depends heavily on the assumed trading cost; the weekly version trades about a quarter as often.
- NVDA, AAPL and GOOGL were chosen knowing they became huge winners, so their buy-and-hold results look good partly in hindsight. Each report says so.