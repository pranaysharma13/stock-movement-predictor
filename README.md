# Stock Direction Predictor

Can a machine learning model predict tomorrow's stock move, better than simple rules can? Tested on SPY (the S&P 500), NVDA, AAPL and GOOGL, and runnable for any stock with one command.

The project asks two questions about tomorrow, never the exact price:

1. **Direction:** will the stock close higher? Judged against "always predict up."
2. **Move size:** will the move, up or down, be bigger than usual? Judged against "Recent swing", a one-line rule: predict a big day if today's high-to-low swing was wider than usual for the past year.

Both are tested honestly:

- **Walk-forward testing:** every prediction comes from a model trained only on earlier years.
- **Real baselines:** models must beat "always predict up," not just a coin flip; for move size, they must beat the strongest simple rule we found.
- **Paired test against every baseline:** a model and a rule predict the same days, so only the days they disagree count. The same goes for the ranges in the comparison charts.
- **Noise check:** every accuracy comes with a 95% range.
- **Backtest with trading costs and interest on cash,** compared against buy-and-hold; Sharpe ratios use returns above the Treasury-bill rate.
- **Luck check:** any backtest edge is compared with sitting out the same number of random days.
- **Pipeline checks both ways:** the same code must find nothing in pure-chance prices (no leaked future information) and must find a pattern planted on purpose (it can detect a real signal).
- **Lucky-winners correction:** when several stocks and models are compared, results must clear a stricter bar (the Holm correction), because some will "win" by chance alone.
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

You need Python 3.11 or newer.

### First time only

```bash
cd Stock_Movement_Predictor

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest                           # optional: check everything works (a few seconds)
pip freeze > requirements-lock.txt   # optional: record the exact versions you used
```

`requirements-lock.txt` lists the exact version of every library. Libraries such as yfinance change often; `pip install -r requirements-lock.txt` recreates exactly the setup the reports were made with.

### Creating reports

```bash
python run.py SPY                       # one stock    -> reports/SPY.ipynb
python run.py SPY NVDA AAPL GOOGL       # several      -> one report each + reports/comparison.ipynb
python run.py AAPL --refresh            # re-download the latest prices first
python run.py SPY NVDA --comparison-only   # rebuild only the comparison, from saved summaries
python run.py --help                    # all options
```

Each report takes about 50 seconds. The first run of a stock downloads its prices from 2005 to today and saves them in `data/`; later runs use that saved copy, so results stay the same until you `--refresh`. Treasury-bill rates (`^IRX`) are downloaded and saved the same way. If a ticker fails (a typo, no internet), the others still run, the message says which one failed, and its last good report is left unchanged; the partly run notebook is saved as `reports/<TICKER>.partial.ipynb` so you can see where it stopped.

Open a finished report in VS Code or with `jupyter notebook reports/NVDA.ipynb`. It already contains every table and chart; there's nothing to run.

| Option | Default | Meaning |
|---|---|---|
| `--start` | `2005-01-01` | First date of data |
| `--first-test-year` | automatic | First year predicted; automatic picks the first year with about 4 years of history before it (2010 for these stocks) |
| `--cost` | `0.001` | Cost per buy or sell in the backtest (0.1%) |
| `--refresh` | off | Re-download prices (and interest rates) before running |
| `--no-comparison` | off | Skip the comparison when running several stocks |
| `--comparison-only` | off | Don't re-run the stocks; rebuild the comparison from their saved summaries |
| `--no-interest` | off | Cash earns nothing in the backtests (by default it earns the Treasury-bill rate) |

### Every time you come back

```bash
cd Stock_Movement_Predictor      # use the full path if you're not in the parent folder,
                                 # e.g. cd ~/Documents/Stock_Movement_Predictor
source .venv/bin/activate        # your prompt should now start with (.venv)
```

Then `python run.py ...` as above. Run `deactivate` when you're done, or just close the terminal.

### Changing the analysis itself

Edit `notebooks/template.ipynb` (or the code in `src/`), then run `python run.py` again to regenerate the reports. Edits made directly in a file in `reports/` are overwritten the next time that stock is run. Keep the template saved without outputs, so it stays small and clean.

If you download before 6 pm New York time, today's row is left out automatically, because its prices may still change after the 4 pm close. Rows where High and Low don't contain Open and Close (an occasional glitch in Yahoo's data) are repaired, and each report lists them.

Tickers without trading volume, such as `^VIX` or currency pairs, stop with a clear message: one of the features needs volume.

## Running the tests

With the virtual environment active, from the project folder:

```bash
pytest
```

It takes a few seconds and needs no internet: every test runs on small fake price series. A green line of dots and "passed" means everything works; a failure names the test and shows what went wrong. Run it after any change to `src/`, `run.py` or the templates.

| Test file | What it checks |
|---|---|
| `test_features.py` | No feature uses future prices, for either task; labels match the next day's real move; the "typical move" and the Recent swing rule use only the past year; known values on simple price series; clear errors for prices without volume or with too little history |
| `test_data.py` | Today's unfinished row is dropped, including shortly after the close; bad High/Low rows are repaired; saved files, safe file names and the download path work (with a fake yfinance); interest rates become daily rates; fake prices look right; the planted pattern really reverses about 60% of the time; clustered volatility really clusters and the random walk doesn't |
| `test_evaluate.py` | Walk-forward never trains on the year it predicts, nor on the day whose label falls in it; default calls work; backtest arithmetic matches hand calculations, including half positions and interest on cash; Sharpe uses returns above cash; the paired test and its range match hand calculations; weekly positions only change at week ends (holiday weeks and the last week included); the luck check works on returns, Sharpe ratio and whole weeks, never reports 0, and gives random versions the same interest; both pipeline checks pass for both tasks |
| `test_report.py` | Summaries for both tasks are correct and saveable, with each p-value the paired test against the baseline it names; the Holm correction matches hand calculations, is applied to each task separately, and counts daily and weekly strategies; a failed run keeps the last good report; `--comparison-only` and `--no-interest` work; `run.py` reads its options and keeps going when one stock fails; both templates have their settings cell and are saved without outputs |

The tests were checked by planting realistic bugs (features that peek at tomorrow, training on the test year, removing the unfinished-day fix, charging half positions as full trades, a baseline that cheats, a paired test that counts agreement days, weekly decisions on the wrong day, training on the day whose label is in the test year, cash that earns nothing, a failed run that overwrites the last good report): each one makes at least one test fail.

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
- Cash earns the 13-week Treasury-bill yield, spread evenly over trading days; a real account would earn a little less.
- Costs are a flat percentage; real costs vary.
- Daily data only.
- The daily half-on-big strategy switches between full and half positions often, so its result depends heavily on the assumed trading cost; the weekly version trades about a quarter as often.
- NVDA, AAPL and GOOGL were chosen knowing they became huge winners, so their buy-and-hold results look good partly in hindsight. Each report says so.