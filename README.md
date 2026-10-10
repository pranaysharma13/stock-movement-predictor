# Stock Direction Predictor

Can a machine learning model predict whether the S&P 500 (SPY) closes higher tomorrow, better than simple rules can?

This project predicts **direction** (up or down), not price, and tests the answer honestly:

- **Walk-forward testing:** every prediction comes from a model trained only on earlier years.
- **Real baselines:** models must beat "always predict up," not just a coin flip.
- **Noise check:** every accuracy comes with a 95% range.
- **Backtest with trading costs,** compared against buy-and-hold.
- **Luck check:** any backtest edge is compared with sitting out the same number of random days.
- **Pipeline checks both ways:** the same code must find nothing in pure-chance prices (no leaked future information) and must find a pattern planted on purpose (it can detect a real signal).

For the full story (decisions, how to read every output, results, issues found and fixed, and lessons learned), see the [Project Log](PROJECT_LOG.md).

## Results

*Fill in after running the notebook: headline accuracy vs. "always up", backtest vs. buy-and-hold, and your one-paragraph conclusion.*

## Project structure

```
Stock_Movement_Predictor/
├── src/
│   ├── data.py         download and cache prices; fake prices for the pipeline checks
│   ├── features.py     features (returns, moving averages, volatility, volume) and label
│   └── evaluate.py     models, walk-forward testing, baselines, accuracy, backtest, luck check
├── notebooks/
│   └── analysis.ipynb  the full analysis, start to finish
├── tests/              automated checks, run with `pytest`
│   ├── test_data.py
│   ├── test_features.py
│   └── test_evaluate.py
├── PROJECT_LOG.md      decisions, results, issues and lessons, kept up to date
├── pytest.ini          test settings
└── requirements.txt
```

The logic lives in `src/` as small, plain functions. The notebook imports them and tells the story.

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
jupyter notebook notebooks/analysis.ipynb
```

Then run all cells (Run → Run All Cells). The first run downloads SPY prices from 2005 to today and saves them in `data/`; later runs use that saved copy.

### Every time you come back

Open Terminal and run:

```bash
cd Stock_Movement_Predictor      # use the full path if you're not in the parent folder,
                                 # e.g. cd ~/Documents/Stock_Movement_Predictor
source .venv/bin/activate        # your prompt should now start with (.venv)
jupyter notebook notebooks/analysis.ipynb
```

No need to create the environment or install anything again.

### When you're done

1. In the browser, save the notebook (File → Save Notebook).
2. In Terminal, press `Ctrl + C` to stop Jupyter (confirm with `y` if asked).
3. Run `deactivate` to leave the virtual environment, or just close Terminal.

### Getting the latest prices

The saved data in `data/` doesn't update by itself. To pull the newest days, either delete the CSV in `data/` and run all cells again, or change the load line in the notebook to:

```python
prices = load_prices(TICKER, START, refresh=True)
```

If you download while the US market is still open, today's row is left out automatically, because its price isn't final until the close (4 pm New York time).

## Running the tests

With the virtual environment active, from the project folder:

```bash
pytest
```

It takes a few seconds and needs no internet: every test runs on small fake price series. A green line of dots and "passed" means everything works; a failure names the test and shows what went wrong. Run it after any change to `src/`.

What the tests check:

| Test file | What it checks |
|---|---|
| `test_features.py` | No feature uses future prices; labels match the next day's real move; known values on a simple price series |
| `test_data.py` | Today's unfinished row is dropped; saved files and the download path work (with a fake yfinance); fake prices look right and the planted pattern really reverses about 60% of the time |
| `test_evaluate.py` | Walk-forward never trains on the year it predicts; default calls work; backtest and performance arithmetic match hand calculations; the luck check spots a perfect strategy; the pipeline finds the planted pattern and finds nothing in noise |

The tests were checked by planting realistic bugs (a feature that peeks at tomorrow, training on the test year, removing the unfinished-day fix): each one makes at least one test fail.

## Settings

At the top of the notebook:

| Setting | Default | Meaning |
|---|---|---|
| `TICKER` | `"SPY"` | Any Yahoo Finance symbol |
| `START` | `"2005-01-01"` | First date of data |
| `FIRST_TEST_YEAR` | `None` | First year predicted; `None` picks the first year with about 4 years of history before it (2010 for SPY) |
| `COST` | `0.001` | Cost per buy or sell in the backtest (0.1%) |

## Features

The models use five features, one per idea:

| Feature | Meaning |
|---|---|
| `ret_1d` | Return over the last day |
| `ret_5d` | Return over the last 5 days |
| `dist_ma50` | Distance of price from its 50-day average |
| `vol_20d` | Standard deviation of daily returns over 20 days |
| `volume_ratio` | Today's volume vs. its 20-day average |

Three more are calculated but left out because they nearly duplicate a kept one: `ret_10d`, `ret_20d` and `dist_ma10`. The notebook shows the overlap and checks that adding them back doesn't help.

## Limitations

- The backtest assumes trades happen exactly at the closing price.
- Costs are a flat percentage; real costs vary.
- One ticker and daily data only.