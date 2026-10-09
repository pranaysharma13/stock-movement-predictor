# Stock Direction Predictor

Can a machine learning model predict whether the S&P 500 (SPY) closes higher tomorrow, better than simple rules can?

This project predicts **direction** (up or down), not price, and tests the answer honestly:

- **Walk-forward testing:** every prediction comes from a model trained only on earlier years.
- **Real baselines:** models must beat "always predict up," not just a coin flip.
- **Noise check:** every accuracy comes with a 95% range.
- **Backtest with trading costs,** compared against buy-and-hold.
- **Random-walk sanity check:** the same pipeline on pure-chance prices, to catch any leak of future information.

## Results

*Fill in after running the notebook: headline accuracy vs. "always up", backtest vs. buy-and-hold, and your one-paragraph conclusion.*

## Project structure

```
Stock_Movement_Predictor/
├── src/
│   ├── data.py         download and cache prices; fake random-walk prices
│   ├── features.py     features (returns, moving averages, volatility, volume) and label
│   └── evaluate.py     models, walk-forward testing, baselines, accuracy, backtest
├── notebooks/
│   └── analysis.ipynb  the full analysis, start to finish
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

Download after the US market closes (4 pm New York time) so today's price is final.

## Settings

At the top of the notebook:

| Setting | Default | Meaning |
|---|---|---|
| `TICKER` | `"SPY"` | Any Yahoo Finance symbol |
| `START` | `"2005-01-01"` | First date of data |
| `FIRST_TEST_YEAR` | `2010` | First year predicted; earlier years are only for training |
| `COST` | `0.001` | Cost per buy or sell in the backtest (0.1%) |

## Features

| Feature | Meaning |
|---|---|
| `ret_1d`, `ret_5d`, `ret_10d`, `ret_20d` | Return over the last 1, 5, 10, 20 days |
| `dist_ma10`, `dist_ma50` | Distance of price from its 10- and 50-day average |
| `vol_20d` | Standard deviation of daily returns over 20 days |
| `volume_ratio` | Today's volume vs. its 20-day average |

## Limitations

- The backtest assumes trades happen exactly at the closing price.
- Costs are a flat percentage; real costs vary.
- One ticker and daily data only.