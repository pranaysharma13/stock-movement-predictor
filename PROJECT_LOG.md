# Stock Movement Predictor — Project Log

*Last updated: 9 October 2026. A running record of decisions, results, issues and lessons, kept up to date with the help of Claude*

## Overview

Neither model predicts SPY's daily direction better than the simple rule "always predict up", and our checks show the reason is the data, not the code: past prices hold no usable signal for tomorrow's direction.

- **Goal:** predict whether a stock closes higher tomorrow, using only information available at today's close, and test the answer honestly.
- **Approach:** daily SPY prices from 2005 to today, 8 price-based features, two models (logistic regression and gradient boosting), walk-forward testing from 2010, compared against simple baselines and a backtest with trading costs.
- **Status:** the first full run on real SPY data is done. The next round of improvements is planned but not built yet.
- **Code:** logic lives in `src/`; the analysis is told in `notebooks/analysis.ipynb`.

This log is updated whenever we find an issue, fix one or add a feature. The Changelog at the end lists every change, newest first.

## Setup and key decisions

Every choice below was made to keep the test honest: no peeking at the future, and a fair bar to beat.

| Decision | Choice | Why |
| --- | --- | --- |
| What to predict | Direction: 1 if tomorrow's close is higher than today's, else 0 | A predicted-vs-actual price chart looks impressive even for a useless model, because tomorrow's price is always close to today's |
| Data | SPY daily prices, adjusted for dividends and splits, 1 Jan 2005 to today, via yfinance | One liquid fund with a long history; adjusted prices give the real return an investor got |
| Data caching | First run downloads and saves `data/SPY_from_2005-01-01.csv`; later runs load that file | Results stay the same between runs until you choose to refresh |
| Project layout | Logic in `src/` (`data.py`, `features.py`, `evaluate.py`); one notebook tells the story | Clean, reusable code plus a readable report |
| Features | 8: returns over 1, 5, 10 and 20 days; distance from 10- and 50-day averages; 20-day volatility; volume vs its 20-day average | Momentum, trend, risk and activity, each using only data up to today's close |
| Models | Logistic regression (features scaled first) and gradient boosting (shallow trees: depth 3, learning rate 0.05, 200 trees), both from scikit-learn | Start simple and readable; LSTM left as a stretch goal |
| Testing | Walk-forward: train on 2005–2009, predict 2010; retrain on 2005–2010, predict 2011; and so on to today | Every prediction comes from a model that never saw that year |
| Baselines | Coin flip, always up, repeat last move | Markets rise on about 55% of days, so "always up" is the real bar, not 50% |
| Noise check | 95% range around each accuracy, plus a p-value against a coin flip | Shows whether a difference could be luck |
| Backtest | Hold SPY into tomorrow when the model says up, else cash; 0.1% cost per buy or sell; compared with buy-and-hold | Accuracy alone doesn't show whether a model would make money |
| Sanity check | The same pipeline on fake random-walk prices | If a model beats pure chance, the code is leaking future data |
| Where it runs | Locally (Mac) | Claude's workspace can't reach Yahoo Finance, so the code was tested on fake prices before delivery |

## How to read the outputs

The two tables that matter most are the accuracy table (section 4) and the backtest table (section 5); the rest support them.

### Accuracy table (notebook section 4)

| Column | What it means | What to look for |
| --- | --- | --- |
| accuracy | Share of test days predicted correctly | Compare it with the "Always up" row, not with 50% |
| range low / range high | The 95% range: accuracies inside it are consistent with luck at this many days | If two methods' ranges overlap heavily, you can't tell them apart |
| predicted up | Share of days the method said "up" | Near 100% means the model mostly learned "always up", not a pattern |
| p-value vs coin flip | Chance of scoring this well by flipping coins | Weak test: "Always up" passes it too, because markets drift upward |
| beats always-up? | True only if even the low end of the range is above "Always up" | The honest test of a real edge |

### Backtest table (notebook section 5)

| Column | What it means | What to look for |
| --- | --- | --- |
| total return | Growth over the whole test period; 8.22 means +822%, so $1 became $9.22 | Compare with buy-and-hold |
| annual return | The same growth expressed per year | The main money comparison |
| annual volatility | How much returns swing in a typical year | Lower is calmer, but only matters next to return |
| Sharpe ratio | Return per unit of volatility | Higher is better; a fair way to compare strategies that hold the stock different amounts of time |
| max drawdown | Worst fall from a peak to a low; -0.34 means -34% | How painful the worst stretch was |
| days in market | Share of days the strategy held the stock | Time out of the market misses up days |
| trades | Number of buys and sells | More trades means more costs |

### Charts and smaller tables

- **Accuracy by year:** shows whether any edge is steady or comes from a few years. Look at which years are bad and what markets did then.
- **Growth of $1:** the backtest as a picture, on a log scale so equal percentage moves look the same size.
- **Cost table:** annual return at 0%, 0.05%, 0.1% and 0.2% per trade. An edge that only exists when trading is free isn't real.
- **Random-walk table (section 6):** models should land near "Always up". Clearly beating it would mean a leak.
- **Logistic regression weights (section 7):** positive pushes toward "up", negative toward "down". Overlapping features can share or cancel weight, so read single bars with care.

## Results so far

On SPY from January 2010 to October 2026 (about 4,200 test days), no model beat "always up", and only buy-and-hold made strong money after costs. Run locally on 9 October 2026; Claude reproduced the same numbers from the saved CSV.

### Accuracy

| Method | Accuracy | 95% range | Predicted up | Beats always-up? |
| --- | --- | --- | --- | --- |
| Always up | 55.2% | 53.7–56.7% | 100% | — |
| Logistic regression | 55.1% | 53.6–56.6% | 97.6% | No |
| Gradient boosting | 52.7% | 51.2–54.2% | 77.0% | No (worse) |
| Repeat last move | 49.7% | 48.2–51.2% | 55.2% | No |
| Coin flip | 49.3% | 47.7–50.8% | 50.4% | No |

- Logistic regression is "always up" in disguise: it said "up" on 97.6% of days.
- Gradient boosting's whole range sits below "always up": it acted on patterns that didn't hold in later years.
- The worst years for every method were 2015, 2018 and 2022, all years with sharp market falls. The models saw none of them coming.

### Backtest (0.1% cost per trade)

| Strategy | Annual return | Sharpe ratio | Max drawdown | Days in market | Trades |
| --- | --- | --- | --- | --- | --- |
| Buy and hold | 14.2% | 0.86 | -33.7% | 100% | 1 |
| Logistic regression | 13.3% | 0.85 | -28.9% | 97.6% | 151 |
| Gradient boosting | 4.1% | 0.35 | -30.9% | 77.0% | 1,053 |

- With free trading, logistic regression earned 14.4% a year, a hair above buy-and-hold's 14.2%. That gap is within luck, and realistic costs erase it.
- Gradient boosting earned only 10.9% a year even with free trading: sitting out 23% of days missed too many up days.
- Logistic regression's smaller drawdown is interesting but not convincing; it was out of the market only about 100 days.

### Checks

- **Random walk:** both models scored 49–50% on fake prices, inside the noise range and not above "always up" (50.3%). No sign of leaked future data.
- **Logistic regression weights:** negative on recent returns (`ret_1d`, `ret_5d`, `ret_10d`), large positive on `dist_ma10`. These features overlap heavily, so the weights partly cancel, and all are small: each shifts the predicted chance of "up" by only 2–3 points.

## Issue tracker

Twelve issues so far: five fixed or worked around, five fixes proposed, one open decision, and one that turned out not to be a bug. Newest first.

| # | Issue | Found | Status | Fix or decision |
| --- | --- | --- | --- | --- |
| 12 | Only one stock runs at a time, and the first test year is fixed at 2010, which fails for stocks listed later | 9 Oct 2026 | Proposed | `run_ticker()` plus a comparison table; first test year set to five years after a stock's data begins |
| 11 | Testing many stocks will produce a few lucky "winners" by chance alone | 9 Oct 2026 | Proposed | A warning in the comparison table |
| 10 | The random-walk check proves the code doesn't invent an edge, but not that it would find a real one | 9 Oct 2026 | Proposed | Planted-pattern test: fake prices with a built-in pattern the models should detect |
| 9 | The last row of downloaded data was an unfinished trading day (9 Oct 2026, volume 14.3M vs about 43M normally), so the 8 Oct label used an unsettled price | 9 Oct 2026 | Proposed | Drop today's row when downloading |
| 8 | Overlapping features (correlation up to 0.91) make logistic regression's weights unreadable | 9 Oct 2026 | Proposed | Trim to 5 features (`ret_1d`, `ret_5d`, `dist_ma50`, `vol_20d`, `volume_ratio`); keep the full set for comparison |
| 7 | No LSTM was built, although the reference repo uses one | 9 Oct 2026 | Open | Decide: add this round or later, judged by the same tests |
| 6 | Neither model beats "always up" | 9 Oct 2026 | Not a bug | The finding, not a bug: the data holds no usable signal. The random-walk check passed |
| 5 | Folder name with spaces needs quotes in Terminal and can break the `jupyter` command | 9 Oct 2026 | Fixed | Renamed to `Stock_Movement_Predictor`; README updated |
| 4 | `.ipynb` files can't be previewed in the Claude app | 9 Oct 2026 | Workaround | Full notebook pasted in chat; open it in Jupyter or VS Code |
| 3 | Claude's workspace can't reach Yahoo Finance | 9 Oct 2026 | Workaround | Code tested on fake prices shaped like yfinance output; real runs on my Mac |
| 2 | The backtest's trades column showed as decimals (1302.0000) | 9 Oct 2026 | Fixed | Stored as whole numbers |
| 1 | Time-zone-aware dates from yfinance shifted to 05:00 UTC | 9 Oct 2026 | Fixed | Keep the local trading date |

## What we struggled with and learned

The biggest lesson so far: when a model predicts badly, first ask whether the code is broken or whether there is simply nothing to learn.

**"Not learning" is different from "broken".** A metal detector on a beach with no metal stays silent, and that doesn't mean it's faulty. The random-walk check showed the code doesn't invent an edge. The planted-pattern test (proposed) will show it can find a real one.

**Beating a coin flip proves nothing.** Markets rise on about 55% of days, so "always up" beats a coin flip without knowing anything. Every model also "beat" the coin flip, with p-values near zero.

**A model can hide a dumb rule.** Logistic regression looked fine at 55.1% accuracy, but it said "up" on 97.6% of days. Always check how often a model predicts each class.

**A more powerful model can do worse.** Gradient boosting fit patterns in the training years that didn't hold later. That is overfitting, and it cost accuracy and money.

**Accuracy isn't money.** Being out of the market on the wrong days misses big up days, and every trade costs something. Gradient boosting's 52.7% accuracy turned into 4.1% a year against 14.2% for buy-and-hold.

**Overlapping features hide what a model learned.** When two features measure nearly the same thing, the model can give them opposite weights that cancel out, so single weights can't be read on their own.

**The limit is the information, not the model.** An LSTM or a bigger network can't find a signal that isn't in the data. Better predictions come from changing what you predict (for example, how big tomorrow's move will be, which is far more predictable than its direction) or what information you use.

**Practical snags:** Claude's workspace can't download market data, the Claude app can't preview notebooks, and spaces in folder names cause Terminal trouble. Each has a workaround in the issue tracker.

## Planned work and open decisions

The next round is planned but not built yet.

### Next round (proposed)

- [ ] Drop the unfinished trading day when downloading
- [ ] Run any list of stocks with one comparison table, first test year set automatically
- [ ] Trim to 5 non-overlapping features, keeping the full set for comparison
- [ ] Add a warning about lucky winners when testing many stocks
- [ ] Planted-pattern test: prove the pipeline finds a real pattern when one exists
- [ ] New section: predict big vs. small moves instead of up vs. down

### My own tasks

- [ ] Write the Conclusions section of the notebook, then copy it into the README's Results section
- [ ] Experiment: `LogisticRegression(max_iter=1000, class_weight="balanced")` in `src/evaluate.py`; guess the outcome before running
- [ ] Experiment: trade only when the model is confident, e.g. `prob_up > 0.55`

### Open decisions

- Go-ahead on the next round above.
- LSTM: add it this round or after the next round?
- Later, maybe: longer horizons and comparing stocks against each other (monthly momentum), a bigger change to the project.

## Glossary

| Term | Meaning |
| --- | --- |
| Adjusted prices | Prices corrected for dividends and splits, so day-to-day changes are the real return |
| Backtest | Replaying a trading rule on past data to see what it would have earned |
| Baseline | A simple rule a model must beat to show it learned something |
| Feature | An input to the model, such as the last 5 days' return |
| Label (target) | What the model predicts: 1 for an up day tomorrow, 0 otherwise |
| Leak (look-ahead) | Future information slipping into features or training, which makes results look better than they could be in real life |
| LSTM | A neural network built for sequences; needs PyTorch or TensorFlow |
| Max drawdown | The worst fall from a peak to a low |
| Multicollinearity | Features that move together, so a model can split or cancel weight between them |
| Overfitting | A model fitting noise in the training data that doesn't repeat later |
| p-value | Chance of a result at least this good from pure luck |
| Random walk | Prices where every day's move is pure chance |
| scikit-learn | The Python library that provides the models and data-preparation tools |
| Sharpe ratio | Return per unit of volatility; higher is better |
| Virtual environment (.venv) | A private set of Python libraries for this project only |
| Volatility clustering | Calm days tend to follow calm days, wild days follow wild days |
| Walk-forward testing | Train on the past, test on the next period, step forward, repeat |

## Changelog

| Date | Change | Details |
| --- | --- | --- |
| 9 Oct 2026 | Project log added to the repo | `PROJECT_LOG.md`, linked from the README |
| 9 Oct 2026 | GitHub prep | `.gitignore` now also skips macOS `.DS_Store` files |
| 9 Oct 2026 | README: returning-session commands | Added "Every time you come back", "When you're done" and "Getting the latest prices"; first-time setup now uses `python3` for Mac |
| 9 Oct 2026 | Project log created | This document |
| 9 Oct 2026 | Next round proposed | Six items, listed under Planned work |
| 9 Oct 2026 | Results reproduced from the saved CSV | Same numbers as the local run; found the unfinished-day issue and the feature overlap |
| 9 Oct 2026 | First real run on SPY | Run locally; results under Results so far |
| 9 Oct 2026 | README updated | Folder renamed to `Stock_Movement_Predictor` |
| 9 Oct 2026 | Version 1 built | `src/data.py`, `src/features.py`, `src/evaluate.py`, `notebooks/analysis.ipynb`, README, requirements; tested on fake prices; two bugs fixed during testing |
| 9 Oct 2026 | Plan agreed | SPY daily data from 2005; logic in `src/` plus one notebook; logistic regression and gradient boosting; walk-forward testing |