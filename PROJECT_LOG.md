# Stock Movement Predictor — Project Log

*Last updated: 16 October 2026 (phase 4.2). A running record of decisions, results, issues and lessons, kept up to date with the help of Claude.*

## Overview

No model predicts tomorrow's direction better than "always predict up" on any of four stocks: past prices hold no usable signal for direction. Move size is different: on all four stocks, both models predict big vs. small days better than the rule "same as today", by 3 to 8 percentage points, and all 8 results survive the correction for testing many things. But on SPY, a one-line rule with no model, "Recent swing" (a big day is likely if today's high-to-low swing was wider than usual), matches the models exactly. On the three single stocks, logistic regression beats that rule by 1.6 to 2.4 points, but only NVDA's edge survives the correction.

- **Goal:** predict whether a stock closes higher tomorrow, using only information available at today's close, and test the answer honestly.
- **Approach:** daily prices from 2005 to today for SPY, NVDA, AAPL and GOOGL; two tasks (direction, and big vs. small moves) with 5 price-based features each; two models (logistic regression and gradient boosting); walk-forward testing from 2010; compared against simple baselines, a backtest with trading costs, a luck check and a lucky-winners correction across stocks.
- **Status:** phases 1–4.2 are built: fixes, 106 automated tests, and one-command reports (`python run.py`) for any stock, with a cross-stock comparison. Phase 4 added a second task, big vs. small moves; phase 4.1 added the tougher Recent swing baseline, a paired test, and a weekly version of the strategy. Phase 4.2 fixed the findings of an independent code review: four bugs, consistent statistics (paired tests everywhere, the Holm correction) and interest on cash. Next: phase 5 (LSTM).
- **Code:** logic lives in `src/`; `notebooks/template.ipynb` is the analysis for one stock; `run.py` saves finished reports in `reports/`.

This log is updated whenever we find an issue, fix one or add a feature. The Changelog at the end lists every change, newest first.

## Setup and key decisions

Every choice below was made to keep the test honest: no peeking at the future, and a fair bar to beat.

| Decision | Choice | Why |
| --- | --- | --- |
| What to predict | Direction: 1 if tomorrow's close is higher than today's, else 0 | A predicted-vs-actual price chart looks impressive even for a useless model, because tomorrow's price is always close to today's |
| Data | Daily prices, adjusted for dividends and splits, 1 Jan 2005 to today, via yfinance; today's row dropped if downloaded before 6 pm New York time; rows whose High and Low don't contain Open and Close are repaired and listed | Adjusted prices give the real return an investor got; Yahoo can still be adjusting a day's prices half an hour after the close |
| Stocks | SPY, NVDA, AAPL, GOOGL; any Yahoo Finance symbol works | One broad fund plus three single stocks |
| Data caching | First run of each stock downloads and saves `data/<TICKER>_from_2005-01-01.csv`; later runs load that file | Results stay the same between runs until you choose to refresh |
| Project layout | Logic in `src/`; `notebooks/template.ipynb` is the analysis for one stock; `run.py` fills it in and saves `reports/<TICKER>.ipynb`; tests in `tests/` | One template means every stock gets exactly the same analysis, and later phases appear in every report automatically |
| Running reports | `python run.py SPY NVDA AAPL GOOGL` runs each report with papermill, then the comparison; a failed ticker doesn't stop the others | One command instead of editing and re-running a notebook per stock |
| Reports on GitHub | Yes, saved with all tables and charts; price data stays out | Visitors see results without running anything |
| Features | 8 calculated; the models use 5: `ret_1d`, `ret_5d`, `dist_ma50`, `vol_20d`, `volume_ratio` | One per idea (momentum, trend, risk, activity); `ret_10d`, `ret_20d` and `dist_ma10` nearly duplicate kept ones |
| Models | Logistic regression (features scaled first) and gradient boosting (shallow trees: depth 3, learning rate 0.05, 200 trees), both from scikit-learn | Start simple and readable; LSTM comes in phase 5 |
| Testing | Walk-forward: train on all years before the test year, predict it, step forward; the last training day is left out, because its label is the first test day's move; first test year = first year with at least 1,000 trading days (about 4 years) before it, 2010 for these stocks, the same for both parts | Every prediction comes from a model that never saw that year, not even one price from it; works for stocks with shorter histories |
| Baselines | Coin flip, always up, repeat last move | Stocks rise on more than half of days, so "always up" is the real bar, not 50% |
| Noise check | 95% range around each accuracy, a p-value against a coin flip, and a paired p-value against the baseline ("always up" or a rule) | Shows whether a difference could be luck; the paired test only counts days where model and baseline disagree |
| Backtest | Hold the stock into tomorrow when the model says up, else cash earning the 13-week Treasury-bill rate (`^IRX`); 0.1% cost per buy or sell; compared with buy-and-hold; Sharpe ratio on returns above cash | Accuracy alone doesn't show whether a model would make money; cash really earns interest, and Sharpe should reward risk, not interest |
| Luck check | The same number of cash days picked at random, 2,000 times; the random versions trade free and earn the same interest; share = (count + 1) / 2,001, so never 0 | Shows whether a backtest edge could come from chance; free trading is generous to luck, so the test is strict |
| Lucky-winners correction | Holm: the smallest p-value must clear 5% divided by the number of comparisons (0.63% for 8), the next 5% / 7, and so on, stopping at the first miss | With 8 comparisons at 5%, about 0.4 pass by luck alone; Holm keeps Bonferroni's guarantee and never passes fewer |
| Hindsight note | Every report says how much $1 grew and warns that well-known winners look good partly in hindsight | NVDA, AAPL and GOOGL were picked knowing how they turned out |
| Pipeline checks | The same code on random-walk prices (should find nothing) and on prices with a planted 60% reversal pattern (should find it) | Proves the code neither invents a pattern nor misses a real one |
| Automated tests | 106 `pytest` tests on small fake price series; no internet needed; run locally, not on GitHub | Catch look-ahead, label, walk-forward, arithmetic, comparison and `run.py` bugs within seconds |
| Move-size label (Part 2) | Big = tomorrow's move, up or down, larger than the median daily move over the previous 250 trading days | A fixed bar like 1% means something different for a calm fund and a wild stock; the past-year median adapts to each stock and era and gives about half big days |
| Move-size features | Today's move size, 5- and 20-day volatility, today's high-low range, volume ratio; all relative to the typical move | All measure recent move size, on the same scale as the label |
| Move-size baselines | Always small / always big; "same as today" (big tomorrow if today's move was bigger than usual); and, since phase 4.1, "Recent swing" (big tomorrow if today's high-to-low swing was wider than its median over the past 250 days) | "Same as today" turned out to be an easy bar. Recent swing is still one line with no fitting, but uses a better measure of how wild today was, so it is the real bar |
| Comparing a model with a rule | Paired test (McNemar's exact test): only the days where model and rule disagree count; "beats <rule>?" means p below 0.05 | A rule predicts the same days and agrees with the model on most of them; treating the rule's accuracy as a fixed number overstates how sure we can be |
| Weekly strategy | Next to the daily half-on-big strategy, a weekly one: decide on the last trading day of each week, hold for the next week; its luck check reshuffles whole weeks | Far fewer trades, so much lower costs; random weeks make the luck check like for like |
| Move-size strategy | Full position on predicted calm days, half on predicted big days; judged on Sharpe ratio and worst drop; luck check on Sharpe ratio | Volatility targeting aims at less risk per unit of return, not more return |
| Move-size pipeline checks | Constant-volatility prices (should find nothing) and GARCH clustered-volatility prices with index-like settings 0.10 / 0.88 (should find it) | Proves the move-size pipeline neither invents nor misses clustering |
| Correction across stocks | Applied to each task separately: 8 direction and 8 move-size accuracy comparisons; the move-size luck check counts 16 (daily and weekly strategy for each), so its corrected bar is 5% / 16 | The two tasks answer different questions; every extra strategy tested is another chance for a lucky pass |
| Python | 3.11 or newer (pandas 3.0 needs it); exact versions recorded with `pip freeze > requirements-lock.txt` | yfinance in particular changes its output often |
| Where it runs | Locally (Mac) | Claude's workspace can't reach Yahoo Finance, so the code was tested on fake prices and the saved SPY CSV |

## How to read the outputs

The two tables that matter most are the accuracy table (section 4) and the backtest table (section 5); the rest support them.

### Accuracy table (notebook section 4)

| Column | What it means | What to look for |
| --- | --- | --- |
| accuracy | Share of test days predicted correctly | Compare it with the "Always up" row, not with 50% |
| range low / range high | The 95% range: accuracies inside it are consistent with luck at this many days | If two methods' ranges overlap heavily, you can't tell them apart |
| predicted up | Share of days the method said "up" | Near 100% means the model mostly learned "always up", not a pattern |
| p-value vs coin flip | Chance of scoring this well by flipping coins | Weak test: "Always up" passes it too, because markets drift upward |
| p-value vs always up | Paired test: only days the model says "down" can differ from "always up"; the chance of being right on that many of them by luck | Small is good |
| beats always up? | True if that p-value is at most 0.05 | The honest test of a real edge |

### Backtest table (notebook section 5)

| Column | What it means | What to look for |
| --- | --- | --- |
| total return | Growth over the whole test period; 8.22 means +822%, so $1 became $9.22 | Compare with buy-and-hold |
| annual return | The same growth expressed per year | The main money comparison |
| annual volatility | How much returns swing in a typical year | Lower is calmer, but only matters next to return |
| Sharpe ratio | Return above what cash earns, per unit of volatility | Higher is better; a fair way to compare strategies that hold the stock different amounts of time |
| max drawdown | Worst fall from a peak to a low; -0.34 means -34% | How painful the worst stretch was |
| days in market | Share of days the strategy held the stock | Time out of the market misses up days |
| trades | Number of buys and sells | More trades means more costs |

### Charts and smaller tables

- **Feature correlation chart (section 2):** values near ±1 mean two features measure nearly the same thing; the five kept features stay at 0.58 or below.
- **5 vs. 8 features (section 4):** if adding the three left-out features barely changes accuracy, they carried no extra information.
- **Accuracy by year:** shows whether any edge is steady or comes from a few years. Look at which years are bad and what markets did then.
- **Growth of $1:** the backtest as a picture, on a log scale so equal percentage moves look the same size.
- **Cost table:** annual return at 0%, 0.05%, 0.1% and 0.2% per trade. An edge that only exists when trading is free isn't real.
- **Luck check (section 5):** "share of random >=" is how often random cash days did at least as well; about 0.05 or less is hard to explain by luck. The biggest days avoided and missed show whether a few extreme days drive the result.
- **Pipeline checks (section 6):** on pure noise, models should land near "Always up"; on the planted pattern, both should show True in "beats always up?".
- **Logistic regression weights (section 7):** positive pushes toward "up", negative toward "down". With five non-overlapping features, each bar can be read on its own.

### Comparison report (`reports/comparison.ipynb`)

| Part | What it shows | What to look for |
| --- | --- | --- |
| Test periods | Each stock's test dates, "always up" accuracy and buy-and-hold return | Stocks with shorter histories start testing later |
| Edge chart | Each model's accuracy minus "always up", with the 95% range of that difference from the paired counts | A range crossing zero means the edge could be luck |
| p-value vs always up | Chance of an accuracy this high if the model were no better than "always up" | Small is good, but see the corrected column |
| 5% vs corrected columns | Whether a result passes as a single test, and after the Holm correction | Only corrected passes count as evidence |
| Luck count | How many passes luck alone would produce, next to the actual passes | Passes close to the luck count mean nothing |

### Part 2: move size (report sections 8–12)

| Part | What it shows | What to look for |
| --- | --- | --- |
| Typical move chart | The bar a "big" day is measured against, over time | It rises in turbulent years (2008, 2020, 2022) and falls in calm ones |
| Feature correlation | How each move-size feature relates to a big day tomorrow | Clearly positive, unlike Part 1 |
| Accuracy table | Each method's accuracy, with paired p-values against "same as today" and "Recent swing", and "beats recent swing?" | The honest test: a p-value against Recent swing below 0.05 |
| Disagreement table | For each model: on how many days it disagrees with Recent swing, and who was right on those days | Only these days decide which is better; a near-even split means the model adds nothing |
| Accuracy by year | Models and both rules year by year | An edge in every year, not just a few |
| Pipeline checks | Constant volatility (about 50%) and clustered volatility (models beat both rules) | Both should behave as described; the rule is weak on fake prices, whose intraday swings are mostly noise |
| Half-on-big backtest | Sharpe ratio, worst drop and trades for each model and rule, daily and weekly, with free trading and with costs | Lower risk only counts if Sharpe holds up after costs; compare daily with weekly to see what trading less does |
| Return per unit of risk | Average return divided by daily swing, on predicted big vs. calm days | If similar, halving big days cuts reward as much as risk |
| Luck check | How often random half-position days (daily) or random whole weeks (weekly) reach the same Sharpe ratio | About 0.05 or less is hard to explain by luck |

In the comparison, Part 2 now shows two edge charts (against "same as today", then against Recent swing), a table with both strategies, and a luck count that includes the weekly strategies.

## Results so far

Move size is predictable on all four stocks; direction is not. On SPY, though, the models predict move size no better than the one-line Recent swing rule; on single stocks, logistic regression beats it by a small margin. No model beats "always up" on any of the four stocks, and on single stocks every direction model loses to buy-and-hold. SPY's borderline backtest edge from phase 1 didn't replicate on NVDA, AAPL or GOOGL. Newest results come first; earlier ones are kept below for comparison.

### After phase 4.2: four stocks, with interest on cash

Run locally on 16 October 2026 with `python run.py SPY NVDA AAPL GOOGL`, on freshly downloaded prices; test period 4 January 2010 to 14 October 2026 (4,221 days) for every stock. Cash earned 1.48% a year on average (13-week Treasury bills), and every Sharpe ratio is now on returns above cash. Edge is accuracy minus Recent swing, in percentage points, with its 95% range from the paired counts; the worst drop and luck share are for the weekly version.

| Stock | Model | Accuracy | Recent swing | Edge | p-value | Sharpe, daily | Sharpe, weekly | Sharpe, buy and hold | Worst drop, weekly | Worst drop, buy and hold | Luck share, weekly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | Logistic regression | 61.7% | 61.6% | +0.05 ± 1.4 | 0.49 | 0.61 | 0.76 | 0.77 | -23.3% | -33.7% | 0.44 |
| SPY | Gradient boosting | 61.6% | 61.6% | -0.05 ± 1.4 | 0.54 | 0.50 | 0.78 | 0.77 | -18.6% | -33.7% | 0.31 |
| NVDA | Logistic regression | 57.4% | 55.1% | +2.3 ± 1.6 | 0.002 | 0.97 | 1.06 | 1.01 | -49.9% | -66.3% | 0.090 |
| NVDA | Gradient boosting | 55.7% | 55.1% | +0.6 ± 1.8 | 0.25 | 0.86 | 1.05 | 1.01 | -50.1% | -66.3% | 0.098 |
| AAPL | Logistic regression | 58.1% | 56.5% | +1.7 ± 1.4 | 0.013 | 0.96 | 0.99 | 0.93 | -35.0% | -43.8% | 0.094 |
| AAPL | Gradient boosting | 56.5% | 56.5% | +0.05 ± 1.6 | 0.49 | 0.77 | 0.90 | 0.93 | -40.4% | -43.8% | 0.43 |
| GOOGL | Logistic regression | 56.4% | 54.7% | +1.7 ± 1.5 | 0.013 | 0.65 | 0.76 | 0.75 | -40.6% | -44.3% | 0.28 |
| GOOGL | Gradient boosting | 55.5% | 54.7% | +0.8 ± 1.7 | 0.19 | 0.61 | 0.70 | 0.75 | -40.4% | -44.3% | 0.60 |

- **The conclusions from phase 4.1 all hold.** Direction: 0 of 8 beat "always up". Move size against Recent swing: 3 of 8 pass at 5%, 1 of 8 after the Holm correction (NVDA logistic regression, p = 0.002); Holm gives the same answer Bonferroni did. Against "same as today": 8 of 8. Strategies: 0 of 16 pass the luck check.
- **Interest on cash lowers every Sharpe ratio by about the same amount,** so no comparison flips. Buy-and-hold drops by 0.04 to 0.09 (SPY 0.86 to 0.77). Logistic regression's weekly version still beats buy-and-hold's Sharpe on NVDA, AAPL and GOOGL by 0.01 to 0.06, as before, and still trails slightly on SPY.
- **Weekly beats daily in all 8 rows,** by 0.03 to 0.28 in Sharpe.
- **SPY's direction lead is unchanged:** logistic regression earns 14.9% a year against 14.1% for buy-and-hold (Sharpe 0.84 vs 0.77), with a luck share of 0.047: a pass at 5%, not after the correction.
- **Gradient boosting moved again; logistic regression didn't.** On SPY, gradient boosting's move-size accuracy went from 61.2% in the check on the old saved file to 61.6% on the fresh download (5 more days, and Yahoo's small revisions to past rows), while logistic regression stayed at 61.7%.

### After phase 4.2: SPY check, without interest on cash

Run on the saved SPY CSV on 16 October 2026 with `--no-interest`, to see what the other fixes change on their own. Real Treasury-bill rates can't be downloaded in Claude's workspace, so the numbers with interest come from the next local run.

| Number | Before 4.2 | After 4.2 | Why |
| --- | --- | --- | --- |
| Logistic regression, direction p-value vs always up | 0.56 | 0.73 | Paired test (64 "down" days: 30 right, 34 wrong) |
| Gradient boosting, direction accuracy | 52.8% | 52.7% | Last training day left out |
| Logistic regression, move-size accuracy | 61.6% | 61.7% | Last training day left out |
| Gradient boosting, move-size accuracy | 61.1% | 61.2% | Last training day left out |
| Logistic regression, weekly Sharpe | 0.85 | 0.86 | Last training day left out |
| Gradient boosting, weekly Sharpe | 0.90 | 0.87 | Last training day left out |
| Luck share, logistic regression direction | 0.0475 | 0.048 | Now (count + 1) / 2,001 |

- **No conclusion changes.**
- **Gradient boosting is fragile:** leaving out one training day per year (17 days in all) changes its move-size predictions on 146 of 4,216 days, 38 of them weekly decision days. Logistic regression barely moves.
- **Repaired rows:** 3 SPY rows had a High or Low that didn't contain the Open or Close (26 Oct 2007, 31 Jul 2012, 19 Jan 2018).

### After phase 4.1: four stocks against the Recent swing rule

Run locally on 16 October 2026 with `python run.py SPY NVDA AAPL GOOGL`; test period January 2010 to October 2026 for every stock. Edge is accuracy minus Recent swing, in percentage points; the p-value is the paired test. Strategies hold half the position on predicted big days, after 0.1% costs; the worst drop and luck share are for the weekly version.

| Stock | Model | Accuracy | Recent swing | Edge | p-value | Sharpe, daily | Sharpe, weekly | Sharpe, buy and hold | Worst drop, weekly | Worst drop, buy and hold | Luck share, weekly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | Logistic regression | 61.6% | 61.6% | 0.0 | 0.53 | 0.72 | 0.85 | 0.86 | -23.3% | -33.7% | 0.39 |
| SPY | Gradient boosting | 61.1% | 61.6% | -0.5 | 0.78 | 0.61 | 0.90 | 0.86 | -18.6% | -33.7% | 0.18 |
| NVDA | Logistic regression | 57.5% | 55.1% | +2.4 | 0.001 | 1.00 | 1.10 | 1.05 | -49.9% | -66.3% | 0.087 |
| NVDA | Gradient boosting | 55.3% | 55.1% | +0.2 | 0.42 | 0.84 | 1.07 | 1.05 | -52.9% | -66.3% | 0.16 |
| AAPL | Logistic regression | 58.1% | 56.5% | +1.7 | 0.013 | 1.01 | 1.05 | 0.99 | -35.0% | -43.8% | 0.083 |
| AAPL | Gradient boosting | 56.2% | 56.5% | -0.3 | 0.64 | 0.84 | 0.92 | 0.99 | -34.0% | -43.8% | 0.58 |
| GOOGL | Logistic regression | 56.3% | 54.7% | +1.6 | 0.017 | 0.71 | 0.82 | 0.81 | -40.9% | -44.3% | 0.26 |
| GOOGL | Gradient boosting | 55.1% | 54.7% | +0.4 | 0.35 | 0.64 | 0.78 | 0.81 | -40.7% | -44.3% | 0.45 |

- **Against Recent swing:** logistic regression beats the rule on all three single stocks and ties on SPY. 3 of 8 pass at 5% and 1 of 8 after the correction (NVDA, p = 0.001 against a bar of 0.0063); luck alone would give about 0.4. Gradient boosting never beats the rule.
- **The rule is much stronger on SPY** (61.6%) than on single stocks (54.7–56.5%). A likely reason: a single stock's intraday swing also reflects company news that doesn't carry over to the next day, so the other features, such as the past month's volatility, add information there.
- **Read the edge with care:** the same direction on three stocks makes logistic regression's small edge more believable than any one p-value, but the three stocks move together, so they aren't three independent confirmations.
- **Against "same as today":** still 8 of 8 after the correction, as in phase 4.
- **Weekly beats daily in all 8 rows,** by 0.03 to 0.29 in Sharpe. Logistic regression's weekly version matches or beats buy-and-hold's Sharpe on NVDA, AAPL and GOOGL (by 0.01 to 0.06) and falls just short on SPY. Its worst drop is smaller on all four stocks, by 3 to 16 points, but its annual return is lower on all four (NVDA 38.8% vs 45.5%).
- **Luck check:** 0 of 16 strategies pass, even at 5%. The closest are logistic regression's weekly versions on AAPL (0.083) and NVDA (0.087).
- **Direction:** unchanged from phase 3; 0 of 8 beat "always up".

### After phase 4.1: SPY against the Recent swing rule

Run on the saved SPY CSV on 16 October 2026; same test period as phase 4 (4 January 2010 to 7 October 2026, 4,216 days).

| Method | Accuracy | 95% range | Paired p-value vs Recent swing | Paired p-value vs Same as today |
| --- | --- | --- | --- | --- |
| Recent swing (one-line rule) | 61.65% | 60.2–63.1% | — | below 0.0001 |
| Logistic regression | 61.62% | 60.2–63.1% | 0.53 | below 0.0001 |
| Gradient boosting | 61.12% | 59.7–62.6% | 0.78 | below 0.0001 |
| Same as today | 53.3% | 51.8–54.9% | — | — |

- **A one-line rule matches the models.** Logistic regression and the rule agree on 78% of days. On the 933 days they disagree, the model is right on 466 and the rule on 467. Year by year, the model is ahead in 7 of 17 years and behind in 10.
- **What this means:** move size is predictable, but the predictability comes from one well-chosen measure, today's high-to-low swing compared with its usual size. The five features and the models add nothing on SPY. Phase 4's "beats same as today" was true but measured against a weak bar.
- **Comparing the swing with its own usual size works better than dividing it by the typical close-to-close move:** 61.7% vs 60.3% for the same idea built on `range_rel` (the version tested before building).
- **Pipeline checks:** on constant-volatility prices every method lands at 49–51%. On clustered-volatility prices, logistic regression (58.2%) and gradient boosting (57.1%) beat both rules (Recent swing 55.3%, same as today 54.3%). On fake prices the intraday swing is mostly random noise, so the rule is weaker there than on real prices.

| Strategy, half on predicted big days (0.1% cost per trade) | Sharpe, daily | Sharpe, weekly | Worst drop, weekly | Trades, daily → weekly | Luck share, daily → weekly |
| --- | --- | --- | --- | --- | --- |
| Buy and hold | 0.86 | — | -33.7% | 1 | — |
| Logistic regression | 0.72 | 0.85 | -23.3% | 789 → 219 | 0.96 → 0.39 |
| Gradient boosting | 0.61 | 0.90 | -18.6% | 1,060 → 262 | 1.00 → 0.18 |
| Recent swing | 0.54 | 0.78 | -19.7% | 1,317 → 315 | — |
| Same as today | 0.34 | 0.81 | -18.6% | 1,964 → 408 | — |

- **Deciding weekly changes the strategy result more than the choice of model.** Trades fall by about three quarters and Sharpe rises by 0.13 to 0.47. Gradient boosting's weekly version edges past buy-and-hold (0.90 vs 0.86) with a much smaller worst drop (-19% vs -34%).
- **But it isn't evidence yet:** random whole weeks, trading free, reach that Sharpe 18% of the time, and 39% for logistic regression. Weekly is clearly the better way to use the prediction, but not a proven edge.

### After phase 4: move size on four stocks

Run locally on 15 October 2026 with `python run.py SPY NVDA AAPL GOOGL`; test period January 2010 to October 2026 for every stock. Edge is accuracy minus "same as today", in percentage points. Sharpe and worst drop are for half on predicted big days, after 0.1% costs.

| Stock | Model | Accuracy | Same as today | Edge | Sharpe (half on big) | Sharpe (buy and hold) | Worst drop (half on big) | Worst drop (buy and hold) | Share of random doing as well |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | Logistic regression | 61.6% | 53.3% | +8.3 | 0.72 | 0.86 | -21.6% | -33.7% | 0.96 |
| SPY | Gradient boosting | 61.1% | 53.3% | +7.8 | 0.61 | 0.86 | -19.9% | -33.7% | 1.00 |
| NVDA | Logistic regression | 57.5% | 51.6% | +5.9 | 1.00 | 1.05 | -50.3% | -66.3% | 0.51 |
| NVDA | Gradient boosting | 55.3% | 51.6% | +3.7 | 0.84 | 1.05 | -56.6% | -66.3% | 0.99 |
| AAPL | Logistic regression | 58.1% | 53.0% | +5.1 | 1.01 | 0.99 | -34.6% | -43.8% | 0.16 |
| AAPL | Gradient boosting | 56.2% | 53.0% | +3.2 | 0.84 | 0.99 | -39.6% | -43.8% | 0.88 |
| GOOGL | Logistic regression | 56.3% | 50.6% | +5.7 | 0.71 | 0.81 | -43.7% | -44.3% | 0.81 |
| GOOGL | Gradient boosting | 55.1% | 50.6% | +4.5 | 0.64 | 0.81 | -44.0% | -44.3% | 0.97 |

- **Accuracy:** all 8 beat "same as today", at 5% and after the correction (8 of 8 both times), against 0 of 8 for direction. Every 95% range sits right of zero.
- **Logistic regression beats gradient boosting on every stock,** by 0.5 to 2.4 points, as in Part 1. It is the move-size model for phase 6.
- **The edge is largest on SPY** (+8.3 vs +5 to 6). A likely reason: an index's swings come from market-wide news, which clusters, while single stocks also jump on company events such as earnings reports, which the features can't see.
- **The models use more than yesterday's move:** on GOOGL, "same as today" barely beats guessing (50.6% vs 50.0%), yet logistic regression reaches 56.3%.
- **Strategy:** halving on predicted big days shrinks the worst drop on every stock but GOOGL, but Sharpe after costs beats buy-and-hold only on AAPL (1.01 vs 0.99), with a luck share of 0.16: easily chance. 0 of 8 pass the luck check.

### After phase 4: move size on SPY

Run on the saved SPY CSV on 15 October 2026; test period 4 January 2010 to 7 October 2026 (4,216 days); 49.2% of days were big.

| Method | Accuracy | 95% range | Beats "same as today"? |
| --- | --- | --- | --- |
| Logistic regression | 61.6% | 60.2–63.1% | Yes |
| Gradient boosting | 61.1% | 59.7–62.6% | Yes |
| Same as today | 53.3% | 51.8–54.9% | — |
| Always small | 50.8% | 49.3–52.3% | No |

- **The signal is real and steady:** both models beat "same as today" in every year from 2010 to 2026. Today's high-low range is the most informative feature (correlation 0.24 with a big day tomorrow).
- **Pipeline checks pass:** about 50% on constant-volatility prices; on clustered-volatility prices logistic regression scores 58.2% and gradient boosting 57.1%, both beating "same as today" (54.3%).

| Strategy (0.1% cost per trade) | Sharpe ratio | Sharpe with free trading | Max drawdown | Trades |
| --- | --- | --- | --- | --- |
| Buy and hold | 0.86 | 0.86 | -33.7% | 1 |
| Logistic regression, half on big | 0.72 | 0.91 | -21.6% | 789 |
| Gradient boosting, half on big | 0.61 | 0.89 | -19.9% | 1,060 |
| Same as today, half on big | 0.34 | 0.82 | -24.0% | 1,964 |

- **Predicting risk works; profiting from it is harder.** Halving on predicted big days cuts the worst drop from -34% to about -20%. With free trading, logistic regression's Sharpe edges past buy-and-hold (0.91 vs 0.86); after costs it falls to 0.72.
- **Why:** days the model calls big are twice as wild (daily swing 1.53% vs 0.77%) but also earn about twice as much (+0.090% vs +0.044% a day), so return per unit of risk is the same (0.059 vs 0.057). Halving big days cuts reward as much as risk.
- **Luck check:** random half-position days, trading free, reach a higher Sharpe than the costed strategy 96% of the time (logistic regression) and 99.8% (gradient boosting).

### After phase 3: four stocks

Run locally on 12 October 2026 with `python run.py SPY NVDA AAPL GOOGL`; test period 2010 to October 2026 for every stock. Edge is accuracy minus "always up", in percentage points. Full reports: [`reports/`](reports/), starting with [`reports/comparison.ipynb`](reports/comparison.ipynb).

| Stock | Model | Accuracy | Always up | Edge | Annual return | Buy and hold | Share of random cash days doing as well |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | Logistic regression | 55.1% | 55.2% | -0.1 | 14.9% | 14.2% | 0.048 |
| SPY | Gradient boosting | 52.8% | 55.2% | -2.4 | 5.4% | 14.2% | 0.999 |
| NVDA | Logistic regression | 51.6% | 52.8% | -1.3 | 33.5% | 45.5% | 0.71 |
| NVDA | Gradient boosting | 51.1% | 52.8% | -1.8 | 22.9% | 45.5% | 0.62 |
| AAPL | Logistic regression | 52.9% | 53.1% | -0.1 | 22.5% | 26.8% | 0.87 |
| AAPL | Gradient boosting | 51.4% | 53.1% | -1.7 | 9.7% | 26.8% | 0.99 |
| GOOGL | Logistic regression | 52.1% | 52.8% | -0.8 | 12.2% | 20.5% | 0.996 |
| GOOGL | Gradient boosting | 51.8% | 52.8% | -1.0 | 5.3% | 20.5% | 0.97 |

- **Accuracy:** all 8 edges are negative; 0 of 8 pass at 5%, 0 of 8 after the correction.
- **Gradient boosting is worse than logistic regression on every stock.** For SPY, NVDA and AAPL its whole 95% range sits below zero: reliably worse than "always up", the same overfitting on every stock.
- **Backtest:** on NVDA, AAPL and GOOGL every model trails buy-and-hold by 4 to 23 points a year. Only SPY's logistic regression passes the luck check at 5% (0.048), and not after the correction (0.0063). The same model does worse than random cash days on the other three stocks, so the phase 1 lead looks like luck.
- **How often a stock rises isn't where returns come from.** NVDA rose on 52.8% of days, fewer than SPY's 55.2%, yet returned 45.5% a year against 14.2%: its up days were much bigger than its down days. A direction-only model can't see that.
- **Hindsight:** the single stocks' buy-and-hold returns (20–46% a year) are high partly because they were picked knowing they became winners.

### After phase 1: SPY (5 features, unfinished day removed)

Run on the saved SPY CSV on 9 October 2026; test period 4 January 2010 to 7 October 2026 (4,216 days).

| Method | Accuracy | 95% range | Predicted up | Beats always-up? |
| --- | --- | --- | --- | --- |
| Always up | 55.2% | 53.7–56.7% | 100% | — |
| Logistic regression | 55.1% | 53.6–56.6% | 98.5% | No |
| Gradient boosting | 52.8% | 51.3–54.3% | 78.3% | No (worse) |

Adding the three left-out features back changes accuracy by 0.1 points or less, as expected.

| Strategy (0.1% cost per trade) | Annual return | Sharpe ratio | Max drawdown | Cash days | Trades | Share of random cash days doing as well |
| --- | --- | --- | --- | --- | --- | --- |
| Buy and hold | 14.2% | 0.86 | -33.7% | 0 | 1 | — |
| Logistic regression | 14.9% | 0.93 | -26.7% | 64 | 97 | 0.048 |
| Gradient boosting | 5.4% | 0.43 | -27.4% | 916 | 1,110 | 0.999 |

- Logistic regression now edges past buy-and-hold, but treat it as a lead, not a finding. Random cash days do as well about 1 time in 20, and one day, the 16 March 2020 crash (-10.9%), supplies more than half of the gain. With all 8 features, the same model lost to buy-and-hold (13.3%).
- Its 64 cash days cluster in market crises: 2010–2011, 2020, 2022 and 2025. Phase 3 tested this on three other stocks, and it didn't hold up there; phase 6 will test it on future data.
- Gradient boosting does worse than random: 99.9% of random choices of its 916 cash days did better.

### After phase 1: pipeline checks

- **Random walk:** both models scored about 48.9%, below "always up" (50.3%). Nothing found, as it should be.
- **Planted pattern:** logistic regression 59.3% and gradient boosting 58.1%, against 49.6% for "always up". Both found the pattern, well beyond noise.
- **Yesterday vs. tomorrow:** the correlation between yesterday's return and tomorrow's is -0.10 on all days, but only -0.03 without the most extreme 1% of days. The small reversal tendency lives mostly in a few wild days.

### Version 1 (8 features): accuracy

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

### Version 1: backtest (0.1% cost per trade)

| Strategy | Annual return | Sharpe ratio | Max drawdown | Days in market | Trades |
| --- | --- | --- | --- | --- | --- |
| Buy and hold | 14.2% | 0.86 | -33.7% | 100% | 1 |
| Logistic regression | 13.3% | 0.85 | -28.9% | 97.6% | 151 |
| Gradient boosting | 4.1% | 0.35 | -30.9% | 77.0% | 1,053 |

- With free trading, logistic regression earned 14.4% a year, a hair above buy-and-hold's 14.2%. That gap is within luck, and realistic costs erase it.
- Gradient boosting earned only 10.9% a year even with free trading: sitting out 23% of days missed too many up days.
- Logistic regression's smaller drawdown is interesting but not convincing; it was out of the market only about 100 days.

### Version 1: checks

- **Random walk:** both models scored 49–50% on fake prices, inside the noise range and not above "always up" (50.3%). No sign of leaked future data.
- **Logistic regression weights:** negative on recent returns (`ret_1d`, `ret_5d`, `ret_10d`), large positive on `dist_ma10`. These features overlap heavily, so the weights partly cancel, and all are small: each shifts the predicted chance of "up" by only 2–3 points.

## Issue tracker

Thirty-seven issues so far: thirty-two fixed or worked around, two open (steps for me), one planned for a later phase, and two that turned out not to be bugs. Issues 23–37 come from an independent review of the code by a separate Claude Code session run locally. Newest first.

| # | Issue | Found | Status | Fix or decision |
| --- | --- | --- | --- | --- |
| 37 | Small inconsistencies: cache file names didn't use the safe ticker name; `walk_forward`'s column was called `prob_up` even for big days; missing predictions were silently counted as wrong; trailing spaces; no way to rebuild only the comparison | 16 Oct 2026 | Fixed | Phase 4.2: safe names (`^IRX` → `_IRX_from_…csv`); column renamed `prob`; missing predictions raise an error; `--comparison-only` |
| 36 | The README said Python 3.10, but pandas 3.0 needs 3.11; no record of exact library versions | 16 Oct 2026 | Open | README now says 3.11; on my side: `pip freeze > requirements-lock.txt` and commit it |
| 35 | `.DS_Store` is committed even though `.gitignore` lists it | 16 Oct 2026 | Open | On my side: `git rm --cached .DS_Store`, then commit |
| 34 | The weekly strategy treated the last day in the data as a week end, even mid-week | 16 Oct 2026 | Fixed | The last day only counts as a week end on a Friday |
| 33 | Yahoo was still adjusting a row downloaded at 4:36 pm (Open above High); a few old SPY rows had High or Low not containing Open or Close | 16 Oct 2026 | Fixed | Cutoff moved to 6 pm New York time; such rows are repaired (High and Low widened; Close never changed) and listed in each report |
| 32 | When predicting year Y, the last training row's label used the first closing price of year Y | 16 Oct 2026 | Fixed | That day is left out of each training set; negligible for logistic regression, but it changes gradient boosting's predictions on 146 SPY days |
| 31 | Cash earned nothing in the backtests, and the Sharpe ratio ignored interest | 16 Oct 2026 | Fixed | Cash earns the 13-week Treasury-bill rate (`^IRX`); Sharpe uses returns above cash; `--no-interest` switches it off |
| 30 | The luck check could report a share of exactly 0, as if luck were impossible | 16 Oct 2026 | Fixed | Share = (count + 1) / (trials + 1) |
| 29 | Bonferroni is stricter than needed | 16 Oct 2026 | Fixed | Holm correction: same guarantee, never fewer passes; same answers on the current results |
| 28 | The comparison charts' error bars showed each model's own accuracy range, not the range of its difference from the baseline, so a bar could cross zero while the table said "significant" | 16 Oct 2026 | Fixed | Ranges now come from the paired counts saved in each summary |
| 27 | Part 1 judged models against "always up" three different ways (range in the table, one-sample test in the summary, own range in the chart) | 16 Oct 2026 | Fixed | The paired test everywhere, as in Part 2; SPY logistic regression's p-value goes from 0.56 to 0.73 |
| 26 | Tickers without volume (`^VIX`, currency pairs) gave an empty dataset and a confusing error | 16 Oct 2026 | Fixed | A clear error saying the volume feature needs volume |
| 25 | The comparison's error handler could itself crash on an error with an empty message | 16 Oct 2026 | Fixed | Same guard as the per-stock handler |
| 24 | A failed run overwrote the last good report with a partly run one | 16 Oct 2026 | Fixed | Reports are written to `<TICKER>.partial.ipynb` and replace the old one only on success |
| 23 | `--first-test-year` only applied to Part 1; Part 2 always chose its own first year | 16 Oct 2026 | Fixed | Both parts use the same year; automatic = the later of the two automatic choices |
| 22 | In the constant-volatility pipeline check, "Recent swing" showed True in "beats same as today?" (50.8% vs 49.4%): one rule beating another by chance, in a check that should find nothing | 16 Oct 2026 | Fixed | The "beats" columns in the pipeline checks are about the models, so they are now left empty for the rules |
| 21 | On the fake prices, the Recent swing rule only reaches about 53%, against about 62% on SPY | 16 Oct 2026 | Workaround | The fake prices' intraday swing is mostly random noise around the day's move, unlike real prices. The pipeline checks are about the models, which behave as expected; the rule's own test only checks that it works on clustered volatility and finds nothing on constant volatility |
| 20 | "Same as today" was an easy bar, and the move-size p-value compared the model's accuracy with the rule's as if the rule's were a fixed number | 15 Oct 2026 | Fixed | Phase 4.1: a one-feature check showed today's swing alone gets most of the way; added the Recent swing baseline and a paired test that only counts days where model and rule disagree |
| 19 | NVDA and GOOGL showed the identical "always up" accuracy, 0.5284495021337127, which could have meant one stock's data was a copy of the other's | 12 Oct 2026 | Not a bug | Checked day by day: both have exactly 2,229 up days out of 4,218, but move the same way on only 66.3% of days. A coincidence; the data is correct |
| 18 | `.gitignore` had no line break at its end, so `echo` glued the new entry onto the last line (`.DS_Storenotebooks/analysis.ipynb`); neither rule worked and VS Code kept offering to add `analysis.ipynb` back | 12 Oct 2026 | Fixed | Split into two lines; checked with `git check-ignore -v`. VS Code's "Insert Final Newline" setting makes every file end with a line break |
| 17 | The comparison chart's legend covered the GOOGL gradient-boosting point | 12 Oct 2026 | Fixed | Legend moved below the chart; found by rendering the chart and looking at it |
| 16 | A comment on the same line as `FIRST_TEST_YEAR = None` in the settings cell made papermill print "Unable to parse line 4" on every run | 12 Oct 2026 | Fixed | Comments moved to their own lines |
| 15 | While copying phase 1 into the project by hand, `walk_forward`'s first line stayed at the old version while its body was new, so every call would crash | 9 Oct 2026 | Fixed | Caught in review and fixed; phase 2's tests now catch it automatically (7 tests fail on that version) |
| 14 | An earlier claim that every feature's correlation with tomorrow's return was close to zero; for `ret_1d` it is -0.10 | 9 Oct 2026 | Fixed | Corrected: -0.10 on all days but -0.03 without the most extreme 1% of days, so the effect lives in a few wild days |
| 13 | First version of the luck check charged random cash days the same trading cost; scattered days trade far more, so random versions looked worse and gradient boosting looked "not luck" | 9 Oct 2026 | Fixed | Random versions now trade free, which is generous to luck and makes the test strict |
| 12 | Only one stock runs at a time | 9 Oct 2026 | Fixed | Phase 3: `python run.py` creates a report per stock from one template |
| 11 | Testing many stocks will produce a few lucky "winners" by chance alone | 9 Oct 2026 | Fixed | Phase 3: the comparison shows passes at 5% and after the Bonferroni correction, next to the number luck would produce |
| 10 | The random-walk check proves the code doesn't invent an edge, but not that it would find a real one | 9 Oct 2026 | Fixed | Planted-pattern check added (phase 1): both models find a 60% reversal pattern |
| 9 | The last row of downloaded data was an unfinished trading day (9 Oct 2026, volume 14.3M vs about 43M normally), so the 8 Oct label used an unsettled price | 9 Oct 2026 | Fixed | Phase 1: today's row is dropped if downloaded before 4:30 pm New York time, including in existing saved files |
| 8 | Overlapping features (correlation up to 0.91) made logistic regression's weights unreadable; the first test year was fixed at 2010 | 9 Oct 2026 | Fixed | Phase 1: models use 5 features (all 8 still available); first test year chosen automatically |
| 7 | No LSTM was built, although the reference repo uses one | 9 Oct 2026 | Proposed | Phase 5, on request only (`RUN_LSTM` switch) |
| 6 | Neither model beats "always up" | 9 Oct 2026 | Not a bug | The finding, not a bug: the data holds no usable signal. Both pipeline checks pass |
| 5 | Folder name with spaces needs quotes in Terminal and can break the `jupyter` command | 9 Oct 2026 | Fixed | Renamed to `Stock_Movement_Predictor`; README updated |
| 4 | `.ipynb` files can't be previewed in the Claude app | 9 Oct 2026 | Workaround | Full notebook pasted in chat; open it in Jupyter or VS Code |
| 3 | Claude's workspace can't reach Yahoo Finance | 9 Oct 2026 | Workaround | Code tested on fake prices and the saved SPY CSV; real runs on my Mac |
| 2 | The backtest's trades column showed as decimals (1302.0000) | 9 Oct 2026 | Fixed | Stored as whole numbers |
| 1 | Time-zone-aware dates from yfinance shifted to 05:00 UTC | 9 Oct 2026 | Fixed | Keep the local trading date |

## What we struggled with and learned

The biggest lesson so far: when a model predicts badly, first ask whether the code is broken or whether there is simply nothing to learn.

**"Not learning" is different from "broken".** A metal detector on a beach with no metal stays silent, and that doesn't mean it's faulty. The random-walk check showed the code doesn't invent an edge. The planted-pattern check showed it can find a real one.

**Beating a coin flip proves nothing.** Markets rise on about 55% of days, so "always up" beats a coin flip without knowing anything. Every model also "beat" the coin flip, with p-values near zero.

**A model can hide a dumb rule.** Logistic regression looked fine at 55.1% accuracy, but it said "up" on 97.6% of days. Always check how often a model predicts each class.

**A more powerful model can do worse.** Gradient boosting fit patterns in the training years that didn't hold later. That is overfitting, and it cost accuracy and money.

**Accuracy isn't money.** Being out of the market on the wrong days misses big up days, and every trade costs something. Gradient boosting's 52.7% accuracy turned into 4.1% a year against 14.2% for buy-and-hold.

**Overlapping features hide what a model learned.** When two features measure nearly the same thing, the model can give them opposite weights that cancel out, so single weights can't be read on their own.

**The limit is the information, not the model.** An LSTM or a bigger network can't find a signal that isn't in the data. Better predictions come from changing what you predict (for example, how big tomorrow's move will be, which is far more predictable than its direction) or what information you use.

**A small edge can live in a few extreme days.** Yesterday's return correlates -0.10 with tomorrow's, but that almost vanishes without the wildest 1% of days. Logistic regression's backtest gain also rests mostly on one crash day in March 2020. Always ask which days drive a result.

**A luck test has to be fair to luck.** The first luck check charged random strategies trading costs they wouldn't really pay, which made a bad model look skilled. When testing whether something beats chance, give chance every advantage.

**Test the tests.** A test that never fails proves nothing. Each test was checked by planting a realistic bug in a copy of the code (a feature that peeks at tomorrow, training on the test year, removing the unfinished-day fix, the old `walk_forward` line) and confirming at least one test failed.

**The more you test, the stricter the bar.** At the 5% level, 1 test in 20 passes by luck. With 4 stocks and 2 models, that's 8 chances, so about 0.4 lucky passes are expected before anything real happens. SPY's borderline backtest result (luck share 0.048) passes at 5% but not at the corrected 0.63%.

**A real pattern should replicate.** SPY's borderline backtest edge looked interesting on its own. Testing the same model on three more stocks showed nothing like it, which is the strongest evidence that it was luck. One good-looking result is a reason to test more, not a finding.

**Direction accuracy isn't where returns come from.** NVDA rose on fewer days than SPY but returned three times as much per year, because its up days were bigger. Predicting up or down ignores the size of the move, and that's where the money is.

**Some things are predictable, some aren't.** The same features, models and tests that found nothing in direction find a strong, steady signal in move size. That contrast is the clearest proof that the pipeline works and that direction really is unpredictable here.

**Predicting risk isn't the same as profiting from it.** The models spot wild days well, but wild days also earn more, so cutting exposure on them cuts reward as much as risk, and frequent switching adds costs. A correct prediction still needs a use that survives costs.

**Make the planted pattern realistic.** The first clustered-volatility check used milder clustering than real markets, and gradient boosting only just missed the bar. Setting it to typical index values (0.10 / 0.88) made the check a fair test of what real data looks like.

**The simpler model keeps winning.** Logistic regression beat gradient boosting on every stock, in both tasks. A more flexible model only helps if the extra patterns it finds carry over to later years; here they didn't.

**Always try the simplest strong rule.** "Same as today" was a fair-sounding bar, and beating it by 8 points looked like a big win. A single better-chosen measure, today's high-to-low swing, matched the models exactly. Before crediting a model, check what one good feature or one line can do.

**Compare like with like.** A rule predicts the same days as the model and agrees with it on most of them. Those shared days say nothing about which is better, so only the days they disagree count (the paired test). On SPY, that's 933 days, split 466 to 467.

**How you use a prediction can matter more than how good it is.** Deciding once a week instead of every day cut trades by about three quarters and lifted the strategy's Sharpe ratio more than any change of model did.

**Get a second pair of eyes.** An independent review found four bugs and several inconsistencies that 77 passing tests had missed, because the tests only checked what had been thought of. Every finding now has a test of its own.

**Small changes can expose a fragile model.** Leaving out 17 training days changed gradient boosting's predictions on 146 days; logistic regression barely moved. A fresh download, with 5 more days and Yahoo's small revisions to old rows, then moved its SPY accuracy by another 0.4 points, while logistic regression stayed put. A model whose output depends on a handful of rows is fitting noise.

**Practical snags:** Claude's workspace can't download market data, the Claude app can't preview notebooks, and spaces in folder names cause Terminal trouble. Each has a workaround in the issue tracker.

## Planned work and open decisions

The roadmap is agreed: seven phases, built in order, each ending with the changed files, this log updated and a local run.

### Roadmap

- [x] **Phase 1, fixes:** drop the unfinished trading day; trim to 5 features (all 8 kept available); first test year set automatically; planted-pattern test
- [x] **Phase 2, automated tests:** `pytest` checks for look-ahead, labels, walk-forward, backtest arithmetic and pattern detection; no internet needed
- [x] **Phase 3, any stock:** notebook becomes a template; `python run.py NVDA` creates and runs `reports/NVDA.ipynb` (papermill); reports for SPY, NVDA, AAPL, GOOGL; comparison notebook with a lucky-winners warning
- [x] **Phase 4, volatility:** predict big vs. small moves, against a "same as today" baseline, in every report
- [x] **Phase 4.1, a tougher bar:** "Recent swing" baseline, paired test against rules, weekly half-on-big strategy with a weekly luck check
- [x] **Phase 4.2, review fixes:** four bugs, paired tests and Holm everywhere, interest on cash, data checks
- [ ] **Phase 5, LSTM:** `src/lstm.py` (PyTorch), same walk-forward test, for direction and volatility; on request only (`RUN_LSTM` switch)
- [ ] **Phase 6, live forward test:** a GitHub Actions job every weekday after the close logs tomorrow's predictions and checks yesterday's
- [ ] **Phase 7, write-up:** result charts in the README; Conclusions written

### Decisions

| Question | Decision |
| --- | --- |
| LSTM in every report? | No, on request: `RUN_LSTM = True` at the top of the template |
| Run tests automatically on GitHub? | No; tests run locally with `pytest` |
| What the live forward test tracks | All four stocks; logistic regression for direction plus logistic regression for move size (best on all four stocks in phase 4); no LSTM |
| Google ticker | `GOOGL` (class A shares) |
| Interest on cash? | Yes: the 13-week Treasury-bill rate (`^IRX`), with Sharpe on returns above cash; `--no-interest` switches it off |
| Correction for many tests | Holm (replaced Bonferroni in phase 4.2) |
| Keep `notebooks/analysis.ipynb`? | Yes, locally only, as a frozen phase 1 snapshot; off GitHub, where it would duplicate `reports/SPY.ipynb`. It may stop running as `src/` changes |
| Known bias | NVDA, AAPL and GOOGL were picked knowing they became huge winners; the reports will say buy-and-hold looks good partly in hindsight |

### My own tasks

- [ ] `git rm --cached .DS_Store`, then commit (issue 35)
- [ ] `pip freeze > requirements-lock.txt`, then commit (issue 36)
- [ ] Experiment: `LogisticRegression(max_iter=1000, class_weight="balanced")` in `src/evaluate.py`; guess the outcome before running
- [ ] Experiment: trade only when the model is confident, e.g. `prob_up > 0.55`
- [ ] Write the Conclusions (phase 7)

## Glossary

| Term | Meaning |
| --- | --- |
| Adjusted prices | Prices corrected for dividends and splits, so day-to-day changes are the real return |
| Backtest | Replaying a trading rule on past data to see what it would have earned |
| Baseline | A simple rule a model must beat to show it learned something |
| Bonferroni correction | Dividing the 5% threshold by the number of comparisons, so that testing many things doesn't produce lucky "wins" |
| Excess return | Return above what cash would have earned; the Sharpe ratio uses it |
| Feature | An input to the model, such as the last 5 days' return |
| GARCH | A standard model of volatility clustering: today's volatility depends on yesterday's move and yesterday's volatility. Used for the fake clustered prices |
| Hindsight bias | Judging a choice by an outcome nobody could know at the time, such as picking stocks that are famous winners today |
| Holm correction | A step-by-step version of Bonferroni: the smallest p-value must clear 5% / m, the next 5% / (m − 1), and so on, stopping at the first miss; same guarantee, never fewer passes |
| Label (target) | What the model predicts: 1 for an up day tomorrow, 0 otherwise |
| Leak (look-ahead) | Future information slipping into features or training, which makes results look better than they could be in real life |
| LSTM | A neural network built for sequences; needs PyTorch or TensorFlow |
| Max drawdown | The worst fall from a peak to a low |
| McNemar's test (paired test) | Comparing two predictors on the same days by counting only the days they disagree: if they were equally good, each would win about half |
| Multicollinearity | Features that move together, so a model can split or cancel weight between them |
| Overfitting | A model fitting noise in the training data that doesn't repeat later |
| p-value | Chance of a result at least this good from pure luck |
| papermill | A tool that fills in a notebook's settings cell and runs every cell, saving the result |
| Random walk | Prices where every day's move is pure chance |
| Recent swing | The one-line move-size rule: predict a big day when today's high-to-low swing was wider than its median over the past 250 trading days |
| scikit-learn | The Python library that provides the models and data-preparation tools |
| Sharpe ratio | Return above cash per unit of volatility; higher is better |
| Treasury bill (`^IRX`) | Short-term US government debt; its yield is the standard "risk-free" rate that cash earns |
| Virtual environment (.venv) | A private set of Python libraries for this project only |
| Volatility clustering | Calm days tend to follow calm days, wild days follow wild days |
| Volatility targeting | Holding less of an investment when it is expected to be wild, to keep risk steadier |
| Walk-forward testing | Train on the past, test on the next period, step forward, repeat |

## Changelog

| Date | Change | Details |
| --- | --- | --- |
| 16 Oct 2026 | Four-stock run with phase 4.2 | Fresh prices to 14 Oct 2026 and interest on cash (1.48% a year on average); conclusions unchanged: 0 of 8 direction, 1 of 8 move size against Recent swing after Holm (NVDA), 0 of 16 strategies beat luck; interest lowers every Sharpe ratio by a similar amount |
| 16 Oct 2026 | Phase 4.2 built | Fixes for an independent code review (issues 23–37): Part 2 uses `--first-test-year`; failed runs keep the last good report; `--comparison-only` and `--no-interest`; clear errors for tickers without volume; paired tests and Holm everywhere, with chart ranges from the paired counts; interest on cash and Sharpe on excess returns; last training day left out; 6 pm cutoff and High/Low repairs; luck share never 0. 29 new tests (106 total), checked by planting 18 bugs (3 missed at first, so 3 tests were added); checked end to end on SPY plus a fake stock |
| 16 Oct 2026 | Four-stock run with phase 4.1 | Logistic regression beats Recent swing on NVDA, AAPL and GOOGL (3 of 8 at 5%, 1 of 8 after the correction) and ties on SPY; weekly beats daily in all 8 rows; 0 of 16 strategies pass the luck check |
| 16 Oct 2026 | Phase 4.1 built | Recent swing baseline (`wide_range` column), paired test (`paired_p_value`), weekly strategy (`hold_for_week`) with a whole-week luck check; Part 2 of the report and the comparison judged against Recent swing; luck correction counts daily and weekly strategies; 10 new tests (77 total), checked by planting 7 bugs (one was missed at first, so a test was added); checked end to end on SPY plus a fake stock. On SPY the rule ties logistic regression (61.65% vs 61.62%) |
| 15 Oct 2026 | Four-stock run with Part 2 | Move size: 8 of 8 beat "same as today" after the correction; half-on-big: 0 of 8 beat luck. Logistic regression chosen as the move-size model for phase 6 |
| 15 Oct 2026 | Phase 4 built | Move-size task in every report (Part 2) and in the comparison; new features, label, baselines, clustered-volatility fake prices, half-position backtest, luck check on Sharpe; 21 new tests (67 total), each new bug type checked by planting it; checked on the saved SPY data plus fake stocks |
| 12 Oct 2026 | NVDA/GOOGL match checked | Identical "always up" accuracy traced to equal up-day counts by coincidence (issue 19) |
| 12 Oct 2026 | First four-stock run | SPY, NVDA, AAPL, GOOGL run locally; results under Results so far; 0 of 8 accuracy passes, 1 of 8 backtest passes at 5% and 0 after the correction |
| 12 Oct 2026 | `analysis.ipynb` kept locally | Untracked from GitHub with `git rm --cached` and listed in `.gitignore`; every earlier version stays in the repo's history. Fixed a `.gitignore` line-break problem on the way (issue 18) |
| 12 Oct 2026 | Phase 3 built | `notebooks/template.ipynb` replaces `analysis.ipynb`; `run.py` creates `reports/<TICKER>.ipynb` per stock and `reports/comparison.ipynb`; `src/report.py` for summaries and the Bonferroni-corrected comparison; hindsight note in every report; 15 new tests (46 total); checked end to end on the saved SPY data plus fake NVDA, AAPL, GOOGL and a short-history stock; `papermill` added to requirements |
| 9 Oct 2026 | Phase 2 built | 31 `pytest` tests in `tests/` (data, features, evaluation); run in about 4 seconds without internet; each checked against planted bugs; README section and `pytest` in requirements |
| 9 Oct 2026 | Phase 1 files reviewed | Local copies of `data.py`, `features.py` and `evaluate.py` checked line by line; one crash-causing mix-up in `walk_forward` found and fixed |
| 9 Oct 2026 | Phase 1 built | Unfinished day dropped; 5 features (all 8 kept); automatic first test year; planted-pattern check; luck check added to the backtest; notebook and README updated; checked on the saved SPY CSV |
| 9 Oct 2026 | Roadmap agreed | Seven phases; decisions recorded under Planned work |
| 9 Oct 2026 | Project log added to the repo | `PROJECT_LOG.md`, linked from the README |
| 9 Oct 2026 | GitHub prep | `.gitignore` now also skips macOS `.DS_Store` files |
| 9 Oct 2026 | README: returning-session commands | Added "Every time you come back", "When you're done" and "Getting the latest prices"; first-time setup now uses `python3` for Mac |
| 9 Oct 2026 | Project log created | This document |
| 9 Oct 2026 | Next round proposed | Six items, later folded into the roadmap |
| 9 Oct 2026 | Results reproduced from the saved CSV | Same numbers as the local run; found the unfinished-day issue and the feature overlap |
| 9 Oct 2026 | First real run on SPY | Run locally; results under Results so far |
| 9 Oct 2026 | README updated | Folder renamed to `Stock_Movement_Predictor` |
| 9 Oct 2026 | Version 1 built | `src/data.py`, `src/features.py`, `src/evaluate.py`, `notebooks/analysis.ipynb`, README, requirements; tested on fake prices; two bugs fixed during testing |
| 9 Oct 2026 | Plan agreed | SPY daily data from 2005; logic in `src/` plus one notebook; logistic regression and gradient boosting; walk-forward testing |