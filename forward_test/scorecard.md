# Forward test scorecard

*Up to the close of 2026-11-24; latest outcome: 2026-11-24. Predictions are written before the day they predict; see `predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-24: 52 in all, 4 still waiting for their day to close.

## Direction: logistic regression vs "always up"

| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |
|---|---|---|---|---|---|
| AAPL | 12 | 50.0% | 50.0% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 12 | 58.3% | 41.7% | +16.7 ± 38.9 | 0.344 |
| NVDA | 12 | 25.0% | 16.7% | +8.3 ± 15.6 | 0.500 |
| SPY | 12 | 33.3% | 25.0% | +8.3 ± 15.6 | 0.500 |
| All four | 48 | 41.7% | 33.3% | +8.3 ± 11.3 | 0.145 |

## Move size: logistic regression vs the Recent swing rule

| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |
|---|---|---|---|---|---|---|
| AAPL | 12 | 75.0% | 75.0% | 41.7% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 12 | 58.3% | 41.7% | 33.3% | +16.7 ± 31.3 | 0.312 |
| NVDA | 12 | 91.7% | 91.7% | 83.3% | +0.0 ± 0.0 | 1.000 |
| SPY | 12 | 58.3% | 58.3% | 33.3% | +0.0 ± 0.0 | 1.000 |
| All four | 48 | 70.8% | 66.7% | 47.9% | +4.2 ± 8.1 | 0.312 |

## How much to read into this

Each stock has at most 12 scored days. An accuracy over n days is uncertain by about ±29 points, so early numbers swing a lot. In the backtests, logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. The two agree on about 4 days in 5, so only the other days carry evidence, and telling an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 years with all four stocks together (the "All four" row), though the stocks move together, so their days aren't fully independent. A year is enough to catch a large edge, or to rule one out. The p-values are paired tests, as in the reports, and are not corrected for the four stocks and two tasks. They are one-sided (is the model better?), while the 95% ranges are two-sided, so a range can just touch zero while the p-value is below 0.05.
