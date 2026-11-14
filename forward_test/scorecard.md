# Forward test scorecard

*Up to the close of 2026-11-13; latest outcome: 2026-11-13. Predictions are written before the day they predict; see `predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-13: 24 in all, 4 still waiting for their day to close.

## Direction: logistic regression vs "always up"

| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |
|---|---|---|---|---|---|
| AAPL | 5 | 60.0% | 60.0% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 5 | 60.0% | 60.0% | +0.0 ± 78.4 | 0.688 |
| NVDA | 5 | 20.0% | 20.0% | +0.0 ± 0.0 | 1.000 |
| SPY | 5 | 40.0% | 40.0% | +0.0 ± 0.0 | 1.000 |
| All four | 20 | 45.0% | 45.0% | +0.0 ± 19.6 | 0.688 |

## Move size: logistic regression vs the Recent swing rule

| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |
|---|---|---|---|---|---|---|
| AAPL | 5 | 80.0% | 80.0% | 20.0% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 5 | 40.0% | 0.0% | 20.0% | +40.0 ± 42.9 | 0.250 |
| NVDA | 5 | 100.0% | 100.0% | 100.0% | +0.0 ± 0.0 | 1.000 |
| SPY | 5 | 40.0% | 40.0% | 20.0% | +0.0 ± 0.0 | 1.000 |
| All four | 20 | 65.0% | 55.0% | 40.0% | +10.0 ± 13.1 | 0.250 |

## How much to read into this

Each stock has at most 5 scored days. An accuracy over n days is uncertain by about ±45 points, so early numbers swing a lot. In the backtests, logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. The two agree on about 4 days in 5, so only the other days carry evidence, and telling an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 years with all four stocks together (the "All four" row), though the stocks move together, so their days aren't fully independent. A year is enough to catch a large edge, or to rule one out. The p-values are paired tests, as in the reports, and are not corrected for the four stocks and two tasks. They are one-sided (is the model better?), while the 95% ranges are two-sided, so a range can just touch zero while the p-value is below 0.05.
