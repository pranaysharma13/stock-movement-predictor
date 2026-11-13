# Forward test scorecard

*Up to the close of 2026-11-12; latest outcome: 2026-11-12. Predictions are written before the day they predict; see `predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-12: 20 in all, 4 still waiting for their day to close.

## Direction: logistic regression vs "always up"

| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |
|---|---|---|---|---|---|
| AAPL | 4 | 50.0% | 50.0% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 4 | 50.0% | 75.0% | -25.0 ± 81.3 | 0.875 |
| NVDA | 4 | 25.0% | 25.0% | +0.0 ± 0.0 | 1.000 |
| SPY | 4 | 50.0% | 50.0% | +0.0 ± 0.0 | 1.000 |
| All four | 16 | 43.8% | 50.0% | -6.2 ± 21.0 | 0.875 |

## Move size: logistic regression vs the Recent swing rule

| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |
|---|---|---|---|---|---|---|
| AAPL | 4 | 75.0% | 75.0% | 0.0% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 4 | 50.0% | 0.0% | 25.0% | +50.0 ± 49.0 | 0.250 |
| NVDA | 4 | 100.0% | 100.0% | 100.0% | +0.0 ± 0.0 | 1.000 |
| SPY | 4 | 50.0% | 50.0% | 0.0% | +0.0 ± 0.0 | 1.000 |
| All four | 16 | 68.8% | 56.2% | 31.2% | +12.5 ± 16.2 | 0.250 |

## How much to read into this

Each stock has at most 4 scored days. An accuracy over n days is uncertain by about ±50 points, so early numbers swing a lot. In the backtests, logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. The two agree on about 4 days in 5, so only the other days carry evidence, and telling an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 years with all four stocks together (the "All four" row), though the stocks move together, so their days aren't fully independent. A year is enough to catch a large edge, or to rule one out. The p-values are paired tests, as in the reports, and are not corrected for the four stocks and two tasks. They are one-sided (is the model better?), while the 95% ranges are two-sided, so a range can just touch zero while the p-value is below 0.05.
