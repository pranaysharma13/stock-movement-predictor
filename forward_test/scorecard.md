# Forward test scorecard

*Up to the close of 2026-11-27; latest outcome: 2026-11-27. Predictions are written before the day they predict; see `predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-27: 60 in all, 4 still waiting for their day to close.

## Direction: logistic regression vs "always up"

| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |
|---|---|---|---|---|---|
| AAPL | 14 | 57.1% | 57.1% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 14 | 57.1% | 42.9% | +14.3 ± 38.9 | 0.363 |
| NVDA | 14 | 28.6% | 21.4% | +7.1 ± 13.5 | 0.500 |
| SPY | 14 | 35.7% | 28.6% | +7.1 ± 13.5 | 0.500 |
| All four | 56 | 44.6% | 37.5% | +7.1 ± 10.9 | 0.172 |

## Move size: logistic regression vs the Recent swing rule

| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |
|---|---|---|---|---|---|---|
| AAPL | 14 | 64.3% | 64.3% | 42.9% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 14 | 57.1% | 35.7% | 35.7% | +21.4 ± 29.2 | 0.188 |
| NVDA | 14 | 85.7% | 85.7% | 78.6% | +0.0 ± 0.0 | 1.000 |
| SPY | 14 | 57.1% | 57.1% | 35.7% | +0.0 ± 0.0 | 1.000 |
| All four | 56 | 66.1% | 60.7% | 48.2% | +5.4 ± 7.7 | 0.188 |

## How much to read into this

Each stock has at most 14 scored days. An accuracy over n days is uncertain by about ±27 points, so early numbers swing a lot. In the backtests, logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. The two agree on about 4 days in 5, so only the other days carry evidence, and telling an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 years with all four stocks together (the "All four" row), though the stocks move together, so their days aren't fully independent. A year is enough to catch a large edge, or to rule one out. The p-values are paired tests, as in the reports, and are not corrected for the four stocks and two tasks. They are one-sided (is the model better?), while the 95% ranges are two-sided, so a range can just touch zero while the p-value is below 0.05.
