# Forward test scorecard

*Up to the close of 2026-11-23; latest outcome: 2026-11-23. Predictions are written before the day they predict; see `predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-23: 48 in all, 4 still waiting for their day to close.

## Direction: logistic regression vs "always up"

| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |
|---|---|---|---|---|---|
| AAPL | 11 | 45.5% | 45.5% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 11 | 63.6% | 45.5% | +18.2 ± 42.3 | 0.344 |
| NVDA | 11 | 18.2% | 18.2% | +0.0 ± 0.0 | 1.000 |
| SPY | 11 | 27.3% | 27.3% | +0.0 ± 0.0 | 1.000 |
| All four | 44 | 38.6% | 34.1% | +4.5 ± 10.8 | 0.344 |

## Move size: logistic regression vs the Recent swing rule

| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |
|---|---|---|---|---|---|---|
| AAPL | 11 | 72.7% | 72.7% | 36.4% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 11 | 63.6% | 45.5% | 36.4% | +18.2 ± 34.0 | 0.312 |
| NVDA | 11 | 90.9% | 90.9% | 81.8% | +0.0 ± 0.0 | 1.000 |
| SPY | 11 | 63.6% | 63.6% | 36.4% | +0.0 ± 0.0 | 1.000 |
| All four | 44 | 72.7% | 68.2% | 47.7% | +4.5 ± 8.8 | 0.312 |

## How much to read into this

Each stock has at most 11 scored days. An accuracy over n days is uncertain by about ±30 points, so early numbers swing a lot. In the backtests, logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. The two agree on about 4 days in 5, so only the other days carry evidence, and telling an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 years with all four stocks together (the "All four" row), though the stocks move together, so their days aren't fully independent. A year is enough to catch a large edge, or to rule one out. The p-values are paired tests, as in the reports, and are not corrected for the four stocks and two tasks. They are one-sided (is the model better?), while the 95% ranges are two-sided, so a range can just touch zero while the p-value is below 0.05.
