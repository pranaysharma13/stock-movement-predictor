# Forward test scorecard

*Up to the close of 2026-11-25; latest outcome: 2026-11-25. Predictions are written before the day they predict; see `predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-25: 56 in all, 4 still waiting for their day to close.

## Direction: logistic regression vs "always up"

| Stock | Days | Model | Always up | Edge (points, 95% range) | p-value |
|---|---|---|---|---|---|
| AAPL | 13 | 53.8% | 53.8% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 13 | 61.5% | 38.5% | +23.1 ± 37.9 | 0.227 |
| NVDA | 13 | 23.1% | 15.4% | +7.7 ± 14.5 | 0.500 |
| SPY | 13 | 30.8% | 23.1% | +7.7 ± 14.5 | 0.500 |
| All four | 52 | 42.3% | 32.7% | +9.6 ± 11.0 | 0.090 |

## Move size: logistic regression vs the Recent swing rule

| Stock | Days | Model | Recent swing | Same as today | Edge over Recent swing (points, 95% range) | p-value |
|---|---|---|---|---|---|---|
| AAPL | 13 | 69.2% | 69.2% | 38.5% | +0.0 ± 0.0 | 1.000 |
| GOOGL | 13 | 61.5% | 38.5% | 38.5% | +23.1 ± 31.3 | 0.188 |
| NVDA | 13 | 92.3% | 92.3% | 84.6% | +0.0 ± 0.0 | 1.000 |
| SPY | 13 | 53.8% | 53.8% | 38.5% | +0.0 ± 0.0 | 1.000 |
| All four | 52 | 69.2% | 63.5% | 50.0% | +5.8 ± 8.3 | 0.188 |

## How much to read into this

Each stock has at most 13 scored days. An accuracy over n days is uncertain by about ±28 points, so early numbers swing a lot. In the backtests, logistic regression beat the Recent swing rule by 1.5 to 2.5 points on single stocks. The two agree on about 4 days in 5, so only the other days carry evidence, and telling an edge that size from luck takes about 1,400 to 3,800 scored days: roughly 1.5 to 4 years with all four stocks together (the "All four" row), though the stocks move together, so their days aren't fully independent. A year is enough to catch a large edge, or to rule one out. The p-values are paired tests, as in the reports, and are not corrected for the four stocks and two tasks. They are one-sided (is the model better?), while the 95% ranges are two-sided, so a range can just touch zero while the p-value is below 0.05.
