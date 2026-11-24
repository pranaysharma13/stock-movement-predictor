# LSTM forward test scorecard (move size)

*Up to the close of 2026-11-23; latest outcome: 2026-11-23. Predictions are written before the day they predict; see `lstm_predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-23: 44 scored, 4 still waiting.

| Stock | Model trained in | Trained through | Epochs chosen (each seed) |
| --- | --- | --- | --- |
| SPY | 2026-11 | 2026-11-05 | 4, 5, 1 |
| NVDA | 2026-11 | 2026-11-05 | 3, 2, 4 |
| AAPL | 2026-11 | 2026-11-05 | 3, 3, 9 |
| GOOGL | 2026-11 | 2026-11-05 | 1, 4, 13 |

## Against the Recent swing rule

| Stock | Days scored | LSTM | Recent swing | Edge | p-value |
| --- | --- | --- | --- | --- | --- |
| SPY | 11 | 63.6% | 63.6% | +0.0 ± 0.0 points | 1.000 |
| NVDA | 11 | 90.9% | 90.9% | +0.0 ± 0.0 points | 1.000 |
| AAPL | 11 | 45.5% | 72.7% | -27.3 ± 36.4 points | 0.969 |
| GOOGL | 11 | 72.7% | 45.5% | +27.3 ± 36.4 points | 0.188 |
| All four | 44 | 68.2% | 68.2% | +0.0 ± 14.1 points | 0.623 |

## Against logistic regression, on the same days

| Stock | Days | LSTM | Logistic regression | p-value |
| --- | --- | --- | --- | --- |
| SPY | 11 | 63.6% | 63.6% | 1.000 |
| NVDA | 11 | 90.9% | 90.9% | 1.000 |
| AAPL | 11 | 45.5% | 72.7% | 0.969 |
| GOOGL | 11 | 72.7% | 63.6% | 0.500 |
| All four | 44 | 68.2% | 72.7% | 0.891 |

## How much to read into this

Each stock has at most 11 scored days, so each accuracy is uncertain by about ±30 points. The backtest edge over Recent swing was 1.5 to 3.7 points; confirming an edge that size takes roughly 1,400 to 3,800 scored days per stock. The p-values are paired tests, one-sided and not corrected for the four stocks; the ranges are two-sided.
