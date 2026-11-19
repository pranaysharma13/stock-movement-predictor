# LSTM forward test scorecard (move size)

*Up to the close of 2026-11-18; latest outcome: 2026-11-18. Predictions are written before the day they predict; see `lstm_predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-18: 32 scored, 4 still waiting.

| Stock | Model trained in | Trained through | Epochs chosen (each seed) |
| --- | --- | --- | --- |
| SPY | 2026-11 | 2026-11-05 | 4, 5, 1 |
| NVDA | 2026-11 | 2026-11-05 | 3, 2, 4 |
| AAPL | 2026-11 | 2026-11-05 | 3, 3, 9 |
| GOOGL | 2026-11 | 2026-11-05 | 1, 4, 13 |

## Against the Recent swing rule

| Stock | Days scored | LSTM | Recent swing | Edge | p-value |
| --- | --- | --- | --- | --- | --- |
| SPY | 8 | 50.0% | 50.0% | +0.0 ± 0.0 points | 1.000 |
| NVDA | 8 | 87.5% | 87.5% | +0.0 ± 0.0 points | 1.000 |
| AAPL | 8 | 37.5% | 75.0% | -37.5 ± 48.2 points | 0.969 |
| GOOGL | 8 | 62.5% | 37.5% | +25.0 ± 45.8 points | 0.312 |
| All four | 32 | 59.4% | 62.5% | -3.1 ± 18.3 points | 0.746 |

## Against logistic regression, on the same days

| Stock | Days | LSTM | Logistic regression | p-value |
| --- | --- | --- | --- | --- |
| SPY | 8 | 50.0% | 50.0% | 1.000 |
| NVDA | 8 | 87.5% | 87.5% | 1.000 |
| AAPL | 8 | 37.5% | 75.0% | 0.969 |
| GOOGL | 8 | 62.5% | 50.0% | 0.500 |
| All four | 32 | 59.4% | 65.6% | 0.891 |

## How much to read into this

Each stock has at most 8 scored days, so each accuracy is uncertain by about ±35 points. The backtest edge over Recent swing was 1.5 to 3.7 points; confirming an edge that size takes roughly 1,400 to 3,800 scored days per stock. The p-values are paired tests, one-sided and not corrected for the four stocks; the ranges are two-sided.
