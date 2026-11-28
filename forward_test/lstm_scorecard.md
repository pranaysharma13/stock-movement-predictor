# LSTM forward test scorecard (move size)

*Up to the close of 2026-11-27; latest outcome: 2026-11-27. Predictions are written before the day they predict; see `lstm_predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-27: 56 scored, 4 still waiting.

| Stock | Model trained in | Trained through | Epochs chosen (each seed) |
| --- | --- | --- | --- |
| SPY | 2026-11 | 2026-11-05 | 4, 5, 1 |
| NVDA | 2026-11 | 2026-11-05 | 3, 2, 4 |
| AAPL | 2026-11 | 2026-11-05 | 3, 3, 9 |
| GOOGL | 2026-11 | 2026-11-05 | 1, 4, 13 |

## Against the Recent swing rule

| Stock | Days scored | LSTM | Recent swing | Edge | p-value |
| --- | --- | --- | --- | --- | --- |
| SPY | 14 | 57.1% | 57.1% | +0.0 ± 0.0 points | 1.000 |
| NVDA | 14 | 85.7% | 85.7% | +0.0 ± 0.0 points | 1.000 |
| AAPL | 14 | 42.9% | 64.3% | -21.4 ± 29.2 points | 0.969 |
| GOOGL | 14 | 64.3% | 35.7% | +28.6 ± 30.9 points | 0.109 |
| All four | 56 | 62.5% | 60.7% | +1.8 ± 11.6 points | 0.500 |

## Against logistic regression, on the same days

| Stock | Days | LSTM | Logistic regression | p-value |
| --- | --- | --- | --- | --- |
| SPY | 14 | 57.1% | 57.1% | 1.000 |
| NVDA | 14 | 85.7% | 85.7% | 1.000 |
| AAPL | 14 | 42.9% | 64.3% | 0.969 |
| GOOGL | 14 | 64.3% | 57.1% | 0.500 |
| All four | 56 | 62.5% | 66.1% | 0.891 |

## How much to read into this

Each stock has at most 14 scored days, so each accuracy is uncertain by about ±27 points. The backtest edge over Recent swing was 1.5 to 3.7 points; confirming an edge that size takes roughly 1,400 to 3,800 scored days per stock. The p-values are paired tests, one-sided and not corrected for the four stocks; the ranges are two-sided.
