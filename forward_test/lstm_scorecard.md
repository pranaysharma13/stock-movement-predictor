# LSTM forward test scorecard (move size)

*Up to the close of 2026-11-25; latest outcome: 2026-11-25. Predictions are written before the day they predict; see `lstm_predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-25: 52 scored, 4 still waiting.

| Stock | Model trained in | Trained through | Epochs chosen (each seed) |
| --- | --- | --- | --- |
| SPY | 2026-11 | 2026-11-05 | 4, 5, 1 |
| NVDA | 2026-11 | 2026-11-05 | 3, 2, 4 |
| AAPL | 2026-11 | 2026-11-05 | 3, 3, 9 |
| GOOGL | 2026-11 | 2026-11-05 | 1, 4, 13 |

## Against the Recent swing rule

| Stock | Days scored | LSTM | Recent swing | Edge | p-value |
| --- | --- | --- | --- | --- | --- |
| SPY | 13 | 53.8% | 53.8% | +0.0 ± 0.0 points | 1.000 |
| NVDA | 13 | 92.3% | 92.3% | +0.0 ± 0.0 points | 1.000 |
| AAPL | 13 | 46.2% | 69.2% | -23.1 ± 31.3 points | 0.969 |
| GOOGL | 13 | 69.2% | 38.5% | +30.8 ± 32.9 points | 0.109 |
| All four | 52 | 65.4% | 63.5% | +1.9 ± 12.5 points | 0.500 |

## Against logistic regression, on the same days

| Stock | Days | LSTM | Logistic regression | p-value |
| --- | --- | --- | --- | --- |
| SPY | 13 | 53.8% | 53.8% | 1.000 |
| NVDA | 13 | 92.3% | 92.3% | 1.000 |
| AAPL | 13 | 46.2% | 69.2% | 0.969 |
| GOOGL | 13 | 69.2% | 61.5% | 0.500 |
| All four | 52 | 65.4% | 69.2% | 0.891 |

## How much to read into this

Each stock has at most 13 scored days, so each accuracy is uncertain by about ±28 points. The backtest edge over Recent swing was 1.5 to 3.7 points; confirming an edge that size takes roughly 1,400 to 3,800 scored days per stock. The p-values are paired tests, one-sided and not corrected for the four stocks; the ranges are two-sided.
