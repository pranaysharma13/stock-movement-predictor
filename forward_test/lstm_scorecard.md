# LSTM forward test scorecard (move size)

*Up to the close of 2026-11-19; latest outcome: 2026-11-19. Predictions are written before the day they predict; see `lstm_predictions.csv` and its git history.*

Predictions made from 2026-11-06 to 2026-11-19: 36 scored, 4 still waiting.

| Stock | Model trained in | Trained through | Epochs chosen (each seed) |
| --- | --- | --- | --- |
| SPY | 2026-11 | 2026-11-05 | 4, 5, 1 |
| NVDA | 2026-11 | 2026-11-05 | 3, 2, 4 |
| AAPL | 2026-11 | 2026-11-05 | 3, 3, 9 |
| GOOGL | 2026-11 | 2026-11-05 | 1, 4, 13 |

## Against the Recent swing rule

| Stock | Days scored | LSTM | Recent swing | Edge | p-value |
| --- | --- | --- | --- | --- | --- |
| SPY | 9 | 55.6% | 55.6% | +0.0 ± 0.0 points | 1.000 |
| NVDA | 9 | 88.9% | 88.9% | +0.0 ± 0.0 points | 1.000 |
| AAPL | 9 | 33.3% | 66.7% | -33.3 ± 43.6 points | 0.969 |
| GOOGL | 9 | 66.7% | 33.3% | +33.3 ± 43.6 points | 0.188 |
| All four | 36 | 61.1% | 61.1% | +0.0 ± 17.2 points | 0.623 |

## Against logistic regression, on the same days

| Stock | Days | LSTM | Logistic regression | p-value |
| --- | --- | --- | --- | --- |
| SPY | 9 | 55.6% | 55.6% | 1.000 |
| NVDA | 9 | 88.9% | 88.9% | 1.000 |
| AAPL | 9 | 33.3% | 66.7% | 0.969 |
| GOOGL | 9 | 66.7% | 55.6% | 0.500 |
| All four | 36 | 61.1% | 66.7% | 0.891 |

## How much to read into this

Each stock has at most 9 scored days, so each accuracy is uncertain by about ±33 points. The backtest edge over Recent swing was 1.5 to 3.7 points; confirming an edge that size takes roughly 1,400 to 3,800 scored days per stock. The p-values are paired tests, one-sided and not corrected for the four stocks; the ranges are two-sided.
