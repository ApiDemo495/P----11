# Alpha lab — purged walk-forward, real Binance 1h history

Data: `backtest/cache/{SYM}_1h.csv` (Binance Vision mirror, 2026-01-01 → 2026-09-30, 6,552 bars/asset).

BEFORE = classic technical composite (momentum / efficiency / range position), |conf| from the composite, fixed 2:1 vol barriers.
AFTER = + taker-imbalance & trade-size & VWAP-deviation block, orthogonalised against the TA block, IC-shrunk weights fitted on TRAIN folds only, meta-label success filter, isotonic confidence, barrier multiples fitted on TRAIN folds only.

| Asset | step | n | B acc | A acc | B acc@floor | A acc@gated | Brier raw→iso | slope | B EV bp | A EV bp | gated EV bp | |IC| ta/flow | IC before→after | gate cov |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 2h | 4446 | 47.8% | 51.6% | 46.3% | 56.4% | 0.259→0.251 | 0.04 | -15.0 | -11.6 | -8.9 | 0.020/0.025 | -0.045→+0.048 | 6% |
| BTC | 4h | 4446 | 47.7% | 50.8% | 45.8% | 56.4% | 0.260→0.252 | 0.03 | -19.4 | -14.1 | -7.9 | 0.020/0.025 | -0.040→+0.026 | 5% |
| ETH | 2h | 4446 | 47.4% | 53.0% | 45.8% | 55.2% | 0.260→0.251 | 0.04 | -19.3 | -8.3 | -11.4 | 0.018/0.028 | -0.062→+0.054 | 7% |
| ETH | 4h | 4446 | 46.4% | 53.1% | 49.1% | 54.0% | 0.263→0.252 | 0.03 | -27.1 | +5.3 | -17.4 | 0.020/0.027 | -0.065→+0.046 | 9% |
| BNB | 2h | 4446 | 47.4% | 52.5% | 49.3% | 50.7% | 0.258→0.254 | 0.02 | -15.6 | -10.8 | -15.6 | 0.025/0.017 | -0.041→+0.034 | 3% |
| BNB | 4h | 4446 | 47.4% | 51.2% | 50.0% | 47.3% | 0.259→0.254 | 0.00 | -16.2 | -6.4 | -20.6 | 0.034/0.028 | -0.034→+0.032 | 4% |
| PAXG | 2h | 4446 | 49.5% | 49.7% | 45.3% | 52.3% | 0.265→0.253 | 0.01 | -17.2 | -18.8 | -16.4 | 0.016/0.027 | -0.008→+0.013 | 6% |
| PAXG | 4h | 4446 | 50.3% | 50.0% | 42.6% | 51.5% | 0.269→0.256 | 0.00 | -16.1 | -19.5 | -18.1 | 0.027/0.020 | +0.011→+0.010 | 9% |
