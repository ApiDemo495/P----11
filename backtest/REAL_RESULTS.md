# REAL BACKTEST RESULTS — Mode A (app logic on real Binance history)

Run: 2026-11-07 (agent sandbox). This file is updated only by real
runs; synthetic smoke runs never write here.

## Data provenance (read this before quoting any number)
- Source: Binance Vision 1h klines (BTCUSDT/ETHUSDT/BNBUSDT/PAXGUSDT),
  mirrored on GitHub (KevinSherry2016/Crypto, marketData/unzipped —
  byte-format identical to data.binance.vision daily files) because
  the sandbox cannot reach Binance directly.
- Validation: 24,096 hourly bars per BTC/ETH/BNB (2024-01-01 →
  2026-09-30) and 13,262 for PAXG (from 2025-03-27); ZERO timestamp
  gaps, ZERO malformed OHLC rows. Spot anchors match known history
  (BTC $124,495 on 2025-10-06 ATH week; PAXG $3,940 same day).
- Window backtested: 2026-07-09 → 2026-09-30 (last 2,000 1h bars,
  all four assets) — ONE ~83-day regime, not a full market cycle.

## Method
App's own logic end-to-end (PredictionEngine direction + moves_risk
2:1 levels frozen at entry), anti-look-ahead DataSlice, one position
at a time, 22-bar horizon, first-touch settlement (SL wins ties),
friction BTC/ETH/BNB 12 bps / PAXG 18 bps round trip charged in R.
Win% intervals are Wilson 95% — with these n they are WIDE.

## Results
| Asset | Win % (95% CI) | n | Breakeven % | EV (R) | Profit factor | Max DD % | Sharpe/trade |
|---|---|---|---|---|---|---|---|
| BNB | 45.21% (34.3–56.6) | 73 | 55.33% | -0.125R | 0.666 | 9.95% | -0.167 |
| BTC | 39.22% (27.0–52.9) | 51 | 46.24% | -0.085R | 0.75 | 9.53% | -0.115 |
| ETH | 21.43% (10.2–39.5) | 28 | 45.02% | -0.235R | 0.333 | 6.46% | -0.394 |
| PAXG | 42.59% (30.3–55.8) | 54 | 35.96% | +0.077R | 1.321 | 5.57% | 0.103 |
| **ALL** | **39.81% (33.4–46.6)** | **206** | **46.39%** | **-0.077R** | **0.764** | **18.99%** | **-0.105** |

## Reading it honestly
- Only PAXG sits above its breakeven (EV +0.077R, PF 1.32) — and its
  CI still includes breakeven-level performance. BTC/ETH/BNB are
  below breakeven on this window: the raw app, unfiltered, LOST
  money on this period (combined EV -0.077R/trade, PF 0.764).
- 23 correlated-drawdown events flagged where BTC+ETH+BNB were
  simultaneously in losing trades (worst: 6 open losers at once,
  2026-08-04 window) — the four assets do NOT diversify each other
  in a sell-off; size accordingly.
- n=28–73 per asset is small. This is a measurement, not a verdict:
  rerun on other windows (the Codespace auto-backtest does 15m
  daily) before changing strategy maths on the strength of it.

Full trade log: backtest/results/mode_a_trades_real_1h.csv
Reproduce (Codespace, Binance reachable):
  python -m backtest.cli --mode A --source ccxt --tf 15m

## Second run — same data, LIVE LEARNING switched ON
`BACKTEST_RECORD=1` (engine record=True: CCI weight learning, Brain-4,
flywire RPE weight updates, physics alpha + emotion reward learning
all active between decisions, as in the live app).

| Asset | Win % | n | Breakeven % | EV (R) | PF | Max DD % |
|---|---|---|---|---|---|---|
| BTC | 43.68% | 87 | 51.09% | -0.090R | 0.742 | 11.72% |
| ETH | 32.18% | 87 | 51.58% | -0.222R | 0.445 | 17.87% |
| BNB | 39.08% | 87 | 52.41% | -0.171R | 0.583 | 15.82% |
| PAXG | 36.78% | 87 | 49.74% | -0.150R | 0.588 | 14.81% |
| ALL | 37.93% | 348 | 51.11% | -0.158R | 0.584 | 43.54% |

Verdict: on this window, learning made it WORSE, not better (EV
-0.077R -> -0.158R per trade; PAXG flipped from +0.077R to -0.150R).
The reward-learning loops as currently tuned chase the last move
and degrade the signal on 1h bars. Trade log:
backtest/results/mode_a_trades_real_1h_learning.csv

## Improvement experiments (walk-forward, same real window)
344 decision points collected once; ~50 variants settled offline
(direction rules incl. inversions, barrier scale k, horizons,
filters). Selection on first 55% of decisions, judged ONLY on the
last 45%. Rejected in-sample-only winners are not listed as wins.

Baseline (engine tf1, k=1, h=22) on test: 37.4% win, EV -0.149R.

REMOVED as harmful on evidence:
- live learning loops (run 2: EV -0.077R -> -0.158R)
- trend/momentum alignment filters (train AND test worse)
- confidence>=median filter (worse both halves)

VALIDATED POLICY (now backtest/policy.py, CLI --policy validated):
direction = INVERSE of engine TF4 direction (the macro timeframe
factor is anti-predictive on 1h bars), only when ATR% is in the
top tercile of its trailing window, barriers k=0.75 (2:1 kept).
- train: 58.6% win, EV +0.252R, PF 1.56 (n=29)
- TEST:  58.3% win, EV +0.254R, PF 1.74 (n=48)
- full window: 58.4%, EV +0.253R (n=77)
Per-asset test inside the policy: BTC 75.0% (n=12), ETH 80.0%
(n=10), BNB 41.7% (n=24, EV +0.061R) — BTC/ETH above 60% but on
small n; pooled 58.3% is the defensible number, NOT 60%+.
Status: candidate. It does NOT touch live direction until forward
data (the Codespace 15m daily backtests) confirms it at n>=200.
