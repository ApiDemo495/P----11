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

TESTED AND NOT ADOPTED (nothing was removed from the app —
every formula/component still computes and votes; these were
backtest-only hypotheses that failed out-of-sample):
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

## TP4 TIMEFRAME — last 1 year (2025-10-01 -> 2026-09-30), real 1h data
The goal restated by the user: win rate on the TP4 timeframe
(tf4, 247-bar macro window), not sub-timeframes. 140 non-overlapping
TP4 decisions across BTC/ETH/BNB/PAXG, engine pack at each, levels
from moves_risk(minutes=247), one trade at a time. Walk-forward:
select on first 55% of the year, judge on last 45% only.

BASELINE (app's raw TF4 direction, k=1):
- train 54.9% win EV +0.245R (n=51) -> TEST 48.6% EV +0.004R PF 1.02 (n=35)
At its native horizon TF4 is NOT anti-predictive (the sub-TF
inversion result was a horizon-mismatch artifact) — but alone it
decays to breakeven out-of-sample.

TP4 POLICY (backtest/policy.py TP4PolicyPredictor): at RSI(14)
extremes (<=30 long / >=70 short) a mean-reversion direction
OVERRIDES; otherwise the engine TF4 direction stands; barriers
k=0.75 of the 247-bar q80-MAE stop, TP = 2x SL.
- train: 65.5% win, EV +0.393R, PF 2.78 (n=55)
- TEST:  61.0% win (95% CI 45.7-74.3), EV +0.238R, PF 1.93 (n=41)
- full year: 63.5% win (CI 53.6-72.5), EV +0.327R, PF 2.39 (n=96)
Per-asset TEST: BNB 69.2% (n=13), ETH 80.0% (n=5), BTC 54.5%
(n=11), PAXG 50.0% (n=12). RSI-revert alone was robust at every
barrier scale (test 73-80%, n=15) but fires rarely (34/140).
Honesty flags: the combined rule was assembled AFTER its two
components validated, so its test half is supportive, not
pristine; one split, one year, CIs wide. Status: candidate for
forward confirmation (Codespace runs), NOT yet live direction.
Trade log: backtest/results/tp4_policy_trades.csv (with split tags).

## ABLATION — every formula/component/brain at TP4 + ratio grid
140 TP4 decisions re-collected with the fusion's exact inputs
captured per decision (comp dict: F1-F27, C31-35, BRAIN, EMOTION,
PHYSICS, NEWS, HEDGE, TRACER + FlyWire/consensus sub-signals).
Each component standalone as direction (sign), k=0.75, walk-forward.

- No single component is strong: best train-validated F25 -> test
  45.3% (EV +0.072R); F2/F5/F17/F23 test EV +0.10..0.14R at
  47-56% win. BRAIN consensus standalone: test 50.0%, EV -0.033R;
  FlyWire vote: 48.4%, EV +0.049R — the brain alone is a coin flip
  at TP4; its value (if any) is only inside the fusion.
- Always ZERO at tf4 (measured fact only — they are KEPT and
  still compute; values may be changed in future, never removed):
  F1, F6, F7, F9, F18, F27, C31, C32, C33, C34, C35.
- Ensembles of train-validated components: equal-vote test 36.5%
  (worse — weak voters dilute), top-5 42.2% EV +0.121R, EV-weighted
  43.8%, top-5 + RSI override 54.7% EV +0.281R PF 2.26 (n=64) —
  best EV, still below the TP4 policy's win rate.
- Volatility gate at TP4: REJECTED (test 33%/25%, tiny n) — the
  tf1-era gate does not transfer horizons.
- RATIO GRID (user bounds 1:1..4:1, pref 2:1), TP4 policy, test:
  win% is 56.8% at EVERY ratio (TP rarely binds; timeouts decide),
  EV: 1:1 +0.073R | 1.5:1 +0.172R | 2:1 +0.153R | 3:1 +0.107R |
  4:1 +0.107R. 1.5:1 and 2:1 are within noise of each other; the
  policy keeps the preferred 2:1 (R4 in policy.py, bounds enforced
  by test).
- RERUN VARIANCE DISCLOSURE: this identical re-collection moved
  the TP4 policy's test reading from 61.0% (n=41, first run) to
  56.8% (n=44). The engine contains nondeterministic elements
  (RL mix selection), so a single run's win% carries run variance
  on top of sampling variance. Defensible pooled TP4 estimate
  today: ~55-61% win, EV positive — NOT a proven 60-80%.
Trade log (run 2): backtest/results/tp4_policy_trades_run2.csv

## VOTE v2 — change every weight, remove NOTHING (user's corrected rule)
All 37 fusion components vote. Changed: each weight = its own
clipped train EV; the 11 anti-predictive voters (BRAIN, EMOTION,
F8, F10, F13, F15, F16, F24, HEDGE, NEWS, TRACER) got NEGATIVE
weights (their votes are flipped — changed, not dropped);
near-zero voters keep a 0.005 floor (nobody zeroed). Trade needs
>=50% of total |weight| agreeing; RSI override kept; k=0.75, 2:1.
Walk-forward both directions (fit logic: backtest/vote_lab.py):
- forward test: 72.2% win, EV +0.606R, PF 7.76 (n=18)
- reverse test: 58.1% win, EV +0.123R, PF 1.37 (n=31)
- POOLED OOS: 63.3% win (95% CI 49.3-75.4), EV +0.300R, PF 2.23 (n=49)
Shipped: ChangedVotePredictor (weights = train55 fit, in code),
CLI --policy changed-vote. tau=0.4 had a prettier forward number
(78.3%) but a weaker reverse (54.3%); tau=0.5 was selected by
worst-direction performance, not by the best-looking number.
Status: cross-validated candidate, NOT live. The next honest
judge is untouched forward data.

## WIN-RATE CHECK (fresh end-to-end run, shipped predictor classes)
Last 1 year, TP4, trade-by-trade through the actual predictor
classes (not offline settles):
- CHANGED-VOTE: ALL 62.63% win, EV +0.311R, PF 2.26 (n=99)
  BNB 62.1% (29) | BTC 70.8% (24) | ETH 47.1% (17) | PAXG 65.5% (29)
- TP4 POLICY (reference): ALL 63.16%, EV +0.288R, PF 2.13 (n=95)
  BNB 71.4% (28) | BTC 70.8% (24) | ETH 50.0% (16) | PAXG 55.6% (27)
Caveat: this full-year pass includes the window the vote weights
were fitted on (first 55%), so it is partly in-sample; the clean
walk-forward estimate for changed-vote remains 63.3% pooled OOS
(n=49, CI 49.3-75.4). The fresh run landing at 62.6% is consistent
with it, not a replacement for it. ETH is the weak asset in both.

## Third run — FULL engine, per asset x per TF (2026-11-09)

Run: `backtest/per_tf_real.py` (this repo), same validated Binance
Vision 1h mirror data (zero gaps; series 2026-05-08 -> 2026-09-30,
decisions in the last 2,000 bars, every 22 bars, 80 packs/asset).
Predictor = `PredictionEngine.compute_asset` with ALL tape-runnable
features voting (37 formulas, CCI fusion + ladder, four real FlyWire
brains, physics/emotion stacks; learning frozen between decisions).
Cross-asset states are set to the same decision bar for every pack
(an earlier draft of this run let relative-strength features read a
stale future state — caught and fixed before quoting any number).
Settlement per TF over its own window: strict sign, tie = loss;
TP/SL first-touch from the app risk block; friction charged in bps.
Trade log: `backtest/results/per_tf_full_engine_real.csv` (893 rows).

| Asset | TF | Win % | 95% CI | n | TP/SL/neither | net bps/window |
|---|---|---|---|---|---|---|
| BTC | tf1 (22b) | 54.24 | 41.7–66.3 | 59 | 4/5/50 | +16.7 |
| BTC | tf2 (44b) | 47.46 | 35.3–60.0 | 59 | 6/4/49 | +33.6 |
| BTC | tf3 (130b) | 50.00 | 37.9–62.1 | 62 | 8/3/51 | +139.3 |
| BTC | tf4 (247b) | 51.72 | 39.2–64.1 | 58 | 9/3/46 | +223.6 |
| ETH | tf1 | 46.67 | 32.9–60.9 | 45 | 4/4/37 | +28.6 |
| ETH | tf2 | 48.89 | 35.0–63.0 | 45 | 4/1/40 | +75.6 |
| ETH | tf3 | 61.70 | 47.4–74.2 | 47 | 6/2/39 | +278.5 |
| ETH | tf4 | 69.57 | 55.2–80.9 | 46 | 8/1/37 | +582.4 |
| BNB | tf1 | 48.39 | 36.4–60.6 | 62 | 5/9/48 | -1.4 |
| BNB | tf2 | 60.00 | 47.9–71.0 | 65 | 7/9/49 | +46.8 |
| BNB | tf3 | 62.32 | 50.5–72.8 | 69 | 12/10/47 | +132.5 |
| BNB | tf4 | 64.18 | 52.2–74.6 | 67 | 15/7/45 | +193.1 |
| PAXG | tf1 | 46.67 | 32.9–60.9 | 45 | 2/11/32 | -34.2 |
| PAXG | tf2 | 42.00 | 29.4–55.8 | 50 | 2/8/40 | -18.7 |
| PAXG | tf3 | 50.00 | 37.3–62.7 | 56 | 4/6/46 | -18.3 |
| PAXG | tf4 | 37.93 | 26.6–50.8 | 58 | 5/13/40 | -17.9 |
| ALL | tf1 | 49.29 | 42.6–56.0 | 211 | | |
| ALL | tf2 | 50.23 | 43.7–56.8 | 219 | | |
| ALL | tf3 | 55.98 | 49.6–62.2 | 234 | | |
| ALL | tf4 | 55.46 | 49.0–61.8 | 229 | | |

### Reading it honestly
- ONE ~83-day regime on 1h bars (TF = 22/44/130/247 *hours* here),
  n=45–69 per cell: almost every cell's CI includes 50%. Only
  ETH tf4 and BNB tf3/tf4 sit clearly above coin-flip, and TF3/TF4
  windows overlap (decisions every 22 bars), so their effective n
  is smaller than printed.
- PAXG is below 50% on all four TFs with negative net bps — on this
  window the full engine LOSING on PAXG is the measured fact.
- Long-TF net bps for BTC/ETH/BNB are flattered by regime drift;
  win% is the headline, bps the context. Neither is a promise.
- Order-book/news features cannot be replayed from klines and are
  absent — "all features" means all features a tape can feed.
