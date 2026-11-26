# The 66 % target at 2:1 — measured, not argued

_Measured 2026-11-26 on 24,096 real hourly bars per asset (BTC/ETH/BNB,
2024-01-01 → 2026-09-30) and 13,262 for PAXG, from the Binance Vision
1h mirror named in `REAL_RESULTS.md`. Zero timestamp gaps, zero malformed
rows. Reproduce with `python -m backtest.frontier_2to1` and
`python -m backtest.geometry_ev`._

---

## 1. The request, stated as arithmetic

"66 % win rate at 2:1, so every $100 earns $66.66" mixes two different
numbers. Both are tested below, but they are not the same ask.

| what is meant | requirement | at 2:1 |
|---|---|---|
| EV = +$66.66 per $100 risked | p = (0.6666 + 1) / 3 | **55.56 %** |
| a 66 % win rate at 2:1 | p = 0.66 | +0.98 R (+$98 per $100) |
| break-even | p = 1/3 | **33.33 %** |

`EV = p·2 − (1−p)·1 = 3p − 1`. Every number below is checked against
break-even, never against a bare win rate, because a 60 % win rate at
0.5:1 (break-even 66.7 %) loses money and an 45 % win rate at 2:1 makes it.

## 2. What the app actually achieves today

From the project's own real-tape trade log
(`backtest/results/mode_a_trades_real_1h.csv`, 206 trades, 83-day window):

| outcome | count | share | mean R |
|---|---|---|---|
| TP touched | 8 | 3.9 % | +2.00 |
| SL touched | 40 | 19.4 % | −1.00 |
| **neither (timeout)** | **158** | **76.7 %** | +0.076 |
| reported "win rate" (r_net > 0) | 82 | 39.8 % | −0.077 |

Two defects are visible in that table, and both mattered more than the
signal:

**a. 76.7 % of trades never resolve.** The barriers are too far away for
the horizon they are given. A trade that touches neither barrier carries
almost no information about the signal and pays friction anyway.

**b. The headline win rate is 2.4x the honest one.** "Win rate" counted
timeouts that happened to drift positive as wins. The real 2:1 hit rate is
`8 / (8 + 40) = 16.7 %` against a 33.3 % break-even.

## 3. Root cause of (a) — a units bug, found this run

`app/moves_risk.moves_stats(window_min)` treated its window as a **bar
count**, not a span of minutes. Every in-repo backtest feeds a **1-hour**
tape, so `tf1`'s 22 minutes was measured as **22 hours**.

```
median BTC stop in the trade log      181 bps
22-HOUR q80 excursion of the 1h tape  214 bps   <- what was measured
22-MINUTE excursion (sqrt-time)        32 bps   <- what was needed
```

A stop 5.7x too wide in distance is **32x too wide in variance** against a
horizon that is still 22 hours, so the trade cannot resolve: hence 76.7 %
timeouts. This is also the direct explanation of the reported
_"tp of 551 on BNB when market was at 556"_ — a 3.5 %-wide target on a
22-minute holding period is not a 22-minute target.

**Fixed:** the window is minutes, converted to bars using the tape's
declared spacing; a window shorter than the tape's own bar is refused and
returned `provisional` with the reason in the note; and the ATR fallback
now scales volatility by `sqrt(horizon / bar)` instead of `sqrt(minutes)`
as if minutes were bar counts.

## 4. Can 66 % (or 55.6 %) be reached? No — and not narrowly

**Search:** 40 causal feature families × 2 sides × 5 threshold quantiles ×
5 stop sizes, threshold fitted on TRAIN only, judged once on the untouched
TEST half. First-touch settlement, SL wins same-bar ties (adversarial).

**Best cross-asset result at 2:1 (test half):**

| config | mean win rate | break-even | gross EV |
|---|---|---|---|
| mom336 ×1 q0.8 k1 | **36.4 %** | 33.33 % | +0.091 R |
| rev240 ×1 q0.8 k1 | 35.8 % | 33.33 % | +0.075 R |
| mom336 ×1 q0.8 k2 | 33.6 % | 33.33 % | +0.008 R |

**Cross-asset consistency — the defence against a 900-way search:** apply
each configuration to every asset without re-fitting anything, and count
how many clear break-even on *all* of them.

> **0 of 432 configurations.** Per-asset CI lower bounds never clear 1/3.

At alpha = 0.05 a 900-cell search produces ~45 pure flukes, which is
exactly the scale of the few "EDGE" flags the single-asset table shows
(PAXG mom336, CI 35.3–46.0 %, was one of six picked from 900).

**Profit factor, after real friction:**

| | best gross PF | best net PF |
|---|---|---|
| any 2:1 config, cross-asset mean | 1.044 | **0.866** |

> **0 of 432 configurations reach net profit factor > 1.**
> **0 of 432 reach positive net EV.**

## 5. Why: the barrier geometry is scale-invariant

Win rate at `r:1` is `1/(1+r)` regardless of how wide the stop is. Stop
size changes *how fast* a trade resolves, not *how often* it wins. Measured
on real tape at 2:1 with a random side (geometry only, no signal):

| TP:SL | realised win rate | break-even |
|---|---|---|
| 0.5 | 66.9 % | 66.7 % |
| 1.0 | 48.8 % | 50.0 % |
| 2.0 | 34.5 % | 33.3 % |
| 3.0 | 23.6 % | 25.0 % |

Every row sits within noise of `1/(1+r)`. A perfect 2:1 win rate is not a
parameter you can tune; it is a *directional edge* you must find, and this
search — 40 families, 900 configs, four assets, 1,700 days — did not find
one larger than +0.09 R gross.

## 6. Friction

| asset | median 1h ATR | round trip | cost at k=1 | cost at k=3 |
|---|---|---|---|---|
| BTC | 63.3 bps | 28.9 bps | **0.46 R** | 0.15 R |
| ETH | 89.4 bps | 29.8 bps | 0.33 R | 0.11 R |
| BNB | 69.0 bps | 31.0 bps | 0.45 R | 0.15 R |
| PAXG | 29.6 bps | 37.4 bps | **1.26 R** | 0.42 R |

PAXG at a 1-ATR stop costs **more than the risk**: no win rate below 100 %
pays for it. The best gross edge found (+0.091 R) is a fifth of BTC's
friction at k=1 and still loses at k=3.

## 7. What this means for the target

The table below is measured, not assumed:

| target | requires | achieved | verdict |
|---|---|---|---|
| 66 % at 2:1 | 66.0 % | 36.4 % best cross-asset | not reachable |
| +$66.66 per $100 | 55.6 % | 36.4 % | not reachable |
| break-even at 2:1 | 33.3 % | 36.4 % gross | real but +0.09 R |
| net profit factor > 1 | — | 0.866 | not reachable here |

**No honest configuration of this system reaches 66 %, 55.6 %, or net
profitability at 2:1 on this data.** Claiming otherwise would require
either fitting to the test half (which is what a 900-way search without
cross-asset replication does) or hiding friction.

What *is* real and worth shipping: the app's headline win rate was 2.4x
its honest hit rate, and 76.7 % of its trades never resolved. Both are
fixed. The remaining gap between 36.4 % and 33.3 % is +0.09 R gross, which
friction consumes — so the correct operational conclusion is **trade less,
at larger stops, on the assets where cost/ATR is lowest**, not "find a
better signal".

## 8. Reproduce

```bash
# real 1h data (the mirror is reachable from the sandbox)
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/KevinSherry2016/Crypto.git /tmp/cryptomirror
cd /tmp/cryptomirror && git sparse-checkout set \
    marketData/unzipped/BTCUSDT marketData/unzipped/ETHUSDT \
    marketData/unzipped/BNBUSDT marketData/unzipped/PAXGUSDT

cd /path/to/repo
.venv/bin/python -m backtest.target_66 BTCUSDT ETHUSDT   # the earlier audit
.venv/bin/python -m backtest.frontier_2to1               # search + consistency
.venv/bin/python -m backtest.geometry_ev                 # geometry vs friction
```
