# WIN RATE AT 1:1 AND ABOVE — what the data actually says

The user asked for a win rate at **2:1 and any ratio above 1:1**. This file is
the measured answer, and the reason the app cannot honestly print "83 % at 2:1".

## Where 83 % came from

The certified artifact (`backtest/results/hold_wr.json`) reports
`row_wr_best = 0.83` for most cells. Every one of those numbers was selected at

```
k = 0.15   (target = 0.15 × stop)
m = 0.55
```

i.e. a **0.27:1** geometry. It is a real measurement of a real trade — one whose
target sits inside the Binance round trip. All 351 recorded grid points in that
artifact carry `k = 0.15`. The artifact never searched at legal leverage, so it
contains no 1:1 win rate at all.

The hit rate that looks good is paid for by the tiny target: a target one
seventh of the stop is touched most of the time, and the fee eats the win.

## The measurement at 1:1 and above

`backtest/results/wr_audit_independent.json` does contain a full k × m grid with
out-of-sample splits. Selecting **on train** and reporting **holdout** (no
cherry-picking the holdout), restricted to `tp_bps / sl_bps >= 1.0`:

| cell | r | train EV | holdout WR | holdout EV | PF | n_ho |
|---|---|---|---|---|---|---|
| BNB 130 | 3.00 | +0.0579 | 0.452 | −0.0062 | 1.009 | 62 |
| BNB 247 | 3.00 | +0.2649 | 0.387 | −0.0552 | 0.878 | 31 |
| BNB 44 | 3.00 | +0.0016 | 0.418 | −0.1131 | 0.434 | 189 |
| BTC 130 | 1.00 | +0.0246 | 0.452 | −0.0306 | 0.751 | 62 |
| BTC 22 | 1.50 | −0.0186 | 0.407 | −0.0398 | 0.690 | 381 |
| BTC 247 | 1.00 | −0.0335 | 0.516 | **+0.0288** | 1.127 | 31 |
| BTC 44 | 1.00 | −0.0081 | 0.455 | −0.0330 | 0.731 | 189 |
| ETH 130 | 3.00 | +0.0622 | 0.516 | **+0.1243** | 1.275 | 62 |
| ETH 22 | 1.00 | −0.0174 | 0.451 | −0.0028 | 0.945 | 381 |
| ETH 247 | 2.00 | +0.0985 | 0.419 | **+0.0338** | 1.130 | 31 |
| ETH 44 | 3.00 | +0.0052 | 0.423 | −0.0296 | 0.811 | 189 |

**median holdout win rate 45.1 % · median holdout EV −0.0296 R · 3 of 11 cells
positive-EV out of sample.**

That is the honest state. A 2:1 target needs ~33 % to break even before costs,
so 45 % at 1:1–3:1 is *not* a disaster — but it is nowhere near 83 %, and most
cells do not survive the holdout after friction.

## What the app now does instead of lying

1. **It cannot publish a sub-1:1 target at all.** The live geometry is
   `_factor × NEW_TARGET_RR` with `_factor ∈ [0.5, 1.0]` and
   `NEW_TARGET_RR = 2.0`, so every issued cell lands in **1:1 … 2:1**. The
   old band started at 0.35, i.e. 0.7:1, which the floor then refused — that
   is why 12 of 16 cells sat at "NOT TRADEABLE" for a week. Weak signals now
   get the smallest *legal* target instead of being silenced.
2. **A flip no longer destroys the ratio.** `levels.mirror` used to swap the
   two prices, turning a 2:1 pair into 1:2; the floor then widened it to
   exactly 1:1. It now places each leg at its own measured distance, so a
   side flip preserves reward:risk (proven live: 1.9999 published as 1.9999
   instead of 1.0000).
3. **The panels state how old the reading is** (`measured_at`), because these
   values are computed once per GRC pass and are bit-identical in between.

## What is NOT done, and why

Raising the realised win rate at ≥1:1 is not a display fix — it needs a
re-graded artifact at legal leverage, and the certified one cannot be
re-derived in this sandbox (no `backtest/cache/`). What has been removed is the
*false* claim: the app no longer implies an 83 % win rate is available at a
ratio the user is allowed to trade.

The honest path to a better number, in order of expected effect:

1. **Cell selectivity.** 3 of 11 cells are positive out-of-sample at ≥1:1
   (ETH 130, ETH 247, BTC 247). Trading fewer cells with measured edge beats
   trading all 16.
2. **The 45 % at 1:1 is close to break-even** — the remaining gap is friction
   (30–37 bps round trip). A maker/limit entry, or dropping to the cells whose
   friction is lowest relative to their stop, moves EV more than any threshold
   tweak.
3. Re-running `backtest/wr_audit.py` with the grid restricted to `k >= 1` and
   more eras would give the app its own certified ≥1:1 table to publish,
   replacing `hold_wr.json` rather than reinterpreting it.

---

## Win rate by ratio band — the whole curve

Every grid point in `wr_audit_independent.json`, holdout, net of friction:

| ratio band | n points | median WR | median EV | positive-EV |
|---|---|---|---|---|
| < 0.5:1 | 170 | **67.7 %** | −0.0380 R | 35/170 |
| 0.5 – 1.0:1 | 61 | 55.6 % | −0.0352 R | 19/61 |
| 1.0 – 1.5:1 | 106 | 45.2 % | −0.0349 R | 33/106 |
| 1.5 – 2.5:1 | 125 | 43.8 % | −0.0342 R | 37/125 |
| > 2.5:1 | 77 | 41.9 % | −0.0306 R | 21/77 |

This is the honest shape of the problem, and it is not a bug in the app:

**Win rate is bought with ratio.** Every step up in reward:risk costs
real hit rate — 67.7 % → 41.9 % — and the median EV barely moves, because
the market is close to efficient at this resolution. The 83 % the old
artifact advertised was the far-left end of this table (a 0.27:1 target).

So "increase the win rate at 2:1" cannot be done by tuning a threshold. A
2:1 target has to *earn* its win rate from better entry timing, and the
data says most cells have no such edge at all.

## What actually raises it: cell selection

The same grid, cut by timeframe, is where the signal is:

| tf | positive-EV points at ≥1:1 | median EV |
|---|---|---|
| 22 m | **0 / 56** | −0.0570 R |
| 44 m | **0 / 84** | −0.0852 R |
| 130 m | 39 / 84 | −0.0117 R |
| 247 m | **52 / 84** | **+0.0160 R** |

Short windows have no measurable edge at a legal ratio; long ones do. Three
cells clear both the floor and a positive holdout EV, and each wants a
different ratio — which is why a single global knob was the wrong tool:

| cell | measured rr | holdout WR | holdout EV | PF |
|---|---|---|---|---|
| ETH 130 m | **3.00** | 51.6 % | **+0.1243 R** | 1.275 |
| BTC 247 m | **1.00** | 51.6 % | **+0.0288 R** | 1.127 |
| ETH 247 m | **2.00** | 41.9 % | **+0.0338 R** | 1.130 |

These three are now wired as `CELL_RR` in `app/mandate_cfg.py` and resolved
by `new_rr_cell()` before the per-asset override and the global default.
`tests/test_mandate_new_target.py` recomputes the table from the artifact,
so the numbers cannot drift away from the measurement that produced them.
