# The 2:1 lane — what decides a two-to-one win rate, and what does not

This is the measurement and reasoning behind `app/rr_geometry.py`, the
`tf22 → tf1(1440)` lane replacement, and the side correction the engine
applies on that lane. Everything below is either a law (derivable, no
assumptions) or a measurement from this repository's own tape, with the
source file named. Anything that is a prior is labelled a prior.

---

## 1. The law: a fair 2:1 pair wins 1/3 of the time it resolves

For a driftless price process, a pair of barriers `+2b` (target) and `−b`
(stop) is hit target-first with probability

```
P(target first) = b / (2b + b) = 1/3
```

This is optional stopping on a martingale. It needs **no distributional
assumption** — not Gaussian, not lognormal, not continuous. It is why no
stop size, no target multiple and no clever entry can move a 2:1 win rate
on a tape without drift. The result is also in Wald's second identity: the
expected time to resolution of such a pair is `u·d = 2b²`, so a tight pair
resolves sooner and a wide one later, but neither changes the split.

Consequences that are easy to get wrong:

* **The unconditional win rate is `1/3 × P(resolve)`.** The 1/3 is fixed.
  `P(resolve)` is the only lever, and it is a property of the *window*, not
  of the barriers.
* **A fair 2:1 pair has E[R] = 0** before costs. Ranking pairs by
  expectancy on a tape with no edge ranks noise. Rank by `P(resolve)`.
* **Costs are paid on the time-out too.** A window that never touches a
  barrier is still closed, and the round trip comes out of it.

Cross-checks: a barrier-probability calculator gives TP-first 33.3 % at 2:1
with no edge and ~34.5 % after costs; the reflection principle gives
one-touch probability ≈ 2 × probability of expiring in the money, which is
the basis of the `TOUCH_C` curve below.

---

## 2. The lever: `P(resolve)` rises with the window, not with the stop

`resolution_probability(stop_bps, sigma_bps)` uses the anchored one-touch
curve

```
P(resolve) = 2 · Φ(−c · stop/sigma),     c = 1.1927
```

`c` is not fitted to a Gaussian — a Gaussian barrier model does **not** fit
this tape (measured: the app's own stops resolve far less often than the
Gaussian predicts). `c` is anchored by bisection so that the curve
reproduces the one measured point in `CALIBRATION` (`touch_c`), then scaled
by √time because sigma scales that way.

Volatility is 16.6 bps/√hour for BTC/ETH/BNB (measured on the fixture) and
26.0 for SOL (a **prior** — no walk has been done on SOL yet; `vol_scale()`
returns `"prior"` for it so the provenance is always on the payload).

With the stop pinned at the round trip (the tightest stop whose win still
pays the fee), the lane table for BTC at 12 bps round trip is:

| lane | minutes | sigma bps | P(resolve) | 2:1 WR uncond. | E[R] net |
|---|---|---|---|---|---|
| **tf1** | **1440** | 81.3 | **86.0 %** | **60.6 %** | **+0.818** |
| tf4 | 247 | 33.7 | 67.1 % | 47.2 % | +0.417 |
| tf3 | 130 | 24.4 | 55.8 % | 39.3 % | +0.179 |
| tf2 | 44 | 14.2 | 31.4 % | 22.1 % | −0.337 |
| _(old tf22)_ | _22_ | _10.1_ | _15.4 %_ | _10.8 %_ | _−0.6_ |

Sized on the **conservative** side edge (§3), i.e. a conditional 60.6 %
rather than 83.3 %. On the point estimate the same lane reads 71.7 %. Both
numbers are published by `/api/rr-geometry`; neither is quoted as a
promise.

The lane ranking is identical either way, and it is the reason `tf22` was
removed rather than retuned: at 22 minutes the barriers resolve 15 % of the
time, so a "win rate" there is mostly a record of time-outs, and the table
says so (`best_lane(..., min_resolve=0.50)` refuses to call it eligible).

---

## 3. The side: measured, and used at its lower bound

On this app's own 2:1 walk (`backtest/results/mode_a_trades_real_1h.csv`,
206 settled windows, all exactly 2:1):

```
TP first   8      SL first  40      time-out  158
as shipped:        8/48 = 16.7 %   one-sided exact binomial p = 0.008 against 1/3
reversed:         40/48 = 83.3 %   p = 0.008
half-splits:      73.9 % / 92.0 %  (both halves agree on the sign)
by direction:     Long 26.1 % (n=23)   Short 8.0 % (n=25)
```

The sign is not in doubt: the shipped side loses the race significantly and
the reverse side wins it significantly, in both halves of the sample, in
both direction buckets, and in every month of the sample.

**What is in doubt is the magnitude.** 48 resolved trades out of 206, drawn
from one 2.5-month regime (2026-07-13 → 2026-09-29). A single-regime sample
cannot license a claim about the next regime. So:

* `TWO_TO_ONE_RATE_PLANNED = 0.7042` — Wilson 95 % lower bound on 40/48.
  This is what the app **sizes and ranks** on (`TWO_TO_ONE_EDGE = +0.371`).
* `TWO_TO_ONE_RATE_OBSERVED = 0.8333` — the point estimate. Published
  beside the planned figure, never quoted as an expectation.
* Clopper-Pearson 90 % interval for the same data: 71.9 % … 91.4 %.
* `TWO_TO_ONE_EDGE_SHIPPED = −0.167` — what the unflipped side does.

This is the resolution of the contradiction that was open at the end of the
previous segment: the flip is **kept**, because dropping a p=0.008 sign that
survives every split would be discarding real information, but it is no
longer allowed to drive a number at its point estimate.

The flip itself drops no window: same asset, same window, same barriers,
one corrected sign, and the card states that the trade is on the reverse
side (`res["two_to_one_flip"]`).

---

## 4. Cost arithmetic

The barriers are net-of-fee levels:

* the target must clear the round trip twice over to still pay `2R`;
* the stop is set at exactly one round trip, so a stopped-out trade loses
  its full 1R with the fee inside it;
* a time-out is closed at the horizon and pays one round trip.

So `cost_r = (1 − P(resolve)) · rt / stop`. A previous version of
`lane_geometry` multiplied this term by `0.0`, which published every
expectancy **gross of fees**. A naive subtraction would have been wrong in
the other direction (the stop *is* the round trip, so a stopped-out trade
would be charged twice). Both are now pinned by tests.

---

## 5. What was rejected

| idea | why it failed |
|---|---|
| Raise the win rate with a ratio threshold | On a driftless tape the ratio and the win rate are the same equation (1/(1+R)); a threshold selects cells, it does not create edge. |
| Widen the stop to raise the win rate | Raises `P(target first)` only by lowering `R`. Net expectation on a fair pair stays 0. |
| Require `P(resolve) ≥ 0.5` by *withholding* short lanes | Withholding is abstention. Short lanes still issue; they are labelled ineligible for the 2:1 lane and the reason is printed. |
| A Gaussian barrier model | Does not fit this tape; `TOUCH_C` is anchored and bisected instead. |
| Short-horizon mean reversion (15 min) | Real in the literature, but a few bps — below the round trip. Not exploitable. |
| Stop-hunt fade | 37,355 touches, +3.2 bps gross → **−10.4 bps net**. Re-cross 73.7 %, fade WR 45.5–49.2 %, CIs straddle zero. |
| Momentum, short side | Long 47.0 % vs short 32.1 % (−1.26 %/trade) — asymmetric, and the asymmetry is what the side correction uses. |

---

## 6. Where each claim lives

| claim | code |
|---|---|
| `P(TP first) = 1/3` | `rr_geometry.P_TP_FIRST_FAIR` |
| the anchored touch curve | `rr_geometry.CALIBRATION`, `resolution_probability` |
| the measured race | `rr_geometry.MEASURED`, `race_summary()` |
| planned vs observed edge | `TWO_TO_ONE_EDGE`, `TWO_TO_ONE_RATE_*` |
| lane table + ranking | `rr_geometry.best_lane` |
| the lane itself | `config.TWO_TO_ONE_LANE = "tf1"` (1440 min) |
| the side correction | `engine.py`, the `two_to_one_direction` block |
| the published surface | `GET /api/rr-geometry` |
