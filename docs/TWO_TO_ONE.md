# The 2:1 lane — what a two-to-one pair can and cannot do

**Revised 2026-11-28.** The previous version of this document claimed a
60.6 % 2:1 win rate at +0.818R. That number cannot exist, and the code that
produced it had an arithmetic error. This version records the error, the
law that rules it out, and what the honest numbers are.

---

## 1. The law: a fair 2:1 pair wins 1/3 of the windows it resolves

For a driftless price process, a pair of barriers `+Rb` (target) and `−b`
(stop) is hit target-first with probability

```
P(target first) = 1 / (1 + R)        ->  1/3 at R = 2
```

Optional stopping on a mean-zero martingale. **No distributional assumption**
— not Gaussian, not lognormal, not continuous. This is why no stop size, no
target multiple and no lane length can move it.

Two consequences that are easy to get wrong, and were both got wrong here:

* **The unconditional rate is `P(resolve)/(1+R)`, so 1/3 is a CEILING.** A
  window that closes before either barrier is touched only drags the
  conditional rate *below* 1/3. Nothing about shortening or lengthening the
  lane can push it above.
* **A fair pair has E[R] = 0 and its costs are not zero**, so a zero-edge
  barrier pair is a guaranteed loss of the round trip. See §4.

Simulated to confirm it (60k random walks, `rr_geometry.SIMULATION`):

| stop/sigma | resolved | P(TP \| resolved) |
|---|---|---|
| 0.25 | 100.0 % | **0.339** |
| 0.50 | 86.8 % | **0.313** |
| 1.00 | 35.7 % | **0.123** |
| 2.00 | 4.4 % | **0.002** |

Never above 1/3, and truncation pulls it down.

---

## 2. The error that produced "60.6 %"

`app/rr_geometry.py` calibrated a one-touch curve

```
P(resolve) = 2 * Phi(-c * stop/sigma)
```

by bisection. The first version solved `2*Phi(-c) = 0.233` — i.e. it assumed
the measured anchor sat at `stop/sigma = 1` — but the anchor actually sits at
`stop/sigma = 185.5/83.5 = 2.22`. Using `c` at the wrong `z` made the model
report **8.1 %** resolved where the tape says **23.3 %**.

It then applied that mis-scaled curve to a stop pinned at the round trip
(12 bps), which is tiny relative to a 24-hour sigma, so `P(resolve)` came out
at 86 %, and — the fatal part — `win_rate_conditional` was computed as
`1/3 + edge` with a **+0.500 edge** taken from a single-regime sample. That
published **P(TP first | resolved) = 0.704**, i.e. more than twice the
theoretical maximum, and every expectancy downstream was computed from it.

Three fixes, all asserted in code:

1. `_calibrate()` now divides by the measured `z` (residual reported as
   `fit_residual`, currently 0.0).
2. `lane_geometry` **asserts** that a zero-edge pair never exceeds 1/3.
3. The edge constant is **0.0**, not +0.500 (see §3).

---

## 3. The side flip is real and still not usable

On the app's own 2:1 walk (206 settled windows, `mode_a_trades_real_1h.csv`):

```
TP first   8      SL first  40      time-out  158
shipped:   8/48 = 16.7 %   one-sided exact binomial p = 0.0081
flipped:  40/48 = 83.3 %   p = 0.0081, half-splits 73.9 % / 92.0 %
```

The signal is not noise. **It is also not profit.** A fixed 2:1 pair maps the
barrier outcome one-to-one onto P&L: target-first books +2R, stop-first books
−1R. Flipping the side therefore moves P&L between one book and the other:

| | resolved book | time-out book | **total** |
|---|---|---|---|
| as shipped | −24.00R | +24.82R | **+0.82R** |
| if flipped | +24.00R | −24.82R | **−0.82R** |

83.3 % of *resolved* trades "win" while the account does not move — and the
sign of the total actually gets worse. The flip is published as a **label**
with `size_authority = 0.0`; the engine does **not** reverse its side. It is
re-enabled only by a locked out-of-sample walk on a new regime that shows
*profit*, not just a race label.

*(This supersedes the previous version of this document, which kept the flip
and sized on a Wilson lower bound. Sizing on a lower bound was still sizing
on a relabelled book.)*

---

## 4. The finding: the arithmetic does not close

With no edge, at ratio `R`, stop `b` and round trip `c`:

```
achieved = P(resolve) / (1 + R)
needed   = (1 + c/b) / (1 + R)
```

The `(1 + R)` **cancels**. The edge is therefore the sign of
`P(resolve) − 1 − c/b`, and since `P(resolve) ≤ 1` and `c/b > 0`, that is
**always negative**. Checked over lane × ratio × stop = 5 × 6 × 198 points:
maximum edge found **+0.000000**.

**A zero-edge barrier pair can never clear its own round trip.** No lane
length, no target multiple and no stop size rescues it. The target multiple
is worse than useless here — it cancels out entirely.

### What the 6-hour lane actually delivers (BTC, measured σ = 49.9 bps/√h)

| stop | P(resolve) | 2:1 WR achieved | 2:1 WR needed | edge |
|---|---|---|---|---|
| 60 bps (best) | 79.2 % | 26.4 % | 40.0 % | −13.6pp |
| 140 bps (1.4 %) | 53.9 % | 18.0 % | 36.2 % | −18.2pp |
| 300 bps | 18.8 % | 6.3 % | 34.7 % | −28.4pp |

The best pair the lane can host, searching ratio 2–5 and every stop: **5:1 at
a 60 bps stop, needs 20.0 %, achieves 13.2 %, −6.8pp.** Still negative.

`/api/rr-geometry` publishes exactly this. It is a measurement of a shortfall,
not a promise.

---

## 5. Volatility is measured, and thin

| asset | σ bps/√h | n obs | provenance |
|---|---|---|---|
| BTC | 49.9 | 51 | `measured_vol` |
| ETH | 57.4 | 28 | `measured_vol` |
| BNB | 39.8 | 73 | `measured_vol` |
| SOL | — | 0 | `unmeasured` |

Estimated from real entry quotes in `mode_a_trades_real_1h.csv`
(2026-07-13 … 2026-09-29): `σ²/h = mean(r²/Δt)` over consecutive entries.
The previous figures (16.6 for every major) were **back-solved from the
resolved-share anchor** — 3× too low for BTC — and are exactly the
circularity described in §2.

**51 observations is not a multi-year history.** There is no price cache in
this checkout and no outbound network, so this is the best available and it
is labelled as thin everywhere it is used. **SOL has no history at all**, so
`vol_scale("SOL")` returns `None` and the endpoint returns
`measured: false` with a reason instead of inventing a rate. The SOL figures
published in the previous release (29.9 % WR, R = 1.3) were priors presented
as measurements; they are withdrawn.

---

## 6. What was rejected

| idea | why it failed |
|---|---|
| Raise the 2:1 win rate with a ratio threshold | The ratio cancels out of achieved-vs-needed (§4). It cannot move a zero-edge book. |
| Widen the stop to raise the win rate | Raises `P(target first)` only by lowering `R`; net expectation on a fair pair stays ≤ 0. |
| Size on the Wilson lower bound of the flip | Still sizing on a relabelled book — §3. |
| A Gaussian barrier model | 8 % where the tape says 23 %; anchored curve instead. |
| Back-solve σ so the model matches the anchor | Circular — feeding it back returns the anchor by construction. |
| Short-horizon mean reversion (15 min) | Real in the literature, a few bps, below the round trip. |
| Stop-hunt fade | 37,355 touches, +3.2 bps gross → −10.4 bps net. |

---

## 7. Where each claim lives

| claim | code |
|---|---|
| `P(TP first) = 1/(1+R)` | `rr_geometry.P_TP_FIRST_FAIR` |
| the ceiling is asserted | `lane_geometry` assertion |
| the zero-edge theorem | `rr_geometry.ZERO_EDGE_THEOREM` |
| the random-walk proof | `rr_geometry.SIMULATION` |
| the measured race + the flip's P&L | `rr_geometry.MEASURED`, `side_policy()` |
| measured volatility | `rr_geometry.VOL_BPS_PER_SQRT_HOUR`, `VOL_OBSERVATIONS` |
| the 6 h lane | `config.TIMEFRAMES["tf1"]` = 360 min |
| the engine keeps its own side | `engine.py`, the `two_to_one_side_note` block |
| the published surface | `GET /api/rr-geometry` |
