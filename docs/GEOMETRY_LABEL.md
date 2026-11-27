# The board said "NOT TRADEABLE" while the app traded it

**Segment 14 · 2026-11-27**

## The contradiction

11 of 16 cells wore:

```
GEOMETRY BELOW 1:1 · NOT TRADEABLE
```

and the app placed those trades anyway. `_cost_aware` widens every target
to `RR_LIVE_MIN = 1.0` and raises `AssertionError` if it cannot, so a
lock can never ship below the floor. Measured live:

| | certified rr | **shipped rr** |
|---|---|---|
| BTC tf1–tf4 | 0.27 | **1.16** |
| ETH tf1–tf4 | 0.27 | **1.16** |
| BNB tf2, tf3 | 0.27 | **1.16** |
| BNB tf4 | 0.38 | **1.08** |
| PAXG tf4 | 0.27 | **1.16** |

Every cell ships at 1.08:1 or better. The badge described the
*certificate*, not the trade.

## Why that was a defect, not pedantry

The label did three wrong things at once:

1. **It was false about the app.** "NOT TRADEABLE" asserts no trade
   happens. A MARKet lock is issued on that cell.
2. **It was an abstention in disguise.** The user's standing rule is that
   no policy may withhold, silence or skip a window. Eleven cells
   announced as permanently untradeable is exactly the screen-of-refusal
   the rule forbids — and it was contradicted by the terminal next to it.
3. **It hid the real uncertainty.** The certified win rate (72.9 %) was
   measured on the 0.27:1 pair, which is *not* the pair that ships. The
   honest statement is that the *shipped* 1.16:1 ratio's hit rate is
   unmeasured — an evidence gap, not a prohibition.

## The fix

The label now says what is true:

```
1:1-REPAIRED · LOCK ORIGIN UNMEASURED
```

and renders in amber (`tier-warn`), not error red — a caveat about where
the number came from, not a broken invariant. `tier_reason()` states both
ratios and both win rates, plus the fact that the artifact's break-even
rate belongs to the old pair.

New `hold_policy.live_rr(symbol, tf)` probes the repair across barrier
scales 0.25×–6× and returns the **worst** ratio it produces, so the figure
is a floor rather than a best case. It is published on `summary()["live_rr"]`
and rendered on the card beside the certified ratio.

## Verification

```
GET /api/cells   → 1:1-REPAIRED · LOCK ORIGIN UNMEASURED × 11
                   NO MEASURED EDGE · PATTERN NOT TRADEABLE × 4
                   UNGRADEABLE / WARMUP × 1
BTC|tf1          → cert rr 0.2727, live rr 1.164
BNB|tf4          → cert rr 0.3750, live rr 1.083
```

Live TF ratios sampled from `/api/snapshot`: min **1.00**, max **2.11**.
Zero below the floor.

`pytest tests -q -p no:randomly` → **1255 passed, 3 skipped**.
`node --check static/app.js` → OK.

## What was NOT changed

The rule itself is untouched. `RR_LIVE_MIN = 1.0`, the `AssertionError` on
a sub-floor pair, and `_cost_aware`'s widen-target-never-narrow-stop
direction are all as shipped in segment 12. This change only stops the
label from claiming the rule was violated when it was not.
