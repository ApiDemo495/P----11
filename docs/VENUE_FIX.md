# The cost model was pricing the wrong venue

**Segment 14 · 2026-11-27 · branch `arena/0e1fb733-p-11`**

## The verdict this answers

Segment 13 dropped 4 PAXG cells and raised mean net per trade from
+100.4 to +137.7 bps. Total net moved +89655 → +94185 bps — **+5.1 %** —
while trade count fell 893 → 684. The user's response was correct:

> "No increase more search internet Fix app and do more"

Raising the mean by removing the worst cells is a **selection** change.
It does not make the account more money. This document records what
actually did.

## What was actually broken

`app/trading.py` posts to `/fapi/v1/order`:

```python
entry_o = self._signed("POST", "/fapi/v1/order", {
    "symbol": sym, "side": side, "type": "MARKET",
    "quantity": qty})
```

`/fapi/v1/order` is Binance **USDⓈ-M futures**, and `type: MARKET` means
**taker on both legs**. The futures taker fee is 0.05 %/side = **10 bps
round trip**.

The cost model charged the *spot* stack: 10 bps/side commission (spot
taker VIP0) + slippage + half the Roll(1984) spread, i.e.
**28.872 bps** for BTC. That is 2.9× what the order path pays.

This was not one stale constant. **Six** places priced the trade, and
they disagreed with each other:

| where | round trip (BTC) | note |
|---|---|---|
| `app/horizon.ASSET_RT_BPS` | 28.872 | spot stack |
| `app/hold_policy.round_trip_bps` | 24.0 | 2 × 12/side |
| `app/cost_gate` sleeve defaults | 8.0 | a third price |
| `backtest/data.FRICTION_BPS` | 28.872 | the origin |
| `backtest/continuous.FRICTION_BPS` | 28.872 | a literal copy |
| `backtest/edge_scan.FRICTION_BPS` | 28.872 | a literal copy |
| `backtest/results/*.csv` `bps_net` | deducts 12 | graded |

The repo's own `backtest/results/winrate_venue.json` already recorded the
right answer and had all along: binance 10.0, hyperliquid4 8.0, aster 8.0,
extended 5.0, mexc_fut 4.0, and "project 28.9" marked **legacy**. The
venue table was never wired into the cost model.

## The fix

Cost is now **derived**, in one place — `backtest/data.py`:

```python
VENUE_RT_BPS = {..., "binance_fut": 10.0, "binance_spot": 20.0,
                "legacy_spot": 28.872, ...}
DEFAULT_VENUE = "binance_fut"          # what /fapi/v1/order actually pays
FRICTION_BPS  = {s: venue_rt(s) / 2.0 for s in SLIPPAGE_BPS}
ROUND_TRIP_BPS = {s: venue_rt(s)     for s in SLIPPAGE_BPS}
```

Every other module imports from it. `VENUE=<key>` or `PENNY_VENUE=1`
re-prices the whole app without touching code. `/api/cost-model` serves
the result so the UI can show the number every EV is divided by.

| asset | old RT | new RT | per side |
|---|---|---|---|
| BTC | 28.872 | **12.0** | 6.0 |
| ETH | 29.808 | **12.0** | 6.0 |
| BNB | 30.963 | **14.0** | 7.0 |
| PAXG | 37.400 | **20.0** | 10.0 |

Unknown symbols now price at the most expensive measured slippage, not
the cheapest — a cost model may fail safe, never flattering.

## What it is worth

Same trades, same cells, cheaper round trip:

| policy | old total net | new total net | delta |
|---|---|---|---|
| all 16 cells (n=893) | +73339 bps | **+88711** | **+21.0 %** |
| drop 4 PAXG (n=684, shipped) | +81924 bps | **+93659** | **+14.3 %** |

Mean net per trade +82.1 → +99.3 bps (+17.2) on all 16, with **no trade
removed**. This is the shape segment 13 failed to produce: more money at
identical trade count.

PAXG stays excluded, and now for a stated reason rather than a
coincidence: its mean gross edge is **−3.7 bps**, i.e. negative before
any fee at all. A zero-fee venue does not rescue it.

## The gate consequences

`horizon.classify` requires σ ≥ 3× the per-side friction. Halving the
cost changes what counts as tradeable:

| lane | σ (bps) | old | new |
|---|---|---|---|
| TF1 | 22.5 | **untradable** (1.51×) | **tradable** (3.75×) |
| TF2 | 33.8 | marginal (2.27×) | tradable (5.63×) |
| TF3 | 59.8 | tradable | tradable |
| TF4 | 81.1 | tradable | tradable |

And the target floor, `MIN_TP_OVER_RT × round trip`, halves. Graded
targets had been sitting *inside* the old round trip — a "win" that could
not pay:

| cell | graded tp | old floor | new floor | |
|---|---|---|---|---|
| BTC tf1 (22) | 22.9 | 43.3 | 18.0 | **newly pays** |
| BTC tf2 (44) | 36.9 | 43.3 | 18.0 | **newly pays** |
| BTC tf3 (130) | 76.1 | 43.3 | 18.0 | pays |
| BTC tf4 (247) | 72.3 | 43.3 | 18.0 | pays |
| ETH tf1 (22) | 34.7 | 44.7 | 18.0 | **newly pays** |
| ETH tf2 (44) | 55.2 | 44.7 | 18.0 | pays |
| ETH tf3 (130) | 118.0 | 44.7 | 18.0 | pays |
| ETH tf4 (247) | 147.5 | 44.7 | 18.0 | pays |
| BNB tf4 (247) | 46.2 | 46.4 | 21.0 | **newly pays** |
| PAXG tf4 (247) | 31.3 | 56.1 | 30.0 | **newly pays** |

**5 of 10 measured cells flip from "target inside the round trip" to
genuinely paying. Zero cells stop paying.** `geometry_cost_check` had
returned `pays_round_trip=None` for all 16 cells because it read
`row_sl_bps_median` out of `hold_wr.json`, a key that artifact has never
had; the graded medians are in `wr_rereg_5m.json`.

Note that this does **not** overturn the segment-12 measurement that the
k=0.15 geometry still needs an 87 % first-touch rate to break even. The
target is 0.15× the stop; a cheap round trip does not repair a payoff
that thin. What it repairs is the claim that the *cost* was the reason.

## Why no new edge was manufactured

Researched before changing anything, per the request:

- **Han, Kang & Ryu (SSRN 4675565)** — most crypto momentum portfolios are
  liquidated under realistic assumptions; significant mean returns
  routinely mean negative profits. Judge by total net, not mean.
- **arXiv 2606.00071** — *"Daily predictability is real but does not
  extend to hourly or monthly horizons, and may not survive transaction
  costs."*
- **Liu, Tsyvinski & Wu (NBER w24877)** — TS momentum is a daily/weekly
  effect (+0.33 % at 1 day, +0.50 % at 6 days per SD).
- **Gbadebo** — the documented edge shape is multi-horizon
  volatility-normalised EMA trend on **daily** bars.
- The app's own `wr_horizon.json` shows in-sample EV rising to +736 bps at
  a 90-day hold while holdout EV falls to **−1345 bps**. That is an
  overfit exhibit, not an edge.

The pooled holdout side accuracy is **0.4907 (n=1608)**. There is no
hourly alpha to find, and long holds are worse, not better. The honest
work was to stop over-charging for trades the app actually takes.

## Files

- `backtest/data.py` — `VENUE_RT_BPS`, `DEFAULT_VENUE`, `venue_rt()`,
  derived `FRICTION_BPS` / `ROUND_TRIP_BPS`, safe fallback.
- `app/horizon.py` — `asset_rt()` delegates to `venue_rt()`, `venue()`.
- `app/hold_policy.py` — `round_trip_bps` / `friction_bps` derive from the
  venue; `summary()` reports the live per-side figure and keeps the stale
  artifact cost as `round_trip_stale_artifact_bps`; `geometry_cost_check`
  reads `wr_rereg_5m.json` and reports `graded_rt_bps` beside the live one.
- `app/cost_gate.py` — one `DEFAULT_FEE_BP`; every sleeve prices from it.
- `backtest/continuous.py`, `backtest/edge_scan.py` — import the table
  instead of copying it.
- `app/main.py` — `GET /api/cost-model`.
- `static/app.js` — the cells header shows venue, fee and round trip.
- `tests/test_cost_model_venue.py` — 15 tests pinning the venue, the
  single source of truth, the safe fallback, the barrier lookup and the
  "no cell may stop paying" invariant.

## Verification

```
pytest tests -q -p no:randomly     → 1249 passed, 3 skipped
node --check static/app.js         → OK
curl /api/cost-model               → venue binance_fut, fee 10.0, BTC RT 12.0
curl /api/cells                    → 16 cells; TF1 lane now tradable
```

## Ledger honesty

The shipped `per_tf_full_engine_real.csv` `bps_net` column still deducts
12 bps (the old grading cost). It was **not** rewritten: it is the record
of what was measured under the old model. The live cost is read from
`ROUND_TRIP_BPS` at display time. Re-grading the CSV would erase the
evidence that the defect existed.
