# Raising the win rate without abstention

**The request:** raise the win rate while still taking a position on every
signal. Skipping weak signals, "no-trade" gates and confidence filters are all
out of bounds — the win rate has to come from the *book*, not from trading less.

**The answer, in one line:** the win rate is a free parameter of the trade
geometry (target ÷ stop), so it can be raised on the same windows at will —
measured **+16 to +23 points on the live 4 h cells** and up to **84 %** at
day scale — but the break-even win rate rises with it at exactly the same rate,
so measured net P&L per trade stays at minus the round trip. The dial ships in
`app/winrate_dial.py`; the one lever that moves the win rate *and* the P&L the
same way is the round trip, not the geometry.

Everything below is measured on real Binance tape, every window graded, no
skips anywhere.

---

## 1. Why a win rate is a dial, not an achievement

Hit a target `k × stop` and you win `k × stop`; hit the stop and you lose one
stop. The win rate needed just to break even is therefore

```
break-even WR = 1 / (1 + k)
```

| target ÷ stop (k) | 0.25 | 0.50 | 1.00 | 2.00 | 3.00 |
|---|---|---|---|---|---|
| needed win rate | 80.0 % | 66.7 % | 50.0 % | 33.3 % | 25.0 % |

A tighter target is touched more often — so tightening `k` raises the win rate
mechanically. It also raises the requirement mechanically. Nothing about the
market, the side rule or the engine changed. That is why the whole repo's
history kept finding "high win rate, negative money": the two columns were the
same number viewed from different sides.

## 2. What was measured (and on what tape)

| artifact | tape | cells |
|---|---|---|
| `backtest/results/winrate_frontier.json` | BTC/ETH 5 m, Jan–Aug 2026 | 22 / 44 / 130 / 247 min |
| `backtest/results/winrate_frontier.json` | BTC/ETH/BNB/PAXG 1 h, Jan 2024–Aug 2026 | 1 d / 2 d / 3 d |
| `backtest/results/winrate_noabstain.json` | BTC/ETH 5 m | live cells × 7 geometries |
| `backtest/results/winrate_noabstain_horizon.json` | BTC/ETH 5 m | 1 d / 2 d / 5 d / 10 d |
| `backtest/results/winrate_noabstain_wf.json` | all four 1 h, 2.7 yr | 4-block walk-forward |
| `backtest/results/winrate_cost_sensitivity.json` | BTC/ETH | round-trip sweep |
| `backtest/results/winrate_dial_report.json` | derived | dial table |

Protocol in all of them: every epoch-aligned window trades, sides are
full-coverage causal rules (`up`, `mom44h`, `revW`, `rsi_fade`), the full
round trip is charged (BTC 28.9 / ETH 29.8 / BNB 31.0 / PAXG 37.4 bps), and a
geometry is only *lockable* if reward:risk ≥ 0.25 — the live
`hold_policy.MIN_REWARD_RISK` — and its target clears 1.5× the round trip,
with the cost floor applied by widening **both** barriers so the promised
ratio is the one that trades.

Reproduce:

```bash
PYTHONPATH=. python3 backtest/winrate_frontier.py
PYTHONPATH=. python3 backtest/winrate_dial_report.py
```

## 3. The dial, on the cells the app trades today

Same windows, same sides. Left column is a 1:1 barrier, right column is the
tightest *lockable* geometry (k = 0.25).

| cell | WR @ 1:1 | WR dialled | Δ | break-even @ k=0.25 | EV @ 1:1 | EV dialled |
|---|---|---|---|---|---|---|
| BTC tf4 (247 m) | 30.2 % | **50.5 %** | +20.3 pp | 80.0 % | −32.2 bps | −25.9 bps |
| ETH tf4 (247 m) | 35.0 % | **57.6 %** | +22.6 pp | 80.0 % | −29.8 bps | −28.4 bps |
| BTC tf3 (130 m) | 26.6 % | 38.9 % | +12.3 pp | 80.0 % | −30.9 bps | −26.3 bps |
| ETH tf3 (130 m) | 29.4 % | 45.7 % | +16.2 pp | 80.0 % | −30.9 bps | −28.1 bps |
| BTC tf2 (44 m) | 16.4 % | 19.0 % | +2.6 pp | 80.0 % | −29.6 bps | −28.5 bps |
| ETH tf2 (44 m) | 20.9 % | 25.1 % | +4.2 pp | 80.0 % | −30.4 bps | −29.8 bps |
| BTC tf1 (22 m) | 9.9 % | 10.7 % | +0.8 pp | 80.0 % | −29.2 bps | −28.8 bps |
| ETH tf1 (22 m) | 13.8 % | 15.3 % | +1.5 pp | 80.0 % | −30.1 bps | −29.8 bps |

Read the two right-hand columns: the win rate moved 20 points and the money did
not move at all. That is the honest content of the request.

* At **22–44 min** the dial barely turns. The move is so small next to 29 bps
  of friction that the cost floor swallows the target — the payable ceiling for
  BTC tf1 is **10.7 %** and no geometry, side or filter lifts it. This is
  arithmetic, not pessimism: the window's own move is the wrong size for the
  toll.
* At **247 min** the dial works: the payable ceiling is 50.5 % (BTC) /
  57.6 % (ETH), and the same tape at a 1:1 barrier shows 30.2 % / 35.0 %.

## 4. Day scale is where a high win rate actually lives

| cell | best lockable WR | k | break-even | gap | EV | n |
|---|---|---|---|---|---|---|
| ETH 2 d | **84.1 %** | 0.25 | 80.0 % | +4.1 pp | −8.5 bps | 195 |
| BTC 3 d | 80.8 % | 0.25 | 80.0 % | +0.8 pp | −25.3 bps | 130 |
| BTC 2 d | 79.5 % | 0.25 | 80.0 % | −0.5 pp | −29.6 bps | 195 |
| BTC 1 d | 77.7 % | 0.25 | 80.0 % | −2.3 pp | −25.1 bps | 390 |
| ETH 1 d | 79.0 % | 0.25 | 80.0 % | −1.0 pp | −25.8 bps | 390 |
| BNB 2 d | 79.5 % | 0.25 | 80.0 % | −0.5 pp | −31.1 bps | 195 |
| PAXG 3 d | 77.1 % | 0.25 | 80.0 % | −2.9 pp | −35.1 bps | 70 |

Every one of those rows lands within a few points of its own break-even line.
The dial does not create an edge at day scale either; it stops the cost from
being the dominant term, so the win rate converges on the honest 50 % of a
coin flip expressed as 4:1 payoff.

**Not wired:** the live cycle has row locks at 22–247 min and an era ride
capped at 7 days, so `app/winrate_dial.day_cell_plan()` returns these numbers
with `wired=False`. Shipping a day-scale lock needs its own walk-forward.

## 5. Walk-forward: does a dialled curve survive?

Selection protocol: on the first 60 % of each asset's history pick the
(side, horizon, geometry) with the highest net win rate **subject to train EV
> 0**; freeze it; grade four sequential out-of-sample blocks.

| asset | selected | OOS win rate | OOS EV | blocks positive |
|---|---|---|---|---|
| BTC | rsi_fade @ 3 d, 1:1 | 45.4 % | −12.9 bps | 1 / 4 |
| ETH | up @ 5 d, 1:2 | 41.0 % | −67.6 bps | 2 / 4 |
| BNB | up @ 3 d, 1:2 | 59.2 % | −19.5 bps | 2 / 4 |
| PAXG | up @ 5 d, 1:1 | 38.1 % | −77.3 bps | 1 / 4 |
| **pooled** | — | **48.4 %** (CI 43.4–53.4) | **−33.5 bps** | — |

The in-sample 2-day fade that looked like a 59–64 % edge on the short tape
collapsed to 48.4 % pooled out of sample. **Do not ship a dialled
configuration chosen on a win-rate criterion** — the criterion has no
predictive content, which is the same conclusion from the other direction.

## 6. The one lever with the right sign on both numbers

Same windows, same geometry, cheaper fill (taker on both sides → maker on
entry only; the cheap column is 14.4 bps for BTC and 14.9 bps for ETH):

| cell / side / geometry | WR @ full RT | WR @ half RT | EV @ full | EV @ half |
|---|---|---|---|---|
| BTC tf4 mom44h k=0.15 m=0.55 | 47.7 % | **70.9 %** | −27.2 bps | **−13.7 bps** |
| BTC tf4 revW k=0.15 m=0.55 | 51.2 % | **71.7 %** | −25.5 bps | **−14.0 bps** |
| BTC tf4 revW k=0.5 m=1.0 | 47.2 % | 52.5 % | −24.3 bps | **−10.8 bps** |
| ETH tf4 revW k=0.15 m=0.55 | 58.0 % | **75.8 %** | −28.2 bps | **−13.5 bps** |
| ETH tf1 revW k=0.15 m=0.55 | 14.7 % | **35.3 %** | −29.9 bps | **−15.2 bps** |

Halving the toll adds 17–21 points of win rate *and* halves the loss per
trade, with no window skipped. That is the only change on this tape that
improves both numbers. Caveat: OHLCV cannot tell you the maker **fill
probability**, so this table is a ceiling on what a better execution layer
would buy, not a promise of it — the honest next project is measuring fills.

## 7. What shipped

| file | role |
|---|---|
| `app/winrate_dial.py` | calibrated dial: `dial(symbol, tf, target_wr=…)` always returns a lockable geometry + its measured price; `build_row_bars_dial()` produces live TP/SL with the cost floor applied; `day_cell_plan()` exposes the day-scale numbers with `wired=False`; `cost_curve()` returns the friction sweep |
| `app/hold_policy.py` | `winrate_dial_plan()` — lazy-import accessor, so nothing about default behaviour changed |
| `backtest/winrate_frontier.py` | the frontier sweep (all cells, all geometries, full coverage) |
| `backtest/winrate_noabstain.py`, `…_horizon.py`, `…_wf.py` | geometry / horizon / walk-forward experiments |
| `backtest/winrate_dial_report.py` | printable dial table with break-even and gap columns |
| `tests/test_winrate_dial.py` | 56 pins: never abstains, never offers a lottery geometry, always quotes the price, monotone in `k`, cost curve monotone in the round trip |

The dial cannot return "no trade": for any cell with a measured frontier and
any requested win rate — including an impossible one — it returns a lockable
geometry, and an unmeasured cell falls back to the certified hold geometry. The
only `None` it can produce is for a degenerate input (zero side, non-positive
price or scale), exactly like `hold_policy.build_row_bars`.

## 8. Verdict

* **Can the win rate be raised without abstention?** Yes — today, by 20+ points
  on the 4 h cells and to 84 % at day scale. The dial is the honest way to do
  it and it reports its own price.
* **Does that make money?** No. Break-even rises with the win rate; net EV
  stays near minus the round trip on every cell, side, horizon and geometry
  tested (the only positive cells are 1–2 σ inside noise).
* **What would make money?** A cheaper round trip (measured: −15 bps of toll
  ≈ +20 points of win rate and half the loss per trade) or a real directional
  edge at ≥ 1-day horizons. Neither exists in the current book; both are
  measurable next steps rather than assumptions.
