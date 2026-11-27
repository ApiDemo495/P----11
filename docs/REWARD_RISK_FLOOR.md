# Reward:risk was below 1:1 — in three places

Reported 2026-11-27: *"didn't you saw all photos eth one was less than 1:1"*.

It was. Not on one photo — on **every graded cell on the board**, and not in
the rendering. Three separate defects, one of which was shipping live trade
geometry that violated the rule.

The previous segment fixed four *explanations*. This one fixes a *number*, and
the number was the important one.

---

## What the board said, and what it was hiding

Every cell card printed:

```
ETH TF1 · 22m      ROWS VERIFIED · SHIP GATE FAIL
74.3% gross · wilson lo 74.1%
```

No ratio anywhere on the card. The `74.3 %` belongs to a certified geometry of
**k = 0.15, m = 0.55** — a target **0.27×** its stop.

That is not a formatting omission, it is the whole story:

| quantity | value |
|---|---|
| target / stop | **0.27 : 1** |
| first-touch rate needed to break even | **78.6 %** |
| certified rate | 74.3 % |
| EV per trade, in units of risk | `0.7425 × 0.15 − 0.2575 × 0.55` = **−0.030 R** |

A 74 % "win rate" that loses three cents on every dollar risked, per window,
before any friction. Every one of the 12 graded cells selected this geometry,
because the research grid **could not select anything else** — `K_ROW` started
at `0.15`.

---

## Defect 1 — the live lock was built at 0.81:1

`app/hold_policy._cost_aware` is the function that turns a certified `(k, m)`
pair into the actual TP/SL every sealed hold lock uses.

It had a cost floor and it worked: a target inside the round trip was widened
to `1.5 × round-trip`. On ETH tf1 that is `1.5 × 29.8 = 44.7 bps`. But the
certified stop is `0.55 × scale = 55 bps`.

```
target 44.7 bps   stop 55.0 bps   →   0.81 : 1
```

…while the function's own note read *"widened to 1.5x RT so a win actually
pays"*. It paid the exchange and not the stop.

**Fixed** (`app/hold_policy.py`):

- `RR_LIVE_MIN = 1.0` with the user's rule quoted on it.
- After the cost floor, a second floor widens the **target** to
  `RR_LIVE_MIN × stop` — never the other way.
- No stop-narrowing branch exists. The stop is already capped at `0.9 × price`,
  so the target can always be widened within that same cap; the floor is
  structural, not aspirational.
- If a pair could still slip under, the builder **raises** instead of
  returning it. A seal cannot be built on a forbidden trade.
- The result publishes `rr_certified`, `rr_live`, `rr_floor` and
  `tp_widened_for_rr`, and `app/engine.py` copies them onto the seal as
  `row_rr` / `row_rr_certified` / `rr_floor` so no surface has to infer a
  payoff from two prices.

Verified across all 4 assets × 4 timeframes × 4 scales × both sides:
**no sub-1:1 live pair exists.**

---

## Defect 2 — the certified geometry could not be traded, and nothing said so

A cell wearing `ROWS VERIFIED · SHIP GATE FAIL` reads as "the measurement is
good, the ship claim is pending". It did not say that the geometry being
measured is one the app is forbidden to place.

**Fixed** (`app/hold_policy.py`):

- `sub_1_to_1_geometry()` reports the certified target as a multiple of the
  certified stop, plus the first-touch rate that ratio would need.
- New tier **`GEOMETRY BELOW 1:1 · NOT TRADEABLE`**, checked *before* every row
  and ship gate — a green gate cannot mask a rule violation.
- The badge explains itself with the arithmetic:

  > the certified geometry pays 0.2727x what it risks (target = 0.15 x stop) —
  > a target an eighth of the stop neither covers its own stop nor the 29.8
  > bps round trip. It needs a 78.57% first-touch rate to break even; the
  > certified rate is 74.2%. Such a pair is not tradeable at ANY win rate and
  > the app enforces a 1:1 floor on every live lock, so this win rate
  > describes a geometry you are not allowed to place

- `summary()` publishes `geometry_rr`, `geometry_feasible`,
  `geometry_win_rate_needed`, `geometry_note`.
- The cell card renders the ratio **above** the win rate, in the error tone,
  so `74.3 %` can never be read without `0.27:1 — BELOW THE 1:1 FLOOR` on the
  line above it.

Cell tier counts now: **12 × GEOMETRY BELOW 1:1**, **4 × UNGRADEABLE/WARMUP**.
`HOLD-OUT VERIFIED` and `SHIP GATE FAIL` no longer appear, because no cell can
honestly wear them.

---

## Defect 3 — the selector could not choose a legal pair

`backtest/hold.py`:

```python
K_ROW = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)     # before
K_ROW = (0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)   # after
```

With `M_ROW` starting at `0.15`, the old grid could not express a ratio above
`0.6 / 0.15 = 4` *only for the largest stop* — and for the stop actually
selected (`m = 0.55`) its maximum was `0.6 / 0.55 = 1.09`. The floor was
technically reachable and empirically never reached: **every cell picked the
tightest target on the grid**, because a tight target maximises first-touch
rate and the selector was scored on first-touch rate.

**Fixed:**

- `rr_floor_ok(k, m)` — the rule as a guard, not a comment.
- The guard is applied **inside** both selection loops (rows and era), so an
  edit to the grid constants cannot reintroduce a legal-looking sub-1:1 pick.
- 25 of the 81 row pairs and 13 era pairs are now skipped as illegal.
- Minimum ratio the row selector can reach: **1.0**.

⚠️ **The artifact itself is not regenerated.** Regenerating
`backtest/results/hold_wr.json` needs the 1m/5m tape (`backtest/cache/`), which
is not in this repo, and `data.binance.vision` is not reachable from this
sandbox. So the shipped numbers on screen are still the OLD geometry's — which
is exactly why they now say `NOT TRADEABLE` rather than being quietly
re-labelled. The new numbers require a re-grade:

```bash
.venv/bin/python -m backtest.hold          # with the tape present
.venv/bin/python -m backtest.wr_rereg
```

Until then the board is honest about what it holds: certified win rates it is
not allowed to trade, and 1:1 live locks whose hit rate has **not** been
measured.

That last point is the real cost of this bug and must not be buried: **there is
no measured win rate for the 1:1 geometry.** The certified 74.3 % belongs to
0.27:1. Quoting it at r ≥ 1.0 would be a fabrication.

---

## Pinned by

`tests/test_rr_never_below_one.py` (16 tests) — the rule as an invariant,
across every cell, side and scale, plus the direction of repair (target widens,
stop never narrows), the unreachability of the violation in the selector, and
the card ordering.

Also updated, each re-pointed at the behaviour it protects rather than weakened:

- `tests/test_cost_floor_winrate.py` — a "target that already pays" case that
  only cleared the cost floor because its stop was enormous now also has to
  clear the reward:risk floor, and is expressed as that.
- `tests/test_v9_real_costs.py` — `K_ROW[0] == 0.5`, and the grid's minimum
  legal ratio ≥ 1.0.
- `tests/test_hold_live.py` — 16-cell block, ship-gate direction restored via
  monkeypatch, plus a new test that a sub-1:1 geometry outranks a *passing*
  ship gate.
- `tests/test_board_explanations.py` — the new tier is in the reason map.

Full suite: **1213 passed, 3 skipped**.
