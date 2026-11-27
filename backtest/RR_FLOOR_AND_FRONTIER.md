# The 1:1 floor: everything within reach at 1:1

_User rule, now enforced in code: **a trade may never pay less than it
risks. 1:1 is the floor.** Measured on 1,004 days of real Binance 1h tape,
walk-forward with purge, **no abstention**._

---

## 1. The rule was being broken on the live board

Before any modelling: I checked whether the app honours your rule, and it
did not. **Seven to nine of sixteen live cells were publishing a target
closer than their stop.**

| cell | published ratio | verdict |
|---|---|---|
| BNB tf1 | **0.474 : 1** | NO CALL |
| BTC tf1 | **0.500 : 1** | ISSUE |
| BTC tf3 | **0.500 : 1** | ISSUE |
| ETH tf3 | **0.545 : 1** | ISSUE |
| PAXG tf2 | **0.543 : 1** | WATCH |
| ETH tf1 / tf2 | **0.800 : 1** | ISSUE |

Three independent paths leaked it, and **each was documented as correct in a
comment while doing the wrong thing**:

1. **`risk.dynamic_level_placement`** — `phi_tp` falls to `0.85` when a
   fractal boundary sits in the target's path; `phi_sl` rises to `1.30` when
   a barrier protects the stop. Two independent multipliers, and *nothing
   compared their product*. An intended 1:1 whose obstacle sat in its path
   became 0.65:1, lower again after `tighten`.
2. **`engine._install_cycle_contract`** — on `rr < 1.0` it demoted the
   cycle to `watch-only` **and published the sub-1:1 pair anyway**. That is
   the abstention you forbade and a bad ratio at the same time.
3. **`engine.build_snapshot`** — the card took the horizon target from
   `tps` and the stop from a *separately* measured `sl`, never compared
   them, then overwrote both with the cycle's frozen block.

### The fix, and its direction

At every one of those points the floor now **widens the target to meet the
measured stop**. It never narrows the stop — the stop is the risk unit, and
shrinking it to buy a prettier ratio would raise the loss frequency, which
is the opposite of what a ratio is for.

`levels.make_contract` enforces it structurally too, so a future caller
cannot reintroduce a sub-1:1 trade by forgetting:

```
live board, after   0 of 16 cells below 1:1
BNB tf1   0.4737 -> 1.000      ETH tf3   0.5455 -> 1.000
BTC tf3   0.5000 -> 1.000      PAXG tf2  0.5435 -> 1.000
```

Each card now shows the correction (`rr_floor.before`) and the panel prints
**1:1 FLOOR — target raised from 0.474:1 to 1:1 (stop unchanged)**. Nothing
abstains.

---

## 2. What is reachable at 1:1 — the honest number

I measured this three times and threw two results away. Recording why,
because the discarded ones are the attractive ones.

### 2a. First attempt said 85.9 % of ten. It was drift.

The first sweep (ratios 1.0 → 4.0) produced this:

| ratio | WR | EV |
|---|---|---|
| 2.0 | 63.6 % | +0.909 |
| 4.0 | **85.9 %** | +3.297 |

Those numbers are nonsense, and the control proved it: **shuffled labels
scored higher than real labels** (68.0 % vs 63.6 %). The cause is
structural — `resolve()` labels are computed with `side=+1` only, so "a
win" meant *a LONG reached its target*, and over 1,004 mostly-rising days
that base rate is high by itself. The model's tiebreak (`p >= 0.5`) simply
kept it long.

### 2b. The control that decides it

Non-degenerate controls (a random side cannot collapse into "always long
because the market rose"):

| ratio | long-only | random side | **model** | model − random |
|---|---|---|---|---|
| 1.0 | 47.8 % | — | 49.3 % | — |
| 1.5 | 38.9 % | 40.5 % | 43.8 % | **+3.3 pp** |
| 2.0 | 32.0 % | 34.1 % | **37.6 %** | **+3.5 pp** |
| 4.0 | 11.7 % | 16.8 % | 24.1 % | **+7.3 pp** |

The model beats a coin-flip side at every ratio — that part is real skill,
and it grows with the ratio. **But it never approaches 60 %.** At 2:1 the
honest figure is **37.6 %**, against 33.3 % break-even.

### 2c. So: the answer within your rule

| target | at 1:1 | at 2:1 | at 4:1 |
|---|---|---|---|
| wins per 10 trades | **4.9** | 3.8 | 2.4 |
| 6 of 10 | no | no | no |
| 8 of 10 | no | no | no |

**At 1:1 or better, 6 of 10 is not reachable.** Break-even at 1:1 is
50 %, the model reaches 49.3 %, and the controls say that is the true
level — not a measurement artefact. The whole "8 of 10" family lives below
1:1, which your rule excludes, and I am not going to argue for it again.

What the model *is* worth: it beats a random side by 3.5 points at 2:1,
where the payout is best. That is the edge, and it is small and honest.

---

## 3. The 44-formula stack contributes nothing

Same windows, same walk-forward, physics outputs vs controls:

| features | WR @ 2:1 |
|---|---|
| physics 44 only | 36.2 % |
| generic 30 only | 36.2 % |
| both, 74 | 36.5 % |
| **physics 44 SHUFFLED (noise)** | **37.1 %** |

A shuffled copy of the formula outputs scores *higher* than the real ones.
The live rung therefore reads price directly.

---

## 4. What shipped

* **`app/risk.py`** — 1:1 floor in `dynamic_level_placement`; published
  `rr_before_floor` / `floor_applied`.
* **`app/levels.py`** — `floor_rr()` (the last line of defence) applied in
  `from_cycle` and `make_contract`; target widened, stop untouched.
* **`app/engine.py`** — floor at the contract install and again where the
  card's levels are written; the old "demote to watch-only" branch replaced
  with the floor so nothing abstains; mandate level scaling no longer
  re-scales an already-scaled stop (that silently turned a requested 2:1
  into 1.667:1 on 8 cells).
* **`static/app.js`** — the 1:1 FLOOR badge, so a corrected ratio is never
  silent.
* **`tests/`** — the floor is asserted across the whole input space (ratio ×
  drhe × ofti × tighten × side), plus "the floor never narrows the stop",
  plus the live-board invariant of 0-of-16 below 1:1.

Suite: **1138 passed, 2 skipped.**

---

## 5. Standing statement

You cannot be under 1:1 in this product any more: it is enforced at
construction, at contract install, at the display write, and at the
execution gate. Within that rule the measured ceiling is **~4.9 wins in 10
at 1:1**, and the best expectancy is at 2:1 where the same model wins ~3.8
in 10. I would rather hand you those numbers than a 85.9 % that was the
bull market wearing a model's clothes.
