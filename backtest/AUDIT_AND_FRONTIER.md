# Per-file audit + the learned-model frontier

_Audit run 2026-11-26 over all 148 Python files in `app/`, `backtest/`,
`scripts/` and the entry points, on the default branch._

---

## 1. Every file: compile, import, conflict

**148 files scanned. 0 hard failures.** All parse; every `app/` and
`backtest/` module imports cleanly. There is no "days-old broken file" —
the rot in this repo was never a file that fails to load. It was files that
load fine and publish a wrong number. That is why the earlier `dead_code`
and import-graph sweeps found only 1 dead function (2 lines).

### 1a. Conflicting constants (the same name, two different values)

| constant | values | verdict |
|---|---|---|
| `CONTRACT_VERSION` | **2** in `app/contracts.py`, **1** in `app/levels.py` | **two different contract systems sharing one name.** Not corrupting live data — the two are never cross-validated — but `_install_cycle_contract` treats `contracts.validate()` on `build_contract` and `levels.contract_valid()` on `make_contract` as interchangeable. A future caller WILL mix them. Should be one system. |
| `BUILD_VERSION` | 1 (`flywire.py`) vs 2 (`rlsys.py`) | different subsystems, harmless, confusing |
| `MIN_N` | 8 (`rlsys`) / 60 (`mandate_wr_search`) / 40 (`wr_audit`) | research thresholds, independent |
| `MIN_BARS` | 30 (`spillover.py`) / 61 (`tracer.py`) | independent modules |
| `LOOKBACK` / `REBALANCE` | research scripts | independent |

### 1b. 77 function names implemented in more than one module

Most are small local helpers (`_clamp`, `_sign`, `_mean`, `_num`, `_finite`)
where duplication is correct — a module should not import a private helper
from a sibling. The ones that matter are the ones that **must agree** and
have no test binding them:

* `_pearson` in `engine.py`, `learning.py`, `physics.py` — three
  implementations of one statistic.
* `_std` in `advanced.py`, `asset_behavior.py`, `physics.py`, `regime.py`.
* `_roll_mean` / `_roll_std` in `derivs.py` and `frontier_2to1.py`.

The one pair that is now **pinned by a test** is the feature transform in
`app/edge_model.py` vs `backtest/edge_model.py`
(`test_live_and_training_feature_transforms_are_identical`, max abs diff
`0.0` on 1,200 bars). That pairing exists on purpose: the live app must not
import the research tree at runtime, so a test is the only thing keeping
them in step.

**Nothing was deleted.** Deleting a working module to look productive would
be the one change here that could actually break the app.

---

## 2. How close can 6-of-10 / 8-of-10 actually get?

The ask, in expectancy terms, is the only way to compare the two:

| promise | needs | EV per 10 trades at 2:1 |
|---|---|---|
| win 6 of 10 | 60 % | +8R (+$60 per $100 risked) |
| win 8 of 10 | 80 % | +14R (+$120 per $100 risked) |
| break-even | 33.33 % | 0 |

`backtest/frontier_2to1.py` searched hand-written signals and topped out at
36.4 %. Hand-written signals are a weak class, so `backtest/edge_model.py`
lets a model learn the ranking instead: a gradient-boosted classifier on 36
causal features (momentum at nine horizons, mean-reversion z-scores,
intrabar close location, range position, volume pressure, acceleration,
volatility ratio, hour/day-of-week), walk-forward with **purge** (no
training label may settle after the test block starts), on real Binance 1h
tape.

### 2a. First, only test barriers that can actually be reached

The old harness let 76.7 % of trades expire. A barrier must resolve before
the horizon ends, so the largest stop whose 2:1 target still resolves in
time is chosen per asset **on the training half only**:

| pair | horizon | stop | resolves | TP share |
|---|---|---|---|---|
| BTC | 336 h (4×ATR) | 96.0 % | 33.4 % |
| ETH | 336 h (4×ATR) | 95.3 % | 32.6 % |
| BNB | 336 h (4×ATR) | 96.5 % | 33.4 % |
| PAXG | 336 h (4×ATR) | 97.2 % | 37.6 % |

### 2b. Out-of-sample result at 2:1

| pair | win rate | 95 % CI | n | EV |
|---|---|---|---|---|
| **BTC** | **39.9 %** | 38.8–41.1 | 7168 | **+0.197 R** |
| **BNB** | **39.5 %** | 38.3–40.6 | 7286 | **+0.184 R** |
| ETH | 34.6 % | 33.5–35.7 | 7082 | +0.037 R |

**This is a real, replicated, positive-expectancy edge at 2:1** — the first
this repo has measured. It is 6.6 points above break-even on BTC.

### 2c. But 6-of-10 is still not reachable at 2:1

The frontier — same model, same windows, only the payout changes:

| ratio | BTC WR | needs for 6/10 (>60 % CI lower) |
|---|---|---|
| 0.25 | **81.6 %** | YES (**8/10 too**) |
| 0.40 | 73.7 % | YES |
| 0.50 | 69.4 % | YES |
| 0.75 | 62.1 % | YES |
| 1.00 | 56.3 % | no |
| 1.50 | 47.1 % | no |
| **2.00** | **39.9 %** | **no** |

So the honest answer to "win 6 or 8 of every 10":

* **8 of 10 is reachable — at a 0.25:1 payout** (81.6 %, BTC and BNB both).
  It earns `0.816·0.25 − 0.184 = +0.020 R` per trade. Almost nothing, because
  the payout is almost nothing.
* **6 of 10 is reachable at anything up to about 0.75:1** (62.1 % at 0.75:1,
  +0.087 R).
* **At 2:1 the number is 39.9 %.** Confidence gating was tested and adds at
  most 1 point (41.0 % if you reject the 30 % least-confident windows, which
  is abstention and is forbidden here anyway).

The reason is arithmetic, not pessimism: `EV = p·r − (1−p)`, and the
measured `p` falls along `1/(1+r)` plus a roughly constant edge. A higher
payout mechanically buys a lower hit rate. **You can have the 80 % hit rate
or the 2:1 payout. At this edge size, not both.**

### 2d. Does the 44-formula stack add anything? No.

Model trained on the app's own formula outputs vs generic features, same
windows, same walk-forward:

| features | WR @ 2:1 | 95 % CI |
|---|---|---|
| physics 44 only | 36.2 % | 31.6–41.2 |
| generic 30 only | 36.2 % | 31.6–41.2 |
| both, 74 | 36.5 % | 31.8–41.4 |
| **physics 44 SHUFFLED (control)** | **37.1 %** | 32.4–42.1 |

A shuffled copy of the physics values scores **higher** than the real ones.
On this tape the formula stack carries no measurable directional
information — which is why the live rung reads price directly.

---

## 3. What was shipped

* `backtest/edge_model.py` — the walk-forward harness (barrier feasibility,
  purged walk-forward, the ratio frontier, block-of-10 scoring).
* `app/edge_model.py` — live consumer. **Optional**: no artifact means
  `available()` is False and the engine is byte-identical. Its feature
  transform is pinned to the training transform by a unit test.
* `app/engine.py` — the model's vote is now a rung in `_direction_ladder`,
  above the fused score, because it is the only component with a measured
  edge at the published reward:risk. It never abstains (side needs only
  `p ≥ 0.5`), never overrides the risk gates, and publishes `prob_up` on the
  card.
* `static/app.js` — the probability is on the CONFIDENCE CHAIN panel beside
  the side it moved.
* `scripts/train_edge_model.py` — reproduces the artifact.

Verify: `python -m backtest.edge_model`, `python -m backtest.frontier_2to1`,
`python scripts/train_edge_model.py`.

---

## 4. What I did not do, and why

I did not force 6-of-10 at 2:1 by relabelling timeouts as wins (the app did
that; it inflated the headline win rate 2.4×) or by fitting the test half.
Either would produce the number you asked for and lose the money.
