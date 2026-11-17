# Alpha research — what was measured, what shipped, what was rejected

Date: 2026-11-16. Branch `arena/f07a39b8-p-11`.
Everything below is reproducible with `python3 -m backtest.alpha_lab`
after `python3 scripts/fetch_binance_mirror.py`.

## 1. Data (real, no synthetic claims)

* Source: Binance Vision 1h klines, GitHub mirror (`KevinSherry2016/Crypto`,
  `marketData/unzipped/…`), byte-identical to the vision daily exports.
* Coverage: **6,552 hourly bars per asset**, 2026-01-01 → 2026-09-30,
  0 missing days, all four of BTC / ETH / BNB / PAXG.
* Columns beyond OHLCV: `quote_volume`, `count` (trades),
  `taker_buy_volume`, `taker_buy_quote_volume` — this is what makes real
  **order-flow** features computable offline.
* Honest limit: the mirror carries **1h bars only**. TF1 (22m) and TF2
  (44m) therefore cannot be validated with it; the horizons tested are
  2 h ≈ TF3 (130 m) and 4 h ≈ TF4 (247 m). Sub-hour claims are not made.

## 2. Literature that shaped the design

| Channel | Source | What it said | What we did |
|---|---|---|---|
| Order-flow imbalance / taker flow | arXiv 2607.09426 (*quarter-hour effect*) | volume-normalised order imbalance predicts crypto returns over hours, with the **persistent** component carrying the signal | `taker_imb`, `taker_persist` features; contrarian prior validated per fold |
| Microstructure feature stability | arXiv 2602.00776 | OFI, spread and VWAP deviation dominate SHAP importance; OFI effect is monotone and **concave** at extremes | VWAP deviation feature; `tanh` saturation instead of linear extrapolation |
| Meta-labeling + triple barrier | López de Prado, *AFML* ch. 3–4 | separate "which way" (primary) from "will it work" (secondary); label on first touch | `triple_barrier()`, meta-label logistic, gate on P(win) |
| Purged / embargoed walk-forward | *AFML* ch. 7 | overlapping label spans leak; purge + embargo at every seam | `purged_folds(n, folds=5, embargo=horizon)` |
| Isotonic vs Platt calibration | MQL5 #12, isotonic write-ups | isotonic when n ≳ 200, Platt below; monitor **Brier, ECE, slope**; a R²-ranked model can still be systematically overconfident | `app/calib.py`: PAV isotonic on settled ledger rows, Brier/ECE/slope telemetry, identity below 40 samples |

## 3. Audit of the existing pipeline (Stage 2)

Measured on the same bars, out-of-fold, horizon 2 h / 4 h:

| Finding | Evidence |
|---|---|
| The composite is **anti-predictive** | equal-weight technical blend IC **−0.036** mean (BTC −0.046, ETH −0.063/−0.065), sign accuracy **47–49 %** — worse than a coin flip |
| Momentum is wrong at these horizons | `mom_z` IC −0.043…−0.093 on every asset; the effect is *not* rescued by trend state (trend-half IC −0.046, chop-half −0.046 on BTC) |
| Single features are weak but non-zero | \|IC\| 0.02–0.09; the strongest consistent one was the **cross-asset** term (BTC's last 6 h, contrarian): sign held in **100 %** of folds on all four assets |
| Confidence ≠ probability | stated confidence is a score magnitude; taken at face value its Brier is **0.26**, its reliability slope ≈ 0 — the classic overconfidence the calibration literature warns about |
| Barriers, not direction, decide the P&L | at 2 h a 1 σ excursion is 15–25 bp; a 12 bp round trip eats ~half of it, so any geometry with `TP < 3×friction` loses regardless of the call |

## 4. What shipped (Stage 3)

1. **`app/alpha_lab.py` + `backtest/alpha_lab.py`** — the rig: causal
   feature builder, triple-barrier labeller, purged walk-forward,
   IC / Brier / ECE / slope / EV, per-feature sign stability.
2. **`app/spillover.py`** — live cross-asset spillover + thesis-signed
   reversal block. Signs come from the validated prior (`SIGN_PRIOR`,
   never re-decided at runtime), horizons are expressed in **hours** and
   converted with the lane's bar length, inputs degrade gracefully
   (`coverage` says how many of the 5 were measurable). Blended into the
   fused score at `w = 0.25 × coverage` **before** the side is derived, so
   card / chart / contract cannot disagree about it, and added as a sixth
   recheck voter.
3. **`app/calib.py`** — isotonic confidence → P(win), fitted only on
   settled, non-seeded ledger rows, per lane with pooled fallback,
   persisted in `data/calibration.json`, identity below 40 samples and it
   *says so*. Publishes Brier raw→iso, ECE, slope.
4. **`app/barriers.py`** — volatility-scaled barrier multiples per TF with
   an explicit **friction floor** (`TP ≥ 3 × round-trip friction`);
   geometry that cannot clear cost is flagged watch-only with the reason.

## 5. Strict before/after (Stage 4)

Out-of-fold, 5-fold purged walk-forward, embargo = horizon, friction 12 bp
(PAXG 18 bp), barriers vol-scaled and fitted on train folds only.
Full table: `backtest/results/ALPHA_LAB.md`.

| Lane | sign acc before → after | IC before → after | Brier before → iso | slope | EV bp before → after |
|---|---|---|---|---|---|
| BTC 2h | 47.9 → **51.6 %** | −0.046 → **+0.048** | 0.259 → 0.251 | 0.03 | −15.0 → −11.3 |
| BTC 4h | 47.7 → **50.7 %** | −0.040 → **+0.026** | 0.260 → 0.252 | 0.02 | −19.4 → −14.0 |
| ETH 2h | 47.4 → **53.1 %** | −0.063 → **+0.055** | 0.260 → 0.251 | 0.03 | −19.4 → −7.9 |
| ETH 4h | 46.4 → **53.1 %** | −0.065 → **+0.046** | 0.263 → 0.250 | 0.04 | −27.4 → **+4.4** |
| BNB 2h | 47.4 → **52.5 %** | −0.041 → **+0.034** | 0.258 → 0.253 | 0.01 | −15.5 → −10.8 |
| BNB 4h | 47.4 → **51.1 %** | −0.034 → **+0.032** | 0.259 → 0.254 | 0.02 | −16.2 → −6.6 |
| PAXG 2h | 49.4 → 49.4 % | −0.008 → +0.014 | 0.265 → 0.253 | 0.03 | −17.6 → −19.0 |
| PAXG 4h | 50.3 → 50.0 % | +0.012 → +0.010 | 0.269 → 0.256 | 0.01 | −16.4 → −20.1 |
| **mean** | **48.0 → 51.4 %** | **−0.036 → +0.033** | **0.261 → 0.252** | 0.02 | **−18.4 → −9.7** |

High-confidence subset (calibrated P(win) ≥ 0.52 and confidence ≥ floor):
coverage 3–9 %, sign accuracy **51–56 %** on seven of eight lanes.

### What this does and does not claim

* **Claimed:** the *direction* module was pointing the wrong way, and
  flipping it with a validated prior — plus the cross-asset term — moves
  out-of-fold sign accuracy by **+3.4 pp** and turns a negative IC
  positive. Calibration removes the overconfidence (Brier 0.261 → 0.252,
  i.e. the probabilities now behave like probabilities).
* **Not claimed:** profitability. After 12 bp friction the net expectancy
  is still negative on seven of eight lanes; only ETH 4h clears (+4.4 bp).
  That is a *geometry* problem, which is exactly why `app/barriers.py`
  exists and why it refuses a target thinner than 3× friction. The honest
  next step is fewer, wider trades — not a bigger claim.
* **PAXG rejected:** the flow block did not help gold (EV slightly worse).
  Its reversal prior is weakest there; the live weight is proportional to
  measured `coverage` and the block can be zeroed per asset, but the
  current state is "neutral on PAXG", stated plainly.


## 7. Deep overhaul (second pass) — sub-hour data, microstructure, cost-aware geometry

The first pass could only speak about TF3/TF4 because the mirror carried 1h bars.

**Data expanded (Phase 1).** `scripts/fetch_binance_mirror.py` now also ingests the
monthly zipped Binance Vision archives through the git-blob API, which is the only
accessible route to sub-hour bars:
**349,920 one-minute bars per asset for BTC / ETH / BNB** (2026-01-01 → 2026-08-31,
with `taker_buy_volume`, `quote_volume`, `trade_count`), resampled to 5m / 15m / 60m.
PAXG has no accessible 1-minute archive in any reachable mirror — TF1/TF2 are therefore
**not validated for PAXG**, and no number is invented for them. TF1 (22 m) and TF2 (44 m)
are now measured directly for the three majors instead of inferred.

**New modules (Phase 3).**

| Module | What it replaces | Core formula |
|---|---|---|
| `app/micro_alpha.py` | ad-hoc flow reads | OFI `(2·taker_buy − V)/V` multi-horizon + persistence; **VPIN** on equal-volume buckets (bulk classification); **Kyle λ** = OLS slope of Δlog p on signed flow; Amihud illiquidity; trade-size skew (large prints vs lane median, aggressor-signed); **Hawkes branching ratio** from trade-count autocorrelation; **adaptive friction** from λ |
| `app/ensemble.py` | equal-weight averaging | noise band `1.5/√n` prunes features whose IC is indistinguishable from zero; prior sets the **sign**, the fit sets only magnitude; ridge shrinkage on standardised features for collinearity; cost-aware label `1[side·ret − hurdle > 0]` |
| `app/horizon.py` | fixed 12 bp assumption | lane is tradable only when σ ≥ 3× measured friction (marginal at 2×, untradable below) — the dynamic horizon rule |
| `app/alpha_lab.py` `evaluate_deep` | single composite | the full pipeline evaluated with purged folds, regression-fitted barriers, adaptive hurdle |

**Results (Phase 4)** — 14 measured lanes, purged walk-forward, embargo = horizon,
none of it fitted on a test fold. Full table: `backtest/results/ALPHA_LAB_DEEP.md`.

| Metric (mean over 14 lanes) | Before | After |
|---|---|---|
| Strict-sign accuracy | 47.9% | **51.0%** |
| Out-of-fold IC | −0.040 | **+0.018** |
| Net EV per window (adaptive friction) | −20 bp | −18 bp |
| Break-even accuracy required by the geometry | — | **53.6%** (gap −2.6 pp) |
| Lanes with positive IC | 1 / 14 | **10 / 14** |

Per-lane highlights: BTC TF1 48.7 → 51.9% (IC −0.036 → +0.033); ETH TF4 46.7 → 51.4%
(−0.068 → +0.019); PAXG TF3 50.0 → 54.3% (+0.025 → **+0.108**). Where nothing cleared
the noise band the ensemble kept **zero** features and refused to guess (ETH TF1,
BNB TF2).

**The honest finding that shapes the architecture.** Net EV is still negative on every
lane. The break-even column explains why: with TP 2σ / SL 1.5σ plus friction, TF1 needs
~62% accuracy (achieved ~51%) while TF4 needs ~51% (achieved ~51%). A 20-minute window
whose σ is 22 bp cannot pay a 15 bp round trip at any plausible hit rate — measured
`tradable` share is 1–4% at TF1 versus 59–100% at TF3/TF4. So the app now *says* that
instead of dressing it up: `app/horizon.py` marks the lane, the hero prints σ vs the
required 3× friction, and the barrier block refuses a geometry that cannot clear cost.
Positive expectancy appears only in the top-20% gated slice of the long horizons, on
3–6% coverage, which is the only place the deep model claims an edge.

## 8. Reproduce

```bash
python3 scripts/fetch_binance_mirror.py            # 1h bars, 2026-01..08
python3 scripts/fetch_binance_mirror.py --source zips --interval 1m \
        --symbols BTC ETH BNB --months 2026-01 … 2026-08   # sub-hour bars
python3 -m backtest.alpha_lab                      # classic before/after
python3 -m backtest.alpha_lab --mode deep          # deep ensemble, TF1–TF4
python3 -m pytest tests/test_alpha_lab.py -q       # 24 contracts
```

## 9. Sequential settlement (third pass) — the cost ceiling, measured

The deep pass left a contradiction worth chasing: accuracy improved on 10/14 lanes and
IC turned positive, yet net EV stayed around −18 bp. That is the signature of a metric
that measures the wrong object. The per-bar lab settles *every* qualifying bar as if it
were a separate bet, on the same price path, and its EV helper stops at the first 500
out-of-fold rows — a sample that is neither independent nor spread over the period.
A book cannot hold 15,000 overlapping positions in one symbol.

### 9.1 What was rebuilt

| module | job | why it had to exist |
|---|---|---|
| `app/portfolio.py` | one position per symbol, entry at the **next bar open** (taker), exits on high/low touches with **stop wins ties**, optional volatility trail, optional time stop, taker fees on every round trip, `neff`/`t_stat` with an overlap correction | the only settlement that matches how a book actually trades |
| `app/alpha_lab.evaluate_cascade` | one **fixed** cascade configuration settled out-of-fold, per-fold EV returned | the previous version searched the grid inside each fold; it kept choosing the rarest cell (30-bar, 4σ) and the test fold then held ~25 trades, so the number described the selector, not the market |
| `app/edge_gate.py` | tradeability verdict from an explicit registry of **out-of-fold gross expectancies**, compared with live friction | the product needs one honest sentence, not a confidence score that ignores cost |
| `backtest/edge_scan.py` | the falsification harness behind the table below | every claim reproducible in one command |

### 9.2 Literature that forced the redesign

* **Maker execution is not free money.** Fill probability and post-fill return are
  anti-correlated (arXiv 2502.18625), and a live limit-order study found entry limits
  filled 19–48 % of the time — and the fills that happened kept going against the
  order, losing 8–14× more per trip than market orders. Every sleeve here is therefore
  modelled as **taker at the next open**; a maker-filled backtest is not admissible.
* **A 2,160-strategy sweep on 1-minute crypto** found nothing positive at taker fees;
  the only survivor under stress was the "crash bid" (buy a limit 2 % below,
  sell +2 %), and its edge decayed on rolling windows. That is why the cascade-fade
  sleeve is tested with **fixed** parameters and reported with per-fold dispersion
  instead of a best-cell number.
* **AFML event sampling** (CUSUM on σ-scaled returns, triple-barrier labels, meta-label
  sizing) moved the target from "sign of the next window" to "did a barrier get hit
  first", which is the quantity a barrier-managed book is actually paid on.

### 9.3 The falsification table

Canonical run: `python3 -m backtest.edge_scan --write` →
`backtest/results/EDGE_SCAN.md`. Friction 12 bp round trip (18 bp PAXG), five purged
folds, all parameters fixed before the folds were cut.

| sleeve | n | gross bp | net bp | PF | t | folds + |
|---|---:|---:|---:|---:|---:|---|
| BTC ensemble h=12h pt2.0/sl1.0 | 425 | +2.6 | −9.4 | 0.71 | −3.30 | 1/5 |
| BTC ensemble h=48h pt3.0/sl1.5 | 261 | +5.4 | −6.6 | 0.86 | −1.18 | 1/5 |
| ETH ensemble h=12h pt2.0/sl1.0 | 376 | −2.5 | −14.5 | 0.67 | −3.73 | 0/5 |
| ETH ensemble h=48h pt3.0/sl1.5 | 285 | −8.6 | −20.6 | 0.68 | −3.09 | 0/5 |
| BNB ensemble h=12h pt2.0/sl1.0 | 396 | +7.3 | −4.7 | 0.84 | −1.64 | 1/5 |
| BNB ensemble h=48h pt3.0/sl1.5 | 286 | +2.7 | −9.3 | 0.80 | −1.83 | 0/5 |
| PAXG ensemble h=12h pt2.0/sl1.0 | 0 | — | — | — | — | no trades |
| PAXG ensemble h=48h pt3.0/sl1.5 | 280 | −0.7 | −18.7 | 0.50 | −5.59 | 0/5 |
| BTC bollinger follow h=24 | 223 | +4.4 | −7.6 | 0.83 | −1.31 | 2/5 |
| ETH bollinger follow h=24 | 210 | +7.7 | −4.3 | 0.93 | −0.52 | 1/5 |
| BNB bollinger follow h=24 | 230 | −6.1 | −18.1 | 0.62 | −3.41 | 0/5 |
| PAXG bollinger follow h=24 | 214 | +3.7 | −14.3 | 0.58 | −3.76 | 1/5 |
| BTC cascade fade k15 z4 | 194 | +3.6 | −8.4 | 0.77 | −1.61 | 1/5 |
| ETH cascade fade k15 z4 | 237 | −2.4 | −14.4 | 0.71 | −2.16 | 2/5 |
| BNB cascade fade k15 z4 | 196 | +1.3 | −10.7 | 0.71 | −2.06 | 1/5 |
| ETH←BTC lead-lag k3 | 57,004 | +0.1 | −11.9 | 0.06 | −305.8 | 0/5 |
| BNB←BTC lead-lag k3 | 56,761 | +0.2 | −11.8 | 0.02 | −398.1 | 0/5 |

Also settled and rejected in the same pass (not in the canonical table because the
constructions are auxiliary): cross-sectional pairs on ETH/BNB, ETH/BTC and BNB/BTC
(`z > 2` log-ratio reversion, 24 bp two-leg cost) — gross +0.3 … +5.6 bp, net −18 … −27 bp,
0–1 of 5 folds positive; and the per-fold grid search over the cascade family, which
degenerated to the rarest cell and returned −6.8 / −22.6 / −15.0 bp.

**No sleeve is net positive, and no sleeve is positive in a majority of folds.** The
gross column is the real finding: the best structure on record is worth **+7.7 bp** per
round trip against a **12 bp** fee.

### 9.4 Why this is a cost ceiling, not a tuning gap

With symmetric ±kσ barriers the gross payout is `(2p−1)·kσ`, so the break-even hit rate is

```
p* = 0.5 + c / (2·k·σ)            c = round-trip friction, k = barrier width in σ
```

`app/edge_gate.required_accuracy` implements exactly this. At 12 bp and k = 2:

| hold | σ (BTC) | p* required | p measured (deep pass) |
|---|---:|---:|---:|
| 1 h | ≈40 bp | 57.5 % | 50.9 % |
| 4 h | ≈80 bp | 53.8 % | 50.5 % |
| 1 day | ≈200 bp | 51.5 % | 50.9 % |
| 5 days | ≈440 bp | 50.7 % | 53.9 % (t ≈ 1) |

The requirement falls as σ grows, and that is precisely what the scan shows: the only
sleeves with positive gross are multi-day ones (BTC +5.4, BNB +7.3, ETH bollinger +7.7 bp).
They are not tradable because 8 months of data gives them ~250 trades and **t ≈ 1**: the
edge cannot be distinguished from zero, and a 2 bp margin over cost is inside the noise.
Reporting them as "positive expectancy" would be the same mistake as the best-cell
cascade number, one level down.

### 9.5 The architecture that follows

The platform stops pretending that a directional score is a tradable edge:

1. `app/edge_gate.py` keeps a **registry of out-of-fold gross expectancies** per
   (asset, horizon bucket). It never reads the live fit — that would be circular.
2. `app/engine.py` attaches `payload["edge"]` to every lane: measured gross, live
   friction, expected net, the gross it *would* need, and an `evidence` flag that is
   `insufficient` whenever `|t| < 2` or fewer than 100 rounds.
3. The hero "cost gate" row prints `blocked / thin / open` with both sides of the
   arithmetic ("OOF gross +5.4 bp vs 12.0 bp cost · net −6.6 bp · needs 14.0 bp").
   On today's measurements every lane is **blocked**, and the row says so.
4. `tests/test_edge_gate.py` pins the invariant that insufficient evidence can never be
   presented as tradable, and that no live lane is called tradable at current costs —
   the guard fails loudly if a future change starts advertising an edge it does not have.

What would legitimately turn a lane green, in order of plausibility: (a) friction at or
below the measured gross (≈6–8 bp round trip, i.e. VIP/maker-adjacent venue economics —
not a modelling change), (b) a signal worth 2.5× the current gross at the same horizons,
(c) far more history so a +5 bp edge on multi-day holds can reach `t > 2`. None of the
three is a parameter tweak, and two of them are outside this repository's data.

## 10. Reproduce (third pass)

```bash
python3 -m backtest.edge_scan                    # canonical sequential table
python3 -m backtest.edge_scan --only cascade -v  # one family only
python3 -m backtest.edge_scan --write            # refresh results/EDGE_SCAN.md
python3 -m pytest tests/test_portfolio.py tests/test_edge_gate.py \
                 tests/test_alpha_lab.py -q      # engine, gate, cascade contracts
```
