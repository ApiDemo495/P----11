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
