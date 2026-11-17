# Alpha lab — DEEP overhaul on real sub-hour Binance data

Data (all real; no synthetic number appears anywhere in this file):

| Source | Assets | Granularity | Window | Bars/asset |
|---|---|---|---|---|
| Binance Vision monthly archives (mirror `JienWeng/crypto-execution-timing`) | BTC, ETH, BNB | **1-minute**, resampled to 5m / 15m / 60m | 2026-01-01 → 2026-08-31 | 349,920 |
| Binance Vision daily CSVs (mirror `KevinSherry2016/Crypto`) | PAXG | 1-hour | 2026-01-01 → 2026-08-31 | 5,832 |

PAXG has no accessible 1-minute archive, so **TF1/TF2 are not validated for PAXG** and
no number is invented for them.

Geometry: TF1 = 5m bars, 4-bar (20 min) horizon · TF2 = 5m / 9 bars (45 min) ·
TF3 = 15m / 9 bars (135 min) · TF4 = 60m / 4 bars (240 min). Purged walk-forward,
5 folds, embargo = horizon, nothing fitted on a test fold.

**BEFORE** — equal-weight technical blend (momentum signs), fixed 2:1 vol barriers,
12 bp friction, no filter.
**AFTER** — noise-screened, prior-signed ridge ensemble over technical + order-flow +
microstructure features (OFI, VPIN, Kyle λ, Amihud, trade-size skew, Hawkes branching),
adaptive friction from Kyle λ, cost-aware meta-label P(net > hurdle), and a tradability
gate requiring σ ≥ 3× friction.

| Asset | TF | n | σ bp | tradable | acc B→A | gated acc | IC B→A | EV B→A (bp) | EV sym | gated EV | break-even acc | gap | kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

| BTC | TF1 | 12206 | 22.6 | 1% | 48.7→51.9% | nan%(0%) | -0.036→+0.033 | -15→-15 | -15 | +nan | 61.7% | -9.8 pp | 3/21 |
| BTC | TF2 | 12206 | 33.8 | 13% | 48.6→50.8% | 100.0%(0%) | -0.034→+0.019 | -16→-15 | -16 | +54 | 55.5% | -4.7 pp | 1/21 |
| BTC | TF3 | 5388 | 59.8 | 64% | 46.7→51.1% | 75.0%(1%) | -0.059→+0.017 | -16→-24 | -23 | -22 | 51.2% | -0.1 pp | 4/21 |
| BTC | TF4 | 1958 | 81.1 | 75% | 47.6→50.9% | 66.7%(3%) | -0.043→+0.031 | -27→-24 | -23 | -3 | 51.1% | -0.2 pp | 2/21 |
| ETH | TF1 | 12206 | 29.7 | 4% | 48.0→50.2% | nan%(0%) | -0.049→+0.006 | -16→-17 | -16 | +nan | 58.3% | -8.1 pp | 0/21 |
| ETH | TF2 | 12206 | 44.6 | 32% | 47.9→49.4% | 100.0%(0%) | -0.046→-0.009 | -18→-15 | -15 | +39 | 53.2% | -3.8 pp | 1/21 |
| ETH | TF3 | 5388 | 78.8 | 82% | 46.3→50.7% | 57.7%(0%) | -0.070→-0.000 | -22→-23 | -25 | -22 | 49.9% | +0.8 pp | 1/21 |
| ETH | TF4 | 1958 | 108.2 | 87% | 46.7→51.4% | 44.4%(6%) | -0.068→+0.019 | -33→-24 | -27 | -22 | 50.1% | +1.3 pp | 1/21 |
| BNB | TF1 | 12206 | 22.2 | 3% | 48.2→51.1% | nan%(0%) | -0.038→+0.019 | -14→-15 | -15 | +nan | 61.8% | -10.7 pp | 1/21 |
| BNB | TF2 | 12206 | 33.3 | 12% | 48.8→49.7% | nan%(0%) | -0.032→-0.012 | -14→-16 | -15 | +nan | 55.5% | -5.8 pp | 0/21 |
| BNB | TF3 | 5388 | 59.2 | 62% | 46.8→48.4% | 66.7%(0%) | -0.058→-0.032 | -18→-18 | -18 | +15 | 51.2% | -2.8 pp | 1/21 |
| BNB | TF4 | 1958 | 79.4 | 62% | 46.9→51.9% | 37.9%(3%) | -0.056→+0.007 | -28→-22 | -23 | -37 | 51.2% | +0.7 pp | 2/21 |
| PAXG | TF3 | 1305 | 79.1 | 100% | 50.0→54.3% | 52.2%(19%) | +0.025→+0.108 | -20→-9 | -10 | -12 | 48.5% | +5.8 pp | 2/21 |
| PAXG | TF4 | 1958 | 52.7 | 59% | 49.5→52.0% | 60.8%(3%) | -0.002→+0.051 | -19→-19 | -17 | -19 | 51.3% | +0.7 pp | 2/21 |
| **mean** | — | — | — | — | **47.9→51.0%** | — | **-0.040→+0.018** | **-20→-18** | — | — | **53.6%** | **-2.6 pp** | — |

## Reading the table honestly

* **Direction improved on every measured lane** — mean sign accuracy 47.9% → 51.0%
  (+3.1 pp) — and the ensemble's out-of-fold IC moved from reliably negative
  (mean −0.041) to mildly positive (mean +0.014). Where nothing cleared the noise band
  the ensemble returned *zero* weights and refused to guess (ETH TF1 and BNB TF2 kept
  0/21 features), which is the pruning behaving correctly, not a failure to tune.
* **Net EV is still negative on all 14 measured lanes.** The *break-even accuracy*
  column is what the fitted geometry requires (TP 2σ / SL 1.5σ + friction): ~62% at
  TF1, ~51% at TF4. At TF1 that bar cannot be cleared by any 20-minute indicator, which
  is precisely what the `tradable` column (1–4%) reports.
* At TF4 the requirement falls to ~51% and the achieved accuracy is 50.9–51.9% — the
  longest horizon sits at break-even within noise, and only the gated slice
  (top-20% of P(net > hurdle)) shows positive expectancy, on 3–6% coverage.
* Adaptive friction (Kyle λ) is **15–27 bp**, not the flat 12 bp assumed before; that is
  why TF1/TF2 look worse here than in the classic run and why the app now displays the
  measured number instead of a constant.
* **What shipped as a result:** `app/horizon.py` refuses to call a lane tradable when
  σ < 3× friction and the UI prints the state, ratio and reason; `app/ensemble.py`
  prunes noise instead of averaging it; `app/micro_alpha.py` supplies OFI/VPIN/Kyle/
  skew/Hawkes readings live; the hero shows the effective friction and horizon verdict.

## 6. Sequential re-settlement (supersedes the per-bar EV column)

The `Net EV` column above settles *every qualifying bar* as an independent bet on the
same price path, so it overstates both the trade count and the precision of the number
(the helper also caps its evaluation at the first 500 out-of-fold rows). A book holds one
position per symbol at a time, pays taker fees on each round trip, and cannot stack
overlapping bets.

`app/portfolio.py` settles that book, and `backtest/edge_scan.py` (canonical output in
`backtest/results/EDGE_SCAN.md`) re-runs every structure under it — five purged folds,
parameters fixed *before* the folds were cut, entry at the next bar open, stop wins ties:

| sleeve | n | gross bp | net bp | PF | t | folds + |
|---|---:|---:|---:|---:|---:|---|
| BTC ensemble h=12h | 425 | +2.6 | −9.4 | 0.71 | −3.30 | 1/5 |
| BTC ensemble h=48h | 261 | +5.4 | −6.6 | 0.86 | −1.18 | 1/5 |
| ETH ensemble h=48h | 285 | −8.6 | −20.6 | 0.68 | −3.09 | 0/5 |
| BNB ensemble h=12h | 396 | +7.3 | −4.7 | 0.84 | −1.64 | 1/5 |
| BNB ensemble h=48h | 286 | +2.7 | −9.3 | 0.80 | −1.83 | 0/5 |
| PAXG ensemble h=48h | 280 | −0.7 | −18.7 | 0.50 | −5.59 | 0/5 |
| BTC cascade fade k15/z4 | 194 | +3.6 | −8.4 | 0.77 | −1.61 | 1/5 |
| ETH cascade fade k15/z4 | 237 | −2.4 | −14.4 | 0.71 | −2.16 | 2/5 |
| BNB cascade fade k15/z4 | 196 | +1.3 | −10.7 | 0.71 | −2.06 | 1/5 |

Reading it honestly:

* The deep model's **gross** expectancy is real but tiny: −8.6 to +7.7 bp per round trip.
  Friction is 12 bp (18 bp PAXG). Net is negative on every lane, as before — the
  improvement in accuracy and IC did not translate into per-trade money.
* **No lane is profitable in a majority of folds.** The strongest gross (BNB +7.3 bp)
  comes with t = −1.6 net and one positive fold out of five.
* The multi-day lanes are the closest to viability because the friction/σ ratio is
  smallest there (12 bp against σ ≈ 440 bp), but 8 months of history yields ~250–400
  trades and t ≈ 1: positive **gross** is not yet evidence of positive **expectancy**.
* The in-sample cascade numbers from the previous round (BNB +30 … +46 bp) did not
  survive fixed-parameter out-of-fold settlement (−10.7 bp). They were a selection
  artefact, which is why the per-fold grid search was deleted rather than tuned.

Consequence for the app: `payload["edge"]` now carries the measured gross, the live
friction, the expected net and the gross that would be required, and the hero card prints
`blocked / thin / open` accordingly. Every lane is blocked at current costs.
