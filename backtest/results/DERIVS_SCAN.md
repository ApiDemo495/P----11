# Edge scan — sequential out-of-fold settlement

> **Superseded in part (2026-11-18 merge).** Block 1's two positive
> readings (ETH catch-up +17.0 bp, BTC squeeze +6.4 bp) did **not**
> reproduce from a from-scratch rebuild — see `docs/ALPHA_RESEARCH.md`
> §0 (overlapping trades counted as independent, friction charged per
> episode in places, best-of-535 selection). They stay on the record so
> nobody re-discovers them as new; the live registry
> (`app/cost_gate.py::SLEEVES`) marks both `rejected`, and the certified
> book is `backtest/results/EDGE_SCAN.md` (§4). What still stands in this
> file: Block 2's fill-model bias probe (live in `--only bias`) and Block
> 3's per-bar falsifications (consistent with §4's closed list).

Every row is a *fixed* configuration, settled on a single-position book with
purged five-fold walk-forward and overlap-aware statistics. Parameters are
never searched inside a fold: a searched number describes the search, not the
market.

**Block 1 — event sleeves** (multi-year archive: spot 1h 2024-01 → 2026-09,
BTC USD-M perp 1h + funding/basis 2023-01 → 2025-11). Asymmetric convex exits
(tight structural stop, uncapped trailing runner), maker-pullback entries
filled **one bar later** at 8 bp round-trip (2 bp maker + 5 bp taker exit +
1 bp adverse-fill penalty).

**Block 2 — fill-model bias probe.** A coin flip on a synthetic random walk.
The naive "fill at the limit inside the bar that touched it" model pays it
~+15 bp per trade, which no venue pays for liquidity provision; the default
`next_bar` fill is the guard against that fiction.

**Block 3 — per-bar sleeves** (same archive, reference only): taker fills at
12 bp (18 bp PAXG). `folds +` counts how many of the five test folds were
profitable.

| sleeve | n | gross bp | net bp | fee bp | PF | boot t | folds + |
|---|---:|---:|---:|---:|---:|---:|---|
| ETH catch-up (BTC impulse) | 180 | +25.0 | +17.0 | 8.0 | 1.31 | +1.26 | 4/5 |
| BNB catch-up (BTC impulse) | 352 | +1.1 | -6.9 | 8.0 | 0.83 | -1.23 | 1/5 |
| PAXG catch-up (gold flight) | 156 | +0.4 | -7.6 | 8.0 | 0.67 | -1.44 | 1/5 |
| BTC squeeze (compression+funding) | 172 | +14.4 | +6.4 | 8.0 | 1.24 | +1.02 | 4/5 |
| BTC basis exhaustion fade | 3 | -33.0 | -41.0 | 8.0 | 0.00 | +nan | 0/5 |
| fill model | n | gross bp | verdict |
|---|---:|---:|---|
| wick | 2333 | +14.76 | fiction — a coin flip cannot earn this |
| next_bar | 1686 | -2.10 | unbiased |


| sleeve | n | gross bp | net bp | PF | t | folds + |
|---|---:|---:|---:|---:|---:|---|
| BTC ensemble h=12h pt2.0/sl1.0 | 1544 | -0.3 | -12.3 | 0.66 | -7.53 | 0/5 |
| BTC ensemble h=48h pt3.0/sl1.5 | 918 | -0.5 | -12.5 | 0.78 | -3.60 | 0/5 |
| ETH ensemble h=12h pt2.0/sl1.0 | 1900 | +0.0 | -12.0 | 0.77 | -5.34 | 0/5 |
| ETH ensemble h=48h pt3.0/sl1.5 | 1320 | -6.3 | -18.3 | 0.76 | -4.61 | 0/5 |
| BNB ensemble h=12h pt2.0/sl1.0 | 1747 | +0.8 | -11.2 | 0.73 | -6.11 | 0/5 |
| BNB ensemble h=48h pt3.0/sl1.5 | 1452 | -0.7 | -12.7 | 0.79 | -4.15 | 0/5 |
| PAXG ensemble h=12h pt2.0/sl1.0 | 0 | — | — | — | — | no trades |
| PAXG ensemble h=48h pt3.0/sl1.5 | 280 | -0.7 | -18.7 | 0.50 | -5.59 | 0/5 |
| BTC bollinger follow h=24 | 944 | -4.8 | -16.8 | 0.69 | -5.40 | 0/5 |
| ETH bollinger follow h=24 | 959 | +1.9 | -10.1 | 0.86 | -2.13 | 0/5 |
| BNB bollinger follow h=24 | 951 | -1.0 | -13.0 | 0.78 | -3.53 | 0/5 |
| PAXG bollinger follow h=24 | 214 | +3.7 | -14.3 | 0.58 | -3.76 | 1/5 |
| BTC cascade fade k15 z4 | 194 | +3.6 | -8.4 | 0.77 | -1.61 | 1/5 |
| ETH cascade fade k15 z4 | 237 | -2.4 | -14.4 | 0.71 | -2.16 | 2/5 |
| BNB cascade fade k15 z4 | 196 | +1.3 | -10.7 | 0.71 | -2.06 | 1/5 |
| ETH<-BTC lead-lag k3 | 57004 | +0.1 | -11.9 | 0.06 | -305.82 | 0/5 |
| BNB<-BTC lead-lag k3 | 56761 | +0.2 | -11.8 | 0.02 | -398.08 | 0/5 |
