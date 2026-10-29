# Crypto Prediction 4

Production-grade real-time cryptocurrency prediction web application targeting
**BTC, PAXG, ETH and BNB**. Built for GitHub Codespaces (4-core) as a
single-process web service with a live streaming dashboard.

> **Part 1 of the specification implemented:** master clock & prediction
> architecture, Formulas 1–21, and the Bgn synchronization system.
> Formulas 22–31, the full Drosophila Brain system, and the 22 emotion
> formulas arrive with part 2 — every integration seam is already wired
> (see *Integration seams* below).

---

## Quick start

```bash
pip install -r requirements.txt
./run.sh                       # serves on http://localhost:8000
```

GitHub Codespaces: open the repo in a Codespace (`.devcontainer/` is
provided), then run `./run.sh` and forward port **8000**.

Run the test suite:

```bash
pytest -q          # 32 tests
```

Environment:

| Variable     | Values                        | Default | Purpose                                   |
|--------------|-------------------------------|---------|-------------------------------------------|
| `FEED_MODE`  | `auto` / `binance` / `sim`    | `auto`  | Binance WS+REST, auto-fallback simulator  |
| `PORT`       | any port number               | `8000`  | HTTP port                                 |

---

## Architecture

### World Clock Alignment (master clock)

* The global clock is divided into **4-minute intervals** → **15 Global
  Recheck Cycles (GRC) per hour**, firing at `:00 :04 :08 … :56`.
* At every GRC the entire system performs a complete recalculation: all
  formulas, brain states, emotion calculations, news impact scores,
  weightages and predictions — across **all four timeframes and all four
  assets simultaneously**.
* Between GRCs the 1-second price stream continues and the UI animates, but
  **no prediction values change** until the next GRC (event-driven exceptions
  defined by the spec: sub-target hit locks, SL protocol, countdown rollover,
  manual override).
* Every GRC-cadence statistic (Γ_max over 500 GRCs, SS baseline over 100,
  τ_half over 300, σ_CWP over 200, …) is kept in ring buffers that are
  appended **exactly once per GRC**, never between GRCs.

### Initial Analysis Phase

* Maximum **4 minutes** with a prominent live countdown and a readiness
  checklist (rolling windows, Formulas 1–21, four brain instances, emotion
  states, news, weightages).
* Prediction locks **early** the moment every subsystem reports ready;
  remaining countdown is skipped.
* If 4 minutes expire first, the best available prediction locks and a
  visible **confidence warning flag** attaches to every timeframe panel.

### Timeframes & prediction lock rules

| TF  | Window | Sub-targets           |
|-----|--------|-----------------------|
| TF1 | 22 min | 5 · 10 · 15 · 22      |
| TF2 | 44 min | 11 · 22 · 33 · 44     |
| TF3 | 130 min| 30 · 40 · 90 · 130    |
| TF4 | 247 min| 60 · 120 · 180 · 247  |

* **Tp4** locks at prediction and never changes during the cycle — except
  under **Extreme Override Conditions** (all three simultaneously:
  Confidence_final < 0.25 **AND** four-brain consensus flips polarity
  **AND** opposing News Impact Aggregate > 0.8).
* **Tp1** locks at prediction and may adjust at every GRC before being hit
  (lowest lock strength).
* **Tp2** is *not calculated* until Tp1 is hit; once locked it only adjusts
  when the CCI shifts by **more than 0.3**.
* **Tp3** is *not calculated* until Tp2 is hit; once locked it only adjusts
  when the CCI shifts by **more than 0.5**.
* Each timeframe has an independent countdown; on zero it immediately resets
  into a fresh cycle. A **↻ FRESH** manual override button per timeframe
  forces an immediate fresh cycle at any moment.

### TP/SL and the SL Hit Protocol

* **Formula 28 (Adaptive Risk Calibration)** selects the bounded TP:SL ratio
  — `3:1` when confidence > 0.8 with full agreement, `3:2` between 0.5–0.8
  with moderate agreement, `1:1` between 0.35–0.5, and **NO TRADE below 0.35**
  (grey overlay: `NO TRADE — INSUFFICIENT CONFIDENCE`).
* **Formula 29 (Dynamic Level Placement)** places levels using the
  volatility surface, liquidity-vacuum map (volume profile) and fractal
  boundary detection instead of arbitrary percentages.
* **SL hit → Full Recalculation Event**, then:
  * new direction **matches** → amber `SAFE — BE CAUTIOUS`, SL tightened
    **15%**, next sub-target recalculated, confidence **−0.15**;
  * new direction **contradicts** → red `CLOSE POSITION IMMEDIATELY` with
    haptic vibration + alert sound, all remaining sub-targets and Tp4 are
    invalidated, and the cycle resets with a new prediction **after the next
    GRC**.

### Formulas 1–21

All formulas are exact transcriptions of the specification, registered in
`app/formulas/`:

`RPIC · CDIF · PEM · SATD · IRKG · SSBD · WRPP · RTDI · CLDF · DVDI ·
MSRC · RMET · DIQ · TC I · ARSI · ISA · VAD · CCMDR · PMTL · DRHE · CCI`

Notes on two spec-critical details:

* **F20 (DRHE)** is *not* bounded to [−1, +1] and is explicitly “not a
  directional signal” — it therefore enters Formula 21 **only** through
  `Confidence_final = |CCI_adjusted| × min(1, DRHE)` and is excluded from the
  weighted sum (F21 requires every summand to already lie in [−1, +1]).
* **F21 (CCI)** implements dynamic weight learning
  (`η = 0.02`, clamped `[0.005, 0.15]`, renormalized), the
  double-check minority-disagreement penalty, the DRHE multiplier, and the
  `> 0.35` issuance floor.

### Bgn (Brain-Grid-Network) synchronization

* 4×4 matrix `s_ij = exp(−|ln(τ_i/τ_j)|² / (2β²)) × coherence_ij`, `β = 1.2`,
  verified against the spec constants (0.846 / 0.334 / 0.131 / 0.664 / 0.354 /
  0.867; diagonal always 1.0).
* `coherence_ij` = |Pearson correlation| of the two timeframes' RPIC outputs
  over the last **50 GRC cycles**; recomputed at every GRC.
* **Every** formula output, brain output, emotion aggregate, physics signal
  and news signal passes through the same matrix before aggregation.

---

## Data feed

1. **Binance** (production): 1-minute kline REST backfill (1000 bars → the
   247-minute TF4 window is ready instantly) + `kline_1m` WebSocket streams
   for BTCUSDT, PAXGUSDT, ETHUSDT, BNBUSDT, with automatic reconnection.
2. **Simulator** (fallback / offline): regime-switching GBM with directional
   drift regimes, aligned 1-minute bars, 1-second ticks — used automatically
   when the exchange is unreachable, so the app always comes up.

---

## Project layout

```
app/
  main.py          FastAPI app: static dashboard, WebSocket, REST
  clock.py         World Clock — GRC scheduler (:00 :04 … :56)
  engine.py        PredictionEngine: phase, locks, SL protocol, cycles
  feed.py          Binance WS/REST + simulator fallback
  history.py       GRC-cadence ring buffers (all rolling baselines)
  bgn.py           Synchronization matrix + adjustment
  cci.py           Formula 21: weights, rewards, double-check, confidence
  risk.py          Formula 28/29 call sites (provisional exact-mapping)
  brain.py         Four Drosophila brain instances (consensus contract)
  emotion.py       22 emotion channels (container; part 2 implements)
  news.py          News impact aggregates (decay + opposing-direction)
  physics.py       Physics signal group (part 2 implements)
  formulas/
    f01_f10.py     Formulas 1–10
    f11_f20.py     Formulas 11–20
    __init__.py    Registry + GRC recording
static/            Dashboard (vanilla JS + canvas, zero CDN dependencies)
tests/             32 tests: spec constants, formulas, engine rules, clock
```

## Integration seams (part 2)

| Component | Seam |
|-----------|------|
| Formulas 22–31 | register in `app/formulas/__init__.py`; add to `N_FORMULAS_ACTIVE` |
| Drosophila brains (4) | `app/brain.py` — `project()` / `consensus()` already feed Extreme Override, F28 and Bgn |
| 22 emotion formulas | `app/emotion.py` — `compute()` / `aggregate()` already feed Bgn and F28 |
| Exact F28/F29 coefficients | `app/risk.py` — inputs & outputs wired end-to-end |
| News pipeline | `app/news.py` — aggregates + opposing-direction API in place |
| Physics signals | `app/physics.py` — neutral group wired into Bgn |

## Protocol (WebSocket `/ws`)

`snapshot` (GRC cadence & spec events) → `tick` (1 Hz prices) →
`phase` (analysis countdown/readiness) → `sl_event` (SAFE / CLOSE banner
payloads). REST: `GET /api/snapshot`, `GET /api/health`,
`POST /api/override/{symbol}/{tf}`. The dashboard automatically falls back
to REST polling if the WebSocket is blocked by a proxy.
