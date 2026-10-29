# Crypto Prediction 4

Production-grade real-time cryptocurrency prediction web application targeting
**BTC, PAXG, ETH and BNB**. Built for GitHub Codespaces (4-core) as a
single-process web service with a live streaming dashboard.

> **Parts 1 and 2 of the specification implemented:** master clock &
> prediction architecture, **Formulas 1–31**, Bgn synchronization, the
> four-instance Drosophila Brain system (Thing 1 + B1–B4, reduced
> 7,700-unit compute), the 22 emotion channels with CERC learning, and the
> exact F28/F29 risk layer.  Part 3 (physics formulas, news categories,
> hedge-fund/dark-pool formulas, visualization) arrives next.

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
python -m pytest -q    # 54 tests (32 part-1 + 22 part-2)
```

Environment:

| Variable        | Values                        | Default | Purpose                                   |
|-----------------|-------------------------------|---------|-------------------------------------------|
| `FEED_MODE`     | `auto` / `binance` / `sim`    | `auto`  | Binance WS+REST, auto-fallback simulator  |
| `PORT`          | any port number               | `8000`  | HTTP port                                 |
| `NEUPRINT_URL`  | neuPrint HTTP endpoint        | *(none)*| Optional live connectome source (Part 4)  |
| `NEUPRINT_TOKEN`| bearer token                  | *(none)*| neuPrint auth                             |

With `NEUPRINT_URL` unset the brains use the deterministic structural
connectome fallback (identical populations, sizes, gain rules and reduced
compute layout) so the app is fully self-contained.

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
  checklist (rolling windows, Formulas 1–31, four brain instances, emotion
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

* **Formula 28 (Adaptive Risk Calibration — exact):**

  ```
  ARC = clamp( 1.5 × (1 + |CCI|)
                 × 1/(1 + σ_current/σ_baseline)
                 × (0.8 + 0.4 × |brain_consensus|),  1, 3 )
  ARC < 1.3 → 1:1 · 1.3 ≤ ARC < 2.0 → 3:2 · ARC ≥ 2.0 → 3:1
  Confidence < 0.35 → NO TRADE (ratio withheld)
  ```

* **Formula 29 (Dynamic Level Placement — exact):**

  ```
  TP = price + dir × ATR × R_ratio × Φ_tp
  Φ_tp = 1 + 0.3 × FBPS_alignment_tp + 0.2 × min(1, DRHE)
  SL = price − dir × ATR × 1 × Φ_sl            (SL base distance = 1 ATR)
  Φ_sl = 1 + 0.3 × FBPS_alignment_sl − 0.2 × OFTI
  ```

  FBPS alignment thresholds (the spec leaves quantification open): TP side
  — boundary farther than 1.05× raw → `+1`, 0.5–1.05× → `0`, closer than
  0.5× → `−0.5` (pulls TP inward); SL side — boundary in `[1.0, 1.5]×`
  beyond SL → `+1`, boundary between price and SL → `−0.3`, else `0`.
  The SAFE protocol's ×0.85 tighten factor stacks on top of Φ_sl.  Fractal
  boundaries are shared with F24; OFTI is a labelled Part-3 seam
  (`app/orderflow.py`).
* **SL hit → Full Recalculation Event**, then:
  * new direction **matches** → amber `SAFE — BE CAUTIOUS`, SL tightened
    **15%**, next sub-target recalculated, confidence **−0.15**;
  * new direction **contradicts** → red `CLOSE POSITION IMMEDIATELY` with
    haptic vibration + alert sound, all remaining sub-targets and Tp4 are
    invalidated, and the cycle resets with a new prediction **after the next
    GRC**.

### Formulas 1–31

All formulas are exact transcriptions of the specification, registered in
`app/formulas/` (F1–F20 in `f01_f10.py` / `f11_f20.py`, F22–F27, F30, F31
in `f22_f31.py`):

`RPIC · CDIF · PEM · SATD · IRKG · SSBD · WRPP · RTDI · CLDF · DVDI ·
MSRC · RMET · DIQ · TCI · ARSI · ISA · VAD · CCMDR · PMTL · DRHE · CCI ·
BAPR · SARD · FBPS · TMDM · IFDS · VCDP · ARC · DLP · PCV · CLRE`

**The 28 weighted CCI components** = F1–F19, F22–F27, F31 (26 signals) +
brain consensus (id 97) + CERC composite (id 98).  Excluded from the sum:

* **F20 (DRHE)** — not bounded to [−1, +1], enters as the multiplier
  `min(1, DRHE)` on confidence;
* **F28 (ARC)** — a risk formula (ratio bands, not a signal);
* **F29 (DLP)** — outputs price levels;
* **F30 (PCV)** — validator that needs the CCI as its input (runs after
  aggregation);
* **F21 (CCI)** — the aggregator itself.

**Confidence pipeline (spec order):**

```
base = |CCI_adjusted| × min(1, DRHE)
     × (0.7 + 0.3 × EQ_B2)                       Brain-2 experience quality
     + 0.1 × P(dominant state)   if sign(CCI) matches the state direction
     − RTP × 0.2                 when RTP > 0.6   (E9 trauma)
     − |PCV| × 0.3               when PCV < 0     (F30 conflict)
clamp ≥ 0; issue when > 0.35
```

### Bgn (Brain-Grid-Network) synchronization

* 4×4 matrix `s_ij = exp(−|ln(τ_i/τ_j)|² / (2β²)) × coherence_ij`, `β = 1.2`,
  verified against the spec constants (0.846 / 0.334 / 0.131 / 0.664 / 0.354 /
  0.867; diagonal always 1.0).
* `coherence_ij` = |Pearson correlation| of the two timeframes' RPIC outputs
  over the last **50 GRC cycles**; recomputed at every GRC.
* **Every** formula output, brain consensus, CERC composite, physics signal
  and news signal passes through the same matrix before aggregation.

### Drosophila Brain System (4 instances)

* Weighted directed graph per brain; activations `A_n(t) ∈ (0,1)` updated
  **once per GRC**: `A_n(t+1) = σ(Σ w_mn·A_m(t)·g_mn + I_n(t))`, weights
  `w = count/max_count ∈ (0,1]`, gains `g = ±1` (inhibitory predominantly
  on mushroom-body → MBON synapses, per spec).
* **Reduced compute** (spec optimization): top 5,000 neurons individually +
  region clusters of 50 → **7,700 units/brain**; one sparse sweep (out-degree
  4) per GRC — the whole 4×4×4 brain pass costs well under a second.
* Connectome source: live neuPrint API when `NEUPRINT_URL`/`NEUPRINT_TOKEN`
  are configured, otherwise the deterministic structural fallback
  (populations: PAM 130 · SE 80 · OA 40 · TA 30 · MBON 34 · KC 2000 ·
  LH 1500 · CC 250 · FSB 150 · EB 100 · DN 500, plus generic/cluster units).
* **Thing 1 (shared):** hormone inputs `I_dopamine = max(0, r30)·V_ratio`,
  `I_serotonin = σ(5×(streak/60 − 0.5))`, `I_octopamine = min(1, |r1|/3σ)`,
  `I_tyramine = σ10/σ60` → PAM/SE/OA/TA populations; MBON-weighted readout
  `H`, `H_signal = −(H − 0.5)/0.5` (>0.6 euphoria/bearish, <0.3
  fear/bullish).
* **Thing 2 (per instance):** DB-1 `B1` Market Pattern Analyzer (KC 2000 ←
  last 20 normalized returns, pattern-MBON valence auto-learn ±0.01),
  DB-2 `B2` UX Predictor (LH 1500, `I+ = TP/(TP+SL)`, `I− = SL/(TP+SL)`,
  experience quality enters the confidence pipeline), DB-3 `B3` State
  Observer (CC 250 ← 5 stats → 5-state distribution, `+0.1·P(dom)` boost),
  DB-4 `B4` Auto-Learning Controller (FSB/EB → DN, secondary
  `ω × (1 + 0.01×(A_DN − 0.5))` on top of the reward rule).
* **Consensus** = mean of the four H_signals → CCI component 97, ARC's
  brain-agreement term, and Extreme Override's polarity-flip condition.
* **Affinity clusters:** 10 neurons per formula (31 formulas); `|F| > 0.7`
  feeds `I = |F|` into the cluster, coupling brain and formula systems.

### Emotional Flow (E1–E22)

Twenty-two dimensions with exact coefficient sets; every E1..E21 output is a
sigmoid/fraction in [0,1]:

`FII · GSL · FPR · PCP · CD · ESI · DPI · ATR_emotion · RTP · ABS · HMC ·
DRD · ERI · OFS · IAF · LAI · CDP · HTI · SMDM · NMS · TEDR · CERC`

* **E22 CERC** = `Σ w_e · E_e · direction_e` over the 21 channels (TEDR
  term carries the `(1 − 0.5·TEDR)` participation multiplier); weights start
  at `1/21`, learn with the **same multiplicative η = 0.02 rule** as the CCI
  weights (clamp `[0.005, 0.15]`, renormalize), and CERC feeds the CCI as
  component 98 after Bgn adjustment.
* GRC-lagged rings (DPI, octopamine, price index, news polarity — 10 GRCs)
  and the 10-GRC composite-emotion ring are appended exactly once per GRC.
* E9 `RTP > 0.6` also applies the explicit `−RTP × 0.2` confidence
  reduction; brain→emotion hormone averaging modulates E6 (serotonergic)
  and E7 (dopaminergic) from the four-brain average.
* Brain-to-agent slot stays `0.5` (neutral) until an agent layer exists.

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
  engine.py        PredictionEngine: phase, locks, SL protocol, pipeline
  feed.py          Binance WS/REST + simulator fallback (OHLC 1-min)
  history.py       GRC-cadence ring buffers + F27/F18/E2/E9 learning stores
  bgn.py           Synchronization matrix + adjustment
  cci.py           Formula 21: 28 weighted components, rewards, confidence
  risk.py          Formula 28 (exact ARC) + Formula 29 (exact DLP)
  brain.py         4 Drosophila instances: connectome, Thing 1 + B1–B4
  emotion.py       E1–E22 with CERC weighting + η reward learning
  orderflow.py     Part-3 seams: OFTI / CMP / news-price correlation
  news.py          News impact aggregates (decay + opposing-direction)
  physics.py       Physics signal group (part 3 extends)
  formulas/
    f01_f10.py     Formulas 1–10
    f11_f20.py     Formulas 11–20
    f22_f31.py     Formulas 22–27, 30, 31 (+ shared F29 boundary helper)
    __init__.py    Registry (27 signal formulas) + GRC recording
static/            Dashboard (vanilla JS + canvas, zero CDN dependencies)
tests/             54 tests: part-1 suite + test_part2.py (F22–F31,
                   exact F28/F29, brains, emotions, confidence pipeline)
```

## Part-3 seams (pending the next specification block)

| Component | Seam |
|-----------|------|
| Physics formulas (22) | `app/physics.py` — neutral group wired into Bgn |
| News categories + weightages | `app/news.py` — aggregates + opposing-direction API in place |
| Hedge fund & dark pool formulas | `app/orderflow.py` — OFTI / CMP / news-price correlation provisional proxies |
| TPDI (E15 IAF direction) | window-displacement fallback in `app/emotion.py` |
| Full visualization system | `static/` — panels render every snapshot field already |

## Protocol (WebSocket `/ws`)

`snapshot` (GRC cadence & spec events) → `tick` (1 Hz prices) →
`phase` (analysis countdown/readiness) → `sl_event` (SAFE / CLOSE banner
payloads). REST: `GET /api/snapshot`, `GET /api/health`,
`POST /api/override/{symbol}/{tf}`. The dashboard automatically falls back
to REST polling if the WebSocket is blocked by a proxy.
