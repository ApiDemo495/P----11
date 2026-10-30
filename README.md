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

**The 31 weighted CCI components** = F1–F19, F22–F27, F31 (26 signals) +
brain consensus (id 97) + CERC composite (id 98) + physics UFP (id 99) +
news impact (id 100) + hedge-fund/dark-pool combined (id 101).  Excluded
from the sum:

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
× (0.5 + 0.5 × WFCP)            Part 3: ground-state overlap (P2)
× (1 − DRPT)                    Part 3: quantum drag (P6)
× (1 − GSPI)                    Part 3: gauge-projected instability (P19)
clamp ≥ 0; issue when > 0.35
```

Inside component 99 the *effective* UFP weight is `stored_UFP × TME`
(P9 Carnot factor), and directional physics inputs are likewise multiplied
by TME before the Σ (non-directional inputs use `2x − 1`, see the
contradiction log below).

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
  engine.py        PredictionEngine: phase, locks, SL protocol, pipeline,
                   rings, co-relation / formula-history / brain payloads
  feed.py          Binance WS/REST + simulator fallback (OHLC 1-min)
  history.py       GRC-cadence ring buffers + F27/F18/E2/E9 learning stores
  bgn.py           Synchronization matrix + adjustment
  cci.py           Formula 21: 31 weighted components, rewards, confidence
  risk.py          Formula 28 (exact ARC) + Formula 29 (exact DLP)
  brain.py         4 Drosophila instances: connectome, Thing 1 + B1–B4,
                   news inputs, 3D geometry + activations payloads
  emotion.py       E1–E22 with CERC weighting + η reward learning
  orderflow.py     OFTI / CMP / news-price correlation
  news.py          News impact engine: categories, decay, per-TF impacts
  hedgefund.py     IFR · DPAD · SMDI (CCI component 101)
  physics.py       P1..P22 — PSSA…QDPC + UFP (exact Part-3 physics)
  formulas/
    f01_f10.py     Formulas 1–10
    f11_f20.py     Formulas 11–20
    f22_f31.py     Formulas 22–27, 30, 31 (+ shared F29 boundary helper)
    __init__.py    Registry (27 signal formulas) + GRC recording
static/
  index.html       Dashboard shell + 5 tabs
  app.js           Tabs, candle chart, gauges, 75 panels, co-relation, news
  brains3d.js      Three.js 2×2 instanced brain visualization
  vendor/          three.module.js v0.160.1 (MIT, vendored — no CDN)
tests/             91 tests: parts 1–3 (formulas, brains, emotions,
                   confidence, news, HF, P1–P22, engine, visualization)
```

## Part 3 — Physics (P1–P22), News, Hedge-Fund, Visualization ✅

Everything below is implemented exactly as specified — no substitution with
existing indicators anywhere.

### Physics group (app/physics.py)

| # | Name | Exact implemented relations |
|---|------|-----------------------------|
| P1 | **PSSA** | N = 21 quantum levels, δP = ATR/21; σ_q = ATR×√(τ/60); κ = 0.5×LVD; h_eff = 2π σ_q v_avg; `Confidence_quantum = max&#124;c&#124;² / (1/N)` (spec-literal — contradiction log). Levels + probs feed the chart overlay. |
| P2 | **WFCP** | 21×21 `H_market` from cross-asset returns, exact `eigh` ground state; CCI × `(0.5 + 0.5 × WFCP)`. |
| P3 | **HPVU** | High-percentile volatility utilization: current ATR vs its rolling window. |
| P4 | **QTPB** | F24 (TMDM) barriers, 20-point Riemann integral of up/down barrier mass; signal = `QTPB_up − QTPB_down` (directional physics only — **F29 TP/SL placement untouched**). |
| P5 | **MEE** | Market energy envelope over the 4-timeframe momentum stack. |
| P6 | **DRPT** | Quantum drag → confidence × `(1 − DRPT)`. Formula `DRPT = 1 − exp(−γτ)`, `γ = (1/τ_deco)·(1−msrc)·(1−cthr)/2`; inputs: msrc = coherence level `&#124;F11 MSRC&#124;` (coherent bears are coherent), cthr = 0.35 confidence floor (documented below). |
| P7 | **BEOFC** | Butterfly / extreme-order-flow structure of the book. |
| P8 | **FDPD** | Fermi–Dirac pressure distribution; μ_chem = V_eff(P_VWAP) (documented). |
| P9 | **TME** | Carnot `1 − Tc/Th`, defaults Tc = 2σ, Th = 0.5σ; multiplies every directional physics input **and** the UFP effective weight (`stored × TME`). |
| P10 | **PIPP** | K = 1000, M = min(20, τ), deterministic seed (symbol/tf/price/minute). |
| P11 | **RGVF** | Ricci-gain volatility flow over GRC returns. |
| P12 | **SSBD** | Signed order parameter; **formula-literal `(T − Tc)/Tc` wins** over the prose form (documented). |
| P13 | **GLPLO** | Newtonian gravity of volume-profile mass nodes → `tanh(net force / Σ&#124;force&#124;)`. |
| P14 | **CELG** | Log-potential gaps: `h` cancels algebraically → `tanh(−sign(d)/d⁴)` (documented). |
| P15 | **DSMS** | Drift–spread with β clipped to ±0.999 (documented). |
| P16 | **RTDMP** | Return-time dispersion, market pull: `tanh((τ_eff/τ − 0.5)·2·direction)`. |
| P17 | **BBIR** | Log-space tanh of the balance ratio (documented). |
| P18 | **TPTI** | ripser point cloud at ε = 2σ, β₁ vs 200-GRC baselines; σ_β1 = 0 → z = 0 (documented). |
| P19 | **GSPI** | 3 simplified gauge transforms (S-whitney, A-milnor, M-metrics), top-5 weighted estimator → confidence × `(1 − GSPI)`. |
| P20 | **PDOBV** | k_cutoff = W/6 price-oscillation modes of the OBV spectrum. |
| P21 | **QDPC** | `conf × exp(−age/τ)`, `τ = h_eff / (2 k_B T γ_env)`; per-minute arrival sum (documented). |
| P22 | **UFP** | `Σ α_p × P_p × coupling_p`; α init 1/21, η-reward (sign-correct grows), renormalized to **exactly 1** at 6-dp storage; `coupling = max(0, corr(P_p, CCI))` over 50 GRCs; directional inputs × TME, non-directional inputs `2x − 1` (documented). |

### News impact engine (app/news.py)

| Category | Weight | Halflife D (min) |
|----------|-------:|-----------------:|
| regulatory | 0.25 | 247 |
| institutional | 0.20 | 130 |
| protocol | 0.15 | 44 |
| macro | 0.15 | 130 |
| structure | 0.10 | 44 |
| sentiment | 0.08 | 22 |
| geopolitical | 0.07 | 247 |

* `impact_score = weight × sentiment × exp(−age/D) × reliability`
  (reliability 1.0 major wires / 0.7 crypto outlets / 0.4 social).
* TF impact = `score × min(1, TF_minutes / D)`; TF4 factor 1.0 **only** when
  `D ≥ 247`.
* Signed aggregate over 247 minutes, clamped [−1, +1] → CCI **component
  100** (literal aggregate — same value on all timeframes), the Extreme
  Override input (`> 0.8` against the prediction) and 10 brain GEN input
  units (news_value × 2.0).
* NewsAPI client with automatic simulator fallback (key optional).

### Hedge-fund & dark-pool group (app/hedgefund.py) → CCI component 101

* **IFR**: V > 3× avg volume ∧ |move| < 1× avg move; upper/lower 30% of the
  candle range classifies institutional buy/sell; strength decays
  `exp(−0.05 × minutes)`.
* **DPAD**: V > 5× avg ∧ |r| < 0.3σ, confirmed by net buying over the next
  3 minutes.
* **SMDI** = `tanh((IFR − retail_flow) / 2)`, `retail_flow = −H_signal`.
* Combined = equal-weighted mean; component 101 initial weight **exactly
  0.08** (then η-learned like every component).

### Visualization (Section 9)

* **5 tabs**: Dashboard · Brains 3D · Formulas ×75 · System Co-relation ·
  News & HF.
* **Three.js 2×2 brains**: one WebGL context with scissor viewports;
  4 × 7,700 instanced spheres + 53,304 edges (green excitatory / red
  inhibitory, brightness = |weight|); activation → colour ramp; **white
  flash (500 ms ease)** on any unit whose activation jumps ≥ 24; projected
  region labels (PAM cluster, Mushroom Body, Kenyon Cells, Central Complex,
  Fan-Shaped Body, Ellipsoid Body, Lateral Horn, Descending Neurons);
  hormone bars; drag to orbit, wheel to zoom.
* **75 formula panels** (31 core + 22 emotions + 22 physics): value,
  100-GRC sparkline, weight %, 50-GRC accuracy, contribution.
* **Candlestick chart** with TP/SL/locked-target layers for the selected
  pair plus the **P1 probability overlay** (21-level ladder, bar width ∝
  normalized max|c|², peak level highlighted).
* **Prediction gauges**: confidence, base CCI, UFP, TME, WFCP, DRPT, GSPI,
  signed news, CERC.
* **System Co-relation tab**: 9-node network (4 brains, EMOTION, PHYSICS,
  CORE + AGENT-G/AGENT-L slots reserved for Part 4), Pearson r over the
  last 50 GRCs — green positive / red negative, thickness = |r|, hover for
  the link table; edges appear once ≥ 3 GRC passes exist.
* **News & HF tab**: signed aggregate card, per-TF impact bars, category
  table with live scores, item feed, IFR/DPAD/SMDI/combined readings.

### REST added in Part 3

`GET /api/formula-history?symbol&tf` · `GET /api/candles?symbol&n` ·
`GET /api/brain-geometry` · `GET /api/brain-activations?symbol&tf` ·
`GET /api/news/{symbol}`.

### Spec-contradiction log (formula literal wins; documented)

| Item | Resolution |
|------|-----------|
| P1 prose vs formula | `max&#124;c&#124;²/(1/N)` literal confidence quantum |
| P12 | `(T − Tc)/Tc` literal (not the prose signed form) |
| P15 | β clipped to ±0.999 |
| P22 non-directional | `2x − 1` mapping |
| P22 coupling | `max(0, corr)` — disagreement removes coupling |
| P8 | μ_chem = V_eff(P_VWAP) |
| P9 / P21 | per-minute arrival sums |
| P14 | h cancels → `tanh(−sign/d⁴)` |
| P17 | tanh in log space |
| P18 | σ_β1 = 0 → z = 0 |
| P21 τ | τ literal ⇒ boot τ ≈ 3.9e11 min (decay ≈ 1 for any realistic age; spec-literal, documented) |
| P6 inputs | Formula unchanged; user delegated: msrc = `&#124;F11&#124;` coherence level (signed MSRC would max DRPT on coherent down-trends) and cthr = 0.35 confidence floor (raw signed DVDI made `(1−cthr)` ≈ 1.76 and pinned DRPT ≈ 1, blocking organic issuance entirely — verified by a 240-GRC probe before the fix) |
| Part-4 η supersession | Section 10's `η_emotion = η_physics = 0.015` with clamps `[0.01, 0.12]` supersedes Part 2's CERC `η=0.02` / `[0.005, 0.15]` (newer spec section wins; formula rule unchanged) |
| S10 vs S14 subsystem step | S10 deep says ±5%, S14 L1 says ±2%: implemented **multiplicative ±5%** (deep-cycle-specific) then the S10 clamps `[0.05,0.25]/[0.05,0.25]/[0.03,0.15]/[0.03,0.12]` then S14's renormalize-to-1 — all three constraints satisfied in that order |

## Protocol (WebSocket `/ws`)

`snapshot` (GRC cadence & spec events) → `tick` (1 Hz prices) →
`phase` (analysis countdown/readiness) → `sl_event` (SAFE / CLOSE banner
payloads). REST: `GET /api/snapshot`, `GET /api/health`,
`GET /api/formula-history`, `GET /api/candles`, `GET /api/brain-geometry`,
`GET /api/brain-activations`, `GET /api/news/{symbol}`,
`POST /api/override/{symbol}/{tf}`. The dashboard automatically falls back
to REST polling if the WebSocket is blocked by a proxy.

---

## Part 4 — Auto-Learning, Agents, UI Redesign, History, Weightage ✅

### Section 10 — the permanently active auto-learning system (`app/learning.py`)

| Cycle | Cadence | What happens (spec-exact) |
|-------|---------|---------------------------|
| **Micro** | every GRC | Core formula weights `ω ← ω·(1 + η_micro·reward)` η=0.02 clamp `[0.005,0.15]` renormalized (engine). Per-brain influence `w_b ← w_b·(1 + 0.01·reward)` clamp `[0.1,0.4]` renormalized across the 4 brains (weighted consensus). CERC 21-member weights η_emotion=0.015 clamp `[0.01,0.12]`. Physics P-group α η_physics=0.015 clamp `[0.01,0.12]`. |
| **Macro** | every 15 GRCs (~60 min) | Rolling-15 accuracy: `<0.4` → halve weight + probation flag until next review; `>0.7` → ×1.2 bonus (scaled by the Brain-4 DN tone). Cross-formula output correlation `>0.85` → the lower-accuracy twin loses 30%. Brain 4 receives the full macro metrics on its FSB drive; DN outputs modulate those weights. Weibull `(k,λ)` snapshots for regime durations; λ_critical streaks keep streaming into F18 every GRC. |
| **Deep** | every 247 min (one TF4 cycle) | Per-timeframe hit-rate check: `HR<0.5` → diagnostic mode — top-3 wrong formulas ×0.6 (`−40%`) and 247-min brain-affinity inhibitory input, top-3 right-despite-wrong ×1.5 (`+50%`). Bgn coherence re-anchored from the full 247-min RPIC history (blended 50/50 with the rolling-50 while fresh). Baum-Welch HMM retrain per (asset, timeframe) — the MSRC **formula is unchanged**; the HMM is its regime-context module. Weibull refit for all five regime types. CERC/physics/news/HF subsystem shares re-evaluated from agreement-vs-outcome contributions with the Section-10 clamps and Section-14 renormalization. |

Adaptive rates: rolling-100 per-timeframe accuracy `<85%` → **high-alert mode** (η_micro=0.04, η_emotion/physics=0.03, probation 0.5); all timeframes `>92%` → normal rates. Target: **90%** Tp-before-SL accuracy per timeframe over the rolling 100.

State persists to **`data/auto_learning_state.json`** (committed; survives restarts — weights, brain influence, HMMs, Weibull, shares, probation). Hourly safety net: `scripts/learning_cron.sh` → `POST /api/learning/macro`.

### Section 11 — agent integration (`app/agents.py`)

* **Gemini** — one call per GRC. Prompt = prices + 10-min returns + CCI direction/confidence + top-5 formula signals + brain consensus + latest news + CERC composite → JSON `{direction, confidence, insight}`.
* Multipliers: cloud-only `CCI × (1 + 0.10·gd·gc)`; local-only `× (1 + 0.08·ld·lc)`; both `× (1 + 0.12·ad·ac)`.
* Failover after 3 consecutive failures: Gemini → Backup1 (Gemini) → Backup2 (OpenAI GPT-4o) → Backup3 (Anthropic Claude); rate-limited slots skip forward; last-good signal reused for 3 GRCs, then neutral + warning.
* **Local model** via llama-cpp-python, auto-selected by RAM: ≥14 GB Phi-3-mini-4k Q4_K_M · ≥7 GB TinyLlama-1.1B Q4_K_M · else Qwen2-0.5B. Doubles as the news sentiment analyzer. Optional deps in `requirements-agents.txt`; keys in `data/settings.json` or env (`GEMINI_API_KEY`, …) — see **API_SETUP.md**.

### Section 12 — UI redesign (`static/`)

Fixed **280px sidebar**, **9 tabs** (Dashboard · Brain Monitor · Formula Lab · Emotion Grid · Physics Engine · News Feed · Agent Control · History · Settings), palette `#0a0a0f / #00e5ff / #ff1744 / #ffd600`, Inter + JetBrains Mono. Dashboard: four TF quadrants, circular countdown rings (cyan flash on GRC reset), vendored **lightweight-charts** candlesticks with dashed→solid TP/SL price lines, translucent quantum-probability strip, Tp1–4 lock dots (Tp4 always filled), circular confidence gauge (red→yellow→green, 0.5 s ease), TP:SL ratio, per-quadrant asset buttons, full-width alert bar (amber SAFE / pulsing-red CLOSE). Formula Lab: searchable/filterable **75 cards** with KaTeX LaTeX + live Term Breakdown. Emotion Grid: 22 alias cells (FOMO…Complacency) with intensity circles, ELEVATED/NEUTRAL/EXTREME/DECAYING tags, pulse `>0.7`, CERC regime labels (EUPHORIC/PANICKED/COMPLACENT/CONFLICTED/EXHAUSTED). Physics: P1 quantum-level probability curve with price/TP/SL markers, Quantum State Summary, density-colored phase-space scatter. News: timestamp / headline / gold-silver-bronze reliability / category+weight / sentiment / impact bar / decay countdown. Agent Control: masked keys, status LEDs, hourly rate bars, local-model stats, **Test Connection** buttons. History + Settings below. Haptics: lock 50 ms · Tp double · SL 200 ms · CLOSE triple-strong · GRC 20 ms.

### Section 13 — history (`app/historydb.py`)

Every GRC writes `data/history_YYYY-MM-DD.json`: timestamp, prices, per-tf/asset predictions (tp/sl + lock_status), outcomes (`target_hit` / `sl_hit` / pnl%), 31 formula outputs, brain states, CERC, physics UFP, news impact, agent signals, weight snapshot. Retention **90 days → weekly summaries** (daily accuracy + GRC counts only). History tab: accuracy calendar (green >80 / yellow 60–80 / red <60), day-detail table, rolling-100 hit-rate chart, formula leaderboard, GRC-by-GRC **Prediction Replay** (prev/next/play).

### Section 14 — weightage (five levels)

* **L1** subsystem shares of the CCI: core 0.45 · brain 0.10 · emotion 0.15 · physics 0.15 · news 0.07 · HF 0.08 — components 97–101 take their share directly; the 26 core signals share `core` ∝ their learned ω. Deep cycle adjusts with the contribution rule + clamps + renormalize.
* **L2** within-core ω per formula (micro η rule).
* **L3** timeframe multiplier: `ω_f × perf_f_TF / avg_perf_f` (rolling 100).
* **L4** asset multiplier: same rule per symbol.
* **L5** sub-target relevance profiles — Tp1 fast: SATD/VAD/WRPP · Tp3: MSRC/DIQ · Tp4 slow: CLRE/DRHE/UFP (exposed per GRC in `weight_profile`).

### Section 15 — API setup

**API_SETUP.md** (repo root): Gemini keys, neuPrint + 3 backups, NewsAPI + 3 backups, Binance + 3 backups, local-model dependencies.

### Section 16 — integration

* `&#124;F_f&#124; > 0.7` → brain affinity clusters (existing coupling, now with deep-cycle inhibitory modulation).
* CERC → Brain-2 LH experience input; brain-3 dominant state → emotion weights (existing); TEDR → P6 DRPT modulation; TME → emotion weight share.
* IFR×OFTI / DPAD×LVD / SMDI×SMDM confirmation flags (hedge panel).
* **Mandatory 4-vote double-check** every GRC: CCI · PCV · agents · brain consensus — ≥3 agree → `confidence × 1.05`; 2 agree → stands with `× 0.85` + **MIXED SIGNALS**; ≤1 agrees → majority direction wins (issued=False — decision requires human confirmation) or **NO TRADE** on a 2–2 tie.
* Optimization priority honored: correctness → data quality → parameter optimization (macro) → execution speed (threaded local model, cached geometry) → redundancy pruning (macro step 4).

### New endpoints

`GET /api/learning` · `POST /api/learning/macro` · `POST /api/learning/deep` · `POST /api/learning/reset` ·
`GET /api/history/days|day|replay|leaderboard` · `GET|PUT /api/settings` · `GET /api/agents/status` ·
`POST /api/agents/test/{slot}` · `GET /api/state/export` · `POST /api/state/import` · `GET /api/systems`.

**Tests: 119 passed** (test_part1/2/3 + `tests/test_part4.py` — learning cycles, HMM, persistence round-trip, two-tier CCI, double-check matrix, agents failover, history retention, UI contract).
