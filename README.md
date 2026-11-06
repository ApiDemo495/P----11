> **Can't open markdown preview?** `markdown.showPreview not found`
> means the editor's *built-in* Markdown extension is disabled/absent —
> no marketplace extension can add that command back. The permanent
> cure (one run): **`bash scripts/fix_markdown_preview.sh`** from a
> terminal inside the editor (or the VS Code task *"Fix Markdown
> Preview (permanent)"*). It installs *Markdown Preview Enhanced*
> (its own preview engine) and **remaps Ctrl+Shift+V / Ctrl+K V to it
> in the real user keybindings file for your editor** — Codespaces
> (`~/.vscode-remote/data/User/`), local, macOS and Windows are all
> detected (`scripts/ensure_md_keybindings.py`; setup/auto-start also
> run it, logging to `data/md_preview.log` — nothing is silent any
> more). If you want the built-in command itself back: Extensions →
> `@builtin markdown` → *Markdown Language Features* → **Enable** →
> reload window. Zero-extension fallbacks: [`docs.html`](docs.html)
> in any browser, or **http://localhost:8000/docs** while the app runs.

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
.venv/bin/python -m pytest -q    # 276 tests (honesty, parts 1-4, news, brain, order-book, UX sync, deep RL)
```

Environment:

| Variable           | Values                        | Default           | Purpose                                  |
|--------------------|-------------------------------|-------------------|------------------------------------------|
| `FEED_MODE`        | `auto` / `binance` / `sim`    | `auto`            | Binance WS+REST, auto-fallback simulator |
| `PORT`             | any port number               | `8000`            | HTTP port                                |
| `SKIP_SETUP`       | `1` to skip bootstrap         | *(unset)*         | `run.sh` fast path                       |
| `OLLAMA_URL`       | Ollama daemon base URL        | `http://127.0.0.1:11434` | local qwen2.5-coder:3b (auditor + tab explainer) |
| `OLLAMA_MODEL`     | model tag                     | `qwen2.5-coder:3b`| pulled automatically when missing        |
| `CONNECTOME_CACHE` | cache directory               | `~/.cache/drosophila-trader/connectome` | FlyWire v783 parquet + npz (never committed) |
| `NEUPRINT_URL`     | neuPrint HTTP endpoint        | *(none)*          | OPTIONAL extra brain queries             |
| `NEUPRINT_TOKEN`   | bearer token (memory only)    | *(none)*          | neuPrint auth — never written to disk    |
| `GEMINI_API_KEY`   | Google AI Studio key          | *(none)*          | primary agent (see `API_SETUP.md`)       |
| `NEWSAPI_KEY`      | newsapi.org key               | *(none)*          | live news (else seeded/synthetic stream) |

The brain is the **real FlyWire v783 connectome** (138,639 neurons ·
15.09 M connections) or it is **absent with weight 0** — there is no
substitute matrix of any size. It downloads once (GitHub, size-checked),
builds in a worker, and hot-swaps in at boot; `POST /api/brain/reconnect`
restarts verification/build. neuPrint is an optional extra, not a
requirement.

---

### Live microstructure feeds (order-book layer)

Alongside the 1-minute klines the app consumes three telemetry streams
(telemetry never writes the tape):

| stream | purpose |
|---|---|
| `<sym>@aggTrade` | millisecond `T`/`E` clocks (clock-skew z), trade-id gaps, aggressor side for fill-vs-cancel accounting |
| `<sym>@depth20@100ms` | top-20 partial book → DBI imbalance, micro-price position, spread, cancel elasticity Λ, resting walls (`app/book.py`) |
| `!forceOrder@arr` (futures) | real liquidation prints → decaying cluster map + net forced flow |

These feed four weighted formulas — **F32 DBI**, **F33 MPD**,
**F34 CWA**, **F35 LNP** — plus the EXCHANGE PHYSICS gauges and the
±10 % confidence micro-band. On the simulator the rings stay empty and
the formulas trace *"no live book / no live forceOrder prints"*
(cause-6 honesty — no synthetic depth is ever invented). Confidence is
additionally shrunk toward the **realised calibration hit-rate** once
n ≥ 40 settled pairs exist (`calibration_factor`, clipped0.6–1.4).

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
BAPR · SARD · FBPS · TMDM · IFDS · VCDP · ARC · DLP · PCV · CLRE ·
DBI · MPD · CWA · LNP  *(F32–F35: live order-book layer)*

**The36 weighted CCI components** = F1–F19, F22–F27, F31, F32–F35
(30 signals) +
brain consensus (id 97) + CERC composite (id 98) + physics UFP (id 99) +
news impact (id 100) + hedge-fund/dark-pool combined (id 101) + the new
**Hedge-Flow Tracer (id 102)** — cross-asset volume-share rotation ×
relative return, a measured real-time "where is the money rotating"
signal with its own6 % subsystem share.  Excluded from the sum:

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
* Instance structure: the four legacy instances run on the seeded graph
  parameterised by the real hemibrain v1.2 stats (`assets/brain_real.json`,
  populations PAM 130 · SE 80 · OA 40 · TA 30 · MBON 34 · KC 2000 · LH 1500 ·
  CC 250 · FSB 150 · EB 100 · DN 500, plus generic/cluster units) — this
  powers B1–B4/hormones/geometry only. The **directional brain VOTE** is
  never taken from a stand-in: it comes from the FlyWire v783 connectome
  below (or is withheld with weight 0). Live neuPrint
  (`NEUPRINT_URL`/`NEUPRINT_TOKEN`) is an optional extra query source.
* **FlyWire v783 whole-brain vote (brain.md):** the MB sub-circuit path
  (685 ALPNs → 5,177 KCs → 96 MBONs, top-5 % APL sparsity, PAM/PPL1
  dopamine gains) presents the 30 signed formula outputs as an odour
  (push-pull ON/OFF groups), reads approach {ACh,GABA} vs avoid {Glu}
  MBON balance, and mirrors it `vote = ½(f(x) − f(−x))` (antisymmetric,
  exact 0 on a flat market). Whole-brain pass (138,639 n · 15.09 M e)
  runs on its own thread every 2 s (~60 ms) for the DN-share confidence
  (0.2–1.0, age-tagged). Learning is **dopamine RPE only**
  (`drg = tanh(2.25·δ)`, α ∈ [0.10,0.50], tonic 0.85) on a KC→MBON delta
  overlay at window boundaries, persisted in the vault, rolled back by
  hindsight when the brain gets worse. Fusion: `CCI × (1 + 0.25 ×
  hindsight × vote × base_conf)`; when the connectome is absent the
  weight is **0** and both brain formulas report `brain absent`
  (CCSv2 signed, KCAE gauge). UI: brain chip, five-step verification,
  polarity map, dopamine panel, reconnect.
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
  brain.py         4 Drosophila instances (Thing 1 + B1–B4) + FlyWire
                   vote integration, probes, status/reconnect
  flywire.py       FlyWire v783 import: acquire/build (worker)/five-step
                   verify, odour encoding, MB pass, valence vote, RPE
  explainer.py     tab explainer: qwen2.5-coder:3b (Ollama, auto-pull)
                   + deterministic fallback with the TP/SL math
  auditor.py       local LLM formula auditor (Ollama, offline-safe)
  emotion.py       E1–E22 with CERC weighting + η reward learning
  orderflow.py     OFTI / CMP / news-price correlation
  news.py          News engine (docs11): ingest/backoff/keyring, 4s
                   evaluation, novelty/habituation, emergency re-lock
  hedgefund.py     IFR · DPAD · SMDI (CCI component 101)
  physics.py       P1..P22 — PSSA…QDPC + UFP (exact Part-3 physics)
  formulas/
    f01_f10.py     Formulas 1–10
    f11_f20.py     Formulas 11–20
    f22_f31.py     Formulas 22–27, 30, 31 (+ shared F29 boundary helper)
    __init__.py    Registry (31 signal formulas, F21 = CCI) + GRC recording
static/
  index.html       Dashboard shell + 9 tabs (dashboard/formulas/emotion/
                   physics/news/brains/agents/history/settings)
  app.js           Tabs, candle chart, fx() formatter, explain panels,
                   contribution ticks, AUTO chip, gauges, news stream
  brains3d.js      Three.js 2×2 instanced brain visualization
  vendor/          three.module.js v0.160.1 (MIT, vendored — no CDN)
tests/             276 tests: parts 1–4, honesty guards, news (docs11),
                   FlyWire brain (10), order-book layer (6), UX sync
pytest.ini         pytest 9 + pytest-asyncio 1, strict asyncio mode,
                   plain `pytest` works (no cache plugin dependency)
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

## One-command automated setup 🤖

Everything — virtualenv, core + AI dependencies, the RAM-tier local model,
data directories, hourly learning cron, and the server itself — bootstraps
automatically:

| Entry point | What it does |
|-------------|--------------|
| `./run.sh` | Foreground start: runs `scripts/setup.sh` if the stamp/venv is stale, then serves on `PORT` (default **8000**). `SKIP_SETUP=1` to bypass. |
| `scripts/setup.sh` | Idempotent bootstrap: `.venv` → `requirements.txt` (hard) → `requirements-agents.txt` incl. **cmake + llama-cpp-python + huggingface-hub** (best-effort) → `data/`, `data/history/`, `models/` → **`scripts/download_models.py`** (auto-picks Phi-3-mini ≥14 GB / TinyLlama ≥7 GB / Qwen2-0.5B, downloads from the HF Hub) → hourly `learning_cron` crontab entry (best-effort). Fast no-op via `.setup_stamp`; `FORCE=1` re-runs everything. |
| `scripts/auto_start.sh` | Background supervisor (devcontainer `postStartCommand`): health-check → setup (only when stale, **niced** so the UI stays responsive) → `nohup` uvicorn → waits for `/api/health`. The editor-preview fix (install MPE only if missing + Ctrl+Shift+V remap) runs **in the background and never delays the boot**. `--stop` / `--restart` supported. |
| devcontainer | `postCreateCommand` = full setup, `postStartCommand` **and `postAttachCommand`** = auto-start (attach is what actually fires on every re-open; `remoteEnv INSTALL_AGENTS=1` turns the background agent-deps + model download ON in the container) — opening the Codespace brings the app up by itself. Outside dev containers, the **folder-open VS Code task** in `.vscode/tasks.json` runs the same auto-start in any editor (allow automatic tasks once when VS Code asks). Note: lifecycle commands only take effect after a container **Rebuild** in an existing Codespace. |
| In-process hourly loop | `main.py` runs a wall-clock safety net every 10 min: if the macro learning cycle is >60 min old it fires macro + deep + save directly — learning keeps running even without system cron (mirrors `scripts/learning_cron.sh`). |

Every AI stage is optional-by-design: no keys → formula-only agents, model
download unreachable → retried on first model use, cron absent → in-process
loop covers it. The core app never blocks on an unavailable service.

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

## Honesty layer (ADVICE_FOR_A_MULTI_ASSET_INTRADAY_APP.md — fully applied)

Every item of the advice document is implemented in this app (translated
from scalping timings to the intraday horizons):

| Advice | Where it lives |
|---|---|
| §1 tape hardening (D-1/D-2/D-3/D-7) | `app/feed.py` — single-writer guard, 120s grace, adopt/demote reconcile, binance.vision rotation, quote-print counting, `/api/feeds/diagnose` |
| §2 formula traces / zero-reasons / double-check / self-test tapes / forward tables / flat watch / superposition | `app/formulas/trace.py`, `app/formulas/selftest.py`, `app/forward.py`, formulas view |
| §3 brain push-pull, polarity calibration, endpoints | `app/brain.py` probes, `/api/brain/{status,wiring,explain,reconnect}` |
| §4 news macro lexicon, relevance gate, critical events, heartbeats | `app/news.py`, `/api/news/{status,poll}` |
| §5 emotion features, complacency gate, ±15% confidence band | `app/emotion.py`, engine band |
| §6 physics mechanism table with reasons + 4 derivation estimators | `app/physics.py` (`mechanisms`, `estimators`) |
| §7 genesis layer (10 domains, 7 metrics, lifecycle + autopsy) | `app/genesis.py`, `/api/genesis/*` |
| §8.1–8.5 ladder, mass, hindsight, meta-combiner, evidence ledger | `app/engine.py`, `app/meta.py` |
| §8.6 edge guard, temporal twin, BOCPD, Hawkes | `engine._learners` |
| §9 vault, watchdog-exit75, supervisor, placeholder, degradation | `run.sh --supervise`, `/api/{ops,watchdog,degrade,secrets/status}`, `/placeholder` |
| §10 one setInterval, 15s silent-socket, status chips, esc() | `static/app.js`, honesty tests |
| §11 key ring, ListModels model choice, deadline + window cache | `app/agents.py`, `engine._apply_agents` |
| §13 Part B detection tests | `tests/test_advice_guards.py` (18 guards) |

### Staged ANALYSE + AUTO-PREDICT

The **▶ ANALYSE** button lives on the **dashboard only**: the pipeline
runs visibly in order — formulas → emotions → physics → news → hedge →
AI review (Gemini + the local qwen2.5-coder, skipped honestly when
keyless) → final box with the BUY/SELL signal, all four TPs, the
stop-loss and the full calculation period. The whole run is under a
**hard 175 s watchdog** (3-minute user cap) and on completion every open
tab refetches — one button, all views in sync.

**AUTO-PREDICT** (dashboard chip `AUTO`/`MANUAL`, also in Settings,
persisted in `/api/settings`): when ON (default) the same staged analysis
re-runs itself at least every 45 s, so every tab is *already analysed*
before you look; when OFF only your manual ▶ ANALYSE runs it. An analysis
is read-only — locks change only at the window boundary.

```bash
curl -X POST "localhost:8000/api/analyse/BTC?tf=tf1"
curl -X PUT localhost:8000/api/settings -H 'Content-Type: application/json' -d '{"auto_predict": false}'
```

### Tab explainer (local qwen2.5-coder:3b)

Every main tab auto-loads a **WHY — plain-language readout** built from
the engine's own numbers: why the value came out this way, what it means,
the current signal, honest zeros, and the **TP/SL math with live
figures** (`TP_k = entry + dir × |conf| × ATR × √minutes_k × 0.5`,
`SL = entry ∓ 1×ATR`, dynamic level placement + fractal boundaries).
Source is the **local** `qwen2.5-coder:3b` through Ollama (temperature
0.1, ≤260 tokens, 10-min cache, `scripts/setup_local_llm.sh` or automatic
`ollama pull`); when the model is not up a deterministic fallback
explains instantly — never a blank panel. `GET /api/explain/status`
shows which source is active.

### Contribution ticks (per window)

`/api/systems` carries `ticks` — for **formulas, emotion channels,
physics mechanisms, hedge legs, news (aggregate + themes) and the brain
vote**, whether each item *actually contributed* this window and why not.
The dashboard shows a `✓ CONTRIBUTION TICKS` summary, the formula table
has a tick column, every formula card carries `✓ tick / ✗ no tick`, and
emotion/physics cards carry the same in their tooltip/why line.

### API index (all live endpoints)

| Group | Endpoints |
|---|---|
| core | `GET /api/health` · `GET /api/snapshot` · `GET /api/ops` · `GET /api/watchdog` · `GET /api/degrade` · `GET /api/secrets/status` · `GET /placeholder` |
| analysis | `POST /api/analyse/{symbol}` · `GET /api/analyse/status` · `POST /api/override/{symbol}/{tf}` |
| explain | `POST /api/explain` · `GET /api/explain/status` |
| rl | `GET /api/rl/status` |
| systems | `GET /api/systems` · `GET /api/candles` · `GET /api/formula-history` · `GET /api/formula-traces` · `GET /api/forward` · `GET /api/ledger` |
| brain | `GET /api/brain/status` · `GET /api/brain/wiring` · `GET /api/brain/explain` · `POST /api/brain/reconnect` · `GET /api/brain-geometry` · `GET /api/brain-activations` |
| news | `GET /api/news/status` · `GET /api/news/feed` · `GET /api/news/impact` · `GET /api/news/{symbol}` · `POST /api/news/poll` · `POST /api/news/emergency` |
| history | `GET /api/history/days` · `GET /api/history/day` · `GET /api/history/replay` · `GET /api/history/leaderboard` |
| learning | `GET /api/learning` · `POST /api/learning/{reset,macro,deep}` · `GET/PUT /api/settings` · `GET /api/settings/effects` |
| agents | `GET /api/agents/{status,catalog}` · `GET /api/agents/pack/{id}` · `POST /api/agents/{triage,auto,import}` · `POST /api/agents/test/{slot}` · `GET /api/auditor/status` |
| genesis | `GET /api/genesis/{status,domains,formulas}` |
| data | `GET /api/state/export` · `POST /api/state/import` · `GET /api/feeds/diagnose` |

### Reports & docs

* `docs/NEWS-REPORT.md` — docs11 news rebuild (§10 format)
* `docs/BRAIN-REPORT.md` — FlyWire component (§12 format)
* `docs/UX-ZERO-REPORT.md` — 0.00/UX pass (asks → fixes table)
* `docs.html` — self-contained renderer of README + API_SETUP
  (regenerate: `python scripts/make_docs_html.py`)

### Reinforcement-learning console

`GET /api/rl/status` + the SETTINGS tab panel expose six controllers,
all learning only from real settled windows (win +1 / loss −1 / 0):

1. **Error-finder** — every graded ring (`facc/eacc/pacc`, brain,
   tracer) is monitored; reward < 0.40 over n≥8 windows DAMPS that
   component's effective weight (×0.75, floor0.4), recovery (≥0.50)
   RESTORES it, sustained failure deepens it — every action logged and
   reversible, never invisible.
1b. **Winner boost** — reward > 0.60 over n≥8 earns weight ×1.10 per
   sustained pass up to ×1.5; the boost is withdrawn when reward fades
   under0.50, before any damp is considered.
2. **Issue bandit** — a5-armed bandit over the confidence threshold
   {0.25…0.45} commits for8 settled windows, scores ±1/0 by realised
   strict-sign outcome and re-picks argmax +15 % exploration. Fresh
   state =0.35 (identical to the legacy floor).
2b. **Contextual bandit** — the same bandit kept per context
   (asset|timeframe|volatility regime); a context speaks for itself
   once it has its own settled history, unseen contexts use the global
   bucket.
3. **Q-table over window states** — state = vol regime × trend/chop
   measured from the tape, action = subsystem-mix preset (balanced,
   core/physics/sentiment/brain-heavy) applied to the CCI shares; the
   (state, action) frozen into each lock is updated Q += 0.25(r − Q)
   on decided settlements. States with <3 decided samples play the
   balanced (legacy) mix.
4. **Honest target tracker** — prints the measured pooled accuracy vs
   the90 % target with n and gap: *RL climbs with real settlements; the
   number is never rewritten.*

### Repository rules

- **No pull requests, no merges.** Work is committed and pushed directly
  to the working branch; PR/merge operations are never used in this
  repository.

Run the detection suite:

```bash
.venv/bin/python -m pytest tests/test_advice_guards.py -q
```

Start under the supervisor (single instance, health loop, exit 75 on a
hung boot):

```bash
./run.sh --supervise
```
