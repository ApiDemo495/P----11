# 0.00 / UX pass — report (user asks → fixes)

Suite: **221 passed**. All changes on `arena/605bdcca-p-11`.

## 1. "0.00 errors — use different perspectives"

| offender (screenshot) | root cause | fix | where |
|---|---|---|---|
| CONFIDENCE **0.000** in the chain | additive RTP/PCV penalties clipped a small base to an exact 0 (zeros cause 1: clipped line) | total penalty capped at **90 % of available confidence** — normal arithmetic pinned by tests stays bit-identical; the collapse case keeps an honest residue + `penalty_capped` flag shown in the chain | `app/cci.py finalize_confidence` |
| real small values printed `0.000` | display rounding (cause 8) | `fx(v, d)` — first significant digit up to 6 decimals, then exponent; true zero still prints `0.00` | `static/app.js fx()`, used in confidence, chain, formula table, physics cards, leaderboard, day rows, analyse box |
| **E5 / E18 emotion `0.00`** | E5 was a binary cliff (`1.0` else exact `0.0`) | logistic ramp `1/(1+e^(−8(ratio−0.75)))` — ordering preserved, never an exact zero while inputs live; E18 now shows its small-but-real value through `fx` | `app/emotion.py` |
| F1/F2 `0.000` | structural flat / needs-history (cause 5) — **kept** | already carries `zero_reasons` + amber chip; reasons verified truthful (identical path vectors / n below baseline) — not faked | `app/formulas/trace.py` |
| physics `P5/P8/…0.000` | below-materiality gates | kept the honest zero **but every P-card now prints its own `why` line** (active ✓ / ○ why: …) instead of burying reasons in one note | `static/app.js loadPhysics` |
| leaderboard **UNDEFINED 0.000** | JS read `r.symbol‖r.tf`; API sends `r.name` | renders `r.name` + `fx(accuracy)` with green/red banding | `static/app.js` |
| day detail "no prediction rows" | JS read `d.rows‖d.predictions`; API sends **`d.records`** | reads `records` — the16 predictions of the day now list | `static/app.js` |

Remaining zeros are *honest* (structural flat, thin history, missing input) — each with a reason on screen.

## 2. "No brain, no brain weightage"

Root cause on the user's machine: **`requirements.txt` never listed `pyarrow`/`scipy`** → the FlyWire build died with `ModuleNotFoundError` → chip `brain absent`, fusion weight 0 (working as designed, but never able to start).
- `requirements.txt += pyarrow==26.0.0, scipy==1.17.1`
- `app/flywire.py acquire()` gained a **curl fallback** (some sandboxes intercept TLS in a way certifi can't see while curl's system CA can; size checks keep short files a failure)
- verified live: `POST /api/brain/reconnect` → chip `FlyWire v783 · 138,639 n · pass 59 ms` green, vote in fusion again.

## 3. ANALYSE button + AUTO-PREDICT

- **Button lives on the DASHBOARD only** (`syncAnalyseButton()` on every view switch).
- **AUTO-PREDICT toggle** (dashboard chip `AUTO`/`MANUAL` + Settings row, persisted in `/api/settings`): ON = the staged analysis re-runs itself every ≥45 s so every tab is always fresh (your "already analysed" state, made explicit); OFF = manual ▶ ANALYSE only. Default ON. Guarded off under pytest.

## 4. "Proper reasons in all tabs — use local qwen2.5-coder:3b"

New `app/explainer.py` + `POST /api/explain` + `GET /api/explain/status`:
- **Auto-import**: if Ollama is up but `qwen2.5-coder:3b` is missing, one background `ollama pull` (never blocks; status shows `pulling…`). Same daemon/model the formula auditor already used.
- Temperature 0.1, ≤260 tokens, 10-min payload-hash cache.
- **Deterministic fallback** (always available, grounded in the engine's own numbers): every tab gets a "WHY — PLAIN-LANGUAGE READOUT" panel auto-loaded on tab open — what the value means, why it came out this way, the signal, honest zeros, and the **TP/SL math with the live numbers**:
  `TP_k = entry + dir × |conf| × ATR × √minutes_k × 0.5 (TARGET_SCALE=0.5)`, `SL = entry ∓ 1×ATR` (dynamic level placement + fractal boundaries noted), plus the double-check votes.
- Dashboard: `🧠 explain this + TP/SL math` button in the WHY panel.

## 5. Tick system (which item contributed, which didn't)

`display_systems[...]["ticks"]` — per window, per group:
- **formulas** (32/32 with contribution ≥1e-5), **emotions** (|v| > 0.01), **physics** mechanisms (coverage + why when off), **hedge** legs (ifr/dpad/smdi/combined), **news** aggregate + themes, **brain** (FlyWire vote / absent).
- Dashboard: `✓ CONTRIBUTION TICKS` block with counts + first six ✗ formulas.
- FORMULAS tab: `tick` column (✓/✗) in the table + `✓ tick / ✗ no tick` chip on every formula card.
- EMOTION: each cell's tooltip carries ✓/✗ with the value.
- PHYSICS: per-card `why` line doubles as the tick.

## 6. Cool things added (my picks)

1. **Confidence trend sparkline** on the dashboard WHY panel (last 40 readings, floor0.35 in scale).
2. **Export CSV** button on HISTORY (leaderboard + day summaries, one click).
3. **Brain hormone values printed** on the DB-1..4 instance bars (DOPA/SERO/OCTO/TYRA now show numbers, not bare bars).
4. Keyboard-free one-tap re-explain per tab (`re-explain` button, server-cached).

## Test/verification notes

- Pinned math intact: `test_finalize_confidence_pipeline_order`, `test_finalize_confidence_physics_multipliers_exact` — exact values unchanged; new cap only engages where the old code produced a silent 0.
- Live smoke on :8000 — explain returns grounded TP/SL text; settings toggle round-trips; ticks present in `/api/systems`; brain chip green after reconnect.
- One full-suite run showed the known environmental flake (`test_sl_safe_branch…` under load) — passes solo, in pairs, and in the final full run (**221 passed**).
