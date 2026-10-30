# Current State
_What is being worked on right now, what is unfinished, and the immediate next steps. Always update this section. If the outcome of the most recent action is unknown, say that explicitly._

**PART 3 COMPLETE, COMMITTED, PUSHED as `1d805a6`** on `arena/605bdcca-p-11` (history: `43ad8a3 → b80eddd → aa9021b → 1d805a6`). 91 tests passed. UI smoke passed (Node DOM harness, 75 panels + all tabs). Live server running on port 8017 (uvicorn, sim feed fallback). **Organic issuance PROVEN**: first issue at GRC+186 on the deterministic trend feed (BNB tf2, conf=0.350, dir=+1, SL 2421.67 / TP 2474.79); live server shows conf rising once coherence builds.

Awaiting **Part 4** from the user (auto-learning, agent integration Gemini/local models, UI design specs, History system, Weightage system, API documentation, remaining end requirements). ask_user sent; on spec arrival, build immediately per the user's standing "build_first" decision.

CRITICAL incident this segment: **Re-clone #3** reset the workspace to `43ad8a3`, destroyed `.venv` and `/tmp`, and dropped branch history. Recovery: `git fetch origin arena/605bdcca-p-11 && git reset --soft FETCH_HEAD && git reset` (index-only) — recovered `aa9021b` (was pushed) with working tree intact; venv rebuilt from requirements.txt (numpy 2.4.6, ripser 0.6.15 pinned). LESSON: **commit + push after every milestone immediately** — pushed commits are the only durable history across re-clones.

# Task
_Production-grade real-time crypto prediction web app (BTC, PAXG, ETH, BNB), GitHub Codespaces 4-core; every formula/coefficient exactly as specified; no substitution with existing indicators._

Part 1 (done): 4-min GRCs; Initial Analysis Phase; TF1=22m/TF2=44m/TF3=130m/TF4=247m sequential Tp locks (0.3/0.5), Tp4 + Extreme Override; F28/F29/SL protocol; Formulas 1–21; Bgn 4×4 (β=1.2).

Part 2 (done, `aa9021b`): F22–F31 exact, ARC/DLP/PCV, 4 brains (7,700 units), 22 emotions, CERC, 28→31-component CCI, confidence pipeline, UI, 54 tests.

Part 3 (DONE, `1d805a6`): P1 PSSA · P2 WFCP · P3 HPVU · P4 QTPB · P5 MEE · P6 DRPT · P7 BEOFC · P8 FDPD · P9 TME · P10 PIPP · P11 RGVF · P12 SSBD · P13 GLPLO · P14 CELG · P15 DSMS · P16 RTDMP · P17 BBIR · P18 TPTI (ripser) · P19 GSPI · P20 PDOBV · P21 QDPC · P22 UFP — all exact with documented contradiction log in README. News engine (7-category exact table, signed 247-min aggregate → component 100 + override + 10 GEN brain inputs, per-TF impacts, NewsAPI sim fallback). HF group IFR/DPAD/SMDI → component 101 (init weight exactly 0.08). CCI: 31 components, order `base→×min(1,DRHE)→×(0.7+0.3·EQ)→+0.1·P(dom)→−RTP·0.2→−|PCV|·0.3→×(0.5+0.5·WFCP)→×(1−DRPT)→×(1−GSPI)→clamp, issue >0.35`; UFP effective = comp×TME. Visualization: 5 tabs; Three.js 2×2 brains (vendored v0.160.1, 4×7,700 instanced spheres, 53,304 edges, white flash 500 ms, region labels, hormone bars, orbit/zoom); 75 formula panels (31+22+22: value/100-GRC sparkline/weight%/50-GRC accuracy/contribution); candle chart with TP/SL/locked-target layers + P1 probability ladder; 9 gauges; System Co-relation tab (9 nodes incl. AGENT-G/L slots, 50-GRC Pearson, edges from ≥3 passes); News & HF tab. REST: /api/formula-history, /api/candles, /api/brain-geometry, /api/brain-activations, /api/news/{symbol}.

**P6 DRPT (user-delegated fix)**: formula `DRPT = 1−exp(−γτ)`, `γ = (1/τ_deco)(1−msrc)(1−cthr)/2` unchanged; call-site now `msrc = |F11 MSRC|` (coherence level; signed punished coherent bears) and `cthr = CONFIDENCE_FLOOR (0.35)` (raw signed DVDI ≈ −0.76 made (1−cthr) ≈ 1.76, pinning DRPT ≈ 0.85–0.93 and blocking ALL issuance forever — proven by a 240-GRC probe: conf plateaued at 0.177 pre-fix; post-fix first issue at GRC+186). User answered "change according to your will and fix that" via ask_user.

# User Constraints & Corrections
- Every formula, coefficient, relationship, term preserved precisely; no substitution with existing indicators. Spec contradictions → formula literal wins + document in README (done for P1/P6/P8/P9/P12/P14/P15/P17/P18/P21/P22).
- Build each part immediately rather than pasting the next spec first (user's explicit decision). Ask for Part 4 only after Part 3 is complete (now satisfied).
- App must run as a web app in GitHub Codespaces 4-core environment.

# Workspace
- /home/user/P----11 — branch `arena/605bdcca-p-11`; commits `43ad8a3` → `b80eddd` → `aa9021b` → **`1d805a6` (Part 3, pushed)**.
- .venv/ — rebuilt after re-clone #3; numpy 2.4.6 + ripser 0.6.15 + fastapi stack; use `/home/user/P----11/.venv/bin/python|pytest` (PEP 668).
- app/config.py CONFIDENCE_FLOOR=0.35 (now also DRPT's cthr).
- app/physics.py — P1..P22 complete; p06_drpt docstring documents the input semantics.
- app/engine.py — pspark1..22 / pacc1..22 / emospark22 / eacc22 / fspark21+facc21 rings; formula_history core=F1..F31; corelation 9 nodes; geometry/activations payloads.
- static/ — index.html (5 tabs), app.js (~1300 lines, Part-3 block appended: tabs/chart/gauges/panels/corel/news/brain3d glue), brains3d.js (module), vendor/three.module.js + THREE_LICENSE.
- tests/test_part3.py — 37 tests incl. extended P6 semantics test.
- README.md — Part-3 section: P1–P22 table, news category table, HF, visualization, REST list, contradiction log (incl. P6 inputs row).
- Live server: uvicorn on 0.0.0.0:8017 (process "Prediction server", sim feed — Binance blocked in sandbox).
- Scratch lost on re-clone (regenerate as needed): /tmp/domharness.js (Node DOM smoke — RECREATED & passing), /tmp/issuance_probe.py (drives engine.on_grc fast-forward).

# Actions Taken
_Terse ordered log:_
- (Part 1) `5de4f1c` lost to re-clone; (Part 2) recovery `b80eddd` → **`aa9021b` pushed**.
- Part 3 backend: config/news/hedgefund/physics/engine/cci/models/risk/history/brain/main — complete; Ring `__iter__`/`__getitem__` added.
- `pytest tests/test_part3.py` 27/37 → fixed (news_units, SMDI sign, DPAD craft, UFP non-dir −1 expectation, P14/P16 tolerances, alpha exact-sum via largest-remainder) → **91 passed**.
- Static: requirements += numpy/ripser; three.js r160.1 vendored via npm tarball; index.html rewritten (tabs); app.js +630 lines; brains3d.js module; style.css +260 lines; `node --check` both JS green.
- Engine fixes: pspark/pacc rings (physics sparks were NEVER written — caught via formula-history probe); emospark22/eacc22; facc21; fspark21.
- E2E: all endpoints 200 (candles incl. live bar, geometry 7700/53304/8/10, activations 4×7700, news, snapshot p1 21 levels probs sum 1.0); UI DOM harness SMOKE PASSED (31+22+22 panels, news tab rendered, corel stats).
- Issuance investigation: 240-GRC probe pre-fix → conf plateau 0.177 (DRPT 0.85–0.93); ask_user → "your will"; P6 input fix; **post-fix probe: FIRST ORGANIC ISSUE GRC+186 BNB tf2 conf 0.35**; live server drpt matches model (msrc 0.327 → drpt 0.665 ✓).
- **Re-clone #3 mid-flight**: recovered history from origin (`fetch` + `reset --soft FETCH_HEAD` + `reset`), rebuilt .venv, 91 passed again, committed+pushed `1d805a6`, restarted server, UI smoke re-passed.
- README Part-3 documentation written (spec tables + contradiction log + P6 decision).

# External Sources
- three.js v0.160.1 fetched from registry.npmjs.org tarball (raw.githubusercontent unreachable — allowlist), vendored with MIT license file.
- neuPrint unreachable (Part-2 note, still true); synthetic connectome fallback with spec populations.

# Errors & Dead Ends
- **Re-clone #3** (this segment): `.git` reset to fresh clone of `43ad8a3` + venv + /tmp wiped. Working tree survived. Recovery via origin fetch + soft reset. ALWAYS push immediately after commits.
- PEP 668 — always `.venv/bin/python -m pytest`.
- `curl | python - <<EOF` swallows stdin → use `-o file` + separate python.
- P6 with signed raw F10/F11 blocked issuance forever (see Current State) — fixed with user delegation; formula body unchanged.
- Node: `global.navigator` is getter-only in v22 → Object.defineProperty. jsdom-free DOM harness pattern (Proxy el/ctx stubs, vm.runInThisContext, no-op timers + fixture fetch router) works well — reuse it.
- history.replaceState / window.addEventListener must be stubbed in harness; `tHead`/`tBodies` needed on table stubs.
- Ring lacks slicing (deque) — `list(ring)[-n:]` only. Ring now iterable/indexable.
- edit_file fuzzy-match fails without verbatim context — use sed output as anchor.
- Scratch files in /tmp do not survive re-clones — keep critical harnesses content in this memory or the repo.

# Key Results
- **Tests: 91 passed** (4s boot engine, ~42s full suite). Push `1d805a6`.
- **Organic issuance (P6-fixed)**: first at GRC+186 on trend feed, BNB tf2 conf=0.350 dir=+1 SL=2421.67 TP=2474.79; pre-fix plateau 0.177 at 240 GRCs. Issuance requires coherence: |MSRC| ≳ 0.7 AND strong base (chop → NO SIGNAL by design).
- **Live (sim feed)**: boot → live in ~6 s; GRC1: BTC tf1 drpt 0.665 (msrc 0.327), ufp 0.069; corelation edges need ≥3 GRC passes (21 edges at GRC3 — all pairs).
- **P1**: probs sum exactly 1.0 (21 levels); **UFP alpha** sums exactly 1.0 at 6-dp (largest-remainder absorption).
- **CCI 31 keys** (F1–F19,F22–F27,F31 + 97..101); prior weights: signals 0.92/30, HF=0.08.
- **Rings**: fspark1–31 (29 fills on first issued DLP), emospark1–22, pspark1–22, facc21/eacc22/pacc1–22 score only when move≠0.
- **Geometry payload**: units 7700, edges 53304, 8 regions (PAM cluster, MB, KC, CC, FSB, EB, LH, DN), news_units 10 (GEN ids ≥ 4814), activations 4×7700 u8 b64 (~10.3 KB each).
- **News**: BTC seed aggregate ≈ 0.054; category table regulatory 0.25/247 … geopolitical 0.07/247; TF impact = score×min(1, TFmin/D); TF4=1.0 only if D≥247.
- **QDPC τ** data-dependent (live GRC1: 1.2e6 min — still effectively no decay; documented).
- DOM smoke recipe: `node /tmp/domharness.js` (fixtures from live endpoints; asserts 31/22/22 panels, news tab, corel stats).
- Probe recipe: `/tmp/issuance_probe.py N` (make_engine → _boot → loop feed.tick(3)+on_grc; watches cycles issued).
