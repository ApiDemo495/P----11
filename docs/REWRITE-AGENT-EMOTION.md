# Rewrite: agent runtime and affect model

This note records what was measured broken on the merged tree, what was
replaced, and how the replacement is verified. It is the audit trail for the
`app/agents.py` and `app/emotion.py` rewrites.

## 1. Why the agent never downloaded

Measured on a fresh sandbox (`data/setup_ai.log`, first boot):

```
installing prebuilt llama.cpp CPU wheel (abetlen index, --only-binary: no source build) …
Could not fetch URL https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/
  There was a problem confirming the ssl certificate: TLS/SSL connection has been closed
ERROR: Could not find a version that satisfies the requirement llama-cpp-python>=0.2.90
prebuilt wheel index unavailable — retrying from PyPI (a source build can take 10-20 min)
Downloading llama_cpp_python-0.3.37.tar.gz (80.9 MB)
Building wheel for llama-cpp-python (pyproject.toml): started
Building wheel for llama-cpp-python (pyproject.toml): still running...   ← 20+ minutes
```

Three independent faults, all fatal on a restricted host:

1. **The fallback could never succeed.** PyPI publishes `llama-cpp-python` as
   **sdist only** (verified against `https://pypi.org/pypi/llama-cpp-python/json`:
   every release, including 0.3.37, has `packagetypes == {'sdist'}`). The
   "retry from PyPI" path therefore *always* compiles llama.cpp from source,
   which needs a linker + 20 minutes and cannot complete inside the boot
   budget.
2. **The wheel index is a single point of failure.** `abetlen.github.io` is a
   GitHub Pages host. Wherever that TLS handshake is blocked, the only
   prebuilt source is gone, and the installer has nothing else to try.
3. **A failed runtime install killed the model download too.** Step 1 raised,
   so step 2 (`download_models.py`) never ran — the model and the runtime were
   needlessly coupled. Neither is required for the app to trade.

## 2. Why the agent "is not happening" in the UI

* `AgentInstaller.status()` reported `state: "installing"` with
  `phase: "runtime"` for the entire 20-minute compile, and
  `/api/agents/status` never carried an `installer` key at all — the Agents
  tab read `installer: null` and had nothing to render.
* `analysis.ai` only ever consulted Gemini keys and `llama_cpp`, so with no
  key and no compiled runtime every analysis printed
  `no AI configured — deterministic pipeline only`. Even a working
  `ollama serve` on 11434 was ignored.
* A failed signal was silently reused and then dropped to neutral; the reason
  never reached the panel.

## 3. What replaced it

### Agent runtime (`app/agents.py`)

* **Runtime ladder, best-effort, never fatal.** The installer now walks a list
  of *backends*, each marked `required=False`, and stops at the first that
  works:
  1. `llama_cpp` if already importable (0 bytes fetched);
  2. prebuilt CPU wheel from the abetlen index (`--only-binary=:all:`);
  3. a *reachable* Python source build, capped at 8 minutes and only when a
     compiler is present (never silently 20-minute compile);
  4. **Ollama** — probed on `127.0.0.1:11434`, started if the binary exists,
     and used over its OpenAI-compatible `/v1/chat/completions` endpoint. This
     is the path that does not need a compiler or the wheel index at all.
  The model download is a **separate step**: if the runtime cannot be
  installed the model still downloads, so a later fix is instant.
* **Honest phase reporting.** `runtime → model → load`, with a `backend`
  field naming what won, a `why` for what did not, and a per-step log. A
  download in progress reports elapsed seconds and the bytes on disk; it no
  longer sits at "installing" with no number.
* **Nothing blocks the app.** Every step is a daemon thread, guarded by the
  same flock the button uses, with bounded retries.

### Agent sources (`app/agents.py::AgentManager`)

* One `AgentRuntime` abstraction for *every* provider, so the local model, the
  cloud keys, Groq and Ollama share one retry/cooldown/score path.
* The local slot resolves in this order: `llama_cpp` GGUF → Ollama HTTP. The
  analysis `ai_review` stage now uses the same resolver, so a review happens
  as soon as *anything* is available, and says which backend answered.
* Every failure carries a plain-word `error` into the status payload.

### Affect model (`app/emotion.py`)

Rebuilt as a measurement, not a formula sheet. Each of the 21 numbered
channels keeps its id (the CCI weights, the reward rule and the UI grid are
keyed by it) but is now derived from a named, measurable statistic:

| # | construct | measurement (published basis) |
|---|---|---|
| 1 | valence | asinh-compressed drift t-stat (Russell 1980) |
| 2 | arousal | short/long realised-vol ratio (Russell 1980) |
| 3 | dominance | move persistence + drift magnitude (Mehrabian & Russell 1974) |
| 4/5 | PA / NA | upside/downside semi-deviation, both activated (PANAS 1988) |
| 6–11 | basic emotions | Gaussian distance to Ekman (1992) prototypes in (v,a,d) |
| 12 | loss pressure | drawdown in σ, λ = 2.25 (Tversky & Kahneman 1992) |
| 13 | somatic marker | EWMA of this system's own called-right rate (Damasio 1994) |
| 14 | herding | CSAD of peer returns (Christie & Huang 1995) |
| 15 | capitulation | NA + low control + volume climax (Frijda 1986) |
| 16 | extrapolation | 5-bar drift over its own dispersion (Barberis et al. 2015) |
| 17 | volatility stress | realised-vol expansion (CBOE VIX concept) |
| 18 | momentum vs mean | log gap to MA30 in standard errors |
| 19 | breadth | share of peers on the same side (F&G dominance slot, labelled) |
| 20 | news sentiment | measured news pipe; abstains when there are no headlines |
| 21 | exhaustion | 1st vs 2nd half displacement decay |

Hard rules enforced and tested:

* **Every channel reports its own neutral and its own citation**, and the
  dominant channel is ranked by deviation from that neutral — never by an
  absolute value that rewards saturation.
* **No channel is allowed to pin at 1.000.** Compression is `tanh`/`asinh`
  based with graded gains; a saturated channel has stopped carrying
  information and used to win the crown by saturation.
* **Abstention is stated, not faked.** Missing headlines, a missing peer
  cross-section, or no settled outcomes each produce a `zero_reasons` entry
  with a cause bucket instead of a middle-of-the-road number.
* **The Fear & Greed index only claims what it measures.** alternative.me's
  volatility (25 %) and momentum/volume (25 %) components are computed; the
  published dominance/social/surveys/trends weights are declared unmeasured
  and the index is renormalised over the measured 50 %.
* **Determinism.** Same tape in → same affect out; the reward update is the
  only stateful path and it needs a settled outcome.

## 4. Verification

`tests/test_agent_download.py`, `tests/test_agent_runtime.py` and
`tests/test_emotion_model.py` pin the behaviour above, including the exact
failure modes found in production:

* a reachable-but-broken wheel index does **not** trigger a source build;
* a missing runtime does **not** block the model download;
* Ollama answering on 11434 makes the local slot available without
  `llama_cpp`;
* status is never `installing` without a phase, and never `null`.

## 5. Post-rewrite fixes found by driving the running app

These were found only after booting the merged tree against a live tape and
reading the panel, which is why they are recorded separately from the design.

### 5.1 Two emotion bugs (found by probe, then by reading the dashboard)

**The drift t-statistic divided two different windows.** The long-window
return was divided by a standard error computed on the *session* window, so a
channel that was flat for 30 bars but had drifted once long ago reported an
enormous "momentum vs mean" t. `r_w` and `se` now both come from the
timeframe window (`cl`); `session_ret` is kept only for the `r30` field, a
separate `window_return` field was added, and `ma30`/`mom_z` use `_mwin =
cl if len(cl) >= 10 else mc`.

**A dead channel voted at full strength.** Deviation from neutral was used
unconditionally, so an *event* channel whose neutral is 0.0 and whose reading
was 0.0005 — "fear is not happening" — was ranked as the single most extreme
signal on the tape and dragged CERC negative. `MOVE_BAND = 0.10` now gates
the vote: a channel inside 0.10 of its own neutral contributes no direction.
Both fixes are visible in `features.window_return` and in the unchanged
`cerc = Σ wᵢeᵢdᵢ` identity.

**The absence of a state must never be crowned.** After the band fix the live
panel still read `Trend exhaustion · 0.92` off a channel value of 0.04,
because 0.04 is far from a 0.0 neutral. Event channels are now eligible for
the crown only when switched on (`value > MOVE_BAND`); state channels
(neutral 0.5) stay eligible on both sides because both sides are states. The
`dominant` payload gained `side` (`high`/`low`/`neutral`) and `directional`,
and the panel prints "… % below neutral" plus a warning when no channel
carries a direction, so a low reading can never render as an extreme high.

### 5.2 Three agent/download faults (found by reading the live status payload)

* **The source build is opt-in.** `llama-cpp-python` is sdist-only on PyPI for
  every release (0.3.35/36/37), so a "fall back to PyPI" rung always compiled
  for 20+ minutes. The ladder is now prebuilt wheel → source build **only
  with `PREDICT_AGENT_SOURCE_BUILD=1`** (budget `SOURCE_BUILD_BUDGET_S=900`)
  → Ollama. Where a toolchain exists the source build does finish (it
  completed here in ~20 min); where it does not, the skip reason says so.
* **`download_models.py` exited 0 after a failed download**, and the installer
  logged `download_models exit 0`, which reads like success. The exit code is
  now never the evidence: the installer re-validates the file on disk and
  reports the real cause (unreachable hub, or a truncated non-GGUF file). The
  script itself resolves the real filename via `list_repo_files` and names
  huggingface.co in its failure message.
* **The log truncated the *front* of long failures** (`line[-300:]`), so the
  panel showed lines beginning mid-word, e.g. `appened while trying to
  locate…` with the actual reason gone. `_say()` keeps the head and the tail
  and elides the middle; `phase` now survives the end of a run (it reported
  `""` when the run failed, hiding *which* step failed), and the compiler
  probe is reported even when a runtime exists, because it is diagnostic.

### 5.3 UI contract fixes

`static/app.js` read `a.local.install`, but the installer is exposed as a
top-level `a.installer`, so the whole panel rendered from `undefined`. That is
fixed, and the panel now renders the ladder step-by-step, the Ollama probe, the
compiler probe, bytes-on-disk vs expected size, and a **re-probe runtime**
button wired to `POST /api/agents/local/retry` (which re-probes without
re-downloading). The `ai_review` stage prints `available/backend/note/reviews/
correction` instead of the old blanket "keyless" text, and a partial Fear &
Greed index says which components are missing rather than presenting a
50 %-measured number as the published index.

### 5.4 Suite

`667 passed, 1 skipped` on the merged tree.

## 6. The two faults behind "what happened" (screenshot, 2026-11-19)

The board showed `ENGINE UNREACHABLE — retrying (start it: bash run.sh)`,
`NO CALL / FLAT · conf 0.000`, `waiting for the first lock…` and four TF cards
reading `AI — not configured`. The server was **running the whole time**. Two
independent bugs produced that picture, and both were only visible in
`data/server.log` under load.

### 6.1 `IndexError` discarded a finished analysis (the "analysis is not happening")

```
File "app/engine.py", line 2991, in _analyse_body
    f"TP1 {tps[0]['level']} · SL {final['sl']} · "
IndexError: list index out of range
```

`tps` is only filled by the preview block, and that block is guarded by
`if direction != 0 and price > 0`. A FLAT call therefore has **no rungs** —
legitimately. But the *final stage summary string* indexed `tps[0]`
unconditionally, the `except` wrapped the whole finale, and so a result that
had already been computed was replaced by `job["status"] = "error"`. The board
said "analysis is not happening" while the engine was finishing and throwing
its work away. The summary now degrades (`TP1 — (no levels: flat call)`,
`SL —`) instead of raising: **a log line may never be able to destroy a
result**. Reproduced in `test_flat_analysis_survives_having_no_levels`
(flat + no locked cycle is the exact state in the screenshot; the test fails
with the same `IndexError` on the old code).

### 6.2 The clock watchdog was self-defeating (the frozen board)

```
WARNING clock: clock watchdog: restarting scheduler (alive=True silent=649.7s)
WARNING clock: clock watchdog: restarting scheduler (alive=True silent=655.2s)   ← every 5 s, for ever
```

`watchdog()` measured silence from `last_fired` **only**. A restart created a
new task but did not touch that timestamp, so the very next pass (the lag
meter calls it every few seconds) declared the fresh task silent and cancelled
it — up to 240 s before its next boundary could ever arrive. One missed GRC
became a permanent outage, which is why "stale" panels never refreshed. The
baseline is now `max(last_fired, started_at)` and the restart sets a fresh
`started_at`, so the watchdog converges. `tests/test_clock_watchdog.py` pins
both halves: a restart must be followed by `healthy` on the next pass, and a
genuinely old clock must still be repaired.

### 6.3 The banner lied about *why* (the red bar)

`api()` aborts every request after **10 s** and the abort fell through to the
offline counter, so a busy engine (warm replay, GRC pass, model load — all
legitimately minutes long) displayed "ENGINE UNREACHABLE … start it", telling
the user to start a process that was running. Now:

* a timeout is reported as `ENGINE BUSY — a long pass is running … standby,
  do not restart` and no longer counts toward "unreachable";
* the banner is sticky for 6 s once shown (`_bannerSince`), because some
  requests succeeded while others timed out and removing it on the first
  success made it strobe;
* it carries a ✕ so a `position:fixed` bar covering the header is dismissible.

`AI — not configured` on the TF cards is unrelated to all of the above: no
cloud key is set and no local model is loadable on this host, and the
`ai_review` stage says exactly that (`no AI answered — deterministic pipeline
only (no cloud key; local agent not loaded …)`).

Verified under load through the real launch path (`bash run.sh`): four assets
analysed, `0` analysis failures, `0` `IndexError`, `0` clock restarts.
