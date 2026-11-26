# §39 — deep per-file audit: what was actually broken

The user asked for a *deeper, more precise* pass than §38: every file, every
line, minor errors included. This is the honest ledger of that pass — what
was measured, what was a real defect, and what was **not** a defect.

Reproduce everything here with:

```bash
.venv/bin/python scripts/deep_audit.py          # 85 files, AST + regex + probe
.venv/bin/python -m pytest tests/test_audit_silence.py -q
```

Suite state after this section: **1092 passed, 2 skipped in 916 s** (was 1066).

---

## 1. The measurement, before the fixes

`scripts/deep_audit.py` scans `app/` and `static/` three ways: a Python AST
pass, a regex pass over `static/app.js`, and a **probe** that calls every
public pure function with adversarial arguments.

```
files 85   functions probed 309   findings 2090
  HIGH  probe-nonfinite      144
  MED   silent-except        128
  MED   float-equality        24
  MED   missing-to-zero        8
  MED   empty-index            2
  MED   lookahead-convolve     1
  LOW   probe-raises        1310
  LOW   unguarded-division   403
  LOW   unguarded-tofixed     46
  LOW   unescaped-html        24
```

The harness itself had to be fixed five times before its output meant
anything — **every one of those bugs made the tool print a plausible clean
result instead of an error**, which is the same failure mode as the app bugs
it was written to find:

| harness bug | how it disguised itself |
|---|---|
| repo root missing from `sys.path` | `probed 0 functions` looked like "nothing to probe" |
| classes have no `__code__` | probe crashed rather than reported |
| coroutines not awaited | `RuntimeWarning`, not an error |
| the probe called `agents.download_gguf()` | spawned a real installer which **closed file descriptor 1**, so every later `print` died and the audit looked empty |
| prints died silently | needed an `os.dup(1)` and an explicit flush |

`download_gguf` is now blocked by a `_SIDE_EFFECT` name filter, pinned by
`test_deep_audit_never_probes_side_effecting_functions`.

---

## 2. Real defects found and fixed

### 2.1 The multiple-testing correction did nothing (HIGH)

`app/metacognition.py::deflated_sharpe` — the gate that decides whether a
cell is `CONFIRMED` or `SHADOW`.

```python
z1 = _norm_cdf(1 - 1.0 / n_trials)      # CDF, not a quantile
z2 = _norm_cdf(1 - 1.0 / (n_trials * math.e))
```

`_norm_cdf` returns a probability in (0, 1). The Bailey–López de Prado
expected-maximum-Sharpe term needs the **z at which P(Z < z) = 1 − 1/N**.
Feeding the CDF in meant that for every N from 10 to 6600 the expression
returned ~0.82–0.84, so `sr0 ≈ 0.019` regardless of how many configurations
were searched:

| trials | DSR before | DSR after |
|---|---|---|
| 1 | 1.0000 | 1.0000 |
| 10 | 1.0000 | 0.6596 |
| 100 | 1.0000 | 0.2962 |
| 1000 | 1.0000 | 0.1052 |
| 6600 | 1.0000 | 0.0400 |

*(SR = 0.20 per observation, n = 100; `DSR_CONFIRM = 0.90`.)*

A search of 6600 configurations scored exactly the same as one honest test.
That is the single most consequential bug of this audit: this repo's own
history includes a 3,168-configuration sweep, and this function was the
gate that was supposed to discount it.

Fixed by adding a real `_norm_ppf` (Acklam's rational approximation, unit
tested against known quantiles 1.959964 / 2.326348 / 3.090232 and round-
tripped through the CDF down to p = 1e−6).

### 2.2 Settings reported success without saving (MED)

`AgentManager.save_settings()` swallowed every `OSError` with `pass` and
returned `None`. On a read-only volume, a full disk or a bad path,
`configure()` still returned success: the dashboard showed the new value and
the server silently reverted to the old one on restart.

Reproduced by pointing `SETTINGS_FILE` at a directory — `open(..., "w")`
raises `IsADirectoryError`, the call returned normally:

```
save_settings() returned normally — no error signalled
SETTINGS_FILE still absent/unchanged -> the callee has no way to know it failed
```

Now returns `bool`, sets `self.last_save_error`, and the engine's
diagnostics payload carries `settings.save_error` so the UI can say
"live in memory only" instead of "saved".

### 2.3 Missing numbers rendered as measured zeros (MED × 8)

Same class as the §38 `pct(NaN) → "0.00%"` fix, in eight more places in
`static/app.js`. `|| 0` / `?? 0` written before `toFixed()`:

| site | before | after |
|---|---|---|
| `live_win_rate` missing | `win 0%` | `win —` |
| `research_pf` missing | `PF 0.00` | `PF —` |
| `age_hours` missing | `0.0h ago` | `—h ago` |
| `er_fast` / `vol_ratio` missing | `ER 0.00, vol×0.00` | `ER —, vol×—` |
| `era.totals.net` missing | `net 0.00%` | `net —` |
| `min_tp_bps` missing | `≥ 0bp` | `≥ —bp` |
| `regime_conf` missing | `conf 0.00` | `conf —` |

One of them was worse than cosmetic: the watchdog's jump column used

```js
color: (r.history_jump_bps || 0) > (r.jump_limit_bps || 300) ? red : green
```

so an **unknown** jump was painted green — "trusted". A watchdog must never
fail in the trusted direction. It now paints `--dim` for unknown.

All eight go through one helper, `num(v, d)`, added next to `fmt`:

```js
const num = (v, d = 2) => {
  if (v == null) return "—";
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "—";
};
```

*(A first attempt used `var(--fg-dim)`, a CSS variable that does not exist
in this stylesheet — caught immediately by grepping the tokens. `--dim`.)*

### 2.4 Best-effort learners that die silently (MED)

`Engine._settle_and_lock` wraps three learning calls in bare
`except Exception: pass`: `MetaCombiner.record` (calibration),
`Genesis.score_walk_forward`, `ForwardTables.record`. Deliberately
best-effort — a bad number must not take down settlement — but `pass` makes
"working" and "dead for three days" identical from the outside, so every
calibration readout can be a fossil.

New `app/silence.py`:

```python
with guard("settle.meta_learn"):
    self.meta.record(symbol, tf, prev["inputs"], up)
```

Still swallows (behaviour unchanged), but counts per site, keeps the first
exception message, and is exposed at `payload["silence"]`. A learner that
has failed 400 times now says so.

### 2.5 Look-ahead in the d08 category feature (MED)

`app/genesis.py::_d08_category` used `np.convolve(x, k, mode="same")`, which
is **centred**: sample `t` depends on `t+1`. Measured directly — perturb only
bar 3 of a 20-bar window and see whether bars 0–2 move:

```
prefix before the perturbed bar unchanged: True      (causal version)
                                              False  (centred version)
```

In production this is mitigated (`ctx.P` is a closed historical window, so
it never reads a bar from the future of a *lock*), and inside a difference
of two smoothed paths most of the bias cancels. But a walk-forward scorer
should not contain a term that would break under streaming, and the first
sample was being averaged against zero-padding that exists in no real
series. Now a trailing kernel with a renormalised short head at t=0,1.

---

## 3. Confirmed **not** bugs (checked, then dismissed)

Recording these matters as much as the defects: 2090 findings were reduced
to 7 real ones, and the user should be able to see why the other 2083 were
dismissed.

| class | n | verdict |
|---|---|---|
| `probe-raises` | 1310 | The probe calls functions with `None`, `-inf` and `10^12` as the **first three positional arguments**. `download_gguf(None, None, None)` raising `TypeError` is correct behaviour for a function whose arguments are a repo name and a destination path. Not a finding. |
| `probe-nonfinite` | 144 | Overwhelmingly the documented contract: `kyle_lambda`/`vpin`/`amihud`/`hawkes_branching` return `nan` when there is not enough history (`i < window`) or the denominator is degenerate (`sd <= 0`, `total <= 0`). Spot-checked against real tapes: given price actually driven by flow at a known coefficient of 0.02, `kyle_lambda` returned 0.00195 — the same order, sign correct. |
| `unguarded-division` | 403 | Every one sampled is guarded. `_dfa_hurst` returns at `n < 64` before `sum(r)/n`; the OLS slopes use `den = (...) or EPS`; `z = dev[-1]/sd_dev` follows `sd_dev = _std(dev) or EPS`; `half_life` is behind `if kappa > EPS`. |
| `float-equality` | 24 | Deliberate exact-zero tie rules, printed in the ledger's own `rule` string: `"strict sign · tie = loss"`. `_row_result` treats `move == 0.0` as a loss on purpose. Nothing here compares two computed floats for approximate equality. |
| `empty-index` | 2 | `app/agents.py:650,652` — `sorted(preferred, key=len)[0]`. `preferred` is only reached inside `if preferred:` and the second is `small or files` where `files` was already returned on empty. Exercised all four branches with a mocked listing: no `IndexError`. |
| `silent-except` | 127 of 128 | Deliberately best-effort I/O and telemetry (`_ram_gb`, `_kill`, log appends, ring buffers). The one that mattered is §2.2; the three learning ones are §2.4. |
| `clamp(inf, inf, inf) -> inf` | — | The probe's own false positive. `clamp(v, lo, hi)` returns `hi` only when `v > hi`; `inf > inf` is `False`, so returning `-inf` for `lo = -inf` is correct. |

---

## 4. What this section does **not** claim

- It does not claim the app is bug-free. 2083 findings were dismissed with
  reasons, but dismissal is a reading of the code, not a proof.
- It does not claim the strategy now beats the benchmark in §36's sense.
  Nothing here changes a signal; §2.5 changes a *feature* by ~0.07 (0.42 →
  0.35 on the same tape) and `deflated_sharpe` now *fails* things it used to
  pass. Both moves are toward fewer false confirmations, not toward a higher
  headline return.
- The `probe-raises` class is not evidence of missing input validation.
  Nothing in the live path calls these functions with `None`.
