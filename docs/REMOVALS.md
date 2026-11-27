# REMOVALS — dead code and dead assets

Records what was deleted in the "remove useless things" pass, and the evidence
that made each removal safe. Kept so a future reader does not re-add a file
speculatively, and does not re-delete something that only looks dead.

Every removal below was proven unreachable by one of: no `<script>` tag in
`static/index.html`, no import in `app/`, no call site anywhere in `app/`,
`tests/`, `scripts/`, or no read of the file at runtime. Anything that was
merely *unreferenced* but is documented, cited in results, or covered by a test
was kept and is listed at the bottom.

## Assets (2.66 MB)

| path | bytes | evidence |
|---|---|---|
| `static/brains3d.js` | 11,306 | never loaded; its DOM ids `brains-canvas`, `region-labels`, `brains-fallback` do not exist in `index.html`. The live brains view (`loadBrains()`) only calls `/api/brain-geometry`, `/api/systems`, `/api/brain/status`. |
| `static/vendor/three.module.js` | 1,272,972 | only imported by `brains3d.js`; `THREE` appears nowhere else. |
| `static/vendor/katex.min.js` | 275,414 | no `<script>` tag, no CSS `@import`. |
| `static/vendor/katex.min.css` | 23,335 | as above. |
| `static/vendor/fonts/` | 1,080,668 | 19 KaTeX font faces; only `katex.min.css` referenced them. |
| `static/vendor/THREE_LICENSE` | 1,081 | licence for the removed `three.module.js`. |

`static/` went from 3.3 MB to 632 KB. `vendor/` now holds only
`lightweight-charts.standalone.production.js`, which `index.html` loads.

## Python (19 functions, 103 lines)

Each had no call site and no decorator (so not a FastAPI route). The AST
scanner used excludes any function with a non-empty `decorator_list`, because
route handlers are reached by decorator rather than by name.

| module | function | lines |
|---|---|---|
| `metacognition.py` | `record_settled` | 15 |
| `metacognition.py` | `apply_audit` | 6 |
| `emotion.py` | `_neg_frac` | 9 |
| `emotion.py` | `_lag_value` | 7 |
| `history.py` | `sl_hit_ratio_recent` | 7 |
| `history.py` | `p_hit` | 6 |
| `brain.py` | `on_grc_outcome` | 7 |
| `brain.py` | `cluster_mean` | 3 |
| `news.py` | `arrival_scores_last` | 7 |
| `news.py` | `agrees_with` | 6 |
| `news.py` | `set_tape_accel` | 3 |
| `formulas/f36_f44.py` | `_minute_series` | 7 |
| `sync_clock.py` | `begin_pass` | 5 |
| `sync_clock.py` | `note_pass` | 2 |
| `bgn.py` | `adjust_vector` | 4 |
| `learning.py` | `_norm_pdf` | 3 |
| `learning.py` | `rates_for` | 2 |
| `agents.py` | `boot_state` | 2 |
| `rlsys.py` | `_mean_reward` | 2 |

`record_settled` deserves a note. Its docstring claimed to be "the ONLY entry
point that moves calibration", but nothing called it: calibration is driven
inline in `engine.py` (`self._calib_pending`, the `calib_p` / `calib_h` rings,
and the lock-learning pass). A docstring asserting sole ownership of a
mechanism it does not own is worse than dead code — it misdirects anyone
debugging calibration. Removed rather than reworded.

## Unused imports (34)

Removed with an AST rewriter so multi-line `from x import (...)` blocks were
edited correctly. Two candidates were **kept** because they are re-exported and
used as attributes elsewhere: `physics.PHYSICS_W_MAX` / `PHYSICS_W_MIN`.

## Dead CSS (17 rules)

Removed from `static/style.css` at rule level, stripping only the dead selector
from grouped rules rather than deleting the whole group:

- `.tf-bar`, `.tf-bar .lbl`, `.tf-btn`, `.tf-btn b`, `.tf-btn span`,
  `.tf-btn.active` — the old timeframe bar. `tests/test_ui_contract.py` asserts
  `"tf-bar" not in INDEX_HTML`, i.e. the markup was already gone.
- `.q-spark`, `.an-an-stages`, `.blk-matrix-wrap`, `.section-h`, `.blk-h`
  and `.q-card` — classes no element carries.
- `.em-tag.DECAYING` — `app.js` can only print `NEUTRAL`, `ELEVATED`, or
  `EXTREME`, so `DECAYING` was unreachable.

### A CSS defect found and fixed on the way

`.em-tag` colouring was split across two declarations, and the one carrying
`.EXTREME { color: red }` also carried the dead `.DECAYING`. Removing the dead
class without care would have silently dropped red from `EXTREME`. It was
replaced with the full set the JS can actually emit:

```css
.em-tag.NEUTRAL  { color: var(--dim); }
.em-tag.ELEVATED { color: var(--gold); }
.em-tag.EXTREME  { color: var(--red); }
```

`NEUTRAL` is now stated explicitly rather than inherited. That matters because
"ELEVATED" previously rendered in the same colour as "NEUTRAL" — the tag text
said one thing and the colour implied another.

- `#mode-chip.mode-drosophila|local|cloud` and
  `.mode-banner.mode-drosophila|local|cloud` — **kept**. `app.js` 4396 sets
  `banner.className = "mode-banner mode-" + mode`, so the element carries both
  classes and these compound selectors are live. `#mode-chip.mode-*` was
  removed instead: the chip only ever gets `textContent`, never a mode class.

A grouped-selector pass was reverted once: an early attempt deleted
`.panel-h, .tbl th, .tab-title, .why-grid h5, .why-grid h6, .q-foot, .seg-lbl,
.day-row.head, #rail-leds, h3, h5, h6` wholesale because one selector in the
group was dead. `tests/test_ux_sync.py` caught it. The final pass edits groups
in place and leaves live selectors untouched.

## Dead markup

- `#rs-win` in `static/index.html` — the rail row labelled "WIN/RUN". It had no
  writer in `app.js` and no CSS rule, so it displayed a permanent `—` in the
  live rail. Its siblings `#rs-grc`, `#rs-next`, `#rs-phase` are all written at
  `app.js` 2714–2717; `#rs-win` was removed rather than left as a placeholder.

## Kept deliberately — do not re-delete

| path | why it stays |
|---|---|
| `/api/ops`, `/api/override/{symbol}/{tf}` | documented in `README.md` (~666, 536) and `docs.html` (~683); `ops_status()` also called at `app/main.py` 954, 972 |
| `assets/connectome/flywire_783.npz.gz` (58.5 MB) | real offline fallback loaded by `app/flywire.py` 495, guarded by `tests/test_brain_bundle_charts.py` |
| `app/orderflow.py` | imported as `from . import hedgefund, orderflow` (`app/engine.py` 140) — a `^import` grep misses this form |
| `probe_v10_train.py` | cited in `backtest/REAL_RESULTS.md` (~1584) |
| `scripts/fetch_oos_tapes.py` | writes `backtest/cache/{SYM}_1m.csv` and `PAXG_1h.csv`, which `backtest/alpha_lab.py` reads |
| `backtest/results/{hold_wr_v10_rejected,hold_wr_v9_rejected,trend_gate_audit}.json` | research audit records cited in results |
| `docs/*.md` (16 files) | records, not code |
| `.mode-banner.mode-*`, `.em-tag.EXTREME` (rewritten) | compound selectors on live elements |

## Verification

- `pytest tests -q -p no:randomly` — full suite green after removal.
- `node --check static/app.js`; CSS braces balanced.
- Live: `/`, `/style.css`, `/app.js` all 200; `POST /api/analyse/BTC?tf=tf1`
  returns the 8-stage pipeline; `POST /api/analyse/PAXG?tf=tf3` queues.
- `python3 scripts/make_docs_html.py` regenerates `docs.html` to the committed
  content, so the README edit and the generated docs agree.
