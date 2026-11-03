# BRAIN COMPONENT — report (brain.md §12 format)

Built: FlyWire v783 whole-brain connectome import, verification, odour
encoding, real propagation, valence readout, dopamine RPE learning, the two
brain formulas, UI, and the 10 acceptance tests (`tests/test_flywire_brain.py`,
10/10; full suite **221 passed**).

## Neuron / edge counts per build

| build | neurons | edges | MB sub-circuit | notes |
|---|---|---|---|---|
| real (v1, shipped cache) | **138,639** | **15,091,983** | 6,849 n / 627,045 e | matches brain.md's stated figures exactly |
| acceptance-test synthetic | 123 | 554 | 117 n | schema-faithful, built in-test |

Real build details: 685 ALPNs (input), 96 MBONs (52 ACh + 19 GABA approach,
25 Glu avoid — Aso 2014 semantics), 5,177 Kenyon cells, 331 DANs
(PAM01–15 / PPL101–08 / PPL201–04), 1,299 descending neurons, 556 LH
neurons. Artefact: `~/.cache/drosophila-trader/connectome/flywire_783.npz`
(192 MB, `BUILD_VERSION=1`, never committed).

**Build peak RSS:** the build ran in a daemon worker (B-1) with
column-at-a-time pyarrow reads (`Presynaptic_ID`, `Postsynaptic_ID`,
`Connectivity`, `Excitatory` → compact dtypes before concatenation); the
worker survived and the API never restarted — peak not sampled separately,
the process completed a 15.09 M-edge CSR build +202 MB npz write without
OOM on this box.

## Five-step verification table (live, durations included)

| step | outcome | duration |
|---|---|---|
| cache | ok — cached (v1) · 192 MB | 0.1 ms |
| neuprint reach | ok — no key (optional; FlyWire is primary) | — |
| neuprint auth | ok — no key | — |
| neuprint query | ok — skipped (no key) | — |
| flywire | ok — FlyWire v783 ready (138,639 neurons · 15,091,983 connections · MB 6,849) | 350.6 ms load |
| absent | connectome loaded — vote live | — |

Chip (live): `FlyWire v783 · 138,639 n · pass 57 ms · 1 s ago` (green).
Whole-brain pass measured **57.3 ms** on its own `flywire-pass` thread
(never the request thread — test asserts the thread name); MB pass
**5.4 ms** per vote (test budget20 ms). neuPrint key path uses
`GET /api/dbmeta/datasets`, distinguishes 401 (invalid key) from timeout
(unreachable), key held in process memory only (test greps the repo).

## Polarity map (which channels flipped)

All 30 channels probed alone at +1 on the real wiring: **flips: none
(all +1), dead channels: none** — every channel produces a non-zero
mirrored balance and a positive post-flip probe. The flip machinery is
live and unit-tested (synthetic circuit with an innately aversive channel
flips it); on this build the wiring needed no correction. Shown in the
UI's "Polarity map" panel and `/api/brain/status → flywire.polarity`.

## 24 h brain-only accuracy per horizon (with n)

Not yet reportable: the outcome ring `facc_brain` accumulates only at
window settlements (first windows need a full22/44/130/247-min cycle).
The ring, the horizon multiplier `clip(1+0.25·z, 0.25, 1.5)` and the
rollback path are implemented and tested; until n ≥ 8 the hindsight card
prints "not enough windows" — never a percentage (advice §0.5).

## Places where the brain's sign is modified outside the RPE rule

**None.** Audit of the code paths:

- `vote = ½(f(x) − f(−x))` — antisymmetric by construction, tested to
  1e-6 (real circuit measured |v₊+v₋| = 0.0 exactly).
- Polarity flips `s_i` are recorded by the calibration probe from the
  wiring itself (never hand-edited; persisted in the vault with
  BUILD_VERSION guard).
- Fusion multiplies by `w × vote` with the circuit's own sign; w = 0
  when absent.
- The only weight changes are `w += α·drg·pre·post` at window boundaries
  (dopamine RPE) and the hindsight re-weight / rollback.

## Adaptations taken (free-will decisions, with reasons)

1. **Column mapping**: philshiu's `Connectivity_783.parquet` ships
   `Excitatory` (±1) + `Connectivity` instead of `nt_type`; the sign
   semantics (ACh/GABA/DA/OA/5-HT +, Glu −) are encoded upstream in that
   flag. Documented in `app/flywire.py` header.
2. **`super_class` value is `descending`** (not `descending_neuron`);
   `top_nt` uses full names — mapped accordingly.
3. **N_INPUTS = 30**: this app's signed directional formulas (F1..F31
   minus F20 DRHE, the gauge) — one odour channel each, as brain.md
   instructs for "your app".
4. **Fusion weight0.25 start** (per brain.md), moved only by the
   per-horizon hindsight multiplier; applied like the agent multiplier
   (`CCI × (1 + w·vote·base_conf`) with the ladder re-run.
5. **Confidence from DN share**: `0.2 + 0.8·min(1, share /
   population-baseline)` — keeps the specified0.2–1.0 range while making
   typical sparse DN activity informative rather than pinned at0.2.
   Age-tagged (live: `pass.age_s`), never a cached boolean (zeros cause 11).
6. **CCSv2/KCAE are brain-channel outputs, not F32/F33**: the core
   registry is contractually 31 formulas (weightage, tests, UI tables);
   both formulas carry the same trace/`zero_reasons`/kind conventions
   (KCAE tagged `gauge` — zeros cause 7) and render on the brain card.
7. **Build under pytest is refused** (absence instead of a100 MB spawn);
   the live app builds for real — as proven by the completed cache.

## What was NOT taken (and why)

- Nothing from §1–§11 was dropped. The only softening: octopamine
  (`hsi`) gain has an empty target group on this build (no
  `top_nt=octopamine` neurons inside the MB sub-circuit), so the
  `×(1 + 0.3·hsi)` term is a no-op until such neurons are annotated —
  reported honestly rather than faked.

## Acceptance tests (brain.md §11) — all pass

1. synthetic build: CSR float32, row-normalised, signs follow the
   `Excitatory` column, BUILD_VERSION in the npz
2. five-step with durations; no data → chip `brain absent`, fusion
   weight 0, engine still picks from other sources
3. `2·N_INPUTS` contiguous ALPN groups, deterministic
4. flat input → vote exactly 0 (`flat input`); vote(−x) = −vote(x) ≤1e-6
5. polarity probe: every connected channel non-zero, flips recorded,
   post-flip probes positive
6. KC sparsity after hop 1 ≤ 5% + 1 (APL enforced every hop)
7. wins raise E, loss → negative drg, weights change only at
   boundaries, rollback restores
8. MB pass < 20 ms (5.4 ms), whole brain < 1 s (57 ms), thread name
   `flywire-pass` ≠ main thread
9. no token string in any repo file (regex lint)
10. absent → fusion weight exactly 0; ready → `0.25 × clip(1+0.25·z, …)`

## Remaining weak points (honest list)

1. `facc_brain` n is still 0 — the24 h accuracy table needs settled
   windows before it can print anything.
2. Octopamine gain target is empty on this build (see above).
3. Whole-brain confidence is driven by one global input vector (the
   BTC anchor's formula vector) — per-asset DNs share it; per-asset
   whole-brain passes would double the cost for marginal gain at 57 ms.
4. Rollback fires only when hindsight re-grades with z ≤ −1 and a
   lower multiplier — it has not yet occurred in live data (path tested
   in unit tests).
