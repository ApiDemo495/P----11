# Cognitive thinking + metacognition

The engine already computed a great deal and *reported* it. Reporting is not
thinking: no panel said how the app got from evidence to a call, what it
considered and rejected, what it assumed, or how much it should trust itself.
`app/cognition.py` adds those four things and the `Ψ MIND` tab shows them.

## The four parts

**1 · Working memory (`working_memory`)** — what actually entered this
window's reasoning, and what was withheld, each with its reason. A fact that
never reached the decision cannot be cited as support for it. Channel-level
abstentions from the emotion model (`zero_reasons`) are carried through
verbatim, so "Fear — prototype 2.40 away in (v,a,d)" appears as a stated
abstention rather than a silent 0.0.

**2 · Deliberation (`deliberation`)** — two independent routes to a side:

| route | statistic | why it is separate |
|---|---|---|
| evidence | fused mass: `Σ supporters − opponents` (what the confidence is built from) | weights the contributions |
| votes | `(agreeing − disagreeing) / voters` | ignores how strongly each voted |

They can disagree, and when they do the trace says so instead of printing one
number. The outcome that was considered and **rejected** is recorded with the
arithmetic that rejected it. Conflict is measured as **sign disagreement**, not
distance — an earlier version scored a +0.023 net against a −0.83 head-count
as 0.79 "conflict" purely because the two routes live on different scales.

**3 · Assumptions (`assumptions`)** — every premise the call depends on, as a
testable claim with a verdict (`holds` / `does not hold` / `unverifiable`).
An unstated assumption is how a model lies without ever printing a wrong
number. Examples that fire today: *"the prices are real market prints"* (fails
on a simulated tape), *"the Fear & Greed composite is fully measured"* (fails
at 50 % coverage), *"no two sources are secretly the same source"* (holds, but
states that correlation above ρ≈0.55 voids the independence).

**4 · Metacognition (`competence`, `abstain`)** — what the app knows about its
own knowing:

* hit-rate over graded outcomes, with a **Wilson 95 % lower bound** (60 % on
  five rows is not 60 % competence);
* **Brier decomposition** (Murphy 1973) into reliability and resolution, which
  fail differently — a model can be perfectly calibrated and useless;
* a **dangling certainty** = claimed confidence − measured hit-rate;
* `trust`: `provisional` (under 30 graded), `calibrated`, `overconfident`,
  `underconfident`;
* **abstention** whenever the certainty is unearned — but never on a small
  sample, because "I have no evidence" is not evidence against.

## It learns from its own thinking

Every analysis stores a record in `data/thoughts.json`
(`_think_and_record`): signature, confidence, signal, whether it abstained,
whether its routes split, whether the trace contradicted itself. When the
window settles, `_grade_last_thought` marks the record win/loss. That is what
makes `thought_stats()` a measure of **reasoning**, not bookkeeping — and it
is why the competence number starts as `unknown` instead of a flattering 0.5.

## Reproducibility

The trace carries a `signature`: a sha256 over each step's name, status,
detail and citations. Timestamps are excluded, so identical evidence yields an
identical signature and two runs can be diffed. `self_check()` then verifies
the trace against itself — step order, no empty step, sources present unless
the app abstained, a split that is actually reported as a conflict, and no
competence claim without a hit-rate.

## The narrator cannot be an analyst

`narrate=1` asks the local model (or a cloud slot) to re-tell the trace.
`guard_narration()` then checks every number it printed against the trace and
requires an abstention to be mentioned. An invented figure throws the model's
text away and the deterministic narration is published instead — a narrator
that invents a number is worse than no narrator.

## Endpoints

| route | purpose |
|---|---|
| `GET /api/think?symbol=&tf=&narrate=` | the full trace (6 steps, assumptions, falsifier, competence, self-check) |
| `GET /api/think/history?symbol=&tf=` | the thought ledger and the reasoning hit-rate |

`POST /api/analyse` now returns a `thinking` summary and records the trace;
`GET /api/snapshot` carries it per window.

## Tests

`tests/test_cognition.py` — 29 tests. Each design claim is falsifiable: ordering
and non-empty steps, the self-check catching a trace it is given deliberately
broken, identical-evidence signatures, withheld inputs naming a reason,
sign-vs-magnitude conflict, an unearned confidence forcing abstention, small
samples never vetoing, Brier separating the two failure modes, the narration
guard rejecting an invented figure and a hidden abstention, a bare payload
producing an honest trace, and the pipeline recording plus grading its own
thoughts.
