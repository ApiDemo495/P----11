# The number, the side, and the reason

Reported 2026-11-27, from four screenshots. Each complaint turned out to be
**two parts of the app describing different objects**.

---

## 1. `n/a` with no reason

PAXG's cells printed `n/a` for the win rate because they are ungradeable —
their own artifact says *"60m bars cannot resolve a 22-min window honestly"*.
The card had the reason in its payload and threw it away.

**Fixed.** `n/a` no longer exists on a cell. A cell without a certified rate
now says which of two different facts it is:

- `no rate — ungradeable` (with the full reason printed underneath), or
- `no rate — not measured`

---

## 2. "Why confidence is too low" — and the chain that disagreed by 20×

The card printed **confidence 0.021**. The CONFIDENCE CHAIN panel, regenerated
from the same inputs, printed **total 0.421**. Twenty times apart.

### The cause

`asset_behavior.temper()` returns the confidence it was given, unchanged, for
any asset with no tempering rule — and **BTC has no rule**. The engine did:

```python
_beh = _temper(symbol, pack["cci_adjusted"], fin["confidence"], closes)
fin["confidence"] = _beh["confidence"]      # ← assigns the input back
```

so the calibrated number was overwritten by whatever `fin["confidence"]`
happened to hold at that moment — a stale 0.02 — for exactly the one asset
with nothing to say. Every panel downstream then read 0.02.

**Fixed.** The calibrated value is captured before `temper()` runs, and the
engine only adopts a confidence that `temper()` actually **changed** (and
states a factor for). `calibrate_confidence` is now the single owner of this
number.

### The audit trail

The user's follow-up — *"you can do anything to increase confidence"* — needs
the app to answer **which stage moved the number**, so the engine now records
one:

```json
"conf_trace": [
  {"stage": "calibrated chain",   "before": null,   "after": 0.3385,
   "why": "SNR→tanh base × bounded modulators"},
  {"stage": "advanced layer ×",   "before": 0.3385, "after": 0.3236,
   "why": "soft_gate × RQA = 0.956"}
]
```

Every stage that moves the number appends an entry, live, and the CONFIDENCE
CHAIN panel renders it verbatim above the eight spec steps.

### The honest answer about "too low"

A confidence at 0.02 is **not a broken formula**. `|CCI_ADJ|` is compared to
that cell's own measured noise floor (0.020) and mapped through `tanh`; a fused
score of `0.0225` against a floor of `0.020` is an SNR of 1.1 — *no edge this
window*. The same formula returns **0.65** at 2.5 noise units and **0.80** at
7.5. Suppressing the floor to manufacture confident-looking numbers would be
the failure mode the chain's own docstring was written to stop.

What the app owes the reader is which of the two situations they are in — and
it now says so:

> **confidence is at the floor** — the fused score sits inside this cell's own
> noise floor (snr 0.00), so the calibrated base is ~0.02 — the chain is
> working, there is no edge to report in this window.

Measured on the live board after the fix: PAXG tf4 **0.6964**, BNB tf4
**0.3236** against a 0.3657 bar — real numbers from the same formula, where
before every lane was pinned at 0.02.

---

## 3. "Chain total and published total" — same defect

Covered by §2. The panel also now prints, above the trace:

```
published 0.3236 · floor 0.366
```

so the number and the bar it is being measured against are never separate
facts on the card.

---

## 4. "Told me to buy, gave me TP and SL — but entry was above TP"

Three different ways this could happen; two were real defects.

### a) The side and the note described different trades

The card read:

```
signal:   BUY
issue_note: "…side FLIPPED to SELL"
```

`verdict.resolve()` is the one place the trend interaction resolves a side —
it can **FLIP** a chase — and it returns four values. The engine took three
and discarded `side_out`, then published `res["direction"]`, which no longer
carried the flip because the double-check pass had overwritten it afterwards.

**Fixed.** The side is resolved **once**, *before any level is built* (a
flipped side with the old levels is a BUY target on a SELL trade), and the
published `direction` / `signal` are overwritten with it. The pre-flip side
stays visible as `direction_cci` for the divergence note.

### b) `risk_block` was already correct

Verified across both sides: TP always ahead of entry, SL always behind, rr
≥ the floor. Re-anchoring moves the anchor and keeps the bps distances, so a
re-lock cannot invert a pair either.

### c) The seal refuses a pair the market has already passed

`predlock._barrier_behind_market` refuses a BUY whose target is at/below spot
and a SELL whose target is at/above it, by name and with the distance, and the
seal path re-anchors on the live price and seals again. Pinned by test.

---

## What I did not do

**I did not raise `ISSUE_BAR` to make more calls issue.** The confidence floor
is 1.1 × `CONFIDENCE_FLOOR` (0.385 for BTC), and the reason cards sit below it
is that the fused score is inside the noise floor — a real absence of edge.
Lowering the bar would manufacture the appearance of conviction without
changing the outcome, which is the exact class of bug this project keeps
having to remove.

**The reward:risk finding from the previous segment still stands.** All 12
graded cells are certified at 0.27:1 and wear
`GEOMETRY BELOW 1:1 · NOT TRADEABLE`; live locks are built on a repaired 1:1
whose hit rate has **not** been measured. No win rate is quoted at r ≥ 1.0
because none exists yet.

---

## Pinned by

`tests/test_number_side_and_reason.py` — 14 tests: the confidence ownership
rule, the trace's existence and behaviour, the "floor is honest" distinction,
`n/a` carrying its reason, the side/note agreement, side-before-levels
ordering, pair orientation on both sides, and the seal's barrier refusal.

Full suite: **1227 passed, 3 skipped**.
