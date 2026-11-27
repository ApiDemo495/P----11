# Why the board says what it says

Four questions about the screenshots, answered with the mechanism and the fix.

---

## 1. "It is showing bullish in a yellow box with impulse bear conf 0.20"

The two labels were computed by **two different code paths, and only one of
them compared them**.

`regime_conflict` — the record that says "this card's badge and this card's
regime chip disagree" — was set **only inside the ANALYSE path**
(`app/engine.py`, in the analyse result builder). Every card that comes from a
**GRC refresh** (which is most of them, every 4 minutes) got the chip from
`chopChip()` and the badge from `cyc.direction`, with nothing in between
comparing the two. So the card could print

> `▲ BULLISH*` … `📉 impulse bear · conf 0.20`

and there was no field anywhere for the page to notice.

**Fixed:**

- `PredictionEngine.regime_conflict(symbol, direction)` is now the **one**
  definition of the comparison. Both writers call it — ANALYSE and every GRC
  refresh (issued and not-issued branches).
- `TFCycle.regime_conflict` is a real field, serialised on the payload, so it
  reaches the page instead of dying in the engine.
- The hero badge **cannot** print two fighting labels. When they disagree it
  prints one sentence naming both facts:

  > `▲ BULLISH*` · `⚠ IMPULSE_BEAR (conf 0.20) OPPOSES THIS BUY · call stands, no order routed`

- The old version also printed `● WITHHELD` in the stats row while its own
  note said "the side stands and is graded" — the label contradicted the
  explanation underneath it. It now reads `▲ BULL` plus a `regime disagreement`
  row.
- A latent `ReferenceError` is gone: that row dereferenced `rawDir`, which was
  never defined anywhere in the file.

---

## 2. "Why asset triggers are unproven and shadow or none"

Those were **three different facts** printed as three unexplained words.

| Label | What it actually meant |
|---|---|
| `UNPROVEN` | a *shadow* sleeve — a candidate the edge scan **rejected**, kept for observation |
| `SHADOW · LONG` | the rejected candidate's live trigger **is firing right now**, and is still never traded |
| `NONE` | two possibilities: the scan tested this asset and everything failed, **or it never tested the asset at all** |

The board said none of that. Worse, it said `UNPROVEN` for a shadow sleeve
**without ever computing that sleeve's live reading**, so the row had no `z`
and no explanation.

**Fixed:**

- Shadow sleeves now get the same frozen-spec `z` the open sleeves use
  (`app/sleeves.py`), namespaced as `shadow:long|short|flat` so no surface can
  ever render it as a tradable signal. The tile prints
  `live z −2.41 / 2.5σ · below the trigger (shadow — never traded)`.
- Every shadow tile names **the gate it failed**, from the registry's own
  reasons:
  `failed: |t| 1.34 < 2.0 (correlation-adjusted); holdout Net EV −1.77 bps ≤ 0`
- `NONE` is split. The registry records which assets it actually covered
  (ADA, BASKET, BNB, BTC, ETH, GOLD, XRP — **821 candidates**). PAXG is
  outside that set, so it now reads:

  > **UNTESTED — PAXG** · *the edge scan (35 candidates across the covered
  > assets) never produced a candidate for PAXG, so there is nothing to
  > certify and nothing to reject* · *not a failure and not a live signal —
  > this asset was outside the scan's search space*

- A warming sleeve says **why** it is warming: certified signals require a
  real Binance tape, and simulated bars are excluded by design.

**What is actually true on your tape (sim):** ETH has two certified `open`
sleeves but their trigger needs ~129 h of continuous Binance 15m bars; the sim
feed cannot supply them, so the honest state is WARMING UP. BNB, BTC, BASKET,
ADA, XRP and GOLD have shadow candidates only. PAXG was never in the scan.

---

## 3. "Why paper execution is paused?"

Because it was **hard-coded**. `_step_portfolio` returned the ledger with

```python
snap["execution_status"] = "paused"
snap["execution_reason"] = "Paper execution paused: next-bar open and
    real-tape provenance are not yet verified; ..."
```

and the page printed the badge, never the reason. The reason was real: the old
live path filled at the **latest 1-minute close** while claiming a
next-15-minute-bar open, and could consume simulated tape — neither of which
the research ledger can reproduce.

**The two missing guarantees are now implemented** (`app/exec_live.py`), so
the freeze is lifted exactly as far as the tape supports it:

1. **Next-bar open.** A signal on bar *N* can only fill at the **open of bar
   N+1**, and only once that bar has *closed*. Never the close of the bar that
   produced the signal, and never early while the fill bar is still forming.
2. **Real-tape provenance.** Binance-sourced, minute-aligned, **gap-free**
   only. Any missing minute inside a bar voids that bar; simulated, unknown or
   mixed tape is refused outright.
3. Pessimistic fills — half the round trip on each side — the same convention
   `backtest/paper_replay.py` uses, so the paper record is comparable to the
   research ledger.
4. Exit is the open of the bar exactly `hold_bars` later, or the catastrophic
   stop if breached first.

The book now reports `live` when it can fill, and `paused` **with the failing
fact, per asset**, printed on the page (`#book-pause-why`):

> **why no fills:** ETH — feed mode is sim — simulated tape is excluded from
> certified fills · BTC — …

On a real Binance feed every one of those lines disappears and the page prints
the fill contract instead. Nothing is paused without naming what paused it.

---

## 4. "Why hold policy have those yellow boxes of gate fail, warmup and other issues"

The badges were correct and **never said why**. Now every one carries the
number that produced it (`hold_policy.tier_reason`, on the API payload, printed
under the badge on each cell card):

| Cell | Badge | Reason now printed |
|---|---|---|
| BTC tf1 | `ROWS VERIFIED · SHIP GATE FAIL` | the rows held up out-of-fold (hold-out row WR **73.2 %**, Wilson lo 72.5 %, holdout era net EV **+40.2 bp**), but the artifact fails its own pre-registered **ship gate (ii)**: it carries no REAL post-cost win rate for any cell, so its mean net WR cannot beat the **v8 reference 0.80 %**. The measurement is real; the ship claim is not earned |
| BTC tf4 | `IN-SAMPLE SHADOW · HOLDOUT MISS` | hold-out era net EV is **−29.3 bp** after friction (needs > 0) |
| PAXG tf1–tf3 | `UNGRADEABLE / WARMUP` | no configuration survives the cost floor on this cell's history — **60m bars cannot resolve a 22-minute window honestly** |
| BNB tf1 | `UNGRADEABLE / WARMUP` | no configuration survives the cost floor on this cell's history |

So the three yellow/amber badges are three different, **measured** statements,
not three ways of saying "something failed":

- **SHIP GATE FAIL** — 7 cells. The rows are genuinely good out-of-fold. The
  *claim* is not earned because the artifact never recorded a post-cost win
  rate, so it cannot beat the v8 reference.
- **HOLDOUT MISS** — 5 cells. Out-of-sample net EV is negative after friction.
- **WARMUP** — 4 cells. No certified configuration exists at all (PAXG's 1h
  history cannot resolve sub-130-minute windows; BNB tf1 has no feasible
  geometry above its round trip).

---

## Pinned by tests

`tests/test_board_explanations.py` — 24 tests: one definition of the conflict
used by both write paths, the field on the payload, the badge sentence, the
removed `rawDir` bug, shadow sleeves carrying `z` + the failed gate, the
untested-vs-failed split, the three asset-tile labels, the resampler dropping
forming bars and refusing gaps/mixed provenance, next-bar-open fill timing, the
simulated-tape refusal, the per-asset pause reason on the page, and a reason for
all 16 hold-policy cells.

```bash
.venv/bin/python -m pytest tests/test_board_explanations.py -q -p no:randomly
```
