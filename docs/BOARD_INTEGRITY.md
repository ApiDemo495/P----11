# Three board defects, and the code that now prevents them

You sent five screenshots of the live page and asked three questions. Each
answer below names the mechanism, the file that had it, and what the board
does now. Nothing here is an explanation in place of a fix.

---

## 1. "Why is there a big BEARISH sign when the analysed prediction is BUY?"

**The sign was right; the target on the same card was not.**

The card was a SELL (the trend rule had flipped BUY → SELL, and the card said
so: `TREND RULE (2 DEC) FLIPPED · BUY→SELL`). But the target printed beside it
came from the *forecast*, which had been computed before the flip — and the
hero card took that number **first, without checking which side of the entry
it fell on**:

```js
// static/app.js, before
const tgt = cyc?.forecast?.price ?? cyc?.tp ?? res.contract_tp ?? null;
```

Your card: short at 3,632, target **3,649 — above its own entry** — stop 3,651.
The page then divided the distance to that impossible target by the risk and
printed `TP:SL 0.87:1`. Three separate problems in one line:

1. a short aimed at a price above where it sold (unexecutable);
2. a displayed ratio below the 1:1 floor you set as absolute;
3. a "BEARISH" badge next to a bull-side number, which is what you saw.

**Now (`static/app.js`):** every candidate target is tested against the call's
own side before it can be used, in order — forecast target, cycle target,
contract TP, locked contract TP. The first one that is *actually ahead* of the
entry wins. Any that were skipped are **named on the card**
(`ignored forecast target 3,649 (wrong side)`). If nothing is ahead of the
entry, the card says so and claims no ratio:

> `NO TARGET ON THE SHORT SIDE — every published target sits on the wrong side
> of entry 3,632 (forecast target 3,649, cycle target 3,649); nothing can be
> executed, so no ratio is claimed`

**And the floor is enforced by widening the target, never by moving the stop**
(your absolute rule). A published 0.89:1 pair is shown as:

> `1.00:1 (forecast target 3,613 vs stop 3,651 from entry 3,632)` ·
> `0.89:1 as published — target widened to the 1:1 floor (3,613); stop left
> unchanged at 3,651`

The cell is **repaired, not demoted**: it still appears, and it still gets
graded.

---

## 2. "Why is a loss already in the prediction history the moment I press analyse?"

Because the lock was being graded against a market it was not sealed from.
Reproduced on the running board before anything was changed:

```
lock   entry 4923.241665   tp 4945.25446313   sl 4920.548755
live   price 4952.501167            <-- 59 bps away from the anchor
BUY target 4952.50 > 4945.25  -> already exceeded at seal time
BUY stop   4952.50 > 4920.55  -> already behind the market
```

The stop was **32 bps behind the live price**. Every barrier was passed before
the prediction existed, so the first completed bar "touched" the stop and the
history recorded `LOST · −3.8bp · SL touched on a closed bar` seconds after
the press. The −3.8 bp is exactly the stop distance — the trade never had a
chance to move.

There were **two** sources of it:

| Source | What happened | Fix |
|---|---|---|
| A stale anchor was accepted | the only anchor guard was `> 20 %` skew, so 59 bps sailed through | `app/engine.py` re-anchors anything past **2 bps** and records the re-anchor; `now_price` is handed to the seal |
| A straddling bar decided the lock | the tape stamps a bar at its *opening* minute, so a bar ending after the seal still holds pre-seal tape | `app/predlock.py` skips any bar whose own interval starts before `created_at` (`bar_s` is now passed as 60 s) |

A third, related bug: pressing ANALYSE twice in the same wall-clock window
handed back the **already-settled** row as "the prediction that does not move"
— which is how a LOST row appeared in the same instant as the press. A settled
row is now returned as labelled history (`status`, `settled: true`,
`locked: false`) and cannot be presented as a new prediction. The refusal text
also stopped shipping a literal `{W}`.

The seal now refuses a stale anchor **by name** (with the distance) instead of
writing an unearned result, and the caller re-measures the same call from the
live price and seals that.

---

## 3. "It is not learning from its mistakes and not detecting the in-app error"

The app now **checks its own records** and publishes what it finds.

`PredictionJournal.audit()` walks every sealed row and tests it against
invariants that need no outside truth:

- `ladder-on-wrong-side` — side, TP and SL are not one ladder (your first
  screenshot, as a machine check);
- `unearned-settlement` — a result recorded less than one bar after the seal;
- `settled-before-sealed` — a settlement timestamp before the seal;
- `graded-at-a-barrier-behind` — a win recorded at a price behind the entry,
  or a loss recorded at a price ahead of it.

It is served two ways:

- `GET /api/predictions/audit` → `{ok, issues, checked, rule}`
- `snapshot.prediction_audit` (rebuilt at most every 5 s, sub-millisecond scan
  of at most 400 rows)

and rendered on the History tab above the table (`#pred-audit`). A clean
journal prints one dim line; a broken one prints each row, by name. The audit
**never raises**: a self-check that dies is the defect it exists to catch, so
its own failure is reported as an error too.

Verified on the live board: `{"ok": true, "issues": [], "checked": 2}`.

---

## What is pinned by tests

`tests/test_board_integrity.py` (17 tests) — wrong-side target selection and
fallback order, the ignored-target label, the 1:1 repair (widening, stop
unchanged), stale-anchor refusal with the exact 59.4 bps message, straddling-bar
rejection, settled-window replay refusal, all four audit kinds, the no-raise
guarantee, and the API/snapshot/UI wiring.

Run just these:

```bash
.venv/bin/python -m pytest tests/test_board_integrity.py -q -p no:randomly
```
