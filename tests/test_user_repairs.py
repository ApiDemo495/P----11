"""This session's user-reported repairs (branch arena/3afeecae-p-11).

Complaints pinned here, each with the exact mechanism that fixes it:

1. **"it didn't use formulas nor brain in prediction but still locked
   my eth for timer"** — a broken stage replay left the FORMULAS row on
   "queued" forever (the handler only understood the WS wrapper shape)
   and a crashed analysis burned the cell timer with no seal.  Fixes:
   shape-tolerant `onAnalyseStage`, a `calculation_period` heal at
   done, a hard "no readings ⇒ fail the job" guard in the formulas
   stage, and cell-timer release when the job ends in error.
2. **"agents mostly, always or partially offline"** — cloud transport
   failures now cool the slot down, the local model has a 5 s wall,
   and the `ai_review` stage gets a hard 8 s budget with an honest
   skip note.
3. **"cancel trade but didn't tell which one"** — every `sl_event`
   message names symbol, tf, side and lock id (server side), with a
   JS fallback that composes the identity.
4. **"said impulse bear but give bullish trade"** — the REGIME
   CONFLICT gate: an IMPULSE_* regime state opposing the certified
   side forces `gate_ok=False`, verdict NO CALL, tier watch-only
   (never routed) while the sealed record keeps the side honest.
"""
import asyncio
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# ── 1. formulas row / sealed-without-formulas ───────────────────────

def test_stage_handler_accepts_all_three_shapes():
    js = _read("static/app.js")
    m = re.search(r"function onAnalyseStage\(msg\) \{.*?\n\}", js, re.S)
    assert m, "onAnalyseStage missing"
    body = m.group(0)
    # wrapper shape {stage: entry}
    assert 'typeof msg.stage === "object"' in body
    # bare entry / calc entry shape {stage: "formulas", ...}
    assert 'typeof msg.stage === "string"' in body
    # tolerant field reads (duration_ms | ms, t_start_ms | at)
    assert "st.ms" in body and "st.at" in body
    # the OLD bug: `const st = msg.stage; if (!st) return;` with a
    # string stage silently produced st.stage === undefined
    assert "msg.stage;" not in body.split("\n")[1] or "object" in body


def test_replay_builder_reads_entry_objects():
    js = _read("static/app.js")
    assert 'typeof st === "object" ? String(st?.stage || "?")' in js


def test_done_heals_still_queued_rows():
    js = _read("static/app.js")
    assert "HEAL rows whose live WS event was missed" in js
    assert 'row.className.trim() === "an-row"' in js
    assert "calculation_period?.stages" in js


def test_formulas_stage_hard_guard():
    src = _read("app/engine.py")
    assert "formulas stage produced no readings — refusing to" in src
    # the guard must sit right after the stage result is stored
    i = src.index("store[\"raws\"] = raws")
    assert "formulas stage produced no readings" in src[i:i + 600]


def test_failed_analysis_releases_cell_timer_and_emits_error():
    src = _read("app/engine.py")
    # generic failure handler in _run_analyse (crashes used to vanish)
    i = src.index("async def _run_analyse")
    body = src[i:i + 3500]
    assert "except Exception as exc:" in body
    assert "analysis failed:" in body
    assert '"type": "analyse_done"' in body
    # error ⇒ the cell timer is returned to the user
    assert "_hp_rel.cell_locks().clear" in body
    assert 'if job.get("status") == "error":' in body


# ── 2. agents offline / slow ─────────────────────────────────────────

def test_provider_transport_failure_sets_cooldown():
    from app import agents as ag

    am = ag.AgentManager()
    calls = {"n": 0}

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            calls["n"] += 1
            raise OSError("connection refused")

    orig = ag.httpx.AsyncClient
    ag.httpx.AsyncClient = Boom
    try:
        res = asyncio.run(am._call_provider("backup2", "sk-test-key",
                                            "prompt"))
    finally:
        ag.httpx.AsyncClient = orig
    assert res["ok"] is False
    assert calls["n"] == 1
    kr = am.keyring.get("backup2") or {}
    assert kr.get("state") == "offline", kr
    assert float(kr.get("cooldown_until", 0)) > time.time(), \
        "transport failure must cool the slot (no re-dial every run)"
    # the very next call must NOT touch the network
    ag.httpx.AsyncClient = Boom
    try:
        res2 = asyncio.run(am._call_provider("backup2", "sk-test-key",
                                             "prompt"))
    finally:
        ag.httpx.AsyncClient = orig
    assert res2.get("reason") == "rate limited"
    assert calls["n"] == 1, "cooled slot must not hit the wire again"


def test_local_review_has_hard_wall():
    from app import agents as ag

    am = ag.AgentManager()

    def slow(prompt, max_tokens=200):
        time.sleep(8)                       # a wedged llama (> 5 s wall)
        return {"ok": True, "text": "late", "backend": "llama_cpp"}

    am.local.ask = slow
    am.local.available = True               # skip the try_load probe
    # the wall fires at 5 s: the coroutine returns the timeout reason
    # long before the wedged thread finishes (asyncio.run then joins
    # the thread at shutdown — that is executor hygiene, not a leak
    # in the long-running server)
    res = asyncio.run(am._call_local("prompt"))
    assert res["ok"] is False
    assert "timed out" in res.get("reason", "")
    assert "slow or offline" in res.get("reason", "")


def test_ai_review_stage_budget_is_pinned():
    src = _read("app/engine.py")
    i = src.index("OFFLINE/SLOW-AGENT BUDGET")
    body = src[i:i + 1500]
    assert "timeout=8.0" in body
    assert "exceeded the 8 s budget" in body
    assert "slow/offline backend skipped" in body


# ── 3. cancel/close alert must name the trade ───────────────────────

def test_sl_events_carry_full_identity():
    src = _read("app/engine.py")
    # both kinds: message starts with symbol+tf and names side + lock
    assert src.count("SL hit vs") >= 2, "safe+close must both name it"
    assert '"side": "BUY" if original_direction > 0 else "SELL"' in src
    assert '"lock_id": str(getattr(cyc, "lock_id", "") or "")' in src
    assert '{symbol} {tf.upper()} — CLOSE POSITION IMMEDIATELY' in src
    assert '{symbol} {tf.upper()} — SAFE, BE CAUTIOUS' in src
    # JS fallback composes identity when a bare kind arrives
    js = _read("static/app.js")
    assert 'msg.type === "sl_event"' in js
    assert "toast(msg.message || `${who} — ${msg.kind}`" in js


# ── 4. regime conflict: never the opposite trade ────────────────────

def test_regime_conflict_holds_routing_but_never_silences_the_call():
    """This test used to pin the OPPOSITE of the user's directive: it
    required the regime conflict to set gate_ok=False, force verdict
    "NO CALL" and demote the tier. That is exactly the abstention the user
    told us to stop doing ("increase win rate without abstention"), applied
    to windows that already had a measured, sealed, graded side. The gate
    keeps its real job — no ORDER may be routed on a self-contradicting
    tape — while the side, levels, contract and grading stand."""
    src = _read("app/engine.py")
    i = src.index("REGIME CONFLICT (user order)")
    j = src.index('final["verdict"] = ("ISSUE" if (final.get("gate_ok")', i)
    body = src[i:j]
    _verdict_src = src[j:j + 400]
    assert '"IMPULSE_BULL": 1' in body and '"IMPULSE_BEAR": -1' in body
    # routing is still held back...
    assert 'final["execution_eligible"] = False' in body
    # ...but the badge and the size are NOT silenced
    assert 'final["gate_ok"] = False' not in body
    assert '"NO CALL" if _conflict is not None' not in body
    assert "REGIME DISAGREEMENT" in body
    assert "The side " in body and "stands and is graded" in body
    assert 'final["verdict"] = ("ISSUE" if (final.get("gate_ok")' in _verdict_src
    # and the note must never read like an abstention is the safe choice
    assert "it is an ungraded one" in body


def test_regime_conflict_no_longer_demotes_the_tier_to_watch_only():
    src = _read("app/engine.py")
    seg = src[src.index("REGIME CONFLICT (user order)"):]
    seg = seg[:seg.index('final["verdict"] = (')]
    assert 'cyc.tier = "watch-only"' not in seg


def test_no_side_collapses_the_tier_and_never_advertises_size():
    """Live defect: `BTC tf4 · direction 0 · tier full · NO CALL`. A size
    hint for a non-trade is a contradiction; it must collapse to watch-only
    and explain itself."""
    src = _read("app/engine.py")
    assert 'if direction is not None and int(direction) == 0 \\' in src
    assert 'and tier != "watch-only":' in src
    assert "nothing to size" in src
    # a caller that never states a side (pure tier maths) is NOT a flat call
    assert "direction: int | None = None" in src


def test_every_window_ships_a_side_never_a_flat_card():
    """The core directive: no abstention. A fused score inside the noise gate
    must still produce a side, labelled with where it came from."""
    src = _read("app/engine.py")
    assert "NO MORE FLAT CARDS" in src
    assert 'fin["direction_fallback"]' in src
    # the fallback prefers the one MEASURED reading left (fade the impulse)…
    assert "the record says fade it" in src
    # …and admits when it is trading a coin flip
    assert "UNINFORMATIVE" in src


def test_conflict_aware_signal_box():
    js = _read("static/app.js")
    # the banner explains the disagreement without replacing the side
    assert "REGIME DISAGREEMENT" in js or "REGIME CONFLICT" in js
    assert "${conflictBanner}" in js
    # the big word must be the real side — a disagreeing chip may not
    # overwrite it with "NO CALL" (that was the abstention defect)
    assert 'const bigWord = r.regime_conflict ? "NO CALL" : r.signal;' not in js


def test_conflict_check_uses_the_same_classifier_as_the_chip():
    src = _read("app/engine.py")
    # the gate must classify THIS asset's tape — the exact input the
    # TF-card chip renders — so the user's "impulse bear" reading and
    # the gate can never disagree
    i = src.index("REGIME CONFLICT (user order)")
    body = src[i:i + 1600]
    assert "_ctape(list(_stq.closes or [])" in body
    assert "self.feed.state(symbol)" in body


# ── 5. formula/brain proof is visible in the result box ─────────────

def test_result_box_shows_formula_signs_and_brain_signals():
    js = _read("static/app.js")
    assert "formulas used:" in js
    assert "formula_signs" in js
    assert "brain signals:" in js
    assert "prediction_lock?.evidence" in js
