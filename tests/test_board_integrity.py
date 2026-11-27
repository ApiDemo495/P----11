"""Board integrity — the three defects the user photographed (2026-11-27).

Each test pins one half of the user's report against the mechanism that
produced it, so the same page can never be shipped again:

1. a BEARISH card whose target sits ABOVE its own entry (the "big BEARISH
   sign when the analysed prediction is BUY" card, TP 3,649 over entry
   3,632 on a short);
2. a prediction recorded LOST in history the moment ANALYSE was pressed,
   before price had gone anywhere near the stop;
3. an app that does not notice — or report — either of the above.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.predlock import PredictionJournal

STATIC = Path(__file__).resolve().parent.parent / "static"
APP_JS = (STATIC / "app.js").read_text()
MAIN_PY = (Path(__file__).resolve().parent.parent / "app" / "main.py").read_text()


def _j(tmp_path) -> PredictionJournal:
    return PredictionJournal(str(tmp_path / "locks.json"))


# ── 1. the wrong-side target on the hero card ──────────────────────────
def test_hero_target_is_chosen_from_the_calls_own_side():
    """The card must never use a target on the wrong side of its entry.

    The defect: `cyc.forecast.price` was taken FIRST with no side check, so
    a forecast computed for the pre-flip BUY was printed on the flipped
    SELL — a short whose target sat above its own entry, rendered as
    "TP:SL 0.87:1 (FORECAST TARGET 3,649 vs STOP 3,651 FROM ENTRY 3,632)".
    """
    js = APP_JS
    i = js.index("TP:SL ratio")
    block = js[i - 4000:i + 4000]
    # the side test exists and gates the candidate list
    assert "_ahead" in block, "the hero must test a target against the side"
    assert "(_side > 0 ? +t > heroEntry : +t < heroEntry)" in block.replace(
        "\n", " ").replace("  ", " ") or "t > heroEntry" in block
    # every source is a candidate, in order, and the first on-side one wins
    assert '"forecast target"' in block and '"cycle target"' in block
    assert "_cands.find(([, v]) => _ahead(v))" in block
    # when nothing is on-side the card says so instead of claiming a ratio
    assert "NO TARGET ON THE" in block
    # and a selected sub-1:1 target is labelled, never silently printed
    assert "target widened to the 1:1 floor" in block


def test_wrong_side_targets_are_named_when_ignored():
    """An ignored target must be visible on the card, not silently dropped."""
    block = APP_JS[APP_JS.index("TP:SL ratio") - 4000:
                   APP_JS.index("TP:SL ratio") + 4000]
    assert "ignored ${_wrong" in block and "wrong side" in block


# ── 2. no loss before the market can move ──────────────────────────────
def test_seal_refuses_an_anchor_that_is_not_the_live_market(tmp_path):
    """A stale anchor is refused by name — the instant-loss root cause.

    The live board sealed ETH at 4,923.24 while the market traded 4,952.50
    (59 bps away) with a 5.5 bp stop: every barrier was already behind the
    market, so the first completed bar graded a loss seconds after the
    press.
    """
    j = _j(tmp_path)
    out = j.seal("ETH", "tf1", direction=1, entry=4923.241665,
                 tp=4945.25446313, sl=4920.548755, confidence=0.7,
                 created_at=time.time(), now_price=4952.501167)
    assert out["locked"] is False and out["status"] == "refused"
    joined = " ".join(out["why"])
    assert "entry anchor" in joined and "59.4 bps" in joined
    assert "target" in joined and "already" in joined
    assert j.rows == [], "a refused seal writes no history row"


def test_seal_accepts_a_barrier_that_still_stands(tmp_path):
    """The same call with honest levels seals normally."""
    j = _j(tmp_path)
    out = j.seal("ETH", "tf1", direction=1, entry=1000.0, tp=1010.0,
                 sl=990.0, confidence=0.7, created_at=time.time(),
                 now_price=1000.05)        # 0.5 bps: the same market
    assert out["locked"] is True
    # a 4 bps quote move is NOT the same market — it must be re-anchored,
    # not silently sealed (the seal tolerance is 2 bps)
    moved = j.seal("ETH", "tf2", direction=1, entry=1000.0, tp=1010.0,
                   sl=990.0, confidence=0.7, created_at=time.time(),
                   now_price=1000.4)
    assert moved["locked"] is False and "anchor" in " ".join(moved["why"])


def test_bar_that_opened_before_the_seal_cannot_grade_it(tmp_path):
    """A bar stamped after the seal but OPENED before it is pre-seal tape.

    The tape stamps a bar at its opening minute, so the bar carrying
    ts = created+50s still contains 50 s of tape from before the prediction
    existed. Grading on it charges the lock for a move it never saw.
    """
    j = _j(tmp_path)
    created = 1_700_000_000.0 + 50.0         # 50 s into a minute
    minute = (int(created) // 60) * 60
    j.seal("ETH", "tf1", direction=1, entry=1743.0, tp=1744.0, sl=1742.34,
           confidence=0.7, created_at=created)
    # this bar ends after the seal but its low is 50 s of PRE-seal tape
    straddling = [(minute + 60, 1743.2, 1742.20)]
    got = j.settle_all({"ETH": 1743.0}, series={"ETH": straddling},
                       now=created + 15, bar_s=60.0)
    assert got == [], "a straddling bar must not decide the lock"
    # the same excursion on a FULLY post-seal bar does grade it
    later = [(minute + 120, 1743.2, 1742.20)]
    got2 = j.settle_all({"ETH": 1743.0}, series={"ETH": later},
                        now=created + 75, bar_s=60.0)
    assert len(got2) == 1 and got2[0]["status"] == "lost"


def test_a_settled_window_is_not_replayed_as_the_new_prediction(tmp_path):
    """Pressing ANALYSE again must not hand back a graded row as "today".

    The second press inside the same wall-clock window used to return the
    OPEN-OR-SETTLED row with `already: True`, so a LOST row appeared in the
    history in the same instant the user pressed the button.
    """
    j = _j(tmp_path)
    j.seal("ETH", "tf1", direction=1, entry=100.0, tp=102.0, sl=99.0,
           confidence=0.7, created_at=10_000)
    j.settle_all({"ETH": 100.0}, series={"ETH": [(10_060, 100.5, 98.5)]},
                 now=10_120, bar_s=60.0)
    again = j.seal("ETH", "tf1", direction=1, entry=101.0, tp=103.0,
                   sl=100.0, confidence=0.8, created_at=10_200)
    assert again["locked"] is False
    assert again.get("settled") is True
    assert "already graded LOST" in " ".join(again["why"])


def test_an_open_window_still_holds_its_lock(tmp_path):
    """The immutability rule itself is unchanged for a live prediction."""
    j = _j(tmp_path)
    a = j.seal("BTC", "tf2", direction=1, entry=100.0, tp=104.0, sl=99.0,
               confidence=0.4, minutes=44, created_at=10_000)
    b = j.seal("BTC", "tf2", direction=-1, entry=300.0, tp=200.0, sl=400.0,
               confidence=0.4, minutes=44, created_at=10_100)
    assert a["locked"] and not a["already"]
    assert b["locked"] and b["already"] and b["id"] == a["id"]


def test_refusal_message_has_no_unformatted_placeholder(tmp_path):
    """The old text shipped a literal `{W}`; a refusal must read as English."""
    j = _j(tmp_path)
    j.seal("BTC", "tf1", direction=1, entry=100.0, tp=101.0, sl=99.0,
           confidence=0.4, created_at=10_000)
    again = j.seal("BTC", "tf1", direction=1, entry=100.0, tp=101.0,
                   sl=99.0, confidence=0.4, created_at=10_010)
    why = " ".join(again["why"])
    assert "{W}" not in why and "22-minute window" in why


# ── 3. the app detects and reports its own inconsistencies ──────────────
def test_audit_is_clean_on_a_coherent_journal(tmp_path):
    j = _j(tmp_path)
    j.seal("ETH", "tf1", direction=1, entry=100.0, tp=102.0, sl=99.0,
           confidence=0.7, created_at=10_000)
    j.settle_all({"ETH": 100.0}, series={"ETH": [(10_060, 102.5, 99.5)]},
                 now=10_120, bar_s=60.0)
    rep = j.audit()
    assert rep["ok"] is True and rep["issues"] == [] and rep["checked"] == 1


def test_audit_catches_an_unearned_settlement(tmp_path):
    """A row graded in under a bar is the defect, and it must be found."""
    j = _j(tmp_path)
    j.seal("ETH", "tf1", direction=1, entry=100.0, tp=102.0, sl=99.0,
           confidence=0.7, created_at=10_000)
    j.rows[-1].update({"settled_at": 10_000.4, "outcome": "lost",
                       "status": "lost", "bps": -100.0,
                       "settle_reason": "SL touched on a closed bar"})
    rep = j.audit()
    assert rep["ok"] is False
    kinds = {i["kind"] for i in rep["issues"]}
    assert "unearned-settlement" in kinds
    detail = " ".join(i["detail"] for i in rep["issues"])
    assert "0.40s" in detail


def test_audit_catches_a_ladder_on_the_wrong_side(tmp_path):
    j = _j(tmp_path)
    j.seal("ETH", "tf1", direction=1, entry=100.0, tp=102.0, sl=99.0,
           confidence=0.7, created_at=10_000)
    # corrupt the stored row exactly as the live board did: a SELL badge
    # with a target above its own entry
    j.rows[-1].update({"direction": -1, "side": "SELL", "tp": 3649.0,
                       "sl": 3651.0, "entry": 3632.0})
    rep = j.audit()
    assert rep["ok"] is False
    assert "ladder-on-wrong-side" in {i["kind"] for i in rep["issues"]}


def test_audit_never_raises_on_a_corrupt_row(tmp_path):
    """A self-check that dies is worse than no self-check."""
    j = _j(tmp_path)
    j.seal("ETH", "tf1", direction=1, entry=100.0, tp=102.0, sl=99.0,
           confidence=0.7, created_at=10_000)
    j.rows[-1].update({"entry": "not a number", "settled_at": "nope",
                       "direction": None})
    rep = j.audit()
    assert isinstance(rep, dict) and "issues" in rep


def test_the_audit_is_published_on_the_api_and_the_snapshot():
    """The result must reach the page, not just the log."""
    assert '@app.get("/api/predictions/audit")' in MAIN_PY
    assert "engine.predlocks.audit()" in MAIN_PY
    engine_py = (Path(__file__).resolve().parent.parent
                 / "app" / "engine.py").read_text()
    assert '"prediction_audit": self._prediction_audit_cached()' in engine_py
    assert "def _prediction_audit_cached" in engine_py


def test_engine_hands_the_live_price_to_the_seal():
    """The engine must give the journal the market it is sealing against."""
    engine_py = (Path(__file__).resolve().parent.parent
                 / "app" / "engine.py").read_text()
    assert "now_price=(_live_seal or None)" in engine_py
    # and it re-anchors instead of leaving the window unsealed
    assert "_reanchored" in engine_py and "re-anchored on the live price" in engine_py
    # the settlement walk knows the bar width
    assert "bar_s=60.0" in engine_py
    # the 20 %-only anchor guard is gone: a 59 bps skew is re-anchored
    assert "_anchor_skew_bps > 2.0" in engine_py


def test_audit_payload_is_json_serialisable(tmp_path):
    j = _j(tmp_path)
    j.seal("ETH", "tf1", direction=1, entry=100.0, tp=102.0, sl=99.0,
           confidence=0.7, created_at=10_000)
    json.dumps(j.audit())          # the API must be able to serve it


def test_the_history_page_renders_the_self_audit():
    """Bug 3 is only fixed if the user can SEE the app flag itself."""
    index_html = (STATIC / "index.html").read_text()
    assert 'id="pred-audit"' in index_html
    assert "async function fetchPredictionAudit()" in APP_JS
    assert 'api("/api/predictions/audit")' in APP_JS
    # a clean journal is quiet, a broken one is loud, and a failed check is
    # itself reported (never rendered as silence)
    assert "SELF-AUDIT FOUND" in APP_JS
    assert "SELF-AUDIT FAILED" in APP_JS
    assert "no wrong-side ladder" in APP_JS


def test_sub_one_to_one_is_repaired_by_widening_the_target():
    """The absolute rule: never below 1:1, and never by narrowing the stop.

    The card may not print a ratio under the floor, and it may not quietly
    drop the cell either (no demotion). It widens the TARGET to 1.0x the
    measured risk, leaves the stop exactly as measured, and prints both
    numbers so the repair is never mistaken for a measurement.
    """
    i = APP_JS.index("TP:SL ratio")
    block = APP_JS[i - 4000:i + 4000]
    assert "THE FLOOR WIDENS THE TARGET, NEVER NARROWS THE STOP" in block
    # the widening is entry + side * risk: the stop is not touched
    assert "heroEntry + _side * risk" in block
    assert "stop left unchanged at" in block
    # a repaired cell is still shown (repaired, never silenced)
    assert "_floorHit ? 1.0 : _rr0" in block
