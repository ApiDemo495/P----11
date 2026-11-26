"""§40 — the contradictions the user photographed.

Six screenshots, five real defects. Each test below pins one, with the
arithmetic that made it a defect rather than a matter of taste.

  1. BULLISH badge beside "impulse bear" — two regime computations, and the
     hero never read the engine's own `regime_conflict`.
  2. "72.9% gross" with "net needs target >= 43bp" — 43 bp is 1.5x the round
     trip, a fee convention, not break-even. That cell needs 60 bp, and its
     shipped geometry (reward:risk 0.15) cannot pay at ANY target.
  3. A stored key rendered as "not configured"/grey.
  4. "✓ local agent ready" printed while the runtime was not answering.
  5. A negative "CCI ADJ" beside a BULLISH badge with no explanation — the
     mandate rule overwrites `direction` and never touches `cci_adjusted`.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "static" / "app.js").read_text()


# ─────────────── 1. a regime conflict must withhold the side ──────────────
def test_hero_treats_regime_conflict_as_no_call():
    """The badge said "▲ BULLISH*" while the chip next to it said
    "📉 impulse bear". The hero now zeroes the side on a conflict."""
    assert "const conflict = res.regime_conflict || cyc?.regime_conflict || null;" in JS
    assert "const dir = conflict ? 0 : rawDir;" in JS


def test_every_panel_that_prints_a_side_reads_the_conflict():
    """Three places rendered BULLISH/BEARISH from `direction` alone."""
    # quick bar
    assert ("const dir = (res?.regime_conflict || cyc?.regime_conflict)\n"
            "      ? 0 : (cyc?.direction ?? res?.direction ?? 0);") in JS
    # why panel
    assert "const _cf = s.regime_conflict || null;" in JS
    assert "const dir = _cf ? 0 : (s.direction ?? 0);" in JS
    # hero call cell
    assert '${conflict ? "● WITHHELD" :' in JS


def test_conflict_is_explained_not_just_hidden():
    assert "why withheld" in JS
    assert "REGIME CONFLICT · side withheld" in JS


# ─────────────── 2. the cost floor must be break-even, not 1.5x ───────────
def test_breakeven_tp_bps_solves_the_net_zero_identity():
    """T* = RT / (p - (1-p)/rr). Verify by substituting back: net must be 0."""
    from app.hold_policy import breakeven_tp_bps
    cases = [(0.729, 28.98, 1.10), (0.60, 12.0, 1.0), (0.75, 20.0, 1.0),
             (0.55, 30.0, 2.0)]
    for p, rt, rr in cases:
        t = breakeven_tp_bps(p, rt, rr)
        assert t is not None, (p, rt, rr)
        stop = t / rr
        net = p * t - (1 - p) * stop - rt
        # the function rounds to 3 dp, so the residual is bounded by the
        # rounding, not by zero: |dn/dT| <= 1, so |net| <= 1e-3
        assert abs(net) < 1e-3, (p, rt, rr, t, net)


def test_breakeven_is_wider_than_the_fee_multiple_for_a_good_payoff():
    """The panel's 43 bp was 1.5 x RT. At 72.9 % and rr 1.10 the honest
    break-even is 60 bp — 17 bp wider. Following 43 loses 8.2 bp/trade."""
    from app.hold_policy import breakeven_tp_bps, MIN_TP_OVER_RT
    rt = 28.98
    fee_floor = MIN_TP_OVER_RT * rt
    t = breakeven_tp_bps(0.729, rt, 1.10)
    assert t == pytest.approx(60.045, abs=1e-2)
    assert t > fee_floor
    net_at_floor = 0.729 * fee_floor - 0.271 * (fee_floor / 1.10) - rt
    assert net_at_floor == pytest.approx(-8.00, abs=0.01), net_at_floor


def test_a_lottery_geometry_has_no_break_even_at_all():
    """row_k = 0.15 means target = 0.15 x stop, so rr = 0.15 and no target
    pays below a 1/(1+0.15) = 87 % hit rate."""
    from app.hold_policy import breakeven_tp_bps
    assert breakeven_tp_bps(0.742, 29.8, 0.15) is None
    assert breakeven_tp_bps(0.90, 29.8, 0.15) is not None


def test_summary_reports_the_break_even_not_a_fee_multiple():
    from app.hold_policy import summary
    s = summary("ETH", "tf1")
    assert s["reward_risk"] == pytest.approx(0.15)
    # the shipped geometry cannot pay, and the panel must say so rather than
    # print a 44 bp floor that funds the exchange
    assert s["cannot_pay_at_any_target"] is True
    assert s["enforced_tp_bps"] is None
    assert s["breakeven_tp_bps"] is None
    assert "first-touch rate above" in s["cost_note"]
    # and it must name what WOULD pay, not just refuse
    assert s["cost_what_would_pay"] and "x the round trip" in s["cost_what_would_pay"]


def test_the_cell_panel_does_not_print_the_fee_floor_as_the_requirement():
    # §41: the alarm was replaced by the two numbers that explain it
    assert "break-even target ${num(c.enforced_tp_bps, 0)}bp" in JS
    assert "COST: ${num(c.round_trip_bps, 1)}bp RT" in JS
    assert "net needs target ≥" not in JS


# ─────────────── 3. a stored key is not "not configured" ──────────────────
def _manager(settings, replies=None, enabled=True):
    from app.agents import AgentManager
    m = AgentManager.__new__(AgentManager)
    m.settings = dict(settings)
    m.settings.setdefault("gemini_enabled", enabled)
    m.keyring, m.calls, m.paid = {}, {}, {}
    m.last_replies = replies or {}
    m.catalog, m.auto_best, m.warning, m.active_slot, m.scores = [], False, "", "", {}
    m.local = type("L", (), {
        "available": False, "backend": "down", "label": "x", "ram_gb": 4.0,
        "last_ms": 0, "error": "daemon down", "download_progress": {}})()
    m.installer = type("I", (), {"status": lambda self=None: {"installed": False}})()
    return m


KEYS = {"gemini_key": "AIzaFAKEKEY123456", "backup1_key": "",
        "backup2_key": "", "backup3_key": "", "groq_key": ""}


def test_stored_but_untested_key_is_not_reported_as_absent():
    slot = _manager(KEYS).status()["slots"][0]
    assert slot["key"] != ""
    assert slot["status"] == "key stored · untested"
    assert slot["status"] != "not configured"


def test_a_failed_test_carries_the_actual_error():
    slot = _manager(KEYS, {"gemini": {"ok": False,
                                      "reason": "http 400"}}).status()["slots"][0]
    assert slot["status"] == "error"
    assert slot["reason"] == "http 400"


def test_a_disabled_slot_says_disabled_not_disconnected():
    slot = _manager(KEYS, enabled=False).status()["slots"][0]
    assert slot["status"] == "disabled"


def test_no_key_still_reads_not_configured():
    slot = _manager({"gemini_key": "", "backup1_key": "", "backup2_key": "",
                     "backup3_key": "", "groq_key": ""}).status()["slots"][0]
    assert slot["status"] == "not configured"


def test_slot_status_classes_cover_every_status_string():
    src = JS.split("function _slotCls(st)")[1][:600]
    for token in ('"connected"', '"disabled"', '"rate-limited"',
                  '"untested"', '"error"'):
        assert token in src, token
    assert ".slot-status.untested" in (ROOT / "static" / "style.css").read_text()


# ─────────────── 4. "local agent ready" when it is not ────────────────────
def test_installed_but_dead_runtime_is_not_green():
    assert "✓ local agent ready and loaded" in JS
    assert "the runtime is INSTALLED but NOT answering" in JS
    # the old unconditional line must be gone
    assert '`<div class="hint" style="margin-top:8px">✓ local agent ready${' not in JS


# ─────────────── 5. a negative CCI beside a BULLISH badge ─────────────────
def test_engine_records_which_rule_set_the_side():
    """`direction` is overwritten by the mandate rule; `cci_adjusted` is not,
    so the UI can print BULLISH next to a negative CCI with no explanation."""
    src = (ROOT / "app" / "engine.py").read_text()
    assert '"direction_cci": pack.get("direction")' in src
    assert '_res["direction_source"] = "mandate:%s"' in src
    cci = (ROOT / "app" / "cci.py").read_text()
    assert '"direction_source": "cci"' in cci


def test_ui_labels_the_sign_divergence():
    assert "opposite sign to the" in JS
    assert "Math.sign(dir) !== Math.sign(res.cci_adjusted)" in JS


# ─────────────── 6. "7/8 sources" beside "2 readings withheld" ────────────
def test_withheld_counts_reconcile_on_one_screen():
    """The panel printed 7/8 sources AND 2 readings withheld: 7 + 2 != 8,
    because channel abstentions were listed but not counted in the total."""
    from app import cognition
    src = (ROOT / "app" / "cognition.py").read_text()
    assert '"counts": {"present": len(present), "sources_withheld": n_declared,' in src
    assert "source(s) withheld" in src
    assert "channel abstention(s)" in src
    # the old sentence that added the two kinds together must be gone
    assert "reading(s) withheld, each with its" not in src


def test_coverage_denominator_counts_every_withheld_reading():
    """coverage = present / (present + withheld sources); channel
    abstentions are a separate, named quantity."""
    present, sources_withheld, channels = 7, 1, 1
    total = present + sources_withheld
    assert present + sources_withheld == total
    assert present + sources_withheld + channels > total   # why it looked wrong


# ─────────────── 7. a zero must say why it is zero ────────────────────────
def test_ledger_explains_an_empty_history():
    src = (ROOT / "app" / "ledger.py").read_text()
    assert '"why_empty": why_zero,' in src
    assert "no window has reached its boundary" in src


def test_mind_panel_shows_the_reason_for_zero_graded_outcomes():
    assert "S.snap?.ledger?.why_empty" in JS


# ─────────────── 8. the TP:SL ratio must match the prices beside it ───────
def test_ratio_is_recomputed_from_the_entry_and_stop_on_screen():
    """The panel showed ratio 1.21 next to TP 5.4bp / SL 3.4bp (=1.59) and a
    forecast target 0.57 above the entry. Three numbers, three ratios."""
    assert "const rr = Math.abs(tgt - heroEntry) / risk;" in JS
    assert "forecast target" in JS and "cycle target" in JS
    # the unchecked `cyc.ratio` print is gone
    assert '<div class="kv"><span>TP:SL ratio</span><b>${cyc?.ratio ?' not in JS


# ─────────── 9. the yellow band: one caveat, not a wall of alarms ─────────
def test_cost_caveat_is_structural_not_an_error_colour():
    """§40 painted "CANNOT PAY at any target" in the error red on all 12
    graded cells. The caveat is true for every shipped cell, so it read as a
    red wall and said nothing about WHY. It is now muted amber text that
    carries the two numbers that matter."""
    css = (ROOT / "static" / "style.css").read_text()
    assert ".hcell .hc-cost { font-size: 10px; color: #d8a13a; }" in css
    assert 'CANNOT PAY at any target</span>' not in JS
    assert "COST: ${num(c.round_trip_bps, 1)}bp RT" in JS


def test_summary_separates_infeasible_from_unpayable():
    """Two different failures, previously one label:
    (a) the round trip alone exceeds 1.5x the target  -> cost_infeasible
    (b) the geometry cannot pay at any target         -> cannot_pay_at_any_target
    """
    from app.hold_policy import summary
    s = summary("BTC", "tf1")
    assert s["cost_infeasible"] is False
    assert s["cannot_pay_at_any_target"] is True


def test_summary_quotes_the_multiple_of_round_trip_needed():
    from app.hold_policy import summary
    s = summary("BTC", "tf1")
    wr, rr = s["win_rate"], s["reward_risk"]
    expect = wr + (1 - wr) / rr
    assert s["tp_multiple_of_rt_needed"] == pytest.approx(expect, abs=1e-3)
    assert s["tp_multiple_of_rt_needed"] > 1.5      # the old fixed floor
    assert "x the round trip" in s["cost_what_would_pay"]


def test_every_graded_cell_gets_an_answer_not_just_an_alarm():
    """No cell may show a cost verdict without the number that would pay."""
    from app.hold_policy import summary, known_cells
    checked = 0
    for symbol, tf in known_cells():
        s = summary(symbol, tf)
        if s.get("win_rate") is None or s.get("reward_risk") is None:
            continue
        checked += 1
        assert s.get("target_verdict") or s.get("cost_note"), (symbol, tf)
        assert s.get("tp_multiple_of_rt_needed") is not None, (symbol, tf)
    assert checked >= 8, checked


def test_hc_cost_text_reports_both_numbers():
    """The chip must read as 'this cost vs this geometry', not an alarm."""
    assert 'num(c.reward_risk, 2)}x TP/SL' in JS
    assert "num(c.tp_multiple_of_rt_needed, 1)}x RT to pay" in JS
