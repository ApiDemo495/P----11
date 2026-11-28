#!/usr/bin/env python3
"""The published number, the published side, and the published reason.

Reported 2026-11-27, four complaints from one session of screenshots:

  1. "why their is n/a?"                      — a card with no rate said n/a
                                                and never said why
  2. "In explain this why confidence is too   — the CONFIDENCE CHAIN panel
     low you can do anything to increase        printed a total (0.421) that
     confidence"                                the published number (0.021)
                                                contradicted by 20x
  3. "chain total and published total problem" — same defect, named
  4. "app told me to buy and gave me tp and    — the card published BUY, the
     sl but on my app entry price was above      note said FLIPPED to SELL,
     tp"                                         and the levels were BUY's

None of these is a rendering preference. Each is two parts of the app
describing different objects.
"""
from __future__ import annotations

import inspect
import math

import pytest


# ────────── 1. the confidence chain owns ONE number ──────────────────
def test_temper_cannot_discard_the_calibrated_confidence():
    """`temper()` returns confidence unchanged for an asset with no rule.

    The engine used to assign that value back over the calibrated number,
    which discarded the whole chain for exactly the assets with nothing to
    say (BTC has no tempering rule). A 0.42 chain landed on a 0.021 card.
    """
    from app.asset_behavior import temper

    # an asset WITH a tempering rule may legitimately damp the number —
    # that is the rule doing its job, and it states a factor when it does
    for symbol in ("BTC", "ETH", "BNB", "SOL"):
        r = temper(symbol, 0.01, 0.421, [100.0] * 60)
        if r["factor"] == 1.0:
            # no rule fired ⇒ the confidence it returns is the one it was
            # given, and the engine must not treat that as a change
            assert r["confidence"] == pytest.approx(0.421), symbol
        else:
            # a rule fired ⇒ the change is stated and bounded
            assert 0.7 <= r["confidence"] / 0.421 <= 1.2, symbol


def test_the_engine_only_adopts_a_confidence_temper_actually_changed():
    src = inspect.getsource(__import__("app.engine", fromlist=["x"]))
    i = src.index("_calibrated_conf = float(fin.get(\"confidence\")")
    block = src[i:i + 1400]
    # the calibrated number is captured before temper runs…
    assert "_calibrated_conf" in block
    # …compared against what temper returned…
    assert "abs(_beh_c - _calibrated_conf) > 1e-9" in block
    # …and restored verbatim when nothing changed
    assert '_beh["confidence"] = _calibrated_conf' in block


def test_every_stage_that_moves_confidence_is_recorded():
    """The user must be able to see WHERE a number collapsed."""
    src = inspect.getsource(__import__("app.engine", fromlist=["x"]))
    assert "_ctrace" in src and '"conf_trace"' in src
    # the recorder only logs real moves, so a stable chain is still
    # auditable ("0.42 in, 0.42 out")
    assert "if abs(float(after) - float(before)) > 1e-9:" in src
    # the advanced layer and the double-check both report
    assert '_conf_stage("advanced layer ×"' in src
    assert '"stage": "4-vote double-check"' in src


def test_the_floor_is_the_honest_explanation_for_a_no_edge_window():
    """Confidence 0.02 is NOT a broken chain — it is |CCI| inside the noise.

    The distinction matters: one is a bug to fix, the other is the market
    having no edge this window. The chain must produce the floor value, and
    the card must SAY that is what happened.
    """
    from app.cci import calibrate_confidence, CONF_MIN

    # a fused score of exactly zero has no signal at all
    out = calibrate_confidence({"cci_adjusted": 0.0, "drhe": 1.0})
    assert out["confidence"] == pytest.approx(CONF_MIN)
    assert out["snr"] == pytest.approx(0.0)
    # a score above its noise floor produces real confidence from the SAME
    # formula, so the chain is not structurally stuck low
    hi = calibrate_confidence({"cci_adjusted": 0.15, "drhe": 1.0})
    assert hi["confidence"] > 0.7
    assert hi["confidence"] > out["confidence"] * 10

    js = open("static/app.js", encoding="utf-8").read()
    assert "confidence is at the floor" in js
    assert "there is no edge to report in this window" in js


def test_the_card_renders_the_trace_and_the_floor_it_was_measured_against():
    js = open("static/app.js", encoding="utf-8").read()
    assert "s.conf_trace" in js
    assert "· floor ${fmt(s.asset_floor, 3)}" in js


# ────────────── 2. "n/a" always carries its reason ───────────────────
def test_no_certified_rate_never_renders_a_bare_n_a():
    js = open("static/app.js", encoding="utf-8").read()
    i = js.index("const wr = c.win_rate != null")
    block = js[i:i + 600]
    assert "no rate — ungradeable" in block
    assert "no rate — not measured" in block
    # the old bare literal is gone from this branch
    assert '"n/a"' not in block


def test_an_ungradeable_cell_prints_the_reason_it_has_no_rate():
    from app import hold_policy as hp

    s = hp.summary("SOL", "tf1")
    # "unknown" (nothing measured at this duration), never "ungradeable"
    # (a measurement exists and failed) — the card must not merge them.
    assert s["status"] == "unknown"
    assert s["win_rate"] is None
    # the payload carries the why, and the card renders it
    assert "no win rate" in (s.get("why") or "")
    assert "never been walked" in (s.get("why") or "")
    js = open("static/app.js", encoding="utf-8").read()
    assert "c.why ?" in js


# ──────── 3. the side and the note describe the same trade ───────────
def test_resolve_reports_the_flipped_side_and_the_engine_adopts_it():
    from app import verdict as vd
    from app import trend_gate as tg

    tg.reset()
    try:
        for c in (100.0, 101.0, 102.0):
            tg.observe("BTC", "tf9", c)
        ok, verd, note, side = vd.resolve("BTC", "tf9", 1, 0.60, "full")
        # the note says FLIPPED and the tuple agrees
        assert "FLIPPED" in note
        assert side == -1
    finally:
        tg.reset()


def test_the_engine_publishes_the_resolved_side_not_the_raw_one():
    """The live defect: `signal` and `direction` kept the pre-flip side."""
    src = inspect.getsource(__import__("app.engine", fromlist=["x"]))
    i_resolve = src.index("_side_resolved = direction")
    # the publish must land before the levels are read off the cycle, so bound
    # the window there rather than at an arbitrary character count
    i_levels = src.index("_lv = _lv_from_cycle(cyc)", i_resolve)
    block = src[i_resolve:i_levels]
    # resolve() is consulted with all four returns captured
    assert "_ok_r, _vd_r, _nt_r, _side_r = _v_resolve(" in block
    # and the published fields are overwritten when it differs
    assert 'res["direction"] = _side_resolved' in block
    assert 'res["signal"] = ("BUY" if _side_resolved > 0' in block
    # the pre-flip side stays visible for the divergence note
    assert 'res["direction_cci"] = direction' in block
    # the 2:1 block no longer REVERSES the side: flipping a fixed 2:1 pair
    # only relabels the book (resolved -24R <-> +24R, total -0.82R either
    # way), so it is published as a note and the engine keeps its own side.
    assert "two_to_one_side_note" in block
    assert '"applied": False' in block
    assert "_side_resolved = _rr_dir" not in block
    assert "two_to_one_direction" not in block


def test_the_side_is_resolved_before_any_level_is_built():
    """A flipped side with the OLD levels is a BUY target on a SELL trade."""
    src = inspect.getsource(__import__("app.engine", fromlist=["x"]))
    i_resolve = src.index("_side_resolved = direction")
    i_levels = src.index("_lv = _lv_from_cycle(cyc)", i_resolve)
    assert i_resolve < i_levels, (
        "the side must be resolved before levels are read off the cycle")


# ───── 4. a published pair always points the right way ───────────────
@pytest.mark.parametrize("side", [1, -1])
def test_a_published_pair_puts_tp_ahead_of_entry_and_sl_behind(side):
    from app.moves_risk import risk_block, reanchor

    closes = [61600 + i for i in range(200)]
    highs = [61700 + i for i in range(200)]
    lows = [61500 + i for i in range(200)]
    b = risk_block(61656.40, side, 22, 0.5, closes, highs, lows,
                   atr=30.0, bar_minutes=1.0)
    assert (b["tp"] - b["entry"]) * side > 0
    assert (b["sl"] - b["entry"]) * side < 0
    assert b["rr"] >= 1.0

    # re-anchoring moves the anchor and keeps the bps distances, so the
    # pair can never end up inverted by a re-lock
    r = reanchor(b, 62000.0)
    assert (r["tp"] - r["entry"]) * side > 0
    assert (r["sl"] - r["entry"]) * side < 0
    assert r["rr"] >= 1.0


def test_a_seal_refuses_a_pair_the_market_has_already_passed():
    """A BUY whose target is already at/below spot cannot run to it."""
    from app.predlock import _barrier_behind_market

    why = _barrier_behind_market(1, 100.0, 99.0, 95.0, 100.5)
    assert "already at/below the live price" in why
    why = _barrier_behind_market(-1, 100.0, 101.0, 105.0, 99.5)
    assert "already at/above the live price" in why
    # a correct pair passes silently
    assert _barrier_behind_market(1, 100.0, 110.0, 95.0, 100.5) == ""


def test_the_seal_overlay_publishes_the_ratio_it_built_on():
    src = inspect.getsource(__import__("app.engine", fromlist=["x"]))
    i = src.index('"row_tp": tp, "row_sl": sl,')
    assert '"row_rr"' in src[i:i + 700]
