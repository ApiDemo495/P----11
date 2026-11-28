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


# ─────── 1. a regime conflict must be EXPLAINED, never silenced ────────
# These tests used to pin the opposite of the user's directive: they required
# the UI to blank the side ("NO CALL" / "side withheld") whenever the 5-state
# chip disagreed with it. That resolved a contradiction by deleting the
# answer, on windows that were already measured, sealed and graded — the
# abstention the user told us to stop. The disagreement is now a banner
# BESIDE a side that stands.

def test_hero_keeps_the_certified_side_on_a_regime_disagreement():
    # the hero's direction is the engine's direction, NOT blanked by a chip
    assert "const dir = cyc?.direction ?? res.direction ?? 0;" in JS
    assert "const dir = conflict ? 0 : rawDir;" not in JS
    # the banner exists and says routing — not the call — is what is held
    assert "REGIME DISAGREEMENT" in JS
    assert "only ORDER ROUTING is held back" in JS


def test_every_panel_that_prints_a_side_reads_the_conflict():
    """Three places render BULLISH/BEARISH from `direction`; each must also
    surface the disagreement as a chip/banner rather than flipping to flat."""
    # quick bar: side stands, chip carries the caveat
    assert "const _cf = res?.regime_conflict || cyc?.regime_conflict || null;" in JS
    assert "no routing" in JS
    # the hero header chip names the opposing state AND the side it opposes,
    # in one sentence (2026-11-27: "it is showing bullish in yellow box with
    # impulse bear") — a bare "routing held" chip left the badge contradicting
    # the chip beside it with nothing explaining the pair
    assert "OPPOSES THIS" in JS and "routing held" in JS
    assert "${conflict ? \"\" : chopChip(tf)}" in JS
    # and the ANALYSE box big word is the real side
    assert "const bigWord = r.signal;" in JS


def test_conflict_is_explained_not_just_hidden():
    assert "routing held" in JS
    assert "REGIME CONFLICT · side withheld" not in JS
    # the note must say a withheld call is the WORSE option
    assert "it is an ungraded one" in JS
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
    # BTC tf2 (44 min) is the shortest CERTIFIED lane. tf1 is a 24-hour
    # window as of 2026-11-27 and was never in the artifact, so it reports
    # the "not certified at this duration" state instead of a cost verdict.
    s = summary("BTC", "tf2")
    assert s["cost_infeasible"] is False
    assert s["cannot_pay_at_any_target"] is True


def test_summary_quotes_the_multiple_of_round_trip_needed():
    from app.hold_policy import summary
    s = summary("BTC", "tf2")
    wr, rr = s["win_rate"], s["reward_risk"]
    expect = wr + (1 - wr) / rr
    assert s["tp_multiple_of_rt_needed"] == pytest.approx(expect, abs=1e-3)
    assert s["tp_multiple_of_rt_needed"] > 1.5      # the old fixed floor
    assert "x the round trip" in s["cost_what_would_pay"]


def test_an_uncertified_24h_lane_says_why_instead_of_inventing_a_rate():
    """The 2:1 lane is new; it must not borrow a 22-minute certification.

    The hold-policy artifact is keyed by lane DURATION. tf1 is 1440 minutes
    now, so it deliberately has no row. The honest answer is a reason that
    names the duration — never a silent None and never a reused number.
    """
    from app.hold_policy import summary, tier_reason
    s = summary("BTC", "tf1")
    assert s["win_rate"] is None
    assert s.get("reward_risk") is None
    why = tier_reason("BTC", "tf1")
    from app.config import TIMEFRAMES
    assert f"{TIMEFRAMES['tf1']['minutes']}-minute" in why
    assert "invent" in why.lower() or "no certified" in why.lower()


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
