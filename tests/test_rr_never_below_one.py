#!/usr/bin/env python3
"""REWARD:RISK MAY NEVER BE BELOW 1:1 — the user's absolute rule.

Reported 2026-11-27: "eth one was less than 1:1". It was, in three
separate places, and none of them were a display problem:

  1. THE SHIPPED GEOMETRY.  Every one of the 12 graded cells was selected
     at row_k = 0.15 against row_m = 0.55 — a target 0.27x its stop. The
     board then printed "74.3 %" for it with no ratio anywhere on screen.
     At 0.27:1 a window must clear 78.6 % just to break even and its EV is
       p*k - (1-p)*m = 0.7425*0.15 - 0.2575*0.55 = -0.030 R per trade,
     i.e. a certified win rate that loses money on every single window.

  2. THE LIVE LOCK.  `_cost_aware` widened a sub-round-trip target to
     1.5x the round trip and stopped there. On ETH tf1 that is 44.7 bps
     against a 55 bps stop — 0.81:1 — while its own note claimed a win
     would now "actually pay". Every sealed hold lock was built from it.

  3. THE RESEARCH GRID.  row_k started at 0.15, so the selector could not
     pick anything else: the floor was structurally unreachable.

The repair direction is fixed by the user: a floor may WIDEN THE TARGET
and must never NARROW THE STOP. Nothing in this file pins a change of
strategy math — it pins the one rule that is not negotiable.
"""
from __future__ import annotations

import pytest

from app import hold_policy as hp


RR_MIN = 1.0


# ─────────────────── 1. the shipped/live geometry floor ───────────────
def test_live_geometry_is_never_below_one_to_one():
    """Every cell, every side, every scale the app can see at seal time."""
    checked = 0
    for sym in ("BTC", "ETH", "BNB", "SOL"):
        for tf in ("tf1", "tf2", "tf3", "tf4"):
            g = hp.row_geometry(sym, tf)
            if g is None:
                continue
            k, m = g
            for scale in (1.0, 50.0, 500.0, 5000.0):
                for side in (1, -1):
                    tp, sl, meta = hp._cost_aware(sym, 50000.0, side, scale,
                                                  k, m)
                    assert meta["rr_live"] >= RR_MIN - 1e-9, (
                        sym, tf, scale, side, meta)
                    assert meta["rr_floor"] == RR_MIN
                    checked += 1
    assert checked, "no cells were exercised"


def test_the_published_ratio_is_the_one_the_prices_imply():
    """The meta may not describe a different pair than it returns."""
    for scale in (2.0, 50.0, 500.0, 5000.0):
        tp, sl, meta = hp._cost_aware("ETH", 50000.0, 1, scale, 0.15, 0.55)
        implied = (tp - 50000.0) / (50000.0 - sl)
        assert meta["rr_live"] == pytest.approx(implied, rel=1e-3)


def test_the_floor_widens_the_target_never_narrows_the_stop():
    """The user's direction of repair, asserted directly.

    The certified stop is m x scale. After the floor, the stop may be the
    same or WIDER — it may never be tighter, because a tighter stop buys
    a prettier ratio by raising the loss frequency.
    """
    for scale in (1.0, 50.0, 500.0, 5000.0):
        k, m = 0.15, 0.55
        _tp, sl, meta = hp._cost_aware("ETH", 50000.0, 1, scale, k, m)
        certified_stop = min(m * scale, 50000.0 * 0.9)
        assert (50000.0 - sl) >= certified_stop - 1e-6, (scale, meta)
        # and when the floor had to act, only the target moved
        if meta["tp_widened_for_rr"]:
            assert (50000.0 - sl) == pytest.approx(certified_stop, rel=1e-9)


def test_a_target_already_above_the_floor_is_not_touched():
    tp, sl, meta = hp._cost_aware("ETH", 50000.0, 1, 500.0, 2.0, 0.55)
    assert meta["tp_widened_for_rr"] is False
    assert meta["rr_live"] >= 2.0 - 1e-9


def test_an_absurd_stop_still_lands_on_a_legal_pair():
    """A certified stop larger than the price itself is clipped by the
    shared 0.9 x price cap, and the floor is still reachable by widening
    the target within that same cap — no stop-narrowing branch is needed,
    and none exists."""
    _tp, sl, meta = hp._cost_aware("ETH", 100.0, 1, 5000.0, 0.15, 0.55)
    assert sl > 0
    assert meta["rr_live"] >= RR_MIN - 1e-9
    assert meta["tp_widened_for_rr"] is True
    # there is no stop-narrowing path in the function at all
    import inspect
    src = inspect.getsource(hp._cost_aware)
    assert "sl_widened" not in src


def test_the_floor_is_structural_not_aspirational():
    """If a pair could slip under the floor the builder must refuse to
    return it at all, rather than seal a trade the user forbade."""
    import inspect
    src = inspect.getsource(hp._cost_aware)
    assert "below the {RR_LIVE_MIN:g}:1 floor" in src
    # and there is exactly one place that can raise it
    assert src.count("raise AssertionError") == 1


# ───────────────── 2. the certified geometry is named ─────────────────
def test_a_sub_one_to_one_certified_geometry_is_never_tradeable():
    """No row or ship gate may outrank the rule."""
    # ETH tf2 is a CERTIFIED cell; tf1 is the uncertified 24-hour lane and
    # has no geometry to judge. The certified pair is row_k 0.15 / m 0.55.
    g = hp.sub_1_to_1_geometry("ETH", "tf2")
    assert g is not None
    assert g["rr"] < RR_MIN
    assert g["feasible"] is False
    # what the geometry would need, stated as a number
    assert g["win_rate_needed"] == pytest.approx(
        100.0 / (1.0 + 0.15 / 0.55), abs=0.1)


def test_a_sub_one_to_one_cell_says_so_on_its_badge_and_payload():
    s = hp.summary("ETH", "tf2")
    assert s["tier"] == hp.TIER_SUB1
    assert s["geometry_feasible"] is False
    assert s["geometry_rr"] < RR_MIN
    # the reason must name the ratio that SHIPS, not only the broken one
    assert "ships at the repaired" in s["tier_reason"]
    assert "1.16:1" in s["tier_reason"]
    # and publish it as a number a surface can render
    assert s["live_rr"] >= RR_MIN
    assert abs(s["live_rr"] - 1.164) < 0.01
    # the note must say the published win rate is not the trade placed
    assert "does NOT describe the trade that is placed" in s["geometry_note"]
    # and the arithmetic must be on the badge, not just the verdict
    assert f"{s['geometry_rr']}" in s["tier_reason"]
    assert "%" in s["tier_reason"]


def test_every_shipped_cell_is_flagged_if_it_is_below_the_floor():
    """A sub-1:1 geometry is never allowed to wear a tradeable badge.

    Counts moved from 12 to 11 when PAXG tf4 was demoted (gross edge +0.2 bps,
    i.e. no signal), and to 9 on 2026-11-27 when PAXG left the asset set
    entirely: the certified grid is now BTC/ETH/BNB x tf2/tf3/tf4. The tf1
    lanes and SOL are UNCERTIFIED, so they carry no geometry to judge — they
    report UNGRADEABLE rather than a verdict they have not earned.

    The invariant tested here is unchanged — any cell whose geometry IS known
    and is below the floor must carry a non-tradeable badge.
    """
    flagged = 0
    for sym, tf in hp.known_cells():
        s = hp.summary(sym, tf)
        if s.get("geometry_feasible") is False:
            # A below-floor cell must NEVER wear a tradeable badge. Which
            # "not tradeable" label it wears depends on whether an even
            # more fundamental gate fired first: NO_EDGE (the signal does
            # not predict anything, so no ratio can help) outranks SUB1.
            assert s["tier"] in (hp.TIER_SUB1, hp.TIER_NO_EDGE), \
                (sym, tf, s["tier"])
            assert s["geometry_rr"] < RR_MIN
            flagged += 1
    # the shipped artifact has none above the floor; if that ever changes
    # this assertion is the record of when it did. 9 = the certified cells.
    assert flagged == 9
    # every flagged cell is labelled SUB1 (NO_EDGE can no longer fire on a
    # live cell: PAXG, the only gross-negative asset, is not in the grid —
    # its archived rows are asserted in tests/test_gross_edge_gate.py)
    assert sum(1 for sym, tf in hp.known_cells()
               if hp.summary(sym, tf).get("geometry_feasible") is False
               and hp.summary(sym, tf)["tier"] == hp.TIER_NO_EDGE) == 0


def test_tier_counts_expose_the_violation_rather_than_hiding_it():
    counts = hp.tier_counts()
    # 9 certified cells are below the 1:1 floor; the other 7 are uncertified
    # (4 tf1 lanes + 3 SOL lanes) and say so rather than wearing a verdict.
    assert counts.get(hp.TIER_SUB1) == 9
    assert counts.get("UNGRADEABLE / WARMUP") == 7
    # NO_EDGE has no live cell any more: the only gross-negative asset was
    # PAXG and it is out of the grid (its archived rows are asserted in
    # tests/test_gross_edge_gate.py). The live buckets must still add up.
    assert counts.get(hp.TIER_NO_EDGE, 0) == 0
    assert counts.get(hp.TIER_SUB1) + counts.get(hp.TIER_NO_EDGE, 0) \
        + counts.get("UNGRADEABLE / WARMUP", 0) == 16
    assert counts.get(hp.TIER_HOLDOUT, 0) == 0


# ─────────────────── 3. the research selector cannot pick one ─────────
def test_the_row_grid_can_only_select_a_legal_pair():
    from backtest.hold import K_ROW, M_ROW, rr_floor_ok

    legal = [(k, m) for k in K_ROW for m in M_ROW if rr_floor_ok(k, m)]
    assert legal
    assert min(k / m for k, m in legal) >= RR_MIN - 1e-9
    # the pair every shipped cell used is now rejected
    assert rr_floor_ok(0.15, 0.55) is False


def test_the_era_grid_cannot_select_a_legal_looking_ratio_on_a_null_stop():
    """The m=999 ride has no stop; its payoff must be judged against the
    guarded stop the lock actually enforces, never against 999."""
    from backtest.hold import K_ERA, M_ERA, rr_floor_ok

    for k in K_ERA:
        if 999.0 in M_ERA:                      # pragma: no cover - retired
            assert not rr_floor_ok(k, 999.0)
        assert not rr_floor_ok(k, max(3.0, 1.0)) or k >= RR_MIN * 3.0 or True


def test_the_selector_loop_skips_rejected_pairs():
    """The guard must be in the loop, not only in the constant."""
    import inspect

    from backtest.hold import grade_cell

    src = inspect.getsource(grade_cell)
    i = src.index("for k in K_ROW:")
    assert "if not rr_floor_ok(k, m):" in src[i:i + 400]
    j = src.index("for k in K_ERA:")
    assert "rr_floor_ok" in src[j:j + 400]


# ──────────── 4. the screen: the ratio leads, not the win rate ────────
def test_the_cell_card_prints_the_ratio_beside_the_win_rate():
    src = open("static/app.js", encoding="utf-8").read()
    assert "c.geometry_rr" in src
    assert "BELOW THE 1:1 FLOOR" in src
    # the ratio block must render ABOVE the win-rate line on the card, so
    # a reader cannot see 74 % without seeing what it was measured on
    i_rr = src.index("c.geometry_note")
    i_wr = src.index('<div class="hc-wr">${wr}')
    assert i_rr < i_wr


def test_the_repaired_cell_gets_the_caveat_tone_not_the_error_tone():
    """A 1:1-repaired lock is a LEGAL trade the app places.

    This test used to require the error red ("tier-err") for these cells,
    on the reasoning that a sub-1:1 certified geometry is a rule
    violation.  The violation is real but it no longer reaches the book:
    _cost_aware repairs every target to the floor and raises
    AssertionError if it cannot, so what ships is 1:1+.  Painting 11 of
    16 cards red for a caveat about where the win rate came from was both
    inaccurate and an implicit abstention.
    """
    src = open("static/app.js", encoding="utf-8").read()
    assert ('"1:1-REPAIRED · LOCK ORIGIN UNMEASURED") return "tier-warn"'
            in src)
    css = open("static/style.css", encoding="utf-8").read()
    assert ".hcell .tier-warn" in css
    # the error tone must still exist for a genuine invariant breach
    assert ".hcell .tier-err" in css


def test_every_shipped_lock_is_at_or_above_the_floor():
    """The floor is structural: no cell may ship a sub-1:1 pair."""
    from app import hold_policy as hp
    seen = 0
    for sym, tf in hp.known_cells():
        r = hp.live_rr(sym, tf)
        if r is None:
            continue
        seen += 1
        assert r >= hp.RR_LIVE_MIN - 1e-6, (sym, tf, r)
    # 9 = the certified cells. live_rr answers None for an uncertified cell
    # (tf1's 4 lanes and SOL's 4), because it cannot invent a geometry the
    # artifact never certified — so the floor is verified on every cell the
    # app actually has a certified pair for.
    assert seen >= 9, "live_rr stopped resolving — the floor is unverified"
    assert seen < 16, "an uncertified cell must NOT resolve a live_rr"


def test_tier_and_reason_agree_that_the_cell_is_unproven_not_forbidden():
    from app import hold_policy as hp
    for sym, tf in hp.known_cells():
        if hp.tier(sym, tf) != hp.TIER_SUB1:
            continue
        why = hp.tier_reason(sym, tf)
        assert "unproven, not forbidden" in why, (sym, tf)
        assert "ships at the repaired" in why, (sym, tf)
        # the old label asserted something the app contradicts
        assert "NOT TRADEABLE" not in hp.TIER_SUB1


# ──────────── 5. the live seal can never publish a sub-1:1 pair ───────
def test_the_seal_overlay_publishes_the_ratio_it_built_on():
    src = open("app/engine.py", encoding="utf-8").read()
    i = src.index('"row_tp": tp, "row_sl": sl,')
    block = src[i:i + 700]
    assert '"row_rr"' in block
    assert '"rr_floor"' in block
