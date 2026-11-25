"""Pins for the no-abstention win-rate dial (``app/winrate_dial.py``).

The dial exists to answer one request honestly: raise the win rate while
still taking every window. These tests pin the three properties that
make it trustworthy:

  1. it never abstains — a lockable geometry comes back for every
     measured cell, at any requested win rate, however unrealistic;
  2. it never trades a win rate the live policy would refuse to lock —
     reward:risk >= MIN_REWARD_RISK and target >= MIN_TP_OVER_RT x round
     trip, always;
  3. it always states the price — a measured net bps/trade figure rides
     with every dialled win rate, so a higher win rate can never be
     presented as free.
"""
from __future__ import annotations

import json
import os

import pytest

from app import hold_policy, winrate_dial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "backtest", "results")


@pytest.fixture(scope="module")
def doc():
    winrate_dial.reload()
    return winrate_dial._doc()


def test_frontier_artifact_is_present_and_measured(doc):
    assert doc, "backtest/results/winrate_frontier.json missing — rerun " \
                "backtest/winrate_frontier.py"
    cells = winrate_dial.known_cells()
    assert len(cells) >= 20, cells
    for key in ("BTC|tf4", "ETH|tf1", "ETH|2d", "BNB|1d", "PAXG|2d"):
        assert key in cells, key
    cell = winrate_dial.frontier("BTC", "tf4")
    assert cell["rt_bps"] > 0 and cell["n"] > 100
    assert cell["sides"], "no measured sides"


@pytest.mark.parametrize("cell", winrate_dial.known_cells())
def test_dial_never_abstains(cell):
    """Every measured cell yields a lockable geometry at every ask."""
    symbol, tf = cell.split("|")
    for target in (None, 0.01, 0.5, 0.99, 1.0 - 1e-9):
        plan = winrate_dial.dial(symbol, tf, target_wr=target, direction=-1)
        assert plan["measured"] is True
        assert plan["k"] and plan["m"], plan
        assert plan["k"] > 0 and plan["m"] > 0
        # the request is answered with a trade, even when unattainable
        assert plan["direction"] == -1, "dial must not touch direction"
        assert plan["expected_wr"] is not None


@pytest.mark.parametrize("cell", winrate_dial.known_cells())
def test_dial_respects_the_live_guardrails(cell):
    symbol, tf = cell.split("|")
    for plan in (winrate_dial.dial(symbol, tf, target_wr=0.9),
                 winrate_dial.dial(symbol, tf)):
        assert plan["k"] >= hold_policy.MIN_REWARD_RISK, plan
        assert plan["min_tp_over_rt"] == hold_policy.MIN_TP_OVER_RT
        # measured, and the interval brackets the point estimate
        assert plan["n"] >= 10
        lo, hi = plan["ci95"]
        assert lo <= plan["expected_wr"] <= hi


def test_unrealistic_target_is_reported_not_hidden():
    plan = winrate_dial.dial("BTC", "tf4", target_wr=0.95, direction=1)
    assert plan["achievable"] is False
    assert plan["k"] is not None and plan["m"] is not None   # still trades
    assert plan["ceiling_wr"] is not None
    assert plan["ceiling_wr"] < 0.95
    assert "above the lockable ceiling" in plan["reason"]


def test_price_is_always_quoted():
    for cell in winrate_dial.known_cells():
        symbol, tf = cell.split("|")
        plan = winrate_dial.dial(symbol, tf, target_wr=0.5)
        assert plan["price_bps"] is not None
        assert isinstance(plan["price_bps"], float)
        assert "round trip" in plan["honest_note"]


@pytest.mark.parametrize("cell", ["BTC|tf4", "ETH|tf4", "ETH|2d", "BNB|1d"])
def test_win_rate_falls_as_the_target_widens(cell):
    """The dial's core claim: win rate is a TP:SL choice, monotone in k."""
    symbol, tf = cell.split("|")
    rows = winrate_dial.addressable(symbol, tf)
    tight = [r for r in rows if r["k"] == 0.25]
    wide = [r for r in rows if r["k"] == 3.0]
    assert tight and wide
    assert max(r["wr"] for r in tight) > max(r["wr"] for r in wide)


def test_ceiling_is_the_lockable_maximum_not_the_artifact_maximum():
    """k < 0.25 rows win more and are refused — that is the guardrail."""
    cell = winrate_dial.frontier("ETH", "2d")
    all_rows = [r for s in cell["sides"].values() for r in s["frontier"]]
    raw_max = max(r["wr"] for r in all_rows)
    ceil = winrate_dial.ceiling("ETH", "2d")
    assert ceil["wr"] < raw_max, "lottery-shaped rows must not be offered"
    assert ceil["k"] >= hold_policy.MIN_REWARD_RISK


def test_build_row_bars_dial_clears_the_cost_floor():
    for symbol, tf in (("BTC", "tf4"), ("ETH", "tf1"), ("BTC", "tf1")):
        plan = winrate_dial.dial(symbol, tf, target_wr=0.0)
        out = winrate_dial.build_row_bars_dial(symbol, tf, 100.0, 1, 2.0,
                                               target_wr=0.0)
        assert out is not None
        tp, sl, meta = out
        assert tp > 100.0 > sl, "long geometry must straddle the entry"
        assert meta["geometry_source"] == "dial"
        assert meta["tp_bps_used"] >= meta["min_tp_bps"] - 1e-9
        assert meta["tp_widened_for_cost"] in (True, False)
        assert meta["dial"]["k"] == plan["k"]


def test_build_row_bars_dial_straddles_both_directions():
    tp_l, sl_l, _ = winrate_dial.build_row_bars_dial("BTC", "tf4", 100.0, 1, 2.0)
    tp_s, sl_s, _ = winrate_dial.build_row_bars_dial("BTC", "tf4", 100.0, -1, 2.0)
    assert tp_l > 100.0 > sl_l
    assert tp_s < 100.0 < sl_s
    # degenerate input only: a zero side is the one thing that yields None
    assert winrate_dial.build_row_bars_dial("BTC", "tf4", 100.0, 0, 2.0) is None
    assert winrate_dial.build_row_bars_dial("BTC", "tf4", 0.0, 1, 2.0) is None


def test_unmeasured_cell_falls_back_to_the_certified_geometry():
    """No frontier for the cell is not a reason to skip the window."""
    plan = winrate_dial.dial("BLUR", "tf4")
    assert plan["measured"] is False
    assert "certified hold geometry" in plan["reason"]
    # a cell that the certified artifact manages still produces a lock
    g = hold_policy.row_geometry("BTC", "tf4")
    if g:
        out = winrate_dial.build_row_bars_dial("BTC", "2d", 100.0, 1, 2.0)
        assert out is not None                      # day cell: measured
    out = winrate_dial.build_row_bars_dial("BTC", "tf1", 100.0, 1, 2.0)
    assert out is not None


def test_cost_curve_shows_the_one_free_lunch():
    """Cheaper fills raise the win rate AND the P&L — no window skipped."""
    curve = winrate_dial.cost_curve("BTC", "tf4")
    assert curve, "winrate_cost_sensitivity.json missing for BTC|tf4"
    for side in {r["side"] for r in curve}:
        for k in {r["k"] for r in curve if r["side"] == side}:
            pts = sorted([r for r in curve
                          if r["side"] == side and r["k"] == k],
                         key=lambda r: r["rt_bps"])
            assert pts, (side, k)
            # pts are ordered cheapest round trip first: the win rate
            # never FALLS as the fill gets cheaper (it only ever rises,
            # because the cost floor also stops widening the target)
            for a, b in zip(pts, pts[1:]):
                assert b["wr"] <= a["wr"] + 1e-9, (side, k, a, b)
            # ... and the net P&L improves with the cheaper round trip
            assert pts[0]["ev_bps"] > pts[-1]["ev_bps"], (side, k)


def test_day_cells_are_computed_but_not_wired():
    plan = winrate_dial.day_cell_plan("ETH", "2d", direction=1, target_wr=0.6)
    assert plan["measured"] is True
    assert plan["wired"] is False
    assert "no live path" in plan["wired_reason"]
    assert plan["expected_wr"] >= 0.6


def test_hold_policy_exposes_the_dial_without_importing_it():
    plan = hold_policy.winrate_dial_plan("ETH", "tf4", target_wr=0.5,
                                         direction=1)
    assert plan["measured"] is True
    assert plan["k"] >= hold_policy.MIN_REWARD_RISK


def test_short_horizons_cannot_reach_a_high_win_rate():
    """The honest reason a 22-min cell will never show 60 % payable."""
    for symbol in ("BTC", "ETH"):
        for tf in ("tf1", "tf2"):
            ceil = winrate_dial.ceiling(symbol, tf)
            assert ceil["wr"] < 0.30, (symbol, tf, ceil)
    # ... while day-scale cells can
    assert winrate_dial.ceiling("ETH", "2d")["wr"] > 0.80


def test_no_abstention_artifact_has_no_no_trade_decision():
    """The dial's measurement run must contain zero skipped windows."""
    wf = os.path.join(RESULTS, "winrate_noabstain_wf.json")
    if not os.path.exists(wf):
        pytest.skip("walk-forward artifact absent")
    doc = json.load(open(wf))
    assert doc["design"]["min_reward_risk"] == hold_policy.MIN_REWARD_RISK
    assert doc["design"]["min_tp_over_rt"] == hold_policy.MIN_TP_OVER_RT
    for symbol, cell in doc["cells"].items():
        # every block graded, no empty block, no "skipped" bucket
        assert len(cell["oos_blocks"]) == doc["design"]["blocks"]
        assert all(b["n"] > 0 for b in cell["oos_blocks"]), symbol
        assert "abstain" not in json.dumps(cell).lower()
