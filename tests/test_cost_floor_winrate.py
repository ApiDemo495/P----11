#!/usr/bin/env python3
"""The cost floor: a "win" must pay the round trip, or it is not a win.

Pins the repair that stops the app reporting an ~85 % win rate while a
winning trade still loses money (target inside the Binance round trip).
"""
from __future__ import annotations

from app import hold_policy as hp


def test_round_trip_uses_authoritative_full_stack():
    from backtest.data import FRICTION_BPS
    for sym in ("BTC", "ETH", "BNB", "SOL"):
        assert hp.round_trip_bps(sym) == 2.0 * FRICTION_BPS[sym]


def test_min_tp_floor_is_stricter_than_the_round_trip():
    rt = hp.round_trip_bps("BTC")
    assert hp.MIN_TP_OVER_RT > 1.0
    assert hp.tp_pays_rt("BTC", 1.5 * rt) is True
    assert hp.tp_pays_rt("BTC", 0.5 * rt) is False


def test_tp_pays_rt_never_guesses_when_distance_unknown():
    assert hp.tp_pays_rt("BTC", None) is None


def test_certified_short_timeframe_target_is_widened_to_pay():
    """BTC tf1's certified k=0.15 target lands inside the round trip;
    the live builder must widen it to the floor (never book the tighter
    target that produces the decorative hit rate)."""
    entry, scale, side = 100.0, 2.0, 1
    tp, sl, meta = hp._cost_aware("BTC", entry, side, scale, 0.15, 0.55)
    tp_bps = (tp - entry) / entry * 1e4 * side
    assert meta["tp_widened_for_cost"] is True
    assert meta["tp_pays_round_trip"] is False
    assert tp_bps >= meta["min_tp_bps"] - 1e-6
    # the stop is untouched by the cost floor
    assert abs((entry - sl) / entry * 1e4 - 0.55 * scale / entry * 1e4) < 1e-6


def test_a_target_that_already_pays_the_round_trip_is_not_widened_for_cost():
    """The COST floor leaves a sufficiently wide target alone.

    2026-11-27: this case used to reuse the certified k=0.15 in a way that
    only cleared the cost floor because the STOP was enormous. It now also
    has to clear the reward:risk floor, so it is expressed as what it is:
    a geometry whose target is already pays-the-round-trip wide AND pays
    its stop. The cost-only assertion is unchanged.
    """
    entry, scale, side = 100.0, 40.0, 2.0
    tp, _sl, meta = hp._cost_aware("BTC", entry, side, scale, 2.0, 0.55)
    assert meta["tp_widened_for_cost"] is False
    assert meta["tp_bps_used"] == meta["tp_bps_certified"]
    assert meta["tp_widened_for_rr"] is False
    assert meta["rr_live"] >= 1.0


def test_widening_only_lowers_the_hit_rate_never_raises_it():
    """A wider target cannot be touched more often than a tighter one —
    this is why the floor cannot be used to flatter a win rate."""
    entry, side = 100.0, 1
    for scale in (0.5, 2.0, 10.0, 40.0):
        tight = 0.15 * 0.55 * scale
        _tp, _sl, meta = hp._cost_aware("BTC", entry, side, scale, 0.15, 0.55)
        used = meta["tp_bps_used"] / 1e4 * entry
        assert used >= tight - 1e-9


def test_summary_pairs_win_rate_with_its_cost_context():
    s = hp.summary("BTC", "tf2")   # tf1 is the 24h lane, uncertified
    assert s["win_rate"] is not None
    # the headline is labelled gross, never money
    assert s["win_rate_label"].startswith("gross")
    assert "does NOT include" in s["win_rate_label"]
    # the authoritative round trip wins over a stale artifact constant
    assert s["round_trip_bps"] == hp.round_trip_bps("BTC")
    assert s["min_tp_bps"] == hp.MIN_TP_OVER_RT * s["round_trip_bps"]
    assert "round trip" in s["cost_note"]


def test_warmup_cell_claims_no_win_rate():
    """An uncertified cell reports status 'unknown' and NO win rate.

    'unknown' and 'ungradeable' are deliberately different words: the first
    says nothing has been measured at this duration, the second says a
    measurement exists and failed. SOL's 24-hour lane is the former.
    """
    s = hp.summary("SOL", "tf1")
    assert s["status"] == "unknown"
    assert s["win_rate"] is None
