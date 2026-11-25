"""The graded record must be able to say "this lost".

Every defect pinned here was found by re-deriving from files the app itself
wrote (data/locks.json, data/history/history_2026-11-24.json), and each one
moved the record in the optimistic direction only.

  * the live ledger booked 0 stop-losses out of 80 snapshots (it read an
    event type the engine never emits);
  * a lane that stopped out and later printed TP1 was graded a WIN;
  * every SHORT printed pnl_pct 0.0 (the formula was long-only);
  * 11 of 16 live lanes carried rung 1 on the wrong side of their own entry.
"""
from __future__ import annotations

import pytest

from app.grading import resolve
from app.contracts import rung_level, build_ladder


# ══════════════ first-touch resolution (app/grading.py) ══════════════

def test_sl_crossed_is_read_as_a_stop():
    """The engine emits "sl_crossed"; the ledger only looked for "sl_hit"."""
    v = resolve([{"ts": 10.0, "type": "sl_crossed", "sl": 90.0}])
    assert v["loss"] is True and v["win"] is False and v["resolved"] is True


def test_stop_then_target_is_a_loss_not_a_win():
    """The SL SAFE branch keeps the cycle alive; TP1 could land later."""
    evs = [{"ts": 10.0, "type": "sl_crossed"},
           {"ts": 99.0, "type": "target_hit", "level": 120.0}]
    v = resolve(evs, tp=120.0)
    assert v["loss"] is True and v["win"] is False


def test_target_then_stop_is_a_win():
    evs = [{"ts": 10.0, "type": "target_hit", "level": 120.0},
           {"ts": 99.0, "type": "sl_crossed"}]
    v = resolve(evs, tp=120.0)
    assert v["win"] is True and v["loss"] is False


def test_same_tick_tie_goes_to_the_stop():
    evs = [{"ts": 10.0, "type": "target_hit", "level": 120.0},
           {"ts": 10.0, "type": "sl_crossed"}]
    v = resolve(evs, tp=120.0)
    assert v["loss"] is True


def test_intermediate_rung_is_not_the_contract_target():
    """TP1 is not a win; only the contract target resolves the position."""
    evs = [{"ts": 10.0, "type": "target_hit", "level": 105.0}]
    v = resolve(evs, tp=120.0)
    assert v["resolved"] is False and v["win"] is False


def test_unresolved_window_is_not_a_win():
    assert resolve([])["resolved"] is False


# ══════════════ canonical rung placement (app/contracts.py) ══════════════

def test_rung_is_never_beyond_the_stop_for_a_short():
    """BNB tf4 as stored in data/locks.json: short 443.06, stop 466.46, but
    the stored rung 1 was 493.50 — past the stop."""
    lvl = rung_level(443.064209, -1, 402.564153, [60, 120, 180, 247],
                     horizon_minutes=247, k=1, sl=466.45574)
    assert lvl < 443.064209, "a short's rung must sit BELOW its entry"
    assert lvl < 466.45574, "and inside the stop"


def test_rung_is_never_beyond_the_stop_for_a_long():
    lvl = rung_level(2646.612892, 1, 2672.528525, [60, 120, 180, 247],
                     horizon_minutes=247, k=1, sl=2620.697259)
    assert lvl > 2646.612892
    assert lvl > 2620.697259


def test_rung_refresh_matches_the_frozen_ladder():
    """The refresh must reproduce the ladder the lock was built from."""
    e, tp, sl = 67202.420382, 71708.94749, 64650.74448
    mins = [30, 40, 90, 130]
    rows = build_ladder(e, 1, tp, mins, horizon_minutes=130)
    assert rows, "the canonical ladder must build for a valid contract"
    for k, row in enumerate(rows, start=1):
        got = rung_level(e, 1, tp, mins, horizon_minutes=130, k=k, sl=sl)
        assert got == pytest.approx(row["level"], rel=1e-9)


def test_ladder_is_monotone_and_ends_at_the_horizon_target():
    e, tp = 100.0, 120.0
    rows = build_ladder(e, 1, tp, [5, 10, 15, 22], horizon_minutes=22)
    lv = [r["level"] for r in rows]
    assert lv == sorted(lv) and len(set(lv)) == len(lv)
    assert lv[-1] == pytest.approx(tp)
    assert all(e < x <= tp for x in lv)
