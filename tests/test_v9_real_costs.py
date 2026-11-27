"""§18 REAL-CONDITIONS v9 — the user's three demands, pinned.

1. ALL Binance costs counted (commission + slippage + half-Roll
   spread per side → 28.9–37.4 bps round trip).
2. The live trade is the GUARDED ERA RIDE (stop ≤ 3σ, exit at
   TP/SL/side-flip/max-hold) — never a window-boundary settle.
3. Confidence issues at a HIGHER bar (floors [0.366, 0.404]) and the
   tradability gate charges the real round trip.
"""
from __future__ import annotations

import os
import tempfile

import pytest


# ─────────────────────────── 1) the cost stack ──────────────────────
def test_full_binance_cost_stack_is_shipped():
    from backtest.data import FRICTION_BPS, ROUND_TRIP_BPS
    # §18.1: commission 10 (spot taker VIP0) + slippage (1/1/2/5)
    # + half the Roll(1984) spread per side
    assert FRICTION_BPS == {"BTC": 14.436, "ETH": 14.904,
                            "BNB": 15.482, "PAXG": 18.700}
    for sym, rt in ROUND_TRIP_BPS.items():
        assert rt == pytest.approx(2.0 * FRICTION_BPS[sym], abs=2e-3)  # 4 dp rounding
    assert ROUND_TRIP_BPS == {"BTC": 28.872, "ETH": 29.808,
                              "BNB": 30.963, "PAXG": 37.400}
    # the harness and the app must bill the SAME stack
    from backtest import continuous as btc
    assert btc.FRICTION_BPS == FRICTION_BPS
    from app.horizon import ASSET_RT_BPS, asset_rt
    for sym, rt in ASSET_RT_BPS.items():
        assert asset_rt(sym) == pytest.approx(rt)
        assert rt == pytest.approx(2.0 * FRICTION_BPS[sym], abs=2e-3)  # 4 dp rounding
    assert asset_rt("UNKNOWN") == ASSET_RT_BPS["BTC"]   # honest fallback


def test_era_grid_is_guarded_and_rows_reach_deep_stops():
    from backtest.hold import K_ROW, M_ROW, K_ERA, M_ERA
    # §18.2: no unbounded rides in selection; rows may use deep stops
    # so the REAL win rate P(gross > RT) is reachable (probe5)
    assert 999.0 not in M_ERA
    assert max(M_ERA) == 3.0
    assert max(M_ROW) == 3.0
    assert min(M_ERA) >= 0.5
    assert K_ERA[0] == 0.6 and max(K_ERA) == 2.0
    # 2026-11-27: the row grid used to start at k=0.15, which let EVERY
    # shipped cell select a 0.27:1 first-touch geometry (k=0.15, m=0.55)
    # and certify a 74 % win rate that loses 0.030 R per trade. The user's
    # absolute rule is that reward:risk may never be below 1:1, so the grid
    # may not contain a lever that cannot pay for the smallest stop.
    from backtest.hold import rr_floor_ok
    assert K_ROW[0] == 0.5
    assert min(k / m for k in K_ROW for m in M_ROW if rr_floor_ok(k, m)) >= 1.0
    # and the guard agrees with the rule it enforces
    assert rr_floor_ok(0.15, 0.55) is False
    assert rr_floor_ok(1.0, 1.0) is True


# ─────────────────────── 2) the era-ride lock ───────────────────────
def _predlock(tmp_path):
    from app.predlock import PredictionJournal
    return PredictionJournal(os.path.join(tmp_path, "locks.json"))


def test_era_ride_seal_lives_past_the_window_and_flips(tmp_path):
    pl = _predlock(str(tmp_path))
    now = 1_700_000_000.0
    out = pl.seal("BTC", "tf2", direction=1, entry=100.0, tp=110.0,
                  sl=95.0, confidence=0.5, created_at=now,
                  max_hold_s=7 * 24 * 3600.0, gate_ok=True)
    assert out["locked"] is True and not out["already"]
    row = pl.rows[-1]
    assert row["settle_mode"] == "era_ride"
    assert row["expires_at"] == now + 7 * 24 * 3600.0

    # a window's worth of time passes with NO barrier touch and the
    # certified side UNCHANGED → the ride stays open (§18: a
    # window-end settle is the −RT trade probes measured)
    later = now + 45 * 60
    settled = pl.settle("BTC", "tf2", 101.0, now=later, side_now=1)
    assert settled == []
    assert pl.rows[-1]["status"] == "open"

    # the certified side FLIPS → exit at market, grade by strict sign
    settled = pl.settle("BTC", "tf2", 103.0, now=later + 60,
                        side_now=-1)
    assert len(settled) == 1
    r = settled[0]
    assert r["status"] == "won" and r["graded"] is True
    assert "flipped" in (r["settle_reason"] or "")
    assert r["bps"] == pytest.approx(300.0, abs=0.01)   # (103-100)/100


def test_era_ride_first_touch_beats_the_flip(tmp_path):
    pl = _predlock(str(tmp_path))
    now = 1_700_000_000.0
    pl.seal("ETH", "tf3", direction=1, entry=100.0, tp=110.0,
            sl=95.0, confidence=0.5, created_at=now,
            max_hold_s=7 * 24 * 3600.0, gate_ok=True)
    # SL touched AND side flipped in the same sample → first-touch
    # grades first (era_sim's barrier-before-boundary order)
    settled = pl.settle("ETH", "tf3", 94.0, high=101.0, low=94.5,
                        now=now + 60, side_now=-1)
    assert len(settled) == 1
    assert settled[0]["status"] == "lost"
    assert "SL touched" in (settled[0]["settle_reason"] or "")


def test_only_one_open_era_ride_per_symbol_tf(tmp_path):
    pl = _predlock(str(tmp_path))
    now = 1_700_000_000.0
    first = pl.seal("BTC", "tf2", direction=1, entry=100.0, tp=110.0,
                    sl=95.0, confidence=0.5, created_at=now,
                    max_hold_s=7 * 24 * 3600.0, gate_ok=True)
    assert first["locked"] and not first["already"]
    second = pl.seal("BTC", "tf2", direction=1, entry=101.0, tp=111.0,
                     sl=96.0, confidence=0.5, created_at=now + 60,
                     max_hold_s=7 * 24 * 3600.0, gate_ok=True)
    assert second["locked"] is True and second["already"] is True
    assert any("already open" in w for w in second["why"])
    # a settled ride releases the slot — the next seal must come from
    # the NEXT window (one lock per window stays a rule)
    pl.settle("BTC", "tf2", 112.0, now=now + 120, side_now=1)
    W_S = 44 * 60
    nxt = float((int(now) // W_S + 1) * W_S + 10)
    third = pl.seal("BTC", "tf2", direction=1, entry=112.0, tp=120.0,
                    sl=108.0, confidence=0.5, created_at=nxt,
                    max_hold_s=7 * 24 * 3600.0, gate_ok=True)
    assert third["locked"] and not third["already"]


def test_window_locks_still_expire_the_old_way(tmp_path):
    """No max_hold → legacy window semantics, zero behaviour change."""
    pl = _predlock(str(tmp_path))
    now = 1_700_000_000.0
    pl.seal("BTC", "tf1", direction=1, entry=100.0, tp=110.0,
            sl=95.0, confidence=0.5, created_at=now, gate_ok=True)
    row = pl.rows[-1]
    assert row["settle_mode"] == "window"
    assert row["expires_at"] == now + 22 * 60
    settled = pl.settle("BTC", "tf1", 101.0, now=now + 23 * 60,
                        side_now=-1)     # flip must NOT close a window lock
    assert len(settled) == 1
    assert settled[0]["status"] == "won"     # sign settle at expiry


# ───────────────────── 3) confidence + gates ────────────────────────
def test_issue_bar_raises_every_asset_floor():
    from app.config import (CONFIDENCE_FLOOR, ISSUE_BAR, ASSET_PROFILES,
                            asset_issue_floor)
    assert CONFIDENCE_FLOOR == 0.35 and ISSUE_BAR == 1.10
    floors = {sym: asset_issue_floor(sym) for sym in ASSET_PROFILES}
    for sym, fl in floors.items():
        # §18.2.5: issued floors land in [0.366, 0.404] — uniformly
        # above the old [0.315, 0.385] band (BNB's gentle 0.95 prior
        # starts at 0.3657, still above the old 0.315)
        assert 0.365 <= fl <= 0.4043, (sym, fl)
    assert floors["BTC"] == pytest.approx(0.385)      # fm 1.00
    assert floors["BNB"] == pytest.approx(0.3657)     # fm 0.95 (gentlest)
    # ETH 1.20 and PAXG 1.10 both clamp to 1.05 — no starvation
    assert floors["ETH"] == floors["PAXG"] == pytest.approx(0.4042)


def test_horizon_gate_charges_the_real_round_trip():
    from app.horizon import classify, asset_rt, DEFAULT_FRICTION_BPS
    assert DEFAULT_FRICTION_BPS == 28.872
    # σ must clear 3× the REAL round trip to be tradable
    btc_rt = asset_rt("BTC")
    assert classify(3 * btc_rt, btc_rt)["state"] == "tradable"
    assert classify(2.5 * btc_rt, btc_rt)["state"] == "marginal"
    assert classify(1.5 * btc_rt, btc_rt)["state"] == "untradable"
    # and the old default (12 bps) would have LIED: 40 bps of σ looks
    # tradable at 12 but cannot pay 28.9
    assert classify(40.0, 12.0)["state"] == "tradable"
    assert classify(40.0, btc_rt)["state"] == "untradable"


def test_v8_net_reference_is_pinned_before_v9():
    """Gate (ii)'s v8 reference — computed before the v9 run (§18.3)."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..",
                        "backtest", "results", "v8_net_reference.json")
    ref = json.load(open(path))
    assert ref["n"] == 12
    assert ref["mean_row_wr_net"] == pytest.approx(0.0080, abs=1e-4)
