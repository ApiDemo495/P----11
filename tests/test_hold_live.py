"""Live hold-policy wiring — ZERO-parity guarantee + cell timers.

Enforces, on synthetic bar arrays (deterministic, no network, no
cache):

  1. side parity      — app.hold_policy side series ==
                        backtest.hold.build_sides (element-wise, 0 mismatches)
  2. era parity       — app.hold_era.track_eras == backtest.hold.era_sim
                        (deep equality — same records, same floats)
  3. row parity       — predlock hold-rule settlement == continuous
                        _simulate outcome on identical windows (ties,
                        flats, touches)
  4. seal geometry    — the engine seals the certified decoupled row
                        (k_row, m_row) TP/SL with hold evidence
  5. cell timers      — one ANALYSE click locks that (asset, tf) only
                        until its tf-minute timer expires; other cells
                        stay free (16 total)
  6. manual default   — auto_predict defaults to False everywhere
  7. committee        — bounded ±5 % confidence contribution, side held
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time

import numpy as np
import pytest

from app import hold_policy as hp
from app import hold_era as he
from backtest import hold as bh
from backtest.continuous import _scale_at, _simulate


# ─────────────────────────── synthetic tape ──────────────────────────
def _bars(n=260_000, step=300, seed=11):
    """2.5 years of 5-minute bars (covers mom525600 = 365 d)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * step + 1_600_000_000
    rets = rng.normal(0, 0.0012, n)
    c = 50_000.0 * np.exp(np.cumsum(rets))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    v = np.ones(n)
    return t, o, h, l, c, v, step


def _windows(bars, w_min):
    """Same window construction as hold.grade_cell / _series_for."""
    t, o, h, l, c, v, step = bars
    step_s = int(step)
    W = w_min * 60
    t0 = int(math.ceil(int(t[0]) / W) * W)
    t1 = int(t[-1]) + step_s
    starts = np.arange(t0, t1 - W + 1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts - step_s, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step_s, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    return starts[ok], i_entry[ok], i_exit[ok]


@pytest.fixture(scope="module")
def big_bars():
    return _bars()


# ─────────────────────────── 1) side parity ──────────────────────────
@pytest.mark.parametrize("side_name", [
    "up", "down", "mom15", "mom1440", "mom43200",
    "mom129600", "mom259200", "mom525600", "mom44h",
    "momMaj", "momMean", "revW", "revW3", "revW5",
    "revWh0.25", "revWh0.5", "revWh0.75", "lr_revW",
])
def test_side_parity_zero_mismatches(big_bars, side_name):
    t, o, h, l, c, v, step = big_bars
    starts, i_entry, i_exit = _windows(big_bars, 22)
    moves = c[i_exit] - c[i_entry]
    sides = bh.build_sides(t, o, h, l, c, v, step, starts, i_entry,
                           moves, 22)
    if side_name in ("lr_revW", "lrE_revW"):
        # composites live OUTSIDE build_sides — grade_cell applies them
        # after the OOF logistic runs (same functions, same order)
        lr_s, lr_sc = bh.lr_sides_oof(t, o, h, l, c, v, step, starts,
                                      i_entry, moves, 22)
        sides = bh.composite_sides(sides, lr_s, lr_sc)
    ref = sides[side_name]
    got = hp._series_for(side_name, big_bars, 22)
    assert len(got) == len(ref)
    mism = int(np.sum(got != ref))
    assert mism == 0, f"{side_name}: {mism} parity mismatches"


def test_certified_sides_all_reachable(big_bars):
    """Every side selected in hold_wr.json exists in build_sides AND is
    reproduced by the live series builder."""
    doc = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "backtest", "results", "hold_wr.json")))
    names = {c["selected"]["side"] for c in doc.values()
             if isinstance(c, dict) and c.get("status") == "ok"}
    assert names <= {"up", "down", "mom44h", "momMaj", "momMean",
                     "revW", "revW3", "revW5", "lr_revW", "lrE_revW"} | {
        f"mom{m}" for m in (7200, 10080, 14400, 43200,
                            129600, 259200, 525600)} | {
        f"revWh{t}" for t in (0.25, 0.5, 0.75)}
    for n in names:
        s = hp._series_for(n, big_bars, 247)
        assert np.all(np.isin(s, (-1, 1))), n


# ─────────────────────────── 2) era parity ───────────────────────────
@pytest.mark.parametrize("side_name,k,m", [
    ("mom259200", 0.6, 999.0),      # ride (the winning geometry)
    ("mom129600", 0.6, 3.0),        # barrier era
    ("down", 1.0, 1.0),             # constant side, tight barriers
])
def test_era_parity_zero_mismatches(big_bars, side_name, k, m):
    t, o, h, l, c, v, step = big_bars
    starts, i_entry, i_exit = _windows(big_bars, 247)
    moves = c[i_exit] - c[i_entry]
    side = hp._series_for(side_name, big_bars, 247)
    scale = _scale_at(t, c, starts - int(step),
                      max(2, int(round(247 / (int(step) / 60.0)))))
    ref = bh.era_sim(side, starts, i_entry, i_exit, c[i_entry], scale,
                     t, h, l, c, k, m, 12.0)
    got = he.track_eras(side, starts, i_entry, i_exit, c[i_entry],
                        scale, t, h, l, c, k, m, 12.0)
    assert len(got) == len(ref) and len(ref) > 0
    for a, b in zip(got, ref):
        assert a["start_i"] == b["start_i"]
        assert a["n_windows"] == b["n_windows"]
        assert a["outcome"] == b["outcome"]
        assert a["pnl_gross"] == b["pnl_gross"]     # bit-identical
        assert a["pnl_net"] == b["pnl_net"]


def test_era_friction_is_never_relaxed():
    """Every closed era pays exactly 2× the certified per-side bps."""
    recs = he.track_eras(np.array([-1, 1, -1]), np.array([0, 1, 2]),
                         np.array([10, 20, 30]), np.array([15, 25, 35]),
                         np.array([100.0, 101.0, 102.0]),
                         np.array([1.0, 1.0, 1.0]),
                         np.arange(60), np.full(60, 101.0),
                         np.full(60, 99.0), np.full(60, 100.0),
                         0.6, 999.0, 12.0)
    assert len(recs) == 3
    for r in recs:
        assert abs((r["pnl_gross"] - r["pnl_net"]) - 0.0024) < 1e-12


# ─────────────────────────── 3) row parity ───────────────────────────
def test_live_row_settlement_marks_same_bar_both_touch_ambiguous(tmp_path, monkeypatch):
    """Live OHLC does not expose intrabar order; never guess SL-first.

    Backtest SL-first remains a pessimistic bound, not live truth.
    """
    from app import predlock
    monkeypatch.setattr(predlock, "JOURNAL_FILE", str(tmp_path / "pl.json"))
    j = predlock.PredictionJournal(str(tmp_path / "pl.json"))
    created = 1_700_000_000.0
    row = j.seal("BTC", "tf1", direction=1, entry=100.0,
                 tp=105.0, sl=95.0, confidence=0.7, created_at=created,
                 evidence={"hold_policy": {"hold_rules": True}})
    assert row["locked"]
    closed = j.settle("BTC", "tf1", 101.0, now=created + 120,
                      series=[(created + 60, 106.0, 94.0)])
    assert len(closed) == 1
    got = closed[0]
    assert got["status"] == got["outcome"] == "ambiguous"
    assert got["graded"] is False and got["bps"] is None
    assert "cannot establish order" in got["settle_reason"]


def test_live_row_settlement_single_touch_and_flat_expiry(tmp_path, monkeypatch):
    from app import predlock
    monkeypatch.setattr(predlock, "JOURNAL_FILE", str(tmp_path / "pl.json"))
    j = predlock.PredictionJournal(str(tmp_path / "pl.json"))
    created = 1_700_000_000.0
    row = j.seal("BTC", "tf1", direction=1, entry=100.0,
                 tp=105.0, sl=95.0, confidence=0.7, created_at=created,
                 evidence={"hold_policy": {"hold_rules": True}})
    j.settle("BTC", "tf1", 105.0, now=created + 120,
             series=[(created + 60, 106.0, 99.0)])
    got = next(r for r in j.rows if r["id"] == row["id"])
    assert got["outcome"] == "won" and got["bps"] == 500.0

    # A bounded hold with no movement at expiry is a push, never a loss.
    j2 = predlock.PredictionJournal(str(tmp_path / "pl2.json"))
    row2 = j2.seal("BTC", "tf1", direction=1, entry=100.0,
                   tp=105.0, sl=95.0, confidence=0.7, created_at=created,
                   max_hold_s=60,
                   evidence={"hold_policy": {"hold_rules": True}})
    j2.settle("BTC", "tf1", 100.0, now=created + 61)
    got2 = next(r for r in j2.rows if r["id"] == row2["id"])
    assert got2["outcome"] == "push" and got2["graded"] is False


def test_engine_never_settles_from_forming_bar_extremes():
    """A still-forming candle and its live quote cannot trigger TP/SL."""
    from tests.test_engine import make_engine
    eng, feed, _ = make_engine()
    minute = int(time.time() // 60) * 60
    created = minute - 10
    st = feed.state("BTC")
    st.push_bar(minute, 106.0, 10.0, open_=100.0,
                high=107.0, low=99.0, source="unknown")
    st.live_price = 106.0
    sealed = eng.predlocks.seal(
        "BTC", "tf1", direction=1, entry=100.0, tp=105.0, sl=95.0,
        confidence=0.7, created_at=created, gate_ok=True,
        evidence={"hold_policy": {"hold_rules": True}})
    assert sealed["locked"]
    assert eng.settle_prediction_locks(now=minute + 10) == []
    row = next(r for r in eng.predlocks.rows if r["id"] == sealed["id"])
    assert row["status"] == "open" and row["outcome"] is None


# ─────────────────────────── 4) seal geometry ────────────────────────
def test_engine_seal_uses_certified_era_ride_geometry():
    # §18: the live lock is the GUARDED ERA RIDE (SL = m_era·scale,
    # TP = k_era·SL, settles at TP/SL/side-flip/max-hold — never at
    # the window boundary).  BNB tf1 (22min) is an honest
    # no_feasible_config on the 9-year tape (nothing certified to
    # seal); BNB tf2 stays gradeable.  Expectations derive from the
    # shipped summary so the pin tracks the artifact, and a legacy
    # m=999 cell (v8 ride) must fall back to the ROW ladder + window
    # settlement instead of sealing an unbounded stop.
    from tests.test_engine import make_engine
    eng, feed, _ = make_engine()
    final = {"direction": -1, "entry": 100.0, "confidence": 0.72,
             "gate_ok": True, "recheck": {"passed": True},
             "tps": [], "sl": None, "atr": 50.0, "voters": {},
             "prediction": {}}
    out = eng._seal_analysis_lock("BNB", "tf2", final, {"raws": {}},
                                  {"atr": 50.0}, None)
    assert out.get("locked") is True, out
    row = next(r for r in eng.predlocks.rows if r["id"] == out["id"])
    pol = hp.summary("BNB", "tf2")
    scale = 50.0                       # atr proxy on the thin DummyFeed
    era_m = float(pol["era_m"] or 0)
    if 0 < era_m < 999.0:
        # guarded era → era-ride seal, barriers capped inside the
        # price (warmup ATR proxies can be fat)
        sl_exp = min(era_m * scale, 100.0 * 0.9)
        tp_exp = min(float(pol["era_k"]) * sl_exp, 100.0 * 0.9)
        assert row.get("settle_mode") == "era_ride"
        assert row.get("max_hold_s") == hp.ERA_MAX_HOLD_S
    else:
        # legacy v8 ride (m=999) → ROW geometry, bounded ride mode
        sl_exp = min(pol["row_m"] * scale, 100.0 * 0.9)
        tp_exp = min(pol["row_k"] * sl_exp, 100.0 * 0.9)
        assert row.get("settle_mode") == "era_ride"
        assert row.get("max_hold_s") == hp.ERA_MAX_HOLD_S
    assert abs(row["sl"] - (100.0 + sl_exp)) < 1e-9      # side −1
    assert abs(row["tp"] - (100.0 - tp_exp)) < 1e-9
    hold = row["evidence"]["hold_policy"]
    assert hold["hold_rules"] is True
    assert hold["tier"] == hp.tier("BNB", "tf2")
    assert hold["win_rate"] == pol["win_rate"]
    # the certified side rule, e.g. hold:mom525600 (v6-era BNB tf1
    # was the "down" side — the rule string tracks the shipped side)
    from app.mandate_cfg import direction_rule
    assert hold["rule"] == direction_rule("BNB", "tf2")
    assert "warmup proxy" in hold["scale_source"]
    # the round trip shown on the seal tracks the SHIPPED artifact's
    # cost stack (§18.1 once v9 ships; 24.0 while v8 is shipped)
    assert (row["evidence"]["hold_policy"]["round_trip_friction_bps"]
            == pol["round_trip_bps"])


# ─────────────────────────── 5) cell timers ──────────────────────────
def test_cell_timer_is_per_asset_tf_only(tmp_path, monkeypatch):
    from app.hold_policy import CellLockBook
    monkeypatch.setenv("HOLD_CELL_LOCKS", "1")
    book = CellLockBook(str(tmp_path / "locks.json"))
    monkeypatch.setattr(hp, "CELL_LOCKS", book)
    # click BTC tf1 → only BTC|tf1 sleeps
    r = book.start("BTC", "tf1", minutes=22)
    assert r["remaining_s"] == pytest.approx(22 * 60, abs=1)
    assert book.locked("BTC", "tf1")
    assert not book.locked("ETH", "tf1")      # other asset free
    assert not book.locked("BTC", "tf2")      # same asset other tf free
    assert not book.locked("BNB", "tf4")
    # full 16-cell snapshot: exactly one locked
    snap = book.snapshot()
    assert len(snap) == 16
    assert sum(1 for v in snap.values() if v["locked"]) == 1
    # expiry re-enables the SAME tf
    book._locks["BTC|tf1"] = time.time() - 1
    assert not book.locked("BTC", "tf1")
    assert book.snapshot()["BTC|tf1"]["can_analyse"] is True


def test_engine_start_analyse_refuses_while_cell_locked(tmp_path,
                                                        monkeypatch):
    from tests.test_engine import make_engine
    from app.hold_policy import CellLockBook
    monkeypatch.setenv("HOLD_CELL_LOCKS", "1")
    book = CellLockBook(str(tmp_path / "locks.json"))
    monkeypatch.setattr(hp, "CELL_LOCKS", book)
    eng, feed, _ = make_engine()

    async def _click(sym, tf):
        out = eng.start_analyse(sym, tf)
        if getattr(eng, "_analyse_task", None):
            eng._analyse_task.cancel()
        return out

    out = asyncio.run(_click("BTC", "tf1"))
    assert out["ok"] is True
    assert book.locked("BTC", "tf1")
    # same cell now refused with the countdown
    again = eng.start_analyse("BTC", "tf1")
    assert again["ok"] is False and again["error"] == "cell_locked"
    assert again["remaining_s"] > 21 * 60
    assert "retry when the countdown" in again["why"]
    # other cells untouched
    out2 = asyncio.run(_click("ETH", "tf1"))
    assert out2["ok"] is True
    out3 = asyncio.run(_click("BTC", "tf2"))
    assert out3["ok"] is True
    # timer expiry → same tf allowed again
    book._locks["BTC|tf1"] = time.time() - 1
    out4 = asyncio.run(_click("BTC", "tf1"))
    assert out4["ok"] is True


def test_cell_timer_defaults_off_under_pytest(monkeypatch):
    """Legacy suites keep working: without HOLD_CELL_LOCKS=1 the
    enforcement is off under pytest (production: always on)."""
    monkeypatch.delenv("HOLD_CELL_LOCKS", raising=False)
    from app.hold_policy import CellLockBook
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        book = CellLockBook(os.path.join(d, "l.json"))
        book.start("BTC", "tf1")
        assert book.enabled() is False
        assert book.remaining("BTC", "tf1") == 0.0


# ────────────────────── 6) manual-default contract ───────────────────
def test_auto_predict_is_manual_by_default_everywhere(monkeypatch):
    src = open(os.path.join("app", "agents.py")).read()
    assert '"auto_predict": False' in src
    assert '"auto_predict_touched": False' in src
    # legacy stored auto-on (untouched) migrates to manual
    assert 'not self.settings.get("auto_predict_touched")' in src
    eng_src = open(os.path.join("app", "engine.py")).read()
    assert 'self.agents.settings.get("auto_predict", False)' in eng_src
    # legacy stored auto-on (untouched) migrates to manual
    from tests.test_engine import make_engine
    eng, _, _ = make_engine()
    # never persist this toggle to the shared data/settings.json — it
    # poisoned every later suite's default assertion
    monkeypatch.setattr(type(eng.agents), "save_settings",
                        lambda self: None)
    assert eng.agents.settings.get("auto_predict") is False
    eng.agents.configure(auto_predict=True)
    assert eng.agents.settings["auto_predict_touched"] is True
    eng.agents.configure(auto_predict=False)
    assert eng.agents.settings.get("auto_predict") is False


# ─────────────────────────── 7) committee ────────────────────────────
def test_committee_contribution_is_bounded_and_holds_side():
    from tests.test_engine import make_engine
    from app.agent_committee import (COMMITTEE, CommitteeResult,
                                     AgentVote)
    eng, _, _ = make_engine()
    final = {"direction": 1, "confidence": 0.80}
    # no vote yet → neutral record
    COMMITTEE.last = None
    eng._committee_contribution("BTC", "tf1", final)
    assert final["confidence"] == 0.80
    # A board vote is telemetry, not accuracy evidence.
    votes = [AgentVote(agent=a, direction=1, confidence=0.9,
                       reason="test") for a in
             ("flow", "lead_lag", "contrarian", "governor")]
    COMMITTEE.last = CommitteeResult(
        votes=votes, consensus=1.0, agreement=0.9,
        may_issue=True, vetoes=[], notes=[])
    final = {"direction": 1, "confidence": 0.80}
    eng._committee_contribution("BTC", "tf1", final)
    assert final["confidence"] == pytest.approx(0.80)
    assert final["committee"]["direction_held"] is True
    assert "confirms" in final["committee"]["effect"]
    assert "confidence unchanged" in final["committee"]["effect"]
    # 0/4 agreement is visible, but still does not mutate confidence.
    votes = [AgentVote(agent=a, direction=-1, confidence=0.9,
                       reason="test") for a in
             ("flow", "lead_lag", "contrarian", "governor")]
    COMMITTEE.last = CommitteeResult(
        votes=votes, consensus=-1.0, agreement=0.9,
        may_issue=True, vetoes=[], notes=[])
    final = {"direction": 1, "confidence": 0.80}
    eng._committee_contribution("BTC", "tf1", final)
    assert final["direction"] == 1
    assert final["confidence"] == pytest.approx(0.80)
    assert "challenges" in final["committee"]["effect"]


# ─────────────────────── live blocks + endpoints ─────────────────────
def test_hold_cells_block_covers_16_cells():
    from tests.test_engine import make_engine
    eng, feed, _ = make_engine()
    block = eng.hold_cells_block()
    assert len(block) == 16
    # 2026-11-27: every CERTIFIED cell was selected at row_k = 0.15 against
    # m = 0.55, i.e. 0.27:1 — below the user's absolute 1:1 floor. Such a
    # geometry is not tradeable whatever its row gates say, so the check
    # for a "rows verified" badge has to be made on a cell whose geometry
    # can legally be traded. The shipped artifact has none, which is exactly
    # what TIER_SUB1 reports.
    #
    # tf1 is now a 24-hour lane that the duration-keyed artifact never
    # walked, so it reports UNGRADEABLE rather than SUB1 — a different and
    # equally honest state. BTC tf2 is a certified cell and still SUB1.
    assert block["BTC|tf2"]["tier"] == hp.TIER_SUB1
    assert block["BTC|tf1"]["tier"] == hp.TIER_WARMUP
    assert "1440-minute" in hp.tier_reason("BTC", "tf1")
    # (btc1 is the certified BTC tf2 cell now — tf1 is uncertified)
    btc1 = block["BTC|tf2"]
    assert btc1["geometry_feasible"] is False
    assert btc1["geometry_rr"] < 1.0
    # the required rate (a percentage) exceeds the cell's own gross win rate
    # (a fraction). BTC tf2's certified 79.5 % is genuinely close to the
    # break-even it needs, which is the honest reading of a 0.27:1 geometry.
    assert btc1["geometry_win_rate_needed"] > 50.0
    assert "1:1" in btc1["geometry_note"]
    assert btc1["win_rate"] and btc1["win_rate"] >= 0.66
    assert btc1["confidence"] and btc1["confidence"] >= 0.66
    assert btc1["analyse"]["tf_minutes"] == 44
    # the 24-hour lane reports its own real duration, not the old 22
    assert block["BTC|tf1"]["analyse"]["tf_minutes"] == 1440
    assert "era" in btc1 and "friction_rule" in btc1["era"]
    # PAXG reports NO MEASURED EDGE rather than WARMUP: its signal has no
    # gross edge at any timeframe, which is a fact about the PATTERN and
    # outranks a limitation of its history (see tests/test_gross_edge_gate.py)
    # PAXG is no longer a live cell (removed 2026-11-27), so it is not in the
    # block. Its archived rows must still answer NO MEASURED EDGE — that is
    # the behaviour, and it is asserted directly rather than through a card
    # for an asset the app no longer trades.
    assert "PAXG|tf1" not in block
    assert hp.tier("PAXG", "tf1") == hp.TIER_NO_EDGE
    # The certified artifact's own ship gates FAIL (gate (ii): it
    # records no REAL post-cost win rate at all), so no cell may wear
    # HOLD-OUT VERIFIED while that artifact is the one shipped.
    ship = hp.artifact_ship_verdict()
    assert ship["checked"] is True
    assert ship["gate_ii_pass"] is False
    assert block["BNB|tf4"]["tier"] == hp.TIER_SUB1
    # and every tradeable cell still carries its GROSS label + cost floor
    assert btc1["win_rate_label"].startswith("gross")
    assert btc1["min_tp_bps"] > btc1["round_trip_bps"]
    assert "round trip" in btc1["cost_note"]


def test_tier_restores_holdout_badge_when_artifact_passes_gates(monkeypatch):
    """A failing artifact must not overclaim — and a passing one must
    still earn the badge.  (Guards both directions of the rule.)

    2026-11-27: the geometry floor is checked BEFORE the ship gate, so a
    cell whose certified payoff is below 1:1 reports that first. To keep
    both directions of the SHIP-GATE rule testable, the geometry floor is
    neutralised here and restored by monkeypatch teardown.
    """
    from app import hold_policy as _hp
    monkeypatch.setattr(_hp, "_SHIP_CACHE", None)
    monkeypatch.setattr(_hp, "sub_1_to_1_geometry", lambda *a, **k: None)
    monkeypatch.setattr(_hp, "artifact_ship_verdict",
                        lambda: {"checked": True, "gate_ii_pass": True})
    assert _hp.tier("BNB", "tf4") == _hp.TIER_HOLDOUT
    monkeypatch.setattr(_hp, "_SHIP_CACHE", None)
    monkeypatch.setattr(_hp, "artifact_ship_verdict",
                        lambda: {"checked": True, "gate_ii_pass": False})
    assert _hp.tier("BNB", "tf4") == _hp.TIER_SHIPGATE


def test_a_sub_1_to_1_geometry_outranks_every_row_and_ship_gate(monkeypatch):
    """The user's absolute rule cannot be masked by a green gate.

    Rows can clear their hold-out bar and the artifact can pass its ship
    gates; a pair that pays less than it risks is still not tradeable.
    """
    from app import hold_policy as _hp
    monkeypatch.setattr(_hp, "_SHIP_CACHE", None)
    monkeypatch.setattr(_hp, "artifact_ship_verdict",
                        lambda: {"checked": True, "gate_ii_pass": True})
    assert _hp.tier("BTC", "tf2") == _hp.TIER_SUB1
    s = _hp.summary("BTC", "tf2")
    assert s["geometry_feasible"] is False
    assert "does NOT describe the trade that is placed" in s["geometry_note"]
    # the reason names the ratio that SHIPS (a legal 1:1+ pair), not just
    # the certification's broken one
    assert "ships at the repaired" in s["tier_reason"]
    assert s["live_rr"] >= 1.0


def test_cells_endpoint_serves_policy():
    from app.main import app  # noqa: F401 — module-level engine
    import app.main as m

    async def go():
        return await m.hold_cells()

    resp = asyncio.run(go())
    body = json.loads(resp.body)
    assert body["ok"] is True
    assert body["count"] == 16
    # the shipped artifact fails its own gate (ii) AND every one of its
    # graded cells sits below the 1:1 reward:risk floor, so none of them is
    # reported as verified, shipped or tradeable
    assert body["tiers"].get("HOLD-OUT VERIFIED", 0) == 0
    assert body["tiers"].get("ROWS VERIFIED · SHIP GATE FAIL", 0) == 0
    # 9 certified cells (BTC/ETH/BNB x tf2/tf3/tf4) are SUB1; the 4 uncertified
    # 24-hour lanes and SOL's 4 lanes report UNGRADEABLE.
    # 9 certified cells (BTC/ETH/BNB x tf2/tf3/tf4) are SUB1. The other 7
    # break down as: 4 uncertified 24-hour lanes + 3 SOL lanes = UNGRADEABLE,
    # and the archived PAXG rows still answer NO MEASURED EDGE for the 4
    # cells they describe (PAXG is not in known_cells, so they are counted
    # from the archive rather than from the live grid).
    assert body["tiers"]["1:1-REPAIRED · LOCK ORIGIN UNMEASURED"] == 9
    assert body["tiers"]["UNGRADEABLE / WARMUP"] == 7
    assert body["manual_default"] is True
    assert body["rule"].startswith("one ANALYSE click")
