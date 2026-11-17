"""Portfolio engine guarantees.

Every test here exists because the corresponding bug actually shipped
during development and was caught by replaying the locked holdout:

* drawdown divided by a near-zero peak -> 159 % "drawdown"
* friction charged twice -> the research's +38 bp became -3.5 bp
* a per-bar volatility stop replaced the certified hold -> different
  strategy, never certified
* one sleeve's bar index used on another sleeve's grid -> exits before
  entries
* 48 hours used as 48 bars on a 15m grid -> every trade exited 4x early
* per-asset budget divided by OPEN positions, not active sleeves ->
  the second sleeve lost 60 of its 167 trades
"""
import math
import random

import pytest

from app.portfolio_live import (
    CATASTROPHIC_STOP_FRACTION,
    DEFAULT_COOLDOWN_BARS,
    LivePortfolio,
    ClosedTrade,
    pearson,
    t_stat,
    vol_target_scale,
)


def _reg(*statuses):
    sleeves = []
    for i, st in enumerate(statuses):
        asset = "ETH"
        sleeves.append({
            "sleeve_id": f"s{i}::{asset}",
            "asset": asset,
            "status": st,
            "net_ev_bps": 50.0,
            "profit_factor": 1.5,
            "t_effective": 2.5,
            "params": {"fold1": "fam|ETH|hold_h=48|thr=2.5"},
        })
    return {"sleeves": sleeves}


def _book(n_open=2, enforce=True):
    return LivePortfolio(registry=_reg(*(["open"] * n_open)),
                         ledger_path="/tmp/_pl_test_ledger.json",
                         enforce_monitors=enforce)


BAR = {"t": 1_700_000_000, "o": 100.0}


# ------------------------------------------------------------------ stats
def test_vol_target_scale_is_clipped_and_safe():
    assert vol_target_scale(0.01) == pytest.approx(1.0)
    assert vol_target_scale(0.001) == 2.0            # never lever past 2x
    assert vol_target_scale(10.0) == 0.25            # never zero the book
    assert vol_target_scale(0.0) == 1.0              # degenerate input
    assert vol_target_scale(float("nan")) == 1.0


def test_t_stat_needs_samples_and_variance():
    assert t_stat([]) == 0.0
    assert t_stat([1.0, 2.0]) == 0.0
    assert t_stat([5.0, 5.0, 5.0, 5.0]) == 0.0       # no variance, no claim
    assert t_stat([1.0, 2.0, 3.0, 4.0]) > 0


def test_pearson_bounds_and_degenerate_cases():
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0      # zero variance
    assert pearson([1], [1]) == 0.0                  # too few


# ------------------------------------------------------------- registry
def test_only_open_sleeves_enter_the_book():
    book = LivePortfolio(registry=_reg("open", "shadow", "closed"),
                         ledger_path="/tmp/_pl_test_ledger.json")
    assert len(book.sleeves) == 1


def test_hold_is_parsed_from_the_registry_params():
    book = _book(1)
    assert book.sleeves[next(iter(book.sleeves))].hold_hours == 48


# ---------------------------------------------------------------- sizing
def test_two_sleeves_on_one_asset_split_the_budget():
    book = _book(2)
    ids = list(book.sleeves)
    assert book.size_for(book.sleeves[ids[0]], 1.0) == pytest.approx(0.25)
    assert book.size_for(book.sleeves[ids[1]], 1.0) == pytest.approx(0.25)


def test_vol_scale_can_shrink_a_position_but_never_grow_it():
    book = _book(1)
    sid = next(iter(book.sleeves))
    assert book.size_for(book.sleeves[sid], 2.0) == pytest.approx(0.5)
    assert book.size_for(book.sleeves[sid], 0.5) == pytest.approx(0.25)


def test_active_sleeve_count_ignores_halted_ones():
    book = _book(2)
    ids = list(book.sleeves)
    book.sleeves[ids[0]].status = "halted"
    assert book.active_sleeves_for("ETH") == 1
    assert book.size_for(book.sleeves[ids[1]], 1.0) == pytest.approx(0.5)


# ------------------------------------------------------------- execution
def test_zero_cooldown_means_no_cooldown():
    assert DEFAULT_COOLDOWN_BARS == 0
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.cooldown_until[sid] = 10_000            # a stale index from a
    ok = book.try_enter(sid, "long", BAR, 5, 0.01)  # different calendar
    assert ok["ok"] is True


def test_entry_pays_half_the_friction_adversely():
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.try_enter(sid, "long", BAR, 0, 0.01)
    pos = book.positions[sid]
    assert pos.entry_price == pytest.approx(100.0 * (1 + 6 / 1e4))   # 12bp; half=6
    book2 = _book(1)
    sid2 = next(iter(book2.sleeves))
    book2.try_enter(sid2, "short", BAR, 0, 0.01)
    assert book2.positions[sid2].entry_price == pytest.approx(100.0 * (1 - 6 / 1e4))


def test_stop_is_a_catastrophic_backstop_not_a_strategy():
    """The certified exit is the hold timer; the stop must be far away."""
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.try_enter(sid, "long", BAR, 0, 0.01)
    pos = book.positions[sid]
    assert pos.stop == pytest.approx(pos.entry_price * (1 - CATASTROPHIC_STOP_FRACTION))
    assert CATASTROPHIC_STOP_FRACTION >= 0.2


def test_a_quiet_bar_does_not_trip_the_stop():
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.try_enter(sid, "long", BAR, 0, 0.01)
    closed = book.manage(1, {"ETH": {"t": 1, "c": 99.0, "h": 100.5, "l": 99.0}})
    assert closed == []          # -1 % is nowhere near a 25 % backstop


def test_flat_signal_never_enters():
    book = _book(1)
    sid = next(iter(book.sleeves))
    assert book.try_enter(sid, "flat", BAR, 0, 0.01)["ok"] is False


def test_unknown_sleeve_is_refused():
    book = _book(1)
    assert book.try_enter("nope", "long", BAR, 0, 0.01)["ok"] is False


# -------------------------------------------------------------- accounting
def test_friction_is_charged_once_not_twice():
    """An adverse fill pair already embodies the round trip."""
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.try_enter(sid, "long", BAR, 0, 0.01)
    entry = book.positions[sid].entry_price
    # flat market: only slippage should show up, and it must be ~12 bp total
    closed = book.manage(48, {"ETH": {"t": 2, "c": entry, "h": entry, "l": entry}})
    assert len(closed) == 1
    # exit from an unmoved market pays the OTHER half of the 12 bp round trip
    exited = entry * (1 - 6 / 1e4)
    assert closed[0].exit_price == pytest.approx(exited)
    assert closed[0].unit_bps == pytest.approx(
        (exited / entry - 1.0) * 1e4, abs=0.5)
    assert closed[0].unit_bps == pytest.approx(-6.0, abs=0.5)


def test_hold_expiry_uses_bars_not_hours():
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.sleeves[sid].hold_bars = 4               # 4 bars, not 4 hours
    book.try_enter(sid, "long", BAR, 0, 0.01)
    px = {"ETH": {"t": 2, "c": 101.0, "h": 101.0, "l": 100.0}}
    assert book.manage(3, px) == []               # too early
    assert len(book.manage(4, px)) == 1           # exactly at the hold


def test_drawdown_is_bounded_and_equity_based():
    book = _book(1)
    assert book.drawdown() == 0.0
    book.equity = 0.9
    book.peak_equity = 1.0
    assert book.drawdown() == pytest.approx(0.10)
    book.equity = 1.2
    book.peak_equity = 1.2
    assert book.drawdown() == 0.0                 # never negative


def test_drawdown_halt_blocks_new_entries():
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.equity, book.peak_equity = 0.5, 1.0      # 50 % drawdown
    r = book.try_enter(sid, "long", BAR, 0, 0.01)
    assert r["ok"] is False and "drawdown" in r["why"]


# ---------------------------------------------------------------- monitors
def test_decay_monitor_halts_a_sleeve_whose_edge_died():
    """Sustained contradiction of a certified edge must eventually halt.

    One contradicting reading only de-risks: a threshold re-tested on every
    new trade is reached by any random walk, and on the real certified
    catchup sleeve the z test first breached -2 at exactly n=88 -- its worst
    running mean -- before recovering to +38 bp by n=165. Decay must be
    SUSTAINED before a sleeve is killed.
    """
    book = _book(1)
    sid = next(iter(book.sleeves))
    # a genuinely dead sleeve: the live mean and the first few moments are
    # far below the certified edge, and it stays that way as trades arrive
    for i in range(120):
        book.closed.append(ClosedTrade(
            sleeve_id=sid, asset="ETH", direction=1, size=0.5,
            entry_t=i, exit_t=i + 1, entry_price=100.0, exit_price=99.0,
            gross_bps=-100.0, slippage_bps=12.0, net_bps=-50.0,
            reason="hold_expired", unit_bps=-100.0 - (i % 7),
        ))
        book._check_decay()
    assert book.sleeves[sid].status == "halted"
    assert book.try_enter(sid, "long", BAR, 0, 0.01)["ok"] is False


def test_one_bad_reading_de_risks_but_does_not_kill():
    from app.portfolio_live import THROTTLE_SIZE_SCALE

    book = _book(1)
    sid = next(iter(book.sleeves))
    rt = book.sleeves[sid]
    full = book.size_for(rt, 1.0)
    for i in range(120):
        book.closed.append(ClosedTrade(
            sleeve_id=sid, asset="ETH", direction=1, size=0.5,
            entry_t=i, exit_t=i + 1, entry_price=100.0, exit_price=99.0,
            gross_bps=-100.0, slippage_bps=12.0, net_bps=-50.0,
            reason="hold_expired", unit_bps=-100.0 - (i % 7),
        ))
    book._check_decay()
    assert rt.status == "throttled"          # first contradiction
    assert rt._decay_streak == 1
    assert book.size_for(rt, 1.0) == pytest.approx(full * THROTTLE_SIZE_SCALE)


def test_throttled_sleeve_actually_trades_smaller():
    """The circuit breaker must reduce risk, not just set a flag.

    ``throttled`` was a decorative status: size_for() never read it, so the
    correlation breaker and the de-risk path changed nothing at all.
    """
    from app.portfolio_live import THROTTLE_SIZE_SCALE

    book = _book(1)
    sid = next(iter(book.sleeves))
    rt = book.sleeves[sid]
    assert rt.status == "open"
    full = book.size_for(rt, 1.0)
    rt.status = "throttled"
    assert book.size_for(rt, 1.0) == pytest.approx(full * THROTTLE_SIZE_SCALE)
    assert THROTTLE_SIZE_SCALE < 1.0


def test_monitor_reasons_never_grow_without_bound():
    """Monitors run every bar; reasons must not accumulate per bar."""
    book = _book(1)
    sid = next(iter(book.sleeves))
    for i in range(40):
        book.closed.append(ClosedTrade(
            sleeve_id=sid, asset="ETH", direction=1, size=0.5,
            entry_t=i, exit_t=i + 1, entry_price=100.0, exit_price=99.0,
            gross_bps=-100.0, slippage_bps=12.0, net_bps=-50.0,
            reason="hold_expired", unit_bps=-100.0,
        ))
    for _ in range(300):
        book._check_decay()
    assert len(book.sleeves[sid].halt_reasons) <= 3


def test_decay_monitor_is_patient_before_it_has_evidence():
    book = _book(1)
    sid = next(iter(book.sleeves))
    for i in range(5):                            # too few to judge
        book.closed.append(ClosedTrade(
            sleeve_id=sid, asset="ETH", direction=1, size=0.5,
            entry_t=i, exit_t=i + 1, entry_price=100.0, exit_price=99.0,
            gross_bps=-100.0, slippage_bps=12.0, net_bps=-50.0,
            reason="hold_expired", unit_bps=-100.0,
        ))
    book._check_decay()
    assert book.sleeves[sid].status == "open"


def test_correlation_monitor_throttles_the_weaker_sleeve():
    book = _book(2)
    a, b = list(book.sleeves)
    rng = random.Random(3)
    for day in range(60):
        common = rng.gauss(0, 50)
        for sid, edge in ((a, 40.0), (b, 5.0)):
            book.closed.append(ClosedTrade(
                sleeve_id=sid, asset="ETH", direction=1, size=0.5,
                entry_t=day * 86400, exit_t=day * 86400 + 3600,
                entry_price=100.0, exit_price=100.0,
                gross_bps=edge + common, slippage_bps=12.0,
                # the book P&L must carry the shared component too, otherwise
                # the daily series is constant and correlation is undefined
                net_bps=(edge + common) * 0.5,
                reason="hold_expired", unit_bps=edge + common,
            ))
    book._check_correlation()
    assert book.sleeves[b].status == "throttled"
    assert book.sleeves[a].status == "open"


def test_report_mode_observes_without_stopping_the_book():
    book = _book(1, enforce=False)
    sid = next(iter(book.sleeves))
    for i in range(120):
        book.closed.append(ClosedTrade(
            sleeve_id=sid, asset="ETH", direction=1, size=0.5,
            entry_t=i, exit_t=i + 1, entry_price=100.0, exit_price=99.0,
            gross_bps=-100.0, slippage_bps=12.0, net_bps=-50.0,
            reason="hold_expired", unit_bps=-100.0 - (i % 7),
        ))
    px = {"ETH": {"t": 3, "c": 100.0, "h": 100.0, "l": 100.0}}
    # a streak means the contradiction survived SUCCESSIVE new trades, so
    # each monitor evaluation must be fed fresh closed evidence
    for bar in range(2, 9):
        book.closed.append(ClosedTrade(
            sleeve_id=sid, asset="ETH", direction=1, size=0.5,
            entry_t=100 + bar, exit_t=101 + bar, entry_price=100.0,
            exit_price=99.0, gross_bps=-100.0, slippage_bps=12.0,
            net_bps=-50.0, reason="hold_expired", unit_bps=-100.0 - (bar % 7),
        ))
        book.manage(bar, px)
    assert book.sleeves[sid].status == "open"     # not enforced
    # the verdict is still reported honestly, it just does not act
    assert book.snapshot()["monitor_verdicts"][sid]["status"] == "halted"


# --------------------------------------------------------------- reporting
def test_snapshot_is_json_safe_and_complete():
    import json
    book = _book(2)
    snap = book.snapshot()
    json.dumps(snap)                              # must not raise
    for key in ("sleeves", "open_positions", "gross_exposure",
                "realized_bps", "drawdown", "limits"):
        assert key in snap
    assert snap["limits"]["max_sleeve_corr"] == 0.8


def test_sleeve_report_separates_unit_edge_from_book_pnl():
    book = _book(1)
    sid = next(iter(book.sleeves))
    book.closed.append(ClosedTrade(
        sleeve_id=sid, asset="ETH", direction=1, size=0.25,
        entry_t=0, exit_t=1, entry_price=100.0, exit_price=101.0,
        gross_bps=100.0, slippage_bps=12.0, net_bps=25.0,
        reason="hold_expired", unit_bps=100.0,
    ))
    rep = book.sleeve_report(sid)
    assert rep["live_ev_bps"] == pytest.approx(100.0)    # comparable to research
    assert rep["live_net_bps"] == pytest.approx(25.0)    # what the book earned


# ------------------------------------------------------------- UI contract
def test_ui_exposes_the_paper_book_and_sleeve_telemetry():
    """The dashboard must show the book and an honest "none/blocked" state."""
    import os

    root = os.path.join(os.path.dirname(__file__), "..")
    html = open(os.path.join(root, "static", "index.html")).read()
    js = open(os.path.join(root, "static", "app.js")).read()
    for el in ("hero-sleeve", "book-sleeves", "book-positions", "book-meta"):
        assert f'id="{el}"' in html, f"missing #{el} in index.html"
    assert "function renderBook()" in js
    assert 'sleeves?.portfolio' in js
    assert "none / blocked" in js, "UI must state when an asset has no edge"
    assert "realised slippage" in js
