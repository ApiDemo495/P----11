"""One asset set, one lane set — and every table must agree.

2026-11-27: PAXG was removed and SOL took its slot (PAXG was the only asset
in the certified walk that lost money GROSS of fees: 44.0 % hit rate, -768 bps
over 209 windows, coefficient of variation 69 against 3-5 for the others).
At the same time `tf22` — the 22-minute lane — was removed and the 24-hour 2:1
lane took its key.

Both changes touch a dozen independent tables, and the failure mode of a
half-applied swap is nasty and quiet: a profile for an asset that no longer
trades, a cost row inherited by an asset that never paid it, a lane that one
module thinks is 22 minutes and another thinks is 24 hours. This file is the
gate on that class of drift. If it fails, the swap is half-done — not "the
test is stale".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (ASSET_PROFILES, SYMBOL_ORDER, SYMBOLS, TF_ORDER,
                        TIMEFRAMES, TWO_TO_ONE_LANE)


def test_asset_set_is_four_and_is_paxg_and_zec_free():
    assert SYMBOL_ORDER == ["BTC", "SOL", "ETH", "BNB"]
    assert set(SYMBOLS) == set(SYMBOL_ORDER)
    assert "PAXG" not in SYMBOLS and "ZEC" not in SYMBOLS
    # exactly ONE of the two candidates was added, per the instruction
    assert ("SOL" in SYMBOLS) != ("ZEC" in SYMBOLS)


def test_every_per_asset_table_covers_exactly_the_asset_set():
    """A table with a stale key is a measurement for a market we left."""
    from app import hold_policy as hp
    from app import mandate_cfg as mc
    from app.exchange_filters import BUNDLED, REFERENCE_PRICE
    from app.horizon import ASSET_RT_BPS
    from app.rr_geometry import VOL_BPS_PER_SQRT_HOUR
    from backtest.data import SLIPPAGE_BPS
    from backtest.data import SYMBOLS as BACKTEST_SYMBOLS

    wanted = set(SYMBOL_ORDER)
    for name, table in (
        ("ASSET_PROFILES", ASSET_PROFILES),
        ("ASSET_RT_BPS", ASSET_RT_BPS),
        ("VOL_BPS_PER_SQRT_HOUR", VOL_BPS_PER_SQRT_HOUR),
        ("MANDATE", mc.MANDATE),
        ("SLIPPAGE_BPS", SLIPPAGE_BPS),
        ("backtest.SYMBOLS", BACKTEST_SYMBOLS),
    ):
        assert set(table) == wanted, (name, set(table) ^ wanted)

    # the ARCHIVE deliberately holds PAXG's rows as the evidence for the
    # removal; it must contain nothing else
    assert set(hp.ARCHIVED_CELL_GROSS_BPS) == {("PAXG", tf) for tf in TF_ORDER}

    # the exchange-keyed tables are the same set in ticker form
    tickers = {SYMBOLS[s] for s in SYMBOL_ORDER}
    assert set(BUNDLED) == tickers
    assert set(REFERENCE_PRICE) == tickers


def test_sol_carries_no_fabricated_measurement():
    """SOL arrived with no walk. Everywhere it would quote a number, it must
    say unknown instead of borrowing PAXG's deleted rows."""
    from app import hold_policy as hp
    from app.cost_gate import MEASURED
    from app.mandate_cfg import MANDATE

    assert MANDATE["SOL"]["measured"] is None
    assert not any(a == "SOL" for a, _ in MEASURED), "cost_gate must not invent"
    assert not any(a == "SOL" for a, _ in hp.CELL_GROSS_BPS)
    for tf in TF_ORDER:
        assert hp.gross_edge("SOL", tf) is None
        assert hp.cell("SOL", tf) is None


def test_the_22_minute_lane_is_gone_and_its_key_is_the_2to1_lane():
    """`tf22` is removed by REPLACING its key's window, not by adding a lane.

    Four lanes, no fifth key, no 22-minute duration anywhere.
    """
    assert TF_ORDER == ["tf1", "tf2", "tf3", "tf4"]
    assert set(TIMEFRAMES) == set(TF_ORDER)
    mins = {tf: TIMEFRAMES[tf]["minutes"] for tf in TF_ORDER}
    assert 22 not in mins.values(), mins
    assert mins["tf1"] == 1440
    assert mins["tf1"] == max(mins.values())
    assert TWO_TO_ONE_LANE == "tf1"
    # tf1's own sub-targets must fit inside its new window
    assert TIMEFRAMES["tf1"]["targets"][-1] <= mins["tf1"]
    for tf in TF_ORDER:
        assert TIMEFRAMES[tf]["targets"][-1] <= TIMEFRAMES[tf]["minutes"]


def test_warmup_gate_clears_the_longest_lane_not_tf4():
    """Hardcoding the gate to tf4 was correct only while tf4 was longest."""
    from app.config import BACKFILL_BARS, MIN_BARS_FULL_WINDOW
    longest = max(m["minutes"] for m in TIMEFRAMES.values())
    assert MIN_BARS_FULL_WINDOW == longest + 1
    # production must actually backfill enough tape to satisfy it
    assert BACKFILL_BARS >= MIN_BARS_FULL_WINDOW


def test_display_metadata_does_not_describe_a_lane_that_is_gone():
    """The UI's own copies of the lane table must match the server's."""
    js = Path(__file__).resolve().parents[1] / "static" / "app.js"
    src = js.read_text()
    assert 'const TF = ["tf1", "tf2", "tf3", "tf4"];' in src
    assert "tf5" not in src, "the seg-16 fifth lane was folded into tf1"
    assert "tf1: { mins: 1440" in src
    # and the asset list matches, exactly
    assert 'const ASSETS = ["BTC", "SOL", "ETH", "BNB"];' in src


def test_the_new_system_has_no_switch_left_to_flip():
    """Deleted, not defaulted off — there must be no selector to turn on."""
    from app import engine as eng_mod
    from app import mandate_cfg as mc

    assert not hasattr(eng_mod.PredictionEngine, "set_system_mode")
    assert not hasattr(eng_mod.PredictionEngine, "_retarget_display")
    assert not hasattr(mc, "mode"), "the per-asset old/new override is gone"
    assert not hasattr(mc, "new_rr"), "the forecast target multiple is gone"


def test_the_measured_2to1_race_and_the_lane_that_hosts_it():
    from app.rr_geometry import (MEASURED, P_TP_FIRST_FAIR, TWO_TO_ONE_SIDE,
                                 race_summary)

    r = race_summary()
    assert r["corrected_side"] == TWO_TO_ONE_SIDE
    assert r["fair_rate"] == pytest.approx(P_TP_FIRST_FAIR, abs=1e-4)
    # the measured race is on the REVERSE of the shipped side, and both
    # one-sided tests must be the ones recorded with the fixture
    assert r["flipped_rate"] > r["shipped_rate"]
    assert r["p_shipped_below_fair"] < 0.01
    assert MEASURED["resolved"] == MEASURED["tp_first"] + MEASURED["sl_first"]
    # and the hosting lane is the 24-hour one
    assert TWO_TO_ONE_LANE == "tf1"
    assert TIMEFRAMES[TWO_TO_ONE_LANE]["minutes"] == 1440


# ────────── the 2:1 arithmetic is sized on the bound, not the point ──────
def test_the_two_to_one_edge_is_the_conservative_bound_not_the_point():
    """40/48 from ONE 2.5-month regime may inform a sign, not a magnitude."""
    from app.rr_geometry import (P_TP_FIRST_FAIR, TWO_TO_ONE_EDGE,
                                 TWO_TO_ONE_EDGE_OBSERVED,
                                 TWO_TO_ONE_EDGE_SHIPPED,
                                 TWO_TO_ONE_RATE_OBSERVED,
                                 TWO_TO_ONE_RATE_PLANNED)
    # the observed point estimate is 40/48 and is published as such
    assert TWO_TO_ONE_RATE_OBSERVED == pytest.approx(40 / 48, abs=1e-9)
    assert TWO_TO_ONE_EDGE_OBSERVED == pytest.approx(40 / 48 - 1 / 3, abs=1e-9)
    # the PLANNING figure is the Wilson 95% lower bound, strictly below it
    assert TWO_TO_ONE_RATE_PLANNED == pytest.approx(0.7042, abs=1e-4)
    assert TWO_TO_ONE_RATE_PLANNED < TWO_TO_ONE_RATE_OBSERVED
    assert TWO_TO_ONE_EDGE == pytest.approx(0.7042 - 1 / 3, abs=1e-4)
    assert TWO_TO_ONE_EDGE < TWO_TO_ONE_EDGE_OBSERVED
    # and the unflipped side is genuinely adverse, not merely neutral
    assert TWO_TO_ONE_EDGE_SHIPPED == pytest.approx(8 / 48 - 1 / 3, abs=1e-9)
    assert TWO_TO_ONE_EDGE_SHIPPED < 0.0 < P_TP_FIRST_FAIR + TWO_TO_ONE_EDGE


def test_cost_is_charged_once_and_only_on_the_time_out():
    """The barriers are net-of-fee, so a stopped trade does not pay twice."""
    from app.rr_geometry import lane_geometry
    g = lane_geometry(1440.0, 16.6, 12.0, edge=0.3709)
    res, rt, stop = (g["resolve_probability"], g["round_trip_bps"],
                     g["stop_bps"])
    # the stop IS the round trip: a loss already eats the fee
    assert stop == pytest.approx(rt)
    assert g["cost_r"] == pytest.approx((1 - res) * rt / stop, abs=1e-6)
    # net is strictly below gross whenever the window can time out
    assert g["expectancy_r"] < g["expectancy_r_gross"]
    # a pair that always resolves charges nothing extra
    always = lane_geometry(10 ** 9, 16.6, 12.0, edge=0.0)
    assert always["resolve_probability"] == pytest.approx(1.0, abs=1e-3)
    assert always["cost_r"] == pytest.approx(0.0, abs=1e-3)
    assert always["expectancy_r"] == pytest.approx(0.0, abs=1e-3)


def test_a_fair_pair_expects_zero_gross_at_every_window():
    """The no-edge null must reproduce, or the whole table is mis-scaled."""
    from app.config import TIMEFRAMES
    from app.rr_geometry import lane_geometry
    for tf, m in TIMEFRAMES.items():
        g = lane_geometry(float(m["minutes"]), 16.6, 12.0, edge=0.0)
        assert g["expectancy_r_gross"] == pytest.approx(0.0, abs=1e-9), tf


def test_the_docs_record_exists_and_names_the_bound():
    """The code cites docs/TWO_TO_ONE.md — it must be there, and be honest."""
    from pathlib import Path
    doc = Path("docs/TWO_TO_ONE.md").read_text(encoding="utf-8")
    assert "1/3" in doc
    assert "0.7042" in doc and "83.3" in doc
    assert "single-regime" in doc.lower() or "one 2.5-month regime" in doc
