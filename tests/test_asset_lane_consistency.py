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
    # the user's directive: six hours, not a 24-hour swing lane
    assert mins["tf1"] == 360
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
    assert "tf1: { mins: 360" in src
    # and the asset list matches, exactly
    assert 'const ASSETS = ["BTC", "SOL", "ETH", "BNB"];' in src
    # the chart tables must agree with the server's lane minutes
    from app.config import TIMEFRAMES
    for tf, m in TIMEFRAMES.items():
        assert f"{tf}: {m['minutes']}" in src, tf


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
    assert r["fair_rate"] == pytest.approx(P_TP_FIRST_FAIR, abs=1e-4)
    # the measured race is on the REVERSE of the shipped side...
    assert r["flipped_rate"] > r["shipped_rate"]
    assert r["p_shipped_at_least_fair"] < 0.01
    assert MEASURED["resolved"] == MEASURED["tp_first"] + MEASURED["sl_first"]
    # ...but it is a LABEL with no size authority, because flipping a fixed
    # 2:1 pair only relabels the book: resolved -24R <-> +24R, time-outs
    # +24.8R <-> -24.8R, total -0.82R either way.
    assert r["side_policy"]["size_authority"] == 0.0
    assert r["side_policy"]["edge_used"] == 0.0
    assert r["side_policy"]["pnl_if_flipped_r"] == pytest.approx(
        -r["side_policy"]["pnl_if_shipped_r"], abs=1e-9)
    assert TWO_TO_ONE_SIDE == "flipped"        # the label
    # and the hosting lane is the SIX-HOUR one
    assert TWO_TO_ONE_LANE == "tf1"
    assert TIMEFRAMES[TWO_TO_ONE_LANE]["minutes"] == 360


# ──────── the 2:1 arithmetic obeys its own ceiling ────────
def test_the_conditional_two_to_one_rate_never_exceeds_one_third():
    """A zero-edge 2:1 pair wins 1/3 of the windows it resolves. Not 0.704.

    The previous release published `win_rate_conditional = 0.704` for a
    zero-edge pair, which is more than twice the theoretical maximum. That
    was an error (a mis-scaled touch curve plus a single-regime `edge`), and
    this test pins the bound that rules it out.
    """
    from app.rr_geometry import (P_TP_FIRST_FAIR, SIMULATION, best_lane,
                                 lane_geometry)
    from app.config import TIMEFRAMES

    # every lane, measured vol, no edge — none may cross 1/3
    from app.rr_geometry import VOL_BPS_PER_SQRT_HOUR
    lanes = {tf: float(m["minutes"]) for tf, m in TIMEFRAMES.items()}
    for sym, v in VOL_BPS_PER_SQRT_HOUR.items():
        if v is None:
            continue
        res = best_lane(12.0, v, lanes, 0.0)
        for tf, d in res["lanes"].items():
            assert d["win_rate_conditional"] <= P_TP_FIRST_FAIR + 1e-9, (sym, tf)
            assert d["win_rate_unconditional"] <= P_TP_FIRST_FAIR + 1e-9, (sym, tf)
            assert d["expectancy_r_gross"] <= 1e-9, (sym, tf)

    # the simulation that proves the ceiling is stored with the code. A
    # coarse monitoring grid may sit a hair above 1/3 (a path can straddle a
    # barrier between samples); that artefact is bounded and converges.
    from app.rr_geometry import MONITORING_SLACK
    for row in SIMULATION["rows"]:
        assert row["p_tp_given_resolved"] <= P_TP_FIRST_FAIR + MONITORING_SLACK, row
    # a shorter window pulls the rate strictly BELOW 1/3
    assert (SIMULATION["rows"][0]["p_tp_given_resolved"]
            > SIMULATION["rows"][-1]["p_tp_given_resolved"])
    # and refining the grid brings the excess back to zero
    conv = SIMULATION["monitoring_convergence"]
    assert abs(conv[-1]["excess_over_one_third"]) < abs(
        conv[0]["excess_over_one_third"])
    assert abs(conv[-1]["excess_over_one_third"]) < 0.01

    g = lane_geometry(360.0, 49.9, 12.0, edge=0.0)
    # rounded to 4dp on the payload
    assert g["win_rate_conditional"] == pytest.approx(P_TP_FIRST_FAIR, abs=1e-4)


def test_the_zero_edge_theorem_holds_across_the_whole_grid():
    """achieved - needed = (P(resolve) - 1 - c/b)/(1+R) < 0, always."""
    from app.rr_geometry import (ZERO_EDGE_THEOREM, breakeven_win_rate,
                                 horizon_sigma, resolution_probability)

    rt = 12.0
    worst = -1e9
    for minutes in (44, 130, 247, 360, 1440):
        sig = horizon_sigma(49.9, minutes)
        for ratio in (2, 3, 4, 5, 10, 20):
            for stop in range(20, 1000, 20):
                pr = resolution_probability(stop, sig)
                achieved = pr / (1.0 + ratio)
                needed = breakeven_win_rate(stop, stop * ratio, rt)
                worst = max(worst, achieved - needed)
    assert worst <= 1e-12, f"a zero-edge pair cleared its round trip: {worst}"
    assert ZERO_EDGE_THEOREM["grid_checked"]["max_edge_found"] == 0.0
    assert "never clear its round trip" in ZERO_EDGE_THEOREM["statement"]


def test_an_unmeasured_asset_gets_no_rate_rather_than_a_prior():
    """SOL has no price history here, so it must get `None`, not a guess."""
    from app.rr_geometry import (VOL_BPS_PER_SQRT_HOUR, VOL_OBSERVATIONS,
                                 vol_scale)

    v, prov = vol_scale("SOL")
    assert v is None
    assert VOL_OBSERVATIONS["SOL"] == 0
    assert "no price history" in prov
    assert "invention" in prov
    # the three majors ARE measured, and the count travels with the value
    for sym in ("BTC", "ETH", "BNB"):
        v2, prov2 = vol_scale(sym)
        assert v2 is not None and v2 > 0
        assert "measured_vol" in prov2
        assert f"n={VOL_OBSERVATIONS[sym]}" in prov2
    # an unknown symbol is unmeasured too — never silently BTC's prior
    assert vol_scale("XRP")[0] is None


def test_the_docs_record_exists_and_records_the_correction():
    """The code cites docs/TWO_TO_ONE.md — it must be there, and be honest."""
    from pathlib import Path
    doc = Path("docs/TWO_TO_ONE.md").read_text(encoding="utf-8")
    assert "1/3" in doc
    # it must record that 60.6% was WRONG, not quietly drop it
    assert "60.6" in doc and "cannot exist" in doc
    assert "cancels" in doc
    assert "never clear its own round trip" in doc
    # and that the flip has no size authority
    assert "size_authority" in doc
