"""The cost model must price the venue the order path actually executes on.

WHY THIS FILE EXISTS
--------------------
For most of this repo's life three modules quoted three different
"round trip" numbers for the same trade:

    app/horizon.ASSET_RT_BPS        28.872 / 29.808 / 30.963 / 37.400
    app/hold_policy.round_trip_bps  24.0 / 24.0 (and 36.0 for PAXG)
    backtest CSV bps_net            12.0 / 18.0 deducted

None of them matched the order path.  ``app/trading.py`` posts

    POST /fapi/v1/order  {"type": "MARKET", ...}

i.e. Binance USDⓈ-M **futures** taker on both legs, which is 0.05 %/side
= 10 bps a round trip, not 28.9.  Every EV and every tradability verdict
in the app was computed against the spot stack, overstating friction by
~17 bps a trade.  That is a defect, not a modelling choice, so these
tests pin the derivation instead of the literal.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_order_path_is_futures_market_taker():
    """The venue claim in the cost model must match the code that trades."""
    src = open(os.path.join("app", "trading.py"), encoding="utf-8").read()
    assert "/fapi/v1/order" in src, "order path moved — re-derive the venue"
    assert '"type": "MARKET"' in src, (
        "orders are no longer MARKET: a maker entry pays a different fee "
        "and ROUND_TRIP_BPS must be re-derived, not left at the taker rate")
    assert "closePosition" in src


def test_default_venue_is_binance_futures():
    from backtest.data import DEFAULT_VENUE, venue_name
    if not os.environ.get("VENUE") and not os.environ.get("PENNY_VENUE"):
        assert DEFAULT_VENUE == "binance_fut"
        assert venue_name() == "binance_fut"


def test_binance_futures_rt_is_10_bps_before_slippage():
    from backtest.data import VENUE_RT_BPS
    assert VENUE_RT_BPS["binance_fut"] == 10.0
    # the legacy figure is still on the books, clearly labelled as such
    assert VENUE_RT_BPS["legacy_spot"] == pytest.approx(28.872)


def test_round_trip_equals_fee_plus_measured_slippage():
    from backtest.data import ROUND_TRIP_BPS, SLIPPAGE_BPS, VENUE_RT_BPS
    fee = VENUE_RT_BPS["binance_fut"]
    for sym, slip in SLIPPAGE_BPS.items():
        assert ROUND_TRIP_BPS[sym] == pytest.approx(fee + 2.0 * slip), sym


def test_the_three_cost_models_agree():
    """horizon / hold_policy / data must never drift apart again."""
    from app.horizon import asset_rt
    from app.hold_policy import friction_bps, round_trip_bps
    from backtest.data import ROUND_TRIP_BPS
    for sym in ("BTC", "ETH", "BNB", "SOL"):
        assert asset_rt(sym) == pytest.approx(ROUND_TRIP_BPS[sym])
        assert round_trip_bps(sym) == pytest.approx(ROUND_TRIP_BPS[sym])
        assert friction_bps(sym) == pytest.approx(ROUND_TRIP_BPS[sym] / 2.0)


def test_cost_gate_reads_the_same_venue():
    import app.cost_gate as cg
    from backtest.data import ROUND_TRIP_BPS
    assert cg.DEFAULT_FEE_BP == pytest.approx(ROUND_TRIP_BPS["BTC"])
    for name, sleeve in cg.SLEEVES.items():
        g = cg.sleeve_gate(name)
        sym = sleeve["asset"]
        assert g["fee_bp"] == pytest.approx(ROUND_TRIP_BPS[sym]), name


def test_override_changes_the_price_not_the_code():
    """VENUE / PENNY_VENUE are the only supported way to re-price."""
    code = ("from backtest.data import ROUND_TRIP_BPS as R;"
            "print(R['BTC'])")
    out = {}
    for env, label in (({}, "default"), ({"VENUE": "binance_spot"}, "spot"),
                       ({"PENNY_VENUE": "1"}, "penny")):
        e = {**os.environ, **env}
        e.pop("VENUE", None) if not env else None
        if label == "default":
            e.pop("VENUE", None)
            e.pop("PENNY_VENUE", None)
        out[label] = float(subprocess.check_output(
            [sys.executable, "-c", code], env=e, text=True).strip())
    assert out["default"] == pytest.approx(12.0)     # 10 fee + 2 slip
    assert out["spot"] == pytest.approx(22.0)        # 20 fee + 2 slip
    assert out["penny"] == pytest.approx(2.0)        # 0 fee + 2 slip
    assert out["default"] < out["spot"] < 28.872


def test_paxg_has_no_edge_even_at_zero_fee():
    """The PAXG exclusion must survive the cheaper venue.

    If the cost fix alone were what killed PAXG, the cells would come back
    at a lower fee.  They do not: mean gross is negative before any cost.
    """
    path = os.path.join("backtest", "results",
                        "per_tf_full_engine_real.csv")
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["asset"] == "PAXG"]
    assert rows
    gross = sum(float(r["bps_gross"]) for r in rows) / len(rows)
    assert gross < 0, "PAXG gross edge is positive — re-test the exclusion"


def test_reprice_increases_total_net_at_identical_trade_count():
    """The whole point: more money, same trades — not fewer trades."""
    path = os.path.join("backtest", "results",
                        "per_tf_full_engine_real.csv")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    old = {"BTC": 28.872, "ETH": 29.808, "BNB": 30.963, "PAXG": 37.400}
    from backtest.data import ROUND_TRIP_BPS as new
    keep = [r for r in rows if r["asset"] != "PAXG"]
    gross = [float(r["bps_gross"]) for r in keep]
    old_net = sum(g - old[r["asset"]] for g, r in zip(gross, keep))
    new_net = sum(g - new[r["asset"]] for g, r in zip(gross, keep))
    assert len(keep) == 684, "trade count must be unchanged by a cost fix"
    assert new_net > old_net
    assert (new_net - old_net) / abs(old_net) > 0.10


def test_tf1_flips_to_tradable_under_the_real_venue():
    """A gate loosening is the measurable consequence of the fix."""
    from app.horizon import MEASURED, classify
    tf1 = MEASURED["TF1"]
    old = classify(tf1["sigma_bps"], tf1["friction_bps"])
    assert old["state"] != "tradable", (
        "the old friction no longer blocks TF1 — this test documents the "
        "before state and must be updated with the new evidence")
    from app.horizon import asset_rt
    new = classify(tf1["sigma_bps"], asset_rt("BTC") / 2.0)
    assert new["state"] == "tradable"


def test_cost_model_endpoint_serves_one_truth():
    import asyncio
    import json
    import app.main as m
    r = asyncio.run(m.cost_model())
    d = json.loads(r.body)
    assert d["ok"] is True
    assert d["fee_bps_round_trip"] == 10.0
    by = {a["symbol"]: a for a in d["assets"]}
    from backtest.data import ROUND_TRIP_BPS
    for sym, a in by.items():
        assert a["round_trip_bps"] == pytest.approx(ROUND_TRIP_BPS[sym])
        assert a["per_side_bps"] == pytest.approx(ROUND_TRIP_BPS[sym] / 2.0)


# ─────────── the geometry lookup was reading the wrong artifact ───────────
def test_geometry_check_finds_the_graded_barrier_distance():
    """"unknown" must mean no evidence, never a wrong-file lookup.

    geometry_cost_check read full["row_sl_bps_median"] out of
    hold_wr.json, which has never carried that key, so all 16 cells
    answered pays_round_trip=None.  The medians live in wr_rereg_5m.json.
    """
    from app import hold_policy as hp
    got_known = 0
    # tf1 is a 24-hour lane as of 2026-11-27 and the artifact is keyed by
    # duration, so it correctly answers "unknown" — it is excluded from the
    # count rather than being allowed to inflate or deflate it. The three
    # certified sub-day lanes on four assets are 12 cells.
    for sym in ("BTC", "ETH", "BNB", "SOL"):
        for tf in ("tf2", "tf3", "tf4"):
            g = hp.geometry_cost_check(sym, tf)
            if g.get("tp_bps") is not None:
                got_known += 1
                assert g["pays_round_trip"] in (True, False)
                assert g["why"]
    # 7 is the CURRENT certified set: BTC/ETH on all three sub-day lanes,
    # BNB on tf4 only. SOL has no artifact row at all (it arrived 2026-11-27
    # with no walk), and tf1 is a duration the artifact has never covered —
    # both answer "unknown", which is the correct answer and not a
    # regression. The floor guards the LOOKUP, not the coverage.
    assert got_known >= 7, (
        f"only {got_known} cells could resolve their graded target — the "
        "artifact lookup has regressed to returning unknown")


def test_no_measured_cell_lost_payability_under_the_cheaper_venue():
    """The cost fix may only unblock cells, never block one."""
    from app import hold_policy as hp
    from backtest.data import ROUND_TRIP_BPS
    OLD = {"BTC": 28.872, "ETH": 29.808, "BNB": 30.963, "PAXG": 37.400}
    for sym in ("BTC", "ETH", "BNB", "SOL"):
        for tf in ("tf1", "tf2", "tf3", "tf4"):
            g = hp.geometry_cost_check(sym, tf)
            tp = g.get("tp_bps")
            if tp is None:
                continue
            old_pays = tp >= hp.MIN_TP_OVER_RT * OLD[sym]
            new_pays = tp >= hp.MIN_TP_OVER_RT * ROUND_TRIP_BPS[sym]
            assert not (old_pays and not new_pays), (
                f"{sym} {tf} stopped paying: the venue change must not "
                "make any cell uneconomic")


def test_summary_does_not_echo_the_historical_per_side_constant():
    from app import hold_policy as hp
    from backtest.data import ROUND_TRIP_BPS
    # tf2 (44 min) — tf1 is the uncertified 24-hour lane and has no summary
    # cost context to echo, by design.
    s = hp.summary("BTC", "tf2")
    assert s["friction_bps_per_side"] == pytest.approx(
        ROUND_TRIP_BPS["BTC"] / 2.0)
    # the stale grading cost is still reported, just not as the live one
    assert s["round_trip_bps"] == pytest.approx(ROUND_TRIP_BPS["BTC"])


def test_unknown_symbol_is_priced_at_least_as_dear_as_any_known_one():
    from backtest.data import ROUND_TRIP_BPS, venue_rt
    assert venue_rt("NOSUCHCOIN") >= max(ROUND_TRIP_BPS.values())
