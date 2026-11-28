"""The gross-edge gate: no signal, no trade.

The app shipped 16 cells for weeks. Measured on its own engine walk
(``backtest/results/per_tf_full_engine_real.csv``, 893 settled windows),
four of them — all PAXG — had no edge BEFORE any fee was charged. That
measurement is exactly why PAXG was REMOVED as an asset on 2026-11-27
(see app/config.py): a cell that is not gross-positive is not a fee
problem, and no amount of geometry or venue choice fixes it. SOL took the
slot, with no measured row — it reports the honest "no out-of-fold
measurement" state instead of inheriting PAXG's numbers.

This file locks in three things:

1. the table in ``app/hold_policy.py`` is exactly re-derivable from the
   raw CSV, so a number can never become a memory;
2. no cell of a shipped asset is excluded on gross edge, and PAXG (which
   was the one excluded asset) is no longer in the set at all;
3. the gate is a pattern-level statement, not a per-window abstention:
   a cell keeps computing, it simply may not present itself as
   tradeable.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "backtest" / "results" / "per_tf_full_engine_real.csv"


def _rows():
    with CSV.open() as fh:
        out = list(csv.DictReader(fh))
    for r in out:
        r["gross"] = float(r["bps_gross"])
        r["net"] = float(r["bps_net"])
        r["win"] = int(r["win"])
        r["conf"] = float(r["conf"] or 0)
    return out


def test_gross_edge_table_reproduces_from_the_raw_walk():
    """Every number in CELL_GROSS_BPS must come out of the CSV again."""
    from app.hold_policy import CELL_GROSS_BPS

    rows = _rows()
    assert len(rows) == 893, "the engine walk changed size; re-derive the table"
    for (symbol, tf), row in CELL_GROSS_BPS.items():
        sub = [r["gross"] for r in rows
               if r["asset"] == symbol and r["tf"] == tf]
        assert sub, f"{symbol} {tf} missing from the walk"
        assert abs(st.mean(sub) - row["gross"]) < 0.05, (symbol, tf)
        assert len(sub) == int(row["n"]), (symbol, tf)


def test_no_shipped_asset_has_a_gross_negative_cell():
    """The asset set is defined BY this test now.

    PAXG was removed because four of its cells had no edge before any fee.
    Nothing in the set may be in that state: a gross-negative cell means the
    asset does not belong here, and the response is to remove the asset —
    not to charge it a different fee.
    """
    from app.config import SYMBOL_ORDER
    from app.hold_policy import has_gross_edge

    assert "PAXG" not in SYMBOL_ORDER, "PAXG must not be in the asset set"
    excluded = {(a, tf) for a in SYMBOL_ORDER
                for tf in ("tf1", "tf2", "tf3", "tf4")
                if has_gross_edge(a, tf) is False}
    assert not excluded, excluded
    # …and the removal is what makes that true: PAXG's archived rows are the
    # only gross-negative cells in the table.
    from app.hold_policy import ARCHIVED_CELL_GROSS_BPS
    assert set(ARCHIVED_CELL_GROSS_BPS) == {
        ("PAXG", t) for t in ("tf1", "tf2", "tf3", "tf4")}
    # the other twelve are all tradeable
    from app.hold_policy import has_gross_edge as h
    for a in ("BTC", "ETH", "BNB"):
        for tf in ("tf1", "tf2", "tf3", "tf4"):
            assert h(a, tf) is True, (a, tf)


def test_the_exclusion_survives_any_fee_assumption():
    """Gross edge is the point: PAXG fails with costs set to ZERO.

    A threshold on NET would be a bet on today's fee stack. The gate is
    expressed gross precisely so lowering fees to nothing — or moving to a
    zero-fee venue — cannot resurrect a signal that predicts no move.
    """
    from app.hold_policy import gross_edge

    for tf in ("tf1", "tf2", "tf3", "tf4"):
        g = gross_edge("PAXG", tf)
        assert g is not None, "archived row must still be readable"
        assert g < 1.0, (tf, g)


def test_crypto_cells_are_gross_positive_at_every_confidence_band():
    """The edge is not concentrated in one confidence slice.

    This is why the gate is per-cell rather than a confidence threshold:
    for BTC/ETH/BNB the mean gross move is positive in all four bands, so
    there is no band in which the pattern stops working.
    """
    rows = [r for r in _rows() if r["asset"] != "PAXG"]   # PAXG archived
    for lo, hi in ((0, 0.2), (0.2, 0.39), (0.39, 0.60), (0.60, 1.01)):
        band = [r for r in rows if lo <= r["conf"] < hi]
        assert band, (lo, hi)
        assert st.mean(r["gross"] for r in band) > 0, (lo, hi)


def test_paxg_flips_sign_across_bands():
    """The record that got PAXG removed — kept as the reason, not as a cell.

    PAXG is no longer an asset, so this reads the archived CSV directly. It
    is deliberately preserved: it is the evidence for the removal, and if
    anyone re-adds a gold token this is the test that says why not.
    """
    rows = [r for r in _rows() if r["asset"] == "PAXG"]
    assert rows, "the archived PAXG rows are the evidence for the removal"
    means = []
    for lo, hi in ((0, 0.2), (0.2, 0.39), (0.39, 0.60), (0.60, 1.01)):
        band = [r for r in rows if lo <= r["conf"] < hi]
        if band:
            means.append(st.mean(r["gross"] for r in band))
    assert any(m > 0 for m in means), "no band is positive"
    assert any(m < 0 for m in means), "no band is negative"
    # a real edge does not change sign four times across confidence
    assert not all(m > 0 for m in means)


def test_no_edge_is_a_pattern_label_not_a_per_window_abstention():
    """The no-abstention rule is about WINDOWS, not about pretending.

    A cell with no measured edge must still compute and must still be shown;
    it may only refuse to present itself as tradeable, and it must say why.
    PAXG used to carry this label. PAXG is gone, so the label is exercised
    on an archived cell instead — the label itself is still load-bearing for
    the next asset that arrives without a walk.
    """
    from app import hold_policy as hp

    assert hp.TIER_NO_EDGE.startswith("NO MEASURED EDGE")
    # the label is a pattern, not a per-window veto
    assert hp.has_gross_edge("PAXG", "tf1") is False
    why = hp.tier_reason("PAXG", "tf1")
    assert "gross" in why.lower() or "no measured edge" in why.lower(), why


def test_unmeasured_cell_is_not_silently_tradeable():
    """None means unknown — never a guessed yes.

    SOL is in exactly the state BTC/ETH/BNB were in before their walk ran:
    real cells, real learning, no measurement. It must report UNKNOWN, not
    borrow a number (and specifically not inherit PAXG's deleted rows).
    """
    from app.config import SYMBOL_ORDER
    from app.hold_policy import CELL_GROSS_BPS, gross_edge, has_gross_edge

    grid = {(s, tf) for s in SYMBOL_ORDER
            for tf in ("tf1", "tf2", "tf3", "tf4")}
    assert not any(a == "SOL" for a, _ in CELL_GROSS_BPS), \
        "SOL must have NO fabricated measurement"
    assert grid  # the grid is real
    for tf in ("tf1", "tf2", "tf3", "tf4"):
        assert gross_edge("SOL", tf) is None
        assert has_gross_edge("SOL", tf) is None


def test_selecting_cells_on_gross_edge_raises_measured_ev():
    """The whole point: the survivors make more money per trade.

    Reported on the same walk the table came from. This is a regression
    guard on the DECISION, not a claim about future tape.
    """
    rows = _rows()
    all_ev = st.mean(r["net"] for r in rows)
    kept = [r for r in rows if r["asset"] != "PAXG"]
    kept_ev = st.mean(r["net"] for r in kept)
    assert kept_ev > all_ev, (kept_ev, all_ev)
    # and the win rate improves too, without dropping below 100 % coverage
    # of the cells we keep
    assert (sum(r["win"] for r in kept) / len(kept)) > \
        (sum(r["win"] for r in rows) / len(rows))
