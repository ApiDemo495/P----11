"""The gross-edge gate: no signal, no trade.

The app shipped 16 cells for weeks. Measured on its own engine walk
(``backtest/results/per_tf_full_engine_real.csv``, 893 settled windows),
four of them — all PAXG — had no edge BEFORE any fee was charged. Their
first-half profit was luck on a zero-expectancy signal, and the second
half gave it back.

This file locks in three things:

1. the table in ``app/hold_policy.py`` is exactly re-derivable from the
   raw CSV, so a number can never become a memory;
2. PAXG is the ONLY asset excluded, and it is excluded on gross edge —
   i.e. the exclusion survives any fee assumption, any venue, any ratio;
3. the gate is a pattern-level statement, not a per-window abstention:
   the cell keeps computing, it simply may not present itself as
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


def test_paxg_is_the_only_asset_without_gross_edge():
    """If this ever changes, the gate should change with it."""
    from app.hold_policy import has_gross_edge

    excluded = {(a, tf) for a in ("BTC", "ETH", "BNB", "PAXG")
                for tf in ("tf1", "tf2", "tf3", "tf4")
                if has_gross_edge(a, tf) is False}
    # every excluded cell belongs to PAXG …
    assert all(a == "PAXG" for a, _ in excluded), excluded
    # … and all four of PAXG's cells are excluded
    assert len(excluded) == 4, excluded
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
        assert g is not None
        assert g < 1.0, (tf, g)


def test_crypto_cells_are_gross_positive_at_every_confidence_band():
    """The edge is not concentrated in one confidence slice.

    This is why the gate is per-cell rather than a confidence threshold:
    for BTC/ETH/BNB the mean gross move is positive in all four bands, so
    there is no band in which the pattern stops working.
    """
    rows = [r for r in _rows() if r["asset"] != "PAXG"]
    for lo, hi in ((0, 0.2), (0.2, 0.39), (0.39, 0.60), (0.60, 1.01)):
        band = [r for r in rows if lo <= r["conf"] < hi]
        assert band, (lo, hi)
        assert st.mean(r["gross"] for r in band) > 0, (lo, hi)


def test_paxg_flips_sign_across_bands():
    """The positive half of PAXG's record is one band, and it is noise."""
    rows = [r for r in _rows() if r["asset"] == "PAXG"]
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

    A cell with no signal must still compute and must still be shown; it
    may only refuse to present itself as tradeable. The label exists so
    the card can say why instead of borrowing a green badge.
    """
    from app import hold_policy as hp

    assert hp.TIER_NO_EDGE.startswith("NO MEASURED EDGE")
    # the cell is still managed (it is shown), it just has no edge
    assert hp.gross_edge("PAXG", "tf1") is not None
    # and the reason string names the measured number, not a vibe
    why = hp.tier_reason("PAXG", "tf1")
    assert "gross" in why.lower() or "no measured edge" in why.lower(), why


def test_unmeasured_cell_is_not_silently_tradeable():
    """None means unknown — never a guessed yes."""
    from app.hold_policy import gross_edge, has_gross_edge

    assert gross_edge("SOL", "tf1") is None
    assert has_gross_edge("SOL", "tf1") is None


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
