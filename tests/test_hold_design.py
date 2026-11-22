"""Hold-design continuous benchmark — the BOTH-targets system.

Targets (REAL_RESULTS §13): every-window first-touch row WR ≥ 66 % AND
system EV > 0 after real friction (charged per era round trip), on real
data, no abstention.  These tests pin the invariants that make the
measurement honest:

* era P&L is signed by the held side (short eras were once inverted —
  this is the regression test),
* friction is charged once per era round trip, never relaxed,
* every side covers every window (no skipped analyses),
* the committed results actually meet both targets where claimed.
"""
from __future__ import annotations

import json
import os

import numpy as np


def _synth_bars(n=4000, step=300, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * step + 1_600_000_000
    rets = rng.normal(0, 0.0008, n)
    c = 50_000.0 * np.exp(np.cumsum(rets))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0004, n)))
    v = np.ones(n)
    return t, o, h, l, c, v, step


def test_era_pnl_signed_by_side():
    """A SHORT era that profits (price falls) must have positive P&L —
    the exit price is raw until multiplied by the held side."""
    from backtest.hold import era_sim
    t, o, h, l, c, v, step = _synth_bars()
    # force a clean −10 % slide over the era's bars
    c = c.copy()
    c[:] = c[0] * 0.9 ** np.linspace(0, 1, len(c))
    h = np.maximum(o, c) * 1.0001
    l = np.minimum(o, c) * 0.9999
    n_win = 200
    W = 22 * 60
    starts = np.arange(t[0] + W, t[0] + (n_win + 1) * W, W, dtype=np.int64)
    ie = np.searchsorted(t, starts - step, side="right") - 1
    ix = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (ie >= 0) & (ix > ie) & (ix < len(t))
    starts, ie, ix = starts[ok], ie[ok], ix[ok]
    entry = c[ie]
    scale = np.full(len(starts), entry[0] * 0.01)
    sides = -np.ones(len(starts), dtype=int)
    recs = era_sim(sides, starts, ie, ix, entry, scale, t, h, l, c,
                   k=1.0, m=999.0, fr_bps=12.0)
    assert len(recs) == 1                       # one short era
    # the era spans part of the slide: short entry→exit is a GAIN
    assert recs[0]["pnl_gross"] > 0.01, recs[0]
    # friction: one round trip at 12 bps/side
    assert abs(recs[0]["pnl_net"] -
               (recs[0]["pnl_gross"] - 0.0024)) < 1e-12


def test_era_friction_charged_once_per_era():
    from backtest.hold import era_sim
    t, o, h, l, c, v, step = _synth_bars()
    n_win = 120
    W = 22 * 60
    starts = np.arange(t[0] + W, t[0] + (n_win + 1) * W, W, dtype=np.int64)
    ie = np.searchsorted(t, starts - step, side="right") - 1
    ix = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (ie >= 0) & (ix > ie) & (ix < len(t))
    starts, ie, ix = starts[ok], ie[ok], ix[ok]
    entry = c[ie]
    scale = np.full(len(starts), entry[0] * 0.01)
    # alternating sides → one era per window
    sides = np.where(np.arange(len(starts)) % 2 == 0, 1, -1).astype(int)
    recs = era_sim(sides, starts, ie, ix, entry, scale, t, h, l, c,
                   k=1.0, m=999.0, fr_bps=12.0)
    assert len(recs) == len(starts)
    for r in recs:
        assert abs(r["pnl_net"] - (r["pnl_gross"] - 0.0024)) < 1e-12


def test_every_side_covers_every_window():
    """No abstention: every candidate side is ±1 on every window."""
    from backtest.hold import build_sides, composite_sides
    t, o, h, l, c, v, step = _synth_bars()
    W = 22 * 60
    starts = np.arange(t[0] + W, t[0] + 800 * W, W, dtype=np.int64)
    ie = np.searchsorted(t, starts - step, side="right") - 1
    ix = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (ie >= 0) & (ix > ie) & (ix < len(t))
    starts, ie, ix = starts[ok], ie[ok], ix[ok]
    entry = c[ie]
    moves = c[ix] - entry
    sides = build_sides(t, o, h, l, c, v, step, starts, ie, moves, 22)
    sides = composite_sides(sides, None, None)
    for name, s in sides.items():
        assert np.all(np.isin(s, (-1, 1))), name
        assert len(s) == len(starts), name


def test_committed_results_meet_both_targets():
    """The shipped hold_wr.json is the system's proof: every ok cell
    meets BOTH targets on the full period, and every multi-year cell
    also reports the holdout split honestly."""
    path = os.path.join(os.path.dirname(__file__), "..",
                        "backtest", "results", "hold_wr.json")
    assert os.path.exists(path), "hold_wr.json missing — run backtest.hold"
    doc = json.load(open(path))
    ok_cells = {k: c for k, c in doc.items() if c.get("status") == "ok"}
    assert ok_cells, "no graded cells"
    for key, cell in ok_cells.items():
        sel = cell["selected"]
        full = sel["full"]
        assert full["row_wr"] >= 0.66, (key, full["row_wr"])
        assert full["era_ev_net"] > 0, (key, full["era_ev_net"])
        assert full["row_n"] == cell["n_windows"], key   # every window
        assert cell["targets"]["full_row_wr_ge_66"] is True, key
        assert cell["targets"]["full_era_ev_net_gt_0"] is True, key
        if not cell.get("single_year"):
            # honest OOF reporting exists for multi-year cells
            assert "holdout_row_wr_ge_66" in cell["targets"], key
            assert sel["selection"] == "train-only", key
        else:
            assert "IN-SAMPLE" in sel["selection"], key
    # ungradeable cells stay disclosed, never forced
    for key, cell in doc.items():
        if cell.get("status") == "ungradeable":
            assert "cannot resolve" in cell.get("why", ""), key
