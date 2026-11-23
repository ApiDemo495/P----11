"""§20 (v10) — TSMOM absolute-momentum era gate + train-robust pins.

Research-backed selection changes (train-only, one-shot holdout):

* ERA gate: an era run whose side trades AGAINST the slow trend
  (sign of the trailing L-minute return disagrees with the side) is
  skipped — Moskowitz–Ooi–Pedersen time-series momentum / regime
  filter family.  Warmup (no L history) → take (full-coverage
  fallback, never a silent skip).  Start-only: an open run is never
  dropped mid-run.
* Row buffer: train WR must clear the holdout-Wilson floor
  (ROW_WR_BUF[tf] + train Wilson lo ≥ 0.66) before a row geometry is
  selectable — gate (iii) needs the Wilson margin on the holdout year.
* Era margin: selected era train mean ≥ ERA_MARGIN (5 bps) so the
  FULL-period sign survives §19's gate-(i) failure mode.
* Maximin era pick within the robustness rung (worst train-year EV
  first, mean tie-break).

These tests pin the constants (pre-registration anchors), the gate
semantics, the batch/live parity twin, and the accountant's gated
state machine.  The holdout is never read here.
"""
from __future__ import annotations

import numpy as np

from app import hold_era as he
from app import hold_policy as hp
from backtest.hold import ERA_GATES, ERA_MARGIN, ROW_WR_BUF, era_sim


# ─────────────────────────── pre-reg anchors ─────────────────────────
def test_selection_constants_are_pinned():
    assert ROW_WR_BUF == {22: 0.665, 44: 0.670, 130: 0.676, 247: 0.690}
    assert ERA_MARGIN == 0.0005
    assert ERA_GATES == (0, 259200, 525600)   # off, 180 d, 365 d
    assert all(v >= 0.66 for v in ROW_WR_BUF.values())


def test_era_gate_field_is_backward_compatible():
    """v8 artifacts predate era_gate_l → 0 → live behaviour unchanged;
    a future artifact's value is read verbatim."""
    exp = int((hp.cell("BTC", "247") or {}).get("selected", {})
              .get("era_gate_l") or 0)
    assert hp.era_gate("BTC", "247") == exp
    # unknown / warmup cells are gated OFF, never crash
    assert hp.era_gate("NOPE", "1") == 0


# ─────────────────────────── gate semantics ──────────────────────────
def _declining_windows(n_win=200, step=300, W=22 * 60, seed=7):
    """Windows over a monotonically DECLINING close series — every
    slow-trend return r < 0 after warmup, deterministically."""
    rng = np.random.default_rng(seed)
    n = n_win * (W // step) + 10
    t = np.arange(n, dtype=np.int64) * step + 1_600_000_000
    c = 50_000.0 * 0.9995 ** np.arange(n)          # strict decline
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * 1.0001
    l = np.minimum(o, c) * 0.9999
    v = np.ones(n)
    starts = np.arange(t[0] + W, t[0] + (n_win + 1) * W, W,
                       dtype=np.int64)
    ie = np.searchsorted(t, starts - step, side="right") - 1
    ix = np.searchsorted(t, starts + W - step, side="right") - 1
    ok = (ie >= 0) & (ix > ie) & (ix < len(t))
    starts, ie, ix = starts[ok], ie[ok], ix[ok]
    entry = c[ie]
    scale = np.full(len(starts), entry[0] * 0.01)
    _ = rng                                        # (kept for parity)
    return t, h, l, c, starts, ie, ix, entry, scale


def test_era_gate_skips_counter_trend_runs_only():
    """gate_l=1100 min → lb_w = round(1100·60/1320) = 50 windows.
    Declining tape ⇒ r < 0: a +1 run starting after warmup is
    skipped; a −1 run (aligned) and any warmup run (w < lb_w) are
    taken."""
    t, h, l, c, starts, ie, ix, entry, scale = _declining_windows()
    n = len(starts)
    assert n >= 100

    # run 0: −1 from the start (warmup, must be taken)
    # run 1: +1 from w=60 (counter-trend, must be skipped when on)
    sides_aligned = -np.ones(n, dtype=np.int64)
    sides_aligned[60:] = 1
    # mirror: +1 warmup run + −1 aligned run (both taken)
    sides_aligned2 = np.ones(n, dtype=np.int64)
    sides_aligned2[60:] = -1

    off = era_sim(sides_aligned, starts, ie, ix, entry, scale,
                  t, h, l, c, 1.0, 999.0, 12.0, gate_l_min=0)
    on = era_sim(sides_aligned, starts, ie, ix, entry, scale,
                 t, h, l, c, 1.0, 999.0, 12.0, gate_l_min=1100)
    assert len(off) == 2 and off[0]["start_i"] == 0 and off[1]["start_i"] == 60
    assert len(on) == 1 and on[0]["start_i"] == 0      # warmup run kept

    on2 = era_sim(sides_aligned2, starts, ie, ix, entry, scale,
                  t, h, l, c, 1.0, 999.0, 12.0, gate_l_min=1100)
    assert len(on2) == 2                               # both aligned


def test_track_eras_gate_parity_twin():
    """app.hold_era.track_eras mirrors era_sim's gate exactly."""
    from app.hold_era import track_eras
    t, h, l, c, starts, ie, ix, entry, scale = _declining_windows()
    n = len(starts)
    sides = -np.ones(n, dtype=np.int64)
    sides[60:] = 1
    for gl in (0, 1100, 525600):
        ref = era_sim(sides, starts, ie, ix, entry, scale,
                      t, h, l, c, 1.5, 2.0, 12.0, gate_l_min=gl)
        got = track_eras(sides, starts, ie, ix, entry, scale,
                         t, h, l, c, 1.5, 2.0, 12.0, gate_l_min=gl)
        assert len(got) == len(ref)
        for a, b in zip(got, ref):
            assert a["start_i"] == b["start_i"]
            assert a["n_windows"] == b["n_windows"]
            assert a["outcome"] == b["outcome"]
            assert a["pnl_gross"] == b["pnl_gross"]
            assert a["pnl_net"] == b["pnl_net"]


# ─────────────────────── accountant (live) gate ──────────────────────
def test_accountant_gate_skip_close_and_persistence(tmp_path):
    path = str(tmp_path / "era.json")
    acct = he.EraAccountant("BTC", "247", path=path)
    # open run A (+1)
    ev = acct.on_window(1, 100.0, 1.0, 1_000.0, gate_ok=True)
    assert ev["event"] == "era_open" and acct.in_position
    # flip into a gate-FAILED run B (−1): A closes at the flip price
    ev = acct.on_window(-1, 99.0, 1.0, 2_000.0, gate_ok=False)
    assert ev["event"] == "era_gated_close"
    assert not acct.in_position and acct.skip_side == -1
    assert acct.history[-1]["outcome"] == 2
    assert acct.history[-1]["exit"] == 99.0
    # same-side window inside the skipped run: nothing happens
    ev = acct.on_window(-1, 98.0, 1.0, 3_000.0, gate_ok=False)
    assert ev["event"] == "window" and not acct.in_position
    # a gate check flipping mid-skip must NOT reopen (start-only rule)
    ev = acct.on_window(-1, 98.5, 1.0, 4_000.0, gate_ok=True)
    assert ev["event"] == "window" and not acct.in_position
    # flip to a gate-PASSING run (+1): opens, skip cleared
    ev = acct.on_window(1, 97.0, 1.0, 5_000.0, gate_ok=True)
    assert ev["event"] == "era_open" and acct.in_position
    assert acct.skip_side is None
    # mid-run gate flip is ignored (batch checks at run start only)
    n0 = acct.n_windows
    ev = acct.on_window(1, 97.5, 1.0, 6_000.0, gate_ok=False)
    assert ev["event"] == "window" and acct.in_position
    assert acct.n_windows == n0 + 1

    # persistence: skip_side and position survive a restart
    again = he.EraAccountant("BTC", "247", path=path)
    assert again.in_position and again.side == 1
    assert again.skip_side is None
