"""Win rate is a DIAL, and the dial is not edge.

Everything here is exact arithmetic on a drift-diffusion (or, in the
drift-free limit, pure first-touch geometry). It exists because the
project's headline claim — "WR >= 0.66 after friction" — has been
re-selected many times without ever stating what directional edge would
be REQUIRED to produce it. In the drift-free limit:

    P(win) = sl / (tp + sl) = 1 / (1 + r),      r = tp/sl
    EV     = P(win)(tp - f) - P(loss)(sl + f) = -f     for EVERY r

so win rate is set by the barrier ratio alone, and moves the expected
value by exactly nothing. Raising the win rate therefore requires
directional edge, not geometry. The numbers below are the required edge,
computed rather than asserted.
"""
from __future__ import annotations

import math

import pytest


def p_hit_first(a: float, b: float, lam: float) -> float:
    """P(touch +a before -b) for X_t = (lam/2)·t + W_t (unit variance)."""
    if abs(lam) < 1e-12:
        return b / (a + b)
    return (math.exp(lam * b) - 1.0) / (math.exp(lam * b) - math.exp(-lam * a))


def fair_win_rate(tp: float, sl: float) -> float:
    """Drift-free first-touch win rate (the geometry floor)."""
    return sl / (tp + sl)


# ══════════════════ geometry sets the win rate ══════════════════

def test_eighty_percent_win_rate_is_exactly_a_quarter_reward_risk():
    assert fair_win_rate(1.0, 4.0) == pytest.approx(0.80)
    # ... and it is not special: 66.7 % is exactly r = 0.5, 50 % is r = 1
    assert fair_win_rate(1.0, 2.0) == pytest.approx(2 / 3)
    assert fair_win_rate(1.0, 1.0) == pytest.approx(0.50)


def test_win_rate_does_not_move_expectancy():
    """EV is -friction at EVERY barrier ratio, in the drift-free limit."""
    friction = 28.872
    for tp, sl in ((1, 4), (1, 2), (1, 1), (2, 1), (3, 1), (10, 1)):
        p = fair_win_rate(tp, sl)
        ev = p * (tp - friction) - (1 - p) * (sl + friction)
        assert ev == pytest.approx(-friction, abs=1e-9), (tp, sl)


def test_a_66_percent_win_rate_is_risking_two_to_make_one():
    """66.7 % is not a good number — it is the win rate of a pair that
    risks 2 to make 1. It needs no edge whatsoever."""
    assert fair_win_rate(1.0, 2.0) == pytest.approx(2 / 3, abs=1e-12)


# ══════════════════ and what edge would actually be required ══════════════════

def _required_drift(tp: float, sl: float, target: float) -> float:
    lo, hi = 0.0, 80.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if p_hit_first(tp, sl, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_required_edge_for_the_headline_win_rates():
    """z = required drift per window, in units of the window's own sigma."""
    # risking 2 to make 1 already reaches 70 % on a ~0.1 sigma edge
    assert _required_drift(1.0, 2.0, 0.70) < 0.12
    # a 1:1 pair needs 0.85 window-sigma of drift for the same 70 %
    # (0.847 for symmetric barriers: lam = 2z = 1.694)
    assert _required_drift(1.0, 1.0, 0.70) == pytest.approx(0.847, abs=0.01)
    # 80 % on a 1:1 pair needs 1.39 window-sigma of drift per window
    assert _required_drift(1.0, 1.0, 0.80) == pytest.approx(1.386, abs=0.01)


def test_the_66_percent_gate_reads_a_geometry_not_an_edge():
    """Risks 2 to make 1 clears "WR >= 0.66" with ZERO edge — which is
    exactly why the gate proves nothing on its own."""
    assert fair_win_rate(1.0, 2.0) >= 0.66


def test_mandate_one_to_one_cells_are_the_hard_case():
    """BTC / BNB ship R = 1.0 in app/mandate_cfg.py while their own direction
    rules are recorded at 52-53 % accuracy. On a 1:1 pair that lands the
    barrier win rate near 53 %, not the documented 66 %.

    SOL carries its OWN prior (R = 1.3), not a copy of BTC's: it is the
    higher-volatility asset and its separate weightage is a requirement, so
    it must NOT be asserted at 1.0.
    """
    from app import mandate_cfg

    for sym in ("BTC", "BNB"):
        cell = mandate_cfg.cfg(sym)
        assert cell is not None
        assert float(cell["R"]) == pytest.approx(1.0), sym
    # SOL is separate-weightage and above the floor
    sol_r = float(mandate_cfg.cfg("SOL")["R"])
    assert sol_r >= 1.0
    assert sol_r != pytest.approx(float(mandate_cfg.cfg("BTC")["R"]))
    # 52.5 % direction accuracy == z 0.063 == 53.1 % barrier win rate,
    # against the 66 % the configuration is documented as having reached
    z = 0.0627
    assert p_hit_first(1.0, 1.0, 2 * z) == pytest.approx(0.531, abs=0.005)
    # and 66 % on that same 1:1 pair would need 0.66 window-sigma of drift
    # (lam = 2z = 1.33), an order of magnitude more edge than the rule
    # was measured to have
    assert _required_drift(1.0, 1.0, 0.66) == pytest.approx(0.663, abs=0.01)
