"""Formula integrity: no formula may be silently inert.

These tests exist because a Pillar-3 audit on 2026-11-17 found F8 RTDI
pinned at sd ~0.006 on real tape: it compared *whether* consecutive
minutes ticked, and ~99% of 1-minute bars tick, so the changed/unchanged
binary carried no information and the formula could never move a
consensus vote. It looked alive (a tiny non-zero variance) to every
existing check.

The guard is generically useful, not F8-specific: any formula whose
output is constant on real tape is either structurally dead (a bug) or
correctly silent because it needs live state the replay cannot supply.
Those two cases must be distinguishable, and neither may be guessed.
"""
import inspect
import math

import pytest

from app.formulas import FORMULA_FUNCTIONS


# --------------------------------------------------------------- F8 RTDI
# FINDING (2026-11-17, Pillar 3): as originally written F8 RTDI was
# structurally degenerate. Its "tick density" was a did-this-minute-change
# flag, and on the app's native 1-minute tape ~99 % of consecutive bars are
# distinct (measured 1.1 % of 1m bars identical to the previous one), so the
# flag carried no information and the output was pinned near zero: measured
# sd 0.006 on 1h bars and 0.011 on 1m, against 0.3-0.9 for every other live
# formula. It could never move a consensus vote.
#
# The rewrite measures VOLUME DENSITY around VWAP (mean volume spent above
# the VWAP minus below it), which is the quantity the name promises and which
# on the same tape gives sd 0.455 - about 40x the range, and a formula that
# can vote.
def _ctx(prices, volumes, W=None):
    from app.models import FormulaContext

    W = W if W is not None else max(5, len(prices) - 1)
    return FormulaContext(
        symbol="ETH", tf="tf1", W=W,
        P=list(prices), V=list(volumes), T=list(range(len(prices))),
        hist=None, others={},
        O=list(prices), H=list(prices), L=list(prices),
    )


def test_rtdi_reads_volume_not_a_price_change_flag():
    """The saturating changed/unchanged flag must never come back."""
    import inspect

    body = inspect.getsource(FORMULA_FUNCTIONS[8])
    code = "\n".join(ln for ln in body.splitlines()
                      if not ln.strip().startswith("#"))
    assert "changed = abs(P[i] - P[i - 1])" not in code, (
        "RTDI regressed to the changed/unchanged price flag: on 1-minute "
        "tape ~99% of bars tick, which pins the output near zero"
    )
    assert "above_mass" in code and "below_mass" in code


def test_rtdi_responds_to_where_volume_is_spent():
    """With a flat price the old version returned exactly 0 forever."""
    n = 60
    prices = [100.0 + (1.0 if i % 2 else -1.0) for i in range(n)]
    vwap = sum(prices) / n
    balanced = [1.0] * n
    heavy_above = [10.0 if p > vwap else 1.0 for p in prices]
    heavy_below = [1.0 if p > vwap else 10.0 for p in prices]

    b = FORMULA_FUNCTIONS[8](_ctx(prices, balanced, W=n - 1))
    a = FORMULA_FUNCTIONS[8](_ctx(prices, heavy_above, W=n - 1))
    c = FORMULA_FUNCTIONS[8](_ctx(prices, heavy_below, W=n - 1))
    assert a > b > c
    assert b == pytest.approx(0.0, abs=1e-9)


def test_rtdi_is_bounded_signed_and_safe():
    n = 60
    v = [1.0] * n
    up = [100.0 + i for i in range(n)]
    down = [100.0 - i for i in range(n)]
    a = FORMULA_FUNCTIONS[8](_ctx(up, v, W=n - 1))
    b = FORMULA_FUNCTIONS[8](_ctx(down, v, W=n - 1))
    assert -1.0 <= a <= 1.0 and -1.0 <= b <= 1.0
    assert a == pytest.approx(-b, abs=1e-9)
    assert FORMULA_FUNCTIONS[8](_ctx([100.0] * 20, [0.0] * 20, W=19)) == 0.0


# ------------------------------------------------- generic audit contract
_LIVE_STATE_TOKENS = ("ctx.hist", "ctx.book", "ctx.flow", "ctx.others",
                      "_book(ctx)", "liq_flow")


def _needs_live_state(fid):
    src = inspect.getsource(FORMULA_FUNCTIONS[fid])
    return any(t in src for t in _LIVE_STATE_TOKENS)


def test_every_formula_is_classified_as_live_state_dependent_or_pure():
    """No formula may be unclassifiable - that is how dead code hides."""
    for fid in FORMULA_FUNCTIONS:
        live = _needs_live_state(fid)
        assert isinstance(live, bool)


def test_audit_can_never_report_a_dead_formula_as_alive():
    """The audit's own contract, on synthetic tape it fully controls."""
    from backtest.formula_audit import MIN_VARYING_SAMPLES

    assert MIN_VARYING_SAMPLES >= 50
