"""Cost gate — the only honest reason to issue a directional call.

Background
==========
The 2026-11 falsification programme (see `backtest/results/EDGE_SCAN.md`)
settled every candidate structure on a *sequential, single-position* book
with taker fills at the next bar open, stop-wins-ties, and per-fold purged
out-of-fold evaluation.  The result is uniform: **gross** expectancy is
between -3 and +8 bp per round trip on every lane, while friction is
12-24 bp.  Net expectancy is therefore negative everywhere, and the deficit
is structural (fee / sigma), not a tuning problem.

This module turns that measurement into a product decision.  A card may
still show a direction — the model is a forecast, and forecasts are allowed
to be uncertain — but the *tradeability* verdict comes from here:

    expected_net_bp = measured_gross_bp - friction_bp

with `measured_gross_bp` read from an explicit registry of out-of-fold
measurements (never from the live fit, which would be circular) and
`friction_bp` from the live microstructure cost model.  When the margin is
not there, the state is `blocked` and the note says by how much.

That is deliberately unflattering: a system that says "no edge at this
cost" is worth more than one that predicts 51 % and loses 12 bp a trade.
"""

from __future__ import annotations

# Minimum expected net edge (bp per round trip) we require before calling a
# lane tradable.  Half of one tick of slippage, so a lane cannot pass on
# rounding noise.
MIN_MARGIN_BP = 2.0

# Horizon buckets in minutes -> the lane a measurement belongs to.
_BUCKETS = ((0, 90, "intraday"), (90, 720, "session"),
            (720, 2880, "swing"), (2880, 10 ** 9, "position"))

# Out-of-fold gross expectancy per round trip, sequential book, taker fills.
# n / t_stat are carried so the note can never overstate the evidence.
MEASURED: dict[tuple[str, str], dict] = {
    ("BTC", "intraday"): {"gross_bp": 0.6, "n": 15664, "t_stat": 3.0,
                          "source": "ensemble 1m h=60, 5 folds"},
    ("ETH", "intraday"): {"gross_bp": -1.0, "n": 858, "t_stat": -3.0,
                          "source": "ensemble 1h h=12, 5 folds"},
    ("BNB", "intraday"): {"gross_bp": 3.0, "n": 899, "t_stat": 2.4,
                          "source": "ensemble 1h h=12, 5 folds"},
    ("PAXG", "intraday"): {"gross_bp": -0.9, "n": 851, "t_stat": -1.0,
                           "source": "ensemble 1h h=24, 5 folds"},
    ("BTC", "session"): {"gross_bp": 5.4, "n": 261, "t_stat": 1.3,
                         "source": "ensemble 1h pt3/sl1.5, 5 folds"},
    ("ETH", "session"): {"gross_bp": 7.7, "n": 241, "t_stat": 0.6,
                         "source": "ensemble 1h pt3/sl1.5, 5 folds"},
    ("BNB", "session"): {"gross_bp": 7.0, "n": 254, "t_stat": 0.9,
                         "source": "ensemble 1h pt3/sl1.5, 5 folds"},
    ("PAXG", "session"): {"gross_bp": -8.4, "n": 219, "t_stat": -7.2,
                          "source": "ensemble 1h pt3/sl1.5, 5 folds"},
    ("BTC", "swing"): {"gross_bp": 3.6, "n": 194, "t_stat": 1.6,
                       "source": "cascade fade k15 z4, 5 folds"},
    ("ETH", "swing"): {"gross_bp": -2.4, "n": 237, "t_stat": -2.2,
                       "source": "cascade fade k15 z4, 5 folds"},
    ("BNB", "swing"): {"gross_bp": 1.3, "n": 196, "t_stat": 2.1,
                       "source": "cascade fade k15 z4, 5 folds"},
    ("PAXG", "swing"): {"gross_bp": 0.0, "n": 0, "t_stat": 0.0,
                        "source": "no sub-hour archive"},
}

_UNMEASURED = {"gross_bp": 0.0, "n": 0, "t_stat": 0.0,
               "source": "no out-of-fold measurement"}


def bucket(minutes: float) -> str:
    """Map a holding horizon in minutes to its measurement bucket."""
    m = float(max(0.0, minutes))
    for lo, hi, name in _BUCKETS:
        if lo <= m < hi:
            return name
    return "position"


def measured(asset: str, tf: str) -> dict:
    """Measured gross expectancy for one lane (never a live fit)."""
    return dict(MEASURED.get((str(asset).upper(), tf), _UNMEASURED))


def required_accuracy(friction_bps: float, sigma_bps: float,
                      k: float = 2.0) -> float:
    """Directional hit rate needed to break even with ±kσ barriers.

    Symmetric barriers of width `k·σ` pay `(2p-1)·k·σ` gross, so the break
    even hit rate is `0.5 + friction / (2·k·σ)`.  This is the number the
    lab should be judged against instead of an absolute accuracy target: at
    1 h on BTC (σ≈40 bp) a 12 bp round trip demands 57.5 % accuracy with
    ±2σ barriers, while at a 5-day hold (σ≈440 bp) it demands 50.7 % — the
    same edge, a totally different requirement.
    """
    s = float(sigma_bps)
    if s <= 0:
        return 1.0
    return 0.5 + float(friction_bps) / (2.0 * float(k) * s)


def breakeven_cost_bps(gross_bp: float) -> float:
    """Round-trip friction at which the measured edge produces zero net."""
    return round(float(gross_bp), 2)


def required_gross_bp(friction_bps: float, margin_bp: float = MIN_MARGIN_BP) -> float:
    """Gross expectancy a lane must show before it may be called tradable."""
    return round(float(friction_bps) + float(margin_bp), 2)


def gate(asset: str, tf: str, friction_bps: float,
         margin_bp: float = MIN_MARGIN_BP) -> dict:
    """Tradeability verdict for one lane at the live cost level.

    state
      open     measured net expectancy clears the margin
      thin     measured net is positive but inside the margin
      blocked  measured net is negative — the fee is not covered
    """
    m = measured(asset, tf)
    gross = float(m["gross_bp"])
    # evidence strength is separate from the cost verdict: a positive but
    # tiny out-of-fold number (|t| < 2 or n < 100 rounds) is a hypothesis,
    # not a measurement, and the note must say so.
    evidence = ("insufficient"
                if (m["n"] < 100 or abs(float(m["t_stat"])) < 2.0)
                else "supported")
    net = gross - float(friction_bps)
    if net >= margin_bp:
        state = "open"
    elif net > 0:
        state = "thin"
    else:
        state = "blocked"
    need = required_gross_bp(friction_bps, margin_bp)
    if m["n"] == 0:
        note = f"no OOF measurement for {asset} {tf}; gross assumed 0"
    else:
        note = (f"OOF gross {gross:+.1f}bp (n={m['n']}, t={m['t_stat']:+.1f}, "
                f"evidence {evidence}) vs {friction_bps:.1f}bp friction — "
                f"needs {need:+.1f}bp")
    return {
        "state": state,
        "asset": str(asset).upper(),
        "tf": tf,
        "gross_bp": round(gross, 2),
        "friction_bps": round(float(friction_bps), 2),
        "expected_net_bp": round(net, 2),
        "required_gross_bp": need,
        "breakeven_cost_bps": breakeven_cost_bps(gross),
        "evidence": evidence,
        "n": int(m["n"]),
        "t_stat": float(m["t_stat"]),
        "source": m["source"],
        "note": note,
    }
