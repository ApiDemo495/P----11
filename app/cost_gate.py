"""Cost gate — the only honest reason to issue a directional call.

(Merged from the f07a39b8 line, where this module lived at
``app/edge_gate.py``; renamed because the 605bdcca line carries a different
statistical sleeve-certification module under that name. Its report is
``backtest/results/DERIVS_SCAN.md`` and its contract suite is
``tests/test_cost_gate.py``.)

Background
==========
The 2026-11 falsification programme (see `backtest/results/DERIVS_SCAN.md`)
settled every candidate per-bar structure on a *sequential, single-position*
book with taker fills at the next bar open, stop-wins-ties, and per-fold
purged out-of-fold evaluation.  The result is uniform on those lanes:
**gross** expectancy is between -3 and +8 bp per round trip, while friction
is 12-24 bp.  Net expectancy is therefore negative on every measured
per-bar lane, and the deficit is structural (fee / sigma), not a tuning
problem.  (Two fourth-pass event sleeves once read positive and now sit in
``SLEEVES`` as ``rejected``: their numbers did not reproduce from a
from-scratch rebuild — see ``docs/ALPHA_RESEARCH.md`` §0.  Certified event
sleeves live in ``app/edge_gate.py``, not here.)

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

# ─────────────────────── event sleeves (third pass) ───────────────────────
# Measured with `backtest.derivs_scan --only events` on the multi-year archive
# (spot 1h 2024-01 -> 2026-09, BTC USD-M perp 1h + funding/basis 2023-01 ->
# 2025-11), five purged folds, asymmetric convex exits, maker-pullback entry
# filled one bar later (adverse selection).  `status` is the verdict, not a
# hope: "rejected" means it was tried and did not clear the fee.
SLEEVES: dict[str, dict] = {
    "eth_catchup": {
        "asset": "ETH", "tf": "session", "status": "rejected",
        "trigger": "BTC 3h impulse >= 0.5σ while ETH has priced < 30 % of it",
        "execution": "maker pullback 0.25σ, next-bar fill, stop 1.0σ, "
                     "trail 3.0σ, 48h horizon",
        "period": "2024-01-01 -> 2026-09-30", "n": 180,
        "gross_bp": 25.0, "fee_bp": 8.0, "net_bp": 17.0,
        "profit_factor": 1.31, "boot_t": 1.26, "folds": "4/5",
        "qualifies": False,
        # RETRACTED (docs/ALPHA_RESEARCH.md §0): the +17 bp / PF 1.31
        # reading did not reproduce from a from-scratch rebuild —
        # overlapping trades counted as independent, friction charged per
        # episode in places, best-of-535 selection. The numbers stay on the
        # record so nobody re-discovers them as new; the verdict is
        # rejected, never shown as live evidence. The certified successor
        # is catchup_subhour_spot_15m::ETH in app/edge_gate.py.
        "note": "RETRACTED per ALPHA_RESEARCH §0 (not reproducible: "
                "overlapping trades, per-episode friction, best-of-535 "
                "selection) — kept on record, never live evidence",
    },
    "btc_squeeze": {
        "asset": "BTC", "tf": "swing", "status": "rejected",
        "trigger": "vol compression + funding crowding + breakout with "
                   "volume surge",
        "execution": "same convex plan", "period": "2023-01 -> 2025-11",
        "n": 172, "gross_bp": 14.4, "fee_bp": 8.0, "net_bp": 6.4,
        "profit_factor": 1.24, "boot_t": 1.02, "folds": "4/5",
        "note": "RETRACTED per ALPHA_RESEARCH §0 (vol_z::BTC scores -24.3 "
                "bp OOF on rebuild) — kept on record, never live evidence",
    },
    "btc_basis_fade": {
        "asset": "BTC", "tf": "swing", "status": "rejected",
        "trigger": "|funding z| >= 1.5 with basis dislocation, volume "
                   "climax and a rejection wick",
        "execution": "same convex plan", "period": "2023-01 -> 2025-11",
        "n": 3, "gross_bp": -33.0, "fee_bp": 8.0, "net_bp": -41.0,
        "profit_factor": 0.0, "boot_t": 0.0, "folds": "0/5",
        "note": "too few exhaustion events to power a test, and negative",
    },
    "bnb_catchup": {
        "asset": "BNB", "tf": "session", "status": "rejected",
        "trigger": "as eth_catchup", "execution": "as eth_catchup",
        "period": "2024-01 -> 2026-09", "n": 352, "gross_bp": 1.1,
        "fee_bp": 8.0, "net_bp": -6.9, "profit_factor": 0.83,
        "boot_t": -1.23, "folds": "1/5",
        "note": "BNB does not catch up to BTC impulses in this sample",
    },
    "paxg_flight": {
        "asset": "PAXG", "tf": "session", "status": "rejected",
        "trigger": "as eth_catchup against gold",
        "execution": "as eth_catchup", "period": "2026-01 -> 2026-09",
        "n": 156, "gross_bp": 0.4, "fee_bp": 8.0, "net_bp": -7.6,
        "profit_factor": 0.67, "boot_t": -1.44, "folds": "1/5",
        "note": "gold decouples, it does not catch up",
    },
}


def bucket(minutes: float) -> str:
    """Map a holding horizon in minutes to its measurement bucket."""
    m = float(max(0.0, minutes))
    for lo, hi, name in _BUCKETS:
        if lo <= m < hi:
            return name
    return "position"


def sleeve_gate(name: str, friction_bps: float = 8.0) -> dict:
    """Verdict for one event sleeve at its own execution cost."""
    s = SLEEVES.get(name)
    if not s:
        return {"state": "unknown", "name": name,
                "note": "no such sleeve"}
    net = float(s["gross_bp"]) - float(friction_bps)
    t = abs(float(s["boot_t"]))
    pf = float(s.get("profit_factor", 0.0))
    # the acceptance bar the brief set: net > 0, |t| > 2, PF > 1.25
    qualifies = bool(net > 0 and t >= 2.0 and pf > 1.25)
    if s["status"] == "rejected" or net <= 0:
        state = "blocked"
    elif qualifies:
        state = "open"
    else:
        state = "unproven"
    return {"state": state, "name": name, "asset": s["asset"], "tf": s["tf"],
            "gross_bp": s["gross_bp"], "fee_bp": float(friction_bps),
            "expected_net_bp": round(net, 2),
            "profit_factor": pf, "qualifies": qualifies, "n": s["n"],
            "boot_t": s["boot_t"], "folds": s["folds"],
            "trigger": s["trigger"], "execution": s["execution"],
            "note": s["note"]}


def gates_for(asset: str, friction_bps: float = 8.0) -> list[dict]:
    """Every measured sleeve that trades this asset (verdicts included)."""
    want = str(asset).upper()
    return [sleeve_gate(k, friction_bps) for k, v in SLEEVES.items()
            if str(v.get("asset", "")).upper() == want]


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
      open     measured net expectancy clears the margin *and* the evidence
               is strong (|t| >= 2)
      thin     positive net, inside the margin
      unproven positive net but |t| < 2 — a hypothesis, not an edge
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
    if state in ("open", "thin") and evidence == "insufficient":
        state = "unproven"
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
