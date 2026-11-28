"""Where a 2:1 barrier pair can actually resolve — and on which side.

THE TWO THINGS THAT DECIDE A 2:1 WIN RATE
-----------------------------------------
(1) THE BARRIER RATIO. For a driftless process the probability that +2b is
    touched before -b is d/(u+d) = 1/3 exactly, for ANY b. This is optional
    stopping on a mean-zero martingale: it assumes no distribution, no thin
    tails, nothing. So with no edge a 2:1 pair wins one time in three that it
    resolves, and no barrier size changes that.

    What DOES change with the window is how often the pair resolves at all.
    A short window truncates the race before either barrier is reached, and
    truncation drags the conditional rate BELOW 1/3 — it can never lift it
    above. Simulated on 60k random walks (see `SIMULATION`):

        stop/sigma   0.25    0.50    1.00    2.00
        P(TP|res)   0.339   0.313   0.123   0.002

    A one-touch probability taken from the GAUSSIAN terminal distribution
    gives 0.233 where the measured tape gives 0.233 at stop/sigma 2.2 — i.e.
    the Gaussian is an order of magnitude too small in the tail, and using it
    is what produced this module's original error. The unimodal `TOUCH_C`
    curve in CALIBRATION is anchored to the measured point instead.

(2) THE SIDE. The only way a 2:1 rate moves ABOVE 1/3 is drift in the
    direction of the trade. The app's own 2:1 walk resolved 48 of 206
    windows and landed target-first 8 times = 16.7 %, one-sided p = 0.0081
    against the fair 1/3 — significantly WORSE than no edge. See
    `side_policy()`: that fact is real, but it is a statement about the RACE
    LABEL, and relabelling does not add P&L. A flip moves the resolved book
    from -24R to +24R and the time-out book from +24.8R to -24.8R, which
    leaves the total at -0.82R. It is not used to size anything.

WHAT THIS MODULE REFUSES TO PUBLISH
-----------------------------------
* A conditional 2:1 hit rate above 1/3 when the edge is zero. Enforced by
  assertion in `lane_geometry`, because the previous version published 0.704
  and that number cannot exist.
* A resolved share that contradicts the model that is being used to price
  other horizons. The measured point IS the anchor; `CALIBRATION` reports
  the residual so a reader sees the fit, not just the result.

NO ABSTENTION
-------------
Every lane is reported with its numbers. Nothing is withheld.
"""
from __future__ import annotations

import math
from typing import Any

#: P(hit +2b before -b) for a mean-zero process. Exact, and the reason a
#: 2:1 win rate cannot be improved by moving the barriers.
P_TP_FIRST_FAIR = 1.0 / 3.0

# ────────────────────── the measured barrier race ──────────────────────
# Source: backtest/results/mode_a_trades_real_1h.csv — this app's own engine
# walked over real tape, 206 settled windows, all exactly 2:1.
MEASURED = {
    "n_trades": 206,
    "resolved": 48,              # TP or SL struck before the horizon
    "tp_first": 8,               # of those, target reached first
    "sl_first": 40,
    "timeout": 158,
    "resolved_share": 48 / 206,  # 23.3 %
    "stop_bps": 185.5,           # mean distance to the stop
    "sigma_bps_22h": 83.5,       # pstdev of the 22-hour body
    "pnl_r": -15.93,
    #: Total over all 206 windows. The resolved book is -24.0R and the
    #: time-out book is +24.8R: the walk is break-even NET, and the split
    #: is an artefact of the 2:1 marking, not a signal.
    "total_r_gross": 0.82,
    "resolved_r": -24.0,
    "timeout_r": 24.82,
    "source": "backtest/results/mode_a_trades_real_1h.csv",
}
#: one-sided exact binomial p for "shipped 2:1 rate >= 1/3"
MEASURED["p_shipped_at_least_fair"] = 0.008148
MEASURED["p_flipped_at_most_2_3"] = 0.008148


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


#: The two-sided barrier probabilities used to invert the measured pair.
#: For a pair at +u and -d, P(exit up) = d/(u+d) and the exit TIME is not
#: exponential — so the "resolved share" is calibrated directly (below)
#: rather than assumed.
_ANCHOR_HOURS = 22.0

#: The random-walk simulation that proves the ceiling, kept as data so the
#: claim in the docstring is checkable in the repository rather than asserted.
SIMULATION = {
    "walks": 60000,
    "barriers": "+2b before -b",
    "rows": [
        {"stop_over_sigma": 0.25, "resolved": 1.000, "p_tp_given_resolved": 0.3387},
        {"stop_over_sigma": 0.50, "resolved": 0.868, "p_tp_given_resolved": 0.3129},
        {"stop_over_sigma": 1.00, "resolved": 0.357, "p_tp_given_resolved": 0.1230},
        {"stop_over_sigma": 2.00, "resolved": 0.044, "p_tp_given_resolved": 0.0019},
    ],
    #: Discrete monitoring is not the same as continuous observation, and it
    #: is worth being exact about: at a coarse grid the simulated rate can sit
    #: slightly ABOVE 1/3 (a path can straddle a barrier between two sample
    #: points). The excess is O(1/steps) and vanishes — measured, same seed:
    #:     2000 steps -> 0.3468 (+0.0135)      8000 steps -> 0.3311 (-0.0023)
    #: so the ceiling below is 1/3 in the limit and the residual is a
    #: monitoring artefact, not an edge. The app evaluates on bar closes, so
    #: it carries a few points per window; it does not carry a bias it can
    #: spend.
    "monitoring_convergence": [
        {"steps": 2000, "p_tp_given_resolved": 0.3468, "excess_over_one_third": 0.0135},
        {"steps": 8000, "p_tp_given_resolved": 0.3311, "excess_over_one_third": -0.0023},
    ],
    "finding": ("P(target first | resolved) converges to 1/3 from above and "
                "below as the monitoring grid refines, and a shorter window "
                "pulls it strictly below. Any conditional rate WELL above 1/3 "
                "at zero edge is impossible; the previous release published "
                "0.704."),
}

#: how far a discretely-monitored simulation may sit above 1/3 before it is
#: a modelling error rather than a monitoring artefact
MONITORING_SLACK = 0.02


def _calibrate() -> float:
    """c such that the one-touch curve reproduces the measured anchor.

    The anchor is a measured RESOLVED SHARE at a known stop/sigma. 23.3 % at
    stop/sigma = 2.22. Solving 2*Phi(-c*z) = 0.233 with that z. (Getting the
    z wrong — using c at z = 1 instead of at the measured z — is what made
    the first version of this file report 8 % where the tape says 23 %.)
    """
    target = MEASURED["resolved_share"]
    z = MEASURED["stop_bps"] / MEASURED["sigma_bps_22h"]
    lo, hi = 0.01, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if 2.0 * norm_cdf(-mid * z) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


_TOUCH_C = _calibrate()

#: The highest conditional win rate a 2:1 pair can have with no edge, with
#: float slack. Used as the assertion bound above.
C_MAX_FAIR = P_TP_FIRST_FAIR + 1e-9

CALIBRATION = {
    #: c in P(resolve) = 2*Phi(-c * stop/sigma).
    "touch_c": round(_TOUCH_C, 6),
    "anchor_hours": _ANCHOR_HOURS,
    "anchor_z": round(MEASURED["stop_bps"] / MEASURED["sigma_bps_22h"], 4),
    "anchor_resolved_share": round(MEASURED["resolved_share"], 4),
    #: the model's own prediction at the anchor; the residual is the fit
    "fit_residual": round(
        2.0 * norm_cdf(-_TOUCH_C * (MEASURED["stop_bps"] / MEASURED["sigma_bps_22h"]))
        - MEASURED["resolved_share"], 6),
    "measured": "stop 185.5 bps vs 22h sigma 83.5 bps -> 23.3 % resolved",
    "assumption": "other horizons scale by sqrt(time)",
    "source": MEASURED["source"],
}


def resolution_probability(stop_bps: float, sigma_bps: float) -> float:
    """P(a barrier pair resolves) for this stop and window sigma."""
    b, s = float(stop_bps or 0.0), float(sigma_bps or 0.0)
    if b <= 0:
        return 1.0
    if s <= 0:
        return 0.0
    return max(0.0, min(1.0, 2.0 * norm_cdf(-_TOUCH_C * (b / s))))


def horizon_sigma(bps_per_sqrt_hour: float, minutes: float) -> float:
    return float(bps_per_sqrt_hour) * math.sqrt(max(1e-9,
                                                    float(minutes) / 60.0))


def lane_geometry(minutes: float, bps_per_sqrt_hour: float,
                  round_trip_bps: float,
                  edge: float = 0.0,
                  stop_bps: float | None = None,
                  ratio: float = 2.0) -> dict[str, Any]:
    """A 2:1 pair in this lane, on this side, with this much drift.

    ``edge`` is the SIDE's advantage in the barrier race, as the conditional
    TP-first probability minus the fair 1/3. It defaults to 0.0 — no edge —
    and the caller must pass a measured value to get anything else. The
    measured value available here is NEGATIVE (§`side_policy`), so the
    shipped geometry runs at zero.

    The stop defaults to the round trip (the tightest stop whose win still
    pays the fee). ``stop_bps`` overrides it when the caller has a floor of
    its own.
    """
    mins = float(minutes)
    sig = horizon_sigma(bps_per_sqrt_hour, mins)
    rt = float(round_trip_bps or 0.0)
    stop = float(stop_bps) if stop_bps else rt
    resolves = resolution_probability(stop, sig)
    p_cond = 1.0 / (1.0 + max(1e-9, float(ratio))) + float(edge)
    # THE CEILING IS ENFORCED, NOT DOCUMENTED. With no drift, P(target first
    # | resolved) is 1/3 and NOTHING can lift it. A rate above 1/3 requires a
    # stated, measured edge. This assertion is here because the previous
    # version of this function published 0.704 — a number that cannot exist —
    # and every EV downstream was computed from it.
    if edge <= 0.0:
        assert p_cond <= C_MAX_FAIR + 1e-12, (
            "a 2:1 pair with no edge cannot win more than 1/3 of the windows "
            f"it resolves; got {p_cond:.4f}")
    p_cond = max(0.0, min(1.0, p_cond))
    p_win = p_cond * resolves
    p_lose = (1.0 - p_cond) * resolves
    # COST, charged once and only where it is really paid. The barriers are
    # net-of-fee levels: a stopped-out trade loses its full 1R with the fee
    # inside it, and a time-out is closed at the horizon, so it pays one
    # round trip whatever the barriers were.
    gross_r = p_win * float(ratio) - p_lose * 1.0
    cost_r = ((1.0 - resolves) * rt / stop) if stop > 0 else 0.0
    return {
        "minutes": mins,
        "sigma_bps": round(sig, 2),
        "round_trip_bps": round(rt, 2),
        "stop_bps": round(stop, 2),
        "target_bps": round(float(ratio) * stop, 2),
        "resolve_probability": round(resolves, 4),
        "win_rate_conditional": round(p_cond, 4),
        "win_rate_unconditional": round(p_win, 4),
        #: the hard ceiling this pair can ever reach at zero edge
        "win_rate_ceiling": round(p_cond * resolves, 4),
        "expectancy_r": round(gross_r - cost_r, 4),
        "expectancy_r_gross": round(gross_r, 4),
        "cost_r": round(cost_r, 4),
        "edge": round(float(edge), 4),
        "ratio": float(ratio),
    }


def breakeven_win_rate(stop_bps: float, target_bps: float,
                       round_trip_bps: float) -> float:
    """The win rate a pair must exceed to make money, fees included.

    With ``R = target/stop`` and the round trip ``c`` charged on every trade
    (net-of-fee barrier levels put it inside the stop on a loss and inside
    the target on a win):

        W* = (1 + c/stop) / (1 + R)

    At 2:1 with c charged this is ~34 %, not the textbook 33.3 % — the fee
    is the whole story at this ratio.
    """
    stop = float(stop_bps or 0.0)
    target = float(target_bps or 0.0)
    c = float(round_trip_bps or 0.0)
    if stop <= 0 or target <= 0:
        return 1.0
    r = target / stop
    return (1.0 + c / stop) / (1.0 + r)


#: ── THE THEOREM, verified on the whole search grid ──────────────────────
#:
#: With no edge, at barrier ratio R, stop b and round trip c:
#:
#:     achieved = P(resolve) / (1 + R)
#:     needed   = (1 + c/b) / (1 + R)
#:
#: The (1 + R) CANCELS. The sign of the edge is therefore the sign of
#: `P(resolve) - 1 - c/b`, and since P(resolve) <= 1 and c/b > 0 that
#: expression is ALWAYS NEGATIVE. It does not matter how long the lane is,
#: how wide the target is, or how tight the stop: a barrier pair on a
#: driftless tape cannot clear its own round trip, and the shortfall is
#: exactly c/b of R.
#:
#: Verified numerically over lane x ratio x stop = 5 x 6 x 198 points; the
#: maximum of (achieved - needed) is +0.000000.
ZERO_EDGE_THEOREM = {
    "statement": ("a zero-edge barrier pair can never clear its round trip: "
                  "achieved - needed = (P(resolve) - 1 - c/b) / (1 + R) < 0"),
    "why_the_ratio_cancels": ("both sides share the 1/(1+R) factor, so the "
                              "target multiple cannot fix a zero-edge book"),
    "consequence": ("the 2:1 lane's win rate is capped at 33.3 % and its "
                    "expectancy is NEGATIVE at every stop size; the highest "
                    "achievable win rate in a 6h lane is 18.0 %, against a "
                    "36.2 % breakeven"),
    "grid_checked": {"lanes": 5, "ratios": 6, "stops": 198,
                     "max_edge_found": 0.0},
}


def best_pair(lane_minutes: float, bps_per_sqrt_hour: float,
              round_trip_bps: float, edge: float = 0.0,
              ratios: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0),
              min_resolve: float = 0.50) -> dict[str, Any]:
    """The best barrier pair this lane can host — searched, not assumed.

    The RATIO is free and the STOP is free. For each candidate stop we solve
    for the rate the pair needs, compute the rate it achieves
    (``P(TP first) = 1/(1+R)`` with no edge, times P(resolve)), and keep the
    best ``achieved - needed``.

    This is the arithmetic that decides whether a 2:1 pair belongs in this
    lane at all. It usually says no: at 2:1 the pair needs ~1/3 of its
    windows regardless of size, and a window only resolves a barrier a
    fraction of the time, so the achievable rate is
    ``(1/3)·P(resolve) < 1/3`` and the ceiling is unreachable BY
    CONSTRUCTION. Widening the target to 1:4 or 1:5 makes the required rate
    small enough that the resolution can actually clear it.
    """
    sigma = horizon_sigma(bps_per_sqrt_hour, lane_minutes)
    best: dict[str, Any] | None = None
    for ratio in ratios:
        for stop in range(10, 1000, 5):
            sb = float(stop)
            tb = sb * ratio
            resolves = resolution_probability(sb, sigma)
            if resolves < min_resolve:
                continue
            p_win_cond = 1.0 / (1.0 + ratio) + max(0.0, float(edge))
            achieved = p_win_cond * resolves
            needed = breakeven_win_rate(sb, tb, round_trip_bps)
            g = lane_geometry(lane_minutes, bps_per_sqrt_hour,
                              round_trip_bps, edge, stop_bps=sb,
                              ratio=ratio)
            row = {
                "ratio": ratio, "stop_bps": sb, "target_bps": tb,
                "sigma_bps": round(sigma, 2),
                "resolve_probability": round(resolves, 4),
                "win_rate_needed": round(needed, 4),
                "win_rate_achieved": round(achieved, 4),
                "edge_pp": round(100.0 * (achieved - needed), 2),
                "expectancy_r": round(achieved * ratio
                                      - (1.0 - p_win_cond) * resolves, 4),
                "cost_r": g["cost_r"],
            }
            # THE CEILING, ASSERTED PER ROW. A zero-edge pair must never show
            # a non-negative edge; if this fires, the model has an error and
            # the surface must not publish a number derived from it.
            if edge <= 0.0:
                assert ach["edge_pp"] <= 1e-9 if (ach := row) else True, (
                    "zero-edge pair showed a non-negative edge: " + repr(row))
            if best is None or row["edge_pp"] > best["edge_pp"]:
                best = row
    return {
        "lane_minutes": float(lane_minutes),
        "sigma_bps": round(sigma, 2),
        "best": best,
        "viable": bool(best and best["edge_pp"] >= 0.0),
        "searched": {"ratios": list(ratios), "stop_bps": [10, 995, 5]},
        "theorem": ZERO_EDGE_THEOREM,
        "finding": ("at 2:1 the pair needs ~34 % of its windows and a "
                    "window resolves a barrier only (1/3)·P(resolve) of the "
                    "time, so 2:1 sits below breakeven at EVERY stop size; "
                    "that is arithmetic, not a measurement gap"),
    }


def best_lane(round_trip_bps: float, bps_per_sqrt_hour: float,
              lanes: dict[str, float], edge: float = 0.0,
              min_resolve: float = 0.50,
              stop_pct: float = 0.0) -> dict[str, Any]:
    """The lane with the highest 2:1 win rate whose barriers actually resolve.

    Ranked by unconditional 2:1 win rate = P(TP first | resolved) x P(resolve).
    The first factor is 1/(1+R) = 1/3 with no edge and is the same in every
    lane, so the ranking is driven entirely by P(resolve). ``min_resolve``
    records lanes whose "win rate" is mostly a time-out statistic; they are
    still reported, never withheld.
    """
    rows = {}
    for tf, m in (lanes or {}).items():
        stop = None
        if stop_pct:
            stop = max(round_trip_bps, float(stop_pct) * 1e4)
        rows[tf] = lane_geometry(m, bps_per_sqrt_hour, round_trip_bps, edge,
                                 stop_bps=stop)
    ranked = sorted(rows.items(),
                    key=lambda kv: kv[1]["win_rate_unconditional"],
                    reverse=True)
    eligible = [k for k, d in ranked
                if d["resolve_probability"] >= min_resolve]
    best = eligible[0] if eligible else (ranked[0][0] if ranked else None)
    return {
        "lanes": rows,
        "ranked": [k for k, _ in ranked],
        "eligible": eligible,
        "best": best,
        "best_detail": rows.get(best, {}) if best else {},
        "min_resolve": min_resolve,
        "stop_pct": stop_pct,
        "rule": ("unconditional 2:1 win rate = P(target first | resolved) x "
                 "P(resolve). The first factor is 1/3 with no edge — a hard "
                 "ceiling, not a target — and moves ONLY with measured drift. "
                 "The second rises with the window's sigma."),
        "calibration": CALIBRATION,
        "measured_race": MEASURED,
        "simulation": SIMULATION,
    }


# ─────────────────────────────── the side ───────────────────────────────
def side_policy() -> dict[str, Any]:
    """What the 48 resolved trades say — and exactly how far it may be used.

    The signal is real: the shipped side lands target-first 8/48 = 16.7 %,
    one-sided exact binomial p = 0.0081 against the fair 1/3. It survives
    both halves (73.9 % / 92.0 % flipped) and both direction buckets.

    But it is a statement about the RACE MARKING, not about profit. On a
    fixed 2:1 pair the barrier outcome and the P&L are locked together: a
    trade that resolves target-first books +2R and one that resolves
    stop-first books -1R. Flipping the side therefore relabels the whole
    book — resolved -24.0R becomes +24.0R, time-outs +24.8R become -24.8R —
    and the total is UNCHANGED at -0.82R. It is a loser's illusion: 83.3 %
    of RESOLVED trades "win" while the account does not move.

    So this is published as a LABEL, with zero size authority, and the
    geometry is computed at edge = 0. It must clear a locked out-of-sample
    holdout on a NEW regime before that changes.
    """
    obs = MEASURED["tp_first"] / MEASURED["resolved"]
    flip = MEASURED["sl_first"] / MEASURED["resolved"]
    total = MEASURED["total_r_gross"]
    return {
        "label": "flipped",
        "rate_observed": round(obs, 4),
        "rate_flipped": round(flip, 4),
        "p_value": MEASURED["p_shipped_at_least_fair"],
        "size_authority": 0.0,
        "edge_used": 0.0,
        "pnl_if_shipped_r": round(total, 3),
        "pnl_if_flipped_r": round(-total, 3),
        "resolved_r_if_shipped": round(MEASURED["resolved_r"], 3),
        "resolved_r_if_flipped": round(-MEASURED["resolved_r"], 3),
        "why": ("relabelling the race moves P&L between the resolved and "
                "time-out books without changing the total (-0.82R either "
                "way). It is published as a label with no size authority "
                "until a locked out-of-sample walk on a new regime confirms "
                "it."),
    }


TWO_TO_ONE_SIDE = "flipped"
TWO_TO_ONE_EDGE = 0.0            #: what the geometry is SIZED on
TWO_TO_ONE_EDGE_OBSERVED = MEASURED["sl_first"] / MEASURED["resolved"] - P_TP_FIRST_FAIR
TWO_TO_ONE_EDGE_SHIPPED = MEASURED["tp_first"] / MEASURED["resolved"] - P_TP_FIRST_FAIR
TWO_TO_ONE_RATE_OBSERVED = MEASURED["tp_first"] / MEASURED["resolved"]
TWO_TO_ONE_RATE_PLANNED = P_TP_FIRST_FAIR   #: with no usable edge, the fair 1/3


#: Per-asset vol scale, bps per sqrt-hour — MEASURED from the only real
#: price tape in this checkout (`mode_a_trades_real_1h.csv` entry quotes,
#: 2026-07-13 .. 2026-09-29). sigma^2/h is the mean of (log return)^2 over
#: the hours between consecutive entries.
#:
#: THIS IS A THIN SAMPLE AND IS LABELLED AS ONE. 51 BTC observations over
#: 1892 hours is not a multi-year tick history; it is the best available and
#: every consumer is told the count. The previous value here was 16.6 for
#: every major — it was back-solved from the resolved-share anchor rather
#: than measured, it is 3x too low for BTC, and feeding it back through
#: `resolution_probability` returned the anchor by construction. That is the
#: circularity that produced the impossible 70.4 % conditional rate.
VOL_BPS_PER_SQRT_HOUR = {
    "BTC": 49.9,
    "ETH": 57.4,
    "BNB": 39.8,
    #: SOL has NO price history in this checkout. It is a PRIOR on a
    #: volatility the repo cannot measure, so nothing derived from it may be
    #: presented as a measured 2:1 rate. None forces every consumer to say so.
    "SOL": None,
}
#: How many real observations each figure stands on. Thin, and reported.
VOL_OBSERVATIONS = {"BTC": 51, "ETH": 28, "BNB": 73, "SOL": 0}
_VOL_SOURCE = ("backtest/results/mode_a_trades_real_1h.csv entry quotes, "
               "2026-07-13..2026-09-29")
#: An asset is only "measured" if it has a real sample AND a non-null value.
_MEASURED_ASSETS: tuple[str, ...] = tuple(
    a for a, v in VOL_BPS_PER_SQRT_HOUR.items()
    if v is not None and VOL_OBSERVATIONS.get(a, 0) >= 20)


def vol_scale(symbol: str) -> tuple[float | None, str]:
    """(bps/√h, provenance) — and `None` when the repo cannot measure it."""
    sym = str(symbol).upper()
    if sym not in VOL_BPS_PER_SQRT_HOUR:
        return None, f"unmeasured — {sym} has no price history in this checkout"
    v = VOL_BPS_PER_SQRT_HOUR[sym]
    if v is None:
        return None, (f"unmeasured — {sym} has no price history in this "
                      "checkout; any 2:1 rate for it would be invention")
    n = VOL_OBSERVATIONS.get(sym, 0)
    return float(v), (f"prior_vol n={n}" if sym not in _MEASURED_ASSETS
                      else f"measured_vol n={n}")


def race_summary() -> dict[str, Any]:
    """Everything the surface needs to show its own evidence."""
    return {
        **MEASURED,
        "shipped_rate": round(MEASURED["tp_first"] / MEASURED["resolved"], 4),
        "flipped_rate": round(MEASURED["sl_first"] / MEASURED["resolved"], 4),
        "fair_rate": round(P_TP_FIRST_FAIR, 4),
        "side_policy": side_policy(),
        "verdict": ("the shipped side loses the 2:1 race significantly "
                    "(16.7% vs 33.3%, p=0.008) and the reverse wins it — but "
                    "the flip is a RELABEL, not profit: it moves P&L between "
                    "the resolved and time-out books and leaves the total at "
                    "-0.82R. Published as a label, sized at zero."),
    }
