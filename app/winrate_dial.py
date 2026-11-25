"""Win-rate dial — raise the win rate WITHOUT ever skipping a window.

The measurement this module is built on
--------------------------------------
``backtest/results/winrate_frontier.json`` sweeps the whole barrier
family (TP = k*SL, SL = m*scale) over every window of real tape, with
the round trip charged and the cost floor applied by widening BOTH
barriers. Two facts come out of it and they are the whole story:

1. **Win rate is a dial, not an edge.** Net win rate falls monotonically
   as k rises: ~85 % at k=0.05, ~50 % at k=1.0, ~30 % at k=3.0 on the
   same windows, same sides. Nothing about the market changed between
   those rows — only the ratio of the target to the stop.
2. **Money is not on the dial.** Mean net P&L per trade sits at roughly
   minus the round trip across the ENTIRE dial, on every asset and every
   horizon. A higher win rate is bought by making each win smaller than
   each loss (EV = WR*TP - (1-WR)*SL - cost), so dialling up is free of
   risk to the win *rate* and pure cost to the account.

So this module gives the honest version of the request: a calibrated
dial that delivers a requested win rate with **no abstention** (a side
is always returned, never 0, never None), and that always reports the
measured price of that win rate in basis points per trade alongside it.

Two guardrails are non-negotiable and both come from the live policy:

* ``MIN_REWARD_RISK`` (0.25) — a geometry whose target is a smaller
  fraction of its stop than this wins often and loses the account. The
  dial therefore refuses k < 0.25, which is why the 80-90 % rows in the
  artifact are reported as a *ceiling* and not as an offer.
* ``MIN_TP_OVER_RT`` (1.5) — the target must clear 1.5x the round trip.
  ``app.hold_policy._cost_aware`` applies the floor at lock time; the
  dial never proposes a geometry that floor would reject.

Usage
-----
    from app.winrate_dial import dial
    dial("BTC", "tf4", target_wr=0.55)
    # {'side': ..., 'k': ..., 'm': ..., 'expected_wr': 0.505, ...,
    #  'achievable': True, 'price_bps': -25.9, 'ceiling_wr': 0.505}

    # the raw frontier for a cell
    from app.winrate_dial import frontier
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

import numpy as np

from .hold_policy import (MIN_REWARD_RISK, MIN_TP_OVER_RT, _cost_aware,
                          round_trip_bps)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FRONTIER_PATH = os.path.join(_ROOT, "backtest", "results",
                                     "winrate_frontier.json")
DEFAULT_COST_PATH = os.path.join(_ROOT, "backtest", "results",
                                 "winrate_cost_sensitivity.json")

#: horizons whose barrier scale is big next to the round trip, so the
#: cost floor does not swallow the target. Cells outside this set can
#: still be dialled — they just cannot reach a high win rate (the
#: artifact has the exact number, e.g. BTC tf1 ceiling 10.7 %).
_DAY_CELLS = ("1d", "2d", "3d", "5d")

_lock = threading.RLock()
_DOC: dict[str, Any] = {}
_COST: dict[str, Any] = {}
_PATHS: tuple[str, str] = ("", "")


def reload(frontier_path: str | None = None,
           cost_path: str | None = None) -> dict[str, Any]:
    """(Re)load the frontier + cost-curve artifacts. Missing → empty."""
    global _DOC, _COST, _PATHS
    fp = frontier_path or _PATHS[0] or DEFAULT_FRONTIER_PATH
    cp = cost_path or _PATHS[1] or DEFAULT_COST_PATH
    with _lock:
        def _read(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return {}
        _DOC = _read(fp) or {}
        _COST = _read(cp) or {}
        _PATHS = (fp, cp)
        return _DOC


def _doc() -> dict[str, Any]:
    if not _DOC:
        reload()
    return _DOC


def known_cells() -> list[str]:
    """Cells with a measured frontier, e.g. ``['BTC|tf4', 'ETH|2d', ...]``."""
    return sorted((_doc().get("cells") or {}).keys())


def frontier(symbol: str, tf: str) -> dict[str, Any] | None:
    """The measured frontier for one cell (``None`` when unmeasured)."""
    return (_doc().get("cells") or {}).get(f"{symbol}|{tf}")


def _rows(cell: dict[str, Any], side: str | None) -> list[dict[str, Any]]:
    sides = cell.get("sides") or {}
    names = [side] if side else list(sides)
    out = []
    for name in names:
        for r in (sides.get(name) or {}).get("frontier") or []:
            out.append({**r, "side": name})
    return out


def addressable(symbol: str, tf: str,
                side: str | None = None) -> list[dict[str, Any]]:
    """Rows the live policy would actually lock: reward:risk >= 0.25 and a
    target that clears the cost floor (the artifact applied it already)."""
    cell = frontier(symbol, tf)
    if not cell:
        return []
    rt = float(cell.get("rt_bps") or round_trip_bps(symbol))
    keep = []
    for r in _rows(cell, side):
        k, m = float(r["k"]), float(r["m"])
        if k < MIN_REWARD_RISK:            # lottery shape — never offered
            continue
        if r.get("ev_bps") is None or not r.get("n"):
            continue
        keep.append({**r, "reward_risk": round(k, 4),
                     "rt_bps": round(rt, 2)})
    return keep


def ceiling(symbol: str, tf: str, side: str | None = None) -> dict[str, Any] | None:
    """Highest *lockable* measured net win rate for the cell."""
    rows = addressable(symbol, tf, side)
    if not rows:
        return None
    best = max(rows, key=lambda r: (r["wr"], r["ev_bps"]))
    return {"wr": best["wr"], "ci95": best["ci95"], "n": best["n"],
            "k": best["k"], "m": best["m"], "side": best["side"],
            "ev_bps": best["ev_bps"], "pf": best["pf"],
            "rt_bps": best["rt_bps"]}


def dial(symbol: str, tf: str, target_wr: float | None = None,
         side_rule: str | None = None,
         direction: int | None = None) -> dict[str, Any]:
    """Pick a geometry that delivers ``target_wr`` — never abstain.

    ``side_rule`` filters the measured side families (``mom44h``,
    ``revW``, ...); ``direction`` is the live +/-1 the caller is about to
    take and is echoed back untouched. The dial never decides *whether*
    to trade and never turns a direction into 0 — that is the whole
    point of it. When the target is above what any lockable geometry
    reaches, the best available geometry is still returned, with
    ``achievable=False`` and the ceiling named.
    """
    cell = frontier(symbol, tf)
    rows = addressable(symbol, tf, side_rule)
    ceil = ceiling(symbol, tf, side_rule)
    if not cell or not rows:
        return {"achievable": False, "measured": False, "symbol": symbol,
                "tf": tf, "target_wr": target_wr, "direction": direction,
                "side_rule": side_rule, "k": None, "m": None,
                "expected_wr": None, "price_bps": None,
                "ceiling_wr": None,
                "reason": "no measured frontier for this cell — the caller "
                          "falls back to the certified hold geometry",
                "honest_note": _HONEST}
    if target_wr is not None and any(r["wr"] >= float(target_wr) for r in rows):
        # widest target that still clears the ask = the cheapest win rate
        best = max([r for r in rows if r["wr"] >= float(target_wr)],
                   key=lambda r: (r["k"], r["ev_bps"]))
    else:
        best = max(rows, key=lambda r: (r["wr"], r["ev_bps"]))
    achievable = bool(target_wr is None or best["wr"] >= float(target_wr))
    ref = max(rows, key=lambda r: r["k"])          # widest geometry in cell
    return {
        "achievable": achievable, "measured": True,
        "symbol": symbol, "tf": tf,
        "direction": direction, "side_rule": best.get("side"),
        "target_wr": target_wr,
        "k": best["k"], "m": best["m"],
        "reward_risk": round(float(best["k"]), 4),
        "expected_wr": best["wr"], "ci95": best["ci95"], "n": best["n"],
        "price_bps": best["ev_bps"],
        "price_vs_widest_geometry_bps": round(float(best["ev_bps"]) -
                                              float(ref["ev_bps"]), 2),
        "pf": best["pf"], "rt_bps": best["rt_bps"],
        "ceiling_wr": (ceil or {}).get("wr"),
        "cost_payable": True,
        "min_reward_risk": MIN_REWARD_RISK,
        "min_tp_over_rt": MIN_TP_OVER_RT,
        "reason": ("measured win rate at a lockable reward:risk"
                   if achievable else
                   f"target {target_wr:.1%} above the lockable ceiling "
                   f"{(ceil or {}).get('wr', 0):.1%} — best available "
                   "returned, the window still trades"),
        "honest_note": _HONEST,
    }


_HONEST = ("win rate is a geometry choice (TP:SL), measured not promised: "
           "the price in bps per trade is quoted with every dialled point, "
           "and net EV sits near minus the round trip across the whole dial")


def build_row_bars_dial(symbol: str, tf: str, entry: float, side: int,
                        scale_px: float, target_wr: float | None = None,
                        side_rule: str | None = None
                        ) -> tuple[float, float, dict[str, Any]] | None:
    """Live TP/SL from the dial, cost floor applied exactly like hold_policy.

    A lock is produced for every window: when the cell has no measured
    frontier the certified hold geometry is used instead of returning
    nothing. ``None`` only for degenerate inputs (non-positive price or
    scale, zero side) — never because a win-rate target was unreachable.
    """
    if entry <= 0 or scale_px <= 0 or not side:
        return None
    plan = dial(symbol, tf, target_wr=target_wr, side_rule=side_rule,
                direction=int(side))
    k, m = plan.get("k"), plan.get("m")
    source = "dial"
    if not k or not m:
        try:                       # certified fallback — still a lock
            from .hold_policy import row_geometry
            g = row_geometry(symbol, tf)
        except Exception:          # pragma: no cover
            g = None
        if not g:
            return None
        k, m = g
        source = "certified_fallback"
    tp, sl, meta = _cost_aware(symbol, float(entry), int(side),
                               float(scale_px), float(k), float(m))
    meta = {**meta, "geometry_source": source,
            "dial": {kk: plan.get(kk) for kk in
                     ("k", "m", "expected_wr", "ci95", "n", "price_bps",
                      "achievable", "ceiling_wr", "measured", "reason")}}
    return tp, sl, meta


def day_cell_plan(symbol: str, horizon: str, direction: int | None = None,
                  target_wr: float | None = None) -> dict[str, Any]:
    """Dial a DAY-scale cell (``1d``/``2d``/``3d``) — measured, not wired.

    These are the only cells where a lockable geometry reaches a high
    net win rate (78-84 % on the 2024-2026 tape) because the window move
    is large next to the round trip. The live cycle has no day-scale
    lock today (``hold_policy`` row locks are 22-247 min and the era
    ride is capped at 7 days), so this returns the plan with
    ``wired=False`` — the number is real, the plumbing does not exist
    yet, and shipping it needs its own walk-forward: the 4-block test in
    ``backtest/results/winrate_noabstain_wf.json`` shows the win rate
    survived out of sample but net EV did not.
    """
    plan = dial(symbol, horizon, target_wr=target_wr, direction=direction)
    if plan.get("measured"):
        plan["wired"] = False
        plan["wired_reason"] = ("day-scale lock has no live path yet; "
                               "row locks are 22-247 min, era ride <= 7d")
    return plan


def cost_curve(symbol: str, tf: str) -> list[dict[str, Any]]:
    """What a cheaper round trip buys — the one lever that moves the win
    rate AND the P&L in the same direction, without skipping a window.

    Rows are ``{rt_bps, side, k, m, wr, ev_bps}`` from
    ``winrate_cost_sensitivity.json`` (same windows, same geometry,
    cheaper fill). Empty when the cell has not been measured.
    """
    if not _COST:
        reload()
    cell = (_COST or {}).get(f"{symbol}|{tf}") or {}
    out = []
    for key, by_rt in (cell.get("rows") or {}).items():
        side, k, m = key.split("|")
        for rt_name, v in by_rt.items():
            out.append({"rt_bps": float(rt_name.replace("rt", "")),
                        "side": side, "k": float(k.replace("k", "")),
                        "m": float(m.replace("m", "")),
                        "wr": v["wr"], "ev_bps": v["ev"],
                        "n": cell.get("n")})
    out.sort(key=lambda r: (r["side"], r["k"], r["rt_bps"]))
    return out


def table(symbol: str, tf: str) -> dict[str, Any]:
    """Everything the UI/docs need for one cell, in one dict."""
    cell = frontier(symbol, tf) or {}
    return {"cell": f"{symbol}|{tf}", "rt_bps": cell.get("rt_bps"),
            "n": cell.get("n"), "source": cell.get("source"),
            "ceiling": ceiling(symbol, tf),
            "addressable": addressable(symbol, tf),
            "cost_curve": cost_curve(symbol, tf),
            "note": _HONEST}
