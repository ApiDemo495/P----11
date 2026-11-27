"""Certified hold policy — the live wiring of the BOTH-targets system.

Loads ``backtest/results/hold_wr.json`` (produced by ``backtest.hold``)
and exposes, per (asset, timeframe) cell:

  * the exact certified side rule (``mom525600``/``mom259200``/…, ``down``),
  * the decoupled TP/SL geometries — tight ROW geometry for the
    every-window first-touch grade and wide ERA geometry for the
    held position,
  * real per-side friction — §18.1 full Binance stack (commission 10
    + slippage + half-Roll spread per side → 28.9 / 29.8 / 31.0 /
    37.4 bps round trip; the artifact carries the exact numbers),
  * honest tier labels: ``HOLD-OUT VERIFIED`` (count from the shipped
    artifact),
    ``HOLDOUT ROWS · ERA n<3`` (BTC 247min, rows verified but a single
    holdout era), ``IN-SAMPLE SHADOW`` (BNB, single-year) / its
    ``HOLDOUT MISS`` variant (PAXG 247min), ``UNGRADEABLE / WARMUP``
    (PAXG tf1–tf3 — 1 h bars cannot resolve 22/44/130-min windows),
  * measured win rate + Wilson interval (confidence source of truth).

Side semantics are byte-identical to ``backtest.hold.build_sides`` for
every non-logistic candidate (the certified selection never picked an
``lr_*`` composite) — ``tests/test_hold_live.py`` enforces ZERO parity
mismatches against ``backtest.hold`` on shared bar arrays, including the
era simulation and row settlement rules.

Every window is analysed (full coverage, no abstention): partial
history falls back to ``up`` exactly like the benchmark's documented
warmup rule, reported through ``meta["warmup"]`` so the UI can say so.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from bisect import bisect_left
from typing import Any

import numpy as np

from .config import TIMEFRAMES

DEFAULT_HOLD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backtest", "results", "hold_wr.json")

TF_BY_MIN = {v["minutes"]: k for k, v in TIMEFRAMES.items()}
TF_MINUTES = {k: int(v["minutes"]) for k, v in TIMEFRAMES.items()}
# sides certified by the selection (hold.py MOM_GRID ∪ LONG_MOM_MIN etc.)
_MOM_MINUTES = tuple(range(15, 45 * 60 + 1, 15)) + (
    7200, 10080, 14400, 43200, 129600, 259200, 525600)

_lock = threading.RLock()
_DOC: dict[str, dict] = {}
_PATH = ""


def _load_doc(path: str | None = None) -> dict[str, dict]:
    global _DOC, _PATH
    with _lock:
        p = path or os.environ.get("HOLD_WR_PATH") or DEFAULT_HOLD_PATH
        if _DOC and p == _PATH:
            return _DOC
        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            # record absent/unreadable → the legacy mandate fallback
            # owns the cells (tests pin this path too)
            doc = {}
        _DOC = {k: v for k, v in doc.items() if isinstance(v, dict)}
        _PATH = p
        return _DOC


def reload(path: str | None = None) -> dict[str, dict]:
    """Drop the cache and re-read (tests use this with HOLD_WR_PATH)."""
    global _DOC, _PATH
    with _lock:
        _DOC, _PATH = {}, ""
    return _load_doc(path)


def cell(symbol: str, tf: str) -> dict | None:
    """Raw hold_wr cell record for (asset, tf), or None if unknown."""
    if tf not in TF_MINUTES:
        return None
    doc = _load_doc()
    return doc.get(f"{symbol}|{TF_MINUTES[tf]}")


def known_cells() -> list[tuple[str, str]]:
    return [(s, t) for s in ("BTC", "ETH", "BNB", "PAXG")
            for t in ("tf1", "tf2", "tf3", "tf4")]


def tier_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol, tf in known_cells():
        t = tier(symbol, tf)
        out[t] = out.get(t, 0) + 1
    return out


# ─────────────────────────────── tiers ───────────────────────────────
TIER_HOLDOUT = "HOLD-OUT VERIFIED"
TIER_ROWS_ONLY = "HOLDOUT ROWS · ERA n<3"
TIER_INSAMPLE = "IN-SAMPLE SHADOW"
TIER_MISS = "IN-SAMPLE SHADOW · HOLDOUT MISS"
TIER_WARMUP = "UNGRADEABLE / WARMUP"
# A cell whose available geometry cannot pay the round trip is NOT
# tradeable, however good its gross hit rate looks.
TIER_UNECONOMIC = "UNECONOMIC · BELOW ROUND TRIP"
# Rows held up out-of-fold, but the artifact the cell came from has
# NOT passed its own ship gates (gate (ii): no REAL post-cost win rate
# recorded, so it cannot beat the v8 reference).  Honest intermediate:
# the measurement is real, the ship claim is not earned.
TIER_SHIPGATE = "ROWS VERIFIED · SHIP GATE FAIL"
# The certified geometry pays less than it risks (every shipped cell was
# selected at row_k = 0.15, i.e. 0.15:1).  The measurement is real, but it
# is a measurement of a trade the app may not place: a target an eighth of
# the stop needs an 87 % first-touch rate just to break even, and the live
# lock is built on a repaired 1:1 geometry whose hit rate has NOT been
# measured.  Neither "verified" nor "failed" is true, so it gets its own
# label rather than quietly borrowing either one.
TIER_SUB1 = "GEOMETRY BELOW 1:1 · NOT TRADEABLE"

# ────────────── cost-aware "is a win actually a win?" ────────────────
# The certified artifact's headline number is the GROSS row win rate
# (first touch of a k=0.15 x SL take-profit wins ~73-85 % of windows).
# That number says nothing about money: the target it measures sits
# INSIDE the Binance round trip, so the app can book an 85 % "win rate"
# while every winning window loses money.  THE rule enforced here:
#
#     a win must pay more than the cost of getting in and out.
#
# MIN_TP_OVER_RT is that floor (target distance >= 1.5x the full round
# trip, i.e. a win nets at least a third of the gross move).  It is a
# cost floor, not a fitted parameter: it holds for any tape, and it is
# what makes a reported win rate comparable to money.
MIN_TP_OVER_RT = 1.5


def breakeven_tp_bps(win_rate: float, rt_bps: float,
                     reward_risk: float) -> float | None:
    """The target a cell must ACTUALLY reach to net zero, given its own
    hit rate and payoff.  ``None`` when the geometry cannot pay at all.

    Bug fixed 2026-11-26: the dashboard printed "net needs target >= 43bp"
    for an ETH cell at 72.9 % gross and 1.10 reward:risk with a 28.98 bp
    round trip. 43 bp is just MIN_TP_OVER_RT (1.5x) x RT — a fee-multiple
    rule, not a break-even calculation. It answers "does a win keep a
    third of its gross move", which is a margin convention; it does not
    ask what THIS cell's hit rate and payoff imply. A payoff of 1.10 needs
    a wider target than a payoff of 1.0 does, and that cell's own 72.9 %
    is a gross first-touch rate, not a cash rate.

    With target T, stop S = T/rr, the net per trade is
        p*T - (1-p)*S - RT   (p = the CASH win rate)
    so the break-even target solves to
        T* = RT / (p - (1-p)/rr)
    At the panel's own numbers that is 60.0 bp, not 43. Following the 43 bp
    floor still loses 8.2 bp per trade. A floor that is 28 % too low is a
    floor that quietly funds the exchange.

    ``win_rate`` here must be the probed CASH win rate for the widened
    target (widening a target lowers the hit rate), so callers that only
    hold the gross first-touch rate should pass an explicit ``None`` rather
    than let this function pretend.
    """
    try:
        p = float(win_rate)
        rr = float(reward_risk)
        rt = float(rt_bps)
    except (TypeError, ValueError):
        return None
    if not (0.0 < p < 1.0) or rr <= 0.0 or rt <= 0.0:
        return None
    edge = p - (1.0 - p) / rr
    if edge <= 1e-9:
        return None                 # no target can pay at this hit rate
    return round(rt / edge, 3)

#: A geometry whose target is a smaller fraction of its stop than this is a
#: lottery ticket: it wins often and loses the account. Measured on 3 years
#: of tape, every >=66 % win-rate geometry had reward:risk below 0.2.
MIN_REWARD_RISK = 0.25


def round_trip_bps(symbol: str) -> float:
    """Full-stack round trip (both sides) in bps for one asset."""
    try:
        from backtest.data import FRICTION_BPS
        return 2.0 * float(FRICTION_BPS.get(symbol, 12.0))
    except Exception:                                   # pragma: no cover
        return 24.0


def tp_pays_rt(symbol: str, tp_bps: float | None) -> bool | None:
    """Does a target ``tp_bps`` away clear the round trip?

    ``None`` when the distance is unknown — never guessed true.
    """
    if tp_bps is None:
        return None
    rt = round_trip_bps(symbol)
    if rt <= 0:
        return None
    return bool(float(tp_bps) >= MIN_TP_OVER_RT * rt)


_SHIP_CACHE: dict[str, Any] | None = None


def artifact_ship_verdict() -> dict[str, Any]:
    """Did the shipped artifact pass its own pre-registered ship gates?

    ``backtest/gates18.py`` is the acceptance bar these tiers were
    defined against (gate (i) row+era viability, gate (ii) mean REAL
    win rate beating the v8 reference, gate (iii) >= 4 holdout-verified
    cells).  The tiers below used to be granted from the per-cell
    ``targets`` flags alone, which let a FAILING artifact keep handing
    out ``HOLD-OUT VERIFIED`` badges.  Cached: the artifact is a file
    on disk, read once per process.
    """
    global _SHIP_CACHE
    with _lock:
        if _SHIP_CACHE is not None:
            return _SHIP_CACHE
        doc = _load_doc()
        out: dict[str, Any] = {"checked": False}
        if doc:
            # gate (ii): mean recorded REAL win rate vs the v8 reference
            nets = [float((c.get("selected") or {}).get("full", {})
                          .get("row_wr_net"))
                    for c in doc.values()
                    if isinstance(c, dict)
                    and (c.get("selected") or {}).get("full", {})
                    .get("row_wr_net") is not None]
            ref = None
            try:
                ref_path = os.path.join(
                    os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))),
                    "backtest", "results", "v8_net_reference.json")
                ref = float(json.load(open(ref_path))["mean_row_wr_net"])
            except Exception:
                ref = None
            mean_net = (sum(nets) / len(nets)) if nets else None
            out = {
                "checked": True,
                "mean_row_wr_net": (round(mean_net, 6)
                                    if mean_net is not None else None),
                "v8_reference": ref,
                "gate_ii_pass": (bool(nets) and ref is not None
                                 and mean_net > ref),
                "note": ("artifact does not carry a REAL (post-cost) win "
                         "rate for any cell, so gate (ii) fails and no "
                         "higher real win rate is certified by this file"
                         if not nets else
                         "artifact carries REAL win rates"),
            }
        _SHIP_CACHE = out
        return out


def tier(symbol: str, tf: str) -> str:
    c = cell(symbol, tf)
    if c is None:
        return TIER_WARMUP
    if c.get("status") != "ok":
        return TIER_WARMUP
    t = c.get("targets") or {}
    # A geometry that pays less than it risks can never be traded, so no
    # row or ship gate can make the cell tradeable.  Checked FIRST, before
    # any gate that could otherwise return a green-ish badge.
    _g = sub_1_to_1_geometry(symbol, tf)
    if _g is not None and not _g["feasible"]:
        return TIER_SUB1
    if c.get("single_year"):
        return TIER_INSAMPLE
    # A cell may only wear HOLD-OUT VERIFIED while the artifact it came
    # from still passes its own ship gates.  On a failing artifact the
    # rows are honest measurements but the ship claim is not earned.
    ship = artifact_ship_verdict()
    rows_ok = (t.get("holdout_row_wr_ge_66")
               and t.get("holdout_row_wilson_lo_ge_66")
               and t.get("holdout_era_ev_net_gt_0"))
    if rows_ok and (not ship.get("checked") or ship.get("gate_ii_pass")):
        return TIER_HOLDOUT
    if rows_ok and not ship.get("gate_ii_pass"):
        return TIER_SHIPGATE
    if t.get("holdout_row_wr_ge_66") and t.get("full_era_ev_net_gt_0"):
        # rows verified out-of-fold but the holdout has < 3 eras to
        # judge system EV (BTC 247min: n=1) — say exactly that
        if t.get("holdout_era_ev_net_gt_0") is None and \
                t.get("holdout_era_n") is not None:
            return TIER_ROWS_ONLY
    return TIER_MISS


def tier_reason(symbol: str, tf: str) -> str:
    """WHY this cell wears the tier it wears — in one readable sentence.

    The user's report: "why hold policy have those yellow boxes of gate fail,
    warmup and other any issues behind them". The badge named the verdict and
    never the evidence, so a failed gate looked like a defect in the app
    rather than a measurement result. Every tier now carries the number that
    produced it, taken from the same artifact the badge is derived from.
    """
    c = cell(symbol, tf)
    t = tier(symbol, tf)
    if c is None:
        return "this cell is not in the certified hold-policy artifact"
    if c.get("status") != "ok":
        why = str(c.get("why") or "").strip()
        return ("no configuration survives the cost floor on this cell's "
                "history, so no win rate can be certified"
                + (f" — {why}" if why else "")
                + " · live tape keeps computing, nothing is certified")
    sel = c.get("selected") or {}
    full = sel.get("full") or {}
    hold = sel.get("holdout") or {}
    tg = c.get("targets") or {}
    if t == TIER_SUB1:
        g = sub_1_to_1_geometry(symbol, tf) or {}
        _wr = g.get("certified_win_rate")
        return (f"the certified geometry pays {g.get('rr')}x what it risks "
                f"(target = {g.get('k'):g} x stop) — a target an eighth of "
                f"the stop neither covers its own stop nor the "
                f"{round(float(c.get('round_trip_bps') or round_trip_bps(symbol)), 1)} bps round "
                f"trip. It needs a "
                f"{g.get('win_rate_needed')}% first-touch rate to break even; "
                "the certified rate is "
                + (f"{round(float(_wr) * 100, 1)}%. " if _wr is not None
                   else "not recorded. ")
                + "Such a pair is not tradeable at ANY win rate and the app "
                "enforces a 1:1 floor on every live lock, so this win rate "
                "describes a geometry you are not allowed to place")
    if t == TIER_HOLDOUT:
        return ("rows AND eras cleared out-of-fold, and the artifact passes "
                "its own ship gates")
    if t == TIER_SHIPGATE:
        ship = artifact_ship_verdict()
        _wil = (hold.get("row_wilson") or [None])[0]
        return ("the rows held up out-of-fold (hold-out row WR "
                f"{(hold.get('row_wr') or 0) * 100:.1f} %"
                + (f", Wilson lo {_wil * 100:.1f} %" if _wil is not None else "")
                + f", holdout era net EV {(hold.get('era_ev_net') or 0) * 1e4:+.1f} bp"
                "), but the artifact that produced them fails its own pre-"
                "registered ship gate (ii): it carries no REAL post-cost win "
                "rate for any cell, so its mean net WR cannot beat the v8 "
                f"reference {(ship.get('v8_reference') or 0) * 100:.2f} %. "
                "The measurement is real; the ship claim is not earned")
    if t == TIER_ROWS_ONLY:
        return ("rows cleared out-of-fold but the hold-out has fewer than 3 "
                f"eras (n={tg.get('holdout_era_n')}), so system EV cannot be "
                "judged")
    if t == TIER_INSAMPLE:
        return ("this cell's artifact is a SINGLE-YEAR fit: every number is "
                "in-sample, so it is shown as a shadow, never as a "
                "certified edge")
    # TIER_MISS and anything else
    misses = []
    _wil = (hold.get("row_wilson") or [None])[0]
    if not tg.get("holdout_row_wr_ge_66"):
        misses.append(f"hold-out row WR {(hold.get('row_wr') or 0) * 100:.1f} % "
                      "is below the 66 % bar")
    elif not tg.get("holdout_row_wilson_lo_ge_66"):
        misses.append("its Wilson lower bound "
                      + (f"({_wil * 100:.1f} %)" if _wil is not None else "")
                      + " is below the 66 % bar")
    if tg.get("holdout_era_ev_net_gt_0") is False:
        misses.append("hold-out era net EV is "
                      f"{(hold.get('era_ev_net') or 0) * 1e4:+.1f} bp after "
                      "friction (needs > 0)")
    if not tg.get("full_era_ev_net_gt_0"):
        misses.append("the full-era net EV does not clear zero")
    return ("missing the hold-out bar — " + "; ".join(misses)
            if misses else "in-sample only — the hold-out never confirmed it")


def is_hold_cell(symbol: str, tf: str) -> bool:
    """True when this cell is governed by the certified hold policy
    (gradeable cells) — warmup cells keep the policy's badges but the
    benchmark never measured a side on them."""
    c = cell(symbol, tf)
    return bool(c and c.get("status") == "ok")

# [removed 2026-11-23] is_warmup_cell — dead code: zero references anywhere


def managed(symbol: str, tf: str) -> bool:
    """True when hold policy owns the cell at all (all 16 cells)."""
    return cell(symbol, tf) is not None


# ───────────────────────── summary for UI / engine ───────────────────
def summary(symbol: str, tf: str) -> dict[str, Any]:
    """Everything the live cell panel needs, honestly labelled."""
    c = cell(symbol, tf)
    t = tier(symbol, tf)
    base: dict[str, Any] = {
        "symbol": symbol, "tf": tf, "minutes": TF_MINUTES.get(tf),
        "tier": t, "managed": c is not None,
        "certified": is_hold_cell(symbol, tf),
        # WHY the badge says what it says — carried on the payload so the
        # card can explain itself instead of showing a bare coloured label.
        "tier_reason": tier_reason(symbol, tf),
    }
    if c is None:
        base.update(status="unknown", side=None, win_rate=None,
                    confidence=None, why="cell not in the hold policy")
        return base
    if c.get("status") != "ok":
        base.update(status="ungradeable", side=None, win_rate=None,
                    confidence=None, why=c.get("why"),
                    note=("UNGRADEABLE on 1 h history — live WARMUP until "
                          "enough sub-hour tape exists; no certified win "
                          "rate is claimed for this cell"))
        return base
    sel = c.get("selected") or {}
    full = sel.get("full") or {}
    hold = sel.get("holdout") or {}
    wil = full.get("row_wilson") or [None, None]
    base.update(
        status="ok",
        side=sel.get("side"),
        row_k=sel.get("row_k"), row_m=sel.get("row_m"),
        era_k=sel.get("era_k"), era_m=sel.get("era_m"),
        ride=bool(float(sel.get("era_m") or 0) >= 999.0),
        win_rate=full.get("row_wr"),
        # reward:risk of the certified row geometry (target / stop). Needed by
        # breakeven_tp_bps below — without it the panel can only show the
        # fee-multiple floor, which is wrong whenever rr != 1.0.
        reward_risk=(round(float(sel["row_k"]), 4)
                     if sel.get("row_k") is not None else None),
        win_rate_label=("gross first-touch hit rate (does NOT include "
                        "costs)"),
        win_rate_wilson=wil,
        confidence=wil[0],                    # Wilson lower bound
        # §18 REAL win rate: share of windows still winners after the
        # full Binance round trip (the honest "win rate on real
        # conditions"); informational — the trade itself is the era.
        win_rate_net=full.get("row_wr_net"),
        win_rate_net_wilson=full.get("row_wr_net_wilson"),
        holdout_win_rate=(hold.get("row_wr")
                          if hold.get("row_n") else None),
        holdout_win_rate_net=hold.get("row_wr_net"),
        n_windows=c.get("n_windows"),
        era_ev_net=full.get("era_ev_net"),
        era_ev_holdout=hold.get("era_ev_net"),
        era_ev_total=full.get("era_ev_total_net"),
        era_wr=full.get("era_wr"), era_n=full.get("era_n"),
        friction_bps_per_side=c.get("friction_bps_per_side"),
        round_trip_bps=2.0 * float(c.get("friction_bps_per_side") or 0.0),
        flip_rate=sel.get("flip_rate"),
        single_year=bool(c.get("single_year")),
        selection=sel.get("selection"),
        targets=c.get("targets"),
    )
    # COST VERDICT — the honest pairing for the number above.  The
    # headline win_rate is a gross first-touch hit rate; if the target
    # the certified geometry aims at sits inside the round trip, that
    # hit rate cannot be booked as profit.
    #
    # The round trip used here is the AUTHORITATIVE full-stack figure
    # (backtest.data.FRICTION_BPS, §18.1: commission 10 + slippage +
    # half-Roll spread per side).  Some shipped artifacts still carry
    # the older 12 bps/side constant; taking the max means the cost
    # verdict can only be conservative, never flattering.
    rt = max(2.0 * float(c.get("friction_bps_per_side") or 0.0),
             round_trip_bps(symbol))
    base["round_trip_bps"] = round(rt, 3)
    # the fee-multiple floor AND the honest break-even for this cell's own
    # gross rate and payoff. The panel shows both; they differ whenever
    # reward:risk is not 1.0, and the break-even is the one that pays.
    _fee_floor = round(MIN_TP_OVER_RT * rt, 3)
    base["min_tp_bps"] = _fee_floor
    _wr = base.get("win_rate")
    _rr = base.get("reward_risk")
    _be = (breakeven_tp_bps(_wr, rt, _rr)
           if _wr is not None and _rr is not None else None)
    base["breakeven_tp_bps"] = _be
    base["win_rate_is_money"] = None
    # ── the reward:risk the CERTIFIED geometry implies, and therefore
    # whether the published win rate describes a trade the app may place.
    _geo = sub_1_to_1_geometry(symbol, tf, base.get("row_k"), base.get("row_m"))
    if _geo is not None:
        base["geometry_rr"] = _geo["rr"]
        base["geometry_feasible"] = _geo["feasible"]
        base["geometry_win_rate_needed"] = _geo["win_rate_needed"]
        base["geometry_note"] = (
            "certified geometry pays "
            f"{_geo['rr']}x what it risks"
            + ("" if _geo["feasible"] else
               f" — BELOW the {RR_LIVE_MIN:g}:1 floor, so live locks are "
               f"built on a repaired {RR_LIVE_MIN:g}:1 geometry and this "
               "win rate does NOT describe the trade that is placed"))
    base["min_tp_note"] = (
        f"{MIN_TP_OVER_RT}x round trip (fee-multiple convention)"
        + (f"; this cell's own break-even is {_be} bp at "
           f"{round(float(_wr) * 100, 1)}% and rr {_rr}"
           if _be is not None else
           "; no break-even exists at this hit rate and payoff"))
    # The floor that is actually enforced must be the break-even, never the
    # fee multiple: 43 bp does not pay a 1.10 payoff. When NO break-even
    # exists the honest answer is None ("no target pays"), not a fallback to
    # the fee floor — every shipped cell carries row_k = 0.15, i.e. a target
    # one-seventh of its stop, which needs an 87 % hit rate to break even.
    # Printing 43 bp there is a floor that funds the exchange.
    _fee_mult = ((_wr + (1.0 - float(_wr)) / float(_rr))
                 if (_wr is not None and _rr) else None)
    base["tp_multiple_of_rt_needed"] = (round(_fee_mult, 3)
                                        if _fee_mult is not None else None)
    base["enforced_tp_bps"] = _be
    base["cannot_pay_at_any_target"] = bool(
        _wr is not None and _rr is not None and _be is None)
    #: NO target can satisfy the 1.5x fee rule, because the ROUND TRIP ALONE
    #: moves the stop by more than 1.5x the target. That is a stronger and
    #: much rarer failure than the one below, and it must not be conflated
    #: with it (it fired on all 12 graded cells before this split).
    base["cost_infeasible"] = bool(_be is None and rt > 0 and _rr
                                   and _rr / (1.0 + float(_rr)) > 0.5)
    if base["cannot_pay_at_any_target"]:
        _need = round(100.0 / (1.0 + float(_rr)), 1)
        _tp_bp = full.get("row_tp_bps_median")
        base["cost_note"] = (
            "at the SHIPPED geometry the target is "
            f"{float(_rr):g}x the stop, so a first-touch rate above {_need}% "
            f"is needed before the {rt:.1f} bps round trip is covered, and "
            f"the certified gross rate is {round(float(_wr) * 100, 1)}%")
        base["cost_what_would_pay"] = (
            "the certified win rate needs a target of at least "
            f"{_fee_mult:.2f}x the round trip ({round(_fee_mult * rt, 1)} bps)"
            if _fee_mult is not None else None)
        base["target_verdict"] = base["cost_note"]
    elif _be is not None and _fee_floor > _be:
        # The distance is the honest one (the stop is where the measurement
        # put it), but no target that is 1.5x the round trip can also be the
        # break-even. Say that precisely instead of "cannot pay".
        base["cost_note"] = (
            f"break-even at this hit rate and payoff needs {_be:.1f} bps, "
            f"which is wider than the {_fee_floor:.1f} bps the cost rule "
            "allows — the target cannot satisfy both")
    else:
        base["cost_note"] = (
            "win_rate is the gross first-touch hit rate; the target must be "
            f">= {MIN_TP_OVER_RT}x the {rt:.1f} bps round trip before a win "
            "pays. Live locks enforce that floor (see geometry_cost_check).")
    return base


def geometry_cost_check(symbol: str, tf: str) -> dict[str, Any]:
    """Does the certified target for this cell clear the round trip?

    Uses the artifact's own selected geometry (k, m) and the grader's
    recorded median barrier scale where present, so the answer is
    derived from the shipped numbers — not from a fresh guess.  When
    the scale is not in the artifact the verdict is ``None`` (unknown),
    never a silent True.
    """
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return {"pays_round_trip": None, "why": "no certified cell"}
    sel = c.get("selected") or {}
    full = sel.get("full") or {}
    rt = 2.0 * float(c.get("friction_bps_per_side") or 0.0)
    # the row target in bps, from the grader's own median stop distance
    sl_bps = full.get("row_sl_bps_median")
    k = sel.get("row_k")
    tp_bps = (float(sl_bps) * float(k)
              if sl_bps is not None and k is not None else None)
    floor = MIN_TP_OVER_RT * rt
    return {
        "rt_bps": round(rt, 3),
        "min_tp_bps": round(floor, 3),
        "tp_bps": (round(float(tp_bps), 3) if tp_bps is not None else None),
        "pays_round_trip": (None if tp_bps is None
                            else bool(float(tp_bps) >= floor)),
        "why": ("target inside the round trip — every 'win' still loses "
                "money" if tp_bps is not None and float(tp_bps) < floor
                else "target clears the round trip" if tp_bps is not None
                else "artifact carries no median stop distance"),
    }


def friction_bps(symbol: str) -> float:
    """Per-side real friction (one round trip = 2× this)."""
    from backtest.data import FRICTION_BPS
    return float(FRICTION_BPS.get(symbol, 12.0))


# §18: an era ride is a POSITION, not a window — the maximum hold is
# a safety net (the normal exit is TP, SL or the certified side
# flipping), never the routine settlement horizon.
ERA_MAX_HOLD_S = 7 * 24 * 3600.0


def row_geometry(symbol: str, tf: str) -> tuple[float, float] | None:
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return None
    sel = c["selected"]
    return float(sel["row_k"]), float(sel["row_m"])


def sub_1_to_1_geometry(symbol: str, tf: str, k: float | None = None,
                        m: float | None = None) -> dict[str, Any] | None:
    """The certified target as a multiple of the certified stop — and the
    exchange fee the target must clear, if the geometry is unsettleable.

    ``None`` when the cell is not graded. Otherwise a dict carrying
    ``rr`` (target / stop), ``feasible`` (False when the ratio is under
    1:1) and, for an infeasible cell, the certified win rate next to the
    first-touch rate the geometry would actually need.
    """
    if k is None or m is None:
        g = row_geometry(symbol, tf)
        if g is None:
            return None
        k, m = g
    k, m = float(k), float(m)
    if k <= 0 or m <= 0:
        return None
    c = cell(symbol, tf) or {}
    wr = ((c.get("selected") or {}).get("full") or {}).get("row_wr")
    return {
        "rr": round(k / m, 4),
        "k": k, "m": m,
        "feasible": bool(k / m >= RR_LIVE_MIN - 1e-9),
        "certified_win_rate": wr,
        "win_rate_needed": round(100.0 / (1.0 + k / m), 2),
    }


def era_geometry(symbol: str, tf: str) -> tuple[float, float] | None:
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return None
    sel = c["selected"]
    return float(sel["era_k"]), float(sel["era_m"])


def era_gate(symbol: str, tf: str) -> int:
    """§20 TSMOM absolute-momentum gate lookback in minutes; 0 = off.

    Read from the certified artifact's ``selected.era_gate_l`` (written
    by ``backtest.hold.grade_cell``).  v8-era artifacts predate the
    field → 0 → live behaviour is byte-identical to before.
    """
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return 0
    return int((c.get("selected") or {}).get("era_gate_l") or 0)


# ─────────────────────────── side semantics ──────────────────────────
def side_value(side_name: str, closes: np.ndarray, i_entry: int,
               step_s: int) -> int:
    """One window's side — exactly ``backtest.build_sides`` semantics.

    ``closes[i_entry]`` is the entry (last bar strictly before the
    window); lookback is measured in BARS for the given step so a
    5-minute replay and a 1-minute live tape express the same
    ``L``-minute momentum.  Partial history → ``up`` (the documented
    full-coverage warmup fallback), never a skip.
    """
    if side_name == "up":
        return 1
    if side_name == "down":
        return -1
    if side_name.startswith("mom") and side_name[3:].isdigit():
        L = int(side_name[3:])
        step_min = max(step_s, 1) / 60.0
        lb = max(1, int(round(L / step_min)))
        if i_entry < lb:
            return 1                     # warmup fallback (hold rule)
        r = float(closes[i_entry]) / float(closes[i_entry - lb]) - 1.0
        if r > 0:
            return 1
        if r < 0:
            return -1
        return 1                         # flat → up (hold rule)
    if side_name == "momMaj":
        from backtest.hold import LONG_MOM_MIN as _LMM
        votes = []
        for L in _LMM:
            lb = max(1, int(round(L / (max(step_s, 1) / 60.0))))
            if i_entry < lb:
                votes.append(1)          # warmup leg → +1 (hold rule)
                continue
            r = float(closes[i_entry]) / float(closes[i_entry - lb]) - 1.0
            votes.append(1 if r > 0 else (-1 if r < 0 else 1))
        tot = sum(votes)
        return 1 if tot >= 0 else -1
    if side_name == "momMean":
        from backtest.hold import LONG_MOM_MIN as _LMM
        lb_max = max(1, int(round(max(_LMM) / (max(step_s, 1) / 60.0))))
        if i_entry < lb_max:
            return 1                      # warmup → +1 (hold rule)
        acc = 0.0
        for L in _LMM:
            lb = max(1, int(round(L / (max(step_s, 1) / 60.0))))
            if i_entry >= lb:
                acc += float(closes[i_entry]) / float(closes[i_entry - lb]) - 1.0
        acc /= float(len(_LMM))
        if acc > 0:
            return 1
        if acc < 0:
            return -1
        return 1
    if side_name in ("revW", "revW3", "revW5") or side_name.startswith("revWh"):
        # needs absolute timestamps (which window is being faded) —
        # the feed path uses ``_revw_live``; series use ``_series_for``
        raise ValueError(f"{side_name} requires timestamps — use _series_for "
                         f"or _revw_live")
    if side_name == "lr_revW":
        raise ValueError("lr_revW needs the full OOF tape — use _series_for")
    if side_name == "mom44h":
        # needs absolute timestamps — the feed path uses
        # ``_mom44h_live``; series paths use ``_series_for``
        raise ValueError(f"mom44h requires timestamps — use _series_for "
                         f"or _mom44h_live")
    raise ValueError(f"unknown certified side {side_name!r}")


def side_series(symbol: str, tf: str, bars: tuple) -> np.ndarray | None:
    """Vectorized certified side over a bar array — parity twin of
    ``backtest.hold.build_sides`` for this cell's selected side."""
    c = cell(symbol, tf)
    if not c or c.get("status") != "ok":
        return None
    return _series_for(str((c.get("selected") or {}).get("side")), bars,
                       TF_MINUTES[tf])


def _series_for(side_name: str, bars: tuple, w_min: int) -> np.ndarray:
    """Vectorized replica of hold.build_sides for ONE side name.

    Uses the identical formulas (mom ratio with warmup=+1, constant
    sides, hourly 44 h difference) so parity tests can compare
    element-wise against ``build_sides(...)[side_name]``.
    """
    t, o, h, l, cl, v, step = bars
    t = np.asarray(t)
    cl = np.asarray(cl, dtype=float)
    step_s = int(step)
    step_min = step_s / 60.0
    W = w_min * 60
    # epoch-aligned windows exactly like hold.grade_cell:
    # starts = ceil(t0 / W) * W … ≤ t1 − W
    t0 = int(math.ceil(int(t[0]) / W) * W)
    t1 = int(t[-1]) + step_s
    starts = np.arange(t0, t1 - W + 1, W, dtype=np.int64)
    if starts.size == 0:
        return np.zeros(0, dtype=int)
    step_s_i = step_s
    i_entry = np.searchsorted(t, starts - step_s_i, side="right") - 1
    i_exit = np.searchsorted(t, starts + W - step_s_i, side="right") - 1
    ok = (i_entry >= 0) & (i_exit > i_entry) & (i_exit < len(t))
    starts, i_entry = starts[ok], i_entry[ok]
    n = len(starts)
    if side_name == "up":
        return np.ones(n, dtype=int)
    if side_name == "down":
        return -np.ones(n, dtype=int)
    if side_name.startswith("mom") and side_name[3:].isdigit():
        L = int(side_name[3:])
        lb = max(1, int(round(L / step_min)))
        s = np.ones(n, dtype=int)
        # i_entry is ALREADY ok-filtered above; masking with the
        # pre-filter `ok` crashed whenever any window was invalid
        # (real BTC 5m tape: 171 such windows) — latent until cache
        # present made the parity test actually reach this branch
        good = i_entry >= lb
        r = np.zeros(n)
        idx = np.clip(i_entry, 0, None)
        r[good] = cl[idx[good]] / cl[np.maximum(idx[good] - lb, 0)] - 1.0
        s[good] = np.where(r[good] > 0, 1, np.where(r[good] < 0, -1, 1))
        return s
    if side_name == "momMaj":
        from backtest.hold import LONG_MOM_MIN as _LMM
        stack = np.zeros((len(_LMM), n))
        for _i, L in enumerate(_LMM):
            lb = max(1, int(round(L / step_min)))
            good = i_entry >= lb
            rr = np.zeros(n)
            idx = np.clip(i_entry, 0, None)
            rr[good] = cl[idx[good]] / cl[np.maximum(idx[good] - lb, 0)] - 1.0
            stack[_i] = np.where(rr > 0, 1, np.where(rr < 0, -1, 1))
        tot = stack.sum(axis=0)
        return np.where(tot >= 0, 1, -1).astype(int)
    if side_name == "momMean":
        from backtest.hold import LONG_MOM_MIN as _LMM
        lb_max = max(1, int(round(max(_LMM) / step_min)))
        good = i_entry >= lb_max
        rbar = np.zeros(n)
        idx = np.clip(i_entry, 0, None)
        for L in _LMM:
            lb = max(1, int(round(L / step_min)))
            okL = i_entry >= lb
            rr = np.zeros(n)
            rr[okL] = cl[idx[okL]] / cl[np.maximum(idx[okL] - lb, 0)] - 1.0
            rbar += rr
        rbar /= float(len(_LMM))
        out = np.ones(n, dtype=int)
        out[good] = np.where(rbar[good] > 0, 1, np.where(rbar[good] < 0, -1, 1))
        return out
    if side_name in ("revW", "revW3", "revW5") or side_name.startswith("revWh"):
        lag_windows = {"revW": 1, "revW3": 3, "revW5": 5}.get(side_name)
        base = np.searchsorted(
            t, starts - (lag_windows * W if lag_windows else W),
            side="right") - 1
        if lag_windows is None:            # revWh* hysteresis state machine
            from backtest.hold import _scale_at as _h_scale
            tau = float(side_name[5:])
            sc0 = _h_scale(t, cl, starts - step_s,
                           max(2, int(round(w_min / (step_s / 60.0)))))
            med = float(np.nanmedian(sc0)) if np.isfinite(sc0).any() else 0.0
            sc_f = np.where(np.isfinite(sc0) & (sc0 > 0), sc0, med)
            ip = np.searchsorted(t, starts - W, side="right") - 1
            st_ = 0
            out = np.ones(n, dtype=int)
            for i in range(n):
                b = int(ip[i])
                ref = 0 if b < 0 else b
                mv = cl[i_entry[i]] - cl[ref]
                prop = -1 if mv > 0 else (1 if mv < 0 else 0)
                if st_ == 0:
                    st_ = prop if prop else 1
                elif prop and abs(mv) > tau * sc_f[i] and prop != st_:
                    st_ = prop
                out[i] = st_
            return out
        out = np.ones(n, dtype=int)        # plain fades
        for i in range(n):
            b = int(base[i])
            if b < 0:
                b0 = int(i_entry[i]) - 1
                if b0 < 0:
                    continue                # first window → +1 (hold rule)
                ref = 0
            else:
                ref = b
            mv = cl[i_entry[i]] - cl[ref]
            if mv > 0:
                out[i] = -1
            elif mv < 0:
                out[i] = 1
        return out
    if side_name in ("lr_revW", "lrE_revW"):
        # composite: OOF logistic where it opines, revW fallback —
        # shares the BENCHMARK's lr_sides_oof/composite_sides (they ARE
        # the spec; build_sides itself never contains these keys — the
        # composite is applied AFTER build in grade_cell)
        from backtest.hold import (lr_sides_oof as _lr_oof,
                                   composite_sides as _comp)
        i_exit = np.searchsorted(t, starts + W - step_s, side="right") - 1
        try:
            moves = cl[i_exit] - cl[i_entry]
            lr_s, lr_sc = _lr_oof(t, o, h, l, cl, v, step_s,
                                  starts, i_entry, moves, w_min)
        except Exception:
            lr_s, lr_sc = None, None
        base_rev = _series_for("revW", bars, w_min)
        comp = _comp({"revW": base_rev}, lr_s, lr_sc)
        return comp[side_name] if side_name in comp else base_rev
    if side_name == "mom44h":
        # hold.py rule: hourly resample, d44 = c[i_entry] − h_c[j−44]
        from backtest.continuous import _hourly_final
        h_ts, h_c = _hourly_final(t, cl)
        hb = (t[np.clip(i_entry, 0, None)] // 3600) * 3600
        j = np.searchsorted(h_ts, hb, side="right") - 1
        mom = np.ones(n, dtype=int)
        with np.errstate(divide="ignore", invalid="ignore"):
            d44 = cl[np.clip(i_entry, 0, None)] - h_c[np.clip(j - 44, 0, None)]
        valid = (i_entry >= 0) & (j >= 44) & np.isfinite(d44)
        mom[valid] = np.where(d44[valid] > 0, 1,
                              np.where(d44[valid] < 0, -1, 1))
        return mom
    raise ValueError(f"unknown certified side {side_name!r}")


def window_entry_index(times, tf: str, now: float | None = None) -> int:
    """Index of the bar strictly before the CURRENT epoch window (live
    analogue of hold.py's ``i_entry``)."""
    W = TF_MINUTES[tf] * 60
    anchor = int(now if now is not None else (times[-1] if times else 0))
    s = anchor // W * W
    return bisect_left(times, s) - 1


def side_at_feed(feed, symbol: str, tf: str, now: float | None = None
                 ) -> tuple[int, dict[str, Any]]:
    """Live certified side for (asset, tf) with honest metadata.

    Reads the feed's CLOSED minute tape only (causal): entry = last bar
    strictly before the current epoch window, exactly the benchmark's
    anchor.  ``meta.warmup`` is True while the lookback history is
    shorter than L (side falls back to ``up`` per the certified rule —
    full coverage, never a skip, never silent).
    """
    pol = summary(symbol, tf)
    if not pol.get("managed"):
        return 0, {"reason": "cell not managed by the hold policy"}
    side_name = pol.get("side")
    meta: dict[str, Any] = {
        "policy_side": side_name, "tier": pol.get("tier"),
        "certified": pol.get("certified"), "warmup": False,
    }
    try:
        st = feed.state(symbol)
        times = list(st.times)
        closes = list(st.closes)
    except Exception:
        return 0, {**meta, "reason": "no tape"}
    if not times or len(times) != len(closes):
        return 0, {**meta, "reason": "empty tape"}
    if not pol.get("certified"):
        # UNGRADEABLE / WARMUP cell: no certified side exists — the
        # live call falls back to the engine raw ladder (side 0 here,
        # caller keeps its own side) and the UI shows the warmup badge
        return 0, {**meta, "warmup": True,
                   "reason": "no certified measurement on 1 h history"}
    step_s = int(round(float(times[1] - times[0]))) if len(times) > 1 else 60
    i_entry = window_entry_index(times, tf, now)
    if i_entry < 0:
        return 1, {**meta, "warmup": True, "reason": "window start has "
                                                     "no prior bar"}
    import numpy as _np
    closes_a = _np.asarray(closes, dtype=float)
    if side_name == "mom44h":
        # pointwise hourly path needs timestamps; bucket math below
        side = _mom44h_live(times, closes_a, i_entry, side_name)
    else:
        side = side_value(str(side_name), closes_a, i_entry, step_s)
    L = None
    if side_name and side_name.startswith("mom") and side_name[3:].isdigit():
        L = int(side_name[3:])
        step_min = step_s / 60.0
        lb = max(1, int(round(L / step_min)))
        if i_entry < lb:
            meta["warmup"] = True
            meta["lookback_bars"] = lb
            meta["have_bars"] = int(i_entry) + 1
            meta["reason"] = (f"warmup: {i_entry + 1}/{lb} bars of "
                              f"{side_name} history — certified fallback "
                              f"side +1 (hold.py full-coverage rule)")
    return int(side), meta


def _mom44h_live(times, closes, i_entry: int, side_name: str) -> int:
    """44 h hourly-difference rule on the live tape."""
    s = int(times[i_entry])
    target_h = s // 3600 - 44
    target_ts = target_h * 3600
    j = bisect_left(times, target_ts) - 1
    if j < 0:
        return 1
    d44 = float(closes[i_entry]) - float(closes[j])
    if d44 > 0:
        return 1
    if d44 < 0:
        return -1
    return 1


# ─────────────────── row settlement (hold rules) ─────────────────────
def row_settle_rules(symbol: str, tf: str) -> dict[str, Any]:
    """How a sealed lock for this cell must be graded — the certified
    benchmark row rules. The backtest reports SL-first as a pessimistic
    bound; live OHLC cannot order same-bar TP/SL, so live grading marks
    that bar ambiguous and never learns a guessed result. A flat expiry
    is a push (friction belongs to measured trades/eras, never rows)."""
    c = cell(symbol, tf)
    if not is_hold_cell(symbol, tf):
        return {"hold_rules": False}
    g = row_geometry(symbol, tf)
    return {
        "hold_rules": True,
        "row_k": g[0] if g else None,
        "row_m": g[1] if g else None,
        "tie_rule": ("live: both barriers in one closed OHLC bar = "
                     "ambiguous, no W/L; backtest SL-first is pessimistic bound"),
        "flat_rule": "zero move at max hold = push (not a fabricated loss)",
        "friction": ("charged once per era round trip by the era "
                     "accountant — rows carry none"),
    }


def barrier_payability(tp_bps: float, sl_bps: float, rt_bps: float) -> dict:
    """Is this barrier geometry structurally payable? (P26, in plain terms)

    The 3-year grading found the decisive defect: the certified geometry
    aimed a target (~23 bps on BTC tf1) INSIDE the round trip (~29 bps),
    so 86 % of windows "won" and the book still lost ~29 bps per trade.

    ``payable`` is False when the target cannot clear the round trip at
    all, or when the reward:risk is lottery-shaped (a 1:40 geometry wins
    often and loses the account).  The caller must treat False as a
    refusal to issue the lock, not as a warning.
    """
    tp = float(tp_bps or 0.0)
    sl = float(sl_bps or 0.0)
    rt = float(rt_bps or 0.0)
    if tp <= 0 or sl <= 0:
        return {"payable": False, "reason": "degenerate barrier",
                "tp_bps": tp, "sl_bps": sl, "rt_bps": rt}
    efficiency = (tp - rt) / tp            # share of the target after cost
    rr = tp / sl
    # the two refusals, stated as numbers rather than adjectives
    below_cost = tp < MIN_TP_OVER_RT * rt
    lottery = rr < MIN_REWARD_RISK
    return {
        "payable": not (below_cost or lottery),
        "below_cost": bool(below_cost),
        "lottery_shape": bool(lottery),
        "tp_bps": round(tp, 2), "sl_bps": round(sl, 2), "rt_bps": round(rt, 2),
        "efficiency": round(efficiency, 4),
        "reward_risk": round(rr, 4),
        "min_tp_bps": round(MIN_TP_OVER_RT * rt, 2),
        "min_reward_risk": MIN_REWARD_RISK,
        "reason": ("target sits inside the round trip" if below_cost else
                   "reward:risk below the lottery floor" if lottery else
                   "payable geometry"),
    }


#: USER RULE (absolute): a trade may never pay less than it risks.
RR_LIVE_MIN = 1.0


def _cost_aware(symbol: str, entry: float, side: int, scale_px: float,
                k: float, m: float) -> tuple[float, float, dict[str, Any]]:
    """(tp_px, sl_px, meta) with the round-trip floor enforced.

    The certified gross geometry (k=0.15, m≈0.55) deliberately aims the
    target at a fraction of the measured window move, because a tight
    target maximises the FIRST-TOUCH hit rate.  That is a hit-rate
    metric, not a money metric: on BTC tf1 the target is ~23 bps while
    the Binance round trip is ~29 bps, so 86 % of windows "win" and the
    book still bleeds ~29 bps per trade (measured 2024-2026, see
    ``backtest/results/wr_audit_independent.json``).

    The floor: a target closer than ``MIN_TP_OVER_RT`` x round trip is
    widened to exactly that floor, keeping the stop where the certified
    rule put it.  Widening a target can only LOWER the hit rate, never
    raise it — so the reported win rate becomes attainable rather than
    decorative.
    """
    # CERTIFIED GEOMETRY: stop = m x scale, target = k x stop, both kept
    # inside the price (a fat warmup ATR proxy must never push a SELL's TP
    # below zero — the seal refuses that).
    sl_px = min(m * float(scale_px), entry * 0.9)
    tp_px = min(k * sl_px, entry * 0.9)
    rt_bps = round_trip_bps(symbol)
    tp_bps = (tp_px / entry * 1e4) if entry > 0 else 0.0
    floor_bps = MIN_TP_OVER_RT * rt_bps
    widened = bool(tp_bps > 0 and floor_bps > 0 and tp_bps < floor_bps
                   and entry > 0)
    if widened:
        # floor the TARGET; the certified stop is left untouched
        tp_px = floor_bps * 1e-4 * entry

    # ── REWARD:RISK FLOOR (user rule, absolute) ───────────────────────
    # Found 2026-11-27. The cost floor above treated "pays the round trip"
    # as sufficient, but on the certified k=0.15 geometry the target is
    # 0.15x the stop: widening it to 1.5x RT (44.7 bps on ETH) against a
    # 55 bps stop still ships 0.81:1 — a target that cannot pay for its
    # own stop no matter how often it is touched. That is a rule
    # violation in the LIVE geometry, not a display bug: every sealed
    # hold lock was built from it.
    #
    # The repair direction is fixed by the user: a floor may WIDEN THE
    # TARGET and may never NARROW THE STOP. Only when the target cannot
    # legally widen any further (the 0.9x price cap) does the stop give
    # way — and it gives way by getting WIDER, never tighter.
    rr_before = (tp_px / sl_px) if sl_px > 0 else None
    tp_widened_for_rr = False
    if sl_px > 0 and tp_px < RR_LIVE_MIN * sl_px:
        # The stop is already capped at 0.9 x price, so the target can
        # always be widened to pay it — this cap is what makes the floor
        # reachable without ever touching the stop. (No stop-narrowing
        # branch exists here on purpose: a tighter stop raises the loss
        # frequency to buy a prettier ratio, which is the opposite of the
        # user's rule.)
        tp_px = min(RR_LIVE_MIN * sl_px, entry * 0.9)
        tp_widened_for_rr = True
    tp_px = min(tp_px, entry * 0.9) if entry > 0 else tp_px
    ratio = (tp_px / sl_px) if sl_px > 0 else None
    # The floor is structural, not aspirational: the stop can never exceed
    # the shared 0.9 x price cap, so a target at the same cap always pays
    # it. Anything else is a bug and must not reach a seal.
    if ratio is not None and ratio < RR_LIVE_MIN - 1e-9:
        raise AssertionError(
            f"reward:risk {ratio:.4f} below the {RR_LIVE_MIN:g}:1 floor "
            f"({symbol} entry={entry} scale={scale_px} k={k} m={m})")
    meta = {
        "rt_bps": round(rt_bps, 3),
        "tp_bps_certified": round(tp_bps, 3),
        "tp_bps_used": (round(min(tp_px / entry, 0.9) * 1e4, 3)
                        if entry else 0.0),
        "min_tp_bps": round(floor_bps, 3),
        "tp_widened_for_cost": widened,
        "tp_pays_round_trip": bool(tp_bps >= floor_bps) if tp_bps else None,
        #: reward:risk the LIVE lock is actually built on, and the
        #: certified one it started from — both published so no surface
        #: has to infer a ratio from prices it was handed.
        "rr_certified": (round(rr_before, 4) if rr_before else None),
        "rr_live": (round(ratio, 4) if ratio else None),
        "rr_floor": RR_LIVE_MIN,
        "tp_widened_for_rr": tp_widened_for_rr,
        "note": "; ".join(filter(None, [
            ("certified target sits inside the round trip — widened to "
             f"{MIN_TP_OVER_RT}x RT so a win actually pays"
             if widened else "certified target clears the round trip"),
            (f"target widened to the {RR_LIVE_MIN:g}:1 reward:risk floor"
             if tp_widened_for_rr else ""),
        ])),
    }
    return entry + tp_px * side, entry - sl_px * side, meta


def build_row_bars(symbol: str, tf: str, entry: float, side: int,
                   scale_px: float) -> tuple[float, float] | None:
    """(tp_px, sl_px) for an EVERY-WINDOW row lock from the certified
    decoupled row geometry.  ``scale_px`` = q80 window-move scale at the
    entry (``_scale_at`` semantics, price units).  Cost-aware: see
    ``_cost_aware``."""
    g = row_geometry(symbol, tf)
    if g is None or entry <= 0 or scale_px <= 0 or not side:
        return None
    k, m = g
    tp, sl, _meta = _cost_aware(symbol, entry, side, scale_px, k, m)
    return tp, sl


def build_era_bars(symbol: str, tf: str, entry: float, side: int,
                   scale_px: float) -> tuple[float, float] | None:
    """(tp_px, sl_px) for an **era-ride** lock (§18): SL = m_era·scale
    (guaranteed stop, m ≤ 3), TP = k_era·SL.  The ride itself exits at
    TP, SL, the certified side flipping, or the maximum hold — never at
    the window boundary (a window-end settle is the −RT trade the §18
    probes measured).  Cost-aware: see ``_cost_aware``."""
    g = era_geometry(symbol, tf)
    if g is None or entry <= 0 or scale_px <= 0 or not side:
        return None
    k, m = g
    tp, sl, _meta = _cost_aware(symbol, entry, side, scale_px, k, m)
    return tp, sl

# [removed 2026-11-23] geometry_for_live — dead code: zero references anywhere


def scale_at_bars(bars: tuple, at_ts, tf: str) -> np.ndarray:
    """Certified barrier scale (q80 of trailing W-length moves) — the
    same call hold.grade_cell makes, imported for parity."""
    from backtest.continuous import _scale_at
    t, o, h, l, c, v, step = bars
    W = TF_MINUTES[tf] * 60
    lb = max(2, int(round(W / (int(step) / 60.0))))
    return _scale_at(np.asarray(t), np.asarray(c, dtype=float),
                     np.asarray(at_ts), lb)


def scale_at_feed(feed, symbol: str, tf: str,
                  now: float | None = None) -> float | None:
    """Live pointwise q80 scale on the feed tape (price units) —
    identical estimator to ``scale_at_bars`` at one boundary."""
    try:
        st = feed.state(symbol)
        times = list(st.times)
        closes = list(st.closes)
    except Exception:
        return None
    if len(times) < 16:
        return None
    W = TF_MINUTES[tf] * 60
    step_s = int(round(float(times[1] - times[0]))) if len(times) > 1 else 60
    lb = max(2, int(round(W / (step_s / 60.0))))
    anchor = int(now if now is not None else times[-1])
    s = anchor // W * W
    i = bisect_left(times, s) - 1
    if i < lb * 2:
        return None
    lo = max(lb, i - 100 * lb)
    js = list(range(lo, i + 1, lb))
    if len(js) < 8:
        return None
    moves = [abs(closes[j] - closes[j - lb]) for j in js]
    return float(np.quantile(np.asarray(moves, dtype=float), 0.80))


# ───────────────────── per-cell ANALYSE timer lock ───────────────────
class CellLockBook:
    """One ANALYSE click → the cell sleeps for its full TF window.

    While ``(asset, tf)`` is locked the app refuses new predictions for
    THAT cell only; every other cell (and the same asset on other TFs)
    stays free.  4 assets × 4 TFs = 16 independent timers.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get(
            "CELL_LOCKS_PATH", os.path.join("data", "cell_locks.json"))
        self._locks: dict[str, float] = {}
        self._load()

    def _key(self, symbol: str, tf: str) -> str:
        return f"{symbol}|{tf}"

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._locks = {k: float(v) for k, v in
                           (raw.get("locks") or {}).items()}
        except (OSError, ValueError, FileNotFoundError):
            self._locks = {}

    def _save(self) -> None:
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "locks": self._locks}, fh)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def enabled(self) -> bool:
        """HOLD_CELL_LOCKS=1 forces on (tests), =0 forces off.
        Under pytest the lock is off by default so legacy suites that
        analyse the same cell repeatedly keep working; the lock tests
        force it on.  Production (no env): always on."""
        v = os.environ.get("HOLD_CELL_LOCKS")
        if v == "1":
            return True
        if v == "0":
            return False
        return not os.environ.get("PYTEST_CURRENT_TEST")

    def remaining(self, symbol: str, tf: str,
                  now: float | None = None) -> float:
        if not self.enabled():
            return 0.0
        now = float(now if now is not None else time.time())
        until = float(self._locks.get(self._key(symbol, tf), 0.0))
        return max(0.0, until - now)

    def locked(self, symbol: str, tf: str,
               now: float | None = None) -> bool:
        return self.remaining(symbol, tf, now) > 0

    def until(self, symbol: str, tf: str) -> float:
        return float(self._locks.get(self._key(symbol, tf), 0.0))

    def start(self, symbol: str, tf: str, *, now: float | None = None,
              minutes: int | None = None) -> dict[str, Any]:
        """Arm the timer for this cell (called at ANALYSE click)."""
        now = float(now if now is not None else time.time())
        mins = int(minutes if minutes is not None
                   else TF_MINUTES.get(tf, 22))
        until = now + mins * 60.0
        self._locks[self._key(symbol, tf)] = until
        self._save()
        return {"symbol": symbol, "tf": tf, "locked_until": until,
                "minutes": mins, "remaining_s": round(until - now, 1)}

    def clear(self, symbol: str, tf: str) -> None:
        self._locks.pop(self._key(symbol, tf), None)
        self._save()

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = float(now if now is not None else time.time())
        out: dict[str, Any] = {}
        for symbol, tf in known_cells():
            rem = self.remaining(symbol, tf, now)
            out[f"{symbol}|{tf}"] = {
                "locked": rem > 0,
                "remaining_s": round(rem, 1),
                "locked_until": self.until(symbol, tf),
                "minutes": TF_MINUTES[tf],
                "can_analyse": rem <= 0,
            }
        return out


CELL_LOCKS: CellLockBook | None = None


def cell_locks() -> CellLockBook:
    global CELL_LOCKS
    with _lock:
        if CELL_LOCKS is None:
            CELL_LOCKS = CellLockBook()
        return CELL_LOCKS
