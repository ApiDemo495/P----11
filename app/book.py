"""Live order book (top-20 partial snapshots) + derived depth physics.

Feed: Binance spot `<symbol>@depth20@100ms` partial-book streams (full
top-20 replacement every100 ms — no U/u diff bookkeeping), plus the
aggTrade stream we already consume for the aggressor side of fills.

Derived, MEASURED signals (all bounded, all honest-None until enough
snapshots exist):

  * dbi20      — depth imbalance (Σbid − Σask)/(Σbid + Σask) in [−1,1]
  * micro_pos  — inventory-weighted micro-price position inside the
                 spread: 2·((P_micro − bid)/(ask − bid) − 0.5) ∈ [−1,1]
  * spread_bps — top-of-book spread in basis points
  * walls      — largest resting size each side (price, bps, qty)
  * lambda     — Reflexive Cancellation Wavefront Elasticity (the
                 Chrono-Sieve gauge, kept and GRADABLE here): within a
                 rolling1 s window, depth reductions NOT explained by
                 executed aggressive volume, distance-weighted across
                 the20 visible ticks, Λ = ln(V_cancel_ask / V_cancel_bid)

Rings (dbi / micro-pos / Λ) are exposed for FormulaContext so F32–F35
read them like any other formula input; the simulator never produces
book packets, so the rings stay empty and the formulas return0 with a
'no live book' reason (cause-6 honesty).
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Any


class DepthBook:
    SNAP_MIN = 10            # snapshots before Λ may speak (~1 s)
    WINDOW_S = 1.0           # cancel-elasticity measurement window
    RING_MAX = 600           # per-symbol formula rings (~60 s @10 Hz)
    BUF_TRIM_S = 2.5         # aggTrade buffer kept slightly longer

    def __init__(self) -> None:
        self._book: dict[str, dict] = {}      # sym -> {bids, asks, ts}
        self._prev: dict[str, tuple] = {}     # sym -> (prev_bids, prev_asks, ts)
        self._cancels: dict[str, deque] = {}  # sym -> (ts, cbid_w, cask_w)
        self._agg: dict[str, deque] = {}      # sym -> (ts, qty, seller_agg)
        self._dbi: dict[str, deque] = {}
        self._dev: dict[str, deque] = {}
        self._lam: dict[str, deque] = {}
        self.snap_count = 0                   # global snapshot counter

    # ------------------------------------------------------------- inputs
    def on_aggtrade(self, symbol: str, t_ms: int, qty: float,
                    seller_aggressor: bool) -> None:
        """seller_aggressor=True (aggTrade m=true) → the sell swept BIDS."""
        buf = self._agg.setdefault(symbol, deque())
        buf.append((time.time(), float(qty), bool(seller_aggressor)))
        now = time.time()
        while buf and now - buf[0][0] > self.BUF_TRIM_S:
            buf.popleft()

    def on_depth(self, symbol: str, bids: list, asks: list,
                 ts_ms: int) -> None:
        try:
            cur_b = {float(p): float(q) for p, q in bids if float(q) > 0}
            cur_a = {float(p): float(q) for p, q in asks if float(q) > 0}
        except Exception:
            return
        if not cur_b or not cur_a:
            return
        now = time.time()
        prev = self._prev.get(symbol)
        cbid_w = cask_w = 0.0
        if prev:
            pb, pa, pts = prev
            # aggressive volume that legitimately consumed the book since
            # the previous snapshot (aggTrade arrived in between)
            exec_buy = exec_sell = 0.0
            buf = self._agg.get(symbol) or deque()
            keep = deque()
            for ts, q, seller_agg in buf:
                if ts < pts - 0.05:      # stale (before previous snapshot)
                    continue
                if ts > now + 0.05:
                    keep.append((ts, q, seller_agg))
                    continue
                if seller_agg:
                    exec_sell += q       # sell aggressor eats BIDS
                else:
                    exec_buy += q        # buy aggressor eats ASKS
                # consumed rows are dropped — kept only within the window
            self._agg[symbol] = keep if keep else deque(
                (ts, q, sa) for ts, q, sa in buf if ts > now - self.WINDOW_S)

            def _lost(prev_d: dict, cur_d: dict, exec_vol: float,
                      desc: bool) -> float:
                """depth reduction NOT explained by fills, distance-weighted
                (level0 = best, weight 1 + d/20 across the20 visible ticks)."""
                order = sorted(prev_d, reverse=desc)
                plain = weighted = 0.0
                for idx, lvl in enumerate(order):
                    q0 = prev_d[lvl]
                    q1 = cur_d.get(lvl, 0.0)
                    if q1 < q0:
                        d_qty = q0 - q1
                        plain += d_qty
                        weighted += d_qty * (1.0 + idx / 20.0)
                cancels = max(0.0, plain - exec_vol)
                if plain > 0:
                    return cancels * (weighted / plain)
                return 0.0

            cbid_w = _lost(pb, cur_b, exec_sell, desc=True)
            cask_w = _lost(pa, cur_a, exec_buy, desc=False)
        self._prev[symbol] = (cur_b, cur_a, now)
        self._book[symbol] = {"bids": cur_b, "asks": cur_a, "ts": now}

        ring = self._cancels.setdefault(symbol, deque())
        ring.append((now, cbid_w, cask_w))
        while ring and now - ring[0][0] > self.WINDOW_S * 8:
            ring.popleft()
        # formula rings (one sample per snapshot)
        d = self._dbi.setdefault(symbol, deque())
        dev = self._dev.setdefault(symbol, deque())
        lam_ring = self._lam.setdefault(symbol, deque())
        bq = sum(cur_b.values())
        aq = sum(cur_a.values())
        if bq + aq > 0:
            d.append(round((bq - aq) / (bq + aq), 6))
            bid = max(cur_b)
            ask = min(cur_a)
            if ask > bid:
                pmicro = (bq * ask + aq * bid) / (bq + aq)
                dev.append(round(2.0 * ((pmicro - bid) / (ask - bid) - 0.5),
                                 6))
        # Λ over the trailing WINDOW (needs both sides to have measured
        # something — a quiet book keeps its honest last value absent)
        w = now - self.WINDOW_S
        v_cbid = sum(c2 for ts, c2, _c in ring if ts >= w)
        v_cask = sum(c2 for ts, _c, c2 in ring if ts >= w)
        if v_cbid > 1e-9 and v_cask > 1e-9:
            lam_ring.append(round(math.log((v_cask + 1e-9) /
                                           (v_cbid + 1e-9)), 6))
        for rr in (d, dev, lam_ring, self._cancels[symbol]):
            while len(rr) > self.RING_MAX:
                rr.popleft()
        self.snap_count += 1

    # ------------------------------------------------------------- status
    def status(self, symbol: str, price: float) -> dict[str, Any]:
        bk = self._book.get(symbol)
        if not bk:
            return {"book": False,
                    "reason": "no live order book yet "
                              "(needs <sym>@depth20@100ms)"}
        bids, asks = bk["bids"], bk["asks"]
        bid, ask = max(bids), min(asks)
        bq, aq = sum(bids.values()), sum(asks.values())
        spread = max(ask - bid, 0.0)
        dbi = round((bq - aq) / (bq + aq), 6) if bq + aq else None
        pmicro = ((bq * ask + aq * bid) / (bq + aq)) if (bq + aq
                  and ask > bid) else None
        dev = (round(2.0 * ((pmicro - bid) / (ask - bid) - 0.5), 6)
               if pmicro is not None and ask > bid else None)
        spread_bps = (round(spread / mid * 1e4, 3)
                      if (mid := (ask + bid) / 2) > 0 else None)
        # walls: largest resting size each side (top visible20)
        wb = max(bids.items(), key=lambda kv: kv[1]) if bids else None
        wa = max(asks.items(), key=lambda kv: kv[1]) if asks else None
        wall_b = ({"price": wb[0], "qty": round(wb[1], 6),
                   "bps": (round((price - wb[0]) / price * 1e4, 1)
                           if price > 0 else None)}
                  if wb else None)
        wall_a = ({"price": wa[0], "qty": round(wa[1], 6),
                   "bps": (round((wa[0] - price) / price * 1e4, 1)
                           if price > 0 else None)}
                  if wa else None)
        # Λ: current window sums
        now = time.time()
        w = now - self.WINDOW_S
        ring = self._cancels.get(symbol) or deque()
        v_cbid = sum(c2 for ts, c2, _c in ring if ts >= w)
        v_cask = sum(c2 for ts, _c, c2 in ring if ts >= w)
        lam = None
        lam_reason = ""
        if self.snap_count < self.SNAP_MIN:
            lam_reason = (f"needs {self.SNAP_MIN} book snapshots "
                          f"(had {self.snap_count})")
        elif v_cbid <= 1e-9 or v_cask <= 1e-9:
            lam_reason = ("quiet book — no balanced cancel flow "
                          "measured in the last second")
        else:
            lam = round(math.log((v_cask + 1e-9) / (v_cbid + 1e-9)), 4)
        return {
            "book": True,
            "dbi": dbi,
            "micro_pos": dev,
            "spread_bps": spread_bps,
            "bid": bid, "ask": ask,
            "bid_qty": round(bq, 6), "ask_qty": round(aq, 6),
            "wall_bid": wall_b, "wall_ask": wall_a,
            "lambda": lam, "lambda_reason": lam_reason,
            "v_cancel_bid": round(v_cbid, 6),
            "v_cancel_ask": round(v_cask, 6),
            "snapshots": self.snap_count,
            "age_s": round(now - bk["ts"], 2),
        }

    # -------------------------------------------------- FormulaContext view
    def view(self, symbol: str) -> dict[str, Any]:
        """Lightweight rings for F32/F33/F35 + latest scalars."""
        st = self.status(symbol, price=0.0)
        return {
            "dbi_ring": list(self._dbi.get(symbol) or ()),
            "dev_ring": list(self._dev.get(symbol) or ()),
            "lambda_ring": list(self._lam.get(symbol) or ()),
            "latest": st,
        }
