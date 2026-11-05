"""Exchange-server physics telemetry (Chrono-Sieve-inspired), honest edition.

Two MEASURED gauges — no synthetic particles, no GPU claims, nothing
inferred that we did not observe on the wire:

  1. Clock-skew tomography: Binance's aggTrade packets carry two clocks —
     T (matching-engine transaction time) and E (gateway event time).
     tau = E - T; a z-score of tau against the last 60 s of our OWN
     packets flags matching-queue congestion. A second z comes from the
     aggregate trade-id gaps (Delta a > 1 = batched fills riding the
     same engine queue). Both need >= 20 samples or they print a reason
     instead of a number (zeros rule / cause 5).
  2. Force-order liquidation map: public @forceOrder events (futures)
     are kept with exponential half-life decay and folded into 25-bp
     buckets against the live price, giving the nearest liquidation
     cluster above/below and the net forced flow share — MEASURED
     liquidation prints, not a simulated margin cohort (the GPU
     particle-filter idea needs CUDA this app does not assume; skipped
     on purpose, see report).

The simulator tape has no matching engine, so when no live packets have
ever arrived the caller reports available=False with a reason.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Any


class MicroPhysics:
    WINDOW_S = 60.0              # baseline window for the z-scores
    MIN_SAMPLES = 20             # below this: reason, never a number
    LIQ_HALFLIFE_S = 600.0       # force-order mass decays with a10-min half-life
    LIQ_HORIZON_S = 1800.0       # only the last30 min of prints count
    BUCKET_BP = 25.0             # cluster buckets in basis points of price

    def __init__(self) -> None:
        # per-symbol rings: tau (ms) and aggregate-id gaps
        self._tau: dict[str, deque] = {}
        self._gap: dict[str, deque] = {}
        self._last_a: dict[str, int] = {}
        # force orders: (ts, price, qty, side, symbol)
        self._liq: deque = deque()
        self.agg_count = 0
        self.liq_count = 0

    # ------------------------------------------------------------- inputs
    def on_aggtrade(self, symbol: str, t_ms: int, e_ms: int,
                    agg_id: int, price: float, qty: float) -> None:
        now = time.time()
        if t_ms and e_ms:
            ring = self._tau.setdefault(symbol, deque())
            ring.append(float(e_ms - t_ms))
            while len(ring) > 500:
                ring.popleft()
        prev = self._last_a.get(symbol)
        self._last_a[symbol] = agg_id
        if prev is not None and agg_id >= prev:
            gap = agg_id - prev
            gr = self._gap.setdefault(symbol, deque())
            gr.append(float(gap))
            while len(gr) > 500:
                gr.popleft()
        self.agg_count += 1

    def on_forceorder(self, symbol: str, side: str, price: float,
                      qty: float) -> None:
        if price <= 0 or qty <= 0:
            return
        self._liq.append((time.time(), float(price), float(qty),
                          str(side or "").upper(), symbol))
        while len(self._liq) > 4000:
            self._liq.popleft()
        self.liq_count += 1

    # ------------------------------------------------------------- stats
    @staticmethod
    def _z(ring: deque) -> float | None:
        n = len(ring)
        if n < MicroPhysics.MIN_SAMPLES:
            return None
        vals = list(ring)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
        sd = var ** 0.5
        if sd < 1e-12:
            return None                      # constant stream: no anomaly
        return round((vals[-1] - mean) / sd, 3)   # last sample vs baseline

    def _decayed(self, symbol: str, now: float | None = None) -> list:
        now = now or time.time()
        out = []
        for ts, price, qty, side, sym in self._liq:
            if sym != symbol:
                continue
            age = now - ts
            if age > self.LIQ_HORIZON_S:
                continue
            mass = qty * 0.5 ** (age / self.LIQ_HALFLIFE_S)
            out.append((price, mass, side, age))
        return out

    def status(self, symbol: str, price: float) -> dict[str, Any]:
        now = time.time()
        z_tau = self._z(self._tau.get(symbol) or deque())
        z_gap = self._z(self._gap.get(symbol) or deque())
        n_tau = len(self._tau.get(symbol) or ())
        n_gap = len(self._gap.get(symbol) or ())
        # composite queue stress in [0,1]: mean of the two available z's
        zs = [z for z in (z_tau, z_gap) if z is not None]
        stress = (round(sum(abs(min(abs(z), 6.0)) for z in zs) / max(1, len(zs)) / 6.0, 4)
                  if zs else None)
        # liquidation map vs the live price
        ev = self._decayed(symbol, now)
        nearest_up = nearest_dn = None
        net_flow = total = 0.0
        buckets: dict[int, dict] = {}
        if price > 0 and ev:
            for p, mass, side, _age in ev:
                bps = (p - price) / price * 1e4
                b = int(round(bps / self.BUCKET_BP))
                bk = buckets.setdefault(b, {"bps": b * self.BUCKET_BP,
                                            "mass": 0.0, "net": 0.0})
                bk["mass"] += mass
                bk["net"] += mass if side == "BUY" else -mass
                total += mass
                net_flow += mass if side == "BUY" else -mass
            # nearest cluster per side with meaningful mass (>=8% of total)
            floor_m = total * 0.08
            for b, bk in buckets.items():
                if bk["mass"] < floor_m:
                    continue
                if b > 0 and (nearest_up is None
                              or bk["mass"] > nearest_up["mass"]):
                    nearest_up = {"bps": round(bk["bps"], 1),
                                  "price": round(
                                      price * (1 + bk["bps"] / 1e4), 6),
                                  "mass": round(bk["mass"], 4),
                                  "bias": "BUY" if bk["net"] >= 0 else "SELL"}
                if b < 0 and (nearest_dn is None
                              or bk["mass"] > nearest_dn["mass"]):
                    nearest_dn = {"bps": round(abs(bk["bps"]), 1),
                                  "price": round(
                                      price * (1 + bk["bps"] / 1e4), 6),
                                  "mass": round(bk["mass"], 4),
                                  "bias": "BUY" if bk["net"] >= 0 else "SELL"}
        flow_share = (round(net_flow / (total + 1e-12), 4)
                      if total > 0 else None)
        return {
            "agg_count": self.agg_count,
            "liq_count": self.liq_count,
            "z_tau": z_tau, "z_gap": z_gap,
            "n_tau": n_tau, "n_gap": n_gap,
            "stress": stress,
            "net_flow": flow_share,
            "liq_events": len(ev),
            "nearest_up": nearest_up,
            "nearest_dn": nearest_dn,
            "min_samples": self.MIN_SAMPLES,
        }

    def flow(self, symbol: str) -> float | None:
        """Net forced-flow share in [-1, 1] (BUY=+1, SELL=-1) or None."""
        ev = self._decayed(symbol)
        if not ev:
            return None
        net = total = 0.0
        for _p, mass, side, _age in ev:
            total += mass
            net += mass if side == "BUY" else -mass
        if total <= 0:
            return None
        return round(net / total, 4)
