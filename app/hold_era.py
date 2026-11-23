"""One-position-per-side-era accountant — live T2 of the hold policy.

``backtest.hold.era_sim`` is the CERTIFIED batch implementation of the
era simulation (position enters at a side flip, TP/SL first-touch on
native bars with SL winning same-bar ties, otherwise exit at the flip
boundary, friction = one round trip per era, ``m >= 999`` rides to the
flip).  This module provides:

  * ``track_eras`` — a window-iterator with the SAME control flow, used
    as the parity twin in tests (``assert track_eras == era_sim``),
  * ``EraAccountant`` — the incremental live state machine: feeds on
    closed native bars + window boundaries, persists its position
    across restarts, and reports realised/unrealised era EV including
    the real 24 bps round-trip friction.

Nothing here is allowed to relax friction or skip an era: every side
run produces exactly one era record (or a documented invalid-scale
skip, mirroring the benchmark).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable

import numpy as np

from .hold_policy import era_geometry, friction_bps, managed


def track_eras(sides, starts, i_entry, i_exit, entry, scale,
               t, h, l, c, k, m, fr_bps, gate_l_min=0) -> list[dict]:
    """Batch era records with era_sim's exact control flow.

    Mirrors ``backtest.hold.era_sim`` (kept byte-for-byte in spirit —
    the certified code stays untouched; this twin is asserted equal in
    ``tests/test_hold_live.py``).  ``gate_l_min`` is the §20 TSMOM
    gate: at each run start, skip the run when
    ``side · (entry[w]/entry[w−lb_w] − 1) < 0``; warmup → take.
    """
    n = len(starts)
    med = float(np.nanmedian(scale)) if np.isfinite(scale).any() else 0.0
    scale_f = np.where(np.isfinite(scale) & (scale > 0), scale, med)
    fr = fr_bps * 1e-4
    lb_w = 0
    if gate_l_min and n > 1:
        w_sec = int(starts[1]) - int(starts[0])
        if w_sec > 0:
            lb_w = max(1, int(round(gate_l_min * 60.0 / w_sec)))
    recs: list[dict] = []
    w = 0
    while w < n:
        s = int(sides[w])
        if s == 0:
            w += 1
            continue
        w2 = w + 1
        while w2 < n and int(sides[w2]) == s:
            w2 += 1
        if lb_w > 0 and w >= lb_w:
            r = float(entry[w]) / float(entry[w - lb_w]) - 1.0
            if s * r < 0.0:
                w = w2
                continue
        e = float(entry[w])
        sl = m * float(scale_f[w])
        if not np.isfinite(sl) or sl <= 0 or e <= 0:
            w = w2
            continue
        tp = k * sl
        tp_px = e + tp * s
        sl_px = e - sl * s
        a = int(i_entry[w]) + 1
        b_last = int(i_exit[w2 - 1])
        b_flip = int(i_entry[w2]) if w2 < n else b_last
        b = min(b_flip, b_last)
        outcome = 0
        exit_px = None
        if m >= 999.0:                    # no-barrier ride to the flip
            outcome = 2
            exit_px = float(entry[w2]) if w2 < n else float(c[b_last])
            pnl_g = (exit_px / e - 1.0) * s
            recs.append({"start_i": w, "n_windows": w2 - w,
                         "outcome": outcome, "pnl_gross": pnl_g,
                         "pnl_net": pnl_g - 2.0 * fr})
            w = w2
            continue
        for j in range(a, b + 1):
            hi, lo = h[j], l[j]
            if s > 0:
                sl_hit = lo <= sl_px
                tp_hit = hi >= tp_px
            else:
                sl_hit = hi >= sl_px
                tp_hit = lo <= tp_px
            if sl_hit:                     # SL wins same-bar ties
                outcome, exit_px = -1, sl_px
                break
            if tp_hit:
                outcome, exit_px = 1, tp_px
                break
        if outcome == 0:
            exit_px = float(entry[w2]) if w2 < n else float(c[b_last])
            outcome = 2                    # 2 = flip exit
        pnl_g = (exit_px / e - 1.0) * s
        recs.append({"start_i": w, "n_windows": w2 - w,
                     "outcome": outcome, "pnl_gross": pnl_g,
                     "pnl_net": pnl_g - 2.0 * fr})
        w = w2
    return recs


# ───────────────────────────── live state ────────────────────────────
class EraAccountant:
    """Incremental era state for ONE (asset, tf) cell.

    Lifecycle per side run (mirrors era_sim):

      * first window of a run opens the position at that window's
        entry price with ``sl = m·scale``, ``tp = k·sl`` (or rides
        when ``m >= 999``),
      * every closed native bar while inside checks SL first, then TP
        (SL wins same-window ties) — a touch closes the position and
        the run stays FLAT until the side flips,
      * a side flip closes an open position at the flip-boundary price
        and opens the new run,
      * §20 TSMOM gate: when the engine reports ``gate_ok=False`` for a
        NEW run (side trades against the slow trend), the run never
        opens — a previously open era still closes at the flip-boundary
        price (batch parity), and the skip persists for the whole run,
      * each closed era pays ``2 × friction_bps`` (real round trip).

    State persists to ``data/hold_era_state.json`` so a restart keeps
    the position (barrier prices are fixed at entry — a resume re-checks
    the first new bar's range and flags ``resumed``).
    """

    def __init__(self, symbol: str, tf: str,
                 path: str | None = None) -> None:
        self.symbol, self.tf = symbol, tf
        self.path = path or os.environ.get(
            "HOLD_ERA_PATH",
            os.path.join("data", f"hold_era_{symbol}_{tf}.json"))
        g = era_geometry(symbol, tf)
        self.k, self.m = g if g else (None, None)
        self.fr_bps = friction_bps(symbol)
        self.side: int | None = None
        self.skip_side: int | None = None   # §20: gate-failed run
        self.entry_px: float | None = None
        self.entry_ts: float | None = None
        self.scale_px: float | None = None
        self.tp_px: float | None = None
        self.sl_px: float | None = None
        self.n_windows: int = 0
        self.in_position: bool = False     # False = flat (touched) in run
        self.touched: bool = False
        self.resumed: bool = False
        self.run_id: int = 0
        self.history: list[dict] = []       # closed eras (capped 500)
        self.totals = {"eras": 0, "net": 0.0, "gross": 0.0,
                       "friction_paid": 0.0, "touches": 0,
                       "flip_exits": 0}
        self._load()

    # ── persistence ────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError, FileNotFoundError):
            return
        self.side = raw.get("side")
        self.entry_px = raw.get("entry_px")
        self.entry_ts = raw.get("entry_ts")
        self.scale_px = raw.get("scale_px")
        self.tp_px = raw.get("tp_px")
        self.sl_px = raw.get("sl_px")
        self.n_windows = int(raw.get("n_windows") or 0)
        self.in_position = bool(raw.get("in_position"))
        self.touched = bool(raw.get("touched"))
        self.run_id = int(raw.get("run_id") or 0)
        self.resumed = bool(raw.get("resumed"))
        self.skip_side = raw.get("skip_side")
        self.history = list(raw.get("history") or [])[-500:]
        tot = raw.get("totals") or {}
        for k in self.totals:
            if k in tot:
                self.totals[k] = tot[k]
        if self.side is not None:
            self.resumed = True            # position survived a restart

    def _save(self) -> None:
        try:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "version": 1, "symbol": self.symbol, "tf": self.tf,
                    "side": self.side, "entry_px": self.entry_px,
                    "entry_ts": self.entry_ts, "scale_px": self.scale_px,
                    "tp_px": self.tp_px, "sl_px": self.sl_px,
                    "n_windows": self.n_windows,
                    "in_position": self.in_position, "touched": self.touched,
                    "run_id": self.run_id, "resumed": self.resumed,
                    "skip_side": self.skip_side,
                    "history": self.history[-500:],
                    "totals": self.totals,
                }, fh)
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ── events ─────────────────────────────────────────────────────
    def on_window(self, side: int, entry_px: float, scale_px: float,
                  window_start: float,
                  gate_ok: bool = True) -> dict[str, Any]:
        """A new window opened with certified side ``side``.

        Handles run boundaries: side change closes an open position at
        this window's entry (flip exit) and opens the new run; same
        side extends the run.  Returns an event dict (for the UI log).

        §20 TSMOM gate (``gate_ok``): consulted only when a NEW run
        would start — a failing gate closes any open era at the flip
        boundary (batch parity) and skips the run entirely for as long
        as its side persists (the batch sim checks the gate at run
        start only, never mid-run).
        """
        side = int(side)
        ev: dict[str, Any] = {"event": "window", "window_start": window_start}
        if side == 0 or entry_px <= 0:
            return ev
        if side == self.skip_side:
            return ev                     # still inside a gate-skipped run
        new_run = self.side != side
        if new_run and not gate_ok:
            if self.in_position and self.entry_px:
                self._close(exit_px=entry_px, outcome=2,
                            at_ts=window_start)
                ev["event"] = "era_gated_close"
                ev["closed"] = self.history[-1] if self.history else None
            else:
                ev["event"] = "era_gated_skip"
            self.skip_side = side
            self.side = None
            self._save()
            return ev
        if new_run:
            self.skip_side = None
        if self.side is None:
            self._open(side, entry_px, scale_px, window_start)
            ev["event"] = "era_open"
        elif side != self.side:
            if self.in_position and self.entry_px:
                self._close(exit_px=entry_px, outcome=2,
                            at_ts=window_start)
                ev["closed"] = self.history[-1] if self.history else None
            elif not self.in_position and not self.touched:
                pass
            self._open(side, entry_px, scale_px, window_start)
            ev["event"] = "era_flip"
        else:
            self.n_windows += 1
        self._save()
        return ev

    def on_bars(self, bars: Iterable[tuple[float, float, float]]
                ) -> list[dict[str, Any]]:
        """Feed CLOSED native bars ``(ts, high, low)`` while a position
        is open — first-touch checks with SL-win ties (era_sim order)."""
        out: list[dict[str, Any]] = []
        if not self.in_position or not self.entry_px or not self.side:
            return out
        if self.m is not None and float(self.m) >= 999.0:
            return out                      # ride: no barriers at all
        s = int(self.side)
        for ts, hi, lo in bars:
            if self.entry_ts is not None and float(ts) <= float(self.entry_ts):
                continue
            sl_hit = (lo <= self.sl_px) if s > 0 else (hi >= self.sl_px)
            tp_hit = (hi >= self.tp_px) if s > 0 else (lo <= self.tp_px)
            if sl_hit:
                rec = self._close(exit_px=self.sl_px, outcome=-1,
                                  at_ts=float(ts))
                out.append(rec)
                break
            if tp_hit:
                rec = self._close(exit_px=self.tp_px, outcome=1,
                                  at_ts=float(ts))
                out.append(rec)
                break
        if out:
            self._save()
        return out

    # ── internals ──────────────────────────────────────────────────
    def _open(self, side: int, entry_px: float, scale_px: float,
              window_start: float) -> None:
        self.side = int(side)
        self.entry_px = float(entry_px)
        self.entry_ts = float(window_start)
        self.scale_px = float(scale_px) if scale_px else None
        self.n_windows = 1
        self.in_position = True
        self.touched = False
        self.run_id += 1
        self.resumed = False
        if self.m is not None and self.k is not None and \
                scale_px and float(scale_px) > 0:
            sl = float(self.m) * float(scale_px)
            tp = float(self.k) * sl
            self.tp_px = self.entry_px + tp * self.side
            self.sl_px = self.entry_px - sl * self.side
        else:
            self.tp_px = self.sl_px = None   # invalid scale → skip era

    def _close(self, *, exit_px: float, outcome: int,
               at_ts: float) -> dict[str, Any]:
        e = float(self.entry_px or 0.0)
        s = int(self.side or 0)
        gross = ((float(exit_px) / e - 1.0) * s) if e > 0 else 0.0
        fr = 2.0 * float(self.fr_bps) * 1e-4
        rec = {
            "symbol": self.symbol, "tf": self.tf, "run_id": self.run_id,
            "side": s, "entry": e, "exit": float(exit_px),
            "entry_ts": self.entry_ts, "closed_ts": float(at_ts),
            "n_windows": self.n_windows, "outcome": int(outcome),
            "pnl_gross": gross, "pnl_net": gross - fr,
            "friction_bps": 2.0 * float(self.fr_bps),
            "closed_at": time.time(),
        }
        self.history.append(rec)
        self.history = self.history[-500:]
        t = self.totals
        t["eras"] = int(t.get("eras", 0)) + 1
        t["gross"] = float(t.get("gross", 0.0)) + gross
        t["net"] = float(t.get("net", 0.0)) + rec["pnl_net"]
        t["friction_paid"] = float(t.get("friction_paid", 0.0)) + fr
        if outcome == 1:
            t["touches"] = int(t.get("touches", 0)) + 1
        elif outcome == -1:
            pass
        else:
            t["flip_exits"] = int(t.get("flip_exits", 0)) + 1
        # run consumed — flat until the next side flip
        self.in_position = False
        self.touched = outcome in (1, -1)
        return rec

    # ── read ───────────────────────────────────────────────────────
    def snapshot(self, price: float | None = None) -> dict[str, Any]:
        unreal = None
        if self.in_position and self.entry_px and self.side and price:
            unreal = ((float(price) / float(self.entry_px) - 1.0)
                      * int(self.side))
        hist = self.history
        n = len(hist)
        wins = sum(1 for r in hist if r.get("pnl_net", 0) > 0)
        return {
            "symbol": self.symbol, "tf": self.tf,
            "certified": managed(self.symbol, self.tf),
            "side": self.side, "in_position": self.in_position,
            "entry": self.entry_px, "entry_ts": self.entry_ts,
            "tp": self.tp_px, "sl": self.sl_px,
            "scale": self.scale_px, "k": self.k, "m": self.m,
            "ride": bool(self.m is not None and float(self.m) >= 999.0),
            "n_windows": self.n_windows, "run_id": self.run_id,
            "touched_this_run": self.touched, "resumed": self.resumed,
            "gate_skip_side": self.skip_side,
            "unrealised_net": unreal,
            "totals": dict(self.totals),
            "era_win_rate": round(wins / n, 4) if n else None,
            "last_era": hist[-1] if hist else None,
            "friction_rule": ("one round trip (2×12 bps, PAXG 2×18) per "
                              "closed era — never relaxed"),
        }


_ACCOUNTANTS: dict[tuple[str, str], EraAccountant] = {}


def accountant(symbol: str, tf: str) -> EraAccountant:
    key = (symbol, tf)
    if key not in _ACCOUNTANTS:
        _ACCOUNTANTS[key] = EraAccountant(symbol, tf)
    return _ACCOUNTANTS[key]


def reset_all() -> None:
    """Tests: drop in-memory accountants (files honour HOLD_ERA_PATH)."""
    _ACCOUNTANTS.clear()
