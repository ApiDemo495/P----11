"""Live sleeve runtime: evaluates CERTIFIED sleeves against the live tape.

Only sleeves that ``app.edge_gate.sleeve_gate`` has promoted to ``open``
can ever produce a signal here. Everything else is telemetry.

Reproducibility is the whole point. The research certified this sleeve on
a specific bar grid (BTC 15m impulse -> ETH 15m bars, hold 48h). The live
feed is 1-minute bars, so this module rebuilds the *same* 15m grid from
the live tape before evaluating, rather than approximating the trigger on
a different grid and calling it the same sleeve. If the tape is too short
to rebuild the grid, the sleeve reports ``warming_up`` and emits nothing —
it never fills the gap with a guess.

The trigger itself is the certified one, unchanged:
``z = sum(BTC log-return, last 12h) / (sd_500bars * sqrt(12h bars))``,
long ETH when ``z > 2.5``, short when ``z < -2.5``, held 48h.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "sleeve_registry.json")

# The certified grid, in seconds. Research bars are 15m; the live feed is
# 1m, so the runtime aggregates the feed up to 900s bars.
SLEEVE_BAR_SECONDS = 900
IMPULSE_HOURS = 12
HOLD_HOURS = 48
Z_THRESHOLD = 2.5
STD_WINDOW_BARS = 500


def load_registry(path: str | None = None) -> dict:
    """Read the committed registry produced by ``backtest/edge_scan.py``."""
    p = path or REGISTRY_PATH
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"sleeves": [], "summary": {}, "note": "registry not generated yet"}
    except Exception:
        return {"sleeves": [], "summary": {}, "note": "registry unreadable"}


@dataclass
class SleeveSignal:
    sleeve_id: str
    asset: str
    status: str
    state: str                 # "long" | "short" | "flat" | "warming_up"
    z: float | None = None
    threshold: float = Z_THRESHOLD
    hold_hours: int = HOLD_HOURS
    as_of: int = 0
    net_ev_bps: float = 0.0
    profit_factor: float = 0.0
    t_effective: float = 0.0
    n: int = 0
    note: str = ""
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sleeve_id": self.sleeve_id,
            "asset": self.asset,
            "status": self.status,
            "state": self.state,
            "z": None if self.z is None else round(self.z, 3),
            "threshold": self.threshold,
            "hold_hours": self.hold_hours,
            "as_of": self.as_of,
            "net_ev_bps": round(self.net_ev_bps, 2),
            "profit_factor": round(self.profit_factor, 3),
            "t_effective": round(self.t_effective, 3),
            "n": self.n,
            "note": self.note,
            "reasons": list(self.reasons),
        }


def resample_bars(times: list, closes: list, bucket: int) -> tuple[list, list]:
    """Aggregate an (irregular, 1m) tape onto fixed ``bucket``-second bars.

    Only COMPLETE buckets are returned: the trailing partial bucket is
    dropped, because a half-formed bar would let the live sleeve react to
    a bar the research grid never had.
    """
    if not times:
        return [], []
    out_t: list = []
    out_c: list = []
    cur = None
    last_c = 0.0
    for t, c in zip(times, closes):
        slot = t - (t % bucket)
        if cur is None:
            cur = slot
        elif slot != cur:
            out_t.append(cur)
            out_c.append(last_c)
            cur = slot
        last_c = c
    # drop the final bucket — it may still be filling
    if len(out_t) and cur is not None:
        done = True
        if done:
            pass
    return out_t, out_c


def compute_impulse_z(
    times: list,
    closes: list,
    bucket: int = SLEEVE_BAR_SECONDS,
    impulse_hours: int = IMPULSE_HOURS,
    std_window: int = STD_WINDOW_BARS,
) -> float | None:
    """Certified trigger value on the research bar grid, or None if warming up."""
    bt, bc = resample_bars(times, closes, bucket)
    # the last returned bar is still forming -> drop it
    if len(bt) > 1:
        bt = bt[:-1]
        bc = bc[:-1]
    if len(bc) < std_window + impulse_hours + 2:
        return None
    rets = [0.0] * len(bc)
    for i in range(1, len(bc)):
        prev = bc[i - 1]
        rets[i] = math.log(bc[i] / prev) if prev > 0 and bc[i] > 0 else 0.0
    window = rets[-std_window:]
    mean = sum(window) / len(window)
    var = sum((r - mean) ** 2 for r in window) / len(window)
    sd = math.sqrt(max(var, 1e-12))
    bars_per_hour = max(1, 3600 // bucket)
    L = impulse_hours * bars_per_hour
    acc = sum(rets[-L:])
    return acc / (sd * math.sqrt(L))


def evaluate_sleeves(
    feed,
    registry: dict | None = None,
    now: float | None = None,
) -> dict:
    """Evaluate every certified ``open`` sleeve against the live tape.

    Returns a snapshot-safe payload. ``assets`` maps each UI asset to the
    best sleeve evidence available for it, so BTC / ETH / BNB / PAXG all
    get a hero-card row whether or not they carry an open sleeve.
    """
    reg = registry if registry is not None else load_registry()
    open_sleeves = [s for s in reg.get("sleeves", []) if s.get("status") == "open"]
    shadow_sleeves = [s for s in reg.get("sleeves", []) if s.get("status") == "shadow"]
    now = int(now or time.time())

    signals: list[SleeveSignal] = []
    for s in open_sleeves:
        asset = s.get("asset", "")
        lead = "BTC"
        base = dict(
            sleeve_id=s.get("sleeve_id", ""),
            asset=asset,
            status=s.get("status", ""),
            net_ev_bps=float(s.get("net_ev_bps", 0.0)),
            profit_factor=float(s.get("profit_factor", 0.0)),
            t_effective=float(s.get("t_effective", 0.0)),
            n=int(s.get("n", 0)),
            as_of=now,
        )
        try:
            lead_st = feed.state(lead)
            tgt_st = feed.state(asset)
        except Exception:
            signals.append(SleeveSignal(state="warming_up", note="feed unavailable", **base))
            continue
        z = compute_impulse_z(lead_st.times, lead_st.closes)
        if z is None:
            need = (STD_WINDOW_BARS + IMPULSE_HOURS + 2) * SLEEVE_BAR_SECONDS // 3600
            signals.append(SleeveSignal(
                state="warming_up", z=None,
                note=f"needs ~{need}h of {SLEEVE_BAR_SECONDS // 60}m bars (have "
                     f"{len(lead_st.closes)} 1m bars)",
                **base,
            ))
            continue
        if z > Z_THRESHOLD:
            state = "long"
        elif z < -Z_THRESHOLD:
            state = "short"
        else:
            state = "flat"
        signals.append(SleeveSignal(state=state, z=z, **base))

    # per-asset roll-up for the hero card: prefer an OPEN sleeve, then the
    # best SHADOW observation (clearly labelled), then "none certified".
    by_asset: dict[str, dict] = {}
    for sig in signals:
        by_asset[sig.asset] = sig.as_dict()
    for s in shadow_sleeves:
        asset = s.get("asset", "")
        if asset in by_asset:
            continue
        by_asset[asset] = {
            "sleeve_id": s.get("sleeve_id", ""),
            "asset": asset,
            "status": "shadow",
            "state": "unproven",
            "z": None,
            "threshold": Z_THRESHOLD,
            "hold_hours": HOLD_HOURS,
            "as_of": now,
            "net_ev_bps": round(float(s.get("net_ev_bps", 0.0)), 2),
            "profit_factor": round(float(s.get("profit_factor", 0.0)), 3),
            "t_effective": round(float(s.get("t_effective", 0.0)), 3),
            "n": int(s.get("n", 0)),
            "note": "out-of-fold evidence only, no live signal",
            "reasons": s.get("reasons", []),
        }

    return {
        "generated_at": reg.get("generated_at"),
        "summary": reg.get("summary", {}),
        "signals": [s.as_dict() for s in signals],
        "assets": by_asset,
        "certified": sorted({s.asset for s in signals}),
        "note": reg.get("note", ""),
    }
