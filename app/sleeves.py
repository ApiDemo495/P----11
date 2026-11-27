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
    if not times or len(times) != len(closes) or bucket <= 0 or bucket % 60:
        return [], []
    out_t: list = []
    out_c: list = []
    cur = None
    rows: list = []
    previous = None
    for t, c in zip(times, closes):
        # The input contract is a strictly ordered UTC one-minute tape.
        # A duplicate or backwards timestamp makes the series unreliable.
        if not math.isfinite(t) or t % 60 or (previous is not None and t <= previous):
            return [], []
        previous = t
        slot = t - (t % bucket)
        if cur is not None and slot != cur:
            expected = list(range(int(cur), int(cur + bucket), 60))
            if ([r[0] for r in rows] == expected and
                    all(math.isfinite(r[1]) and r[1] > 0 for r in rows)):
                out_t.append(cur)
                out_c.append(rows[-1][1])
            rows = []
        cur = slot
        rows.append((t, c))
    # Always exclude the last bucket; it may still be forming.
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
    # resample_bars already excludes the forming bucket. Do not discard a
    # second, completed bar: that shifts the certified trigger by 15 minutes.
    required = max(std_window, impulse_hours * max(1, 3600 // bucket)) + 1
    if len(bc) < required:
        return None
    recent = bt[-required:]
    if (bt[-1] + bucket != times[-1] - times[-1] % bucket or
            any(b - a != bucket for a, b in zip(recent, recent[1:]))):
        return None  # missing minutes/buckets cannot become a multi-bar return
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


def parse_frozen(frozen: str) -> dict:
    """Parse the registry's ``frozen_candidate`` string into parameters.

    ``"spillover_spot_15m|ETH|hold_h=48|lead=BNB|lookback_h=24|thr=2.0"``
    -> ``{"hold_h": 48, "lead": "BNB", "lookback_h": 24, "thr": 2.0}``

    The registry string is the frozen modal pick from the purged
    walk-forward, i.e. the exact parameters that passed the gate. The live
    engine must trade THOSE, never a module-level default: until this
    existed every sleeve was evaluated with catchup's settings
    (lead=BTC, lookback 12 h, thr 2.5), so the spillover sleeve -- which
    was certified as a BNB-led 24 h / 2.0 trigger -- was running a
    different strategy live than the one that passed the gate.
    """
    out: dict = {}
    for part in str(frozen or "").split("|"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def impulse_z_for(
    feed,
    asset: str,
    lead: str = "BTC",
    impulse_hours: int = IMPULSE_HOURS,
    threshold: float = Z_THRESHOLD,
    std_window: int = STD_WINDOW_BARS,
    bucket: int = SLEEVE_BAR_SECONDS,
) -> tuple[float | None, str]:
    """Certified trigger ``z`` for ONE sleeve's frozen specification.

    Returns ``(z, state)`` where state is long/short/flat, or
    ``(None, "warming_up")`` when the feed does not yet hold enough
    history to compute the trigger honestly.
    """
    try:
        st = feed.state(lead)
    except Exception:
        return None, "warming_up"
    # A certified live signal must not be inferred from simulated, unknown,
    # or mixed-provenance candles during feed handover.
    from .feed import _family
    sources = getattr(st, "sources", [])
    if (getattr(feed, "mode", None) != "binance" or
            len(sources) != len(st.times) or
            any(_family(source) != "binance" for source in sources)):
        return None, "warming_up"
    z = compute_impulse_z(st.times, st.closes, bucket=bucket,
                          impulse_hours=impulse_hours, std_window=std_window)
    if z is None:
        return None, "warming_up"
    if z > threshold:
        return z, "long"
    if z < -threshold:
        return z, "short"
    return z, "flat"


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
        # trade the FROZEN specification that passed the gate
        spec = parse_frozen((s.get("details") or {}).get("frozen_candidate") or "")
        lead = str(spec.get("lead") or "BTC")
        impulse_h = int(spec.get("lookback_h") or IMPULSE_HOURS)
        thr = float(spec.get("thr") or Z_THRESHOLD)
        hold_h = int(spec.get("hold_h") or HOLD_HOURS)
        base = dict(
            sleeve_id=s.get("sleeve_id", ""),
            asset=asset,
            status=s.get("status", ""),
            net_ev_bps=float(s.get("net_ev_bps", 0.0)),
            profit_factor=float(s.get("profit_factor", 0.0)),
            t_effective=float(s.get("t_effective", 0.0)),
            n=int(s.get("n", 0)),
            threshold=thr,
            hold_hours=hold_h,
            as_of=now,
        )
        try:
            feed.state(asset)                      # target must exist too
        except Exception:
            signals.append(SleeveSignal(state="warming_up", note="feed unavailable", **base))
            continue
        z, state = impulse_z_for(feed, asset, lead=lead, impulse_hours=impulse_h,
                                 threshold=thr)
        if z is None:
            need = (STD_WINDOW_BARS + impulse_h + 2) * SLEEVE_BAR_SECONDS // 3600
            try:
                have = len(feed.state(lead).closes)
            except Exception:
                have = 0
            if getattr(feed, "mode", None) != "binance":
                note = "live signal unavailable: feed is not Binance; simulated tape is excluded"
            else:
                note = (f"needs ~{need}h of continuous, Binance-sourced "
                        f"{SLEEVE_BAR_SECONDS // 60}m bars on the {lead} lead "
                        f"(have {have} 1m bars); gaps or mixed sources are excluded")
            signals.append(SleeveSignal(state="warming_up", z=None, note=note, **base))
            continue
        signals.append(SleeveSignal(
            state=state, z=z,
            note=f"lead {lead} · {impulse_h}h impulse · |z| > {thr:g} · "
                 f"hold {hold_h}h",
            **base,
        ))

    # per-asset roll-up for the hero card: prefer an OPEN sleeve, then the
    # best SHADOW observation (clearly labelled), then "none certified".
    by_asset: dict[str, dict] = {}
    for sig in signals:
        by_asset[sig.asset] = sig.as_dict()
    for s in shadow_sleeves:
        asset = s.get("asset", "")
        if asset in by_asset:
            continue
        # A SHADOW SLEEVE STILL HAS A LIVE READING (user report 2026-11-27:
        # "why asset triggers are unproven and shadow or none"). The card
        # printed the word UNPROVEN and a bare row of statistics with no
        # indication of what the trigger is doing NOW, so "unproven" read as
        # "the app is not running this" rather than "the gate rejected this
        # candidate". Compute the same frozen-spec z the open sleeves use —
        # it is a measurement either way — and say on the row exactly which
        # gate the candidate failed.
        z = None
        state = "unproven"
        _spec = parse_frozen((s.get("details") or {}).get("frozen_candidate") or "")
        _lead = str(_spec.get("lead") or "BTC")
        try:
            z, state = impulse_z_for(
                feed, asset, lead=_lead,
                impulse_hours=int(_spec.get("lookback_h") or IMPULSE_HOURS),
                threshold=float(_spec.get("thr") or Z_THRESHOLD))
        except Exception:
            z, state = None, "unproven"
        # A SHADOW SLEEVE MAY NOT EMIT A TRADABLE STATE. `state` stays
        # "unproven" — that is the contract every consumer and test reads —
        # and the live reading travels in its own field (`trigger`) so the
        # card can show what the trigger is doing WITHOUT any surface being
        # able to mistake it for a position.
        trigger = (str(state) if state in ("long", "short", "flat")
                   else "warming_up")
        state = "unproven"
        reasons = [str(r) for r in (s.get("reasons") or [])]
        by_asset[asset] = {
            "sleeve_id": s.get("sleeve_id", ""),
            "asset": asset,
            "status": "shadow",
            "state": state,
            "z": (round(float(z), 3) if z is not None else None),
            "trigger": trigger,
            "threshold": float(_spec.get("thr") or Z_THRESHOLD),
            "hold_hours": int(_spec.get("hold_h") or HOLD_HOURS),
            "as_of": now,
            "net_ev_bps": round(float(s.get("net_ev_bps", 0.0)), 2),
            "profit_factor": round(float(s.get("profit_factor", 0.0)), 3),
            "t_effective": round(float(s.get("t_effective", 0.0)), 3),
            "n": int(s.get("n", 0)),
            "note": ("the gate rejected this candidate, so it is watched and "
                     "reported but never traded"
                     + (" · the live trigger reading needs a real Binance "
                        "tape (simulated bars are excluded)"
                        if z is None else
                        " · the live trigger reading is shown below")),
            "gate_note": ("; ".join(reasons) if reasons
                          else "no gate reason recorded"),
            "reasons": reasons,
        }

    return {
        "generated_at": reg.get("generated_at"),
        "summary": reg.get("summary", {}),
        "signals": [s.as_dict() for s in signals],
        "assets": by_asset,
        "certified": sorted({s.asset for s in signals}),
        "note": reg.get("note", ""),
    }
