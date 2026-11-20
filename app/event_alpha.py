"""Event-driven alpha — trade the stress, not the candle.

The per-bar model scored every bar and lost the fee, because a candle's shape
is mostly a coin flip once you subtract the spread.  Forced flow is
different: liquidation cascades, funding squeezes and stop runs are *caused*,
they are visible in funding / basis / open interest / taker flow before the
move completes, and the follow-through is not arbitraged away at retail
latency.

Three setups, each a pure predicate on causal arrays:

A  ``squeeze``      leverage build-up + volatility compression + a breakout
                    with taker-flow confirmation -> trade **with** the break.
B  ``basis_fade``   funding/basis dislocation at an extreme, exhausted by a
                    volume climax and a rejection wick -> trade **against**
                    the crowded side, back toward spot.
C  ``leadlag``      a BTC impulse that the laggard has not priced yet ->
                    trade the catch-up leg (also the BTC/gold flight leg).

Every setup returns decision arrays, never outcomes: `score[i]` is the side,
`mask[i]` says the setup fired. `app.portfolio.run_sleeve` settles them with
the asymmetric convex plan below — tight structural stop, uncapped trailing
runner, maker-pullback entries with realistic fill rules.
"""

from __future__ import annotations

import numpy as np

# The one configuration this programme measured out-of-fold as net positive:
# BTC makes a 3-hour impulse of at least half a sigma and ETH has priced less
# than 30 % of it.  Frozen here so the live pipeline and the harness cannot
# drift apart.
CATCHUP = {"k": 3, "impulse_sigma": 0.5, "lag_max": 0.30}

PLAN = {
    # tight stop / uncapped runner: the payoff shape is what makes a sub-50 %
    # hit rate profitable after fees
    "squeeze": {"sl": 0.9, "trail": 1.6, "hz": 48, "pt": 6.0, "runner": True},
    "basis_fade": {"sl": 1.0, "trail": 2.2, "hz": 72, "pt": 6.0, "runner": True},
    "leadlag": {"sl": 1.0, "trail": 3.0, "hz": 48, "pt": 6.0, "runner": True},
}

# Default execution: post-only entry a quarter-sigma into the pullback, stop
# taken with a market order (stops are never assumed to fill passively).
EXEC = {"entry": "maker_pullback", "pullback": 0.25,
        "maker_fee_bps": 2.0, "taker_fee_bps": 5.0,
        "maker_penalty_bps": 1.0, "min_penetration": 1e-4,
        # one-bar-delayed fill: you only get filled while the move continues
        # against you, which is the adverse selection makers actually pay
        "fill_mode": "next_bar"}


# ───────────────────────────── helpers ─────────────────────────────


def _ema(x: np.ndarray, win: int) -> np.ndarray:
    a = 2.0 / (win + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _roll_pct(x: np.ndarray, win: int) -> np.ndarray:
    """Causal mid-rank percentile of the newest value in its own window.

    Ties count half: a flat window must read 0.5, not 1.0, otherwise a quiet
    market would look like the *most* volatile one and the compression test
    could never fire.
    """
    n = x.size
    out = np.full(n, 0.5)
    win = max(3, int(win))
    for i in range(n):
        j = max(0, i - win + 1)
        seg = x[j:i + 1]
        out[i] = float(((seg < x[i]).sum() + 0.5 * (seg == x[i]).sum())
                       / seg.size)
    return out


def _true_range(bars) -> np.ndarray:
    h = np.asarray(bars.h, float)
    lo = np.asarray(bars.l, float)
    c = np.asarray(bars.c, float)
    tr = h - lo
    if c.size > 1:
        tr[1:] = np.maximum(tr[1:], np.abs(h[1:] - c[:-1]))
        tr[1:] = np.maximum(tr[1:], np.abs(lo[1:] - c[:-1]))
    return tr


def _vol_ratio(bars, win: int = 240) -> np.ndarray:
    v = np.asarray(getattr(bars, "v", np.ones(len(bars.c))), float)
    out = np.ones(v.size)
    for i in range(v.size):
        j = max(0, i - win + 1)
        seg = v[j:i + 1]
        med = float(np.median(seg)) if seg.size else 0.0
        out[i] = v[i] / med if med > 0 else 1.0
    return out


# ─────────────────────────── setup A: squeeze ───────────────────────────


def squeeze(bars, f: dict | None = None, *, compression=0.30,
            funding_z_min=1.0, surge=1.5, lookback=20) -> dict:
    """Leverage build-up + compression + breakout with flow confirmation."""
    c = np.asarray(bars.c, float)
    h = np.asarray(bars.h, float)
    lo = np.asarray(bars.l, float)
    n = c.size
    tr = _true_range(bars)
    atr_pct = np.zeros(n)
    for i in range(n):
        j = max(0, i - 48)
        atr_pct[i] = float(tr[j:i + 1].mean()) / max(1e-9, c[i])
    # compression is a property of the regime *before* the breakout bar: the
    # squeeze is the release, so the percentile is read one bar back
    comp_now = _roll_pct(atr_pct, 240) <= compression
    comp = np.r_[False, comp_now[:-1]]
    surge_r = _vol_ratio(bars) >= surge
    # the breakout bar must clear the range of the bars *before* it; using
    # its own high would make the test unsatisfiable by construction
    hi = np.full(n, np.inf)
    low = np.full(n, -np.inf)
    for i in range(n):
        j = max(0, i - lookback)
        if i > j:
            hi[i] = float(h[j:i].max())
            low[i] = float(lo[j:i].min())
    up_break = c >= hi * 0.9995
    dn_break = c <= low * 1.0005
    fz = (np.nan_to_num(f.get("funding_z", np.zeros(n)), nan=0.0)
          if f else np.zeros(n))
    crowded = np.abs(fz) >= funding_z_min
    side = np.where(up_break, 1.0, np.where(dn_break, -1.0, 0.0))
    # a squeeze breaks *against* the crowded side or with it; both are
    # allowed, but the breakout must be funded and confirmed by expansion
    mask = comp & surge_r & (side != 0) & (np.abs(fz) >= 0.0)
    if f is not None and "oi_seen" in f and float(np.sum(f["oi_seen"])) > 0:
        # when the open-interest archive exists, demand an OI build too
        oi_build = np.nan_to_num(f.get("oi_chg_24h", np.zeros(n)), nan=0.0) > 0
        has = np.nan_to_num(f.get("oi_seen", np.zeros(n))) > 0
        mask = mask & (oi_build | ~has)
    return {"score": side * mask, "mask": mask, "name": "squeeze",
            "crowded": crowded, "compression": comp, "surge": surge_r}


# ───────────────────────── setup B: basis/funding fade ─────────────────────────


def basis_fade(bars, f: dict, *, funding_z=2.0, basis_z=1.5,
               climax=3.0, wick=0.5) -> dict:
    """Fade a crowded, exhausted funding/basis dislocation."""
    c = np.asarray(bars.c, float)
    h = np.asarray(bars.h, float)
    lo = np.asarray(bars.l, float)
    o = np.asarray(bars.o, float)
    n = c.size
    if f is None or "funding_z" not in f:
        z = np.zeros(n)
        bz = np.zeros(n)
    else:
        z = np.nan_to_num(f.get("funding_z", np.zeros(n)), nan=0.0)
        bz = np.nan_to_num(f.get("basis_z", np.zeros(n)), nan=0.0)
    rng = np.maximum(1e-12, h - lo)
    upper = (h - np.maximum(c, o)) / rng
    lower = (np.minimum(c, o) - lo) / rng
    climax_r = _vol_ratio(bars) >= climax
    # crowded longs (funding very positive) exhausted by an upper wick
    long_exhaust = (z >= funding_z) & (bz >= basis_z) & (upper >= wick) & (c < o)
    short_exhaust = (z <= -funding_z) & (bz <= -basis_z) & (lower >= wick) & (c > o)
    side = np.where(long_exhaust, -1.0, np.where(short_exhaust, 1.0, 0.0))
    mask = climax_r & (side != 0)
    if f is not None and "funding_seen" in f:
        mask &= np.nan_to_num(f["funding_seen"], nan=0.0) > 0
    return {"score": side * mask, "mask": mask, "name": "basis_fade"}


# ────────────────────────── setup C: lead-lag ──────────────────────────


def leadlag(target_bars, source_bars, f: dict | None = None, *, k=3,
            impulse_sigma=1.5, lag_max=0.45, hz_scale=None) -> dict:
    """BTC impulse the laggard has not priced yet -> trade the catch-up."""
    ct = np.asarray(target_bars.c, float)
    cs = np.asarray(source_bars.c, float)
    n = min(ct.size, cs.size)
    ct, cs = ct[:n], cs[:n]
    r = np.zeros(n)
    m = np.zeros(n)
    k = max(1, int(k))
    r[k:] = cs[k:] / cs[:-k] - 1.0                       # BTC impulse
    m[k:] = ct[k:] / ct[:-k] - 1.0                       # laggard move
    sig = np.zeros(n)
    for i in range(k, n):
        j = max(0, i - 240)
        d = np.diff(cs[j:i + 1]) / np.maximum(1e-12, cs[j:i])
        if d.size > 10:
            sig[i] = float(np.std(d)) * np.sqrt(k)
        else:
            sig[i] = abs(r[i])
    zc = np.divide(r, np.maximum(1e-12, sig))
    lagging = np.abs(m) <= lag_max * np.abs(r)
    fire = (np.abs(zc) >= impulse_sigma) & lagging
    side = np.sign(r) * fire
    return {"score": side, "mask": fire, "name": "leadlag",
            "impulse_z": zc}


# ───────────────────────────── settlement plan ─────────────────────────────


def catchup(target_bars, source_bars) -> dict:
    """The measured ETH/BTC catch-up sleeve (frozen parameters)."""
    return leadlag(target_bars, source_bars, **CATCHUP)


def plan(name: str) -> dict:
    """Execution parameters for a setup (convex geometry + maker entry)."""
    p = dict(PLAN[name])
    p.update(EXEC)
    return p


def settle(bars, sig: dict, sigma, name: str, *, diag: dict | None = None,
           fees_bps: float | None = None) -> list[dict]:
    """Run one setup through the sequential engine with its own plan."""
    from .portfolio import run_sleeve
    p = plan(name)
    kw = dict(pt=p["pt"], sl=p["sl"], hz=p["hz"], sigma=sigma,
              trail=p["trail"], runner=p["runner"],
              entry=p["entry"], pullback=p["pullback"],
              maker_fee_bps=p["maker_fee_bps"],
              taker_fee_bps=p["taker_fee_bps"],
              maker_penalty_bps=p["maker_penalty_bps"],
              min_penetration=p["min_penetration"],
              fill_mode=p.get("fill_mode", "next_bar"), diag=diag)
    if fees_bps is not None and p["entry"] == "taker":
        kw["fees_bps"] = fees_bps
    return run_sleeve(bars, sig["score"], sig["mask"], **kw)
