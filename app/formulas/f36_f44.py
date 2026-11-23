"""F36–F44 — the DIAGNOSIS-DRIVEN formulas.

Every formula here answers a defect found by measuring 3 years of real
tape (``backtest/diagnose_all.py`` and ``backtest/money_machine.py``),
not a new theoretical story.  The diagnosis was blunt:

  * F1–F35 measure *shape* (path identity, inversions, elasticity) and
    mostly have no directional information at any horizon;
  * the book flips side every window (149,038 changes in 3 years) and
    pays ~43,000 % of capital in cost — total ruin;
  * the target projection sits BELOW the round trip at tf1/tf2, so a
    window "win" cannot pay for itself.

So the new block is deliberately different in KIND:

  F36 TVF  trend vs fast — the one effect that survived (low turnover)
  F37 RPG  regime persistence — is the trend still intact, or late?
  F38 VRP  volatility risk premium proxy — do we get paid to hold now?
  F39 CTX  cost-to-edge ratio — is this whole window worth trading at all?
  F40 VLQ  liquidity vacuum — thin tape amplifies moves (fill realism)
  F41 FND  funding / carry tilt — who pays whom for holding
  F42 MDC  move-decay / exhaustion — has the impulse already spent itself
  F43 CRC  cross-asset confirmation — BTC/ETH must agree, or stand down
  F44 HDG  hold-duration advisor — how long the current state usually lasts

Contract shared with F1–F35: output in [-1, +1], never NaN/inf, and a
zero must have a reason (the trace layer records it).
"""
from __future__ import annotations

import math

EPS = 1e-12


def _px(ctx) -> list[float]:
    return list(getattr(ctx, "P", []) or [])


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x != x:
        return 0.0
    return lo if x < lo else (hi if x > hi else x)


def _sma(xs: list[float], n: int) -> float | None:
    if n <= 0 or len(xs) < n:
        return None
    seg = xs[-n:]
    return sum(seg) / n


def _ret(xs: list[float], n: int) -> float | None:
    if n <= 0 or len(xs) <= n or xs[-n - 1] <= 0:
        return None
    return xs[-1] / xs[-n - 1] - 1.0


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _diffs(xs: list[float]) -> list[float]:
    return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]


def _z_move(r: float, vol: float) -> float:
    """Vol-normalised move, with an explicit zero-vol case.

    A perfectly linear tape has zero diff-stdev, and dividing by it
    silently returns 0 — which reads as "no signal" exactly when the
    signal is strongest.  With no noise, a non-zero move IS a trend, so
    map it to the +/-1 rail instead of to zero.
    """
    if vol > 1e-12:
        return r / vol
    if r > 1e-12:
        return 3.0
    if r < -1e-12:
        return -3.0
    return 0.0


def _minute_series(ctx, n: int) -> list[float]:
    """Resample the context to ~n evenly spaced samples (cheap + honest)."""
    P = _px(ctx)
    if len(P) <= n or n <= 1:
        return P
    step = len(P) / float(n)
    return [P[min(len(P) - 1, int(i * step))] for i in range(n)]


# ─────────────────────────────────────────────────────────────────────
# F36 TVF — Trend vs Fast  (the only effect that survived 3 years)
# ─────────────────────────────────────────────────────────────────────
def f36_tvf(ctx) -> float:
    """F36 TVF — multi-horizon trend agreement, normalised.

    Three horizons (short/medium/long) each vote on direction; the output
    is the *agreement-weighted* mean.  A trend that points the same way
    on all three horizons is worth far more than one that fights itself,
    and that distinction is what the F1–F35 block was missing.

    Zero when there is not enough data for the long leg.
    """
    P = _px(ctx)
    if len(P) < 60:
        return 0.0
    W = max(20, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60)))
    legs = (max(3, W // 8), max(6, W // 4), max(12, W // 2))
    votes: list[float] = []
    weights: list[float] = []
    for i, n in enumerate(legs):
        r = _ret(P, n)
        if r is None:
            continue
        # scale each leg by its own volatility so legs are comparable
        seg = _diffs(P[-n - 1:])
        vol = _stdev(seg)
        unit = (vol / max(P[-1], EPS))
        z = _z_move(r, unit) if unit > 0 else _z_move(r, 0.0)
        votes.append(math.tanh(z / 2.0))
        weights.append(1.0 + 0.5 * i)          # longer legs weigh more
    if not votes:
        return 0.0
    tot = sum(weights)
    mean = sum(v * w for v, w in zip(votes, weights)) / tot
    # agreement: 1 when all legs point the same way, 0 when split
    agree = abs(sum(votes)) / len(votes)
    return _clamp(mean * (0.5 + 0.5 * agree))


# ─────────────────────────────────────────────────────────────────────
# F37 RPG — Regime Persistence Gap
# ─────────────────────────────────────────────────────────────────────
def f37_rpg(ctx) -> float:
    """F37 RPG — how much of the current move is already spent.

    Enters LATE when the impulse is nearly exhausted.  Compares the move
    over the recent third of the window against the move over the whole
    window: if the recent leg is most of the total, the trend is
    exhausted and the formula points the other way (mean reversion at the
    end of a run); if it is early, the trend has room and it agrees.
    """
    P = _px(ctx)
    if len(P) < 24:
        return 0.0
    W = max(12, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60)))
    full = _ret(P, min(W, len(P) - 1))
    third = max(2, W // 3)
    recent = _ret(P, min(third, len(P) - 1))
    if full is None or recent is None:
        return 0.0
    seg = _diffs(P[-(min(W, len(P) - 1)):])
    vol = _stdev(seg) / max(P[-1], EPS)
    zf = _z_move(full, vol)
    zr = _z_move(recent, vol)
    if abs(zf) < 1e-9:
        return 0.0
    spent = zr / zf                     # 1 = all of it was recent, 0 = early
    spent = max(-2.0, min(2.0, spent))
    # early (spent<0.5) -> agree with direction; late (>1) -> fade it
    tilt = math.tanh(zf / 2.0) * (0.5 - 0.5 * spent)
    return _clamp(tilt)


# ─────────────────────────────────────────────────────────────────────
# F38 VRP — Volatility Risk Premium proxy
# ─────────────────────────────────────────────────────────────────────
def f38_vrp(ctx) -> float:
    """F38 VRP — are we being paid to carry risk right now?

    Compares realised volatility over a short window with a longer
    baseline.  Cheap vol (short << long) is when trend-following earns
    its keep; expensive vol means stand down.  The output scales a
    *long* bias when vol is cheap and turns defensive when it is rich —
    it is a sizing signal, not a direction call, so its sign is the
    sign of the current drift weighted by the vol ratio.
    """
    P = _px(ctx)
    if len(P) < 40:
        return 0.0
    W = max(30, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60)))
    short_n = max(5, W // 4)
    long_n = max(20, W)
    s = _stdev(_diffs(P[-short_n - 1:]))
    l = _stdev(_diffs(P[-long_n - 1:]))
    if l <= 1e-12:
        # no volatility at all: the tape is a straight line, so carry is
        # free and the only thing to do is follow it
        drift = _ret(P, min(short_n * 2, len(P) - 1)) or 0.0
        return _clamp(1.0 if drift > 0 else (-1.0 if drift < 0 else 0.0))
    ratio = s / l
    # ratio < 1 -> calm; > 1 -> stressed. Map to [-1, +1] around 1.
    calm = math.tanh((1.0 - ratio) * 2.0)
    drift = _ret(P, min(short_n * 2, len(P) - 1)) or 0.0
    direction = 1.0 if drift > 0 else (-1.0 if drift < 0 else 0.0)
    return _clamp(calm * direction * 0.7 + calm * 0.3)


# ─────────────────────────────────────────────────────────────────────
# F39 CTX — Cost-To-Edge Ratio  (the one that stops the bleeding)
# ─────────────────────────────────────────────────────────────────────
def f39_ctx(ctx) -> float:
    """F39 CTX — is this window worth trading at all?

    THE central defect this project found: the target projection is
    often SMALLER than the round trip, so a "win" cannot pay for itself.
    This formula makes that condition observable: it returns the signed
    edge-to-cost ratio, clamped — positive when the recent move is large
    relative to the app's own measured round trip, strongly negative when
    the window is pure noise around the cost line.

    It is intentionally capable of returning ~0 for a live window: zero
    here means "cost eats the move", which the trace layer records as a
    real reason rather than a data gap.
    """
    P = _px(ctx)
    if len(P) < 12:
        return 0.0
    w = max(6, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60)))
    r = _ret(P, w)
    if r is None:
        return 0.0
    # The edge actually on offer is the realised move; the question is
    # whether it clears the round trip.  (Using *expected* vol here was a
    # bug: a clean trend has ZERO diff-variance, so the ratio collapsed to
    # zero exactly when the move was largest, inverting the sign.)
    move_bps = abs(r) * 1e4
    try:                                    # the app's own friction table
        from ..hold_policy import round_trip_bps as _rtb
        rt = float(_rtb(getattr(ctx, "symbol", "BTC") or "BTC"))
    except Exception:
        rt = 30.0
    if not (rt > 0):
        rt = 30.0
    ratio = move_bps / rt
    sgn = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
    pay = math.tanh(math.log(max(ratio, 1e-6)))     # 0 at ratio=1, +1 at 7x
    return _clamp(sgn * pay)


# ─────────────────────────────────────────────────────────────────────
# F40 VLQ — Liquidity Vacuum
# ─────────────────────────────────────────────────────────────────────
def f40_vlq(ctx) -> float:
    """F40 VLQ — thin tape amplifies moves, so a break is more real.

    Volume collapse + a directional move is the classic vacuum: price
    travels on little size and the move tends to continue.  Volume
    *without* a move is absorption and points the other way.
    """
    V = list(getattr(ctx, "V", []) or [])
    P = _px(ctx)
    if len(V) < 20 or len(P) < 20:
        return 0.0
    n = max(10, min(len(V) - 1, int(getattr(ctx, "W", 60) or 60) // 2))
    recent_v = sum(V[-n:]) / n
    base_v = sum(V[-(4 * n):]) / max(1, len(V[-(4 * n):]))
    if base_v <= 1e-12:
        return 0.0
    v_ratio = recent_v / base_v
    r = _ret(P, n) or 0.0
    seg = _diffs(P[-n - 1:])
    vol = _stdev(seg) / max(P[-1], EPS)
    move_z = math.tanh(_z_move(r, vol) / 2.0)
    vacuum = math.tanh((1.0 - v_ratio) * 1.5)      # +1 on thin tape
    return _clamp(move_z * (0.5 + 0.5 * vacuum))


# ─────────────────────────────────────────────────────────────────────
# F41 FND — Funding / Carry Tilt
# ─────────────────────────────────────────────────────────────────────
def f41_fnd(ctx) -> float:
    """F41 FND — who pays whom to hold this position.

    Perpetual funding is the closest thing to a free lunch in crypto:
    when funding is strongly positive, longs pay shorts, so the crowd is
    long and the carry favours the other side.  The signal is *contrarian
    on crowding, not on price* — a positive funding rate tilts the
    output negative.

    Zero (with a recorded reason) when no funding series is attached —
    the honest state on an offline tape.
    """
    f = getattr(ctx, "funding", None)
    if f is None:
        return 0.0
    try:
        vals = list(f)[-8:]
    except TypeError:
        return 0.0
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    # 0.01 % per 8h is a typical neutral rate; 0.1 % is extreme crowding
    return _clamp(-math.tanh(m / 0.0005))


# ─────────────────────────────────────────────────────────────────────
# F42 MDC — Move-Decay / Exhaustion
# ─────────────────────────────────────────────────────────────────────
def f42_mdc(ctx) -> float:
    """F42 MDC — has the impulse already spent itself?

    Fits the decay of per-bar absolute returns over the window.  A move
    that is *accelerating* (recent bars bigger than earlier bars) has
    continuation odds; a move that is fading has mean-reversion odds.
    Output follows the direction that the decay pattern favours.
    """
    P = _px(ctx)
    if len(P) < 24:
        return 0.0
    n = max(12, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60)))
    seg = P[-(n + 1):]
    d = _diffs(seg)
    if len(d) < 12:
        return 0.0
    half = len(d) // 2
    early = sum(abs(x) for x in d[:half]) / half
    late = sum(abs(x) for x in d[half:]) / max(1, len(d) - half)
    if early <= 1e-12:
        # zero early movement: any late movement is pure acceleration
        accel = 3.0 if late > 1e-12 else 0.0
    else:
        accel = (late / early) - 1.0
    direction = 1.0 if (seg[-1] - seg[0]) > 0 else (
        -1.0 if (seg[-1] - seg[0]) < 0 else 0.0)
    # accelerating -> follow; decelerating -> fade
    strength = math.tanh(accel)
    if accel < 0:
        return _clamp(-direction * math.tanh(-accel) * 0.8)
    return _clamp(direction * strength * 0.8)


# ─────────────────────────────────────────────────────────────────────
# F43 CRC — Cross-Asset Confirmation
# ─────────────────────────────────────────────────────────────────────
def f43_crc(ctx) -> float:
    """F43 CRC — BTC and ETH must agree, or stand down.

    Crypto majors are one trade with two tickers; a signal on one that
    the other contradicts is usually noise.  Reads ``ctx.others`` (the
    aligned series the engine already builds for F19) and returns the
    agreement-weighted direction of the *other* asset, which acts as a
    confirmation/denial on this one.
    """
    others = getattr(ctx, "others", None) or {}
    P = _px(ctx)
    if not others or len(P) < 12:
        return 0.0
    w = max(6, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60)))
    mine = _ret(P, w)
    if mine is None:
        return 0.0
    agree = 0.0
    n = 0
    for _sym, pair in others.items():
        try:
            ts, ps = pair
            if not ps or len(ps) < w + 1:
                continue
            theirs = ps[-1] / ps[-w - 1] - 1.0
        except Exception:
            continue
        agree += 1.0 if (mine > 0) == (theirs > 0) else -1.0
        n += 1
    if n == 0:
        return 0.0
    conf = agree / n                       # +1 all agree, -1 all disagree
    sgn = 1.0 if mine > 0 else (-1.0 if mine < 0 else 0.0)
    return _clamp(sgn * conf)


# ─────────────────────────────────────────────────────────────────────
# F44 HDG — Hold-Duration Advisor
# ─────────────────────────────────────────────────────────────────────
def f44_hdg(ctx) -> float:
    """F44 HDG — how long does the current state usually last?

    Measures how quickly the tape is changing character (sign flips of
    the per-bar move, normalised).  A choppy tape needs short holds; a
    persistent one supports a long hold.  The output is the *persistence*
    in [-1, +1]: +1 = strongly trending (hold), -1 = chop (exit fast).

    This is the direct answer to the app's core defect — it forces a
    holding period decision instead of flipping every window.
    """
    P = _px(ctx)
    if len(P) < 20:
        return 0.0
    n = max(15, min(len(P) - 1, int(getattr(ctx, "W", 60) or 60) * 2))
    d = _diffs(P[-(n + 1):])
    d = [x for x in d if abs(x) > 1e-12]
    if len(d) < 10:
        return 0.0
    flips = sum(1 for i in range(len(d) - 1)
                if (d[i] > 0) != (d[i + 1] > 0))
    flip_rate = flips / (len(d) - 1)       # 0.5 = coin, 0 = one-way
    # persistence: 1 - 2*flip_rate  ->  +1 one-way, 0 coin, -1 alternating
    pers = 1.0 - 2.0 * flip_rate
    # efficiency: net move vs total path travelled
    net = abs(P[-1] - P[-(n + 1)])
    path = sum(abs(x) for x in d)
    eff = (net / path) if path > 1e-12 else 0.0
    return _clamp(0.5 * pers + 0.5 * math.tanh((eff - 0.2) * 2.0))
