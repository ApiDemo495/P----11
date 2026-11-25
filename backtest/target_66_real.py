"""Can TP:SL = 2:1 reach a 66 % win rate? Measured on real Binance 1h tape.

The claim to test, stated exactly: with the reward twice the risk, a trade's
geometric win rate is 1/(1+2) = 33.3 %. Asking for 66 % is asking for DOUBLE
the geometric rate, i.e. a directional edge that has to survive friction.

Design (no abstention, no look-ahead, no tuning on the test half):

  * tape      : Binance Vision 1h klines, real OHLC, no gaps (asserted)
  * geometry  : TP = 2.0 * s, SL = 1.0 * s, s = k * ATR(14) at entry
  * settle    : bar-by-bar first touch, SL wins same-bar ties, and a
                conservative intra-bar rule (see _walk) — never the mid
  * direction : the ONLY edge in the test is the signed entry signal under
                test; a random side is the control
  * friction  : the asset's real Binance round trip charged once per trade
  * split     : first 65 % of time = train, last 35 % = TEST (one look)

If no signal reaches 66 % on TEST, the honest answer is that the target is
not reachable at 2:1 by that class of signal — not that it is unreachable
in principle, since the market could always contain edge we cannot measure.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys

import numpy as np

DATA = "/tmp/cryptomirror/marketData/unzipped"
RT_BPS = {"BTCUSDT": 28.872, "ETHUSDT": 29.808, "BNBUSDT": 30.963,
          "PAXGUSDT": 37.400, "SOLUSDT": 31.5}
TRAIN_FRAC = 0.65


def load(pair: str) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(os.path.join(DATA, pair, pair + "-1h-*", "*.csv")))
    rows = []
    for path in files:
        with open(path) as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for r in rd:
                if len(r) >= 6:
                    rows.append((int(r[0]) // 1000, float(r[1]), float(r[2]),
                                 float(r[3]), float(r[4]), float(r[5])))
    rows.sort(key=lambda x: x[0])
    t = np.array([r[0] for r in rows], dtype=np.int64)
    gaps = int(np.sum(np.diff(t) != 3600))
    assert gaps == 0, f"{pair}: {gaps} gaps/dups"
    return {"t": t,
            "o": np.array([r[1] for r in rows]),
            "h": np.array([r[2] for r in rows]),
            "l": np.array([r[3] for r in rows]),
            "c": np.array([r[4] for r in rows]),
            "v": np.array([r[5] for r in rows])}


def atr(b: dict[str, np.ndarray], n: int = 14) -> np.ndarray:
    h, l, c = b["h"], b["l"], b["c"]
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.full_like(tr, np.nan)
    csum = np.cumsum(np.insert(tr, 0, 0.0))
    out[n:] = (csum[n + 1:] - csum[1:-n]) / n
    return out


def _walk(o, h, l, c, i, side, tp, sl, use_low, use_high):
    """First-touch between the two barriers. Returns +1 / -1 / 0 (open).

    `use_low`/`use_high` select which side is allowed to trigger first when a
    single bar spans both barriers. The adversarial setting (the default used
    for the verdict) assumes the STOP was touched first whenever a bar spans
    both — the pessimistic reading.
    """
    n = len(c)
    for j in range(i + 1, n):
        hi, lo = h[j], l[j]
        if side > 0:
            hit_tp = hi >= tp
            hit_sl = lo <= sl
        else:
            hit_tp = lo <= tp
            hit_sl = hi >= sl
        if hit_tp and hit_sl:
            return -1 if use_low else 1        # pessimistic: stop first
        if hit_sl:
            return -1
        if hit_tp:
            return 1
    return 0


def walk_fast(o, h, l, c, entry_idx, side, tp, sl, max_bars=4000,
              pessimistic=True):
    """Vectorised-ish first touch over a bounded horizon."""
    idx = entry_idx + np.arange(1, max_bars + 1)
    idx = idx[idx < len(c)]
    if len(idx) == 0:
        return None
    if side > 0:
        tp_hit = h[idx] >= tp
        sl_hit = l[idx] <= sl
    else:
        tp_hit = l[idx] <= tp
        sl_hit = h[idx] >= sl
    both = tp_hit & sl_hit
    first_tp = np.argmax(tp_hit) if tp_hit.any() else -1
    first_sl = np.argmax(sl_hit) if sl_hit.any() else -1
    if first_tp < 0 and first_sl < 0:
        return 0
    if first_tp < 0:
        return -1
    if first_sl < 0:
        return 1
    if first_tp == first_sl:
        return -1 if pessimistic else 1     # same bar: stop wins
    return 1 if first_tp < first_sl else -1


# ────────────────────────────── signals ──────────────────────────────
# Every signal is CAUSAL: it reads only bars <= i.

def sig_trend(b, lb):
    c = b["c"]
    s = np.zeros(len(c), dtype=int)
    r = np.full(len(c), np.nan)
    r[lb:] = c[lb:] / c[:-lb] - 1.0
    s[lb:] = np.sign(r[lb:]).astype(int)
    return s


def sig_revert(b, lb, zwin=96):
    """Fade the extension: z-score of price vs its trailing mean."""
    c = b["c"]
    n = len(c)
    out = np.zeros(n, dtype=int)
    ma = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    cs = np.cumsum(np.insert(c, 0, 0.0))
    ma[zwin:] = (cs[zwin + 1:] - cs[1:-zwin]) / zwin
    for i in range(zwin, n):
        out[i] = 0
    cs2 = np.cumsum(np.insert(c * c, 0, 0.0))
    var = np.full(n, np.nan)
    var[zwin:] = (cs2[zwin + 1:] - cs2[1:-zwin]) / zwin - ma[zwin:] ** 2
    sd[zwin:] = np.sqrt(np.maximum(var[zwin:], 0.0))
    ok = np.isfinite(ma) & np.isfinite(sd) & (sd > 0)
    z = np.zeros(n)
    z[ok] = (c[ok] - ma[ok]) / sd[ok]
    out[ok] = (-np.sign(z[ok])).astype(int)
    # lb filters weak extensions
    zmin = 1.0
    weak = np.abs(z) < zmin
    out[weak] = 0
    out[:zwin] = 0
    return out


def sig_breakout(b, lb):
    """Donchian breakout of the last `lb` bars."""
    h, l, c = b["h"], b["l"], b["c"]
    n = len(c)
    out = np.zeros(n, dtype=int)
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    for i in range(lb, n):
        hi[i] = h[i - lb:i].max()
        lo[i] = l[i - lb:i].min()
    out = np.where(c > hi, 1, np.where(c < lo, -1, 0)).astype(int)
    out[:lb] = 0
    return out


def sig_mom_vol(b, lb):
    """Trend only when volatility is in the top tercile (regime filter)."""
    base = sig_trend(b, lb)
    a = atr(b, 14)
    n = len(a)
    pct = np.full(n, np.nan)
    for i in range(160, n):
        w = a[i - 120:i]
        w = w[np.isfinite(w)]
        if len(w) > 20 and np.isfinite(a[i]):
            pct[i] = float(np.mean(w <= a[i]))
    out = np.where(pct > 0.66, base, 0).astype(int)
    out[~np.isfinite(pct)] = 0
    return out


def test_signal(b, side_sig, k=5.0, horizon=400, pessimistic=True):
    """Walk every bar of the tape; return first-touch outcomes."""
    a = atr(b, 14)
    c, h, l = b["c"], b["h"], b["l"]
    n = len(c)
    wins = losses = 0
    warm = 200
    for i in range(warm, n - 2):
        s = int(side_sig[i])
        if s == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        dist = k * a[i]
        entry = c[i]
        tp = entry + s * 2.0 * dist
        sl = entry - s * 1.0 * dist
        r = walk_fast(None, h, l, c, i, s, tp, sl,
                      max_bars=horizon, pessimistic=pessimistic)
        if r is None or r == 0:
            continue
        if r > 0:
            wins += 1
        else:
            losses += 1
    return wins, losses


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    cen = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((cen - m) / d, (cen + m) / d)


SIGNALS = {
    "trend-24h": lambda b: sig_trend(b, 24),
    "trend-72h": lambda b: sig_trend(b, 72),
    "trend-168h": lambda b: sig_trend(b, 168),
    "revert-z96": lambda b: sig_revert(b, 0, 96),
    "revert-z240": lambda b: sig_revert(b, 0, 240),
    "breakout-48h": lambda b: sig_breakout(b, 48),
    "breakout-168h": lambda b: sig_breakout(b, 168),
    "momvol-72h": lambda b: sig_mom_vol(b, 72),
}


def main() -> None:
    pairs = [p for p in os.listdir(DATA)
             if os.path.isdir(os.path.join(DATA, p))]
    if len(sys.argv) > 1:
        pairs = sys.argv[1:]
    print(f"pairs: {pairs}\n")
    tapes = {}
    for p in pairs:
        b = load(p)
        tapes[p] = b
        print(f"{p}: {len(b['c'])} hourly bars  "
              f"{min(b['t'])}..{max(b['t'])}")
    print()
    for p, b in tapes.items():
        n = len(b["c"])
        cut = int(n * TRAIN_FRAC)
        print(f"═══ {p}  train=<{cut}  TEST={n - cut} bars")
        for name, fn in SIGNALS.items():
            sig = fn(b)
            # split by masking the signal outside each half
            tr = sig.copy(); tr[cut:] = 0
            te = sig.copy(); te[:cut] = 0
            w1, l1 = test_signal(b, tr)
            w2, l2 = test_signal(b, te)
            def fmt(w, l):
                if w + l == 0:
                    return "n=0"
                p_ = w / (w + l)
                lo, hi = wilson(w, w + l)
                return (f"{p_:6.1%} [{lo:.1%},{hi:.1%}] n={w+l:5d}")
            print(f"  {name:<13} train {fmt(w1,l1):28}  TEST {fmt(w2,l2)}")
        print()


if __name__ == "__main__":
    main()
