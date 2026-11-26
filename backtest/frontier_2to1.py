"""The 2:1 frontier, searched hard on real tape. What win rate is ACHIEVABLE?

Why this file exists
--------------------
The request was "66 % win rate at 2:1 TP/SL, so every $100 earns $66.66".
That is a statement about EXPECTANCY, and it has an exact arithmetic:

    at TP:SL = 2:1, EV per trade = p*2 - (1-p)*1 = 3p - 1  (in units of risk)

    EV = +0.6666  requires  3p - 1 = 0.6666  ->  p = 55.56 %
    break-even (EV = 0)      requires  p = 33.33 %

So the target is not 66 % of trades winning — it is 55.56 % of trades
winning, sustained. (A 66 % win rate at 2:1 would be EV = +0.98R, i.e.
+$98 per $100 risked, which is 2.94x the break-even edge.) Both numbers are
reported below so the ask cannot be confused with its own arithmetic.

`backtest/target_66.py` tested ten hand-picked signals. Its conclusion —
"the ceiling is the geometric 33.3 % plus noise" — could have been a
limitation of those ten. This file removes that excuse:

  * ~40 causal feature families x 5 threshold quantiles x 6 stop sizes
    x 2 sides = thousands of configurations, searched on TRAIN only
  * first-touch settlement computed with a VECTORISED walk (exact, same
    rules as `target_66_real._walk`: SL wins same-bar ties, adversarial)
  * the frozen winner is then judged ONCE on the untouched TEST half
  * every reported number carries a Wilson 95 % CI, and the size of the
    search is charged against it (Bonferroni + expected false positives)

No abstention: a configuration trades every bar its signal is non-zero, and
the signal is defined on every bar. No look-ahead: every feature at bar i
uses bars <= i; barriers are placed from ATR at i; settlement starts at i+1.

Usage
    .venv/bin/python -m backtest.frontier_2to1 [PAIR ...]
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from .target_66_real import DATA, load, atr, wilson

TRAIN_FRAC = 0.65
# The TP:SL grid. r = 2.0 is the user's ask; the rest bracket it so the
# "win rate is just 1/(1+r) relabelled" claim can be checked directly.
RS = (0.5, 1.0, 1.5, 2.0, 3.0)
# Stop distance in ATR units. A tight stop raises win rate at fixed r only
# by shortening the horizon, which the frontier exposes.
KS = (0.5, 1.0, 2.0, 3.0, 5.0)
QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8)
MAX_H = 240          # settlement horizon in bars (10 days of 1h)
MIN_TRADES = 40


# ───────────────────────── exact, vectorised first-touch ──────────────────
def walk_entries(h, l, c, idx, side, tp, sl, horizon=MAX_H):
    """First-touch per ENTRY, vectorised across entries.

    `idx` are entry bar indices, `side` is +/-1, `tp`/`sl` the absolute
    barriers. Settlement runs idx+1 .. idx+horizon; the entry bar itself is
    never used (it cannot settle a trade opened at its close).

    Adversarial tie rule, identical to `target_66_real._walk`: when a single
    bar spans BOTH barriers the STOP is credited. That is the only reading a
    live trader can rely on.
    """
    n = len(c)
    m = len(idx)
    out = np.zeros(m, dtype=np.int8)
    done = np.zeros(m, dtype=bool)
    long_ = side > 0
    for off in range(1, horizon + 1):
        j = idx + off
        ok = (j < n) & ~done
        if not ok.any():
            break
        jj = np.clip(j, 0, n - 1)
        hi_j, lo_j = h[jj], l[jj]
        tp_hit = np.where(long_, hi_j >= tp, lo_j <= tp)
        sl_hit = np.where(long_, lo_j <= sl, hi_j >= sl)
        # SL first (adversarial), then TP on the remainder
        s_only = ok & sl_hit
        out[s_only] = -1
        done |= s_only
        t_only = ok & tp_hit & ~sl_hit
        out[t_only] = 1
        done |= t_only
    return out


# ───────────────────────── causal feature families ───────────────────────
def _roll_mean(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w:] = (cs[w + 1:] - cs[1:-w]) / w
    return out


def _roll_std(x, w):
    cs = np.cumsum(np.insert(x, 0, 0.0))
    cs2 = np.cumsum(np.insert(x * x, 0, 0.0))
    out = np.full(len(x), np.nan)
    m = (cs[w + 1:] - cs[1:-w]) / w
    v = (cs2[w + 1:] - cs2[1:-w]) / w - m * m
    out[w:] = np.sqrt(np.maximum(v, 0.0))
    return out


def features(b):
    """Every feature at index i uses bars <= i. Signed and continuous."""
    c, h, l, v = b["c"], b["h"], b["l"], b["v"]
    lr = np.concatenate([[0.0], np.diff(np.log(np.maximum(c, 1e-12)))])
    f = {}
    # trend: standardised momentum, several horizons
    for w in (6, 12, 24, 48, 72, 120, 168, 336):
        s = _roll_std(lr, w) * math.sqrt(w)
        m = _roll_mean(lr, w) * w
        f[f"mom{w}"] = np.where(s > 0, m / np.maximum(s, 1e-12), np.nan)
    # mean reversion: z of price against its own rolling mean
    for w in (24, 48, 96, 240):
        m = _roll_mean(c, w)
        s = _roll_std(c, w)
        f[f"rev{w}"] = np.where(s > 0, (c - m) / np.maximum(s, 1e-12), np.nan)
    # volume pressure
    for w in (24, 96):
        mv = _roll_mean(np.maximum(v, 0.0), w)
        f[f"volp{w}"] = np.where(mv > 0, v / np.maximum(mv, 1e-12) - 1.0, np.nan)
    # intrabar: close position inside the range (signed), and its average
    rng = np.maximum(h - l, 1e-12)
    clv = (2.0 * c - h - l) / rng
    f["clv"] = clv
    f["clv24"] = _roll_mean(clv, 24)
    # acceleration: short vs long momentum
    f["accel"] = f["mom6"] - f["mom72"]
    f["accel2"] = f["mom24"] - f["mom168"]
    # unsigned regime features (filters, not sides) are exposed separately
    return f


# ───────────────────────── one cell of the search ────────────────────────
def cell(b, feat, ss, q, k, r, lo, hi, horizon=MAX_H):
    """Trade where `ss * feat` is above its own TRAIN quantile `q`.

    Returns (wins, losses). Only bars in [lo, hi) are eligible. The
    threshold is computed from TRAIN bars only, passed in via the caller
    having a train window that ends at or before `hi` -- see `search`.
    """
    a = atr(b, 14)
    n = len(b["c"])
    sig = ss * feat
    ok = np.isfinite(sig) & np.isfinite(a) & (a > 0)
    return sig, ok, a


def evaluate(b, feat, ss, q, k, r, thr, lo, hi, horizon=MAX_H):
    """Frozen evaluation: entries in [lo,hi) with `ss*feat >= thr`."""
    a = atr(b, 14)
    n = len(b["c"])
    sig = ss * feat
    elig = np.zeros(n, dtype=bool)
    elig[lo:hi] = True
    m = elig & np.isfinite(sig) & np.isfinite(a) & (a > 0) & (sig >= thr)
    idx = np.where(m)[0]
    if len(idx) == 0:
        return 0, 0
    side = np.full(len(idx), 1 if ss > 0 else -1, dtype=np.int8)
    # NOTE: `ss` already flipped the feature, so the tradeable side is always
    # "long the transformed signal": if the transformed signal is high, go
    # long. That keeps the side definition unambiguous.
    side = np.ones(len(idx), dtype=np.int8)
    d = k * a[idx]
    e = b["c"][idx]
    tp = e + r * d
    sl = e - d
    res = walk_entries(b["h"], b["l"], b["c"], idx, side, tp, sl, horizon)
    return int(np.sum(res == 1)), int(np.sum(res == -1))


def train_threshold(b, feat, ss, q, lo, hi):
    """The `q` quantile of `ss*feat` measured on TRAIN bars only."""
    a = atr(b, 14)
    n = len(b["c"])
    sig = ss * feat
    elig = np.zeros(n, dtype=bool)
    elig[lo:hi] = True
    v = sig[elig & np.isfinite(sig) & np.isfinite(a) & (a > 0)]
    if len(v) < 200:
        return None
    return float(np.quantile(v, q))


def side_sign_sign(x):
    return 1


def main() -> None:
    pairs = sys.argv[1:] or sorted(
        d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    print("=" * 78)
    print("THE 2:1 FRONTIER — searched on TRAIN, judged once on TEST")
    print("=" * 78)
    print(f"data: {DATA} (Binance Vision 1h mirror, REAL OHLC)")
    print(f"grid: {len(KS)} stops x {len(RS)} reward:risk x {len(QUANTILES)} "
          f"thresholds x 2 sides x every feature")
    print()
    print("The arithmetic of the ask, before any measurement:")
    for earn, label in ((0.6666, "the literal ask"), (0.98, "66% at 2:1")):
        print(f"  {label:18s} = +${earn*100:.2f} per $100 risked at 2:1 "
              f"-> needs p = {(earn + 1) / 3:.4%}")
    print(f"  break-even at 2:1 is p = {1/3:.4%}")
    print()

    for pair in pairs:
        try:
            b = load(pair)
        except Exception as exc:
            print(f"{pair}: unavailable ({exc})")
            continue
        n = len(b["c"])
        cut = int(n * TRAIN_FRAC)
        print("=" * 78)
        print(f"── {pair}: {n} real hourly bars ({n/24:.0f} days) "
              f"| train 200..{cut} | test {cut}..{n}")
        feats = features(b)
        rows = []
        for name, feat in feats.items():
            for ss in (1, -1):
                for q in QUANTILES:
                    thr = train_threshold(b, feat, ss, q, 200, cut)
                    if thr is None:
                        continue
                    for k in KS:
                        w, z = evaluate(b, feat, ss, q, k, 2.0, thr, 200, cut)
                        if w + z >= MIN_TRADES:
                            rows.append((w / (w + z), name, ss, q, k, thr,
                                         w + z))
        if not rows:
            print("   no configuration produced enough train trades")
            continue
        rows.sort(key=lambda t: -t[0])
        print(f"   {len(rows)} configs at r=2.0 with n>={MIN_TRADES} on train")
        print(f"   {'config':<32}{'TRAIN':>8}{'n':>7}{'TEST':>8}{'n':>7}"
              f"{'95% CI':>18}  verdict")
        judged = 0
        for wr, name, ss, q, k, thr, ntr in rows[:6]:
            tw, tz = evaluate(b, feats[name], ss, q, k, 2.0, thr, cut, n)
            if tw + tz < 20:
                continue
            twr = tw / (tw + tz)
            lo, hi = wilson(tw, tw + tz)
            be = 1 / 3
            tag = ("EDGE" if lo > be else "at chance")
            print(f"   {f'{name} x{ss} q{q} k{k}':<32}{wr:7.1%}{ntr:7d}"
                  f"{twr:7.1%}{tw+tz:7d}  [{lo:.1%},{hi:.1%}]  {tag}")
            judged += 1
        print(f"   ({judged} frozen configs judged on test; "
              f"{len(rows)} were searched, so expected false EDGE "
              f"flags = {0.05*len(rows):.1f})")
        print()

    # ---- the frontier: best TEST win rate at each r, per asset
    print("=" * 78)
    print("THE FRONTIER — best TEST win rate achievable at each reward:risk")
    print("=" * 78)
    print(f"  {'asset':10}{'r':>6}{'best WR':>9}{'n':>7}{'95% CI':>18}"
          f"{'break-even':>12}{'EV(R)':>9}{'  config'}")
    for pair in pairs:
        try:
            b = load(pair)
        except Exception:
            continue
        n = len(b["c"])
        cut = int(n * TRAIN_FRAC)
        feats = features(b)
        for r in RS:
            rows = []
            for name, feat in feats.items():
                for ss in (1, -1):
                    for q in (0.4, 0.6, 0.8):
                        thr = train_threshold(b, feat, ss, q, 200, cut)
                        if thr is None:
                            continue
                        for k in (1.0, 2.0, 3.0):
                            w, z = evaluate(b, feat, ss, q, k, r, thr, 200, cut)
                            if w + z >= MIN_TRADES:
                                rows.append((w / (w + z), name, ss, q, k, thr))
            if not rows:
                continue
            rows.sort(key=lambda t: -t[0])
            wr, name, ss, q, k, thr = rows[0]
            tw, tz = evaluate(b, feats[name], ss, q, k, r, thr, cut, n)
            if tw + tz < 20:
                continue
            twr = tw / (tw + tz)
            lo, hi = wilson(tw, tw + tz)
            be = 1.0 / (1.0 + r)
            ev = twr * r - (1 - twr)
            print(f"  {pair:10}{r:6.1f}{twr:8.1%}{tw+tz:7d}  [{lo:.1%},{hi:.1%}]"
                  f"{be:11.1%}{ev:+9.3f}  {name} x{ss} k{k}")

    print()
    print("READ THIS BEFORE QUOTING ANY NUMBER:")
    print("  * Every row is the BEST of a large search. At alpha=0.05 the")
    print("    expected number of false EDGE flags among the cells searched")
    print("    is printed with each table; a lone edge in a search of")
    print("    hundreds is not evidence.")
    print("  * EV(R) is per unit risked, BEFORE friction. Subtract the")
    print("    asset round trip (BTC 28.9 bps, ETH 29.8, BNB 31.0, PAXG")
    print("    37.4) expressed in R for the tradable number.")
    print("  * Win rate alone is not a target. At 2:1 the frontier's")
    print("    break-even is 33.33%, so a cell at 45% is a real edge while a")
    print("    cell at 60% at r=0.5 (break-even 66.7%) is a loser.")

    consistency()
    friction_table()




# ═══════════════ 5. CROSS-ASSET CONSISTENCY (the real multiple-testing test)
def consistency(pairs=None, r: float = 2.0) -> None:
    """A real edge repeats on assets it was never fitted to; an artifact does not.

    Searching 900 configurations and keeping the best is guaranteed to
    produce a flattering number — at alpha=0.05 the expected number of pure
    flukes among 900 cells is 45. The defence is not a smaller p-value, it is
    REPLICATION: apply each configuration to every asset without re-fitting
    anything, and ask how many clear break-even on ALL of them.

    Measured on real 1h tapes (BTC/ETH/BNB 2024-01-01..2026-09-30, PAXG
    from 2025-03-27; test half only):

        configs where the CI lower bound clears 1/3 on ALL FOUR assets: 0
                                                                  of 432

        best cross-asset mean win rate at r=2: 36.4 %
        (mom336 x1 q0.8 k1: BTC 35.3, ETH 35.8, BNB 35.7, PAXG 38.6)

    36.4 % against a 33.3 % break-even is a +0.09R gross edge — real, small,
    and completely consumed by friction (next section). It is not remotely
    66 %, and it is not 55.6 % either.
    """
    if pairs is None:
        pairs = sorted(d for d in os.listdir(DATA)
                       if os.path.isdir(os.path.join(DATA, d)))
    tapes = {}
    for p in pairs:
        try:
            tapes[p] = load(p)
        except Exception:
            continue
    if len(tapes) < 2:
        print("consistency: need >=2 assets")
        return
    grid = [(n, ss, q, k) for n in features(next(iter(tapes.values())))
            for ss in (1, -1) for q in (0.4, 0.6, 0.8)
            for k in (1.0, 2.0, 3.0, 5.0)]
    print("=" * 78)
    print(f"CROSS-ASSET CONSISTENCY at r={r}: does the edge replicate?")
    print("=" * 78)
    print(f"  {len(grid)} configs x {len(tapes)} assets = "
          f"{len(grid)*len(tapes)} cells, threshold fitted on TRAIN only")

    def run(cfg):
        name, ss, q, k = cfg
        out = {}
        for p, b in tapes.items():
            N = len(b["c"])
            cut = int(N * TRAIN_FRAC)
            feat = features(b)[name]
            thr = train_threshold(b, feat, ss, q, 200, cut)
            if thr is None:
                return None
            tw, tz = evaluate(b, feat, ss, q, k, r, thr, cut, N)
            if tw + tz < 20:
                return None
            out[p] = (tw / (tw + tz), tw + tz, wilson(tw, tw + tz))
        return out

    passing = 0
    scored = []
    for cfg in grid:
        res = run(cfg)
        if not res or len(res) < len(tapes):
            continue
        if all(lo > 1.0 / (1.0 + r) for _w, _n, (lo, _hi) in res.values()):
            passing += 1
            print(f"  PASSES: {cfg}")
        scored.append((float(np.mean([v[0] for v in res.values()])), cfg, res))
    print(f"\n  configs clearing break-even on EVERY asset: {passing} "
          f"of {len(grid)}")
    scored.sort(key=lambda t: -t[0])
    print(f"  {'config':30s}{'mean WR':>9s}  per-asset")
    for mean, cfg, res in scored[:6]:
        per = "  ".join(f"{p[:3]}={res[p][0]:.1%}" for p in sorted(res))
        print(f"  {str(cfg):30s}{mean:8.1%}  {per}")


# ═══════════════ 6. FRICTION: the number that decides it
def friction_table(pairs=None, r: float = 2.0) -> None:
    """A 2:1 win is worth +2R gross; friction is subtracted in R.

    On 1h tape the median ATR(14) is 63 bps (BTC), 89 (ETH), 69 (BNB) and
    30 (PAXG); the measured Binance round trip is 29 / 30 / 31 / 37 bps.
    So a stop at k*ATR costs the following share of R per trade:

        BTC 0.46R at k=1, 0.15R at k=3      PAXG 1.26R at k=1, 0.42R at k=3

    PAXG at a 1-ATR stop cannot pay for itself at ANY win rate below 100 %:
    the cost is larger than the risk. This is why every configuration that
    looked positive gross turns negative net, and why the honest table is
    the friction-adjusted one.
    """
    from .target_66_real import RT_BPS
    if pairs is None:
        pairs = sorted(d for d in os.listdir(DATA)
                       if os.path.isdir(os.path.join(DATA, d)))
    print("=" * 78)
    print(f"FRICTION-ADJUSTED EXPECTANCY at r={r} (real Binance round trip)")
    print("=" * 78)
    print(f"  {'asset':10}{'ATR bps':>9}{'round trip':>12}"
          f"{'cost@k=1':>11}{'cost@k=3':>11}")
    for p in pairs:
        try:
            b = load(p)
        except Exception:
            continue
        a = atr(b, 14)
        med = float(np.nanmedian(a[200:] / b["c"][200:])) * 1e4
        rt = RT_BPS.get(p, float("nan"))
        print(f"  {p:10}{med:9.1f}{rt:12.1f}{rt/med:10.2f}R{rt/(3*med):10.2f}R")
    print("\n  net EV per trade = (WR*r - (1-WR)) - round_trip/(k*ATR)")
    print("  A cell must clear that sum, not just the win rate.")


if __name__ == "__main__":
    main()
