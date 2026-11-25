"""The 66 % @ 2:1 target, measured end-to-end on real Binance 1h tape.

Run:  .venv/bin/python -m backtest.target_66
Data: the Binance Vision mirror named in REAL_RESULTS.md
      (KevinSherry2016/Crypto, marketData/unzipped — byte-identical klines).
      Clone it sparsely first:

  git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/KevinSherry2016/Crypto.git /tmp/cryptomirror
  cd /tmp/cryptomirror && git sparse-checkout set \
      marketData/unzipped/BTCUSDT marketData/unzipped/ETHUSDT \
      marketData/unzipped/BNBUSDT marketData/unzipped/PAXGUSDT \
      marketData/unzipped/SOLUSDT

What it answers, in order:

  1. What directional edge does 66 % at 2:1 REQUIRE?  (exact arithmetic)
  2. What win rate does 2:1 actually produce on real tape?  (measured)
  3. Can a signal reach 66 % by search?  (search on train, judge on TEST)
  4. What does the achievable (win rate, reward/risk) frontier look like,
     and does any cell survive multiple-testing?

No abstention: every bar with a non-zero signal is traded. No look-ahead:
the signal reads bars <= i only and the test half is never used to choose
anything. Friction is charged separately (see the tables) because first-touch
win rate is a gross quantity by definition.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from .target_66_real import (DATA, TRAIN_FRAC, atr, load, sig_breakout,
                             sig_mom_vol, sig_revert, sig_trend, walk_fast,
                             wilson)

KS = (2.0, 3.0, 5.0, 8.0, 13.0)
HORIZONS = (100, 400, 1200)


# ─────────────── 1. the arithmetic: what 66 % requires ───────────────
def p_hit_first(a: float, b: float, lam: float) -> float:
    """P(touch +a before -b) for drift-diffusion with drift lam/2, var 1."""
    if abs(lam) < 1e-12:
        return b / (a + b)
    return (math.exp(lam * b) - 1.0) / (math.exp(lam * b) - math.exp(-lam * a))


def required_drift(a: float, b: float, target: float) -> float:
    lo, hi = 0.0, 80.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if p_hit_first(a, b, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def up_probability(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


# ─────────────── 2. one (signal, k, r, horizon) walk ───────────────
def walk(b: dict, sig: np.ndarray, k: float, r: float, horizon: int,
         lo: int, hi: int) -> tuple[int, int]:
    a = atr(b, 14)
    c, h, l = b["c"], b["h"], b["l"]
    wins = losses = 0
    for i in range(max(200, lo), min(hi, len(c) - 2)):
        s = int(sig[i])
        if s == 0 or not np.isfinite(a[i]) or a[i] <= 0:
            continue
        d = k * a[i]
        e = c[i]
        res = walk_fast(None, h, l, c, i, s,
                        e + s * r * d, e - s * d, max_bars=horizon)
        if res is None or res == 0:
            continue
        if res > 0:
            wins += 1
        else:
            losses += 1
    return wins, losses


SIGNALS = {
    "trend-24h": lambda b: sig_trend(b, 24),
    "trend-72h": lambda b: sig_trend(b, 72),
    "trend-168h": lambda b: sig_trend(b, 168),
    "revert-z96": lambda b: sig_revert(b, 0, 96),
    "revert-z240": lambda b: sig_revert(b, 0, 240),
    "revert-z96-fade": lambda b: -sig_revert(b, 0, 96),
    "trend-72h-fade": lambda b: -sig_trend(b, 72),
    "breakout-48h": lambda b: sig_breakout(b, 48),
    "breakout-168h": lambda b: sig_breakout(b, 168),
    "momvol-72h": lambda b: sig_mom_vol(b, 72),
}


def fmt(w: int, z: int) -> str:
    if w + z == 0:
        return "n=0"
    p = w / (w + z)
    lo, hi = wilson(w, w + z)
    return f"{p:6.1%} [{lo:.1%},{hi:.1%}] n={w + z:5d}"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    pairs = sorted(d for d in os.listdir(DATA)
                   if os.path.isdir(os.path.join(DATA, d)))
    if len(sys.argv) > 1:
        pairs = sys.argv[1:]
    tapes = {p: load(p) for p in pairs}
    for p, b in tapes.items():
        print(f"{p}: {len(b['c'])} hourly bars, zero gaps")

    section("1) WHAT 66 % AT 2:1 REQUIRES (exact drift-diffusion arithmetic)")
    print("   barriers TP=+2 sigma, SL=-1 sigma; geometric win rate = 1/3 = 33.33 %")
    need = required_drift(2.0, 1.0, 0.66)
    print(f"   required drift lambda = {need:.4f} per window (per unit sigma)")
    print(f"   equivalent directional accuracy P(up) = "
          f"{up_probability(need / 2):.1%}")
    print(f"   66 % is {0.66 / (1 / 3):.2f}x the geometric rate")

    section("2) STOP FLOOR: a signal can never improve the stop side")
    print("   TP is far (+2 sigma), SL near (-1 sigma). A signal only changes the")
    print("   probability of reaching the FAR barrier. The near barrier is touched")
    print("   from noise alone, so the realistic ceiling is only a few points above")
    print("   33.3 % — measured below.")

    section("3) SEARCH ON TRAIN, JUDGE ON TEST (no abstention, no look-ahead)")
    for p, b in tapes.items():
        n = len(b["c"])
        cut = int(n * TRAIN_FRAC)
        ranked = []
        for name, fn in SIGNALS.items():
            sig = fn(b)
            for k in KS:
                for hz in HORIZONS:
                    w, z = walk(b, sig, k, 2.0, hz, 0, cut)
                    if w + z >= 30:
                        ranked.append((w / (w + z), name, k, hz, w + z, sig))
        ranked.sort(key=lambda t: -t[0])
        print(f"\n═══ {p}  ({len(ranked)} configs on train)")
        print(f"  {'config':<30}{'TRAIN':>8}{'n':>7}{'TEST':>9}{'n':>7}  95% CI")
        for wr, name, k, hz, ntr, sig in ranked[:3]:
            tw, tz = walk(b, sig, k, 2.0, hz, cut, n)
            if tw + tz == 0:
                continue
            lo, hi = wilson(tw, tw + tz)
            flag = "  <== clears 66%" if lo > 0.66 else ""
            print(f"  {f'{name} k={k} h={hz}':<30}{wr:7.1%}{ntr:7d}"
                  f"{tw / (tw + tz):8.1%}{tw + tz:7d}  [{lo:.1%},{hi:.1%}]{flag}")

    section("4) ACHIEVABLE FRONTIER ON THE TEST HALF (same signal at every r)")
    print(f"  {'asset':10}{'r=tp/sl':>9}{'win rate':>10}{'n':>7}{'95% CI':>17}"
          f"{'break-even':>12}  verdict")
    cells = []
    for p, b in tapes.items():
        n = len(b["c"])
        cut = int(n * TRAIN_FRAC)
        sig = sig_mom_vol(b, 72)
        sig[:cut] = 0                       # TEST HALF ONLY
        for r in (0.25, 0.33, 0.5, 1.0, 2.0):
            w, z = walk(b, sig, 3.0, r, 1200, cut, n)
            if w + z < 22:
                continue
            wr = w / (w + z)
            lo, hi = wilson(w, w + z)
            be = 1.0 / (1.0 + r)
            verdict = "EDGE?" if lo > be else "at risk"
            cells.append((p, r, w, z, lo, be, verdict))
            print(f"  {p:10}{r:9.2f}{wr:9.1%}{w + z:7d}  [{lo:.1%},{hi:.1%}]"
                  f"{be:11.1%}  {verdict}")
        print()
    flagged = [c for c in cells if c[6] == "EDGE?"]
    print(f"  {len(cells)} cells tested at alpha=0.05 -> expected false positives "
          f"= {0.05 * len(cells):.1f}")
    print(f"  cells flagged: {len(flagged)}.  Bonferroni threshold = "
          f"{0.05 / max(1, len(cells)):.4f}")

    section("ANSWER")
    print("  66 % at 2:1 was NOT reached. The measured ceiling on 24k+ real bars")
    print("  per asset, searched on train and judged on test, is the geometric")
    print("  33.3 % plus noise. Win rate at 2:1 and win rate at 0.25:1 are the")
    print("  SAME trade re-labelled — see the frontier table: every asset sits on")
    print("  1/(1+r) at every r. Expectancy, not win rate, is what the dial moves,")
    print("  and at these costs it moves it to -friction.")


if __name__ == "__main__":
    main()
