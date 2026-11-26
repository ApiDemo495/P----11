"""Pick the stop size / horizon geometry that MAXIMISES net EV on real tape.

This isolates GEOMETRY from signal quality by using a RANDOM side. Barrier
hit rates are fixed by the geometry alone; a signal can only tilt the balance
around the 1/(1+r) geometric rate. So this table is a floor, not a forecast.

Units matter and were wrong once: the median ATR is a RATIO of price, so it
must be multiplied by 1e4 to become bps before dividing the round-trip cost
into it.
"""
import sys

import numpy as np

sys.path.insert(0, '.')
from backtest.target_66_real import load, atr, RT_BPS

TRAIN_FRAC = 0.65


def run(pair, sl_mult, r, H, ndraw=3, seed=11):
    b = load(pair)
    h, l, c = b["h"], b["l"], b["c"]
    n = len(c)
    a = atr(b, 14)
    cut = int(n * TRAIN_FRAC)
    idx = np.arange(cut, n - 1)
    idx = idx[idx + H < n]
    med_ratio = float(np.nanmedian(a[cut:] / c[cut:]))
    med_bps = med_ratio * 1e4                 # <-- the units fix
    sl_bps = sl_mult * med_bps
    fr = RT_BPS[pair] / max(sl_bps, 1e-9)     # friction in R
    rng = np.random.default_rng(seed)
    tot_tp = tot_sl = tot_to = 0
    for _ in range(ndraw):
        s = rng.choice([-1, 1], size=len(idx))
        e = c[idx]
        d = sl_mult * a[idx]
        tp = e + s * r * d
        sl = e - s * d
        out = np.zeros(len(idx), dtype=np.int8)
        done = np.zeros(len(idx), dtype=bool)
        for off in range(1, H + 1):
            j = idx + off
            ok = (j < n) & ~done
            if not ok.any():
                break
            hi, lo = h[j], l[j]
            L = s > 0
            tph = np.where(L, hi >= tp, lo <= tp)
            slh = np.where(L, lo <= sl, hi >= sl)
            dd = ok & slh
            out[dd] = -1
            done |= dd
            dd = ok & tph & ~slh
            out[dd] = 1
            done |= dd
            if done.all():
                break
        tot_tp += int((out == 1).sum())
        tot_sl += int((out == -1).sum())
        tot_to += int((out == 0).sum())
    N = len(idx) * ndraw
    ev = (tot_tp * (r - fr) - tot_sl * (1.0 + fr)) / N
    return tot_tp / N, tot_sl / N, tot_to / N, fr, ev


def main():
    print("random-side control: geometry only, no signal.")
    print("net EV charges the real round trip and does NOT credit drift on")
    print("timeouts (conservative). 2:1 means TP = 2x the stop distance.\n")
    for pair in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "PAXGUSDT"):
        print(f"── {pair}")
        print(f"  {'k*ATR':>6}{'H(bars)':>8}{'TP%':>7}{'SL%':>7}{'to%':>7}"
              f"{'fric(R)':>9}{'net EV(R)':>11}")
        rows = []
        for k in (0.5, 0.8, 1.2, 2.0, 2.8, 4.0):
            for H in (22, 48, 96, 168, 336):
                tp, sl, to, fr, ev = run(pair, k, 2.0, H)
                rows.append((ev, k, H, tp, sl, to, fr))
                print(f"  {k:6.1f}{H:8d}{tp:6.1%}{sl:6.1%}{to:6.1%}"
                      f"{fr:9.2f}{ev:+11.3f}")
        best = max(rows)
        print(f"  BEST net EV {best[0]:+.3f}R at k={best[1]} H={best[2]}"
              f"  ({best[3]+best[4]:.1%} resolve, {best[5]:.1%} timeout)\n")


if __name__ == "__main__":
    main()
