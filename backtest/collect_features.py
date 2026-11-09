"""Collect per-decision features + outcomes from the FULL engine on
real tape, for walk-forward meta-gate research (which existing
signals actually predict success, out-of-sample?).

Same data + anti-look-ahead discipline as per_tf_real.py. Every
numeric leaf of the engine pack (formulas CCI, brains, physics,
emotion, confidence machinery) is flattened, plus tape regime
features computed ONLY from bars[:t+1]. Outcomes: strict-sign win
at the TF's own window close (tie=loss) + net bps after friction.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

from .data import Bar, FRICTION_BPS
from .per_tf_real import SYMS, TF_W, load_bars
from .predictor import AppPredictor
from .slice import DataSlice

import os as _os
KEEP = int(_os.environ.get("CF_KEEP", "3500"))
EVAL = int(_os.environ.get("CF_EVAL", "2000"))
STEP = int(_os.environ.get("CF_STEP", "22"))
SKIP = int(_os.environ.get("CF_SKIP", "0"))   # bars dropped from the end
OUT = _os.environ.get("CF_OUT", os.path.join(
    os.path.dirname(__file__), "results", "decision_features_real.jsonl"))


def flat(prefix, obj, out, depth=0):
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(f"{prefix}.{k}", v, out, depth + 1)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(obj):
            out[prefix] = float(obj)
    elif isinstance(obj, bool):
        out[prefix] = 1.0 if obj else 0.0


def regime(bars, t, W):
    c = [b.c for b in bars[:t + 1]]
    rets = [(c[i] / c[i - 1] - 1) for i in range(max(1, len(c) - 100), len(c))]
    vol = (sum(r * r for r in rets) / max(1, len(rets))) ** 0.5
    ema = c[-1]
    k = 2 / 51
    e = c[0]
    for x in c:
        e += k * (x - e)
    ema50 = e
    mom_w = c[-1] / c[-W - 1] - 1 if len(c) > W else 0.0
    hi = max(b.h for b in bars[max(0, t - W):t + 1])
    lo = min(b.l for b in bars[max(0, t - W):t + 1])
    return {"reg.vol100": vol, "reg.ema_dist": c[-1] / ema50 - 1,
            "reg.momW": mom_w, "reg.range_pos": (c[-1] - lo) / (hi - lo)
            if hi > lo else 0.5}


def main():
    series = {s: load_bars("/tmp/crypto", p)[-KEEP - SKIP:]
              for s, p in SYMS.items()}
    if SKIP:
        series = {s: b[:-SKIP] for s, b in series.items()}
    pred = AppPredictor(); pred.boot()

    def set_state(s2, b2):
        st = pred.feed.state(s2)
        st.times = [b.t for b in b2]; st.opens = [b.o for b in b2]
        st.highs = [b.h for b in b2]; st.lows = [b.l for b in b2]
        st.closes = [b.c for b in b2]; st.volumes = [b.v for b in b2]
        st.sources = ["backtest"] * len(b2)
        st.live_price = b2[-1].c
        st.live_open, st.live_high = b2[-1].o, b2[-1].h
        st.live_low = b2[-1].l; st.day_open = b2[0].o

    n = 0
    with open(OUT, "w") as fh:
        for sym, bars in series.items():
            ds = DataSlice(bars)
            t = len(bars) - EVAL
            while t + max(TF_W.values()) < len(bars):
                for s2, b2 in series.items():
                    set_state(s2, b2[:t + 1])
                pack = pred.pack(ds.at(t), sym)
                entry = bars[t].c
                for tf, W in TF_W.items():
                    res = pack.get(tf) or {}
                    d = int(res.get("direction") or 0)
                    if not d:
                        continue
                    f = {}
                    flat("pk", res, f)
                    f.update(regime(bars, t, W))
                    settle = bars[t + W].c
                    bps = d * (settle / entry - 1) * 1e4
                    fh.write(json.dumps({
                        "asset": sym, "tf": tf, "t": bars[t].t,
                        "dir": d, "win": 1 if bps > 0 else 0,
                        "bps_net": round(bps - FRICTION_BPS[sym], 2),
                        "f": f}) + "\n")
                    n += 1
                t += STEP
            print(sym, "done", flush=True)
    print("rows", n, "->", OUT)


if __name__ == "__main__":
    main()
