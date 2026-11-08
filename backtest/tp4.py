"""TP4-timeframe analysis: the app's macro target (tf4 = 247-bar
window, targets 60/120/180/247), judged on the LAST 1 YEAR of real
1h data. Decisions are non-overlapping (one TP4 trade at a time,
247-bar horizon, levels from moves_risk at minutes=247, TP = 2x SL
unless a barrier-scale variant is being tested). Walk-forward:
policies selected on the first 55% of the year's decisions and
judged only on the last 45%. No number here is in-sample.
"""
from __future__ import annotations

import json

from .data import FRICTION_BPS, Bar
from .experiment import _atr_pct, _ema, _m, _rsi, settle_dir
from .predictor import AppPredictor
from .slice import DataSlice

H4 = 247                      # tf4 window / TP4 horizon (bars)
YEAR_START = 1759276800       # 2025-10-01 00:00 UTC
YEAR_END = 1790812800         # 2026-09-30 00:00 UTC (last bar < this)


def collect_tp4(series, predictor: AppPredictor, warmup=600):
    from app.moves_risk import risk_block
    rows = []
    for asset, bars in series.items():
        ds = DataSlice(bars)
        idx = [i for i, b in enumerate(bars)
               if YEAR_START <= b.t < YEAR_END - H4 * 3600]
        i = next((i for i in idx if i >= warmup), None)
        while i is not None and i + H4 < len(bars) and bars[i].t < YEAR_END:
            view = ds.at(i)
            pack = predictor.pack(view, asset)
            closes = view.closes
            entry = closes[-1]
            risks = {}
            for d in (1, -1):
                blk = risk_block(entry, d, H4, 0.5, closes,
                                 view.highs, view.lows)
                if blk.get("sl") is not None:
                    risks[str(d)] = abs(entry - float(blk["sl"]))
            if len(risks) == 2:
                rows.append({
                    "asset": asset, "t": i, "ts": bars[i].t, "entry": entry,
                    "dirs": {tf: int(pack[tf].get("direction") or 0)
                             for tf in ("tf1", "tf2", "tf3", "tf4")},
                    "conf4": float(pack["tf4"].get("confidence") or 0.0),
                    "conf1": float(pack["tf1"].get("confidence") or 0.0),
                    "ema": 1 if _ema(closes, 50) >= _ema(closes, 200) else -1,
                    "mom": 1 if closes[-1] >= closes[-H4] else -1,
                    "rsi": round(_rsi(closes), 1),
                    "atr_pct": round(_atr_pct(view.bars), 3),
                    "risks": risks,
                    "fut": [[b.h, b.l, b.c]
                            for b in ds.future(i, i + H4 + 1)]})
            i += H4
    return rows


def rule_dir(row, rule):
    d = row["dirs"]
    if rule == "tf4":
        return d["tf4"]
    if rule == "inv_tf4":
        return -d["tf4"]
    if rule == "tf3":
        return d["tf3"]
    if rule == "inv_tf3":
        return -d["tf3"]
    if rule == "majority":
        s = sum(d.values())
        return 1 if s > 0 else -1 if s < 0 else 0
    if rule == "inv_majority":
        return -rule_dir(row, "majority")
    if rule == "momentum":
        return row["mom"]
    if rule == "fade_mom":
        return -row["mom"]
    if rule == "trend":
        return row["ema"]
    if rule == "tf4_x_mom":          # only when macro agrees with momentum
        return d["tf4"] if d["tf4"] == row["mom"] else 0
    if rule == "inv_tf4_x_mom":
        return -d["tf4"] if -d["tf4"] == row["mom"] else 0
    if rule == "tf4_x_trend":
        return d["tf4"] if d["tf4"] == row["ema"] else 0
    if rule == "rsi_revert":
        return 1 if row["rsi"] <= 30 else -1 if row["rsi"] >= 70 else 0
    return 0


def settle_row(row, rule, k, h=H4):
    d = rule_dir(row, rule)
    if d == 0:
        return None
    fut = [Bar(row["ts"], 0.0, fh, fl, fc) for fh, fl, fc in row["fut"]]
    return settle_dir(row["asset"], row["entry"], d,
                      row["risks"][str(d)], k, h, fut)


RULES = ("tf4", "inv_tf4", "tf3", "inv_tf3", "majority", "inv_majority",
         "momentum", "fade_mom", "trend", "tf4_x_mom", "inv_tf4_x_mom",
         "tf4_x_trend", "rsi_revert")
KS = (0.5, 0.75, 1.0, 1.5, 2.0)


def split(rows):
    train, test = [], []
    for a in sorted({r["asset"] for r in rows}):
        rs = [r for r in rows if r["asset"] == a]
        cut = int(len(rs) * 0.55)
        train += rs[:cut]
        test += rs[cut:]
    return train, test


def grid(rows):
    train, test = split(rows)
    scored = []
    for rule in RULES:
        for k in KS:
            tr = [settle_row(r, rule, k) for r in train]
            tr = [x for x in tr if x is not None]
            m = _m(tr)
            if m["n"] >= 15 and m["ev_r"] is not None:
                scored.append((m["ev_r"], rule, k, m))
    scored.sort(reverse=True)
    out = {"baseline": {}, "top_by_train": [], "per_asset_best": {}}
    for part_name, part in (("train", train), ("test", test)):
        vals = [settle_row(r, "tf4", 1.0) for r in part]
        out["baseline"][part_name] = _m([x for x in vals if x is not None])
    for ev, rule, k, mtr in scored[:8]:
        te = [settle_row(r, rule, k) for r in test]
        out["top_by_train"].append({"rule": rule, "k": k, "train": mtr,
                                     "test": _m([x for x in te
                                                 if x is not None])})
    chosen, agg = {}, []
    for a in sorted({r["asset"] for r in rows}):
        tr_a = [r for r in train if r["asset"] == a]
        te_a = [r for r in test if r["asset"] == a]
        best = None
        for rule in RULES:
            for k in KS:
                vals = [settle_row(r, rule, k) for r in tr_a]
                vals = [x for x in vals if x is not None]
                m = _m(vals)
                if m["n"] >= 5 and m["ev_r"] is not None and (
                        best is None or m["ev_r"] > best[0]):
                    best = (m["ev_r"], rule, k)
        chosen[a] = best[1:] if best else None
        if best:
            vals = [settle_row(r, best[1], best[2]) for r in te_a]
            agg += [x for x in vals if x is not None]
    out["per_asset_best"] = {"chosen_on_train": chosen, "test": _m(agg)}
    return out


if __name__ == "__main__":
    raw = json.load(open("/home/user/realdata/full_series.json"))
    series = {s.replace("USDT", ""): [Bar(int(t), o, h, l, c, v)
              for t, o, h, l, c, v in rows] for s, rows in raw.items()}
    rows = collect_tp4(series, AppPredictor())
    json.dump(rows, open("/home/user/realdata/tp4_decisions.json", "w"))
    print(f"decisions: {len(rows)}")
    print(json.dumps(grid(rows), indent=1))
