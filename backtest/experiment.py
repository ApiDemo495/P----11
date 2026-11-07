"""Walk-forward improvement lab on REAL data.

Collects every decision point once (engine pack + slice features +
future bars), then settles MANY variants offline: direction rules
(engine tf1 / TF-majority / tf4 / momentum / trend), barrier scale
k (SL/TP distances x k, ratio stays 2:1), horizon (22/44 bars),
and meta-filters (TF agreement, trend/momentum alignment,
confidence, side). Policies are SELECTED on the first 55% of
decisions and judged only on the untouched last 45% (out-of-sample).
Anything that wins only in-sample is rejected here, visibly.
"""
from __future__ import annotations

import json
import math

from .data import FRICTION_BPS, Bar
from .predictor import HORIZON, AppPredictor
from .slice import DataSlice

KS = (0.5, 0.75, 1.0, 1.5)
HS = (22, 44)


def _ema(xs, n):
    k = 2 / (n + 1)
    e = xs[0]
    for x in xs[1:]:
        e = x * k + e * (1 - k)
    return e


def _rsi(closes, n=14):
    if len(closes) < n + 1:
        return 50.0
    g = l = 0.0
    for a, b in zip(closes[-n - 1:-1], closes[-n:]):
        d = b - a
        g += max(d, 0)
        l += max(-d, 0)
    return 100.0 if l == 0 else 100 - 100 / (1 + (g / n) / (l / n))


def _atr_pct(bars, n=14):
    trs = []
    for a, b in zip(bars[-n - 1:-1], bars[-n:]):
        trs.append(max(b.h - b.l, abs(b.h - a.c), abs(b.l - a.c)))
    return (sum(trs) / len(trs)) / bars[-1].c * 100 if trs else 0.0


def settle_dir(asset, entry, direction, risk, k, horizon, future):
    """Net R for barriers entry -/+ direction*risk*k (SL) and 2x (TP)."""
    if direction == 0 or risk <= 0:
        return None
    sl = entry - direction * risk * k
    tp = entry + direction * 2 * risk * k
    exit_price = future[horizon - 1].c
    for b in future[:horizon]:
        if (b.l <= sl if direction > 0 else b.h >= sl):
            exit_price = sl
            break
        if (b.h >= tp if direction > 0 else b.l <= tp):
            exit_price = tp
            break
    r = direction * (exit_price - entry) / (risk * k)
    return r - (entry * FRICTION_BPS[asset] / 1e4) / (risk * k)


def collect(series, predictor: AppPredictor, warmup=80, step=22):
    from app.moves_risk import risk_block
    rows = []
    for asset, bars in series.items():
        ds = DataSlice(bars)
        for t in range(warmup, len(bars) - max(HS), step):
            view = ds.at(t)
            pack = predictor.pack(view, asset)
            dirs = {tf: int(pack[tf].get("direction") or 0)
                    for tf in ("tf1", "tf2", "tf3", "tf4")}
            closes = view.closes
            entry = closes[-1]
            ema_s = 1 if _ema(closes, 20) >= _ema(closes, 50) else -1
            mom = 1 if closes[-1] >= closes[-22] else -1
            future = ds.future(t, t + max(HS) + 1)
            risks = {}
            for d in set(dirs.values()) | {mom, ema_s}:
                if d == 0:
                    continue
                blk = risk_block(entry, d, HORIZON, 0.5, closes,
                                 view.highs, view.lows)
                if blk.get("sl") is not None:
                    risks[d] = abs(entry - float(blk["sl"]))
            rules = {"tf1": dirs["tf1"], "tf2": dirs["tf2"],
                     "tf4": dirs["tf4"],
                     "majority": (1 if sum(dirs.values()) > 0 else
                                  -1 if sum(dirs.values()) < 0 else 0),
                     "momentum": mom, "trend": ema_s}
            out = {}
            for name, d in rules.items():
                if d in risks:
                    for k in KS:
                        for h in HS:
                            r = settle_dir(asset, entry, d, risks[d], k, h,
                                           future)
                            if r is not None:
                                out[f"{name}|k{k}|h{h}"] = round(r, 4)
            rows.append({"asset": asset, "t": t, "ts": bars[t].t,
                         "dirs": dirs, "conf": float(pack["tf1"].get(
                             "confidence") or 0.0),
                         "ema": ema_s, "mom": mom,
                         "rsi": round(_rsi(closes), 1),
                         "atr_pct": round(_atr_pct(view.bars), 3),
                         "out": out})
    return rows


def _m(rs):
    if not rs:
        return {"n": 0, "win_pct": None, "ev_r": None, "pf": None}
    w = [r for r in rs if r > 0]
    lo = [r for r in rs if r <= 0]
    return {"n": len(rs), "win_pct": round(100 * len(w) / len(rs), 1),
            "ev_r": round(sum(rs) / len(rs), 3),
            "pf": round(sum(w) / abs(sum(lo)), 2) if lo and sum(lo) else None}


def evaluate(rows):
    """Select on train (first 55% per asset), judge on test."""
    assets = sorted({r["asset"] for r in rows})
    train, test = [], []
    for a in assets:
        rs = [r for r in rows if r["asset"] == a]
        cut = int(len(rs) * 0.55)
        train += rs[:cut]
        test += rs[cut:]
    keys = sorted({k for r in rows for k in r["out"]})
    # global direction-rule x barrier policies, selected on train
    cands = {}
    for kkey in keys:
        tr = [r["out"][kkey] for r in train if kkey in r["out"]]
        if len(tr) >= 40:
            cands[kkey] = _m(tr)
    best = sorted(cands, key=lambda k: cands[k]["ev_r"] or -9)[-6:]
    report = {"n_train": len(train), "n_test_rows": len(test),
              "policies": {}}
    for kkey in ["tf1|k1.0|h22"] + best:
        te = [r["out"][kkey] for r in test if kkey in r["out"]]
        report["policies"][kkey] = {"train": cands.get(kkey) or _m(
            [r["out"][kkey] for r in train if kkey in r["out"]]),
            "test": _m(te)}
    # meta-filters on the tf1 baseline, thresholds from train only
    base_tr = [r for r in train if "tf1|k1.0|h22" in r["out"]]
    med_conf = sorted(r["conf"] for r in base_tr)[len(base_tr) // 2]
    filters = {
        "tf_agree(tf1==tf2)": lambda r: r["dirs"]["tf1"] == r["dirs"]["tf2"],
        "trend_align": lambda r: r["ema"] == r["dirs"]["tf1"],
        "mom_align": lambda r: r["mom"] == r["dirs"]["tf1"],
        "tf+trend": lambda r: (r["dirs"]["tf1"] == r["dirs"]["tf2"]
                               and r["ema"] == r["dirs"]["tf1"]),
        "conf>=train_median": lambda r: r["conf"] >= med_conf,
        "long_only": lambda r: r["dirs"]["tf1"] > 0,
        "rsi_not_extreme": lambda r: 25 <= r["rsi"] <= 75,
    }
    frep = {}
    for name, f in filters.items():
        tr = [r["out"]["tf1|k1.0|h22"] for r in base_tr if f(r)]
        te = [r["out"]["tf1|k1.0|h22"] for r in test
              if "tf1|k1.0|h22" in r["out"] and f(r)]
        frep[name] = {"train": _m(tr), "test": _m(te)}
    report["filters_tf1_k1_h22"] = frep
    # per-asset best-of (rule policies + filters) chosen on train
    menu = {**{k: ("out", k) for k in best},
            **{f"flt:{n}": ("flt", n) for n in filters}}
    chosen, per_asset_test = {}, []
    for a in assets:
        tr_a = [r for r in train if r["asset"] == a]
        te_a = [r for r in test if r["asset"] == a]
        best_name, best_ev = "tf1|k1.0|h22", -9.0
        for name, (kind, ref) in menu.items():
            if kind == "out":
                vals = [r["out"][ref] for r in tr_a if ref in r["out"]]
            else:
                vals = [r["out"]["tf1|k1.0|h22"] for r in tr_a
                        if "tf1|k1.0|h22" in r["out"] and filters[ref](r)]
            m = _m(vals)
            if m["n"] >= 8 and (m["ev_r"] or -9) > best_ev:
                best_name, best_ev = name, m["ev_r"]
        chosen[a] = best_name
        kind, ref = menu.get(best_name, ("out", "tf1|k1.0|h22"))
        if kind == "out":
            per_asset_test += [r["out"][ref] for r in te_a if ref in r["out"]]
        else:
            per_asset_test += [r["out"]["tf1|k1.0|h22"] for r in te_a
                               if "tf1|k1.0|h22" in r["out"]
                               and filters[ref](r)]
    report["per_asset_best"] = {"chosen_on_train": chosen,
                                "test": _m(per_asset_test)}
    base_test = [r["out"]["tf1|k1.0|h22"] for r in test
                 if "tf1|k1.0|h22" in r["out"]]
    report["baseline_test"] = _m(base_test)
    return report




# ── full-optionality collection: everything later is offline ─────────

def collect_full(series, predictor: AppPredictor, warmup=80, step=22):
    """Like collect(), but stores entry, per-direction risk distances
    and the next 44 bars (h/l/c) so ANY direction rule, inversion,
    barrier scale or horizon can be settled offline forever."""
    from app.moves_risk import risk_block
    rows = []
    for asset, bars in series.items():
        ds = DataSlice(bars)
        for t in range(warmup, len(bars) - max(HS), step):
            view = ds.at(t)
            pack = predictor.pack(view, asset)
            closes = view.closes
            entry = closes[-1]
            risks = {}
            for d in (1, -1):
                blk = risk_block(entry, d, HORIZON, 0.5, closes,
                                 view.highs, view.lows)
                if blk.get("sl") is not None:
                    risks[str(d)] = abs(entry - float(blk["sl"]))
            if len(risks) < 2:
                continue
            rows.append({
                "asset": asset, "t": t, "ts": bars[t].t, "entry": entry,
                "dirs": {tf: int(pack[tf].get("direction") or 0)
                         for tf in ("tf1", "tf2", "tf3", "tf4")},
                "conf": float(pack["tf1"].get("confidence") or 0.0),
                "ema": 1 if _ema(closes, 20) >= _ema(closes, 50) else -1,
                "mom": 1 if closes[-1] >= closes[-22] else -1,
                "rsi": round(_rsi(closes), 1),
                "atr_pct": round(_atr_pct(view.bars), 3),
                "risks": risks,
                "fut": [[b.h, b.l, b.c] for b in ds.future(t, t + 45)]})
    return rows


def rule_dir(row, rule):
    d = row["dirs"]
    if rule == "tf1":
        return d["tf1"]
    if rule == "inv_tf1":
        return -d["tf1"]
    if rule == "tf4":
        return d["tf4"]
    if rule == "inv_tf4":
        return -d["tf4"]
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
    if rule == "inv_trend":
        return -row["ema"]
    if rule == "rsi_revert":
        return 1 if row["rsi"] <= 30 else -1 if row["rsi"] >= 70 else 0
    if rule == "rsi_trend":
        return 1 if row["rsi"] >= 55 else -1 if row["rsi"] <= 45 else 0
    return 0


def settle_row(row, rule, k, h):
    d = rule_dir(row, rule)
    if d == 0:
        return None
    fut = [Bar(row["ts"], 0.0, fh, fl, fc) for fh, fl, fc in row["fut"]]
    return settle_dir(row["asset"], row["entry"], d,
                      row["risks"][str(d)], k, h, fut)


RULES = ("tf1", "inv_tf1", "tf4", "inv_tf4", "majority", "inv_majority",
         "momentum", "fade_mom", "trend", "inv_trend", "rsi_revert",
         "rsi_trend")
KS2 = (0.4, 0.5, 0.75, 1.0, 1.5, 2.0)
HS2 = (11, 22, 44)


def grid_report(rows):
    assets = sorted({r["asset"] for r in rows})
    train, test = [], []
    for a in assets:
        rs = [r for r in rows if r["asset"] == a]
        cut = int(len(rs) * 0.55)
        train += rs[:cut]
        test += rs[cut:]
    scored = []
    for rule in RULES:
        for k in KS2:
            for h in HS2:
                tr = [settle_row(r, rule, k, h) for r in train]
                tr = [x for x in tr if x is not None]
                m = _m(tr)
                if m["n"] >= 40 and m["ev_r"] is not None:
                    scored.append((m["ev_r"], rule, k, h, m))
    scored.sort(reverse=True)
    out = {"top_by_train": [], "baseline_test": None}
    for ev, rule, k, h, mtr in scored[:8]:
        te = [settle_row(r, rule, k, h) for r in test]
        te = [x for x in te if x is not None]
        out["top_by_train"].append(
            {"rule": rule, "k": k, "h": h, "train": mtr, "test": _m(te)})
    te = [settle_row(r, "tf1", 1.0, 22) for r in test]
    out["baseline_test"] = _m([x for x in te if x is not None])
    # per-asset: pick each asset's best rule/k/h on ITS train slice
    chosen, agg = {}, []
    for a in assets:
        tr_a = [r for r in train if r["asset"] == a]
        te_a = [r for r in test if r["asset"] == a]
        best = None
        for rule in RULES:
            for k in KS2:
                for h in HS2:
                    vals = [settle_row(r, rule, k, h) for r in tr_a]
                    vals = [x for x in vals if x is not None]
                    m = _m(vals)
                    if m["n"] >= 10 and m["ev_r"] is not None and (
                            best is None or m["ev_r"] > best[0]):
                        best = (m["ev_r"], rule, k, h)
        chosen[a] = best[1:]
        vals = [settle_row(r, best[1], best[2], best[3]) for r in te_a]
        agg += [x for x in vals if x is not None]
    out["per_asset_best"] = {"chosen_on_train": chosen, "test": _m(agg)}
    return out


if __name__ == "__main__":
    raw = json.load(open("/home/user/realdata/real_series.json"))
    series = {s.replace("USDT", ""): [Bar(int(t), o, h, l, c, v)
              for t, o, h, l, c, v in rows] for s, rows in raw.items()}
    rows = collect_full(series, AppPredictor())
    json.dump(rows, open("/home/user/realdata/decisions_full.json", "w"))
    print(json.dumps(grid_report(rows), indent=1))
