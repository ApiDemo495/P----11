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


def _flat(d, prefix="", out=None, depth=0):
    """Numeric leaves of a nested result dict -> flat ablation inputs."""
    out = {} if out is None else out
    if depth > 2 or not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, dict):
            _flat(v, key + ".", out, depth + 1)
        elif isinstance(v, list):
            for i, item in enumerate(v[:6]):
                if isinstance(item, (int, float)):
                    out[f"{key}[{i}]"] = float(item)
                elif isinstance(item, dict):
                    _flat(item, f"{key}[{i}].", out, depth + 1)
    return out


def _install_capture(predictor):
    """Wrap (not modify) engine methods to record, per (symbol, tf),
    the exact fusion inputs + brain/physics/emotion outputs of the
    pass that produced the pack. Returns the capture dict."""
    eng = predictor.engine
    cap: dict = {}
    orig_cci = eng.cci.compute
    def cci_wrap(symbol, tf, comp, *a, **kw):
        cap[("comp", symbol, tf)] = {int(k): float(v) for k, v in comp.items()}
        return orig_cci(symbol, tf, comp, *a, **kw)
    eng.cci.compute = cci_wrap
    orig_brains = eng.brains.evaluate
    def brains_wrap(symbol, tf, *a, **kw):
        ret = orig_brains(symbol, tf, *a, **kw)
        cap[("brain", symbol, tf)] = _flat(ret)
        return ret
    eng.brains.evaluate = brains_wrap
    orig_phys = eng.physics.evaluate
    def phys_wrap(symbol, tf, *a, **kw):
        ret = orig_phys(symbol, tf, *a, **kw)
        cap[("phys", symbol, tf)] = _flat(ret)
        return ret
    eng.physics.evaluate = phys_wrap
    orig_emo = eng.emotions.compute
    def emo_wrap(ctx, *a, **kw):          # compute(ctx, market, ...)
        ret = orig_emo(ctx, *a, **kw)
        cap[("emo", ctx.symbol, ctx.tf)] = _flat(ret)
        return ret
    eng.emotions.compute = emo_wrap
    return cap


SPECIAL = {97: "BRAIN", 98: "EMOTION", 99: "PHYSICS"}


def _comp_names(comp: dict) -> dict:
    try:
        from app import config as _cfg
        for name in ("NEWS_COMPONENT_ID", "HEDGE_COMPONENT_ID",
                     "TRACER_COMPONENT_ID"):
            if hasattr(_cfg, name):
                SPECIAL[int(getattr(_cfg, name))] = name.replace(
                    "_COMPONENT_ID", "")
    except Exception:
        pass
    return {SPECIAL.get(int(k), f"F{int(k)}" if int(k) <= 30
                        else f"C{int(k)}"): float(v)
            for k, v in comp.items()}


def collect_tp4(series, predictor: AppPredictor, warmup=600,
                capture: bool = False):
    from app.moves_risk import risk_block
    cap = _install_capture(predictor) if capture else {}
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
                row = {
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
                            for b in ds.future(i, i + H4 + 1)]}
                if capture:
                    row["comp"] = _comp_names(
                        cap.get(("comp", asset, "tf4")) or {})
                    row["brain"] = cap.get(("brain", asset, "tf4")) or {}
                    row["phys"] = cap.get(("phys", asset, "tf4")) or {}
                    row["emo"] = cap.get(("emo", asset, "tf4")) or {}
                rows.append(row)
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
    raw = json.load(open("/home/user/P----11/.data_tmp/full_series.json"))
    series = {s.replace("USDT", ""): [Bar(int(t), o, h, l, c, v)
              for t, o, h, l, c, v in rows] for s, rows in raw.items()}
    rows = collect_tp4(series, AppPredictor(), capture=True)
    json.dump(rows, open("/home/user/P----11/.data_tmp/tp4_decisions.json", "w"))
    print(f"decisions: {len(rows)}")
    print(json.dumps(grid(rows), indent=1))
    print(json.dumps(ablation_report(rows), indent=1))


RS = (1.0, 1.5, 2.0, 3.0, 4.0)      # user bounds: 1:1 .. 4:1, pref 2:1


def settle_r(row, direction, k, r, h=H4):
    """TP4 barriers with TP = r x SL distance (r within [1, 4])."""
    if direction == 0:
        return None
    risk = row["risks"][str(direction)]
    entry = row["entry"]
    sl = entry - direction * risk * k
    tp = entry + direction * r * risk * k
    fut = [Bar(row["ts"], 0.0, fh, fl, fc) for fh, fl, fc in row["fut"]]
    exit_price = fut[h - 1].c
    for b in fut[:h]:
        if (b.l <= sl if direction > 0 else b.h >= sl):
            exit_price = sl
            break
        if (b.h >= tp if direction > 0 else b.l <= tp):
            exit_price = tp
            break
    rr = direction * (exit_price - entry) / (risk * k)
    return rr - (entry * FRICTION_BPS[row["asset"]] / 1e4) / (risk * k)


def _sign(v):
    return 1 if v > 0 else -1 if v < 0 else 0


def ablation_report(rows):
    """Every formula/component/brain as a standalone TP4 direction,
    then ensembles of the components that validated on train."""
    train, test = split(rows)
    names = sorted({n for r in rows for n in (r.get("comp") or {})})
    per_comp = {}
    for n in names:
        tr = [settle_r(r, _sign((r.get("comp") or {}).get(n, 0.0)), 0.75, 2.0)
              for r in train]
        te = [settle_r(r, _sign((r.get("comp") or {}).get(n, 0.0)), 0.75, 2.0)
              for r in test]
        per_comp[n] = {"train": _m([x for x in tr if x is not None]),
                       "test": _m([x for x in te if x is not None])}
    good = [n for n, v in per_comp.items()
            if (v["train"]["ev_r"] or -9) > 0 and v["train"]["n"] >= 20]
    good.sort(key=lambda n: -(per_comp[n]["train"]["ev_r"] or 0))

    def ensemble(part, members, weighted=False):
        vals = []
        for r in part:
            comp = r.get("comp") or {}
            if weighted:
                s = sum((per_comp[n]["train"]["ev_r"] or 0)
                        * _sign(comp.get(n, 0.0)) for n in members)
            else:
                s = sum(_sign(comp.get(n, 0.0)) for n in members)
            vals.append(settle_r(r, _sign(s), 0.75, 2.0))
        return _m([x for x in vals if x is not None])

    def with_rsi(part, members):
        vals = []
        for r in part:
            d = rule_dir(r, "rsi_revert")
            if d == 0:
                comp = r.get("comp") or {}
                d = _sign(sum(_sign(comp.get(n, 0.0)) for n in members))
            vals.append(settle_r(r, d, 0.75, 2.0))
        return _m([x for x in vals if x is not None])

    out = {"components": per_comp, "validated_on_train": good,
           "ensemble_good_equal": {"train": ensemble(train, good),
                                   "test": ensemble(test, good)},
           "ensemble_top5": {"train": ensemble(train, good[:5]),
                             "test": ensemble(test, good[:5])},
           "ensemble_weighted": {"train": ensemble(train, good, True),
                                 "test": ensemble(test, good, True)},
           "ensemble_plus_rsi": {"train": with_rsi(train, good),
                                 "test": with_rsi(test, good)}}
    # ratio grid for the headline policies (user: 1:1..4:1, pref 2:1)
    def policy_dir(r, kind):
        if kind == "tf4":
            return r["dirs"]["tf4"]
        d = rule_dir(r, "rsi_revert")
        return d if d != 0 else r["dirs"]["tf4"]
    out["ratio_grid"] = {}
    for kind in ("tf4", "combined"):
        out["ratio_grid"][kind] = {}
        for rr in RS:
            tr = [settle_r(r, policy_dir(r, kind), 0.75, rr) for r in train]
            te = [settle_r(r, policy_dir(r, kind), 0.75, rr) for r in test]
            out["ratio_grid"][kind][str(rr)] = {
                "train": _m([x for x in tr if x is not None]),
                "test": _m([x for x in te if x is not None])}
    return out
