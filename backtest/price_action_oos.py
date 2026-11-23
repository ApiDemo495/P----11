"""Purged walk-forward audit of the shipped context-gated PA veto.

This runner compares the immutable v8 certified direction/geometry with
v8 plus exactly the application's `price_action.read(...).strong` opposing
veto. No thresholds are tuned here. Entries use next-bar OPEN, exits use
completed OHLC bars, both-touch candles are excluded as AMBIGUOUS, and the
reported fixed stress cost is exactly 24 bps round trip for every trade.

The locked holdout is the final UTC calendar year available at the preregistered
cutoff (partial 2026 through 2026-11-20 23:59 UTC, matching the v8 artifact's
year-based convention). Up to five chronological validation folds are built only
from years strictly before that holdout; each test fold is embargoed from any
train window ending within max(247 min, 7 days) before the fold. Tape rows after
the preregistered cutoff are truncated, so no future data is used. The PA function
has no fitted parameters, so the train portion is not used to tune it.

Run: .venv/bin/python -m backtest.price_action_oos
Requires real tapes in backtest/cache/{SYM}_1m.csv and PAXG_1h.csv. Missing
or too-coarse data is reported as ungradeable; no synthetic fill is allowed.
"""
from __future__ import annotations
import csv, json, math, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from app import hold_policy as hp
from app import price_action as pa
from backtest.continuous import load_bars, _utc_years, _year_start
from backtest.data import ROUND_TRIP_BPS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ("BTC", "ETH", "BNB", "PAXG")
TF_MINUTES = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}
RT_BPS = 24.0
PURGE_S = 7 * 86400
N_FOLDS = 5
EXPECTED_V8_MD5 = "cee99dc26d359a3de2cbf7c7a2751b8f"
END_CUT_EPOCH = int(datetime(2026, 11, 20, 23, 59, tzinfo=timezone.utc).timestamp())


class PAStream:
    """O(1)-per-bar replica of read() on every causal prefix.

    read() only needs the last 14 true-ranges, last 20 volumes, last two
    confirmed pivots for structure, and last six pivots per side for level.
    A pivot at i is available only once i+5 has closed. This preserves the
    app's exact 5-left/5-right pivot and score rules without re-scanning the
    full multi-year prefix at each window.
    """
    def __init__(self, t, o, h, l, c, v):
        self.t, self.o, self.h, self.l, self.c, self.v = t, o, h, l, c, v
        self.i = 0
        self.hi_p, self.lo_p = [], []

    def advance(self, cut: int) -> None:
        while self.i < cut:
            i = self.i
            self.i += 1
            pivot = i - 5
            if pivot < 5:
                continue
            # This is exactly _pivots()'s candidate test on the prefix
            # ending at i, where pivot has five bars on both sides.
            hs = self.h[pivot - 5:pivot + 6]
            ls = self.l[pivot - 5:pivot + 6]
            if (self.h[pivot] >= float(np.max(hs))
                    and self.h[pivot] > max(float(np.max(self.h[pivot-5:pivot])),
                                            float(np.max(self.h[pivot+1:pivot+6])))):
                self.hi_p.append(pivot)
            if (self.l[pivot] <= float(np.min(ls))
                    and self.l[pivot] < min(float(np.min(self.l[pivot-5:pivot])),
                                            float(np.min(self.l[pivot+1:pivot+6])))):
                self.lo_p.append(pivot)

    def read(self) -> dict:
        n = self.i
        neutral = {"direction": 0, "score": 0.0, "strong": False}
        if n < 60:
            return neutral
        a = n - min(14, n - 1)
        atr = 0.0
        m = min(14, n - 1)
        for i in range(n - m, n):
            prev = self.c[i - 1]
            atr += max(self.h[i] - self.l[i], abs(self.h[i] - prev), abs(self.l[i] - prev))
        atr /= m
        structure = 0
        if len(self.hi_p) >= 2 and len(self.lo_p) >= 2:
            h2, h1 = self.h[self.hi_p[-1]], self.h[self.hi_p[-2]]
            l2, l1 = self.l[self.lo_p[-1]], self.l[self.lo_p[-2]]
            if h2 > h1 and l2 > l1: structure = 1
            elif h2 < h1 and l2 < l1: structure = -1
        # Exact nearest recent swing level, support below = +1, resistance above = -1.
        level, dist = 0, float("inf")
        band = 1.5 * atr if atr > 0 else None
        price = float(self.c[n - 1])
        for pivots, values in ((self.hi_p, self.h), (self.lo_p, self.l)):
            for p in pivots[-6:]:
                d = abs(price - values[p])
                if band is not None and d > band: continue
                side = 1 if values[p] < price else -1
                if d < dist:
                    level, dist = side, d
        trigger, _ = pa._trigger(
            float(self.o[n-1]), float(self.h[n-1]), float(self.l[n-1]), float(self.c[n-1]),
            float(self.o[n-2]), float(self.h[n-2]), float(self.l[n-2]), float(self.c[n-2]))
        vol_ok = False
        if n >= 21:
            base = sum(self.v[n-21:n-1]) / 20.0
            vol_ok = base > 0 and self.v[n-1] >= 1.5 * base
        score = pa._W_STRUCTURE * structure
        if trigger and level and trigger == level:
            score += pa._W_TRIGGER_AT_LEVEL
            if vol_ok: score += pa._W_VOLUME_CONFIRM
        elif trigger and structure and trigger == structure:
            score += pa._W_TRIGGER_ALIGNED
            if vol_ok: score += pa._W_TRIGGER_ALIGNED_VOL
        elif trigger:
            score += pa._W_TRIGGER_SOLO
        score = max(-1.0, min(1.0, score))
        direction = 0 if abs(score) < pa._DIR_FLOOR else (1 if score > 0 else -1)
        return {"direction": direction, "score": round(score, 4),
                "strong": bool(direction and abs(score) >= pa._STRONG)}


def _windows(t, step_s: int, w_min: int):
    W = w_min * 60
    t0 = int(math.ceil(int(t[0]) / W) * W)
    starts = np.arange(t0, int(t[-1]) + step_s - W + 1, W, dtype=np.int64)
    nxt = np.searchsorted(t, starts, side="left")
    end = np.searchsorted(t, starts + W - step_s, side="right") - 1
    good = ((nxt < len(t)) & (end > nxt) & (end < len(t))
            & (t[np.minimum(nxt, len(t)-1)] >= starts)
            & ((t[np.minimum(nxt, len(t)-1)] - starts) <= step_s))
    # Every observed bar inside the decision window must be present at
    # the declared resolution; never let a multi-hour hole masquerade as
    # a completed-bar path or next-bar fill.
    valid=np.zeros(len(starts),dtype=bool)
    for q in np.flatnonzero(good):
        valid[q]=(t[end[q]]-t[nxt[q]]) == (int(end[q])-int(nxt[q]))*step_s
    good &= valid
    return starts[good], nxt[good], end[good]


def _pa_directions(t,o,h,l,c,v,starts):
    cuts = np.searchsorted(t, starts, side="left")
    direction = np.zeros(len(starts), dtype=np.int8)
    strong = np.zeros(len(starts), dtype=bool)
    order = np.argsort(cuts, kind="stable")
    stream = PAStream(t,o,h,l,c,v)
    for ix in order:
        stream.advance(int(cuts[ix]))   # excludes the decision bar at t == start
        result = stream.read()
        direction[ix] = int(result["direction"])
        strong[ix] = bool(result["strong"])
    return direction, strong


def _rows_for(t,o,h,l,c,starts,nxt,end,sides,scale,k,m,blocked,entry_mask=None):
    """Next-bar entries, completed-bar first-touch; output one row/window."""
    allowed = np.ones(len(starts), dtype=bool) if entry_mask is None else entry_mask.copy()
    allowed &= ~blocked
    n=len(starts)
    result=np.zeros(n,dtype=np.int8)  # 1 target, -1 stop, 0 horizon, 2 ambiguous, 3 veto
    gross=np.full(n,np.nan)
    entry=np.full(n,np.nan)
    for q in range(n):
        if not allowed[q]:
            result[q]=3
            continue
        d=int(sides[q]); e=float(o[nxt[q]])
        if not d or e<=0 or not np.isfinite(scale[q]) or scale[q]<=0:
            result[q]=3; continue
        sl=min(float(m)*float(scale[q]),e*0.9)
        tp=min(float(k)*sl,e*0.9)
        tp_px=e+tp*d; sl_px=e-sl*d
        entry[q]=e
        code=0; ex=float(c[end[q]])
        for j in range(int(nxt[q]),int(end[q])+1):
            sl_hit=(l[j]<=sl_px) if d>0 else (h[j]>=sl_px)
            tp_hit=(h[j]>=tp_px) if d>0 else (l[j]<=tp_px)
            if sl_hit and tp_hit:
                code=2; break
            if sl_hit:
                code=-1; ex=sl_px; break
            if tp_hit:
                code=1; ex=tp_px; break
        result[q]=code
        if code!=2:
            gross[q]=d*(ex/e-1.0)*1e4
    return allowed,result,gross,entry


def _metrics(allowed,result,gross):
    graded=allowed & (result!=2) & np.isfinite(gross)
    net=gross[graded]-RT_BPS
    touched=graded & np.isin(result,(1,-1))
    wins=int(np.sum(net>0)); n=int(len(net))
    target=int(np.sum(result[graded]==1)); stop=int(np.sum(result[graded]==-1))
    pos=float(net[net>0].sum()); neg=float(-net[net<0].sum())
    return {"issued_n":int(allowed.sum()),"graded_n":n,
            "ambiguous_n":int(np.sum(allowed & (result==2))),
            "strict_wr_net":round(wins/n,6) if n else None,
            "first_touch_wr":round(target/(target+stop),6) if target+stop else None,
            "first_touch_n":target+stop,
            "pf_net":round(pos/neg,6) if neg else (None if not pos else "inf"),
            "net_ev_bps":round(float(net.mean()),6) if n else None,
            "target_n":target,"stop_n":stop}


def _era_holdout_ev(t,o,h,l,c,starts,nxt,end,sides,scale,k,m,blocked,holdout_year,step_s):
    """One position per contiguous certified-side era, next-open fill.

    The PA veto is applied only at era entry (issuance); once entered, the
    position is held through the era and exits at a single closed-bar touch
    or next-bar open on side flip. Both-touch eras are excluded and their
    unresolved cost is reported, not treated as zero-cost.
    """
    years=_utc_years(starts)
    records=[]; i=0
    while i<len(starts):
        d=int(sides[i]); j=i+1
        while j<len(starts) and int(sides[j])==d: j+=1
        if d and years[i]==holdout_year:
            enter=i
            while enter<j and blocked[enter]: enter+=1
            if enter<j:
                e=float(o[nxt[enter]])
                if e>0 and np.isfinite(scale[enter]) and scale[enter]>0:
                    sl=min(float(m)*float(scale[enter]),e*0.9); tp=min(float(k)*sl,e*0.9)
                    tpx=e+tp*d; slx=e-sl*d
                    last=int(end[j-1]); stop_at=int(nxt[j]) if j<len(nxt) else last
                    hi_end=min(last,stop_at)
                    outcome=0; ex=float(c[hi_end]); gap=False
                    # Missing native bars inside a held era make the path
                    # unknowable. Count a data gap separately, never as zero EV.
                    if hi_end > int(nxt[enter]):
                        gap=bool(np.any(np.diff(t[int(nxt[enter]):hi_end+1]) != step_s))
                    if gap:
                        outcome=3
                    else:
                        for b in range(int(nxt[enter]),hi_end+1):
                            slhit=(l[b]<=slx) if d>0 else (h[b]>=slx)
                            tphit=(h[b]>=tpx) if d>0 else (l[b]<=tpx)
                            if slhit and tphit: outcome=2; break
                            if slhit: outcome=-1; ex=slx; break
                            if tphit: outcome=1; ex=tpx; break
                    net=d*(ex/e-1)*1e4-RT_BPS if outcome not in (2,3) else None
                    records.append({"outcome":outcome,"net_bps":net})
        i=j
    resolved=[r for r in records if r["outcome"] not in (2,3)]
    return {"n":len(records),"graded_n":len(resolved),
            "ambiguous_n":sum(r["outcome"]==2 for r in records),
            "data_gap_n":sum(r["outcome"]==3 for r in records),
            "ev_net_bps":(round(float(np.mean([r["net_bps"] for r in resolved])),6)
                          if resolved else None),
            "win_rate_net":(round(float(np.mean([r["net_bps"]>0 for r in resolved])),6)
                            if resolved else None)}


def _fold_report(starts, allowed0, result0, gross0, allowed1, result1, gross1, holdout_year):
    years=_utc_years(starts)
    train_years=sorted(set(int(y) for y in years if int(y)<holdout_year))
    chunks=[x for x in np.array_split(np.asarray(train_years,dtype=int), min(N_FOLDS,len(train_years))) if len(x)]
    folds=[]
    for fi,chunk in enumerate(chunks,1):
        lo,hi=int(chunk[0]),int(chunk[-1])
        test=(years>=lo)&(years<=hi)
        # Explicit purge boundary: no observation in this fold can use a
        # training label that would mature inside the 7-day + max-horizon embargo.
        train=(years<lo)&(starts < _year_start(lo)-PURGE_S-247*60)
        folds.append({"fold":fi,"test_years":[lo,hi],"train_n":int(train.sum()),
                      "purge_before":datetime.fromtimestamp(_year_start(lo),timezone.utc).date().isoformat(),
                      "baseline":_metrics(allowed0&test,result0,gross0),
                      "after":_metrics(allowed1&test,result1,gross1)})
    return folds


def run() -> dict:
    digest=__import__("hashlib").md5((ROOT/"results"/"hold_wr.json").read_bytes()).hexdigest()
    if digest!=EXPECTED_V8_MD5:
        raise RuntimeError(f"refusing run: v8 artifact md5 {digest} != expected {EXPECTED_V8_MD5}")
    report={"method":"fixed v8 side/geometry; exact app PA opposition predicate; next_bar open; closed OHLC; ambiguous excluded; fixed 24 bps RT",
            "holdout":"final UTC calendar year in v8 convention (partial 2026 through 2026-11-20 23:59 UTC), with all later tape rows truncated",
            "walk_forward":"up to five chronological pre-holdout folds, 7-day + 247-minute purging/embargo; PAXG has one resolvable fold; no PA parameters fitted",
            "v8_md5":digest,"round_trip_bps":RT_BPS,"cells":{}}
    for asset in ASSETS:
        bars=load_bars(asset)
        if bars is None:
            for tf in TF_MINUTES:
                report["cells"][f"{asset}|{tf}"]={"status":"MISSING REAL TAPE","baseline":None,"after":None}
            continue
        t,o,h,l,c,v,step=bars
        keep=np.asarray(t)<END_CUT_EPOCH
        t,o,h,l,c,v=(x[keep] for x in (t,o,h,l,c,v))
        bars=(t,o,h,l,c,v,step)
        if len(t)<2:
            for tf in TF_MINUTES:
                report["cells"][f"{asset}|{tf}"]={"status":"INSUFFICIENT DATA AT PREREGISTERED CUTOFF","baseline":None,"after":None}
            continue
        for tf,w in TF_MINUTES.items():
            key=f"{asset}|{tf}"
            record=hp.cell(asset,tf) or {}
            selected=record.get("selected") or {}
            if not bars or int(step)*4>w*60 or record.get("status")!="ok" or not selected:
                report["cells"][key]={"status":("UNGRADEABLE RESOLUTION" if int(step)*4>w*60 else "NO CERTIFIED V8 ROW"),
                                       "baseline":None,"after":None}
                continue
            starts,nxt,end=_windows(t,int(step),w)
            sides_all=hp.side_series(asset,tf,bars)
            W=w*60
            first=int(math.ceil(int(t[0])/W)*W)
            starts_all=np.arange(first,int(t[-1])+int(step)-W+1,W,dtype=np.int64)
            entry_all=np.searchsorted(t,starts_all-int(step),side="right")-1
            exit_all=np.searchsorted(t,starts_all+W-int(step),side="right")-1
            side_ok=(entry_all>=0)&(exit_all>entry_all)&(exit_all<len(t))
            aligned_starts=starts_all[side_ok]
            if sides_all is None or len(sides_all)!=len(aligned_starts):
                report["cells"][key]={"status":"V8 SIDE PARITY/ARTIFACT UNAVAILABLE","baseline":None,"after":None}
                continue
            side_ix=np.searchsorted(aligned_starts,starts)
            if len(side_ix) and not np.array_equal(aligned_starts[side_ix],starts):
                report["cells"][key]={"status":"V8 SIDE ALIGNMENT FAILURE","baseline":None,"after":None}
                continue
            sides=sides_all[side_ix]
            scale=hp.scale_at_bars(bars,starts-int(step),tf)
            rowk,rowm=hp.row_geometry(asset,tf)
            pa_dir,pa_strong=_pa_directions(t,o,h,l,c,v,starts)
            blocked=pa_strong & (pa_dir != sides)
            baseline_allowed,bres,bgross,_=_rows_for(t,o,h,l,c,starts,nxt,end,sides,scale,rowk,rowm,
                                                      np.zeros(len(starts),bool))
            after_allowed,ares,agross,_=_rows_for(t,o,h,l,c,starts,nxt,end,sides,scale,rowk,rowm,blocked)
            years=_utc_years(starts); hold_year=int(max(years))
            base=_metrics(baseline_allowed,bres,bgross)
            after=_metrics(after_allowed,ares,agross)
            base["holdout_era"]=_era_holdout_ev(t,o,h,l,c,starts,nxt,end,sides,scale,
                                                   float(selected["era_k"]),float(selected["era_m"]),
                                                   np.zeros(len(starts),bool),hold_year,int(step))
            after["holdout_era"]=_era_holdout_ev(t,o,h,l,c,starts,nxt,end,sides,scale,
                                                    float(selected["era_k"]),float(selected["era_m"]),
                                                    blocked,hold_year,int(step))
            hold_mask=years==hold_year
            base["locked_holdout"]=_metrics(baseline_allowed&hold_mask,bres,bgross)
            after["locked_holdout"]=_metrics(after_allowed&hold_mask,ares,agross)
            folds=_fold_report(starts,baseline_allowed,bres,bgross,after_allowed,ares,agross,hold_year)
            hold_min_n=max(100,int(0.7*base["locked_holdout"]["issued_n"]))
            hold_wr_ok=(after["locked_holdout"]["strict_wr_net"] is not None
                        and base["locked_holdout"]["strict_wr_net"] is not None
                        and after["locked_holdout"]["strict_wr_net"] >= base["locked_holdout"]["strict_wr_net"])
            hold_touch_ok=(after["locked_holdout"]["first_touch_wr"] is not None
                           and base["locked_holdout"]["first_touch_wr"] is not None
                           and after["locked_holdout"]["first_touch_wr"] >= base["locked_holdout"]["first_touch_wr"])
            promotion=(after["locked_holdout"]["issued_n"] >= hold_min_n
                       and hold_wr_ok and hold_touch_ok
                       and after["holdout_era"]["ev_net_bps"] is not None
                       and after["holdout_era"]["ev_net_bps"] > 0
                       and after["holdout_era"]["ev_net_bps"] >= (base["holdout_era"]["ev_net_bps"] or 0))
            report["cells"][key]={"status":"measured","bar_step_s":int(step),
                "v8_rule":selected.get("side"),"row_geometry":{"k":rowk,"m":rowm},
                "holdout_year":hold_year,"baseline":base,"after":after,
                "folds":folds,"promotion":"ISSUANCE GATE" if promotion else "TELEMETRY ONLY",
                "promotion_rule":{"min_holdout_n":hold_min_n,
                    "strict_sign_net_wr_nondecrease":True,
                    "first_touch_wr_nondecrease":True,
                    "holdout_era_ev_net_positive":True,
                    "holdout_era_ev_non_decrease":True}}
            print(f"{key}: {report['cells'][key]['promotion']} n {base['issued_n']}→{after['issued_n']} hold-era EV {base['holdout_era']['ev_net_bps']}→{after['holdout_era']['ev_net_bps']} bps",flush=True)
    return report


def write(report):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/"price_action_oos.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["# Context-gated price-action OOS audit","",f"Method: {report['method']}",
           f"Holdout: {report['holdout']}",f"Folds: {report['walk_forward']}",
           f"v8 artifact md5: `{report['v8_md5']}`","",
           "| cell | status | baseline n/WR/PF/EV bp/hold-era EV bp | after n/WR/PF/EV bp/hold-era EV bp | promotion |",
           "|---|---|---:|---:|---|"]
    for key,row in report["cells"].items():
        def show(x):
            if not x:return "—"
            e=x.get("holdout_era",{}).get("ev_net_bps")
            return f"{x.get('issued_n')} / {x.get('strict_wr_net')} / {x.get('pf_net')} / {x.get('net_ev_bps')} / {e}"
        lines.append(f"| {key} | {row['status']} | {show(row.get('baseline'))} | {show(row.get('after'))} | {row.get('promotion','—')} |")
    lines += ["","AMBIGUOUS rows are counted and excluded from WR/PF/EV; their counts are disclosed in `ambiguous_n`. `strict_wr_net` counts net PnL > 0 after the fixed 24 bp fee. `first_touch_wr` is target touches / target+stop touches before fee classification. PF and EV use net bps.",
              "A promoted cell is still a holdout-selected promotion and should be treated as a validation result, not proof of future profits. Any missing tape, insufficient resolution, or absent certified v8 geometry stays telemetry-only."]
    (RESULTS/"price_action_oos.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    out=run();write(out)
    print(f"wrote {RESULTS/'price_action_oos.md'}",flush=True)
