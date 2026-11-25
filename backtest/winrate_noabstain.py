"""No-abstention win-rate experiment on REAL 5m tape (BTC/ETH Jan-Aug 2026).

Question: how to raise the win rate when EVERY window must trade?

Design (all causal, every window graded, no skips):
  sides (full coverage): up, mom44h, revW, rsi_fade
  geometries: time-exit + 5 barrier geometries (live TP-only widening
    vs proportional widening at RR 0.15/0.5/1.0/2.0)
  split: first 65% windows train, last 35% holdout (by time)
  metrics: sign WR (gross/net), barrier net WR (pnl>0), EV bps, PF

Usage: python3 backtest/winrate_noabstain.py
"""
from __future__ import annotations
import csv, math, os, sys
import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), "cache")
RESULTS = os.path.join(os.path.dirname(__file__), "results")
FRICTION = {"BTC": 14.436, "ETH": 14.904}  # per side bps (data.py)
TF_MIN = {"tf1": 22, "tf2": 44, "tf3": 130, "tf4": 247}

def load_5m(asset):
    path = os.path.join(CACHE, f"{asset}_5m.csv")
    rows = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0,1,2,3,4,5))
    rows = rows[np.argsort(rows[:,0], kind="stable")]
    t = rows[:,0].astype(np.int64)
    o,h,l,c,v = rows[:,1],rows[:,2],rows[:,3],rows[:,4],rows[:,5]
    return t,o,h,l,c,v

def wilson(k,n,z=1.96):
    if n==0: return (0.0,1.0)
    p=k/n; d=1+z*z/n; cc=(p+z*z/(2*n))/d
    hh=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0.0, cc - hh), min(1.0, cc + hh))

def build_windows(t, w_min):
    W = w_min*60
    t0 = int(math.ceil(int(t[0])/W)*W)
    t1 = int(t[-1])+300
    starts = np.arange(t0, t1-W+1, W, dtype=np.int64)
    i_entry = np.searchsorted(t, starts-300, side="right")-1
    i_exit = np.searchsorted(t, starts+W-300, side="right")-1
    ok = (i_entry>=0)&(i_exit>i_entry)&(i_exit<len(t))
    return starts[ok], i_entry[ok], i_exit[ok]

def scale_at(t,c,starts,w_min):
    # q80 of trailing W-length moves, causal (ends at/before start)
    W = w_min*60
    lb = max(1, int(round(W/300)))
    idx = np.searchsorted(t, starts, side="right")-1
    out = np.full(len(starts), np.nan)
    for j,i in enumerate(idx):
        if i < lb*2: continue
        lo = max(lb, i-100*lb)
        js = np.arange(lo, i+1, lb)
        if len(js)<8: continue
        moves = np.abs(c[js]-c[js-lb])
        out[j] = float(np.quantile(moves,0.80))
    # fill warmup with median
    med = np.nanmedian(out) if np.isfinite(out).any() else 0.0
    out = np.where(np.isfinite(out)&(out>0), out, med)
    return out

def hourly_closes(t,c):
    hour=(t//3600)*3600
    keep=np.ones(len(t),bool); keep[:-1]=hour[1:]!=hour[:-1]
    return hour[keep], c[keep]

def rsi_series(closes, n=14):
    # Wilder-ish simple RSI on hourly closes (live semantics approx)
    out=np.full(len(closes),50.0)
    for i in range(n, len(closes)):
        g=l=0.0
        for k in range(i-n+1,i+1):
            d=closes[k]-closes[k-1]
            if d>=0: g+=d
            else: l-=d
        out[i]=100.0 if l<=0 else 100-100/(1+g/l)
    return out

def sides_for(t,c,starts,i_entry,w_min):
    n=len(starts)
    up=np.ones(n,dtype=int)
    # mom44h: hourly 44h diff
    h_ts,h_c=hourly_closes(t,c)
    hb=(t[i_entry]//3600)*3600
    j=np.searchsorted(h_ts,hb,side="right")-1
    mom=np.ones(n,dtype=int)
    with np.errstate(divide="ignore",invalid="ignore"):
        d44=c[i_entry]-h_c[np.clip(j-44,0,None)]
    valid=(j>=44)&np.isfinite(d44)
    mom[valid]=np.where(d44[valid]>0,1,np.where(d44[valid]<0,-1,1))
    # revW: fade previous window [s-W,s)
    W=w_min*60
    base=np.searchsorted(t,starts-W,side="right")-1
    rev=np.ones(n,dtype=int)
    for i in range(n):
        b=int(base[i])
        if b<0:
            b0=int(i_entry[i])-1
            if b0<0: continue
            ref=0
        else: ref=b
        mv=c[i_entry[i]]-c[ref]
        if mv>0: rev[i]=-1
        elif mv<0: rev[i]=1
    # rsi_fade: -sign(RSI-50) on hourly, every window
    rsi_h=rsi_series(h_c,14)
    # map each window to hourly idx j
    rsi_f=np.ones(n,dtype=int)
    for i in range(n):
        jj=int(j[i])
        if jj<0 or jj>=len(rsi_h): continue
        r=rsi_h[jj]
        if r>50: rsi_f[i]=-1
        elif r<50: rsi_f[i]=1
        else: rsi_f[i]=1
    return {"up":up,"mom44h":mom,"revW":rev,"rsi_fade":rsi_f}

def settle_barrier(sides, entry, exit_, scale, h,l, i_entry,i_exit, rt_bps, geom):
    # geom: dict with mode
    n=len(sides)
    pnl=np.zeros(n); tp_hit=np.zeros(n,bool); sl_hit=np.zeros(n,bool); timeout=np.zeros(n,bool)
    rt=rt_bps*1e-4
    for i in range(n):
        side=int(sides[i]); e=float(entry[i])
        if geom["mode"]=="time":
            ret=side*(float(exit_[i])/e-1.0)-rt
            pnl[i]=ret
            timeout[i]=True
            continue
        sl_px_dist=float(scale[i])*geom["m"]
        tp_px_dist=float(scale[i])*geom["m"]*geom["k"]
        # cost floor handling
        tp_bps=tp_px_dist/e*1e4
        floor=1.5*rt_bps
        if tp_bps < floor and tp_bps>0:
            if geom["widen"]=="tp_only":
                tp_px_dist=floor*1e-4*e
            elif geom["widen"]=="prop":
                f=floor/tp_bps
                tp_px_dist*=f; sl_px_dist*=f
        tp_px=e+tp_px_dist*side
        sl_px=e-sl_px_dist*side
        a=int(i_entry[i])+1; b=int(i_exit[i])
        outcome=0
        for jj in range(a,b+1):
            hi=float(h[jj]); lo=float(l[jj])
            if side>0:
                s_h=lo<=sl_px; t_h=hi>=tp_px
            else:
                s_h=hi>=sl_px; t_h=lo<=tp_px
            if s_h: outcome=-1; break
            if t_h: outcome=1; break
        if outcome<0:
            sl_hit[i]=True; pnl[i]=-sl_px_dist/e-rt
        elif outcome>0:
            tp_hit[i]=True; pnl[i]=tp_px_dist/e-rt
        else:
            timeout[i]=True
            pnl[i]=side*(float(exit_[i])/e-1.0)-rt
    return pnl, tp_hit, sl_hit, timeout

def metrics(pnl):
    n=len(pnl); wins=int(np.sum(pnl>0))
    lo,hi=wilson(wins,n)
    ev=float(np.mean(pnl)*1e4) if n else 0.0
    # PF
    gp=float(np.sum(pnl[pnl>0])); gl=float(-np.sum(pnl[pnl<0]))
    pf=(gp/gl) if gl>0 else float("inf")
    return {"n":n,"wins":wins,"wr":round(wins/n,4) if n else 0.0,"ci95":[round(lo,4),round(hi,4)],
            "ev_bps":round(ev,2),"pf":round(pf,3) if np.isfinite(pf) else None,
            "worst_bps":round(float(np.min(pnl)*1e4),1) if n else 0.0}

def main():
    import json
    geoms={
        "time": {"mode":"time"},
        "live_tp_only_k0.15_m0.55": {"mode":"bar","k":0.15,"m":0.55,"widen":"tp_only"},
        "prop_k0.15_m0.55": {"mode":"bar","k":0.15,"m":0.55,"widen":"prop"},
        "prop_k0.5_m1.0": {"mode":"bar","k":0.5,"m":1.0,"widen":"prop"},
        "prop_k1.0_m1.0": {"mode":"bar","k":1.0,"m":1.0,"widen":"prop"},
        "prop_k2.0_m1.0": {"mode":"bar","k":2.0,"m":1.0,"widen":"prop"},
        "prop_k1.0_m2.0": {"mode":"bar","k":1.0,"m":2.0,"widen":"prop"},
    }
    out={}
    for asset in ["BTC","ETH"]:
        t,o,h,l,c,v=load_5m(asset)
        rt=2.0*FRICTION[asset]
        for tf,w_min in TF_MIN.items():
            starts,i_entry,i_exit=build_windows(t,w_min)
            entry=c[i_entry]; exit_=c[i_exit]
            scale=scale_at(t,c,starts,w_min)
            sides=sides_for(t,c,starts,i_entry,w_min)
            n=len(starts); cut=int(n*0.65)
            print(f"\n=== {asset} {tf} ({w_min}m) n={n} RT={rt:.1f}bps ===")
            # sign WR (gross/net) per side
            moves=exit_/entry-1.0
            for sname,s in sides.items():
                gross=np.sum(((moves>0)&(s>0))|((moves<0)&(s<0)))/n
                net=np.sum(s*moves-rt*1e-4>0)/n
                print(f"  sign {sname:9s} gross {gross*100:5.2f}% net {net*100:5.2f}%")
            cell={}
            for sname,s in sides.items():
                cell[sname]={}
                for gname,g in geoms.items():
                    pnl,tp,sl,to=settle_barrier(s,entry,exit_,scale,h,l,i_entry,i_exit,rt,g)
                    tr=metrics(pnl[:cut]); ho=metrics(pnl[cut:])
                    cell[sname][gname]={"train":tr,"holdout":ho,
                        "tp_rate":round(float(np.mean(tp[cut:])),4),
                        "sl_rate":round(float(np.mean(sl[cut:])),4),
                        "timeout_rate":round(float(np.mean(to[cut:])),4)}
                    print(f"  {sname:9s} {gname:24s} train WR {tr['wr']*100:5.1f}% EV {tr['ev_bps']:+7.1f} | hold WR {ho['wr']*100:5.1f}% EV {ho['ev_bps']:+7.1f} PF {ho['pf']}")
            out[f"{asset}|{tf}"]={"asset":asset,"tf":tf,"w_min":w_min,"rt_bps":rt,"n":n,"cells":cell}
    os.makedirs(RESULTS,exist_ok=True)
    json.dump(out,open(os.path.join(RESULTS,"winrate_noabstain.json"),"w"),indent=1)
    print("\n-> backtest/results/winrate_noabstain.json")

if __name__=="__main__":
    main()
