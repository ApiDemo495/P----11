"""Train the walk-forward-validated edge model and save it to data/models/.

Reproduce:
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/KevinSherry2016/Crypto.git /tmp/cryptomirror
    cd /tmp/cryptomirror && git sparse-checkout set \
        marketData/unzipped/BTCUSDT marketData/unzipped/ETHUSDT \
        marketData/unzipped/BNBUSDT marketData/unzipped/PAXGUSDT

    .venv/bin/python scripts/train_edge_model.py

The out-of-sample numbers this artifact was validated on are in
backtest/TARGET_66_VERDICT.md and `python -m backtest.edge_model`.
"""
import sys, numpy as np, warnings, json, time
warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
from sklearn.ensemble import HistGradientBoostingClassifier
from backtest.target_66_real import load, atr, wilson
from backtest.edge_model import build_features, resolve, tradeable_barriers, TRAIN_FRAC

pair="BTCUSDT"; hz,k=336,4.0
b=load(pair); n=len(b["c"]); a=atr(b,14); X,names=build_features(b)
idx=np.arange(400, n-hz-1)
d=k*a[idx]; ok=np.isfinite(d)&(d>0); idx,d=idx[ok],d[ok]
e=b["c"][idx]
rl=resolve(b["h"],b["l"],b["c"],idx,np.ones(len(idx),np.int8),e+2.0*d,e-d,hz)
ys=np.full(len(idx),-1,np.int8); ys[rl==1]=1; ys[rl==-1]=0
keep=ys>=0; idx,ys=idx[keep],ys[keep]
m=HistGradientBoostingClassifier(max_iter=180,learning_rate=0.06,max_depth=4,
    min_samples_leaf=60,l2_regularization=1.0,random_state=7)
m.fit(X[idx],ys)
import joblib, os
os.makedirs("data/models",exist_ok=True)
joblib.dump({"model":m,"features":names,"horizon":hz,"stop_atr":k,
             "trained_on":pair,"bars":int(n)}, "data/models/edge_btc.joblib")
print(f"saved: {len(names)} features, {len(idx)} training rows, "
      f"horizon {hz}h, stop {k}xATR")
p=m.predict_proba(X[idx])[:,1]
print(f"in-sample p_up: mean {p.mean():.3f} sd {p.std():.3f}")
