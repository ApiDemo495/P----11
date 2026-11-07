"""Mode A — automated historical backtest (app only).

Walks each asset's series; every `step` bars the predictor sees a
DataSlice up to t and may open one trade, settled by first touch of
TP/SL over the next `horizon` bars (SL wins a same-bar tie — the
conservative rule). Friction is charged per asset in bps of entry
and converted into R so results stay comparable across assets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import FRICTION_BPS, Bar
from .predictor import HORIZON, AppPredictor, Signal
from .slice import DataSlice


@dataclass
class Trade:
    asset: str
    ts: int
    direction: str        # Long / Short
    entry: float
    sl: float
    tp: float
    exit_ts: int
    exit_reason: str      # TP / SL / TIMEOUT
    r_gross: float
    r_net: float
    pnl: float            # quote PnL per 1 unit, after friction
    tier: str = ""
    source: str = "raw"   # raw (Mode A) / discretionary (Mode B)


def settle(asset: str, sig: Signal, entry_bar: Bar,
           future: list[Bar]) -> Trade:
    risk = abs(sig.entry - sig.sl)
    if risk <= 0:
        raise ValueError("zero risk")
    s = sig.direction
    exit_price, reason, exit_bar = future[-1].c, "TIMEOUT", future[-1]
    for b in future:
        hit_sl = b.l <= sig.sl if s > 0 else b.h >= sig.sl
        hit_tp = b.h >= sig.tp if s > 0 else b.l <= sig.tp
        if hit_sl:                       # SL wins ties (conservative)
            exit_price, reason, exit_bar = sig.sl, "SL", b
            break
        if hit_tp:
            exit_price, reason, exit_bar = sig.tp, "TP", b
            break
    r_gross = s * (exit_price - sig.entry) / risk
    cost_quote = sig.entry * FRICTION_BPS[asset] / 1e4      # per unit
    r_net = r_gross - cost_quote / risk
    return Trade(asset, entry_bar.t, "Long" if s > 0 else "Short",
                 sig.entry, sig.sl, sig.tp, exit_bar.t, reason,
                 round(r_gross, 4), round(r_net, 4),
                 round(r_net * risk, 4), sig.tier)


def run_mode_a(series: dict[str, list[Bar]], predictor: AppPredictor,
               warmup: int = 80, step: int = HORIZON,
               horizon: int = HORIZON) -> list[Trade]:
    trades: list[Trade] = []
    for asset, bars in series.items():
        ds = DataSlice(bars)
        t = warmup
        while t + horizon < len(bars):
            sig = predictor.predict(ds.at(t), asset)
            if sig is not None:
                trades.append(settle(asset, sig, bars[t],
                                     ds.future(t, t + horizon + 1)))
                t += horizon            # one position at a time
            else:
                t += step
    return trades


def trades_to_csv(trades: list[Trade], path: str) -> str:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys())
                           if trades else [f.name for f in Trade.__dataclass_fields__.values()])
        w.writeheader()
        for tr in trades:
            w.writerow(asdict(tr))
    return path
