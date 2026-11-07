"""Performance analytics (definitions printed in every report).

Sizing model for DD/Sharpe: fixed-fractional — each trade risks
1% of a 10,000 quote starting equity, sequentially. Win% and EV are
on NET R (after friction). Breakeven win rate = avgLoss/(avgWin +
avgLoss) in R. Profit factor = sum(winning R)/|sum(losing R)|.
Sharpe is per-trade (mean/std of net R) — compare like with like.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev

from .runner import Trade

START_EQUITY = 10_000.0
RISK_FRAC = 0.01


def metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0}
    rs = [t.r_net for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    aw = mean(wins) if wins else 0.0
    al = abs(mean(losses)) if losses else 0.0
    eq, peak, maxdd = START_EQUITY, START_EQUITY, 0.0
    for t in sorted(trades, key=lambda x: x.exit_ts):
        eq += t.r_net * RISK_FRAC * eq       # risk 1% of equity at entry
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak * 100 if peak > 0 else 0.0)
    return {
        "n": len(trades),
        "win_pct": round(100 * len(wins) / len(rs), 2),
        "breakeven_pct": round(100 * al / (aw + al), 2) if (aw + al) > 0 else None,
        "ev_r": round(mean(rs), 4),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 3) if losses and sum(losses) != 0 else None,
        "max_dd_pct": round(maxdd, 2),
        "sharpe_per_trade": round(mean(rs) / pstdev(rs), 3) if len(rs) > 1 and pstdev(rs) > 0 else None,
        "total_pnl": round(sum(t.pnl for t in trades), 2),
    }




def by_asset(trades: list[Trade]) -> dict[str, dict]:
    out: dict[str, list[Trade]] = {}
    for t in trades:
        out.setdefault(t.asset, []).append(t)
    return {a: metrics(ts) for a, ts in sorted(out.items())}


def correlated_drawdowns(trades: list[Trade],
                         assets=("BTC", "ETH", "BNB")) -> list[dict]:
    """Flag moments where >=3 of BTC/ETH/BNB were simultaneously in
    open, eventually-losing trades — correlated drawdown spikes."""
    losers = [t for t in trades if t.asset in assets and t.r_net < 0]
    events = []
    points = sorted({t.ts for t in losers} | {t.exit_ts for t in losers})
    for p in points:
        open_now = [t.asset for t in losers if t.ts <= p <= t.exit_ts]
        if len(open_now) >= 3:
            events.append({"ts": p, "assets_open_losing": sorted(set(open_now)),
                           "open_count": len(open_now)})
    # collapse consecutive points into one event
    collapsed = []
    for e in events:
        if not collapsed or e["ts"] - collapsed[-1]["ts"] > 3600:
            collapsed.append(e)
    return collapsed
