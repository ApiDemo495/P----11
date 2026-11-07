"""Comparison report: Raw App Signals vs Discretionary Filtered."""
from __future__ import annotations

from .analytics import by_asset, correlated_drawdowns, metrics
from .runner import Trade


def _row(name: str, m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return f"{name:<12} no trades"
    return (f"{name:<12} n={m['n']:<4} win {m['win_pct']:>6}%  "
            f"breakeven {m['breakeven_pct']}%  EV {m['ev_r']:+.3f}R  "
            f"PF {m['profit_factor']}  maxDD {m['max_dd_pct']}%  "
            f"Sharpe {m['sharpe_per_trade']}  PnL {m['total_pnl']:+.2f}")


def comparison_report(raw: list[Trade], filtered: list[Trade]) -> str:
    lines = ["═" * 78,
             "RAW APP SIGNALS (Mode A — every signal taken)",
             "─" * 78]
    for a, m in by_asset(raw).items():
        lines.append(_row(a, m))
    lines.append(_row("COMBINED", metrics(raw)))
    lines += ["", "═" * 78,
              "DISCRETIONARY FILTERED (Mode B — only trades you took)",
              "─" * 78]
    for a, m in by_asset(filtered).items():
        lines.append(_row(a, m))
    lines.append(_row("COMBINED", metrics(filtered)))
    ev_raw, ev_f = metrics(raw).get("ev_r"), metrics(filtered).get("ev_r")
    if ev_raw is not None and ev_f is not None:
        lines += ["",
                  f"Your filter ΔEV: {ev_f - ev_raw:+.3f}R per trade "
                  f"({'your discretion ADDS value' if ev_f > ev_raw else 'your discretion SUBTRACTS value — the raw app beats the filter on this sample'})"]
    events = correlated_drawdowns(raw)
    lines += ["", "─" * 78,
              f"CORRELATED DRAWDOWN (BTC+ETH+BNB simultaneously in losing trades): "
              f"{len(events)} event(s)"]
    for e in events[:10]:
        lines.append(f"  ⚠ t={e['ts']} open losing: {', '.join(e['assets_open_losing'])} x{e['open_count']}")
    return "\n".join(lines)
