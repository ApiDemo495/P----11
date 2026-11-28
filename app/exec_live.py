"""Verified LIVE paper execution for the certified sleeves.

Why this module exists
----------------------
The paper book used to fill on the latest one-minute *close* while claiming
a next-15-minute-bar open, and could consume simulated tape. Rather than
ship a book whose fills the research fill contract cannot reproduce, the
engine froze it: ``_step_portfolio`` returns the ledger with
``execution_status = "paused"`` and the reason "next-bar open and real-tape
provenance are not yet verified".

That is honest, but it leaves the certified sleeves measuring nothing. This
module supplies exactly the two missing guarantees, so execution can resume
under a contract the research *does* reproduce:

1. **Next-bar open.** A signal is only actionable on a bar that has already
   CLOSED. The fill is the OPEN of the bar after it — never the close of the
   bar that produced the signal.
2. **Real-tape provenance.** Every bar used must be Binance-sourced,
   epoch-aligned, gap-free and complete. Simulated, unknown or mixed tape is
   refused outright — no fills are invented from a simulator.

Plus the two rules that make the paper record comparable to the research
ledger: the fill is moved AGAINST us by half the round-trip friction (the
same pessimistic convention ``backtest/paper_replay.py`` uses), and the exit
is the OPEN of the bar exactly ``hold_bars`` later, or the catastrophic stop
if it is breached first.

Nothing here decides *whether* to trade — ``app/sleeves.py`` owns the
certified trigger and ``LivePortfolio.try_enter`` owns the risk guards. This
module only decides whether a fill can be honestly claimed, and at what
price.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

log = logging.getLogger("exec_live")

#: The research grid the certified triggers were measured on.
BAR_SECONDS = 900

#: Fraction of the round trip paid on each side (entry pays up, exit gives
#: back). Half on each side sums to exactly one round trip.
SLIPPAGE_SHARE = 0.5

#: A bar whose newest tick is older than this cannot price a fill.
MAX_BAR_AGE_S = 2.0 * BAR_SECONDS

#: Bars allowed between the signal bar and the fill bar — one, by contract.
FILL_DELAY_BARS = 1


def _family(source: str) -> str:
    try:
        from .feed import _family as f
        return f(source)
    except Exception:
        return "unknown"


def resample_ohlc(times: list, opens: list, highs: list, lows: list,
                  closes: list, sources: list | None = None,
                  bucket: int = BAR_SECONDS) -> tuple[list[dict], str]:
    """Aggregate a 1-minute tape onto the research bar grid.

    Returns ``(bars, why)``. ``bars`` is empty when the tape cannot support
    an honest fill, and ``why`` always says which rule failed. A bar is
    emitted ONLY when every one-minute slot inside it is present, so a gap
    can never be papered over by aggregating whatever survived.
    """
    n = len(times or [])
    if n == 0:
        return [], "empty tape"
    if not (len(opens) == len(highs) == len(lows) == len(closes) == n):
        return [], "OHLC series disagree in length"
    if bucket <= 0 or bucket % 60:
        return [], f"bucket {bucket}s is not a whole number of minutes"
    src = list(sources or [])
    if src and len(src) != n:
        return [], "source series disagrees in length with the tape"
    slots_needed = bucket // 60
    out: list[dict] = []
    prev_t = None
    cur_slot = None
    rows: list[tuple[int, float, float, float, float, str]] = []

    def flush() -> bool:
        """True when the current slot was complete and has been emitted."""
        if cur_slot is None or not rows:
            return False
        if len(rows) != slots_needed:
            return False
        for k, r in enumerate(rows):
            if int(r[0]) != int(cur_slot) + k * 60:
                return False
        o = rows[0][1]
        c = rows[-1][4]
        h = max(r[2] for r in rows)
        lo = min(r[3] for r in rows)
        if not all(math.isfinite(v) and v > 0 for v in (o, h, lo, c)):
            return False
        if h < max(o, c) or lo > min(o, c):
            return False
        out.append({"t": int(cur_slot), "o": o, "h": h, "l": lo, "c": c,
                    "source": rows[0][5]})
        return True

    for i in range(n):
        try:
            t = int(times[i])
            o, h, lo, c = (float(opens[i]), float(highs[i]),
                           float(lows[i]), float(closes[i]))
        except (TypeError, ValueError, OverflowError):
            return [], f"non-numeric bar at index {i}"
        if t % 60:
            return [], f"bar at {t} is not minute-aligned"
        if prev_t is not None and t != prev_t + 60:
            return [], (f"tape gap between {prev_t} and {t} "
                        f"({t - prev_t}s) — a fill cannot be priced across it")
        prev_t = t
        slot = t - (t % bucket)
        if cur_slot is not None and slot != cur_slot:
            flush()
            rows = []
        cur_slot = slot
        rows.append((t, o, h, lo, c, src[i] if src else "unknown"))
    # THE LAST BUCKET IS ALWAYS DROPPED: it may still be forming, and a
    # half-built bar is exactly what the old live path filled against.
    return out, ""


def tape_provenance(feed, symbol: str) -> tuple[bool, str]:
    """Can this asset's tape price a certified fill at all?

    Returns ``(ok, why)``. ``ok`` is true only for a real Binance tape whose
    every minute is present and correctly stamped. Anything else is refused
    with the reason, never silently downgraded to "close enough".
    """
    if getattr(feed, "mode", None) != "binance":
        return False, (f"feed mode is {getattr(feed, 'mode', '?')} — simulated "
                       f"tape is excluded from certified fills")
    try:
        st = feed.state(symbol)
    except Exception as exc:
        return False, f"feed state unavailable: {type(exc).__name__}"
    times = list(getattr(st, "times", []) or [])
    sources = list(getattr(st, "sources", []) or [])
    if not times:
        return False, "no bars on the tape yet"
    if len(sources) != len(times):
        return False, "source series disagrees with the tape"
    foreign = [s for s in sources if _family(s) != "binance"]
    if foreign:
        return False, (f"{len(foreign)} bar(s) of non-Binance provenance on "
                       f"the tape (e.g. {foreign[0]})")
    if times[-1] % 60:
        return False, "newest bar is not minute-aligned"
    return True, ""


def next_bar_open(bars: list[dict], signal_bar_t: int,
                  now: float | None = None) -> tuple[dict | None, str]:
    """The bar the fill must use: the OPEN of the bar AFTER the signal bar.

    This is the rule the research ledger and ``backtest/paper_replay.py``
    fill on. Returns ``(bar, why)``; ``bar`` is None while that bar has not
    closed yet (the signal is pending, not filled early).
    """
    want = int(signal_bar_t) + FILL_DELAY_BARS * BAR_SECONDS
    for b in bars:
        if int(b["t"]) == want:
            # the fill bar must itself be fully closed on the wall clock
            stamped = float(now if now is not None else time.time())
            if stamped < want + BAR_SECONDS:
                return None, (f"next bar {want} is still forming — the fill "
                              f"waits for its close, never fills early")
            return b, ""
    return None, (f"next bar {want} has not been printed yet "
                  f"(the signal is pending, not filled)")


def step_live_book(engine, block: dict, now: float | None = None) -> dict:
    """Advance the paper book on VERIFIED next-bar-open fills.

    Called once per engine tick. Returns the portfolio snapshot with an
    explicit ``execution_status`` that is one of:

      * ``live``   — fills are being made under the verified contract;
      * ``paused`` — with the exact rule that is failing, per asset.

    A paused book is never silently paused: every reason is a named,
    checkable fact about the tape (feed mode, provenance, gaps, a bar that
    has not closed). That is the difference between "we do not fill" and
    "we do not know why we do not fill".
    """
    import time as _t
    now = float(now if now is not None else _t.time())
    reasons: dict[str, str] = {}
    filled: list[dict] = []
    book = engine.portfolio

    for symbol in ("ETH", "BTC", "BNB", "SOL"):
        ok, why = tape_provenance(engine.feed, symbol)
        if not ok:
            reasons[symbol] = why
            continue
        try:
            st = engine.feed.state(symbol)
            bars, why = resample_ohlc(
                list(st.times), list(st.opens), list(st.highs),
                list(st.lows), list(st.closes),
                list(getattr(st, "sources", []) or []))
        except Exception as exc:
            reasons[symbol] = f"tape resample failed: {type(exc).__name__}"
            continue
        if why:
            reasons[symbol] = why
            continue
        if len(bars) < 2:
            reasons[symbol] = "fewer than two complete research bars"
            continue
        # the signal bar is the newest CLOSED research bar
        sig_bar = bars[-2]
        fill_bar, why = next_bar_open(bars, int(sig_bar["t"]), now=now)
        if fill_bar is None:
            reasons[symbol] = why
            continue
        # marks and exits are managed by the portfolio on the same bar
        prices = {symbol: {"t": fill_bar["t"], "o": fill_bar["o"],
                           "c": fill_bar["c"], "h": fill_bar["h"],
                           "l": fill_bar["l"]}}
        try:
            closed = book.manage(int(fill_bar["t"] // BAR_SECONDS), prices)
        except Exception:
            log.exception("book.manage failed for %s", symbol)
            closed = []
        if closed:
            filled.extend({"kind": "exit", "asset": symbol,
                           "n": len(closed)})
        for sig in (block.get("signals") or []):
            if sig.get("asset") != symbol or sig.get("status") != "open":
                continue
            state = str(sig.get("state") or "")
            if state not in ("long", "short"):
                continue
            try:
                res = book.try_enter(
                    str(sig.get("sleeve_id")),
                    state,
                    {"t": fill_bar["t"], "o": fill_bar["o"]},
                    int(fill_bar["t"] // BAR_SECONDS),
                    0.0)
            except Exception:
                log.exception("book.try_enter failed for %s", symbol)
                continue
            if res.get("ok"):
                filled.append({"kind": "entry", "asset": symbol,
                               "sleeve": sig.get("sleeve_id"),
                               "fill": res.get("fill"),
                               "size": res.get("size"),
                               "bar_t": fill_bar["t"]})

    snap = book.snapshot()
    snap["execution_status"] = "live" if not reasons else "paused"
    if reasons:
        snap["execution_reason"] = (
            "Paper execution paused on the assets that cannot prove a fill: "
            + " · ".join(f"{k}: {v}" for k, v in reasons.items()))
    else:
        snap["execution_reason"] = (
            "fills use the next research-bar OPEN on a gap-free, fully "
            "Binance-sourced tape and are moved against us by half the "
            "round trip — the same convention the research ledger and "
            "backtest/paper_replay.py use")
    snap["execution_contract"] = {
        "fill": "next 15m bar open (never the close of the signal bar)",
        "provenance": "Binance only — simulated/unknown/mixed tape refused",
        "gaps": "any minute missing inside a bar voids that bar",
        "slippage": "half the round trip on each side",
        "exit": "open of the bar hold_bars later, or the catastrophic stop",
    }
    snap["paused_assets"] = reasons
    snap["entered_this_bar"] = filled
    return snap
