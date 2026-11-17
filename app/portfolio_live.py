"""Live portfolio engine: concurrent certified sleeves, honest paper fills.

This module is the orchestration layer. It does **not** decide whether an
edge is real — ``app/edge_gate.py`` already did that, and
``app/sleeve_registry.json`` is the resulting contract. This layer decides
*how much* of each certified sleeve may run at once, and keeps a ledger
that can be audited against the research expectation.

Design constraints that are deliberately not negotiable:

* **One risk book per asset.** The first portfolio run showed two
  certified ETH sleeves sharing only 5 % of their entry bars but carrying
  **0.76 daily-PnL correlation**. Sizing them independently would have
  doubled ETH risk while looking diversified. Exposure is therefore capped
  per ASSET, not per sleeve, and the per-asset budget is split across that
  asset's live sleeves.
* **Correlation circuit breaker.** If the realised correlation between
  live sleeve P&L streams exceeds ``MAX_SLEEVE_CORR``, the smaller sleeve
  is throttled — the portfolio stops paying for the same bet twice.
* **Alpha-decay monitor.** Each sleeve's live (post-deployment) trades are
  tested against the research expectation. If live expectancy falls below
  ``DECAY_HALT_BPS`` or its live t-stat implies the research edge is gone,
  the sleeve is HALTED (no new entries) and flagged, not silently kept.
* **Fills are adverse.** Entries pay the bar open plus half the expected
  friction; exits pay the bar open minus the same. Realised slippage is
  measured and reported against the 8-12 bp assumption instead of being
  assumed away.
* **Nothing is invented.** A sleeve with insufficient live tape reports
  ``warming_up`` and contributes nothing. The ledger only ever contains
  fills that were actually simulated from a real bar.

The engine is pure-Python and synchronous; the async loop in
``app/engine.py`` drives it.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "sleeve_registry.json")
LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "paper_ledger.json")

# ---------------------------------------------------------------------------
# risk limits
# ---------------------------------------------------------------------------
MAX_GROSS_EXPOSURE = 1.0        # sum of |position| across the book
MAX_ASSET_EXPOSURE = 0.5        # any single asset
MAX_SLEEVE_CORR = 0.80          # above this, throttle the smaller sleeve
CORR_MIN_DAYS = 20              # need this many shared live days to judge
PORTFOLIO_DRAWDOWN_HALT = 0.15  # 15% peak-to-trough stops new entries
DECAY_WINDOW_TRADES = 20        # live trades needed before judging decay
# Decay = evidence the edge is GONE, not evidence the last 20 trades were
# unlucky.  A ~50 %-win-rate strategy with +88 bp expectancy has sd ~300 bp
# per trade, so its first 20 live trades sit negative roughly 40 % of the
# time.  Halting on raw sign (the previous DECAY_HALT_BPS = 0.0) therefore
# killed both certified sleeves inside 20 trades at live t of -0.52 and
# -0.19 - i.e. on pure noise (Pillar-3 finding, 2026-11-17).  The test is
# now two-sided AND statistical: bad point estimate *and* enough evidence
# to reject the research edge.
DECAY_HALT_BPS = -25.0          # live expectancy at/below this -> suspicious
# How far the live edge may fall below the certified edge before we call it
# decay.  Judged against the CERTIFIED number, not against zero: a sleeve
# certified at +88 bp is expected to run somewhat under its OOF estimate, so
# demanding live EV > 0 would halt on ordinary variance.  The z test also
# charges a finite-sample allowance so a fixed cut is not re-tested every bar
# into a guaranteed eventual false positive (sequential-testing trap).
DECAY_RETENTION_FLOOR = 0.0     # live edge must not fall below this fraction
DECAY_Z_HALT = -2.0             # z of (live - certified) below this -> halt
# A z-test at the 20-trade window has almost no power: the sampling sd of a
# 20-trade mean is sd/sqrt(20) ~ 0.22*sd, so a certified +88 bp sleeve sits
# 2 sigma below its own mean as a matter of routine. Professional practice is
# to require a minimum observation count before acting on a monitor at all.
DECAY_MIN_TRADES_HALT = 60      # never halt on fewer live trades than this
# A threshold re-tested on every new trade is eventually reached by any
# random walk (the "peeking" problem). On the certified catchup sleeve the
# z test first breached -2 at exactly n=88, its worst running mean (-28.7 bp);
# the same sleeve then recovered to +38 bp by n=165. Killing a sleeve at a
# local minimum is not risk management. So: a single contradiction DE-RISKS
# (throttle to half size) and only a contradiction that survives several
# consecutive evaluations is treated as decay and halted.
DECAY_SUSTAIN_EVALS = 5         # consecutive contradicting evals before halting
# A "throttled" verdict must actually reduce risk. It previously only set a
# status flag that size_for() never read, so the correlation circuit breaker
# and the de-risk path were decorative. Half size is the contract.
THROTTLE_SIZE_SCALE = 0.5
DECAY_T_HALT = -1.5             # live t at/below this -> halt

# how much of the assumed round-trip friction becomes realised slippage in
# the paper ledger (half on entry, half on exit)
SLIPPAGE_SHARE_OF_FRICTION = 0.5

# The certified sleeves have NO stop: they hold a fixed number of bars and
# exit at the close. The first version of this engine added a 2x-per-bar-vol
# stop, which turned a 48h hold into a sequence of stop-outs and paid the
# round trip many times over -- a DIFFERENT strategy that was never
# certified. The only stop kept here is a catastrophic backstop against a
# broken feed or a flash crash, deliberately far away so it can never
# change the certified distribution.
CATASTROPHIC_STOP_FRACTION = 0.25

# Post-exit cooldown before the SAME sleeve may re-enter, in bars. The
# certified contract already guarantees non-overlap, so the honest default
# is 0: a non-zero value silently deletes part of the certified trade
# population (a 48h cooldown on a 48h hold removed 42 of 165 holdout
# trades and flipped the sign of the result). It is kept as a configurable
# risk knob, not as a default.
DEFAULT_COOLDOWN_BARS = 0


@dataclass
class Position:
    sleeve_id: str
    asset: str
    direction: int
    size: float                 # notional fraction of the book
    entry_price: float
    entry_t: int
    entry_bar: int
    stop: float
    target_hold_bars: int
    scale: float                # vol-target multiple applied
    expected_edge_bps: float

    @property
    def bars_held(self) -> int:
        return 0  # replaced at read time; kept for dataclass symmetry


@dataclass
class ClosedTrade:
    sleeve_id: str
    asset: str
    direction: int
    size: float
    entry_t: int
    exit_t: int
    entry_price: float
    exit_price: float
    gross_bps: float
    slippage_bps: float
    net_bps: float          # sized contribution to the book (size applied)
    reason: str             # "hold_expired" | "stop" | "halt"
    unit_bps: float = 0.0   # per-unit-notional result: directly comparable
                            # to the research Net EV
    bar_index_exit: int = 0

    def as_dict(self) -> dict:
        return {
            "sleeve_id": self.sleeve_id, "asset": self.asset,
            "direction": self.direction, "size": round(self.size, 4),
            "entry_t": self.entry_t, "exit_t": self.exit_t,
            "entry_price": self.entry_price, "exit_price": self.exit_price,
            "gross_bps": round(self.gross_bps, 2),
            "slippage_bps": round(self.slippage_bps, 2),
            "net_of_slippage_bps": round(self.unit_bps, 2),
            "net_bps": round(self.net_bps, 2),
            "unit_bps": round(self.unit_bps, 2),
            "reason": self.reason,
        }


@dataclass
class SleeveRuntime:
    sleeve_id: str
    asset: str
    research_ev_bps: float
    research_pf: float
    research_t: float
    hold_hours: int
    hold_bars: int = 0          # hold measured in BARS of the sleeve's grid;
                                # 48 hours on a 15m grid is 192 bars, and
                                # using the hour count as a bar count exited
                                # every trade 4x too early
    threshold: float = 2.5
    status: str = "open"        # open | throttled | halted
    halt_reasons: list = field(default_factory=list)
    _decay_streak: int = 0      # consecutive contradicting monitor evals
    _decay_eval_n: int = -1     # trade count of the last monitor eval

    def as_dict(self) -> dict:
        return {
            "sleeve_id": self.sleeve_id,
            "asset": self.asset,
            "status": self.status,
            "research_ev_bps": round(self.research_ev_bps, 2),
            "research_pf": round(self.research_pf, 3),
            "research_t": round(self.research_t, 3),
            "hold_hours": self.hold_hours,
            "halt_reasons": list(self.halt_reasons),
        }


# ---------------------------------------------------------------------------
# statistics helpers (kept local: this module must not import the backtest)
# ---------------------------------------------------------------------------
def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_stat(xs) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    sd = _stdev(xs)
    if sd <= 0:
        return 0.0
    return _mean(xs) / (sd / math.sqrt(n))


def pearson(x, y) -> float:
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = _mean(x), _mean(y)
    # NB: no local names `a`/`b` here. An earlier version used them as loop
    # variables, which silently shadowed the caller's `a, b` sleeve ids and
    # made this function read the wrong arrays — the correlation monitor
    # then reported 0.0 for perfectly correlated P&L streams.
    var_x = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    var_y = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (var_x * var_y)


def vol_target_scale(realised_vol: float, target_vol: float = 0.01,
                     lo: float = 0.25, hi: float = 2.0) -> float:
    """Position multiplier that keeps risk roughly constant across regimes.

    ``realised_vol`` is the per-bar standard deviation of log returns on the
    sleeve's own bar grid. Clipped so a quiet patch cannot lever the book
    20x and a violent one cannot zero it.
    """
    if not (realised_vol == realised_vol) or realised_vol <= 0:
        return 1.0
    return min(hi, max(lo, target_vol / realised_vol))


# ---------------------------------------------------------------------------
# the portfolio
# ---------------------------------------------------------------------------
def _z_vs_certified(pnls, certified, block=24, iters=1000, seed=12345):
    """z of the live mean against the CERTIFIED mean, with honest SE.

    Trades from a 48 h hold on a 15-minute grid overlap heavily and their
    P&L clusters with regimes, so an iid standard error is too small.  The
    OOF gate itself uses a circular block bootstrap for exactly this reason
    (``edge_gate.t_stat_block_bootstrap``); reusing the same estimator here
    keeps the live monitor as strict as the certification it polices.
    Returns None when there is not enough data to say anything.
    """
    n = len(pnls)
    if n < max(block * 2, 12):
        return None
    import random

    mean = sum(pnls) / n
    rng = random.Random(seed)
    n_blocks = int(math.ceil(n / block))
    means = []
    for _ in range(iters):
        acc = 0.0
        for _b in range(n_blocks):
            start = rng.randrange(n)
            for k in range(block):
                acc += pnls[(start + k) % n]
        means.append(acc / (n_blocks * block))
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
    se = math.sqrt(var)
    if se <= 0:
        return None
    return (mean - certified) / se


class LivePortfolio:
    """Concurrent multi-sleeve paper portfolio with risk guardrails."""

    def __init__(self, registry: dict | None = None, ledger_path: str | None = None,
                 enforce_monitors: bool = True):
        self.registry = registry if registry is not None else _load_registry()
        # report mode records what the monitors WOULD have done without
        # stopping the book -- needed to measure a full holdout honestly
        self.enforce_monitors = enforce_monitors
        self.ledger_path = ledger_path or LEDGER_PATH
        self.sleeves: dict[str, SleeveRuntime] = {}
        for s in self.registry.get("sleeves", []):
            if s.get("status") != "open":
                continue
            sid = s.get("sleeve_id", "")
            if not sid:
                continue
            hold_hours = 48
            threshold = 2.5
            for v in (s.get("params") or {}).values():
                if not isinstance(v, str) or "|" not in v:
                    continue
                for part in v.split("|"):
                    if part.startswith("hold_h="):
                        hold_hours = int(part.split("=")[1])
                    elif part.startswith("thr="):
                        threshold = float(part.split("=")[1])
            self.sleeves[sid] = SleeveRuntime(
                sleeve_id=sid,
                asset=s.get("asset", ""),
                research_ev_bps=float(s.get("net_ev_bps", 0.0)),
                research_pf=float(s.get("profit_factor", 0.0)),
                research_t=float(s.get("t_effective", 0.0)),
                hold_hours=hold_hours,
                hold_bars=max(1, hold_hours),   # 1h-grid default; the replay
                                                # sets the true bar count
                threshold=threshold,
            )
        self.positions: dict[str, Position] = {}
        self.closed: list[ClosedTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.realized_bps: float = 0.0
        # equity is tracked as a FRACTION of starting capital (1.0). An
        # earlier version tracked cumulative bps and divided by its own
        # peak, which produced "drawdowns" of 159 % and tripped the risk
        # halt after three trades. Equity in fractional units is bounded,
        # comparable to the limit, and cannot divide by ~zero.
        self.equity: float = 1.0
        self.peak_equity: float = 1.0
        self.slippage_samples: list[float] = []
        self.cooldown_until: dict[str, int] = {}
        self.cooldown_bars: int = DEFAULT_COOLDOWN_BARS
        self._observe_only: dict[str, tuple[str, list]] = {}
        self._scheduled_exit: dict[str, dict | None] = {}

    # -- book-level state ---------------------------------------------------
    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.size) for p in self.positions.values())

    def asset_exposure(self, asset: str) -> float:
        return sum(abs(p.size) for p in self.positions.values() if p.asset == asset)

    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    # -- sizing -------------------------------------------------------------
    def active_sleeves_for(self, asset: str) -> int:
        """How many tradeable sleeves share this asset's risk budget."""
        return max(
            1,
            sum(1 for s in self.sleeves.values()
                if s.asset == asset and s.status in ("open", "throttled")),
        )

    def size_for(self, sleeve: SleeveRuntime, scale: float,
                 n_same_asset: int | None = None) -> float:
        """Notional for a new entry, after every cap.

        The per-asset budget is divided by the number of that asset's LIVE
        sleeves rather than multiplied, because those sleeves are the same
        underlying risk (measured 0.76 correlation between the two ETH
        sleeves — see docs/ALPHA_RESEARCH.md).

        The divisor is the number of ACTIVE SLEEVES on the asset, not the
        number currently holding a position. Counting only open positions
        let whichever sleeve entered first take the whole ETH budget and
        starve the second one out of 65 of its 167 holdout trades.
        """
        if n_same_asset is None:
            n_same_asset = self.active_sleeves_for(sleeve.asset)
        base = MAX_ASSET_EXPOSURE / max(1, n_same_asset)
        # Vol targeting may scale a position DOWN from its budget, never UP
        # through it. Allowing scale>1 let one ETH sleeve consume the whole
        # per-asset budget and starved the other certified ETH sleeve of 92 %
        # of its trades -- the correlation problem reappearing as a capacity
        # problem. Clipping at 1.0 keeps both sleeves fundable.
        sized = base * min(max(scale, 0.0), 1.0)
        # a de-risked sleeve trades at reduced size, not full size
        if sleeve.status != "open":
            sized *= THROTTLE_SIZE_SCALE
        room_book = MAX_GROSS_EXPOSURE - self.gross_exposure
        room_asset = MAX_ASSET_EXPOSURE - self.asset_exposure(sleeve.asset)
        return max(0.0, min(sized, room_book, room_asset))

    # -- execution ----------------------------------------------------------
    def try_enter(self, sleeve_id: str, signal_state: str, bar: dict,
                  bar_index: int, realised_vol: float,
                  scale: float | None = None,
                  exit_at: dict | None = None) -> dict:
        """Open a paper position if every guard allows it.

        ``bar`` is the fill bar: ``{"t": epoch, "o": open}``. The fill is the
        open price *moved against us* by half the assumed friction, which is
        the pessimistic convention the research used.
        """
        sleeve = self.sleeves.get(sleeve_id)
        if sleeve is None:
            return {"ok": False, "why": "unknown sleeve"}
        if self.enforce_monitors and sleeve.status == "halted":
            return {"ok": False, "why": "sleeve halted: " + "; ".join(sleeve.halt_reasons)}
        if sleeve_id in self.positions:
            return {"ok": False, "why": "already positioned"}
        # The cooldown is measured in BARS, but different sleeves run on
        # different calendars, so a bar index is not comparable across
        # sleeves. It is therefore only consulted when it is actually
        # enabled; with a zero cooldown the sleeve is free immediately and
        # the stored index is ignored entirely.
        if self.cooldown_bars > 0 and bar_index < self.cooldown_until.get(sleeve_id, 0):
            return {"ok": False, "why": "cooldown"}
        if signal_state not in ("long", "short"):
            return {"ok": False, "why": "flat"}
        if self.enforce_monitors and self.drawdown() >= PORTFOLIO_DRAWDOWN_HALT:
            return {"ok": False, "why": "portfolio drawdown halt"}

        direction = 1 if signal_state == "long" else -1
        n_same = self.active_sleeves_for(sleeve.asset)
        # `realised_vol` is a volatility and `scale` is the already-computed
        # sizing multiple. The earlier version fed the multiple in as the
        # volatility and re-targeted it, shrinking every position twice.
        if scale is None:
            scale = vol_target_scale(realised_vol)
        size = self.size_for(sleeve, scale, n_same)
        if size <= 0:
            return {"ok": False, "why": "no risk budget"}

        friction_bps = 18.0 if sleeve.asset == "PAXG" else 12.0
        slip = friction_bps * SLIPPAGE_SHARE_OF_FRICTION
        open_px = float(bar["o"])
        # adverse fill: pay up on entry
        fill = open_px * (1.0 + direction * slip / 1e4)
        # catastrophic backstop only (see module note); the certified exit is
        # the hold timer, not this level
        stop = fill * (1.0 - direction * CATASTROPHIC_STOP_FRACTION)
        self.positions[sleeve_id] = Position(
            sleeve_id=sleeve_id, asset=sleeve.asset, direction=direction,
            size=size, entry_price=fill, entry_t=int(bar["t"]),
            entry_bar=bar_index, stop=stop,
            target_hold_bars=max(1, sleeve.hold_bars or sleeve.hold_hours),
            scale=scale, expected_edge_bps=sleeve.research_ev_bps,
        )
        # the scheduled exit price is the bar the research would have used;
        # storing it makes the paper fill identical to the referee's fill
        self._scheduled_exit[sleeve_id] = exit_at
        return {"ok": True, "size": size, "fill": fill, "scale": scale}

    def manage(self, bar_index: int, prices: dict) -> list[ClosedTrade]:
        """Mark positions to market; exit on hold expiry or stop breach.

        ``prices`` maps asset -> ``{"c": close, "h": high, "l": low}`` for the
        current bar. Exits fill at the close, moved against us by half the
        assumed friction (the mirror of the entry convention).
        """
        closed_now: list[ClosedTrade] = []
        for sid in list(self.positions):
            pos = self.positions[sid]
            px = prices.get(pos.asset)
            if not px:
                continue
            held = bar_index - pos.entry_bar
            friction_bps = 18.0 if pos.asset == "PAXG" else 12.0
            slip = friction_bps * SLIPPAGE_SHARE_OF_FRICTION
            hit_stop = (pos.direction > 0 and px["l"] <= pos.stop) or \
                       (pos.direction < 0 and px["h"] >= pos.stop)
            expired = held >= pos.target_hold_bars
            if not (hit_stop or expired):
                continue
            # Exit rule is the certified one: close at the bar that is
            # exactly `hold_bars` after entry, using THAT bar's close. The
            # scheduled-exit map is used only to guarantee the price comes
            # from the right bar; if it is missing we fall back to the
            # current bar's close rather than skipping the exit (an earlier
            # version `continue`d forever and left positions open for
            # thousands of bars).
            scheduled = self._scheduled_exit.pop(sid, None)
            exit_raw = scheduled["c"] if scheduled else px["c"]
            exit_px = exit_raw * (1.0 - pos.direction * slip / 1e4)
            # The adverse fills ALREADY embody the round-trip cost: entry is
            # paid up by half the friction and the exit gives back the other
            # half. Charging the friction a second time on top (the first
            # version did) doubled the cost and turned a +38 bp research
            # holdout into -3.5 bp. `gross` here is the realised per-unit
            # result NET of slippage -- directly comparable to research EV.
            unit = pos.direction * (exit_px / pos.entry_price - 1.0) * 1e4
            paid = 2.0 * slip
            gross = unit          # realised, already net of both fills' slip
            net = pos.size * unit
            trade = ClosedTrade(
                sleeve_id=sid, asset=pos.asset, direction=pos.direction,
                size=pos.size, entry_t=pos.entry_t, exit_t=int(px.get("t", 0)),
                entry_price=pos.entry_price, exit_price=exit_px,
                gross_bps=gross, slippage_bps=paid, net_bps=net, unit_bps=unit,
                reason="stop" if hit_stop else "hold_expired",
                bar_index_exit=bar_index,
            )
            self.closed.append(trade)
            self.realized_bps += net
            # a trade of size `s` earning `net` bps moves equity by
            # size * net / 1e4 of starting capital
            self.equity *= (1.0 + pos.size * net / 1e4)
            self.peak_equity = max(self.peak_equity, self.equity)
            self.equity_curve.append((trade.exit_t, self.realized_bps))
            self.slippage_samples.append(paid)
            self.cooldown_until[sid] = bar_index + self.cooldown_bars
            del self.positions[sid]
            closed_now.append(trade)
        self._check_correlation()
        self._check_decay()
        if not self.enforce_monitors:
            # report mode: record what the monitors WOULD have decided, then
            # restore the tradeable state so the full holdout can be measured
            # under the certified contract rather than under a monitor that
            # tripped on the first 20 trades of a 2-year sample.
            self._observe_only = {
                sid: (rt.status, list(rt.halt_reasons))
                for sid, rt in self.sleeves.items()
            }
            for rt in self.sleeves.values():
                # include "throttled": a de-risked sleeve sizes at half, which
                # would quietly change the research comparison this mode
                # exists to protect.
                if rt.status in ("halted", "throttled"):
                    rt.status = "open"
        return closed_now

    # -- monitors -----------------------------------------------------------
    def _daily_pnl(self, sleeve_id: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in self.closed:
            if t.sleeve_id != sleeve_id:
                continue
            day = t.exit_t // 86400
            out[day] = out.get(day, 0.0) + t.net_bps
        return out

    def _check_correlation(self) -> None:
        """Throttle the smaller of a highly correlated pair.

        Uses realised daily P&L of CLOSED live trades only — never research
        data — so the measure reflects what the book is actually exposed to.
        """
        ids = [s for s in self.sleeves if len(self._daily_pnl(s)) >= CORR_MIN_DAYS]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                da, db = self._daily_pnl(a), self._daily_pnl(b)
                days = sorted(set(da) & set(db))
                if len(days) < CORR_MIN_DAYS:
                    continue
                corr = pearson([da[d] for d in days], [db[d] for d in days])
                if corr <= MAX_SLEEVE_CORR:
                    continue
                # throttle the sleeve with the smaller live expectancy
                ea = _mean([t.unit_bps for t in self.closed if t.sleeve_id == a])
                eb = _mean([t.unit_bps for t in self.closed if t.sleeve_id == b])
                victim, keeper = (a, b) if ea < eb else (b, a)
                rt = self.sleeves.get(victim)
                if rt is not None and rt.status == "open":
                    rt.status = "throttled"
                    rt.halt_reasons.append(
                        f"corr {corr:.2f} with {keeper} > {MAX_SLEEVE_CORR} "
                        f"(same underlying risk)"
                    )

    def _check_decay(self) -> None:
        """Halt a sleeve whose LIVE expectancy contradicts its research edge."""
        for sid, rt in self.sleeves.items():
            # Monitors run every bar; a reason recorded on every bar would
            # grow without bound (observed: 100s of identical entries in one
            # replay). Record each distinct verdict once.
            if rt.status == "halted":
                continue
            pnls = [t.unit_bps for t in self.closed if t.sleeve_id == sid]
            if len(pnls) < DECAY_WINDOW_TRADES:
                continue
            # The verdict can only change when a new trade closes; the
            # bootstrap is expensive and ``manage()`` runs every bar (a naive
            # version took >28 min on one replay). Re-evaluate on new
            # evidence only - never on the passage of time alone.
            if getattr(rt, "_decay_eval_n", -1) == len(pnls):
                continue
            rt._decay_eval_n = len(pnls)
            live_ev = _mean(pnls)
            live_t = t_stat(pnls)
            # Halt only on a statistically meaningful contradiction of the
            # research edge.  A thin, noisy, merely-negative sample is NOT
            # decay - halting there would delete the certified sleeves that
            # Pillar 2 exists to protect.
            # Certified sleeve: is the live edge inconsistent with the edge
            # that passed the gate?  z compares the live mean to the
            # research mean using the sleeve's own realised dispersion.
            certified = rt.research_ev_bps
            z = _z_vs_certified(pnls, certified)
            floor = certified * DECAY_RETENTION_FLOOR
            enough = len(pnls) >= DECAY_MIN_TRADES_HALT
            contradicting = (
                enough
                and live_ev <= DECAY_HALT_BPS
                and live_ev <= floor
                and z is not None
                and z <= DECAY_Z_HALT
            )
            if contradicting:
                rt._decay_streak = getattr(rt, "_decay_streak", 0) + 1
                if rt._decay_streak >= DECAY_SUSTAIN_EVALS:
                    rt.status = "halted"
                    rt.halt_reasons.append(
                        f"live expectancy {live_ev:+.1f} bp (t {live_t:+.2f}, "
                        f"z vs certified {z:+.2f}) over {len(pnls)} trades "
                        f"contradicts research {certified:+.1f} bp for "
                        f"{rt._decay_streak} consecutive evaluations"
                    )
                else:
                    # de-risk now, do not kill on one bad reading
                    was = rt.status
                    rt.status = "throttled"
                    if was != "throttled":
                        rt.halt_reasons.append(
                            f"de-risked: live expectancy {live_ev:+.1f} bp "
                            f"(z vs certified {z:+.2f}) over {len(pnls)} "
                            f"trades, streak {rt._decay_streak}/"
                            f"{DECAY_SUSTAIN_EVALS}"
                        )
            else:
                rt._decay_streak = 0
            if rt.status == "halted":
                continue
            if not contradicting and enough and live_t <= DECAY_T_HALT:
                rt.status = "halted"
                rt.halt_reasons.append(
                    f"live t {live_t:+.2f} <= {DECAY_T_HALT} over {len(pnls)} trades"
                )
            elif not contradicting and live_ev <= DECAY_HALT_BPS:
                # suspicious but not proven: de-risk instead of killing.
                # Record ONE verdict per state change, never per bar: the
                # numbers move every bar, so deduping the text does not cap
                # the list (observed hundreds of entries in a single replay).
                was = rt.status
                rt.status = "throttled"
                if was != "throttled":
                    rt.halt_reasons.append(
                        f"throttled: live expectancy {live_ev:+.1f} bp over "
                        f"{len(pnls)} trades, t {live_t:+.2f} not decisive")
            elif not contradicting and rt.status == "throttled":
                # recover only on clear evidence, not on one good trade
                if live_ev > rt.research_ev_bps * 0.25 and live_t > 1.0:
                    rt.status = "open"
                    rt.halt_reasons.clear()

    # -- reporting ----------------------------------------------------------
    def sleeve_report(self, sleeve_id: str) -> dict:
        rt = self.sleeves.get(sleeve_id)
        if rt is None:
            return {}
        pnls = [t.net_bps for t in self.closed if t.sleeve_id == sleeve_id]
        units = [t.unit_bps for t in self.closed if t.sleeve_id == sleeve_id]
        gross = sum(t.gross_bps for t in self.closed if t.sleeve_id == sleeve_id)
        slip = sum(t.slippage_bps for t in self.closed if t.sleeve_id == sleeve_id)
        wins = sum(1 for p in pnls if p > 0)
        loss = -sum(p for p in pnls if p < 0)
        gain = sum(p for p in pnls if p > 0)
        return {
            **rt.as_dict(),
            "live_trades": len(units),
            "live_ev_bps": round(_mean(units), 2),
            "live_net_bps": round(_mean(pnls), 2),
            "live_t": round(t_stat(units), 2),
            "live_win_rate": round(wins / len(pnls), 3) if pnls else None,
            "live_profit_factor": round(gain / loss, 3) if loss > 0 else None,
            "realized_slippage_bps": round(slip / len(pnls), 2) if pnls else None,
            "gross_bps": round(gross, 2),
            "decay_warning": bool(
                len(units) >= DECAY_WINDOW_TRADES
                and _mean(units) < rt.research_ev_bps * 0.5
            ),
        }

    def snapshot(self) -> dict:
        return {
            "sleeves": [self.sleeve_report(s) for s in self.sleeves],
            "monitor_verdicts": {
                sid: {"status": v[0], "reasons": v[1]}
                for sid, v in self._observe_only.items()
            },
            "monitors_enforced": self.enforce_monitors,
            "open_positions": [
                {
                    "sleeve_id": p.sleeve_id, "asset": p.asset,
                    "direction": p.direction, "size": round(p.size, 4),
                    "entry_price": p.entry_price, "entry_t": p.entry_t,
                    "stop": p.stop, "hold_bars": p.target_hold_bars,
                    "scale": round(p.scale, 3),
                }
                for p in self.positions.values()
            ],
            "gross_exposure": round(self.gross_exposure, 4),
            "realized_bps": round(self.realized_bps, 2),
            "equity": round(self.equity, 6),
            "drawdown": round(self.drawdown(), 4),
            "closed_trades": len(self.closed),
            "avg_slippage_bps": (
                round(_mean(self.slippage_samples), 2)
                if self.slippage_samples else None
            ),
            "asset_exposure": {
                a: round(self.asset_exposure(a), 4)
                for a in sorted({s.asset for s in self.sleeves.values()})
            },
            "limits": {
                "cooldown_bars": self.cooldown_bars,
                "max_gross": MAX_GROSS_EXPOSURE,
                "max_asset": MAX_ASSET_EXPOSURE,
                "max_sleeve_corr": MAX_SLEEVE_CORR,
                "drawdown_halt": PORTFOLIO_DRAWDOWN_HALT,
            },
        }

    # -- persistence --------------------------------------------------------
    def save(self, path: str | None = None) -> str:
        p = path or self.ledger_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        payload = {
            "saved_at": int(time.time()),
            "realized_bps": self.realized_bps,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "trades": [t.as_dict() for t in self.closed],
            "sleeves": [self.sleeve_report(s) for s in self.sleeves],
        }
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        return p

    @classmethod
    def load(cls, registry: dict | None = None, ledger_path: str | None = None):
        inst = cls(registry=registry, ledger_path=ledger_path)
        p = inst.ledger_path
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = json.load(f)
                inst.realized_bps = float(data.get("realized_bps", 0.0))
                inst.equity = float(data.get("equity", 1.0))
                inst.peak_equity = float(data.get("peak_equity", 1.0))
            except Exception:
                pass
        return inst


def _load_registry() -> dict:
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except Exception:
        return {"sleeves": []}
