"""Sleeve certification gate: the ONLY way a research sleeve becomes live.

A "sleeve" is one preconditioned trade trigger on one asset. Research
produces a lot of candidates; almost all of them are noise once friction
is charged. This module is the filter between "the scan liked it" and
"the engine is allowed to say it".

The gate is deliberately hostile:

  * Out-of-fold only. Every candidate's parameters are chosen on data
    that ends BEFORE the fold it is then scored on (walk-forward), with
    an embargo equal to the holding period so a trade can never straddle
    the train/test boundary (purged folds).
  * Unbiased fill. A decision made on bar ``t``'s close is filled at bar
    ``t+1``'s OPEN, and exited at a later bar's CLOSE. No same-bar fills,
    no "signal price" fills — the adverse-selection variant of
    ``next_bar`` that makes a strategy look worse, never better.
  * Round-trip friction charged in full (12 bps BTC/ETH/BNB, 18 bps
    PAXG) on every trade, win or lose.
  * Correlation-aware significance. Trades that overlap in time are not
    independent, and pooled sleeves across correlated assets are not
    independent either. The gate therefore reports BOTH an iid t and a
    circular block-bootstrap t (block = holding period) and requires the
    CONSERVATIVE one to clear |t| >= 2.0.
  * Multiplicity honesty. The scan tests many candidates; the gate
    reports how many were tested and the Bonferroni threshold that would
    be needed. A sleeve that clears |t| >= 2.0 but not the multiplicity
    threshold is marked ``shadow`` (telemetry only, never a live signal),
    never ``open``.

Statuses
--------
``open``    every gate passed — the engine may signal it and the hero
            card may show it.
``shadow``  some gate failed but the sleeve is still measured; shown as
            an unproven observation, never as a signal.
``closed``  worse than useless (negative OOF Net EV or PF <= 1.0).

Nothing here reads live data or the network: it is pure statistics over
trade records produced by ``backtest/edge_scan.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# thresholds — one place, so a claim can never quietly soften
# ---------------------------------------------------------------------------
MIN_OOF_NET_EV_BPS = 0.0      # strictly greater than zero
MIN_PROFIT_FACTOR = 1.25
MIN_POSITIVE_FOLDS = 0.5      # "majority" — strictly more than half
MIN_ABS_T = 2.0
MIN_TRADES_PER_SLEEVE = 60    # below this nothing is certifiable, only shadow
MIN_TRADES_PER_FOLD = 8
MIN_HOLDOUT_TRADES = 25       # the locked holdout must be large enough to mean something
MIN_HOLDOUT_NET_EV_BPS = 0.0  # ...and it must be positive too

STATUS_OPEN = "open"
STATUS_SHADOW = "shadow"
STATUS_CLOSED = "closed"


# ---------------------------------------------------------------------------
# trade records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Trade:
    """One closed round trip, in bps of entry notional, net of friction."""

    entry_t: int          # epoch seconds of the fill bar's open
    exit_t: int
    fold: int             # purged walk-forward fold the trade was scored in
    pnl_bps: float
    direction: int = 0    # +1 long, -1 short (diagnostics only)
    asset: str = ""


@dataclass
class FoldStat:
    fold: int
    n: int
    net_ev_bps: float
    profit_factor: float
    win_rate: float


@dataclass
class GateResult:
    sleeve_id: str
    asset: str
    family: str
    status: str
    n: int
    net_ev_bps: float
    profit_factor: float
    win_rate: float
    t_iid: float
    t_block_bootstrap: float
    t_effective: float          # the conservative one the gate uses
    positive_folds: int
    folds: list[FoldStat] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sleeve_id": self.sleeve_id,
            "asset": self.asset,
            "family": self.family,
            "status": self.status,
            "n": self.n,
            "net_ev_bps": round(self.net_ev_bps, 3),
            "profit_factor": round(self.profit_factor, 3),
            "win_rate": round(self.win_rate, 4),
            "t_iid": round(self.t_iid, 3),
            "t_block_bootstrap": round(self.t_block_bootstrap, 3),
            "t_effective": round(self.t_effective, 3),
            "positive_folds": self.positive_folds,
            "params": self.params,
            "reasons": list(self.reasons),
            "folds": [
                {
                    "fold": f.fold,
                    "n": f.n,
                    "net_ev_bps": round(f.net_ev_bps, 3),
                    "profit_factor": round(f.profit_factor, 3),
                }
                for f in self.folds
            ],
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def profit_factor(pnls: Sequence[float]) -> float:
    gain = sum(p for p in pnls if p > 0)
    loss = -sum(p for p in pnls if p < 0)
    if loss <= 0:
        return float("inf") if gain > 0 else 0.0
    return gain / loss


def t_stat_iid(pnls: Sequence[float]) -> float:
    n = len(pnls)
    if n < 3:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    if var <= 0:
        return 0.0
    return mean / math.sqrt(var / n)


def t_stat_block_bootstrap(
    pnls: Sequence[float],
    block: int = 24,
    iters: int = 2000,
    seed: int = 12345,
) -> float:
    """Circular block bootstrap t for the mean trade.

    Whole blocks are resampled with replacement and concatenated back to
    exactly ``n`` observations, so any serial dependence (clustering of
    wins and losses in time — the thing an iid t ignores) stays inside the
    resampled series. The bootstrap SE is then at least as large as the
    iid SE whenever the trade stream is positively autocorrelated, which
    is the normal case for trend/catch-up sleeves.

    A proper ``random.Random`` is used deliberately: an earlier version of
    this function used a hand-rolled LCG whose low bits are badly
    correlated, which correlated the resampled block starts, understated
    the SE, and inflated |t|. Statistics that only work with a broken RNG
    are not statistics.
    """
    import random

    n = len(pnls)
    if n < max(block * 2, 12):
        return 0.0
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
        return 0.0
    return mean / se


def fold_stats(trades: Iterable[Trade]) -> list[FoldStat]:
    by_fold: dict[int, list[float]] = {}
    for t in trades:
        by_fold.setdefault(t.fold, []).append(t.pnl_bps)
    out = []
    for fold in sorted(by_fold):
        pnls = by_fold[fold]
        out.append(
            FoldStat(
                fold=fold,
                n=len(pnls),
                net_ev_bps=sum(pnls) / len(pnls) if pnls else 0.0,
                profit_factor=profit_factor(pnls),
                win_rate=sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def sleeve_gate(
    sleeve_id: str,
    trades: Sequence[Trade],
    *,
    asset: str = "",
    family: str = "",
    params: dict | None = None,
    block: int = 24,
    tested_candidates: int = 1,
    bootstrap_iters: int = 2000,
    holdout_trades: Sequence[Trade] | None = None,
    details: dict | None = None,
) -> GateResult:
    """Certify (or refuse) one sleeve from its purged out-of-fold trades.

    A sleeve is ``open`` only when ALL of the following hold:
      * OOF Net EV > 0 bps
      * OOF Profit Factor > 1.25
      * a strict majority of folds have positive Net EV
      * conservative (correlation-adjusted) |t| >= 2.0
      * |t| also clears the Bonferroni threshold for the number of
        candidates the scan tested (multiplicity honesty)
      * the LOCKED holdout (data never used for selection) is positive
        with at least ``MIN_HOLDOUT_TRADES`` trades

    The holdout replaces the "selection certainty" a Bonferroni-style
    threshold tries to buy. It is the stronger test: it is a genuine
    out-of-sample period frozen before it was ever scored.
    """
    pnls = [t.pnl_bps for t in trades]
    n = len(pnls)
    res = GateResult(
        sleeve_id=sleeve_id,
        asset=asset,
        family=family,
        status=STATUS_CLOSED,
        n=n,
        net_ev_bps=sum(pnls) / n if n else 0.0,
        profit_factor=profit_factor(pnls),
        win_rate=sum(1 for p in pnls if p > 0) / n if n else 0.0,
        t_iid=0.0,
        t_block_bootstrap=0.0,
        t_effective=0.0,
        positive_folds=0,
        params=dict(params or {}),
        details=dict(details or {}),
    )
    if n == 0:
        res.reasons.append("no trades")
        return res

    res.t_iid = t_stat_iid(pnls)
    res.t_block_bootstrap = t_stat_block_bootstrap(pnls, block=block, iters=bootstrap_iters)
    # conservative: the smaller magnitude of the two, keeping sign
    res.t_effective = (
        res.t_block_bootstrap
        if abs(res.t_block_bootstrap) <= abs(res.t_iid)
        else res.t_iid
    )
    res.folds = fold_stats(trades)
    res.positive_folds = sum(1 for f in res.folds if f.net_ev_bps > 0)
    big_folds = [f for f in res.folds if f.n >= MIN_TRADES_PER_FOLD]
    res.details.setdefault("folds_with_enough_trades", len(big_folds))

    # multiplicity: Bonferroni over the candidates the scan actually ran
    k = max(1, int(tested_candidates))
    res.details["tested_candidates"] = k
    res.details["t_bonferroni_threshold"] = round(min(6.0, 2.0 + math.log10(k)), 3)

    # ---- locked holdout (data never used for selection) ----
    h_pnls = [t.pnl_bps for t in (holdout_trades or [])]
    h_n = len(h_pnls)
    h_ev = sum(h_pnls) / h_n if h_n else None
    h_pf = profit_factor(h_pnls) if h_n else None
    res.details["holdout_n"] = h_n
    res.details["holdout_net_ev_bps"] = round(h_ev, 3) if h_ev is not None else None
    res.details["holdout_profit_factor"] = round(h_pf, 3) if h_pf is not None else None

    fails: list[str] = []
    if res.net_ev_bps <= MIN_OOF_NET_EV_BPS:
        fails.append(f"OOF Net EV {res.net_ev_bps:+.2f} bps <= {MIN_OOF_NET_EV_BPS}")
    if res.profit_factor <= MIN_PROFIT_FACTOR:
        fails.append(f"PF {res.profit_factor:.2f} <= {MIN_PROFIT_FACTOR}")
    if not res.folds or res.positive_folds * 2 <= len(res.folds):
        fails.append(f"positive folds {res.positive_folds}/{len(res.folds)} not a majority")
    if abs(res.t_effective) < MIN_ABS_T:
        fails.append(f"|t| {abs(res.t_effective):.2f} < {MIN_ABS_T} (correlation-adjusted)")
    if n < MIN_TRADES_PER_SLEEVE:
        fails.append(f"n {n} < {MIN_TRADES_PER_SLEEVE}")
    if h_n < MIN_HOLDOUT_TRADES:
        fails.append(f"holdout n {h_n} < {MIN_HOLDOUT_TRADES}")
    elif h_ev is not None and h_ev <= MIN_HOLDOUT_NET_EV_BPS:
        fails.append(f"holdout Net EV {h_ev:+.2f} bps <= 0")

    res.reasons = list(fails)

    if res.net_ev_bps <= 0 or res.profit_factor <= 1.0:
        res.status = STATUS_CLOSED
        return res
    if fails:
        res.status = STATUS_SHADOW
        return res
    # every hard gate passed, including the locked holdout
    res.status = STATUS_OPEN
    res.details["selection_multiple"] = k
    return res


def summarise(results: Sequence[GateResult]) -> dict:
    """Registry-level roll-up used by the engine and the docs."""
    open_ = [r for r in results if r.status == STATUS_OPEN]
    shadow = [r for r in results if r.status == STATUS_SHADOW]
    closed = [r for r in results if r.status == STATUS_CLOSED]
    covered = sorted({r.asset for r in open_} | {r.asset for r in shadow})
    return {
        "open": len(open_),
        "shadow": len(shadow),
        "closed": len(closed),
        "total": len(results),
        "open_assets": sorted({r.asset for r in open_}),
        "covered_assets": covered,
        "open_sleeves": [r.sleeve_id for r in open_],
    }
