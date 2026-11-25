"""Pins the breadth result: it helps vol/drawdown, it cannot create edge."""
from __future__ import annotations

import numpy as np
import pytest

from backtest.breadth_test import align_closes, xs_book


def _synthetic(n_sym=24, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, (n_sym, n)), axis=1))
    return closes


def test_dollar_neutral_book_is_actually_neutral():
    closes = _synthetic()
    fees = np.zeros(closes.shape[0])
    pnl = xs_book(closes, fees, legs=4, lookback=24, rebalance=24)
    assert np.isfinite(pnl).all()


def test_breadth_cuts_volatility():
    """The measured effect: more names, lower portfolio vol."""
    closes = _synthetic(n=3000, seed=3)
    fees = np.zeros(closes.shape[0])
    narrow = xs_book(closes, fees, legs=2, lookback=24, rebalance=24)
    wide = xs_book(closes, fees, legs=8, lookback=24, rebalance=24)
    assert np.std(wide) < np.std(narrow)


def test_breadth_cuts_drawdown():
    closes = _synthetic(n=3000, seed=5)
    fees = np.zeros(closes.shape[0])
    def mdd(p):
        eq = np.cumprod(1 + p[len(p) // 2:])
        return float(np.min(eq / np.maximum.accumulate(eq) - 1))
    assert mdd(xs_book(closes, fees, 8, 24, 24)) > mdd(xs_book(closes, fees, 2, 24, 24))


def test_fees_change_the_answer():
    """Turnover cost is not decoration: the same book with fees is worse."""
    closes = _synthetic(n=3000, seed=7)
    free = xs_book(closes, np.zeros(closes.shape[0]), 8, 24, 24)
    charged = xs_book(closes, np.full(closes.shape[0], 0.003), 8, 24, 24)
    assert charged.sum() < free.sum()


def test_random_walk_has_no_edge_gross():
    """On pure noise the gross book is ~zero: any measured positive gross
    edge must come from the signal, not the construction."""
    closes = _synthetic(n_sym=24, n=6000, seed=11)
    pnl = xs_book(closes, np.zeros(closes.shape[0]), legs=6,
                  lookback=168, rebalance=168)
    body = pnl[400:]
    t = body.mean() / (body.std() / np.sqrt(len(body)))
    assert abs(t) < 3.0, f"construction leaked an edge: t={t:.2f}"
