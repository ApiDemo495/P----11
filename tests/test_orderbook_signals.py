"""Order-book layer acceptance (user: 'make it deep — order book things').

F32 DBI · F33 MPD · F34 CWA · F35 LNP — measured depth/liquidation
signals enter the WEIGHTED CCI (count updated31→35 components), graded
like every other input. Cold feed →0 + cause-6 reason, never silent.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.book import DepthBook

ROOT = Path(__file__).resolve().parent.parent


def _pump(bk: DepthBook, n: int = 12, ask_cut: float = 0.0) -> None:
    """Feed n snapshots: bid side stable, ask side losing `ask_cut` per
    level each pass with NO buys (→ ask-side cancellations dominate)."""
    import time as _t
    for i in range(n):
        # bids shed a LITTLE (steady two-sided flow), asks shed a LOT
        # when ask_cut>0 → Λ = ln(ask cancels / bid cancels) > 0
        bids = [[str(100.0 - k * 0.1),
                 str(round(max(0.1, 5.0 - i * 0.08 - k * 0.01), 3))]
                for k in range(5)]
        asks = [[str(100.1 + k * 0.1),
                 str(round(max(0.1, 5.0 - ask_cut - i * 0.3 + k), 3))]
                for k in range(5)]
        bk.on_depth("BTC", bids, asks, int(_t.time() * 1000))
        bk.on_aggtrade("BTC", int(_t.time() * 1000), 0.01,
                       seller_aggressor=False)   # buys consume asks (fills)


def test_depthbook_status_fields():
    bk = DepthBook()
    cold = bk.status("BTC", 100.0)
    assert cold["book"] is False and "depth20" in cold["reason"]
    _pump(bk, n=12, ask_cut=0.5)
    st = bk.status("BTC", 100.0)
    assert st["book"] is True
    # ask side shed more size than bids → DBI positive (bid-heavy)
    assert st["dbi"] is not None and st["dbi"] > 0
    assert st["spread_bps"] and st["spread_bps"] > 0
    assert st["micro_pos"] is not None and -1.0 <= st["micro_pos"] <= 1.0
    assert st["snapshots"] >= DepthBook.SNAP_MIN
    assert st["wall_bid"] and st["wall_bid"]["price"] == pytest.approx(100.0)
    assert st["wall_ask"] and st["wall_ask"]["price"] > 100.0
    # Λ measurable: ask cancels > bid cancels → ln(ask/bid) > 0
    assert st["lambda"] is not None and st["lambda"] > 0
    assert st["v_cancel_ask"] > st["v_cancel_bid"]


def test_depthbook_rings_feed_formulas():
    from app.formulas.f32_f35 import f32_dbi, f33_mpd, f34_cwa, f35_lnp
    bk = DepthBook()
    # cold ctx → every formula reads0 (EVENT_REASON carries the reason)
    cold = SimpleNamespace(book=bk.view("BTC"), liq_flow=None)
    assert f32_dbi(cold) == 0.0 and f33_mpd(cold) == 0.0
    assert f34_cwa(cold) == 0.0 and f35_lnp(cold) == 0.0
    from app.formulas.trace import zero_reason, EVENT_REASON
    for fid in (32, 33, 34, 35):
        r = zero_reason(fid, 0.0, w_eff=50)
        assert r and "flat this pass" in r, (fid, r)
        assert fid in EVENT_REASON
    # warm it up
    _pump(bk, n=14, ask_cut=0.5)
    ctx = SimpleNamespace(book=bk.view("BTC"), liq_flow=0.4)
    for fn in (f32_dbi, f33_mpd, f34_cwa, f35_lnp):
        v = fn(ctx)
        assert -1.0 <= v <= 1.0 and v != 0.0, (fn.__name__, v)
    assert f35_lnp(ctx) == pytest.approx(0.4)


def test_registry_and_brain_channel_budget():
    from app.formulas import FORMULA_FUNCTIONS, FORMULA_NAMES
    from app.config import WEIGHT_IDS, N_WEIGHT_COMPONENTS
    from app.flywire import N_INPUTS
    assert len(WEIGHT_IDS) == 35 == N_WEIGHT_COMPONENTS
    assert [FORMULA_NAMES[i] for i in (32, 33, 34, 35)] == \
        ["DBI", "MPD", "CWA", "LNP"]
    # the brain's odour channels must fit every directional formula —
    # this invariant pins N_INPUTS whenever the registry grows
    directional = [f for f in FORMULA_FUNCTIONS if f != 20]
    assert len(FORMULA_FUNCTIONS) == 31
    assert len(directional) == N_INPUTS, (len(directional), N_INPUTS)


def test_cci_weight_migration_for_old_state():
    from app.cci import CCIEngine, WEIGHT_IDS
    eng = CCIEngine()
    w = eng.get_weights("BTC", "tf1")
    assert set(w) == set(WEIGHT_IDS)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    # simulate a state saved before F32..F35
    old = {k: v for k, v in w.items() if k <= 31}
    eng.weights[("BTC", "tf2")] = old
    w2 = eng.get_weights("BTC", "tf2")
    assert set(w2) == set(WEIGHT_IDS) and sum(w2.values()) == pytest.approx(1.0, abs=1e-9)


def test_calibration_factor_guards():
    from tests.test_engine import make_engine, run, _boot
    eng, _f, _e = make_engine()
    # cold: n<40 → silent
    assert eng.calibration_factor() is None
    # inject50 settled pairs: stated0.5, hit-rate0.4 →0.8
    pk = eng.hist.tf_key("BTC", "tf1", "calib_p")
    hk = eng.hist.tf_key("BTC", "tf1", "calib_h")
    for _ in range(50):
        eng.hist.ring(pk, 400).append(0.5)
        eng.hist.ring(hk, 400).append(1.0 if _ < 20 else 0.0)   #40% hits
    f = eng.calibration_factor()
    assert f is not None and f == pytest.approx(0.8, abs=0.01)
    # noise floor: stated confidence <0.15 → refuse to speak
    # (a FRESH engine whose only pairs are noise-stated)
    eng2, _f2, _e2 = make_engine()
    for _ in range(50):
        eng2.hist.ring(eng2.hist.tf_key("BTC", "tf1", "calib_p"),
                       400).append(0.05)
        eng2.hist.ring(eng2.hist.tf_key("BTC", "tf1", "calib_h"),
                       400).append(0.0)
    assert eng2.calibration_factor() is None


def test_wiring_contracts():
    feed = (ROOT / "app" / "feed.py").read_text()
    assert "@depth20@100ms" in feed and "book.on_depth" in feed
    eng = (ROOT / "app" / "engine.py").read_text()
    assert "_attach_flow" in eng and "calibration_factor" in eng
    assert '"book": self.feed.book_status' in eng
    cfg = (ROOT / "app" / "config.py").read_text()
    assert "32, 33, 34, 35" in cfg
