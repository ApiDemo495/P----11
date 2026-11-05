"""Exchange-physics telemetry (Chrono-Sieve-inspired gauges) — acceptance.

TAKE: measured clock-skew (E−T) z, trade-id gap z, and real @forceOrder
liquidation clusters as gauges + confidence micro-band + hindsight ring.
SKIP (reasons in the report): GPU particle filter (no CUDA assumption),
Rust/SHM rewrite, live order execution, and any hard trigger that would
replace the graded fusion.
"""
import time
from pathlib import Path

import pytest

from app.micro import MicroPhysics

ROOT = Path(__file__).resolve().parent.parent


def _feed_packets(m: MicroPhysics, n: int = 40, tau: float = 12.0) -> None:
    """A steady stream with one jittery tail sample."""
    for i in range(n):
        t0 = 1_700_000_000_000 + i * 50
        m.on_aggtrade("BTC", t0, t0 + int(tau), i + 1, 80000.0, 0.5)
    # last packet: a real skew spike (E−T jumps)
    t0 = 1_700_000_000_000 + n * 50
    m.on_aggtrade("BTC", t0, t0 + int(tau + 60), n + 1, 80000.0, 0.5)


def test_clock_skew_z_and_gap_z():
    m = MicroPhysics()
    # cold: honest None below MIN_SAMPLES (cause5 — reason, not a number)
    st = m.status("BTC", 80000.0)
    assert st["z_tau"] is None and st["z_gap"] is None
    assert st["n_tau"] < m.MIN_SAMPLES
    _feed_packets(m, n=40)
    st = m.status("BTC", 80000.0)
    assert st["agg_count"] == 41
    # one jittery tail vs a flat baseline → large positive z
    assert st["z_tau"] is not None and st["z_tau"] > 3.0, st
    assert st["z_gap"] is None or isinstance(st["z_gap"], float)
    assert st["stress"] is not None and 0 < st["stress"] <= 1.0


def test_forceorder_clusters_and_flow():
    m = MicroPhysics()
    price = 80000.0
    now = time.time()
    # a BUY-liquidation cluster +30 bps ABOVE and a SELL cluster −50 bp below
    for _ in range(6):
        m.on_forceorder("BTC", "BUY", price * 1.003, 1.5)
    for _ in range(3):
        m.on_forceorder("BTC", "SELL", price * 0.995, 0.8)
    st = m.status("BTC", price)
    assert st["available"] if "available" in st else True
    assert st["liq_events"] == 9
    assert st["nearest_up"] and st["nearest_up"]["price"] > price
    assert st["nearest_dn"] and st["nearest_dn"]["price"] < price
    # net flow is BUY-heavy (6×1.5 BUY vs 3×0.8 SELL)
    fl = m.flow("BTC")
    assert fl is not None and fl > 0.3
    # wrong symbol sees nothing (per-asset honesty)
    assert m.flow("ETH") is None


def test_forceorder_half_life_decay():
    m = MicroPhysics()
    # force an old event into the past by rewriting timestamps
    m.on_forceorder("BTC", "BUY", 80000.0, 10.0)
    old = m._liq[-1]
    m._liq[-1] = (time.time() - 1801,) + tuple(old[1:])  # beyond horizon
    assert m.status("BTC", 80000.0)["liq_events"] == 0    # expired


def test_feed_micro_status_honest_on_sim():
    from app.feed import MarketFeed
    f = MarketFeed()          # never started → no packets anywhere
    st = f.micro_status("BTC")
    assert st["available"] is False
    assert "sim" in st["reason"] or "waiting" in st["reason"]
    assert f.micro_flow("BTC") is None      # band stays silent (0)


def test_engine_micro_band_and_hindsight_wired():
    eng = (ROOT / "app" / "engine.py").read_text()
    assert '"micro": self.feed.micro_status(symbol)' in eng
    assert "micro_band" in eng and "0.10 * float(_mflow)" in eng
    assert '"facc_micro"' in eng
    assert '"agent", "brain", "micro"' in eng     # hindsight grades it
    js = (ROOT / "static" / "app.js").read_text()
    html = (ROOT / "static" / "index.html").read_text()
    assert "micro-panel" in html and "renderMicro" in js
    assert "LIQ↑" in js                          # chart overlays
    assert "clock-skew" in js
    # feed listens to the right streams
    feed = (ROOT / "app" / "feed.py").read_text()
    assert "@aggTrade" in feed and "!forceOrder@arr" in feed
    assert "_forceorder_loop" in feed
