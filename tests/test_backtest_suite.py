"""Multi-asset testing suite: anti-look-ahead, friction, analytics."""
from backtest.analytics import correlated_drawdowns, metrics
from backtest.data import Bar, sim_bars
from backtest.predictor import Signal
from backtest.runner import Trade, settle
from backtest.slice import DataSlice


def _bars(n=50, start=100.0):
    return [Bar(1_700_000_000 + i * 60, start, start + 1, start - 1,
                start, 10.0) for i in range(n)]


def test_slice_never_exposes_future():
    bars = _bars()
    ds = DataSlice(bars)
    v = ds.at(9)
    assert len(v) == 10 and v.last.t == bars[9].t
    assert max(b.t for b in v.bars) == bars[9].t      # nothing beyond t
    assert v.closes == [b.c for b in bars[:10]]


def test_settle_sl_wins_tie_and_friction_in_r():
    sig = Signal(1, 100.0, 99.0, 102.0, 0.5)          # 2:1 long
    future = [Bar(1, 100, 103, 98, 101, 1)]           # hits BOTH -> SL
    tr = settle("BTC", sig, future[0], future)
    assert tr.exit_reason == "SL" and tr.r_gross == -1.0
    # friction 12 bps of 100 = 0.12 quote = 0.12R (risk 1.0)
    assert abs(tr.r_net - (-1.12)) < 1e-9
    assert abs(tr.pnl - (-1.12)) < 1e-9


def test_settle_tp_and_paxg_friction_higher():
    sig = Signal(-1, 100.0, 101.0, 98.0, 0.5)         # short
    future = [Bar(1, 100, 100.5, 97.5, 99, 1)]
    tr = settle("PAXG", sig, future[0], future)
    assert tr.exit_reason == "TP" and tr.r_gross == 2.0
    assert tr.r_net < 2.0                              # 18 bps charged
    tr2 = settle("BTC", sig, future[0], future)
    assert tr2.r_net > tr.r_net                        # BTC cheaper to trade


def test_metrics_breakeven_and_ev():
    ts = [Trade("BTC", 0, "Long", 100, 99, 102, 60, "TP", 2, 1.9, 1.9),
          Trade("BTC", 100, "Long", 100, 99, 102, 160, "SL", -1, -1.1, -1.1),
          Trade("BTC", 200, "Long", 100, 99, 102, 260, "SL", -1, -1.1, -1.1)]
    m = metrics(ts)
    assert m["n"] == 3 and m["win_pct"] == 33.33
    assert abs(m["ev_r"] - round((1.9 - 1.1 - 1.1) / 3, 4)) < 1e-9
    # breakeven = avgLoss/(avgWin+avgLoss) = 1.1/(1.9+1.1) = 36.67%
    assert m["breakeven_pct"] == 36.67


def test_correlated_drawdown_flag():
    mk = lambda a, ts: Trade(a, ts, "Long", 100, 99, 102, ts + 1000,
                             "SL", -1, -1.1, -1.1)
    trades = [mk("BTC", 0), mk("ETH", 10), mk("BNB", 20), mk("PAXG", 30)]
    ev = correlated_drawdowns(trades)
    assert ev and set(ev[0]["assets_open_losing"]) == {"BTC", "ETH", "BNB"}


def test_replay_scripted_logs_to_sqlite(tmp_path):
    from backtest.replay import load_filtered, replay

    class Stub:
        def predict(self, view, symbol):
            return Signal(1, view.last.c, view.last.c * 0.99,
                          view.last.c * 1.02, 0.5)

    series = {"BTC": sim_bars("BTC", n=200, seed=3)}
    db = str(tmp_path / "r.db")
    tr = replay(series, Stub(), asset="BTC", start=100,
                actions=["t"], db_path=db)
    assert tr is not None and tr.source == "discretionary"
    taken = load_filtered(db)
    assert len(taken) == 1 and taken[0].asset == "BTC"


def test_backtest_endpoint_empty_state_is_honest(tmp_path, monkeypatch):
    from app.main import backtest_report   # import BEFORE chdir
    monkeypatch.chdir(tmp_path)            # no backtest/out here
    import conftest
    d = conftest.run_sync(backtest_report())
    assert d["ok"] is False and "no real backtest" in d["reason"]
