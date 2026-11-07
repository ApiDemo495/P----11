# Multi-asset testing suite

Two modes over BTC/ETH/BNB/PAXG, built on this app's own logic.

**Data** — `data.py`: CCXT (Binance) OHLCV with CSV cache in
`backtest/cache/`; per-asset friction BTC/ETH/BNB 12 bps round trip
(8 fee + 4 slippage), PAXG 18 bps (8 + 10, thinner book).

**Anti-look-ahead** — predictors receive only `DataSlice.at(t)`, a
view structurally limited to `bars[:t+1]`. Settlement (`future()`)
is a separate object the predictor never sees.

**Predictor** — `predictor.py`: the app's `PredictionEngine`
direction (CCI fusion + ladder) on the slice, levels from the app's
`moves_risk.risk_block` (q80-MAE stop, 2:1 TP, frozen at entry).

**Mode A** — `python -m backtest.cli --mode A --source ccxt --tf 15m`
walks history, settles first-touch TP/SL (SL wins ties), writes
`backtest/out/mode_a_trades.csv`, prints per-asset + combined
win% / breakeven% / EV(R) / PF / maxDD% / Sharpe(per-trade).

**Mode B** — `--mode B`: blind replay (asset/date hidden), step
candle-by-candle with the app's call at t; Take/Skip/Adjust;
outcomes logged to `backtest/replay.db`.

**Compare** — `--report backtest/out/mode_a_trades.csv` prints Raw
vs Discretionary (EV delta of your filter) and flags correlated
BTC+ETH+BNB losing-overlap drawdown events.

`--source sim` is an offline smoke test on SYNTHETIC data — useful
to verify the harness, never a performance claim. Real numbers
require `--source ccxt` in an environment that can reach Binance.
