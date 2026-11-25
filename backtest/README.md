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

## Edge scan (certified sleeves)

`backtest/edge_scan.py` searches a fixed grammar of trigger families over
the cached real tape and hands every candidate to `app/edge_gate.py`,
which is the only thing allowed to promote a sleeve to `open`.

```
python -m backtest.edge_scan          # full grid -> results/EDGE_SCAN.{json,md}
python -m backtest.edge_scan --quick  # smaller grid, identical rules
```

It also rewrites `app/sleeve_registry.json`, which the live engine reads
through `app/sleeves.py`. Data lives in `backtest/cache/` (git-ignored);
provenance and every retraction are in `docs/ALPHA_RESEARCH.md`.

Gate, out of fold, all required: Net EV > 0 bp, PF > 1.25, strict
majority of folds positive, correlation-adjusted |t| >= 2.0, and a
positive locked holdout (last 35 % of each asset's tape, never used for
selection). Fills are `next_bar` open, exits at bar close, friction
charged on every trade.

## Alpha lab + derivatives scan (direction research, falsifications)

`backtest/alpha_lab.py` is the direction-accuracy rig (causal features,
triple-barrier labels, purged walk-forward, isotonic calibration):

```
python -m backtest.alpha_lab              # classic before/after -> results/ALPHA_LAB.md
python -m backtest.alpha_lab --mode deep  # sub-hour ensemble, TF1-TF4 -> results/ALPHA_LAB_DEEP.md
```

`backtest/derivs_scan.py` is the sequential-settlement harness
(single-position book, fixed parameters, convex exits) plus the
fill-model bias probe (coin flip on a random walk: +14.76 bp under
wick fills, -2.10 bp under `next_bar`):

```
python -m backtest.derivs_scan            # everything -> results/DERIVS_SCAN.md
python -m backtest.derivs_scan --only events -v   # event sleeves, per-fold EVs + CI
python -m backtest.derivs_scan --only bias        # random-walk fill-model guard
python -m backtest.derivs_scan --write    # refresh results/DERIVS_SCAN.md
```

Caveat: the scan's two once-positive event sleeves are **retracted**
(`docs/ALPHA_RESEARCH.md` §0, flagged in the report itself) — the
harness and its falsifications stand, the positives do not. Cost
verdicts from these measurements come from `app/cost_gate.py`; the
certified book stays in `app/edge_gate.py` + `results/EDGE_SCAN.md`.

## Win-rate dial (raise the win rate, never skip a window)

`winrate_frontier.py` sweeps the whole barrier family (`TP = k·SL`,
`SL = m·scale`) over every window of real tape with the round trip charged and
the cost floor applied by widening **both** barriers, so a geometry cannot buy
win rate by aiming the target inside the toll. Output
`results/winrate_frontier.json` feeds `app/winrate_dial.py`, which returns a
lockable geometry (reward:risk ≥ 0.25, target ≥ 1.5× round trip) for any
requested win rate — and the measured net bps/trade that win rate costs.

* `winrate_noabstain.py` — live cells × 7 geometries × 4 full-coverage sides.
* `winrate_noabstain_horizon.py` — the same at 1/2/5/10 d.
* `winrate_noabstain_wf.py` — 4-block walk-forward: selection by train win
  rate collapses to 48.4 % pooled OOS (EV −33.5 bps). A win rate is not a
  forward-looking criterion.
* `winrate_dial_report.py` — printable table of win rate, break-even
  `1/(1+k)`, the gap between them, and the price per trade.

Measured headline: +20.3 pp (BTC tf4) / +22.6 pp (ETH tf4) at the lockable
floor; 77–84 % at day scale; net EV ≈ −round trip everywhere. Full write-up in
`WINRATE_NO_ABSTENTION.md`; §36 of `REAL_RESULTS.md`.
