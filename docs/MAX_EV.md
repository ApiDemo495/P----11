# MAXIMISING POSITIVE-EV POINTS — what the data said to do

The request was "maximum positive EV points", with full latitude. This is the
measurement behind every change, so the decision can be re-checked rather than
trusted.

Everything here is computed from one file:
`backtest/results/per_tf_full_engine_real.csv` — the app's **own engine**
walked over real tape, 893 settled windows, every row carrying the realised
`bps_gross`, `bps_net` and `conf`. Nothing below is a simulation.

## The one number that explains the whole year

Pooled **direction accuracy on holdout: 0.4907** over 1608 windows.

The engine cannot call the side better than a coin. That is the reason no
target ratio, confidence threshold or fee assumption has rescued the
short-timeframe cells: reward:risk multiplies an expectancy that is
approximately zero.

## Why win rate and EV pull in opposite directions

Every grid point in the independent audit, holdout, net of friction:

| ratio band | n | median WR | median EV | positive-EV |
|---|---|---|---|---|
| < 0.5:1 | 170 | **67.7 %** | −0.0380 R | 35/170 |
| 0.5 – 1.0:1 | 61 | 55.6 % | −0.0352 R | 19/61 |
| 1.0 – 1.5:1 | 106 | 45.2 % | −0.0349 R | 33/106 |
| 1.5 – 2.5:1 | 125 | 43.8 % | −0.0342 R | 37/125 |
| > 2.5:1 | 77 | 41.9 % | −0.0306 R | 21/77 |

Higher ratio buys lower win rate and barely moves EV. So "raise the win rate at
2:1" and "raise EV" are not the same request and cannot both be answered by one
threshold.

## Where the EV actually is

Per cell, on the engine walk. `gross` is the signal's own expectation before
fees; `net` is after the app's 12 bps (18 bps PAXG) round trip:

| cell | n | WR | gross bps | net bps |
|---|---|---|---|---|
| ETH tf4 | 46 | 0.696 | **+594.4** | +582.4 |
| ETH tf3 | 47 | 0.617 | +290.5 | +278.5 |
| BTC tf4 | 58 | 0.517 | +235.6 | +223.6 |
| BNB tf4 | 67 | 0.642 | +205.1 | +193.1 |
| BTC tf3 | 62 | 0.500 | +151.3 | +139.3 |
| BNB tf3 | 69 | 0.623 | +144.5 | +132.5 |
| ETH tf2 | 45 | 0.489 | +87.6 | +75.6 |
| BNB tf2 | 65 | 0.600 | +58.8 | +46.8 |
| BTC tf2 | 59 | 0.475 | +45.6 | +33.6 |
| ETH tf1 | 45 | 0.467 | +40.6 | +28.6 |
| BTC tf1 | 59 | 0.542 | +28.7 | +16.7 |
| BNB tf1 | 62 | 0.484 | +10.6 | −1.4 |
| PAXG tf4 | 58 | 0.379 | **+0.2** | −17.8 |
| PAXG tf3 | 56 | 0.500 | **−0.3** | −18.3 |
| PAXG tf2 | 50 | 0.420 | **−0.7** | −18.7 |
| PAXG tf1 | 45 | 0.467 | **−16.1** | −34.1 |

**Twelve cells have a real gross edge. PAXG has none at any timeframe.**

## Why PAXG is excluded on GROSS, not net

This is the important methodological point. A threshold on *net* would encode
today's fee stack: if fees went to zero, PAXG would spring back to "profitable"
while still predicting nothing. Gross edge is a property of the signal and
survives any venue, any fee, any ratio.

The evidence that PAXG's edge is noise rather than small:

- its mean gross flips sign across confidence bands: **−9.5 / −35.9 / +60.0 /
  −28.9** bps. The crypto cells are positive in **every** band:
  **+29.9 / +218.2 / +267.0 / +148.2**.
- its first half of the sample made **+65 bps**, the second **−101 bps** — a
  zero-expectancy signal doing what zero-expectancy does.
- structurally, PAXG pays 18 bps a round trip against a median |move| of
  109 bps; the majors pay 12 bps against 147–154 bps. Gold's microstructure is
  not crypto's, and the crypto-calibrated engine does not transfer.

So the first-half profit was luck, and selecting PAXG on full-sample EW would
have been curve-fitting. It is excluded because its **gross** expectancy is
zero.

## What changed

| change | effect on the measurement |
|---|---|
| `CELL_GROSS_BPS` + `has_gross_edge()` in `app/hold_policy.py` | the gate exists and is re-derivable from the CSV |
| `TIER_NO_EDGE` checked before the history and geometry gates | a no-signal cell is named, not hidden behind "warmup" or "geometry" |
| PAXG's 4 cells demoted | 12 of 16 cells are now tradeable instead of 16 that all lose on 4 of them |

Selection result on the same walk it was derived from:

| policy | n | WR | mean net |
|---|---|---|---|
| all 16 cells (before) | 893 | 52.9 % | +100.4 bps |
| **12 gross-positive cells** | **684** | **55.6 %** | **+137.7 bps** |

## What was deliberately NOT done

- **No confidence threshold was added.** Measured, a floor of 0.10 lifts
  per-trade EV to +188 bps but keeps only 72 % of trades, and the effect is
  almost entirely first-half — the second half has only 5 low-confidence rows.
  The app's existing floor (0.385 per asset) already sits near the optimum
  (+183.5 bps at n=389). Adding a second gate would have been a number that
  looked better for a reason that does not replicate.
- **No cell was silenced.** The no-abstention rule is about *windows*. A
  no-edge cell still computes and is still shown on the board; it may only
  refuse to present itself as tradeable, and it now says why with the measured
  figure.
- **The venue switch was not made.** `winrate_venue.json` shows a zero-fee
  venue makes every cell positive, but that analysis is a null-skill baseline,
  not this engine's edge. Moving venues would not create skill.

## The honest limit

This selects the best cells an 893-window walk supports. It does not create
edge, and 45–69 windows per cell is a thin sample. The direction accuracy of
0.49 is the real ceiling on everything else: until the model can call the side,
the long timeframes will keep carrying the book because their moves are large
enough to overcome a coin-flip direction and the fees, and the short ones will
not.
