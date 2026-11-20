# Integration audit

The branch histories shown in the screenshot are ancestors of this branch
at the commits listed below (checked against the remote on 2026-11-20).
A branch can advance later, so this is a snapshot, not a promise about
future commits. The earlier combined tree (`d969c40`) had already incorporated
parts of the two independent lines without recording their ancestry. The
merge commits for those two lines retain that later tree rather than blindly
overwriting it with older code; Git ancestry alone does not prove all
conflicting features were retained or work correctly.

| Remote branch | Inspected commit | Integration |
|---|---|---|
| `arena/023f99d5-p-11` | `63082af` | Ancestor of the imported default history |
| `arena/0c8f7b9b-p-11` | `70d02f5` | Three-way merged after the initial consolidation; adds its audit note |
| `arena/4977d7a0-p-11` | `e4241a0` | Original base of this branch |
| `arena/605bdcca-p-11` | `269caa7` | Imported default-branch history |
| `arena/68bbb922-p-11` | `d1469ce` | History-only merge, earlier integrated implementations retained |
| `arena/e4edde2f-p-11` | `e200cad` | Ancestor of the imported default history |
| `arena/f07a39b8-p-11` | `c31d0d4` | History-only merge, earlier integrated implementations retained |
| `main` | `43ad8a3` | Ancestor of the imported history |

At the time of this audit, the combined branch had not yet been merged into
the default branch; pull request #4 subsequently merged it. Keep the original
branches until you have reviewed the combined app and its limitations; no
branches were deleted in this consolidation.

The branch comparison found that the independent alpha-lab, calibration,
contract, event-alpha, derivatives, portfolio and related test files are
present. Its older cost-gate implementation is retained as `app/cost_gate.py`;
`app/edge_gate.py` is the later, separate certified-sleeve gate. Earlier
positive research claims remain explicitly retracted where the later audit
could not reproduce them. The GitHub connection demonstration text file from
the other line is preserved too, although it is not part of the application.

## Safety changes in this audit

* `FEED_MODE` now reads the documented environment variable; unsupported
  values stop startup rather than silently choosing a different feed.
* Active manual and automatic emergency re-locks both require a valid
  price, a known ATR, a move strictly greater than 3 ATR and a sufficiently
  old lock. Drills without an active lock remain audit-only.
* A retracted research sleeve cannot report itself as qualifying even if
  its historical numbers happen to pass numerical thresholds.
* A failed dependency installation is no longer stamped as successful, so
  a subsequent launch retries provisioning. Constant meta-input channels no
  longer produce NaN correlation output; unavailable correlations are named.
* Certified trigger calculations reject incomplete or discontinuous
  15-minute buckets and simulated/mixed-source lead data. The newest fully
  completed bucket is used once, not discarded twice.
* **Paper execution is paused.** The old runtime passed the latest minute
  close as the next 15-minute open and could use simulator prices while
  describing fills as real. Existing positions and results are retained but
  frozen and are not verified evidence. The UI states the pause. The
  standalone research/backtest portfolio remains available.

## Remaining work / limits

A replacement live-paper execution adapter needs timestamped, verified real
OHLC provenance, causal next-bar fills, correct gap handling, persistence and
restart/handover tests before execution can be resumed. Previously recorded
paper results need independent reconciliation, not automatic promotion to
verified data. The sandbox cannot reach the exchange feeds, so a live-market
end-to-end check and any claimed trading edge cannot be validated here. The
app is research software, not proof of profitability or investment advice.
