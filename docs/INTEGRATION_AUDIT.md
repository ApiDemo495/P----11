# Integration audit

The branch histories shown in the screenshot are ancestors of this branch.
The earlier combined tree (`d969c40`) had already incorporated the two
independent lines without recording their ancestry. The merge commits retain
that later tree rather than blindly overwriting it with older code.

The branch comparison found that the independent alpha-lab, calibration,
contract, event-alpha, derivatives, portfolio and related test files are
present. Its older cost-gate implementation is retained as `app/cost_gate.py`;
`app/edge_gate.py` is the later, separate certified-sleeve gate. Earlier
positive research claims remain explicitly retracted where the later audit
could not reproduce them. A small GitHub connection demonstration text file
from the other line was not part of the integrated application.

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
