# Context-gated price-action OOS audit

Method: fixed v8 side/geometry; exact app PA opposition predicate; next_bar open; closed OHLC; ambiguous excluded; fixed 24 bps RT
Holdout: final UTC calendar year in v8 convention (partial 2026 through 2026-11-20 23:59 UTC), with all later tape rows truncated
Folds: up to five chronological pre-holdout folds, 7-day + 247-minute purging/embargo; PAXG has one resolvable fold; no PA parameters fitted
v8 artifact md5: `cee99dc26d359a3de2cbf7c7a2751b8f`

| cell | status | baseline n/WR/PF/EV bp/hold-era EV bp | after n/WR/PF/EV bp/hold-era EV bp | promotion |
|---|---|---:|---:|---|
| BTC|tf1 | measured | 221031 / 0.17403 / 0.07998 / -24.132438 / 139.99572 | 212313 / 0.175197 / 0.08107 / -24.171218 / 147.402648 | TELEMETRY ONLY |
| BTC|tf2 | measured | 110513 / 0.27365 / 0.173568 / -23.901596 / 436.208859 | 106116 / 0.274022 / 0.173844 / -23.945886 / 436.208859 | TELEMETRY ONLY |
| BTC|tf3 | measured | 37394 / 0.396597 / 0.372779 / -23.891567 / 339.778757 | 35886 / 0.399452 / 0.371801 / -23.902939 / 339.778757 | TELEMETRY ONLY |
| BTC|tf4 | measured | 19672 / 0.453655 / 0.506306 / -22.411646 / -23.759951 | 18574 / 0.452019 / 0.499644 / -22.691788 / -23.766451 | TELEMETRY ONLY |
| ETH|tf1 | measured | 221031 / 0.270202 / 0.150627 / -24.13666 / 12.234831 | 208709 / 0.26908 / 0.147802 / -24.166646 / 13.235408 | TELEMETRY ONLY |
| ETH|tf2 | measured | 110513 / 0.363083 / 0.267832 / -24.282174 / -129.479572 | 104529 / 0.365127 / 0.264022 / -24.340257 / -129.479572 | TELEMETRY ONLY |
| ETH|tf3 | measured | 37394 / 0.437124 / 0.484601 / -22.86999 / 78.600926 | 35289 / 0.436656 / 0.48275 / -22.782892 / 80.32888 | TELEMETRY ONLY |
| ETH|tf4 | measured | 19672 / 0.467045 / 0.619373 / -21.318554 / 4.172971 | 18586 / 0.46508 / 0.614483 / -21.515424 / 3.761988 | TELEMETRY ONLY |
| BNB|tf1 | NO CERTIFIED V8 ROW | — | — | — |
| BNB|tf2 | measured | 107871 / 0.339171 / 0.249156 / -24.595494 / -41.30653 | 105167 / 0.339899 / 0.249737 / -24.622748 / -41.473006 | TELEMETRY ONLY |
| BNB|tf3 | measured | 36500 / 0.437155 / 0.445247 / -24.277085 / 17.158248 | 35035 / 0.436733 / 0.441696 / -24.406255 / 17.883308 | TELEMETRY ONLY |
| BNB|tf4 | measured | 19202 / 0.490591 / 0.49233 / -24.439909 / 44.578677 | 18446 / 0.489945 / 0.48846 / -24.746611 / 50.337193 | TELEMETRY ONLY |
| PAXG|tf1 | UNGRADEABLE RESOLUTION | — | — | — |
| PAXG|tf2 | UNGRADEABLE RESOLUTION | — | — | — |
| PAXG|tf3 | UNGRADEABLE RESOLUTION | — | — | — |
| PAXG|tf4 | measured | 3220 / 0.283991 / 0.110723 / -23.490598 / -53.51386 | 3076 / 0.284336 / 0.107925 / -23.620383 / -54.8641 | TELEMETRY ONLY |

AMBIGUOUS rows are counted and excluded from WR/PF/EV; their counts are disclosed in `ambiguous_n`. `strict_wr_net` counts net PnL > 0 after the fixed 24 bp fee. `first_touch_wr` is target touches / target+stop touches before fee classification. PF and EV use net bps.
A promoted cell is still a holdout-selected promotion and should be treated as a validation result, not proof of future profits. Any missing tape, insufficient resolution, or absent certified v8 geometry stays telemetry-only.
