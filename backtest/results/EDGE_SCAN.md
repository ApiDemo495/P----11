# Edge scan — sequential out-of-fold settlement

Every sleeve is a *fixed* configuration settled on a single-position book: entry at the next bar open, taker fills, stop wins ties, purged 5-fold walk-forward, friction 12 bp (18 bp PAXG). `folds +` counts how many of the five test folds were profitable.

| sleeve | n | gross bp | net bp | PF | t | folds + |
|---|---:|---:|---:|---:|---:|---|
| BTC ensemble h=12h pt2.0/sl1.0 | 425 | +2.6 | -9.4 | 0.71 | -3.30 | 1/5 |
| BTC ensemble h=48h pt3.0/sl1.5 | 261 | +5.4 | -6.6 | 0.86 | -1.18 | 1/5 |
| ETH ensemble h=12h pt2.0/sl1.0 | 376 | -2.5 | -14.5 | 0.67 | -3.73 | 0/5 |
| ETH ensemble h=48h pt3.0/sl1.5 | 285 | -8.6 | -20.6 | 0.68 | -3.09 | 0/5 |
| BNB ensemble h=12h pt2.0/sl1.0 | 396 | +7.3 | -4.7 | 0.84 | -1.64 | 1/5 |
| BNB ensemble h=48h pt3.0/sl1.5 | 286 | +2.7 | -9.3 | 0.80 | -1.83 | 0/5 |
| PAXG ensemble h=12h pt2.0/sl1.0 | 0 | — | — | — | — | no trades |
| PAXG ensemble h=48h pt3.0/sl1.5 | 280 | -0.7 | -18.7 | 0.50 | -5.59 | 0/5 |
| BTC bollinger follow h=24 | 223 | +4.4 | -7.6 | 0.83 | -1.31 | 2/5 |
| ETH bollinger follow h=24 | 210 | +7.7 | -4.3 | 0.93 | -0.52 | 1/5 |
| BNB bollinger follow h=24 | 230 | -6.1 | -18.1 | 0.62 | -3.41 | 0/5 |
| PAXG bollinger follow h=24 | 214 | +3.7 | -14.3 | 0.58 | -3.76 | 1/5 |
| BTC cascade fade k15 z4 | 194 | +3.6 | -8.4 | 0.77 | -1.61 | 1/5 |
| ETH cascade fade k15 z4 | 237 | -2.4 | -14.4 | 0.71 | -2.16 | 2/5 |
| BNB cascade fade k15 z4 | 196 | +1.3 | -10.7 | 0.71 | -2.06 | 1/5 |
| ETH<-BTC lead-lag k3 | 57004 | +0.1 | -11.9 | 0.06 | -305.82 | 0/5 |
| BNB<-BTC lead-lag k3 | 56761 | +0.2 | -11.8 | 0.02 | -398.08 | 0/5 |
