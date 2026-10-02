# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/combo

- 35 finished cells in 4 variants; 0 failed, 5 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| quantum | quantum | combo | — | 56 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30` | 10 | 7/10 |
| mlp | mlp | combo | — | 76 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30` | 10 | 9/10 |
| quantum_ft | quantum | combo | — | 56 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30` | 5 | 5/5 |
| mlp_trunc | mlp | combo | — | 76 | `training.bootstrap_truncation=true` | 10 | 9/10 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 0.79 [0.40, 0.97] | 0.03 [0.00, 0.22] | 0.15 [0.03, 0.27] | 46.4 [42.3, 50.1] (n=9) | 1563 [1382, 2108] (n=6) |
| mlp | 0.89 [0.70, 0.98] | 0.01 [0.00, 0.47] | 0.11 [0.04, 0.20] | 31.0 [27.1, 34.2] | 1712 [1593, 2251] (n=4) |
| quantum_ft | 0.98 [0.83, 1.00] | 0.19 [0.00, 0.55] | 0.12 [0.09, 0.14] | 44.9 [42.3, 52.1] | 1619 [1444, 1915] |
| mlp_trunc | 0.82 [0.63, 0.94] | 0.02 [0.00, 0.08] | 0.08 [0.03, 0.14] | 28.6 [26.7, 32.6] | 1505 [1391, 1630] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 0.86 [0.33, 1.00] | 0.00 [0.00, 0.25] | 0.16 [0.02, 0.28] | 47.7 [41.5, 52.4] (n=9) | 1480 [1374, 2208] (n=6) |
| mlp | 0.92 [0.74, 1.00] | 0.00 [0.00, 0.47] | 0.10 [0.03, 0.22] | 31.3 [26.1, 34.9] | 1712 [1593, 2251] (n=4) |
| quantum_ft | 1.00 [0.78, 1.00] | 0.22 [0.00, 0.64] | 0.12 [0.08, 0.14] | 44.4 [41.4, 54.1] | 1657 [1439, 1993] |
| mlp_trunc | 0.86 [0.61, 0.97] | 0.00 [0.00, 0.06] | 0.08 [0.03, 0.14] | 28.5 [26.3, 33.0] | 1517 [1375, 1638] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 6/10 (60%) | 1400 | 1225 [925, 1725] | 1233 [958, 1675] |
| mlp | 9/10 (90%) | 900 | 900 [151, 2100] | 1119 [486, 1814] |
| quantum_ft | 5/5 (100%) | 800 | 800 [50, 2300] | 710 [155, 1880] |
| mlp_trunc | 9/10 (90%) | 1650 | 1650 [750, 2200] | 1481 [933, 1939] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 6/10 (60%) | 1800 | 1350 [1075, 2100] | 1400 [1108, 2000] |
| mlp | 6/10 (60%) | 1000 | 925 [126, 1375] | 808 [250, 1358] |
| quantum_ft | 4/5 (80%) | 1450 | 1250 [950, 2300] | 1250 [950, 2300] |
| mlp_trunc | 5/10 (50%) | 2350 | 1650 [800, 2350] | 1515 [860, 2200] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `quantum`

| variant | P(final-params lapped > quantum) [95% CI] | P(stability > quantum) [95% CI] |
|---|---|---|
| mlp | 0.52 [0.30, 0.71] | 0.43 [0.18, 0.72] |
| quantum_ft | 0.65 [0.37, 0.90] | 0.48 [0.20, 0.80] |
| mlp_trunc | 0.51 [0.29, 0.72] | 0.41 [0.15, 0.69] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| quantum | 0 | 36/36 | 0/36 | 0.34 | 41.7 | 2054 | 2050 | 2400 | 300 |
| quantum | 1 | 0/36 | 0/36 | 0.00 | — | — | — | — | 205 |
| quantum | 2 | 6/36 | 0/36 | 0.01 | 48.0 | — | — | — | 316 |
| quantum | 3 | 36/36 | 0/36 | 0.23 | 37.1 | 1403 | 1100 | 1200 | 553 |
| quantum | 4 | 12/36 | 12/36 | 0.03 | 52.4 | 2361 | — | — | 315 |
| quantum | 5 | 29/36 | 0/36 | 0.10 | 48.0 | — | 1400 | 1800 | 342 |
| quantum | 6 | 36/36 | 20/36 | 0.25 | 41.5 | 1344 | 1300 | 1300 | 433 |
| quantum | 7 | 34/36 | 9/36 | 0.26 | 54.7 | 1507 | 1150 | 1400 | 368 |
| quantum | 8 | 22/36 | 0/36 | 0.02 | 45.8 | — | — | — | 202 |
| quantum | 9 | 33/36 | 0/36 | 0.31 | 47.7 | 1454 | 750 | 950 | 340 |
| mlp | 0 | 32/36 | 0/36 | 0.30 | 23.9 | 1663 | 850 | 950 | 101 |
| mlp | 1 | 36/36 | 0/36 | 0.14 | 34.7 | — | 900 | 900 | 84 |
| mlp | 2 | 28/36 | 0/36 | 0.11 | 25.4 | — | 700 | 1000 | 81 |
| mlp | 3 | 36/36 | 0/36 | 0.02 | 30.9 | — | 151 | 151 | 84 |
| mlp | 4 | 2/36 | 0/36 | 0.00 | 28.8 | — | — | — | 80 |
| mlp | 5 | 34/36 | 34/36 | 0.09 | 35.2 | — | 2100 | — | 103 |
| mlp | 6 | 35/36 | 35/36 | 0.24 | 33.1 | 2251 | 100 | 100 | 137 |
| mlp | 7 | 21/36 | 0/36 | 0.06 | 31.7 | — | 2450 | — | 85 |
| mlp | 8 | 36/36 | 2/36 | 0.22 | 36.3 | 1762 | 1750 | 1750 | 100 |
| mlp | 9 | 28/36 | 0/36 | 0.03 | 26.1 | 1593 | 1450 | — | 66 |
| quantum_ft | 0 | 34/36 | 0/36 | 0.14 | 44.4 | 1657 | 800 | 1050 | 395 |
| quantum_ft | 1 | 28/36 | 12/36 | 0.13 | 41.4 | 1733 | 900 | — | 366 |
| quantum_ft | 2 | 36/36 | 8/36 | 0.10 | 54.1 | 1439 | 401 | 1450 | 360 |
| quantum_ft | 3 | 36/36 | 0/36 | 0.12 | 43.2 | 1454 | 50 | 950 | 432 |
| quantum_ft | 4 | 36/36 | 23/36 | 0.08 | 47.4 | 1993 | 2300 | 2300 | 396 |
| mlp_trunc | 0 | 2/36 | 0/36 | 0.00 | 29.3 | 1407 | — | — | 164 |
| mlp_trunc | 1 | 22/36 | 2/36 | 0.03 | 29.7 | 1617 | 2450 | — | 174 |
| mlp_trunc | 2 | 36/36 | 2/36 | 0.08 | 37.0 | 1579 | 850 | 1850 | 166 |
| mlp_trunc | 3 | 31/36 | 0/36 | 0.01 | 26.3 | 1698 | 1450 | — | 108 |
| mlp_trunc | 4 | 27/36 | 14/36 | 0.09 | 27.7 | 1393 | 1700 | — | 118 |
| mlp_trunc | 5 | 31/36 | 0/36 | 0.26 | 27.7 | 1527 | 1650 | — | 134 |
| mlp_trunc | 6 | 35/36 | 3/36 | 0.14 | 25.4 | 1267 | 750 | 1000 | 88 |
| mlp_trunc | 7 | 31/36 | 0/36 | 0.07 | 30.4 | 1832 | 2200 | 2350 | 92 |
| mlp_trunc | 8 | 22/36 | 0/36 | 0.11 | 36.3 | 1507 | 1650 | 1650 | 102 |
| mlp_trunc | 9 | 36/36 | 0/36 | 0.16 | 25.6 | 1343 | 700 | 800 | 82 |
