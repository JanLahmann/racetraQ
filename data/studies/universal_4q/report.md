# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/universal

- 15 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| quantum | quantum | multi | — | 56 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30`, `training.epsilon_decay_episodes=2000`, `training.gamma=0.99` | 5 | 5/5 |
| quantum_ft | quantum | multi | — | 56 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30`, `training.epsilon_decay_episodes=2000`, `training.gamma=0.99` | 5 | 4/5 |
| mlp | mlp | multi | — | 76 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30`, `training.epsilon_decay_episodes=2000`, `training.gamma=0.99` | 5 | 5/5 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 0.89 [0.68, 1.00] | 0.47 [0.25, 0.76] | 0.35 [0.28, 0.48] | 23.8 [18.8, 29.4] | 1446 [1270, 1644] |
| quantum_ft | 0.64 [0.49, 0.85] | 0.07 [0.00, 0.64] | 0.12 [0.10, 0.17] | 28.3 [19.1, 40.7] | 1478 [1155, 1721] |
| mlp | 0.95 [0.60, 1.00] | 0.50 [0.50, 0.85] | 0.48 [0.46, 0.49] | 24.9 [17.3, 28.4] | 1270 [1030, 1746] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 0.94 [0.67, 1.00] | 0.42 [0.25, 0.78] | 0.31 [0.28, 0.48] | 25.1 [18.5, 30.7] | 1438 [1198, 1721] |
| quantum_ft | 0.61 [0.47, 0.94] | 0.00 [0.00, 0.81] | 0.10 [0.09, 0.17] | 29.6 [18.8, 43.2] | 1434 [1046, 1770] |
| mlp | 1.00 [0.50, 1.00] | 0.50 [0.50, 1.00] | 0.48 [0.46, 0.49] | 24.5 [14.3, 29.2] | 1197 [985, 1852] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 5/5 (100%) | 900 | 900 [450, 1300] | 825 [480, 1210] |
| quantum_ft | 4/5 (80%) | 2300 | 2125 [1500, 2850] | 2125 [1500, 2850] |
| mlp | 5/5 (100%) | 200 | 200 [50, 300] | 200 [95, 270] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 3/5 (60%) | 1800 | 1300 [1100, 1800] | 1350 [1100, 1800] |
| quantum_ft | 1/5 (20%) | — | 2400 [—] | 2400 [—] |
| mlp | 3/5 (60%) | 3000 | 2850 [2300, 3000] | 2783 [2300, 3000] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `quantum`

| variant | P(final-params lapped > quantum) [95% CI] | P(stability > quantum) [95% CI] |
|---|---|---|
| quantum_ft | 0.24 [0.00, 0.60] | 0.00 [0.00, 0.00] |
| mlp | 0.68 [0.20, 1.00] | 0.84 [0.52, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| quantum | 0 | 36/36 | 27/36 | 0.31 | 18.5 | 1465 | 1300 | 1300 | 494 |
| quantum | 1 | 34/36 | 15/36 | 0.46 | 25.1 | 1198 | 900 | 1100 | 496 |
| quantum | 2 | 26/36 | 9/36 | 0.29 | 19.5 | 1721 | 550 | — | 431 |
| quantum | 3 | 36/36 | 28/36 | 0.48 | 30.7 | 1438 | 450 | 1800 | 600 |
| quantum | 4 | 24/36 | 9/36 | 0.28 | 26.4 | 1437 | 1000 | — | 436 |
| quantum_ft | 0 | 19/36 | 0/36 | 0.10 | 34.7 | 1607 | 1950 | — | 486 |
| quantum_ft | 1 | 34/36 | 0/36 | 0.10 | 43.2 | 1434 | 2300 | 2400 | 553 |
| quantum_ft | 2 | 22/36 | 9/36 | 0.15 | 29.6 | 1409 | 2850 | — | 525 |
| quantum_ft | 3 | 29/36 | 29/36 | 0.17 | 19.9 | 1046 | 1500 | — | 453 |
| quantum_ft | 4 | 17/36 | 0/36 | 0.09 | 18.8 | 1770 | — | — | 468 |
| mlp | 0 | 18/36 | 18/36 | 0.49 | 14.3 | 1136 | 200 | — | 240 |
| mlp | 1 | 36/36 | 18/36 | 0.48 | 24.5 | 985 | 200 | 2850 | 234 |
| mlp | 2 | 36/36 | 18/36 | 0.47 | 26.3 | 1197 | 50 | 2300 | 213 |
| mlp | 3 | 36/36 | 36/36 | 0.46 | 24.1 | 1852 | 300 | 3000 | 205 |
| mlp | 4 | 30/36 | 18/36 | 0.48 | 29.2 | 1500 | 200 | — | 184 |
