# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/gp_q10_july

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 3dceebe90baaaeea2755c0ff0625f9078ee5ffab

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| L4 | quantum | gp | — | 128 | `training.bootstrap_truncation=false`, `training.epsilon_end=0.05`, `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]` | 8 | 5/8 |
| L6 | quantum | gp | — | 188 | `training.bootstrap_truncation=false`, `training.epsilon_end=0.05`, `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `circuit.n_layers=6` | 8 | 3/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.44 [0.10, 0.82] | 0.02 [0.00, 0.30] | 0.07 [0.00, 0.16] | 28.0 [23.7, 31.2] (n=6) | 2194 [1830, 2289] (n=7) |
| L6 | 0.42 [0.21, 0.79] | 0.06 [0.00, 0.30] | 0.07 [0.03, 0.11] | 22.2 [20.7, 24.9] | 1673 [1544, 2042] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.53 [0.07, 0.92] | 0.01 [0.00, 0.19] | 0.08 [0.00, 0.19] | 28.0 [23.5, 31.4] (n=6) | 2215 [1765, 2271] (n=7) |
| L6 | 0.32 [0.17, 0.86] | 0.03 [0.00, 0.22] | 0.07 [0.01, 0.12] | 22.1 [20.8, 24.2] | 1660 [1503, 2166] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 4/8 (50%) | 3000 | 2725 [2600, 3000] | 2725 [2600, 3000] |
| L6 | 5/8 (62%) | 2400 | 2400 [1800, 2900] | 2265 [1845, 2750] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 2/8 (25%) | — | 2850 [2700, 3000] | 2850 [2700, 3000] |
| L6 | 0/8 (0%) | — | — | — |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `L4`

| variant | P(final-params lapped > L4) [95% CI] | P(stability > L4) [95% CI] |
|---|---|---|
| L6 | 0.52 [0.26, 0.80] | 0.51 [0.20, 0.81] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 5/36 | 1/36 | 0.00 | 29.2 | 1765 | — | — | 12694 |
| L4 | 1 | 33/36 | 33/36 | 0.19 | 33.7 | 2210 | 3000 | 3000 | 13351 |
| L4 | 2 | 21/36 | 0/36 | 0.10 | 20.0 | 2244 | 2600 | — | 10774 |
| L4 | 3 | 19/36 | 0/36 | 0.10 | 27.6 | 2379 | — | — | 7563 |
| L4 | 4 | 0/36 | 0/36 | 0.00 | — | 2271 | — | — | 6623 |
| L4 | 5 | 0/36 | 0/36 | 0.00 | — | — | — | — | 6147 |
| L4 | 6 | 36/36 | 7/36 | 0.19 | 26.9 | 2215 | 2650 | 2700 | 8419 |
| L4 | 7 | 19/36 | 2/36 | 0.06 | 28.4 | 1620 | 2800 | — | 8269 |
| L6 | 0 | 11/36 | 6/36 | 0.08 | 22.1 | 1629 | 1950 | — | 15194 |
| L6 | 1 | 26/36 | 27/36 | 0.18 | 20.8 | 1503 | 1800 | — | 24232 |
| L6 | 2 | 3/36 | 2/36 | 0.00 | 20.8 | 1672 | — | — | 15524 |
| L6 | 3 | 36/36 | 0/36 | 0.07 | 19.0 | 1649 | 2900 | — | 13350 |
| L6 | 4 | 6/36 | 0/36 | 0.01 | 24.2 | 2581 | — | — | 11168 |
| L6 | 5 | 11/36 | 8/36 | 0.07 | 23.9 | 1498 | 2400 | — | 13517 |
| L6 | 6 | 12/36 | 0/36 | 0.07 | 22.2 | 2166 | — | — | 10848 |
| L6 | 7 | 33/36 | 0/36 | 0.12 | 29.3 | 1743 | 2400 | — | 16313 |
