# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/q8_oval

- 18 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| L4 | quantum | oval | q8 | 104 | `training.bootstrap_truncation=true` | 6 | 5/6 |
| L5 | quantum | oval | q8 | 128 | `training.bootstrap_truncation=true`, `circuit.n_layers=5` | 6 | 6/6 |
| mlp | mlp | oval | q8 | 108 | `training.bootstrap_truncation=true` | 6 | 6/6 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 1.00 [0.60, 1.00] | 0.36 [0.05, 0.60] | 0.46 [0.20, 0.66] | 14.2 [13.5, 14.4] | 285 [217, 354] |
| L5 | 0.99 [0.90, 1.00] | 0.88 [0.62, 0.99] | 0.53 [0.45, 0.65] | 14.4 [13.7, 14.8] | 255 [208, 386] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.84, 0.95] | 12.7 [12.6, 12.8] | 132 [118, 155] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 1.00 [0.60, 1.00] | 0.39 [0.01, 0.62] | 0.48 [0.17, 0.67] | 14.2 [13.4, 14.4] | 290 [205, 361] |
| L5 | 1.00 [0.89, 1.00] | 0.90 [0.58, 1.00] | 0.51 [0.44, 0.67] | 14.4 [13.6, 14.9] | 248 [204, 403] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.90 [0.83, 0.96] | 12.7 [12.6, 12.8] | 132 [117, 156] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 5/6 (83%) | 350 | 350 [300, 450] | 365 [315, 435] |
| L5 | 6/6 (100%) | 250 | 325 [125, 500] | 317 [150, 475] |
| mlp | 6/6 (100%) | 100 | 100 [50, 100] | 92 [58, 100] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 5/6 (83%) | 400 | 400 [300, 600] | 400 [315, 555] |
| L5 | 6/6 (100%) | 350 | 375 [250, 550] | 383 [267, 525] |
| mlp | 6/6 (100%) | 100 | 100 [50, 100] | 92 [58, 100] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `L4`

| variant | P(final-params lapped > L4) [95% CI] | P(stability > L4) [95% CI] |
|---|---|---|
| L5 | 0.92 [0.72, 1.00] | 0.67 [0.33, 0.97] |
| mlp | 1.00 [1.00, 1.00] | 0.97 [0.83, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 36/36 | 0/36 | 0.31 | 14.2 | 188 | 350 | 350 | 2151 |
| L4 | 1 | 35/36 | 24/36 | 0.48 | 12.9 | 327 | 350 | 450 | 2070 |
| L4 | 2 | 36/36 | 9/36 | 0.78 | 14.3 | 222 | 300 | 300 | 2671 |
| L4 | 3 | 36/36 | 19/36 | 0.47 | 14.2 | 296 | 450 | 600 | 1765 |
| L4 | 4 | 8/36 | 1/36 | 0.04 | 13.9 | 395 | — | — | 561 |
| L4 | 5 | 36/36 | 21/36 | 0.56 | 14.6 | 285 | 400 | 400 | 1804 |
| L5 | 0 | 33/36 | 36/36 | 0.39 | 14.1 | 271 | 400 | 400 | 2430 |
| L5 | 1 | 36/36 | 31/36 | 0.48 | 14.6 | 330 | 400 | 500 | 2084 |
| L5 | 2 | 36/36 | 25/36 | 0.54 | 13.9 | 476 | 600 | 600 | 1691 |
| L5 | 3 | 36/36 | 34/36 | 0.48 | 13.4 | 207 | 50 | 350 | 2570 |
| L5 | 4 | 36/36 | 36/36 | 0.69 | 14.9 | 202 | 200 | 200 | 2641 |
| L5 | 5 | 31/36 | 17/36 | 0.64 | 14.8 | 225 | 250 | 300 | 2354 |
| mlp | 0 | 36/36 | 36/36 | 0.78 | 12.7 | 142 | 100 | 100 | 186 |
| mlp | 1 | 36/36 | 36/36 | 0.91 | 12.6 | 132 | 51 | 51 | 194 |
| mlp | 2 | 36/36 | 36/36 | 0.94 | 12.7 | 124 | 100 | 100 | 230 |
| mlp | 3 | 36/36 | 36/36 | 0.89 | 12.8 | 110 | 50 | 50 | 187 |
| mlp | 4 | 36/36 | 36/36 | 0.98 | 12.7 | 131 | 100 | 100 | 259 |
| mlp | 5 | 36/36 | 36/36 | 0.90 | 12.6 | 171 | 100 | 100 | 183 |
