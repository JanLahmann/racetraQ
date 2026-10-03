# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/q10_oval

- 18 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| L4 | quantum | oval | q10 | 128 | `training.bootstrap_truncation=true` | 6 | 6/6 |
| L6 | quantum | oval | q10 | 188 | `training.bootstrap_truncation=true`, `circuit.n_layers=6` | 6 | 6/6 |
| mlp | mlp | oval | q10 | 124 | `training.bootstrap_truncation=true` | 6 | 6/6 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.92 [0.86, 0.97] | 0.36 [0.06, 0.85] | 0.42 [0.27, 0.50] | 14.3 [13.7, 14.7] | 293 [234, 343] |
| L6 | 0.95 [0.77, 1.00] | 0.68 [0.19, 1.00] | 0.51 [0.26, 0.63] | 13.3 [12.8, 14.5] | 292 [255, 376] |
| mlp | 1.00 [0.97, 1.00] | 0.99 [0.55, 1.00] | 0.89 [0.87, 0.93] | 12.6 [12.6, 12.6] | 118 [107, 141] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.92 [0.85, 0.99] | 0.29 [0.03, 0.94] | 0.44 [0.24, 0.51] | 14.3 [13.6, 14.7] | 294 [230, 344] |
| L6 | 0.97 [0.75, 1.00] | 0.69 [0.15, 1.00] | 0.51 [0.26, 0.65] | 13.2 [12.7, 14.6] | 286 [246, 382] |
| mlp | 1.00 [0.97, 1.00] | 1.00 [0.54, 1.00] | 0.89 [0.87, 0.94] | 12.6 [12.6, 12.6] | 119 [106, 144] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 6/6 (100%) | 350 | 375 [75, 500] | 333 [117, 500] |
| L6 | 6/6 (100%) | 350 | 350 [300, 525] | 358 [308, 508] |
| mlp | 6/6 (100%) | 100 | 100 [75, 125] | 100 [75, 125] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 6/6 (100%) | 600 | 600 [500, 626] | 592 [508, 625] |
| L6 | 5/6 (83%) | 450 | 450 [450, 600] | 480 [450, 585] |
| mlp | 6/6 (100%) | 100 | 100 [75, 150] | 108 [75, 142] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `L4`

| variant | P(final-params lapped > L4) [95% CI] | P(stability > L4) [95% CI] |
|---|---|---|
| L6 | 0.71 [0.38, 1.00] | 0.61 [0.25, 0.92] |
| mlp | 0.86 [0.58, 1.00] | 1.00 [1.00, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 36/36 | 13/36 | 0.53 | 14.6 | 356 | 400 | 450 | 6292 |
| L4 | 1 | 30/36 | 0/36 | 0.46 | 14.9 | 250 | 350 | 550 | 6149 |
| L4 | 2 | 32/36 | 2/36 | 0.49 | 14.2 | 258 | 400 | 600 | 4830 |
| L4 | 3 | 35/36 | 34/36 | 0.28 | 14.0 | 331 | 600 | 650 | 4345 |
| L4 | 4 | 31/36 | 8/36 | 0.20 | 13.2 | 330 | 100 | 600 | 3500 |
| L4 | 5 | 34/36 | 34/36 | 0.42 | 14.5 | 211 | 50 | 601 | 4992 |
| L6 | 0 | 29/36 | 14/36 | 0.40 | 13.1 | 347 | 450 | 450 | 8481 |
| L6 | 1 | 36/36 | 11/36 | 0.39 | 12.4 | 262 | 350 | 550 | 9187 |
| L6 | 2 | 36/36 | 36/36 | 0.68 | 13.3 | 234 | 300 | 450 | 9284 |
| L6 | 3 | 34/36 | 36/36 | 0.62 | 15.2 | 259 | 300 | 450 | 7624 |
| L6 | 4 | 36/36 | 36/36 | 0.62 | 14.1 | 418 | 600 | 600 | 7377 |
| L6 | 5 | 25/36 | 0/36 | 0.13 | 13.0 | 310 | 350 | — | 8742 |
| mlp | 0 | 36/36 | 36/36 | 0.92 | 12.6 | 113 | 100 | 100 | 291 |
| mlp | 1 | 36/36 | 36/36 | 0.88 | 12.6 | 127 | 100 | 100 | 285 |
| mlp | 2 | 36/36 | 36/36 | 0.95 | 12.6 | 105 | 100 | 150 | 197 |
| mlp | 3 | 36/36 | 5/36 | 0.88 | 12.6 | 125 | 50 | 50 | 214 |
| mlp | 4 | 36/36 | 36/36 | 0.90 | 12.6 | 160 | 100 | 100 | 206 |
| mlp | 5 | 34/36 | 34/36 | 0.87 | 12.6 | 107 | 150 | 150 | 187 |
