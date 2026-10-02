# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/q8_chicane

- 18 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| L4 | quantum | chicane | q8 | 104 | `training.bootstrap_truncation=true` | 6 | 5/6 |
| L5 | quantum | chicane | q8 | 128 | `training.bootstrap_truncation=true`, `circuit.n_layers=5` | 6 | 6/6 |
| mlp | mlp | chicane | q8 | 108 | `training.bootstrap_truncation=true` | 6 | 6/6 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.90 [0.47, 1.00] | 0.28 [0.04, 0.57] | 0.41 [0.28, 0.74] | 13.3 [12.9, 14.0] | 345 [251, 495] |
| L5 | 1.00 [0.93, 1.00] | 0.37 [0.07, 0.80] | 0.47 [0.31, 0.65] | 13.2 [12.7, 14.3] | 277 [235, 425] |
| mlp | 1.00 [1.00, 1.00] | 0.99 [0.62, 1.00] | 0.95 [0.88, 0.98] | 12.9 [12.7, 13.6] | 143 [138, 162] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.96 [0.40, 1.00] | 0.29 [0.00, 0.60] | 0.40 [0.26, 0.75] | 13.2 [12.8, 14.1] | 337 [232, 522] |
| L5 | 1.00 [0.93, 1.00] | 0.35 [0.04, 0.88] | 0.47 [0.29, 0.67] | 13.2 [12.7, 14.4] | 280 [230, 430] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [0.61, 1.00] | 0.95 [0.88, 0.99] | 12.9 [12.7, 13.6] | 144 [137, 162] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 5/6 (83%) | 550 | 550 [250, 750] | 520 [280, 720] |
| L5 | 6/6 (100%) | 400 | 425 [300, 550] | 425 [317, 533] |
| mlp | 6/6 (100%) | 50 | 75 [50, 100] | 75 [50, 100] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 4/6 (67%) | 550 | 550 [350, 750] | 550 [350, 750] |
| L5 | 5/6 (83%) | 450 | 450 [350, 650] | 450 [365, 590] |
| mlp | 6/6 (100%) | 50 | 75 [50, 125] | 75 [50, 125] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `L4`

| variant | P(final-params lapped > L4) [95% CI] | P(stability > L4) [95% CI] |
|---|---|---|
| L5 | 0.61 [0.25, 0.92] | 0.58 [0.22, 0.92] |
| mlp | 0.92 [0.67, 1.00] | 0.89 [0.67, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 36/36 | 13/36 | 0.46 | 12.7 | 213 | 350 | 350 | 1486 |
| L4 | 1 | 21/36 | 0/36 | 0.33 | 13.1 | 468 | 650 | — | 1176 |
| L4 | 2 | 35/36 | 18/36 | 0.53 | 12.9 | 252 | 250 | 550 | 1936 |
| L4 | 3 | 36/36 | 25/36 | 0.97 | 13.9 | 363 | 550 | 550 | 1663 |
| L4 | 4 | 8/36 | 8/36 | 0.18 | 13.4 | 311 | — | — | 715 |
| L4 | 5 | 34/36 | 0/36 | 0.33 | 14.3 | 575 | 750 | 750 | 955 |
| L5 | 0 | 31/36 | 3/36 | 0.21 | 13.4 | 299 | 500 | — | 1865 |
| L5 | 1 | 36/36 | 36/36 | 0.78 | 13.7 | 286 | 450 | 450 | 1983 |
| L5 | 2 | 36/36 | 10/36 | 0.57 | 12.7 | 560 | 600 | 650 | 1356 |
| L5 | 3 | 36/36 | 15/36 | 0.37 | 12.6 | 215 | 350 | 350 | 2500 |
| L5 | 4 | 36/36 | 27/36 | 0.51 | 15.2 | 273 | 400 | 500 | 2055 |
| L5 | 5 | 36/36 | 0/36 | 0.43 | 13.0 | 245 | 250 | 400 | 2531 |
| mlp | 0 | 36/36 | 36/36 | 0.93 | 12.9 | 145 | 100 | 100 | 197 |
| mlp | 1 | 36/36 | 34/36 | 1.00 | 12.9 | 178 | 50 | 50 | 206 |
| mlp | 2 | 36/36 | 36/36 | 0.96 | 12.9 | 137 | 100 | 100 | 205 |
| mlp | 3 | 36/36 | 36/36 | 0.97 | 12.7 | 143 | 50 | 50 | 243 |
| mlp | 4 | 36/36 | 36/36 | 0.93 | 14.2 | 137 | 101 | 150 | 240 |
| mlp | 5 | 36/36 | 10/36 | 0.84 | 12.7 | 147 | 50 | 50 | 292 |
