# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/q10_chicane

- 18 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| L4 | quantum | chicane | q10 | 128 | `training.bootstrap_truncation=true` | 6 | 6/6 |
| L6 | quantum | chicane | q10 | 188 | `training.bootstrap_truncation=true`, `circuit.n_layers=6` | 6 | 6/6 |
| mlp | mlp | chicane | q10 | 124 | `training.bootstrap_truncation=true` | 6 | 6/6 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.99 [0.95, 1.00] | 0.55 [0.24, 0.85] | 0.46 [0.31, 0.70] | 13.6 [13.2, 13.9] | 354 [274, 525] |
| L6 | 0.99 [0.76, 1.00] | 0.55 [0.12, 0.93] | 0.45 [0.32, 0.66] | 13.1 [12.7, 13.4] | 341 [298, 416] |
| mlp | 1.00 [1.00, 1.00] | 0.97 [0.44, 1.00] | 0.93 [0.80, 0.98] | 12.7 [12.7, 12.8] | 168 [156, 179] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 1.00 [0.94, 1.00] | 0.53 [0.21, 0.89] | 0.44 [0.29, 0.75] | 13.7 [13.1, 13.9] | 332 [274, 556] |
| L6 | 1.00 [0.75, 1.00] | 0.56 [0.04, 1.00] | 0.43 [0.31, 0.69] | 13.2 [12.6, 13.4] | 335 [294, 430] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [0.42, 1.00] | 0.96 [0.77, 0.99] | 12.7 [12.7, 12.8] | 168 [155, 179] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 6/6 (100%) | 600 | 625 [200, 675] | 567 [258, 667] |
| L6 | 6/6 (100%) | 400 | 400 [350, 550] | 408 [350, 542] |
| mlp | 6/6 (100%) | 100 | 100 [100, 125] | 100 [100, 125] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 6/6 (100%) | 650 | 650 [575, 725] | 650 [583, 717] |
| L6 | 5/6 (83%) | 600 | 600 [450, 700] | 585 [480, 670] |
| mlp | 6/6 (100%) | 100 | 100 [100, 150] | 108 [100, 142] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `L4`

| variant | P(final-params lapped > L4) [95% CI] | P(stability > L4) [95% CI] |
|---|---|---|
| L6 | 0.51 [0.17, 0.88] | 0.50 [0.17, 0.83] |
| mlp | 0.79 [0.47, 1.00] | 0.97 [0.83, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 36/36 | 30/36 | 0.46 | 13.7 | 305 | 600 | 600 | 8167 |
| L4 | 1 | 34/36 | 24/36 | 0.31 | 14.2 | 305 | 150 | 550 | 7737 |
| L4 | 2 | 36/36 | 3/36 | 0.42 | 13.7 | 493 | 700 | 700 | 7469 |
| L4 | 3 | 36/36 | 12/36 | 0.72 | 13.7 | 358 | 650 | 650 | 4754 |
| L4 | 4 | 36/36 | 14/36 | 0.78 | 13.3 | 619 | 650 | 650 | 3524 |
| L4 | 5 | 34/36 | 34/36 | 0.28 | 12.9 | 244 | 250 | 750 | 5168 |
| L6 | 0 | 36/36 | 0/36 | 0.32 | 13.4 | 450 | 450 | 550 | 9800 |
| L6 | 1 | 20/36 | 20/36 | 0.36 | 13.4 | 291 | 400 | — | 9928 |
| L6 | 2 | 36/36 | 20/36 | 0.50 | 12.6 | 297 | 300 | 450 | 11988 |
| L6 | 3 | 36/36 | 36/36 | 0.66 | 13.1 | 307 | 400 | 600 | 7690 |
| L6 | 4 | 34/36 | 3/36 | 0.29 | 13.3 | 409 | 650 | 700 | 5939 |
| L6 | 5 | 36/36 | 36/36 | 0.72 | 12.7 | 363 | 400 | 600 | 7797 |
| mlp | 0 | 36/36 | 36/36 | 0.94 | 12.7 | 171 | 100 | 150 | 612 |
| mlp | 1 | 36/36 | 36/36 | 0.79 | 12.8 | 162 | 100 | 100 | 579 |
| mlp | 2 | 36/36 | 36/36 | 0.97 | 12.7 | 187 | 150 | 150 | 189 |
| mlp | 3 | 36/36 | 36/36 | 0.99 | 12.7 | 170 | 100 | 100 | 252 |
| mlp | 4 | 36/36 | 0/36 | 0.76 | 12.7 | 167 | 100 | 100 | 355 |
| mlp | 5 | 36/36 | 30/36 | 0.98 | 12.7 | 148 | 100 | 100 | 265 |
