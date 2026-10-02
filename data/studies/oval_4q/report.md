# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/study_oval

- 56 finished cells in 7 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| base | quantum | oval | — | 56 | — | 8 | 8/8 |
| trunc | quantum | oval | — | 56 | `training.bootstrap_truncation=true` | 8 | 8/8 |
| eps30 | quantum | oval | — | 56 | `training.epsilon_end=0.30` | 8 | 7/8 |
| eps30_trunc | quantum | oval | — | 56 | `training.epsilon_end=0.30`, `training.bootstrap_truncation=true` | 8 | 7/8 |
| eps30_trunc_b128 | quantum | oval | — | 56 | `training.epsilon_end=0.30`, `training.bootstrap_truncation=true`, `training.batch_size=128` | 8 | 7/8 |
| mlp | mlp | oval | — | 76 | — | 8 | 8/8 |
| mlp_trunc | mlp | oval | — | 76 | `training.bootstrap_truncation=true` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| base | 1.00 [0.98, 1.00] | 0.52 [0.01, 1.00] | 0.58 [0.49, 0.71] | 13.8 [13.1, 14.3] | 294 [238, 369] |
| trunc | 1.00 [0.98, 1.00] | 0.84 [0.57, 0.97] | 0.69 [0.57, 0.75] | 13.9 [13.3, 14.4] | 294 [238, 369] |
| eps30 | 1.00 [0.81, 1.00] | 0.96 [0.56, 1.00] | 0.68 [0.34, 0.85] | 14.2 [13.7, 15.9] | 374 [332, 517] |
| eps30_trunc | 1.00 [0.81, 1.00] | 0.76 [0.24, 0.96] | 0.65 [0.35, 0.85] | 14.0 [13.5, 14.6] | 374 [332, 517] |
| eps30_trunc_b128 | 1.00 [0.83, 1.00] | 0.94 [0.72, 1.00] | 0.71 [0.53, 0.80] | 13.7 [13.2, 14.4] | 270 [241, 405] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [0.87, 1.00] | 0.92 [0.86, 0.96] | 13.3 [12.9, 13.7] | 153 [134, 168] |
| mlp_trunc | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.00] | 0.94 [0.87, 0.99] | 12.9 [12.6, 13.2] | 153 [134, 168] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| base | 1.00 [0.99, 1.00] | 0.53 [0.01, 1.00] | 0.57 [0.49, 0.76] | 13.8 [13.1, 14.4] | 290 [232, 364] |
| trunc | 1.00 [1.00, 1.00] | 0.83 [0.50, 1.00] | 0.69 [0.56, 0.74] | 13.9 [13.2, 14.5] | 290 [232, 364] |
| eps30 | 1.00 [0.78, 1.00] | 1.00 [0.47, 1.00] | 0.74 [0.23, 0.85] | 14.2 [13.8, 15.2] | 370 [319, 499] |
| eps30_trunc | 1.00 [0.78, 1.00] | 0.82 [0.24, 0.97] | 0.65 [0.23, 0.86] | 14.0 [13.4, 14.6] | 370 [319, 499] |
| eps30_trunc_b128 | 1.00 [1.00, 1.00] | 0.96 [0.71, 1.00] | 0.72 [0.64, 0.83] | 13.6 [13.2, 14.4] | 270 [250, 329] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.85, 0.97] | 13.3 [12.9, 13.7] | 157 [133, 165] |
| mlp_trunc | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.93 [0.87, 1.00] | 12.9 [12.6, 13.2] | 157 [133, 165] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| base | 8/8 (100%) | 350 | 350 [250, 400] | 325 [250, 425] |
| trunc | 8/8 (100%) | 350 | 350 [250, 400] | 325 [250, 425] |
| eps30 | 8/8 (100%) | 300 | 325 [300, 400] | 325 [275, 475] |
| eps30_trunc | 8/8 (100%) | 300 | 325 [300, 400] | 325 [275, 475] |
| eps30_trunc_b128 | 7/8 (88%) | 201 | 201 [150, 250] | 215 [168, 282] |
| mlp | 8/8 (100%) | 100 | 100 [100, 200] | 112 [100, 175] |
| mlp_trunc | 8/8 (100%) | 100 | 100 [100, 200] | 112 [100, 188] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| base | 8/8 (100%) | 400 | 425 [250, 500] | 388 [250, 500] |
| trunc | 8/8 (100%) | 400 | 425 [250, 450] | 388 [250, 462] |
| eps30 | 6/8 (75%) | 400 | 400 [350, 600] | 392 [358, 600] |
| eps30_trunc | 6/8 (75%) | 400 | 400 [350, 500] | 392 [358, 500] |
| eps30_trunc_b128 | 7/8 (88%) | 300 | 300 [200, 350] | 282 [204, 339] |
| mlp | 8/8 (100%) | 100 | 100 [100, 200] | 112 [100, 175] |
| mlp_trunc | 8/8 (100%) | 100 | 100 [100, 200] | 112 [100, 200] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `base`

| variant | P(final-params lapped > base) [95% CI] | P(stability > base) [95% CI] |
|---|---|---|
| trunc | 0.56 [0.25, 0.89] | 0.64 [0.36, 0.92] |
| eps30 | 0.61 [0.34, 0.84] | 0.61 [0.30, 0.91] |
| eps30_trunc | 0.47 [0.18, 0.76] | 0.58 [0.27, 0.88] |
| eps30_trunc_b128 | 0.59 [0.30, 0.89] | 0.69 [0.41, 0.95] |
| mlp | 0.72 [0.51, 0.92] | 1.00 [1.00, 1.00] |
| mlp_trunc | 0.72 [0.51, 0.93] | 1.00 [1.00, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 35/36 | 2/36 | 0.37 | 14.1 | 470 | 550 | 550 | 166 |
| base | 1 | 36/36 | 0/36 | 0.76 | 14.5 | 354 | 350 | 450 | 245 |
| base | 2 | 36/36 | 36/36 | 0.49 | 12.7 | 219 | 200 | 200 | 148 |
| base | 3 | 34/36 | 0/36 | 0.52 | 14.7 | 320 | 350 | 400 | 146 |
| base | 4 | 36/36 | 36/36 | 0.59 | 13.4 | 232 | 350 | 550 | 268 |
| base | 5 | 36/36 | 1/36 | 0.67 | 14.0 | 374 | 450 | 450 | 198 |
| base | 6 | 36/36 | 36/36 | 0.77 | 13.6 | 244 | 250 | 250 | 242 |
| base | 7 | 36/36 | 36/36 | 0.56 | 12.6 | 259 | 250 | 250 | 233 |
| trunc | 0 | 36/36 | 28/36 | 0.42 | 14.4 | 470 | 550 | 550 | 230 |
| trunc | 1 | 36/36 | 26/36 | 0.85 | 14.7 | 354 | 350 | 450 | 258 |
| trunc | 2 | 36/36 | 32/36 | 0.56 | 13.2 | 219 | 200 | 200 | 176 |
| trunc | 3 | 36/36 | 18/36 | 0.71 | 14.2 | 320 | 350 | 450 | 265 |
| trunc | 4 | 36/36 | 36/36 | 0.74 | 12.9 | 232 | 350 | 400 | 250 |
| trunc | 5 | 33/36 | 12/36 | 0.66 | 13.6 | 374 | 450 | 450 | 176 |
| trunc | 6 | 36/36 | 35/36 | 0.66 | 14.5 | 244 | 250 | 250 | 238 |
| trunc | 7 | 36/36 | 36/36 | 0.73 | 13.5 | 259 | 250 | 250 | 250 |
| eps30 | 0 | 36/36 | 30/36 | 0.77 | 14.4 | 357 | 350 | 350 | 89 |
| eps30 | 1 | 36/36 | 36/36 | 1.00 | 14.6 | 401 | 350 | 400 | 117 |
| eps30 | 2 | 36/36 | 36/36 | 0.82 | 14.0 | 319 | 300 | 400 | 124 |
| eps30 | 3 | 36/36 | 36/36 | 0.85 | 13.1 | 306 | 300 | 350 | 161 |
| eps30 | 4 | 28/36 | 0/36 | 0.23 | 15.2 | 367 | 300 | — | 67 |
| eps30 | 5 | 36/36 | 36/36 | 0.70 | 13.8 | 372 | 200 | 400 | 102 |
| eps30 | 6 | 17/36 | 17/36 | 0.19 | 13.9 | 800 | 800 | — | 68 |
| eps30 | 7 | 36/36 | 36/36 | 0.42 | 19.5 | 499 | 400 | 800 | 110 |
| eps30_trunc | 0 | 36/36 | 30/36 | 0.77 | 14.4 | 357 | 350 | 350 | 88 |
| eps30_trunc | 1 | 36/36 | 36/36 | 1.00 | 14.6 | 401 | 350 | 400 | 101 |
| eps30_trunc | 2 | 36/36 | 36/36 | 0.82 | 14.0 | 319 | 300 | 400 | 132 |
| eps30_trunc | 3 | 36/36 | 29/36 | 0.86 | 13.4 | 306 | 300 | 350 | 157 |
| eps30_trunc | 4 | 28/36 | 0/36 | 0.23 | 15.2 | 367 | 300 | — | 67 |
| eps30_trunc | 5 | 36/36 | 0/36 | 0.49 | 13.8 | 372 | 200 | 400 | 95 |
| eps30_trunc | 6 | 17/36 | 17/36 | 0.19 | 13.9 | 800 | 800 | — | 66 |
| eps30_trunc | 7 | 36/36 | 34/36 | 0.52 | 12.9 | 499 | 400 | 600 | 122 |
| eps30_trunc_b128 | 0 | 36/36 | 22/36 | 0.83 | 13.1 | 329 | 201 | 350 | 238 |
| eps30_trunc_b128 | 1 | 36/36 | 36/36 | 0.88 | 13.8 | 286 | 150 | 150 | 160 |
| eps30_trunc_b128 | 2 | 36/36 | 34/36 | 0.76 | 14.4 | 266 | 200 | 200 | 272 |
| eps30_trunc_b128 | 3 | 36/36 | 36/36 | 0.66 | 13.0 | 274 | 150 | 250 | 240 |
| eps30_trunc_b128 | 4 | 11/36 | 20/36 | 0.09 | 14.4 | 739 | — | — | 99 |
| eps30_trunc_b128 | 5 | 36/36 | 36/36 | 0.75 | 13.4 | 252 | 250 | 300 | 172 |
| eps30_trunc_b128 | 6 | 36/36 | 31/36 | 0.64 | 14.6 | 187 | 350 | 350 | 165 |
| eps30_trunc_b128 | 7 | 36/36 | 35/36 | 0.68 | 13.3 | 250 | 250 | 300 | 220 |
| mlp | 0 | 36/36 | 36/36 | 0.78 | 12.4 | 165 | 250 | 250 | 90 |
| mlp | 1 | 36/36 | 36/36 | 0.92 | 13.4 | 186 | 200 | 200 | 101 |
| mlp | 2 | 36/36 | 36/36 | 1.00 | 12.9 | 134 | 100 | 100 | 119 |
| mlp | 3 | 36/36 | 36/36 | 0.97 | 13.0 | 156 | 150 | 150 | 95 |
| mlp | 4 | 36/36 | 17/36 | 0.89 | 14.3 | 158 | 100 | 100 | 93 |
| mlp | 5 | 36/36 | 36/36 | 0.95 | 13.7 | 165 | 100 | 100 | 108 |
| mlp | 6 | 36/36 | 36/36 | 0.85 | 13.6 | 133 | 100 | 100 | 110 |
| mlp | 7 | 36/36 | 36/36 | 0.93 | 13.2 | 130 | 100 | 100 | 118 |
| mlp_trunc | 0 | 36/36 | 36/36 | 0.81 | 12.7 | 165 | 300 | 350 | 98 |
| mlp_trunc | 1 | 36/36 | 36/36 | 0.92 | 13.4 | 186 | 200 | 200 | 138 |
| mlp_trunc | 2 | 36/36 | 36/36 | 1.00 | 12.9 | 134 | 100 | 100 | 135 |
| mlp_trunc | 3 | 36/36 | 36/36 | 1.00 | 13.2 | 156 | 150 | 150 | 131 |
| mlp_trunc | 4 | 36/36 | 36/36 | 0.84 | 12.6 | 158 | 100 | 100 | 91 |
| mlp_trunc | 5 | 36/36 | 36/36 | 0.93 | 13.0 | 165 | 100 | 100 | 129 |
| mlp_trunc | 6 | 36/36 | 33/36 | 0.99 | 12.6 | 133 | 100 | 100 | 97 |
| mlp_trunc | 7 | 36/36 | 36/36 | 0.93 | 13.2 | 130 | 100 | 100 | 95 |
