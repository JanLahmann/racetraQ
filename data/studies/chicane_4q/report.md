# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/study_chicane

- 56 finished cells in 7 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| base | quantum | chicane | — | 56 | — | 8 | 6/8 |
| trunc | quantum | chicane | — | 56 | `training.bootstrap_truncation=true` | 8 | 7/8 |
| eps30 | quantum | chicane | — | 56 | `training.epsilon_end=0.30` | 8 | 7/8 |
| eps30_trunc | quantum | chicane | — | 56 | `training.epsilon_end=0.30`, `training.bootstrap_truncation=true` | 8 | 7/8 |
| eps30_trunc_b128 | quantum | chicane | — | 56 | `training.epsilon_end=0.30`, `training.bootstrap_truncation=true`, `training.batch_size=128` | 8 | 7/8 |
| mlp | mlp | chicane | — | 76 | — | 8 | 8/8 |
| mlp_trunc | mlp | chicane | — | 76 | `training.bootstrap_truncation=true` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| base | 0.78 [0.51, 0.97] | 0.29 [0.05, 0.78] | 0.36 [0.23, 0.47] | 13.9 [13.2, 14.4] | 387 [279, 550] |
| trunc | 0.94 [0.68, 0.99] | 0.54 [0.22, 0.78] | 0.47 [0.41, 0.55] | 13.9 [13.4, 14.2] | 387 [279, 555] |
| eps30 | 0.99 [0.65, 1.00] | 0.28 [0.08, 0.64] | 0.39 [0.19, 0.56] | 16.1 [14.8, 18.8] | 468 [385, 577] |
| eps30_trunc | 0.99 [0.65, 1.00] | 0.23 [0.05, 0.64] | 0.37 [0.19, 0.52] | 16.1 [14.8, 18.8] | 468 [385, 577] |
| eps30_trunc_b128 | 0.98 [0.75, 1.00] | 0.42 [0.07, 0.73] | 0.60 [0.35, 0.74] | 16.1 [13.4, 20.2] | 309 [252, 412] |
| mlp | 1.00 [0.99, 1.00] | 0.91 [0.88, 0.96] | 0.92 [0.85, 0.97] | 14.0 [13.7, 14.2] | 183 [160, 196] |
| mlp_trunc | 1.00 [0.96, 1.00] | 0.92 [0.81, 0.97] | 0.97 [0.93, 0.99] | 13.9 [13.5, 14.1] | 183 [160, 196] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| base | 0.79 [0.44, 0.97] | 0.33 [0.04, 0.94] | 0.40 [0.22, 0.44] | 13.9 [13.0, 14.4] | 388 [278, 618] |
| trunc | 0.94 [0.58, 0.99] | 0.53 [0.22, 0.78] | 0.48 [0.40, 0.55] | 13.9 [13.4, 14.1] | 388 [278, 635] |
| eps30 | 0.99 [0.62, 1.00] | 0.32 [0.08, 0.75] | 0.39 [0.15, 0.57] | 16.1 [14.8, 17.6] | 472 [373, 546] |
| eps30_trunc | 0.99 [0.62, 1.00] | 0.21 [0.06, 0.75] | 0.39 [0.15, 0.51] | 16.1 [14.8, 17.6] | 472 [373, 546] |
| eps30_trunc_b128 | 0.97 [0.97, 1.00] | 0.49 [0.07, 0.74] | 0.64 [0.33, 0.74] | 15.6 [13.2, 20.5] | 307 [252, 412] |
| mlp | 1.00 [1.00, 1.00] | 0.92 [0.88, 0.97] | 0.91 [0.85, 0.97] | 14.0 [13.7, 14.2] | 186 [160, 190] |
| mlp_trunc | 1.00 [0.94, 1.00] | 0.92 [0.86, 0.97] | 0.97 [0.93, 0.99] | 13.9 [13.5, 14.1] | 186 [160, 190] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| base | 7/8 (88%) | 550 | 550 [450, 600] | 532 [396, 600] |
| trunc | 8/8 (100%) | 500 | 525 [474, 650] | 538 [425, 650] |
| eps30 | 7/8 (88%) | 500 | 500 [400, 550] | 482 [393, 593] |
| eps30_trunc | 7/8 (88%) | 500 | 500 [400, 550] | 482 [393, 593] |
| eps30_trunc_b128 | 7/8 (88%) | 300 | 300 [250, 450] | 321 [200, 414] |
| mlp | 8/8 (100%) | 100 | 150 [100, 200] | 150 [100, 200] |
| mlp_trunc | 8/8 (100%) | 100 | 150 [100, 200] | 150 [100, 200] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| base | 4/8 (50%) | 800 | 625 [450, 800] | 625 [450, 800] |
| trunc | 6/8 (75%) | 750 | 650 [500, 775] | 642 [508, 758] |
| eps30 | 6/8 (75%) | 600 | 575 [400, 725] | 558 [425, 708] |
| eps30_trunc | 6/8 (75%) | 600 | 575 [400, 725] | 558 [425, 708] |
| eps30_trunc_b128 | 7/8 (88%) | 550 | 550 [350, 550] | 521 [371, 564] |
| mlp | 8/8 (100%) | 100 | 150 [100, 200] | 150 [100, 200] |
| mlp_trunc | 8/8 (100%) | 100 | 150 [100, 200] | 150 [100, 200] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `base`

| variant | P(final-params lapped > base) [95% CI] | P(stability > base) [95% CI] |
|---|---|---|
| trunc | 0.59 [0.29, 0.89] | 0.75 [0.50, 0.97] |
| eps30 | 0.52 [0.23, 0.80] | 0.50 [0.22, 0.78] |
| eps30_trunc | 0.47 [0.19, 0.77] | 0.50 [0.22, 0.78] |
| eps30_trunc_b128 | 0.61 [0.31, 0.88] | 0.75 [0.47, 0.97] |
| mlp | 0.79 [0.50, 1.00] | 1.00 [1.00, 1.00] |
| mlp_trunc | 0.81 [0.53, 1.00] | 1.00 [1.00, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 23/36 | 1/36 | 0.42 | 14.2 | 468 | 550 | — | 123 |
| base | 1 | 13/36 | 13/36 | 0.38 | 13.8 | 618 | — | — | 73 |
| base | 2 | 34/36 | 34/36 | 0.25 | 12.8 | 216 | 200 | 800 | 156 |
| base | 3 | 34/36 | 11/36 | 0.42 | 13.5 | 302 | 600 | 700 | 147 |
| base | 4 | 36/36 | 36/36 | 0.44 | 13.0 | 386 | 500 | 550 | 172 |
| base | 5 | 36/36 | 0/36 | 0.65 | 14.4 | 391 | 450 | 450 | 174 |
| base | 6 | 16/36 | 16/36 | 0.22 | 14.1 | 254 | 550 | — | 137 |
| base | 7 | 22/36 | 2/36 | 0.08 | 15.1 | 648 | 650 | — | 104 |
| trunc | 0 | 35/36 | 30/36 | 0.61 | 14.1 | 468 | 550 | 750 | 128 |
| trunc | 1 | 17/36 | 17/36 | 0.46 | 13.5 | 635 | 800 | — | 88 |
| trunc | 2 | 34/36 | 2/36 | 0.38 | 12.9 | 216 | 200 | 750 | 136 |
| trunc | 3 | 32/36 | 32/36 | 0.43 | 13.9 | 302 | 600 | 800 | 150 |
| trunc | 4 | 34/36 | 14/36 | 0.50 | 14.7 | 386 | 500 | 550 | 181 |
| trunc | 5 | 36/36 | 0/36 | 0.59 | 14.1 | 391 | 450 | 500 | 128 |
| trunc | 6 | 36/36 | 26/36 | 0.51 | 13.3 | 254 | 500 | 500 | 165 |
| trunc | 7 | 21/36 | 21/36 | 0.36 | 13.9 | 648 | 650 | — | 104 |
| eps30 | 0 | 10/36 | 1/36 | 0.15 | 15.3 | 726 | — | — | 83 |
| eps30 | 1 | 35/36 | 35/36 | 0.39 | 17.2 | 518 | 500 | 800 | 82 |
| eps30 | 2 | 36/36 | 11/36 | 0.57 | 24.7 | 373 | 550 | 650 | 100 |
| eps30 | 3 | 36/36 | 15/36 | 0.64 | 14.7 | 358 | 350 | 400 | 120 |
| eps30 | 4 | 19/36 | 12/36 | 0.23 | 17.0 | 536 | 700 | — | 73 |
| eps30 | 5 | 36/36 | 3/36 | 0.39 | 14.8 | 425 | 400 | 400 | 70 |
| eps30 | 6 | 36/36 | 3/36 | 0.05 | 14.6 | 546 | 550 | 550 | 80 |
| eps30 | 7 | 35/36 | 27/36 | 0.56 | 18.0 | 394 | 400 | 600 | 86 |
| eps30_trunc | 0 | 10/36 | 1/36 | 0.15 | 15.3 | 726 | — | — | 83 |
| eps30_trunc | 1 | 35/36 | 35/36 | 0.39 | 17.2 | 518 | 500 | 800 | 70 |
| eps30_trunc | 2 | 36/36 | 0/36 | 0.46 | 24.7 | 373 | 550 | 650 | 94 |
| eps30_trunc | 3 | 36/36 | 15/36 | 0.64 | 14.7 | 358 | 350 | 400 | 119 |
| eps30_trunc | 4 | 19/36 | 12/36 | 0.23 | 17.0 | 536 | 700 | — | 73 |
| eps30_trunc | 5 | 36/36 | 3/36 | 0.39 | 14.8 | 425 | 400 | 400 | 71 |
| eps30_trunc | 6 | 36/36 | 3/36 | 0.05 | 14.6 | 546 | 550 | 550 | 80 |
| eps30_trunc | 7 | 35/36 | 27/36 | 0.56 | 18.0 | 394 | 400 | 600 | 89 |
| eps30_trunc_b128 | 0 | 35/36 | 2/36 | 0.77 | 19.7 | 297 | 450 | 550 | 168 |
| eps30_trunc_b128 | 1 | 36/36 | 18/36 | 0.70 | 13.6 | 412 | 450 | 550 | 163 |
| eps30_trunc_b128 | 2 | 35/36 | 30/36 | 0.80 | 13.2 | 230 | 300 | 600 | 193 |
| eps30_trunc_b128 | 3 | 36/36 | 17/36 | 0.65 | 13.0 | 317 | 350 | 350 | 167 |
| eps30_trunc_b128 | 4 | 3/36 | 2/36 | 0.01 | 14.4 | 566 | — | — | 88 |
| eps30_trunc_b128 | 5 | 35/36 | 35/36 | 0.42 | 20.5 | 352 | 300 | 500 | 139 |
| eps30_trunc_b128 | 6 | 35/36 | 22/36 | 0.33 | 16.9 | 270 | 50 | 550 | 155 |
| eps30_trunc_b128 | 7 | 36/36 | 3/36 | 0.62 | 23.7 | 233 | 250 | 250 | 186 |
| mlp | 0 | 36/36 | 31/36 | 0.92 | 14.0 | 172 | 200 | 200 | 98 |
| mlp | 1 | 36/36 | 35/36 | 0.91 | 14.2 | 218 | 250 | 250 | 85 |
| mlp | 2 | 36/36 | 33/36 | 0.97 | 13.3 | 138 | 100 | 100 | 108 |
| mlp | 3 | 35/36 | 33/36 | 0.98 | 14.2 | 190 | 200 | 200 | 74 |
| mlp | 4 | 36/36 | 36/36 | 0.73 | 14.5 | 187 | 200 | 200 | 86 |
| mlp | 5 | 36/36 | 31/36 | 0.90 | 13.7 | 189 | 100 | 100 | 102 |
| mlp | 6 | 36/36 | 33/36 | 0.85 | 14.0 | 147 | 100 | 100 | 84 |
| mlp | 7 | 36/36 | 32/36 | 0.97 | 13.6 | 184 | 100 | 100 | 72 |
| mlp_trunc | 0 | 36/36 | 34/36 | 0.93 | 14.1 | 172 | 200 | 200 | 132 |
| mlp_trunc | 1 | 36/36 | 36/36 | 1.00 | 14.1 | 218 | 250 | 250 | 116 |
| mlp_trunc | 2 | 36/36 | 34/36 | 0.98 | 13.3 | 138 | 100 | 100 | 146 |
| mlp_trunc | 3 | 33/36 | 36/36 | 0.99 | 13.8 | 190 | 200 | 200 | 132 |
| mlp_trunc | 4 | 36/36 | 32/36 | 0.97 | 14.4 | 187 | 200 | 200 | 137 |
| mlp_trunc | 5 | 34/36 | 32/36 | 0.98 | 13.7 | 189 | 100 | 100 | 135 |
| mlp_trunc | 6 | 36/36 | 21/36 | 0.95 | 14.0 | 147 | 100 | 100 | 79 |
| mlp_trunc | 7 | 36/36 | 31/36 | 0.90 | 13.3 | 184 | 100 | 100 | 70 |
