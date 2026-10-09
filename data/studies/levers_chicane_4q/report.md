# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/levers_chicane

- 32 finished cells in 4 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 3dceebe90baaaeea2755c0ff0625f9078ee5ffab

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| none | quantum | chicane | — | 56 | `training.action_gap=0.0`, `training.act_noise={}` | 8 | 7/8 |
| gap | quantum | chicane | — | 56 | `training.act_noise={}` | 8 | 7/8 |
| noise | quantum | chicane | — | 56 | `training.action_gap=0.0` | 8 | 6/8 |
| both | quantum | chicane | — | 56 | — | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| none | 0.94 [0.68, 0.99] | 0.54 [0.22, 0.78] | 0.47 [0.41, 0.55] | 13.9 [13.4, 14.2] | 387 [279, 555] |
| gap | 0.94 [0.68, 1.00] | 0.42 [0.14, 0.94] | 0.48 [0.30, 0.64] | 13.8 [12.9, 14.6] | 230 [214, 302] |
| noise | 0.96 [0.47, 1.00] | 0.47 [0.00, 0.99] | 0.42 [0.16, 0.54] | 14.2 [13.9, 14.9] (n=7) | 365 [323, 540] |
| both | 1.00 [0.94, 1.00] | 0.60 [0.41, 0.89] | 0.64 [0.53, 0.75] | 13.5 [12.9, 14.3] | 301 [234, 450] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| none | 0.94 [0.58, 0.99] | 0.53 [0.22, 0.78] | 0.48 [0.40, 0.55] | 13.9 [13.4, 14.1] | 388 [278, 635] |
| gap | 0.96 [0.78, 1.00] | 0.31 [0.14, 0.94] | 0.48 [0.32, 0.64] | 13.8 [12.8, 14.8] | 226 [214, 264] |
| noise | 0.96 [0.47, 1.00] | 0.44 [0.00, 0.99] | 0.43 [0.18, 0.54] | 14.2 [13.9, 14.7] (n=7) | 349 [331, 582] |
| both | 1.00 [0.97, 1.00] | 0.58 [0.50, 1.00] | 0.65 [0.50, 0.70] | 13.4 [12.7, 14.4] | 255 [230, 450] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| none | 8/8 (100%) | 500 | 525 [474, 650] | 538 [425, 650] |
| gap | 7/8 (88%) | 250 | 250 [200, 500] | 279 [218, 472] |
| noise | 6/8 (75%) | 500 | 475 [450, 600] | 483 [450, 592] |
| both | 8/8 (100%) | 300 | 300 [300, 475] | 338 [275, 475] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| none | 6/8 (75%) | 750 | 650 [500, 775] | 642 [508, 758] |
| gap | 6/8 (75%) | 500 | 450 [375, 650] | 467 [392, 617] |
| noise | 3/8 (38%) | — | 750 [600, 800] | 733 [600, 800] |
| both | 8/8 (100%) | 450 | 475 [400, 550] | 462 [400, 562] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `none`

| variant | P(final-params lapped > none) [95% CI] | P(stability > none) [95% CI] |
|---|---|---|
| gap | 0.54 [0.23, 0.81] | 0.48 [0.19, 0.78] |
| noise | 0.50 [0.19, 0.80] | 0.41 [0.12, 0.70] |
| both | 0.60 [0.28, 0.88] | 0.84 [0.59, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 35/36 | 30/36 | 0.61 | 14.1 | 468 | 550 | 750 | 155 |
| none | 1 | 17/36 | 17/36 | 0.46 | 13.5 | 635 | 800 | — | 99 |
| none | 2 | 34/36 | 2/36 | 0.38 | 12.9 | 216 | 200 | 750 | 155 |
| none | 3 | 32/36 | 32/36 | 0.43 | 13.9 | 302 | 600 | 800 | 194 |
| none | 4 | 34/36 | 14/36 | 0.50 | 14.7 | 386 | 500 | 550 | 238 |
| none | 5 | 36/36 | 0/36 | 0.59 | 14.1 | 391 | 450 | 500 | 158 |
| none | 6 | 36/36 | 26/36 | 0.51 | 13.3 | 254 | 500 | 500 | 190 |
| none | 7 | 21/36 | 21/36 | 0.36 | 13.9 | 648 | 650 | — | 127 |
| gap | 0 | 3/36 | 1/36 | 0.04 | 15.5 | 466 | — | — | 143 |
| gap | 1 | 36/36 | 36/36 | 0.36 | 12.8 | 227 | 500 | 500 | 232 |
| gap | 2 | 36/36 | 34/36 | 0.54 | 14.8 | 264 | 600 | 600 | 183 |
| gap | 3 | 31/36 | 5/36 | 0.32 | 13.0 | 211 | 200 | 700 | 201 |
| gap | 4 | 28/36 | 5/36 | 0.42 | 13.9 | 213 | 200 | — | 246 |
| gap | 5 | 34/36 | 34/36 | 0.67 | 12.7 | 214 | 250 | 400 | 233 |
| gap | 6 | 36/36 | 5/36 | 0.71 | 14.5 | 252 | 250 | 400 | 301 |
| gap | 7 | 35/36 | 17/36 | 0.61 | 13.8 | 225 | 300 | 350 | 281 |
| noise | 0 | 36/36 | 17/36 | 0.53 | 13.9 | 337 | 500 | — | 217 |
| noise | 1 | 0/36 | 0/36 | 0.00 | — | 730 | — | — | 70 |
| noise | 2 | 1/36 | 0/36 | 0.07 | 15.7 | 331 | — | — | 140 |
| noise | 3 | 36/36 | 36/36 | 0.62 | 14.7 | 424 | 550 | 750 | 218 |
| noise | 4 | 36/36 | 36/36 | 0.47 | 13.6 | 279 | 450 | — | 173 |
| noise | 5 | 35/36 | 35/36 | 0.28 | 14.2 | 582 | 650 | 800 | 147 |
| noise | 6 | 34/36 | 15/36 | 0.40 | 14.0 | 348 | 450 | — | 146 |
| noise | 7 | 33/36 | 0/36 | 0.54 | 14.4 | 350 | 450 | 600 | 195 |
| both | 0 | 35/36 | 21/36 | 0.50 | 14.7 | 449 | 450 | 500 | 249 |
| both | 1 | 29/36 | 18/36 | 0.42 | 12.7 | 246 | 200 | 450 | 166 |
| both | 2 | 36/36 | 36/36 | 0.99 | 14.7 | 494 | 550 | 550 | 285 |
| both | 3 | 36/36 | 36/36 | 0.70 | 13.1 | 253 | 300 | 400 | 325 |
| both | 4 | 36/36 | 24/36 | 0.60 | 13.2 | 212 | 300 | 400 | 249 |
| both | 5 | 36/36 | 21/36 | 0.66 | 14.0 | 257 | 300 | 500 | 260 |
| both | 6 | 36/36 | 0/36 | 0.67 | 13.7 | 230 | 300 | 300 | 299 |
| both | 7 | 36/36 | 20/36 | 0.64 | 12.6 | 450 | 500 | 700 | 192 |
