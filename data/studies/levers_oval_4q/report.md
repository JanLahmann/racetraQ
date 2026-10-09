# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/levers_oval

- 32 finished cells in 4 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 3dceebe90baaaeea2755c0ff0625f9078ee5ffab

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| none | quantum | oval | — | 56 | `training.action_gap=0.0`, `training.act_noise={}` | 8 | 8/8 |
| gap | quantum | oval | — | 56 | `training.act_noise={}` | 8 | 8/8 |
| noise | quantum | oval | — | 56 | `training.action_gap=0.0` | 8 | 7/8 |
| both | quantum | oval | — | 56 | — | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| none | 1.00 [0.98, 1.00] | 0.84 [0.57, 0.97] | 0.69 [0.57, 0.75] | 13.9 [13.3, 14.4] | 294 [238, 369] |
| gap | 1.00 [0.97, 1.00] | 0.89 [0.62, 1.00] | 0.83 [0.68, 0.93] | 13.1 [12.7, 14.0] | 220 [203, 268] |
| noise | 0.99 [0.79, 1.00] | 0.83 [0.58, 1.00] | 0.66 [0.45, 0.77] | 14.1 [13.9, 14.5] | 281 [233, 369] |
| both | 1.00 [0.93, 1.00] | 0.73 [0.37, 1.00] | 0.78 [0.53, 0.95] | 13.8 [13.1, 14.2] | 255 [224, 335] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| none | 1.00 [1.00, 1.00] | 0.83 [0.50, 1.00] | 0.69 [0.56, 0.74] | 13.9 [13.2, 14.5] | 290 [232, 364] |
| gap | 1.00 [1.00, 1.00] | 0.90 [0.69, 1.00] | 0.85 [0.66, 0.94] | 12.8 [12.7, 14.2] | 215 [206, 276] |
| noise | 1.00 [0.83, 1.00] | 0.82 [0.60, 1.00] | 0.64 [0.39, 0.78] | 14.0 [13.9, 14.5] | 276 [230, 333] |
| both | 1.00 [0.92, 1.00] | 0.72 [0.33, 1.00] | 0.81 [0.49, 0.95] | 13.8 [12.9, 14.2] | 253 [221, 339] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| none | 8/8 (100%) | 350 | 350 [250, 400] | 325 [250, 425] |
| gap | 8/8 (100%) | 250 | 250 [200, 325] | 262 [200, 325] |
| noise | 8/8 (100%) | 250 | 275 [250, 400] | 288 [250, 400] |
| both | 8/8 (100%) | 250 | 275 [200, 300] | 262 [200, 325] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| none | 8/8 (100%) | 400 | 425 [250, 450] | 388 [250, 462] |
| gap | 8/8 (100%) | 250 | 275 [225, 350] | 288 [225, 375] |
| noise | 6/8 (75%) | 400 | 375 [300, 675] | 400 [308, 650] |
| both | 8/8 (100%) | 300 | 300 [200, 400] | 300 [225, 400] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `none`

| variant | P(final-params lapped > none) [95% CI] | P(stability > none) [95% CI] |
|---|---|---|
| gap | 0.55 [0.27, 0.82] | 0.75 [0.48, 0.97] |
| noise | 0.52 [0.23, 0.80] | 0.42 [0.14, 0.75] |
| both | 0.46 [0.18, 0.75] | 0.61 [0.28, 0.89] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 36/36 | 28/36 | 0.42 | 14.4 | 470 | 550 | 550 | 446 |
| none | 1 | 36/36 | 26/36 | 0.85 | 14.7 | 354 | 350 | 450 | 362 |
| none | 2 | 36/36 | 32/36 | 0.56 | 13.2 | 219 | 200 | 200 | 609 |
| none | 3 | 36/36 | 18/36 | 0.71 | 14.2 | 320 | 350 | 450 | 585 |
| none | 4 | 36/36 | 36/36 | 0.74 | 12.9 | 232 | 350 | 400 | 442 |
| none | 5 | 33/36 | 12/36 | 0.66 | 13.6 | 374 | 450 | 450 | 278 |
| none | 6 | 36/36 | 35/36 | 0.66 | 14.5 | 244 | 250 | 250 | 339 |
| none | 7 | 36/36 | 36/36 | 0.73 | 13.5 | 259 | 250 | 250 | 404 |
| gap | 0 | 36/36 | 25/36 | 0.70 | 12.9 | 276 | 300 | 500 | 1251 |
| gap | 1 | 36/36 | 33/36 | 0.66 | 12.7 | 214 | 350 | 350 | 1283 |
| gap | 2 | 36/36 | 27/36 | 0.91 | 13.9 | 339 | 350 | 350 | 1008 |
| gap | 3 | 36/36 | 36/36 | 0.56 | 12.8 | 181 | 150 | 200 | 648 |
| gap | 4 | 32/36 | 32/36 | 0.94 | 14.2 | 206 | 150 | 200 | 559 |
| gap | 5 | 36/36 | 36/36 | 0.99 | 14.5 | 208 | 250 | 300 | 496 |
| gap | 6 | 36/36 | 5/36 | 0.90 | 12.7 | 241 | 250 | 250 | 552 |
| gap | 7 | 36/36 | 36/36 | 0.81 | 12.6 | 216 | 250 | 250 | 498 |
| noise | 0 | 36/36 | 26/36 | 0.78 | 14.7 | 210 | 200 | 400 | 619 |
| noise | 1 | 36/36 | 36/36 | 0.64 | 14.0 | 333 | 400 | 550 | 842 |
| noise | 2 | 30/36 | 19/36 | 0.26 | 14.0 | 225 | 250 | — | 325 |
| noise | 3 | 36/36 | 33/36 | 0.63 | 13.8 | 331 | 350 | 350 | 406 |
| noise | 4 | 34/36 | 24/36 | 0.39 | 14.3 | 302 | 250 | — | 258 |
| noise | 5 | 14/36 | 14/36 | 0.60 | 15.0 | 512 | 600 | 800 | 158 |
| noise | 6 | 36/36 | 36/36 | 0.76 | 13.9 | 249 | 250 | 250 | 396 |
| noise | 7 | 36/36 | 36/36 | 0.82 | 14.0 | 241 | 300 | 350 | 362 |
| both | 0 | 36/36 | 3/36 | 0.56 | 12.7 | 268 | 300 | 350 | 582 |
| both | 1 | 36/36 | 36/36 | 0.35 | 12.9 | 221 | 200 | 200 | 847 |
| both | 2 | 36/36 | 12/36 | 0.71 | 14.0 | 339 | 400 | 400 | 762 |
| both | 3 | 36/36 | 33/36 | 0.95 | 14.1 | 246 | 300 | 300 | 705 |
| both | 4 | 36/36 | 36/36 | 0.97 | 13.7 | 184 | 200 | 200 | 582 |
| both | 5 | 29/36 | 19/36 | 0.95 | 14.5 | 259 | 300 | 300 | 644 |
| both | 6 | 33/36 | 17/36 | 0.92 | 13.6 | 247 | 250 | 250 | 496 |
| both | 7 | 36/36 | 36/36 | 0.49 | 14.2 | 475 | 200 | 550 | 294 |
