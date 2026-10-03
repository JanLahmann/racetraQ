# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/robust_chicane

- 10 finished cells in 1 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| gap8an | quantum | chicane | — | 56 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}` | 10 | 9/10 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| gap8an | 1.00 [0.87, 1.00] | 0.63 [0.39, 0.88] | 0.63 [0.49, 0.73] | 13.6 [13.0, 14.3] | 355 [247, 478] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| gap8an | 1.00 [0.90, 1.00] | 0.58 [0.39, 1.00] | 0.65 [0.50, 0.73] | 13.5 [12.9, 14.4] | 353 [244, 489] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| gap8an | 10/10 (100%) | 300 | 375 [300, 550] | 395 [300, 525] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| gap8an | 9/10 (90%) | 500 | 500 [400, 550] | 481 [408, 550] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `gap8an`

(no other variant to compare)

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| gap8an | 0 | 35/36 | 21/36 | 0.50 | 14.7 | 449 | 450 | 500 | 229 |
| gap8an | 1 | 29/36 | 18/36 | 0.42 | 12.7 | 246 | 200 | 450 | 139 |
| gap8an | 2 | 36/36 | 36/36 | 0.99 | 14.7 | 494 | 550 | 550 | 222 |
| gap8an | 3 | 36/36 | 36/36 | 0.70 | 13.1 | 253 | 300 | 400 | 266 |
| gap8an | 4 | 36/36 | 24/36 | 0.60 | 13.2 | 212 | 300 | 400 | 223 |
| gap8an | 5 | 36/36 | 21/36 | 0.66 | 14.0 | 257 | 300 | 500 | 205 |
| gap8an | 6 | 36/36 | 0/36 | 0.67 | 13.7 | 230 | 300 | 300 | 207 |
| gap8an | 7 | 36/36 | 20/36 | 0.64 | 12.6 | 450 | 500 | 700 | 133 |
| gap8an | 8 | 36/36 | 36/36 | 0.79 | 13.3 | 489 | 550 | 550 | 122 |
| gap8an | 9 | 17/36 | 8/36 | 0.25 | 14.4 | 604 | 650 | — | 75 |
