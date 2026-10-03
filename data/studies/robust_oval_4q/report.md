# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/robust_oval

- 10 finished cells in 1 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| gap8an | quantum | oval | — | 56 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}` | 10 | 10/10 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| gap8an | 1.00 [0.94, 1.00] | 0.80 [0.47, 0.99] | 0.75 [0.52, 0.92] | 13.7 [13.2, 14.1] | 264 [236, 341] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| gap8an | 1.00 [0.96, 1.00] | 0.88 [0.43, 1.00] | 0.77 [0.49, 0.95] | 13.7 [13.1, 14.1] | 258 [234, 339] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| gap8an | 10/10 (100%) | 250 | 275 [200, 300] | 260 [206, 300] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| gap8an | 10/10 (100%) | 300 | 300 [250, 400] | 315 [250, 395] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `gap8an`

(no other variant to compare)

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| gap8an | 0 | 36/36 | 3/36 | 0.56 | 12.7 | 268 | 300 | 350 | 301 |
| gap8an | 1 | 36/36 | 36/36 | 0.35 | 12.9 | 221 | 200 | 200 | 264 |
| gap8an | 2 | 36/36 | 12/36 | 0.71 | 14.0 | 339 | 400 | 400 | 376 |
| gap8an | 3 | 36/36 | 33/36 | 0.95 | 14.1 | 246 | 300 | 300 | 559 |
| gap8an | 4 | 36/36 | 36/36 | 0.97 | 13.7 | 184 | 200 | 200 | 552 |
| gap8an | 5 | 29/36 | 19/36 | 0.95 | 14.5 | 259 | 300 | 300 | 542 |
| gap8an | 6 | 33/36 | 17/36 | 0.92 | 13.6 | 247 | 250 | 250 | 438 |
| gap8an | 7 | 36/36 | 36/36 | 0.49 | 14.2 | 475 | 200 | 550 | 242 |
| gap8an | 8 | 35/36 | 30/36 | 0.42 | 13.1 | 414 | 201 | 450 | 199 |
| gap8an | 9 | 36/36 | 36/36 | 0.83 | 13.7 | 256 | 300 | 300 | 357 |
