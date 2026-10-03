# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/q6_chicane

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| quantum | quantum | chicane | q6 | 80 | `training.bootstrap_truncation=true` | 8 | 8/8 |
| mlp | mlp | chicane | q6 | 92 | `training.bootstrap_truncation=true` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 1.00 [0.98, 1.00] | 0.78 [0.40, 1.00] | 0.68 [0.45, 0.83] | 13.4 [13.2, 14.0] | 304 [252, 357] |
| mlp | 1.00 [1.00, 1.00] | 0.99 [0.40, 1.00] | 0.89 [0.81, 0.97] | 14.0 [13.5, 14.2] | 168 [144, 184] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 1.00 [0.97, 1.00] | 0.85 [0.40, 1.00] | 0.71 [0.41, 0.88] | 13.4 [13.2, 14.1] | 308 [242, 372] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [0.19, 1.00] | 0.88 [0.81, 0.99] | 14.1 [13.5, 14.2] | 171 [140, 185] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 8/8 (100%) | 350 | 350 [300, 550] | 362 [325, 500] |
| mlp | 8/8 (100%) | 100 | 125 [50, 150] | 125 [75, 150] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 8/8 (100%) | 450 | 450 [350, 550] | 438 [350, 550] |
| mlp | 8/8 (100%) | 150 | 150 [100, 200] | 150 [100, 200] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `quantum`

| variant | P(final-params lapped > quantum) [95% CI] | P(stability > quantum) [95% CI] |
|---|---|---|
| mlp | 0.54 [0.28, 0.81] | 0.81 [0.56, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| quantum | 0 | 36/36 | 36/36 | 0.76 | 13.6 | 236 | 300 | 450 | 789 |
| quantum | 1 | 36/36 | 25/36 | 0.41 | 15.0 | 242 | 300 | 300 | 628 |
| quantum | 2 | 36/36 | 14/36 | 0.88 | 13.3 | 328 | 350 | 350 | 677 |
| quantum | 3 | 36/36 | 36/36 | 0.89 | 12.9 | 342 | 400 | 450 | 700 |
| quantum | 4 | 35/36 | 15/36 | 0.22 | 13.4 | 372 | 700 | 700 | 315 |
| quantum | 5 | 36/36 | 36/36 | 0.67 | 13.2 | 287 | 350 | 500 | 496 |
| quantum | 6 | 34/36 | 7/36 | 0.52 | 13.4 | 379 | 550 | 550 | 325 |
| quantum | 7 | 36/36 | 36/36 | 0.78 | 14.1 | 261 | 350 | 350 | 536 |
| mlp | 0 | 36/36 | 36/36 | 1.00 | 14.1 | 136 | 150 | 150 | 148 |
| mlp | 1 | 36/36 | 36/36 | 0.87 | 13.9 | 185 | 50 | 50 | 146 |
| mlp | 2 | 36/36 | 36/36 | 0.85 | 14.1 | 184 | 150 | 150 | 169 |
| mlp | 3 | 36/36 | 1/36 | 0.76 | 14.4 | 186 | 150 | 200 | 170 |
| mlp | 4 | 36/36 | 36/36 | 0.79 | 13.1 | 164 | 100 | 200 | 177 |
| mlp | 5 | 36/36 | 7/36 | 0.88 | 13.2 | 178 | 150 | 200 | 175 |
| mlp | 6 | 36/36 | 36/36 | 0.99 | 14.1 | 140 | 50 | 100 | 142 |
| mlp | 7 | 36/36 | 35/36 | 0.95 | 14.4 | 147 | 100 | 100 | 146 |
