# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/q6_oval

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| quantum | quantum | oval | q6 | 80 | `training.bootstrap_truncation=true` | 8 | 8/8 |
| mlp | mlp | oval | q6 | 92 | `training.bootstrap_truncation=true` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 1.00 [0.99, 1.00] | 1.00 [0.81, 1.00] | 0.75 [0.58, 0.85] | 13.3 [13.1, 13.7] | 276 [227, 311] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.87 [0.81, 0.93] | 12.8 [12.7, 13.3] | 134 [111, 160] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| quantum | 1.00 [1.00, 1.00] | 1.00 [0.81, 1.00] | 0.73 [0.66, 0.86] | 13.3 [13.1, 13.6] | 276 [222, 311] |
| mlp | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.86 [0.79, 0.93] | 12.8 [12.7, 12.9] | 136 [111, 164] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 8/8 (100%) | 350 | 350 [300, 401] | 338 [300, 400] |
| mlp | 8/8 (100%) | 100 | 100 [50, 126] | 88 [50, 126] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| quantum | 8/8 (100%) | 350 | 400 [300, 500] | 400 [325, 538] |
| mlp | 8/8 (100%) | 100 | 100 [50, 126] | 88 [50, 126] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `quantum`

| variant | P(final-params lapped > quantum) [95% CI] | P(stability > quantum) [95% CI] |
|---|---|---|
| mlp | 0.62 [0.50, 0.81] | 0.81 [0.56, 0.98] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| quantum | 0 | 34/36 | 36/36 | 0.85 | 14.5 | 242 | 250 | 300 | 749 |
| quantum | 1 | 36/36 | 36/36 | 0.66 | 13.5 | 261 | 350 | 350 | 732 |
| quantum | 2 | 36/36 | 25/36 | 0.72 | 13.1 | 312 | 350 | 350 | 804 |
| quantum | 3 | 36/36 | 36/36 | 0.93 | 13.4 | 290 | 300 | 450 | 747 |
| quantum | 4 | 36/36 | 36/36 | 0.69 | 13.0 | 192 | 500 | 500 | 529 |
| quantum | 5 | 36/36 | 22/36 | 0.86 | 13.1 | 310 | 401 | 450 | 485 |
| quantum | 6 | 36/36 | 36/36 | 0.23 | 13.3 | 336 | 350 | 800 | 371 |
| quantum | 7 | 36/36 | 36/36 | 0.73 | 13.6 | 222 | 300 | 300 | 610 |
| mlp | 0 | 36/36 | 36/36 | 0.84 | 12.8 | 135 | 150 | 150 | 204 |
| mlp | 1 | 36/36 | 36/36 | 0.93 | 13.0 | 119 | 50 | 50 | 210 |
| mlp | 2 | 36/36 | 36/36 | 0.93 | 14.7 | 103 | 150 | 150 | 131 |
| mlp | 3 | 36/36 | 36/36 | 0.93 | 12.7 | 146 | 50 | 50 | 148 |
| mlp | 4 | 36/36 | 36/36 | 0.79 | 12.8 | 181 | 100 | 100 | 179 |
| mlp | 5 | 36/36 | 36/36 | 0.86 | 12.7 | 167 | 101 | 101 | 213 |
| mlp | 6 | 36/36 | 36/36 | 0.87 | 12.8 | 137 | 50 | 50 | 214 |
| mlp | 7 | 36/36 | 36/36 | 0.79 | 12.9 | 103 | 100 | 100 | 175 |
