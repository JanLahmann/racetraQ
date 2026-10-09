# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/gp4_pace30

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 6556e70dadfc63d6b8c2805ed0d9edc86ab5d40e

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| rel30 | quantum | gp | — | 56 | `training.epsilon_start=0.3`, `training.epsilon_end=0.3` | 8 | 8/8 |
| fast30 | quantum | gp | — | 56 | `training.epsilon_start=0.3`, `training.epsilon_end=0.3`, `training.snapshot_rank="pace"` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| rel30 | 0.94 [0.83, 1.00] | 0.03 [0.00, 0.22] | 0.30 [0.22, 0.41] | 33.6 [28.7, 36.2] | 59 [26, 103] |
| fast30 | 0.94 [0.83, 0.99] | 0.03 [0.00, 0.22] | 0.30 [0.22, 0.41] | 32.8 [27.3, 36.0] | 59 [26, 103] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| rel30 | 0.94 [0.86, 1.00] | 0.00 [0.00, 0.22] | 0.32 [0.22, 0.45] | 33.8 [29.0, 36.5] | 56 [26, 104] |
| fast30 | 0.94 [0.86, 0.99] | 0.00 [0.00, 0.22] | 0.32 [0.22, 0.45] | 33.8 [27.3, 36.5] | 56 [26, 104] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| rel30 | 8/8 (100%) | 100 | 125 [75, 250] | 150 [75, 300] |
| fast30 | 8/8 (100%) | 100 | 125 [75, 250] | 150 [75, 300] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| rel30 | 6/8 (75%) | 250 | 200 [50, 425] | 192 [67, 400] |
| fast30 | 6/8 (75%) | 250 | 200 [50, 425] | 192 [67, 400] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `rel30`

| variant | P(final-params lapped > rel30) [95% CI] | P(stability > rel30) [95% CI] |
|---|---|---|
| fast30 | 0.50 [0.25, 0.73] | 0.50 [0.21, 0.78] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| rel30 | 0 | 31/36 | 5/36 | 0.19 | 34.7 | 65 | 250 | — | 235 |
| rel30 | 1 | 23/36 | 0/36 | 0.24 | 33.1 | 113 | 250 | — | 147 |
| rel30 | 2 | 36/36 | 0/36 | 0.16 | 36.5 | 121 | 550 | 550 | 162 |
| rel30 | 3 | 32/36 | 0/36 | 0.31 | 34.5 | 27 | 150 | 300 | 206 |
| rel30 | 4 | 36/36 | 8/36 | 0.55 | 32.4 | 48 | 100 | 250 | 171 |
| rel30 | 5 | 33/36 | 0/36 | 0.33 | 24.3 | 26 | 50 | 50 | 178 |
| rel30 | 6 | 35/36 | 18/36 | 0.45 | 25.6 | 96 | 50 | 50 | 268 |
| rel30 | 7 | 36/36 | 0/36 | 0.33 | 39.3 | 24 | 100 | 150 | 174 |
| fast30 | 0 | 31/36 | 5/36 | 0.19 | 34.7 | 65 | 250 | — | 214 |
| fast30 | 1 | 23/36 | 0/36 | 0.24 | 33.1 | 113 | 250 | — | 223 |
| fast30 | 2 | 36/36 | 0/36 | 0.16 | 36.5 | 121 | 550 | 550 | 194 |
| fast30 | 3 | 32/36 | 0/36 | 0.31 | 34.5 | 27 | 150 | 300 | 242 |
| fast30 | 4 | 35/36 | 8/36 | 0.55 | 29.0 | 48 | 100 | 250 | 168 |
| fast30 | 5 | 33/36 | 0/36 | 0.33 | 24.3 | 26 | 50 | 50 | 179 |
| fast30 | 6 | 35/36 | 18/36 | 0.45 | 25.6 | 96 | 50 | 50 | 268 |
| fast30 | 7 | 36/36 | 0/36 | 0.33 | 39.3 | 24 | 100 | 150 | 225 |
