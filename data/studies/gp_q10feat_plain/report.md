# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/gp_q10_plain

- 8 finished cells in 1 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 6556e70dadfc63d6b8c2805ed0d9edc86ab5d40e

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| plainL6 | quantum | gp | q10 | 188 | `training.bootstrap_truncation=false`, `training.epsilon_end=0.05` | 8 | 1/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| plainL6 | 0.02 [0.00, 0.27] | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.01] | 21.4 [19.8, 24.3] (n=4) | 1854 [1705, 2042] (n=6) |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| plainL6 | 0.01 [0.00, 0.22] | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.00] | 21.4 [19.8, 24.3] (n=4) | 1832 [1705, 2067] (n=6) |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| plainL6 | 1/8 (12%) | — | 2350 [—] | 2350 [—] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| plainL6 | 0/8 (0%) | — | — | — |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `plainL6`

(no other variant to compare)

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| plainL6 | 0 | 1/36 | 0/36 | 0.00 | 20.6 | 1770 | — | — | 16711 |
| plainL6 | 1 | 2/36 | 2/36 | 0.00 | 24.3 | 1770 | — | — | 19258 |
| plainL6 | 2 | 0/36 | 0/36 | 0.00 | — | — | — | — | 15376 |
| plainL6 | 3 | 0/36 | 0/36 | 0.00 | — | 2029 | — | — | 18739 |
| plainL6 | 4 | 0/36 | 0/36 | 0.00 | — | 1894 | — | — | 19588 |
| plainL6 | 5 | 8/36 | 0/36 | 0.00 | 22.2 | 1640 | — | — | 18344 |
| plainL6 | 6 | 28/36 | 1/36 | 0.05 | 19.8 | 2105 | 2350 | — | 18436 |
| plainL6 | 7 | 0/36 | 0/36 | 0.00 | — | — | — | — | 15777 |
