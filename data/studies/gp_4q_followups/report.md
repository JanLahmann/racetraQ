# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/gp4_follow

- 24 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 6556e70dadfc63d6b8c2805ed0d9edc86ab5d40e

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| feat4 | quantum | gp | — | 56 | `observation.ray_angles_deg=[-45.0,45.0]`, `observation.features=["rays","speed","corner_speed_ratio"]` | 8 | 2/8 |
| pace_rel | quantum | gp | — | 56 | — | 8 | 5/8 |
| pace_fast | quantum | gp | — | 56 | `training.snapshot_rank="pace"` | 8 | 5/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| feat4 | 0.11 [0.00, 0.46] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 32.5 [22.7, 49.1] (n=5) | 2947 [—] (n=1) |
| pace_rel | 0.56 [0.30, 0.80] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.05] | 28.1 [24.0, 36.7] (n=7) | 11 [6, 14] |
| pace_fast | 0.56 [0.30, 0.80] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.05] | 28.1 [24.0, 36.7] (n=7) | 11 [6, 14] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| feat4 | 0.06 [0.00, 0.46] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 27.5 [21.3, 50.5] (n=5) | 2947 [—] (n=1) |
| pace_rel | 0.58 [0.28, 0.83] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.05] | 27.0 [24.2, 35.3] (n=7) | 11 [6, 15] |
| pace_fast | 0.58 [0.28, 0.83] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.05] | 27.0 [24.2, 35.3] (n=7) | 11 [6, 15] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| feat4 | 2/8 (25%) | — | 2100 [1450, 2750] | 2100 [1450, 2750] |
| pace_rel | 6/8 (75%) | 150 | 125 [75, 325] | 142 [75, 300] |
| pace_fast | 6/8 (75%) | 150 | 125 [75, 325] | 142 [75, 300] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| feat4 | 0/8 (0%) | — | — | — |
| pace_rel | 2/8 (25%) | — | 250 [100, 400] | 250 [100, 400] |
| pace_fast | 2/8 (25%) | — | 250 [100, 400] | 250 [100, 400] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `feat4`

| variant | P(final-params lapped > feat4) [95% CI] | P(stability > feat4) [95% CI] |
|---|---|---|
| pace_rel | 0.50 [0.50, 0.50] | 0.63 [0.44, 0.81] |
| pace_fast | 0.50 [0.50, 0.50] | 0.63 [0.44, 0.81] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| feat4 | 0 | 3/36 | 0/36 | 0.00 | 50.5 | 2947 | — | — | 1633 |
| feat4 | 1 | 12/36 | 0/36 | 0.00 | 26.0 | — | — | — | 1251 |
| feat4 | 2 | 30/36 | 0/36 | 0.03 | 27.5 | — | 1450 | — | 1145 |
| feat4 | 3 | 0/36 | 0/36 | 0.00 | — | — | — | — | 865 |
| feat4 | 4 | 0/36 | 0/36 | 0.00 | — | — | — | — | 782 |
| feat4 | 5 | 21/36 | 0/36 | 0.00 | 45.8 | — | 2750 | — | 798 |
| feat4 | 6 | 1/36 | 0/36 | 0.00 | 21.3 | — | — | — | 478 |
| feat4 | 7 | 0/36 | 0/36 | 0.00 | — | — | — | — | 422 |
| pace_rel | 0 | 26/36 | 0/36 | 0.00 | 24.6 | 7 | 100 | — | 196 |
| pace_rel | 1 | 13/36 | 0/36 | 0.00 | 31.8 | 14 | — | — | 345 |
| pace_rel | 2 | 10/36 | 0/36 | 0.05 | 21.2 | 10 | 250 | — | 512 |
| pace_rel | 3 | 22/36 | 0/36 | 0.02 | 27.0 | 2 | 50 | — | 326 |
| pace_rel | 4 | 20/36 | 0/36 | 0.00 | 43.5 | 16 | 150 | — | 324 |
| pace_rel | 5 | 0/36 | 0/36 | 0.00 | — | 15 | — | — | 227 |
| pace_rel | 6 | 34/36 | 0/36 | 0.00 | 35.3 | 6 | 100 | 100 | 281 |
| pace_rel | 7 | 31/36 | 0/36 | 0.13 | 24.2 | 12 | 400 | 400 | 214 |
| pace_fast | 0 | 26/36 | 0/36 | 0.00 | 24.6 | 7 | 100 | — | 161 |
| pace_fast | 1 | 13/36 | 0/36 | 0.00 | 31.8 | 14 | — | — | 714 |
| pace_fast | 2 | 10/36 | 0/36 | 0.05 | 21.2 | 10 | 250 | — | 327 |
| pace_fast | 3 | 22/36 | 0/36 | 0.02 | 27.0 | 2 | 50 | — | 360 |
| pace_fast | 4 | 20/36 | 0/36 | 0.00 | 43.5 | 16 | 150 | — | 112 |
| pace_fast | 5 | 0/36 | 0/36 | 0.00 | — | 15 | — | — | 181 |
| pace_fast | 6 | 34/36 | 0/36 | 0.00 | 35.3 | 6 | 100 | 100 | 272 |
| pace_fast | 7 | 31/36 | 0/36 | 0.13 | 24.2 | 12 | 400 | 400 | 206 |
