# Study report: scienceF/gp_q10_pace_L4

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 6556e70dadfc63d6b8c2805ed0d9edc86ab5d40e

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| pace_rel | quantum | gp | — | 128 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]` | 8 | 1/8 |
| pace_fast | quantum | gp | — | 128 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `training.snapshot_rank="pace"` | 8 | 0/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| pace_rel | 0.21 [0.08, 0.38] | 0.08 [0.00, 0.17] | 0.03 [0.00, 0.14] (n=7) | 27.3 [22.6, 31.0] (n=7) | 49 [19, 124] (n=7) |
| pace_fast | 0.12 [0.04, 0.22] | 0.08 [0.00, 0.17] | 0.03 [0.00, 0.14] (n=7) | 24.6 [21.1, 30.2] (n=7) | 49 [19, 124] (n=7) |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| pace_rel | 0.19 [0.07, 0.32] | 0.08 [0.00, 0.19] | 0.04 [0.00, 0.07] (n=7) | 27.1 [20.6, 30.4] (n=7) | 44 [17, 124] (n=7) |
| pace_fast | 0.14 [0.03, 0.22] | 0.08 [0.00, 0.19] | 0.04 [0.00, 0.07] (n=7) | 23.8 [20.6, 30.4] (n=7) | 44 [17, 124] (n=7) |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| pace_rel | 2/8 (25%) | — | 375 [350, 400] | 375 [350, 400] |
| pace_fast | 2/8 (25%) | — | 375 [350, 400] | 375 [350, 400] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| pace_rel | 0/8 (0%) | — | — | — |
| pace_fast | 0/8 (0%) | — | — | — |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `pace_rel`

| variant | P(final-params lapped > pace_rel) [95% CI] | P(stability > pace_rel) [95% CI] |
|---|---|---|
| pace_fast | 0.50 [0.24, 0.77] | 0.50 [0.21, 0.79] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| pace_rel | 0 | 11/36 | 5/36 | 0.04 | 28.7 | 28 | — | — | 4416 |
| pace_rel | 1 | 12/36 | 7/36 | 0.29 | 30.4 | 17 | 350 | — | 5374 |
| pace_rel | 2 | 26/36 | 1/36 | 0.07 | 26.9 | 44 | 400 | — | 3756 |
| pace_rel | 3 | 2/36 | 0/36 | 0.00 | 20.3 | 186 | — | — | 2016 |
| pace_rel | 4 | 0/36 | 0/36 | 0.00 | — | — | — | — | 1949 |
| pace_rel | 5 | 6/36 | 0/36 | 0.00 | 20.6 | 124 | — | — | 1790 |
| pace_rel | 6 | 8/36 | 8/36 | 0.06 | 27.1 | 4 | — | — | 2325 |
| pace_rel | 7 | 5/36 | 5/36 | — | 34.1 | 65 | — | — | 2076 |
| pace_fast | 0 | 5/36 | 5/36 | 0.04 | 23.8 | 28 | — | — | 4422 |
| pace_fast | 1 | 12/36 | 7/36 | 0.29 | 30.4 | 17 | 350 | — | 5523 |
| pace_fast | 2 | 1/36 | 1/36 | 0.07 | 22.3 | 44 | 400 | — | 4006 |
| pace_fast | 3 | 2/36 | 0/36 | 0.00 | 20.3 | 186 | — | — | 1938 |
| pace_fast | 4 | 0/36 | 0/36 | 0.00 | — | — | — | — | 1943 |
| pace_fast | 5 | 6/36 | 0/36 | 0.00 | 20.6 | 124 | — | — | 1781 |
| pace_fast | 6 | 8/36 | 8/36 | 0.06 | 27.1 | 4 | — | — | 2338 |
| pace_fast | 7 | 5/36 | 5/36 | — | 34.1 | 65 | — | — | 2072 |
