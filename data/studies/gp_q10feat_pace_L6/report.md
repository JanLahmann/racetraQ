# Study report: scienceF/gp_q10_pace_L6

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 6556e70dadfc63d6b8c2805ed0d9edc86ab5d40e

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| pace_rel | quantum | gp | — | 188 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]` | 8 | 3/8 |
| pace_fast | quantum | gp | — | 188 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `training.snapshot_rank="pace"` | 8 | 3/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| pace_rel | 0.40 [0.19, 0.72] | 0.01 [0.00, 0.06] | 0.10 [0.04, 0.15] | 24.3 [22.6, 25.5] | 12 [6, 40] |
| pace_fast | 0.37 [0.12, 0.72] | 0.01 [0.00, 0.06] | 0.10 [0.04, 0.15] | 24.0 [22.0, 25.9] (n=7) | 12 [6, 40] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| pace_rel | 0.32 [0.14, 0.78] | 0.01 [0.00, 0.03] | 0.11 [0.03, 0.14] | 24.5 [22.8, 24.6] | 11 [4, 48] |
| pace_fast | 0.33 [0.11, 0.78] | 0.01 [0.00, 0.03] | 0.11 [0.03, 0.14] | 24.4 [22.1, 24.6] (n=7) | 11 [4, 48] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| pace_rel | 5/8 (62%) | 250 | 150 [50, 500] | 165 [65, 425] |
| pace_fast | 5/8 (62%) | 250 | 150 [50, 500] | 165 [65, 425] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| pace_rel | 1/8 (12%) | — | 150 [—] | 150 [—] |
| pace_fast | 1/8 (12%) | — | 150 [—] | 150 [—] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `pace_rel`

| variant | P(final-params lapped > pace_rel) [95% CI] | P(stability > pace_rel) [95% CI] |
|---|---|---|
| pace_fast | 0.50 [0.24, 0.75] | 0.50 [0.23, 0.77] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| pace_rel | 0 | 28/36 | 0/36 | 0.23 | 23.4 | 78 | 150 | 150 | 3267 |
| pace_rel | 1 | 29/36 | 6/36 | 0.14 | 28.3 | 3 | 100 | — | 7496 |
| pace_rel | 2 | 5/36 | 0/36 | 0.00 | 22.1 | 8 | — | — | 3964 |
| pace_rel | 3 | 13/36 | 1/36 | 0.12 | 24.6 | 7 | 250 | — | 3882 |
| pace_rel | 4 | 4/36 | 1/36 | 0.03 | 24.4 | 14 | — | — | 2972 |
| pace_rel | 5 | 24/36 | 0/36 | 0.06 | 24.6 | 21 | — | — | 4439 |
| pace_rel | 6 | 10/36 | 1/36 | 0.12 | 20.7 | 48 | 50 | — | 4415 |
| pace_rel | 7 | 10/36 | 0/36 | 0.09 | 24.5 | 4 | 500 | — | 4318 |
| pace_fast | 0 | 28/36 | 0/36 | 0.23 | 23.4 | 78 | 150 | 150 | 3231 |
| pace_fast | 1 | 29/36 | 6/36 | 0.14 | 28.3 | 3 | 100 | — | 7536 |
| pace_fast | 2 | 5/36 | 0/36 | 0.00 | 22.1 | 8 | — | — | 4036 |
| pace_fast | 3 | 13/36 | 1/36 | 0.12 | 24.6 | 7 | 250 | — | 3865 |
| pace_fast | 4 | 4/36 | 1/36 | 0.03 | 24.4 | 14 | — | — | 3006 |
| pace_fast | 5 | 24/36 | 0/36 | 0.06 | 24.6 | 21 | — | — | 4465 |
| pace_fast | 6 | 11/36 | 1/36 | 0.12 | 20.2 | 48 | 50 | — | 4379 |
| pace_fast | 7 | 0/36 | 0/36 | 0.09 | — | 4 | 500 | — | 4316 |
