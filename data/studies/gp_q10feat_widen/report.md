# Study report: scienceF/gp_q10_widen

- 24 finished cells in 3 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 6556e70dadfc63d6b8c2805ed0d9edc86ab5d40e

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| w4 | quantum | gp | — | 188 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `training.bootstrap_truncation=false`, `training.epsilon_end=0.05`, `training.epsilon_start=0.3`, `training.epsilon_decay_episodes=1000` | 8 | 4/8 |
| w6 | quantum | gp | — | 192 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `training.bootstrap_truncation=false`, `training.epsilon_end=0.05`, `training.epsilon_start=0.3`, `training.epsilon_decay_episodes=1000` | 8 | 6/8 |
| w8 | quantum | gp | — | 196 | `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `training.bootstrap_truncation=false`, `training.epsilon_end=0.05`, `training.epsilon_start=0.3`, `training.epsilon_decay_episodes=1000` | 8 | 5/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| w4 | 0.62 [0.29, 0.97] | 0.12 [0.02, 0.26] | 0.11 [0.06, 0.16] | 25.9 [22.4, 30.0] | 47 [16, 158] |
| w6 | 0.83 [0.49, 0.97] | 0.01 [0.00, 0.12] | 0.09 [0.05, 0.17] | 24.2 [18.8, 33.6] | 31 [12, 142] |
| w8 | 0.65 [0.16, 0.92] | 0.02 [0.00, 0.11] | 0.08 [0.03, 0.12] | 20.7 [19.0, 23.4] (n=7) | 47 [21, 197] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| w4 | 0.60 [0.28, 1.00] | 0.12 [0.00, 0.22] | 0.12 [0.05, 0.16] | 25.9 [22.2, 31.0] | 50 [13, 158] |
| w6 | 0.89 [0.49, 1.00] | 0.00 [0.00, 0.12] | 0.07 [0.05, 0.20] | 23.8 [18.7, 32.0] | 14 [12, 118] |
| w8 | 0.75 [0.15, 0.93] | 0.00 [0.00, 0.14] | 0.09 [0.02, 0.12] | 20.8 [18.8, 22.5] (n=7) | 42 [20, 85] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| w4 | 6/8 (75%) | 750 | 700 [325, 900] | 692 [358, 892] |
| w6 | 6/8 (75%) | 800 | 800 [475, 950] | 808 [483, 933] |
| w8 | 5/8 (62%) | 950 | 950 [250, 1300] | 875 [385, 1195] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| w4 | 3/8 (38%) | — | 700 [100, 1000] | 650 [100, 1000] |
| w6 | 3/8 (38%) | — | 1000 [800, 1450] | 1042 [800, 1450] |
| w8 | 4/8 (50%) | 1450 | 1325 [700, 1450] | 1325 [700, 1450] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `w4`

| variant | P(final-params lapped > w4) [95% CI] | P(stability > w4) [95% CI] |
|---|---|---|
| w6 | 0.30 [0.06, 0.56] | 0.42 [0.16, 0.73] |
| w8 | 0.26 [0.05, 0.51] | 0.33 [0.09, 0.62] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| w4 | 0 | 5/36 | 0/36 | 0.04 | 27.1 | 52 | 650 | — | 9985 |
| w4 | 1 | 36/36 | 0/36 | 0.11 | 31.0 | 19 | 550 | 700 | 16951 |
| w4 | 2 | 12/36 | 8/36 | 0.12 | 24.8 | 158 | — | — | 9287 |
| w4 | 3 | 29/36 | 6/36 | 0.07 | 21.2 | 8 | 100 | 100 | 18272 |
| w4 | 4 | 14/36 | 7/36 | 0.12 | 23.5 | 304 | 1000 | — | 12869 |
| w4 | 5 | 34/36 | 17/36 | 0.25 | 28.3 | 49 | 750 | — | 16977 |
| w4 | 6 | 10/36 | 3/36 | 0.05 | 20.8 | 67 | — | — | 12077 |
| w4 | 7 | 36/36 | 1/36 | 0.16 | 33.1 | 7 | 800 | 1000 | 16654 |
| w6 | 0 | 21/36 | 0/36 | 0.03 | 21.6 | 118 | — | — | 12354 |
| w6 | 1 | 31/36 | 1/36 | 0.07 | 33.6 | 11 | 200 | — | 17591 |
| w6 | 2 | 4/36 | 0/36 | 0.04 | 48.0 | 15 | — | — | 17268 |
| w6 | 3 | 36/36 | 0/36 | 0.23 | 18.8 | 12 | 800 | 800 | 16657 |
| w6 | 4 | 34/36 | 0/36 | 0.14 | 18.7 | 351 | 1000 | 1000 | 14361 |
| w6 | 5 | 36/36 | 6/36 | 0.20 | 30.4 | 85 | 800 | 1450 | 14962 |
| w6 | 6 | 33/36 | 9/36 | 0.08 | 17.9 | 13 | 900 | — | 16705 |
| w6 | 7 | 14/36 | 0/36 | 0.06 | 26.0 | 12 | 750 | — | 17242 |
| w8 | 0 | 0/36 | 0/36 | 0.01 | — | 85 | — | — | 6010 |
| w8 | 1 | 35/36 | 5/36 | 0.11 | 18.8 | 16 | 950 | — | 8454 |
| w8 | 2 | 34/36 | 3/36 | 0.14 | 18.3 | 23 | 950 | 1350 | 12079 |
| w8 | 3 | 23/36 | 0/36 | 0.09 | 21.9 | 18 | 250 | 1450 | 11811 |
| w8 | 4 | 6/36 | 0/36 | 0.00 | 22.5 | 582 | — | — | 10826 |
| w8 | 5 | 5/36 | 5/36 | 0.03 | 20.8 | 80 | — | — | 9428 |
| w8 | 6 | 31/36 | 0/36 | 0.08 | 19.5 | 42 | 700 | 700 | 12751 |
| w8 | 7 | 33/36 | 0/36 | 0.13 | 26.5 | 42 | 1300 | 1300 | 9946 |
