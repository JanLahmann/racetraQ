# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/gp_q10feat

- 6 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| L4 | quantum | gp | — | 128 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30`, `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]` | 3 | 0/3 |
| L6 | quantum | gp | — | 188 | `training.bootstrap_truncation=true`, `training.epsilon_end=0.30`, `circuit.n_qubits=10`, `observation.ray_angles_deg=[-60.0,-30.0,0.0,30.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `circuit.n_layers=6` | 3 | 0/3 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.02 [0.00, 0.11] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 31.8 [—] (n=1) | 2249 [—] (n=1) |
| L6 | 0.13 [0.06, 0.19] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.02] | 34.0 [21.7, 38.7] | 2236 [1786, 2479] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| L4 | 0.00 [0.00, 0.11] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 31.8 [—] (n=1) | 2249 [—] (n=1) |
| L6 | 0.14 [0.06, 0.19] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.02] | 36.0 [21.7, 38.7] | 2287 [1786, 2479] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 0/3 (0%) | — | — | — |
| L6 | 0/3 (0%) | — | — | — |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| L4 | 0/3 (0%) | — | — | — |
| L6 | 0/3 (0%) | — | — | — |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `L4`

| variant | P(final-params lapped > L4) [95% CI] | P(stability > L4) [95% CI] |
|---|---|---|
| L6 | 0.50 [0.50, 0.50] | 0.83 [0.50, 1.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 0 | 0/36 | 0/36 | 0.00 | — | — | — | — | 6489 |
| L4 | 1 | 4/36 | 0/36 | 0.00 | 31.8 | 2249 | — | — | 9166 |
| L4 | 2 | 0/36 | 0/36 | 0.00 | — | — | — | — | 6949 |
| L6 | 0 | 2/36 | 0/36 | 0.01 | 36.0 | 1786 | — | — | 13174 |
| L6 | 1 | 7/36 | 0/36 | 0.00 | 38.7 | 2287 | — | — | 13280 |
| L6 | 2 | 5/36 | 0/36 | 0.02 | 21.7 | 2479 | — | — | 11323 |
