# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/pro

- 3 finished cells in 1 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| pro | mlp | multi | — | 2436 | `training.bootstrap_truncation=true`, `circuit.n_qubits=14`, `observation.ray_angles_deg=[-60.0,-45.0,-30.0,-15.0,0.0,15.0,30.0,45.0,60.0]`, `observation.features=["rays","speed","curvature_ahead","lateral_offset","heading_error","corner_speed_ratio"]`, `mlp.hidden=128`, `training.epsilon_decay_episodes=2500`, `training.gamma=0.99` | 3 | 3/3 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| pro | 0.81 [0.75, 1.00] | 0.97 [0.83, 1.00] | 0.68 [0.67, 0.72] | 14.5 [14.0, 15.3] | 1518 [1322, 1584] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| pro | 0.78 [0.75, 1.00] | 1.00 [0.83, 1.00] | 0.68 [0.67, 0.72] | 14.4 [14.0, 15.3] | 1551 [1322, 1584] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| pro | 3/3 (100%) | 250 | 250 [100, 300] | 233 [100, 300] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| pro | 3/3 (100%) | 2100 | 2100 [300, 2250] | 1825 [300, 2250] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `pro`

(no other variant to compare)

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| pro | 0 | 28/36 | 36/36 | 0.72 | 14.4 | 1584 | 300 | 2100 | 1222 |
| pro | 1 | 36/36 | 36/36 | 0.68 | 15.3 | 1322 | 250 | 300 | 1253 |
| pro | 2 | 27/36 | 30/36 | 0.67 | 14.0 | 1551 | 100 | 2250 | 1228 |
