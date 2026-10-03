# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-traQmania/auditA/retrain/robust_q6_oval

- 8 finished cells in 1 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| gap8an | quantum | oval | q6 | 80 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| gap8an | 1.00 [0.99, 1.00] | 0.89 [0.55, 1.00] | 0.86 [0.79, 0.93] | 13.4 [13.2, 13.7] | 235 [179, 299] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| gap8an | 1.00 [1.00, 1.00] | 1.00 [0.53, 1.00] | 0.86 [0.82, 0.93] | 13.4 [13.2, 13.6] | 231 [157, 314] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| gap8an | 8/8 (100%) | 250 | 275 [250, 350] | 275 [238, 350] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| gap8an | 8/8 (100%) | 300 | 300 [250, 400] | 300 [250, 388] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `gap8an`

(no other variant to compare)

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| gap8an | 0 | 35/36 | 36/36 | 0.68 | 13.3 | 348 | 450 | 500 | 532 |
| gap8an | 1 | 36/36 | 36/36 | 0.93 | 13.5 | 203 | 200 | 200 | 654 |
| gap8an | 2 | 36/36 | 19/36 | 0.86 | 13.1 | 275 | 300 | 300 | 634 |
| gap8an | 3 | 36/36 | 20/36 | 0.92 | 13.5 | 208 | 250 | 250 | 1145 |
| gap8an | 4 | 36/36 | 19/36 | 0.85 | 14.1 | 157 | 250 | 250 | 1037 |
| gap8an | 5 | 36/36 | 36/36 | 0.82 | 13.1 | 254 | 300 | 300 | 1054 |
| gap8an | 6 | 36/36 | 36/36 | 0.82 | 13.6 | 149 | 250 | 400 | 939 |
| gap8an | 7 | 36/36 | 36/36 | 1.00 | 13.2 | 314 | 350 | 350 | 874 |
