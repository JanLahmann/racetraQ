# Study report: runs/entangler_oval

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): b901ebc4a694ad930207b99b590340e471c53b3f

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| cz_ring | quantum | oval | — | 56 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}`, `circuit.entangler=cz_ring` | 8 | 8/8 |
| none | quantum | oval | — | 56 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}`, `circuit.entangler=none` | 8 | 8/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| cz_ring | 1.00 [0.93, 1.00] | 0.73 [0.37, 1.00] | 0.78 [0.53, 0.95] | 13.8 [13.1, 14.2] | 255 [224, 335] |
| none | 1.00 [0.92, 1.00] | 1.00 [0.90, 1.00] | 0.65 [0.43, 0.80] | 14.0 [13.9, 14.2] | 282 [238, 321] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| cz_ring | 1.00 [0.92, 1.00] | 0.72 [0.33, 1.00] | 0.81 [0.49, 0.95] | 13.8 [12.9, 14.2] | 253 [221, 339] |
| none | 1.00 [1.00, 1.00] | 1.00 [0.94, 1.00] | 0.70 [0.43, 0.77] | 14.0 [13.9, 14.2] | 286 [238, 320] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| cz_ring | 8/8 (100%) | 250 | 275 [200, 300] | 262 [200, 325] |
| none | 8/8 (100%) | 400 | 400 [150, 450] | 388 [212, 475] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| cz_ring | 8/8 (100%) | 300 | 300 [200, 400] | 300 [225, 400] |
| none | 7/8 (88%) | 550 | 550 [400, 650] | 532 [354, 654] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `cz_ring`

| variant | P(final-params lapped > cz_ring) [95% CI] | P(stability > cz_ring) [95% CI] |
|---|---|---|
| none | 0.75 [0.50, 0.94] | 0.38 [0.09, 0.66] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| cz_ring | 0 | 36/36 | 3/36 | 0.56 | 12.7 | 268 | 300 | 350 | 225 |
| cz_ring | 1 | 36/36 | 36/36 | 0.35 | 12.9 | 221 | 200 | 200 | 163 |
| cz_ring | 2 | 36/36 | 12/36 | 0.71 | 14.0 | 339 | 400 | 400 | 236 |
| cz_ring | 3 | 36/36 | 33/36 | 0.95 | 14.1 | 246 | 300 | 300 | 369 |
| cz_ring | 4 | 36/36 | 36/36 | 0.97 | 13.7 | 184 | 200 | 200 | 373 |
| cz_ring | 5 | 29/36 | 19/36 | 0.95 | 14.5 | 259 | 300 | 300 | 359 |
| cz_ring | 6 | 33/36 | 17/36 | 0.92 | 13.6 | 247 | 250 | 250 | 321 |
| cz_ring | 7 | 36/36 | 36/36 | 0.49 | 14.2 | 475 | 200 | 550 | 196 |
| none | 0 | 36/36 | 36/36 | 0.96 | 14.0 | 217 | 350 | 400 | 169 |
| none | 1 | 36/36 | 34/36 | 0.77 | 14.0 | 260 | 150 | 150 | 246 |
| none | 2 | 36/36 | 36/36 | 0.77 | 13.8 | 352 | 400 | 550 | 179 |
| none | 3 | 24/36 | 24/36 | 0.35 | 14.5 | 236 | 650 | — | 111 |
| none | 4 | 36/36 | 36/36 | 0.70 | 13.9 | 240 | 400 | 600 | 204 |
| none | 5 | 36/36 | 36/36 | 0.70 | 13.9 | 315 | 450 | 450 | 173 |
| none | 6 | 36/36 | 36/36 | 0.44 | 14.1 | 324 | 400 | 650 | 147 |
| none | 7 | 36/36 | 36/36 | 0.42 | 14.2 | 313 | 100 | 750 | 133 |
