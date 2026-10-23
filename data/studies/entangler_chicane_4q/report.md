# Study report: runs/entangler_chicane

- 16 finished cells in 2 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): b901ebc4a694ad930207b99b590340e471c53b3f

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| cz_ring | quantum | chicane | — | 56 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}`, `circuit.entangler=cz_ring` | 8 | 8/8 |
| none | quantum | chicane | — | 56 | `training.bootstrap_truncation=true`, `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}`, `circuit.entangler=none` | 8 | 0/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| cz_ring | 1.00 [0.94, 1.00] | 0.60 [0.41, 0.89] | 0.64 [0.53, 0.75] | 13.5 [12.9, 14.3] | 301 [234, 450] |
| none | 0.06 [0.00, 0.17] | 0.00 [0.00, 0.05] | 0.00 [0.00, 0.00] (n=7) | 15.8 [15.1, 16.4] (n=5) | 535 [352, 652] |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| cz_ring | 1.00 [0.97, 1.00] | 0.58 [0.50, 1.00] | 0.65 [0.50, 0.70] | 13.4 [12.7, 14.4] | 255 [230, 450] |
| none | 0.06 [0.00, 0.17] | 0.00 [0.00, 0.03] | 0.00 [0.00, 0.00] (n=7) | 15.9 [14.9, 16.6] (n=5) | 544 [302, 673] |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| cz_ring | 8/8 (100%) | 300 | 300 [300, 475] | 338 [275, 475] |
| none | 1/8 (12%) | — | 800 [—] | 800 [—] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| cz_ring | 8/8 (100%) | 450 | 475 [400, 550] | 462 [400, 562] |
| none | 0/8 (0%) | — | — | — |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `cz_ring`

| variant | P(final-params lapped > cz_ring) [95% CI] | P(stability > cz_ring) [95% CI] |
|---|---|---|
| none | 0.08 [0.00, 0.26] | 0.00 [0.00, 0.00] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| cz_ring | 0 | 35/36 | 21/36 | 0.50 | 14.7 | 449 | 450 | 500 | 201 |
| cz_ring | 1 | 29/36 | 18/36 | 0.42 | 12.7 | 246 | 200 | 450 | 149 |
| cz_ring | 2 | 36/36 | 36/36 | 0.99 | 14.7 | 494 | 550 | 550 | 198 |
| cz_ring | 3 | 36/36 | 36/36 | 0.70 | 13.1 | 253 | 300 | 400 | 267 |
| cz_ring | 4 | 36/36 | 24/36 | 0.60 | 13.2 | 212 | 300 | 400 | 218 |
| cz_ring | 5 | 36/36 | 21/36 | 0.66 | 14.0 | 257 | 300 | 500 | 227 |
| cz_ring | 6 | 36/36 | 0/36 | 0.67 | 13.7 | 230 | 300 | 300 | 218 |
| cz_ring | 7 | 36/36 | 20/36 | 0.64 | 12.6 | 450 | 500 | 700 | 155 |
| none | 0 | 0/36 | 0/36 | 0.00 | — | 420 | — | — | 84 |
| none | 1 | 2/36 | 0/36 | 0.00 | 16.6 | 709 | — | — | 80 |
| none | 2 | 5/36 | 0/36 | 0.00 | 15.5 | 302 | — | — | 92 |
| none | 3 | 11/36 | 1/36 | 0.00 | 16.0 | 209 | — | — | 92 |
| none | 4 | 6/36 | 6/36 | — | 15.9 | 595 | 800 | — | 83 |
| none | 5 | 0/36 | 0/36 | 0.00 | — | 631 | — | — | 77 |
| none | 6 | 0/36 | 0/36 | 0.00 | — | 673 | — | — | 89 |
| none | 7 | 2/36 | 0/36 | 0.01 | 14.9 | 494 | — | — | 70 |
