# Study report: /Users/majl/.claude/projects/-Users-majl-GitHub-racetraQ/scienceF/levers_gp

- 32 finished cells in 4 variants; 0 failed, 0 not run yet.
- Intervals: 95% stratified percentile bootstrap over seeds, 2000 resamples, RNG seed 0. IQM = mean of the middle 50% of seeds. With fewer than about 10 seeds these intervals understate the uncertainty; a single seed has none ([—]).
- Lapped fractions come from the reliability evals (distinct greedy episodes of the best snapshot and of the final params); stability and sample complexity from the in-training eval log.
- git commit(s): 3dceebe90baaaeea2755c0ff0625f9078ee5ffab

## Variants

| variant | agent | track | profile | params | overrides | seeds | seeds whose best snapshot laps >= 50% |
|---|---|---|---|---|---|---|---|
| none | quantum | gp | — | 56 | — | 8 | 8/8 |
| gap | quantum | gp | — | 56 | `training.action_gap=0.8` | 8 | 5/8 |
| noise | quantum | gp | — | 56 | `training.act_noise={attenuation=0.95,shots=1024}` | 8 | 3/8 |
| both | quantum | gp | — | 56 | `training.action_gap=0.8`, `training.act_noise={attenuation=0.95,shots=1024}` | 8 | 2/8 |

## IQM over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| none | 0.99 [0.93, 1.00] | 0.01 [0.00, 0.12] | 0.17 [0.12, 0.26] | 34.9 [30.1, 39.2] | 1733 [1645, 1818] |
| gap | 0.75 [0.33, 0.94] | 0.24 [0.00, 0.69] | 0.08 [0.02, 0.22] | 28.7 [24.9, 31.1] (n=7) | 2530 [1968, 2921] (n=6) |
| noise | 0.42 [0.04, 0.83] | 0.00 [0.00, 0.24] | 0.14 [0.08, 0.27] | 44.5 [38.7, 51.4] (n=6) | 2275 [1782, 2704] |
| both | 0.05 [0.00, 0.41] | 0.00 [0.00, 0.10] | 0.02 [0.00, 0.09] | 27.0 [22.4, 28.6] (n=4) | 2686 [2591, 2824] (n=3) |

## Median over seeds [95% CI]

| variant | best-snapshot lapped | final-params lapped | stability | best-snapshot mean lap (s) | first clean lap (episode) |
|---|---|---|---|---|---|
| none | 0.99 [0.94, 1.00] | 0.01 [0.00, 0.08] | 0.16 [0.13, 0.22] | 34.9 [30.1, 40.2] | 1743 [1658, 1825] |
| gap | 0.86 [0.22, 0.97] | 0.14 [0.00, 0.71] | 0.08 [0.00, 0.21] | 29.2 [23.8, 29.8] (n=7) | 2566 [1882, 2957] (n=6) |
| noise | 0.40 [0.00, 0.89] | 0.00 [0.00, 0.03] | 0.14 [0.08, 0.28] | 44.3 [37.7, 52.4] (n=6) | 2261 [1782, 2686] |
| both | 0.03 [0.00, 0.50] | 0.00 [0.00, 0.03] | 0.01 [0.00, 0.07] | 27.0 [22.4, 28.6] (n=4) | 2675 [2591, 2824] (n=3) |

Stability = mean lapped fraction over all in-training evals after the first eval that lapped (a seed that never laps scores 0; one whose first lapping eval is its last has no value). Mean lap and first clean lap are over the n seeds that lapped / drove a clean lap at all.

## Sample complexity

Training episodes until the running-best in-training eval laps in >= 50% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| none | 8/8 (100%) | 1100 | 1225 [1000, 1550] | 1238 [975, 1500] |
| gap | 5/8 (62%) | 1800 | 1450 [700, 2350] | 1465 [925, 2185] |
| noise | 7/8 (88%) | 1050 | 1050 [651, 1850] | 1193 [679, 1868] |
| both | 3/8 (38%) | — | 2550 [800, 2600] | 2267 [800, 2600] |

Training episodes until the running-best in-training eval laps in >= 90% of its episodes:

| variant | seeds that get there | episodes until half of ALL seeds are there | median [95% CI], seeds that get there | IQM [95% CI], seeds that get there |
|---|---|---|---|---|
| none | 8/8 (100%) | 1350 | 1475 [1150, 2275] | 1588 [1125, 2238] |
| gap | 5/8 (62%) | 2550 | 1750 [700, 2850] | 1810 [1015, 2760] |
| noise | 2/8 (25%) | — | 1925 [1650, 2200] | 1925 [1650, 2200] |
| both | 1/8 (12%) | — | 800 [—] | 800 [—] |

"Half of ALL seeds" counts the seeds that never get there (— = fewer than half do), so it is the column to compare variants by; median and IQM are conditional on getting there.

## Versus baseline `none`

| variant | P(final-params lapped > none) [95% CI] | P(stability > none) [95% CI] |
|---|---|---|
| gap | 0.61 [0.33, 0.87] | 0.27 [0.03, 0.55] |
| noise | 0.39 [0.13, 0.63] | 0.42 [0.14, 0.72] |
| both | 0.39 [0.12, 0.64] | 0.11 [0.00, 0.33] |

Probability that a random seed of the variant beats a random seed of the baseline (ties count half): 0.5 = no difference; an interval entirely above 0.5 is a statistically supported improvement.

## Per-seed results

| variant | seed | best snapshot lapped | final params lapped | stability | mean lap (s) | first clean lap | episodes to >= 50% | episodes to >= 90% | train wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 36/36 | 1/36 | 0.21 | 42.0 | 1658 | 550 | 600 | 683 |
| none | 1 | 36/36 | 0/36 | 0.17 | 40.2 | 1950 | 1450 | 2450 | 573 |
| none | 2 | 30/36 | 1/36 | 0.13 | 36.1 | 1695 | 1550 | 2350 | 501 |
| none | 3 | 36/36 | 3/36 | 0.15 | 32.3 | 1750 | 1000 | 1200 | 415 |
| none | 4 | 35/36 | 0/36 | 0.07 | 33.8 | 1495 | 1600 | 1600 | 508 |
| none | 5 | 35/36 | 0/36 | 0.22 | 37.5 | 1748 | 1100 | 1100 | 581 |
| none | 6 | 33/36 | 0/36 | 0.15 | 28.0 | 1738 | 1350 | 1350 | 622 |
| none | 7 | 36/36 | 12/36 | 0.43 | 27.9 | 1825 | 1050 | 2200 | 721 |
| gap | 0 | 32/36 | 32/36 | 0.13 | 29.8 | 2734 | 1800 | 2850 | 661 |
| gap | 1 | 30/36 | 0/36 | 0.04 | 29.4 | — | 700 | 700 | 417 |
| gap | 2 | 8/36 | 10/36 | 0.03 | 22.4 | 2949 | — | — | 405 |
| gap | 3 | 33/36 | 27/36 | 0.34 | 34.0 | 2398 | 2350 | 2550 | 400 |
| gap | 4 | 36/36 | 24/36 | 0.29 | 29.2 | 1968 | 1151 | 1151 | 583 |
| gap | 5 | 0/36 | 0/36 | 0.00 | — | — | — | — | 638 |
| gap | 6 | 35/36 | 0/36 | 0.11 | 28.5 | 1796 | 1450 | 1750 | 514 |
| gap | 7 | 13/36 | 0/36 | 0.00 | 23.8 | 2965 | — | — | 475 |
| noise | 0 | 0/36 | 0/36 | 0.18 | — | 1653 | 651 | — | 722 |
| noise | 1 | 3/36 | 0/36 | 0.08 | 37.5 | 2961 | 2200 | — | 508 |
| noise | 2 | 14/36 | 0/36 | 0.32 | 52.7 | 2522 | 1850 | 2200 | 483 |
| noise | 3 | 34/36 | 34/36 | 0.17 | 38.0 | 1626 | 1050 | — | 473 |
| noise | 4 | 15/36 | 0/36 | 0.08 | 46.0 | 2000 | 600 | — | 567 |
| noise | 5 | 28/36 | 0/36 | 0.11 | 52.0 | 2686 | 1751 | — | 561 |
| noise | 6 | 0/36 | 0/36 | 0.06 | — | 2667 | — | — | 501 |
| noise | 7 | 36/36 | 1/36 | 0.39 | 42.6 | 1912 | 750 | 1650 | 625 |
| both | 0 | 36/36 | 0/36 | 0.07 | 28.6 | 2824 | 800 | 800 | 599 |
| both | 1 | 18/36 | 1/36 | 0.21 | 26.7 | 2591 | 2550 | — | 386 |
| both | 2 | 0/36 | 0/36 | 0.00 | — | — | — | — | 392 |
| both | 3 | 0/36 | 0/36 | 0.00 | — | — | — | — | 306 |
| both | 4 | 2/36 | 0/36 | 0.01 | 27.4 | — | — | — | 540 |
| both | 5 | 5/36 | 13/36 | 0.06 | 22.4 | 2675 | 2600 | — | 637 |
| both | 6 | 0/36 | 0/36 | 0.00 | — | — | — | — | 469 |
| both | 7 | 0/36 | 0/36 | 0.00 | — | — | — | — | 450 |
