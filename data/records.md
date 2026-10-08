# Lap records: every bundled driver on every bundled track

Rendered from `data/records.json` (`python -m racetraq.records --markdown`).
The cells were produced by `python -m racetraq.records --episodes 36 --seed 47000`
on 2026-10-08 in 35 min of wall-clock time on an Apple M1 Max: a greedy
evaluation of each bundled driver on each bundled track, 36 standing-start
episodes per cell from the env seed 47000 (an episode set no driver selection
used), each episode ending at the first crash or at the decision budget.
"Laps" counts every lap in the 36 episodes; "best" and "mean" are over those
laps, the first lap from standstill included. The 10-qubit oval and chicane
drivers are the 6-block files bundled on 2026-10-03 and `quantum_gp_q10` is
still the July 2026 file; every other driver was re-selected on 2026-10-02 (see
each `racetraq/weights/<name>.meta.json`). Rows are sorted by best lap.

### chicane

| driver | kind | qubits | best | mean | laps | episodes lapped |
|---|---|---|---|---|---|---|
| quantum_chicane | quantum | 4 | 12.00 s | 12.65 s | 137 | 35/36 |
| hero | hero | — | 12.10 s | 12.60 s | 144 | 36/36 |
| quantum_chicane_q10 | quantum | 10 | 12.10 s | 12.62 s | 140 | 35/36 |
| quantum_chicane_q8 | quantum | 8 | 12.10 s | 12.64 s | 144 | 36/36 |
| mlp_oval | mlp | — | 12.20 s | 12.71 s | 144 | 36/36 |
| quantum_oval | quantum | 4 | 12.30 s | 13.50 s | 3 | 2/36 |
| quantum_chicane_q6 | quantum | 6 | 12.40 s | 12.94 s | 144 | 36/36 |
| pro | pro | — | 12.50 s | 13.12 s | 144 | 36/36 |
| quantum_oval_q6 | quantum | 6 | 12.70 s | 13.18 s | 144 | 36/36 |
| mlp_chicane | mlp | — | 12.80 s | 13.31 s | 144 | 36/36 |
| quantum_oval_q10 | quantum | 10 | 12.90 s | 13.76 s | 24 | 11/36 |
| quantum_oval_q8 | quantum | 8 | 12.90 s | 13.45 s | 144 | 36/36 |
| mlp_gp | mlp | — | 17.00 s | 17.93 s | 108 | 36/36 |
| quantum_gp | quantum | 4 | 21.50 s | 22.77 s | 72 | 36/36 |
| quantum_combo | quantum | 4 | 25.80 s | 26.80 s | 72 | 36/36 |
| quantum_universal | quantum | 4 | 26.20 s | 27.37 s | 72 | 36/36 |
| mlp_combo | mlp | — | 31.30 s | 32.03 s | 26 | 26/36 |
| quantum_gp_q10 | quantum | 10 | — | — | 0 | 0/36 |

### combo

| driver | kind | qubits | best | mean | laps | episodes lapped |
|---|---|---|---|---|---|---|
| hero | hero | — | 19.00 s | 19.73 s | 45 | 17/36 |
| pro | pro | — | 19.10 s | 20.23 s | 80 | 35/36 |
| mlp_gp | mlp | — | 27.50 s | 27.50 s | 1 | 1/36 |
| quantum_gp | quantum | 4 | 32.40 s | 32.89 s | 21 | 21/36 |
| mlp_combo | mlp | — | 35.70 s | 36.90 s | 36 | 36/36 |
| quantum_combo | quantum | 4 | 36.00 s | 37.22 s | 36 | 36/36 |
| quantum_universal | quantum | 4 | 37.50 s | 38.19 s | 36 | 36/36 |
| mlp_chicane | mlp | — | — | — | 0 | 0/36 |
| mlp_oval | mlp | — | — | — | 0 | 0/36 |
| quantum_chicane_q10 | quantum | 10 | — | — | 0 | 0/36 |
| quantum_chicane_q6 | quantum | 6 | — | — | 0 | 0/36 |
| quantum_chicane_q8 | quantum | 8 | — | — | 0 | 0/36 |
| quantum_chicane | quantum | 4 | — | — | 0 | 0/36 |
| quantum_gp_q10 | quantum | 10 | — | — | 0 | 0/36 |
| quantum_oval_q10 | quantum | 10 | — | — | 0 | 0/36 |
| quantum_oval_q6 | quantum | 6 | — | — | 0 | 0/36 |
| quantum_oval_q8 | quantum | 8 | — | — | 0 | 0/36 |
| quantum_oval | quantum | 4 | — | — | 0 | 0/36 |

### gp

| driver | kind | qubits | best | mean | laps | episodes lapped |
|---|---|---|---|---|---|---|
| hero | hero | — | 16.40 s | 17.03 s | 108 | 36/36 |
| pro | pro | — | 16.50 s | 17.76 s | 108 | 36/36 |
| quantum_gp_q10 | quantum | 10 | 19.70 s | 22.23 s | 44 | 24/36 |
| mlp_gp | mlp | — | 21.40 s | 22.47 s | 71 | 36/36 |
| quantum_gp | quantum | 4 | 26.60 s | 27.93 s | 72 | 36/36 |
| mlp_combo | mlp | — | 32.00 s | 33.08 s | 36 | 36/36 |
| quantum_combo | quantum | 4 | 33.20 s | 34.53 s | 36 | 36/36 |
| quantum_universal | quantum | 4 | 34.00 s | 35.18 s | 36 | 36/36 |
| mlp_chicane | mlp | — | — | — | 0 | 0/36 |
| mlp_oval | mlp | — | — | — | 0 | 0/36 |
| quantum_chicane_q10 | quantum | 10 | — | — | 0 | 0/36 |
| quantum_chicane_q6 | quantum | 6 | — | — | 0 | 0/36 |
| quantum_chicane_q8 | quantum | 8 | — | — | 0 | 0/36 |
| quantum_chicane | quantum | 4 | — | — | 0 | 0/36 |
| quantum_oval_q10 | quantum | 10 | — | — | 0 | 0/36 |
| quantum_oval_q6 | quantum | 6 | — | — | 0 | 0/36 |
| quantum_oval_q8 | quantum | 8 | — | — | 0 | 0/36 |
| quantum_oval | quantum | 4 | — | — | 0 | 0/36 |

### oval

| driver | kind | qubits | best | mean | laps | episodes lapped |
|---|---|---|---|---|---|---|
| mlp_oval | mlp | — | 12.00 s | 12.55 s | 144 | 36/36 |
| quantum_chicane_q10 | quantum | 10 | 12.00 s | 12.55 s | 144 | 36/36 |
| quantum_chicane | quantum | 4 | 12.00 s | 12.58 s | 140 | 35/36 |
| hero | hero | — | 12.10 s | 12.58 s | 144 | 36/36 |
| pro | pro | — | 12.10 s | 12.65 s | 144 | 36/36 |
| quantum_chicane_q8 | quantum | 8 | 12.10 s | 12.59 s | 144 | 36/36 |
| quantum_oval | quantum | 4 | 12.10 s | 12.65 s | 144 | 36/36 |
| quantum_chicane_q6 | quantum | 6 | 12.50 s | 13.05 s | 144 | 36/36 |
| quantum_oval_q6 | quantum | 6 | 12.50 s | 13.06 s | 144 | 36/36 |
| mlp_chicane | mlp | — | 12.60 s | 13.10 s | 144 | 36/36 |
| quantum_oval_q10 | quantum | 10 | 12.80 s | 13.30 s | 144 | 36/36 |
| quantum_oval_q8 | quantum | 8 | 12.80 s | 13.35 s | 144 | 36/36 |
| mlp_gp | mlp | — | 16.50 s | 17.30 s | 108 | 36/36 |
| quantum_gp | quantum | 4 | 21.40 s | 22.23 s | 72 | 36/36 |
| quantum_combo | quantum | 4 | 25.90 s | 26.65 s | 72 | 36/36 |
| quantum_universal | quantum | 4 | 26.20 s | 27.64 s | 72 | 36/36 |
| mlp_combo | mlp | — | 30.60 s | 31.06 s | 36 | 36/36 |
| quantum_gp_q10 | quantum | 10 | — | — | 0 | 0/36 |

