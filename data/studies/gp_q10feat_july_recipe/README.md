# 10-qubit gp, July recipe (audit A/B, 2026-10-01)

Six single runs from the first audit experiment (not a tools/study.py study):
10 qubits, feature observation (5 rays, speed, curvature ahead, lateral offset,
heading error, corner-speed ratio), gp preset as of July 2026 (3000 episodes,
epsilon 1.0 -> 0.05 over 2000, gamma 0.99, time limit terminal), snapshot
selection on 4 distinct eval episodes, 36-episode reliability evals afterwards.

Layouts: `base` = 4 blocks, features in config order; `perm` = 4 blocks, a
hand-made feature-to-qubit reordering (speed, corner-speed ratio, curvature and
the centre ray on the four readout qubits); `L6` = 6 blocks (full light-cone
visibility). Seeds 42 and 0 each. `results.json` has one record per run.

Code: not recorded. The runs used an untracked A/B script and the main
checkout's working tree. They started at about 12:40 on 2026-10-01, when that
checkout's HEAD was `801a5ce` (from the reflog). Any uncommitted changes in it
cannot be checked any more, so this commit is the closest one, not an exact
match (docs/REPORT.md §7).
