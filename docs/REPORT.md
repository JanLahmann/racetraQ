# racetraQ: technical report on the October 2026 re-measurement

*A variational quantum circuit as the Q-function of a double DQN, measured
against a matched classical baseline with a multi-seed protocol.*

Written 2026-10-03 from the project's records, in this order of authority:
[docs/SCIENCE.md](SCIENCE.md); the study summaries in
[`data/studies/`](../data/studies) (`report.md`, `report.json`, `cells.json`);
the `selection` blocks of `racetraq/weights/*.meta.json`;
[`data/records.md`](../data/records.md); [docs/ARCHITECTURE.md](ARCHITECTURE.md);
notebooks 03–07; the README. Every number is copied from one of those files and
sourced next to its table; nothing was re-run. A non-technical companion is
[EXPLAINER.md](EXPLAINER.md).

## Abstract

racetraQ trains a data re-uploading variational quantum circuit (4–10 qubits,
56–188 parameters) as the Q-function of a double DQN that drives a car around
four tracks from lidar and speed, against a parameter-matched MLP under the
same trainer. An October 2026 audit found the 8- and 10-qubit circuits
structurally blind to some inputs (readout light cones), snapshot selection
made on four episodes counted three times, and headline claims resting on one
to three seeds. Everything was re-measured — 505 training runs in 16 studies,
8–10 seeds per recipe at 4 qubits, interquartile means with stratified
bootstrap intervals, stability, sample complexity, probability of improvement.
The earlier "parity" reading is reversed: the MLP drives its first clean lap in
about half the circuit's episodes on the easy tracks and is far steadier there;
on the hard tracks it laps about 10 s (gp) and 18 s (combo) faster, and neither
agent's end-of-training parameters lap. The circuit does learn all four tracks,
and all four at once; a per-agent 0.30 exploration floor removes a greedy-policy
collapse specific to the 4-qubit circuit; advantage learning plus acting under
emulated noise makes 4-qubit drivers lap a simulated IBM Nighthawk (10 of 10
oval seeds, 8 of 10 chicane); classical kernel surrogates reproduce the trained
driver's laps. No quantum advantage is claimed or possible here. One full lap
on a physical QPU (ibm_marrakesh, 141 decisions, 14.1 s, raw noise, 90 %
action agreement with the exact simulator) is reported as a single data point.

## 1 Setting

### 1.1 Environment

| Item | Definition | Source |
|---|---|---|
| Tracks | `oval`, `chicane` (flat-out), `gp` (hairpin, braking required), `combo` (hairpin plus chicane); half-width 7.0 (oval, chicane) or 6.0 (gp, combo); four checkpoints each | `racetraq/env/tracks/*.json` |
| Time base | 60 Hz physics, 6 substeps per decision: agents decide at 10 Hz and hold the action | `racetraq/config/default.toml` `[physics]` |
| Car (physics v2) | `accel` 11, `brake` 16, `drag` 0.35, `v_max` 25 | same |
| Observation | n features for n qubits: n − 1 lidar rays evenly spaced over [−60°, +60°], normalised by `ray_max_dist` 30, plus speed v/v_max; every scalar in [0, 1]; default 3 rays + speed. Egocentric, no absolute position | SCIENCE.md "The circuit"; `racing_env.py` |
| Actions | 4 discrete: right / straight / left at full throttle, and coast-brake (steer 0, throttle 0, brake 1). 6- and 8-action sets exist; no weights with more than 4 actions are bundled | `racetraq/agents/base.py` |
| Reward | signed centerline progress × 1.0; `checkpoint_bonus` 5 per checkpoint; `lap_bonus` 50; `offtrack_penalty` 10, and leaving the track ends the episode | `default.toml` `[reward]` |
| Time limit | `max_decisions` 600 = 60 s, a truncation (no penalty, `truncated` flag); since the audit the TD target bootstraps through it | same; SCIENCE.md "Trainer options" |
| Training env | 8 parallel cars | `default.toml` `[training]` |

### 1.2 Agents

**Quantum circuit** (`racetraq/agents/quantum/circuit.py`, one definition for
the numpy, `EstimatorQNN` and hardware paths): n qubits, L re-uploading blocks
on |0…0⟩. Block l: encoding `RY(λ[l,i]·s[i])` on every qubit (all n features
re-uploaded every block); variational `RY(θ[l,i,0])` then `RZ(θ[l,i,1])`; CZ
ring `CZ(0,1) … CZ(n−1,0)`. Readout `E_a = ⟨Z_a⟩` on qubits 0–3 — four actions
at any n; extra qubits widen the feature register. Head `Q_a = w[a]·E_a + b[a]`,
essential because useful Q-values are of order 100 while ⟨Z⟩ ∈ [−1, 1]
(bundled drivers end with |w| between 25 and 95).

**P = 3·L·n + 8**, flat vector `[λ, θ, w, b]`: λ L×n (init π), θ L×n×2 (init
U(−0.1, 0.1)), w 4 (init 1), b 4 (init 0).

| n qubits × L blocks | 4 × 4 | 6 × 4 | 8 × 4 | 8 × 5 | 10 × 4 | 10 × 6 |
|---|---|---|---|---|---|---|
| P | 56 | 80 | 104 | 128 | 128 | 188 |

(source: SCIENCE.md "The circuit"; "Variants" tables in `data/studies/*/report.md`.)

Training runs on `fastsim`, a numpy statevector simulator with adjoint
gradients, pinned against `EstimatorQNN` on Aer and finite differences
(`tests/test_fastsim_vs_qiskit.py`, `tests/test_gradients.py`). One batch-32
double-DQN update: ~3.4 ms against ~20.5 s with parameter-shift gradients
(timed before the October 2026 stack upgrade).

**Classical baseline.** A 76-parameter MLP (4-8-4, tanh), chosen for comparable
parameter count and not tuned to win, behind the same flat-vector `QFunction`
interface. Wider observations: 92 / 108 / 124 parameters for 5 / 7 / 9 rays;
`mlp_h32` 292; the expert-menu `pro` driver 14-input, 128-hidden, 2,436
parameters (source: `data/studies/*/report.md` "Variants"; SCIENCE.md "The ceiling").

### 1.3 Training algorithm and shipped recipes

Double DQN in numpy (`agents/training/dqn.py`): replay 10 000, batch 32, Adam
lr 0.01, hard target sync every 200 updates, linear epsilon decay, γ 0.98,
`bootstrap_truncation = true`; the target network is a second flat parameter
vector. Optional keys (Huber, soft targets, per-group rates, lr anneal, grad
clip, reward scale, `act_noise`, `action_gap`) are off unless a recipe sets
them. Recipes resolve per track and agent: `[training]`, then
`[training_presets.<track>]`, then `[training_presets_<agent>.<track>]`.

| Track | Episodes | ε decay (episodes) → floor | γ | Quantum-only layer | MLP |
|---|---|---|---|---|---|
| oval | 800 (studies and bundled drivers; 400 is the live-demo default) | 250 → 0.05 | 0.98 | `action_gap = 0.8`, `act_noise = {attenuation 0.95, shots 1024}` (noise-robust) | base recipe |
| chicane | 800 | 250 → 0.05 | 0.98 | same noise-robust layer | base recipe |
| gp | 3000 | 2000 → **0.30** | 0.99 | `epsilon_end = 0.30` | floor 0.05 |
| combo | 2500 | 1500 → **0.30** | 0.99 | `epsilon_end = 0.30` | floor 0.05 |
| universal (`--track multi`) | 3000 | 2000 → 0.30 | 0.99 | gp-style recipe for both agents | same |

(source: `default.toml`; ARCHITECTURE.md "Training recipes"; SCIENCE.md
"Trainer options".) The 6-, 8- and 10-qubit oval/chicane studies ran
`[training]` with truncation, 800 episodes, no robustness keys; `q8` has 5
blocks and `q10` 6 (section 3); `quantum_oval_q6` comes from a separate
noise-robust study (`data/studies/robust_oval_q6`); `quantum_gp_q10` is a July
2026 single run (4 blocks, engineered-feature observation).

## 2 Evaluation protocol

**Snapshot selection.** Every `eval_every` = 50 training episodes the current
parameters drive `eval_episodes` = 12 greedy episodes on a fresh evaluation
env — 12 parallel cars with their own spawn jitter, so 12 distinct episodes.
Snapshots rank by (episodes lapped, mean lap over all laps, mean return); the
final parameters compete too. Under the noise-robust recipe these evals act
under the emulated noise (SCIENCE.md "Which parameters ship").

**Reliability eval.** `tools/study.py` evaluates each cell's best snapshot
*and* final parameters over 36 distinct greedy episodes (env seed 20 000);
"best-snapshot lapped" and "final-params lapped" are the fractions with a lap.

**Bundling.** `tools/bundle_driver.py` ranks a variant's seeds by that eval,
re-evaluates the top three (four for hardware-demo drivers) on 72 *fresh*
episodes on env seed 31 000, and chooses by (episodes lapped, mean lap) — or by
laps on the simulated device first (`rank_by: "device"`, 8 device episodes) for
`quantum_oval`, `quantum_chicane`, `quantum_oval_q6`. The sidecar records the
fresh numbers, never the shortlisting ones, every candidate and the recipe's
seed spread. A later re-check used 36 episodes on env seed 47 000
(`python -m racetraq.records --episodes 36 --seed 47000`), then 61 000 and
88 000, which no selection saw (sidecars `selection.rule`; SCIENCE.md "The
bundled drivers").

**Seeds.** 8–10 per recipe at 4 qubits (29 gp variants, 6 or 10 seeds each);
8 at 6 qubits; 6 at 8 and at 10; 5 for the universal driver; 3 for `pro`.

**Statistics** (`tools/study.py`, lines 108–234):

- *IQM*: mean of the middle 50 % of per-seed values, the value straddling a
  quartile weighted by the fraction inside.
- *Intervals*: 95 % percentile bootstrap over seeds, 2000 resamples, RNG seed
  0, **stratified** (each variant resampled on its own, size kept); a single
  seed gets none. Every `x [a, b]` below is such an IQM with its interval.
- *Stability*: mean lapped fraction over all in-training evals after the first
  that lapped (1 = it keeps lapping; a run that never laps scores 0).
- *Sample complexity*: episodes until the in-training eval laps in ≥ 50 % (or
  ≥ 90 %) of its episodes, reported as "episodes until half of *all* seeds are
  there", so seeds that never learn count against a variant.
- *Probability of improvement*: P(a random seed of X beats a random seed of Y),
  Mann–Whitney U over n·m pairs with ties half, with a stratified bootstrap
  interval. **Supported** means the interval excludes 0.5. Below about ten
  seeds the intervals understate the uncertainty; every report says so.

**What was wrong before.** Snapshots were chosen on a single 4-episode eval,
later on a 4-car env rebuilt from one seed three times — 4 distinct episodes
counted three times; comparisons rested on one to three seeds and best-snapshot
numbers; the 60 s time limit was a terminal state (Pardo et al. [9]); the wide
circuits hid inputs (section 3); the hardware path targeted retired Falcons. A
snapshot chosen on four episodes is a lucky draw as often as a policy, and
"broadly similar sample efficiency and comparable lap times" did not survive
8–10 seeds and a separate 36-episode evaluation of every run (SCIENCE.md audit
note; "Honest claims").

## 3 Light cone

Pushed backwards through the circuit (Heisenberg picture), a readout `Z_a`
meets only the gates in its light cone, which a nearest-neighbour CZ ring
widens by one qubit per block in each direction. Three exact structural facts
(`agents/quantum/lightcone.py`; `tests/test_lightcone.py` checks them against
fastsim values and adjoint gradients): (1) the last CZ ring never matters and
the final block's RZ angles are dead; (2) block l (from 0) is live only on
qubits within ring distance L − 1 − l of a readout qubit, its RZ gates within
L − 2 − l; (3) `⟨Z_a⟩` sees exactly the features within ring distance L − 1 of
qubit a, so every action sees every feature only when **L ≥ ⌊n/2⌋ + 1**.

| At L = 4, readouts on qubits 0–3 | 4 qubits | 6 qubits | 8 qubits | 10 qubits |
|---|---|---|---|---|
| Circuit parameters (3·L·n) | 48 | 72 | 96 | 120 |
| Dead (zero gradient for every input) | 4 | 12 | 26 | 46 |
| CZ gates that can reach a readout | 12 of 16 | 17 of 24 | 20 of 32 | 21 of 40 |
| Features hidden from each action | 0 | 0 | 1 | 3 |
| Blocks needed for full visibility | 3 | 4 | 5 | 6 |

(source: SCIENCE.md "Light cones"; `python -m racetraq.agents.quantum.lightcone --qubits 10 --layers 4`.)

"Dead" is structural: exactly zero gradient for every input. At 4 qubits the
dead set is the four final-block RZ angles; beyond that whole encoding and RY
gates drop out, so the count grows faster than n — the earlier "n dead
parameters at any qubit count" holds only at n = 4. The 5-block `q8` circuit
has 120 circuit parameters, 26 dead, 28 of 40 CZ live, no hidden feature.
Blind spots at 4 blocks (feature j on qubit j): at 8 qubits Right cannot see
ray +20°, Straight +40°, Left +60°, and **Brake cannot see speed**; at 10 qubits
each action misses three rays (Brake: rays +45°/+60° and speed); the July
`quantum_gp_q10` hides lateral offset, heading error and corner speed from
Brake — the feature built to say "too fast for this corner".

**Consequences.** Every July 2026 result at 8 and 10 qubits was measured with
partially blind circuits on one to three seeds; those results are withdrawn,
not reversed. `q8` now runs 5 blocks and `q10` 6 (section 4.3);
`circuit_spec()` now exports the visibility map and the dead gates, and
`train_headless` warns before training a blind circuit. Parameter count becomes
an imperfect fairness measure (12 of 72 dead at 6 qubits, 26 of 120 at 8
qubits and 5 blocks, 46 of 120 at 10 qubits and 4).

**Hardware pruning.** `lightcone.pruned_circuit` runs only the live gates with
identical expectation values: 12 instead of 16 CZ at 4 qubits. A CZ ring on an
even number of qubits embeds in the Nighthawk square lattice with zero SWAPs;
on heavy-hex Herons it is routed — 37 CZ unpruned and 27 pruned on `fake_fez`
(best of 8 layout seeds), 50 / 38 CX on the retired `fake_manila` (SCIENCE.md
"Hardware"). A readout that depends on a bounded neighbourhood can also be
simulated on that neighbourhood alone (notebook 07).

## 4 Results

Measured 2026-10-01/03, physics v2, `fastsim`, distinct episodes. "Mean lap"
averages every lap of an episode, the standing-start lap included; "first clean
lap" is the training episode in which an exploring car first completes a lap;
"P" is a probability of improvement with its interval.

### 4.1 Circuit against the matched MLP at 4 qubits

Each agent under the recipe that ships for it (section 1.3).

| Track | Agent | Seeds | Best-snapshot lapped | Final-params lapped | Stability | Best-snapshot mean lap | First clean lap (episode) |
|---|---|---|---|---|---|---|---|
| oval | circuit | 10 | 1.00 [0.94, 1.00] | 0.80 [0.47, 0.99] | 0.75 [0.52, 0.92] | 13.7 s [13.2, 14.1] | 264 [236, 341] |
| oval | MLP | 8 | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.00] | 0.94 [0.87, 0.99] | 12.9 s [12.6, 13.2] | 153 [134, 168] |
| chicane | circuit | 10 | 1.00 [0.87, 1.00] | 0.63 [0.39, 0.88] | 0.63 [0.49, 0.73] | 13.6 s [13.0, 14.3] | 355 [247, 478] |
| chicane | MLP | 8 | 1.00 [0.96, 1.00] | 0.92 [0.81, 0.97] | 0.97 [0.93, 0.99] | 13.9 s [13.5, 14.1] | 183 [160, 196] |
| gp | circuit | 10 | 0.96 [0.82, 1.00] | 0.02 [0.00, 0.08] | 0.17 [0.12, 0.22] | 35.0 s [30.8, 40.1] | 1768 [1690, 1916] |
| gp | MLP | 10 | 0.74 [0.37, 0.97] | 0.01 [0.00, 0.38] | 0.14 [0.05, 0.21] | 25.4 s [23.3, 28.1] | 1868 [1761, 1979] |
| combo | circuit | 10 | 0.79 [0.40, 0.97] | 0.03 [0.00, 0.22] | 0.15 [0.03, 0.27] | 46.4 s [42.3, 50.1] | 1563 [1382, 2108] |
| combo | MLP | 10 | 0.82 [0.63, 0.94] | 0.02 [0.00, 0.08] | 0.08 [0.03, 0.14] | 28.6 s [26.7, 32.6] | 1505 [1391, 1630] |

(source: SCIENCE.md "The circuit against the matched MLP"; per-seed data
`data/studies/robust_oval_4q`, `robust_chicane_4q` (circuit, `gap8an`),
`oval_4q`, `chicane_4q` (MLP, `mlp_trunc`), `gp_4q` (`eps30_trunc`,
`mlp_trunc`), `combo_4q` (`quantum`, `mlp_trunc`). Oval and chicane 800
episodes, gp 3000, combo 2500. Mean lap and first clean lap are over the seeds
that lapped at all: 9 of 10 for the MLP on gp and the circuit on combo; 6 of 10
circuit seeds drove a clean lap while exploring on combo. The circuit's
in-training evals on oval and chicane run under emulated device noise.)

- *Easy tracks: the MLP is ahead* — first clean lap in about half the episodes
  (P 0.99 [0.93, 1.00] on both), stability P 0.78 [0.53, 0.95] on the oval and
  0.93 [0.74, 1.00] on chicane, 0.8 s faster on the oval (P 0.82 [0.59, 1.00]);
  chicane lap times do not differ (P 0.36 [0.10, 0.65]).
- *Hard tracks: the MLP is faster, neither is stable* — about 10 s on gp and
  18 s on combo (P 0.92 [0.77, 1.00] and 1.00); stability does not differ
  (P 0.40 [0.15, 0.67] gp, 0.41 [0.15, 0.69] combo).
- *One thing goes the circuit's way on gp — speed of learning, not quality*:
  half of all circuit seeds have a half-lapping greedy eval by episode 1100 and
  all 10 get there; the MLP needs 2150 and 7 of 10 get there
  (`data/studies/gp_4q/report.md`, "Sample complexity"; under the shared July
  recipe 800 against 2250). The circuit finds a slow lapping policy early and
  does not keep it.

**Bundled drivers** (one seed each; "fresh" = the 72-episode selection eval;
"re-check" = 36 episodes on env seed 47 000):

| Driver | Qubits × blocks | Seeds in study | Fresh (72) | Re-check (36) |
|---|---|---|---|---|
| `quantum_oval` | 4 × 4 | 10 | 72, 12.7 s | 36, 12.7 s |
| `quantum_chicane` | 4 × 4 | 10 | 72, 12.6 s | 35, 12.7 s |
| `quantum_gp` | 4 × 4 | 10 | 72, 27.9 s | 36, 27.9 s |
| `quantum_combo` | 4 × 4 | 10 | 72, 37.1 s | 36, 37.2 s |
| `quantum_oval_q6` | 6 × 4 | 8 | 72, 13.1 s | 36, 13.1 s |
| `quantum_chicane_q6` | 6 × 4 | 8 | 72, 12.9 s | 36, 12.9 s |
| `quantum_oval_q8` | 8 × 5 | 6 | 72, 13.4 s | 36, 13.4 s |
| `quantum_chicane_q8` | 8 × 5 | 6 | 72, 12.6 s | 36, 12.6 s |
| `quantum_oval_q10` | 10 × 6 | 6 | 72, 13.3 s | 36, 13.3 s |
| `quantum_chicane_q10` | 10 × 6 | 6 | 72, 12.6 s | 35, 12.6 s |
| `mlp_oval` / `mlp_chicane` | — | 8 / 8 | 72, 12.6 s / 72, 13.3 s | 36, 12.6 s / 36, 13.3 s |
| `mlp_gp` / `mlp_combo` | — | 10 / 10 | 72, 22.5 s / 72, 36.9 s | 36, 22.5 s / 36, 36.9 s |

(source: SCIENCE.md "The bundled drivers"; `racetraq/weights/<name>.meta.json`
`selection.fresh_eval`; `data/records.md`. The 6-block 10-qubit drivers were
bundled after the first re-check, from `data/studies/oval_q10` and
`chicane_q10`, variant `L6`, seed 2 in both; their re-check is the records run
of 2026-10-08. `quantum_gp_q10` stays the July file: 24 of 36 re-check
episodes at 22.2 s.)
Reliability ranks before pace: `mlp_combo` (36.9 s) was preferred to a seed
lapping in 25.6 s that missed one of 72 episodes. On env seeds 61 000 and
88 000 mean laps stay within 0.3 s.

### 4.2 The stabilisation study (gp, 4 qubits, 3000 episodes)

29 variants, 6 seeds each and 10 for the baseline and the variants that
mattered; baseline = July recipe (epsilon 1.0 → 0.05 over 2000 episodes,
γ 0.99, time limit terminal). Selected rows; all 29 are in
`data/studies/gp_4q/report.md` ("IQM over seeds", "Versus baseline `base`").

| Change from the July recipe | Seeds | Best-snapshot lapped | Final-params lapped | Stability | P(stability > baseline) |
|---|---|---|---|---|---|
| none (baseline `base`) | 10 | 0.89 [0.74, 0.97] | 0.00 [0.00, 0.03] | 0.08 [0.04, 0.15] | — |
| Huber loss | 6 | 0.78 [0.39, 1.00] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.10] | 0.25 [0.02, 0.55] |
| soft target update | 6 | 0.92 [0.74, 1.00] | 0.00 [0.00, 0.00] | 0.06 [0.04, 0.18] | 0.47 [0.17, 0.77] |
| lr annealed to 0.001 | 6 | 0.67 [0.20, 0.93] | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.05] | 0.13 [0.00, 0.37] |
| per-group rates (angles 0.003, head 0.03) | 6 | 0.02 [0.00, 0.15] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.03 [0.00, 0.13] |
| reward scale 0.05 | 6 | 0.90 [0.43, 0.98] | 0.00 [0.00, 0.02] | 0.12 [0.05, 0.20] | 0.60 [0.30, 0.90] |
| replay 200 000 | 6 | 0.94 [0.62, 1.00] | 0.00 [0.00, 0.39] | 0.09 [0.04, 0.29] | 0.60 [0.30, 0.88] |
| batch 128 | 6 | 0.91 [0.81, 0.96] | 0.01 [0.00, 0.38] | 0.15 [0.09, 0.30] | 0.78 [0.52, 0.98] |
| target sync every 1000 | 6 | 0.24 [0.05, 0.44] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.06] | 0.13 [0.00, 0.40] |
| bootstrap through the time limit | 6 | 0.94 [0.78, 0.99] | 0.06 [0.00, 0.44] | 0.13 [0.04, 0.47] | 0.57 [0.25, 0.87] |
| exploration floor 0.30 | 10 | 0.94 [0.81, 1.00] | 0.16 [0.03, 0.48] | 0.18 [0.13, 0.21] | 0.77 [0.51, 1.00] |
| floor 0.30 + truncation (**ships**) | 10 | 0.96 [0.82, 1.00] | 0.02 [0.00, 0.08] | 0.17 [0.12, 0.22] | 0.75 [0.49, 0.95] |
| floor 0.30 + truncation + batch 128 | 10 | 0.75 [0.29, 0.97] | 0.00 [0.00, 0.14] | 0.12 [0.05, 0.20] | 0.57 [0.32, 0.82] |
| MLP, July recipe | 10 | 0.87 [0.54, 0.97] | 0.17 [0.00, 0.48] | 0.17 [0.05, 0.34] | 0.63 [0.35, 0.87] |
| MLP + truncation (**ships for the MLP**) | 10 | 0.74 [0.37, 0.97] | 0.01 [0.00, 0.38] | 0.14 [0.05, 0.21] | 0.61 [0.33, 0.85] |
| MLP, floor 0.30 + truncation | 10 | 0.21 [0.02, 0.61] | 0.01 [0.00, 0.03] | 0.03 [0.00, 0.11] | 0.32 [0.11, 0.58] |
| MLP, hidden 32 (292 parameters) | 6 | 1.00 [0.98, 1.00] | 0.31 [0.04, 0.86] | 0.24 [0.19, 0.34] | 0.88 [0.68, 1.00] |

- *What did not help*: Huber, soft target, lr decay, per-group rates, reward
  scaling, a 20× replay, slower target sync, removing the lap and checkpoint
  bonuses, or Huber + soft target + lr decay + truncation together (2 of 6
  seeds reliable); several hurt. Truncation — a real bug, fixed — is neutral
  on gp. Batch 128 just clears the bar at 6 seeds
  (0.78 [0.52, 0.98]; lower end 0.50–0.53 under other bootstrap seeds) and
  added nothing on top of the floor at 10 seeds, so it was not adopted. In no
  quantum variant do the end-of-training parameters lap reliably.
- *The exploration diagnostic* — in-training greedy evals of 10 seeds pooled
  and binned by training episode (about 1200 episodes per cell):

| Lapped fraction of greedy evals, episodes | 1–500 | 501–1000 | 1001–1500 | 1501–2000 | 2001–2500 | 2501–3000 |
|---|---|---|---|---|---|---|
| circuit, floor 0.05 (July recipe) | 0.00 | 0.12 | 0.15 | 0.12 | 0.00 | 0.07 |
| circuit, floor 0.30 + truncation | 0.00 | 0.05 | 0.15 | 0.17 | 0.25 | 0.15 |
| MLP, floor 0.05 (July recipe) | 0.00 | 0.02 | 0.00 | 0.04 | 0.26 | 0.28 |
| MLP, floor 0.05 + truncation | 0.00 | 0.02 | 0.00 | 0.03 | 0.27 | 0.16 |
| MLP, floor 0.30 + truncation | 0.01 | 0.03 | 0.00 | 0.00 | 0.02 | 0.06 |

(source: SCIENCE.md "Training stability", from the eval logs in
`data/studies/gp_4q/cells.json`.) The circuit drives best while epsilon is
still falling and collapses once it sits at 0.05 — 3 lapped episodes of 1224 in
episodes 2001–2500; the MLP does the opposite.

- *The adopted recipe: a modest, partly unsupported gain.* The 0.30 floor
  removes the collapse: stability 0.18 against 0.08 (P 0.77 [0.51, 1.00]), all
  10 seeds reliable (9 of 10 before). It costs pace on the easy tracks (chicane
  16.1 s [14.8, 18.8] against 13.9 s [13.2, 14.4], 8 seeds;
  `data/studies/chicane_4q`), so it is a gp and combo preset only. Adding
  truncation leaves stability where it was (0.17; P 0.75 [0.49, 0.95], at the
  edge of what 10 seeds support) and makes the end of training worse (0.02
  [0.00, 0.08] against 0.16 [0.03, 0.48]).
- *Not universal.* The MLP does worse under the floor on every measure, without
  a supported margin: 3 of 10 seeds reliable against 7 of 10; P(0.30 seed beats
  0.05 seed) 0.28 [0.07, 0.51] for the best snapshot, 0.27 [0.06, 0.52] for
  stability (10 seeds each, both with truncation; notebook 04). Hence the
  per-agent preset. The 10-qubit engineered-feature circuit fails under it too:
  best snapshots of three seeds lapped 0, 4 and 0 of 36 at 4 blocks and 2, 7
  and 5 at 6 blocks (`data/studies/gp_q10feat`), where single July-recipe runs
  lapped 6 to 24 of 36 (`data/studies/gp_q10feat_july_recipe`).
- *Capacity is not the lever*: 6 blocks 27.4 s with 3 of 6 reliable seeds, 8
  blocks 26.3 s with 5 of 6, baseline 34.7 s with 9 of 10; plain 6 qubits does
  not learn gp under the July recipe (0 of 6). The one clearly steadier learner
  is the MLP with 32 hidden units.

### 4.3 Scaling: 4, 6, 8, 10 qubits and depth

Oval and chicane under one recipe (`[training]` with truncation, 800 episodes,
no noise-robust options), next to an MLP on the same observation.

| Track | Agent | Params | Seeds | Best-snapshot lapped | Final-params lapped | Stability | Mean lap | First clean lap |
|---|---|---|---|---|---|---|---|---|
| oval | 4 qubits, 4 blocks | 56 | 8 | 1.00 [0.98, 1.00] | 0.84 [0.57, 0.97] | 0.69 [0.57, 0.75] | 13.9 s [13.3, 14.4] | 294 [238, 369] |
| oval | 6 qubits, 4 blocks | 80 | 8 | 1.00 [0.99, 1.00] | 1.00 [0.81, 1.00] | 0.75 [0.58, 0.85] | 13.3 s [13.1, 13.7] | 276 [227, 311] |
| oval | 8 qubits, 4 blocks | 104 | 6 | 1.00 [0.60, 1.00] | 0.36 [0.05, 0.60] | 0.46 [0.20, 0.66] | 14.2 s [13.5, 14.4] | 285 [217, 354] |
| oval | 8 qubits, 5 blocks | 128 | 6 | 0.99 [0.90, 1.00] | 0.88 [0.62, 0.99] | 0.53 [0.45, 0.65] | 14.4 s [13.7, 14.8] | 255 [208, 386] |
| oval | 10 qubits, 4 blocks | 128 | 6 | 0.92 [0.86, 0.97] | 0.36 [0.06, 0.85] | 0.42 [0.27, 0.50] | 14.3 s [13.7, 14.7] | 293 [234, 343] |
| oval | 10 qubits, 6 blocks | 188 | 6 | 0.95 [0.77, 1.00] | 0.68 [0.19, 1.00] | 0.51 [0.26, 0.63] | 13.3 s [12.8, 14.5] | 292 [255, 376] |
| oval | MLP, 3 rays | 76 | 8 | 1.00 [1.00, 1.00] | 1.00 [0.98, 1.00] | 0.94 [0.87, 0.99] | 12.9 s [12.6, 13.2] | 153 [134, 168] |
| oval | MLP, 5 rays | 92 | 8 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.87 [0.81, 0.93] | 12.8 s [12.7, 13.3] | 134 [111, 160] |
| oval | MLP, 7 rays | 108 | 6 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.91 [0.84, 0.95] | 12.7 s [12.6, 12.8] | 132 [118, 155] |
| oval | MLP, 9 rays | 124 | 6 | 1.00 [0.97, 1.00] | 0.99 [0.55, 1.00] | 0.89 [0.87, 0.93] | 12.6 s [12.6, 12.6] | 118 [107, 141] |
| chicane | 4 qubits, 4 blocks | 56 | 8 | 0.94 [0.68, 0.99] | 0.54 [0.22, 0.78] | 0.47 [0.41, 0.55] | 13.9 s [13.4, 14.2] | 387 [279, 555] |
| chicane | 6 qubits, 4 blocks | 80 | 8 | 1.00 [0.98, 1.00] | 0.78 [0.40, 1.00] | 0.68 [0.45, 0.83] | 13.4 s [13.2, 14.0] | 304 [252, 357] |
| chicane | 8 qubits, 4 blocks | 104 | 6 | 0.90 [0.47, 1.00] | 0.28 [0.04, 0.57] | 0.41 [0.28, 0.74] | 13.3 s [12.9, 14.0] | 345 [251, 495] |
| chicane | 8 qubits, 5 blocks | 128 | 6 | 1.00 [0.93, 1.00] | 0.37 [0.07, 0.80] | 0.47 [0.31, 0.65] | 13.2 s [12.7, 14.3] | 277 [235, 425] |
| chicane | 10 qubits, 4 blocks | 128 | 6 | 0.99 [0.95, 1.00] | 0.55 [0.24, 0.85] | 0.46 [0.31, 0.70] | 13.6 s [13.2, 13.9] | 354 [274, 525] |
| chicane | 10 qubits, 6 blocks | 188 | 6 | 0.99 [0.76, 1.00] | 0.55 [0.12, 0.93] | 0.45 [0.32, 0.66] | 13.1 s [12.7, 13.4] | 341 [298, 416] |
| chicane | MLP, 3 rays | 76 | 8 | 1.00 [0.96, 1.00] | 0.92 [0.81, 0.97] | 0.97 [0.93, 0.99] | 13.9 s [13.5, 14.1] | 183 [160, 196] |
| chicane | MLP, 5 rays | 92 | 8 | 1.00 [1.00, 1.00] | 0.99 [0.40, 1.00] | 0.89 [0.81, 0.97] | 14.0 s [13.5, 14.2] | 168 [144, 184] |
| chicane | MLP, 7 rays | 108 | 6 | 1.00 [1.00, 1.00] | 0.99 [0.62, 1.00] | 0.95 [0.88, 0.98] | 12.9 s [12.7, 13.6] | 143 [138, 162] |
| chicane | MLP, 9 rays | 124 | 6 | 1.00 [1.00, 1.00] | 0.97 [0.44, 1.00] | 0.93 [0.80, 0.98] | 12.7 s [12.7, 12.8] | 168 [156, 179] |

(source: SCIENCE.md "Scaling and the light cone"; per-seed data
`data/studies/oval_4q`, `chicane_4q` (`trunc`, `mlp_trunc`), `oval_q6`,
`chicane_q6`, `oval_q8`, `chicane_q8`, `oval_q10`, `chicane_q10`.)

- *Six qubits is where the circuit does best*: 8 of 8 reliable seeds on both
  tracks; the one supported difference against 4 qubits is the chicane
  best-snapshot fraction (P 0.80 [0.59, 0.98]) — "no worse, probably steadier".
  Chicane laps are about as fast as the matched MLP's (13.4 s against 14.0 s;
  P 0.66 [0.36, 0.94], not supported) while the MLP is steadier (P 0.81
  [0.56, 1.00]); on the oval the MLP is faster (P 0.88 [0.62, 1.00]) and
  steadier (P 0.81 [0.56, 0.98]). Under the noise-robust recipe the 6-qubit
  circuit reaches the oval MLP's stability (0.86 [0.79, 0.93] against 0.87
  [0.81, 0.93]; `data/studies/robust_oval_q6`).
- *8 qubits, 4 against 5 blocks.* Five blocks repair the oval's end of training
  (P 0.92 [0.72, 1.00] for final-params lapped — the one supported difference
  of ten comparisons) and give 6 of 6 reliable seeds on both tracks against 5
  of 6; on chicane the end of training stays poor at either depth (P 0.61
  [0.25, 0.92]). Lap times do not differ. Against 6 qubits the 5-block circuit
  is less stable and slower on the oval (P 0.16 [0.00, 0.41], 0.12
  [0.00, 0.38]). Notebook 06's control — deeper 4-qubit circuits on gp, more
  blocks without more visibility, did not become more reliable — makes
  visibility the simplest reading of the fifth block's effect, not a proof.
- *10 qubits, 4 against 6 blocks.* Oval: the same direction, unsupported at 6
  seeds (final-params 0.68 against 0.36, P 0.71 [0.38, 1.00]; laps 13.3 s
  against 14.3 s, P 0.75 [0.42, 1.00]). Chicane: indistinguishable (stability
  P 0.50 [0.17, 0.83]; laps P 0.51 [0.17, 0.88]). The 4-block circuit is less
  stable than the 4-qubit one on the oval (P 0.07 [0.00, 0.25]); the 6-block
  one is less stable than the 6-qubit circuit (P 0.12 [0.00, 0.38]) and no
  faster; the 9-ray MLP is steadier and laps earlier in every pairing (P 1.00).
  Decision: `q10` runs 6 blocks — full visibility costs nothing measurable.
- *gp above 4 qubits.* Plain 6 qubits: 0 of 6 seeds (July recipe). The
  10-qubit feature circuit under the July recipe, single runs on 36 episodes:
  4 blocks lapped 6 (seed 0) and 24 (seed 42); 6 blocks 16 and 17; a hand-made
  reordering (speed, corner speed, curvature and the centre ray on the readout
  qubits) 2 and 0 — steadier at 6 blocks, not better, two runs per depth
  (`data/studies/gp_q10feat_july_recipe/results.json`, `best_snapshot_36`).
  The bundled `quantum_gp_q10` is faster than the bundled 4-qubit gp driver
  (22.2 s against 27.9 s), level with the bundled MLP (22.5 s) and the least
  reliable of the three.

### 4.4 Generalisation

Zero-shot, greedy, 36 episodes per cell on env seed 47 000 — episodes that lap,
and the mean lap:

| Driver | oval | chicane | gp | combo |
|---|---|---|---|---|
| `quantum_oval` | 36, 12.7 s | 2 | 0 | 0 |
| `quantum_chicane` | 35, 12.6 s | 35, 12.7 s | 0 | 0 |
| `quantum_gp` | 36, 22.2 s | 36, 22.8 s | 36, 27.9 s | 21, 32.9 s |
| `quantum_combo` | 36, 26.7 s | 36, 26.8 s | 36, 34.5 s | 36, 37.2 s |
| `quantum_universal` | 36, 27.6 s | 36, 27.4 s | 36, 35.2 s | 36, 38.2 s |
| `mlp_oval` | 36, 12.6 s | 36, 12.7 s | 0 | 0 |
| `mlp_chicane` | 36, 13.1 s | 36, 13.3 s | 0 | 0 |
| `mlp_gp` | 36, 17.3 s | 36, 17.9 s | 36, 22.5 s | 1 |
| `mlp_combo` | 36, 31.1 s | 26, 32.0 s | 36, 33.1 s | 36, 36.9 s |

(source: SCIENCE.md "One driver, every track"; `data/records.md`. On env seeds
61 000 and 88 000 no cell moves by more than three episodes except `quantum_gp`
on combo, 16–22 of 36.) Transfer runs downhill: braking-trained drivers lap
the flat-out tracks; the reverse never happens.

**Universal driver and the selection lesson.** One 4-qubit circuit trained
round-robin on all four tracks (3000 episodes, 5 seeds): best snapshots lap in
0.89 [0.68, 1.00] of 36 episodes (9 per track); fine-tuning the July universal
driver did worse (0.64 [0.49, 0.85]; stability P 1.00 for the fresh run);
final-params 0.47 [0.25, 0.76], stability 0.35 [0.28, 0.48]. An MLP under the
identical recipe: 0.95 [0.60, 1.00], stability 0.48 [0.46, 0.49] — the 0.30
floor being the circuit's recipe (`data/studies/universal_4q/report.md`). The
rule would have shipped seed 0; on ten generated tracks (`env/trackgen`, seeds
100–109, 12 episodes each) it completes no lap:

| Seed | Four bundled tracks (36 episodes each) | Generated, difficulty 0.5 | Generated, difficulty 0.65 |
|---|---|---|---|
| 0 (top-ranked) | 36, 36, 36, 35 at 13.7 / 13.9 / 32.3 / 41.8 s | 0 of 120 | 0 of 120 |
| 1 | 36, 34, 36, 31 at 20.8 / 21.8 / 31.0 / 36.9 s | 120 of 120 | 41 of 120 |
| 2 | 36, 1, 21, 28 | 6 of 120 | 0 of 120 |
| 3 (**bundled**) | 36, 36, 36, 36 at 27.6 / 27.4 / 35.2 / 38.2 s | 120 of 120 | 120 of 120 |
| 4 | 36, 20, 13, 18 | 115 of 120 | 47 of 120 |

(source: SCIENCE.md "Unseen tracks". Two further sets — seeds 300–309 on env
seed 61 000, 500–509 on 88 000 — repeat it: seed 0 0 of 120 at both
difficulties, seed 3 120 of 120 at both; `quantum_gp` 120 of 120 everywhere,
`quantum_combo` 120 and 106–108, `mlp_gp` 120 and 113–116.) Seed 3 ships,
at about twice seed 0's easy-track lap time. A rule that ranks only on the
training tracks picks a specialist. It was first chosen by hand; since
2026-10-09 `tools/bundle_driver.py --unseen 10` ranks by tracks lapped
reliably over the bundled and ten generated tracks and picks seed 3 (287 of
288 fresh bundled episodes, 120 of 120 generated; seed 0: 283 and 0). For scale, the model-based `hero` reference drives best laps of
12.1 / 12.1 / 16.5 / 19.0 s and the 2,436-parameter `pro` MLP matches it on
oval and gp with the same four actions (SCIENCE.md "The ceiling").

### 4.5 Hardware under noise

Device numbers are on the local Aer twin of `fake_miami`, a calibration
snapshot of an IBM Nighthawk (120 qubits, square lattice); only the 4 or 6
physical qubits the routed circuit touches are simulated (~0.1 s per
decision), 1024 shots per decision, `resilience_level` 0, no rescale. Source
for this subsection unless stated: SCIENCE.md "Hardware".

**Noise model and validation** (`agents/quantum/noise.py`, fitted with one
job): `⟨Z⟩_device ≈ f · ⟨Z⟩_exact`, f = 0.95 (0.945–0.956 over 25 parameter
sets), per-readout slopes 0.94–0.96; TREX moves f to about 0.98; shot-noise
variance matched the binomial formula within 4 %. Over 25 drivers the
emulation's lap counts correlated 0.96 with 12 device episodes each, but it is
a screening tool — judge a driver on the device path.

**The knife-edge finding.** The July 4-qubit oval driver completed no lap on
the simulated device (0 of 14 logged runs): the median gap between its best and
second-best Q-value was 0.46 at |w| ≈ 52, about one standard deviation of
1024-shot noise. Exactly it lapped 27 of 36; emulated, 10 of 36 with shot noise
alone, 33 of 36 with the common attenuation alone, 2 of 36 with only the ±1 %
differences between readout slopes. For eight fresh default-recipe drivers
(seeds 100–107; mean laps of 36): exact 27.5, attenuation 25.1, slopes and
biases 27.6, shot noise 22.6, all together 16.6. Inference-time mitigation (more
shots, per-readout rescale, TREX) did not rescue the July driver; it does
rescue drivers with moderate margins.

**The noise-robust recipe** — `action_gap = 0.8` [23] and `act_noise =
{attenuation 0.95, shots 1024}` — was chosen among seven on seeds 0–7. The
800-episode studies measure it on every seed, nothing selected:

| Track, qubits | Recipe | Seeds | Device laps over all seeds | Seeds lapping ≥ 11 of 12 | Laps per seed (of 12) |
|---|---|---|---|---|---|
| oval, 4 | truncation only | 8 | 73 / 96 | 5 | 12, 11, 0, 12, 10, 5, 11, 12 |
| oval, 4 | noise-robust | 10 | 118 / 120 | 10 | 12, 12, 12, 12, 12, 12, 11, 11, 12, 12 |
| chicane, 4 | truncation only | 8 | 55 / 96 | 0 | 10, 5, 0, 10, 9, 8, 8, 5 |
| chicane, 4 | noise-robust | 10 | 108 / 120 | 8 | 9, 12, 12, 11, 12, 12, 12, 12, 12, 4 |
| oval, 6 | truncation only | 8 | 92 / 96 | 7 | 12, 12, 12, 12, 11, 12, 9, 12 |
| oval, 6 | noise-robust | 8 | 95 / 96 | 8 | 12, 12, 12, 12, 11, 12, 12, 12 |

(source: SCIENCE.md "What survives the noise", `tools/hw_reliability.py`; the
same seeds' exact-simulator statistics are in `data/studies/oval_4q`,
`chicane_4q`, `robust_oval_4q`, `robust_chicane_4q`, `oval_q6`,
`robust_oval_q6`.) At 4 qubits the recipe decides it; at 6 the circuit is
nearly as robust without it. The mechanism is the margin: the bundled 4-qubit
oval driver's median gap is 4.0 at |w| ≈ 65 (about three standard deviations);
truncation-only seed 4 (36 of 36 exact, 52 of 72 device episodes) has 0.7.
Costs at 400 episodes: 5 of 24 recipe seeds did not learn to lap reliably, and
its exact lap count was no better than the default's (25.9 against 27.5 of
36). Which lever does the work is unresolved; they were only tested together.

| Bundled driver | Exact simulator | Emulated noise | Device path |
|---|---|---|---|
| `quantum_oval`, 4 qubits | 36 / 36 | 36 / 36 | 24 / 24 |
| `quantum_chicane`, 4 qubits | 35 / 36 | 36 / 36 | 24 / 24 |
| `quantum_oval_q6`, 6 qubits | 36 / 36 | 36 / 36 | 24 / 24 |
| `quantum_chicane_q6`, 6 qubits | 36 / 36 | 36 / 36 | 11 / 12 |
| `quantum_gp`, 4 qubits | 36 / 36 | 25 / 36 | 9 / 12 |
| `quantum_combo`, 4 qubits | 36 / 36 | 12 / 36 | 4 / 12 |
| `quantum_universal` on oval | 36 / 36 | 25 / 36 | 9 / 12 |
| `quantum_universal` on chicane | 36 / 36 | 11 / 36 | 3 / 12 |

(device episodes capped at 200 decisions, 400 on gp, 520 on combo; three
further sets gave 72 of 72 device episodes for each of the first three
drivers; 8- and 10-qubit drivers not measured on the device path; the lower
rows were trained without the robustness keys.)

**SPSA sprint.** Replay batch and TD targets are computed once on the exact
simulator; the device evaluates only the TD loss, two Estimator jobs per
iteration regardless of parameter count [11]. A loss evaluation scatters by
10–18 %, so the default sprint is hedged: output head only, calibrated bounded
steps, blocking, and a simulator guard that refuses a step costing more than
10 % of the exact greedy return; ten iterations cost about 40 jobs (39–45). On
the July knife-edge driver the first version (all 56 parameters, every step
taken) kept the policy in 6 of 16 runs, while the hardware loss still fell in
7 of the last 8; the default sprint kept it in 20 of 20, with the loss lower by
more than two standard errors in 15 of 20 — and never made that driver lap on
the device (1 of 8 before, 0 of 8 after each of four sprints). On the current
driver two sprints took 2 and 3 of 10 steps and changed neither loss nor return
measurably. The sprint demonstrates the mechanics of updating parameters
against a device; it has not been shown to improve a driver.

### 4.6 Classical surrogates

The trained circuit is a partial Fourier series whose frequencies are fixed by
the encoding and the trained scalings λ [3]; `agents/quantum/surrogate.py` and
notebook 07 fit classical surrogates [13] from black-box samples (executed
2026-10-03 on the drivers bundled now; source for every row: SCIENCE.md
"Honest claims"):

| Quantity | Value |
|---|---|
| Frequency vectors the 4-qubit, 4-block encoding allows per readout | 531,441 |
| … that can carry a non-zero coefficient (trained and random angles alike) | 45,072 |
| Kernel surrogate fitted to 1,000 driving states, 36-episode eval | 36 of 36 on oval and gp (4 qubits) and oval (6 qubits), like the circuits; single fits, two refits each on gp and the 6-qubit oval |
| Controls on the oval: untrained-frequency kernel, generic Gaussian kernel | 36 of 36 each; nRMSE on driving states 0.090 and 0.108 against 0.082 for the trained-frequency kernel (0.02–0.08 over 20 fits in `tests/test_surrogate.py`) |
| Least squares on the predicted frequencies, small circuits | exact to better than 10⁻¹³; frequencies scaled by 1.3 leave errors of 0.05 to 1.7 in four of five circuits |
| Fitted on the uniform cube: true-frequency kernel against Gaussian | nRMSE 0.006 against 0.245; laps 36 of 36 against 0 of 36 |

A surrogate that laps is weak evidence for the Fourier picture — the controls
lap too. The evidence is in the function: exact reproduction, and the cube fit
where only the true frequencies still drive. Only inference is dequantized;
fitted Q-iteration on the surrogate [15] was not run.

## 5 Honest assessment

**Shown.** A 56-parameter circuit learns each track from scratch and one
4-qubit circuit learns all four at once; at 6 qubits it is reliable on both easy tracks in 8 of 8 seeds with
chicane laps level with the MLP's; on gp its seeds reach a lapping greedy
policy in about half the MLP's episodes; every re-selected bundled driver laps
72 of 72 fresh episodes; three lap 72 of 72 device episodes on the simulated
Nighthawk, and the recipe behind that works on 10 of 10 and 8 of 10 unselected
seeds.

**Not shown, not claimed.** Parity or advantage: under 8–10 seeds the MLP is
ahead on first clean lap and stability on the easy tracks and on pace on the
hard ones, as the benchmarking literature predicts [6, 7, 8]. "The agent
learned gp" overstates what either optimizer holds on to — the shipped gp and
combo drivers are selected snapshots of an unstable optimisation (stability
0.17 and 0.15, final-params 0.02 and 0.03) and a 29-variant study found no
setting that fixes it. No quantum advantage is possible here: 4–10 qubits are
trivially simulable, training ran on a classical simulation, the function is a
Fourier series a classical regression recovers, and a racing toy has none of
the properties an advantage claim needs [16, 19, 20, 21]. Engineered features
and the July 8/10-qubit mechanisms were never re-measured.

**Threats to validity.**

- *Simulated device only*: one calibration snapshot of one Nighthawk, a 4- or
  6-qubit patch — no drift, crosstalk or queue; one lap on heavy-hex `fake_fez`
  (27 CZ, TREX) is one run; no physical-QPU lap.
- *Seed counts*: 6 at 8 and 10 qubits, 5 for the universal driver, 3 for
  `pro`, 2–3 per arm at 10-qubit gp; intervals at these n understate the
  uncertainty, and single supported marks among many comparisons deserve little
  weight (notebook 06 counts the marks chance alone would produce).
- *Recipes tuned on the same tracks*: the circuit's recipe has options the
  MLP's does not and was developed on these four tracks; the robustness recipe
  was chosen among seven on seeds 0–7.
- *Bootstrap at small n*: a percentile bootstrap over 6–10 seeds; "supported"
  (interval excluding 0.5) is a threshold, not a controlled error rate across
  the dozens of comparisons made.
- *Selection effects*: a bundled driver's numbers come after two selection
  steps (best snapshot, best seed), and the 72-episode fresh eval still picks
  among 3–5 candidates.
- *Denominators*: comparisons are per training episode; per second the MLP
  trains far faster, and from 6 qubits a growing share of circuit parameters is
  dead.

## 6 Relation to the literature

(References are SCIENCE.md's, by number.) The method is Chen et al.'s [1] — a
VQC as the Q-function of experience-replay Q-learning — with Skolik, Jerbi &
Dunjko's [2] data re-uploading and trainable input/output scaling; Schuld,
Sweke & Meyer [3] explain why re-uploading works and underpin section 4.6;
Meyer et al.'s survey [4] situates value-based QRL; qiskit-machine-learning's
`EstimatorQNN` [5] is the reference the simulator is pinned against. The
protocol follows Agarwal et al. [7] (interquartile means, interval estimates
over runs) and Meyer et al.'s QRL benchmarking [6], whose outperformance
criterion and sample-complexity estimator are the model for sections 2 and 4.
Bowles, Ahmed & Schuld [8] supply the prior: out-of-the-box
classical models win, and "it learns" is not evidence that "quantum" is the
active ingredient. Pardo et al. [9] motivate `bootstrap_truncation`. Skolik et
al. [10] studied shot, coherent and incoherent noise in variational RL; the
narrow addition here is that a closed-loop policy with a 25–95× output head
lives or dies by its action margins, and that margins can be trained with
Bellemare et al.'s gap-increasing operator [23]. SPSA is Spall's [11]; QN-SPSA
[12] was not used. Section 4.6 is the classical surrogate of Schreiber, Eisert
& Meyer [13], with RFF dequantization conditions from Sweke et al. [14],
kernelized fitted Q-iteration guarantees from Rodriguez-Grasa et al. [15] (not
run) and Cerezo et al.'s [16] perspective that provable trainability tends to
come with classical simulability. IBM's plan and device documentation [17, 18,
22] fixes the hardware path; the advantage framework [20], the tracker [21] and
IBM's teaching material [19] set the bar this project does not meet.

## 7 Reproducibility

- `tools/study.py run` trains a (variant × seed) grid, one resumable
  subprocess per cell, and evaluates every best snapshot and final parameter
  set over 36 distinct episodes; `study.py report` writes `report.md` /
  `report.json`, a pure function of the result files (fixed bootstrap seed).
- `tools/export_study.py STUDY_DIR --name NAME` writes the committable
  `data/studies/NAME/{report.md, report.json, cells.json}`; `cells.json` holds
  every seed's eval log and both 36-episode evals. Sixteen studies are
  committed, 505 finished cells by their headers (`gp_4q` 202, `oval_4q` 56,
  `chicane_4q` 56, `combo_4q` 35, `oval_q6` 16, `chicane_q6` 16, `oval_q8` 18,
  `chicane_q8` 18, `oval_q10` 18, `chicane_q10` 18, `gp_q10feat` 6,
  `universal_4q` 15, `pro` 3, `robust_oval_4q` 10, `robust_chicane_4q` 10,
  `robust_oval_q6` 8), plus six single runs in
  `gp_q10feat_july_recipe/results.json`. Weights and full logs are not in the
  repository; SCIENCE.md's "487 training runs" predates the `chicane_q10` export.
- `tools/bundle_driver.py --study DIR --variant V --name N` ranks,
  re-evaluates on 72 fresh episodes (a seed the study used is refused),
  optionally drives candidates on the device path, copies the weights byte for
  byte and writes the `selection` block (`candidates`, `seed_spread`,
  `fresh_eval`, `device_eval`, `weights_sha256`).
- Sidecars `racetraq/weights/<name>.meta.json` record circuit shape,
  observation, action count, the resolved training table and provenance; every
  loader builds the circuit a file needs from them.
- `tools/make_stages.py --track T` replays the bundled driver's run from its
  sidecar and writes evolution-stage weights only if the replay's best snapshot
  equals the driver parameter for parameter.
- Hardware: `tools/hw_reliability.py`, `python -m racetraq.hardware lap|sprint
  --fake`, `python -m racetraq.agents.quantum.noise calibrate|validate`;
  cross-track matrix: `python -m racetraq.records --episodes 36 --seed 47000`.
- Notebooks 03–07 regenerate the light-cone derivations, the multi-seed tables
  from `data/studies/`, the hardware measurements and the surrogates; the
  `tests/` suite pins the light cone, the surrogates, the simulator parity and
  the study and bundling tools.

## 8 Open questions and next steps

1. **Selection on unseen tracks** — done 2026-10-09: `bundle_driver.py
   --unseen 10` adds ten generated tracks to the ranking; re-selecting
   `quantum_universal` with it picks seed 3 (the former hand choice).
2. **A physical QPU.** Every multi-seed hardware number is a simulated snapshot;
   one real lap exists (ibm_marrakesh, 2026-10-03, Open Plan batch mode, 27 CZ,
   ~280 QPU-seconds, 27 minutes wall clock; `data/qpu/`). Repeating it across
   days and devices, and on a Nighthawk, is the open item (Open Plan: job and
   batch mode only, 10 QPU-minutes per 28 days, heavy-hex Herons).
3. **Which robustness lever matters** — measured 2026-10-09 (SCIENCE.md,
   "Follow-ups to the audit"): on oval and chicane the two levers are
   complementary (device laps of 96: 73/48 with neither, 81/71 gap only,
   79/54 noise only, 95/95 both); on gp they hurt learning (best-snapshot
   lapped 0.99 without, 0.05 with both), so gp still trains without them
   and remains fragile on the device (36 of 96). Combo and the universal
   driver were not tested.
4. **gp warm start and the end of training.** No variant makes the
   end-of-training parameters lap on gp, and warm live-training on gp mostly
   fails (`default.toml` `[training_warm_gp]`: 3 of 11 seeds end lapping).
5. **10 qubits on gp.** The October recipe fails at both depths (3 seeds
   each). Under the July recipe, 8 seeds per depth: the engineered-feature
   circuit learns gp at 4 and 6 layers (best snapshots lap 0.44 and 0.42 of
   their test episodes; 6 layers 22.2 s), but no run keeps what it learned.
   A dependable 10-qubit gp driver is still open; the plain-ray profile, a
   pace phase and scaled action sets are being measured.
6. **Feature engineering and pace**: at 4 qubits on gp, two rays plus the
   corner feature do not learn (0.11), and a pace phase either loses
   reliability (exploration to 0.02) or keeps it without faster laps (held
   at 0.30); a pace selection rule (`snapshot_rank = "pace"`) exists but
   changed nothing there. Scaled action sets are being measured at 10
   qubits.
7. **Kernelized fitted Q-iteration** on the product kernel [15] was not run;
   the surrogate dequantizes inference only.
8. **Stale records** — done 2026-10-08: `data/records.md` regenerated with
   the 6-block 10-qubit drivers and the bundled universal driver (seed 3; the
   2026-10-03 matrix still showed seed 0, 13.7 s on the oval).
