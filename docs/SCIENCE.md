# The science behind traQmania

What the quantum agent actually is, how it is trained, what runs on real
hardware, and — importantly — what this demo does and does not show. The
[notebooks](../notebooks/) build all of this up from scratch with runnable
code; this page is the condensed reference.

> **October 2026 audit.** A soundness audit on 2026-10-01 found that the 8-
> and 10-qubit circuits were partially blind ("Light cones" below), that the
> DQN's greedy performance is unstable after the first lap — most of all on
> gp ("Training") — that the headline numbers rested on one to three seeds
> and on snapshots chosen from four evaluation episodes, and that the
> hardware path targeted retired devices ("Hardware"). The code is fixed or
> instrumented for all of it. On 2026-10-02/03 every bundled driver was
> retrained and re-selected from multi-seed studies ("Measured results"
> reports those studies; their per-seed summaries are in `data/studies/`),
> except `quantum_gp_q10`, which stays a July single run. The headline
> changed with them: under 8–10 seeds the matched classical baseline is
> ahead of the circuit on most metrics ("Honest claims"). Still open: a
> universal driver selected on unseen tracks as well (the bundled one was
> chosen by hand — "One driver, every track"), and a full lap on a physical
> QPU ("Hardware": a first attempt on `ibm_marrakesh` is recorded there).

## The circuit

A data re-uploading variational quantum circuit (VQC) acting as the Q-function
of a DQN agent. Canonical definition: `traqmania/agents/quantum/circuit.py`
(single source of truth for the numpy fast path, the Qiskit/`EstimatorQNN`
path, and the hardware path).

- **n qubits** (default **4**), **L re-uploading blocks** on |0…0⟩ (4 by
  default, 5 in the `q8` profile — "Light cones" below says why).
  Block *l*:
  1. encoding: `RY(λ[l,i] · s[i])` on each qubit *i* — the n observation
     features `s` (n − 1 lidar rays evenly spaced over [−60°, +60°] plus
     speed, each normalized to [0, 1]; 3 rays + speed at the default size)
     are re-uploaded in **every** block;
  2. variational: `RY(θ[l,i,0])` then `RZ(θ[l,i,1])` on each qubit;
  3. entanglement: a CZ ring `CZ(0,1) … CZ(n−1,0)`.
- **Readout:** one Pauli-Z expectation per action, `E_a = ⟨Z_a⟩` on the
  first four qubits — the **4 actions stay fixed at any n**; extra qubits
  only widen the feature register — mapped through a trainable classical
  head `Q_a = w[a]·E_a + b[a]`. The output scaling is essential: useful
  Q-values for this task are of order 100 while ⟨Z⟩ ∈ [−1, 1] (the bundled
  drivers end up with |w| between 25 and 95).
- **Trainable parameters, P = 3·L·n + 8** (**56** at n = 4, **80** at
  n = 6, 128 at n = 8 with 5 blocks), one flat vector `[λ, θ, w, b]`: input
  scalings λ (L×n, init π),
  variational angles θ (L×n×2, init U(−0.1, 0.1)), output weights w
  (4, init 1) and biases b (4, init 0).

### Light cones: what each readout can see

A readout ⟨Z_a⟩ does not depend on the whole circuit. Push Z_a backwards
through the gates (Heisenberg picture) and it only meets the gates inside its
backward *light cone*; everything outside commutes with it and cancels. With
a nearest-neighbour CZ ring the cone widens by one qubit per block in each
direction, which gives three exact, purely structural facts
(`traqmania/agents/quantum/lightcone.py`; `tests/test_lightcone.py` checks
them against fastsim values and adjoint gradients):

- **The last CZ ring never matters.** It is diagonal and so is Z_a: they
  commute. For the same reason the RZ angles of the final block are dead —
  the original teaching point of this section.
- **Block *l* (counted from 0) only matters near the readout qubits.** Its
  encoding and RY gates are live on qubits within ring distance L − 1 − l of
  a readout qubit, its RZ gates within L − 2 − l.
- **⟨Z_a⟩ sees exactly the features within ring distance L − 1 of qubit a.**
  Every action sees every feature only when L ≥ ⌊n/2⌋ + 1.

At L = 4 — the depth of every profile except `q8` — with the four readouts
on qubits 0–3:

| | 4 qubits | 6 qubits | 8 qubits | 10 qubits |
|---|---|---|---|---|
| Circuit parameters (3·L·n) | 48 | 72 | 96 | 120 |
| Dead (zero gradient for every input) | 4 | 12 | 26 | 46 |
| CZ gates that can reach a readout | 12 of 16 | 17 of 24 | 20 of 32 | 21 of 40 |
| Features hidden from each action | 0 | 0 | 1 | 3 |
| Blocks needed for full visibility | 3 | 4 | 5 | 6 |

(`python -m traqmania.agents.quantum.lightcone --qubits 10 --layers 4`
prints the full map.) "Dead" is structural: the gate commutes with every
back-propagated readout, so its angle has exactly zero gradient for every
input and can never train. At 4 qubits the dead parameters are the four
final-block RZ angles. Beyond that the count grows faster than n, because
whole encoding and RY gates on the far side of the ring drop out too. An
earlier version of this page said "n dead parameters at any qubit count";
that is true only at n = 4.

The `q8` profile therefore runs **5 blocks** since October 2026: 120 circuit
parameters, 26 of them dead, 28 of 40 CZ gates live, no hidden feature.

**Blind spots at 4 blocks.** Feature j sits on qubit j and the actions read
qubits 0–3, so which feature is hidden from which action is an accident of
the feature order:

- 8 qubits, 7 rays + speed (the July `q8` drivers; the bundled
  `quantum_oval_q8` and `quantum_chicane_q8` are 5-block drivers and see
  everything): Right cannot see ray +20°, Straight ray +40°, Left ray +60° —
  and **Brake cannot see speed**.
- `q10`, 9 rays + speed (`quantum_oval_q10`, `quantum_chicane_q10` — still
  the July files): Right misses rays 0°/+15°/+30°, Straight +15°/+30°/+45°,
  Left +30°/+45°/+60°, Brake rays +45°/+60° and speed.
- `quantum_gp_q10` (also still the July file), 5 rays + speed + four
  engineered features: Right misses ray +60°, speed and curvature ahead;
  Straight speed, curvature ahead and lateral offset; Left curvature ahead,
  lateral offset and heading error; **Brake lateral offset, heading error
  and corner speed** — the brake action decides without the one feature
  built to say "too fast for this corner".
- The 8-qubit engineered-feature observation (3 rays + speed + the four
  features in the order of the `q8` profile's alternative) hides exactly one
  engineered feature from each action (curvature ahead, lateral offset,
  heading error, corner speed for Right, Straight, Left, Brake).

The default 4-qubit circuit and the `q6` profile (plain or with features)
are fully visible.

**What the re-measurement says.** Every 8- and 10-qubit result of July 2026
was measured with these partially blind circuits, on one to three seeds.
Those results are withdrawn, not reversed. What replaced them so far (tables
under "Scaling and the light cone"):

- *8 qubits, 5 blocks against 4* (oval and chicane, 6 seeds per depth, 800
  episodes): the evidence for full visibility is thin but one-directional.
  On the oval the end-of-training parameters lap more often — 0.88
  [0.62, 0.99] of the evaluation episodes against 0.36 [0.05, 0.60]
  (probability of improvement 0.92 [0.72, 1.00]), the one supported
  difference of the ten comparisons made — and everything else points the
  same way without support: on both tracks the best snapshot of 6 of 6
  seeds laps in at least half of its episodes, against 5 of 6 at 4 blocks.
  Lap times do not differ. Hence the profile's new depth.
- *10 qubits on gp* (engineered-feature observation): under the July
  recipe, 6 blocks lapped in 16 and 17 of 36 episodes (seeds 0 and 42)
  where 4 blocks lapped in 6 and 24 — steadier, not better, and two single
  runs per depth. A hand-made reordering that puts speed, corner speed,
  curvature and the centre ray on the four readout qubits did worse (2 and
  0 of 36): the layout matters, and that guess at a good one was wrong.
  Under the recipe that now ships for 4-qubit gp, neither depth learns the
  track (3 seeds each; "Training stability" below).
- *10 qubits on the oval, 6 blocks against 4* (6 seeds per depth, 800
  episodes): the same direction as at 8 qubits, but not supported at this
  sample. With 6 blocks the end-of-training parameters lap in 0.68
  [0.19, 1.00] of the evaluation episodes against 0.36 [0.06, 0.85]
  (probability of improvement 0.71 [0.38, 1.00]), and the best snapshots
  lap in 13.3 s [12.8, 14.5] against 14.3 s [13.7, 14.7] (0.75
  [0.42, 1.00]). At either depth the best snapshot of 6 of 6 seeds laps in
  at least half of its episodes.
- *10 qubits on chicane, 6 blocks against 4* (6 seeds per depth, 800
  episodes, `data/studies/chicane_q10`): indistinguishable. End-of-training
  parameters lap in 0.55 [0.12, 0.93] of the evaluation episodes at 6
  blocks against 0.55 [0.24, 0.85] at 4, stability 0.45 [0.32, 0.66]
  against 0.46 [0.31, 0.70] (probability of improvement 0.50 [0.17, 0.83]);
  the best snapshots lap in 13.1 s [12.7, 13.4] against 13.6 s [13.2, 13.9]
  (0.51 [0.17, 0.88]). The matched MLP on the same nine rays: stability
  0.93 [0.80, 0.98], 12.7 s. Decision: the `q10` profile runs 6 blocks —
  full visibility costs nothing measurable and helped on the oval — and the
  bundled 10-qubit oval and chicane drivers are 6-block files re-selected
  from these studies (72 of 72 fresh episodes each, 13.3 s and 12.6 s).
  `quantum_gp_q10` stays the July 4-block file: under the recipe that now
  ships for 4-qubit gp neither depth learns the track (above).

The stack now shows the light cone instead of hiding it: `circuit_spec()`
carries `visibility`, `dead_params`, `dead_gates`,
`min_layers_full_visibility` and a `live` flag on every gate; the web UI
dims the dead gates in the circuit diagram and draws a "Who sees what"
matrix of inputs against actions; `train_headless` prints the blind spots
before it trains a too-shallow circuit; and the hardware path executes only
the live gates (`lightcone.pruned_circuit` — identical expectation values,
12 instead of 16 CZ at 4 qubits). The lesson is the old one, sharpened: what
a parameter — or an input — can do is decided by the ansatz and the
observable *together*. Notebook 03 derives the dead sets at 4 and 6 qubits
and the 8- and 10-qubit blind spots from the light cone and confirms them by
taking the derivative rather than trusting the commutator argument; notebook
07 — light cones and classical surrogates — works through the general case,
down to the pruned circuit the hardware runs.

## Algorithm lineage

The method:

- **Chen, Yang, Qi, Chen, Ma & Goan, “Variational Quantum Circuits for Deep
  Reinforcement Learning”, IEEE Access 8 (2020)** [1] — the paper this demo
  follows (co-authored by Pin-Yu Chen of IBM Research): a VQC as the
  Q-function of experience-replay Q-learning.
- **Skolik, Jerbi & Dunjko, “Quantum agents in the Gym”, Quantum 6, 720
  (2022)** [2] — data re-uploading, trainable input/output scaling, and the
  observable-design considerations adopted here.
- **Schuld, Sweke & Meyer, Phys. Rev. A 103, 032430 (2021)** [3] — why
  re-uploading works at all: such a model is a partial Fourier series in the
  data whose accessible frequencies are fixed by the encoding gates. Here
  the trainable input scalings λ set those frequencies. The same structure
  is what makes classical surrogates possible (see "Honest claims").
- **Meyer et al., “A Survey on Quantum Reinforcement Learning”
  (arXiv:2211.03464)** [4] — situates this family of value-based QRL methods
  in the broader landscape.
- **Sahin et al., arXiv:2505.17756** [5] — the qiskit-machine-learning
  library paper (STFC Hartree Centre and IBM authors); its `EstimatorQNN` is
  the reference implementation our numpy simulator is pinned against.

How to evaluate it — the literature the October 2026 audit measured this
project against:

- **Meyer et al., “Benchmarking Quantum Reinforcement Learning”, ICML 2025**
  [6] — a benchmarking methodology built on a statistical estimator for
  sample complexity and a definition of statistical outperformance; applied
  to QRL it "casts doubt on some previous claims regarding its superiority".
- **Agarwal et al., “Deep Reinforcement Learning at the Edge of the
  Statistical Precipice”, NeurIPS 2021** [7] — interval estimates and the
  interquartile mean over runs instead of point estimates from a handful of
  seeds.
- **Bowles, Ahmed & Schuld, arXiv:2403.07059** [8] — a large benchmark of
  quantum classifiers in which out-of-the-box classical models win overall
  and removing entanglement often does not hurt: "it learns" is not evidence
  that "quantum" is the active ingredient.
- **Pardo et al., “Time Limits in Reinforcement Learning”, ICML 2018** [9] —
  an episode cut off by a time limit is not a terminal state; the value
  target should keep bootstrapping (`bootstrap_truncation` below).
- **Skolik et al., “Robustness of quantum reinforcement learning under
  hardware errors”, EPJ Quantum Technology 10, 8 (2023)** [10] — shot noise,
  coherent and incoherent errors during training and evaluation of
  variational RL agents.
- **Spall (1992)** [11] for SPSA, the optimizer of the hardware sprint;
  **Gacon, Zoufal, Carleo & Woerner, Quantum 5, 567 (2021)** [12] (IBM
  Research – Zurich, ETH Zurich and EPFL) apply the same
  simultaneous-perturbation idea to the quantum Fisher information at
  constant cost per step (QN-SPSA) — not used here.

## Training

**Double DQN in pure numpy** (`agents/training/dqn.py`): experience replay
(ring buffer, 10 000 transitions), epsilon-greedy exploration with linear
decay, Adam, and a periodically-synced target network — which is just a
second flat parameter vector, so the identical loop trains the quantum
circuit and the classical baseline. The environment is vectorized (8 parallel
cars by default); reward is signed centerline progress plus checkpoint/lap
bonuses minus an off-track penalty. The classical baseline is a 76-parameter
MLP (4-8-4, tanh) — chosen for comparable parameter count, not tuned to win.

**Which parameters ship: the snapshot eval.** The weights that are saved are
not the last ones but the best *snapshot*. Every `eval_every` (50) training
episodes the current parameters drive one round of `eval_episodes` (12)
greedy episodes on a fresh evaluation env — 12 cars in parallel, each from
its own spawn jitter, so 12 distinct episodes. Snapshots are ranked by
(episodes that produced a lap, mean lap time over all laps, mean return):
reliability first, average pace second, and one lucky lap never decides. The
final parameters compete too. `history["eval_log"]` keeps every eval,
`final_eval` the last one, and `train_headless --save-final` writes the
end-of-training parameters next to the best snapshot so the two can be
compared. Before the audit the "12 episodes" were a 4-car env rebuilt from
the same seed three times — 4 distinct episodes counted three times — and
the rule before that was a single 4-episode eval. The bundled 10-qubit
drivers still date from then.

**Trainer options** (`[training]` keys). `bootstrap_truncation` is on since
the October 2026 studies; every other key defaults to the pre-audit
behaviour:

| Key | Default | What it is for |
|---|---|---|
| `bootstrap_truncation` | `true` | Treat the 60 s time limit as a truncation, not a terminal state: the TD target keeps bootstrapping from the last observation (Pardo et al. [9]). Off-track stays terminal. `false` restores the pre-audit behaviour. |
| `loss`, `huber_delta` | `"mse"`, 10 | `"huber"` clips the TD error in the gradient at ±`huber_delta`; Q-values here are of order 100, so a few large errors otherwise dominate a batch. |
| `lr_groups` | — | Separate learning rates per parameter group (quantum: `lam`, `theta`, `head`; MLP: `body`, `head`). Circuit angles are of order 1, the output head grows to 25–95 — one rate may fit neither. |
| `lr_end` | = `lr` | Linear learning-rate anneal over the run. |
| `target_update`, `tau` | `"hard"`, 0.005 | `"soft"` replaces the copy every `target_sync_every` updates by Polyak averaging on every update. |
| `grad_clip` | 0 (off) | Clip the gradient's global L2 norm. |
| `reward_scale` | 1 | Scales rewards as stored in replay (reported returns stay raw), e.g. to shrink Q-values toward the ⟨Z⟩ range. |
| `eval_every`, `eval_episodes` | 50, 12 | The snapshot eval above. |
| `act_noise` | — (off) | `{ attenuation, shots, bias }` (quantum agent): rollouts and snapshot evals pick actions from *noisy* readouts, ⟨Z⟩ → attenuation·⟨Z⟩ + bias + the shot noise of `shots` shots (`agents/quantum/noise.py`). TD targets and gradients stay exact, and the snapshot kept is the one that drives best under the noise. |
| `action_gap` | 0 (off) | Advantage learning (Bellemare et al. [23]): the TD target loses `action_gap` · (max_a Q(s, a) − Q(s, a_taken)), which widens the gap between the best action and the rest by 1 / (1 − `action_gap`) and leaves the best action unchanged. |

**The recipe is per track and per agent.** `config.resolve_training_cfg`
merges three layers: `[training]`, then `[training_presets.<track>]` (gp:
3000 episodes, epsilon decayed over 2000, gamma 0.99; combo: 2500, 1500,
0.99), then `[training_presets_<agent>.<track>]`. Only the quantum agent has
the third layer today: `action_gap = 0.8` and `act_noise = { attenuation =
0.95, shots = 1024 }` on oval and chicane (the noise-robust recipe —
"Hardware"), and `epsilon_end = 0.30` on gp and combo ("Training stability"
below). The MLP keeps `epsilon_end = 0.05` everywhere. The oval and chicane
studies and the bundled drivers of those tracks ran 800 episodes (`--episodes
800`); the `[training]` default of 400 is the live-demo length.

From the command line: `train_headless --set training.loss=huber --set
circuit.n_layers=6 …` overrides any config value (an unknown `training.` key
is an error, not a silent baseline run), `--preset none` trains on plain
`[training]` without either preset layer, and the resolved recipe is
recorded in the weights' `.meta.json`.

**Many seeds, proper statistics: `tools/study.py`.** One training run is an
anecdote. The study tool trains a (variant × seed) grid, one subprocess per
cell, resumable, and then evaluates every cell's best snapshot *and* its
final parameters over 36 distinct greedy episodes. `study.py report`
aggregates over seeds, following Agarwal et al. [7] and Meyer et al. [6]:

- interquartile mean and median with 95 % percentile-bootstrap confidence
  intervals (2000 resamples over seeds) for the lapped fraction of the best
  snapshot, the lapped fraction of the final parameters, the best snapshot's
  mean lap and the first clean lap;
- **stability** — the mean lapped fraction over all in-training evals after
  the first one that lapped (1 = once it laps it keeps lapping);
- **sample complexity** — training episodes until an eval laps in ≥ 50 % /
  ≥ 90 % of its episodes, reported as "episodes until half of *all* seeds
  are there" so seeds that never learn count against a variant;
- **probability of improvement** over a baseline variant — the chance that a
  random seed of one beats a random seed of the other, with a stratified
  bootstrap interval; an interval entirely above 0.5 is a supported
  improvement.

With fewer than about ten seeds the intervals understate the uncertainty,
and a single seed gets none; the report says so itself. Every number on this
page written as `x [a, b]` is such an interquartile mean over seeds with its
95 % interval. `tools/export_study.py STUDY_DIR --name NAME` writes the
committable summary of a study to `data/studies/NAME/`: `report.md`,
`report.json`, and `cells.json` with every seed's eval log and both
36-episode evals. The studies behind this page are there (487 training
runs); their weights and full logs are not in the repository.

**From a study to a bundled driver: `tools/bundle_driver.py`.** A bundled
driver is one seed's best snapshot, so choosing it is a second selection
step and needs its own control. The tool ranks a variant's seeds by the
study's 36-episode eval, re-evaluates the top three or four on 72 *fresh*
greedy episodes (another env seed), and chooses by that result — episodes
lapped first, mean lap second; for the three hardware-demo drivers, laps on
the simulated device first. The sidecar records the fresh numbers, never
the ones a seed was shortlisted by, in a `selection` block: study, variant,
chosen seed, every candidate's numbers, the recipe's spread over all seeds,
`fresh_eval` and, where it ran, `device_eval`. The fresh eval still picks
among the candidates, so a small selection effect remains in it; the
36-episode re-check quoted under "Measured results" used a third env seed
that nothing was selected on.

**Two executions of one circuit, verified equal.** The training path is
`fastsim`, a hand-written numpy statevector simulator with **adjoint**
(backprop-style) gradients: one forward plus one backward sweep for the whole
gradient. The same circuit runs through Qiskit / qiskit-machine-learning's
`EstimatorQNN` on Aer; `tests/test_fastsim_vs_qiskit.py` and
`tests/test_gradients.py` pin forward values and gradients against each other
and against finite differences. Measured cost of one batch-32 double-DQN
update: **~3.4 ms** (fastsim + adjoint) vs **~20.5 s** (`EstimatorQNN` +
parameter-shift, 2 evaluations per parameter) — a ~6 000× gap, which is *the*
practical reason training runs on the simulator. (That pair was timed before
the October 2026 stack upgrade and has not been re-timed on a quiet machine.
Notebook 03's batch-8 cell, run four times on 2026-10-01/02 on a heavily
loaded machine with qiskit-machine-learning 0.9.1, gave 3–9 ms against
13–27 s — ratios of 2 600 to 4 300, the same order of magnitude.)

### Training stability

**Greedy performance is not stable — stated plainly.** A policy that laps
at one snapshot eval frequently laps in none of the episodes at the next.
Under the recipes that now ship (4 qubits; 10 seeds each), stability is
0.75 [0.52, 0.92] on the oval, 0.63 [0.49, 0.73] on chicane, 0.17
[0.12, 0.22] on gp and 0.15 [0.03, 0.27] on combo, and the parameters at
the end of training lap in 0.80 [0.47, 0.99], 0.63 [0.39, 0.88], 0.02
[0.00, 0.08] and 0.03 [0.00, 0.22] of 36 greedy episodes. The matched MLP
holds on to the easy tracks far better (stability 0.94 and 0.97, 8 seeds)
and to the hard ones no better (0.14 on gp, 0.08 on combo; end-of-training
parameters 0.01 and 0.02; 10 seeds). On gp and combo the shipped driver of
either agent is therefore a selected snapshot: a real policy, but "the
agent learned gp" overstates what the optimizer holds on to.

**The stabilisation study** asked what would fix that for the circuit: gp,
4 qubits, 3000 episodes, 29 variants, 6 seeds each and 10 for the baseline
and the variants that mattered (`data/studies/gp_4q`; all 29 are in its
`report.md`). The baseline is the July recipe: epsilon from 1.0 to 0.05
over the first 2000 episodes, gamma 0.99, the time limit a terminal state.

| Change from the July recipe | Seeds | Best-snapshot lapped | Final-params lapped | Stability | P(stability beats baseline) |
|---|---|---|---|---|---|
| none (baseline) | 10 | 0.89 [0.74, 0.97] | 0.00 [0.00, 0.03] | 0.08 [0.04, 0.15] | — |
| Huber loss | 6 | 0.78 [0.39, 1.00] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.10] | 0.25 [0.02, 0.55] |
| soft target update | 6 | 0.92 [0.74, 1.00] | 0.00 [0.00, 0.00] | 0.06 [0.04, 0.18] | 0.47 [0.17, 0.77] |
| learning rate annealed to 0.001 | 6 | 0.67 [0.20, 0.93] | 0.00 [0.00, 0.00] | 0.02 [0.00, 0.05] | 0.13 [0.00, 0.37] |
| per-group rates (angles 0.003, head 0.03) | 6 | 0.02 [0.00, 0.15] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] | 0.03 [0.00, 0.13] |
| reward scale 0.05 | 6 | 0.90 [0.43, 0.98] | 0.00 [0.00, 0.02] | 0.12 [0.05, 0.20] | 0.60 [0.30, 0.90] |
| no lap / checkpoint bonuses | 6 | 0.46 [0.26, 0.83] | 0.00 [0.00, 0.07] | 0.04 [0.01, 0.09] | 0.35 [0.07, 0.68] |
| replay 200 000 | 6 | 0.94 [0.62, 1.00] | 0.00 [0.00, 0.39] | 0.09 [0.04, 0.29] | 0.60 [0.30, 0.88] |
| batch 128 | 6 | 0.91 [0.81, 0.96] | 0.01 [0.00, 0.38] | 0.15 [0.09, 0.30] | 0.78 [0.52, 0.98] |
| target sync every 1000 updates | 6 | 0.24 [0.05, 0.44] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.06] | 0.13 [0.00, 0.40] |
| bootstrap through the time limit | 6 | 0.94 [0.78, 0.99] | 0.06 [0.00, 0.44] | 0.13 [0.04, 0.47] | 0.57 [0.25, 0.87] |
| exploration floor 0.15 | 6 | 0.70 [0.62, 0.93] | 0.11 [0.00, 0.43] | 0.07 [0.04, 0.15] | 0.50 [0.22, 0.82] |
| exploration floor 0.30 | 10 | 0.94 [0.81, 1.00] | 0.16 [0.03, 0.48] | 0.18 [0.13, 0.21] | 0.77 [0.51, 1.00] |
| floor 0.30 + truncation (**ships**) | 10 | 0.96 [0.82, 1.00] | 0.02 [0.00, 0.08] | 0.17 [0.12, 0.22] | 0.75 [0.49, 0.95] |
| floor 0.30 + truncation + batch 128 | 10 | 0.75 [0.29, 0.97] | 0.00 [0.00, 0.14] | 0.12 [0.05, 0.20] | 0.57 [0.32, 0.82] |
| MLP, July recipe | 10 | 0.87 [0.54, 0.97] | 0.17 [0.00, 0.48] | 0.17 [0.05, 0.34] | 0.63 [0.35, 0.87] |
| MLP + truncation (**ships for the MLP**) | 10 | 0.74 [0.37, 0.97] | 0.01 [0.00, 0.38] | 0.14 [0.05, 0.21] | 0.61 [0.33, 0.85] |
| MLP, floor 0.30 + truncation | 10 | 0.21 [0.02, 0.61] | 0.01 [0.00, 0.03] | 0.03 [0.00, 0.11] | 0.32 [0.11, 0.58] |
| MLP, hidden 32 (292 parameters) | 6 | 1.00 [0.98, 1.00] | 0.31 [0.04, 0.86] | 0.24 [0.19, 0.34] | 0.88 [0.68, 1.00] |

- *What did not help.* None of the standard stabilisers: Huber loss, a soft
  target, learning-rate decay, per-group learning rates, reward scaling, a
  20× larger replay buffer, slower target sync, removing the lap and
  checkpoint bonuses, or Huber, soft target, learning-rate decay and
  truncation together (2 of 6 seeds lap in half their episodes). Several
  made it worse; a learning rate of 0.003 never lapped at all.
  Bootstrapping through the time limit — a real bug, fixed (Pardo et al.
  [9]) — is neutral on gp. A batch of 128 is the one optimizer setting
  whose stability gain just clears the bar at 6 seeds (0.78 [0.52, 0.98];
  under other bootstrap seeds the interval's lower end sits at 0.50–0.53),
  and on top of the floor below it added nothing at 10 seeds (0.57
  [0.32, 0.82]; 3 of 10 seeds no longer reach a snapshot that laps in half
  its episodes), so it was not adopted. In no quantum variant do the
  end-of-training parameters lap reliably.
- *The diagnostic.* Pool the in-training greedy evals (12 episodes every 50
  training episodes) over the 10 seeds and bin them by training episode.
  Epsilon falls linearly over the first 2000 episodes and then sits at its
  floor:

  | Lapped fraction of greedy evals, episodes | 1–500 | 501–1000 | 1001–1500 | 1501–2000 | 2001–2500 | 2501–3000 |
  |---|---|---|---|---|---|---|
  | circuit, floor 0.05 (July recipe) | 0.00 | 0.12 | 0.15 | 0.12 | 0.00 | 0.07 |
  | circuit, floor 0.30 + truncation | 0.00 | 0.05 | 0.15 | 0.17 | 0.25 | 0.15 |
  | MLP, floor 0.05 (July recipe) | 0.00 | 0.02 | 0.00 | 0.04 | 0.26 | 0.28 |
  | MLP, floor 0.05 + truncation | 0.00 | 0.02 | 0.00 | 0.03 | 0.27 | 0.16 |
  | MLP, floor 0.30 + truncation | 0.01 | 0.03 | 0.00 | 0.00 | 0.02 | 0.06 |

  (About 1200 episodes per cell.) The circuit's greedy driving is at its
  best while epsilon is still falling — 0.12 to 0.15 from episode 501 to
  2000 — and collapses once epsilon sits at 0.05: 3 lapped episodes of 1224
  in episodes 2001–2500. The MLP does the opposite: next to nothing until
  exploration is low, then it climbs. Most of the 6-seed variants that
  keep the 0.05 floor show the same drop from episodes 1501–2000 to
  2001–2500 (soft target 0.15 → 0.00, gamma 0.98 0.23 → 0.03, batch 128
  0.27 → 0.09, replay 200 000 0.23 → 0.08), but not all: reward scaling
  holds its level (0.12 → 0.14) and so does truncation alone (0.13 →
  0.13) — there through two of its six seeds, which lap in 95 of 240
  episodes while the other four lap in none of 480.
- *The floor.* Holding epsilon at 0.30 removes the collapse. The gain is
  modest: stability 0.18 against 0.08 (probability of improvement 0.77
  [0.51, 1.00]), and all 10 seeds reach a snapshot that laps in at least
  half its episodes (9 of 10 before). It costs pace on the easy tracks
  (chicane: 16.1 s [14.8, 18.8] against 13.9 s [13.2, 14.4], 8 seeds), so
  it is a gp and combo preset only. And it does not make the end of
  training usable. The shipped recipe adds truncation, which leaves
  stability where it was (0.17; 0.75 [0.49, 0.95], at the edge of what 10
  seeds support) and the end-of-training parameters worse: they lap in
  0.02 [0.00, 0.08] of the episodes, against 0.16 [0.03, 0.48] with the
  floor alone.
- *Not a universal recipe.* The MLP does worse under the same floor on
  every measure, but not by a supported margin: 3 of 10 seeds reach a
  half-lapping snapshot, against 7 of 10 at 0.05, and the probability that
  a 0.30-floor seed beats a 0.05-floor seed is 0.28 [0.07, 0.51] for the
  best snapshot, 0.26 [0.04, 0.52] for the lapped fraction of the snapshot
  evals in episodes 2050–3000 and 0.27 [0.06, 0.52] for stability (10
  seeds each, both with truncation; notebook 04 prints them). Nothing
  suggests the floor helps the MLP, which is why the preset is per agent.
  And the 10-qubit circuit on the
  engineered-feature observation fails under it too: with floor 0.30 and
  truncation the best snapshots of three seeds lapped in 0, 4 and 0 of 36
  episodes at 4 blocks and in 2, 7 and 5 at 6 blocks
  (`data/studies/gp_q10feat`), where single July-recipe runs of the same
  circuit — two seeds per depth — lapped in 6 to 24 of 36
  (`data/studies/gp_q10feat_july_recipe`).
  The bundled `quantum_gp_q10` therefore stays the July driver. What the
  0.30 floor fixes is specific to the 4-qubit circuit on its 4-feature
  observation.
- *Capacity is not the lever either.* Deeper 4-qubit circuits lap gp faster
  but less reliably (6 blocks: 27.4 s, 3 of 6 seeds; 8 blocks: 26.3 s, 5 of
  6; baseline 34.7 s, 9 of 10), and the plain 6-qubit profile with 5 rays
  does not learn gp at all under the July recipe (0 of 6 seeds). The one
  clearly steadier learner in the study is classical: the MLP with 32
  hidden units.

So the shipped quantum drivers for gp and combo, and the universal driver,
are selected snapshots of an unstable optimisation. What changed is that
the selection is now reproducible and controlled: many seeds, a fixed
ranking rule, a fresh re-evaluation, and the whole decision recorded next
to the weights.

## Hardware

(`traqmania/hardware.py`, via `qiskit-ibm-runtime`. Everything below also
runs without an account on a *fake backend*: the calibration snapshot of a
real IBM device — coupling map, gate and readout errors, T1/T2 — simulated
locally with Aer.)

**The devices.** IBM's fleet is CZ-based today: Heron processors (156 qubits
on a heavy-hex lattice) and Nighthawk (120 qubits on a square lattice with
218 couplers; `ibm_miami` since January 2026, a second revision since
September 2026). The Eagle generation left the fleet in April 2026, and the
5–7-qubit CX-based Falcon devices this page used to target were retired in
2023 [18]. The lattice matters for this circuit. A CZ ring on an even
number of qubits is the perimeter of a 2 × n/2 rectangle, so it embeds in
the square lattice with **zero SWAPs**: 16 CZ at 4 qubits, 12 after
light-cone pruning. Heavy-hex has no cycle that short, so the ring must be
routed: 37 CZ unpruned and 27 pruned on `fake_fez` (best of 8 layout seeds),
50 / 38 CX on the retired `fake_manila`.

- **Simulated device.** The default fake is `fake_miami`, a Nighthawk
  calibration snapshot (`[hardware] fake_name`, CLI `--fake-name`; any
  spelling, an unknown name is an error that lists what exists). A 120–156
  qubit noise model is far too large to simulate per decision, so the
  circuit is routed onto the full device once and only the physical qubits
  it touches are simulated — a *device patch* with those qubits' real
  errors and couplings (4 qubits at the default size, ~0.1 s per decision).
  `fake_fez`, `fake_marrakesh`, `fake_kingston` are heavy-hex Herons;
  `fake_manila` and `fake_lagos` are the retired Falcons, kept as an option
  and simulated whole.
- **Estimator and mitigation.** Expectation values come from the client-side
  `qiskit_ibm_runtime.executor_estimator.Estimator`, with a fallback to
  `EstimatorV2` on runtimes that do not have it yet (qiskit-ibm-runtime
  0.50.0 of 2026-09-24 added the former and deprecated the latter [22]).
  Error mitigation is explicit: `resilience_level` 0 — raw device noise, the
  default here and the honest number — 1 TREX readout mitigation, 2 TREX
  plus ZNE with gate twirling.
- **Execution mode.** Session → Batch → plain jobs, whichever the account
  grants first; the mode used and the reason for any fallback are reported
  (CLI, `hardware_status`). This is what a free IBM Quantum **Open Plan**
  account needs: per IBM's documentation it gets 10 minutes of QPU time per
  28-day rolling window (plus an opt-in promotion of 180 minutes over 12
  months as of March 2026), instances in the us-east region only, and no
  Sessions — "Workloads on the Open Plan can run only in job mode or batch
  mode" [17]. Without a Session every decision queues like any other job.
  IBM's changelog lists Herons (`ibm_fez`, `ibm_marrakesh`, `ibm_kingston`)
  for Open Plan users and the Nighthawks for paid plans [18], so on a free
  account expect the ring to be routed with SWAPs.
- **Inference laps** (`run_hardware_lap`): drive one car greedily with every
  10 Hz-equivalent decision evaluated on the backend. `HardwareQFunction`
  implements the same `QFunction` contract and parameter layout as the
  simulator: the light-cone-pruned circuit is transpiled to ISA form once
  (`--no-prune` runs the full one), then each decision is one Estimator job
  — all observation rows × all four Z_a observables in a single PUB, 1024
  shots by default (`[hardware] shots`; `decision_shots` sets a separate
  default for laps and applies when the caller names no shot count — the
  web panel always sends its own).
  Gradients are deliberately `NotImplementedError` — parameter-shift would
  cost 2 × 48 circuit evaluations per batch at the default 4 qubits
  (2 × 3·L·n in general; 2 × 44 if the four dead angles are skipped).
- **SPSA sprint** (`spsa_sprint`): the only sensible way to *update*
  parameters on hardware today. The replay batch and double-DQN TD targets
  are computed **once** in the exact simulator (with the discount, reward
  scale and action-gap term the weights' sidecar records); the hardware then
  only evaluates the MSE TD loss, one Estimator job per evaluation — SPSA
  [11] needs **2 per iteration regardless of parameter count**. One probe
  pair is a very noisy gradient: on the simulated device a single loss
  evaluation (16 rows × 1024 shots) scatters by 10–18 %, about as much as
  the two probes differ. So the sprint is hedged by default. SPSA moves only
  the output head `w, b` (`[hardware] spsa_groups`, CLI `--groups`; to first
  order a device's error is a slope and a bias per readout, which is what
  the head can absorb). Four up-front probe pairs calibrate the gain and a
  trust region caps the step, so the first step has a known size whatever
  the loss scale of the weights file. A step is taken only if the hardware
  loss at the proposed parameters is not worse (*blocking*, one more job per
  iteration; `--no-blocking`). And a step that would cost the driver more
  than 10 % of its greedy return on the exact simulator is refused before a
  job is spent on it (the *guard*: `[hardware] spsa_guard`, 0 or
  `--no-guard` turns it off). Ten iterations cost about 40 jobs (39–45 in
  our runs): 8 for the calibration, 20 probes, 1 for the starting point
  plus 1 for every step the guard lets through, and 6 for the fresh loss
  before and after (3 when no step was taken). Deliberate asymmetry: the loss is evaluated on the noisy
  backend, the greedy returns before and after (mean of 12 episodes) on the
  exact simulator.

  What was measured (2026-10-01/02, with the July 4-qubit oval driver that
  was bundled until the October retrain — a driver with knife-edge action
  margins, see "Why a driver fails under noise"; ten iterations, 1024
  shots, batch 16, the local `fake_miami` patch; a run "kept" the policy if
  it kept at least 80 % of its 12-episode simulator return):

  - *The first version of the sprint* — all 56 parameters, the gain from
    one probe pair, every step taken — kept the policy in 6 of 16 seeded
    runs. On 36 fresh episodes the driver went from 27 laps to a median of
    12.5 (8 runs) and from 26 to a median of 5.5 (8 runs with other seeds).
    The hardware loss still fell in 7 of those last 8 runs: a falling loss
    does not mean a surviving policy.
  - *The default sprint* kept it in 20 of 20 seeded runs. The guard
    guarantees that by construction, because it checks the very episodes
    the return is reported on. The independent check is 36 episodes the
    sprint never saw: 27 → median 32 (range 27–36; 12 runs) and 26 → median
    36 (28–36; 8 runs). It took between 0 and 8 of the 10 steps. The
    hardware loss, re-measured with 16 fresh evaluations per side, was lower
    by more than two standard errors in 15 of the 20 runs, unresolved in 4
    (one of which took no step) and higher in 1.
  - *The guard is what protects a fragile driver.* With head-only updates
    and blocking but no guard, 1 of 8 device runs lost the policy (return
    1254 → 60) while its hardware loss went down (13.4 → 8.3). On a
    noise-free "device" with a planted slope and bias per readout, the
    unguarded sprint lowered the loss (67.9 → 51.2) and took the simulator
    return from 1130 to 195: descending a 16-row TD loss can wreck this
    driver with no noise at all. How often that happens is not settled:
    without the guard the policy was lost in 1 of 8 runs on the device path
    and in 13 of 32 in emulation, while in four further seeded runs with
    every safeguard switched off it was kept all four times.
  - *It does not make the driver lap on the device.* On the device path
    before and after four default sprints (8 episodes each), that driver
    drove 1 lap of 8 before and 0 of 8 after each of the four.
    The gain in fresh-episode laps on the simulator is a side effect of
    taking only steps the exact simulator approves, for a driver that
    decides on knife-edge margins — it is not hardware learning.
  - On a driver trained for wide action gaps (see below) both versions left
    the policy alone: the first version kept it in 8 of 8 runs, the default
    in 14 of 14, with 36 of 36 fresh laps before and after in every default
    run.

  So the sprint is now a guarded, small adjustment of the output head that
  usually lowers the hardware loss a little. It demonstrates the mechanics
  of updating parameters against a device; it has not been shown to improve
  a driver, and everything above is the 4-qubit oval on one simulated
  device, measured before the retrain. On the driver bundled now there are
  two default sprints from the command line and nine seeded ones in
  notebook 05 so far — observations, not a result. The command-line sprints
  took 2 and 3 of 10 steps and did not lower the hardware loss (40.0 → 40.4
  and 35.5 → 40.3; each number is the mean of 3 fresh evaluations that
  scatter by 10–18 % apiece), with the simulator return unchanged (1892 →
  1889 and 1892 → 1899). The notebook's three default sprints (seeds 0–2)
  took 1, 3 and 1 steps (loss 40.5 → 32.4, 25.8 → 26.9, 100.6 → 84.9), and
  after each the driver lapped 36 of 36 fresh simulator episodes and 8 of 8
  device episodes, as before; its three sprints with the rebuilt original
  recipe and three with every safeguard off also left the policy intact
  (36 of 36 laps after each). The same notebook's contrast driver, trained
  without the noise-robust recipe, lost its policy to the rebuilt original
  recipe in one of three seeds (36 → 0 laps while the hardware loss fell
  6.1 → 3.0).
- **Why full training stays simulated:** a typical run is ~20 000 gradient
  steps. At param-shift pricing that is ~5 days of pure compute before queue
  time; even SPSA's 2 jobs/iteration makes hardware training a demonstration
  of mechanics, not a competitive training method at this scale.

**What survives the noise.** Measured on 2026-10-02 (the further sets
below partly on 2026-10-03) with
`tools/hw_reliability.py` on the local `fake_miami` patch: 1024 shots per
decision, resilience 0, no rescale; episodes from the eval env's jittered
spawns, on an env seed that no training or selection step used; a device
episode is capped at 200 decisions, enough for the first lap from a
standing start. Counts are episodes that completed a lap / episodes:

| Bundled driver | Exact simulator | Emulated device noise | Device path |
|---|---|---|---|
| `quantum_oval`, 4 qubits | 36 / 36 | 36 / 36 | 24 / 24 |
| `quantum_chicane`, 4 qubits | 35 / 36 | 36 / 36 | 24 / 24 |
| `quantum_oval_q6`, 6 qubits | 36 / 36 | 36 / 36 | 24 / 24 |
| `quantum_chicane_q6`, 6 qubits | 36 / 36 | 36 / 36 | 11 / 12 |
| `quantum_gp`, 4 qubits | 36 / 36 | 25 / 36 | 9 / 12 |
| `quantum_combo`, 4 qubits | 36 / 36 | 12 / 36 | 4 / 12 |
| `quantum_universal` on oval | 36 / 36 | 25 / 36 | 9 / 12 |
| `quantum_universal` on chicane | 36 / 36 | 11 / 36 | 3 / 12 |

(Device episodes are capped at 400 decisions on gp and 520 on combo. The
8- and 10-qubit drivers were not measured on the device path.)

The top three rows were repeated on three further sets of episodes and
shot-noise seeds (env seeds 61000, 73000 and 88000): 24 of 24, 12 of 12
and 12 of 12 device episodes for each of the three drivers — 72 of 72 per
driver over the four sets.

The first three are the hardware-demo drivers: trained with the
noise-robust recipe (`action_gap = 0.8`, `act_noise = { attenuation = 0.95,
shots = 1024 }`) and picked among the four best seeds of their study with
the device path ranking first. That ranking decided nothing — all twelve
shortlisted seeds lapped 8 of 8 — so the pick fell to pace.

It is the recipe that does it, not the pick. The same measurement for
*every* seed of the studies, nothing selected (12 device episodes per seed;
"truncation only" is `[training]` without the two robustness keys, 800
episodes in both cases):

| Track, qubits | Recipe | Seeds | Device laps over all seeds | Seeds lapping ≥ 11 of 12 | Laps per seed (of 12) |
|---|---|---|---|---|---|
| oval, 4 | truncation only | 8 | 73 / 96 | 5 | 12, 11, 0, 12, 10, 5, 11, 12 |
| oval, 4 | noise-robust | 10 | 118 / 120 | 10 | 12, 12, 12, 12, 12, 12, 11, 11, 12, 12 |
| chicane, 4 | truncation only | 8 | 55 / 96 | 0 | 10, 5, 0, 10, 9, 8, 8, 5 |
| chicane, 4 | noise-robust | 10 | 108 / 120 | 8 | 9, 12, 12, 11, 12, 12, 12, 12, 12, 4 |
| oval, 6 | truncation only | 8 | 92 / 96 | 7 | 12, 12, 12, 12, 11, 12, 9, 12 |
| oval, 6 | noise-robust | 8 | 95 / 96 | 8 | 12, 12, 12, 12, 11, 12, 12, 12 |

At 4 qubits the recipe decides it: without it 5 of 8 oval seeds and none of
8 chicane seeds reach 11 of 12; with it, 10 of 10 and 8 of 10. At 6 qubits
the circuit is nearly as robust without it.
The mechanism is the margin between actions. Along its own oval trajectory
the bundled 4-qubit driver's best and second-best Q-values are a median of
4.0 apart at |w| ≈ 65, about three standard deviations of the 1024-shot
noise on that difference. For the truncation-only driver of seed 4 —
which laps in 36 of 36 exact episodes and in 16 of 24 on the device path
(18 of 24, 8 of 12 and 10 of 12 on the three further sets: 52 of 72 in
all) — the median gap is 0.7, about 0.7 standard deviations.

**A lap on a physical QPU (2026-10-03).** The bundled 4-qubit oval driver
(`quantum_oval.npz`, trained with the noise-robust recipe and selected on
the simulated Nighthawk) drove one full oval lap on **`ibm_marrakesh`**, a
156-qubit Heron r2, through the IBM Quantum Open Plan: 141 decisions at
1024 shots each, resilience level 0 (raw device noise, no mitigation),
light-cone-pruned circuit routed to 27 CZ gates on the heavy-hex lattice.
It completed the lap in **14.1 s** of simulated driving (the same driver
laps in 12.7 s on the exact simulator) without leaving the track, in 27
minutes of wall-clock time and about 280 QPU-seconds. Sessions are refused
on the Open Plan (error 1352), so the run used a Batch, which the service
closes after 10 minutes; the hardware path now re-opens the batch and
retries the job (it happened twice; each re-open cost one queue wait of
2–4 minutes, the other decisions took 7 s median). Against the exact
simulator on the same car states, the device's greedy action agreed in 90
% of the 140 decisions, and the device Q-values were lower by 4–12 units
per action (the attenuation the noise model predicts; root-mean-square
difference 8.4 against an exact top-2 action gap of 8.3 median). A first
attempt without the re-open logic reached decision 75, still on track,
before the batch lifetime ended. One lap on one device on one day is a
data point, not a reliability number; the per-decision log is in
`data/qpu/ibm_marrakesh_oval_2026-10-03.json`.

What this does not show:

- **A simulated device.** `fake_miami` is a calibration snapshot of one
  Nighthawk, and only the 4 or 6 physical qubits the circuit lands on are
  simulated: no drift, no crosstalk from the rest of the chip, no queue.
  Other devices were not sampled; a single lap of the chicane driver on the
  heavy-hex `fake_fez` (27 CZ, TREX) completed, which is one run. The one
  physical-QPU lap above is a single run on a Heron, not the device the
  driver was selected on.
- **Not a guarantee per seed.** Chicane seed 9 of the noise-robust study
  did not learn the track well (17 of 36 exact episodes in the study's
  eval) and laps in 4 of 12 on the device path; seed 0 laps in 9. In the
  earlier 400-episode runs 5 of 24 seeds trained with the recipe did not
  learn to lap reliably (below).
- **Only the easy tracks.** gp, combo and the universal driver train with
  the 0.30 exploration floor and without the robustness keys; the 6-qubit
  chicane driver and the 8-qubit drivers come from studies without them
  too. The lower rows of the first table are what to expect from such
  drivers: laps in 11, 9, 4, 9 and 3 of 12 device episodes.
- **Not which lever matters.** Action gap and acting noise were only
  tested together at 800 episodes.

**Why a driver fails under noise, and what helps.** The analysis that led
to the recipe, measured on 2026-10-01/02 with `agents/quantum/noise.py` (an
expectation-noise model: attenuation, bias and shot noise per readout,
fitted to the device with one job) and `tools/hw_reliability.py`.
Everything in this list is the 4-qubit oval, and the freshly trained
drivers in it ran 400 episodes; "device" is the local `fake_miami` patch
(12 CZ); "the July driver" is the 4-qubit oval driver that was bundled
until the retrain, and "default recipe" the one of that time, without
action gap or acting noise.

- *What the device does to a readout.* Fitted against the exact simulator on
  random inputs, ⟨Z⟩_device ≈ f · ⟨Z⟩_exact with f = 0.95 (0.945–0.956 over
  25 parameter sets), and every readout has its own slope, between 0.94 and
  0.96. TREX moves f to about 0.98. On top comes the shot noise, whose
  variance matched the binomial formula within 4 %.
- *Why the July driver fell off the track.* It did not complete a lap on
  the simulated device (0 of 14 logged runs), because it decided on
  knife-edge margins. Along its own trajectory the gap between its best and
  second-best Q-value has a median of 0.46, at a head of |w| ≈ 52 — about
  one standard deviation of 1024-shot noise. Even on the exact simulator it
  lapped in 27 of 36 episodes, and a random ±0.2 on its head moves that
  anywhere between 5 and 35. In emulation it lapped in 10 of 36 with shot
  noise alone, in 33 of 36 with the common attenuation alone, and in 2 of
  36 with nothing but the differences of about ±1 % between the readouts'
  slopes. For this driver the differences between readouts, not the overall
  shrink, are what re-rank the actions.
- *That is this driver, not the rule.* For eight freshly trained
  default-recipe drivers (seeds 100–107; emulation, mean laps of 36): exact
  27.5, common attenuation alone 25.1, per-readout slopes and biases alone
  27.6, 1024-shot noise alone 22.6, everything together 16.6. Shot noise is
  the largest single component, the systematic errors alone cost little on
  average, and the combination is worse than either.
- *Mitigation at inference time* — more shots, the per-readout rescale
  (`[hardware] rescale = "readout"`, CLI `--rescale readout`, one extra
  calibration job), TREX — did not rescue the July driver: 8 laps in 36
  device episodes over twelve settings in one sweep, 3 in 32 over four
  settings in a second one with other seeds. It does rescue drivers with
  moderate margins. Two default-recipe drivers that lapped 1 of 12 device
  episodes at 1024 raw shots lapped 8 of 8 with the readout rescale at 4096
  shots (3 and 4 of 8 at 4096 shots without it, 8 and 7 of 8 at 16384); a
  third (5 of 12) gained nothing from the rescale (4 of 8 with it at 4096
  shots, 7 of 8 without). The global rescale (`--rescale global`) did
  nothing for the July driver (0 of 3 device episodes).
- *Training for it works more often, not every time.* The recipe —
  `action_gap = 0.8`, advantage learning [23] for wider gaps, and
  `act_noise = { attenuation = 0.95, shots = 1024 }`, acting under emulated
  device noise — was chosen among seven on seeds 0–7. On 16 seeds it was
  not chosen on (8–15 and 100–107; 400 episodes each), 9 of its drivers
  still lapped in at least 32 of 36 episodes under emulated 1024-shot
  device noise, against 2 of 16 for the default recipe. On the device path
  (seeds 100–107, 12 episodes per driver, 1024 raw shots) 5 of 8 lapped at
  least 11 of 12, against 1 of 8. The costs at 400 episodes: 5 of the 24
  seeds trained with it did not learn to lap reliably (the default recipe
  has such seeds too); on the fresh seeds its exact-simulator lap count was
  no better than the default's (25.9 against 27.5 of 36). Notebook 05
  trains two further seeds per recipe (200 and 201, 400 episodes) and runs
  them on the device path: 8 and 8 of 8 with the recipe against 6 and 4 of
  8 without — a direction, not a rate, at two seeds. Which of the two
  levers does the work is unresolved. The 800-episode studies in the table above are the
  larger and later measurement, and the one the quantum presets rest on.
- *The emulation is a screening tool.* Over 25 drivers, the per-readout
  emulation's lap counts correlated 0.96 with 12 device episodes each, but
  for 2 drivers it was off by 4 or more laps of 12, and it is optimistic
  about the rescale, which inverts exactly the model the emulation is built
  from (July driver, readout rescale at 16384 shots: 21 of 36 emulated,
  0 of 8 on the device path). Compare recipes with it; judge a single
  driver on the device path.

`aer_noisy`, the noisy training/eval backend of the `EstimatorQNN` path
(`agents/quantum/qnn.py`), uses the same local twin of the same fake device.

## Measured results

Everything here except the last subsection was measured on 2026-10-01/02
under one protocol:

- **Many seeds.** Each recipe is a `tools/study.py` variant: 8–10 seeds at 4
  qubits, 8 at 6, 6 at 8 and at 10, 5 for the universal driver, 3 for the
  pro driver.
  A number written `x [a, b]` is the interquartile mean over seeds with its
  95 % bootstrap interval; "P" is the probability of improvement with its
  interval, and a difference counts as supported when that interval
  excludes 0.5. The per-seed data are in `data/studies/<name>/`.
- **Distinct episodes.** Snapshot selection during training uses 12 distinct
  greedy episodes; afterwards every seed's best snapshot and final
  parameters drive 36 further ones ("best-snapshot lapped", "final-params
  lapped": the fraction of those episodes with a lap).
- **Bundled drivers** are one seed each, chosen and re-evaluated as
  described under "From a study to a bundled driver".
- Physics v2 (`accel` 11, `brake` 16, `v_max` 25; since July 2026),
  `fastsim`. "Mean lap" averages every lap of an episode, the first one
  from a standing start included. "First clean lap" is the training episode
  in which an exploring car first completes a lap.

### The circuit against the matched MLP (4 qubits)

Each agent under the recipe that ships for it ("The recipe is per track and
per agent"): 56-parameter circuit, 76-parameter MLP, same observation, same
double DQN.

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

(oval and chicane: 800 episodes; gp 3000; combo 2500. Mean lap and first
clean lap are over the seeds that lapped at all: 9 of 10 for the MLP on gp
and the circuit on combo, and only 6 of 10 circuit seeds drove a clean lap
while exploring on combo. For the circuit on oval and chicane the
in-training evals behind "stability" are driven under the recipe's emulated
device noise; the 36-episode evals are exact.)

What the table supports, and what it does not:

- **Easy tracks: the MLP is ahead.** It drives its first clean lap in about
  half the episodes (oval 153 against 264, chicane 183 against 355; P 0.99
  [0.93, 1.00] on both) and holds on to what it learned (stability P 0.78
  [0.53, 0.95] on the oval, 0.93 [0.74, 1.00] on chicane). On the oval it
  is also 0.8 s faster (P 0.82 [0.59, 1.00]); on chicane the lap times do
  not differ (P 0.36 [0.10, 0.65]).
- **Hard tracks: the MLP is faster, and neither agent is stable.** Its
  best snapshots lap gp about 10 s and combo about 18 s faster (P 0.92
  [0.77, 1.00] and 1.00). Stability is low for both and does not differ
  (P 0.40 [0.15, 0.67] on gp, 0.41 [0.15, 0.69] on combo), and neither
  agent's end-of-training parameters lap.
- **One thing goes the circuit's way on gp, and it is about speed of
  learning, not quality.** Half of all circuit seeds have a greedy eval
  that laps in at least half its episodes by episode 1100, and all 10 get
  there; the MLP needs 2150 episodes and 7 of 10 get there. Under the
  shared July recipe the gap is similar (800 against 2250), and there the
  circuit's first clean lap is also earlier (1457 [1305, 1755] against
  1875 [1778, 2054]). The circuit finds a slow lapping policy early and
  does not keep it; the MLP finds a faster one late.

So the earlier claim on this page — "broadly similar sample efficiency and
comparable lap times" — is withdrawn. It rested on one to three seeds.

### The bundled drivers

One seed each (`traqmania/weights/<name>.meta.json`, block `selection`).
"Fresh" is the 72-episode evaluation that chose among the shortlisted
seeds; "re-check" is 36 episodes on an env seed no selection used
(`python -m traqmania.records --episodes 36 --seed 47000`, run on
2026-10-02). Lapped episodes, then mean lap:

| Driver | Qubits × blocks | Seeds in its study | Fresh (72 episodes) | Re-check (36 episodes) |
|---|---|---|---|---|
| `quantum_oval` | 4 × 4 | 10 | 72, 12.7 s | 36, 12.7 s |
| `quantum_chicane` | 4 × 4 | 10 | 72, 12.6 s | 35, 12.7 s |
| `quantum_gp` | 4 × 4 | 10 | 72, 27.9 s | 36, 27.9 s |
| `quantum_combo` | 4 × 4 | 10 | 72, 37.1 s | 36, 37.2 s |
| `quantum_oval_q6` | 6 × 4 | 8 | 72, 13.1 s | 36, 13.1 s |
| `quantum_chicane_q6` | 6 × 4 | 8 | 72, 12.9 s | 36, 12.9 s |
| `quantum_oval_q8` | 8 × 5 | 6 | 72, 13.4 s | 36, 13.4 s |
| `quantum_chicane_q8` | 8 × 5 | 6 | 72, 12.6 s | 36, 12.6 s |
| `mlp_oval` | — | 8 | 72, 12.6 s | 36, 12.6 s |
| `mlp_chicane` | — | 8 | 72, 13.3 s | 36, 13.3 s |
| `mlp_gp` | — | 10 | 72, 22.5 s | 36, 22.5 s |
| `mlp_combo` | — | 10 | 72, 36.9 s | 36, 36.9 s |

Every re-selected driver laps in all 72 fresh episodes; that is what the
selection optimises, and it says nothing about the average seed (previous
table). It is not a guarantee for the next 36 episodes either: repeating
the re-check on a further env seed (61000, same command), ten of the
twelve lap in all 36 again, `quantum_chicane` in 35 again and `quantum_gp`
in 34; on a third (88000) eleven do and `mlp_combo` laps in 35. Mean laps
stay within 0.3 s of the first re-check on both. Reliability ranks before pace, and on combo that shows: `mlp_combo`
(36.9 s) was preferred to a seed that laps in 25.6 s but missed one of its
72 episodes, so on combo the bundled MLP is no faster than the bundled
circuit although the MLP recipe is, by 18 s. On gp the rule cost nothing:
`quantum_gp` (27.9 s) is the fastest of its ten seeds as well as a
reliable one. It is the recipe that is slow (35.0 s [30.8, 40.1] over the
seeds), and the bundled driver is slower than the July one it replaces,
which was quoted at 23–24 s and lapped in 29 of 36 old-protocol episodes.
`quantum_oval`, `quantum_chicane` and `quantum_oval_q6`
were trained with the noise-robust recipe and additionally ranked by laps
on the simulated device ("Hardware"). The 8-qubit and the 6-qubit chicane
drivers come from studies without that recipe.

The files are reproducible. Bundling copies a study cell's weights byte
for byte, training is deterministic per seed, and the shipped config now
resolves to the studied recipe: `python -m traqmania.train_headless --agent
quantum --track oval --seed 0 --episodes 800`, run in this tree on
2026-10-02, reproduced `quantum_oval.npz` bit for bit (the sha256 its
sidecar records) on the machine the studies ran on.

Not re-selected: the three 10-qubit drivers are still the July files (4
blocks, chosen under the old protocol). On the same 36 re-check episodes
`quantum_oval_q10` laps in 34 (12.6 s), `quantum_chicane_q10` in 34
(13.7 s) and `quantum_gp_q10` in 24 (22.2 s, best 19.7 s); on the second
re-check seed in 32, 34 and 26, on the third in 35, 36 and 25.

### Scaling and the light cone

Oval and chicane at 4, 6 and 8 qubits and the oval at 10, all with the same
recipe (`[training]` with truncation, 800 episodes, no noise-robust
options), next to an MLP on the same observation:

| Track | Agent | Parameters | Seeds | Best-snapshot lapped | Final-params lapped | Stability | Best-snapshot mean lap | First clean lap (episode) |
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

- **Six qubits is where the circuit does best.** All 8 seeds give a
  reliable driver on both tracks, and every metric is at least as good as
  at 4 qubits. Only one of those differences is supported — the best
  snapshot's lapped fraction on chicane, P 0.80 [0.59, 0.98] — so read it
  as "no worse, probably steadier". On chicane its laps are about as fast
  as the matched MLP's or faster (13.4 s against 14.0 s; P 0.66
  [0.36, 0.94], not supported), while the MLP is still the steadier
  learner (stability P 0.81 [0.56, 1.00]). On the oval the matched MLP is
  ahead on both counts: faster (12.8 s against 13.3 s; P 0.88
  [0.62, 1.00]) and steadier (P 0.81 [0.56, 0.98]). With the noise-robust
  recipe on the oval the 6-qubit circuit reaches the MLP's stability (0.86
  [0.79, 0.93] against 0.87 [0.81, 0.93]; 8 seeds each).
- **Eight qubits need the fifth block, and still gain nothing over six.**
  At 4 blocks — every action blind to one input — the end of training is
  poor (final-params lapped 0.36 on the oval) and one seed in six fails on
  each track. Five blocks repair that on the oval (P 0.92 [0.72, 1.00] for
  the final parameters) and give 6 of 6 reliable seeds on both tracks,
  which is why the `q8` profile and the bundled 8-qubit drivers now have
  5. On chicane the end of training stays poor at either depth (0.37
  [0.07, 0.80] against 0.28 [0.04, 0.57]; P 0.61 [0.25, 0.92], not
  supported). And against 6 qubits the 5-block circuit is less stable and
  slower on the oval (P 0.16 [0.00, 0.41] and 0.12 [0.00, 0.38]); on
  chicane it trends less stable too, without support (P 0.29
  [0.04, 0.60]), at the same lap time.
- **Seven rays help the MLP on chicane; extra qubits make the circuit no
  faster anywhere.** With 7 rays the MLP laps chicane in 12.9 s against
  13.9 s with 3 (P 0.85 [0.56, 1.00]) and drives its first clean lap
  earlier (P 0.86 [0.65, 1.00]); on the oval, and with 5 rays on either
  track, its gains are not supported (oval, 7 rays: P 0.67 [0.38, 0.96]).
  With 9 rays on the oval its first clean lap comes earlier than with 3
  (P 0.90 [0.67, 1.00]); its lap time is not supportedly better (P 0.77
  [0.50, 1.00]).
  The 8-qubit circuit is not faster than the 4-qubit one on either track
  (P 0.31 [0.06, 0.62] on the oval, 0.71 [0.38, 0.98] on chicane), and on
  the oval the 7-ray MLP beats it in every pairing of seeds (12.7 s
  [12.6, 12.8] against 14.4 s; P 1.00). What grows reliably with the qubit
  count is compute: the statevector goes from 16 to 256 amplitudes, 1024
  at 10 qubits.
- **gp above 4 qubits.** The plain 6-qubit profile does not learn gp under
  the July recipe (0 of 6 seeds, `data/studies/gp_4q`, variant `q6`), and
  the 10-qubit feature circuit does not learn it under the new one
  ("Training stability"). The bundled `quantum_gp_q10` is a July single
  run with a Brake readout that cannot see corner speed; it laps in 24 of
  36 re-check episodes at 22.2 s — faster than the bundled 4-qubit gp
  driver (27.9 s), level with the bundled MLP (22.5 s), and the least
  reliable of the three.
- **Ten qubits on the oval: no gain, at either depth.** With 4 blocks —
  every action blind to three inputs — the circuit is less stable than the
  4-qubit one (0.42 [0.27, 0.50] against 0.69 [0.57, 0.75]; P 0.07
  [0.00, 0.25]) and its best snapshots are less reliable (P 0.14
  [0.00, 0.33]). Six blocks give full visibility and point the same way as
  the fifth block at 8 qubits — end-of-training parameters 0.68 against
  0.36, laps 13.3 s against 14.3 s — without either difference being
  supported at 6 seeds (P 0.71 [0.38, 1.00] and 0.75 [0.42, 1.00]). Against
  the 6-qubit circuit the 6-block one is less stable (P 0.12 [0.00, 0.38])
  and no faster (P 0.58 [0.23, 0.92]). The MLP on the same 9 rays is
  steadier than either depth and drives its first clean lap earlier, in
  every pairing of seeds (P 1.00 for both); its laps are faster than the
  4-block circuit's in every pairing and not supportedly faster than the
  6-block one's (P 0.83 [0.50, 1.00]).
- **10 qubits on chicane** (`data/studies/chicane_q10`, 6 seeds per depth):
  4 and 6 blocks are indistinguishable — end-of-training lapped 0.55 against
  0.55, stability 0.46 [0.31, 0.70] against 0.45 [0.32, 0.66], laps 13.6 s
  against 13.1 s (P 0.51 [0.17, 0.88]) — and the matched MLP is again
  steadier (0.93 [0.80, 0.98]) and faster (12.7 s). The `q10` profile runs 6
  blocks because full visibility costs nothing measurable and helped on the
  oval.

### One driver, every track (cross-track generalization)

The observation is egocentric (lidar rays + speed, no absolute position),
so a policy is not tied to its training track. Zero-shot, greedy, 36
episodes per cell on the re-check seed — episodes that lap, and the mean
lap:

| Driver | oval | chicane | gp | combo |
|---|---|---|---|---|
| `quantum_oval` | 36, 12.7 s | 2 | 0 | 0 |
| `quantum_chicane` | 35, 12.6 s | 35, 12.7 s | 0 | 0 |
| `quantum_gp` | 36, 22.2 s | 36, 22.8 s | 36, 27.9 s | 21, 32.9 s |
| `quantum_combo` | 36, 26.7 s | 36, 26.8 s | 36, 34.5 s | 36, 37.2 s |
| `quantum_universal` | 36, 13.7 s | 36, 13.9 s | 36, 32.3 s | 35, 41.8 s |
| `mlp_oval` | 36, 12.6 s | 36, 12.7 s | 0 | 0 |
| `mlp_chicane` | 36, 13.1 s | 36, 13.3 s | 0 | 0 |
| `mlp_gp` | 36, 17.3 s | 36, 17.9 s | 36, 22.5 s | 1 |
| `mlp_combo` | 36, 31.1 s | 26, 32.0 s | 36, 33.1 s | 36, 36.9 s |

Transfer runs downhill: drivers trained where braking matters also lap the
flat-out tracks, far off the specialists' pace; the oval and chicane
specialists never lap gp or combo. The new 4-qubit oval
specialist does not even transfer to chicane (2 of 36), while the chicane
specialist drives the oval. On the second re-check seed (61000) no cell of
the matrix moves by more than three episodes (`mlp_combo` on chicane 29,
`quantum_universal` on combo 33, `quantum_gp` on gp 34 and on combo 22,
`quantum_oval` on chicane 3); on a third (88000) one does — `quantum_gp`
on combo laps in 16 — and every other cell is within one episode. The gp
specialist's combo transfer is the one number here to quote as a range,
16–22 of 36.

The bundled **`quantum_universal.npz`** is one 4-qubit circuit trained
fresh on all four tracks round-robin (3000 episodes, gp-style recipe with
the 0.30 floor). It is no longer a warm migration of an older driver:
across 5 seeds the fresh recipe gives best snapshots that lap in 0.89
[0.68, 1.00] of 36 episodes (9 per track), and fine-tuning the July
universal driver under the same recipe did worse (0.64 [0.49, 0.85];
stability P 1.00 for fresh). The top three seeds lapped in 144, 143 and 141
of 144 fresh episodes. The bundled one is seed 3 (143 of 144: oval 27.7 s,
chicane 27.4 s, gp 35.3 s, combo 38.5 s), chosen by hand over the
top-ranked seed 0 (144 of 144 at 13.7 / 13.9 / 32.2 / 41.5 s) for the
reason the next paragraph measures: seed 0 drives only the four tracks it
was trained on. The end of training is again not the
driver (final-params lapped 0.47 [0.25, 0.76], stability 0.35
[0.28, 0.48]). An MLP under the identical recipe reaches 0.95 [0.60, 1.00]
and stability 0.48 [0.46, 0.49] — with the caveat that the 0.30 floor is
the circuit's recipe, not the MLP's.

**Unseen tracks: selection on the training tracks alone picks a
specialist.** The top-ranked seed 0, on ten generated tracks
(`env/trackgen.generate_track`, seeds 100–109, 12 episodes each,
zero-shot), completes no lap at all — 0 of 120 episodes at difficulty 0.5,
and 0 of 120 at the 0.65 that the demo's 🎲 tracks use; in the demo's
random- and drawn-track modes it brakes to a stop. The July driver lapped
all ten (120 of 120 and 116 of 120 episodes) while being less reliable on
the four bundled tracks (26 of 36 on chicane and 30 of 36 on gp on the
re-check episodes). The selection rule ranked the study's seeds on the
four training tracks only, and the seed it picked had specialised on
exactly those. All five seeds of the study, on the same tracks:

| Seed | Four bundled tracks (36 episodes each) | Generated, difficulty 0.5 | Generated, difficulty 0.65 |
|---|---|---|---|
| 0 (top-ranked) | 36, 36, 36, 35 at 13.7 / 13.9 / 32.3 / 41.8 s | 0 of 120 | 0 of 120 |
| 1 | 36, 34, 36, 31 at 20.8 / 21.8 / 31.0 / 36.9 s | 120 of 120 | 41 of 120 |
| 2 | 36, 1, 21, 28 | 6 of 120 | 0 of 120 |
| 3 (bundled) | 36, 36, 36, 36 at 27.6 / 27.4 / 35.2 / 38.2 s | 120 of 120 | 120 of 120 |
| 4 | 36, 20, 13, 18 | 115 of 120 | 47 of 120 |

Seed 3 is the one driver here that deserves the name — every bundled and
every generated track, at about twice the oval and chicane lap time of
seed 0 — and it is the one that ships (`tools/bundle_driver.py --seed 3`,
recorded in the sidecar; the automatic rule would have shipped seed 0).
Ranking seeds on unseen tracks as well is the open item. The hard-track specialists generalize too: `quantum_gp` and
`quantum_combo` each lap all ten generated tracks in 120 of 120 episodes
at both difficulties (gp: mean laps of 30–39 s and 33–41 s), and `mlp_gp`
in 120 and 113 of 120. Ten other generated tracks (seeds 300–309, env seed
61000) give the same picture: the bundled universal driver 0 of 120 at
both difficulties, seed 3 120 of 120 at both, `quantum_gp` 120 of 120 at
both, `quantum_combo` 120 and 106 (it laps one of the ten in none of its
12 episodes), `mlp_gp` 120 and 113, the July universal driver 116 and 118,
seed 1 120 and 68. A third set (seeds 500–509, env seed 88000) repeats it
once more: the bundled universal driver 0 of 120 at both difficulties,
seed 3 and `quantum_gp` 120 of 120 at both, `quantum_combo` 120 and 108,
`mlp_gp` 120 and 116. So in the bundle as it stands, "universal" means
"all four bundled tracks", and the random- and drawn-track modes, which
hand the car to `quantum_universal`, get a driver that brakes to a stop
within the first tens of metres (in the demo's attract mode on 20 generated
tracks it never left the start line on one and was standing still at the
end on the other 19, mostly on the track with Brake as its greedy action;
the July driver and seed 3 lapped all 20). A universal driver has to be
ranked on tracks it was not trained on.

### The ceiling: a model-based reference driver

To know how good the learned drivers are, the expert demo includes a
**hero** driver that is not learned at all: it builds a family of candidate
racing lines from the track geometry (curve-shortening flow, blended wide
through slow corners), derives brake/accelerate-feasible speed profiles from
the `[physics]` constants, *simulates itself* on each candidate with the real
car physics, and drives the fastest crash-free combination with continuous
steering (pure pursuit). The expert menu's **pro** driver is the strongest
learned one: the same double DQN as every agent, with a wide MLP (hidden
128; 2,436 parameters) and a 14-feature observation (9 rays, speed, four
track scalars), trained on all four tracks at once for 5000 episodes.

Measured on the re-check seed, 12 episodes per track from the eval env's
jittered standing starts — episodes with a lap, best lap, and the mean
over all laps (the first one from standstill included):

| Driver | oval | chicane | gp | combo |
|---|---|---|---|---|
| hero | 12, 12.1 s, 12.6 s | 12, 12.1 s, 12.6 s | 12, 16.5 s, 17.0 s | 6, 19.0 s, 19.7 s |
| pro (`mlp_pro`) | 12, 12.1 s, 12.7 s | 12, 12.5 s, 13.1 s | 12, 16.5 s, 17.7 s | 12, 19.1 s, 20.3 s |
| bundled 4-qubit circuit (36 episodes) | 36, 12.1 s, 12.7 s | 35, 12.0 s, 12.7 s | 36, 26.6 s, 27.9 s | 36, 36.0 s, 37.2 s |
| bundled MLP (36 episodes) | 36, 12.0 s, 12.6 s | 36, 12.8 s, 13.3 s | 36, 21.4 s, 22.5 s | 36, 35.7 s, 36.9 s |

Three things follow.

- **The hero's best laps are the pace target** — 12.1, 12.1, 16.5 and
  19.0 s — and, being derived from the `[physics]` constants, they need no
  retraining when those change. It is less crash-proof than its
  construction suggests: from jittered spawns it left the track before the
  first lap in 6 of 12 combo episodes (8 of 12 on each of two further sets
  of starts, env seeds 61000 and 88000; on the other three tracks 12 of 12
  every time, with a 16.4 s gp lap on the third set), and on 1 of 7
  generated tracks (difficulty 0.5, one episode each).
- **The gap to the ceiling is not the action interface.** The pro driver
  steers with the same four bang-bang actions at 10 Hz as every other agent
  and matches the hero's best lap on oval and gp, within 0.1 s on combo and
  0.4 s on chicane (on two further sets of 12 starts its best laps were
  12.1, 12.6, 16.8 and 19.2 s, then 12.1, 12.6, 16.8 and 19.3 s, each time
  a lap in 12 of 12 on every track). What
  separates the small agents from it on the hard
  tracks — 10 s per lap for the circuit and 5 s for the MLP on gp — is what
  a 14-feature observation and 2,436 parameters buy. An earlier version of
  this page attributed most of that gap to the 4-action interface; the pro
  driver's laps do not support that.
- **The pro driver is a selected seed too.** Of three seeds, the best
  snapshots lapped in 36, 28 and 27 of the study's 36 episodes; the
  bundled one laps in 144 of 144 fresh episodes (the other two in 128 and
  109), and in 23 of 24 episodes on eight generated tracks (difficulty 0.5,
  three episodes each; at least two laps of three on every track). On the
  flat-out oval nothing separates the learned drivers from the hero: the
  bundled 56-parameter circuit and 76-parameter MLP reach best laps of
  12.1 and 12.0 s there.

Hero and pro laps are excluded from ghost records — records stay with the
standard demo agents (and humans).

### July 2026 exploratory results (old protocol)

Two campaigns of July 2026 have no successor measurement: observation
engineering, and a set of mechanisms meant to make extra qubits pay off.
Their numbers came from one to three seeds, snapshots chosen on four
evaluation episodes, and — at 8 and 10 qubits — circuits that hid inputs
from actions. **None of them was re-measured, and none is quoted here.**
(The earlier text is in the repository history.) What remains in the code
and is worth knowing:

- **The feature registry** (`[observation] features`;
  `traqmania/env/racing_env.py`). Any qubit count can trade lidar rays for
  engineered scalars, one feature per qubit, all normalized to [0, 1]:
  `rays`, `speed`, `curvature_ahead` (max centerline |κ| over a lookahead
  window; a suffix sets the horizon, e.g. `"curvature_ahead:30"`),
  `lateral_offset`, `heading_error`, and `corner_speed_ratio` = v /
  v_safe(R), with v_safe derived from the car model's steering kinematics —
  "am I too fast for the corner coming up?". A weights file records the
  observation it was trained on and every loader adopts it, which is how
  `quantum_gp_q10` (5 rays, speed, four features) coexists with plain-ray
  drivers at the same qubit count. July's reading was that features neither
  reliably helped the circuit nor reliably hurt the MLP at three seeds;
  with one feature hidden from each action at 8 qubits and three at 10,
  that was not a fair test in either direction.
- **Scaled action sets** (`[circuit] n_actions`, `train_headless
  --actions`). The readout takes Q_a = ⟨Z_a⟩ from the first *k* qubits: 6
  actions add trail braking (full steer + brake), 8 add half-steer
  (`traqmania/agents/base.py`; prefix-compatible, so the 4-action default
  is unchanged). In July no scaled-action run on gp converged to reliable
  greedy lapping; no weights with more than 4 actions are bundled.
- **A pace objective** (`[reward] time_penalty`, `train_headless --pace
  --init <lapping snapshot>`): a low-epsilon fine-tune whose reward charges
  each decision, so that lap time and not only lap completion is optimised.
  July's two single runs were inconclusive: one traded reliability for
  pace, the other the reverse. A pace-phase
  selection rule — best mean lap above a reliability floor — was never
  built, so the reliability-first ranking still applies inside a pace run.

## Honest claims

Being honest matters more than being exciting:

- **The classical baseline is ahead; "parity" is withdrawn.** A
  56-parameter circuit and a 76-parameter MLP both learn every track with
  the same double DQN, and that a VQC *can* do this is still the point,
  echoing Chen et al. [1] and Skolik et al. [2]. But by the standard the
  field now applies — many seeds, interval estimates, a statistical
  definition of outperformance [6, 7] — the two are not at parity. Over
  8–10 seeds the MLP drives its first clean lap in about half the episodes
  on oval and chicane and is far more stable there, and its laps are about
  10 s faster on gp and 18 s on combo ("Measured results"). An earlier
  version of this page spoke of "broadly similar sample efficiency and
  comparable lap times"; that rested on one to three seeds and
  best-snapshot numbers. It is the outcome the benchmarking literature
  would have predicted: in a large benchmark of quantum classifiers,
  out-of-the-box classical models won overall [8].
- **What the circuit does show.** It learns each of the four tracks from
  scratch, and a single 4-qubit circuit learns all four at once; one seed
  of five also drives every unseen track we generated, though not the seed
  the selection rule bundled. On gp its
  seeds reach a lapping greedy policy in about half the MLP's episodes — a
  slower policy, and one that training does not keep. At 6 qubits it gives
  a reliable driver on both easy tracks in 8 of 8 seeds, with chicane laps
  level with the matched MLP's. Every re-selected bundled driver laps in 72
  of 72 fresh episodes, and three of them complete their laps on a
  simulated IBM device. None of this is an advantage. It is what "it works"
  means here, with intervals.
- **Shipped drivers are selected, and we say how.** On gp and combo the
  end-of-training parameters of either agent almost never lap; the bundled
  driver is the best snapshot of the best of ten seeds. "72 of 72"
  describes that one file, the tables describe the recipe, and both are
  recorded next to the weights. A study of 29 training variants found no
  setting that makes the optimisation itself stable ("Training
  stability").
- **No quantum advantage is claimed — or possible here.** Four qubits — or
  ten — are trivially simulable; training literally ran on a classical
  simulation of the circuit. The argument is stronger than "small enough to
  brute-force". The trained circuit is a Fourier series with a finite set of
  known frequencies per input [3], so a purely classical model can be fitted
  to reproduce it — a *classical surrogate* [13]; necessary and sufficient
  conditions for an efficient random-Fourier-feature dequantization are
  established for regression [14], and first guarantees exist for
  Q-learning in a simplified setting [15]. More generally, Cerezo et al.
  [16] collect evidence, case by case, that many commonly used variational
  models whose loss landscapes avoid barren plateaus can also be simulated
  classically, given some data collected from a quantum device first.
  Notebook 07 — light cones and classical surrogates — builds such
  surrogates (`agents/quantum/surrogate.py`); the numbers here are from its
  execution on 2026-10-03 against the drivers bundled now (a run on the
  July drivers on 2026-10-01/02 gave the same picture). The encoding of the
  4-qubit, 4-block circuit allows 531,441 frequency vectors per readout,
  but only 45,072 of them can carry a non-zero coefficient, for trained and
  for random angles alike. Kernel surrogates fitted to 1,000 states sampled
  while the quantum driver drove lap exactly like that driver in every
  comparison made (36 of 36 episodes on oval and gp at 4 qubits and on the
  oval at 6 — as the circuits themselves; single fits, with two refits each
  on gp and the 6-qubit oval) — but on the oval so do a kernel built from
  the *untrained* frequencies and a generic Gaussian kernel (36 of 36 each),
  and on driving states those controls fit the Q-values nearly as well
  (normalized error 0.090 and 0.108 against 0.082 for the trained-frequency
  kernel in the notebook's single fits; 0.02–0.08 over 20 fits with other
  sample seeds in `tests/test_surrogate.py`). A surrogate that laps is
  therefore weak evidence for the Fourier picture. The evidence is in the
  function: least squares on the predicted frequencies reproduces small
  circuits to better than 10⁻¹³, where frequencies scaled by 1.3 leave
  errors of 0.05 to 1.7 in four of the five circuits tried (the fifth,
  whose readouts each depend on one feature through 27 sinusoids, fits
  either way); and fitted on uniformly sampled inputs, the true-frequency
  kernel reaches a normalized error of 0.006 on the cube where the Gaussian
  kernel reaches 0.245 — and only the former still laps (36 of 36 against
  0 of 36). Only inference is dequantized there — fitted Q-iteration on the
  surrogate [15] was not run. IBM's own teaching material makes the same point about
  expectations: "It is not realistic to expect a quantum speed-up for
  machine learning tasks that classical computers already do quite well"
  [19]. An advantage claim needs an output
  whose correctness can be validated and a demonstrated separation from the
  best classical methods [20]; the open community tracker for such claims
  follows observable estimation, variational problems and classically
  verifiable problems [21]. A racing toy is none of these.
- **Engineered features and 10 qubits: no claim.** The July comparisons of
  engineered observations, and every 8- and 10-qubit result of that
  campaign, were measured with circuits that could not show every input to
  every action, on one to three seeds. An asymmetry we once reported —
  features helping the circuit and hurting the MLP — had already dissolved
  at three seeds; neither it nor its absence was ever tested fairly.
  Nothing about features has been re-measured. Ten qubits on plain rays
  have been, on both easy tracks: no gain over 6 qubits at 4 or at 6
  blocks.
- **Watch the denominators.** Every comparison here is per *episode*; per
  *second* the MLP trains far faster (cheaper gradients). Parameter count
  is an imperfect fairness measure — expressivity per parameter differs,
  and at 6 qubits and up a growing share of the circuit's parameters is
  dead (12 of 72 at 6 qubits, 26 of 120 at 8 qubits and 5 blocks, 46 of 120
  at 10 and 4). Each agent is compared under the recipe that ships for it,
  and the recipes differ: the circuit's has options the MLP's does not, so
  where the two were run under a shared recipe the text says so. And a
  bundled driver's numbers come after two selection steps (best snapshot,
  best seed).
- **Noise robustness: what is known, what is open.** It is not an untouched
  question — Skolik et al. [10] studied shot noise, coherent and incoherent
  errors in the training and evaluation of variational RL agents. What this
  testbed adds is narrow and concrete: a policy trained on an exact
  simulator, with an output head that multiplies ⟨Z⟩ by 25–95, driven
  closed-loop where one flipped argmax can end the episode. An earlier
  version of this page said the trained policy "laps at reference pace"
  under shot noise; for the driver bundled at the time that was wrong — it
  completed no lap in 14 logged runs on the simulated device. What can be
  said now:
  whether a driver survives is decided by the margins between its action
  values, and margins can be trained. With advantage learning and acting
  noise in training, 10 of 10 oval seeds and 8 of 10 chicane seeds lap in
  at least 11 of 12 simulated-device episodes at 4 qubits (5 of 8 and 0 of
  8 without), and the three bundled hardware-demo drivers lap in 72 of 72
  device episodes over four sets ("Hardware"). At inference time, more shots and a per-readout rescale
  restore drivers with moderate margins, not a knife-edge one. Still open:
  which of the two training levers matters, the hard tracks, whatever a
  calibration snapshot of one device leaves out — and physical QPUs beyond
  the single `ibm_marrakesh` lap of 2026-10-03.

## References

Every entry was checked against its arXiv, journal, Crossref or IBM page on
2026-10-01 (entry 23 on 2026-10-02).

1. S. Y.-C. Chen, C.-H. H. Yang, J. Qi, P.-Y. Chen, X. Ma, H.-S. Goan,
   *Variational Quantum Circuits for Deep Reinforcement Learning*, IEEE
   Access 8, 141007–141024 (2020), doi:10.1109/ACCESS.2020.3010470.
   <https://research.ibm.com/publications/variational-quantum-circuits-for-deep-reinforcement-learning>
2. A. Skolik, S. Jerbi, V. Dunjko, *Quantum agents in the Gym: a variational
   quantum algorithm for deep Q-learning*, Quantum 6, 720 (2022).
   <https://quantum-journal.org/papers/q-2022-05-24-720/>
3. M. Schuld, R. Sweke, J. J. Meyer, *The effect of data encoding on the
   expressive power of variational quantum machine learning models*, Phys.
   Rev. A 103, 032430 (2021). <https://arxiv.org/abs/2008.08605>
4. N. Meyer, C. Ufrecht, M. Periyasamy, D. D. Scherer, A. Plinge,
   C. Mutschler, *A Survey on Quantum Reinforcement Learning*,
   arXiv:2211.03464 (v2, March 2024; no journal reference listed).
   <https://arxiv.org/abs/2211.03464>
5. M. E. Sahin, E. Altamura, O. Wallis, S. P. Wood, A. Dekusar,
   D. A. Millar, T. Imamichi, A. Matsuo, S. Mensa, *Qiskit Machine Learning:
   an open-source library for quantum machine learning tasks at scale on
   quantum hardware and classical simulators*, arXiv:2505.17756 (2025).
   <https://arxiv.org/abs/2505.17756>
6. N. Meyer, C. Ufrecht, G. Yammine, G. Kontes, C. Mutschler, D. D. Scherer,
   *Benchmarking Quantum Reinforcement Learning*, Proceedings of the 42nd
   International Conference on Machine Learning (ICML 2025), PMLR 267,
   43934–43964. <https://arxiv.org/abs/2501.15893>
7. R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare,
   *Deep Reinforcement Learning at the Edge of the Statistical Precipice*,
   NeurIPS 2021. <https://arxiv.org/abs/2108.13264>
8. J. Bowles, S. Ahmed, M. Schuld, *Better than classical? The subtle art of
   benchmarking quantum machine learning models*, arXiv:2403.07059 (2024).
   <https://arxiv.org/abs/2403.07059>
9. F. Pardo, A. Tavakoli, V. Levdik, P. Kormushev, *Time Limits in
   Reinforcement Learning*, ICML 2018, PMLR 80, 4045–4054.
   <https://arxiv.org/abs/1712.00378>
10. A. Skolik, S. Mangini, T. Bäck, C. Macchiavello, V. Dunjko, *Robustness
    of quantum reinforcement learning under hardware errors*, EPJ Quantum
    Technology 10, 8 (2023). <https://doi.org/10.1140/epjqt/s40507-023-00166-1>
11. J. C. Spall, *Multivariate stochastic approximation using a simultaneous
    perturbation gradient approximation*, IEEE Trans. Autom. Control 37(3),
    332–341 (1992), doi:10.1109/9.119632 — the SPSA method used by the
    hardware sprint.
12. J. Gacon, C. Zoufal, G. Carleo, S. Woerner, *Simultaneous Perturbation
    Stochastic Approximation of the Quantum Fisher Information*, Quantum 5,
    567 (2021). <https://quantum-journal.org/papers/q-2021-10-20-567/>
13. F. J. Schreiber, J. Eisert, J. J. Meyer, *Classical surrogates for
    quantum learning models*, Phys. Rev. Lett. 131, 100803 (2023).
    <https://arxiv.org/abs/2206.11740>
14. R. Sweke, E. Recio-Armengol, S. Jerbi, E. Gil-Fuster, B. Fuller,
    J. Eisert, J. J. Meyer, *Potential and limitations of random Fourier
    features for dequantizing quantum machine learning*, Quantum 9, 1640
    (2025) — Sweke and Fuller at IBM Quantum.
    <https://quantum-journal.org/papers/q-2025-02-20-1640/>
15. P. Rodriguez-Grasa, S. Jerbi, M. Sanz, R. Sweke, *Towards Surrogate
    Based Dequantization of Quantum Reinforcement Learning*,
    arXiv:2609.16266 (September 2026) — finite-sample guarantees for
    classical kernelized fitted Q-iteration, in a simplified setting with
    uniformly sampled state-action pairs. <https://arxiv.org/abs/2609.16266>
16. M. Cerezo, M. Larocca, D. García-Martín, N. L. Diaz, P. Braccia,
    E. Fontana, M. S. Rudolph, P. Bermejo, A. Ijaz, S. Thanasilp,
    E. R. Anschuetz, Z. Holmes, *Does provable absence of barren plateaus
    imply classical simulability?*, Nature Communications 16, 7907 (2025).
    <https://arxiv.org/abs/2312.09121>
17. IBM Quantum documentation, *Plans overview*, *Introduction to IBM
    Quantum Compute Service execution modes* and *Run jobs in a session*
    (fetched 2026-10-01).
    <https://quantum.cloud.ibm.com/docs/en/guides/plans-overview>,
    <https://quantum.cloud.ibm.com/docs/en/guides/execution-modes>,
    <https://quantum.cloud.ibm.com/docs/en/guides/run-jobs-session>
18. IBM Quantum documentation, *Processor types*, *Retired cloud QPUs*,
    *View backend details* (supported instructions) and the *IBM Quantum
    Compute Service changelog* (fetched 2026-10-01).
    <https://quantum.cloud.ibm.com/docs/en/guides/processor-types>,
    <https://quantum.cloud.ibm.com/docs/en/guides/retired-qpus>,
    <https://quantum.cloud.ibm.com/docs/en/guides/qpu-information>,
    <https://quantum.cloud.ibm.com/docs/en/guides/changelog-quantum-compute-service>
19. IBM Quantum Learning, *Quantum Machine Learning* course, lesson
    "Introduction to Quantum Machine Learning" (fetched 2026-10-01).
    <https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/introduction>
20. O. Lanes, M. Beji, A. D. Corcoles, C. Dalyac, J. M. Gambetta,
    L. Henriet, A. Javadi-Abhari, A. Kandala, A. Mezzacapo, C. Porter,
    S. Sheldon, J. Watrous, C. Zoufal, A. Dauphin, B. Peropadre, *A
    Framework for Quantum Advantage*, arXiv:2506.20658 (2025) — IBM and
    Pasqal authors. <https://arxiv.org/abs/2506.20658>
21. *Quantum Advantage Tracker*, an open community tracker.
    <https://quantum-advantage-tracker.github.io/>; described in J. Gambetta,
    R. Davis, "Quantum Advantage Tracker: the race to advantage", IBM
    Quantum blog, 23 Feb 2026.
    <https://www.ibm.com/quantum/blog/quantum-advantage-tracker>
22. *IBM Quantum Compute client release notes* (qiskit-ibm-runtime), 0.50.0
    (2026-09-24): client-side `executor_estimator.Estimator` added,
    `EstimatorV2` deprecated; 0.47.0 (2026-05-12): `FakeMiami` added
    (fetched 2026-10-01).
    <https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes>
23. M. G. Bellemare, G. Ostrovski, A. Guez, P. S. Thomas, R. Munos,
    *Increasing the Action Gap: New Operators for Reinforcement Learning*,
    AAAI 2016 — Baird's advantage learning and other gap-increasing
    operators; the `action_gap` trainer option.
    <https://arxiv.org/abs/1512.04860>
