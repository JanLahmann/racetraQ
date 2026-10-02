# The science behind traQmania

What the quantum agent actually is, how it is trained, what runs on real
hardware, and — importantly — what this demo does and does not show. The
[notebooks](../notebooks/) build all of this up from scratch with runnable
code; this page is the condensed reference.

> **October 2026 audit.** A soundness audit on 2026-10-01 found three things
> that change how this page must be read: the 8- and 10-qubit circuits were
> partially blind ("Light cones" below), the DQN's greedy performance is
> unstable after the first lap — most of all on gp ("Training") — and the
> hardware path targeted retired devices ("Hardware"). The code is fixed or
> instrumented for all three. The numbers under "Measured results" predate
> the audit and are being re-measured. Measured since: what device noise
> does to a driver and what helps ("Hardware"), and a classical surrogate of
> the trained circuit ("Honest claims", notebook 07).

## The circuit

A data re-uploading variational quantum circuit (VQC) acting as the Q-function
of a DQN agent. Canonical definition: `traqmania/agents/quantum/circuit.py`
(single source of truth for the numpy fast path, the Qiskit/`EstimatorQNN`
path, and the hardware path).

- **n qubits** (default **4**), **L = 4 re-uploading blocks** on |0…0⟩.
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
  drivers end up with |w| between 31 and 104).
- **Trainable parameters, P = 3·L·n + 8** (**56** at n = 4, **80** at
  n = 6), one flat vector `[λ, θ, w, b]`: input scalings λ (L×n, init π),
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

At the shipped depth L = 4, with the four readouts on qubits 0–3:

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

**Blind spots of the shipped wider circuits.** Feature j sits on qubit j and
the actions read qubits 0–3, so which feature is hidden from which action is
an accident of the feature order:

- `q8`, 7 rays + speed (`quantum_oval_q8`, `quantum_chicane_q8`): Right
  cannot see ray +20°, Straight ray +40°, Left ray +60° — and **Brake cannot
  see speed**.
- `q10`, 9 rays + speed (`quantum_oval_q10`, `quantum_chicane_q10`): Right
  misses rays 0°/+15°/+30°, Straight +15°/+30°/+45°, Left +30°/+45°/+60°,
  Brake rays +45°/+60° and speed.
- `quantum_gp_q10`, 5 rays + speed + four engineered features: Right misses
  ray +60°, speed and curvature ahead; Straight speed, curvature ahead and
  lateral offset; Left curvature ahead, lateral offset and heading error;
  **Brake lateral offset, heading error and corner speed** — the brake
  action decides without the one feature built to say "too fast for this
  corner".
- The 8-qubit engineered-feature observation (3 rays + speed + the four
  features in the order of the `q8` profile's alternative) hides exactly one
  engineered feature from each action (curvature ahead, lateral offset,
  heading error, corner speed for Right, Straight, Left, Brake).

The default 4-qubit circuit and the `q6` profile (plain or with features)
are fully visible.

**What this corrects.** Every 8- and 10-qubit result further down — the
scaling table, the engineered-feature comparison, the 10-qubit gp driver,
the scaled-action and braking-ladder campaigns — was measured with these
partially blind circuits. "More qubits barely help" and "features neither
help nor hurt" were therefore never clean statements about qubits or
features: part of every added input was invisible to part of the policy.
Those conclusions are suspended, not reversed. The re-measurement at
sufficient depth (L = 5 at 8 qubits, L = 6 at 10), with a permuted feature
order as a control, decides what survives.
<!-- RESULTS-PENDING: 8/10-qubit scaling and feature results re-measured at L >= n//2 + 1 plus the permuted-feature control; say here which conclusions survive -->

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
the rule before that was a single 4-episode eval. Either way, every snapshot
behind the current "Measured results" was selected on 4 distinct episodes.

**Greedy performance is not stable — stated plainly.** A policy that laps
at one snapshot eval frequently laps in none of the episodes at the next.
What the audit measured (4 qubits, default recipe, seeds 0, 1 and 42, one
run each): on gp, 86–89 % of the evals after the first lapping one scored
zero laps, and the parameters at the end of training lapped in 0 of 36
greedy episodes at all three seeds. It is not only the hard track: on
chicane the final parameters lapped in 0 of 36 episodes at two of the three
seeds (29 of 36 at the third). Only the oval held on (36 of 36 at all
three). combo was not measured. Best-snapshot selection hides this — it
reports the peak, not the plateau. The shipped weights are such peaks: real
policies, but "the agent learned gp" overstates what the optimizer holds on
to. A stabilisation study over the options below is in progress; no result
is claimed here yet.
<!-- RESULTS-PENDING: stabilisation study (tools/study.py, >= 10 seeds per variant): stability and final-params lapped fraction with CIs per variant, and the recipe adopted as the new default -->

**Trainer options** (`[training]` keys; each defaults to the behaviour that
produced the bundled weights, so default training stays bit-identical):

| Key | Default | What it is for |
|---|---|---|
| `bootstrap_truncation` | `false` | Treat the 60 s time limit as a truncation, not a terminal state: the TD target keeps bootstrapping from the last observation (Pardo et al. [9]). Off-track stays terminal. |
| `loss`, `huber_delta` | `"mse"`, 10 | `"huber"` clips the TD error in the gradient at ±`huber_delta`; Q-values here are of order 100, so a few large errors otherwise dominate a batch. |
| `lr_groups` | — | Separate learning rates per parameter group (quantum: `lam`, `theta`, `head`; MLP: `body`, `head`). Circuit angles are of order 1, the output head grows to 30–100 — one rate may fit neither. |
| `lr_end` | = `lr` | Linear learning-rate anneal over the run. |
| `target_update`, `tau` | `"hard"`, 0.005 | `"soft"` replaces the copy every `target_sync_every` updates by Polyak averaging on every update. |
| `grad_clip` | 0 (off) | Clip the gradient's global L2 norm. |
| `reward_scale` | 1 | Scales rewards as stored in replay (reported returns stay raw), e.g. to shrink Q-values toward the ⟨Z⟩ range. |
| `eval_every`, `eval_episodes` | 50, 12 | The snapshot eval above. |
| `act_noise` | — (off) | `{ attenuation, shots, bias }` (quantum agent): rollouts and snapshot evals pick actions from *noisy* readouts, ⟨Z⟩ → attenuation·⟨Z⟩ + bias + the shot noise of `shots` shots (`agents/quantum/noise.py`). TD targets and gradients stay exact, and the snapshot kept is the one that drives best under the noise. |
| `action_gap` | 0 (off) | Advantage learning (Bellemare et al. [23]): the TD target loses `action_gap` · (max_a Q(s, a) − Q(s, a_taken)), which widens the gap between the best action and the rest by 1 / (1 − `action_gap`) and leaves the best action unchanged. |

(The last two exist for robustness to device noise; what they buy, and what
they cost, is measured under "Hardware".)

From the command line: `train_headless --set training.loss=huber --set
circuit.n_layers=6 …` overrides any config value (an unknown `training.` key
is an error, not a silent baseline run), `--preset none` trains on plain
`[training]` without the track's `[training_presets.<track>]`, and the
resolved recipe is recorded in the weights' `.meta.json`.

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
and a single seed gets none; the report says so itself.

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
  our runs): 8 for the calibration, 20 probes, 1 for the starting point plus 1 for every step the
  guard lets through, and 6 for the fresh loss before and after (3 when no
  step was taken). Deliberate asymmetry: the loss is evaluated on the noisy
  backend, the greedy returns before and after (mean of 12 episodes) on the
  exact simulator.

  What was measured (2026-10-01/02: bundled 4-qubit oval driver, ten
  iterations, 1024 shots, batch 16, the local `fake_miami` patch; a run
  "kept" the policy if it kept at least 80 % of its 12-episode simulator
  return):

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
    and in 13 of 32 in emulation, while in notebook 05's four seeds the
    current code with every safeguard switched off kept it all four times.
  - *It does not make the driver lap on the device.* Notebook 05 puts the
    bundled driver on the device path before and after four default sprints
    (8 episodes each): 1 lap of 8 before, 0 of 8 after each of the four.
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
  device.
- **Why full training stays simulated:** a typical run is ~20 000 gradient
  steps. At param-shift pricing that is ~5 days of pure compute before queue
  time; even SPSA's 2 jobs/iteration makes hardware training a demonstration
  of mechanics, not a competitive training method at this scale.

**What survives the noise today — not the default driver.** Measured on
2026-10-01 with the bundled weights (first lap from a standing start, one
car, 1024 shots; counts are laps completed / attempts):

| Bundled driver | Exact simulator | Shot noise only | `fake_miami`, resilience 0 |
|---|---|---|---|
| oval, 4 qubits (the default) | 7 / 10 | 9 / 30 | 0 / 14 — off track after 24–43 of the ~160 decisions a lap takes |
| chicane, 4 qubits | 10 / 10 | 27 / 30 | 12 / 20 |
| oval, 6 qubits | 10 / 10 | 27 / 30 | 13 / 13 (14.9–15.4 s) |
| gp, 4 qubits | 6 / 10 | 12 / 30 | 2 / 6 |

(Exact and shot-noise columns: fastsim, spawn seeds 0–9, × 3 noise seeds
with each ⟨Z_a⟩ sampled independently from 1024 shots; a re-run that draws
all four readouts jointly from the same 1024 shots gave 9, 29, 29 and 16 of
30. Device column: every logged `python -m traqmania.hardware lap --fake`
run of that day, all from the seed-42 spawn, the simulator's shot noise
unseeded. The columns use different spawns, so compare them row by row, not
cell by cell.) The default oval driver does not complete a lap on the
simulated device, and device errors are not needed to explain it: from that
same seed-42 spawn, where the exact simulator laps, shot noise alone left
1 lap in 20 attempts at 1024 shots and 1 in 20 at 4096. Noise-free it laps
from only 7 of 10 spawn positions. Its action ranking does not survive the
up to ~0.03·|w| that 1024 shots put on every Q-value. TREX (resilience 1)
helps a little and not enough: 2 laps in 11 runs, the others off track
after 26–116 decisions. A different device did not help either (one run
each on `fake_fez` and `fake_manila`: off track after 21 decisions). Other
bundled drivers do better: the 6-qubit oval driver lapped every time, the
chicane driver in 12 of 20 runs and the gp driver in 2 of 6; single runs of
the 8- and 10-qubit oval drivers lapped on `fake_miami` too (one run each —
an observation, not a result). No lap on a physical QPU is reported
in this document. With an account the same code submits to a real device,
and the table says what to expect from it.
<!-- RESULTS-PENDING: this table re-measured for the retrained drivers (more runs per cell, all tracks and sizes); any real-QPU lap -->

**Why it fails, and what helps.** Measured on 2026-10-01/02 with
`agents/quantum/noise.py` (an expectation-noise model: attenuation, bias and
shot noise per readout, fitted to the device with one job) and
`tools/hw_reliability.py` (lap completion of one weights file, emulated and
on the device path). Everything here is the 4-qubit oval; "device" is always
the local `fake_miami` patch (12 CZ), and episodes start from the eval
env's jittered spawns, not from the seed-42 spawn of the table above.

- *What the device does to a readout.* Fitted against the exact simulator on
  random inputs, ⟨Z⟩_device ≈ f · ⟨Z⟩_exact with f = 0.95 (0.945–0.956 over
  25 parameter sets), and every readout has its own slope, between 0.94 and
  0.96. TREX moves f to about 0.98. On top comes the shot noise, whose
  variance matched the binomial formula within 4 %.
- *Why the bundled oval driver falls off.* It decides on knife-edge
  margins. Along its own trajectory the gap between its best and
  second-best Q-value has a median of 0.46, at a head of |w| ≈ 52 — about
  one standard deviation of 1024-shot noise. Even on the exact simulator it
  laps in 27 of 36 episodes, and a random ±0.2 on its head moves that
  anywhere between 5 and 35. In emulation it laps in 10 of 36 with shot
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
  calibration job), TREX — does not rescue the bundled driver: 8 laps in 36
  device episodes over twelve settings in one sweep, 3 in 32 over four
  settings in a second one with other seeds. It does rescue drivers with
  moderate margins. Two default-recipe drivers that lapped 1 of 12 device
  episodes at 1024 raw shots lapped 8 of 8 with the readout rescale at 4096
  shots (3 and 4 of 8 at 4096 shots without it, 8 and 7 of 8 at 16384); a
  third (5 of 12) gained nothing from the rescale (4 of 8 with it at 4096
  shots, 7 of 8 without). The global rescale (`--rescale global`) did
  nothing for the bundled driver (0 of 3 device episodes).
- *Training for it works more often, not every time.* The recipe `--set
  training.action_gap=0.8 --set
  'training.act_noise={attenuation=0.95,shots=1024}'` — advantage learning
  [23] for wider gaps, acting under emulated device noise — was chosen among
  seven on seeds 0–7. On 16 seeds it was not chosen on (8–15 and 100–107;
  400 episodes each), 9 of its drivers still lapped in at least 32 of 36
  episodes under emulated 1024-shot device noise, against 2 of 16 for the
  default recipe. On the device path (seeds 100–107, 12 episodes per driver,
  1024 raw shots) 5 of 8 lapped at least 11 of 12, against 1 of 8. The
  costs: 5 of the 24 seeds trained with it did not learn to lap reliably
  within 400 episodes (the default recipe has such seeds too); on the fresh
  seeds its exact-simulator lap count is no better than the default's (25.9
  against 27.5 of 36); and notebook 05's three further seeds show no
  difference on the device (8, 8 and 5 of 8 against 7, 5 and 8). Which of
  the two levers does the work is unresolved, and nothing is known beyond
  the 4-qubit oval. No bundled driver has been retrained this way yet.
- *The emulation is a screening tool.* Over 25 drivers, the per-readout
  emulation's lap counts correlated 0.96 with 12 device episodes each, but
  for 2 drivers it was off by 4 or more laps of 12, and it is optimistic
  about the rescale, which inverts exactly the model the emulation is built
  from (bundled driver, readout rescale at 16384 shots: 21 of 36 emulated,
  0 of 8 on the device path). Compare recipes with it; judge a single
  driver on the device path.

`aer_noisy`, the noisy training/eval backend of the `EstimatorQNN` path
(`agents/quantum/qnn.py`), uses the same local twin of the same fake device.

## Measured results

> **Read this first (2026-10-01).** Every number in this section predates
> the October 2026 audit. Snapshot selection at the time used 4 distinct
> greedy eval episodes (counted three times as "12"), headline numbers rest
> on 1–3 training seeds, and the 8- and 10-qubit circuits were partially
> blind (see "Light cones"). All bundled drivers are being retrained and
> every table re-measured with the multi-seed harness (`tools/study.py`).
> Until then, read what follows as the project's pre-audit record, with the
> corrections marked inline.

<!-- RESULTS-PENDING: replace this note and every number/table in "Measured results" with the re-measured multi-seed values (IQM + 95% CI, seeds listed) -->

**Physics v2 (2026-07).** All numbers in this section are measured under the
current physics constants (`accel` 11, `brake` 16, `v_max` 25 — faster
straights, unchanged hairpin discipline). The change was made so speed
visibly *varies*: the model-based hero driver's max/min speed ratio is 1.81
on gp and 1.79 on combo (≈1.5 under v1), while the oval/chicane corners are
gentle enough to stay flat-out at any of these speeds. It was tuned as far
as the 4-qubit circuit could follow — `v_max` 28–30 variants left gp and
combo unlearnable at 4 qubits even when warm-started — and it still has a
real cost, reported below: the hard tracks got harder for the tiny circuit.
Unless noted, headline numbers are seed 42; the v2 seed spreads (seeds
42/0/1 per variant, quoted inline below) were re-run in the overnight
campaign of 2026-07-11; histories in `data/histories/`.

Apple Silicon laptop, `fastsim`, greedy best-snapshot eval:

| Track | Quantum first clean lap | Quantum best lap (greedy) | Classical MLP best lap |
|---|---|---|---|
| oval | ~18 s wall-clock training (ep ≈ 381) | 14.1 s (spread 13.5–14.1) | 13.2 s |
| chicane | ~25 s (ep ≈ 450) | 12.5 s (spread 12.5–14.8) | 13.5 s |
| gp | ~123 s (ep ≈ 1794) | 23.2 s | 20.3 s |
| combo | ~79 s (ep ≈ 1034, warm-started) | 27.5 s (fresh s0: 30.7 s) | 30.8 s |

The gp story improved after an overnight recipe sweep (8 epsilon-decay ×
gamma combinations plus learning-rate, target-sync, curriculum and
warm-start probes). The winning recipe — epsilon decayed over 2000 of 3000
episodes, gamma 0.99, now the shipped `[training_presets.gp]` — moves the
best snapshot from an early high-epsilon fluke (episode ≈ 250 under the old
recipe) to a genuinely learned late-training policy (episode 1950,
epsilon 0.07): greedy eval 7 laps at 23.2 s, and over a 36-episode
evaluation it laps in a mean of 24.0 s (best 20.4 s) with the same 7/36
no-lap episode rate as the old 27.8 s driver. It is also the first v2 gp
recipe that is seed-robust: all three seeds lap (best-snapshot 23.2 /
31.0 / 24.8 s at seeds 42/0/1) where the old recipe lapped at one seed of
three. That closes most, not all, of the gap to the MLP's 20.3 s (under v1
the two were at parity). One honest
regression stands: **combo** — fresh 4-qubit training never laps under v2;
every lapping combo driver is a warm migration of the v1 weights (seeds
42/0/1/7 reach 27.5 / 30.7 / 37.7 / 33.6 s snapshots — the shipped driver
is the seed-42 run), still ahead of the fresh-trained MLP (30.8 s).

Warm-start live demo: from the bundled pre-first-lap checkpoints the quantum
agent reaches its first clean lap in **~2.2 s** of training on the oval,
**~2.7 s** on the chicane, and **~9 s** on combo (seed-0 verification runs).
gp's warm training improved with the new checkpoint (episode 1100 of the
winning recipe) but stays honestly a coin flip: the shipped 900-episode
schedule laps on 2 of 3 seeds tried (~20–40 s wall); when it misses it
misses outright, and every shorter or hotter schedule probed did worse —
gp's braking discovery remains exploration-luck-dependent even
warm-started, which the config notes honestly. One double-DQN update:
**~3.4 ms** fastsim/adjoint vs **~20.5 s** `EstimatorQNN`/param-shift
(timed before the October 2026 stack upgrade; see "Training").

### Scaling the qubit count

The circuit generalizes over `n_qubits` (config profiles `q6`, `q8`, `q10`;
the default stays 4 and is bit-identical to the pre-scaling stack, pinned by
a regression test). Extra qubits widen the *feature* register — by default
n − 1 lidar rays evenly spaced over [−60°, +60°] plus speed, one feature per
qubit — while the 4 actions and the Z_0…Z_3 readout stay fixed. Trained
weights now ship for **oval and chicane at every size** (`quantum_<track>_q6/
q8/q10.npz`), plus a pre-first-lap oval warm-start checkpoint and evolution
stages at q6, so the live Qubits selector works out of the box on those
tracks. Measured on the oval (greedy best-snapshot eval, physics v2; the
headline row is the bundled driver's seed — 42 everywhere except q10,
whose bundled driver is the seed-7 run):

| | 4 qubits | 6 qubits | 8 qubits | 10 qubits |
|---|---|---|---|---|
| Lidar rays (plain profile) | 3 | 5 | 7 | 9 |
| Trainable parameters | 56 | 80 | 104 | 128 |
| First clean lap (episode) | ≈ 381 | ≈ 392 | ≈ 202 | ≈ 358 |
| Best greedy lap (bundled driver) | 14.1 s | 13.3 s | 13.3 s | **12.0 s** |
| Spread over 3 seeds | 13.5–14.1 s | 12.9–13.3 s | 12.5–13.3 s | 12.0–14.2 s |
| ms per greedy decision (fastsim) | < 1 | 0.6–1 | 1.2–2 | 4.6–8.7 |

The honest reading: sample efficiency per *episode* stays roughly flat from
4 to 10 qubits (first clean lap between episode ~200 and ~515 across all
sizes and both tracks) — a real scaling data point, not a success story.
Under v2 the wider registers still trend faster on the easy tracks, but the
seed spreads overlap heavily: the q10 driver's 12.0 s headline is the best
of three seeds (its spread 12.0–14.2 s is the *widest* of any size), so
"more qubits = faster laps" is a tendency, not a clean ordering. Chicane
tells the same story (q6 12.6–13.4 s, q8 12.2–13.1 s, q10 13.2–13.6 s
across seeds). What grows reliably is per-decision compute, since the
statevector goes 16 → 1024 amplitudes. The fastsim/`EstimatorQNN` parity and gradient checks
run at 6 qubits too (forward agreement ≤ 1e-9; all 80 gradients verified
against finite differences). *Audit correction:* at 8 and 10 qubits every
action in this table decided without 1 and 3 of its inputs respectively
("Light cones" above), so the flat scaling is a statement about partially
blind circuits. The hard tracks above 4 qubits went through two campaigns
with opposite outcomes, and the difference was the recipe, not the circuit.
The first campaign trained gp at q6 (plain and feature observations), q8
and q10 (features) plus combo at q6/q8/q10 — all with the *old* fast
epsilon decay (1200 of 2000 episodes) — and every run finished with zero
greedy laps (a few lapped during exploration only). A follow-up campaign
re-ran the same configurations with the swept winning recipe (decay over
2000 of 3000 episodes), and gp now laps greedily at **every** qubit count.
36-episode reliability evals (12 episodes × 3 env seeds, same protocol as
the 4-qubit numbers): q6-feat laps in 27/36 episodes but slowly
(mean 41.5 s), q8-feat 18/36 (mean 40.2 s), and **q10-feat 25/36 with
mean 22.3 s / best 20.0 s — pace-competitive with the bundled 4-qubit
driver's 29/36 / mean 23.9 s / best 20.4 s** (one training seed vs the
4-qubit driver's three, so treat it as an existence proof, not a spread).
q6-plain is the cautionary tale: its snapshot posted an 18.1 s lap on the
4-episode training-time eval, but the full check shows 5/36, and seeds 0/1
give 1 and 0 greedy laps — a fluke, reported as such. Combo moved too:
with the slow decay, *fresh* 4-qubit combo training laps for the first
time under v2 (23/36, mean 38.1 s — the warm-migrated bundled driver
stays much faster at 21/36, mean 28.7 s), while combo above 4 qubits
still never laps greedily (fresh slow-decay and warm-from-chicane both
fail at q6/q10; cross-track warm starts have never worked in this
project). Net verdict at the time: the braking problem was a
recipe/exploration problem, not a capacity one — with the right decay,
capacity is fine all the way to 10 qubits. *Audit correction:* that
campaign could not see the light cone. The q10 gp driver's Brake readout
cannot see corner speed, lateral offset or heading error, so "capacity is
fine" was never tested with a fully sighted circuit. The q10 gp winner ships as
`quantum_gp_q10.npz`: weight resolution is observation-aware — each weights
file's `.meta.json` records the `[observation]` it was trained with
(`runtime.weights_observation`), and the server overlays it on the profile
when that driver is active, so a feature-observation driver coexists with
the plain-rays oval/chicane weights at the same qubit count. The q6/q8 gp
candidates stay unbundled (they lap, but ~17 s off the pace), as does
every combo candidate above 4 qubits.

### Observation engineering

The observation registry (`[observation] features` in the config;
`traqmania/env/racing_env.py`) lets any qubit count trade lidar rays for
engineered scalars, one feature per qubit, all normalized to [0, 1]:

- `rays` — lidar distances (the default), `speed` — car speed;
- `curvature_ahead` — max centerline |κ| over a lookahead window;
- `lateral_offset` — signed centerline offset / half-width;
- `heading_error` — wrapped angle to the track tangent / π;
- `corner_speed_ratio` — v / v_safe(R) with R = 1/max(|κ_ahead|, 1e-6) and
  v_safe = √(max(0, 2·k_steer·v_turn·R − v_turn²)), derived from the car
  model's steering kinematics (the same “why you must brake for hairpins”
  analysis as notebook 01; derivation in the `racing_env.py` docstring) —
  “am I too fast for the corner coming up?”

Measured variants on the oval (same DQN hyperparameters as the plain runs;
physics v2, seed 42, greedy best-snapshot eval):

- **`q6feat`** (80 params: 3 rays + speed + curvature_ahead +
  corner_speed_ratio): 13.2 s at seed 42; spread 13.0–13.6 s vs plain q6's
  12.9–13.3 s — indistinguishable.
- **`q8feat`** (104: adds lateral_offset + heading_error): **12.5 s** at
  seed 42, which looked like the clearest feature win at v2 — but the seed
  spread (12.5–13.8 s) overlaps plain q8's (12.5–13.3 s), so it does not
  survive as a claim.
- **`q10feat`** (128: 5 rays + speed + all four features): 13.5 s at seed
  42, spread 12.5–13.5 s vs plain q10's 12.0–14.2 s — overlapping again;
  neither the v1 "features stop helping at ten qubits" reading nor its
  opposite survives the seeds.

The **asymmetry** we observed under v1 is now, with three seeds per variant
under v2, best described as *gone at this sample size*: engineered features
neither reliably help the VQC (spreads overlap at every width) nor reliably
hurt the matched MLP baselines (92/108/124 params at the q6/q8/q10
observation widths, rays-only greedy laps 12.1–12.5 / 12.2–12.3 /
11.9–12.1 s over seeds, ~0.01 ms/decision; the q6-width feature MLP runs
12.9–13.0 s — slightly slower than its rays-only twin, the one remnant of
the v1 finding). The v1 asymmetry rested on single seeds; the v2 spreads
absorb it. What does survive is the parity headline: at the q10 observation
the MLP's 11.9 s vs quantum's 12.0 s is the closest the two stacks have
ever been. Notebook 06 has the learning curves and the
full comparison (v1-physics campaign; re-execution under v2 refreshes the
plotted histories).

### One driver, every track (cross-track generalization)

Because the observation is egocentric (lidar rays + speed, no absolute
position), a trained policy is not tied to its training track. Measured
zero-shot under v2 (greedy, no fine-tuning): the gp-trained 4-qubit
specialist laps the oval (13.5 s, 24 laps in 6 episodes) and even combo
(40.6 s) — though under v2 it does not transfer to the chicane (0/6). The bundled
**`quantum_universal.npz`** is now a *warm migration*: the v1 universal
driver fine-tuned on the four-track round-robin under the new physics
(3000 episodes, seed 42). It laps everything — oval 13.0 s, chicane 13.3 s,
gp 30.3 s, combo 52.3 s — and 10/10 unseen generated tracks at difficulty
0.5 (best 23.0 s). Seed-honesty, two layers of it: *fresh* multi-track
training under v2 does not produce a fully universal driver — four runs
tried (seeds 42, 0, 1, 7, one of them 5000 episodes); the best laps
oval/chicane and 9–10/10 generated tracks but fails gp and combo outright.
And the warm migration itself is seed-sensitive: re-running the same v1 →
v2 fine-tune at seeds 0 and 1 *collapses to an oval specialist* (24/24
greedy oval laps at 12.1 s, zero laps on chicane/gp/combo/generated) —
only the seed-42 run kept all-track coverage, and that is the one that
ships. Knowledge transfer across a physics change preserved universality
once, not reproducibly — a nice RL lesson in itself. All learning curves
are in `data/histories/` (`uni2_*`).

### The ceiling: a model-based reference driver

To know how good the learned drivers actually are, the expert demo includes a
**hero** driver that is not learned at all: it builds a family of candidate
racing lines from the track geometry (curve-shortening flow, blended wide
through slow corners), derives brake/accelerate-feasible speed profiles from
the `[physics]` constants, *simulates itself* on each candidate with the real
car physics, and drives the fastest provably crash-free combination with
continuous steering (pure pursuit). Because everything derives from the
`[physics]` constants, the hero adapted to the v2 physics with no retraining.
Measured: oval 12.1 s, chicane 12.1 s, gp 16.4 s, combo 19.0 s — ahead of
every learned agent wherever braking and line choice matter. Two useful
facts follow. First, the RL agents' gap to this ceiling is mostly their
action set: they steer with 4 bang-bang actions at 10 Hz, the hero steers
continuously — on gp that is worth ~7 s per lap for the 4-qubit
circuit (16.4 vs 23.2) and ~1.4 s for the strongest MLP. Second, capacity
alone is not the constraint at baseline scale, but the expert menu's **pro**
driver shows what capacity plus rich sensing buys: the same double-DQN
recipe as every agent in the demo, with a wide MLP (hidden 128, 2,436
params) and the 14-feature observation (9 rays + speed + 4 track scalars),
trained fresh on all four tracks at once for 5000 episodes (`mlp_pro.npz`,
seed 0; the 3000-episode recipe that sufficed under v1 fails gp under v2).
Measured: 11.9 / 12.4 / 17.8 / 20.4 s and 10/10 generated tracks — the
strongest learned driver we have, and unlike the quantum universal driver
it is seed-robust: an independent seed-1 run of the same recipe lands at
11.9 / 12.5 / 17.3 / 20.0 s with 10/10 generated tracks. It even edges the hero on the flat-out
oval by 0.2 s (that track is pure path geometry, and the bang-bang zigzag
traces a fractionally shorter path than smooth line-tracking — documented as
a near-tie, not chased further), while trailing by 1–1.4 s everywhere
control quality matters. The remaining gap is dominated by the 4-action
interface, not model size. Hero and pro laps are excluded from ghost
records — records stay with the standard demo agents (and humans).

### Making qubits matter: scaled actions, longer sensing, a pace objective

Through ten qubits, the scaling story above kept its uncomfortable shape:
more qubits widen the *observation*, yet lap times barely move. Analyzing
the trained drivers against the hero ceiling identified four concrete
bottlenecks — none of them circuit capacity — and each now has a mechanism
in the stack (measured single-seed results from the follow-up campaign are
inline below; the standard eval is 36 greedy episodes). *Audit correction:*
there was a fifth, inside the circuit, that this analysis missed — at L = 4
the 8- and 10-qubit circuits hide inputs from actions ("Light cones"). Every
measurement in this subsection was taken with such a circuit, including the
6- and 8-action readouts, each of which also misses 1 (8 qubits) or 3
(10 qubits) features:

1. **The action set caps the racing line.** Every driver picked from the
   same 4 bang-bang actions, and crucially could not steer while braking, so
   hairpin entries alternate brake/steer decisions at 10 Hz. The readout now
   scales with the register: `[circuit] n_actions` reads Q_a = ⟨Z_a⟩ off the
   first *k* qubits — 6 actions add trail braking (full steer + brake), 8
   add half-steer (`traqmania/agents/base.py`; prefix-compatible, so the
   default stays bit-identical and every existing weights file keeps its
   meaning). A pure coast action was evaluated and rejected: at drag 0.35
   it is dominated by brake/throttle nearly everywhere. Weights record
   their action count in the `.meta.json` sidecar and the server adopts it
   per driver, exactly like the recorded observation.
   *Measured (campaign E, seed 42):* the mechanism works, the recipe
   doesn't yet. During exploration the 8-action q10 driver drove **18.9 s**
   on gp — the fastest quantum lap ever recorded in this project — but no
   scaled-action lane converged to greedy lapping under the campaign-D
   recipe (q8 6-action 0/36 where its 4-action same-observation control
   lapped 29/36; q10 8-action 0/36, partially rescued to 7/36 @ best
   20.0 s by a pace fine-tune). On the flat-out oval, 6 actions lap
   reliably (36/36) but *slower* (14.8 s vs ~13 s plain q8): extra actions
   only have value where braking matters, and everywhere they enlarge the
   exploration problem. Scaled action sets need their own recipe — the
   promising untried route is warm-starting the widened head from a
   trained 4-action policy (the prefix property makes the padding
   well-defined).
2. **The sensing horizon was shorter than the braking distance.** Slowing
   from v_max 25 to hairpin speed 9 takes ≈(25²−9²)/(2·16) ≈ 17 m, but
   `curvature_ahead` looked only 15 m ahead — optimal braking was literally
   invisible. Curvature kinds now take a per-feature horizon suffix
   (`"curvature_ahead:30"`, `"curvature_ahead:50"`), so a 10-qubit register
   can afford a *braking ladder* of near/mid/far curvature instead of more
   rays.
   *Measured:* honest null-to-negative so far. The ladder observation
   (rays + speed + κ@15/30/50 + corner-speed) is learnable at q8 (29/36,
   but slow at 36.4 s mean) and converged *worse* at q10 (1/36) than
   campaign D's mixed-feature observation (25/36) under the same recipe —
   seeing farther did not make braking easier to learn at this seed.
3. **The reward never asked for speed.** Progress reward pays the same
   total per lap however slow the lap; discounting was the only pace
   pressure. `[reward] time_penalty` (default 0 — bit-identical) charges
   each decision, so a lap's net value rises ~5 per second saved at the
   `[training_pace]` setting. The intended use is two-phase:
   `train_headless --pace --init <lapping snapshot>` runs a low-epsilon
   fine-tune whose objective is lap time, on top of a reliability-trained
   policy.
   *Measured:* **the clear win of the campaign.** A 600-episode pace
   fine-tune of the bundled 10-qubit gp driver moved it from 25/36 @
   22.3 s mean / 20.0 s best to 14/36 @ **20.2 s mean / 17.5 s best** —
   the first greedy quantum gp lap under 18 s, closing a third of the gap
   to the hero's 16.4 s, at a real reliability cost (the bundled driver
   is unchanged; the pace variant lives in the campaign archive). The
   same fine-tune on the 4-qubit gp driver went the other way: **36/36**
   lapped (perfect, up from 30/36) but 28.1 s mean — see (4) for why.
4. **Snapshot selection rewarded lucky laps.** The shipped driver used to
   be the best *4-episode* greedy eval by (laps, best-lap) — the recipe that
   crowned an 18.1 s gp headline which a 36-episode recheck put at 5/36
   lapped episodes. Selection was changed to 12 greedy episodes ranked by
   (lapped episodes, mean lap): reliability first, average pace second, one
   lucky lap never decides. *Audit correction:* as implemented for this
   campaign the 12 were 4 distinct episodes evaluated three times; since
   October 2026 they are 12 distinct ones (see "Training").
   *Measured:* selection did what it promises — every campaign-E lane that
   lapped shipped a snapshot whose 36-episode recheck matches its
   (nominally) 12-episode eval (no more mirages). It also exposed a design tension:
   *inside a pace fine-tune*, reliability-first ranking keeps the most
   reliable snapshot even when a slightly less reliable one is seconds
   faster (that is how the 4-qubit pace run "won" 36/36 at 28.1 s). A
   pace-phase selection rule — best mean lap above a reliability floor —
   is the natural follow-up.

The model-based hero (gp 16.4 s) remains the pace target; the bundled
4-qubit gp driver stands at 23.7 s mean under the 36-episode protocol, and
the pace-tuned 10-qubit driver has touched 17.5 s. All campaign-E numbers
are one seed (42) — the same caveat as every headline in this document
until a seed spread says otherwise.

## Honest claims

Being honest matters more than being exciting:

- **Parity at tiny scale is an observation, not yet a result.** A
  56-parameter VQC and a 76-parameter MLP both learn the task, with broadly
  similar sample efficiency and comparable lap times in the runs so far.
  That a VQC *can* do this is the point, echoing Chen et al. [1] and Skolik
  et al. [2]. But those comparisons rest on 1–3 seeds and best-snapshot
  numbers. By the standard the field now applies — many seeds, interval
  estimates, a statistical definition of outperformance [6, 7] — and with
  the instability described under "Training", "parity" is a claim we still
  have to earn. It is being re-measured with `tools/study.py`. The same
  literature is why the classical baseline matters: in a large benchmark of
  quantum classifiers, out-of-the-box classical models won overall [8].
  <!-- RESULTS-PENDING: quantum vs MLP per track with IQM + CI and probability of improvement, >= 10 seeds each -->
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
  surrogates for the bundled drivers (`agents/quantum/surrogate.py`), and
  two of its measurements belong here. The encoding of the 4-qubit, 4-block
  circuit allows 531,441 frequency vectors per readout, but only 45,072 of
  them can carry a non-zero coefficient, for trained and for random angles
  alike. And kernel surrogates fitted to 1,000 states sampled while the
  quantum driver drove lapped within two episodes of 36 of that driver in
  every comparison made (oval and gp at 4 qubits, oval at 6; two to five
  fits each) — but on the oval so did a kernel built from the *untrained*
  frequencies and a generic Gaussian kernel (27 of 36 each, like the
  circuit). A surrogate that laps is therefore weak evidence for the Fourier
  picture. The evidence is in the function: least squares on the predicted
  frequencies reproduces small circuits to better than 10⁻¹³, where
  frequencies scaled by 1.3 leave errors of 0.05 to 1.7 in four of the five
  circuits tried (the fifth, with a single feature, fits either way); and
  fitted on uniformly sampled inputs, the true-frequency kernel reaches a
  normalized error of 0.004 where the Gaussian kernel reaches 0.054. Only
  inference is dequantized there — fitted Q-iteration on the surrogate
  [15] was not run. IBM's own teaching material makes the same point about
  expectations: "It is not realistic to expect a quantum speed-up for
  machine learning tasks that classical computers already do quite well"
  [19]. An advantage claim needs an output
  whose correctness can be validated and a demonstrated separation from the
  best classical methods [20]; the open community tracker for such claims
  follows observable estimation, variational problems and classically
  verifiable problems [21]. A racing toy is none of these.
- **We do not claim engineered features prove anything.** Under v1 physics,
  hand-engineered observations appeared to make the VQC faster and the
  matched MLP slower — an intriguing asymmetry. The v2 three-seed spreads
  dissolved it. We report the dissolution as prominently as we reported the
  observation: single-seed asymmetries in small RL benchmarks usually are
  noise (notebook 06 walks through both). The audit adds a second reason for
  caution: at 8 and 10 qubits the circuit could not show every feature to
  every action, so neither the asymmetry nor its absence was tested fairly.
- **Watch the denominators.** “Similar sample efficiency” is per *episode*;
  per *second* the MLP trains far faster (cheaper gradients). Parameter count
  is an imperfect fairness measure — expressivity per parameter differs, and
  at 6 qubits and up a growing share of the circuit's parameters is dead
  (12 of 72 at 6 qubits, 46 of 120 at 10).
- **Noise robustness: what is known, what is open.** It is not an untouched
  question — Skolik et al. [10] studied shot noise, coherent and incoherent
  errors in the training and evaluation of variational RL agents. What this
  testbed adds is narrow and concrete: a policy trained on an exact
  simulator, with an output head that multiplies ⟨Z⟩ by 30–100, driven
  closed-loop where one flipped argmax can end the episode. For that setting
  our own measurements are sobering: the default 4-qubit oval driver laps
  in 9 of 30 attempts under 1024-shot sampling and in none of 14 on a device
  noise model, while the 6-qubit oval driver laps in 27 of 30 and 13 of 13
  (table under "Hardware"). An earlier version of this page said the
  trained policy "laps at reference pace" under shot noise; that does not
  hold for the current default driver. The
  question this left open — whether training with noise in the loop, or
  rescaling the readout, makes the *ranking* of the four Q-values robust at
  the shot counts and error rates of today's devices — has first answers
  ("Hardware", "Why it fails, and what helps"). More shots and a per-readout
  rescale restore drivers with moderate margins, but not the bundled
  knife-edge one. Training for wider action gaps under acting noise gave an
  oval driver that laps on the simulated device in 5 of 8 fresh seeds,
  against 1 of 8 for the default recipe — and did not learn to lap in about
  one seed in five. That is one track, four qubits and one simulated
  device. Still open: other tracks and sizes, which of the two training
  levers matters, and any physical QPU.
  <!-- RESULTS-PENDING: noise survival of the retrained (and noise-robustly trained) bundled drivers; real-device lap if one is run -->

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
