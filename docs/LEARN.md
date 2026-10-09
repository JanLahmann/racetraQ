# Learning with racetraQ

racetraQ is a small, complete quantum reinforcement learning (QRL) system: a
race car, a 4-qubit circuit that learns to drive it, a classical network
trained the same way, and the measurements that compare them. This page
suggests what to read, in which order, depending on where you start from.
Pick one of the five tracks below. Each lists the material in order, the
time it takes, what you should know beforehand, and where you can stop.

Terms in **bold** are in the [glossary](GLOSSARY.md). Teachers: the
[workshop plans](TEACHING.md) build on these tracks.

| Track | You are | Time | You need |
|---|---|---|---|
| [1. Curious visitor](#1-curious-visitor) | anyone with a browser | 5 min | nothing |
| [2. Student](#2-student) | school or first-year university | 1–2 h | a little Python, sine and cosine |
| [3. ML practitioner, new to quantum](#3-ml-practitioner-new-to-quantum) | you know DQN and backprop | 2–3 h | Python, numpy, linear algebra |
| [4. Physicist, new to ML](#4-physicist-new-to-ml) | you know qubits and Pauli operators | 2–3 h | Python, numpy |
| [5. Researcher](#5-researcher) | you work on QML or RL | half a day | the literature of your field |

How to run the notebooks: every one has a **Launch** badge in the
[README](../README.md#notebooks) that opens it on Binder (QuBins images,
nothing to install). Locally: `pip install -e ".[notebooks]"`, then
`jupyter lab`. The README's notebook table lists each notebook's level and
how long it takes to read and to run.

## 1. Curious visitor

**Time:** 5 minutes. **Prerequisites:** none.

1. Open [racetraq.org](https://racetraq.org/) and watch the quantum driver
   for a lap. Pause it: the side panel shows one decision — the sensors,
   the circuit, the four qubit readings and the four scores. (2 min)
2. Press **Race** and try to beat it. (2 min)
3. Read "What is this?" and "What did we find?" in
   [racetraQ in plain words](EXPLAINER.md). (1–2 min)

**After this you can** say what the car sees (three distance rays and its
speed), where the qubits are (they turn the sensor readings into one score
per move), and the honest result: a classical network of similar size
learns faster and more reliably here. No quantum advantage is claimed.

**Going further:** track 2.

## 2. Student

**Time:** 1–2 hours. **Prerequisites:** basic Python (variables, functions,
loops), numpy arrays, sine and cosine. No quantum physics and no machine
learning needed.

1. [racetraQ in plain words](EXPLAINER.md), all of it. (10 min)
2. [00 — Qubits for drivers](../notebooks/00_qubits_for_drivers.ipynb): one
   qubit, $\langle Z\rangle = P(0) - P(1)$, shots, RZ, CZ, and one block of
   the circuit built by hand. (15 min)
3. [01 — The racing environment](../notebooks/01_the_racing_env.ipynb): the
   game, the physics, and a hand-written driver; do exercises 2 and 3 (make
   it brake for the hairpin). (20 min)
4. [02 — Q-learning from scratch](../notebooks/02_q_learning_from_scratch.ipynb):
   a Q-table, a five-minute neural network, then deep Q-learning that laps
   the oval in seconds. Do exercise 1 (Bellman by hand). (30 min)
5. [03 — Quantum circuits as Q-functions](../notebooks/03_quantum_circuits_as_q_functions.ipynb)
   up to and including "A teaching point: the provably dead parameters".
   (20 min)
6. Optional: in the full demo, train your own driver in the **Studio**, or
   race the circuit at [racetraq.org](https://racetraq.org/). (10 min)

Answer the checkpoint questions at the end of each notebook before moving
on.

**After this you can** explain what a Q-value is and how Q-learning learns
one, say what a neuron, a loss and a learning rate are, compute
$\langle Z\rangle$ for a rotated qubit, build one block of the racetraQ
circuit by hand, and say why some of its angles can never learn.

**Going further:** the rest of notebook 03, then notebook 04 Part A and Part B.

## 3. ML practitioner, new to quantum

**Time:** 2–3 hours. **Prerequisites:** DQN (replay buffer, target network,
ε-greedy), backpropagation, Adam; numpy; linear algebra. Complex numbers
help.

1. [00 — Qubits for drivers](../notebooks/00_qubits_for_drivers.ipynb),
   quickly: it is the minimum quantum mechanics you need, in numpy.
   (15 min)
2. [Classical and quantum, side by side](#classical-and-quantum-side-by-side)
   below: map what you know onto the circuit. (5 min)
3. [02](../notebooks/02_q_learning_from_scratch.ipynb), skimmed: the trainer
   and the four-method Q-function protocol that both models implement.
   (10 min)
4. [03 — Quantum circuits as Q-functions](../notebooks/03_quantum_circuits_as_q_functions.ipynb),
   all of it, with its exercises: re-uploading as a Fourier series, two
   implementations checked against each other, light cones and dead
   parameters, adjoint against parameter shift, gradient variance. (45 min)
5. [04 — Training the quantum driver](../notebooks/04_training_the_quantum_driver.ipynb),
   Parts A and B and the closing section: the same DQN on both models, over
   8–10 seeds, with interval statistics. (40 min)
6. [07 — Light cones and classical surrogates](../notebooks/07_light_cones_and_classical_surrogates.ipynb),
   Part B: the trained circuit as an explicit Fourier series, and a classical
   model that drives like it. (30 min)
7. [SCIENCE.md](SCIENCE.md), "Honest claims". (10 min)

**After this you can** translate every part of the quantum model into the
vocabulary of neural networks (and say where the analogy breaks), explain
why training runs on a simulator with adjoint gradients, read a light-cone
map, and judge the multi-seed comparison the project reports.

**Going further:** notebook 05 (shots, device noise, training for the
device) and notebook 06 (wider circuits).

## 4. Physicist, new to ML

**Time:** 2–3 hours. **Prerequisites:** qubits, Pauli operators,
expectation values, the Heisenberg picture; Python and numpy. No machine
learning needed.

1. [01 — The racing environment](../notebooks/01_the_racing_env.ipynb): the
   Markov decision process the agent faces. (20 min)
2. [02 — Q-learning from scratch](../notebooks/02_q_learning_from_scratch.ipynb),
   all of it, with its exercises: Bellman equation, a Q-table, the neural
   network primer (loss, gradient descent, backprop, learning rate), DQN and
   its stabilisers. (40 min)
3. [03](../notebooks/03_quantum_circuits_as_q_functions.ipynb), focusing on
   the light-cone sections (they use the Heisenberg picture you know) and
   "Trainability". (30 min)
4. [04 — Training the quantum driver](../notebooks/04_training_the_quantum_driver.ipynb):
   why one training run is an anecdote, the interquartile mean, bootstrap
   intervals, the probability of improvement, best snapshot against final
   parameters. Do all three exercises. (45 min)
5. [05 — Real quantum hardware](../notebooks/05_real_quantum_hardware.ipynb),
   sections 1–4: transpilation, device noise and what it does to action
   choices. (30 min)

**After this you can** explain DQN and why it needs a replay buffer and a
target network, what a learning rate and a loss do, why reinforcement
learning results need many seeds, and how the project's statistics decide
whether a difference is supported.

**Going further:** [REPORT.md](REPORT.md), section 2 (evaluation protocol).

## 5. Researcher

**Time:** half a day and more. **Prerequisites:** variational circuits,
data re-uploading, barren plateaus, dequantization; deep RL evaluation
practice (Agarwal et al. 2021).

1. [REPORT.md](REPORT.md): abstract, §2 evaluation protocol, §3 light cone,
   §4 results, §5 honest assessment. (60 min)
2. [SCIENCE.md](SCIENCE.md), "Honest claims" and the references. (20 min)
3. Notebooks [06](../notebooks/06_scaling_and_features.ipynb) (qubits and
   depth over many seeds) and
   [07](../notebooks/07_light_cones_and_classical_surrogates.ipynb) (light
   cones, Fourier spectrum, classical surrogates). (90 min)
4. The data: per-seed summaries in [`data/studies/`](../data/studies),
   bundled-driver records in [`data/records.json`](../data/records.json),
   the one physical-QPU lap in [`data/qpu/`](../data/qpu). Recompute one
   table (notebook 04's helpers do it in a few lines). (30 min)
5. Run a study of your own with `tools/study.py` (see the README,
   "Developer quick reference") and compare it with the committed one.
   (hours)
6. [REPORT.md](REPORT.md), §8 open questions. (10 min)

**After this you can** reproduce the reported tables from committed data,
run and report a new recipe variant under the same protocol, and say which
claims are supported at which number of seeds.

## Classical and quantum, side by side

The two drivers solve the same task with the same trainer, the same
observation and the same four actions. Only the Q-function differs.

| | Classical MLP | Quantum circuit (default, 4 qubits) |
|---|---|---|
| Building block | a **layer**: 4 inputs → 8 tanh neurons → 4 outputs | a **block**: encode each input as $RY(\lambda s)$, rotate with $RY(\theta)\,RZ(\theta')$, entangle with a CZ ring; repeated 4 times |
| What is learned | **weights** and biases | rotation **angles** $\theta$, input scales $\lambda$, and an output weight and bias per action |
| Parameters | 76 | 56, of which 4 (the final RZ angles) can never change an output |
| Inputs enter | once, at the first layer | in every block (data re-uploading) |
| Nonlinearity | the **activation** $\tanh$, values in $(-1, 1)$ | the encoding: outputs are sums of sines and cosines of the inputs (a truncated Fourier series); the circuit itself is linear in the quantum state |
| Output per action | a linear output neuron | the readout $\langle Z_a\rangle \in [-1, 1]$, scaled by $w_a$ and shifted by $b_a$ |
| Who sees which input | every output sees every input after one layer | the **light cone**: at 4 qubits every action sees every input from 3 blocks on; at 8 or 10 qubits 4 blocks are too few |
| Gradient | **backpropagation**: one forward and one backward pass | on a simulator, the **adjoint method** (one forward and one backward sweep of the statevector); on hardware, the **parameter-shift rule** (two circuit runs per parameter) |
| Noise in training | **minibatch noise**: each update sees 32 random transitions from the replay buffer | the same minibatch noise; training uses exact expectation values. On a device each $\langle Z\rangle$ would also carry **shot noise**, about $\pm 0.03$ at 1024 shots |
| Cost of one decision | microseconds (about 0.003 ms in notebook 06's stored run) | exact simulator: about 0.3 ms (about 100 times the MLP); simulated IBM device: about 0.1 s; the physical QPU lap: 141 decisions in 27 minutes of wall clock, about 280 QPU-seconds |
| Cost of one training update (batch 32) | about 0.01 ms for the gradient (notebook 06) | a few milliseconds with the adjoint method (1.9 ms for the gradient in notebook 06, 3.4 ms for a whole double-DQN update in SCIENCE.md); about 20 s with parameter shift through `EstimatorQNN` |
| Result (8–10 seeds per recipe) | learns oval and chicane in about half the episodes and keeps lapping more reliably; faster laps on gp and combo | learns all four tracks, and all four at once; on gp its policy laps earlier in training; no quantum advantage, and none is claimed |

Sources: the parameter counts and the circuit from notebook 03; the timings
from notebooks 03 and 06 and [SCIENCE.md](SCIENCE.md) ("Training",
"Hardware"; they depend on the machine); the results from the README's
"Measured results" and notebook 04.

Where the analogy breaks: a network's width costs memory linearly, while a
statevector simulation doubles with every qubit. And a circuit's output is
an estimate from shots on real hardware, never an exact number.

## More

- [Glossary](GLOSSARY.md) — about 60 terms in one sentence each.
- [Teaching with racetraQ](TEACHING.md) — a 90-minute and a half-day
  workshop, with an answer key.
- [racetraQ in plain words](EXPLAINER.md), [SCIENCE.md](SCIENCE.md),
  [REPORT.md](REPORT.md).
