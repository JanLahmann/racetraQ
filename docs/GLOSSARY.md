# Glossary

One plain sentence per term, and where to learn more. "nb03" means
[notebook 03](../notebooks/03_quantum_circuits_as_q_functions.ipynb), and so
on; the notebooks are listed in the [README](../README.md#notebooks). For an
order to read them in, see the [learning path](LEARN.md).

## Racing and reinforcement learning

**Action** — One of the four moves the car can make ten times a second:
steer right, go straight, steer left (all at full throttle) or brake.
*More:* nb01, "Actions and reward".

**Agent** — The learner that picks the actions; in racetraQ either the
quantum circuit or the classical network, trained by the same code.
*More:* nb02.

**Bellman equation** — The rule that the value of an action equals its
immediate reward plus the discounted value of the best action in the next
state; Q-learning turns it into an update. *More:* nb02, "The 60-second
theory".

**Best snapshot** — The parameters, among all those evaluated during a
training run, that drove best in the greedy evaluations; this is what a
weights file holds, not the parameters at the end of training. *More:* nb04,
Part A.

**Checkpoint (track)** — A marker along the track that pays a small bonus
(+5) when the car passes it during a lap. *More:* nb01, "Tracks".

**Discount factor (γ)** — How much a reward one step later is worth compared
with one now (0.98 in racetraQ); the agent looks roughly $1/(1-\gamma)$
decisions ahead. *More:* nb02, exercise 2.

**Double DQN** — A DQN variant in which the online network picks the best
next action and the target network evaluates it, which reduces
overestimated Q-values. *More:* nb02, "Deep Q-learning".

**DQN (deep Q-network)** — Q-learning with a parameterized function in place
of a table, trained by gradient descent on the TD error, with a replay buffer
and a target network. *More:* nb02.

**Episode** — One drive from a start position until the car leaves the track
or the 60-second time limit runs out. *More:* nb01, "Actions and reward".

**ε-greedy exploration** — Acting randomly with probability ε and greedily
otherwise; ε falls from 1.0 to a floor during training (0.05, or 0.30 for the
circuit on gp and combo). *More:* nb02; nb04, Part C.

**Exploration floor** — The final value of ε, below which exploration never
drops; raising it to 0.30 removed the 4-qubit circuit's late collapse on gp.
*More:* nb04, Part C.

**Greedy policy** — Always taking the action with the highest Q-value, with no
random exploration; all evaluations drive greedily. *More:* nb02, "Watch what
it learned".

**Lidar ray** — A distance sensor: how far the track edge is in one direction
(60° right, ahead, 60° left), divided by 30 units so that it lies in
$[0, 1]$. *More:* nb01, "What the agent sees".

**Markov decision process (MDP)** — The formal setting of reinforcement
learning: states, actions, rewards and transitions, where the next state
depends only on the current state and action. *More:* nb02.

**Observation** — What the agent is given each decision: in the default
setup three lidar rays and the speed, four numbers in $[0, 1]$. *More:* nb01.

**Policy** — The rule that maps what the agent sees to an action; for a
Q-learner, "take the action with the largest Q-value". *More:* nb02.

**Q-function / Q-value** — $Q(s, a)$ is the total discounted reward the agent
can still expect after taking action $a$ in state $s$ and playing well
afterwards; each driver outputs one Q-value per action. *More:* nb02.

**Q-table (tabular Q-learning)** — A Q-function stored as one number per state
and action, updated entry by entry; it works only when there are few states.
*More:* nb02, "Warm-up 1".

**Reinforcement learning (RL)** — Learning to act by trial and error from
rewards, without being shown the correct action. *More:* nb02.

**Replay buffer** — A memory of past transitions (10,000 in racetraQ) from
which each update draws a random batch of 32, so that consecutive, correlated
steps are not learned from in order. *More:* nb02, "Deep Q-learning".

**Reward** — The number the environment pays after each decision: progress
along the track, +5 per checkpoint, +50 per lap, −10 for leaving the track.
*More:* nb01, "Actions and reward".

**Target network** — A slowly updated copy of the parameters that computes
the learning targets, so that the targets do not move with every update.
*More:* nb02, "Deep Q-learning".

**TD error (temporal-difference error)** — The difference between the
Bellman target $r + \gamma \max_{a'} Q(s', a')$ and the current estimate
$Q(s, a)$; learning shrinks it. *More:* nb02, "Warm-up 1".

**Truncation** — Ending an episode at the time limit while the car is still
on track; the trainer keeps bootstrapping there, unlike after a crash.
*More:* nb02, "Deep Q-learning".

## Neural networks

**Activation function** — The nonlinear function a neuron applies to its
weighted sum; racetraQ's network uses tanh. *More:* nb02, "Warm-up 2".

**Adam** — The optimizer used for both drivers: gradient descent with a
per-parameter step size adapted from running averages of the gradient.
*More:* `racetraq/agents/training/dqn.py`.

**Backpropagation** — Computing the gradient of the loss with respect to every
weight in one backward pass through the network, by the chain rule. *More:*
nb02, "Warm-up 2".

**Gradient descent** — Repeatedly changing every parameter a small step
against the slope of the loss. *More:* nb02, "Warm-up 2".

**Learning rate** — The size of each gradient-descent step; too small and
learning crawls, too large and it overshoots or blows up. *More:* nb02,
"Warm-up 2" and exercise 3.

**Loss** — A number that measures how wrong the model is; here the squared
TD error. *More:* nb02, "Warm-up 2".

**MLP (multilayer perceptron)** — racetraQ's classical baseline: 4 inputs, 8
tanh neurons, 4 outputs, 76 weights and biases. *More:* nb02, "The classical
Q-function".

**Neuron** — A unit that multiplies its inputs by weights, adds a bias and
applies an activation function. *More:* nb02, "Warm-up 2".

## Quantum computing

**Amplitude** — One of the complex numbers that describe a quantum state; the
probability of a measurement outcome is the squared size of its amplitude.
*More:* nb00, section 1.

**Bloch sphere** — A picture of one qubit as an arrow on a sphere, $|0\rangle$
at the north pole and $|1\rangle$ at the south pole; its height is
$\langle Z\rangle$. *More:* nb00, sections 1–2.

**CZ gate** — The two-qubit gate of the racetraQ circuit: it flips the sign of
the $|11\rangle$ amplitude, which matters only together with rotations after
it. *More:* nb00, section 5.

**Entanglement** — A joint state of several qubits that cannot be written as
separate states of each; the CZ ring creates it. *More:* nb00, section 5.

**Expectation value ⟨Z⟩** — The average of +1 (outcome 0) and −1 (outcome 1)
over many measurements, $P(0) - P(1)$, a number in $[-1, 1]$; each action's
score is read this way. *More:* nb00, section 2.

**Measurement** — Reading a qubit, which gives 0 or 1 at random with the
probabilities its state sets. *More:* nb00, section 1.

**Qubit** — The quantum bit: a two-level system whose state is two
amplitudes, measured as 0 or 1. *More:* nb00.

**RY / RZ rotation** — Single-qubit gates: RY tilts the Bloch arrow and
changes the measurement odds; RZ spins it around the vertical axis and
changes only phases. *More:* nb00, sections 1 and 4.

**Shot** — One run of a circuit ending in one measured bit string; an
expectation value from $N$ shots has a statistical error of about
$1/\sqrt{N}$. *More:* nb00, section 3; nb05.

**Statevector simulator** — A classical program that stores all $2^n$
amplitudes of an $n$-qubit state and applies the gates exactly; its cost
doubles with every qubit. *More:* nb03, "Two implementations, one circuit".

**Superposition** — A qubit state with non-zero amplitudes for both 0 and 1,
so that the measurement outcome is random. *More:* nb00, section 1.

## The quantum driver

**Adjoint differentiation** — Computing the gradient of a simulated circuit
with one forward and one backward sweep of the statevector, the circuit
analogue of backpropagation; racetraQ trains this way. *More:* nb03, "What
gradients cost".

**Ansatz** — The fixed layout of gates whose angles are trained; racetraQ's
is encode, rotate, entangle, repeated in blocks. *More:* nb03, "The circuit".

**Block (layer)** — One repetition of encode (RY of each input), rotate (RY
and RZ) and entangle (CZ ring); the default circuit has 4. *More:* nb00,
section 7; nb03.

**Data re-uploading** — Encoding the inputs again in every block, which lets
the circuit represent richer functions of them. *More:* nb03, "Why
re-uploading?".

**Dead parameter** — An angle that can never change any readout, so its
gradient is exactly zero; at 4 qubits these are the 4 final RZ angles.
*More:* nb03, "A teaching point".

**fastsim** — racetraQ's own numpy statevector simulator with adjoint
gradients, checked against Qiskit's `EstimatorQNN` to numerical precision.
*More:* nb03.

**Fourier series (of a circuit)** — Because inputs enter as rotation angles,
every readout is a sum of sines and cosines of the inputs with a known set of
frequencies. *More:* nb03, "Why re-uploading?"; nb07, Part B.

**Input scale (λ)** — A trainable factor that multiplies each input before
it becomes a rotation angle. *More:* nb00, section 6; nb03.

**Light cone** — The set of gates and inputs that can influence one readout;
with nearest-neighbour CZ gates it grows by one qubit in each direction per
block. *More:* nb03; nb07, Part A.

**Output head** — The trainable weight $w_a$ and bias $b_a$ that turn
$\langle Z_a\rangle$ into a Q-value. *More:* nb03, "The circuit".

**Parameter-shift rule** — Computing a circuit's gradient from two extra
circuit runs per parameter, at shifted angles; it works on real hardware but
costs thousands of times more than the adjoint method here. *More:* nb03,
"What gradients cost".

**VQC (variational quantum circuit)** — A circuit with trainable angles used
as a model, here as the Q-function. *More:* nb03.

## Hardware

**Action gap** — The distance between the best and second-best Q-value;
racetraQ's `action_gap` option (advantage learning) trains it wider, so that
device noise is less likely to swap the actions. *More:* nb04, "The recipe";
nb05, section 5.

**Error mitigation** — Classical post-processing that reduces the effect of
device noise on expectation values; racetraQ's default is none (resilience
level 0). *More:* nb05, sections 3 and 6.

**Fake backend (simulated device)** — A local noise model built from a real
IBM processor's calibration data, such as `fake_miami` (Nighthawk); no account
needed. *More:* nb05, section 1.

**Noise model** — A description of how a device's gates and readout go
wrong, used to simulate it. *More:* nb05, section 4.

**QPU** — Quantum processing unit: a physical quantum processor; racetraQ's
oval driver completed one lap on IBM's `ibm_marrakesh` on 2026-10-03. *More:*
nb05; SCIENCE.md, "Hardware".

**SPSA** — A gradient-free optimizer that estimates a direction from two
evaluations with all parameters shifted at once; the hardware sprint uses it.
*More:* nb05, section 8.

**Transpilation** — Rewriting a circuit into a device's native gates and
qubit connections. *More:* nb05, section 2.

## Measuring results

**Bootstrap interval** — An uncertainty range found by resampling the seeds
with replacement many times and taking the middle 95 % of the recomputed
statistic. *More:* nb04, Part B.

**Classical surrogate** — A classical model fitted to reproduce a trained
circuit's outputs; one drives racetraQ's oval as the circuit does. *More:*
nb07, Part B.

**IQM (interquartile mean)** — The mean of the middle half of the values,
less swayed by one lucky or unlucky seed than the mean. *More:* nb04, Part B.

**Probability of improvement** — The chance that a random seed of one recipe
beats a random seed of another; 0.5 means no difference. *More:* nb04, Part B.

**Quantum advantage** — A quantum computer solving a problem better or faster
than any classical method; racetraQ does not claim it, and its circuits are
easy to simulate classically. *More:* nb04, closing section; SCIENCE.md,
"Honest claims".

**Reliability eval** — Driving the best snapshot and the final parameters on
36 greedy episodes that no snapshot was selected on. *More:* nb04, Part A.

**Seed** — The number that fixes a run's random choices; different seeds of
the same recipe can learn very differently, so results are reported over
8–10 seeds. *More:* nb04, Part B and exercise 1.

**Stability** — After a run's greedy evaluation first laps, the fraction of
later evaluation episodes that still lap; 1 means it keeps what it learned.
*More:* nb04, Part A.

## The demo

**Studio** — The mode in the full demo where a visitor configures and trains
a quantum driver live (at most 5 minutes) and enters it on a per-track board.
*More:* README, "Modes"; [TEACHING.md](TEACHING.md).

**Browser edition** — The install-free version at
[racetraq.org](https://racetraq.org/): the bundled drivers race in the
browser, with every decision's circuit shown. *More:* README, "Browser
edition".
