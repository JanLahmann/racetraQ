# Teaching with racetraQ

Two workshop plans built on the [learning path](LEARN.md): 90 minutes for
school students or a first university course, and half a day for students
who already program. Both use the notebooks' "predict, then check"
exercises, end with a 10-minute debate on quantum advantage, and point to an
[answer key](#answer-key) at the bottom. The [glossary](GLOSSARY.md) is a
good handout.

The tone to keep, because it is what the data say: a classical network of
similar size learns this task faster and more reliably than the quantum
circuit. The workshop is about how quantum machine learning works and how to
measure it honestly, not about a quantum win.

## Before the workshop

- **Notebooks.** Every notebook has a Launch badge in the
  [README](../README.md#notebooks) that opens it on Binder from a
  [QuBins](https://qubins.org) image with Qiskit preinstalled. Open each one
  you plan to use on the day before, on the network you will teach on. The
  first launch of an image can take several minutes; ask students to click
  the badge at the start of the session, not when you need it.
- **Studio laptop** (for the competition). The Studio is part of the full
  demo, not the browser edition: `./run.sh` on a laptop
  ([README](../README.md#try-it)); `python -m racetraq --host 0.0.0.0`
  also serves it to other devices on the room's network. Do one training run
  before the students arrive. Clear old boards by deleting
  `racetraq/data/leaderboard/studio_<track>.json` while the server is
  stopped.
- **Phones.** [racetraq.org](https://racetraq.org/) works on a phone with
  nothing installed: enough for the opening and for anyone whose notebook
  will not start.
- **Exercises.** Each of notebooks 01–04 ends with three exercises. The
  question is in a text cell; the reasoning is in a collapsed "Solution" box
  below it; the code that checks it is in a cell marked `# Solution N`,
  which JupyterLab opens collapsed. Ask students to write a prediction down
  before they open anything.

### If Binder is slow or down

1. **Local install** (the most reliable on a good laptop): clone the
   repository, then `pip install -e ".[notebooks]"` and `jupyter lab`
   (Python 3.11 or newer). Notebooks 00–03 run in under two minutes each.
2. **Read instead of run.** The notebooks are committed with their outputs,
   so GitHub shows every plot and number. Students can still predict first
   and then read the stored result; only the exercise cells cannot be
   changed.
3. **The browser edition** at [racetraq.org](https://racetraq.org/) for the
   hands-on parts: watch a decision step by step, race the circuit, open a
   decision's circuit in IBM Quantum Composer.

## 90-minute workshop

For: school students (upper secondary) or a first university course.
Needs: a little Python; no quantum or machine learning background.

| Time | Activity | Material |
|---|---|---|
| 0:00–0:10 | **Hook.** On the projector and on phones: watch the quantum driver, pause on one decision (sensors → circuit → four scores), then two volunteers race it. Ask: what does the car know? | [racetraq.org](https://racetraq.org/) |
| 0:10–0:25 | **Qubits in 15 minutes.** Sections 1–5: RY and $\langle Z\rangle$, shots as coin flips, RZ that the measurement cannot see, CZ. Before each plot: predict the curve. | nb00 |
| 0:25–0:40 | **The game.** Car physics and the scripted driver. In pairs: exercises 2 and 3 (will the scripted driver lap gp? make it brake). | nb01 |
| 0:40–1:00 | **Learning from rewards.** Warm-up 1 (Q-table) with exercise 1 by hand on paper; warm-up 2 (a neural network, three learning rates); run the DQN training cell and watch the greedy lap. | nb02 |
| 1:00–1:10 | **The circuit as a driver.** Section 7 of nb00: one block by hand, and why one block is blind. Then the dead-parameter cell of nb03. | nb00, nb03 |
| 1:10–1:20 | **Debate:** is this quantum advantage? | [below](#the-10-minute-debate-is-this-quantum-advantage) |
| 1:20–1:30 | **Checkpoints.** The three questions at the end of nb00–02, in pairs, then the answers. | nb00–02 |

Take-home: the rest of nb03 and the student track of the
[learning path](LEARN.md).

## Half-day workshop

For: university students who program (computer science, physics,
engineering). About 3 h 45 min with two breaks. One laptop runs the Studio
for the competition.

| Time | Activity | Material |
|---|---|---|
| 0:00–0:15 | **Hook** as in the 90-minute plan; explain the competition rules (below) and start the first Studio run. | racetraq.org, Studio |
| 0:15–0:45 | **Qubits and the game.** nb00 (all) and nb01, with nb01's exercises. | nb00, nb01 |
| 0:45–1:30 | **Q-learning from scratch.** nb02 with all three exercises. Discuss: why a target network, why a replay buffer. | nb02 |
| 1:30–1:45 | Break. Studio runs continue. | |
| 1:45–2:35 | **Circuits as Q-functions.** nb03: re-uploading, the two implementations, dead parameters and light cones, adjoint against parameter shift, trainability. All three exercises; exercise 2 on paper first. | nb03 |
| 2:35–2:45 | Break. | |
| 2:45–3:15 | **Measuring honestly.** nb04 Part B and its three exercises (one seed against one seed, four seeds against ten, what 72 of 72 proves). Re-run nothing: Part A trains for minutes; read its stored output. | nb04 |
| 3:15–3:25 | **Competition results** on the Studio board, read with nb04's lesson in mind. | Studio |
| 3:25–3:35 | **Debate:** is this quantum advantage? | below |
| 3:35–3:45 | **Checkpoints** of nb03 and nb04; where to go next (tracks 3–5 of the learning path). | nb03, nb04 |

## The Studio board as a classroom competition

The Studio trains a quantum driver live: pick a track, 4–10 qubits, the
sensors and the action set, then press Start. A run stops after at most 5
minutes, sooner once its best test laps all 12 test drives and six more
tests bring no improvement. Named runs that lap enter a per-track board,
ranked by the share of test drives lapped, then by mean lap time. The result
screen compares each run with the study runs of the same track and size.

Rules that work in a classroom:

1. Teams of 2–4 choose a name, a qubit count and the sensors. Everyone
   races on the **oval**, so the board compares like with like.
2. Each team writes down a prediction before its run: when will it first
   lap? Will 6 qubits beat 4?
3. One run per team on the shared laptop; the others work on the notebooks
   meanwhile. Measured on an M1 Max laptop
   ([EXHIBITION.md](EXHIBITION.md)): 4 qubits on the oval lap within about
   20–30 s and finish in about 2 minutes; 6 qubits take about 25–45 s to a
   first lap; 8 qubits about 2 minutes; 10 qubits about 9 minutes, so they
   will not lap within one turn (the setup screen says so). On a Raspberry
   Pi training is slower.
4. Read the board together at the end, with nb04 exercise 1 in mind: the
   same recipe with a different seed can land anywhere from the top to the
   bottom of the board. The winner may have had a lucky seed. Ask: how many
   runs per team would you need to call a winner?

Then "Race your model": the team that trained the winning driver races it.

## The 10-minute debate: "is this quantum advantage?"

Motion: *"racetraQ shows that quantum computers are better at learning to
drive."* Two groups, two minutes to prepare from the evidence below, two
minutes per side, then four minutes of open discussion.

Evidence for the motion side to work with:

- The circuit learns all four tracks from scratch, and one 4-qubit circuit
  learns all four at once.
- It has fewer parameters (56) than the classical network (76).
- On gp its greedy policy reaches a lapping driver earlier in training
  (about half the MLP's episodes).
- A trained driver completed a full lap on a physical IBM quantum processor.

Evidence for the opposing side:

- Over 8–10 seeds the classical network learns oval and chicane in about
  half the episodes, keeps lapping more reliably, and laps gp and combo about
  10 s and 18 s faster.
- Four qubits are trivial to simulate. Training ran entirely on a classical
  simulation of the circuit.
- A classical surrogate fitted to the trained circuit drives like it
  (notebook 07).
- One decision costs about 100 times more on the exact simulator than with
  the network, and seconds on real hardware (the physical lap took 27
  minutes for 141 decisions).
- Most of the tuning effort went into the circuit (24 of 29 variants in the
  gp study).

How to close: there is no quantum advantage here, and the project does not
claim one. A real claim would need a problem where the quantum model
provably captures structure that comparable classical models cannot, and a
comparison that could come out against the quantum side. Ask what makes
racetraQ worth studying anyway: it is a complete, honest testbed in which
the comparison *can* come out against the quantum model — and does.
Sources: [SCIENCE.md](SCIENCE.md), "Honest claims"; notebook 04, closing
section; notebook 07, Part C.

## Answer key

Every notebook carries its own solutions: a collapsed "Solution" box with
the reasoning, and a cell marked `# Solution N` that checks it. The numbers
below are those of the stored outputs; the training exercises of nb02 are
single runs, so a different machine can show slightly different numbers.

| Notebook | Exercise | Answer in one line |
|---|---|---|
| [01](../notebooks/01_the_racing_env.ipynb) | 1. Hairpin speed by hand | about 18.4 units/s, 0.74 of top speed |
| | 2. Scripted driver on gp | 0 of 12 cars lap; the brake fires (47 times) but too late |
| | 3. Teach it to brake | brake earlier: with `front < 0.6` all 12 lap; with `front < 0.6` and speed > 0.6 the mean lap is about 25 s |
| [02](../notebooks/02_q_learning_from_scratch.ipynb) | 1. Bellman by hand | $Q(6, \text{throttle}) = -1 + 0.98 \cdot 10 = 8.8$; $Q(7, \text{throttle}) = -10$ |
| | 2. $\gamma = 0.9$ | Q-values about 25 instead of about 68; it still laps the oval (one seed) |
| | 3. Learning rate 0.1 | learning collapses: mean return of the last 25 episodes about 240 against about 1600 (one seed) |
| [03](../notebooks/03_quantum_circuits_as_q_functions.ipynb) | 1. Delete the final RZ layer | Q-values change by about $10^{-13}$; zeroing the final RYs changes the action in most states |
| | 2. Cone map, 6 qubits, 2 blocks | 12 of 36 dead: all final RZs, the final RY and λ of qubits 4–5, the first-block RZs of qubits 4–5 |
| | 3. Count the parameters | `q8`: 128, `q10`: 188 ($P = 3Ln + 8$) |
| [04](../notebooks/04_training_the_quantum_driver.ipynb) | 1. One seed against one seed | on the oval the circuit looks more stable in about 1 pairing in 5 |
| | 2. Four seeds instead of ten | interval 0.60 wide instead of 0.40; the estimate ranges from 0.45 to 0.95 over all choices of 4 seeds |
| | 3. What 72 of 72 proves | a failure rate of up to about 4 % is still plausible (rule of three: 3/72) |

The checkpoint questions at the end of notebooks 00–04 have their answers
in a collapsed box right below them.
