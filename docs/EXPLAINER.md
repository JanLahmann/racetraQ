# racetraQ in plain words

A short explainer for visitors, journalists and students. The long version,
with every table and caveat, is [SCIENCE.md](SCIENCE.md).

## What is this?

racetraQ is a racing game in which the driver is a tiny quantum circuit. A
small car laps a track; ten times a second it reads its sensors, and a
circuit of four qubits decides whether to steer left, steer right, go
straight or brake. Nobody programmed the circuit to drive. It learned by
trial and error, the way a small neural network learns to play a video game.

Next to it sits a matched classical driver: an ordinary small neural network
with a similar number of adjustable numbers, trained with exactly the same
method on exactly the same tracks. The demo exists to put the two side by
side and report what was actually measured — including where the quantum
driver loses. It runs on a laptop or a Raspberry Pi.

## What does the car see and do?

Very little. Three distance sensors — straight ahead, 60 degrees left, 60
degrees right — and its own speed: four numbers, each between 0 and 1. It has
no map and does not know where on the track it is, which is why a driver
trained on one track can sometimes drive another.

From those four numbers it picks one of four actions: full throttle while
steering right, straight or left — or brake. Two of the four bundled tracks
(oval and chicane) can be driven flat out; the other two (gp and combo) have
hairpins where a car that does not brake flies off.

## What does "learning" mean here?

The method is deep Q-learning, a standard recipe from classical AI. The
driver keeps a value for every action in every situation: roughly, how much
progress around the track it expects if it takes that action now. It starts
with random values and random driving, and every crash, completed lap or
metre of progress nudges the values toward what actually happened.

The quantum part is only *where the values live*. Instead of a neural
network, the value function is a quantum circuit with 56 adjustable numbers,
most of them rotation angles; the classical baseline is a network with 76.
The same code trains both.

One detail matters for everything below: training does not settle. A driver
that laps cleanly at one checkpoint can crash in every test 50 episodes later
and recover after that. What the demo ships is therefore not the circuit as
training left it but the best snapshot along the way — and the science page
says so, and records how each snapshot was chosen.

## Where are the qubits?

Picture four qubits in a ring. Each sensor reading is turned into a rotation
of its own qubit — a long distance ahead rotates its qubit further than a
short one. Then adjustable rotations are applied (the knobs that training
turns), and neighbouring qubits are entangled, which mixes their information.
That block — encode, rotate, entangle — is repeated four times, the sensor
readings fed in afresh each time, which is what lets the circuit represent
more complicated functions. Finally each qubit is measured. The result is
four numbers between −1 and +1, which a trained scale turns into the four
action values; the biggest wins. The four gauges wobbling on the demo screen
are those readouts, live.

Because mixing happens only between neighbours, a measured qubit can be
influenced only by inputs a few qubits away: one qubit further per repeated
block, in each direction around the ring. Physicists call this the readout's
*light cone*. With four blocks the cone reaches three qubits around the ring —
everything, when the ring has four or six qubits. But at eight qubits every
action was blind to one sensor (the brake action could not see the speed),
and at ten each action was blind to three. No amount of training can fix
that: the information never reaches the qubit that is measured. The October
2026 audit found this by checking our own work; the bigger circuits now have
an extra block or two so that every action sees every sensor.

## What did we find?

In October 2026 everything was re-measured: eight to ten training runs per
setting instead of one to three, each judged on fresh test episodes it was
not selected on. The headline changed.

**The classical network learns faster and more reliably.** On the oval it
drove its first clean lap after about 150 training episodes, the circuit
after about 260 (eight classical and ten quantum runs). Once the network can
lap, it keeps lapping in well over nine of ten later checks on both easy
tracks; the circuit keeps it in roughly three quarters of them on the oval
and about two thirds on the chicane (ten runs each). On the hairpin tracks
neither driver's training is stable, and the network's laps are faster — by
about 10 seconds on gp and 18 on combo (ten runs each). One thing goes the
circuit's way: on gp its runs reach a lapping policy roughly twice as early
in training as the network's — a slower policy, and one that training does
not keep.

**The circuit does learn every track.** All four, from scratch, and one
four-qubit circuit learns all four at once. At six qubits every one of eight
training runs gave a reliable driver on both easy tracks, with chicane laps
level with the network's. Each bundled single-track driver lapped in 72 of 72
fresh test episodes.

**More qubits did not buy faster laps.** Eight and ten qubits are no faster
than six on either easy track, and if anything less steady.

**No quantum advantage, and none was expected.** Four qubits — or ten — are
trivial to simulate; training literally ran on a classical simulation of the
circuit, because one training step is thousands of times cheaper there than
on a device. More than that: the trained circuit computes a function of a
known mathematical form, and a classical model can be fitted to copy it —
notebook 07 does so, and the copy drives the car. A racing toy that a small
classical network also learns is not where quantum computers can show an
advantage. The point is different: a quantum model can be a drop-in part of a
real learning system, measured carefully enough that the comparison can come
out against it — as it does.

## Does it run on a real quantum computer?

Partly. Hardware mode sends every steering decision as a job to a
*simulated* IBM device: a calibration snapshot of a real 120-qubit Nighthawk
processor, with its measured gate and readout errors, of which only the four
qubits the circuit lands on are simulated. On a noisy device, 1,024
measurements per decision leave the four action values blurred. The driver
shipped until October 2026 never finished a lap there (0 of 14 tries): its
best two action values sat so close that noise swapped them. The current oval
and chicane drivers at the default size were trained with that noise in the
loop and rewarded for keeping their action values apart; each has lapped in
72 of 72 episodes on the simulated device. Over all training runs, not just
the chosen ones, 10 of 10 oval runs and 8 of 10 chicane runs trained that way
lap almost every device episode, against 5 of 8 and 0 of 8 without.

With an IBM Quantum account the same code submits to a physical machine. A
lap is about 140 queued jobs; on 3 October 2026 the oval driver completed
one full lap on IBM's `ibm_marrakesh` processor in 27 minutes of waiting
and computing, with no error correction of any kind. One lap on one day is
a data point, not a track record.

## What can I try?

At the booth: **Watch** the circuit drive. **Train** a fresh driver and see
the first clean lap land within minutes on the oval (on a Raspberry Pi, tick
*warm start*). **Evolution** races four snapshots of one training run against
each other. **Race** it yourself with the arrow keys — most first-timers
lose. **Hardware** runs a lap on the simulated device. Roll a random track or
draw your own, and pick the *gp* driver, which laps tracks it has never seen;
the default *universal* driver currently stops on them, a known open item.

The seven notebooks launch in a browser with nothing installed and build the
whole stack from scratch: environment, classical driver, quantum circuit and
its dead angles, the many-run comparison, the simulated device, more qubits,
and finally the trained circuit taken apart.

## Where to read more

- [SCIENCE.md](SCIENCE.md) — every number above with its uncertainty, the
  protocol, what the audit corrected, and the literature.
- [The notebooks](../notebooks/) — the whole stack built from scratch.
- [EXHIBITION.md](EXHIBITION.md) — the runbook for showing it, with a
  scripted five-minute demo and what to expect from each mode.
