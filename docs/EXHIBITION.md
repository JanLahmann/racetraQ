# Exhibiting traQmania

A practical runbook for running traQmania at a booth, in a classroom, or on a
museum kiosk. For what the science means, see [SCIENCE.md](SCIENCE.md); for
how the system works, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setups

### Laptop (the simple case)

```sh
./run.sh          # venv + install + launch, opens http://127.0.0.1:8000
```

Requires Python ≥ 3.11. Everything (training included) runs locally on CPU.
Pass server flags straight through, e.g. `./run.sh --port 8010`
(set `TRAQMANIA_PORT=8010` too so the auto-opened browser URL matches).

### Raspberry Pi (QuBins)

The demo runs on a Pi 4 or Pi 5 — e.g. on [QuBins](https://qubins.org) images,
which ship Qiskit preinstalled. Use the matching profile:

```sh
./run.sh --profile pi5     # or pi4
```

The Pi profiles lower the broadcast rate to 15 Hz and telemetry to 5 Hz and
shrink training batches (pi4 also drops default episodes to 150). For live
training on a Pi, always tick **Warm start** — cold training on a Pi 4 is a
coffee break; a warm start skips most of it.

Or containerized (the image is multi-arch and built on a QuBins base):

```sh
docker run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania
# podman works identically:
podman run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania
```

To use a profile inside the container, override the command:

```sh
docker run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania \
  python -m traqmania --host 0.0.0.0 --port 8000 --profile pi5
```

### Kiosk / exhibition mode

```sh
./run.sh --profile exhibition
```

The `exhibition` profile (`traqmania/config/exhibition.toml`):

- binds `0.0.0.0` — the UI is reachable from other devices on the LAN
  (visitors' phones can watch);
- `attract_idle_seconds = 20` — the client returns to attract mode after 20 s
  without keyboard/mouse activity, so the exhibit never sits on a stale
  screen (default is 45 s; set `0` to disable);
- `kiosk = true` — larger captions, hidden mouse cursor.

Profiles stack with an extra overlay via `--config <file.toml>`, and any
`./config/<name>.toml` in the working directory shadows the packaged profile
of the same name — so a Pi kiosk is `./run.sh --profile pi5 --config
traqmania/config/exhibition.toml`. Run the browser fullscreen, e.g.
`chromium-browser --kiosk http://localhost:8000`.

### The 6-qubit variant

```sh
./run.sh --profile q6
```

What visibly changes: the car senses with **5 lidar rays** instead of 3, the
circuit diagram shows **6 qubits**, and the parameter count reads **80**
(actions stay 4). Bundled 6-qubit weights cover **oval and chicane**;
**Evolution** (beyond the oval) and the **MLP opponent** are 4-qubit-only,
and the UI reports a clear error if selected rather than crashing. Live
training works on any track, and the oval ships a 6-qubit warm-start
checkpoint and evolution stages too. Hardware mode needs no special
handling: the simulated device (`fake_miami`, 120 qubits) fits every size
and simulates just the 6 physical qubits the circuit lands on, and the
6-qubit oval driver completes its lap there (see "Hardware-mode
prerequisites"). 8- and 10-qubit oval/chicane weights ship as well. The
8-qubit drivers are new: five layers instead of four, mean lap 13.4 s on
the oval and 12.6 s on chicane, a lap in 72 of 72 test episodes each. The
10-qubit oval and chicane drivers are new as well: six layers (every action
sees every input), 13.3 s and 12.6 s, 72 of 72 test episodes each. At 10
qubits even **gp** has a bundled driver, still from July: a four-layer
circuit that senses 5 rays plus four engineered track features (the gauge
labels change accordingly when it drives) and laps in 24 of 36 test
episodes at about 22 s.

Talking point — what the audit found, and what the re-measurement says:
*"With four layers, each action's readout only sees inputs within three
qubits of its own. At 4 and 6 qubits that is everything. At 8 qubits every
action misses one input — the Brake action cannot see the speed — and at 10
qubits three. The bigger circuits were driving partly blind; we found it by
checking our own work. The 8-qubit drivers now have a fifth layer; with it
the trained parameters lap more often at the end of training — a modest
gain on six runs per depth. It did not make the car faster: six
qubits is where this circuit does best, and a small classical network with
the same sensors is still ahead of it."* The numbers behind that are six to
eight training runs per size (SCIENCE.md, "Scaling and the light cone"). At
10 qubits only the oval has been re-measured — six runs each with four and
with six layers, neither better than six qubits. Do not quote a 10-qubit
chicane result: those runs are not analysed yet.
`python -m traqmania.agents.quantum.lightcone --qubits 8` prints the map if
a physicist asks.

## The 5-minute demo

A narrative that works cold, in order. Controls for the race segment:
**arrow keys or WASD** (up/W throttle, down/S brake, left/right steer).

1. **Attract — "this driver is a quantum circuit"** (~1 min).
   The screen already shows it: a car lapping, four wobbling gauges, a
   circuit diagram. Say: *"Every tenth of a second, the car's three distance
   sensors and its speed are encoded into rotations on 4 qubits; the four
   ⟨Z⟩ readouts you see on the gauges become the four action values — the
   biggest one steers the car."* Point at a corner: watch the brake action
   win just before the hairpin.

2. **Train — "watch it learn its first lap"** (~1 min).
   On the oval: mode **Train** → agent *Quantum* → tick **Warm start** →
   *Start training*. Eight cars flail, the return curve climbs, the first
   clean lap lands, and the best-lap banner fires as laps keep improving.
   Mention: *"This is real double-DQN training against a simulated version
   of the circuit — the approach of the 2020 quantum-RL paper by Chen et
   al., co-authored at IBM Research, that this demo follows."* (Without
   warm start a cold run on the oval drives its first clean lap around
   episode 260 — anywhere from 184 to 475 over ten training runs, so in
   two of the ten later than the 400 episodes of a default live run. On a
   Pi, warm only. gp and combo are the hard tracks: cold, the first clean
   lap comes around episode 1770 on gp, and on combo only 6 of 10 training
   runs drove one at all — train oval or chicane live for a payoff you can
   count on.)
   Measured in October 2026 over eleven training seeds per track through
   the server's own training path (seconds are CPU seconds of training on
   an M1 Max; a busy machine or a Pi takes longer): with **Warm start** a
   training car completes its first lap on the oval after 15–36 episodes
   (median 23; 1.4–2.9 s), on the chicane after 14–78 (median 50;
   1.3–7.2 s) and on the 6-qubit oval after 11–64 (median 29; 3.4–9.9 s) —
   in eleven of eleven seeds each — and on combo, which has a warm recipe
   of its own, after 148–284 episodes (median 208; 13–23 s), also eleven of
   eleven. Cold, the same 400-episode live run reaches its first lap on the
   oval after 184–339 episodes (median 257; 7.2–18.0 s; six of six seeds)
   and on the chicane in four of six seeds (212–397 episodes). Warm start
   does not rescue gp: a training car lapped in three of eleven seeds
   (episodes 200–274, 15–27 s). Whether the run *ends* with a lapping
   driver is a separate question: on the oval and on combo it did in eleven
   of eleven seeds, on the 6-qubit oval in nine, on the chicane in seven
   (not on the demo's own seed), on gp in three — which is why this step
   says "on the oval".

3. **Evolution — "the same circuit at four ages"** (~30 s).
   Mode **Evolution**: four numbered, colour-coded cars drive weights
   snapshotted at increasing points of one training run — the corner legend
   maps each number to how many episodes it trained ("ep N"). The youngest
   car wobbles and crashes; the oldest is smooth. One picture of what
   training buys. Works on all four bundled tracks (on gp/combo the young
   stages mostly crawl and crash — which is exactly the story).

4. **Race — "beat the quantum driver"** (~1.5 min).
   Mode **Race**, opponent *Quantum*, hand over the keyboard. Going off
   track freezes the visitor's car for a second, then respawns it; the ghost
   car is the all-time best lap on this machine. Most first-timers lose —
   that lands the point better than any slide.

5. **Hardware — "and now on a simulated quantum device"** (~1 min).
   Mode **Hardware**, backend *Simulated device*, run a **lap**: every
   steering decision is now an Estimator job against a local twin of a real
   IBM processor (`fake_miami`, a calibration snapshot of a 120-qubit
   Nighthawk) — its gate and readout errors, its coupling map, 1024 shots
   per decision. The status panel walks through connecting → transpiling →
   running and shows where it runs (the 4 physical qubits used), the
   execution mode, the two-qubit gate count (12) and the shots per job; then
   the run replays next to a simulator car driving the same weights.
   **Know what will happen before you press the button.** On the default
   track and size (oval, 4 qubits) the car now completes its lap: that
   driver lapped in 24 of 24 simulated-device episodes when we measured it,
   and so did **chicane** at 4 qubits and the **oval at 6 qubits**. Say
   why, because that is the interesting part: *"Same circuit, same weights
   — the only change is who executes it. With real-device noise and only
   1024 measurements per decision, the four action values blur. A driver
   trained on a perfect simulator often lets two of them sit so close that
   they swap places, and one wrong decision ends the lap — the driver we
   shipped until October 2026 got round in none of 14 tries. This one was
   trained to keep a wide margin between its action values, with that
   noise already in the loop."* Stay on those three combinations. The
   other bundled drivers were not trained that way and are a gamble on the
   device:
   chicane at 6 qubits lapped in 11 of 12 device episodes, gp at 4 qubits
   in 9 of 12, combo in 4 of 12, the universal driver in 9 of 12 on the
   oval and in 3 of 12 on chicane; the 8- and 10-qubit drivers were not
   measured.
   Do not say "with a token this runs on a real quantum computer" as if it
   were routine: the code path exists, but a full lap on a physical device
   is ~140 queued jobs and this documentation reports none. What you are
   showing is a calibration snapshot of a real device, simulated.
   A lap is about 140 decisions: ~20–30 s wall-clock on an idle laptop
   (0.1–0.2 s per decision), and a minute or more when the CPU is busy
   with something else (0.4 s per decision in our last check).

## Per-mode talking points

- **Watch (attract):** 4 qubits, 4 layers, 56 trainable parameters — of which
  4 are provably dead (a fun aside for physicists: the final RZ commutes with
  the Z measurement, and so does the whole last CZ ring — only 12 of the 16
  CZ gates can reach a readout). Gauges are live ⟨Z_a⟩; bars are Q-values
  after the trained output head. The circuit diagram is the actual gate
  sequence.
  The **Driver** dropdown swaps which training drives — put the gp-trained
  specialist on the oval to show zero-shot transfer, or pick *universal* (one
  circuit trained on all four tracks at once).
- **Surprise tracks (🎲 random):** every roll is a fresh procedurally
  generated circuit with hairpins and chicanes. **Set the Driver dropdown
  to *gp* first.** By default the universal weights drive it, and the
  universal driver bundled since October 2026 laps the four bundled tracks
  but none of the ten generated tracks we tested (0 of 120 episodes); the
  gp specialist lapped all ten (120 of 120), at 30–40 s a lap. Type a seed
  (shown in the track label) to reload a favourite; the size dropdown gives
  short/medium/long layouts. Long tracks are for driving and watching, not
  training.
- **Draw your own (✏️):** the crowd-pleaser — let a visitor sketch a loop on
  the race view; the server smooths it into a drivable track and the agent
  drives it zero-shot. Same advice as for 🎲: keep the Driver dropdown on
  *gp* (drawn tracks were not measured separately; they use the same
  default driver). Impossible drawings come back with a friendly hint (open
  loop, crossing, too-tight corners); drawing again is the adjust flow.
  Talking point: *"this driver was trained on a single track and has never
  seen yours — it only ever sees three distance rays and its speed, and
  that is why it can drive a track it was not trained on."*
- **Sharing the demo (driver lock + turn queue):** when several browsers are
  connected — a public deployment, or visitors' phones plus the booth screen
  — only one client at a time holds the wheel: the first to interact.
  Anyone else who presses a control joins a waiting line ("⏳ in line — 2
  ahead of you"); while someone waits, turns last at most 2 minutes (the
  driver sees the countdown), and the wheel also frees after ~90 s of driver
  inactivity or when the driver's tab closes. Tune via `[server]
  driver_turn_s` / `driver_idle_s`. No configuration needed for a single
  kiosk — a solo driver has no time limit and never notices any of it.
- **Train:** double DQN, epsilon-greedy, replay buffer — the classical RL
  recipe, with the neural network swapped for a quantum circuit. Choosing
  *Both* races quantum vs MLP learning curves live. Honest line: *"the
  small classical network learns the easy tracks in about half the
  episodes and more steadily, and it drives the hard ones faster — we
  measured that over eight to ten training runs each. The interesting part
  is that a 56-parameter quantum model does this at all. And it can be
  simulated classically, so nobody should call this quantum advantage."* If
  the lap
  times get worse again after a good lap: that is real. This kind of
  training is not stable on the hard tracks, which is why the saved driver
  is the best snapshot along the way, not the last one.
- **Studio:** the hands-on station. A visitor types a name, picks track,
  qubits, sensors and action set, and presses Start; training runs at most
  5 minutes (`[studio] time_limit_s`) and usually ends sooner on the easy
  tracks — once the best test laps all 12 test drives and six more tests
  bring no improvement. While it runs, the exhibit does not idle back to
  Watch. Measured on an M1 Max laptop: 4 qubits on the oval lap within
  ~20–30 s and finish in about 2 minutes; 6 qubits ~25–45 s to a first lap; 8
  qubits ~2 minutes; 10 qubits ~9 minutes — so 10 qubits will not lap in
  one turn (the setup screen says so), and the hard tracks (gp, combo) need
  a few minutes and may not lap at all. On a Raspberry Pi training is
  slower (not measured for the studio): try a run before the visitors come,
  and raise `time_limit_s` or steer them to 4 qubits. The result screen compares
  the run with the study runs of the same track and size ("your first lap
  came sooner than 9 of 10 study runs"); choices no study covers (corner
  sensors, 6/8 actions, warm start) are labelled experiments. Then "Race
  your model" — a visitor racing the circuit they just trained is the best
  moment of the booth. The studio board (one per track) ranks named runs
  that lapped: share of test drives lapped first, then mean lap time.
  Clear it by deleting `traqmania/data/leaderboard/studio_<track>.json`.
- **Evolution:** all cars run the identical architecture; only the training
  amount differs. Labels show "ep N" for mid-training checkpoints and "best"
  for the shipped driver. Tracks without stage snapshots show just two cars:
  warm-start vs best.
- **Race:** the agent decides 10 times per second and gets exactly the same
  observation a human gets from the screen: three distance rays and speed.
  Clean laps that beat the record become the new ghost.
- **Hardware:** inference on hardware, training on simulator — one gradient
  step is ~3.4 ms simulated vs ~20.5 s with circuit-evaluation gradients
  (before queue time!). The *sprint* action shows the middle path: SPSA
  estimates a gradient from 2 quantum jobs per iteration, whatever the
  parameter count. It is deliberately cautious — it only adjusts the output
  head, a third job per iteration checks each step, which is rejected if
  the measured loss got worse, and a step that would make the driver
  clearly worse on the exact simulator is refused before a job is spent on
  it. Present it as a demonstration of the mechanics, not as a way to
  improve the driver. What we measured on the simulated device in October
  2026, with the oval driver bundled before the retrain (not re-measured
  with the current one): in 20 seeded ten-iteration sprints it took between 0
  and 8 of the 10 steps, never lost the simulator's greedy return (the
  simulator check guarantees that) and lowered the hardware loss clearly in
  15 — but the adjusted driver lapped the simulated device no more often
  than before (1 of 8 episodes before, 0 of 8 after each of four sprints).
  If someone asks why it is so cautious: the first version, without these
  safeguards, wrecked the same driver in 10 of 16 runs, sometimes while the
  measured loss went down. The "eval return" line in the panel is the exact
  simulator's verdict, and the final message says how many steps were taken
  and how many the simulator check refused.
  Two details worth pointing at in the status panel: the **two-qubit gate
  count** (12 — the circuit's CZ ring fits IBM's square-lattice Nighthawk
  with no SWAPs, and the gates that cannot reach a readout are pruned) and
  the **mitigation level** (raw device noise by default: what you see is
  what the device would give). What the visitor should take away: the
  circuit is small enough for today's devices, and whether the *policy*
  survives their noise depends on how it was trained.

## Expert mode: the hero driver

Open the demo with `#expert` in the URL (`http://127.0.0.1:8000/#expert`) and
the Watch-mode Driver dropdown gains **hero — racing line**: a cyan car driven
by a model-based controller, *not* a learned agent. It computes the
family of candidate racing lines and physics-derived braking/speed profiles
straight from the track geometry, picks the fastest combination by simulating
itself with the real car physics (crash-free laps only), and tracks it with
continuous steering — the "perfect drive" ceiling for this car model.
Best laps, re-measured in October 2026: oval 12.1 s, chicane 12.1 s, gp
16.5 s, combo 19.0 s — the pace target for every learned driver — and it
adapts to physics changes with no retraining. Operator note: do not promise
that it never crashes. From the randomised start positions of our
evaluation it lapped in 12 of 12 runs on oval, chicane and gp but left the
track before the first lap in 6 of 12 on combo (8 of 12 on each of two
further sets of starts), and on one of seven generated tracks. One talking point: the hero's line visibly differs (wide
entries into hairpins, earlier braking). Notes: the first hero lap on a
track pauses ~5-8 s while the candidate search runs (cached afterwards),
and hero laps never become ghost records — the record board stays reserved
for learned and human drivers.

Expert mode also offers **pro — big classical DQN**: the biggest classical
agent we train, with the exact same double-DQN recipe as every other agent —
just more parameters (a wide MLP) and a richer observation (9 lidar rays,
speed and four track-aware scalars), trained on all four tracks at once
(5000 episodes; the best of three training runs). Best laps: oval 12.1 s,
chicane 12.5 s, gp 16.5 s, combo 19.1 s — level with the hero on oval and
gp, 0.1 and 0.4 s behind on combo and chicane — and a lap in 144 of 144
test episodes, plus 23 of 24 on eight generated tracks. The talking point
changed with the re-measurement: the pro driver steers with the same four
on/off actions as the quantum driver, so what separates the small agents
from the ceiling is sensing and model size, not the controls.

## Hardware-mode prerequisites

**Install the hardware extra** (not needed for anything else):

```sh
pip install -e ".[hardware]"     # qiskit-ibm-runtime
```

It must be qiskit-ibm-runtime ≥ 0.47 (the first release with `fake_miami`).

**Simulated device: nothing else.** No account, no network,
exhibition-safe. The default is `fake_miami` — the calibration snapshot of
a real 120-qubit Nighthawk processor (square lattice, CZ gates), of which
only the handful of physical qubits the circuit lands on are simulated.
The operator steps:

1. Pick a track and size whose driver was trained for device noise:
   **oval** or **chicane** at 4 qubits (the default size), or the **oval**
   at 6 qubits. Each lapped in 24 of 24 simulated-device episodes; the
   other bundled drivers are a gamble (step 5 of the demo has the counts).
2. Mode **Hardware** → backend *Simulated device* → **Run hardware lap**.
3. Read the panel: backend (`fake_miami (4-qubit patch: physical qubits
   …)`), execution mode (*session (dedicated)*), two-qubit gates, circuit
   depth, shots per job, decisions and seconds per decision. The run then
   replays as a ghost.

A different device or mitigation level is a config choice, not a UI one —
put it in an overlay passed with `--config`:

```toml
[hardware]
fake_name = "fake_fez"     # heavy-hex Heron: the CZ ring needs SWAPs (27 CZ instead of 12)
resilience_level = 1       # 0 raw noise (default) | 1 TREX | 2 TREX + ZNE with gate twirling
# rescale = "readout"      # per-readout attenuation rescale, calibrated with one extra job before the run
```

(The rescale helped drivers with moderate decision margins in our tests,
not a driver with knife-edge margins — SCIENCE.md, "Why a driver fails
under noise, and what helps". The three drivers of step 1 do not need it.)

`fake_manila` / `fake_lagos` (the retired 5- and 7-qubit Falcon devices
the demo used before October 2026) still work the same way.

**Real backend:** an IBM Quantum account. Create a token at
[quantum.cloud.ibm.com](https://quantum.cloud.ibm.com), then either
`export QISKIT_IBM_TOKEN=<token>` before starting the server, or save it
once with `QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
token=...)`. `[hardware] backend_name` picks the device (empty: least
busy). What to expect:

- **Free Open Plan account:** IBM documents 10 minutes of QPU time per
  28-day rolling window and no Sessions ("Workloads on the Open Plan can
  run only in job mode or batch mode"). The demo falls back to a Batch,
  then to plain jobs, and shows the reason in the panel. Without a Session
  every decision waits in the queue like any other job, so a full lap of
  ~140 decisions is not a booth activity; cap it (`--max-decisions 5`) to
  prove the loop closes.
- **The device will be a Heron** (heavy-hex) on the Open Plan as far as
  IBM's changelog shows; the SWAP-free Nighthawk embedding needs a paid
  plan. Expect the two-qubit gate count in the panel to be higher than on
  the simulated Nighthawk.
- **Do not promise a lap.** The drivers of step 1 lap on a *simulated*
  Nighthawk snapshot at 12 two-qubit gates (17 at 6 qubits). A real Heron
  runs more gates, drifts, and has not been tried: nothing here says what
  it will do.

For a booth: run the simulated device live and describe the real path
truthfully, or pre-run the CLI and show the transcript:

```sh
python -m traqmania.hardware lap --track oval --fake                # simulated Nighthawk, 4 qubits
python -m traqmania.hardware lap --track chicane --fake
python -m traqmania.hardware lap --track oval --fake --profile q6   # 6 qubits
python -m traqmania.hardware lap --track chicane --fake-name fake_fez --resilience 1
python -m traqmania.hardware lap --track chicane --fake --no-prune   # full circuit, 16 CZ
python -m traqmania.hardware sprint --track oval --fake --iterations 20
python -m traqmania.hardware lap --track oval --backend ibm_kingston --max-decisions 5   # real device
```

The CLI prints the backend (and patch), execution mode, any fallback
note, the two-qubit gate count and depth, the shots and the mitigation
level before the first decision. A sprint ends with the steps taken (and
how many the simulator guard refused or the hardware loss did not confirm),
the hardware loss before → after and the simulator's greedy return before →
after. `--rescale readout` adds the per-readout rescale to a lap or a
sprint; `tools/hw_reliability.py --weights W.npz --track oval` counts how
often a weights file laps under noise, emulated and on the simulated device
(about ten minutes at its defaults).

Never paste a token into a notebook or config file that might get
committed.

## Troubleshooting

**Port already in use** (`[Errno 48] address already in use`): a previous
instance is still running. `lsof -ti:8000 | xargs kill`, or start on another
port: `./run.sh --port 8010`.

**Status pill stuck on "reconnecting…":** the browser lost the websocket. The
client auto-reconnects with backoff (0.5 s → 8 s), so if the server is up it
recovers by itself — if it stays stuck, the server process died: check the
terminal, `curl http://127.0.0.1:8000/health`, and restart `./run.sh`. When
serving other devices, remember plain `./run.sh` binds `127.0.0.1` only — use
the exhibition profile or `--host 0.0.0.0`.

**Choppy rendering / sluggish Pi:** use the right profile (`--profile pi4` or
`pi5` — lower broadcast/telemetry rates, smaller training batches). Train
warm-start only, keep episodes modest, and prefer oval/chicane; gp is the
heavyweight track (cold training takes ~3000 episodes). Close other browser
tabs — the canvas renderer is cheap but not free.

**"training is already running" / "cannot change track while training is
running":** press **Stop** on the Training tab first; track switches and new
runs are blocked while a job is live.

**Resetting ghosts:** the best-lap ghost per track lives in
`traqmania/data/ghosts/<track>.json` and is overwritten whenever anyone —
including a talented visitor — beats it with a clean lap. To reset to the
bundled records in a git checkout: `git checkout -- traqmania/data/ghosts/`.
To simply clear one: delete the file and restart (no ghost is shown until a
new clean lap is driven). In a container, ghosts reset with the container.

**Hardware mode fails immediately:** with backend *real*, the server needs
`QISKIT_IBM_TOKEN` (see prerequisites); the error message contains the setup
steps. With the simulated device, check that `qiskit-ibm-runtime` is
installed (`pip install -e ".[hardware]"`); `unknown fake backend …` means
`[hardware] fake_name` names a device this runtime version does not ship —
the message lists the ones it does.

**The hardware car leaves the track after a few seconds:** on oval or
chicane at 4 qubits and on the oval at 6 that should be rare now (24 laps
in 24 measured episodes each) — check that the bundled weights were not
overwritten by a local training run (`git status traqmania/weights`). On
any other track or size it is expected: those drivers were not trained for
device noise (see step 5 of the demo).

**The panel shows "Session unavailable … using a Batch" (or job mode):** the
IBM account cannot open a Session — normal on the free Open Plan. The run
continues; each decision just queues on its own.
