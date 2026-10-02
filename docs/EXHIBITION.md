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
coffee break, warm-start is seconds.

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
and simulates just the 6 physical qubits the circuit lands on — and the
6-qubit oval driver is the one to show there (see "Hardware-mode
prerequisites"). 8- and 10-qubit oval/chicane weights ship as well — the
q10 oval driver's 12.0 s is the fastest quantum lap in the demo (at ~5–9 ms
per decision instead of <1 ms) — and at 10 qubits even **gp** has a bundled
driver: it senses 5 rays plus four engineered track features (the gauge
labels change accordingly when it drives), and laps at 20–22 s.
<!-- RESULTS-PENDING: lap times of the retrained 8/10-qubit drivers quoted in this paragraph -->

Talking point — what the audit found, which is the better story anyway:
*"With four layers, each action's readout only sees inputs within three
qubits of its own. At 4 and 6 qubits that is everything. At 8 qubits every
action misses one input — the Brake action cannot see the speed — and at 10
qubits three. The bigger circuits were driving partly blind, and our own
'more qubits don't help much' result was measured that way. We found it by
checking our own work; the corrected runs use more layers."* Do not quote
the old scaling line ("sample efficiency stays flat from 4 to 10 qubits")
as a result until it has been re-measured.
<!-- RESULTS-PENDING: scaling talking point from the re-measured, fully visible 8/10-qubit circuits -->
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

2. **Train — "watch it learn its first lap in seconds"** (~1 min).
   Mode **Train** → agent *Quantum* → tick **Warm start** → *Start
   training*. Eight cars flail, the return curve climbs, and the first clean
   lap lands in a couple of seconds (measured ~2.2 s on oval, ~2.7 s on
   chicane, ~9 s on combo); the best-lap banner fires as laps keep
   improving. Mention: *"This is real double-DQN training against a
   simulated version of the circuit — the approach of the 2020 quantum-RL
   paper by Chen et al., co-authored at IBM Research, that this demo
   follows."* (Without warm start a full cold run on oval is ~18 s to the
   first clean lap — still demoable; on a Pi, warm only. gp's warm training
   is a coin flip — it laps in ~20–40 s on most seeds but can miss outright,
   so prefer the other three tracks for a guaranteed payoff.)
   <!-- RESULTS-PENDING: warm-start and cold first-lap times after the retrain -->

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
   track and size (oval, 4 qubits) the noisy car leaves the track after a
   few seconds — 0 laps in 14 runs when we measured it. That is the honest
   result, and it demos well if you say it first: *"Same circuit, same
   weights — the only change is who executes it. The exact simulator laps;
   with real-device noise and only 1024 measurements per decision, this
   driver's four action values get close enough to swap places, and one
   wrong decision ends the lap. Training the policy to keep a wider margin
   between its action values fixes that in most of our test runs; the
   driver you are watching has not been retrained that way yet."* If you
   want a completed lap instead, set Qubits to 6 (or start
   with `--profile q6`) and stay on the oval (13 of 13 runs lapped);
   **chicane** at 4 qubits lapped in 12 of 20. Do not say "with a token this
   runs on a real quantum computer" as if it were routine: the code path
   exists, but a full lap on a physical device is ~150 queued jobs and this
   documentation reports none.
   <!-- RESULTS-PENDING: which bundled drivers complete a simulated-device lap after the retrain; update the track/profile recommendation here, in step 1 of "Hardware-mode prerequisites" and in the Troubleshooting entry about the hardware car leaving the track -->
   A completed lap takes ~20–30 s wall-clock (0.1–0.2 s per decision).

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
  generated circuit with hairpins and chicanes; the universal weights drive
  it. Type a seed (shown in the track label) to reload a favourite; the size
  dropdown gives short/medium/long layouts. Long tracks are for driving and
  watching, not training.
- **Draw your own (✏️):** the crowd-pleaser — let a visitor sketch a loop on
  the race view; the server smooths it into a drivable track and the same
  universal circuit drives it zero-shot. Impossible drawings come back with a
  friendly hint (open loop, crossing, too-tight corners); drawing again is
  the adjust flow. Talking point: *"the agent has never seen this track —
  the egocentric lidar view is why one driver generalizes."*
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
  *Both* races quantum vs MLP learning curves live. Honest line: *"similar
  learning in the runs we have, no speedup — the interesting part is that a
  56-parameter quantum model does this at all. And it can be simulated
  classically, so nobody should call this quantum advantage."* If the lap
  times get worse again after a good lap: that is real. This kind of
  training is not stable on the hard tracks, which is why the saved driver
  is the best snapshot along the way, not the last one.
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
  improve the driver. What we measured on the simulated device with the
  default oval driver: in 20 seeded ten-iteration sprints it took between 0
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
  circuit is small enough for today's devices; the *policy* is not yet
  robust enough for their noise.

## Expert mode: the hero driver

Open the demo with `#expert` in the URL (`http://127.0.0.1:8000/#expert`) and
the Watch-mode Driver dropdown gains **hero — racing line**: a cyan car driven
by a model-based controller, *not* a learned agent. It computes the
family of candidate racing lines and physics-derived braking/speed profiles
straight from the track geometry, picks the fastest combination by simulating
itself with the real car physics (crash-free laps only), and tracks it with
continuous steering — the "perfect drive" ceiling for this car model.
Measured (physics v2): oval 12.1 s, chicane 12.1 s, gp 16.4 s, combo 19.0 s
— ahead of every learned driver everywhere it takes skill (the pro driver
edges it by 0.2 s on the flat-out oval, which is pure path geometry) — and
it handles every generated and drawn track, adapting to physics changes with
no retraining. Two talking points: the learned agents' gap to this ceiling
is mostly the 4-action bang-bang control, not intelligence, and the hero's
line visibly differs (wide entries into hairpins, earlier braking). Notes:
the first hero lap on a track pauses ~5-8 s while the candidate search runs
(cached afterwards), and hero laps never become ghost records — the record
board stays reserved for learned and human drivers.

Expert mode also offers **pro — big classical DQN**: the biggest classical
agent we train, with the exact same double-DQN recipe as every other agent —
just more parameters (a wide MLP) and a richer observation (9 lidar rays,
speed and four track-aware scalars), trained on all four tracks at once
(5000 episodes under v2; seed 0). Measured: oval 11.9 s, chicane 12.4 s,
gp 17.8 s, combo 20.4 s and 10/10 generated tracks — the strongest learned
driver in the demo, 1–1.4 s behind the hero on the hard tracks. That gap is
the 4-action control interface, not model size.
<!-- RESULTS-PENDING: pro driver lap times after the retrain (the hero is model-based and does not change) -->

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

1. Pick the track and size that will complete a lap (today: **oval** at
   6 qubits; **chicane** at 4 qubits laps more often than not; the default
   oval/4-qubit driver leaves the track after a few seconds — see step 5
   of the demo).
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
not the default oval driver — SCIENCE.md, "Why it fails, and what helps".)

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
  ~150 decisions is not a booth activity; cap it (`--max-decisions 5`) to
  prove the loop closes.
- **The device will be a Heron** (heavy-hex) on the Open Plan as far as
  IBM's changelog shows; the SWAP-free Nighthawk embedding needs a paid
  plan. Expect the two-qubit gate count in the panel to be higher than on
  the simulated Nighthawk.
- **Do not promise a lap.** The default driver does not survive the
  simulated device's noise; there is no reason to expect better from the
  real one.

For a booth: run the simulated device live and describe the real path
truthfully, or pre-run the CLI and show the transcript:

```sh
python -m traqmania.hardware lap --track oval --fake --profile q6   # simulated Nighthawk
python -m traqmania.hardware lap --track chicane --fake
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

**The hardware car leaves the track after a few seconds:** expected for the
default oval / 4-qubit driver under device noise — not a bug in your setup.
Use 6 qubits on the oval (see prerequisites); chicane at 4 qubits also
fails in roughly 4 runs of 10.

**The panel shows "Session unavailable … using a Batch" (or job mode):** the
IBM account cannot open a Session — normal on the free Open Plan. The run
continues; each decision just queues on its own.
