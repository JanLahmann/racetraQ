# traQmania architecture

How the pieces fit: a numpy physics/RL core, a FastAPI + WebSocket demo server
that ticks one shared session, a vanilla-ES-module browser frontend, and an
optional bridge to IBM Quantum hardware.

## System overview

```mermaid
flowchart LR
    subgraph browser [Browser - traqmania/web]
        main[main.js\nboot + wiring]
        net[net.js\nws client, reconnect]
        race[race.js\ncanvas renderer]
        panels[quantum-panel.js / charts.js\ncircuit.js / attract.js / explain.js]
        input[input.js\nkeys -> bitmask]
    end

    subgraph server [Server - traqmania/server]
        app[app.py\nFastAPI: /health, /ws, static]
        ws[ws.py\nHub + per-socket loop]
        proto[protocol.py\ntyped messages, strict validation]
        session[session.py\nDemoSession: 60 Hz tick,\nmode state machine]
        runtime[runtime.py\nagents/tracks/ghosts/config glue]
    end

    subgraph core [Core - traqmania]
        env[env/: track.py, car.py,\nracing_env.py]
        agents[agents/: quantum fastsim+adjoint,\nlight cone, MLP, DQNTrainer, SPSA]
        hw[hardware.py\nEstimator lap + SPSA sprint]
    end

    ibm[(IBM Quantum QPU\nor local device twin)]

    main --- net
    net <-->|JSON over /ws| ws
    ws --> proto
    ws --> session
    session --> runtime
    session --> env
    session --> agents
    session --> hw
    hw --> ibm
    app --> ws
```

## Module map

| Module | Responsibility |
|---|---|
| `traqmania/__main__.py` | CLI entry point: profile/config/host/port flags, starts uvicorn. |
| `traqmania/config.py` | `default.toml` + profile overlay (`pi4`, `pi5`, `exhibition`, plus the circuit-size overlays `q6`/`q8`/`q10` — each sets `[circuit] n_qubits` and the matching `[observation] ray_angles_deg`) + optional extra TOML; `./config/*.toml` in the working dir shadows packaged profiles. Also `resolve_training_cfg` (`[training]` + the track's preset + warm-start recipes — one rule for the server and headless training) and `parse_override` / `apply_overrides` (the dotted `section.key=value` overrides behind `train_headless --set` and `tools/study.py`). |
| `traqmania/env/track.py` | Closed-loop track geometry: resampling, arc-length projection, lidar raycasts, spatial-hash acceleration, validation. |
| `traqmania/env/car.py` | Vectorized bicycle-ish car physics (throttle/brake/drag, speed-dependent steering). |
| `traqmania/env/racing_env.py` | Gym-style vector env: obs = lidar rays (`[observation] ray_angles_deg`, 3 by default) + speed, progress reward, checkpoint/lap bonuses, off-track penalty, auto-reset. `step()` info also reports `truncated` (ended by the time limit, still on track) and `final_obs` (the observation before the auto-reset) for `[training] bootstrap_truncation`. |
| `traqmania/env/trackgen.py` | Procedural track generator (numpy-only): `generate_track(seed, difficulty, length)` builds a deterministic closed loop that passes the exact `Track.load` validation (see "Random tracks" below); `LENGTH_PRESETS` maps short/medium/long to base-radius ranges and perimeter caps. |
| `traqmania/env/racing_line.py` | The model-based "hero" driver (expert demo): curvature-minimizing racing line, brake/accel-feasible speed profile from `[physics]`, pure-pursuit `RacingLineController` returning continuous (steer, throttle, brake). Not a learned agent. |
| `traqmania/env/multi_track.py` | `MultiTrackEnv`: round-robin mixture of per-track `RacingEnv`s behind the identical vector-env interface, so `DQNTrainer` trains one policy over several tracks unchanged; `random_pool` builds the mixture from generated tracks. |
| `traqmania/agents/base.py` | `QFunction` protocol + the 4 discrete actions (right/straight/left at full throttle, coast-brake — car steer +1 turns left on screen). |
| `traqmania/agents/quantum/circuit.py` | Canonical Qiskit circuit (single source of truth) + JSON `circuit_spec` for the browser diagram, including the light-cone fields `visibility`, `dead_params`, `min_layers_full_visibility`, `dead_gates`, a `live` flag on every gate and `pruned_on_hardware`. |
| `traqmania/agents/quantum/lightcone.py` | Structural light-cone analysis (numpy only): `feature_visibility` (which features each readout ⟨Z_a⟩ can depend on), `live_gates` / `live_parameter_mask` (what can influence a readout at all — the rest has exactly zero gradient), `min_layers_full_visibility`, `blind_spots` / `blind_spot_warning` (human-readable, printed by `train_headless`), `pruned_circuit` (the live gates only, same expectation values — what the hardware path runs). CLI: `python -m traqmania.agents.quantum.lightcone --qubits 10 --layers 4`. |
| `traqmania/agents/quantum/fastsim.py`, `adjoint.py` | Hand-written numpy statevector simulator and adjoint (backprop-style) gradients. |
| `traqmania/agents/quantum/noise.py` | Expectation-value noise model (numpy only at import): `ExpectationNoise` (attenuation, shots, bias — what a device does to ⟨Z_a⟩), `NoisyQFunction`, `ReadoutCorrection` (the calibrated rescale behind `HardwareQFunction(rescale=...)`), `calibrate` / `validate` against the local device patch (CLI: `python -m traqmania.agents.quantum.noise calibrate|validate --fake fake_miami`). The model is semi-quantitative — what it gets right and wrong is measured in SCIENCE.md, "Why it fails, and what helps". |
| `traqmania/agents/quantum/surrogate.py` | Classical Fourier surrogates of the trained circuit (numpy only): the frequency spectrum set by the trained input scalings, (`frequency_spectrum` / `spectrum_size`: the product upper bound the encoding allows), `gate_coefficients` (the exact Fourier coefficients of a readout — which of those frequencies are really used), and `fit_surrogate` (`full` least squares, random Fourier features, kernel ridge) returning a `FourierSurrogate` that can drive in place of the circuit (`drive_laps`, `compare`). Backs notebook 07 — light cones and classical surrogates. |
| `traqmania/agents/quantum/qdqn.py` | `QuantumQFunction`: fastsim-backed `QFunction`, flat `[lam, theta, w, b]` layout, P = 3·L·n + 8 params (56 at 4 qubits, 80 at 6); `param_groups()` names the `lam` / `theta` / `head` slices for per-group learning rates. |
| `traqmania/agents/quantum/qnn.py` | Same circuit via qiskit-machine-learning `EstimatorQNN` (parity checks, shots/noise backends; `aer_noisy` uses the local device twin from `hardware.local_simulator`). |
| `traqmania/agents/classical/mlp.py` | 76-parameter numpy MLP baseline (4-8-4, tanh) with analytic backprop; `param_groups()`: `body` / `head`. |
| `traqmania/agents/training/dqn.py` | Double-DQN loop over vectorized envs, Adam, replay buffer — shared by both backends. Snapshot eval (one round of `eval_episodes` distinct greedy episodes every `eval_every` episodes; `history["eval_log"]`, `final_eval`, `trainer.final_params`) and the optional `[training]` knobs `bootstrap_truncation`, `loss` / `huber_delta`, `lr_groups`, `lr_end`, `target_update` / `tau`, `grad_clip`, `reward_scale` — each defaulting to the legacy behaviour — plus, for robustness to device noise and also off by default, `act_noise` (act and select snapshots under an `ExpectationNoise`; TD targets and gradients stay exact) and `action_gap` (advantage learning). |
| `traqmania/agents/training/spsa.py` | Minimal SPSA minimizer used by hardware sprints: two loss evaluations per iteration, plus the optional safeguards the sprint turns on — per-parameter `scale` (0 freezes a parameter), a `max_step` trust region, `blocking` (reject a step the loss does not confirm), `accept` (a veto on a proposed point, costing no loss evaluation) and `calibrate_gain`. |
| `traqmania/hardware.py` | IBM Quantum via `qiskit-ibm-runtime`, real or simulated. `get_backend` (real QPU, or a fake by name — default `fake_miami`, a Nighthawk calibration snapshot; unknown names raise), `local_simulator` / `execution_backend` (the Aer twin of a fake, built once: fakes of up to 7 qubits whole, larger devices — the 120–156-qubit ones above all — as a *device patch* of just the physical qubits the routed circuit touches), `open_execution_mode` (Session → Batch → job fallback, with the reason), `HardwareQFunction` (inference-only; light-cone-pruned ISA circuit, client-side `executor_estimator.Estimator` with `EstimatorV2` fallback, `resilience_level` 0/1/2, optional calibrated attenuation `rescale`), `run_hardware_lap`, `spsa_sprint` (by default: output head only, blocking, and a guard that vetoes steps costing more than 10 % of the exact-simulator greedy return; TD targets follow the recipe in the weights' sidecar; the result lists `accepted` and `vetoed` per iteration, and `sprint_steps_text` words them). CLI: `python -m traqmania.hardware lap|sprint [--track T] [--profile q6] [--fake] [--fake-name NAME] [--backend NAME] [--weights W.npz] [--shots N] [--resilience 0|1|2] [--rescale off|global|readout] [--no-prune]`, plus `--max-decisions N` for a lap and `--iterations N --batch N --groups lam,theta,head --no-blocking --no-guard` for a sprint. `[hardware]` config: `backend_name`, `fake_name`, `shots`, `decision_shots`, `spsa_iterations`, `spsa_groups`, `spsa_blocking`, `spsa_guard`, `resilience_level`, `prune_light_cone`, `rescale`, `calibration_samples`, `calibration_shots`. |
| `traqmania/server/protocol.py` | Typed WS messages; strict client-side validation (`ProtocolError`). |
| `traqmania/server/session.py` | `DemoSession`: the mode state machine and synchronous 60 Hz `tick()`; training threads; ghost recording. |
| `traqmania/server/runtime.py` | Loading bundled agents/weights/tracks/ghosts, track payloads; re-exports `config.resolve_training_cfg`. |
| `traqmania/server/ws.py` | Connection `Hub`, broadcast fan-out, per-socket receive loop, `DriverLock` (exclusive control, spectators watch). |
| `traqmania/server/app.py` | FastAPI factory: `/health`, `/ws`, `/api/docs` + `/api/docs/{id}` (repo markdown for the in-UI docs browser; empty outside a source checkout) and `/docs-assets` (images), static frontend mounted last. |
| `traqmania/train_headless.py` | Offline training CLI that produces the bundled `weights/*.npz` (+ `.meta.json`, history JSON). The recipe is `[training]` with the track's `[training_presets.<track>]` merged on top (`--preset none` skips that); `--set section.key=value` overrides any config value (an unknown `training.` key is an error), `--episodes` / `--seed` win over everything, and `--save-final` also writes `<name>.final.npz`, the end-of-training parameters. The sidecar records the circuit shape, observation, action count and the resolved training table. Besides the bundled names, `--track multi` trains one policy on the oval+chicane+gp+combo mixture and `--track random` on a `MultiTrackEnv.random_pool` of generated tracks (seeded from `--seed`); weights save under the literal names (`quantum_multi.npz` / `quantum_random.npz`) — the universal-driver candidates. |
| `traqmania/records.py` | `python -m traqmania.records`: greedy evaluation of every bundled driver on every bundled track into `data/records.json`. |
| `traqmania/bench.py` | Micro-benchmarks (env steps, forward passes, DQN updates). |
| `tools/study.py` | Multi-seed study harness. `run` trains a (variant × seed) grid, one subprocess per cell, resumable; each cell saves best-snapshot and final weights and evaluates both over 36 distinct greedy episodes. `report` aggregates over seeds: IQM and median with bootstrap confidence intervals, stability, sample complexity, probability of improvement over a baseline (`report.md`, `report.json`). |
| `tools/hw_reliability.py` | Lap completion and decisions-until-crash of one weights file under device noise: many emulated episodes (the `noise.py` model) plus a few on the local device patch, across shots (`--shots`, default 1024,4096,16384), rescale settings (`--rescale`, default off,readout) and resilience levels (`--resilience`, default 0,1); `--device-episodes N` sets the device-path sample (default 3 — use 8 or more to judge a driver), `--no-device` emulates only. Emulated rescale rows are optimistic, and "mean lap" is not comparable between emulated and device rows (different decision caps). |
| `tools/make_stages.py` | Trains a fresh quantum agent, snapshots parameters as it learns, and saves 4 evolution-stage weights `quantum_<track>_stage{1..4}.npz` (+ `.meta.json` with the episode count shown as the car label). |
| `traqmania/web/` | Frontend ES modules (`main`, `net`, `race`, `input`, `charts`, `circuit`, `quantum-panel`, `hardware-panel`, `attract`, `explain`, `docs`, `md`, `draw`, `tooltip`); no build step, served statically. |

## Qubit-count profiles (q6 / q8 / q10)

The quantum stack is generalized over `[circuit] n_qubits`. Qubits map 1:1 to
observation features — (n − 1) lidar rays evenly spaced over [−60°, +60°]
plus normalized speed — while the **4 actions and the Z_0…Z_3 readout stay
fixed** (extra qubits only widen the feature register). The overlays set
exactly that pair of config values:

- `--profile q6` — 6 qubits, 5 rays; trained oval and chicane weights are
  bundled (`weights/quantum_oval_q6.npz`, `quantum_chicane_q6.npz`).
- `--profile q8` / `q10` — 8/10 qubits, 7/9 rays; oval and chicane weights
  are bundled at both sizes, plus `quantum_gp_q10.npz` (trained on an
  engineered-feature observation recorded in its sidecar). On a track
  without weights at the active size, the modes that need them stay
  unavailable until you train, e.g.
  `python -m traqmania.train_headless --agent quantum --profile q8 --track gp`.

**Depth and the light cone.** All profiles keep `[circuit] n_layers = 4`.
That is enough for every action's readout to see every feature at 4 and 6
qubits, but not at 8 (needs 5 blocks) or 10 (needs 6): each readout only
depends on features within ring distance `n_layers − 1` of its qubit
(`agents/quantum/lightcone.py`; SCIENCE.md, "Light cones"). `circuit_spec`
reports it, and `train_headless` warns before training such a circuit (live
training writes the same warning to the server log). To
train with full visibility: `--set circuit.n_layers=6`. The sidecar records
the depth, but weight *filenames* do not (use `--out`), and the server
loaders adopt a driver's recorded observation and action count, not its
depth — such weights only load under a config with the same `n_layers`.
Under any other depth the session reports them as unavailable (an `error`
naming the parameter counts, car-less attract mode) instead of loading
them.

The default stays 4 qubits and is bit-identical to the pre-scaling stack
(pinned by a regression test). The evolution stages and the bundled MLP
(agent and race opponent) are 4-qubit-only: at n ≠ 4 the session rejects
those modes/opponents with an `error` message instead of crashing (evolution
comes back if you produce `_stage<i>` weights at that qubit count).

Weight filenames follow one rule (`server/session.quantum_weights_path`):
`quantum_<track>[_warmstart|_stage<i>][_q<n>].npz` — the `_q<n>` tag is
appended at any non-default qubit count, so a `--profile q6` run reads and
writes `quantum_<track>_q6.npz` and never clobbers the 4-qubit weights.

## Random tracks (🎲)

`env/trackgen.generate_track(seed, difficulty=0.5)` builds a procedural
closed-loop track, deterministic per `(seed, difficulty)`: random polar
harmonics around a loop, amplitude-normalized so the minimum corner radius
lands ~3× the car's kinematic minimum turn radius at difficulty 0, tightening
toward ~1.6× (and narrowing from wide to gp-like width) at difficulty 1. The
candidate is constructed through the same code path `Track.load` uses after
parsing JSON (resample + validation + precompute), so a generated track can
never be unlearnable by the load-time rules; a failing candidate retries with
rng-derived jitter (≤ 50 attempts — in practice the first attempt passes).

**Server flow** — `set_track {track: "random", seed?}` generates a track
named `random #<seed>` (a seedless request rolls a fresh seed) and broadcasts
the normal `track` payload; the UI's 🎲 picker entry sends exactly that, shows
the seed in the picker label so the track is reproducible, and offers a
reroll button. Attract / human race / live training all work as on any track.
Differences from bundled tracks:

- **Weights fallback chain** — no per-track specialist exists, so the quantum
  driver resolves to `quantum_universal[_q<n>].npz` when bundled, else the gp
  specialist `quantum_gp[_q<n>].npz` (under the current physics it was
  measured to lap oval and combo zero-shot, but not chicane — SCIENCE.md,
  "One driver, every track"). The car is labelled honestly: `driver: universal` or
  `driver: gp-trained generalist` (`session.random_track_weights`). Warm
  starts and evolution stages follow the same chain (`_warmstart` /
  `_stage<i>` suffixes).
  <!-- RESULTS-PENDING: which tracks the retrained gp specialist laps zero-shot -->
- **No ghost persistence** — best-lap ghosts are never written for random
  tracks (`ghosts_dir` only ever holds bundled-track files).
- **Graceful rejections** — hardware mode needs a bundled track name (the
  runner loads tracks by name); evolution/mlp modes reject with the existing
  missing-weights `error` when their files are absent.

The universal-driver *candidates* are trained offline with
`train_headless --track multi` (bundled-track mixture) or `--track random`
(generated pool via `MultiTrackEnv.random_pool`); promoting one to
`quantum_universal.npz` is a manual bundling decision.

## WebSocket protocol reference

> **Keep in sync:** this section is hand-generated from
> `traqmania/server/protocol.py`. If you change the protocol, change this
> table in the same commit. Every message is a JSON object with a `"type"`
> field. Client→server messages are **strictly validated**: unknown types,
> unknown fields, wrong value types, and out-of-enum values all get an
> `error` reply on the offending socket (other clients are unaffected).
> Fields marked *omitted-if-null* are dropped from the wire when `None`.

### Client → server

**`hello`** `{}` — request a fresh `welcome` on this socket (also sent
automatically to every client on connect).

**`input`** — human driving input, last-writer-wins across all sockets (one
shared seat). `keys` is always required.

| field | type | required | meaning |
|---|---|---|---|
| `keys` | int 0–15 | yes | bitmask: 1 throttle, 2 brake, 4 left, 8 right (left+right cancel, brake overrides throttle) |
| `steer` | float [-1, 1] | no | analog steering (clamped at parse time) |
| `throttle` | float [0, 1] | no | analog throttle (clamped) |
| `brake` | float [0, 1] | no | analog brake (clamped) |

When **any** analog field is present the server drives from the analog values
instead of the bitmask (missing axes default to 0.0; send `keys: 0`
alongside). The analog override persists until a later keys-only `input`.

**`set_mode`** — `{mode}` with `mode` ∈ `attract | train | race | evolution |
hardware`.

**`set_track`** — `{track, seed?, length?}` with `track` a name from
`welcome.tracks` or the special name `"random"` (procedurally generated — see
"Random tracks"). `seed` (int ≥ 0, *omitted-if-null*) is only meaningful with
`track: "random"` and makes the generated track reproducible; without it the
server rolls a fresh one. `length` (`"short" | "medium" | "long"`,
*omitted-if-null*, default medium) picks a size preset; non-default lengths
are tagged into the track name (`random #7 (long)`), and long tracks are for
live driving only (laps can exceed the 60 s training-episode cap). Rejected
with an `error` while training or a hardware job is running, or for unknown
names.

**`draw_track`** — `{points}` with `points` a list of 8–5000 `[x, y]` pairs:
a freehand centerline stroke from the UI's ✏️ draw overlay (world
coordinates, any scale). `env/trackgen.track_from_drawing` recentres and
rescales it to a drivable perimeter, low-passes pointer jitter (coarse
arc-length resample) and Chaikin-rounds the corners just enough to clear the
6-unit minimum corner radius, narrows the track a little if two sections run
close, and finally constructs it through the same `Track` path as every other
track. The result is named `drawn #<n>` and behaves like a generated track
(universal-driver fallback, no ghost persistence). Drawings that cannot be
made drivable — open strokes, self-crossings, unrecoverably tight corners —
come back as an `error` naming what to fix; drawing again is the adjust flow.

**`set_driver`** — `{driver}`: pick which trained quantum weights drive the
agent car. `"auto"` (default) uses the current track's specialist with the
honest universal/gp fallback on generated tracks; a training name from
`welcome.drivers` (e.g. `"oval"`, `"gp"`, `"universal"`) forces that
training's weights on any track; `"hero"` swaps in the model-based
racing-line controller (expert demo — no weights, continuous controls, works
on every track; the stock UI only offers it with `#expert` in the URL), and
`"pro"` the big classical DQN reference driver (`mlp_pro.npz`, same expert
menu). The attract car rebuilds immediately; race/evolution are unaffected
(`"hero"` and `"pro"` behave like `"auto"` for them). Unavailable names (not bundled at the active
qubit count) get an `error`; a qubit switch silently resets an unavailable
pick to `"auto"`.

**`train`**

| field | type | required | meaning |
|---|---|---|---|
| `action` | `"start"` \| `"stop"` | yes | |
| `agent` | `"quantum"` \| `"mlp"` \| `"both"` | yes | `both` starts two side-by-side jobs |
| `track` | str | no | switch track first |
| `warm` | bool | no (default false) | start from bundled warm-start weights (quantum only) |
| `episodes` | int ≥ 1 | no | overrides the config default |

**`race`**

| field | type | required | meaning |
|---|---|---|---|
| `action` | `"start"` \| `"reset"` | yes | `reset` respawns all cars (race mode only) |
| `opponent` | `"quantum"` \| `"mlp"` | yes | |
| `track` | str | no | switch track first |

**`qubits`** — `{n}` (int ≥ 1): live circuit-size switch. The server takes
`[circuit]` and `[observation]` from the packaged `q{n}` profile (`n = 4` →
the plain default config) and keeps every other section as it was started
(`[ui]`, `[server]`, `[hardware]`, `[training]`… — an exhibition kiosk stays
a kiosk), rebuilds
track/env/agents/circuit state in place, resets to attract mode (car-less if
that qubit count has no bundled weights for the current track — an `error` is
emitted alongside, same as any weight-missing rejection), and re-broadcasts
the `welcome` payload so every client re-renders. Rejected with an `error`
while training or a hardware job is running, or when no packaged `q{n}`
profile exists.

**`set_name`** — `{name}`: the racer's display name for the leaderboard (a
string of at most 24 characters, stripped; empty clears it — unnamed laps
are not recorded).

**`hardware`** — run the quantum policy on an IBM backend (or a local fake
noise-model twin).

| field | type | required | meaning |
|---|---|---|---|
| `action` | `"lap"` \| `"sprint"` \| `"abort"` | yes | one inference lap / SPSA fine-tune / cancel |
| `backend` | `"fake"` \| `"real"` | yes | fake = the local twin of `[hardware] fake_name` (default `fake_miami`), no account needed; real = `[hardware] backend_name`, else the least busy QPU of the saved account |
| `iterations` | int ≥ 1 | no | SPSA iterations (sprint only; default `[hardware] spsa_iterations`) |
| `shots` | int ≥ 1 | no | shots per Estimator job (default `[hardware] shots`, 1024; for a lap `[hardware] decision_shots` when set). The stock web panel always sends this field, so `decision_shots` only matters for other clients and the CLI |
| `max_decisions` | int ≥ 1 | no | lap only: cap the rollout at N backend decisions |

Error-mitigation level, light-cone pruning, the attenuation rescale and the
sprint's SPSA settings are not message fields: the server takes them from
`[hardware] resilience_level` (0), `prune_light_cone` (true), `rescale`
(off), `spsa_groups` (`["head"]`), `spsa_blocking` (true) and `spsa_guard`
(0.9).

### Server → client

**`welcome`** — sent on connect and in reply to `hello` (and re-broadcast to
everyone after a `qubits` or `set_driver` switch):
`{mode, track: TrackPayload, tracks: [str], circuit_spec, ui, obs_labels,
driver, drivers}`. `driver` is the active `set_driver` selection (`"auto"`
default) and `drivers` the currently valid choices (`"auto"`, each bundled
training at the active qubit count, `"hero"`, and `"pro"` when
`mlp_pro.npz` is bundled).
`circuit_spec` is the JSON gate-by-gate circuit description from
`agents/quantum/circuit.circuit_spec` (qubit/layer/gate list, parameter
counts, readout observables, action labels) plus the structural light cone:
`visibility` (`[a][j]` = 1 iff readout Z_a can depend on the feature on
qubit j), `dead_params` (circuit parameters with zero gradient for every
input), `dead_gates` (gates outside every readout's light cone, per gate
type), `min_layers_full_visibility`, and a `live` flag on each gate (false:
it cannot influence any action and the hardware path drops it).
`pruned_on_hardware` (bool) is `[hardware] prune_light_cone`: whether the
hardware path really drops the dead gates — the page says "skipped on
hardware" only when it is true. For a circuit shape the analysis rejects,
the light-cone fields and every `live` are null. `ui` is the
`[ui]` config section (`attract_idle_seconds`, `kiosk`). `obs_labels` (*omitted-if-null*) is the
display name of each observation feature feeding the circuit, in qubit order
(`env.feature_names`, e.g. `["ray -60°", "ray 0°", "ray +60°", "speed"]`).

**`control`** — `{driving, locked, watchers, waiting, queue_pos,
turn_ends_in_s}`, per-client driver-lock + turn-queue status. One client at a
time controls the shared session: the first to send any control message
takes the wheel; anyone else who interacts joins a FIFO line (their control
messages are dropped and answered with this status). A solo driver keeps the
wheel indefinitely, but while the line is non-empty a turn lasts at most
`[server].driver_turn_s` (default 120 s); the wheel also frees after
`[server].driver_idle_s` (default 90 s) of driver inactivity or on
disconnect — a 1 Hz ticker (`ws.control_ticker`) performs the handover to
the next in line and keeps countdowns fresh. `driving` = this client holds
the wheel; `locked` = someone does; `watchers` = other connected clients;
`waiting` = line length; `queue_pos` (*null when not queued*) = this
client's 1-based place; `turn_ends_in_s` (*null when nobody waits*) = the
running countdown. Protects public deployments and exhibit screens from
visitors' phones fighting over the demo — and turns contention into an
arcade-style rotation.

**`track`** — `{track: TrackPayload}` after a successful track switch.

*TrackPayload* (built in `runtime.track_payload`): `name`, `half_width`,
`total_length`, `checkpoints` (fractions of a lap), `theme` (free-form dict
from the track JSON), `start: {x, y, theta}`, and polylines `centerline`,
`left`, `right` as `[[x, y], ...]`.

**`state`** — `{t: float, mode, cars: [CarState]}` at the broadcast rate.

*CarState:*

| field | type | notes |
|---|---|---|
| `id` | str | `"human"`, `"quantum"`, `"mlp"`, `"stage1..4"` (evolution), `"ghost"`, `"hardware"`, `"hero"`, `"pro"` |
| `kind` | `"human"` \| `"quantum"` \| `"mlp"` \| `"hero"` \| `"pro"` | |
| `x`, `y`, `theta`, `v` | float | pose + speed |
| `lap` | int | completed laps |
| `progress` | float | signed arc-length progress |
| `last_lap_time` | float \| null | null until the first lap |
| `off_track` | bool | |
| `rays` | [float] | *omitted-if-null*; normalized lidar distances (agent cars) |
| `label` | str | *omitted-if-null*; e.g. `"ep 250"` (evolution), `"best 14.4s"` (ghost), `"hardware lap"`, `"driver: gp-trained generalist"` (random track) |
| `ghost` | bool | *omitted-if-null*; true for replay cars |

**`quantum`** — live circuit introspection for a quantum car, throttled to
≤ 10 Hz per car: `{car_id, expectations: [float], q_values: [float],
action: int}` (`expectations` are the raw ⟨Z_a⟩ readouts).

**`telemetry`** — training progress per agent at the telemetry rate:
`{agent, episode, mean_return, epsilon, loss: float|null,
returns_tail: [float]}` (last ≤ 100 episode returns), plus *omitted-if-null*
`best_lap_s: float` and `lap_times: [[episode, lap_s], ...]` (last ≤ 50).

**`event`** — `{kind, car_id?, lap_time?, agent?}` with `kind` ∈
`lap | crash | clean_lap | training_done | new_best_lap` (optional fields
*omitted-if-null*; `new_best_lap` carries `agent` during training and
`car_id` for ghost records).

**`hardware_status`** — progress of a hardware lap/sprint:

| field | type | meaning |
|---|---|---|
| `phase` | `"idle"` \| `"connecting"` \| `"transpiling"` \| `"running"` \| `"replay"` \| `"done"` \| `"error"` | |
| `backend_name` | str? | where the circuit runs; for a simulated large device it names the patch, e.g. `fake_miami (4-qubit patch: physical qubits 4, 5, 14, 15)` |
| `message` | str? | human-readable status / error text |
| `decision` | int? | lap: decisions completed so far |
| `seconds_per_decision` | float? | lap: measured latency |
| `iteration` | int? | sprint: SPSA iterations completed so far |
| `loss` | float? | sprint: current TD loss |
| `eval_return_before` / `eval_return_after` | float? | sprint: greedy fastsim return before/after (mean of 12 episodes) |
| `lap_time` | float? | lap: completed lap time (absent when the run ended without a lap) |
| `execution_mode` | `"session"` \| `"batch"` \| `"job"`? | the mode the account/backend granted |
| `note` | str? | only after a fallback: which more exclusive modes were refused and why (e.g. an Open Plan account cannot open a Session), or why a requested rescale was switched off |
| `two_qubit_gates` | int? | two-qubit gates of the transpiled (ISA) circuit |
| `circuit_depth` | int? | depth of the transpiled (ISA) circuit |
| `shots` | int? | shots per Estimator job of this run |
| `rescale` | `"global"` \| `"readout"`? | only when the attenuation rescale is applied |
| `attenuation` | float? | with `rescale`: the calibrated attenuation f |

All optional fields are *omitted-if-null*; `running`-phase messages are
throttled to ≤ 5 Hz. The job itself runs in a background thread (like
training) so the tick loop never blocks on a quantum backend. The run fields
(`execution_mode`, `note`, `two_qubit_gates`, `circuit_depth`, `shots`,
`rescale`, `attenuation`) arrive with a second `transpiling` message once
the circuit is transpiled and the execution mode is open — its `message`
also names the mitigation level and whether the circuit is light-cone
pruned — and again with `done`. A sprint's `done` message summarizes what
SPSA moved, how many steps were taken and why not the rest, the fresh
hardware loss before → after and the number of jobs, in the form `SPSA on
head: steps taken 1/10 (3 refused by the simulator guard (the policy would
keep less than 90% of its return); 6 did not lower the hardware loss);
hardware loss <before> → <after> (<N> jobs)`.

A finished hardware run **replays in the race canvas** as a car
`{id: "hardware", kind: "quantum", ghost: true, label: "hardware lap"}`,
alongside a fastsim car driving the same weights for comparison — whether
or not it completed the lap (a run that left the track replays up to that
point).

**`leaderboard`** — `{track, entries, references}`, sent on connect and
whenever the board or the track changes. `entries` are the ranked named
human race laps (`{name, lap_s, date}`, fastest first); `references` the AI
drivers' best laps (`{kind, driver, lap_s}`), shown for comparison and never
ranked. Generated and drawn tracks get a session-lifetime board that is not
persisted.

**`error`** — `{message}`; sent on the offending socket for malformed input,
broadcast for session-level failures (unknown track, training already
running, a training thread crashing, ...).

## Data flow and rates

All timing derives from `[physics] dt = 1/60` and
`substeps_per_decision = 6`:

- **60 Hz physics** — `DemoSession.tick()` advances one substep per call; the
  asyncio loop (`DemoSession.run`) drives it with drift correction (and
  resyncs rather than spiraling if it falls > 0.5 s behind). Tests call
  `tick()` directly; nothing in the session requires an event loop.
- **10 Hz decisions** — every 6th substep each agent car observes (its lidar
  rays + speed), takes `argmax Q`, and holds that action for the next 6
  substeps —
  exactly matching training. `quantum` introspection messages are throttled
  to ≤ 10 Hz per car.
- **20 Hz broadcast** — `state` messages at `[server] broadcast_hz`
  (15 on the Pi profiles). Ghost replay cars are stored at decision rate
  (10 Hz) and lerped back to 60 Hz for smooth playback.
- **10 Hz telemetry** — in train mode, `telemetry`/event messages at
  `[server] telemetry_hz` (5 on Pi). Training itself runs in **background
  threads** (`DQNTrainer` on vectorized envs); the tick loop samples live car
  states via `RacingEnv.state_snapshot()` and reads shared job stats under a
  lock — the GIL-released numpy inner loops keep the server responsive.

The session is **shared**: every websocket client sees the same world; human
input is last-writer-wins, and input resets to 0 when the last client
disconnects.

## How-to guides

### Add a track

Drop `traqmania/env/tracks/<name>.json`:

```json
{
  "name": "hairpin",
  "centerline": [[x0, y0], [x1, y1], ...],
  "half_width": 7.0,
  "checkpoints": [0.25, 0.5, 0.75],
  "theme": {"surface": "asphalt", "edge": "kerb-red"}
}
```

- `centerline`: ≥ 3 `[x, y]` points tracing a **closed loop** in world units.
  A gap between the first and last point of up to 25 % of the loop length is
  auto-closed with a straight segment; larger gaps are rejected. The polyline
  is resampled to ~`[track] resample_spacing` (1.5 units) spacing on load.
- `half_width`: ≥ 3.0 (validated).
- Minimum corner radius after resampling: **6.0 units** (validated —
  `Track.load` raises `ValueError` naming the offending value).
- `checkpoints`: optional lap fractions, strictly increasing in `[0, 1)`
  (each grants `checkpoint_bonus` reward once per lap).
- `theme`: optional, passed through to the renderer untouched.

The track appears automatically in `welcome.tracks` and the UI picker
(`runtime.available_tracks()` globs the directory). Attract/race/evolution
need weights though: train and save them with

```sh
python -m traqmania.train_headless --agent quantum --track hairpin
python -m traqmania.train_headless --agent mlp --track hairpin
python tools/make_stages.py --track hairpin   # evolution-mode snapshots
```

which writes `traqmania/weights/{quantum,mlp}_hairpin.npz` + `.meta.json`
(at a non-default qubit count, pass e.g. `--profile q6` and the quantum
filenames gain the `_q6` tag — see the filename rule above).
Slow-to-learn tracks can get a `[training_presets.hairpin]` section in
`default.toml` (merged onto `[training]` by `runtime.resolve_training_cfg`).

### Add an agent

Implement the `QFunction` protocol from `traqmania/agents/base.py` — batched
numpy, parameters in one flat vector:

```python
class MyQFunction:
    n_features: int   # lidar rays + speed (4 at the default config)
    n_actions: int    # 4 = right / straight / left / brake (6 or 8 with [circuit] n_actions)

    def q_values(self, obs):            # (B, F) -> (B, A)
    def grad_selected(self, obs, action_idx, upstream):  # -> (P,)
        # gradient of sum_b upstream[b] * Q[b, action_idx[b]] wrt flat params
    def get_params(self):               # -> (P,)
    def set_params(self, params): ...
```

`DQNTrainer` works unchanged with any implementation (the target network is
just a second flat parameter vector). An optional
`param_groups() -> dict[str, slice]` names slices of the flat vector so
`[training] lr_groups` can give them separate learning rates. To expose it in the demo, wire it into
`runtime.load_agent` and add the kind to the relevant enums in `protocol.py`
(`CAR_KINDS`, plus `OPPONENTS`/`TRAIN_AGENTS` if it should be raceable /
trainable) — and update the protocol reference above. `QuantumQFunction` and
`MLPQFunction` are the reference implementations; `HardwareQFunction` shows a
legitimate partial one (inference-only, `grad_selected` raises).

### Add a mode

1. Add the name to `MODES` in `protocol.py` (client `set_mode` validation).
2. In `session.py`, handle it in `_set_mode` and write an `_enter_<mode>`
   that builds `self.cars` (see `_enter_evolution` for a multi-car example);
   extend `tick()` if the mode needs non-default per-substep behavior.
3. Frontend: add a `<button data-mode="...">` in `index.html` and any
   mode-specific UI in `main.js` (`applyMode`).
4. Update the protocol reference above and `docs/EXHIBITION.md` talking
   points.
