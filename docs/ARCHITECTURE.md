# racetraQ architecture

How the pieces fit: a numpy physics/RL core, a FastAPI + WebSocket demo server
that ticks one shared session, a vanilla-ES-module browser frontend, and an
optional bridge to IBM Quantum hardware.

## System overview

```mermaid
flowchart LR
    subgraph browser [Browser - racetraq/web]
        main[main.js\nboot + wiring]
        net[net.js\nws client, reconnect]
        race[race.js\ncanvas renderer]
        panels[quantum-panel.js / charts.js\ncircuit.js / attract.js / explain.js]
        input[input.js\nkeys -> bitmask]
    end

    subgraph server [Server - racetraq/server]
        app[app.py\nFastAPI: /health, /ws, static]
        ws[ws.py\nHub + per-socket loop]
        proto[protocol.py\ntyped messages, strict validation]
        session[session.py\nDemoSession: 60 Hz tick,\nmode state machine]
        runtime[runtime.py\nagents/tracks/ghosts/config glue]
    end

    subgraph core [Core - racetraq]
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
| `racetraq/__main__.py` | CLI entry point: profile/config/host/port flags, starts uvicorn. |
| `racetraq/config.py` | `default.toml` + profile overlay (`pi4`, `pi5`, `exhibition`, plus the circuit-size overlays `q6`/`q8`/`q10` — each sets `[circuit] n_qubits` and the matching `[observation] ray_angles_deg`, `q8` also `n_layers = 5`) + optional extra TOML; `./config/*.toml` in the working dir shadows packaged profiles. Also `resolve_training_cfg(config, track, warm, agent)` — the one precedence rule for training recipes, shared by the server and headless training: `[training]`, then `[training_presets.<track>]`, then `[training_presets_<agent>.<track>]`, then the warm-start recipes (see "Training recipes, studies and bundled drivers") and `parse_override` / `apply_overrides` (the dotted `section.key=value` overrides behind `train_headless --set` and `tools/study.py`). |
| `racetraq/env/track.py` | Closed-loop track geometry: resampling, arc-length projection, lidar raycasts, spatial-hash acceleration, validation. |
| `racetraq/env/car.py` | Vectorized bicycle-ish car physics (throttle/brake/drag, speed-dependent steering). |
| `racetraq/env/racing_env.py` | Gym-style vector env: obs = lidar rays (`[observation] ray_angles_deg`, 3 by default) + speed, progress reward, checkpoint/lap bonuses, off-track penalty, auto-reset. `step()` info also reports `truncated` (ended by the time limit, still on track) and `final_obs` (the observation before the auto-reset) for `[training] bootstrap_truncation`. |
| `racetraq/env/trackgen.py` | Procedural track generator (numpy-only): `generate_track(seed, difficulty, length)` builds a deterministic closed loop that passes the exact `Track.load` validation (see "Random tracks" below); `LENGTH_PRESETS` maps short/medium/long to base-radius ranges and perimeter caps. |
| `racetraq/env/racing_line.py` | The model-based "hero" driver (expert demo): curvature-minimizing racing line, brake/accel-feasible speed profile from `[physics]`, pure-pursuit `RacingLineController` returning continuous (steer, throttle, brake). Not a learned agent. |
| `racetraq/env/multi_track.py` | `MultiTrackEnv`: round-robin mixture of per-track `RacingEnv`s behind the identical vector-env interface, so `DQNTrainer` trains one policy over several tracks unchanged; `random_pool` builds the mixture from generated tracks. |
| `racetraq/agents/base.py` | `QFunction` protocol + the 4 discrete actions (right/straight/left at full throttle, coast-brake — car steer +1 turns left on screen). |
| `racetraq/agents/quantum/circuit.py` | Canonical Qiskit circuit (single source of truth) + JSON `circuit_spec` for the browser diagram, including the light-cone fields `visibility`, `dead_params`, `min_layers_full_visibility`, `dead_gates`, a `live` flag on every gate and `pruned_on_hardware`. |
| `racetraq/agents/quantum/lightcone.py` | Structural light-cone analysis (numpy only): `feature_visibility` (which features each readout ⟨Z_a⟩ can depend on), `live_gates` / `live_parameter_mask` (what can influence a readout at all — the rest has exactly zero gradient), `min_layers_full_visibility`, `blind_spots` / `blind_spot_warning` (human-readable, printed by `train_headless`), `pruned_circuit` (the live gates only, same expectation values — what the hardware path runs). CLI: `python -m racetraq.agents.quantum.lightcone --qubits 10 --layers 4`. |
| `racetraq/agents/quantum/fastsim.py`, `adjoint.py` | Hand-written numpy statevector simulator and adjoint (backprop-style) gradients. |
| `racetraq/agents/quantum/noise.py` | Expectation-value noise model (numpy only at import): `ExpectationNoise` (attenuation, shots, bias — what a device does to ⟨Z_a⟩), `NoisyQFunction`, `ReadoutCorrection` (the calibrated rescale behind `HardwareQFunction(rescale=...)`), `calibrate` / `validate` against the local device patch (CLI: `python -m racetraq.agents.quantum.noise calibrate\|validate --fake fake_miami`). The model is semi-quantitative — what it gets right and wrong is measured in SCIENCE.md, "Why a driver fails under noise, and what helps". |
| `racetraq/agents/quantum/surrogate.py` | Classical Fourier surrogates of the trained circuit (numpy only): the frequency spectrum set by the trained input scalings, (`frequency_spectrum` / `spectrum_size`: the product upper bound the encoding allows), `gate_coefficients` (the exact Fourier coefficients of a readout — which of those frequencies are really used), and `fit_surrogate` (`full` least squares, random Fourier features, kernel ridge) returning a `FourierSurrogate` that can drive in place of the circuit (`drive_laps`, `compare`). Backs notebook 07 — light cones and classical surrogates. |
| `racetraq/agents/quantum/qdqn.py` | `QuantumQFunction`: fastsim-backed `QFunction`, flat `[lam, theta, w, b]` layout, P = 3·L·n + 8 params (56 at 4 qubits, 80 at 6); `param_groups()` names the `lam` / `theta` / `head` slices for per-group learning rates. |
| `racetraq/agents/quantum/qnn.py` | Same circuit via qiskit-machine-learning `EstimatorQNN` (parity checks, shots/noise backends; `aer_noisy` uses the local device twin from `hardware.local_simulator`). |
| `racetraq/agents/classical/mlp.py` | 76-parameter numpy MLP baseline (4-8-4, tanh) with analytic backprop; `param_groups()`: `body` / `head`. |
| `racetraq/agents/training/dqn.py` | Double-DQN loop over vectorized envs, Adam, replay buffer — shared by both backends. Snapshot eval (one round of `eval_episodes` distinct greedy episodes every `eval_every` episodes; `history["eval_log"]`, `final_eval`, `trainer.final_params`) and the optional `[training]` knobs `bootstrap_truncation`, `loss` / `huber_delta`, `lr_groups`, `lr_end`, `target_update` / `tau`, `grad_clip`, `reward_scale`, plus, for robustness to device noise, `act_noise` (act and select snapshots under an `ExpectationNoise`; TD targets and gradients stay exact) and `action_gap` (advantage learning). In the trainer every knob is off unless the recipe sets it; `default.toml` turns `bootstrap_truncation` on for everyone and `act_noise` / `action_gap` on for the quantum agent on oval and chicane. |
| `racetraq/agents/training/spsa.py` | Minimal SPSA minimizer used by hardware sprints: two loss evaluations per iteration, plus the optional safeguards the sprint turns on — per-parameter `scale` (0 freezes a parameter), a `max_step` trust region, `blocking` (reject a step the loss does not confirm), `accept` (a veto on a proposed point, costing no loss evaluation) and `calibrate_gain`. |
| `racetraq/hardware.py` | IBM Quantum via `qiskit-ibm-runtime`, real or simulated. `get_backend` (real QPU, or a fake by name — default `fake_miami`, a Nighthawk calibration snapshot; unknown names raise), `local_simulator` / `execution_backend` (the Aer twin of a fake, built once: fakes of up to 7 qubits whole, larger devices — the 120–156-qubit ones above all — as a *device patch* of just the physical qubits the routed circuit touches), `open_execution_mode` (Session → Batch → job fallback, with the reason), `HardwareQFunction` (inference-only; light-cone-pruned ISA circuit, client-side `executor_estimator.Estimator` with `EstimatorV2` fallback, `resilience_level` 0/1/2, optional calibrated attenuation `rescale`), `run_hardware_lap`, `spsa_sprint` (by default: output head only, blocking, and a guard that vetoes steps costing more than 10 % of the exact-simulator greedy return; TD targets follow the recipe in the weights' sidecar; the result lists `accepted` and `vetoed` per iteration, and `sprint_steps_text` words them). CLI: `python -m racetraq.hardware lap\|sprint [--track T] [--profile q6] [--fake] [--fake-name NAME] [--backend NAME] [--weights W.npz] [--shots N] [--resilience 0\|1\|2] [--rescale off\|global\|readout] [--no-prune]`, plus `--max-decisions N` for a lap and `--iterations N --batch N --groups lam,theta,head --no-blocking --no-guard` for a sprint. `[hardware]` config: `backend_name`, `fake_name`, `shots`, `decision_shots`, `spsa_iterations`, `spsa_groups`, `spsa_blocking`, `spsa_guard`, `resilience_level`, `prune_light_cone`, `rescale`, `calibration_samples`, `calibration_shots`. |
| `racetraq/server/protocol.py` | Typed WS messages; strict client-side validation (`ProtocolError`). |
| `racetraq/studio.py` | Training studio, the pure part: sensor presets (`lidar`: n−1 rays + speed, what every bundled driver and study uses; `corner`: the rays + `corner_speed_ratio`), `studio_config` (the q<n> profile with the run's sensors and action set), `option_problem`, `catalog` (setup options, study numbers and time estimates per track × size), `compare` (a run against the study runs), the per-track booth board (`studio_<track>.json` in the leaderboard dir; ranked by the best test's lapped share, then mean lap). |
| `racetraq/server/studio.py` | `StudioController`, the session side of studio mode: phases setup → training → done → race / watch; starts the quantum job under the studio config with a random seed, follows the trainer's periodic tests, stops at `[studio] time_limit_s` (300) or once the best test laps every drive and six more tests bring no improvement; files named runs that lapped; gives the session its profile config back when another mode takes over and re-applies the run's config and track when the model drives again. |
| `racetraq/data/studio_stats.json` | Generated by `tools/export_studio.py`: per (track, qubits) the closest study runs (first-lap episodes, best tests) and the live training speed per size measured through the server's own training path (`--calibrate`, stored in `data/studies/studio_calibration.json`). |
| `racetraq/server/session.py` | `DemoSession`: the mode state machine and synchronous 60 Hz `tick()`; training threads; ghost recording. |
| `racetraq/server/runtime.py` | Loading bundled agents/weights/tracks/ghosts, track payloads; what a weights file brings along (`weights_observation`, `weights_actions`, `weights_circuit`: its circuit depth and action count; `with_weights_config` overlays all of it on a config); re-exports `config.resolve_training_cfg`. |
| `racetraq/server/ws.py` | Connection `Hub`, broadcast fan-out, per-socket receive loop, `DriverLock` (exclusive control, spectators watch). |
| `racetraq/server/app.py` | FastAPI factory: `/health`, `/ws`, `/api/docs` + `/api/docs/{id}` (repo markdown for the in-UI docs browser; empty outside a source checkout) and `/docs-assets` (images), static frontend mounted last. |
| `racetraq/train_headless.py` | Offline training CLI that produces the bundled `weights/*.npz` (+ `.meta.json`, history JSON). The recipe is `[training]` with the track's `[training_presets.<track>]` and then the agent's `[training_presets_<agent>.<track>]` merged on top (`--preset none` skips both); `--set section.key=value` overrides any config value (an unknown `training.` key is an error), `--episodes` / `--seed` win over everything, and `--save-final` also writes `<name>.final.npz`, the end-of-training parameters. The sidecar records the circuit shape, observation, action count and the resolved training table. Besides the bundled names, `--track multi` trains one policy on the oval+chicane+gp+combo mixture and `--track random` on a `MultiTrackEnv.random_pool` of generated tracks (seeded from `--seed`); weights save under the literal names (`quantum_multi.npz` / `quantum_random.npz`) — the universal-driver candidates. |
| `racetraq/records.py` | `python -m racetraq.records [--episodes N] [--seed S] [--drivers a,b] [--tracks x,y] [--out FILE]`: greedy evaluation of every bundled driver on every bundled track (each under its own recorded observation, depth and action count) into `data/records.json`. |
| `racetraq/bench.py` | Micro-benchmarks (env steps, forward passes, DQN updates). |
| `tools/study.py` | Multi-seed study harness. `run` trains a (variant × seed) grid, one subprocess per cell, resumable; each cell saves best-snapshot and final weights and evaluates both over 36 distinct greedy episodes. `report` aggregates over seeds: IQM and median with bootstrap confidence intervals, stability, sample complexity, probability of improvement over a baseline (`report.md`, `report.json`). |
| `tools/bundle_driver.py` | From a study to a bundled driver: ranks a variant's seeds by the study's 36-episode eval, re-runs that eval for the top candidates (the check that this checkout still drives the weights as the study did; `--skip-replay`), re-evaluates them on 72 fresh episodes (`--eval-seed`, refused when it coincides with a seed the study used), optionally drives them on the simulated device (`--device-episodes N`, ranking first with `--rank-by device`), and writes `<name>.npz` plus a sidecar with a `selection` block. `--list` prints a study's variants with their seed spreads; `--dry-run` writes nothing; a driver below `--min-lapped` (0.9) or an existing target needs `--force` / `--overwrite`. |
| `tools/export_study.py` | `python tools/export_study.py STUDY_DIR --name NAME [--out data/studies]`: the committable summary of a study — `report.md`, `report.json`, and `cells.json` with one record per finished cell (variant, seed, overrides, parameter count, first clean lap, the in-training eval log, both 36-episode evals, mean return per 100 episodes, wall time). |
| `data/studies/<name>/` | Those summaries for every study behind SCIENCE.md's "Measured results" (the weights and full logs of the runs are not in the repository). |
| `tools/hw_reliability.py` | Lap completion and decisions-until-crash of one weights file under device noise: many emulated episodes (the `noise.py` model) plus a few on the local device patch, across shots (`--shots`, default 1024,4096,16384), rescale settings (`--rescale`, default off,readout) and resilience levels (`--resilience`, default 0,1); `--device-episodes N` sets the device-path sample (default 3 — use 8 or more to judge a driver), `--no-device` emulates only. Emulated rescale rows are optimistic, and "mean lap" is not comparable between emulated and device rows (different decision caps). |
| `tools/make_stages.py` | Evolution-stage weights and the warm-start checkpoint, as earlier snapshots of the bundled driver's own training run: replays that run from the driver's sidecar (seed, episodes, recipe, depth, observation) and writes nothing unless the replay's best snapshot equals the driver parameter for parameter (`--allow-mismatch` overrides; `--fresh` trains the config's recipe instead). Saves 4 snapshots that improve from one to the next and end on the driver, `quantum_<track>_stage{1..4}[_q<n>].npz` (+ `.meta.json` with the episode count shown as the car label), and `quantum_<track>_warmstart[_q<n>].npz`, the last snapshot that does not lap before the run starts lapping in half of an eval's episodes. |
| `racetraq/web/` | Frontend ES modules (`main`, `net`, `race`, `input`, `charts`, `circuit`, `quantum-panel`, `hardware-panel`, `studio-panel`, `attract`, `explain`, `docs`, `md`, `draw`, `tooltip`); no build step, served statically. |

## Qubit-count profiles (q6 / q8 / q10)

The quantum stack is generalized over `[circuit] n_qubits`. Qubits map 1:1 to
observation features — (n − 1) lidar rays evenly spaced over [−60°, +60°]
plus normalized speed — while the **4 actions and the Z_0…Z_3 readout stay
fixed** (extra qubits only widen the feature register). The overlays set
exactly that pair of config values, and `q8` the circuit depth as well:

- `--profile q6` — 6 qubits, 5 rays; trained oval and chicane weights are
  bundled (`weights/quantum_oval_q6.npz`, `quantum_chicane_q6.npz`).
- `--profile q8` / `q10` — 8/10 qubits, 7/9 rays (and, at `q8`, 5
  re-uploading blocks); oval and chicane weights are bundled at both sizes,
  plus `quantum_gp_q10.npz` (trained on an engineered-feature observation
  recorded in its sidecar). On a track
  without weights at the active size, the modes that need them stay
  unavailable until you train, e.g.
  `python -m racetraq.train_headless --agent quantum --profile q8 --track gp`.

**Depth and the light cone.** Each readout only depends on features within
ring distance `n_layers − 1` of its qubit (`agents/quantum/lightcone.py`;
SCIENCE.md, "Light cones"), so 4 blocks show every feature to every action
at 4 and 6 qubits, but 8 qubits need 5 blocks and 10 need 6. The default
config and `q6` keep `[circuit] n_layers = 4`; **`q8` sets 5** (the October
2026 study, 6 seeds per depth: with 5 blocks the end-of-training parameters
lap more often on the oval — the one supported difference of ten — and
everything else points the same way without support; the bundled 8-qubit
drivers are 5-block, 128-parameter files); `q10` has 6 (full visibility;
on the oval 6 blocks pointed the same way as the fifth block at 8 qubits
without the difference being supported, on chicane the depths are
indistinguishable — `data/studies/oval_q10`, `chicane_q10`), and its
bundled oval and chicane drivers are 6-block, 188-parameter files;
`quantum_gp_q10` is still a 4-block file, which the loaders read from its
parameter count.
`circuit_spec` reports the visibility, and `train_headless` warns before
training a circuit with blind spots (live training writes the same warning
to the server log). To train with full visibility at 10 qubits: `--set
circuit.n_layers=6`. Weight *filenames*
do not carry the depth (use `--out` to keep two depths apart), but a driver
brings its own: every loader builds the circuit a weights file needs
(`server/runtime.weights_circuit`) — the sidecar's `circuit` block
(`n_qubits`, `n_layers`, `n_actions`) when there is one, else the depth its
parameter count implies, `P = 3·L·n + 2·A` (sidecars written before the
block existed: the July files). The session adopts that
depth per driver exactly like the recorded observation and action count —
attract, race, hardware, driver / track / qubit switches; evolution builds
each stage car at its own file's depth — so drivers of different depth can
sit side by side at one qubit count, and `circuit_spec` in the `welcome`
(depth, gates, visibility, dead parameters) and the hardware job's
light-cone-pruned circuit are the active driver's. A `hero` / `pro` pick
only replaces the attract car: in race, evolution and hardware mode the
session syncs to the quantum car that drives there. Live training always
builds the profile's `[circuit] n_layers` (warm-start weights of another
depth are skipped with an `error`: cold start). Outside the session the
same rules apply (`runtime.with_weights_config`): the hardware lap / sprint
and their CLI, `agents.quantum.noise validate`, `tools/hw_reliability.py`
and `records` run a weights file under its recorded observation and at its
own depth and action count (the CLIs print one line when the sidecar's
observation differs from the profile's); `train_headless --init` and
`tools/make_stages.py --init` continue at the init weights' depth and
action count, and an explicit `--set circuit.n_layers` / `--actions` that
contradicts them is an error. Only a parameter count that fits no depth
`L ≥ 1` (or a sidecar naming another qubit count) is still refused: an
`error` naming the file and the count, car-less attract mode.

The default stays 4 qubits and is bit-identical to the pre-scaling stack
(pinned by a regression test). The evolution stages and the bundled MLP
(agent and race opponent) are 4-qubit-only: at n ≠ 4 the session rejects
those modes/opponents with an `error` message instead of crashing (evolution
comes back if you produce `_stage<i>` weights at that qubit count).

Weight filenames follow one rule (`server/session.quantum_weights_path`):
`quantum_<track>[_warmstart|_stage<i>][_q<n>].npz` — the `_q<n>` tag is
appended at any non-default qubit count, so a `--profile q6` run reads and
writes `quantum_<track>_q6.npz` and never clobbers the 4-qubit weights.

## Training recipes, studies and bundled drivers

**One recipe rule.** `config.resolve_training_cfg(config, track, warm,
agent)` merges, later layers winning:

1. `[training]` — the base (400 episodes, epsilon 1.0 → 0.05 over 250
   episodes, `bootstrap_truncation = true`);
2. `[training_presets.<track>]` — gp (3000 episodes, epsilon decay over
   2000, gamma 0.99) and combo (2500 / 1500 / 0.99);
3. `[training_presets_<agent>.<track>]` — the per-agent layer, today only
   for `quantum`: `action_gap = 0.8` and `act_noise = { attenuation = 0.95,
   shots = 1024 }` on oval and chicane, `epsilon_end = 0.30` on gp and
   combo. The MLP has no such table and keeps `epsilon_end = 0.05`: on 10
   seeds the MLP tends to do worse with the 0.30 floor on gp — a consistent
   trend, not a supported difference (SCIENCE.md, "Training stability");
4. with `warm`: `[training_warm]`, plus `[training_warm_<track>]` where one
   exists (gp, combo).

Live training in the session and `train_headless` both call it with the
agent. `train_headless --preset none` skips layers 2 and 3, `--set
training.<key>=…` wins over all of them, and the resolved table is what the
sidecar's `training` block records.

**From a study to the weights directory.**

```
tools/study.py run     -> <study>/cells/<variant>/seed<k>/   weights, final weights, history, result.json
tools/study.py report  -> <study>/report.md, report.json     statistics over seeds
tools/export_study.py  -> data/studies/<name>/               report.md, report.json, cells.json (committed)
tools/bundle_driver.py -> racetraq/weights/<name>.npz + <name>.meta.json
```

**The sidecar** (`<name>.meta.json`). Loaders read three blocks — `circuit`
(`n_qubits`, `n_layers`, `n_actions`), `observation` and `actions` — and
the hardware sprint reads `training` for its TD targets. The rest documents
where the driver came from: `provenance` (one paragraph) and, for every
driver written by `tools/bundle_driver.py`, a `selection` block:

| Key | Content |
|---|---|
| `study`, `variant`, `profile`, `overrides` | which recipe of which study |
| `study_eval` | episodes and env seed of the study's own 36-episode eval that shortlisted the seeds |
| `chosen_seed`, `n_seeds`, `rank_by`, `rule` | the seed taken, out of how many, by which rule (`fresh` or `device`) |
| `source`, `weights_sha256` | the cell file the weights were copied from, byte for byte |
| `candidates` | every shortlisted seed with its study, fresh and device numbers |
| `seed_spread` | the recipe over all its seeds: IQM, 95 % interval and median of best-snapshot lapped, final-params lapped and stability; `seeds_lapping_half` |
| `fresh_eval` | the chosen seed on fresh episodes (72; 36 per track for a multi-track driver, under `tracks`): `lapped`, `mean_lap`, `best_lap`, `eval_seed` |
| `device_eval` | when it ran: fake backend and patch, shots, rescale, resilience, episodes, `lapped`, `crashed`, two-qubit gates |

Files that predate the tool (the 10-qubit drivers) have no `selection`
block, and a sidecar without a `circuit` block gets its depth from the
parameter count ("Depth and the light cone" above).

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
  specialist `quantum_gp[_q<n>].npz`. The car is labelled honestly:
  `driver: universal` or `driver: gp-trained generalist`
  (`session.random_track_weights`). Warm starts and evolution stages follow
  the same chain (`_warmstart` / `_stage<i>` suffixes). **As bundled in
  October 2026 the first link is the weak one**: `quantum_universal` laps
  the four bundled tracks but none of ten generated tracks (0 of 120
  episodes at difficulty 0.5 and at the 0.65 the server generates), while
  `quantum_gp` laps all ten (120 of 120 at both) and, zero-shot, oval and
  chicane (36 of 36 each) and combo (16–22 of 36) — SCIENCE.md, "One driver,
  every track". Until the universal driver is re-selected, `set_driver
  {driver: "gp"}` is the working choice on generated and drawn tracks. At
  10 qubits there is no universal file and the fallback is the July
  `quantum_gp_q10`, which laps neither oval nor chicane (0 of 36 each).
- **No ghost persistence** — best-lap ghosts are never written for random
  tracks (`ghosts_dir` only ever holds bundled-track files).
- **Graceful rejections** — hardware mode needs a bundled track name (the
  runner loads tracks by name); evolution/mlp modes reject with the existing
  missing-weights `error` when their files are absent.

The universal-driver *candidates* are trained offline with
`train_headless --track multi` (bundled-track mixture) or `--track random`
(generated pool via `MultiTrackEnv.random_pool`). The bundled
`quantum_universal.npz` was promoted from a `--track multi` study with
`tools/bundle_driver.py`, which ranks seeds on the four bundled tracks
only — not on generated ones (see the fallback chain above).

## WebSocket protocol reference

> **Keep in sync:** this section is hand-generated from
> `racetraq/server/protocol.py`. If you change the protocol, change this
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
are not recorded). Switching to attract mode (`set_mode`) clears it, so the
next visitor starts anonymous.

**`idle_reset`** — `{}`: the booth went idle. Back to attract mode with the
booth defaults: the name is cleared, the studio forgets its model and
returns to setup, and with `[ui] kiosk = true` the startup track, qubit
count and driver come back (`track` / `welcome` are re-broadcast). Ignored
while training, a hardware job or its replay, the evolution show or a
studio run is on. Only the browser holding the wheel sends it (after
`attract_idle_seconds` without input, or 3× that, at least 90 s, while a
studio result is shown); when nobody holds the wheel, the server's 1 Hz
control ticker applies the same reset after the same time.

**`studio`** — the training studio (mode `studio`).

| field | type | required | meaning |
|---|---|---|---|
| `action` | `"start"` \| `"stop"` \| `"race"` \| `"watch"` \| `"result"` \| `"setup"` | yes | train / end the run early (best snapshot so far becomes the model) / race your model / let it drive alone / take it off the track / back to the setup screen |
| `track` | str | no | start: one of oval, chicane, gp, combo (default: the current track) |
| `qubits` | int | no | start: 4, 6, 8 or 10 (default 4) |
| `sensors` | `"lidar"` \| `"corner"` | no | start: sensor preset (default lidar) |
| `actions` | int | no | start: 4, 6 or 8, at most `qubits` (default 4) |
| `warm` | bool | no | start: warm start from the bundled checkpoint (lidar, 4 actions, where one exists) |

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
everyone after a `qubits` or `set_driver` switch, and whenever the active
driver's observation, action count or circuit depth differs from the
previous one's):
`{mode, track: TrackPayload, tracks: [str], circuit_spec, ui, obs_labels,
driver, drivers}`. `driver` is the active `set_driver` selection (`"auto"`
default) and `drivers` the currently valid choices (`"auto"`, each bundled
training at the active qubit count, `"hero"`, and `"pro"` when
`mlp_pro.npz` is bundled). `qubit_options` lists the circuit sizes with a
trained quantum driver for the current track (the UI greys out the rest).
`circuit_spec` is the JSON gate-by-gate description of the active driver's
circuit (the profile's while training or without a quantum driver) from
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

**`track`** — `{track: TrackPayload, qubit_options: [int]}` after a successful
track switch.

*TrackPayload* (built in `runtime.track_payload`): `name`, `half_width`,
`total_length`, `checkpoints` (fractions of a lap), `theme` (free-form dict
from the track JSON), `start: {x, y, theta}`, and polylines `centerline`,
`left`, `right` as `[[x, y], ...]`.

**`state`** — `{t: float, mode, cars: [CarState]}` at the broadcast rate,
plus `countdown: float` (seconds to GO) while a race start counts down:
`race start` and `race reset` hold every car on the grid for 3 s (visitor
on the right, opponent on the left), the lap clocks and the ghost's lap
start at GO.

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
| `lap_t` | float | running time of the current lap (0 during a race countdown) |

**`quantum`** — live circuit introspection for a quantum car, throttled to
≤ 10 Hz per car: `{car_id, expectations: [float], q_values: [float],
action: int}` (`expectations` are the raw ⟨Z_a⟩ readouts).

**`telemetry`** — training progress per agent at the telemetry rate:
`{agent, episode, mean_return, epsilon, loss: float|null,
returns_tail: [float]}` (last ≤ 100 episode returns), plus *omitted-if-null*
`best_lap_s: float` and `lap_times: [[episode, lap_s], ...]` (last ≤ 50).

**`event`** — `{kind, car_id?, lap_time?, agent?}` with `kind` ∈
`lap | crash | clean_lap | timeout | training_done | new_best_lap | lap_result`
(optional fields *omitted-if-null*; `lap_result` ends each lap of the
visitor in Race mode and adds `clean: bool`, `rank: int` — the board place
the lap took, or would take without a name; null for a lap that left the
track or misses the board — `named: bool` and `board_size: int`. A visitor
who leaves the track is put back on the centerline a few metres behind
after 1 s and the lap goes on, marked dirty; `new_best_lap` carries `agent` during training and
`car_id` for ghost records). `timeout`: an agent car went `[reward] max_decisions`
decisions (60 s) without finishing a lap — the training env's episode cap — and
was respawned at the start line.

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

**`studio`** — training-studio state, on every phase change and twice a
second while training: `{phase, time_limit_s, spec?, elapsed_s?, live?,
result?, rank?, name?, board?, catalog?}`. `phase` ∈ `setup | training |
done | race | watch`; `spec` the run's choices plus `n_params`, `n_layers`
and `studied` (whether the studies cover it); `live` `{episode, first_lap,
best_lap_s, last_eval, best_test}`; `result` `{episodes, seconds,
first_lap, best_eval, stop_reason (time | converged | stopped | error),
error, comparison}`; `board` `{track, entries}` (top 10); `catalog` (on
phase changes) the setup options with, per `<track>_q<n>`, the study
numbers, a time estimate and whether a warm start exists.

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

Drop `racetraq/env/tracks/<name>.json`:

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
python -m racetraq.train_headless --agent quantum --track hairpin
python -m racetraq.train_headless --agent mlp --track hairpin
python tools/make_stages.py --track hairpin   # evolution stages + warm-start checkpoint (replays the quantum run above)
```

which writes `racetraq/weights/{quantum,mlp}_hairpin.npz` + `.meta.json`
(at a non-default qubit count, pass e.g. `--profile q6` and the quantum
filenames gain the `_q6` tag — see the filename rule above).
Slow-to-learn tracks can get a `[training_presets.hairpin]` section in
`default.toml`, and an agent-specific `[training_presets_quantum.hairpin]`
or `[training_presets_mlp.hairpin]` on top (merged onto `[training]` in
that order by `config.resolve_training_cfg`).

### Add an agent

Implement the `QFunction` protocol from `racetraq/agents/base.py` — batched
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
