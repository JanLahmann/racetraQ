"""Hardware reliability of a trained driver: does it still lap under device noise?

A driver that laps on the exact simulator can crash within a few dozen
decisions once its readout expectations come from a noisy device.  This tool
measures that for one weights file and track — lap completion and
decisions-until-crash — on two paths:

- **emulation**: fastsim plus the expectation-noise model of
  ``traqmania.agents.quantum.noise`` (a slope and a bias per readout, fitted
  to the device once, plus shot noise) — cheap, so it runs many distinct
  episodes (default 36);
- **device**: the real local device-patch path (``traqmania.hardware`` on the
  Aer twin of a fake backend, ~0.1-0.2 s per decision) — a few episodes,
  capped at ``--device-max-decisions``.

Both across per-decision shots, the attenuation rescale (off | global |
readout — ``HardwareQFunction(rescale=...)``, emulated with a calibration of
the same size) and the Estimator's resilience level::

    python tools/hw_reliability.py --weights W.npz --track oval \\
        --shots 1024,4096,16384 --rescale off,global,readout --resilience 0,1 \\
        --out reliability.json

It prints a markdown table and writes every number as JSON to ``--out``.
Two things to keep in mind when reading it.  The emulated rescale undoes
exactly the slope-and-bias model the emulation itself is built from, so its
rows are optimistic about what the rescale does: on the device path the
state-dependent part of the error stays (the bundled oval driver at 16384
shots with the readout rescale lapped 21 of 36 emulated episodes and 0 of 8
on the device path, 2026-10-01).  And "mean lap" averages every lap an
episode drove, so the emulated rows (up to the env's time limit: the first
lap from the spawn plus flying laps) read faster than the device rows (capped
soon after the first lap).
``--no-device`` skips the device path and emulates ``--attenuation`` /
``--bias`` instead of a fitted model (no qiskit needed); ``--model
earlier.json`` reuses the fitted models of an earlier run.

Only numpy and the standard library are imported at module level;
``traqmania`` (and, for the device path, qiskit) is loaded lazily.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1

EPISODES = 36  # distinct emulated episodes per setting (records.py's sample)
EVAL_SEED = 20_000  # env seed of the episodes (study.py's reliability eval)
DEVICE_EPISODES = 3
DEVICE_MAX_DECISIONS = 200  # one oval lap is ~160 decisions
MODEL_SHOTS = 65_536  # shots of the job that fits the emulation's device model
MODEL_SAMPLES = 64
RESCALE_CHOICES = ("off", "global", "readout")


def _find_traqmania() -> None:
    if importlib.util.find_spec("traqmania") is None:  # checkout without an install
        sys.path.insert(0, str(REPO_ROOT))


def _int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _float_list(text: str) -> float | list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    return values[0] if len(values) == 1 else values


def weights_config(weights: Path, profile: str | None) -> dict:
    """The config the weights drive under: the profile, then the observation
    of the weights' ``.meta.json`` sidecar and the circuit shape the weights
    need (``runtime.weights_circuit``: the sidecar's ``circuit`` block, else
    the depth their parameter count implies at the profile's qubit count).
    Raises ``ValueError`` for weights that fit no depth."""
    from traqmania.config import load_config
    from traqmania.server.runtime import weights_circuit, with_weights_observation

    config = with_weights_observation(load_config(profile), weights)
    circuit = config.setdefault("circuit", {})
    circuit.update(weights_circuit(weights, int(circuit.get("n_qubits", 4)),
                                   circuit.get("n_actions")))
    return config


# ------------------------------------------------------------------ rollouts


def rollout(qfunc: Any, env: Any, max_decisions: int | None = None) -> dict:
    """One greedy episode per sub-env, all in parallel, each frozen at its
    first done (crash or time limit) or at ``max_decisions``.

    Only the envs still running are evaluated (a device job costs per row).
    Returns per-episode arrays ``decisions`` (survived), ``crashed``,
    ``lapped``, ``laps`` and the list of ``lap_times``.
    """
    obs = env.reset()
    n_envs = obs.shape[0]
    cap = int(env.max_decisions if max_decisions is None else max_decisions)
    alive = np.ones(n_envs, dtype=bool)
    crashed = np.zeros(n_envs, dtype=bool)
    decisions = np.zeros(n_envs, dtype=np.int64)
    laps = np.zeros(n_envs, dtype=np.int64)
    prev_lap = np.zeros(n_envs, dtype=np.int64)
    lap_times: list[float] = []
    for _ in range(cap):
        actions = np.zeros(n_envs, dtype=np.int64)
        actions[alive] = np.argmax(qfunc.q_values(obs[alive]), axis=1)
        obs, _reward, done, info = env.step(actions)
        done = np.asarray(done, dtype=bool)
        decisions[alive] += 1
        lap = np.asarray(info["lap"], dtype=np.int64)
        event = (lap > prev_lap) & alive
        laps[event] += 1
        times = np.asarray(info["last_lap_time"], dtype=np.float64)
        lap_times.extend(float(t) for t in times[event & ~np.isnan(times)])
        prev_lap = np.where(done, 0, lap)
        crashed |= done & alive & np.asarray(info["off_track"], dtype=bool)
        alive &= ~done
        if not alive.any():
            break
    return {"decisions": decisions, "crashed": crashed, "lapped": laps > 0, "laps": laps,
            "lap_times": lap_times}


def summarize(result: dict) -> dict:
    """Episode counts and decision statistics of one :func:`rollout`."""
    decisions = result["decisions"]
    crashed = result["crashed"]
    return {
        "episodes": int(decisions.size),
        "lapped": int(result["lapped"].sum()),
        "crashed": int(crashed.sum()),
        "laps": int(result["laps"].sum()),
        "decisions_median": float(np.median(decisions)),
        "decisions_min": int(decisions.min()),
        "crash_decisions_median": float(np.median(decisions[crashed])) if crashed.any()
        else None,
        "mean_lap": float(np.mean(result["lap_times"])) if result["lap_times"] else None,
    }


def _make_env(track_name: str, config: dict, n_envs: int, seed: int):
    from traqmania.env.racing_env import RacingEnv
    from traqmania.env.track import Track

    track = Track.load(track_name, config["track"]["resample_spacing"])
    return RacingEnv(track, config, n_envs=n_envs, seed=seed)


# ---------------------------------------------------------------- the paths


def fit_device_model(args: argparse.Namespace, config: dict, params: np.ndarray,
                     resilience: int) -> dict:
    """Calibration fit of the fake device for this driver (one high-shot job)."""
    from traqmania.agents.quantum import noise

    circuit = config["circuit"]
    return noise.calibrate(
        args.fake, int(circuit["n_qubits"]), int(circuit.get("n_layers", 4)),
        circuit.get("n_actions"), samples=MODEL_SAMPLES, shots=args.model_shots,
        seed=args.seed, params=params, resilience_level=resilience,
        seed_simulator=args.seed,
    )


def emulation_row(fast: Any, config: dict, args: argparse.Namespace, model: dict,
                  shots: int, rescale: str, resilience: int | None, row_seed: int) -> dict:
    """``args.episodes`` emulated episodes under ``model`` at ``shots`` shots."""
    from traqmania.agents.quantum.noise import ExpectationNoise, NoisyQFunction

    device_noise = ExpectationNoise(attenuation=model["attenuation"], shots=shots,
                                    bias=model["bias"])
    qfunc = NoisyQFunction(fast, device_noise, rng=np.random.default_rng([args.seed, row_seed]))
    correction = None
    if rescale != "off":
        # the mitigation's own calibration job, seen through the same noise
        qfunc.calibrate(rescale, samples=args.calibration_samples,
                        shots=max(shots, args.calibration_shots), seed=args.seed)
        correction = {"slope": qfunc.correction.slope, "bias": qfunc.correction.bias}
    env = _make_env(args.track, config, args.episodes, args.eval_seed)
    return {
        "path": "emulation", "resilience": resilience, "rescale": rescale, "shots": shots,
        **summarize(rollout(qfunc, env, args.max_decisions)),
        "noise": {"attenuation": model["attenuation"], "bias": model["bias"]},
        "correction": correction,
    }


def device_row(config: dict, params: np.ndarray, args: argparse.Namespace, backend: Any,
               shots: int, rescale: str, resilience: int, row_seed: int) -> dict:
    """``args.device_episodes`` episodes on the local device-patch path."""
    from traqmania import hardware

    hw = hardware.HardwareQFunction(config["circuit"], backend, shots=shots,
                                    resilience_level=resilience,
                                    rescale=None if rescale == "off" else rescale)
    hw.set_params(params)
    hw.seed_simulator(args.seed + row_seed)
    correction = None
    if rescale != "off":
        hw.calibrate(samples=args.calibration_samples,
                     shots=max(shots, args.calibration_shots), seed=args.seed)
        if hw.correction is not None:
            correction = {"slope": hw.correction.slope, "bias": hw.correction.bias}
    env = _make_env(args.track, config, args.device_episodes, args.eval_seed)
    t0 = time.perf_counter()
    result = rollout(hw, env, args.device_max_decisions)
    seconds = time.perf_counter() - t0
    jobs = int(result["decisions"].max())
    return {
        "path": "device", "resilience": resilience, "rescale": rescale, "shots": shots,
        **summarize(result),
        "backend": hw.backend_name,
        "two_qubit_gates": hw.two_qubit_gates,
        "max_decisions": int(args.device_max_decisions),
        "seconds_per_job": seconds / max(1, jobs),
        "correction": correction,
        "rescale_note": hw.rescale_note,
    }


# -------------------------------------------------------------------- report


def _cell(value: Any, fmt: str = "{}") -> str:
    return "—" if value is None else fmt.format(value)


def markdown(report: dict) -> str:
    """The report as a markdown table plus the fitted device models."""
    lines = [
        f"### Hardware reliability: {Path(report['weights']).name} on {report['track']}",
        "",
        "| path | resilience | rescale | shots | episodes | lapped | crashed | decisions "
        "(median / min) | crash at (median) | mean lap |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['path']} | {_cell(row['resilience'])} | {row['rescale']} | "
            f"{_cell(row['shots'])} | {row['episodes']} | {row['lapped']} | {row['crashed']} | "
            f"{row['decisions_median']:.0f} / {row['decisions_min']} | "
            f"{_cell(row['crash_decisions_median'], '{:.0f}')} | "
            f"{_cell(row['mean_lap'], '{:.1f} s')} |"
        )
    lines.append("")
    lines.append(f"Emulated episodes run up to {report['max_decisions']} decisions, device "
                 f"episodes up to {report['device_max_decisions']} "
                 "(\"lapped\": episodes that completed a lap within that).")
    for level, model in report["models"].items():
        source = f"fitted to {model['backend']}" if model.get("backend") else "given"
        slope = model["attenuation"]
        slope = ", ".join(f"{v:.3f}" for v in slope) if isinstance(slope, list) else f"{slope:.3f}"
        bias = model["bias"]
        bias = ", ".join(f"{v:+.4f}" for v in bias) if isinstance(bias, list) else f"{bias:+.4f}"
        label = "" if level == "given" else f" at resilience {level}"
        lines.append(f"Emulation model{label} ({source}): slope [{slope}], bias [{bias}]"
                     + (f", unexplained rms {model['excess_rms']:.4f}"
                        if "excess_rms" in model else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/hw_reliability.py",
        description="Lap completion and decisions-until-crash of a trained quantum driver "
                    "under device noise: fast emulation and the local device-patch path.",
    )
    parser.add_argument("--weights", required=True, help="weights .npz of a quantum driver")
    parser.add_argument("--track", default="oval")
    parser.add_argument("--profile", default=None, help="config profile overlay (e.g. q6)")
    parser.add_argument("--fake", default="fake_miami", help="fake backend, any spelling")
    parser.add_argument("--shots", type=_int_list, default=[1024, 4096, 16384],
                        help="per-decision shots, comma separated (default 1024,4096,16384)")
    parser.add_argument("--rescale", default="off,readout",
                        help="attenuation rescale settings, comma separated out of "
                             f"{', '.join(RESCALE_CHOICES)} (default off,readout; global "
                             "divides by one fitted attenuation, readout corrects each "
                             "readout's own slope and bias)")
    parser.add_argument("--resilience", type=_int_list, default=[0, 1],
                        help="Estimator resilience levels, comma separated (default 0,1)")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help=f"distinct emulated episodes per setting (default {EPISODES})")
    parser.add_argument("--max-decisions", type=int, default=None,
                        help="cap on emulated episodes (default: the env's time limit)")
    parser.add_argument("--device-episodes", type=int, default=DEVICE_EPISODES,
                        help=f"episodes per setting on the device path (default "
                             f"{DEVICE_EPISODES}; 0 skips it but still fits the model)")
    parser.add_argument("--device-max-decisions", type=int, default=DEVICE_MAX_DECISIONS,
                        help=f"cap on device episodes (default {DEVICE_MAX_DECISIONS})")
    parser.add_argument("--no-device", action="store_true",
                        help="emulation only, from --attenuation / --bias (no qiskit)")
    parser.add_argument("--attenuation", type=_float_list, default=0.95,
                        help="with --no-device: attenuation, one value or one per readout "
                             "(default 0.95)")
    parser.add_argument("--bias", type=_float_list, default=0.0,
                        help="with --no-device: additive bias, one value or one per readout")
    parser.add_argument("--model", default=None,
                        help="reuse the fitted device models of an earlier --out JSON")
    parser.add_argument("--model-shots", type=int, default=MODEL_SHOTS,
                        help=f"shots of the model-fitting job (default {MODEL_SHOTS})")
    parser.add_argument("--calibration-samples", type=int, default=None,
                        help="inputs of the rescale's calibration job (default: hardware's)")
    parser.add_argument("--calibration-shots", type=int, default=None,
                        help="minimum shots of that job (default: hardware's)")
    parser.add_argument("--seed", type=int, default=0, help="noise / simulator seed")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help=f"env seed of the episodes (default {EVAL_SEED})")
    parser.add_argument("--out", default=None, help="write the report as JSON here")
    return parser


def run(args: argparse.Namespace) -> dict:
    """Run every requested setting; returns the report dict."""
    _find_traqmania()
    from traqmania import hardware
    from traqmania.agents.quantum.qdqn import QuantumQFunction

    rescales = [item.strip() for item in args.rescale.split(",") if item.strip()]
    unknown = sorted(set(rescales) - set(RESCALE_CHOICES))
    if unknown:
        raise ValueError(f"--rescale: unknown setting(s) {unknown}; choose from "
                         f"{list(RESCALE_CHOICES)}")
    if args.calibration_samples is None:
        args.calibration_samples = hardware.CALIBRATION_SAMPLES
    if args.calibration_shots is None:
        args.calibration_shots = hardware.CALIBRATION_MIN_SHOTS

    weights = Path(args.weights)
    config = weights_config(weights, args.profile)
    from traqmania.config import load_config
    from traqmania.server.runtime import observation_note

    note = observation_note(load_config(args.profile), config)
    if note:  # stderr: stdout is the report
        print(note, file=sys.stderr, flush=True)
    params = np.load(weights)["params"]
    fast = QuantumQFunction(config["circuit"])
    fast.set_params(params)

    exact = summarize(rollout(fast, _make_env(args.track, config, args.episodes,
                                              args.eval_seed), args.max_decisions))
    rows = [{"path": "exact", "resilience": None, "rescale": "off", "shots": None, **exact}]

    # the device model(s) the emulation runs under
    models: dict[str, dict] = {}
    if args.no_device:
        models["given"] = {"attenuation": args.attenuation, "bias": args.bias}
    else:
        saved = {}
        if args.model:
            saved = json.loads(Path(args.model).read_text(encoding="utf-8"))["models"]
        for level in args.resilience:
            if str(level) in saved:
                models[str(level)] = saved[str(level)]
                continue
            fit = fit_device_model(args, config, params, level)
            models[str(level)] = {
                "attenuation": fit["slope_per_readout"], "bias": fit["bias_per_readout"],
                "global_attenuation": fit["attenuation"], "excess_rms": fit["excess_rms"],
                "affine_residual_rms": fit["affine_residual_rms"],
                "backend": fit["backend"], "shots": fit["shots"], "samples": fit["samples"],
            }

    row_seed = 0
    for level, model in models.items():
        for rescale in rescales:
            for shots in args.shots:
                row_seed += 1
                rows.append(emulation_row(fast, config, args, model, shots, rescale,
                                          None if level == "given" else int(level), row_seed))
    if not args.no_device and args.device_episodes > 0:
        backend = hardware.get_backend(use_fake=True, fake_name=args.fake,
                                       min_qubits=max(5, int(config["circuit"]["n_qubits"])))
        for level in args.resilience:
            for rescale in rescales:
                for shots in args.shots:
                    row_seed += 1
                    rows.append(device_row(config, params, args, backend, shots, rescale,
                                           level, row_seed))
                    print(f"device: resilience {level}, rescale {rescale}, {shots} shots: "
                          f"{rows[-1]['lapped']}/{rows[-1]['episodes']} lapped",
                          file=sys.stderr, flush=True)

    return {
        "schema": SCHEMA,
        "weights": str(weights),
        "track": args.track,
        "profile": args.profile,
        "fake": None if args.no_device else args.fake,
        "episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "seed": args.seed,
        "max_decisions": args.max_decisions or int(config["reward"]["max_decisions"]),
        "device_max_decisions": args.device_max_decisions,
        "calibration": {"samples": args.calibration_samples,
                        "min_shots": args.calibration_shots},
        "models": models,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(markdown(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
