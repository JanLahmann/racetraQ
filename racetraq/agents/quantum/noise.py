"""Expectation-value noise: what a device does to ``<Z_a>``, cheap enough to train with.

The policy reads ``Q_a = w[a] * <Z_a> + b[a]`` and acts on the argmax, so what
matters about a noisy device is what it does to the readout expectations.
:class:`ExpectationNoise` models that with three numbers:

- ``attenuation`` f — gate and readout errors shrink every expectation toward
  zero, ``<Z>_device ~ f * <Z>_exact`` (exact for a global depolarizing channel,
  a least-squares summary for a real device);
- ``shots`` — each expectation is the mean of ``shots`` +-1 outcomes, so it
  carries Gaussian noise of ``sigma = sqrt((1 - <Z>^2) / shots)``;
- ``bias`` — an additive offset (asymmetric readout error, relaxation).

``attenuation`` and ``bias`` may be given per readout: readout error differs
from qubit to qubit, and it is the DIFFERENCE between readouts — not the
common shrink — that re-ranks actions.  :class:`ReadoutCorrection` is the
matching inference-side mitigation (``HardwareQFunction(rescale=...)``):
divide by a calibrated attenuation, globally or per readout with its bias
removed.

The model is deliberately small.  It has no state-dependent error and treats
the readouts' shot noise as independent (on a device the ``Z_a`` come from
the same shots); :func:`calibrate` reports how much of a device's error it
leaves unexplained and :func:`validate` compares its action-flip statistics
with the real device-patch simulation.  How far to trust it (measured
2026-10-01 on the fake_miami patch at 1024 shots, 25 four-qubit oval drivers,
12 episodes each on the device path against the same 12 starts emulated ten
times): the per-readout model's shot-noise variance matched the device's
within 4 %, its on-trajectory flip rate was 0.128 against 0.140, and its lap
counts correlated 0.96 with the device's — but for 2 of the 25 drivers it
predicted about 6 laps of 12 where the device path drove 10 and 11.  One
global attenuation did worse (flip rate 0.084, correlation 0.89, 3 drivers
off by 4 laps or more).  Use it to compare recipes over many seeds; check a
single driver on the device path (``tools/hw_reliability.py``).

:class:`NoisyQFunction` wraps a ``QuantumQFunction`` so that every
``q_values`` call sees the noise — the fast stand-in for a hardware lap or an
SPSA sprint (``tools/hw_reliability.py``).  The trainer's ``[training]
act_noise`` option uses ``QuantumQFunction.noisy_q_values`` instead and keeps
TD targets and gradients exact.

numpy/stdlib only at import time; :func:`calibrate` and :func:`validate`
import ``racetraq.hardware`` (and with it qiskit) lazily.  Command line::

    python -m racetraq.agents.quantum.noise calibrate --fake fake_miami --qubits 4
    python -m racetraq.agents.quantum.noise validate --fake fake_miami --track oval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NOISE_KEYS = ("attenuation", "shots", "bias")
RESCALE_MODES = ("global", "readout")
# rng stream tag of the trainer's acting noise (spells "nois")
ACT_NOISE_STREAM = 0x6E6F6973


def _per_readout(value: Any, name: str) -> float | tuple[float, ...]:
    """A float, or a tuple of floats (one per readout), from a config value."""
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a number or a list of numbers, got {value!r}")
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(float(v) for v in value)
    return float(value)


def _fmt(value: float | tuple[float, ...]) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(f"{v:.3f}" for v in value) + "]"
    return f"{value:.3f}"


@dataclass(frozen=True)
class ExpectationNoise:
    """``<Z>_noisy = clip(f * <Z> + bias + shot noise, -1, 1)``.

    ``attenuation`` and ``bias`` are scalars or one value per readout;
    ``shots = 0`` means no shot noise.
    """

    attenuation: float | tuple[float, ...] = 1.0
    shots: int = 0
    bias: float | tuple[float, ...] = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "attenuation", _per_readout(self.attenuation, "attenuation"))
        object.__setattr__(self, "bias", _per_readout(self.bias, "bias"))
        if np.any(np.asarray(self.attenuation) <= 0.0):
            raise ValueError(f"attenuation must be > 0, got {self.attenuation}")
        if isinstance(self.shots, (bool, str)) or int(self.shots) != self.shots or self.shots < 0:
            raise ValueError(f"shots must be an integer >= 0, got {self.shots!r}")
        object.__setattr__(self, "shots", int(self.shots))

    @classmethod
    def from_config(cls, table: Mapping[str, Any] | ExpectationNoise | None):
        """The noise a config table (``{attenuation, shots, bias}``, every key
        optional) describes — or ``None`` when it is absent, empty or the
        identity, so "no noise" always takes the exact code path."""
        if table is None:
            return None
        if isinstance(table, cls):
            noise = table
        elif isinstance(table, Mapping):
            unknown = sorted(set(table) - set(NOISE_KEYS))
            if unknown:
                raise ValueError(f"unknown key(s) {unknown}; known: {list(NOISE_KEYS)}")
            noise = cls(**table)
        else:
            raise ValueError(
                f"expected a table like {{ attenuation = 0.95, shots = 1024 }}, got {table!r}"
            )
        return None if noise.is_identity else noise

    @property
    def is_identity(self) -> bool:
        """True when :meth:`apply` returns its input unchanged."""
        return (
            bool(np.all(np.asarray(self.attenuation) == 1.0))
            and self.shots == 0
            and bool(np.all(np.asarray(self.bias) == 0.0))
        )

    def mean(self, expectations: np.ndarray) -> np.ndarray:
        """Noise-free part of the device value: ``f * <Z> + bias``."""
        e = np.asarray(expectations, dtype=np.float64)
        return np.asarray(self.attenuation) * e + np.asarray(self.bias)

    def sigma(self, expectations: np.ndarray) -> np.ndarray:
        """Shot-noise standard deviation of each device value."""
        mean = self.mean(expectations)
        if self.shots == 0:
            return np.zeros_like(mean)
        return np.sqrt(np.clip(1.0 - mean**2, 0.0, None) / self.shots)

    def apply(self, expectations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One noisy estimate per entry of ``expectations`` (any shape; per-readout
        ``attenuation`` / ``bias`` broadcast over the last axis)."""
        value = self.mean(expectations)
        if self.shots > 0:
            sigma = np.sqrt(np.clip(1.0 - value**2, 0.0, None) / self.shots)
            value = np.clip(value + sigma * rng.standard_normal(value.shape), -1.0, 1.0)
        return value

    def describe(self) -> str:
        parts = []
        if not np.all(np.asarray(self.attenuation) == 1.0):
            parts.append(f"attenuation {_fmt(self.attenuation)}")
        if self.shots:
            parts.append(f"{self.shots} shots")
        if not np.all(np.asarray(self.bias) == 0.0):
            parts.append(f"bias {_fmt(self.bias)}")
        return ", ".join(parts) if parts else "no noise"


@dataclass(frozen=True)
class ReadoutCorrection:
    """Inference-side mitigation: ``<Z>_corrected = (<Z>_device - bias) / slope``.

    ``slope`` / ``bias`` are scalars or one value per readout.  Built from a
    calibration fit (:func:`fit_attenuation`) by :meth:`from_fit`: mode
    ``"global"`` divides by the one fitted attenuation f (the
    global-depolarizing rescale), ``"readout"`` inverts each readout's own
    affine fit.  The correction restores the scale of ``w * <Z>`` against the
    bias ``b`` — and amplifies the shot noise by ``1 / slope``.
    """

    slope: float | tuple[float, ...] = 1.0
    bias: float | tuple[float, ...] = 0.0
    mode: str = "global"

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", _per_readout(self.slope, "slope"))
        object.__setattr__(self, "bias", _per_readout(self.bias, "bias"))
        if self.mode not in RESCALE_MODES:
            raise ValueError(f"rescale mode must be one of {RESCALE_MODES}, got {self.mode!r}")
        if not np.all(np.asarray(self.slope) > 0.0):
            raise ValueError(f"correction slope must be > 0, got {self.slope}")

    @classmethod
    def from_fit(cls, fit: Mapping[str, Any], mode: str = "global") -> ReadoutCorrection:
        if mode == "readout":
            return cls(tuple(fit["slope_per_readout"]), tuple(fit["bias_per_readout"]), mode)
        return cls(float(fit["attenuation"]), 0.0, mode)

    @property
    def min_slope(self) -> float:
        return float(np.min(np.asarray(self.slope)))

    def apply(self, expectations: np.ndarray) -> np.ndarray:
        e = np.asarray(expectations, dtype=np.float64)
        return (e - np.asarray(self.bias)) / np.asarray(self.slope)

    def describe(self) -> str:
        if self.mode == "readout":
            return f"per-readout rescale, slope {_fmt(self.slope)}, bias {_fmt(self.bias)}"
        return f"global rescale, attenuation f = {_fmt(self.slope)}"


def rescale_mode(value: Any) -> str | None:
    """Normalise a ``rescale`` setting: false / none / "" / "off" -> ``None``,
    true -> "global", else one of ``RESCALE_MODES``."""
    if value is None or value is False or value in ("", "off", "none"):
        return None
    if value is True:
        return "global"
    if value in RESCALE_MODES:
        return str(value)
    raise ValueError(f"rescale must be false, true or one of {RESCALE_MODES}, got {value!r}")


def act_noise_rng(seed: int | None) -> np.random.Generator:
    """The generator the trainer draws acting noise from: its own stream, so
    exploration and replay sampling keep theirs."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng([int(seed), ACT_NOISE_STREAM])


class NoisyQFunction:
    """A Q-function whose EVERY evaluation sees expectation noise.

    Wraps a ``QuantumQFunction`` (anything with ``expectations`` and a ``w`` /
    ``b`` head): same parameters, same ``QFunction`` reading contract, but
    ``q_values`` is ``w * correction(noise(<Z>)) + b`` with a fresh noise draw
    per call — what ``HardwareQFunction`` returns on a device, at fastsim
    cost.  ``correction`` (optional) is the mitigation a hardware run would
    apply.  Inference only.
    """

    def __init__(self, qfunc: Any, noise: ExpectationNoise | None,
                 rng: np.random.Generator | int | None = None,
                 correction: ReadoutCorrection | None = None) -> None:
        self.qfunc = qfunc
        self.noise = noise
        self.correction = correction
        self.rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        self.n_features = qfunc.n_features
        self.n_actions = qfunc.n_actions

    def expectations(self, obs: np.ndarray) -> np.ndarray:
        """Device-like readout expectations: noisy, uncorrected."""
        exact = self.qfunc.expectations(obs)
        return exact if self.noise is None else self.noise.apply(exact, self.rng)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        expectations = self.expectations(obs)
        if self.correction is not None:
            expectations = self.correction.apply(expectations)
        return expectations * self.qfunc.w + self.qfunc.b

    def calibrate(self, mode: str = "global", samples: int = 32, shots: int = 4096,
                  seed: int = 0) -> dict:
        """Emulate ``HardwareQFunction.calibrate``: fit the correction from
        ``samples`` random inputs seen through the noise at ``shots`` shots
        (so the emulated mitigation carries a realistic calibration error)."""
        obs = calibration_inputs(self.n_features, samples, seed)
        exact = self.qfunc.expectations(obs)
        measured = exact
        if self.noise is not None:
            job = ExpectationNoise(self.noise.attenuation, int(shots), self.noise.bias)
            measured = job.apply(exact, self.rng)
        fit = fit_attenuation(measured, exact, shots=shots)
        self.correction = ReadoutCorrection.from_fit(fit, mode)
        return fit

    def grad_selected(self, obs, action_idx, upstream):
        raise NotImplementedError("NoisyQFunction is inference-only")

    @property
    def n_params(self) -> int:
        return self.qfunc.n_params

    def get_params(self) -> np.ndarray:
        return self.qfunc.get_params()

    def set_params(self, params: np.ndarray) -> None:
        self.qfunc.set_params(params)


# ------------------------------------------------------------------ fitting


def fit_attenuation(noisy: np.ndarray, exact: np.ndarray, shots: int = 0) -> dict:
    """Least-squares attenuation of ``noisy ~ f * exact`` (both ``(K, A)``).

    Returns the global fit — ``attenuation`` f, its ``residual_rms``, the
    ``shot_rms`` that ``shots`` alone would leave and the ``excess_rms``
    beyond it (structure the one-number model misses; 0 when ``shots`` is
    unknown) — plus per-readout fits: ``attenuation_per_readout`` (through
    the origin) and the affine ``slope_per_readout`` / ``bias_per_readout``
    with ``affine_residual_rms``.  ``rms_error`` is the raw rms of ``noisy -
    exact``.
    """
    noisy = np.atleast_2d(np.asarray(noisy, dtype=np.float64))
    exact = np.atleast_2d(np.asarray(exact, dtype=np.float64))
    if noisy.shape != exact.shape:
        raise ValueError(f"noisy {noisy.shape} and exact {exact.shape} must have the same shape")
    power = float(np.sum(exact**2))
    if power <= 0.0:
        raise ValueError("cannot fit an attenuation: every exact expectation is zero")
    f = float(np.sum(noisy * exact) / power)
    residual = noisy - f * exact
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    shot_rms = 0.0
    if shots:
        shot_rms = float(np.sqrt(np.mean(np.clip(1.0 - (f * exact) ** 2, 0.0, None)) / shots))
    column_power = np.sum(exact**2, axis=0)
    per_readout = np.sum(noisy * exact, axis=0) / np.where(column_power > 0.0, column_power, 1.0)
    slopes, biases = [], []
    affine_residual = np.zeros_like(noisy)
    for a in range(exact.shape[1]):
        design = np.stack([exact[:, a], np.ones(exact.shape[0])], axis=1)
        (slope, bias), *_ = np.linalg.lstsq(design, noisy[:, a], rcond=None)
        slopes.append(float(slope))
        biases.append(float(bias))
        affine_residual[:, a] = noisy[:, a] - design @ np.array([slope, bias])
    return {
        "attenuation": f,
        "residual_rms": residual_rms,
        "shot_rms": shot_rms,
        "excess_rms": float(math.sqrt(max(0.0, residual_rms**2 - shot_rms**2))),
        "rms_error": float(np.sqrt(np.mean((noisy - exact) ** 2))),
        "attenuation_per_readout": [float(v) for v in per_readout],
        "slope_per_readout": slopes,
        "bias_per_readout": biases,
        "affine_residual_rms": float(np.sqrt(np.mean(affine_residual**2))),
        "samples": int(exact.shape[0]),
        "shots": int(shots),
    }


def calibration_inputs(n_features: int, samples: int, seed: int = 0) -> np.ndarray:
    """``samples`` random observations in [0, 1]^n — the calibration job's inputs."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(int(samples), int(n_features)))


def random_circuit_params(n_qubits: int, n_layers: int, n_actions: int,
                          seed: int = 0) -> np.ndarray:
    """Flat ``[lam, theta, w, b]`` with lam = pi, theta ~ U(-pi, pi), w = 1, b = 0:
    generic angles whose expectations spread over [-1, 1]."""
    rng = np.random.default_rng(seed)
    lam = np.full(n_layers * n_qubits, np.pi)
    theta = rng.uniform(-np.pi, np.pi, size=n_layers * n_qubits * 2)
    return np.concatenate([lam, theta, np.ones(n_actions), np.zeros(n_actions)])


# ------------------------------------------------------- device calibration


def _device_qfunction(fake: str, circuit_cfg: dict, shots: int, resilience_level: int,
                      prune: bool, seed_simulator: int | None):
    """HardwareQFunction on the local twin of ``fake`` (lazy qiskit), job mode."""
    from racetraq import hardware

    n_qubits = int(circuit_cfg.get("n_qubits", 4))
    backend = hardware.get_backend(use_fake=True, fake_name=fake, min_qubits=max(5, n_qubits))
    hw = hardware.HardwareQFunction(circuit_cfg, backend, shots=shots,
                                    resilience_level=resilience_level, prune_light_cone=prune)
    if seed_simulator is not None:
        hw.seed_simulator(seed_simulator)
    return hw


def calibrate(
    fake: str = "fake_miami",
    n_qubits: int = 4,
    n_layers: int = 4,
    n_actions: int | None = None,
    samples: int = 64,
    shots: int = 4096,
    seed: int = 0,
    params: np.ndarray | None = None,
    resilience_level: int = 0,
    prune: bool = True,
    seed_simulator: int | None = None,
) -> dict:
    """Fit the attenuation of a fake device for one circuit shape.

    Runs ``samples`` random observations through the device-patch simulation
    of ``fake`` (one Estimator job) and through fastsim, with ``params`` (a
    flat ``[lam, theta, w, b]`` vector; default: generic random angles), and
    fits ``<Z>_device ~ f * <Z>_exact``.  Returns :func:`fit_attenuation`'s
    fields plus ``backend``, ``two_qubit_gates``, ``depth``, the circuit shape
    and two tables to hand to ``[training] act_noise`` or
    :class:`ExpectationNoise`: ``noise`` (the global ``{attenuation, shots}``)
    and ``noise_per_readout`` (a slope and a bias per readout — the model that
    reproduces the device's action flips).
    """
    from racetraq.agents.quantum.qdqn import QuantumQFunction

    circuit_cfg: dict[str, Any] = {"n_qubits": int(n_qubits), "n_layers": int(n_layers)}
    if n_actions is not None:
        circuit_cfg["n_actions"] = int(n_actions)
    fast = QuantumQFunction(circuit_cfg, seed=seed)
    if params is None:
        params = random_circuit_params(fast.n_qubits, fast.n_layers, fast.n_actions, seed)
    fast.set_params(params)
    hw = _device_qfunction(fake, circuit_cfg, shots, resilience_level, prune, seed_simulator)
    hw.set_params(params)

    obs = calibration_inputs(fast.n_features, samples, seed)
    fit = fit_attenuation(hw.expectations(obs), fast.expectations(obs), shots=shots)
    return {
        **fit,
        "backend": hw.backend_name,
        "two_qubit_gates": hw.two_qubit_gates,
        "depth": hw.depth,
        "n_qubits": fast.n_qubits,
        "n_layers": fast.n_layers,
        "n_actions": fast.n_actions,
        "resilience_level": int(resilience_level),
        "pruned": bool(prune),
        "noise": {"attenuation": round(fit["attenuation"], 4), "shots": int(shots)},
        "noise_per_readout": {
            "attenuation": [round(v, 4) for v in fit["slope_per_readout"]],
            "bias": [round(v, 4) for v in fit["bias_per_readout"]],
            "shots": int(shots),
        },
    }


# ----------------------------------------------------------- flip statistics


def trajectory_states(qfunc: Any, track_name: str, config: dict, states: int,
                      seed: int = 20_000, episodes: int = 4) -> np.ndarray:
    """``states`` observations the greedy (exact) policy visits on ``track_name``:
    evenly thinned from ``episodes`` parallel episodes of one greedy round."""
    from racetraq.env.racing_env import RacingEnv
    from racetraq.env.track import Track

    track = Track.load(track_name, config["track"]["resample_spacing"])
    env = RacingEnv(track, config, n_envs=int(episodes), seed=seed)
    obs = env.reset()
    alive = np.ones(env.n_envs, dtype=bool)
    visited = []
    for _ in range(env.max_decisions):
        visited.append(obs[alive])
        obs, _reward, done, _info = env.step(np.argmax(qfunc.q_values(obs), axis=1))
        alive &= ~np.asarray(done, dtype=bool)
        if not alive.any():
            break
    visited = np.concatenate(visited)
    keep = np.linspace(0, visited.shape[0] - 1, min(int(states), visited.shape[0]))
    return visited[np.round(keep).astype(int)]


def flip_rate(expectations: np.ndarray, w: np.ndarray, b: np.ndarray,
              greedy: np.ndarray) -> np.ndarray:
    """Per state, the fraction of noisy repeats whose argmax differs from
    ``greedy``: ``expectations`` is ``(R, K, A)``, the result ``(K,)``."""
    actions = np.argmax(np.asarray(expectations) * w + b, axis=-1)
    return np.mean(actions != np.asarray(greedy)[None, :], axis=0)


def validate(
    weights: str | Path,
    track: str = "oval",
    fake: str = "fake_miami",
    shots: int = 1024,
    states: int = 64,
    repeats: int = 8,
    seed: int = 0,
    profile: str | None = None,
    resilience_level: int = 0,
    emulation_repeats: int = 400,
    reference_shots: int = 100_000,
) -> dict:
    """Compare the emulation's action flips with the device-patch simulation's.

    Takes ``states`` observations from the driver's own greedy trajectory and
    counts how often the greedy action differs from the exact one

    - on the local twin of ``fake``: ``repeats`` jobs at ``shots`` shots, plus
      one at ``reference_shots`` whose flips are the SYSTEMATIC part (what no
      number of shots removes);
    - under two emulations fitted on an independent ``reference_shots``
      calibration job (random inputs, the driver's parameters): the global
      model (one attenuation f) and the per-readout affine model (a slope and
      a bias per readout), ``emulation_repeats`` draws each, plus their
      infinite-shot flips.

    ``per_state_correlation`` compares the per-state flip rates of the device
    and of each emulation.
    """
    from racetraq.agents.quantum.qdqn import QuantumQFunction
    from racetraq.config import load_config
    from racetraq.server.runtime import with_weights_config

    # the profile, with the observation the weights record and at the depth
    # and action count they need
    config = with_weights_config(load_config(profile), Path(weights))
    fast = QuantumQFunction(config["circuit"])
    params = np.load(weights)["params"]
    fast.set_params(params)
    obs = trajectory_states(fast, track, config, states)
    exact = fast.expectations(obs)
    greedy = np.argmax(exact * fast.w + fast.b, axis=1)
    ordered = np.sort(exact * fast.w + fast.b, axis=1)
    gaps = ordered[:, -1] - ordered[:, -2]

    hw = _device_qfunction(fake, config["circuit"], shots, resilience_level, True, seed)
    hw.set_params(params)
    device = np.stack([hw.expectations(obs) for _ in range(int(repeats))])  # seed + job index
    hw._estimator.options.default_shots = int(reference_shots)
    reference = hw.expectations(obs)
    cal_obs = calibration_inputs(fast.n_features, 64, seed)
    fit = fit_attenuation(hw.expectations(cal_obs), fast.expectations(cal_obs),
                          shots=reference_shots)

    def flips(expectations: np.ndarray) -> np.ndarray:
        return flip_rate(expectations, fast.w, fast.b, greedy)

    device_flips = flips(device)
    rng = np.random.default_rng(seed)
    models = {
        "global": ExpectationNoise(attenuation=fit["attenuation"], shots=shots),
        "per_readout": ExpectationNoise(attenuation=tuple(fit["slope_per_readout"]),
                                        bias=tuple(fit["bias_per_readout"]), shots=shots),
    }
    emulation = {}
    for name, model in models.items():
        per_state = flips(np.stack([model.apply(exact, rng)
                                    for _ in range(int(emulation_repeats))]))
        correlation = float("nan")
        if device_flips.std() > 0.0 and per_state.std() > 0.0:
            correlation = float(np.corrcoef(device_flips, per_state)[0, 1])
        emulation[name] = {
            "flip_rate": float(per_state.mean()),
            "systematic_flip_rate": float(flips(model.mean(exact)[None]).mean()),
            "per_state_correlation": correlation,
        }
    rate = float(device_flips.mean())
    return {
        "backend": hw.backend_name,
        "track": track,
        "shots": int(shots),
        "reference_shots": int(reference_shots),
        "resilience_level": int(resilience_level),
        "states": int(obs.shape[0]),
        "repeats": int(repeats),
        "calibration": fit,
        "gap_median": float(np.median(gaps)),
        "gap_p10": float(np.percentile(gaps, 10)),
        "device_flip_rate": rate,
        "device_flip_stderr": float(math.sqrt(max(rate * (1.0 - rate), 0.0) / device.size
                                              * device.shape[2])),
        "device_systematic_flip_rate": float(flips(reference[None]).mean()),
        "emulation": emulation,
    }


# ------------------------------------------------------------------------ CLI


def _calibration_text(result: dict) -> str:
    per = ", ".join(f"{v:.3f}" for v in result["attenuation_per_readout"])
    bias = ", ".join(f"{v:+.4f}" for v in result["bias_per_readout"])
    return "\n".join([
        f"backend: {result['backend']}",
        f"circuit: {result['n_qubits']} qubits, {result['n_layers']} blocks, "
        f"{result['two_qubit_gates']} two-qubit gates, depth {result['depth']}"
        f"{' (light-cone pruned)' if result['pruned'] else ''}",
        f"calibration: {result['samples']} random inputs x {result['n_actions']} readouts, "
        f"{result['shots']} shots, resilience level {result['resilience_level']}",
        f"attenuation f = {result['attenuation']:.4f}   "
        f"(<Z>_device ~ f * <Z>_exact; raw rms error {result['rms_error']:.4f})",
        f"residual rms after the fit: {result['residual_rms']:.4f}   "
        f"(shot noise alone: {result['shot_rms']:.4f}, unexplained: {result['excess_rms']:.4f})",
        f"per readout: f = [{per}]; affine fit bias = [{bias}], "
        f"residual rms {result['affine_residual_rms']:.4f}",
        f"act_noise = {{ attenuation = {result['noise']['attenuation']}, shots = <per-decision "
        f"shots> }}",
        f"act_noise = {{ attenuation = {result['noise_per_readout']['attenuation']}, "
        f"bias = {result['noise_per_readout']['bias']}, shots = <per-decision shots> }}",
    ])


def _validation_text(result: dict) -> str:
    lines = [
        f"backend: {result['backend']}",
        f"{result['states']} on-trajectory states of {result['track']} x {result['repeats']} "
        f"device repeats at {result['shots']} shots "
        f"(top-2 Q gap: median {result['gap_median']:.2f}, 10th percentile "
        f"{result['gap_p10']:.2f})",
        "action flips vs the exact policy          at these shots   systematic   "
        "per-state corr.",
        f"  device                                  {result['device_flip_rate']:.3f} "
        f"+-{result['device_flip_stderr']:.3f}    "
        f"{result['device_systematic_flip_rate']:.3f}",
    ]
    labels = {"global": "emulation, one attenuation f", "per_readout":
              "emulation, slope + bias per readout"}
    for name, label in labels.items():
        emu = result["emulation"][name]
        lines.append(f"  {label:<38}  {emu['flip_rate']:.3f}           "
                     f"{emu['systematic_flip_rate']:.3f}        "
                     f"{emu['per_state_correlation']:.2f}")
    lines.append(f"(systematic: device at {result['reference_shots']} shots, emulation "
                 f"without shot noise)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m racetraq.agents.quantum.noise",
        description="Calibrate and validate racetraQ's expectation-noise model "
                    "against a local fake device.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="fit the attenuation of a fake device")
    cal.add_argument("--fake", default="fake_miami", help="fake backend, any spelling")
    cal.add_argument("--qubits", type=int, default=4)
    cal.add_argument("--layers", type=int, default=4)
    cal.add_argument("--actions", type=int, default=None,
                     help="readout qubits (default min(4, qubits))")
    cal.add_argument("--samples", type=int, default=64, help="random calibration inputs")
    cal.add_argument("--shots", type=int, default=4096)
    cal.add_argument("--weights", default=None,
                     help="calibrate with these weights (.npz) instead of random angles")
    val = sub.add_parser("validate",
                         help="compare emulated and device-patch action flips on a "
                              "driver's own trajectory")
    val.add_argument("--fake", default="fake_miami", help="fake backend, any spelling")
    val.add_argument("--track", default="oval")
    val.add_argument("--profile", default=None, help="config profile overlay (e.g. q6)")
    val.add_argument("--weights", default=None, help="weights .npz (default: bundled)")
    val.add_argument("--shots", type=int, default=1024)
    val.add_argument("--states", type=int, default=64, help="on-trajectory states")
    val.add_argument("--repeats", type=int, default=8, help="device jobs per state")
    for p in (cal, val):
        p.add_argument("--resilience", type=int, default=0, choices=[0, 1, 2],
                       help="Estimator error mitigation level (default 0: raw noise)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    try:
        if args.command == "calibrate":
            params = np.load(args.weights)["params"] if args.weights else None
            result = calibrate(args.fake, args.qubits, args.layers, args.actions,
                               samples=args.samples, shots=args.shots, seed=args.seed,
                               params=params, resilience_level=args.resilience,
                               seed_simulator=args.seed)
            text = _calibration_text(result)
        else:
            from racetraq import hardware
            from racetraq.config import load_config
            from racetraq.server.runtime import observation_note, with_weights_observation

            profile_config = load_config(args.profile)
            n_qubits = int(profile_config["circuit"]["n_qubits"])
            weights = args.weights or hardware._default_weights(args.track, n_qubits)
            note = observation_note(
                profile_config, with_weights_observation(profile_config, Path(weights)))
            if note:  # stderr: stdout may be the --json result
                print(note, file=sys.stderr)
            result = validate(weights, args.track, args.fake, shots=args.shots,
                              states=args.states, repeats=args.repeats, seed=args.seed,
                              profile=args.profile, resilience_level=args.resilience)
            text = _validation_text(result)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2) if args.json else text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
