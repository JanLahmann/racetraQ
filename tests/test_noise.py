"""Expectation noise: the model's statistics, the attenuation fit and the
correction built from it, the trainer's ``act_noise`` option (default off =
bit-identical; noise reaches acting and snapshot evals, never the TD targets)
and tools/hw_reliability.py.  Everything here runs on numpy; the two tests
that touch the device-patch simulation skip without qiskit-ibm-runtime."""

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from racetraq.agents.classical import MLPQFunction
from racetraq.agents.quantum import noise
from racetraq.agents.quantum.noise import (
    ExpectationNoise,
    NoisyQFunction,
    ReadoutCorrection,
    fit_attenuation,
)
from racetraq.agents.quantum.qdqn import QuantumQFunction
from racetraq.agents.training import DQNTrainer
from racetraq.config import load_config
from racetraq.env.racing_env import RacingEnv
from racetraq.env.track import Track

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "hw_reliability.py"
OVAL_WEIGHTS = REPO_ROOT / "racetraq" / "weights" / "quantum_oval.npz"

TRAINING_CFG = {
    "episodes": 24,
    "replay_size": 4000,
    "batch_size": 16,
    "gamma": 0.98,
    "lr": 0.01,
    "target_sync_every": 50,
    "epsilon_start": 0.5,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 10,
    "n_parallel_envs": 4,
    "seed": 5,
    "eval_every": 8,
}
NOISE = {"attenuation": 0.9, "shots": 64}


@pytest.fixture(scope="module")
def config():
    cfg = load_config()
    cfg["reward"]["max_decisions"] = 60  # keeps greedy evals short
    return cfg


@pytest.fixture(scope="module")
def oval(config):
    return Track.load("oval", config["track"]["resample_spacing"])


def _driver(config) -> QuantumQFunction:
    qfunc = QuantumQFunction(config["circuit"], seed=7)
    qfunc.set_params(np.load(OVAL_WEIGHTS)["params"])
    return qfunc


# ---------------------------------------------------------------- the model


def test_noise_mean_and_spread_match_the_model():
    exact = np.array([[-0.9, -0.3, 0.0, 0.6]])
    model = ExpectationNoise(attenuation=0.9, shots=400, bias=0.02)
    draws = model.apply(np.repeat(exact, 20_000, axis=0), np.random.default_rng(0))
    mean = 0.9 * exact[0] + 0.02
    sigma = np.sqrt((1.0 - mean**2) / 400)
    np.testing.assert_allclose(model.mean(exact)[0], mean)
    np.testing.assert_allclose(model.sigma(exact)[0], sigma)
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=4 * sigma.max() / np.sqrt(20_000))
    np.testing.assert_allclose(draws.std(axis=0), sigma, rtol=0.03)
    assert np.all(np.abs(draws) <= 1.0)


def test_noise_without_shots_is_deterministic_and_stays_in_range():
    exact = np.linspace(-1.0, 1.0, 9).reshape(3, 3)
    rng = np.random.default_rng(1)
    state = copy.deepcopy(rng.bit_generator.state)
    np.testing.assert_allclose(ExpectationNoise(attenuation=0.8).apply(exact, rng), 0.8 * exact)
    assert rng.bit_generator.state == state  # no draw taken
    # a single shot at <Z> = +-1 would overshoot without the clip
    tiny = ExpectationNoise(shots=1, bias=0.0).apply(np.zeros((500, 4)), rng)
    assert tiny.min() >= -1.0 and tiny.max() <= 1.0 and tiny.std() > 0.5


def test_per_readout_attenuation_and_bias_broadcast_over_the_last_axis():
    model = ExpectationNoise(attenuation=[0.9, 0.8, 1.0, 0.5], bias=(0.0, 0.1, -0.1, 0.0))
    exact = np.full((3, 4), 0.5)
    np.testing.assert_allclose(model.apply(exact, np.random.default_rng(0)),
                               np.tile([0.45, 0.5, 0.4, 0.25], (3, 1)))
    assert model.attenuation == (0.9, 0.8, 1.0, 0.5)  # lists are stored as tuples
    assert "attenuation [0.900, 0.800, 1.000, 0.500]" in model.describe()


def test_from_config_treats_absent_and_identity_tables_as_no_noise():
    assert ExpectationNoise.from_config(None) is None
    assert ExpectationNoise.from_config({}) is None
    assert ExpectationNoise.from_config({"attenuation": 1.0, "shots": 0, "bias": 0.0}) is None
    model = ExpectationNoise.from_config({"attenuation": 0.95, "shots": 1024})
    assert model == ExpectationNoise(0.95, 1024)
    assert ExpectationNoise.from_config(model) is model
    assert model.describe() == "attenuation 0.950, 1024 shots"
    assert not model.is_identity and ExpectationNoise().is_identity


@pytest.mark.parametrize("bad", [{"attenuation": 0.0}, {"attenuation": [0.9, -0.1]},
                                 {"shots": -1}, {"shots": 10.5}, {"shots": "many"},
                                 {"attenuation": "weak"}, {"bias": True}, {"shot": 1024}])
def test_invalid_noise_tables_are_rejected(bad):
    with pytest.raises(ValueError):
        ExpectationNoise.from_config(bad)
    with pytest.raises(ValueError):
        ExpectationNoise.from_config("0.95")


# -------------------------------------------------------- fit and correction


def test_fit_recovers_a_planted_attenuation():
    rng = np.random.default_rng(3)
    exact = rng.uniform(-1.0, 1.0, size=(200, 4))
    fit = fit_attenuation(0.87 * exact, exact)
    assert fit["attenuation"] == pytest.approx(0.87, abs=1e-12)
    assert fit["residual_rms"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(fit["attenuation_per_readout"], 0.87)
    np.testing.assert_allclose(fit["bias_per_readout"], 0.0, atol=1e-12)

    shots = 2048
    noisy = ExpectationNoise(attenuation=0.87, shots=shots).apply(exact, rng)
    fit = fit_attenuation(noisy, exact, shots=shots)
    assert fit["attenuation"] == pytest.approx(0.87, abs=0.005)
    # what is left is shot noise: nothing unexplained beyond it
    assert fit["residual_rms"] == pytest.approx(fit["shot_rms"], rel=0.1)
    assert fit["excess_rms"] < 0.5 * fit["shot_rms"]
    assert fit["rms_error"] > fit["residual_rms"]  # the attenuation itself is an error
    assert (fit["samples"], fit["shots"]) == (200, shots)


def test_fit_separates_per_readout_slopes_and_biases():
    rng = np.random.default_rng(4)
    exact = rng.uniform(-1.0, 1.0, size=(300, 4))
    slopes, biases = np.array([0.95, 0.90, 0.97, 0.80]), np.array([0.01, -0.02, 0.0, 0.03])
    fit = fit_attenuation(slopes * exact + biases, exact)
    np.testing.assert_allclose(fit["slope_per_readout"], slopes, atol=1e-12)
    np.testing.assert_allclose(fit["bias_per_readout"], biases, atol=1e-12)
    assert fit["affine_residual_rms"] == pytest.approx(0.0, abs=1e-12)
    assert 0.80 < fit["attenuation"] < 0.97  # one number cannot describe four readouts
    assert fit["residual_rms"] > 0.01

    with pytest.raises(ValueError, match="same shape"):
        fit_attenuation(exact[:10], exact)
    with pytest.raises(ValueError, match="zero"):
        fit_attenuation(np.zeros((4, 4)), np.zeros((4, 4)))


def test_correction_inverts_the_fitted_device_model():
    rng = np.random.default_rng(5)
    exact = rng.uniform(-1.0, 1.0, size=(100, 4))
    device = ExpectationNoise(attenuation=(0.95, 0.90, 0.97, 0.80), bias=(0.01, -0.02, 0.0, 0.03))
    measured = device.apply(exact, rng)
    fit = fit_attenuation(measured, exact)

    readout = ReadoutCorrection.from_fit(fit, "readout")
    np.testing.assert_allclose(readout.apply(measured), exact, atol=1e-12)
    assert readout.min_slope == pytest.approx(0.80)
    assert readout.describe().startswith("per-readout rescale")

    overall = ReadoutCorrection.from_fit(fit, "global")
    assert overall.slope == pytest.approx(fit["attenuation"]) and overall.bias == 0.0
    np.testing.assert_allclose(overall.apply(measured), measured / fit["attenuation"])
    assert np.max(np.abs(overall.apply(measured) - exact)) > 0.02  # a global f is not enough

    with pytest.raises(ValueError, match="mode"):
        ReadoutCorrection(0.9, 0.0, "local")
    with pytest.raises(ValueError, match="slope"):
        ReadoutCorrection((0.9, 0.0), 0.0)


def test_rescale_mode_normalises_the_config_spellings():
    for off in (None, False, "", "off", "none"):
        assert noise.rescale_mode(off) is None
    assert noise.rescale_mode(True) == "global"
    assert noise.rescale_mode("global") == "global"
    assert noise.rescale_mode("readout") == "readout"
    with pytest.raises(ValueError, match="rescale"):
        noise.rescale_mode("per-qubit")


# ----------------------------------------------------------- NoisyQFunction


def test_noisy_qfunction_applies_noise_before_the_head(config):
    fast = _driver(config)
    obs = np.random.default_rng(0).uniform(0.0, 1.0, size=(6, 4))
    model = ExpectationNoise(attenuation=0.9, bias=0.01)  # no shots: deterministic
    noisy = NoisyQFunction(fast, model, rng=0)
    np.testing.assert_allclose(noisy.q_values(obs),
                               (0.9 * fast.expectations(obs) + 0.01) * fast.w + fast.b)
    assert (noisy.n_features, noisy.n_actions, noisy.n_params) == (4, 4, 56)
    np.testing.assert_array_equal(NoisyQFunction(fast, None).q_values(obs), fast.q_values(obs))
    with pytest.raises(NotImplementedError):
        noisy.grad_selected(obs, np.zeros(6, dtype=int), np.ones(6))

    # a fresh draw per call, reproducible from the seed
    shot = ExpectationNoise(shots=256)
    first = NoisyQFunction(fast, shot, rng=3)
    a, b = first.q_values(obs), first.q_values(obs)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(NoisyQFunction(fast, shot, rng=3).q_values(obs), a)
    # same greedy action wherever the gap is far above the shot noise: 4 sigma
    # of the difference of two readouts at 256 shots (sigma <= 1/16 in <Z>
    # each), in the Q units of this driver's own output head
    gap = np.diff(np.sort(fast.q_values(obs), axis=1)[:, -2:], axis=1)[:, 0]
    clear = gap > 4.0 * np.sqrt(2.0) * np.max(np.abs(fast.w)) / np.sqrt(256)
    assert clear.any()
    np.testing.assert_array_equal(np.argmax(a, axis=1)[clear],
                                  np.argmax(fast.q_values(obs), axis=1)[clear])


def test_emulated_calibration_restores_the_exact_q_values(config):
    fast = _driver(config)
    obs = np.random.default_rng(1).uniform(0.0, 1.0, size=(8, 4))
    device = ExpectationNoise(attenuation=(0.95, 0.94, 0.97, 0.93), bias=(0.0, 0.004, 0.0, 0.008))
    noisy = NoisyQFunction(fast, device, rng=0)
    raw_error = np.max(np.abs(noisy.q_values(obs) - fast.q_values(obs)))

    fit = noisy.calibrate("readout", samples=64, shots=200_000)
    assert noisy.correction.mode == "readout"
    np.testing.assert_allclose(noisy.correction.slope, device.attenuation, atol=0.004)
    np.testing.assert_allclose(noisy.correction.bias, device.bias, atol=0.004)
    assert fit["shots"] == 200_000
    corrected_error = np.max(np.abs(noisy.q_values(obs) - fast.q_values(obs)))
    assert raw_error > 1.0 and corrected_error < 0.25 * raw_error

    noisy.calibrate("global", samples=64, shots=200_000)
    assert noisy.correction.mode == "global"
    assert 0.93 < noisy.correction.slope < 0.97 and noisy.correction.bias == 0.0


# ------------------------------------------------------- trainer: act_noise


def _trainer(oval, config, qfunc=None, **options):
    env = RacingEnv(oval, config, n_envs=4, seed=5)
    qfunc = qfunc if qfunc is not None else QuantumQFunction(config["circuit"], seed=5)
    return DQNTrainer(qfunc, env, {**TRAINING_CFG, **options}, rng=np.random.default_rng(5),
                      env_factory=lambda: RacingEnv(oval, config, n_envs=4, seed=77))


def _run(oval, config, **options):
    trainer = _trainer(oval, config, **options)
    return trainer.train(), trainer


@pytest.fixture(scope="module")
def exact_run(oval, config):
    return _run(oval, config)


@pytest.mark.parametrize("table", [None, {}, {"attenuation": 1.0, "shots": 0}])
def test_act_noise_off_is_bit_identical_to_absent(oval, config, exact_run, table):
    hist_a, trainer_a = exact_run
    hist_b, trainer_b = _run(oval, config, act_noise=table, action_gap=0.0)
    assert trainer_b.act_noise is None
    assert len(hist_a["losses"]) > 50  # the run really trained
    assert hist_a["episode_returns"] == hist_b["episode_returns"]
    assert hist_a["losses"] == hist_b["losses"]
    assert hist_a["eval_log"] == hist_b["eval_log"]
    np.testing.assert_array_equal(trainer_a.final_params, trainer_b.final_params)


def test_act_noise_changes_the_rollout(oval, config, exact_run):
    hist_a, trainer_a = exact_run
    hist_b, trainer_b = _run(oval, config, act_noise=NOISE)
    assert trainer_b.act_noise == ExpectationNoise(0.9, 64)
    assert hist_a["episode_returns"] != hist_b["episode_returns"]
    assert not np.array_equal(trainer_a.final_params, trainer_b.final_params)
    # reproducible: the acting noise has its own seeded stream
    hist_c, trainer_c = _run(oval, config, act_noise=NOISE)
    assert hist_b["episode_returns"] == hist_c["episode_returns"]
    np.testing.assert_array_equal(trainer_b.final_params, trainer_c.final_params)


def test_act_noise_reaches_acting_and_evals_but_not_the_td_targets(oval, config):
    """Same replay batch, same parameters: the update of a trainer with acting
    noise is the update of one without, to the last bit."""
    exact = _trainer(oval, config)
    noisy = _trainer(oval, config, act_noise=NOISE)
    rng = np.random.default_rng(2)
    env = RacingEnv(oval, config, n_envs=4, seed=1)
    obs = env.reset()
    for _ in range(20):
        actions = rng.integers(4, size=4)
        next_obs, reward, done, _ = env.step(actions)
        for trainer in (exact, noisy):
            trainer.buffer.add(obs, actions, reward, next_obs, done)
        obs = next_obs
    assert exact._update() == noisy._update()
    np.testing.assert_array_equal(exact.qfunc.get_params(), noisy.qfunc.get_params())

    # acting: exact Q-values without the option, noisy ones with it
    noisy_calls: list = []
    original = noisy.qfunc.noisy_q_values

    def recording(*args):
        noisy_calls.append(args)
        return original(*args)

    noisy.qfunc.noisy_q_values = recording
    np.testing.assert_array_equal(exact._acting_q(obs), exact.qfunc.q_values(obs))
    acted = noisy._acting_q(obs)
    assert len(noisy_calls) == 1 and noisy_calls[0][1] is noisy.act_noise
    assert not np.allclose(acted, noisy.qfunc.q_values(obs))
    noisy._select_actions(obs, 0.0)
    assert len(noisy_calls) == 2
    # the snapshot eval drives under the noise too
    noisy._greedy_eval_round(RacingEnv(oval, config, n_envs=2, seed=3), max_steps=5)
    assert len(noisy_calls) == 7
    # ... and the update still never asks for a noisy value
    noisy._update()
    assert len(noisy_calls) == 7


def test_act_noise_needs_readout_expectations(oval, config):
    mlp = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=1)
    with pytest.raises(ValueError, match="act_noise needs a Q-function with readout"):
        _trainer(oval, config, qfunc=mlp, act_noise=NOISE)
    _trainer(oval, config, qfunc=mlp, act_noise={})  # off: the MLP path is untouched
    with pytest.raises(ValueError, match=r"\[training\] act_noise: unknown key"):
        _trainer(oval, config, act_noise={"shotz": 3})
    with pytest.raises(ValueError, match=r"\[training\] act_noise"):
        _trainer(oval, config, act_noise="noisy")
    # a per-readout list must name every readout (not fail on the first step)
    with pytest.raises(ValueError, match="attenuation lists 2 values, the Q-function has 4"):
        _trainer(oval, config, act_noise={"attenuation": [0.95, 0.94], "shots": 64})
    with pytest.raises(ValueError, match="bias lists 3 values"):
        _trainer(oval, config, act_noise={"bias": [0.0, 0.01, 0.0]})
    per_readout = _trainer(oval, config, act_noise={"attenuation": [0.95, 0.94, 0.96, 0.93]})
    assert per_readout.act_noise.attenuation == (0.95, 0.94, 0.96, 0.93)


def test_noisy_q_values_of_the_quantum_qfunction(config):
    fast = _driver(config)
    obs = np.random.default_rng(0).uniform(0.0, 1.0, size=(5, 4))
    model = ExpectationNoise(attenuation=0.5)
    np.testing.assert_allclose(fast.noisy_q_values(obs, model, np.random.default_rng(0)),
                               0.5 * fast.expectations(obs) * fast.w + fast.b)
    a = noise.act_noise_rng(5).standard_normal(4)
    np.testing.assert_array_equal(noise.act_noise_rng(5).standard_normal(4), a)
    assert not np.array_equal(np.random.default_rng(5).standard_normal(4), a)  # own stream


# ---------------------------------------------------------- import hygiene


def test_importing_the_noise_module_does_not_import_qiskit():
    check = ("import racetraq.agents.quantum.noise, sys; "
             "assert not [m for m in sys.modules if m.split('.')[0].startswith('qiskit')]")
    result = subprocess.run([sys.executable, "-c", check], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------- hw_reliability tool


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("racetraq_hw_reliability", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_imports_only_numpy_and_stdlib_at_module_level():
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots - set(sys.stdlib_module_names) == {"numpy"}


class _ScriptedEnv:
    """Three cars: one crashes at step 2, one laps at step 3 and runs into the
    time limit at step 5, one is still driving when the cap ends the rollout."""

    max_decisions = 9

    def reset(self):
        self.t = 0
        return np.zeros((3, 2))

    def step(self, actions):
        self.t += 1
        self.seen = actions
        done = np.array([self.t == 2, self.t == 5, False])
        info = {
            "lap": np.array([0, int(self.t >= 3), 0]),
            "last_lap_time": np.array([np.nan, 14.0 if self.t >= 3 else np.nan, np.nan]),
            "off_track": np.array([self.t == 2, False, False]),
        }
        return np.full((3, 2), float(self.t)), np.ones(3), done, info


class _CountingPolicy:
    def __init__(self):
        self.rows: list[int] = []

    def q_values(self, obs):
        self.rows.append(obs.shape[0])
        return np.tile([0.0, 1.0], (obs.shape[0], 1))


def test_rollout_freezes_each_episode_at_its_first_done(tool):
    policy = _CountingPolicy()
    result = tool.rollout(policy, _ScriptedEnv(), max_decisions=7)
    np.testing.assert_array_equal(result["decisions"], [2, 5, 7])
    np.testing.assert_array_equal(result["crashed"], [True, False, False])
    np.testing.assert_array_equal(result["lapped"], [False, True, False])
    assert result["lap_times"] == [14.0]
    assert policy.rows == [3, 3, 2, 2, 2, 1, 1]  # finished cars are not evaluated again

    summary = tool.summarize(result)
    assert summary == {"episodes": 3, "lapped": 1, "crashed": 1, "laps": 1,
                       "decisions_median": 5.0, "decisions_min": 2,
                       "crash_decisions_median": 2.0, "mean_lap": 14.0}


def test_tool_emulation_end_to_end(tool, tmp_path, capsys):
    out = tmp_path / "reliability.json"
    code = tool.main(["--weights", str(OVAL_WEIGHTS), "--track", "oval", "--no-device",
                      "--attenuation", "0.95,0.94,0.96,0.93", "--shots", "256,65536",
                      "--rescale", "off,readout", "--episodes", "4", "--max-decisions", "30",
                      "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "| path | resilience | rescale | shots |" in text
    assert "Emulation model (given): slope [0.950, 0.940, 0.960, 0.930]" in text

    report = json.loads(out.read_text(encoding="utf-8"))
    rows = report["rows"]
    assert [(r["path"], r["rescale"], r["shots"]) for r in rows] == [
        ("exact", "off", None), ("emulation", "off", 256), ("emulation", "off", 65536),
        ("emulation", "readout", 256), ("emulation", "readout", 65536)]
    assert all(r["episodes"] == 4 and 0 < r["decisions_median"] <= 30 for r in rows)
    assert rows[1]["correction"] is None
    np.testing.assert_allclose(rows[4]["correction"]["slope"], [0.95, 0.94, 0.96, 0.93],
                               atol=0.01)
    assert report["fake"] is None and report["max_decisions"] == 30

    with pytest.raises(SystemExit):
        tool.main(["--weights", str(OVAL_WEIGHTS), "--no-device", "--rescale", "sideways"])


# --------------------------------------------------- against the device patch


def test_calibration_and_validation_on_the_device_patch(capsys):
    """The patch twin of the default fake: a few percent of attenuation, and
    an emulation whose per-readout model explains the device's systematic
    action flips where one global attenuation does not."""
    pytest.importorskip("qiskit_ibm_runtime")
    result = noise.calibrate("fake_miami", 4, samples=48, shots=20_000, seed_simulator=1)
    assert result["backend"].startswith("fake_miami (4-qubit patch")
    assert (result["two_qubit_gates"], result["n_actions"], result["pruned"]) == (12, 4, True)
    assert 0.90 < result["attenuation"] < 0.99
    assert result["rms_error"] > result["residual_rms"]  # the fit explains most of the error
    assert result["residual_rms"] < 0.03
    assert all(0.88 < f < 1.0 for f in result["slope_per_readout"])
    assert result["noise"] == {"attenuation": round(result["attenuation"], 4), "shots": 20_000}

    assert noise.main(["calibrate", "--fake", "fake_miami", "--samples", "16",
                       "--shots", "2048"]) == 0
    out = capsys.readouterr().out
    assert "backend: fake_miami (4-qubit patch" in out and "attenuation f = 0.9" in out

    report = noise.validate(OVAL_WEIGHTS, "oval", "fake_miami", shots=1024, states=32,
                            repeats=3, reference_shots=40_000, emulation_repeats=100)
    assert report["states"] == 32 and report["repeats"] == 3
    readout, overall = report["emulation"]["per_readout"], report["emulation"]["global"]
    device = report["device_systematic_flip_rate"]
    assert 0.0 <= device <= 1.0 and 0.0 <= report["device_flip_rate"] <= 1.0
    # Whatever the driver: the per-readout model is at least as close to the
    # device's systematic flips as one global attenuation.  (The oval driver
    # bundled in October 2026, trained with action_gap 0.8, is rarely flipped
    # at all: 0.06 on the device, 0.06 for the per-readout model, 0.00 for the
    # global one.  The July 2026 driver, whose on-trajectory gaps were far
    # below the device error, gave 0.56 on the device, 0.55 vs 0.02.)
    assert abs(readout["systematic_flip_rate"] - device) \
        <= abs(overall["systematic_flip_rate"] - device) + 0.03
    assert abs(readout["systematic_flip_rate"] - device) < 0.25
    assert abs(readout["flip_rate"] - report["device_flip_rate"]) < 0.25
