"""Trainer correctness options: truncation info from the envs, the optional
DQNTrainer knobs (each must default to the legacy behaviour and move things in
the documented direction when set), distinct eval episodes, and the
train_headless preset / override / final-weights plumbing."""

import copy
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from traqmania.agents.classical import MLPQFunction
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.agents.training import Adam, DQNTrainer
from traqmania.agents.training.dqn import OPTION_KEYS
from traqmania.config import apply_overrides, load_config, parse_override, resolve_training_cfg
from traqmania.env.multi_track import MultiTrackEnv
from traqmania.env.racing_env import RacingEnv
from traqmania.env.track import Track

REPO_ROOT = Path(__file__).resolve().parents[1]
STEER_NEG, STRAIGHT, STEER_POS, BRAKE = 0, 1, 2, 3
NO_UPDATES = 10**9  # batch_size the replay buffer never reaches: params stay put

TRAINING_CFG = {
    "episodes": 30,
    "replay_size": 4000,
    "batch_size": 16,
    "gamma": 0.98,
    "lr": 0.01,
    "target_sync_every": 50,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 20,
    "n_parallel_envs": 4,
    "seed": 5,
    "eval_every": 10,
}
# every optional knob spelled out at its default: must change nothing
EXPLICIT_DEFAULTS = {
    "bootstrap_truncation": False,
    "loss": "mse",
    "huber_delta": 10.0,
    "lr_groups": {},
    "lr_end": TRAINING_CFG["lr"],
    "target_update": "hard",
    "tau": 0.005,
    "grad_clip": 0.0,
    "reward_scale": 1.0,
    "eval_episodes": 12,
    "act_noise": {},
    "action_gap": 0.0,
}


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def oval(config):
    return Track.load("oval", config["track"]["resample_spacing"])


class ConstantPolicy:
    """Minimal QFunction that always prefers one action and never learns."""

    def __init__(self, action: int, n_features: int = 4, n_actions: int = 4):
        self.n_features, self.n_actions = n_features, n_actions
        self._q = np.zeros(n_actions)
        self._q[action] = 1.0
        self._params = np.zeros(1)

    def q_values(self, obs):
        return np.tile(self._q, (np.asarray(obs).shape[0], 1))

    def grad_selected(self, obs, action_idx, upstream):
        return np.zeros(1)

    def get_params(self):
        return self._params.copy()

    def set_params(self, params):
        self._params = np.asarray(params, dtype=np.float64).copy()


class ScriptedEnv:
    """Two sub-envs, two features; the obs after step t is t everywhere.
    Sub-env 0 hits the time limit at step 3, sub-env 1 goes off track at
    step 5; a finished sub-env's pre-reset observation is -t."""

    n_envs = 2

    def __init__(self, with_info: bool = True):
        self.with_info = with_info
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros((2, 2))

    def step(self, actions):
        self.t += 1
        obs = np.full((2, 2), float(self.t))
        done = np.array([self.t == 3, self.t == 5])
        info = {}
        if self.with_info:
            final_obs = None
            if done.any():
                final_obs = obs.copy()
                final_obs[done] = -float(self.t)
            info = {"truncated": np.array([self.t == 3, False]), "final_obs": final_obs}
        return obs, np.ones(2), done, info


def _filled_trainer(oval, config, steps: int = 40, **options):
    """MLP trainer on the oval with a warm replay buffer and no update taken yet."""
    env = RacingEnv(oval, config, n_envs=4, seed=1)
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=1)
    trainer = DQNTrainer(qfunc, env, {**TRAINING_CFG, **options}, rng=np.random.default_rng(1))
    rng = np.random.default_rng(2)
    obs = env.reset()
    for _ in range(steps):
        actions = rng.integers(4, size=4)
        next_obs, reward, done, _ = env.step(actions)
        trainer.buffer.add(obs, actions, reward, next_obs, done)
        obs = next_obs
    return trainer


def _td_errors(trainer) -> np.ndarray:
    """TD errors of the batch the NEXT ``_update()`` call will draw."""
    rng = copy.deepcopy(trainer.rng)
    idx = rng.integers(0, trainer.buffer.size, size=trainer.batch_size)
    buf, qfunc = trainer.buffer, trainer.qfunc
    rows = np.arange(trainer.batch_size)
    a_star = np.argmax(qfunc.q_values(buf.next_obs[idx]), axis=1)
    online = qfunc.get_params()
    qfunc.set_params(trainer.target_params)
    q_next = qfunc.q_values(buf.next_obs[idx])[rows, a_star]
    qfunc.set_params(online)
    target = buf.reward[idx] + trainer.gamma * (1.0 - buf.done[idx]) * q_next
    return qfunc.q_values(buf.obs[idx])[rows, buf.action[idx]] - target


def _spy(obj, name: str) -> list:
    """Record the positional args of every ``obj.name(...)`` call."""
    calls: list = []
    original = getattr(obj, name)

    def wrapper(*args):
        calls.append(args)
        return original(*args)

    setattr(obj, name, wrapper)
    return calls


# ------------------------------------------------------------- env info keys


def test_info_keys_at_off_track(oval, config):
    env = RacingEnv(oval, config, n_envs=3, seed=2)
    twin = RacingEnv(oval, config, n_envs=3, seed=2)  # same cars, never reset
    env.reset()
    twin.reset()
    twin._spawn = lambda mask: None
    for _ in range(200):
        obs, _, done, info = env.step(np.full(3, STEER_POS))
        pre_reset_obs, *_ = twin.step(np.full(3, STEER_POS))
        if np.any(done):
            break
        assert info["final_obs"] is None
        assert info["truncated"].shape == (3,) and not np.any(info["truncated"])
    assert np.any(done) and np.all(info["off_track"][done])
    assert info["truncated"].dtype == bool and not np.any(info["truncated"])
    final_obs = info["final_obs"]
    assert final_obs.shape == obs.shape
    # every row is the observation BEFORE the auto-reset ...
    np.testing.assert_allclose(final_obs, pre_reset_obs, atol=1e-12)
    # ... which for a finished car is not the fresh spawn the env returned
    assert np.all(obs[done, 3] == 0.0) and np.all(final_obs[done, 3] > 0.0)
    np.testing.assert_array_equal(final_obs[~done], obs[~done])


def test_info_keys_at_timeout(oval, config):
    cfg = copy.deepcopy(config)
    cfg["reward"]["max_decisions"] = 5
    env = RacingEnv(oval, cfg, n_envs=2, seed=0)
    spawn_obs = env.reset()
    for step in range(5):
        obs, _, done, info = env.step(np.full(2, BRAKE))  # cars never move
        if step < 4:
            assert not np.any(done) and not np.any(info["truncated"])
            assert info["final_obs"] is None
    assert np.all(done) and np.all(info["truncated"]) and not np.any(info["off_track"])
    np.testing.assert_allclose(info["final_obs"], spawn_obs, atol=1e-12)
    assert not np.array_equal(info["final_obs"], obs)  # obs: re-jittered fresh spawns


def test_multi_track_env_merges_info_keys(config):
    spacing = config["track"]["resample_spacing"]
    tracks = [Track.load(name, spacing) for name in ("oval", "chicane")]
    env = MultiTrackEnv(tracks, config, n_envs=5, seed=4)
    env.reset()
    partial = False
    for _ in range(300):
        obs, _, done, info = env.step(np.full(5, STEER_POS))
        np.testing.assert_array_equal(info["truncated"], done & ~info["off_track"])
        if not np.any(done):
            assert info["final_obs"] is None
            continue
        final_obs = info["final_obs"]
        assert final_obs.shape == obs.shape
        np.testing.assert_array_equal(final_obs[~done], obs[~done])
        assert np.all(final_obs[done, 3] > 0.0) and np.all(obs[done, 3] == 0.0)
        if not np.all(done):
            partial = True  # only some sub-envs (maybe only one track) finished
            break
    assert partial


def test_multi_track_final_obs_is_the_pre_reset_obs_in_global_order(config):
    cfg = copy.deepcopy(config)
    cfg["reward"]["max_decisions"] = 25  # timeouts as well as crashes
    spacing = cfg["track"]["resample_spacing"]
    tracks = [Track.load(name, spacing) for name in ("oval", "chicane", "gp")]
    env = MultiTrackEnv(tracks, cfg, n_envs=7, seed=5)
    env.reset()
    pre_reset: dict[int, np.ndarray] = {}  # track -> its cars' obs just before a respawn
    for t, sub in env._envs:
        def spawn(mask, t=t, sub=sub, original=sub._spawn):
            pre_reset[t] = sub.observer.observe(sub.state)
            original(mask)

        sub._spawn = spawn
    rng = np.random.default_rng(1)
    truncated = crashed = 0
    for _ in range(120):
        pre_reset.clear()
        obs, _, done, info = env.step(rng.integers(4, size=7))
        if not np.any(done):
            assert info["final_obs"] is None
            continue
        for i in np.flatnonzero(done):  # global env i = car i // 3 of track i % 3
            np.testing.assert_array_equal(info["final_obs"][i], pre_reset[i % 3][i // 3])
        np.testing.assert_array_equal(info["final_obs"][~done], obs[~done])
        truncated += int(info["truncated"].sum())
        crashed += int(info["off_track"].sum())
    assert truncated > 0 and crashed > 0


# ------------------------------------------------------------ legacy defaults


@pytest.fixture(scope="module")
def short_config(config):
    cfg = copy.deepcopy(config)
    cfg["reward"]["max_decisions"] = 120  # keeps idling greedy evals short
    return cfg


def _short_run(oval, config, training_cfg: dict):
    env = RacingEnv(oval, config, n_envs=4, seed=5)
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=5)
    trainer = DQNTrainer(qfunc, env, training_cfg, rng=np.random.default_rng(5),
                         env_factory=lambda: RacingEnv(oval, config, n_envs=6, seed=77))
    history = trainer.train()
    return history, trainer, qfunc


@pytest.fixture(scope="module")
def legacy_run(oval, short_config):
    """A short run with no optional knob in the training table."""
    return _short_run(oval, short_config, dict(TRAINING_CFG))


def test_explicit_defaults_reproduce_the_legacy_run(oval, short_config, legacy_run):
    hist_a, trainer_a, qfunc_a = legacy_run
    hist_b, trainer_b, qfunc_b = _short_run(oval, short_config,
                                            {**TRAINING_CFG, **EXPLICIT_DEFAULTS})
    assert len(hist_a["losses"]) > 50  # the run really trained
    assert hist_a["episode_returns"] == hist_b["episode_returns"]
    assert hist_a["losses"] == hist_b["losses"]
    assert hist_a["eval_log"] == hist_b["eval_log"]
    np.testing.assert_array_equal(trainer_a.final_params, trainer_b.final_params)
    np.testing.assert_array_equal(qfunc_a.get_params(), qfunc_b.get_params())
    assert np.isscalar(trainer_b.optimizer.lr) and trainer_b.optimizer.lr == TRAINING_CFG["lr"]


def test_uniform_lr_groups_equal_the_scalar_lr(oval, short_config, legacy_run):
    hist_a, trainer_a, _ = legacy_run
    groups = {"body": TRAINING_CFG["lr"], "head": TRAINING_CFG["lr"]}
    hist_b, trainer_b, _ = _short_run(oval, short_config,
                                      {**TRAINING_CFG, "lr_groups": groups})
    assert hist_a["episode_returns"] == hist_b["episode_returns"]
    np.testing.assert_array_equal(trainer_a.final_params, trainer_b.final_params)


# ------------------------------------------------------ bootstrap_truncation


def _scripted_buffer(**options):
    env = ScriptedEnv(with_info=options.pop("with_info", True))
    cfg = {**TRAINING_CFG, "batch_size": NO_UPDATES, **options}
    trainer = DQNTrainer(ConstantPolicy(STRAIGHT, n_features=2), env, cfg,
                         rng=np.random.default_rng(0))
    history = trainer.train(episodes=2)
    assert trainer.buffer.size == 10  # 5 steps x 2 sub-envs, row = 2 * (t - 1) + env
    return trainer.buffer, history


def test_bootstrap_truncation_stores_final_obs_and_keeps_truncations_alive():
    buf, _ = _scripted_buffer(bootstrap_truncation=True)
    truncated_row, terminal_row = 2 * 2 + 0, 2 * 4 + 1
    np.testing.assert_array_equal(buf.next_obs[truncated_row], [-3.0, -3.0])  # final_obs
    assert buf.done[truncated_row] == 0.0  # time limit: keep bootstrapping
    np.testing.assert_array_equal(buf.next_obs[terminal_row], [-5.0, -5.0])
    assert buf.done[terminal_row] == 1.0  # off track stays terminal
    assert buf.done[:10].sum() == 1.0
    # unfinished rows of a step where the other sub-env finished are untouched
    np.testing.assert_array_equal(buf.next_obs[2 * 2 + 1], [3.0, 3.0])


def test_without_bootstrap_truncation_the_time_limit_is_terminal():
    buf, _ = _scripted_buffer()  # legacy default, info keys ignored
    truncated_row, terminal_row = 2 * 2 + 0, 2 * 4 + 1
    np.testing.assert_array_equal(buf.next_obs[truncated_row], [3.0, 3.0])  # fresh obs
    assert buf.done[truncated_row] == 1.0 and buf.done[terminal_row] == 1.0


def test_bootstrap_truncation_needs_the_info_keys():
    with pytest.raises(ValueError, match="bootstrap_truncation"):
        _scripted_buffer(bootstrap_truncation=True, with_info=False)


def test_bootstrap_truncation_on_the_racing_env(oval, config):
    cfg = copy.deepcopy(config)
    cfg["reward"]["max_decisions"] = 6  # too short to crash: every ending is a timeout
    env = RacingEnv(oval, cfg, n_envs=3, seed=8)
    tcfg = {**TRAINING_CFG, "batch_size": NO_UPDATES, "bootstrap_truncation": True}
    trainer = DQNTrainer(ConstantPolicy(STRAIGHT), env, tcfg, rng=np.random.default_rng(8))
    trainer.train(episodes=6)
    buf = trainer.buffer
    assert buf.size == 36 and not np.any(buf.done[:36])
    # the stored successor of a truncated transition is the moving car, not a
    # fresh spawn at standstill
    last_rows = np.arange(36).reshape(12, 3)[5::6].ravel()
    assert np.all(buf.next_obs[last_rows, 3] > 0.0)


def test_bootstrap_truncation_on_the_multi_track_env(config):
    cfg = copy.deepcopy(config)
    cfg["reward"]["max_decisions"] = 6
    spacing = cfg["track"]["resample_spacing"]
    tracks = [Track.load(name, spacing) for name in ("oval", "chicane")]
    env = MultiTrackEnv(tracks, cfg, n_envs=3, seed=8)
    tcfg = {**TRAINING_CFG, "batch_size": NO_UPDATES, "bootstrap_truncation": True}
    trainer = DQNTrainer(ConstantPolicy(STRAIGHT), env, tcfg, rng=np.random.default_rng(8))
    trainer.train(episodes=6)
    buf = trainer.buffer
    assert buf.size == 36 and not np.any(buf.done[:36])
    last_rows = np.arange(36).reshape(12, 3)[5::6].ravel()
    assert np.all(buf.next_obs[last_rows, 3] > 0.0)  # the moving car, not a respawn


# ------------------------------------------------- loss / clipping / targets


def test_huber_gradient_is_the_clipped_td_error(oval, config):
    delta = 0.6
    trainer = _filled_trainer(oval, config, loss="huber", huber_delta=delta)
    td = _td_errors(trainer)
    assert np.any(np.abs(td) > delta) and np.any(np.abs(td) < delta)  # both regimes
    calls = _spy(trainer.qfunc, "grad_selected")
    loss = trainer._update()
    upstream = calls[0][2]
    np.testing.assert_allclose(upstream, np.clip(td, -delta, delta) / trainer.batch_size,
                               rtol=0, atol=1e-15)
    assert np.max(np.abs(upstream)) == pytest.approx(delta / trainer.batch_size)
    expected = np.where(np.abs(td) <= delta, 0.5 * td**2, delta * (np.abs(td) - 0.5 * delta))
    assert loss == pytest.approx(float(expected.mean()))


def test_mse_gradient_is_the_raw_td_error(oval, config):
    trainer = _filled_trainer(oval, config)
    td = _td_errors(trainer)
    calls = _spy(trainer.qfunc, "grad_selected")
    loss = trainer._update()
    np.testing.assert_array_equal(calls[0][2], 2.0 * td / trainer.batch_size)
    assert loss == float(np.mean(td**2))


def test_grad_clip_bounds_the_global_norm(oval, config):
    raw = _filled_trainer(oval, config)
    raw_calls = _spy(raw.optimizer, "step")
    raw._update()
    raw_grad = raw_calls[0][1]
    raw_norm = float(np.linalg.norm(raw_grad))
    clip = 0.25 * raw_norm

    clipped = _filled_trainer(oval, config, grad_clip=clip)
    clipped_calls = _spy(clipped.optimizer, "step")
    clipped._update()
    grad = clipped_calls[0][1]
    assert float(np.linalg.norm(grad)) == pytest.approx(clip)
    np.testing.assert_allclose(grad, raw_grad * 0.25, rtol=1e-12)  # direction kept

    loose = _filled_trainer(oval, config, grad_clip=10.0 * raw_norm)
    loose_calls = _spy(loose.optimizer, "step")
    loose._update()
    np.testing.assert_array_equal(loose_calls[0][1], raw_grad)  # below the bound: untouched


def test_soft_target_moves_by_tau(oval, config):
    trainer = _filled_trainer(oval, config, target_update="soft", tau=0.1)
    before = trainer.target_params.copy()
    trainer._update()
    online = trainer.qfunc.get_params()
    assert not np.array_equal(online, before)
    np.testing.assert_allclose(trainer.target_params, 0.9 * before + 0.1 * online, rtol=1e-12)
    np.testing.assert_allclose(trainer.target_params - before, 0.1 * (online - before),
                               atol=1e-15)


def test_hard_target_only_syncs_every_n_updates(oval, config):
    trainer = _filled_trainer(oval, config, target_sync_every=3)
    before = trainer.target_params.copy()
    trainer._update()
    trainer._update()
    np.testing.assert_array_equal(trainer.target_params, before)
    trainer._update()
    np.testing.assert_array_equal(trainer.target_params, trainer.qfunc.get_params())


def test_action_gap_subtracts_the_gap_to_the_best_action(oval, config):
    """Advantage learning: target -= action_gap * (max_a Q(s, a) - Q(s, a_taken)),
    both from the target network; the best action's own target is untouched."""
    alpha = 0.7
    plain = _filled_trainer(oval, config)
    gapped = _filled_trainer(oval, config, action_gap=alpha)
    td = _td_errors(plain)  # same seeds: both trainers draw the same batch

    idx = copy.deepcopy(gapped.rng).integers(0, gapped.buffer.size, size=gapped.batch_size)
    rows = np.arange(gapped.batch_size)
    q_now = gapped.qfunc.q_values(gapped.buffer.obs[idx])  # target net == online net here
    gap = q_now.max(axis=1) - q_now[rows, gapped.buffer.action[idx]]
    assert np.any(gap > 0.0) and np.any(gap == 0.0)  # best and non-best actions in the batch

    calls = _spy(gapped.qfunc, "grad_selected")
    loss = gapped._update()
    np.testing.assert_allclose(calls[0][2], 2.0 * (td + alpha * gap) / gapped.batch_size,
                               rtol=0, atol=1e-15)
    assert loss == pytest.approx(float(np.mean((td + alpha * gap) ** 2)))
    np.testing.assert_array_equal(calls[0][2][gap == 0.0],
                                  (2.0 * td / gapped.batch_size)[gap == 0.0])


@pytest.mark.parametrize("bad", [{"loss": "l1"}, {"target_update": "polyak"}, {"tau": 0.0},
                                 {"huber_delta": 0.0}, {"grad_clip": -1.0},
                                 {"reward_scale": 0.0}, {"lr_end": -0.001},
                                 {"bootstrap_truncation": "off"}, {"action_gap": 1.0},
                                 {"action_gap": -0.1}])
def test_invalid_option_values_are_rejected(oval, config, bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        _filled_trainer(oval, config, steps=0, **bad)


# ----------------------------------------------------------- learning rates


def test_adam_accepts_a_per_parameter_lr_vector():
    grad = np.array([0.3, -2.0, 0.01, 5.0])
    lr = np.array([0.1, 0.1, 0.5, 0.001])
    vector = Adam(4, lr=lr).step(np.zeros(4), grad)
    for i in range(4):
        scalar = Adam(4, lr=float(lr[i])).step(np.zeros(4), grad)
        assert vector[i] == scalar[i]


def test_param_groups_partition_the_flat_vector():
    quantum = QuantumQFunction({"n_qubits": 6, "n_layers": 3, "n_actions": 4})
    groups = quantum.param_groups()
    assert groups == {"lam": slice(0, 18), "theta": slice(18, 54), "head": slice(54, 62)}
    params = quantum.get_params()
    np.testing.assert_array_equal(params[groups["lam"]], quantum.lam.ravel())
    np.testing.assert_array_equal(params[groups["theta"]], quantum.theta.ravel())
    np.testing.assert_array_equal(params[groups["head"]],
                                  np.concatenate([quantum.w, quantum.b]))

    mlp = MLPQFunction(n_features=4, hidden=8, n_actions=4)
    groups = mlp.param_groups()
    assert groups == {"body": slice(0, 40), "head": slice(40, 76)}
    np.testing.assert_array_equal(mlp.get_params()[groups["head"]],
                                  np.concatenate([mlp.W2.ravel(), mlp.b2]))


def test_lr_groups_apply_per_slice(oval, config):
    trainer = _filled_trainer(oval, config, lr=0.01, lr_groups={"head": 0.5})
    groups = trainer.qfunc.param_groups()
    lr = trainer.optimizer.lr
    assert np.all(lr[groups["body"]] == 0.01) and np.all(lr[groups["head"]] == 0.5)
    before = trainer.qfunc.get_params()
    calls = _spy(trainer.optimizer, "step")
    trainer._update()
    moved = np.abs(trainer.qfunc.get_params() - before)
    live = np.abs(calls[0][1]) > 1e-6  # the first Adam step is lr * sign(grad)
    assert live[groups["body"]].any() and live[groups["head"]].any()
    np.testing.assert_allclose(moved[groups["body"]][live[groups["body"]]], 0.01, rtol=1e-3)
    np.testing.assert_allclose(moved[groups["head"]][live[groups["head"]]], 0.5, rtol=1e-3)


def test_lr_groups_on_the_quantum_qfunction():
    qfunc = QuantumQFunction({"n_qubits": 4, "n_layers": 2})
    tcfg = {**TRAINING_CFG, "lr_groups": {"lam": 0.001, "head": 0.1}}
    lr = DQNTrainer(qfunc, None, tcfg).optimizer.lr
    np.testing.assert_array_equal(lr, [0.001] * 8 + [0.01] * 16 + [0.1] * 8)


def test_lr_groups_reject_unknown_groups_and_groupless_qfunctions():
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4)
    with pytest.raises(ValueError, match=r"unknown group.*theta.*body.*head"):
        DQNTrainer(qfunc, None, {**TRAINING_CFG, "lr_groups": {"theta": 0.1}})
    with pytest.raises(ValueError, match="param_groups"):
        DQNTrainer(ConstantPolicy(STRAIGHT), None, {**TRAINING_CFG, "lr_groups": {"head": 0.1}})


def test_lr_end_anneals_every_rate_linearly(oval, config):
    trainer = _filled_trainer(oval, config, lr=0.01, lr_end=0.001, lr_groups={"head": 0.5})
    groups = trainer.qfunc.param_groups()
    for frac, scale in ((0.0, 1.0), (0.5, 0.55), (1.0, 0.1), (3.0, 0.1)):
        trainer._anneal_lr(frac)
        np.testing.assert_allclose(trainer.optimizer.lr[groups["body"]], 0.01 * scale)
        np.testing.assert_allclose(trainer.optimizer.lr[groups["head"]], 0.5 * scale)


def test_lr_end_lowers_the_rate_over_a_run(oval, config):
    env = RacingEnv(oval, config, n_envs=4, seed=5)
    qfunc = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=5)
    tcfg = {**TRAINING_CFG, "lr_end": 0.001}
    trainer = DQNTrainer(qfunc, env, tcfg, rng=np.random.default_rng(5))
    seen: list[float] = []
    trainer.train(episodes=20, callback=lambda ep, stats: seen.append(trainer.optimizer.lr))
    assert seen[0] == pytest.approx(0.01, rel=0.2)
    assert all(a >= b for a, b in zip(seen, seen[1:], strict=False))
    assert 0.001 <= seen[-1] < 0.002


# --------------------------------------------------------------- reward_scale


def test_reward_scale_only_touches_the_replay_rewards(oval, config):
    def run(**options):
        env = RacingEnv(oval, config, n_envs=4, seed=3)
        tcfg = {**TRAINING_CFG, "batch_size": NO_UPDATES, **options}
        trainer = DQNTrainer(ConstantPolicy(STRAIGHT), env, tcfg, rng=np.random.default_rng(3))
        return trainer.train(episodes=8), trainer.buffer

    hist_raw, buf_raw = run()
    hist_scaled, buf_scaled = run(reward_scale=0.01)
    assert hist_raw["episode_returns"] == hist_scaled["episode_returns"]  # reported unscaled
    assert buf_raw.size == buf_scaled.size and np.any(buf_raw.reward != 0.0)
    np.testing.assert_allclose(buf_scaled.reward, 0.01 * buf_raw.reward, rtol=1e-12)


# ----------------------------------------------------------- eval episodes


def test_eval_runs_one_round_of_distinct_episodes(oval, config):
    built: list[RacingEnv] = []

    def env_factory():
        built.append(RacingEnv(oval, config, n_envs=12, seed=10_042))
        return built[-1]

    # flat out and straight: every car leaves the track at the first corner,
    # each from its own spawn jitter
    trainer = DQNTrainer(ConstantPolicy(STRAIGHT), RacingEnv(oval, config, n_envs=4, seed=0),
                         dict(TRAINING_CFG), rng=np.random.default_rng(0),
                         env_factory=env_factory)
    best = trainer._eval_snapshot(None, episode=0)
    assert len(built) == 1  # ONE env, ONE round
    assert best["eval_episodes"] == 12 and trainer.last_eval["eval_episodes"] == 12

    _, _, returns = trainer._greedy_eval_round(RacingEnv(oval, config, n_envs=12, seed=10_042))
    assert len(returns) == 12
    assert len({round(r, 9) for r in returns}) > 6  # distinct episodes, not repeats


def test_history_reports_every_eval_and_the_final_params(oval, short_config, legacy_run):
    history, trainer, qfunc = legacy_run
    log = history["eval_log"]
    assert len(log) >= 3
    for entry in log:
        assert set(entry) >= {"episode", "lapped_episodes", "eval_episodes", "mean_lap",
                              "best_lap"}
        assert entry["eval_episodes"] == 6
        assert 0 <= entry["lapped_episodes"] <= 6
    episodes = [entry["episode"] for entry in log]
    assert episodes == sorted(set(episodes))  # one entry per eval, none repeated
    assert episodes[-1] == len(history["episode_returns"])
    assert history["final_eval"] == log[-1]
    assert history["best_eval"]["episode"] in episodes
    json.dumps(history["eval_log"])  # plain JSON

    # final_params are the end-of-training params, kept even though qfunc is
    # left at the best snapshot: an eval-free twin run ends on exactly them
    env = RacingEnv(oval, short_config, n_envs=4, seed=5)
    twin = MLPQFunction(n_features=4, hidden=8, n_actions=4, seed=5)
    DQNTrainer(twin, env, dict(TRAINING_CFG), rng=np.random.default_rng(5)).train()
    np.testing.assert_array_equal(trainer.final_params, twin.get_params())
    if history["best_eval"]["episode"] != episodes[-1]:
        assert not np.array_equal(qfunc.get_params(), trainer.final_params)


def test_final_params_get_their_own_eval_when_the_run_ends_between_evals(oval, short_config):
    history, trainer, qfunc = _short_run(oval, short_config, {**TRAINING_CFG, "episodes": 25})
    n_done = len(history["episode_returns"])
    assert n_done % TRAINING_CFG["eval_every"] != 0  # no periodic eval on the last step
    log = history["eval_log"]
    assert len(log) == 3 and all(entry["episode"] < n_done for entry in log[:-1])
    assert history["final_eval"] == log[-1] and log[-1]["episode"] == n_done
    # ... and it really scored the final params
    qfunc.set_params(trainer.final_params)
    trainer._eval_snapshot(None, n_done)
    assert trainer.last_eval == history["final_eval"]


# ------------------------------------------------------ config helpers / CLI


def test_parse_override_reads_toml_literals():
    assert parse_override("circuit.n_layers=6") == ("circuit.n_layers", 6)
    assert parse_override("training.lr = 0.5") == ("training.lr", 0.5)
    assert parse_override('training.loss="huber"') == ("training.loss", "huber")
    assert parse_override("training.loss=huber") == ("training.loss", "huber")  # bare word
    assert parse_override("training.bootstrap_truncation=true")[1] is True
    assert parse_override("training.bootstrap_truncation=False")[1] is False
    assert parse_override("training.lr_groups={ head = 0.1 }")[1] == {"head": 0.1}
    assert parse_override("observation.ray_angles_deg=[-60.0, 0.0, 60.0]")[1] == [-60.0, 0.0, 60.0]
    for bad in ("training.lr", "lr=0.1", "training.lr=", ".lr=1", "training.lr=1 2",
                "training.lr=1\nseed=2"):
        with pytest.raises(ValueError, match="bad config override"):
            parse_override(bad)


def test_apply_overrides_sets_nested_keys():
    config = {"training": {"lr": 0.01}, "reward": {"lap_bonus": 50.0}}
    out = apply_overrides(config, ["training.lr_groups.head=0.1", "reward.lap_bonus=0",
                                   "mlp.hidden=16"])
    assert out is config
    assert config == {"training": {"lr": 0.01, "lr_groups": {"head": 0.1}},
                      "reward": {"lap_bonus": 0}, "mlp": {"hidden": 16}}
    apply_overrides(config, {"training.lr": 0.02})
    assert config["training"]["lr"] == 0.02
    assert apply_overrides(config, None) is config


def test_resolve_training_cfg_merges_track_presets(config):
    base = resolve_training_cfg(config, "oval")
    assert base == config["training"]
    gp = resolve_training_cfg(config, "gp")
    assert gp["gamma"] == config["training_presets"]["gp"]["gamma"] != base["gamma"]
    assert gp["lr"] == base["lr"]
    warm_gp = resolve_training_cfg(config, "gp", warm=True)
    assert warm_gp["episodes"] == config["training_warm_gp"]["episodes"]
    assert config["training"]["gamma"] == base["gamma"]  # inputs untouched


FAST = ["reward.max_decisions=25", "training.eval_episodes=5", "training.eval_every=4"]


def test_train_headless_records_overrides_evals_and_final_params(tmp_path):
    from traqmania.train_headless import train

    history_path = tmp_path / "history.json"
    summary = train("quantum", "oval", episodes=16, seed=1, profile=None,
                    out_dir=str(tmp_path), history_path=str(history_path),
                    overrides=[*FAST, "training.loss=huber", "circuit.n_layers=3"],
                    save_final=True)

    meta = json.loads((tmp_path / "quantum_oval.meta.json").read_text())
    assert meta["circuit"] == {"n_qubits": 4, "n_layers": 3, "n_actions": 4}
    assert meta["training"]["loss"] == "huber"
    assert meta["training"]["episodes"] == 16 and meta["training"]["seed"] == 1
    assert meta["training"]["eval_episodes"] == 5

    best = np.load(tmp_path / "quantum_oval.npz")["params"]
    final = np.load(tmp_path / "quantum_oval.final.npz")["params"]
    assert best.shape == final.shape == (3 * 3 * 4 + 8,)  # L = 3 via --set
    assert summary["final_weights_path"] == str(tmp_path / "quantum_oval.final.npz")

    assert len(summary["eval_log"]) >= 2
    assert all(entry["eval_episodes"] == 5 for entry in summary["eval_log"])
    assert summary["final_eval"] == summary["eval_log"][-1]
    assert summary["best_eval"]["eval_episodes"] == 5
    payload = json.loads(history_path.read_text())
    for key in ("eval_log", "best_eval", "final_eval"):
        assert payload[key] == summary[key]
    assert payload["training"]["loss"] == "huber"


def test_train_headless_takes_act_noise_and_action_gap(tmp_path, capsys):
    from traqmania.train_headless import train

    robust = ["training.act_noise={ attenuation = 0.95, shots = 256 }",
              "training.action_gap=0.5"]
    train("quantum", "oval", episodes=8, seed=1, profile=None, out_dir=str(tmp_path),
          overrides=[*FAST, *robust])
    out = capsys.readouterr().out
    assert "acting noise: attenuation 0.950, 256 shots" in out
    meta = json.loads((tmp_path / "quantum_oval.meta.json").read_text())
    assert meta["training"]["act_noise"] == {"attenuation": 0.95, "shots": 256}
    assert meta["training"]["action_gap"] == 0.5

    # plain [training] has neither; the quantum oval preset
    # ([training_presets_quantum.oval], the default recipe) switches both on
    train("quantum", "oval", episodes=8, seed=1, profile=None, out_dir=str(tmp_path),
          overrides=FAST, preset="none")
    assert "acting noise" not in capsys.readouterr().out
    meta = json.loads((tmp_path / "quantum_oval.meta.json").read_text())
    assert "act_noise" not in meta["training"] and "action_gap" not in meta["training"]
    recipe = load_config()["training_presets_quantum"]["oval"]
    train("quantum", "oval", episodes=8, seed=1, profile=None, out_dir=str(tmp_path),
          overrides=FAST)
    assert "acting noise: " in capsys.readouterr().out
    meta = json.loads((tmp_path / "quantum_oval.meta.json").read_text())
    assert meta["training"]["act_noise"] == recipe["act_noise"]
    assert meta["training"]["action_gap"] == recipe["action_gap"]
    # ... for the circuit only: the MLP on the same track trains without them
    train("mlp", "oval", episodes=8, seed=1, profile=None, out_dir=str(tmp_path),
          overrides=FAST)
    assert "acting noise" not in capsys.readouterr().out
    meta = json.loads((tmp_path / "mlp_oval.meta.json").read_text())
    assert "act_noise" not in meta["training"] and "action_gap" not in meta["training"]

    # the classical baseline has no readout expectations to add noise to
    with pytest.raises(ValueError, match="act_noise needs a Q-function with readout"):
        train("mlp", "oval", episodes=8, seed=1, profile=None, out_dir=str(tmp_path),
              overrides=[*FAST, robust[0]])


def test_save_final_writes_the_final_params_not_the_best_snapshot(tmp_path, monkeypatch):
    from traqmania import train_headless

    trainers: list[DQNTrainer] = []

    class FirstEvalWins(DQNTrainer):
        """Keeps the FIRST eval as the best snapshot, so best != final params."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trainers.append(self)

        def _eval_snapshot(self, best, episode):
            new = super()._eval_snapshot(best, episode)
            return new if best is None else best

    monkeypatch.setattr(train_headless, "DQNTrainer", FirstEvalWins)
    summary = train_headless.train("mlp", "oval", episodes=16, seed=1, profile=None,
                                   out_dir=str(tmp_path), overrides=FAST, save_final=True)
    best = np.load(tmp_path / "mlp_oval.npz")["params"]
    final = np.load(summary["final_weights_path"])["params"]
    assert summary["best_eval"]["episode"] < summary["final_eval"]["episode"]
    np.testing.assert_array_equal(final, trainers[0].final_params)
    assert not np.array_equal(best, final)


def test_train_headless_preset_precedence(tmp_path, monkeypatch, config):
    from traqmania import train_headless

    preset = config["training_presets"]["gp"]

    def trained(*argv: str) -> dict:
        out = tmp_path / str(len(list(tmp_path.iterdir())))
        monkeypatch.setattr(sys, "argv", ["train_headless", "--agent", "mlp", "--track", "gp",
                                          "--episodes", "4", "--out", str(out),
                                          *(arg for spec in FAST for arg in ("--set", spec)),
                                          *argv])
        train_headless.main()
        assert not (out / "mlp_gp.final.npz").exists()  # only with --save-final
        return json.loads((out / "mlp_gp.meta.json").read_text())["training"]

    auto = trained()
    assert auto["gamma"] == preset["gamma"]
    assert auto["epsilon_decay_episodes"] == preset["epsilon_decay_episodes"]
    assert auto["episodes"] == 4  # explicit --episodes beats the preset's 3000

    plain = trained("--preset", "none")
    assert plain["gamma"] == config["training"]["gamma"]
    assert plain["epsilon_decay_episodes"] == config["training"]["epsilon_decay_episodes"]

    forced = trained("--set", "training.gamma=0.9", "--seed", "7")
    assert forced["gamma"] == 0.9  # --set beats the preset
    assert forced["epsilon_decay_episodes"] == preset["epsilon_decay_episodes"]
    assert forced["seed"] == 7


def test_train_headless_rejects_unknown_training_overrides(tmp_path):
    from traqmania.train_headless import train

    for typo in ("training.target_updates=soft", "training.huber_dleta=5"):
        with pytest.raises(ValueError, match=r"unknown \[training\] override.*" + typo[:16]):
            train("mlp", "oval", episodes=2, seed=1, profile=None, out_dir=str(tmp_path),
                  overrides=[typo])
    assert not list(tmp_path.iterdir())  # rejected before anything trained


def test_train_headless_warns_about_light_cone_blind_spots(tmp_path, capsys):
    from traqmania.train_headless import train

    # 8 qubits at 4 blocks (set explicitly: the shipped q8 profile has 5 since
    # October 2026): every action is blind to one feature
    train("quantum", "oval", episodes=2, seed=1, profile="q8", out_dir=str(tmp_path),
          overrides=[*FAST, "circuit.n_layers=4"])
    out = capsys.readouterr().out
    assert "WARNING: light cone: n_layers = 4 is too shallow for 8 qubits" in out
    assert "full visibility needs n_layers >= 5" in out
    assert "WARNING:   Brake (Z_3) cannot see: speed" in out
    assert (tmp_path / "quantum_oval_q8.npz").is_file()  # a warning, not an error

    # deep enough, the default 4 qubits, or no circuit at all: silent
    train("quantum", "oval", episodes=2, seed=1, profile="q8", out_dir=str(tmp_path),
          overrides=[*FAST, "circuit.n_layers=5"])
    train("quantum", "oval", episodes=2, seed=1, profile=None, out_dir=str(tmp_path),
          overrides=FAST)
    train("mlp", "oval", episodes=2, seed=1, profile="q8", out_dir=str(tmp_path),
          overrides=FAST)
    assert "light cone" not in capsys.readouterr().out


def test_option_keys_list_every_optional_key_the_trainer_reads():
    read = set(re.findall(r'training_cfg\.get\(\s*"(\w+)"', inspect.getsource(DQNTrainer)))
    assert read - {"seed"} == set(OPTION_KEYS)
