"""Double-DQN training loop over vectorized environments, in pure numpy.

Works with any Q-function implementing the :class:`~racetraq.agents.base.QFunction`
protocol: the target network is just a second flat parameter vector that gets
swapped in to evaluate target Q-values, so quantum and classical backends share
this loop unchanged.

Env protocol: ``env.reset() -> obs (n_envs, F)``;
``env.step(actions (n_envs,) int) -> (obs, reward (n_envs,), done (n_envs,) bool, info)``,
where done sub-envs are auto-reset (the returned obs is the fresh one — safe here
because done transitions never bootstrap from next_obs).  With
``[training] bootstrap_truncation`` the env must also report ``info["truncated"]``
and ``info["final_obs"]`` (see ``RacingEnv.step``): time-limit endings are then
stored as non-terminal transitions into the pre-reset observation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

DEFAULT_EVAL_EPISODES = 12  # greedy episodes per snapshot eval ([training] eval_episodes)
LOSSES = ("mse", "huber")
TARGET_UPDATES = ("hard", "soft")
# Optional [training] keys the trainer reads (absent = the legacy behaviour).
# train_headless checks ``--set training.<key>`` against these plus the keys
# of the loaded [training] table, so a typo cannot silently train the baseline.
OPTION_KEYS = (
    "eval_every", "eval_episodes", "bootstrap_truncation", "loss", "huber_delta",
    "lr_groups", "lr_end", "target_update", "tau", "grad_clip", "reward_scale",
    "act_noise", "action_gap",
)


class Adam:
    """Standard bias-corrected Adam optimizer on a flat parameter vector.

    ``lr`` is a scalar or a per-parameter vector of shape ``(n_params,)``.
    """

    def __init__(
        self,
        n_params: int,
        lr: float | np.ndarray = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """One descent step; returns the updated parameter vector."""
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad**2
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class _ReplayBuffer:
    """Uniform ring-buffer replay memory backed by preallocated numpy arrays."""

    def __init__(self, capacity: int, n_features: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, n_features))
        self.action = np.zeros(capacity, dtype=np.intp)
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, n_features))
        self.done = np.zeros(capacity)
        self._ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        n = obs.shape[0]
        idx = (self._ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_obs[idx] = next_obs
        self.done[idx] = done
        self._ptr = (self._ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)


class DQNTrainer:
    """Double-DQN with epsilon-greedy exploration, replay, and a numpy Adam optimizer."""

    def __init__(
        self,
        qfunc: Any,
        env: Any,
        training_cfg: dict,
        rng: np.random.Generator | None = None,
        env_factory: Callable[[], Any] | None = None,
        stop_event: Any = None,
    ) -> None:
        """``env_factory`` (optional): zero-arg callable returning a FRESH eval env;
        when given, a greedy eval runs every ``eval_every`` (default 50) episodes and
        ``train()`` leaves ``qfunc`` at the best-scoring snapshot.  ``stop_event``
        (optional): ``threading.Event``-like; when set, ``train()`` returns early.

        Optional ``training_cfg`` keys (each defaults to the legacy behaviour):
        ``bootstrap_truncation`` (false), ``loss`` ("mse" | "huber") with
        ``huber_delta`` (10.0), ``lr_groups`` ({group: lr} over
        ``qfunc.param_groups()``), ``lr_end`` (= lr), ``target_update``
        ("hard" | "soft") with ``tau`` (0.005), ``grad_clip`` (0 = off) and
        ``reward_scale`` (1.0), ``act_noise`` (none: a ``{attenuation, shots,
        bias}`` table of expectation noise the rollouts and snapshot evals act
        under) and ``action_gap`` (0.0: advantage-learning coefficient) — see
        ``config/default.toml`` for what each does.
        """
        self.qfunc = qfunc
        self.env = env
        self.cfg = training_cfg
        self.rng = rng if rng is not None else np.random.default_rng(training_cfg.get("seed"))
        self.env_factory = env_factory
        self.stop_event = stop_event
        self.eval_every = int(training_cfg.get("eval_every", 50))
        # Greedy episodes per snapshot eval — the size ``env_factory`` is
        # expected to give its env (n_envs = eval_episodes): one eval is ONE
        # round of that many parallel episodes, each from its own spawn jitter.
        # (Re-running a small env built from the same seed only repeats the
        # same few episodes.)  12 instead of the old 4: a 4-episode eval once
        # crowned an 18.1 s gp headline that a 36-episode recheck put at 5/36
        # lapped episodes — reliability needs a sample, not a lucky roll.
        self.eval_episodes = int(training_cfg.get("eval_episodes", DEFAULT_EVAL_EPISODES))

        self.gamma = training_cfg["gamma"]
        self.batch_size = training_cfg["batch_size"]
        self.target_sync_every = training_cfg["target_sync_every"]
        self.epsilon_start = training_cfg["epsilon_start"]
        self.epsilon_end = training_cfg["epsilon_end"]
        self.epsilon_decay_episodes = training_cfg["epsilon_decay_episodes"]

        # Time-limit endings are truncations, not terminal states: when set,
        # they are stored with done = 0 and the pre-reset observation, so the
        # TD target keeps bootstrapping through them.
        flag = training_cfg.get("bootstrap_truncation", False)
        if isinstance(flag, str):  # bool("false") / bool("off") would be True
            raise ValueError(
                f"[training] bootstrap_truncation must be true or false, got '{flag}'"
            )
        self.bootstrap_truncation = bool(flag)
        self.loss = str(training_cfg.get("loss", "mse"))
        if self.loss not in LOSSES:
            raise ValueError(f"[training] loss must be one of {LOSSES}, got '{self.loss}'")
        self.huber_delta = float(training_cfg.get("huber_delta", 10.0))
        if self.huber_delta <= 0.0:
            raise ValueError(f"[training] huber_delta must be > 0, got {self.huber_delta}")
        self.target_update = str(training_cfg.get("target_update", "hard"))
        if self.target_update not in TARGET_UPDATES:
            raise ValueError(
                f"[training] target_update must be one of {TARGET_UPDATES}, "
                f"got '{self.target_update}'"
            )
        self.tau = float(training_cfg.get("tau", 0.005))
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"[training] tau must be in (0, 1], got {self.tau}")
        self.grad_clip = float(training_cfg.get("grad_clip", 0.0))
        if self.grad_clip < 0.0:
            raise ValueError(f"[training] grad_clip must be >= 0, got {self.grad_clip}")
        self.reward_scale = float(training_cfg.get("reward_scale", 1.0))
        if self.reward_scale <= 0.0:
            raise ValueError(f"[training] reward_scale must be > 0, got {self.reward_scale}")

        # Advantage learning: subtract action_gap * (V(s) - Q(s, a)) from the TD
        # target, which widens the gap between the best action and the rest
        # by 1 / (1 - action_gap) without changing which action is best.
        self.action_gap = float(training_cfg.get("action_gap", 0.0))
        if not 0.0 <= self.action_gap < 1.0:
            raise ValueError(f"[training] action_gap must be in [0, 1), got {self.action_gap}")
        # Acting noise: rollouts and snapshot evals pick actions from NOISY
        # readout expectations (what a device returns); TD targets and
        # gradients stay exact.  None = act on the exact Q-values.
        self.act_noise = None
        self._act_noise_rng: np.random.Generator | None = None
        if training_cfg.get("act_noise"):
            from racetraq.agents.quantum.noise import ExpectationNoise, act_noise_rng

            try:
                self.act_noise = ExpectationNoise.from_config(training_cfg["act_noise"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[training] act_noise: {exc}") from exc
            if self.act_noise is not None:
                if not hasattr(qfunc, "noisy_q_values"):
                    raise ValueError(
                        "[training] act_noise needs a Q-function with readout expectations "
                        f"(noisy_q_values); {type(qfunc).__name__} has none"
                    )
                for name in ("attenuation", "bias"):  # per-readout lists: one per action
                    value = getattr(self.act_noise, name)
                    if isinstance(value, tuple) and len(value) not in (1, qfunc.n_actions):
                        raise ValueError(
                            f"[training] act_noise: {name} lists {len(value)} values, "
                            f"the Q-function has {qfunc.n_actions} readouts"
                        )
                self._act_noise_rng = act_noise_rng(training_cfg.get("seed"))

        self.buffer = _ReplayBuffer(training_cfg["replay_size"], qfunc.n_features)
        lr = training_cfg["lr"]
        # Start-of-run learning rate(s): the scalar lr, or a per-parameter
        # vector when lr_groups assigns group-specific rates.
        self._lr_start = self._learning_rates(lr, training_cfg.get("lr_groups"))
        lr_end = training_cfg.get("lr_end", lr)
        if lr_end != lr and (not lr > 0.0 or lr_end < 0.0):  # negative rates ascend
            raise ValueError(
                f"[training] lr_end needs lr > 0 and lr_end >= 0, "
                f"got lr = {lr}, lr_end = {lr_end}"
            )
        self._lr_ratio = 1.0 if lr_end == lr else float(lr_end) / float(lr)
        self.optimizer = Adam(qfunc.get_params().shape[0], lr=self._lr_start)
        self.target_params = qfunc.get_params()
        self._updates = 0
        # Params at the end of the last train() call, BEFORE the best snapshot
        # is restored (None until train() has run).
        self.final_params: np.ndarray | None = None
        self.last_eval: dict | None = None  # eval_log entry of the latest snapshot eval

    def _learning_rates(self, lr: float, lr_groups: dict | None) -> float | np.ndarray:
        """The scalar ``lr``, or — with ``lr_groups`` — a per-parameter vector:
        each named group of ``qfunc.param_groups()`` gets its own rate, groups
        not listed keep ``lr``."""
        if not lr_groups:
            return lr
        param_groups = getattr(self.qfunc, "param_groups", None)
        if param_groups is None:
            raise ValueError(
                f"[training] lr_groups needs a Q-function with param_groups(); "
                f"{type(self.qfunc).__name__} has none"
            )
        groups = param_groups()
        unknown = sorted(set(lr_groups) - set(groups))
        if unknown:
            raise ValueError(
                f"[training] lr_groups: unknown group(s) {unknown}; "
                f"{type(self.qfunc).__name__} has {sorted(groups)}"
            )
        rates = np.full(self.qfunc.get_params().shape[0], float(lr))
        for name, rate in lr_groups.items():
            rates[groups[name]] = float(rate)
        return rates

    def _anneal_lr(self, frac: float) -> None:
        """Linear lr schedule: every rate goes from its start value (frac = 0)
        to start * lr_end / lr (frac = 1).  No-op without ``lr_end``."""
        if self._lr_ratio != 1.0:
            self.optimizer.lr = self._lr_start * (1.0 + (self._lr_ratio - 1.0) * min(1.0, frac))

    def _epsilon(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def _acting_q(self, obs: np.ndarray) -> np.ndarray:
        """Q-values the behaviour policy acts on: exact, or — with ``act_noise``
        — computed from noisy readout expectations."""
        if self.act_noise is None:
            return self.qfunc.q_values(obs)
        return self.qfunc.noisy_q_values(obs, self.act_noise, self._act_noise_rng)

    def _select_actions(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        greedy = np.argmax(self._acting_q(obs), axis=1)
        random_a = self.rng.integers(self.qfunc.n_actions, size=obs.shape[0])
        explore = self.rng.random(obs.shape[0]) < epsilon
        return np.where(explore, random_a, greedy)

    def _loss_and_upstream(self, td: np.ndarray) -> tuple[float, np.ndarray]:
        """Batch-mean loss of the TD errors and its gradient wrt each selected Q.

        MSE: mean(td^2).  Huber: 0.5 * td^2 inside +-huber_delta, linear
        outside — so its gradient is the TD error CLIPPED to +-huber_delta,
        which keeps a few huge errors (Q-values here are of order 100) from
        dominating the batch.
        """
        if self.loss == "huber":
            delta = self.huber_delta
            abs_td = np.abs(td)
            loss = np.where(abs_td <= delta, 0.5 * td**2, delta * (abs_td - 0.5 * delta))
            return float(np.mean(loss)), np.clip(td, -delta, delta) / self.batch_size
        return float(np.mean(td**2)), 2.0 * td / self.batch_size  # d(MSE)/d(Q_sel)

    def _update(self) -> float:
        """One double-DQN gradient step on a uniform replay batch; returns the loss."""
        idx = self.rng.integers(0, self.buffer.size, size=self.batch_size)
        obs = self.buffer.obs[idx]
        action = self.buffer.action[idx]
        reward = self.buffer.reward[idx]
        next_obs = self.buffer.next_obs[idx]
        done = self.buffer.done[idx]
        rows = np.arange(self.batch_size)

        # Double DQN: online net picks a*, target net evaluates it.
        a_star = np.argmax(self.qfunc.q_values(next_obs), axis=1)
        online_params = self.qfunc.get_params()
        self.qfunc.set_params(self.target_params)
        q_next = self.qfunc.q_values(next_obs)[rows, a_star]
        if self.action_gap > 0.0:
            q_now = self.qfunc.q_values(obs)  # target net, for the gap term
        self.qfunc.set_params(online_params)
        target = reward + self.gamma * (1.0 - done) * q_next
        if self.action_gap > 0.0:
            target = target - self.action_gap * (q_now.max(axis=1) - q_now[rows, action])

        q_sel = self.qfunc.q_values(obs)[rows, action]
        td = q_sel - target
        loss, upstream = self._loss_and_upstream(td)
        grad = self.qfunc.grad_selected(obs, action, upstream)
        if self.grad_clip > 0.0:  # global-norm clipping
            norm = float(np.linalg.norm(grad))
            if norm > self.grad_clip:
                grad = grad * (self.grad_clip / norm)
        self.qfunc.set_params(self.optimizer.step(online_params, grad))

        self._updates += 1
        if self.target_update == "soft":  # Polyak averaging, every update
            self.target_params = (
                (1.0 - self.tau) * self.target_params + self.tau * self.qfunc.get_params()
            )
        elif self._updates % self.target_sync_every == 0:
            self.target_params = self.qfunc.get_params()
        return loss

    def _stored_transition(
        self, next_obs: np.ndarray, done: np.ndarray, info: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """(next_obs, done) as stored in replay under ``bootstrap_truncation``:
        finished sub-envs get the observation BEFORE their auto-reset, and only
        off-track endings stay terminal — a time-limit truncation keeps
        bootstrapping from that final observation."""
        if not isinstance(info, dict) or "truncated" not in info or "final_obs" not in info:
            raise ValueError(
                "[training] bootstrap_truncation needs an env whose step() info "
                "reports 'truncated' and 'final_obs' (RacingEnv / MultiTrackEnv do); "
                f"{type(self.env).__name__} does not"
            )
        done = np.asarray(done, dtype=bool)
        if not done.any():
            return next_obs, done
        if info["final_obs"] is None:
            raise ValueError(
                "[training] bootstrap_truncation: a sub-env finished but the env "
                "reported info['final_obs'] = None"
            )
        next_obs = np.where(done[:, None], info["final_obs"], next_obs)
        return next_obs, done & ~np.asarray(info["truncated"], dtype=bool)

    def _greedy_eval_round(self, env, max_steps: int = 5000):
        """One greedy (epsilon = 0) round: one full episode per env, in parallel
        (under ``act_noise`` when set: greedy on the noisy Q-values).

        Counts laps INCREMENTALLY while each env's first episode is still
        running and stops once every env has finished one episode (crash or
        step cap) — counting laps only at ``done`` undercounts exactly the
        good policies, whose envs lap on without finishing, while crashing
        envs "finish" fast.  Returns (lapped_episodes, lap_times, episode_returns)
        for the round.  Expects the racing-env info dict (``lap``,
        ``last_lap_time``).
        """
        obs = env.reset()
        n_envs = obs.shape[0]
        done_seen = np.zeros(n_envs, dtype=bool)
        prev_lap = np.zeros(n_envs, dtype=np.int64)
        lapped = np.zeros(n_envs, dtype=bool)
        lap_times: list[float] = []
        return_acc = np.zeros(n_envs)
        episode_returns: list[float] = []
        for _ in range(max_steps):
            actions = np.argmax(self._acting_q(obs), axis=1)
            obs, reward, done, info = env.step(actions)
            done = np.asarray(done, dtype=bool)
            active = ~done_seen
            return_acc[active] += np.asarray(reward)[active]
            if isinstance(info, dict) and "lap" in info:
                lap = np.asarray(info["lap"], dtype=np.int64)
                event = (lap > prev_lap) & active  # a lap finished this step
                lapped |= event
                lt = np.asarray(info.get("last_lap_time", np.nan), dtype=np.float64)
                lap_times.extend(float(t) for t in lt[event & ~np.isnan(lt)])
                prev_lap = np.where(done, 0, lap)  # env auto-resets on done
            for i in np.flatnonzero(done & active):
                episode_returns.append(float(return_acc[i]))
            done_seen |= done
            if done_seen.all():
                break
        return int(lapped.sum()), lap_times, episode_returns

    def _eval_snapshot(self, best: dict | None, episode: int) -> dict:
        """Greedy-eval the current params on a fresh env; keep the best snapshot.

        Calls ``env_factory()`` ONCE and runs one round of ``env.n_envs``
        parallel greedy episodes — distinct episodes, one per spawn jitter
        (the factory sizes the env; ``eval_episodes`` is the intended size).
        Scores lexicographically (lapped_episodes, -mean_lap,
        mean_return): reliability first — the number of EPISODES that produced
        a lap — then average pace over every lap driven, then eval return.
        Mean lap over all laps (not the single best) keeps one lucky lap from
        crowning an unreliable snapshot; the return tie-breaker matters before
        the first lap, where (0, inf) would otherwise tie forever and pin the
        earliest (untrained) snapshot.  The eval's summary is left in
        ``self.last_eval`` whether or not it becomes the best.
        """
        env = self.env_factory()
        lapped, lap_times, episode_returns = self._greedy_eval_round(env)
        episodes_run = env.n_envs if hasattr(env, "n_envs") else max(1, len(episode_returns))
        mean_lap = float(np.mean(lap_times)) if lap_times else float("inf")
        mean_return = float(np.mean(episode_returns)) if episode_returns else float("-inf")
        score = (lapped, -mean_lap, mean_return)
        self.last_eval = {
            "episode": episode,
            "lapped_episodes": lapped,
            "eval_episodes": episodes_run,
            "mean_lap": None if np.isinf(mean_lap) else mean_lap,
            "best_lap": min(lap_times) if lap_times else None,
            "mean_return": None if np.isinf(mean_return) else mean_return,
        }
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "params": self.qfunc.get_params(),
                "episode": episode,
                "lapped_episodes": lapped,
                "eval_episodes": episodes_run,
                "mean_lap": None if np.isinf(mean_lap) else mean_lap,
                "laps": len(lap_times),
                "best_lap": min(lap_times) if lap_times else None,
            }
        return best

    def train(
        self,
        episodes: int | None = None,
        callback: Callable[[int, dict], None] | None = None,
    ) -> dict:
        """Run until ``episodes`` sub-env episodes have completed; returns history.

        With ``env_factory`` set, a greedy eval runs every ``eval_every`` episodes
        (plus once at the end); ``qfunc`` is left at the BEST-scoring params and
        ``history["best_eval"]`` reports {episode, laps, best_lap}.
        ``history["eval_log"]`` lists EVERY eval ({episode, lapped_episodes,
        eval_episodes, mean_lap, best_lap, mean_return}) and
        ``history["final_eval"]`` is its last entry — the eval of the final
        params, which stay available as ``self.final_params``.
        """
        episodes = episodes if episodes is not None else self.cfg["episodes"]
        t_start = time.perf_counter()

        obs = self.env.reset()
        n_envs = obs.shape[0]
        return_acc = np.zeros(n_envs)
        episode_returns: list[float] = []
        losses: list[float] = []
        best: dict | None = None
        evals_done = 0
        eval_log: list[dict] = []
        evaluated_at = -1  # self._updates at the latest eval (params change only there)

        while len(episode_returns) < episodes:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            epsilon = self._epsilon(len(episode_returns))
            self._anneal_lr(len(episode_returns) / max(1, episodes))
            actions = self._select_actions(obs, epsilon)
            next_obs, reward, done, info = self.env.step(actions)

            stored_next, stored_done = next_obs, done
            if self.bootstrap_truncation:
                stored_next, stored_done = self._stored_transition(next_obs, done, info)
            # reward_scale shapes the replay targets only; reported returns stay raw
            stored_reward = reward if self.reward_scale == 1.0 else reward * self.reward_scale
            self.buffer.add(obs, actions, stored_reward, stored_next, stored_done)
            return_acc += reward
            obs = next_obs

            # ONE gradient update per env decision-step once the buffer is warm.
            if self.buffer.size >= self.batch_size:
                losses.append(self._update())

            for i in np.flatnonzero(done):
                episode_returns.append(float(return_acc[i]))
                if callback is not None:
                    last_loss = losses[-1] if losses else float("nan")
                    callback(
                        len(episode_returns) - 1,
                        dict(returns=episode_returns[-1], epsilon=epsilon, loss=last_loss),
                    )
            return_acc[done] = 0.0

            if (self.env_factory is not None
                    and len(episode_returns) >= (evals_done + 1) * self.eval_every):
                evals_done = len(episode_returns) // self.eval_every
                best = self._eval_snapshot(best, len(episode_returns))
                eval_log.append(self.last_eval)
                evaluated_at = self._updates

        history = {
            "episode_returns": episode_returns,
            "losses": losses,
        }
        self.final_params = self.qfunc.get_params()
        if self.env_factory is not None:
            # Final params compete too — unless the last periodic eval already
            # scored exactly these params (no update since).
            if evaluated_at != self._updates:
                best = self._eval_snapshot(best, len(episode_returns))
                eval_log.append(self.last_eval)
            self.qfunc.set_params(best["params"])
            history["best_eval"] = {
                k: best[k]
                for k in ("episode", "lapped_episodes", "eval_episodes",
                          "mean_lap", "laps", "best_lap")
            }
            history["eval_log"] = eval_log
            history["final_eval"] = eval_log[-1]
        history["wall_time_s"] = time.perf_counter() - t_start
        return history
