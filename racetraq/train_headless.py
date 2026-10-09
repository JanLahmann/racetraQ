"""Headless training entry point for racetraQ baselines.

Run as ``python -m racetraq.train_headless --agent mlp --track oval
[--episodes N --seed S --profile P]``.  Builds the vectorized racing env,
a Q-function, and the double-DQN trainer, trains for the requested number of
sub-env episodes, prints a per-20-episode mean-return trace, reports the first
episode (and wall-clock second) at which a full clean lap occurred, and saves
the learned weights as ``<agent>_<track>.npz`` plus a JSON metadata sidecar:
to ``--out DIR``, by default a new ``runs/<agent>_<track>_<time>/`` folder.
``--out bundled`` writes ``racetraq/weights/`` and replaces the shipped
driver.

The training recipe is ``[training]`` with the track's
``[training_presets.<track>]`` merged on top (``--preset none`` skips that);
``--set section.key=value`` overrides any config value and ``--episodes`` /
``--seed`` win over everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from racetraq.agents.base import N_ACTIONS
from racetraq.agents.classical import MLPQFunction
from racetraq.agents.training import DQNTrainer
from racetraq.agents.training.dqn import DEFAULT_EVAL_EPISODES, OPTION_KEYS
from racetraq.config import apply_overrides, load_config, parse_override, resolve_training_cfg
from racetraq.env.multi_track import MultiTrackEnv
from racetraq.env.racing_env import RacingEnv
from racetraq.env.track import Track

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
BUNDLED = "bundled"  # --out value that writes WEIGHTS_DIR
REPORT_EVERY = 20  # episodes per mean-return line
MULTI_TRACK_NAMES = ("oval", "chicane", "gp", "combo")  # the --track multi mixture
PRESET_MODES = ("auto", "none")  # --preset: merge [training_presets.<track>] or not


class CleanLapMonitor:
    """Env wrapper that records the first CLEAN lap seen during training.

    A clean lap means a sub-env reached ``lap >= 1`` while still on track;
    because off-track immediately ends an episode, any completed lap was
    driven entirely on the racing surface.  Records the 1-based index of the
    episode it happened in (the episode then in progress) and the wall-clock
    seconds since ``reset()``.
    """

    def __init__(self, env: RacingEnv | MultiTrackEnv) -> None:
        self.env = env
        self.episodes_done = 0
        self.first_clean_episode: int | None = None
        self.first_clean_wall_s: float | None = None
        self.best_lap_s: float = float("nan")
        self._t0 = time.perf_counter()

    def reset(self) -> np.ndarray:
        self._t0 = time.perf_counter()
        return self.env.reset()

    def step(self, actions: np.ndarray):
        obs, reward, done, info = self.env.step(actions)
        if self.first_clean_episode is None:
            clean = (info["lap"] >= 1) & ~info["off_track"]
            if np.any(clean):
                self.first_clean_episode = self.episodes_done + 1
                self.first_clean_wall_s = time.perf_counter() - self._t0
        laps = info["last_lap_time"]
        if np.any(~np.isnan(laps)):
            self.best_lap_s = np.nanmin([self.best_lap_s, np.nanmin(laps)])
        self.episodes_done += int(np.sum(done))
        return obs, reward, done, info


def build_qfunc(agent: str, n_features: int, seed: int, config: dict,
                n_actions: int = N_ACTIONS):
    """Q-function factory: classical MLP baseline or the quantum circuit."""
    if agent == "mlp":
        hidden = int(config.get("mlp", {}).get("hidden", 8))
        return MLPQFunction(n_features=n_features, hidden=hidden,
                            n_actions=n_actions, seed=seed)
    if agent == "quantum":
        # numpy-only fast path (fastsim + adjoint); keeps headless training qiskit-free
        from racetraq.agents.quantum.qdqn import QuantumQFunction

        return QuantumQFunction(config["circuit"], seed=seed)
    raise ValueError(f"unknown agent '{agent}' (expected 'mlp' or 'quantum')")


def _json_default(value: Any):
    """JSON fallback for numpy scalars/arrays in programmatic config overrides."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def config_hash(config: dict) -> str:
    """Short stable hash of the fully-resolved config dict."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def save_weights(qfunc, agent: str, track_name: str, config: dict, episodes: int,
                 out_dir: Path | None = None, training_cfg: dict | None = None) -> Path:
    """Write ``<agent>_<track>.npz`` (params) + ``.meta.json`` sidecar; returns npz path.

    Non-default circuit sizes get a ``_q<n>`` filename tag (both agents, so a
    q6/q8/q10 training run never clobbers the bundled 4-feature weights).
    ``training_cfg`` (the resolved training table the run actually used:
    preset, pace and overrides merged) is recorded in the sidecar when given.
    """
    out_dir = Path(out_dir) if out_dir is not None else WEIGHTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    n_qubits = int(config.get("circuit", {}).get("n_qubits", 4))
    qtag = "" if n_qubits == 4 else f"_q{n_qubits}"
    npz_path = out_dir / f"{agent}_{track_name}{qtag}.npz"
    np.savez(npz_path, params=qfunc.get_params())
    obs_cfg = config["observation"]
    meta = {
        "agent": agent,
        "track": track_name,
        "config_hash": config_hash(config),
        "episodes": episodes,
        # the circuit shape the params belong to (n_layers is not recoverable
        # from the filename; an MLP only uses n_qubits as its feature count)
        "circuit": {
            "n_qubits": n_qubits,
            "n_layers": int(config.get("circuit", {}).get("n_layers", 4)),
            "n_actions": int(qfunc.n_actions),
        },
        # what the driver was trained to see; loaders (see
        # runtime.weights_observation) overlay this on the profile obs
        "observation": {
            "ray_angles_deg": [float(a) for a in obs_cfg["ray_angles_deg"]],
            "features": [str(k) for k in obs_cfg.get("features", ["rays", "speed"])],
            **({"lookahead_m": float(obs_cfg["lookahead_m"])}
               if "lookahead_m" in obs_cfg else {}),
        },
        # how many discrete actions the driver was trained to pick between;
        # loaders (see runtime.weights_actions) adopt this per driver so a
        # 6/8-action policy drives with the action table it learned
        "actions": {"n_actions": int(qfunc.n_actions)},
        "date": "DATE",
    }
    if training_cfg is not None:
        meta["training"] = dict(training_cfg)
    meta_path = npz_path.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default) + "\n",
                         encoding="utf-8")
    return npz_path


def _adopt_init_circuit(config: dict, init: str | Path, explicit_layers: bool,
                        explicit_actions: bool) -> int:
    """``--init`` on a quantum agent: the run continues at the circuit depth
    and action count of the init weights (``runtime.weights_circuit``: their
    sidecar's ``circuit`` block, else their parameter count), written into
    ``config["circuit"]`` — with one printed line when that changes anything.

    A depth the caller asked for EXPLICITLY (``--set circuit.n_layers``) that
    contradicts the weights raises ``ValueError``, as do weights of another
    qubit count or of a parameter count that fits no depth.  An explicit
    action count (``--actions`` / ``circuit.n_actions``) LARGER than the
    weights' widens them (:func:`widen_actions`: the action tables are
    prefixes of each other); a smaller one raises.  Returns the init weights'
    action count.
    """
    from racetraq.server.runtime import weights_circuit

    name = Path(init).name
    circuit = config.setdefault("circuit", {})
    n_qubits = int(circuit.get("n_qubits", 4))
    have = {"n_layers": int(circuit.get("n_layers", 4)),
            "n_actions": int(circuit.get("n_actions", min(4, n_qubits)))}
    shape = weights_circuit(Path(init), n_qubits, circuit.get("n_actions"))
    if shape["n_qubits"] != n_qubits:
        raise ValueError(f"--init weights '{name}' belong to a {shape['n_qubits']}-qubit "
                         f"circuit, but this run has [circuit] n_qubits = {n_qubits} — "
                         "use the matching --profile")
    if explicit_layers and shape["n_layers"] != have["n_layers"]:
        raise ValueError(f"--init weights '{name}' have {shape['n_layers']} blocks, but "
                         f"circuit.n_layers = {have['n_layers']} was requested — drop the "
                         "override (the run continues at the init weights' depth) or "
                         "start from weights of that depth")
    if explicit_actions and shape["n_actions"] > have["n_actions"]:
        raise ValueError(f"--init weights '{name}' use {shape['n_actions']} actions, but "
                         f"{have['n_actions']} were requested (--actions / "
                         "circuit.n_actions) — an action set can be widened, not "
                         "narrowed; drop the override or start from weights with at "
                         "most that many actions")
    keys = ("n_layers",) if explicit_actions else tuple(have)
    changed = {key: shape[key] for key in keys if shape[key] != have[key]}
    if changed:
        circuit.update(changed)
        print(f"circuit shape from --init {name}: {shape['n_layers']} blocks, "
              f"{shape['n_actions']} actions (config: {have['n_layers']} blocks, "
              f"{have['n_actions']} actions)")
    return int(shape["n_actions"])


def widen_actions(params: np.ndarray, n_from: int, n_to: int) -> np.ndarray:
    """Quantum weights for ``n_from`` actions -> the same driver with ``n_to``.

    The parameter vector ends in the readout head (w, b), one entry per
    action; the circuit parameters before it do not depend on the action
    count.  The added actions read their own qubits (``Q_a = w_a <Z_a> +
    b_a``) and get w = mean(w) and a PESSIMISTIC bias: below every old
    action's lowest possible value, b = min_a(b_a - |w_a|) - |mean(w)|.  So
    the widened driver starts out driving exactly like the old one (greedy
    never picks a new action until training has raised its value) and
    exploration has to earn each new action its place.
    """
    params = np.asarray(params, dtype=np.float64)
    if n_to < n_from:
        raise ValueError(f"cannot narrow {n_from} actions to {n_to}")
    n_circuit = params.size - 2 * n_from
    w, b = params[n_circuit:n_circuit + n_from], params[n_circuit + n_from:]
    w_new = float(np.mean(w))
    b_new = float(np.min(b - np.abs(w))) - abs(w_new)
    extra = n_to - n_from
    return np.concatenate([params[:n_circuit], w, np.full(extra, w_new),
                           b, np.full(extra, b_new)])


def train(agent: str, track_name: str, episodes: int | None, seed: int | None,
          profile: str | None, out_dir: str | None = None, init: str | None = None,
          history_path: str | None = None, actions: int | None = None,
          pace: bool = False, preset: str = "auto",
          overrides: Mapping[str, Any] | Iterable[str] | None = None,
          save_final: bool = False) -> dict:
    """Build env/agent/trainer from config, train, print progress, save weights.

    ``actions`` overrides ``[circuit] n_actions`` (the 6/8-action scaled
    readout).  ``pace`` merges the ``[training_pace]`` fine-tune recipe onto
    the training config — low epsilon plus a per-decision time penalty, so the
    objective becomes lap time rather than reliable progress; meant to be
    combined with ``--init`` on an already-lapping snapshot.  A quantum
    ``init`` brings its circuit depth and action count along
    (:func:`_adopt_init_circuit`); asking explicitly for another one
    (``actions``, a ``circuit.n_layers`` / ``circuit.n_actions`` override)
    raises instead of failing in ``set_params``.

    ``preset="auto"`` merges ``[training_presets.<track>]`` onto ``[training]``
    (the server's rule, :func:`racetraq.config.resolve_training_cfg`);
    ``"none"`` trains on plain ``[training]``.  ``overrides`` are dotted-key
    config overrides — ``"section.key=value"`` strings or a
    ``{"section.key": value}`` mapping — applied last, so they beat the
    preset and pace recipes; explicit ``episodes`` / ``seed`` beat those too.
    A ``training.<key>`` override the trainer does not know raises (a typo
    would otherwise train the baseline unnoticed).
    ``save_final`` also writes ``<name>.final.npz``: the params at the end of
    training, next to the best-snapshot weights.

    Returns a summary dict: episode returns, wall time, first-clean-lap
    episode/second (None if no clean lap happened), every greedy eval
    (``eval_log``) with the best and final ones, and the saved paths.
    """
    if preset not in PRESET_MODES:
        raise ValueError(f"preset must be one of {PRESET_MODES}, got '{preset}'")
    if overrides is None:
        overrides = {}
    elif not isinstance(overrides, Mapping):
        overrides = dict(parse_override(spec) for spec in overrides)
    config = load_config(profile=profile)
    # A mistyped training key would be ignored and silently train the baseline.
    known = set(config["training"]) | set(OPTION_KEYS)
    unknown = sorted(key for key in overrides
                     if key.startswith("training.") and key.split(".")[1] not in known)
    if unknown:
        raise ValueError(f"unknown [training] override(s) {unknown}; "
                         f"known keys: {sorted(known)}")
    apply_overrides(config, overrides)
    if actions is not None:
        config.setdefault("circuit", {})["n_actions"] = int(actions)
    if preset == "auto":
        training_cfg = resolve_training_cfg(config, track_name, agent=agent)
    else:
        training_cfg = dict(config["training"])
    if pace:
        pace_cfg = dict(config.get("training_pace", {}))
        config["reward"]["time_penalty"] = float(pace_cfg.pop("time_penalty", 0.5))
        training_cfg.update(pace_cfg)
        if init is None:
            print("WARNING: --pace without --init fine-tunes random weights; "
                  "expected use is on an already-lapping snapshot")
    # Explicit overrides win over the recipes merged above: re-apply them to
    # the config (pace sets reward.time_penalty) and to the resolved table.
    apply_overrides(config, overrides)
    apply_overrides({"training": training_cfg},
                    {k: v for k, v in overrides.items() if k.startswith("training.")})
    if seed is not None:
        training_cfg["seed"] = seed
    seed = int(training_cfg["seed"])
    episodes = int(episodes) if episodes is not None else int(training_cfg["episodes"])
    training_cfg["episodes"] = episodes  # the resolved table records the actual run
    init_actions = None
    if init is not None and agent == "quantum":
        init_actions = _adopt_init_circuit(
            config, init, explicit_layers="circuit.n_layers" in overrides,
            explicit_actions=actions is not None or "circuit.n_actions" in overrides)

    spacing = config["track"]["resample_spacing"]
    n_parallel = training_cfg["n_parallel_envs"]
    # One snapshot eval = one round of n_eval parallel greedy episodes, each
    # from its own spawn jitter (a small env rebuilt from the same seed would
    # just replay the same few episodes).
    n_eval = int(training_cfg.get("eval_episodes", DEFAULT_EVAL_EPISODES))
    if track_name in ("multi", "random"):
        # Mixture training: one policy over several tracks (the universal
        # candidates); weights are saved under the literal name multi/random.
        if track_name == "multi":
            tracks = [Track.load(name, spacing) for name in MULTI_TRACK_NAMES]
            env = MultiTrackEnv(tracks, config, n_envs=n_parallel, seed=seed)
        else:
            env = MultiTrackEnv.random_pool(config, n_envs=n_parallel, seed=seed)
            tracks = env.tracks
        save_name = track_name

        def env_factory(tracks=tracks, config=config, seed=seed) -> MultiTrackEnv:
            return MultiTrackEnv(tracks, config, n_envs=n_eval, seed=seed + 10_000)
    else:
        track = Track.load(track_name, spacing)
        env = RacingEnv(track, config, n_envs=n_parallel, seed=seed)
        save_name = track.name

        def env_factory(track=track, config=config, seed=seed) -> RacingEnv:
            return RacingEnv(track, config, n_envs=n_eval, seed=seed + 10_000)

    monitor = CleanLapMonitor(env)
    qfunc = build_qfunc(agent, env.n_features, seed, config, n_actions=env.n_actions)
    if qfunc.n_actions != env.n_actions:
        raise ValueError(f"agent has {qfunc.n_actions} actions but the env's "
                         f"action table has {env.n_actions}")
    if init is not None:
        params = np.load(init)["params"]
        if init_actions is not None and init_actions < qfunc.n_actions:
            params = widen_actions(params, init_actions, qfunc.n_actions)
            print(f"widened the init weights from {init_actions} to {qfunc.n_actions} actions")
        qfunc.set_params(params)
        print(f"warm-started from {init}")
    if agent == "quantum":
        # too few blocks for the ring: some actions decide without some features
        from racetraq.agents.quantum.lightcone import blind_spot_warning

        for line in blind_spot_warning(env.feature_names, qfunc.n_layers, qfunc.n_actions):
            print(f"WARNING: {line}")

    trainer = DQNTrainer(qfunc, monitor, training_cfg, rng=np.random.default_rng(seed),
                         env_factory=env_factory)

    print(f"training agent={agent} track={save_name} episodes={episodes} seed={seed}")
    if trainer.act_noise is not None:
        print(f"acting noise: {trainer.act_noise.describe()} (rollouts and snapshot evals "
              "act on noisy expectations; TD targets and gradients stay exact)")
    t0 = time.perf_counter()
    returns: list[float] = []

    def callback(episode: int, stats: dict) -> None:
        returns.append(stats["returns"])
        if (episode + 1) % REPORT_EVERY == 0:
            mean = float(np.mean(returns[-REPORT_EVERY:]))
            wall = time.perf_counter() - t0
            print(
                f"episode {episode + 1:>4}  mean return (last {REPORT_EVERY}): "
                f"{mean:>8.1f}  eps={stats['epsilon']:.2f}  wall={wall:6.1f}s"
            )

    history = trainer.train(episodes=episodes, callback=callback)

    print(f"trained {len(history['episode_returns'])} episodes "
          f"in {history['wall_time_s']:.1f}s wall clock")
    if monitor.first_clean_episode is not None:
        print(f"first clean lap: episode {monitor.first_clean_episode} "
              f"at {monitor.first_clean_wall_s:.1f}s wall clock")
    else:
        print("first clean lap: NEVER (no clean lap this run)")
    if "best_eval" in history:
        be = history["best_eval"]
        mean = f"{be['mean_lap']:.1f}s" if be["mean_lap"] is not None else "none"
        lap = f"{be['best_lap']:.1f}s" if be["best_lap"] is not None else "none"
        print(f"saved snapshot: episode {be['episode']} (greedy eval: "
              f"{be['lapped_episodes']}/{be['eval_episodes']} episodes lapped, "
              f"mean lap {mean}, best {lap})")
        fe = history["final_eval"]
        print(f"final params: {fe['lapped_episodes']}/{fe['eval_episodes']} episodes "
              f"lapped; {sum(e['lapped_episodes'] > 0 for e in history['eval_log'])}"
              f"/{len(history['eval_log'])} evals lapped at all")

    npz_path = save_weights(qfunc, agent, save_name, config, episodes,
                            out_dir=Path(out_dir) if out_dir else None,
                            training_cfg=training_cfg)
    print(f"weights saved to {npz_path}")
    final_path = None
    if save_final:
        final_path = npz_path.with_suffix("").with_suffix(".final.npz")
        np.savez(final_path, params=trainer.final_params)
        print(f"final params saved to {final_path}")
    if not np.isnan(monitor.best_lap_s):
        print(f"best lap: {monitor.best_lap_s:.2f}s")

    summary = {
        "episode_returns": history["episode_returns"],
        "wall_time_s": history["wall_time_s"],
        "first_clean_episode": monitor.first_clean_episode,
        "first_clean_wall_s": monitor.first_clean_wall_s,
        "best_lap_s": None if np.isnan(monitor.best_lap_s) else float(monitor.best_lap_s),
        "eval_log": history.get("eval_log", []),
        "best_eval": history.get("best_eval"),
        "final_eval": history.get("final_eval"),
        "weights_path": str(npz_path),
        "final_weights_path": str(final_path) if final_path is not None else None,
    }
    if history_path:
        payload = dict(summary, agent=agent, track=save_name, seed=seed, episodes=episodes,
                       training=training_cfg)
        Path(history_path).write_text(json.dumps(payload, default=_json_default) + "\n",
                                      encoding="utf-8")
    return summary


def cli_out_dir(out: str | None, agent: str, track: str) -> str | None:
    """The CLI's --out: a run of its own unless the bundled weights are asked for
    by name (None: ``train`` writes WEIGHTS_DIR)."""
    if out == BUNDLED:
        return None
    if out is not None:
        return out
    return str(Path("runs") / f"{agent}_{track}_{time.strftime('%Y%m%d-%H%M%S')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="racetraQ headless training")
    parser.add_argument("--agent", default="mlp", choices=["mlp", "quantum"],
                        help="Q-function backend")
    parser.add_argument("--track", default="oval",
                        help="track name (oval | chicane | gp | combo), 'multi' (mixture of "
                             "all four) or 'random' (pool of generated tracks from --seed)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="sub-env episodes (default: [training].episodes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: [training].seed)")
    parser.add_argument("--profile", default=None, help="config profile overlay (e.g. pi5)")
    parser.add_argument("--out", default=None,
                        help="weights output dir (default: a new runs/<agent>_<track>_<time>/); "
                             f"'{BUNDLED}' writes racetraq/weights/ and replaces the shipped driver")
    parser.add_argument("--init", default=None,
                        help="warm-start from a weights .npz (a quantum run continues at "
                             "its circuit depth and action count)")
    parser.add_argument("--history", default=None,
                        help="write returns/lap summary JSON to this path")
    parser.add_argument("--actions", type=int, default=None, choices=[4, 6, 8],
                        help="action-set size ([circuit] n_actions override): "
                             "6 adds trail braking, 8 adds half-steer")
    parser.add_argument("--pace", action="store_true",
                        help="pace fine-tune: merge [training_pace] (low epsilon "
                             "+ per-decision time penalty); combine with --init")
    parser.add_argument("--preset", default="auto", choices=list(PRESET_MODES),
                        help="auto (default): merge [training_presets.<track>] onto "
                             "[training], as the server does; none: plain [training]")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="config override, repeatable; the value is a TOML literal "
                             "(bare words count as strings), e.g. --set training.loss=huber "
                             "--set circuit.n_layers=6 --set reward.lap_bonus=0. "
                             "Wins over the preset; --episodes/--seed win over it; "
                             "an unknown training.<key> is an error")
    parser.add_argument("--save-final", action="store_true",
                        help="also write <name>.final.npz: the params at the end of "
                             "training, next to the best-snapshot weights")
    args = parser.parse_args()
    train(args.agent, args.track, args.episodes, args.seed, args.profile,
          out_dir=cli_out_dir(args.out, args.agent, args.track), init=args.init, history_path=args.history,
          actions=args.actions, pace=args.pace, preset=args.preset,
          overrides=args.overrides, save_final=args.save_final)


if __name__ == "__main__":
    main()
