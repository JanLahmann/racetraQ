"""Multi-seed training studies: run a (variant x seed) grid, then report it.

One training run is an anecdote; DQN on this task is noisy enough that a claim
needs many seeds and interval estimates.  This tool trains every (variant,
seed) cell of a study in its own subprocess and aggregates the results the way
the current benchmarking literature asks for:

- Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical
  Precipice" (NeurIPS 2021): interquartile mean (IQM), stratified-bootstrap
  confidence intervals, probability of improvement.
- Meyer et al., "Benchmarking Quantum Reinforcement Learning" (ICML 2025,
  arXiv:2501.15893): many seeds, sample-complexity estimates, statistical
  outperformance instead of best-run comparisons.

Run a study (resumable — finished cells are skipped)::

    python tools/study.py run --out DIR --agent quantum --track gp --seeds 0-9 \\
        --variant base \\
        --variant 'huber:training.loss="huber",training.huber_delta=10' \\
        --variant L6:circuit.n_layers=6 --jobs 6

A variant is ``name`` or ``name:item,item,...``; an item is a dotted config
override forwarded to the trainer exactly like ``train_headless --set``
(``training.loss="huber"``), or one of the pseudo-keys ``agent=``, ``track=``,
``profile=``, ``episodes=``, ``preset=``, ``actions=``, ``pace=true``,
``init=<weights.npz>`` (``{seed}`` in the path is replaced by the cell's
seed), so one study can compare agents, profiles or fine-tune recipes.
``--set section.key=value`` applies an override to every variant.

Report it::

    python tools/study.py report DIR [--baseline base] [--json]

Study directory layout::

    DIR/manifest.json                    commands, git commit, resolved variants
    DIR/cells/<variant>/seed<k>/
        cell.json                        the cell's recipe (worker input)
        result.json                      metrics (its existence marks the cell done)
        cell.error                       written instead when the cell failed
        train.log                        the worker's stdout/stderr
        history.json                     train_headless history (all returns)
        <agent>_<track>[_q<n>].npz       best-snapshot weights + .meta.json sidecar
        <agent>_<track>[_q<n>].final.npz params at the end of training
    DIR/DONE                             written when every cell is finished
    DIR/report.md, DIR/report.json       written by ``report``

Only numpy and the standard library are imported at module level; ``racetraq``
is loaded lazily — its config loader by ``run`` (to check override keys), the
trainer inside the cell worker.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = 1

MANIFEST_NAME = "manifest.json"
DONE_NAME = "DONE"
SPEC_NAME = "cell.json"
RESULT_NAME = "result.json"
ERROR_NAME = "cell.error"
LOG_NAME = "train.log"
HISTORY_NAME = "history.json"

NICENESS = 5  # cells run at +5 so a study never starves the desktop
# One BLAS/OpenMP thread per cell: the pool is the parallelism, and the small
# matrices here gain nothing from threads (oversubscription only slows cells).
THREAD_ENV = ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
ERROR_LOG_TAIL = 40  # lines of train.log copied into cell.error

EVAL_EPISODES = 36  # distinct greedy episodes per reliability eval (records.py's sample)
EVAL_SEED = 20_000  # records.evaluate's default env seed
N_BOOT = 2000  # bootstrap resamples per confidence interval
BOOTSTRAP_SEED = 0  # fixed: a report is a pure function of the result files
CI_LEVEL = 0.95
RELIABLE_FRACTION = 0.5  # a seed is "reliable" when its best snapshot laps this often
THRESHOLDS = (0.5, 0.9)  # sample-complexity targets (lapped fraction of one eval)

# Variant items without a dot: train() arguments rather than config overrides.
PSEUDO_KEYS = ("agent", "track", "profile", "episodes", "preset", "actions", "pace", "init")
_VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")


# --------------------------------------------------------------- statistics


def iqm(values: Iterable[float]) -> float:
    """Interquartile mean: the mean of the middle 50% of ``values``.

    Exactly a quarter of the sample is dropped at each end: when ``n / 4`` is
    not whole, the value straddling a quartile counts with the fraction of it
    that lies inside (10 values: 2 dropped, the 3rd and 8th at weight 0.5).
    Equal to rliable's ``scipy.stats.trim_mean(x, 0.25)`` when ``n`` is a
    multiple of 4 — that rule rounds the cut down, which for 10 seeds would
    average the middle 60% and for 3 seeds trim nothing.  One or two values
    give their mean; NaN for an empty input.
    """
    x = np.sort(np.asarray(list(values), dtype=np.float64).ravel())
    n = x.size
    if n == 0:
        return float("nan")
    whole = n // 4  # values dropped outright at each end
    part = 0.25 * n - whole  # ...plus this fraction of the next one in
    weights = np.ones(n)
    weights[:whole] = 0.0
    weights[n - whole:] = 0.0
    weights[whole] -= part
    weights[n - whole - 1] -= part
    return float(weights @ x / (0.5 * n))


def median(values: Iterable[float]) -> float:
    """Median of ``values``; NaN for an empty input."""
    x = np.asarray(list(values), dtype=np.float64).ravel()
    return float(np.median(x)) if x.size else float("nan")


def prob_improvement(x: Iterable[float], y: Iterable[float]) -> float:
    """Probability of improvement P(X > Y) over all (x, y) pairs; ties count half.

    The Mann-Whitney U statistic divided by ``len(x) * len(y)``: 1.0 when
    every x beats every y, 0.5 for identical samples, 0.0 when every x loses.
    NaN when either sample is empty.
    """
    a = np.asarray(list(x), dtype=np.float64).ravel()
    b = np.asarray(list(y), dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def bootstrap_ci(
    *strata: Iterable[float],
    statistic: Callable[..., float] = iqm,
    n_boot: int = N_BOOT,
    ci: float = CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Stratified percentile bootstrap: ``(point estimate, ci low, ci high)``.

    Each positional array is one stratum — an independent group of per-seed
    values — and is resampled with replacement ON ITS OWN, keeping its size;
    ``statistic`` receives one array per stratum.  One stratum is the plain
    bootstrap over seeds (``bootstrap_ci(x, statistic=iqm)``); two strata
    compare a variant with the baseline
    (``bootstrap_ci(x, y, statistic=prob_improvement)``).  Deterministic: the
    generator is rebuilt from ``seed`` on every call.  All NaN when a stratum
    is empty; a NaN interval around the point estimate when a stratum holds a
    single value — one seed says nothing about the seed-to-seed spread, and
    resampling it would report a zero-width interval, i.e. certainty.
    """
    groups = [np.asarray(list(s), dtype=np.float64).ravel() for s in strata]
    if not groups:
        raise ValueError("bootstrap_ci needs at least one stratum")
    if any(g.size == 0 for g in groups):
        return float("nan"), float("nan"), float("nan")
    if any(g.size == 1 for g in groups):
        return float(statistic(*groups)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = [g[rng.integers(0, g.size, size=(n_boot, g.size))] for g in groups]
    boot = np.array([statistic(*(d[b] for d in draws)) for b in range(n_boot)],
                    dtype=np.float64)
    tail = 100.0 * (1.0 - ci) / 2.0
    low, high = np.percentile(boot, [tail, 100.0 - tail])
    return float(statistic(*groups)), float(low), float(high)


def _lapped_fraction(entry: dict) -> float:
    """Lapped fraction of one eval_log entry (0.0 for a malformed/empty one)."""
    n = entry.get("eval_episodes") or 0
    return float(entry.get("lapped_episodes", 0)) / n if n else 0.0


def sample_complexity(eval_log: Sequence[dict], threshold: float) -> int | None:
    """Training episodes until the running-best eval laps in >= ``threshold``
    of its episodes: the ``episode`` of the first ``eval_log`` entry that gets
    there, or None when the run never does."""
    for entry in eval_log:
        if _lapped_fraction(entry) >= threshold:
            return int(entry["episode"])
    return None


def episodes_for_fraction(episodes: Iterable[int | None], fraction: float = 0.5) -> int | None:
    """Training episodes by which at least ``fraction`` of ALL seeds got there.

    ``episodes`` holds one :func:`sample_complexity` value per seed (None =
    never).  Unlike a median over the seeds that made it, this does not
    reward a variant for seeds that never learn: None when fewer than
    ``fraction`` of the seeds ever get there.
    """
    values = list(episodes)
    reached = sorted(int(v) for v in values if v is not None)
    needed = max(1, int(np.ceil(fraction * len(values) - 1e-9)))
    return reached[needed - 1] if values and len(reached) >= needed else None


def stability(eval_log: Sequence[dict]) -> float | None:
    """Mean lapped fraction over every eval AFTER the first eval that lapped.

    1.0 = once the policy laps it keeps lapping; near 0 = the lapping policy
    was a transient the optimiser walked away from.  A run that never laps
    scores 0.0 (there is nothing to keep); None when the first lapping eval
    is also the last one, i.e. no later eval exists to judge it by.
    """
    first = next((i for i, e in enumerate(eval_log) if e.get("lapped_episodes", 0) > 0), None)
    if first is None:
        return 0.0
    later = eval_log[first + 1:]
    if not later:
        return None
    return float(np.mean([_lapped_fraction(e) for e in later]))


# ------------------------------------------------------- variants and seeds


def parse_seeds(spec: str) -> list[int]:
    """``"0-9"``, ``"42,0,1"`` or ``"0-2,42"`` -> seed list (order kept, no repeats)."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        span = re.fullmatch(r"(\d+)-(\d+)", part)
        if span:
            low, high = int(span.group(1)), int(span.group(2))
            if high < low:
                raise ValueError(f"bad seed range '{part}': end before start")
            seeds.extend(range(low, high + 1))
        elif re.fullmatch(r"\d+", part):
            seeds.append(int(part))
        else:
            raise ValueError(f"bad --seeds '{spec}': expected e.g. 0-9 or 0,1,42")
    return list(dict.fromkeys(seeds))


def _split_items(text: str) -> list[str]:
    """Split on top-level commas: commas inside quotes, [] or {} belong to a
    value (``training.lr_groups={ head = 0.1, lam = 0.001 }``)."""
    items, depth, quote, start, escaped = [], 0, "", 0, False
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:])
    return [item.strip() for item in items if item.strip()]


def parse_variant(spec: str) -> tuple[str, dict[str, str], list[str]]:
    """``"name:item,item"`` -> (name, {pseudo-key: raw value}, [override strings]).

    Items with a dotted key are config overrides (kept verbatim, in order, for
    ``train(overrides=...)``); items whose key is one of :data:`PSEUDO_KEYS`
    are returned separately.  A bare ``"name"`` is the unmodified recipe.
    """
    name, _, body = spec.partition(":")
    name = name.strip()
    if not _VARIANT_NAME.fullmatch(name):
        raise ValueError(
            f"bad variant name '{name}' in '{spec}': use letters, digits and _ . + - "
            "(it names a directory)"
        )
    pseudo: dict[str, str] = {}
    overrides: list[str] = []
    for item in _split_items(body):
        key, sep, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or not key or not raw:
            raise ValueError(f"bad item '{item}' in variant '{name}': expected key=value")
        if "." in key:
            overrides.append(f"{key}={raw}")
        elif key in PSEUDO_KEYS:
            pseudo[key] = raw
        else:
            raise ValueError(
                f"bad item '{item}' in variant '{name}': '{key}' is neither a dotted "
                f"config key (section.key) nor one of {PSEUDO_KEYS}"
            )
    return name, pseudo, overrides


def _unquote(raw: str) -> str:
    return raw[1:-1] if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'" else raw


def _parse_bool(raw: str, what: str) -> bool:
    value = _unquote(raw).lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{what} must be true or false, got '{raw}'")


def resolve_variant(spec: str, defaults: dict[str, Any],
                    common_overrides: Sequence[str] = ()) -> dict[str, Any]:
    """The full recipe of one variant: the study-wide ``defaults`` (agent,
    track, profile, episodes, preset, actions, pace, init) with the variant's
    pseudo-keys on top, and ``common_overrides`` followed by its own overrides
    (later entries win in the trainer)."""
    name, pseudo, overrides = parse_variant(spec)
    recipe = {"name": name, **{key: defaults.get(key) for key in PSEUDO_KEYS}}
    for key, raw in pseudo.items():
        if key == "pace":
            recipe[key] = _parse_bool(raw, f"variant '{name}': pace")
        elif key in ("episodes", "actions"):
            try:
                recipe[key] = int(_unquote(raw))
            except ValueError:
                raise ValueError(
                    f"variant '{name}': {key} must be an integer, got '{raw}'"
                ) from None
        else:
            value = _unquote(raw)
            # "none" clears a study-wide --profile / --init for this variant
            recipe[key] = None if key in ("profile", "init") and value.lower() == "none" \
                else value
    recipe["pace"] = bool(recipe["pace"])
    recipe["overrides"] = [*common_overrides, *overrides]
    return recipe


def _find_racetraq() -> None:
    if importlib.util.find_spec("racetraq") is None:  # checkout without an install
        sys.path.insert(0, str(REPO_ROOT))


def new_config_keys(recipe: dict) -> list[str]:
    """Override keys of ``recipe`` that the config it overlays does not have.

    Outside ``[training]`` (whose keys the trainer checks itself) nothing
    rejects an unknown key: ``circuit.n_layer=6`` adds a key nobody reads and
    trains the unmodified recipe under the variant's name.  Optional keys
    (``circuit.n_actions``, ``mlp.hidden``) are absent from the config too,
    so the caller prints these as a note, not an error.  Raises ValueError
    for an override that does not parse.
    """
    try:
        _find_racetraq()
        from racetraq.config import load_config, parse_override

        config = load_config(profile=recipe["profile"])
    except (ImportError, OSError):  # e.g. an unknown profile: the cells report it
        return []
    new = []
    for item in recipe["overrides"]:
        key = parse_override(item)[0]
        table: Any = config
        for part in key.split("."):
            if not isinstance(table, dict) or part not in table:
                if not key.startswith("training."):
                    new.append(key)
                break
            table = table[part]
    return list(dict.fromkeys(new))


# ------------------------------------------------------------- cell worker


def _write_json(path: Path, payload: Any, indent: int | None = None) -> None:
    """Atomic JSON write: a killed process never leaves a half-written file
    (a result file's existence is what marks its cell done)."""

    def default(value: Any):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, default=default) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def cell_dir(out: Path, variant: str, seed: int) -> Path:
    return Path(out) / "cells" / variant / f"seed{seed}"


def weights_config(spec: dict, weights_path: Path) -> dict:
    """The config a cell's weights drive under: the cell's profile and
    overrides, then the observation / circuit shape / action count recorded in
    the weights' own ``.meta.json`` sidecar on top (the loaders' rule — see
    ``runtime.weights_observation`` / ``weights_actions``)."""
    from racetraq.config import apply_overrides, load_config

    config = load_config(profile=spec["profile"])
    apply_overrides(config, spec["overrides"])
    if spec["actions"] is not None:
        config.setdefault("circuit", {})["n_actions"] = int(spec["actions"])
    meta_path = weights_path.with_suffix("").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    config["observation"].update(meta.get("observation") or {})
    config.setdefault("circuit", {}).update(meta.get("circuit") or {})
    n_actions = (meta.get("actions") or {}).get("n_actions")
    if n_actions:
        config["circuit"]["n_actions"] = int(n_actions)
    return config


def _eval_env(spec: dict, config: dict):
    """Fresh env of ``eval_episodes`` parallel cars on the cell's track(s)."""
    from racetraq.env.multi_track import MultiTrackEnv
    from racetraq.env.racing_env import RacingEnv
    from racetraq.env.track import Track
    from racetraq.train_headless import MULTI_TRACK_NAMES

    n_envs, seed = int(spec["eval_episodes"]), int(spec["eval_seed"])
    spacing = config["track"]["resample_spacing"]
    if spec["track"] == "multi":
        tracks = [Track.load(name, spacing) for name in MULTI_TRACK_NAMES]
        return MultiTrackEnv(tracks, config, n_envs=n_envs, seed=seed)
    if spec["track"] == "random":
        # the generated pool is a function of the TRAINING seed
        pool = MultiTrackEnv.random_pool(config, n_envs=n_envs, seed=int(spec["seed"]))
        return MultiTrackEnv(pool.tracks, config, n_envs=n_envs, seed=seed)
    return RacingEnv(Track.load(spec["track"], spacing), config, n_envs=n_envs, seed=seed)


def greedy_eval(qfunc: Any, env: Any, max_steps: int) -> dict:
    """Reliability eval: one greedy episode per sub-env, all in parallel.

    Every sub-env starts from its own spawn jitter, so ``env.n_envs`` episodes
    are that many DISTINCT episodes; each is frozen at its first done (crash
    or time limit), so auto-reset never starts a second one — the
    ``records._eval_batched`` rule.
    """
    obs = env.reset()
    n_envs = obs.shape[0]
    done_mask = np.zeros(n_envs, dtype=bool)
    lapped = np.zeros(n_envs, dtype=bool)
    prev_lap = np.zeros(n_envs, dtype=int)
    lap_times: list[float] = []
    for _ in range(max_steps):
        obs, _, done, info = env.step(np.argmax(qfunc.q_values(obs), axis=1))
        cur_lap = np.asarray(info["lap"], dtype=int)
        lt = np.asarray(info["last_lap_time"], dtype=np.float64)
        event = (cur_lap > prev_lap) & ~done_mask
        lap_times.extend(lt[event & ~np.isnan(lt)].tolist())
        lapped |= event
        done = np.asarray(done, dtype=bool)
        done_mask |= done
        prev_lap = np.where(done, 0, cur_lap)
        if done_mask.all():
            break
    return {
        "episodes": int(n_envs),
        "lapped_episodes": int(lapped.sum()),
        "laps": len(lap_times),
        "mean_lap": float(np.mean(lap_times)) if lap_times else None,
        "best_lap": float(min(lap_times)) if lap_times else None,
    }


def run_cell(cell: Path) -> None:
    """Worker, one subprocess per cell: train, then greedy-eval the best
    snapshot and the final params, and write ``result.json``."""
    spec = json.loads((cell / SPEC_NAME).read_text(encoding="utf-8"))
    if hasattr(os, "nice"):
        os.nice(NICENESS)
    _find_racetraq()
    from racetraq.config import parse_override
    from racetraq.train_headless import build_qfunc, train

    t0 = time.perf_counter()
    summary = train(
        spec["agent"], spec["track"], spec["episodes"], spec["seed"], spec["profile"],
        out_dir=str(cell), init=spec["init"], history_path=str(cell / HISTORY_NAME),
        actions=spec["actions"], pace=spec["pace"], preset=spec["preset"],
        overrides=spec["overrides"], save_final=True,
    )
    history = json.loads((cell / HISTORY_NAME).read_text(encoding="utf-8"))
    weights = Path(summary["weights_path"])
    final_weights = Path(summary["final_weights_path"])

    config = weights_config(spec, weights)
    max_steps = int(config["reward"]["max_decisions"]) + 1
    evals = {}
    n_params = 0
    for key, path in (("best_snapshot_eval", weights), ("final_params_eval", final_weights)):
        env = _eval_env(spec, config)
        qfunc = build_qfunc(spec["agent"], env.n_features, spec["seed"], config,
                            n_actions=env.n_actions)
        qfunc.set_params(np.load(path)["params"])
        n_params = int(qfunc.get_params().size)
        evals[key] = greedy_eval(qfunc, env, max_steps)

    returns = np.asarray(summary["episode_returns"], dtype=np.float64)
    result = {
        "schema": SCHEMA,
        "variant": spec["variant"],
        "seed": spec["seed"],
        "agent": spec["agent"],
        "track": spec["track"],
        "profile": spec["profile"],
        "preset": spec["preset"],
        "actions": spec["actions"],
        "pace": spec["pace"],
        "init": spec["init"],
        "episodes": history["episodes"],
        "overrides": dict(parse_override(item) for item in spec["overrides"]),
        "training": history["training"],
        "n_params": n_params,
        "wall_time_s": summary["wall_time_s"],
        "cell_wall_s": time.perf_counter() - t0,
        "first_clean_episode": summary["first_clean_episode"],
        "eval_log": summary["eval_log"],
        "best_eval": summary["best_eval"],
        "final_eval": summary["final_eval"],
        **evals,
        "mean_return_last200": float(returns[-200:].mean()) if returns.size else None,
        "returns_by_100": [float(returns[i:i + 100].mean())
                           for i in range(0, returns.size, 100)],
        "weights": weights.name,
        "final_weights": final_weights.name,
    }
    _write_json(cell / RESULT_NAME, result)
    print(f"cell result written to {cell / RESULT_NAME}")


# --------------------------------------------------------------------- run


def _git_state() -> dict:
    """Commit and dirty flag of the checkout this tool lives in (read-only)."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, timeout=30, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def update_manifest(out: Path, variants: list[dict], seeds: list[int], evals: dict,
                    command: str, jobs: int) -> dict:
    """Create or extend ``DIR/manifest.json``; returns the manifest.

    A resumed or extended study appends its command to ``runs`` and merges new
    variants/seeds.  Re-using a variant NAME with a different recipe (or other
    reliability-eval settings) is refused: the cells already on disk would be
    reported under a recipe they were not trained with.
    """
    path = out / MANIFEST_NAME
    manifest = {"schema": SCHEMA, "variants": {}, "seeds": [], "eval": evals, "runs": []}
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("eval") != evals:
            raise ValueError(
                f"{out} was run with reliability-eval settings {manifest.get('eval')}, "
                f"not {evals}; use a new --out"
            )
    for variant in variants:
        clash = next((name for name in manifest["variants"]
                      if name != variant["name"] and name.lower() == variant["name"].lower()),
                     None)
        if clash is not None:
            raise ValueError(
                f"variant '{variant['name']}' differs from the existing variant '{clash}' "
                f"in {out} only by upper/lower case; pick a new variant name"
            )
        known = manifest["variants"].get(variant["name"])
        if known is not None and known != variant:
            raise ValueError(
                f"variant '{variant['name']}' already exists in {out} with a different "
                f"recipe ({known}); pick a new variant name or a new --out"
            )
        manifest["variants"][variant["name"]] = variant
    manifest["seeds"] = list(dict.fromkeys([*manifest["seeds"], *seeds]))
    manifest["runs"].append({
        "command": command,
        "cwd": os.getcwd(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_git_state(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "jobs": jobs,
        "variants": [variant["name"] for variant in variants],
        "seeds": seeds,
    })
    _write_json(path, manifest, indent=2)
    return manifest


def _cell_summary(result: dict) -> str:
    best, final = result["best_snapshot_eval"], result["final_params_eval"]
    lap = f"{best['mean_lap']:.1f}s" if best["mean_lap"] is not None else "none"
    clean = result["first_clean_episode"]
    return (
        f"best {best['lapped_episodes']}/{best['episodes']} lapped (mean lap {lap}), "
        f"final {final['lapped_episodes']}/{final['episodes']}, "
        f"first clean lap {'never' if clean is None else f'ep {clean}'}, "
        f"{result['cell_wall_s']:.0f}s"
    )


class _CellPool:
    """Runs cell subprocesses; remembers the live ones so Ctrl-C can stop them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: set[subprocess.Popen] = set()
        self._stopping = False

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            for proc in self._procs:
                proc.terminate()

    def run(self, cell: Path) -> str:
        """Run one cell to completion; returns its progress-line text.  Never
        raises: any failure becomes a ``cell.error`` file."""
        error_path = cell / ERROR_NAME
        try:
            error_path.unlink(missing_ok=True)
            env = dict(os.environ, PYTHONUNBUFFERED="1", **dict.fromkeys(THREAD_ENV, "1"))
            with (cell / LOG_NAME).open("w", encoding="utf-8") as log:
                with self._lock:
                    if self._stopping:
                        return "not started (interrupted)"
                    proc = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "_cell", str(cell)],
                        stdout=log, stderr=subprocess.STDOUT, env=env,
                    )
                    self._procs.add(proc)
                try:
                    code = proc.wait()
                finally:
                    with self._lock:
                        self._procs.discard(proc)
            if self._stopping:
                return "interrupted"
            if code == 0 and (cell / RESULT_NAME).is_file():
                return _cell_summary(json.loads((cell / RESULT_NAME).read_text("utf-8")))
            tail = (cell / LOG_NAME).read_text(encoding="utf-8", errors="replace")
            tail = "\n".join(tail.splitlines()[-ERROR_LOG_TAIL:])
            reason = f"exit code {code}" if code else "exit code 0 but no result file"
            error_path.write_text(f"{reason}\nlog: {cell / LOG_NAME}\n\n{tail}\n",
                                  encoding="utf-8")
            return f"FAILED ({reason}) -> {error_path}"
        except Exception as exc:  # one broken cell must not stop the study
            with contextlib.suppress(OSError):
                error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return f"FAILED ({type(exc).__name__}: {exc}) -> {error_path}"


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def cmd_run(args: argparse.Namespace, command: str) -> int:
    """Train every missing cell; exit code 0 (all cells have results), 1 (some
    failed — see their ``cell.error``) or 130 (interrupted)."""
    out = Path(args.out)
    defaults = {"agent": args.agent, "track": args.track, "profile": args.profile,
                "episodes": args.episodes, "preset": args.preset, "actions": None,
                "pace": False, "init": None}
    variants = [resolve_variant(spec, defaults, args.overrides)
                for spec in (args.variant or ["base"])]
    # A name is a directory, and macOS / Windows file systems fold case: "l6"
    # and "L6" would train into the same cells.
    names = [variant["name"] for variant in variants]
    folded = [name.lower() for name in names]
    repeated = sorted({name for name in names if folded.count(name.lower()) > 1})
    if repeated:
        raise ValueError(f"variant name(s) given twice (upper/lower case does not tell "
                         f"two variants apart): {repeated}")
    seeds = parse_seeds(args.seeds)
    if args.jobs < 1:
        raise ValueError(f"--jobs must be >= 1, got {args.jobs}")
    if args.eval_episodes < 1:
        raise ValueError(f"--eval-episodes must be >= 1, got {args.eval_episodes}")
    evals = {"episodes": int(args.eval_episodes), "seed": int(args.eval_seed)}
    for variant in variants:
        new = new_config_keys(variant)
        if new:
            print(f"note: variant '{variant['name']}': {', '.join(new)} not in the loaded "
                  "config — a new key, or a typo that leaves the recipe unchanged",
                  file=sys.stderr, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    update_manifest(out, variants, seeds, evals, command, args.jobs)
    (out / DONE_NAME).unlink(missing_ok=True)

    # Seeds outermost: an interrupted study has every variant at similar depth.
    cells = [(variant, seed) for seed in seeds for variant in variants]
    todo = []
    for variant, seed in cells:
        cell = cell_dir(out, variant["name"], seed)
        if (cell / RESULT_NAME).is_file():
            continue
        cell.mkdir(parents=True, exist_ok=True)
        init = variant["init"]
        spec = {
            **{key: variant[key] for key in PSEUDO_KEYS},
            "variant": variant["name"],
            "seed": seed,
            "init": init.replace("{seed}", str(seed)) if init else None,
            "overrides": variant["overrides"],
            "eval_episodes": evals["episodes"],
            "eval_seed": evals["seed"],
        }
        _write_json(cell / SPEC_NAME, spec, indent=2)
        todo.append((variant["name"], seed, cell))
    print(f"study {out}: {len(cells)} cells ({len(variants)} variants x {len(seeds)} seeds), "
          f"{len(cells) - len(todo)} already done, {len(todo)} to run, --jobs {args.jobs}",
          flush=True)

    t0 = time.perf_counter()
    pool = _CellPool()
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    # A plain `kill` stops the study like Ctrl-C does: without this the cell
    # workers would outlive it and race a resumed run for the same cells.
    on_main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGTERM, _raise_interrupt) if on_main_thread else None
    try:
        futures = {executor.submit(pool.run, cell): (name, seed) for name, seed, cell in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            name, seed = futures[future]
            print(f"[{n}/{len(todo)}] {name} seed {seed}: {future.result()} "
                  f"(elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
    except KeyboardInterrupt:
        pool.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        print("interrupted — finished cells are kept; re-run the same command to resume",
              flush=True)
        return 130
    finally:
        if on_main_thread:
            signal.signal(signal.SIGTERM, previous)
    executor.shutdown()

    done = sum((cell_dir(out, variant["name"], seed) / RESULT_NAME).is_file()
               for variant, seed in cells)
    failed = len(cells) - done
    summary = f"{len(cells)} cells: {done} ok, {failed} failed"
    (out / DONE_NAME).write_text(summary + "\n", encoding="utf-8")
    print(f"DONE {summary} ({time.perf_counter() - t0:.0f}s) -> {out}", flush=True)
    return 1 if failed else 0


# ------------------------------------------------------------------ report


def load_study(directory: Path) -> tuple[dict, dict[str, list[dict]], list[dict]]:
    """(manifest or {}, {variant: [result, ...] by seed}, failed cells as
    [{variant, seed, error_file}]).

    Variants come in manifest order, then any further cell directories by name.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
    cells_root = directory / "cells"
    on_disk = sorted(p.name for p in cells_root.iterdir() if p.is_dir()) \
        if cells_root.is_dir() else []
    order = list(dict.fromkeys([*manifest.get("variants", {}), *on_disk]))
    results: dict[str, list[dict]] = {}
    failed: list[dict] = []
    for name in order:
        runs = [json.loads(path.read_text(encoding="utf-8"))
                for path in (cells_root / name).glob(f"seed*/{RESULT_NAME}")]
        if runs:
            results[name] = sorted(runs, key=lambda r: r["seed"])
        for path in sorted((cells_root / name).glob(f"seed*/{ERROR_NAME}")):
            if not (path.parent / RESULT_NAME).is_file():
                failed.append({"variant": name, "seed": int(path.parent.name[4:]),
                               "error_file": str(path)})
    return manifest, results, failed


def seed_metrics(result: dict) -> dict:
    """The per-seed numbers the report aggregates."""
    best, final = result["best_snapshot_eval"], result["final_params_eval"]
    eval_log = result.get("eval_log") or []
    return {
        "seed": result["seed"],
        "best_lapped": best["lapped_episodes"],
        "final_lapped": final["lapped_episodes"],
        "eval_episodes": best["episodes"],
        "best_lapped_frac": best["lapped_episodes"] / best["episodes"],
        "final_lapped_frac": final["lapped_episodes"] / final["episodes"],
        "best_mean_lap": _num(best["mean_lap"]),  # None and NaN both mean "no lap"
        "first_clean_episode": result.get("first_clean_episode"),
        "stability": stability(eval_log),
        **{f"episodes_to_{round(100 * t)}": sample_complexity(eval_log, t)
           for t in THRESHOLDS},
        "wall_time_s": result.get("wall_time_s"),
    }


def _num(value: float | None) -> float | None:
    """JSON-safe number: NaN / inf become None."""
    return None if value is None or not np.isfinite(value) else float(value)


def _aggregate(values: Iterable[float | None], n_boot: int) -> dict:
    """IQM and median with bootstrap CIs over the seeds that HAVE the value."""
    present = [float(v) for v in values if v is not None]
    out: dict[str, Any] = {"n": len(present)}
    for name, statistic in (("iqm", iqm), ("median", median)):
        point, low, high = bootstrap_ci(present, statistic=statistic, n_boot=n_boot)
        out[name] = {"value": _num(point), "ci_low": _num(low), "ci_high": _num(high)}
    return out


# (per-seed key, column label, number format); the last two are conditional:
# aggregated over the seeds that lapped / drove a clean lap at all.
METRICS = (
    ("best_lapped_frac", "best-snapshot lapped", ".2f"),
    ("final_lapped_frac", "final-params lapped", ".2f"),
    ("stability", "stability", ".2f"),
    ("best_mean_lap", "best-snapshot mean lap (s)", ".1f"),
    ("first_clean_episode", "first clean lap (episode)", ".0f"),
)
COMPARED = (("final_lapped_frac", "final-params lapped"), ("stability", "stability"))


def build_report(directory: Path, baseline: str | None = None,
                 n_boot: int = N_BOOT) -> dict:
    """Aggregate a study directory into the report dict (``report.json``)."""
    manifest, results, failed = load_study(directory)
    if not results:
        raise ValueError(f"no finished cells under {Path(directory) / 'cells'}")
    if baseline is None:
        baseline = "base" if "base" in results else next(iter(results))
    if baseline not in results:
        raise ValueError(f"baseline variant '{baseline}' has no finished cells "
                         f"(variants with results: {list(results)})")
    per_seed = {name: [seed_metrics(r) for r in runs] for name, runs in results.items()}
    planned = manifest.get("seeds", [])

    variants: dict[str, dict] = {}
    for name, rows in per_seed.items():
        seeds = [row["seed"] for row in rows]
        failed_seeds = [cell["seed"] for cell in failed if cell["variant"] == name]
        entry: dict[str, Any] = {
            "recipe": manifest.get("variants", {}).get(name),
            "n_params": results[name][0].get("n_params"),
            "n_seeds": len(rows),
            "seeds": seeds,
            "failed_seeds": failed_seeds,
            "missing_seeds": [s for s in planned if s not in seeds and s not in failed_seeds],
            "reliable_seeds": sum(r["best_lapped_frac"] >= RELIABLE_FRACTION for r in rows),
            "metrics": {key: _aggregate([r[key] for r in rows], n_boot)
                        for key, _, _ in METRICS},
            "sample_complexity": {},
            "per_seed": rows,
        }
        for threshold in THRESHOLDS:
            key = f"episodes_to_{round(100 * threshold)}"
            stats = _aggregate([r[key] for r in rows], n_boot)
            entry["sample_complexity"][key] = {
                "threshold": threshold,
                "reached": stats["n"],
                "fraction_reached": stats["n"] / len(rows),
                "episodes_half_of_seeds": episodes_for_fraction([r[key] for r in rows], 0.5),
                "iqm": stats["iqm"],
                "median": stats["median"],
            }
        if name != baseline:
            entry["vs_baseline"] = {}
            for key, _ in COMPARED:
                mine = [r[key] for r in rows if r[key] is not None]
                base = [r[key] for r in per_seed[baseline] if r[key] is not None]
                point, low, high = bootstrap_ci(mine, base, statistic=prob_improvement,
                                                n_boot=n_boot)
                entry["vs_baseline"][key] = {
                    "p_improvement": _num(point), "ci_low": _num(low), "ci_high": _num(high),
                    "n": len(mine), "n_baseline": len(base),
                }
        variants[name] = entry

    return {
        "schema": SCHEMA,
        "directory": str(directory),
        "baseline": baseline,
        "bootstrap": {"resamples": n_boot, "seed": BOOTSTRAP_SEED, "ci": CI_LEVEL,
                      "method": "stratified percentile bootstrap over seeds"},
        "reliable_fraction": RELIABLE_FRACTION,
        "thresholds": list(THRESHOLDS),
        "eval": manifest.get("eval"),
        "git_commits": sorted({run["git_commit"] for run in manifest.get("runs", [])
                               if run.get("git_commit")}),
        "failed_cells": failed,
        "variants": variants,
    }


def _ci_text(stat: dict | None, fmt: str, key: str = "value") -> str:
    if stat is None or stat.get(key) is None:
        return "—"
    if stat["ci_low"] is None or stat["ci_high"] is None:  # a single seed: no interval
        return f"{stat[key]:{fmt}} [—]"
    return f"{stat[key]:{fmt}} [{stat['ci_low']:{fmt}}, {stat['ci_high']:{fmt}}]"


def _cell_text(value: float | None, fmt: str) -> str:
    return "—" if value is None else f"{value:{fmt}}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_markdown(report: dict) -> str:
    """The report dict as markdown tables."""
    variants = report["variants"]
    boot = report["bootstrap"]
    pct = round(100 * boot["ci"])
    n_cells = sum(v["n_seeds"] for v in variants.values())
    missing = sum(len(v["missing_seeds"]) for v in variants.values())
    lines = [
        f"# Study report: {report['directory']}",
        "",
        f"- {n_cells} finished cells in {len(variants)} variants; "
        f"{len(report['failed_cells'])} failed, {missing} not run yet.",
        f"- Intervals: {pct}% {boot['method']}, {boot['resamples']} resamples, "
        f"RNG seed {boot['seed']}. IQM = mean of the middle 50% of seeds. With fewer "
        "than about 10 seeds these intervals understate the uncertainty; a single seed "
        "has none ([—]).",
        "- Lapped fractions come from the reliability evals (distinct greedy episodes of "
        "the best snapshot and of the final params); stability and sample complexity "
        "from the in-training eval log.",
    ]
    if report["git_commits"]:
        lines.append(f"- git commit(s): {', '.join(report['git_commits'])}")
    lines.append("")

    lines += ["## Variants", ""]
    rows = []
    for name, v in variants.items():
        recipe = v["recipe"] or {}
        rows.append([
            name, str(recipe.get("agent", "?")), str(recipe.get("track", "?")),
            str(recipe.get("profile") or "—"), _cell_text(v["n_params"], "d"),
            ", ".join(f"`{item}`" for item in recipe.get("overrides", [])) or "—",
            f"{v['n_seeds']}",
            f"{v['reliable_seeds']}/{v['n_seeds']}",
        ])
    lines += _table(
        ["variant", "agent", "track", "profile", "params", "overrides", "seeds",
         f"seeds whose best snapshot laps >= {round(100 * report['reliable_fraction'])}%"],
        rows)

    for stat, title in (("iqm", "IQM"), ("median", "Median")):
        lines += [f"## {title} over seeds [{pct}% CI]", ""]
        rows = []
        for name, v in variants.items():
            row = [name]
            for key, _, fmt in METRICS:
                metric = v["metrics"][key]
                text = _ci_text(metric[stat], fmt)
                if metric["n"] != v["n_seeds"]:  # conditional metric: say over how many
                    text += f" (n={metric['n']})"
                row.append(text)
            rows.append(row)
        lines += _table(["variant", *(label for _, label, _ in METRICS)], rows)
    lines += [
        "Stability = mean lapped fraction over all in-training evals after the first eval "
        "that lapped (a seed that never laps scores 0; one whose first lapping eval is its "
        "last has no value). Mean lap and first clean lap are over the n seeds that lapped "
        "/ drove a clean lap at all.",
        "",
    ]

    lines += ["## Sample complexity", ""]
    for threshold in report["thresholds"]:
        t = round(100 * threshold)
        key = f"episodes_to_{t}"
        lines += [f"Training episodes until the running-best in-training eval laps in "
                  f">= {t}% of its episodes:", ""]
        rows = []
        for name, v in variants.items():
            sc = v["sample_complexity"][key]
            rows.append([
                name,
                f"{sc['reached']}/{v['n_seeds']} ({sc['fraction_reached']:.0%})",
                _cell_text(sc["episodes_half_of_seeds"], "d"),
                _ci_text(sc["median"], ".0f"),
                _ci_text(sc["iqm"], ".0f"),
            ])
        lines += _table(
            ["variant", "seeds that get there", "episodes until half of ALL seeds are there",
             f"median [{pct}% CI], seeds that get there",
             f"IQM [{pct}% CI], seeds that get there"],
            rows)
    lines += [
        "\"Half of ALL seeds\" counts the seeds that never get there (— = fewer than half "
        "do), so it is the column to compare variants by; median and IQM are conditional "
        "on getting there.",
        "",
    ]

    baseline = report["baseline"]
    lines += [f"## Versus baseline `{baseline}`", ""]
    rows = [[name, *(_ci_text(v["vs_baseline"][key], ".2f", "p_improvement")
                     for key, _ in COMPARED)]
            for name, v in variants.items() if "vs_baseline" in v]
    if rows:
        lines += _table(["variant", *(f"P({label} > {baseline}) [{pct}% CI]"
                                      for _, label in COMPARED)], rows)
        lines += [
            "Probability that a random seed of the variant beats a random seed of the "
            "baseline (ties count half): 0.5 = no difference; an interval entirely above "
            "0.5 is a statistically supported improvement.",
            "",
        ]
    else:
        lines += ["(no other variant to compare)", ""]

    lines += ["## Per-seed results", ""]
    rows = []
    for name, v in variants.items():
        for r in v["per_seed"]:
            rows.append([
                name, str(r["seed"]),
                f"{r['best_lapped']}/{r['eval_episodes']}",
                f"{r['final_lapped']}/{r['eval_episodes']}",
                _cell_text(r["stability"], ".2f"),
                _cell_text(r["best_mean_lap"], ".1f"),
                _cell_text(r["first_clean_episode"], "d"),
                *(_cell_text(r[f"episodes_to_{round(100 * t)}"], "d")
                  for t in report["thresholds"]),
                _cell_text(r["wall_time_s"], ".0f"),
            ])
    lines += _table(
        ["variant", "seed", "best snapshot lapped", "final params lapped", "stability",
         "mean lap (s)", "first clean lap",
         *(f"episodes to >= {round(100 * t)}%" for t in report["thresholds"]),
         "train wall (s)"],
        rows)

    if report["failed_cells"]:
        lines += ["## Failed cells", "",
                  *(f"- {cell['variant']} seed {cell['seed']}: `{cell['error_file']}`"
                    for cell in report["failed_cells"]), ""]
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    report = build_report(directory, baseline=args.baseline, n_boot=args.resamples)
    markdown = render_markdown(report)
    _write_json(directory / "report.json", report, indent=2)
    (directory / "report.md").write_text(markdown + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json else markdown)
    return 0


# --------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/study.py",
        description="Multi-seed training studies: run a (variant x seed) grid, report "
                    "IQM / bootstrap CIs / sample complexity / probability of improvement.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run,report}")

    run = sub.add_parser("run", help="train every (variant, seed) cell (resumable)")
    run.add_argument("--out", required=True, help="study directory")
    run.add_argument("--agent", default="quantum", choices=["mlp", "quantum"],
                     help="Q-function backend of every variant (default quantum)")
    run.add_argument("--track", default="gp",
                     help="track name, 'multi' or 'random' (default gp)")
    run.add_argument("--profile", default=None, help="config profile overlay (e.g. q10)")
    run.add_argument("--episodes", type=int, default=None,
                     help="training episodes (default: the resolved [training] value)")
    run.add_argument("--preset", default="auto", choices=["auto", "none"],
                     help="train_headless --preset (default auto)")
    run.add_argument("--seeds", required=True, help="e.g. 0-9 or 0,1,42")
    run.add_argument("--variant", action="append", metavar="NAME[:ITEM,ITEM...]",
                     help="repeatable; an item is a config override "
                          "(training.loss=\"huber\") or a pseudo-key "
                          f"({' | '.join(PSEUDO_KEYS)}); default: one variant 'base'")
    run.add_argument("--set", dest="overrides", action="append", default=[],
                     metavar="SECTION.KEY=VALUE",
                     help="config override applied to EVERY variant (a variant's own "
                          "items win), repeatable")
    run.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                     help="cells trained in parallel (default: half the cores)")
    run.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                     help=f"episodes per reliability eval (default {EVAL_EPISODES})")
    run.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                     help=f"env seed of the reliability evals (default {EVAL_SEED})")

    report = sub.add_parser("report", help="aggregate a study directory")
    report.add_argument("dir", help="study directory")
    report.add_argument("--baseline", default=None,
                        help="variant the others are compared with (default: 'base', "
                             "else the first variant)")
    report.add_argument("--json", action="store_true",
                        help="print report.json instead of the markdown report")
    report.add_argument("--resamples", type=int, default=N_BOOT,
                        help=f"bootstrap resamples per interval (default {N_BOOT})")

    cell = sub.add_parser("_cell")  # internal: the per-cell worker
    cell.add_argument("cell_dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "_cell":
        run_cell(Path(args.cell_dir))
        return 0
    try:
        if args.command == "run":
            command = shlex.join([sys.executable, str(Path(__file__).resolve()), *argv])
            return cmd_run(args, command)
        return cmd_report(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
