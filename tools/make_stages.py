"""Generate HONEST evolution-mode stage weights: earlier snapshots of the
training run that produced the bundled driver.

Run as ``python tools/make_stages.py --track oval [--profile q6]``.

Evolution mode shows four cars "at different stages of training" and ends on
the bundled driver ``quantum_<track>[_q<n>].npz``.  That is only true if the
earlier cars are snapshots of THAT driver's training run, so this tool
replays it:

- seed, episode count, the resolved ``training`` table, circuit depth and
  observation are read from the driver's ``.meta.json`` sidecar (``--seed``,
  ``--episodes`` and ``--set`` override them; ``--fresh`` ignores the driver
  and trains the config's recipe);
- the run goes through ``traqmania.train_headless.train`` — the code path
  that trained the driver (``tools/study.py`` calls it once per cell) — with
  a trainer that also keeps the parameters of every snapshot eval;
- the best snapshot of the replay must equal the bundled driver PARAMETER
  FOR PARAMETER, otherwise nothing is written (``--allow-mismatch`` writes
  the stages anyway, ends them on a copy of the bundled driver and says so
  in every sidecar).

Every snapshot has two evals.  The trainer's own, which picked the driver:
lexicographically ``(episodes lapped, -mean lap, mean return)`` over
``eval_episodes`` distinct greedy episodes (acting under the recipe's
``act_noise`` when it has one).  And an independent one made here:
``EXACT_EVAL_EPISODES`` distinct greedy episodes on the exact simulator
(``records.evaluate``), which is how evolution mode drives the cars.  Four
snapshots spanning early -> late are selected — the first of the chain that
laps, two intermediates, the best — such that each scores STRICTLY better
than the one before in the trainer's eval and NO WORSE in the exact one, and
saved to ``traqmania/weights/quantum_<track>_stage<i>[_q<n>].npz`` plus a
``.meta.json`` sidecar: ``episodes`` (the "ep N" car label in evolution
mode), both evals of that snapshot and the circuit / observation / training
blocks of a ``train_headless`` sidecar.

The ``_warmstart`` checkpoint of the warm live-training demo is the last
snapshot that laps in NEITHER eval before the run's breakthrough — the first
snapshot that laps in at least half of an eval's episodes.  Usually that is
a pre-first-lap snapshot; where an earlier snapshot lapped in a few episodes
and lost it again (DQN churn), the sidecar names those episodes.

Rationale: raw parameters at fixed episode counts are NOT monotonically
better — DQN policy churn made an ep-400 snapshot beat ep-800 on screen, and
a 12-episode eval alone can still rank two snapshots the wrong way round.
Selecting on both evals keeps the evolution-mode story (later stage = better
driver) true where the audience sees it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from traqmania import train_headless
from traqmania.agents.training import DQNTrainer
from traqmania.config import load_config, parse_override
from traqmania.server.runtime import WEIGHTS_DIR, with_weights_config

N_STAGES = 4
DEFAULT_SEED = 42  # --fresh without --seed
EXACT_EVAL_EPISODES = 36  # distinct greedy episodes of the independent eval
EXACT_EVAL_SEED = 20_000  # records.evaluate's default env seed
BREAKTHROUGH = 0.5  # lapped share of an eval from which a run "has got it"
# sidecar blocks copied from the replayed run's train_headless sidecar
RUN_BLOCKS = ("circuit", "observation", "actions", "training")


def _sidecar(npz: Path) -> Path:
    return npz.with_suffix("").with_suffix(".meta.json")


def _read_sidecar(npz: Path) -> dict:
    try:
        meta = json.loads(_sidecar(npz).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _write_sidecar(npz: Path, meta: dict) -> None:
    _sidecar(npz).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def driver_recipe(driver: Path, profile: str | None = None) -> dict | None:
    """What it takes to replay the run behind a bundled driver, from its
    sidecar: ``{"seed", "episodes", "overrides"}`` — the resolved training
    table as ``training.<key>`` overrides (so the replay does not depend on
    today's presets), the circuit depth and action count, and the recorded
    observation.  None when the sidecar is missing or records no
    ``training`` table with a seed and an episode count (a pre-October-2026
    driver: its run cannot be replayed from what it recorded)."""
    meta = _read_sidecar(driver)
    training = meta.get("training")
    if not isinstance(training, dict) or "seed" not in training or "episodes" not in training:
        return None
    overrides: dict[str, Any] = {f"training.{key}": value for key, value in training.items()}
    circuit = meta.get("circuit") if isinstance(meta.get("circuit"), dict) else {}
    if "n_layers" in circuit:
        overrides["circuit.n_layers"] = int(circuit["n_layers"])
    profile_circuit = load_config(profile=profile).get("circuit", {})
    default_actions = int(profile_circuit.get(
        "n_actions", min(4, int(profile_circuit.get("n_qubits", 4)))))
    if int(circuit.get("n_actions", default_actions)) != default_actions:
        overrides["circuit.n_actions"] = int(circuit["n_actions"])
    observation = meta.get("observation") if isinstance(meta.get("observation"), dict) else {}
    for key, value in observation.items():
        overrides[f"observation.{key}"] = value
    return {"seed": int(training["seed"]), "episodes": int(training["episodes"]),
            "overrides": overrides}


@contextlib.contextmanager
def recorded_evals(record: list[dict]) -> Iterator[None]:
    """While active, ``train_headless.train`` builds a trainer that appends
    every snapshot eval to ``record`` — the ``eval_log`` entry plus
    ``params``, the parameters it scored.  Nothing else about the run
    changes: no extra RNG draws, no extra evals."""

    class RecordingTrainer(DQNTrainer):
        def _eval_snapshot(self, best, episode):
            best = super()._eval_snapshot(best, episode)
            record.append({**self.last_eval, "params": self.qfunc.get_params().copy()})
            return best

    original = train_headless.DQNTrainer
    train_headless.DQNTrainer = RecordingTrainer
    try:
        yield
    finally:
        train_headless.DQNTrainer = original


def eval_score(entry: Mapping[str, Any]) -> tuple:
    """The trainer's lexicographic snapshot score of one eval-log entry:
    (episodes lapped, -mean lap, mean return) — ``DQNTrainer._eval_snapshot``."""
    mean_lap = entry.get("mean_lap")
    mean_return = entry.get("mean_return")
    return (int(entry.get("lapped_episodes", 0)),
            -float(mean_lap) if mean_lap is not None else float("-inf"),
            float(mean_return) if mean_return is not None else float("-inf"))


def best_index(scores: Sequence[tuple]) -> int:
    """Index of the best snapshot — the EARLIEST one with the top score, as
    the trainer keeps it (a later eval must beat the best strictly)."""
    return max(range(len(scores)), key=lambda i: (scores[i], -i))


def exact_score(result: Mapping[str, Any]) -> tuple:
    """Ordering of an :func:`exact_eval` result: episodes lapped, then mean lap."""
    mean_lap = result.get("mean_lap")
    return (int(result["lapped"]), -float(mean_lap) if mean_lap is not None else float("-inf"))


def select_stages(scores: Sequence[tuple],
                  exact: Sequence[tuple] | None = None) -> list[int]:
    """Pick ``N_STAGES`` snapshot indices with strictly improving eval score.

    ``scores[i]`` is the trainer's lexicographic score of the i-th snapshot
    (episode order; see :func:`eval_score`), ``exact[i]`` — optional — its
    score in the independent exact-simulator eval (:func:`exact_score`).  A
    snapshot may follow another when its trainer score is strictly better
    and its exact score is not worse.  The longest such chain ending at the
    best-scoring snapshot is computed (O(n^2) DP); the returned four indices
    are the chain start (preferring the first member that completed a lap),
    the chain end (the best snapshot) and two intermediates spread evenly
    between them.
    """
    n = len(scores)
    best_idx = best_index(scores)

    def improves(j: int, i: int) -> bool:
        return scores[j] < scores[i] and (exact is None or exact[j] <= exact[i])

    length = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if improves(j, i) and length[j] + 1 > length[i]:
                length[i] = length[j] + 1
                prev[i] = j
    chain: list[int] = []
    k = best_idx
    while k != -1:
        chain.append(k)
        k = prev[k]
    chain.reverse()

    if len(chain) < N_STAGES:
        raise RuntimeError(
            f"only {len(chain)} strictly improving snapshots found "
            f"(need {N_STAGES}); scores: {list(scores)}"
            + (f"; exact-simulator scores: {list(exact)}" if exact is not None else "")
        )
    # Prefer starting at the first chain member that completes a lap, as long
    # as at least N_STAGES chain members remain after it.
    start = next(
        (k for k, idx in enumerate(chain)
         if scores[idx][0] > 0 and len(chain) - k >= N_STAGES),
        0,
    )
    sub = chain[start:]
    if len(sub) < N_STAGES:
        sub = chain[-N_STAGES:]
    return [sub[round(t * (len(sub) - 1) / (N_STAGES - 1))] for t in range(N_STAGES)]


def warmstart_choice(lapped: Sequence[float],
                     breakthrough: float = BREAKTHROUGH) -> tuple[int, int] | None:
    """``(warm-start index, breakthrough index)`` of a run, or None.

    ``lapped[i]`` is the lapped share of snapshot i — the larger one of its
    evals.  The breakthrough is the first snapshot that laps in at least
    ``breakthrough`` of an eval's episodes; the warm-start snapshot is the
    last one before it that laps in no eval episode at all — the "about to
    get it" starting point.  None when the run never breaks through or no
    lap-less snapshot precedes the breakthrough.
    """
    first = next((i for i, share in enumerate(lapped) if share >= breakthrough), None)
    if first is None:
        return None
    warm = next((j for j in range(first - 1, -1, -1) if lapped[j] == 0.0), None)
    return None if warm is None else (warm, first)


def exact_eval(npz: Path, track_name: str, profile: str | None,
               episodes: int = EXACT_EVAL_EPISODES, seed: int = EXACT_EVAL_SEED) -> dict:
    """Independent eval of a saved snapshot: ``episodes`` distinct greedy
    episodes on the exact simulator (``records.evaluate``; no acting noise),
    driven as the demo loads the file — at the depth, action count and
    observation its sidecar records."""
    from traqmania import records

    config = with_weights_config(load_config(profile=profile), npz)
    driver = records.Driver(npz.name.split(".")[0], "quantum", track_name,
                            int(config["circuit"]["n_qubits"]), npz, config)
    result = records.evaluate(driver, track_name, episodes, seed)
    return {"episodes": episodes, "eval_seed": seed, "lapped": result["lapped_episodes"],
            "laps": result["laps"], "mean_lap": result["mean_s"], "best_lap": result["best_s"]}


def _eval_fields(entry: Mapping[str, Any], acting_noise: str | None) -> dict:
    """Sidecar fields for the trainer's eval of one snapshot."""
    def rounded(value: Any) -> float | None:
        return None if value is None else round(float(value), 3)

    return {
        "eval_episodes": int(entry["eval_episodes"]),
        "eval_lapped": int(entry["lapped_episodes"]),
        "eval_mean_lap": rounded(entry.get("mean_lap")),
        "eval_best_lap": rounded(entry.get("best_lap")),
        "eval_mean_return": rounded(entry.get("mean_return")),
        # None: the eval acted on exact Q-values
        "eval_acting_noise": acting_noise,
    }


def _exact_text(result: Mapping[str, Any]) -> str:
    lap = f"mean lap {result['mean_lap']:.2f} s" if result["mean_lap"] is not None else "no lap"
    return f"exact simulator {result['lapped']}/{result['episodes']} episodes lapped, {lap}"


def make_stages(track_name: str = "oval", seed: int | None = None,
                episodes: int | None = None, profile: str | None = None,
                init: str | None = None,
                overrides: Mapping[str, Any] | Sequence[str] | None = None,
                weights_dir: Path | str | None = None, fresh: bool = False,
                allow_mismatch: bool = False,
                exact_episodes: int = EXACT_EVAL_EPISODES) -> list[Path]:
    """Replay the bundled quantum driver's training run on ``track_name`` and
    save four eval-selected stages of it; returns the stage paths.

    Also saves the last lap-less snapshot before the run's breakthrough as
    the track's ``_warmstart`` checkpoint (:func:`warmstart_choice`; the warm
    live-training demo needs a starting point that does not lap yet), so one
    run per track regenerates the whole evolution + warm-start weight family.
    ``profile`` (e.g. ``q6``) trains at that circuit size and names files by
    the usual ``_q{n}`` rule.

    The recipe is the driver's own (:func:`driver_recipe`), with ``seed`` /
    ``episodes`` / ``overrides`` on top; ``fresh`` — or a ``weights_dir``
    without a replayable driver — trains the config's recipe from
    ``seed`` (default 42) and ends the stages on that run's best snapshot.
    With ``init`` the run continues from those weights at their circuit depth
    and action count (``train_headless --init``); a driver that was itself
    fine-tuned needs it to be replayed.  Raises ``RuntimeError`` when the
    replay's best snapshot is not the bundled driver (unless
    ``allow_mismatch``: the last stage is then a copy of the driver, flagged
    in the sidecars) or when fewer than four strictly improving snapshots
    exist.  ``exact_episodes = 0`` skips the independent exact-simulator
    eval: the snapshots are then selected on the trainer's evals alone.
    """
    weights_dir = Path(weights_dir) if weights_dir is not None else WEIGHTS_DIR
    n_qubits = int(load_config(profile=profile)["circuit"]["n_qubits"])
    qtag = "" if n_qubits == 4 else f"_q{n_qubits}"
    driver = weights_dir / f"quantum_{track_name}{qtag}.npz"

    recipe = None if fresh or not driver.is_file() else driver_recipe(driver, profile)
    run_overrides: dict[str, Any] = dict(recipe["overrides"]) if recipe else {}
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            overrides = dict(parse_override(spec) for spec in overrides)
        run_overrides.update(overrides)
    if seed is None:
        seed = recipe["seed"] if recipe else DEFAULT_SEED
    if episodes is None and recipe:
        episodes = recipe["episodes"]
    if recipe:
        print(f"replaying the run behind {driver.name}: seed {seed}, {episodes} episodes",
              flush=True)
    else:
        why = "--fresh" if fresh else f"no replayable {driver.name}"
        print(f"fresh run ({why}): seed {seed}", flush=True)

    record: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="make_stages_") as tmp, recorded_evals(record):
        summary = train_headless.train("quantum", track_name, episodes, seed, profile,
                                       out_dir=tmp, init=init, overrides=run_overrides)
        run_npz = Path(summary["weights_path"])
        run_meta = _read_sidecar(run_npz)
        best_params = np.load(run_npz)["params"]
        # the independent eval of every snapshot, each loaded from a file with
        # the run's sidecar — as the demo will load the stages
        exact_results: list[dict] | None = None
        if exact_episodes > 0:
            print(f"\nexact-simulator eval of {len(record)} snapshots "
                  f"({exact_episodes} greedy episodes each)", flush=True)
            exact_results = []
            snapshot = Path(tmp) / f"quantum_snapshot{qtag}.npz"
            _write_sidecar(snapshot, run_meta)
            for entry in record:
                np.savez(snapshot, params=entry["params"])
                exact_results.append(exact_eval(snapshot, track_name, profile, exact_episodes))
    if not record:
        raise RuntimeError("the run recorded no snapshot eval")
    scores = [eval_score(entry) for entry in record]
    exact = [exact_score(result) for result in exact_results] if exact_results else None
    best = best_index(scores)
    if not np.array_equal(record[best]["params"], best_params):
        raise RuntimeError("the recorded best snapshot is not the one the run saved "
                           "(trainer scoring changed? see eval_score)")
    total_episodes = int(run_meta["episodes"])
    acting_noise = None
    if run_meta.get("training", {}).get("act_noise"):
        from traqmania.agents.quantum.noise import ExpectationNoise

        noise = ExpectationNoise.from_config(run_meta["training"]["act_noise"])
        acting_noise = noise.describe() if noise is not None else None

    print(f"\n{len(record)} snapshot evals ({record[0]['eval_episodes']} greedy episodes "
          f"each{f', acting under {acting_noise}' if acting_noise else ''}"
          f"{f'; exact simulator: {exact_episodes} episodes' if exact_results else ''}):",
          flush=True)
    for i, entry in enumerate(record):
        lap = f"{entry['mean_lap']:6.2f}s" if entry["mean_lap"] is not None else "   --  "
        line = (f"ep {entry['episode']:>4}  lapped={entry['lapped_episodes']:>2}  "
                f"mean_lap={lap}  return={entry['mean_return']:8.1f}")
        if exact_results:
            line += f"  | {_exact_text(exact_results[i])}"
        print(line + ("  <- best" if i == best else ""), flush=True)

    # Is the replay the run that produced the bundled driver?
    same_run: bool | None = None
    run_block: dict[str, Any] = {"seed": int(seed), "episodes": total_episodes,
                                 "best_episode": int(record[best]["episode"])}
    if init is not None:
        run_block["init"] = Path(init).name
    if driver.is_file() and not fresh:
        bundled = np.load(driver)["params"]
        same_run = bundled.shape == best_params.shape and np.array_equal(bundled, best_params)
        run_block.update(driver=driver.name, driver_sha256=_sha256(driver),
                         reproduces_driver=same_run)
        if same_run:
            print(f"\nbest snapshot (ep {record[best]['episode']}) == {driver.name}, "
                  "parameter for parameter", flush=True)
        else:
            diff = (f"max |difference| {float(np.max(np.abs(bundled - best_params))):.3g}"
                    if bundled.shape == best_params.shape
                    else f"{bundled.size} vs {best_params.size} parameters")
            message = (f"the replay's best snapshot (ep {record[best]['episode']}) is NOT "
                       f"{driver.name} ({diff}): this is not the run that produced the "
                       "bundled driver (other seed / episodes / recipe / --init, or the "
                       "training code changed)")
            if not allow_mismatch:
                raise RuntimeError(message + " — nothing written; --allow-mismatch writes "
                                   "the stages anyway and ends them on the bundled driver")
            print(f"\nWARNING: {message}", flush=True)

    def base_meta(entry: Mapping[str, Any]) -> dict:
        return {
            "agent": "quantum",
            "track": track_name,
            "config_hash": run_meta.get("config_hash"),
            "episodes": int(entry["episode"]),  # the "ep N" car label
        }

    def run_blocks() -> dict:
        return {key: run_meta[key] for key in RUN_BLOCKS if key in run_meta}

    def run_text() -> str:
        text = f"seed {seed}, {total_episodes} episodes"
        if init is not None:
            text += f", continued from {Path(init).name}"
        if same_run:
            return (f"the training run that produced {driver.name} ({text}; replayed by "
                    "tools/make_stages.py, best snapshot identical to the driver)")
        if same_run is False:
            return (f"a tools/make_stages.py run ({text}) that did NOT reproduce "
                    f"{driver.name}")
        return f"a tools/make_stages.py run ({text})"

    def evals(index: int) -> dict:
        """Both evals of snapshot ``index`` as sidecar fields."""
        fields = _eval_fields(record[index], acting_noise)
        if exact_results:
            fields["exact_eval"] = exact_results[index]
        return fields

    # Select before writing anything: a run without four improving snapshots
    # must not leave half a family behind.
    picked = select_stages(scores, exact)
    # per snapshot, the larger lapped share of its two evals
    lapped = [max(entry["lapped_episodes"] / max(1, entry["eval_episodes"]),
                  exact_results[i]["lapped"] / exact_results[i]["episodes"]
                  if exact_results else 0.0)
              for i, entry in enumerate(record)]
    warm = warmstart_choice(lapped)
    print(f"\nselected stages: {[record[i]['episode'] for i in picked]}", flush=True)

    warm_path = weights_dir / f"quantum_{track_name}_warmstart{qtag}.npz"
    if warm is not None:
        warm_idx, first = warm
        entry = record[warm_idx]
        earlier = [int(record[j]["episode"]) for j in range(warm_idx) if lapped[j] > 0.0]
        if earlier:
            history = ("NOT the run's first lap: the snapshot(s) at episode(s) "
                       f"{', '.join(map(str, earlier))} lapped in a few eval episodes and "
                       "lost it again")
        else:
            history = "no earlier snapshot lapped: a pre-first-lap checkpoint"
        np.savez(warm_path, params=entry["params"])
        _write_sidecar(warm_path, {
            **base_meta(entry),
            "run": run_block,
            **evals(warm_idx),
            "warmstart": {
                "breakthrough_episode": int(record[first]["episode"]),
                "breakthrough_fraction": BREAKTHROUGH,
                "earlier_lapping_episodes": earlier,
            },
            **run_blocks(),
            "provenance": (f"warm-start checkpoint: snapshot at episode {entry['episode']} "
                           f"of {run_text()}; it laps in neither eval, and it is the last "
                           f"such snapshot before the run's breakthrough at episode "
                           f"{record[first]['episode']} (the first snapshot that laps in at "
                           f"least half of an eval's episodes); {history}"),
            "date": time.strftime("%Y-%m-%d"),
        })
        print(f"saved {warm_path}  (ep {entry['episode']}, laps in neither eval; "
              f"breakthrough at ep {record[first]['episode']}; {history})", flush=True)
    else:
        stale = (" (the existing file is from another run: remove it)"
                 if warm_path.is_file() else "")
        print("WARNING: no lap-less snapshot before a breakthrough (the run never laps "
              f"in half of an eval, or laps from its first snapshot) — {warm_path.name} "
              f"NOT written{stale}", flush=True)

    paths = []
    for stage, idx in enumerate(picked, start=1):
        entry = record[idx]
        npz_path = weights_dir / f"quantum_{track_name}_stage{stage}{qtag}.npz"
        last = stage == N_STAGES
        if last and same_run is not None:
            # the last stage IS the bundled driver: the same file, byte for byte
            shutil.copyfile(driver, npz_path)
        else:
            np.savez(npz_path, params=entry["params"])
        provenance = f"snapshot at episode {entry['episode']} of {run_text()}"
        if last and same_run:
            provenance += f"; the run's best snapshot, a copy of {driver.name}"
        meta = {
            **base_meta(entry),
            "stage": stage,
            "run": run_block,
            **evals(idx),
            **run_blocks(),
            "provenance": provenance,
            "date": time.strftime("%Y-%m-%d"),
        }
        if last and same_run is False:
            # ends on the bundled driver, which is NOT a snapshot of this run:
            # its own sidecar says what it is
            bundled_meta = _read_sidecar(driver)
            meta = {
                "agent": "quantum",
                "track": track_name,
                "config_hash": bundled_meta.get("config_hash"),
                "episodes": bundled_meta.get("episodes"),
                "stage": stage,
                "run": run_block,
                **{key: bundled_meta[key] for key in RUN_BLOCKS if key in bundled_meta},
                "provenance": (f"a copy of {driver.name} — NOT a snapshot of the run the "
                               f"earlier stages come from ({run_text()})"),
                "date": time.strftime("%Y-%m-%d"),
            }
        _write_sidecar(npz_path, meta)
        print(f"saved {npz_path}  (ep {meta['episodes']}, "
              f"lapped={entry['lapped_episodes']}/{entry['eval_episodes']}"
              f"{f'; {_exact_text(exact_results[idx])}' if exact_results else ''})",
              flush=True)
        paths.append(npz_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="generate evolution-mode stage weights by replaying the bundled "
                    "driver's training run")
    parser.add_argument("--track", default="oval", help="track name (default oval)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: the bundled driver's; 42 for a fresh run)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="training episodes (default: the bundled driver's; the "
                             "track's recipe for a fresh run)")
    parser.add_argument("--profile", default=None,
                        help="config profile overlay (e.g. q6) — files get the _q{n} tag")
    parser.add_argument("--init", default=None,
                        help="warm-start the stage training from a weights .npz")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="config override on top of the recipe, repeatable "
                             "(as train_headless --set)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the bundled driver: train the config's recipe and "
                             "end the stages on that run's best snapshot")
    parser.add_argument("--allow-mismatch", action="store_true",
                        help="write the stages even if the replay does not reproduce the "
                             "bundled driver (the last stage is then a copy of it)")
    parser.add_argument("--out", default=None,
                        help="weights directory to read the driver from and write to "
                             "(default: the bundled traqmania/weights)")
    args = parser.parse_args()
    make_stages(args.track, args.seed, args.episodes, args.profile, args.init,
                overrides=args.overrides, weights_dir=args.out, fresh=args.fresh,
                allow_mismatch=args.allow_mismatch)


if __name__ == "__main__":
    main()
