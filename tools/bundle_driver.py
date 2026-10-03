"""Bundle a driver from a multi-seed study: rank the seeds, re-evaluate, write it.

A bundled driver is ``traqmania/weights/<name>.npz`` plus a ``.meta.json``
sidecar.  Picking it by hand from one run reports that run's lucky numbers;
this tool picks it from a ``tools/study.py`` study by a fixed rule and writes
down what the choice was made from::

    python tools/bundle_driver.py --study DIR --variant trunc --name quantum_oval

1. The variant's seeds are ranked by the study's stored reliability eval of
   each best snapshot: lapped fraction (descending), then mean lap.
2. Winner's-curse control: the seed that tops that table was picked BY those
   episodes, so its number there is biased upwards.  The top ``--top`` seeds
   are re-evaluated on fresh distinct greedy episodes (``--eval-episodes``,
   env seed ``--eval-seed`` — refused when it coincides with a seed the study
   used) under each cell's own observation / action / circuit settings, and
   the driver is chosen by that result.  The sidecar records the FRESH numbers
   of the chosen driver, never the ones it was shortlisted by.  (The fresh
   eval still picks among ``--top`` candidates, so a small selection effect
   remains in it; with one candidate, ``--top 1``, there is none.)
3. ``--device-episodes N`` (quantum only) also drives each candidate on the
   local simulated-device path of ``tools/hw_reliability.py``; the result is
   reported, and ranks first with ``--rank-by device``.
4. Unless ``--dry-run``: the cell's weights are copied byte for byte to
   ``<out-dir>/<name>.npz`` and a sidecar is written with the cell's agent /
   track / observation / actions / circuit / training table, a real date, a
   ``selection`` record (study, variant, rule, every candidate's numbers, the
   seed-to-seed spread of the recipe, the fresh eval) and a one-paragraph
   ``provenance``.  A chosen driver below ``--min-lapped`` is not written
   (exit code 1) without ``--force``; an existing target is not replaced
   without ``--overwrite``.

A study of ``--track multi`` is re-evaluated on each of the four tracks
separately; the choice is by the number of tracks lapped in at least
``--min-lapped`` of the episodes, then total episodes lapped, and
``--min-lapped`` must hold on every track.

``--list`` prints every variant of a study with its seed-spread statistics::

    python tools/bundle_driver.py --study DIR --list

Only numpy and the standard library are imported at module level (plus the
sibling ``tools/study.py``, which follows the same rule); ``traqmania`` and,
for the device path, qiskit are loaded lazily.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
import shutil
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DEFAULT_OUT_DIR = REPO_ROOT / "traqmania" / "weights"

TOP = 3  # candidates re-evaluated on fresh episodes
EVAL_EPISODES = 72  # fresh distinct greedy episodes per candidate (and per track)
EVAL_SEED = 31_000  # fresh env seed: neither study.EVAL_SEED nor a trainer's seed + 10_000
MIN_LAPPED = 0.9  # a bundled driver laps in at least this fraction of fresh episodes
SEED_SPAN = 4  # MultiTrackEnv seeds its per-track envs seed, seed + 1, ...: keep clear of all
RANK_BY = ("fresh", "device")
RESCALE_CHOICES = ("off", "global", "readout")


def _load_tool(name: str):
    """A sibling ``tools/<name>.py`` as a module (``tools`` is not a package)."""
    spec = importlib.util.spec_from_file_location(f"traqmania_tool_{name}",
                                                  TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = _load_tool("study")
_hw_module = None


def hw_tool():
    """``tools/hw_reliability.py``, loaded on first use (the device path only)."""
    global _hw_module
    if _hw_module is None:
        _hw_module = _load_tool("hw_reliability")
    return _hw_module


# ------------------------------------------------------------------ helpers


def _num(value: Any, digits: int | None = None) -> float | None:
    """JSON-safe number: None / NaN / inf become None; optionally rounded."""
    if value is None or not np.isfinite(value):
        return None
    return float(value) if digits is None else round(float(value), digits)


def _text(value: Any, fmt: str = ".1f") -> str:
    return "—" if value is None else f"{value:{fmt}}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _sidecar_path(npz: Path) -> Path:
    return npz.with_suffix("").with_suffix(".meta.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ----------------------------------------------------------- study contents


def load_cells(directory: Path, variant: str) -> tuple[dict, list[dict], list[int]]:
    """(manifest, the variant's finished cells by seed, its failed seeds).

    A cell is ``{seed, dir, spec, result, weights}``: ``spec`` is its
    ``cell.json`` recipe, ``result`` its ``result.json``, ``weights`` the
    best-snapshot ``.npz`` (sidecar next to it).
    """
    directory = Path(directory)
    if not (directory / "cells").is_dir():
        raise ValueError(f"{directory} is not a study directory (no cells/ in it)")
    manifest, results, failed = study.load_study(directory)
    if variant not in results:
        raise ValueError(f"variant '{variant}' has no finished cells in {directory} "
                         f"(variants with results: {list(results)})")
    cells = []
    for result in results[variant]:
        cell = study.cell_dir(directory, variant, result["seed"])
        spec_path = cell / study.SPEC_NAME
        if not spec_path.is_file():
            raise ValueError(f"{cell} has a result but no {study.SPEC_NAME}")
        cells.append({
            "seed": int(result["seed"]),
            "dir": cell,
            "spec": json.loads(spec_path.read_text(encoding="utf-8")),
            "result": result,
            "weights": cell / result["weights"],
        })
    failed_seeds = sorted(cell["seed"] for cell in failed if cell["variant"] == variant)
    return manifest, cells, failed_seeds


def study_key(result: dict) -> tuple:
    """Sort key of step 1: the study's reliability eval of the best snapshot —
    lapped fraction descending, then mean lap ascending (no lap last), then seed."""
    best = result["best_snapshot_eval"]
    lap = _num(best["mean_lap"])
    return (-best["lapped_episodes"] / best["episodes"],
            math.inf if lap is None else lap, result["seed"])


def _spread(values: Iterable[float | None], n_boot: int) -> dict:
    """IQM with its bootstrap interval, and the median, over the seeds that
    HAVE the value (``n`` of them)."""
    present = [float(v) for v in values if v is not None]
    point, low, high = study.bootstrap_ci(present, statistic=study.iqm, n_boot=n_boot)
    return {"iqm": _num(point, 4), "ci_low": _num(low, 4), "ci_high": _num(high, 4),
            "median": _num(study.median(present), 4), "n": len(present)}


def seed_spread(results: Sequence[dict], n_boot: int | None = None) -> dict:
    """Seed-to-seed spread of one variant: what the RECIPE delivers, as opposed
    to the one seed that gets bundled."""
    n_boot = study.N_BOOT if n_boot is None else int(n_boot)
    rows = [study.seed_metrics(result) for result in results]
    return {
        "n_seeds": len(rows),
        "best_snapshot_lapped": _spread([r["best_lapped_frac"] for r in rows], n_boot),
        "final_params_lapped": _spread([r["final_lapped_frac"] for r in rows], n_boot),
        "stability": _spread([r["stability"] for r in rows], n_boot),
        "seeds_lapping_half": sum(r["best_lapped_frac"] >= study.RELIABLE_FRACTION
                                  for r in rows),
        "interval": f"{round(100 * study.CI_LEVEL)}% percentile bootstrap of the IQM over "
                    f"seeds, {n_boot} resamples, RNG seed {study.BOOTSTRAP_SEED}",
    }


def _spread_text(stat: dict) -> str:
    if stat["iqm"] is None:
        return "—"
    interval = "—" if stat["ci_low"] is None else f"{stat['ci_low']:.2f}, {stat['ci_high']:.2f}"
    return f"{stat['iqm']:.2f} [{interval}]"


# --------------------------------------------------------------- evaluation


def eval_tracks(track: str) -> list[str]:
    """Tracks a fresh eval runs on, one after the other: the four of the
    mixture for ``multi``, else the cell's own track."""
    if track == "multi":
        study._find_traqmania()
        from traqmania.train_headless import MULTI_TRACK_NAMES

        return list(MULTI_TRACK_NAMES)
    return [track]


def evaluate(spec: dict, weights: Path, episodes: int, eval_seed: int,
             tracks: Sequence[str]) -> dict[str, dict]:
    """``{track: study.greedy_eval result}`` of ``episodes`` distinct greedy
    episodes per track, under the config the weights drive with: the cell's
    profile and overrides, then the observation / circuit shape / action count
    of the weights' sidecar (``study.weights_config``)."""
    study._find_traqmania()
    from traqmania.train_headless import build_qfunc

    config = study.weights_config(spec, weights)
    max_steps = int(config["reward"]["max_decisions"]) + 1
    params = np.load(weights)["params"]
    out = {}
    for track in tracks:
        env = study._eval_env({**spec, "track": track, "eval_episodes": int(episodes),
                               "eval_seed": int(eval_seed)}, config)
        qfunc = build_qfunc(spec["agent"], env.n_features, int(spec["seed"]), config,
                            n_actions=env.n_actions)
        qfunc.set_params(params)
        out[track] = study.greedy_eval(qfunc, env, max_steps)
    return out


def summarize(per_track: dict[str, dict], reliable: float) -> dict:
    """Totals of a per-track eval, plus how many tracks lap ``reliable``-ly."""
    evals = list(per_track.values())
    episodes = sum(e["episodes"] for e in evals)
    lapped = sum(e["lapped_episodes"] for e in evals)
    laps = sum(e["laps"] for e in evals)
    lap_time = sum(e["mean_lap"] * e["laps"] for e in evals if e["laps"])
    best = [e["best_lap"] for e in evals if e["best_lap"] is not None]
    out = {
        "episodes": episodes,
        "lapped": lapped,
        "lapped_fraction": _num(lapped / episodes, 4),
        "laps": laps,
        "mean_lap": _num(lap_time / laps, 3) if laps else None,
        "best_lap": _num(min(best), 3) if best else None,
        "tracks_reliable": sum(e["lapped_episodes"] >= reliable * e["episodes"] - 1e-9
                               for e in evals),
    }
    if len(per_track) > 1:
        out["tracks"] = {
            name: {"episodes": e["episodes"], "lapped": e["lapped_episodes"],
                   "lapped_fraction": _num(e["lapped_episodes"] / e["episodes"], 4),
                   "laps": e["laps"], "mean_lap": _num(e["mean_lap"], 3),
                   "best_lap": _num(e["best_lap"], 3)}
            for name, e in per_track.items()
        }
    return out


def reproduces(replay: dict, stored: dict) -> bool:
    """Does re-running the study's own eval give the numbers it stored?"""
    if (replay["lapped_episodes"], replay["laps"]) != (stored["lapped_episodes"],
                                                       stored.get("laps", replay["laps"])):
        return False
    mine, theirs = _num(replay["mean_lap"]), _num(stored["mean_lap"])
    if mine is None or theirs is None:
        return mine is theirs
    return math.isclose(mine, theirs, rel_tol=1e-6, abs_tol=1e-6)


def _call_known(func: Callable, **available: Any) -> Any:
    """Call ``func`` with those of ``available`` it takes — so a parameter
    added to or dropped from a sibling tool's function does not break the
    call; a new REQUIRED one that is not on offer is reported by name."""
    kwargs = {}
    for name, param in inspect.signature(func).parameters.items():
        if name in available:
            kwargs[name] = available[name]
        elif param.default is inspect.Parameter.empty and param.kind in (
                param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise ValueError(
                f"tools/hw_reliability.py: {func.__name__}() now needs '{name}', which "
                "tools/bundle_driver.py does not provide — update its device_eval()")
    return func(**kwargs)


def device_eval(spec: dict, weights: Path, args: argparse.Namespace,
                tracks: Sequence[str], backends: dict) -> dict:
    """``args.device_episodes`` episodes per track on the local simulated-device
    path (``hw_reliability.device_row``: the Aer twin of the fake backend).

    ``backends`` caches the fake backend per qubit count across candidates.
    """
    hw = hw_tool()
    study._find_traqmania()
    from traqmania import hardware

    config = study.weights_config(spec, weights)
    params = np.load(weights)["params"]
    n_qubits = int(config["circuit"]["n_qubits"])
    cap = int(args.device_max_decisions or config["reward"]["max_decisions"])
    if n_qubits not in backends:
        backends[n_qubits] = hardware.get_backend(use_fake=True, fake_name=args.fake,
                                                  min_qubits=max(5, n_qubits))
    rows = {}
    for track in tracks:
        # hw_reliability's own argument namespace (its defaults for whatever
        # else it reads), with this run's settings on top
        hw_args = hw.build_parser().parse_args(["--weights", str(weights), "--track", track])
        settings = {
            "track": track, "profile": spec["profile"], "fake": args.fake,
            "device_episodes": int(args.device_episodes), "device_max_decisions": cap,
            "eval_seed": int(args.eval_seed), "seed": int(args.device_seed),
        }
        for key, value in settings.items():
            setattr(hw_args, key, value)
        if getattr(hw_args, "calibration_samples", None) is None:
            hw_args.calibration_samples = hardware.CALIBRATION_SAMPLES
        if getattr(hw_args, "calibration_shots", None) is None:
            hw_args.calibration_shots = hardware.CALIBRATION_MIN_SHOTS
        rows[track] = _call_known(
            hw.device_row, config=config, params=params, args=hw_args,
            backend=backends[n_qubits], shots=int(args.shots), rescale=args.device_rescale,
            resilience=int(args.device_resilience), row_seed=0,
            weights=weights, weights_path=weights, track=track, profile=spec["profile"])
    first = next(iter(rows.values()))
    out = {
        "fake": args.fake,
        "backend": first.get("backend"),
        "shots": int(args.shots),
        "rescale": args.device_rescale,
        "resilience": int(args.device_resilience),
        "episodes": sum(int(row["episodes"]) for row in rows.values()),
        "lapped": sum(int(row["lapped"]) for row in rows.values()),
        "crashed": sum(int(row.get("crashed", 0)) for row in rows.values()),
        "max_decisions": cap,
        "eval_seed": int(args.eval_seed),
        "simulator_seed": int(args.device_seed),
        "two_qubit_gates": first.get("two_qubit_gates"),
    }
    if len(rows) == 1:
        out["mean_lap"] = _num(first.get("mean_lap"), 3)
    else:
        out["tracks"] = {name: {"episodes": int(row["episodes"]), "lapped": int(row["lapped"]),
                                "mean_lap": _num(row.get("mean_lap"), 3)}
                         for name, row in rows.items()}
    return out


# ---------------------------------------------------------------- selection


def choice_key(candidate: dict, rank_by: str) -> tuple:
    """Sort key of step 2 — the fresh eval: tracks lapped reliably (one track:
    the same order as its lapped fraction), then episodes lapped, then mean
    lap, then the study rank; ``rank_by="device"`` puts the device laps first."""
    fresh = candidate["fresh"]
    lap = fresh["mean_lap"]
    key = (-fresh["tracks_reliable"], -fresh["lapped"], math.inf if lap is None else lap,
           candidate["rank"])
    if rank_by == "device":
        return (-candidate["device"]["lapped"], *key)
    return key


def check_eval_seed(eval_seed: int, cells: Sequence[dict]) -> None:
    """Refuse a "fresh" env seed whose episodes the candidates have already
    been selected on: the study's reliability eval, a trainer's snapshot evals
    (training seed + 10_000) or its training env (the training seed)."""
    def near(seed: int) -> bool:
        return abs(int(eval_seed) - int(seed)) < SEED_SPAN

    for cell in cells:
        spec = cell["spec"]
        used = {"the study's reliability eval": spec["eval_seed"],
                f"seed {spec['seed']}'s in-training snapshot evals": spec["seed"] + 10_000,
                f"seed {spec['seed']}'s training env": spec["seed"]}
        for what, seed in used.items():
            if near(seed):
                raise ValueError(
                    f"--eval-seed {eval_seed} is not fresh: {what} used env seed {seed}; "
                    f"pick one at least {SEED_SPAN} away")


def rule_text(args: argparse.Namespace, n_top: int, multi: bool, study_episodes: int) -> str:
    order = (f"tracks lapped in >= {args.min_lapped:.0%} of the episodes, then episodes lapped"
             if multi else "lapped fraction, then mean lap")
    if args.rank_by == "device":
        order = f"episodes lapped on the simulated device, then {order}"
    per_track = " per track" if multi else ""
    if getattr(args, "seed", None) is not None:
        return (f"seed {args.seed} chosen by hand (--seed; reason in the note) and re-evaluated on "
                f"{args.eval_episodes} fresh distinct greedy episodes{per_track} (env seed "
                f"{args.eval_seed}); the fresh numbers are the ones recorded")
    rule = (
        f"seeds ranked by the study's {study_episodes}-episode reliability eval of the best "
        f"snapshot (lapped fraction, then mean lap); the top {n_top} re-evaluated on "
        f"{args.eval_episodes} fresh distinct greedy episodes{per_track} (env seed "
        f"{args.eval_seed}) and the driver chosen by that result ({order}); the fresh numbers "
        "are the ones recorded")
    if n_top > 1:
        rule += (f" (they also picked among the {n_top} candidates, so a small selection "
                 "effect remains in them)")
    return rule


def _eval_phrase(fresh: dict) -> str:
    """'laps in 70/72 episodes, best 13.7 s, mean 14.4 s' (per track for multi)."""
    def one(lapped: int, episodes: int, best: float | None, mean: float | None) -> str:
        text = f"laps in {lapped}/{episodes} episodes"
        if mean is not None:
            text += f", best {best:.1f} s, mean {mean:.1f} s"
        return text

    if "tracks" in fresh:
        return "; ".join(
            f"{name} {one(t['lapped'], t['episodes'], t['best_lap'], t['mean_lap'])}"
            for name, t in fresh["tracks"].items())
    return one(fresh["lapped"], fresh["episodes"], fresh["best_lap"], fresh["mean_lap"])


def provenance_text(meta: dict, selection: dict) -> str:
    """One paragraph, in the style of the hand-written sidecars, built from
    the numbers of ``selection``."""
    spread = selection["seed_spread"]
    recipe = [f"{meta.get('episodes', '?')} episodes"]
    if selection["profile"]:
        recipe.append(f"profile {selection['profile']}")
    if selection.get("init"):
        recipe.append(f"warm-started from {selection['init']}")
    if selection.get("pace"):
        recipe.append("pace fine-tune")
    recipe += selection["overrides"]
    n_top = len(selection["candidates"])
    fresh = selection["fresh_eval"]
    parts = [
        f"best snapshot of seed {selection['chosen_seed']}, chosen from the "
        f"{selection['n_seeds']}-seed study {selection['study']} variant "
        f"{selection['variant']} ({meta.get('agent', '?')} on {meta.get('track', '?')}, "
        f"{', '.join(recipe)})",
        f"seeds ranked by the study's {selection['study_eval']['episodes']}-episode "
        f"reliability eval, the top {n_top} re-evaluated on fresh episodes and chosen by "
        + ("the simulated-device laps, then " if selection["rank_by"] == "device" else "")
        + "that result",
        (f"fresh greedy eval of {fresh['episodes_per_track']} episodes per track"
         if "tracks" in fresh else f"{fresh['episodes']}-episode fresh greedy eval")
        + f" (env seed {fresh['eval_seed']}): {_eval_phrase(fresh)}",
    ]
    if "device_eval" in selection:
        dev = selection["device_eval"]
        parts.append(
            f"simulated device ({dev['backend'] or dev['fake']}, {dev['shots']} shots, rescale "
            f"{dev['rescale']}, resilience {dev['resilience']}): laps in "
            f"{dev['lapped']}/{dev['episodes']} episodes")
    stability = spread["stability"]
    parts.append(
        f"the recipe across its {spread['n_seeds']} seeds (IQM [95% CI]): best snapshot laps "
        f"in {_spread_text(spread['best_snapshot_lapped'])} of the study's eval episodes, "
        f"final params in {_spread_text(spread['final_params_lapped'])}, stability "
        f"{_spread_text(stability)}"
        + (f" (n={stability['n']})" if stability["n"] != spread["n_seeds"] else "")
        + f"; {spread['seeds_lapping_half']}/{spread['n_seeds']} seeds lap in at least half")
    if selection.get("failed_seeds"):
        parts.append(f"{len(selection['failed_seeds'])} more seed(s) failed to train")
    if selection.get("note"):
        parts.append(selection["note"])
    return "; ".join(parts)


def describe_existing(npz: Path, new_weights: Path) -> list[str]:
    """What an ``--overwrite`` is about to replace."""
    lines = []
    if npz.is_file():
        same = _sha256(npz) == _sha256(new_weights)
        try:
            size = f"{np.load(npz)['params'].size} params"
        except (OSError, ValueError, KeyError):
            size = "unreadable params"
        lines.append(f"replacing {npz} ({size}, sha256 {_sha256(npz)[:12]}; "
                     + ("the SAME weights as the chosen cell" if same else "different weights")
                     + ")")
    meta_path = _sidecar_path(npz)
    if meta_path.is_file():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        chosen = (old.get("selection") or {}).get("chosen_seed")
        lines.append(
            f"replacing {meta_path}: agent {old.get('agent', '?')}, track "
            f"{old.get('track', '?')}, {old.get('episodes', '?')} episodes, date "
            f"{old.get('date', '?')}"
            + (f", study seed {chosen}" if chosen is not None else "")
            + f", provenance: {old.get('provenance') or '(none)'}")
    return lines


def name_notes(name: str, meta: dict) -> list[str]:
    """Mismatches between ``--name`` and the loaders' filename rules
    (``runtime.load_agent``: ``<agent>_<track>[_q<n>].npz``)."""
    notes = []
    agent = meta.get("agent")
    if agent and not name.startswith(f"{agent}_"):
        notes.append(f"note: --name {name} does not start with '{agent}_' — the server looks "
                     f"a {agent} driver up as {agent}_<track>.npz")
    n_qubits = int((meta.get("circuit") or {}).get("n_qubits", 4))
    tagged = f"_q{n_qubits}" in name
    if agent == "quantum" and n_qubits != 4 and not tagged:
        notes.append(f"note: --name {name} has no '_q{n_qubits}' tag, but these are "
                     f"{n_qubits}-qubit weights (4-qubit loaders would pick them up)")
    return notes


# ----------------------------------------------------------------- commands


def cmd_list(args: argparse.Namespace) -> int:
    """Every variant of the study with its seed-spread statistics."""
    directory = Path(args.study)
    manifest, results, failed = study.load_study(directory)
    if args.variant:
        results = {name: runs for name, runs in results.items() if name == args.variant}
    if not results:
        raise ValueError(f"no finished cells under {directory / 'cells'}"
                         + (f" for variant '{args.variant}'" if args.variant else ""))
    rows = []
    for name, runs in results.items():
        spread = seed_spread(runs, args.resamples)
        recipe = (manifest.get("variants") or {}).get(name) or {}
        first = runs[0]
        overrides = recipe.get("overrides")
        if overrides is None:
            overrides = [f"{k}={json.dumps(v)}" for k, v in (first.get("overrides") or {}).items()]
        # what else tells this variant's recipe from a plain run of its agent and track
        extras = [f"actions={first['actions']}"] if first.get("actions") is not None else []
        extras += [f"init={Path(first['init']).name}"] if first.get("init") else []
        extras += ["pace=true"] if first.get("pace") else []
        best_run = min(runs, key=study_key)
        best = best_run["best_snapshot_eval"]
        n_failed = sum(cell["variant"] == name for cell in failed)
        stability = spread["stability"]
        rows.append([
            name, str(first.get("agent", "?")), str(first.get("track", "?")),
            str(first.get("profile") or "—"),
            f"{len(runs)}" + (f" (+{n_failed} failed)" if n_failed else ""),
            _spread_text(spread["best_snapshot_lapped"]),
            _text(spread["best_snapshot_lapped"]["median"], ".2f"),
            _spread_text(spread["final_params_lapped"]),
            _spread_text(stability)
            + (f" (n={stability['n']})" if stability["n"] != len(runs) else ""),
            f"{spread['seeds_lapping_half']}/{len(runs)}",
            f"seed {best_run['seed']}: {best['lapped_episodes']}/{best['episodes']}"
            + (f", {best['mean_lap']:.1f} s" if _num(best["mean_lap"]) is not None else ""),
            ", ".join(f"`{item}`" for item in [*extras, *overrides]) or "—",
        ])
    print(f"## Study {directory.resolve().name}: seed spread per variant")
    print()
    print(_table(
        ["variant", "agent", "track", "profile", "seeds", "best-snapshot lapped IQM [95% CI]",
         "median", "final-params lapped IQM [95% CI]", "stability IQM [95% CI]",
         f"seeds lapping >= {study.RELIABLE_FRACTION:.0%}", "best seed (study eval)",
         "recipe"], rows))
    print()
    print("Lapped fractions are over the study's reliability eval "
          f"({(manifest.get('eval') or {}).get('episodes', '?')} distinct greedy episodes); "
          "IQM = mean of the middle half of the seeds, interval = percentile bootstrap over "
          "seeds (it understates the uncertainty below about 10 seeds). The best seed's own "
          "number is a maximum over seeds — bundle_driver re-evaluates it on fresh episodes.")
    return 0


def _ranking_table(ranked: Sequence[dict], n_top: int) -> str:
    rows = []
    for cell in ranked:
        metrics = study.seed_metrics(cell["result"])
        best = cell["result"]["best_snapshot_eval"]
        rows.append([
            str(cell["rank"]), str(cell["seed"]),
            f"{best['lapped_episodes']}/{best['episodes']}",
            _text(_num(best["mean_lap"])), _text(_num(best["best_lap"])),
            f"{metrics['final_lapped']}/{metrics['eval_episodes']}",
            _text(metrics["stability"], ".2f"),
            _text(metrics["first_clean_episode"], "d"),
            "candidate" if cell["rank"] <= n_top else "",
        ])
    return _table(["rank", "seed", "best snapshot lapped", "mean lap (s)", "best lap (s)",
                   "final params lapped", "stability", "first clean lap (episode)", ""], rows)


def _candidate_table(top: Sequence[dict], chosen: dict, tracks: Sequence[str]) -> str:
    multi = len(tracks) > 1
    header = ["seed", "study lapped", "study mean lap (s)", "study eval reproduced"]
    header += [f"fresh {name}" for name in tracks] if multi else []
    header += ["fresh lapped", "fresh mean lap (s)", "fresh best lap (s)"]
    if multi:
        header.append("tracks reliable")
    if "device" in chosen:
        header.append("device lapped")
    header.append("")
    rows = []
    for cand in top:
        best, fresh = cand["result"]["best_snapshot_eval"], cand["fresh"]
        row = [str(cand["seed"]), f"{best['lapped_episodes']}/{best['episodes']}",
               _text(_num(best["mean_lap"])),
               {True: "yes", False: "NO", None: "not checked"}[cand["reproduced"]]]
        if multi:
            row += [f"{t['lapped']}/{t['episodes']} ({_text(t['mean_lap'])} s)"
                    for t in fresh["tracks"].values()]
        row += [f"{fresh['lapped']}/{fresh['episodes']}", _text(fresh["mean_lap"]),
                _text(fresh["best_lap"])]
        if multi:
            row.append(f"{fresh['tracks_reliable']}/{len(tracks)}")
        if "device" in cand:
            row.append(f"{cand['device']['lapped']}/{cand['device']['episodes']}")
        row.append("CHOSEN" if cand is chosen else "")
        rows.append(row)
    return _table(header, rows)


def build_selection(directory: Path, manifest: dict, cells: Sequence[dict],
                    top: Sequence[dict], chosen: dict, spread: dict,
                    failed_seeds: Sequence[int], multi: bool,
                    args: argparse.Namespace) -> dict:
    """The sidecar's ``selection`` record: where the driver comes from, the
    rule, every candidate's study and fresh numbers, the recipe's seed spread
    and the chosen driver's fresh (and device) eval."""
    study_episodes = int(chosen["result"]["best_snapshot_eval"]["episodes"])
    fresh_eval = {"episodes": chosen["fresh"]["episodes"], "eval_seed": int(args.eval_seed),
                  **{key: chosen["fresh"][key]
                     for key in ("lapped", "lapped_fraction", "laps", "mean_lap", "best_lap")}}
    if multi:
        fresh_eval.update(episodes_per_track=int(args.eval_episodes),
                          tracks_reliable=chosen["fresh"]["tracks_reliable"],
                          reliable_fraction=float(args.min_lapped),
                          tracks=chosen["fresh"]["tracks"])
    candidates = []
    for cell in top:
        best = cell["result"]["best_snapshot_eval"]
        entry = {
            "seed": cell["seed"],
            "study_lapped": best["lapped_episodes"],
            "study_episodes": best["episodes"],
            "study_mean_lap": _num(best["mean_lap"], 3),
            "study_eval_reproduced": cell["reproduced"],
            "fresh_lapped": cell["fresh"]["lapped"],
            "fresh_episodes": cell["fresh"]["episodes"],
            "fresh_mean_lap": cell["fresh"]["mean_lap"],
            "fresh_best_lap": cell["fresh"]["best_lap"],
        }
        if multi:
            entry["fresh_tracks"] = {name: t["lapped"]
                                     for name, t in cell["fresh"]["tracks"].items()}
            entry["fresh_tracks_reliable"] = cell["fresh"]["tracks_reliable"]
        if "device" in cell:
            entry["device_lapped"] = cell["device"]["lapped"]
            entry["device_episodes"] = cell["device"]["episodes"]
        candidates.append(entry)
    selection: dict[str, Any] = {
        "study": directory.resolve().name,
        "variant": args.variant,
        "profile": chosen["spec"]["profile"],
        "overrides": list(chosen["spec"]["overrides"]),
        "chosen_seed": chosen["seed"],
        "n_seeds": len(cells),
        "rank_by": args.rank_by,
        "rule": rule_text(args, len(top), multi, study_episodes),
        "study_eval": {"episodes": study_episodes, "eval_seed": chosen["spec"]["eval_seed"]},
        "source": chosen["weights"].relative_to(directory).as_posix(),
        "weights_sha256": _sha256(chosen["weights"]),
        "candidates": candidates,
        "seed_spread": spread,
        "fresh_eval": fresh_eval,
    }
    commits = sorted({run["git_commit"] for run in manifest.get("runs", [])
                      if run.get("git_commit")})
    if chosen["spec"].get("init"):  # a fine-tune: what it started from (file name only)
        selection["init"] = Path(chosen["spec"]["init"]).name
    if chosen["spec"].get("pace"):
        selection["pace"] = True
    if chosen["spec"].get("actions") is not None:
        selection["actions"] = int(chosen["spec"]["actions"])
    if commits:  # the checkout(s) that trained the study, when it ran from a git tree
        selection["study_commits"] = commits
    if failed_seeds:
        selection["failed_seeds"] = failed_seeds
    if "device" in chosen:
        selection["device_eval"] = chosen["device"]
    if args.note:
        selection["note"] = args.note
    return selection


def cmd_bundle(args: argparse.Namespace) -> int:
    """Steps 1-5 of the module docstring; exit code 0, or 1 when the chosen
    driver misses ``--min-lapped`` (nothing written without ``--force``)."""
    directory = Path(args.study)
    name = args.name[:-4] if args.name.endswith(".npz") else args.name
    if not name or Path(name).name != name:
        raise ValueError(f"--name must be a bare file stem like quantum_oval, got '{args.name}'")
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    target = out_dir / f"{name}.npz"
    target_meta = _sidecar_path(target)

    manifest, cells, failed_seeds = load_cells(directory, args.variant)
    spec0 = cells[0]["spec"]
    agent, track = spec0["agent"], spec0["track"]
    study_episodes = int(cells[0]["result"]["best_snapshot_eval"]["episodes"])
    if args.device_episodes > 0 and agent != "quantum":
        raise ValueError(f"--device-episodes needs a quantum driver; variant "
                         f"'{args.variant}' trains agent '{agent}'")
    if args.device_episodes > 0 and track == "random":
        raise ValueError("--device-episodes cannot run on --track random studies")

    # ---- step 1: the study's own ranking
    ranked = sorted(cells, key=lambda cell: study_key(cell["result"]))
    for rank, cell in enumerate(ranked, start=1):
        cell["rank"] = rank
    if args.seed is not None:  # a human-chosen seed: still fresh-evaluated and recorded as such
        top = [cell for cell in ranked if cell["seed"] == args.seed]
        if not top:
            raise ValueError(f"--seed {args.seed} is not a finished seed of variant {args.variant}")
    else:
        top = ranked[:args.top]
    spread = seed_spread([cell["result"] for cell in cells], args.resamples)
    print(f"## {directory.resolve().name} / {args.variant}: {agent} on {track}, "
          f"{len(cells)} seeds" + (f" ({len(failed_seeds)} more failed: {failed_seeds})"
                                   if failed_seeds else ""))
    print()
    print(f"recipe: profile {spec0['profile'] or '—'}, overrides "
          f"{', '.join(spec0['overrides']) or '—'}"
          + (f", warm-started from {Path(spec0['init']).name}" if spec0.get("init") else "")
          + (", pace fine-tune" if spec0.get("pace") else "")
          + f"; study eval: {study_episodes} episodes, env seed {spec0['eval_seed']}")
    print()
    print("### Seeds ranked by the study's reliability eval of the best snapshot")
    print()
    print(_ranking_table(ranked, len(top)))
    print()
    stability = spread["stability"]
    print(f"Seed spread (IQM [95% CI]; median): best snapshot lapped "
          f"{_spread_text(spread['best_snapshot_lapped'])}; "
          f"{_text(spread['best_snapshot_lapped']['median'], '.2f')} — final params lapped "
          f"{_spread_text(spread['final_params_lapped'])}; "
          f"{_text(spread['final_params_lapped']['median'], '.2f')} — stability "
          f"{_spread_text(stability)}; {_text(stability['median'], '.2f')} "
          f"(n={stability['n']}) — {spread['seeds_lapping_half']}/{len(cells)} seeds lap in "
          f">= {study.RELIABLE_FRACTION:.0%} of the study's eval episodes.")
    print()

    # ---- refusals that need no evaluation
    check_eval_seed(args.eval_seed, top)
    for cell in top:
        if not cell["weights"].is_file() or not _sidecar_path(cell["weights"]).is_file():
            raise ValueError(f"seed {cell['seed']}: {cell['weights']} or its .meta.json "
                             "sidecar is missing from the study")
    existing = [path for path in (target, target_meta) if path.exists()]
    if existing and not args.overwrite:
        if not args.dry_run:
            raise ValueError(f"{existing[0]} exists; pass --overwrite to replace it")
        print(f"note: {existing[0]} exists — a real run needs --overwrite")

    # ---- step 2 (+3): fresh episodes, optionally the simulated device
    tracks = eval_tracks(track)
    backends: dict = {}
    for i, cell in enumerate(top, start=1):
        print(f"evaluating seed {cell['seed']} ({i}/{len(top)}) ...", file=sys.stderr,
              flush=True)
        cell["reproduced"] = None
        if not args.skip_replay:
            # the study's own eval again: tells episode luck from a checkout
            # that drives these weights differently from the one that trained them
            replay = evaluate(cell["spec"], cell["weights"], cell["spec"]["eval_episodes"],
                              cell["spec"]["eval_seed"], [track])[track]
            cell["reproduced"] = reproduces(replay, cell["result"]["best_snapshot_eval"])
            cell["replay"] = replay
        cell["fresh_tracks"] = evaluate(cell["spec"], cell["weights"], args.eval_episodes,
                                        args.eval_seed, tracks)
        cell["fresh"] = summarize(cell["fresh_tracks"], args.min_lapped)
        if args.device_episodes > 0:
            cell["device"] = device_eval(cell["spec"], cell["weights"], args, tracks, backends)
            print(f"device: seed {cell['seed']}: {cell['device']['lapped']}/"
                  f"{cell['device']['episodes']} lapped", file=sys.stderr, flush=True)
    chosen = min(top, key=lambda cell: choice_key(cell, args.rank_by))

    multi = len(tracks) > 1
    print(f"### Fresh re-evaluation of the top {len(top)}: {args.eval_episodes} episodes"
          f"{' per track' if multi else ''}, env seed {args.eval_seed}"
          + (f"; device path {args.fake}, {args.shots} shots, {args.device_episodes} episodes"
             f"{' per track' if multi else ''}" if args.device_episodes > 0 else ""))
    print()
    print(_candidate_table(top, chosen, tracks))
    print()
    for cell in top:
        if cell["reproduced"] is False:
            stored, replay = cell["result"]["best_snapshot_eval"], cell["replay"]
            print(f"note: seed {cell['seed']}: re-running the study's own eval here gives "
                  f"{replay['lapped_episodes']}/{replay['episodes']} lapped (mean lap "
                  f"{_text(_num(replay['mean_lap']), '.2f')} s), the study recorded "
                  f"{stored['lapped_episodes']}/{stored['episodes']} "
                  f"({_text(_num(stored['mean_lap']), '.2f')} s) — this checkout drives these "
                  "weights differently from the one that trained them (physics, observation "
                  "or config changed); the fresh numbers are the ones that hold here")

    # ---- the sidecar: the cell's own, plus a real date and the selection on record
    meta = json.loads(_sidecar_path(chosen["weights"]).read_text(encoding="utf-8"))
    selection = build_selection(directory, manifest, cells, top, chosen, spread,
                                failed_seeds, multi, args)
    provenance = provenance_text(meta, selection)
    meta["date"] = datetime.date.today().isoformat()
    meta["provenance"] = provenance
    meta["selection"] = selection
    sidecar = json.dumps(meta, indent=2, allow_nan=False) + "\n"

    print(f"### Chosen: seed {chosen['seed']} -> {name}")
    print()
    print(provenance)
    print()
    for note in name_notes(name, meta):
        print(note)

    # ---- step 4: the reliability gate
    if multi:
        passed = chosen["fresh"]["tracks_reliable"] == len(tracks)
        got = (f"{chosen['fresh']['tracks_reliable']}/{len(tracks)} tracks lap in >= "
               f"{args.min_lapped:.0%} of the fresh episodes")
    else:
        passed = chosen["fresh"]["lapped"] >= args.min_lapped * chosen["fresh"]["episodes"] - 1e-9
        got = (f"the fresh eval laps in {chosen['fresh']['lapped']}/"
               f"{chosen['fresh']['episodes']} episodes")
    if not passed:
        if not args.force:
            print(f"REFUSED: {got}; --min-lapped {args.min_lapped:g} is not reached. Nothing "
                  "written (--force bundles it anyway).")
            return 1
        print(f"note: {got}, below --min-lapped {args.min_lapped:g}; bundling it anyway "
              "(--force)")

    if args.dry_run:
        print(f"dry run: nothing written. {target_meta} would be:")
        print()
        print(sidecar, end="")
        return 0

    # ---- step 5: write (weights first: a sidecar never describes other weights for long)
    for line in describe_existing(target, chosen["weights"]):
        print(line)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(chosen["weights"], tmp)
    os.replace(tmp, target)
    _write_atomic(target_meta, sidecar.encode("utf-8"))
    print(f"weights written to {target}")
    print(f"sidecar written to {target_meta}")
    return 0


# ---------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/bundle_driver.py",
        description="Select a bundled driver from a multi-seed study (tools/study.py): rank "
                    "the seeds, re-evaluate the best on fresh episodes, write weights + "
                    "sidecar with the selection on record.")
    parser.add_argument("--study", required=True, help="study directory (tools/study.py --out)")
    parser.add_argument("--variant", default=None, help="variant of the study to bundle from")
    parser.add_argument("--name", default=None,
                        help="file stem of the bundled driver, e.g. quantum_oval")
    parser.add_argument("--list", action="store_true",
                        help="print every variant with its seed-spread statistics and exit")
    parser.add_argument("--seed", type=int, default=None,
                        help="bundle this seed instead of the top candidates (it is still "
                             "re-evaluated on fresh episodes; the sidecar records the choice)")
    parser.add_argument("--top", type=int, default=TOP,
                        help=f"candidates re-evaluated on fresh episodes (default {TOP})")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help="fresh distinct greedy episodes per candidate, per track for a "
                             f"multi study (default {EVAL_EPISODES})")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help=f"env seed of the fresh episodes (default {EVAL_SEED}); must "
                             "differ from the seeds the study evaluated and trained on")
    parser.add_argument("--min-lapped", type=float, default=MIN_LAPPED,
                        help="lapped fraction the chosen driver's fresh eval must reach, on "
                             f"every track of a multi study (default {MIN_LAPPED})")
    parser.add_argument("--force", action="store_true",
                        help="bundle the chosen driver even below --min-lapped")
    parser.add_argument("--rank-by", default="fresh", choices=list(RANK_BY),
                        help="fresh (default): choose by the fresh eval; device: by the "
                             "simulated-device laps first (needs --device-episodes)")
    parser.add_argument("--device-episodes", type=int, default=0,
                        help="quantum only: episodes per candidate on the local "
                             "simulated-device path (default 0 = skip)")
    parser.add_argument("--fake", default="fake_miami",
                        help="fake backend of the device path (default fake_miami)")
    parser.add_argument("--shots", type=int, default=1024,
                        help="shots per decision on the device path (default 1024)")
    parser.add_argument("--device-rescale", default="off", choices=list(RESCALE_CHOICES),
                        help="attenuation rescale on the device path (default off)")
    parser.add_argument("--device-resilience", type=int, default=0,
                        help="Estimator resilience level on the device path (default 0)")
    parser.add_argument("--device-max-decisions", type=int, default=None,
                        help="cap on device episodes (default: the env's time limit; one "
                             "decision is one simulated job)")
    parser.add_argument("--device-seed", type=int, default=0,
                        help="simulator seed of the device path (default 0)")
    parser.add_argument("--out-dir", default=None,
                        help=f"where <name>.npz and <name>.meta.json go (default "
                             f"{DEFAULT_OUT_DIR})")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing <name>.npz / .meta.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="do everything but write; prints the sidecar")
    parser.add_argument("--note", default=None,
                        help="free text recorded in the sidecar (selection.note, provenance)")
    parser.add_argument("--skip-replay", action="store_true",
                        help="do not re-run the study's own eval of each candidate (the check "
                             "that this checkout still drives the weights as the study did)")
    parser.add_argument("--resamples", type=int, default=None,
                        help="bootstrap resamples of the seed-spread intervals (default: "
                             "the study tool's)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list:
            return cmd_list(args)
        if not args.variant or not args.name:
            raise ValueError("--variant and --name are required (or --list)")
        if args.top < 1:
            raise ValueError(f"--top must be >= 1, got {args.top}")
        if args.eval_episodes < 1:
            raise ValueError(f"--eval-episodes must be >= 1, got {args.eval_episodes}")
        if not 0.0 <= args.min_lapped <= 1.0:
            raise ValueError(f"--min-lapped is a fraction in [0, 1], got {args.min_lapped}")
        if args.device_episodes < 0:
            raise ValueError(f"--device-episodes must be >= 0, got {args.device_episodes}")
        if args.rank_by == "device" and args.device_episodes < 1:
            raise ValueError("--rank-by device needs --device-episodes N > 0")
        return cmd_bundle(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
