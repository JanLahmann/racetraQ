"""Export what the browser-only variant (``browser/``) needs as static JSON.

The browser app has no server: it ports the track geometry, car physics,
observation pipeline and both Q-functions to TypeScript and runs the circuit
on QAMPoser's in-browser state-vector simulator.  This tool writes its data
and the numpy reference values its parity tests check the port against:

- ``browser/public/data/manifest.json`` — physics, tracks, drivers (with the
  honest provenance and fresh-eval numbers of every bundled driver), ghosts
- ``browser/public/data/tracks/<id>.json`` — the raw track JSON (the browser
  resamples it exactly like :class:`~racetraq.env.track.Track`)
- ``browser/public/data/drivers/<id>.json`` — flat params, circuit shape or
  hidden width, the observation the driver was trained with, action count
- ``browser/public/data/ghosts/<track>.json`` — a ghost lap per track: the
  track's quantum driver's greedy lap from a standing start (reproducible from
  the bundled weights; the server's recorded ghosts are local, not in git)
- ``browser/tests/fixtures/parity.json`` — numpy results for track queries,
  physics substeps, observations, circuit expectations, Q-values and greedy
  closed-loop rollouts

Usage: python tools/export_browser.py [--out browser] [--check]

``--check`` exports to a temporary directory and fails when the data files
differ from the committed ones (the simulated files — parity fixture, ghost
laps, manifest — are compared with a float tolerance, since numpy's last bits
vary across platforms).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from racetraq.agents.base import action_labels, action_set
from racetraq.agents.classical import MLPQFunction
from racetraq.agents.quantum.qdqn import QuantumQFunction
from racetraq.config import load_config
from racetraq.env.car import CarPhysics
from racetraq.env.racing_env import CarObserver
from racetraq.env.track import TRACKS_DIR, Track
from racetraq.server import runtime

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "racetraq" / "weights"
TRACKS = ("oval", "chicane", "gp", "combo")
TRACK_NAMES = {"oval": "Oval", "chicane": "Chicane", "gp": "Grand Prix", "combo": "Combo"}
# multi-track drivers race on every track
MULTI_TRACK = {"quantum_universal": "multi", "mlp_pro": "multi"}
DATA_VERSION = 1

# parity fixture: which drivers get Q-value checks and closed-loop rollouts
CIRCUIT_CASES = ("quantum_oval", "quantum_gp", "quantum_oval_q6", "quantum_oval_q8",
                 "quantum_oval_q10", "quantum_universal")
MLP_CASES = ("mlp_gp", "mlp_pro")
ROLLOUTS = (("quantum_oval", "oval"), ("quantum_chicane", "chicane"), ("quantum_gp", "gp"),
            ("quantum_combo", "combo"), ("mlp_gp", "gp"), ("mlp_pro", "chicane"),
            ("quantum_oval_q10", "oval"))
MAX_ROLLOUT_DECISIONS = 600


def _floats(values) -> list:
    """numpy array -> nested lists of Python floats (exact repr in JSON)."""
    return np.asarray(values, dtype=np.float64).tolist()


def _write_json(path: Path, payload, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


# ----------------------------------------------------------------- drivers

def _meta(path: Path) -> dict:
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _driver_kind(stem: str) -> tuple[str, int | None]:
    """('driver' | 'stage' | 'warmstart', stage index)."""
    for part in stem.split("_"):
        if part.startswith("stage") and part[5:].isdigit():
            return "stage", int(part[5:])
        if part == "warmstart":
            return "warmstart", None
    return "driver", None


def _driver_track(stem: str, meta: dict) -> str:
    if stem in MULTI_TRACK:
        return MULTI_TRACK[stem]
    track = meta.get("track")
    if track in TRACKS:
        return track
    for name in TRACKS:
        if stem.split("_")[1:2] == [name]:
            return name
    raise ValueError(f"cannot tell which track {stem} was trained on")


def _fresh_eval(meta: dict) -> dict | None:
    fresh = (meta.get("selection") or {}).get("fresh_eval") or meta.get("fresh_eval")
    if not isinstance(fresh, dict):
        return None
    keys = ("episodes", "lapped_fraction", "mean_lap", "best_lap")
    return {key: fresh.get(key) for key in keys if fresh.get(key) is not None} or None


def load_driver(path: Path, base_config: dict) -> tuple[dict, object, dict]:
    """(driver record, Q-function, resolved config) for a bundled weights file."""
    stem = path.stem
    agent = stem.split("_")[0]
    meta = _meta(path)
    params = np.load(path)["params"].astype(np.float64)
    if agent == "quantum":
        n_qubits = runtime.weights_circuit(path)["n_qubits"]
        profile_config = base_config if n_qubits == 4 else load_config(f"q{n_qubits}")
        config = runtime.with_weights_config(profile_config, path)
        circuit = {key: int(config["circuit"][key])
                   for key in ("n_qubits", "n_layers", "n_actions")}
        qfunc = QuantumQFunction(dict(config["circuit"]))
        qfunc.set_params(params)
        shape = {"circuit": circuit}
        n_actions = circuit["n_actions"]
    elif agent == "mlp":
        config = runtime.with_weights_observation(base_config, path)
        n_features = CarObserver(Track.load("oval"), {**config, "circuit": {}}).n_features
        n_actions = runtime.weights_actions(path) or 4
        hidden = (params.size - n_actions) // (n_features + 1 + n_actions)
        qfunc = MLPQFunction(n_features=n_features, hidden=hidden, n_actions=n_actions)
        qfunc.set_params(params)
        shape = {"hidden": int(hidden)}
    else:
        raise ValueError(f"unknown agent in {path.name}")
    obs_cfg = config["observation"]
    observation = {
        "ray_angles_deg": [float(a) for a in obs_cfg["ray_angles_deg"]],
        "ray_max_dist": float(obs_cfg["ray_max_dist"]),
        "features": [str(f) for f in obs_cfg.get("features", ["rays", "speed"])],
        "lookahead_m": float(obs_cfg.get("lookahead_m", 15.0)),
    }
    kind, stage = _driver_kind(stem)
    record = {
        "id": stem,
        "agent": agent,
        "track": _driver_track(stem, meta),
        "kind": kind,
        "stage": stage,
        "episodes": meta.get("episodes"),
        "n_params": int(params.size),
        "n_actions": int(n_actions),
        **shape,
        "observation": observation,
        "provenance": meta.get("provenance"),
        "fresh_eval": _fresh_eval(meta),
    }
    if agent == "quantum":
        config = {**config, "circuit": dict(config["circuit"])}
    return record, qfunc, config


def driver_files() -> list[Path]:
    return sorted(WEIGHTS_DIR.glob("*.npz"))


# ------------------------------------------------------------------ fixture

def _track_cases(track: Track, rng: np.random.Generator) -> dict:
    n = len(track.centerline)
    picks = sorted({0, 1, 2, n - 1, *rng.integers(0, n, 12).tolist()})
    # query points scattered around the centerline, some well off the track
    idx = rng.integers(0, n, 40)
    offsets = rng.uniform(-2.5, 2.5, 40) * track.half_width
    points = track.centerline[idx] + track.normals[idx] * offsets[:, None]
    points += rng.normal(0.0, 0.5, points.shape)
    s_vals, lateral = track.project(points)
    origins = track.centerline[idx[:30]] + track.normals[idx[:30]] * (
        rng.uniform(-0.9, 0.9, 30) * track.half_width)[:, None]
    angles = rng.uniform(-math.pi, math.pi, 30)
    dist = track.raycast(origins, angles, 30.0)
    s_query = rng.uniform(0.0, track.total_length, 20)
    return {
        "n_points": n,
        "total_length": float(track.total_length),
        "max_abs_curvature": float(track.max_abs_curvature),
        "start_pose": list(track.start_pose()),
        "samples": {
            "index": picks,
            "centerline": _floats(track.centerline[picks]),
            "normals": _floats(track.normals[picks]),
            "curvature": _floats(track.curvature[picks]),
            "s": _floats(track.s[picks]),
        },
        "project": {"points": _floats(points), "s": _floats(s_vals),
                    "lateral": _floats(lateral)},
        "raycast": {"origins": _floats(origins), "angles": _floats(angles),
                    "max_dist": 30.0, "dist": _floats(dist)},
        "curvature_ahead": {"s": _floats(s_query), "lookahead": 15.0,
                            "kappa": _floats(track.curvature_ahead(s_query, 15.0))},
        "tangent_angle": {"s": _floats(s_query), "angle": _floats(track.tangent_angle(s_query))},
    }


def _physics_cases(config: dict, rng: np.random.Generator) -> dict:
    car = CarPhysics(config["physics"])
    states = np.column_stack([rng.uniform(-50, 50, 24), rng.uniform(-50, 50, 24),
                              rng.uniform(-math.pi, math.pi, 24), rng.uniform(0, 26, 24)])
    table = np.asarray(action_set(8), dtype=np.float64)
    controls = table[rng.integers(0, len(table), 24)]
    out = car.step(states, controls[:, 0], controls[:, 1], controls[:, 2])
    return {"states": _floats(states), "controls": _floats(controls), "next": _floats(out)}


def _observer_cases(track: Track, config: dict, rng: np.random.Generator) -> dict:
    n = len(track.centerline)
    idx = rng.integers(0, n, 16)
    # off the resampled vertices: a car projecting exactly onto one sits on a
    # curvature-window boundary, where floor(s / ds) flips on the last bit
    along = rng.uniform(0.2, 0.8, 16) * (track.total_length / n)
    pos = (track.centerline[idx] + track.tangents[idx] * along[:, None]
           + track.normals[idx] * (rng.uniform(-1.1, 1.1, 16) * track.half_width)[:, None])
    heading = np.arctan2(track.tangents[idx, 1], track.tangents[idx, 0])
    states = np.column_stack([pos, heading + rng.normal(0.0, 0.4, 16),
                              rng.uniform(0.0, 25.0, 16)])
    observer = CarObserver(track, {**config, "circuit": {}})
    return {"states": _floats(states), "obs": _floats(observer.observe(states))}


def rollout(track: Track, config: dict, qfunc, max_decisions: int) -> dict:
    """Greedy closed-loop drive from the exact start pose at rest, the way the
    demo server drives an agent car: decide every ``substeps_per_decision``
    substeps (the first one at t = 0), project after every substep, stop at
    the first completed lap or the first off-track substep."""
    car = CarPhysics(config["physics"])
    observer = CarObserver(track, {**config, "circuit": {}})
    table = np.asarray(action_set(qfunc.n_actions), dtype=np.float64)
    x, y, theta = track.start_pose()
    state = np.array([x, y, theta, 0.0])
    s_prev = float(track.project(state[None, :2])[0][0])
    total = track.total_length
    progress, substep = 0.0, 0
    states, obs_log, q_log, actions = [], [], [], []
    outcome = {"result": "timeout", "lap_time": None, "substeps": None}
    for _ in range(max_decisions):
        obs = observer.observe(state[None])[0]
        q = qfunc.q_values(obs[None])[0]
        action = int(np.argmax(q))
        states.append(state.copy())
        obs_log.append(obs)
        q_log.append(q)
        actions.append(action)
        steer, throttle, brake = table[action]
        done = False
        for _ in range(car.substeps_per_decision):
            state = car.step(state[None], np.array([steer]), np.array([throttle]),
                             np.array([brake]))[0]
            substep += 1
            s_new, lateral = track.project(state[None, :2])
            delta = (float(s_new[0]) - s_prev + 0.5 * total) % total - 0.5 * total
            s_prev = float(s_new[0])
            progress += delta
            if progress >= total:
                outcome = {"result": "lap", "lap_time": substep * car.dt, "substeps": substep}
                done = True
                break
            if abs(float(lateral[0])) > track.half_width:
                outcome = {"result": "crash", "lap_time": None, "substeps": substep}
                done = True
                break
        if done:
            break
    return {"states": _floats(states), "obs": _floats(obs_log), "q": _floats(q_log),
            "actions": actions, **outcome}


def build_fixture(base_config: dict, drivers: dict) -> dict:
    rng = np.random.default_rng(2026_10_08)
    tracks = {name: Track.load(name, base_config["track"]["resample_spacing"])
              for name in TRACKS}
    pro_config = drivers["mlp_pro"][2]
    fixture: dict = {
        "tracks": {name: _track_cases(track, rng) for name, track in tracks.items()},
        "physics": _physics_cases(base_config, rng),
        "observer": {
            "config": {key: pro_config["observation"][key]
                       for key in ("ray_angles_deg", "ray_max_dist", "features", "lookahead_m")},
            "cases": {name: _observer_cases(track, pro_config, rng)
                      for name, track in tracks.items()},
        },
        "q_values": {},
        "rollouts": [],
    }
    for stem in (*CIRCUIT_CASES, *MLP_CASES):
        _, qfunc, _ = drivers[stem]
        obs = rng.uniform(0.0, 1.0, (8, qfunc.n_features))
        entry = {"obs": _floats(obs), "q": _floats(qfunc.q_values(obs))}
        if isinstance(qfunc, QuantumQFunction):
            entry["expectations"] = _floats(qfunc.all_expectations(obs))
        fixture["q_values"][stem] = entry
    for stem, track_name in ROLLOUTS:
        _, qfunc, config = drivers[stem]
        result = rollout(tracks[track_name], config, qfunc, MAX_ROLLOUT_DECISIONS)
        fixture["rollouts"].append({"driver": stem, "track": track_name, **result})
    return fixture


# ------------------------------------------------------------------- export

def export(out: Path) -> dict:
    """Write the data files under ``out``; returns a short summary."""
    base_config = load_config()
    data = out / "public" / "data"
    if data.exists():
        shutil.rmtree(data)
    drivers: dict[str, tuple] = {}
    for path in driver_files():
        drivers[path.stem] = load_driver(path, base_config)

    for stem, (record, qfunc, _) in drivers.items():
        payload = {key: record[key] for key in
                   ("id", "agent", "n_actions", "observation") if key in record}
        if "circuit" in record:
            payload["circuit"] = record["circuit"]
        if "hidden" in record:
            payload["hidden"] = record["hidden"]
        payload["params"] = _floats(qfunc.get_params())
        _write_json(data / "drivers" / f"{stem}.json", payload)

    for name in TRACKS:
        raw = json.loads((TRACKS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        _write_json(data / "tracks" / f"{name}.json", raw)

    ghosts = []
    for name in TRACKS:
        stem = f"quantum_{name}"
        _, qfunc, config = drivers[stem]
        track = Track.load(name, base_config["track"]["resample_spacing"])
        lap = rollout(track, config, qfunc, MAX_ROLLOUT_DECISIONS)
        if lap["result"] != "lap":
            continue
        ghost = {"track": name, "lap_time": lap["lap_time"], "kind": "quantum",
                 "driver": f"{stem} (greedy lap from a standing start)",
                 "points": [state[:3] for state in lap["states"]]}
        _write_json(data / "ghosts" / f"{name}.json", ghost)
        ghosts.append({key: ghost[key] for key in ("track", "lap_time", "kind", "driver")})

    manifest = {
        "version": DATA_VERSION,
        "physics": {key: int(value) if key == "substeps_per_decision" else float(value)
                    for key, value in base_config["physics"].items()},
        "resample_spacing": float(base_config["track"]["resample_spacing"]),
        "max_decisions": int(base_config["reward"]["max_decisions"]),
        "action_set": [list(a) for a in action_set(8)],
        "action_labels": list(action_labels(8)),
        "tracks": [{"id": name, "name": TRACK_NAMES[name]} for name in TRACKS],
        "drivers": [{key: value for key, value in record.items() if key != "observation"}
                    | {"features": record["observation"]["features"],
                       "n_rays": len(record["observation"]["ray_angles_deg"])}
                    for record, _, _ in drivers.values()],
        "ghosts": ghosts,
    }
    _write_json(data / "manifest.json", manifest, indent=1)

    fixture = build_fixture(base_config, drivers)
    _write_json(out / "tests" / "fixtures" / "parity.json", fixture)
    return {"drivers": len(drivers), "tracks": len(TRACKS), "ghosts": len(ghosts),
            "rollouts": [(r["driver"], r["track"], r["result"], r["lap_time"])
                         for r in fixture["rollouts"]]}


def _close(a, b, tol: float = 1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, int | float) and isinstance(b, int | float)
                and abs(a - b) <= tol * max(1.0, abs(a), abs(b)))
    return a == b


def check(out: Path) -> list[str]:
    """Paths whose committed content differs from a fresh export."""
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        export(fresh)
        fresh_files = {p.relative_to(fresh) for p in fresh.rglob("*.json")}
        committed = {p.relative_to(out) for p in (out / "public" / "data").rglob("*.json")}
        committed.add(Path("tests/fixtures/parity.json"))
        for rel in sorted(fresh_files | committed):
            a, b = fresh / rel, out / rel
            if not a.is_file() or not b.is_file():
                problems.append(f"{rel}: missing on one side")
                continue
            # files computed by simulation (fixture, ghost laps, their lap times
            # in the manifest) compare with a float tolerance
            if rel.name in ("parity.json", "manifest.json") or rel.parent.name == "ghosts":
                if not _close(json.loads(a.read_text()), json.loads(b.read_text())):
                    problems.append(f"{rel}: differs beyond tolerance")
            elif a.read_bytes() != b.read_bytes():
                problems.append(f"{rel}: differs")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "browser",
                        help="browser app directory (default: browser/)")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed export is up to date")
    args = parser.parse_args(argv)
    if args.check:
        problems = check(args.out)
        for line in problems:
            print(line, file=sys.stderr)
        print("browser export up to date" if not problems else
              f"{len(problems)} file(s) out of date — run python tools/export_browser.py")
        return 1 if problems else 0
    summary = export(args.out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
