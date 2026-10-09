"""Training studio (#29): a visitor configures a quantum driver — track,
circuit size, sensors, action set, warm start — trains it live within a time
limit, sees the run next to the multi-seed studies, races it, and files it
on a per-track booth leaderboard.

This module is the pure part (options, configs, study comparison, board);
:class:`racetraq.server.studio.StudioController` drives the session.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from racetraq.agents.base import ACTION_SIZES, action_labels

STUDIO_TRACKS = ("oval", "chicane", "gp", "combo")
STUDIO_QUBITS = (4, 6, 8, 10)
STATS_PATH = Path(__file__).resolve().parent / "data" / "studio_stats.json"
BOARD_MAX_ENTRIES = 20

# Sensor presets: one feature per qubit. "lidar" is what every bundled driver
# and every study used (n-1 rays over +/-60 degrees plus speed); "corner"
# swaps raw speed for the speed relative to the safe speed of the next corner
# (the engineered corner_speed_ratio feature) — no study covers it.
SENSOR_PRESETS = {
    "lidar": {
        "label": "Distance + speed",
        "blurb": "distance sensors (lidar rays fanned over ±60°) plus the car's "
                 "speed — what every bundled driver senses",
    },
    "corner": {
        "label": "Distance + corner warning",
        "blurb": "distance sensors (lidar rays fanned over ±60°) plus, instead of "
                 "the speed, how fast the car is compared with the safe speed for "
                 "the next corner (experimental)",
    },
}

ACTION_BLURBS = {
    4: "right, straight, left, brake",
    6: "adds braking while steering (experimental)",
    8: "adds a gentle steer at full throttle (experimental)",
}

TRACK_NOTES = {
    "oval": "easy — two long bends",
    "chicane": "easy — a quick left-right",
    "gp": "hard — a hairpin; training is unstable",
    "combo": "hard — hairpins and chicanes",
}


def ray_angles(n_rays: int) -> list[float]:
    """n rays fanned evenly over [-60, +60] degrees (one straight ahead when odd)."""
    if n_rays == 1:
        return [0.0]
    return [round(float(a), 6) for a in np.linspace(-60.0, 60.0, n_rays)]


def studio_observation(n_qubits: int, sensors: str, base_obs: dict) -> dict:
    """The [observation] table for ``n_qubits`` features of preset ``sensors``."""
    if sensors not in SENSOR_PRESETS:
        raise ValueError(f"unknown sensor preset '{sensors}'")
    obs = {k: (list(v) if isinstance(v, list) else v) for k, v in base_obs.items()}
    obs["ray_angles_deg"] = ray_angles(n_qubits - 1)
    obs["features"] = ["rays", "speed" if sensors == "lidar" else "corner_speed_ratio"]
    return obs


def option_problem(n_qubits: int, sensors: str, n_actions: int) -> str | None:
    """Why a studio combination is not allowed, or None."""
    if n_qubits not in STUDIO_QUBITS:
        return f"qubits must be one of {STUDIO_QUBITS}"
    if sensors not in SENSOR_PRESETS:
        return f"sensors must be one of {tuple(SENSOR_PRESETS)}"
    if n_actions not in ACTION_SIZES:
        return f"actions must be one of {ACTION_SIZES}"
    if n_actions > n_qubits:
        return f"{n_actions} actions need at least {n_actions} qubits (one readout each)"
    return None


def studio_config(profile_config: dict, n_qubits: int, sensors: str, n_actions: int) -> dict:
    """The session config a studio run trains and drives under: the q<n>
    profile with the chosen sensors and action set."""
    problem = option_problem(n_qubits, sensors, n_actions)
    if problem is not None:
        raise ValueError(problem)
    config = copy.deepcopy(profile_config)
    config["observation"] = studio_observation(n_qubits, sensors, config["observation"])
    config.setdefault("circuit", {})["n_actions"] = int(n_actions)
    config["circuit"]["n_qubits"] = int(n_qubits)
    return config


def is_studied(sensors: str, n_actions: int, warm: bool) -> bool:
    """Whether the multi-seed studies cover this kind of run (cold start,
    lidar sensors, 4 actions)."""
    return sensors == "lidar" and n_actions == 4 and not warm


# ------------------------------------------------------------------ stats


def load_stats(path: Path = STATS_PATH) -> dict:
    """studio_stats.json (tools/export_studio.py), or empty when missing."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cells": {}, "speed": {}}


def stats_key(track: str, n_qubits: int) -> str:
    return f"{track}_q{n_qubits}"


def estimate_seconds(stats: dict, track: str, n_qubits: int,
                     local_speed: dict | None = None) -> float | None:
    """Expected seconds to the study's median first lap at the live training
    speed of this circuit size — measured on this machine by an earlier
    studio run when there was one (``local_speed``), else the exporter's
    laptop speed — or None without data."""
    cell = stats.get("cells", {}).get(stats_key(track, n_qubits))
    s_per_episode = (local_speed or {}).get(str(n_qubits))
    if s_per_episode is None:
        speed = stats.get("speed", {}).get(str(n_qubits))
        s_per_episode = speed["s_per_episode"] if speed else None
    if not cell or s_per_episode is None or cell.get("first_lap") is None:
        return None
    return float(cell["first_lap"]["median"]) * float(s_per_episode)


SPEED_FILE = "studio_speed.json"  # per installation, next to the boards
SPEED_MIN_EPISODES = 20  # shorter runs say too little about the speed


def load_local_speed(board_dir: Path) -> dict:
    """Seconds per training episode per circuit size measured by this
    machine's own studio runs: {"4": 0.21, ...} (empty before the first run)."""
    try:
        data = json.loads((Path(board_dir) / SPEED_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): float(v) for k, v in data.items()
            if isinstance(v, (int, float)) and v > 0}


def record_local_speed(board_dir: Path, n_qubits: int, episodes: int, seconds: float) -> None:
    """Remember this machine's training speed for ``n_qubits`` (the latest run
    wins; the setup screen's estimates use it from then on)."""
    if episodes < SPEED_MIN_EPISODES or seconds <= 0:
        return
    speed = load_local_speed(board_dir)
    speed[str(int(n_qubits))] = round(seconds / episodes, 4)
    path = Path(board_dir) / SPEED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(speed, indent=1) + "\n", encoding="utf-8")


def catalog(stats: dict, warm_available: dict, time_limit_s: float,
            local_speed: dict | None = None) -> dict:
    """Everything the studio setup screen shows: tracks, sizes, sensors,
    action sets and, per (track, qubits), the study numbers and a time
    estimate (``estimate_here``: measured on this machine, not on the
    exporter's laptop). ``warm_available`` maps "<track>_q<n>" to bool."""
    cells = stats.get("cells", {})
    local_speed = local_speed or {}
    combos = {}
    for track in STUDIO_TRACKS:
        for n in STUDIO_QUBITS:
            key = stats_key(track, n)
            seconds = estimate_seconds(stats, track, n, local_speed)
            combos[key] = {
                "study": cells.get(key),
                "estimate_s": None if seconds is None else round(seconds, 1),
                "estimate_here": str(n) in local_speed,
                "fits": None if seconds is None else bool(seconds <= time_limit_s),
                "warm": bool(warm_available.get(key, False)),
            }
    return {
        "tracks": [{"id": t, "note": TRACK_NOTES[t]} for t in STUDIO_TRACKS],
        "qubits": list(STUDIO_QUBITS),
        "sensors": [{"id": k, **v} for k, v in SENSOR_PRESETS.items()],
        "actions": [{"n": n, "labels": list(action_labels(n)), "blurb": ACTION_BLURBS[n]}
                    for n in ACTION_SIZES],
        "combos": combos,
        "time_limit_s": float(time_limit_s),
        "speed_machine": stats.get("machine"),
    }


def compare(stats: dict, track: str, n_qubits: int, first_lap: int | None,
            best_eval: dict | None) -> dict | None:
    """The visitor's run next to the study runs of the same track and size:
    first-lap episode rank and the best test (eval) against the studies'.
    None when the studies have no such runs."""
    cell = stats.get("cells", {}).get(stats_key(track, n_qubits))
    if not cell:
        return None
    out: dict[str, Any] = {"study": cell}
    firsts = cell.get("first_laps") or []
    if first_lap is not None and firsts:
        out["first_lap_faster_than"] = sum(1 for f in firsts if f > first_lap)
        out["first_lap_runs"] = len(firsts)
    if best_eval and best_eval.get("mean_lap") is not None and cell.get("best_mean_laps"):
        laps = cell["best_mean_laps"]
        out["mean_lap_faster_than"] = sum(1 for m in laps if m > best_eval["mean_lap"])
        out["mean_lap_runs"] = len(laps)
    return out


# ------------------------------------------------------------------ board


def board_path(track: str, board_dir: Path) -> Path:
    return Path(board_dir) / f"studio_{track}.json"


def load_board(track: str, board_dir: Path) -> list[dict]:
    try:
        entries = json.loads(board_path(track, board_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return entries if isinstance(entries, list) else []


def save_board(track: str, entries: list[dict], board_dir: Path) -> None:
    path = board_path(track, board_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")


def rank_key(entry: dict) -> tuple:
    """Reliability first (share of test episodes that lapped), then pace."""
    share = entry["lapped"] / max(1, entry["eval_episodes"])
    mean_lap = entry["mean_lap"] if entry.get("mean_lap") is not None else float("inf")
    return (-share, mean_lap)


def add_entry(entries: list[dict], entry: dict) -> tuple[list[dict], int | None]:
    """Insert, rank, cap; returns (entries, 1-based rank or None if cut)."""
    entry = {**entry, "date": entry.get("date") or time.strftime("%Y-%m-%d")}
    ranked = sorted([*entries, entry], key=rank_key)[:BOARD_MAX_ENTRIES]
    rank = next((i + 1 for i, e in enumerate(ranked) if e is entry), None)
    return ranked, rank
