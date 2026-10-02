"""Server-side glue: bundled-agent loading, TrackPayload building, and training
config resolution (per-track presets + warm-start recipes from default.toml —
``resolve_training_cfg``, defined in ``traqmania.config`` where headless
training shares it and re-exported here for the session)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from traqmania.agents.base import N_ACTIONS
from traqmania.agents.classical import MLPQFunction
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.config import load_config, resolve_training_cfg  # noqa: F401  (re-export)
from traqmania.env.track import TRACKS_DIR, Track

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
GHOSTS_DIR = Path(__file__).resolve().parent.parent / "data" / "ghosts"
LEADERBOARD_DIR = Path(__file__).resolve().parent.parent / "data" / "leaderboard"
N_EVOLUTION_STAGES = 4
LEADERBOARD_MAX_ENTRIES = 10


TRACK_ORDER = ("oval", "chicane", "gp", "combo")  # simple -> complex, UI order


def available_tracks() -> list[str]:
    """Bundled track names, simple-to-complex (unknown extras sorted last)."""
    found = {p.stem for p in TRACKS_DIR.glob("*.json")}
    return [n for n in TRACK_ORDER if n in found] + sorted(found - set(TRACK_ORDER))


def load_track(config: dict, name: str) -> Track:
    return Track.load(name, config["track"]["resample_spacing"])


def load_agent(kind: str, track_name: str, warm: bool = False, config: dict | None = None):
    """Build a Q-function of ``kind`` ('quantum' | 'mlp') with bundled weights loaded.

    ``warm=True`` (quantum only) loads ``quantum_<track>_warmstart.npz`` instead of
    the fully-trained weights.  Quantum weight filenames gain a ``_q<n>`` tag when
    ``config`` sets ``circuit.n_qubits`` to anything other than the default 4.
    """
    if config is None:
        config = load_config()
    if kind == "mlp":
        qfunc: Any = MLPQFunction(n_features=4, n_actions=N_ACTIONS)
        path = WEIGHTS_DIR / f"mlp_{track_name}.npz"
    elif kind == "quantum":
        # n-qubit filename rule: no tag at the default 4 qubits, _q<n> otherwise.
        n_qubits = int(config.get("circuit", {}).get("n_qubits", 4))
        qtag = "" if n_qubits == 4 else f"_q{n_qubits}"
        suffix = "_warmstart" if warm else ""
        path = WEIGHTS_DIR / f"quantum_{track_name}{suffix}{qtag}.npz"
        # the depth and action count are the file's own (see weights_circuit)
        qfunc = QuantumQFunction(with_weights_circuit(config, path)["circuit"])
    else:
        raise ValueError(f"unknown agent kind '{kind}' (expected 'quantum' or 'mlp')")

    qfunc.set_params(np.load(path)["params"])
    return qfunc


def _weights_meta(path: Path) -> dict:
    """The parsed ``.meta.json`` sidecar of a weights file ({} when it is
    missing, unreadable or not a JSON object)."""
    meta_path = Path(path).with_suffix("").with_suffix(".meta.json")
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def weights_observation(path: Path) -> dict | None:
    """The ``[observation]`` settings a weights file was trained with, from its
    ``.meta.json`` sidecar (``ray_angles_deg``/``features``/``lookahead_m``),
    or None when the sidecar is missing or records no observation.

    Loaders overlay this on the profile's observation so a driver trained on
    engineered features is fed the scalars it actually learned from — the
    circuit encodes one feature per qubit, so driving it under a different
    ray layout silently scrambles its inputs.
    """
    obs = _weights_meta(path).get("observation")
    return dict(obs) if isinstance(obs, dict) else None


def _recorded_count(table: Any, key: str) -> int | None:
    """``table[key]`` as a positive int, or None (no table, no key, garbage)."""
    if not isinstance(table, dict):
        return None
    try:
        value = int(table[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _meta_actions(meta: dict) -> int | None:
    """Recorded action count of a parsed sidecar: ``actions.n_actions``, else
    the ``circuit`` block's."""
    actions = meta.get("actions")
    if isinstance(actions, dict):
        try:
            return int(actions["n_actions"])
        except (KeyError, TypeError, ValueError):
            return None
    return _recorded_count(meta.get("circuit"), "n_actions")


def weights_actions(path: Path) -> int | None:
    """The action-set size a weights file was trained with, from its
    ``.meta.json`` sidecar (``actions.n_actions``, else ``circuit.n_actions``),
    or None when the sidecar is missing or records no action count
    (pre-scaled-readout weights: 4).

    Loaders adopt this per driver — a 6/8-action policy reads Q_a = <Z_a> off
    the first n_actions qubits and its parameter head has one w/b pair per
    action, so the count must match to load (and to drive) it.
    """
    return _meta_actions(_weights_meta(path))


_QTAG = re.compile(r"_q(\d+)$")


def _observation_width(obs: Any) -> int | None:
    """Scalars per observation of a RECORDED ``observation`` table (one per
    qubit), or None unless it records the feature list — and the ray layout,
    when ``rays`` is one of the features."""
    if not isinstance(obs, dict) or not isinstance(obs.get("features"), list):
        return None
    width = 0
    for kind in obs["features"]:
        if str(kind) == "rays":
            if not isinstance(obs.get("ray_angles_deg"), list):
                return None
            width += len(obs["ray_angles_deg"])
        else:
            width += 1
    return width or None


def weights_circuit(path: Path, n_qubits: int | None = None,
                    n_actions: int | None = None) -> dict:
    """The circuit shape ``{n_qubits, n_layers, n_actions}`` a QUANTUM weights
    file needs — what every loader builds its Q-function with, so a driver
    brings its own depth the way it brings its observation and action count.

    - ``n_qubits``: the sidecar's ``circuit.n_qubits``; else the caller's
      ``n_qubits`` (the size it loads the file at); else the recorded
      observation width; else the filename rule (``_q<n>`` tag, none = 4).
      A caller that needs one particular size compares the result with it.
    - ``n_actions``: the recorded count (:func:`weights_actions`); else the
      caller's ``n_actions``; else ``min(4, n_qubits)``.
    - ``n_layers``: the sidecar's ``circuit.n_layers`` when it fits the
      parameter count ``P = 3 * L * n + 2 * A`` (lam, theta, output head);
      else the L that count implies — sidecars written before the ``circuit``
      block existed record no depth.

    Raises ``ValueError`` (with a message for the user) when the file is not
    a readable weights archive, records more actions than qubits, or holds a
    parameter count that fits no depth ``L >= 1`` — the one case loaders
    still refuse. A missing file raises ``FileNotFoundError`` as usual.
    """
    path = Path(path)
    try:
        with np.load(path) as data:
            n_params = int(data["params"].size)
    except FileNotFoundError:
        raise
    except Exception as exc:  # truncated, not an npz, no "params" array
        raise ValueError(
            f"cannot read weights '{path.name}' ({type(exc).__name__}: {exc})") from exc
    meta = _weights_meta(path)
    block = meta.get("circuit")

    n = _recorded_count(block, "n_qubits")
    if n is None and n_qubits is not None:
        n = int(n_qubits)
    if n is None:
        n = _observation_width(meta.get("observation"))
    if n is None:
        tag = _QTAG.search(path.name.split(".")[0])
        n = int(tag.group(1)) if tag else 4
    a = _meta_actions(meta) or (int(n_actions) if n_actions is not None else min(4, n))
    if not 1 <= a <= n:
        raise ValueError(f"weights '{path.name}' need {a} action readouts on a "
                         f"{n}-qubit circuit (one qubit per action, at least one)")

    layers = _recorded_count(block, "n_layers")
    if layers is None or 3 * layers * n + 2 * a != n_params:
        layers, rest = divmod(n_params - 2 * a, 3 * n)
        if rest or layers < 1:
            raise ValueError(
                f"weights '{path.name}' hold {n_params} parameters, which fits no depth "
                f"of a {n}-qubit, {a}-action circuit ({3 * n} per block + {2 * a} for "
                "the output head) — retrain, or correct its .meta.json")
    return {"n_qubits": n, "n_layers": int(layers), "n_actions": a}


def with_weights_circuit(config: dict, path: Path) -> dict:
    """``config`` with the ``[circuit]`` depth and action count the quantum
    weights at ``path`` need (:func:`weights_circuit`; the config's own
    ``n_actions`` is the fallback for a file that records none). A shallow
    copy — only the circuit table is new; ``config`` itself is untouched.

    Raises ``ValueError`` when the weights fit no depth or belong to another
    qubit count than the config's: the observation that goes with a size
    comes from the profile, so the size is the caller's to pick.
    """
    circuit = dict(config.get("circuit", {}))
    n_qubits = int(circuit.get("n_qubits", 4))
    shape = weights_circuit(path, n_qubits, circuit.get("n_actions"))
    if shape["n_qubits"] != n_qubits:
        raise ValueError(
            f"weights '{Path(path).name}' belong to a {shape['n_qubits']}-qubit circuit, "
            f"but the config has [circuit] n_qubits = {n_qubits} — use the matching "
            "profile")
    circuit["n_layers"] = shape["n_layers"]
    circuit["n_actions"] = shape["n_actions"]
    return {**config, "circuit": circuit}


def with_weights_observation(config: dict, path: Path) -> dict:
    """``config`` with the ``[observation]`` the weights at ``path`` record
    (:func:`weights_observation`) overlaid on its own — a shallow copy with a
    new observation table, or ``config`` itself when the sidecar records
    none. Whether the result is a valid observation for the circuit is
    checked where the env is built (``CarObserver``)."""
    obs = weights_observation(Path(path))
    if not obs:
        return config
    return {**config, "observation": {**config["observation"], **obs}}


def with_weights_config(config: dict, path: Path) -> dict:
    """The config a quantum weights file drives under, for loaders outside the
    session (hardware lap / sprint, the noise tools): ``config`` with the
    file's recorded observation (:func:`with_weights_observation`) and the
    circuit depth and action count it needs (:func:`with_weights_circuit`) —
    the rule the demo server and ``records`` apply per driver. ``config``
    itself is untouched; raises ``ValueError`` like ``with_weights_circuit``."""
    return with_weights_circuit(with_weights_observation(config, path), path)


def observation_note(config: dict, resolved: dict) -> str | None:
    """One line for a command-line tool when ``resolved`` (``config`` after
    :func:`with_weights_config`) observes something else than ``config``: the
    weights' sidecar, not the profile, decides what the driver sees. None
    when both observe the same."""
    theirs, ours = resolved["observation"], config["observation"]
    if theirs == ours:
        return None

    def brief(obs: dict) -> str:
        return (f"{len(obs['ray_angles_deg'])} rays, features "
                f"{list(obs.get('features', ['rays', 'speed']))}")

    return (f"observation from the weights' sidecar: {brief(theirs)} "
            f"(profile: {brief(ours)})")


def _weights_label(path: Path, fallback: str) -> str:
    """Evolution-car label from a weights file's .meta.json ('ep N'), or ``fallback``."""
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            episodes = json.load(f)["episodes"]
        return f"ep {int(episodes)}"
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


def best_stage_label(path: Path) -> str:
    """"best" plus what it was trained with: the shipped driver is the best
    greedy snapshot OF an N-episode run (its .meta.json ``episodes``), not
    the params at a fixed episode — phrase the label accordingly."""
    ep = _weights_label(path, "")
    return f"best (of {ep[3:]} ep run)" if ep.startswith("ep ") else "best"


def evolution_stage_specs(track_name: str) -> list[tuple[str, Path]]:
    """(label, weights_path) for the 4 evolution-mode cars on ``track_name``.

    Prefers the bundled ``quantum_<track>_stage{1..4}.npz`` snapshots; when none
    exist, falls back to [warmstart, final] duplicated to 4 cars.
    """
    specs = [
        (_weights_label(path, f"stage {i}"), path)
        for i in range(1, N_EVOLUTION_STAGES + 1)
        if (path := WEIGHTS_DIR / f"quantum_{track_name}_stage{i}.npz").is_file()
    ]
    if specs:
        best = WEIGHTS_DIR / f"quantum_{track_name}.npz"
        if best.is_file():
            # the last car runs the shipped best-snapshot driver rather than
            # the last training snapshot (final-episode params drift)
            specs[-1] = (best_stage_label(best), best)
        return specs
    warm = WEIGHTS_DIR / f"quantum_{track_name}_warmstart.npz"
    final = WEIGHTS_DIR / f"quantum_{track_name}.npz"

    def pair_label(kind: str, path: Path) -> str:
        ep = _weights_label(path, "")
        return f"{kind} ({ep})" if ep else kind

    return [(pair_label("warm-start", warm), warm), (best_stage_label(final), final)]


def ghost_path(track_name: str, ghosts_dir: Path | None = None) -> Path:
    return (ghosts_dir if ghosts_dir is not None else GHOSTS_DIR) / f"{track_name}.json"


def load_ghost(track_name: str, ghosts_dir: Path | None = None) -> dict | None:
    """Best-lap ghost record for ``track_name``: {lap_time, kind, points} or None.

    Returns None when no ghost is stored or the file fails basic validation.
    """
    path = ghost_path(track_name, ghosts_dir)
    try:
        with path.open("r", encoding="utf-8") as f:
            ghost = json.load(f)
        lap_time = float(ghost["lap_time"])
        points = ghost["points"]
        if lap_time <= 0.0 or not isinstance(points, list) or len(points) < 2:
            return None
        return {
            "lap_time": lap_time,
            "kind": str(ghost.get("kind", "quantum")),
            "driver": str(ghost["driver"]) if ghost.get("driver") else None,
            "points": [[float(p[0]), float(p[1]), float(p[2])] for p in points],
        }
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def save_ghost(track_name: str, ghost: dict, ghosts_dir: Path | None = None) -> Path:
    """Persist a best-lap ghost record as ``<ghosts_dir>/<track>.json``."""
    path = ghost_path(track_name, ghosts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"track": track_name, **ghost}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------- leaderboards


def leaderboard_path(track_name: str, board_dir: Path | None = None) -> Path:
    return (board_dir if board_dir is not None else LEADERBOARD_DIR) / f"{track_name}.json"


def load_leaderboard(track_name: str, board_dir: Path | None = None) -> dict:
    """Per-track leaderboard: named human entries (ranked) plus per-kind AI
    reference laps (shown, never ranked). Empty board when nothing stored or
    the file fails basic validation."""
    empty = {"entries": [], "references": {}}
    path = leaderboard_path(track_name, board_dir)
    try:
        with path.open("r", encoding="utf-8") as f:
            board = json.load(f)
        entries = [
            {"name": str(e["name"])[:24], "lap_s": float(e["lap_s"]),
             "date": str(e.get("date", ""))}
            for e in board["entries"]
            if float(e["lap_s"]) > 0.0 and str(e["name"]).strip()
        ]
        references = {
            str(kind): {"driver": str(ref.get("driver", kind)),
                        "lap_s": float(ref["lap_s"])}
            for kind, ref in dict(board.get("references", {})).items()
            if float(ref["lap_s"]) > 0.0
        }
        entries.sort(key=lambda e: e["lap_s"])
        return {"entries": entries[:LEADERBOARD_MAX_ENTRIES], "references": references}
    except (OSError, ValueError, KeyError, TypeError):
        return empty


def save_leaderboard(track_name: str, board: dict,
                     board_dir: Path | None = None) -> Path:
    """Persist a leaderboard as ``<board_dir>/<track>.json``."""
    path = leaderboard_path(track_name, board_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"track": track_name, **board}) + "\n",
                    encoding="utf-8")
    return path


def _track_theme(name: str) -> dict:
    """Theme dict from the bundled track JSON (Track itself does not keep it)."""
    path = TRACKS_DIR / f"{name}.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return dict(json.load(f).get("theme", {}))
    return {}


def track_payload(track: Track) -> dict:
    """TrackPayload dict per the WS protocol (boundaries recomputed as
    centerline +/- normal * half_width, matching Track's internal ones)."""
    x, y, theta = track.start_pose()
    left = track.centerline + track.normals * track.half_width
    right = track.centerline - track.normals * track.half_width
    return {
        "name": track.name,
        "half_width": float(track.half_width),
        "total_length": float(track.total_length),
        "checkpoints": [float(c) for c in track.checkpoints],
        "theme": _track_theme(track.name),
        "start": {"x": x, "y": y, "theta": theta},
        "centerline": track.centerline.tolist(),
        "left": left.tolist(),
        "right": right.tolist(),
    }
