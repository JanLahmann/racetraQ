"""Bundled evolution-stage and warm-start snapshots: what their sidecars claim
must hold.

Evolution mode shows four cars "at different stages of training" and ends on
the bundled driver; the warm live-training demo starts from a pre-first-lap
checkpoint.  ``tools/make_stages.py`` regenerates each family by REPLAYING the
training run of the bundled driver (seed, episodes and recipe from the
driver's sidecar) and refuses to write unless the replay's best snapshot is
that driver, parameter for parameter.  These tests pin what a regeneration
must leave behind, so a bad one cannot land silently:

- the stages are snapshots of ONE run, in training order, each scoring
  strictly better than the one before in the trainer's own snapshot eval and
  no worse in the independent eval on the exact simulator (36 episodes —
  how evolution mode drives them);
- that run is the bundled driver's (same seed, length and recipe) and the
  last stage is the driver file itself, byte for byte;
- the warm-start checkpoint is a snapshot of the same run that laps in
  neither eval, taken before the run's breakthrough (the first snapshot that
  laps in at least half of an eval's episodes), and its sidecar says whether
  an earlier snapshot — or an exploring training car — had lapped already;
- every file records the circuit it needs and loads at that shape;
- the real session fields them: evolution mode drives the four stages (the
  last car being the shipped driver) and a warm training start begins from
  the checkpoint's parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from traqmania.config import load_config, resolve_training_cfg
from traqmania.server import protocol as P
from traqmania.server.runtime import weights_circuit
from traqmania.server.session import DemoSession

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "traqmania" / "weights"
N_STAGES = 4
# (track, filename tag) of every stage family that ships
FAMILIES = [("oval", ""), ("chicane", ""), ("gp", ""), ("combo", ""), ("oval", "_q6")]
EVAL_KEYS = ("eval_episodes", "eval_lapped", "eval_mean_lap", "eval_best_lap",
             "eval_mean_return", "eval_acting_noise")
RUN_BLOCKS = ("circuit", "observation", "actions", "training")

family = pytest.mark.parametrize("track,qtag", FAMILIES,
                                 ids=[f"{track}{qtag}" for track, qtag in FAMILIES])


def _meta(npz: Path) -> dict:
    meta_path = npz.with_suffix("").with_suffix(".meta.json")
    assert npz.is_file(), f"missing weights file {npz.name}"
    assert meta_path.is_file(), f"missing meta file {meta_path.name}"
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _driver(track: str, qtag: str) -> Path:
    return WEIGHTS_DIR / f"quantum_{track}{qtag}.npz"


def _stage(track: str, qtag: str, i: int) -> Path:
    return WEIGHTS_DIR / f"quantum_{track}_stage{i}{qtag}.npz"


def _warmstart(track: str, qtag: str) -> Path:
    return WEIGHTS_DIR / f"quantum_{track}_warmstart{qtag}.npz"


def _quality(meta: dict) -> tuple:
    """The trainer's snapshot score: episodes lapped, then mean lap, then return."""
    lap = meta["eval_mean_lap"]
    return (meta["eval_lapped"], -lap if lap is not None else -float("inf"),
            meta["eval_mean_return"])


def _exact_quality(meta: dict) -> tuple:
    """The independent exact-simulator eval: episodes lapped, then mean lap."""
    exact = meta["exact_eval"]
    lap = exact["mean_lap"]
    return (exact["lapped"], -lap if lap is not None else -float("inf"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@family
def test_stage_meta_fields_present(track, qtag):
    for i in range(1, N_STAGES + 1):
        meta = _meta(_stage(track, qtag, i))
        assert meta["agent"] == "quantum" and meta["track"] == track
        assert meta["stage"] == i
        for key in EVAL_KEYS:
            assert key in meta, f"stage{i} meta missing '{key}'"
        assert meta["eval_episodes"] >= 1
        assert 0 <= meta["eval_lapped"] <= meta["eval_episodes"]
        lap = meta["eval_mean_lap"]
        assert (lap is None) == (meta["eval_lapped"] == 0)
        assert lap is None or 0 < meta["eval_best_lap"] <= lap
        for block in RUN_BLOCKS:
            assert isinstance(meta.get(block), dict), f"stage{i} meta missing '{block}'"
        # the independent eval of the saved file on the exact simulator
        exact = meta["exact_eval"]
        assert exact["episodes"] >= 36 and 0 <= exact["lapped"] <= exact["episodes"]


@family
def test_stages_are_improving_snapshots_of_one_run(track, qtag):
    metas = [_meta(_stage(track, qtag, i)) for i in range(1, N_STAGES + 1)]
    episodes = [m["episodes"] for m in metas]
    assert episodes == sorted(episodes), f"stage episodes not increasing: {episodes}"
    assert len(set(episodes)) == N_STAGES, f"stage episodes not distinct: {episodes}"
    qualities = [_quality(m) for m in metas]
    assert qualities == sorted(qualities) and len(set(qualities)) == N_STAGES, (
        f"stage eval quality does not improve strictly across stages 1..4: {qualities}")
    exact = [_exact_quality(m) for m in metas]
    assert exact == sorted(exact), (
        f"a later stage drives worse on the exact simulator: {exact}")
    # one run: the same seed, length, best snapshot and recipe in every sidecar
    assert all(m["run"] == metas[0]["run"] for m in metas)
    assert all(m[block] == metas[0][block] for m in metas for block in RUN_BLOCKS)
    run = metas[0]["run"]
    assert episodes[-1] == run["best_episode"] <= run["episodes"]


@family
def test_the_run_is_the_bundled_drivers_and_ends_on_it(track, qtag):
    driver = _driver(track, qtag)
    driver_meta = _meta(driver)
    last = _stage(track, qtag, N_STAGES)
    meta = _meta(last)
    run = meta["run"]
    assert run["reproduces_driver"] is True
    assert run["driver"] == driver.name
    # the driver has not been replaced since the stages were made ...
    assert run["driver_sha256"] == _sha256(driver), (
        f"{driver.name} changed after its stages were generated — rerun "
        f"tools/make_stages.py --track {track}" + (f" --profile {qtag[1:]}" if qtag else ""))
    # ... and the last stage is that very file
    assert last.read_bytes() == driver.read_bytes()
    # same seed, length and recipe as the driver's sidecar records
    assert run["seed"] == driver_meta["training"]["seed"]
    if "selection" in driver_meta:
        assert run["seed"] == driver_meta["selection"]["chosen_seed"]
    assert run["episodes"] == driver_meta["episodes"]
    for block in RUN_BLOCKS:
        assert meta[block] == driver_meta[block], block
    assert meta["exact_eval"]["lapped"] > 0  # it laps on the exact simulator


@family
def test_warmstart_is_a_lapless_snapshot_of_the_same_run(track, qtag):
    meta = _meta(_warmstart(track, qtag))
    stages = [_meta(_stage(track, qtag, i)) for i in range(1, N_STAGES + 1)]
    assert meta["agent"] == "quantum" and meta["track"] == track
    assert meta["run"] == stages[0]["run"]
    assert all(meta[block] == stages[0][block] for block in RUN_BLOCKS)
    # neither eval of this snapshot saw a lap ...
    assert meta["eval_lapped"] == 0 and meta["eval_mean_lap"] is None
    assert meta["exact_eval"]["lapped"] == 0
    # ... and it precedes the run's breakthrough, which precedes the driver
    warm = meta["warmstart"]
    assert meta["episodes"] < warm["breakthrough_episode"] <= meta["run"]["best_episode"]
    assert warm["breakthrough_fraction"] == 0.5
    # the sidecar is honest about laps before the checkpoint
    earlier = warm["earlier_lapping_episodes"]
    assert all(episode < meta["episodes"] for episode in earlier)
    for stage in stages:  # a stage that lapped before the checkpoint must be listed
        if stage["episodes"] < meta["episodes"] and (
                stage["eval_lapped"] > 0 or stage["exact_eval"]["lapped"] > 0):
            assert stage["episodes"] in earlier
    # ... and about the training cars: the run's first clean lap (exploring,
    # under the recipe's acting noise) may have come before the checkpoint
    first_lap = warm["first_training_lap_episode"]
    assert first_lap is None or 1 <= first_lap <= meta["run"]["episodes"]
    pre_first_lap = not earlier and (first_lap is None or first_lap > meta["episodes"])
    assert ("a pre-first-lap checkpoint" in meta["provenance"]) == pre_first_lap
    if earlier:
        assert "NOT the run's first lap" in meta["provenance"]
    elif not pre_first_lap:
        assert "NOT strictly pre-first-lap" in meta["provenance"]
        assert f"episode {first_lap}" in meta["provenance"]


@family
def test_stage_and_warmstart_weights_load_at_the_recorded_shape(track, qtag):
    driver_shape = weights_circuit(_driver(track, qtag))
    files = [_stage(track, qtag, i) for i in range(1, N_STAGES + 1)]
    files.append(_warmstart(track, qtag))
    for path in files:
        shape = weights_circuit(path)
        assert shape == driver_shape == _meta(path)["circuit"], path.name
        params = np.load(path)["params"]
        assert params.shape == (3 * shape["n_layers"] * shape["n_qubits"]
                                + 2 * shape["n_actions"],)
        assert np.all(np.isfinite(params))
    # distinct snapshots, not one file under four names
    assert len({np.load(path)["params"].tobytes() for path in files}) == len(files)


@family
def test_session_evolution_and_warm_start_use_the_family(track, qtag, tmp_path):
    """Through the real session, as the demo loads them."""
    config = load_config(qtag[1:] or None)
    config["training"] = dict(config["training"], n_parallel_envs=2)  # a cheap warm job
    session = DemoSession(config, ghosts_dir=tmp_path)
    session.drain_outbox()
    if session.track_name != track:
        session.handle_message(P.SetTrack(track=track))
        assert not [m for m in session.drain_outbox() if m["type"] == "error"]

    # evolution mode: one car per stage, in stage order, the last one the driver
    stages = [_stage(track, qtag, i) for i in range(1, N_STAGES + 1)]
    metas = [_meta(path) for path in stages]
    session.handle_message(P.SetMode(mode="evolution"))
    msgs = session.drain_outbox()
    assert not [m for m in msgs if m["type"] == "error"] and session.mode == "evolution"
    assert [car.id for car in session.cars] == [f"stage{i}" for i in range(1, N_STAGES + 1)]
    for car, path, meta in zip(session.cars, stages, metas, strict=True):
        np.testing.assert_array_equal(car.qfunc.get_params(), np.load(path)["params"])
        assert car.qfunc.n_layers == meta["circuit"]["n_layers"]
    np.testing.assert_array_equal(session.cars[-1].qfunc.get_params(),
                                  np.load(_driver(track, qtag))["params"])
    labels = [car.label for car in session.cars]
    assert labels[-1] == f"best (of {metas[-1]['run']['episodes']} ep run)"
    # the earlier cars are labelled by their sidecar episode at every qubit count
    assert labels[:-1] == [f"ep {meta['episodes']}" for meta in metas[:-1]]
    for _ in range(120):  # 2 s: every car drives, nothing errors
        session.tick()
    assert not [m for m in session.drain_outbox() if m["type"] == "error"]
    assert all(np.hypot(*(car.state[:2] - session.track.start_pose()[:2])) > 1.0
               for car in session.cars)

    # the warm button: the quantum job starts from the checkpoint's parameters
    # (0 episodes: the trainer only evaluates them, so they stay what they were)
    warm = _warmstart(track, qtag)
    session.handle_message(P.Train(action="start", agent="quantum", warm=True, episodes=0))
    assert not [m for m in session.drain_outbox() if m["type"] == "error"]
    job = session.jobs["quantum"]
    job.thread.join(timeout=300.0)
    assert not job.thread.is_alive() and job.error is None
    np.testing.assert_array_equal(job.trainer.qfunc.get_params(), np.load(warm)["params"])
    assert job.trainer.qfunc.n_layers == _meta(warm)["circuit"]["n_layers"]
    assert job.trainer.cfg == resolve_training_cfg(config, track, warm=True, agent="quantum")
