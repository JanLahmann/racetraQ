"""Training studio (#29): options and configs, study comparison, booth board,
protocol messages, and short real runs through the demo session."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from racetraq import studio
from racetraq.config import load_config
from racetraq.env.racing_env import CarObserver
from racetraq.env.track import Track
from racetraq.server import protocol as P
from racetraq.server.session import DemoSession
from racetraq.server.studio import StudioController

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import export_studio  # noqa: E402

# ------------------------------------------------------------- options


@pytest.mark.parametrize("n", studio.STUDIO_QUBITS)
@pytest.mark.parametrize("sensors", list(studio.SENSOR_PRESETS))
def test_one_feature_per_qubit(n, sensors):
    profile = load_config() if n == 4 else load_config(f"q{n}")
    config = studio.studio_config(profile, n, sensors, 4)
    observer = CarObserver(Track.load("oval"), config)
    assert observer.n_features == n
    assert config["observation"]["ray_angles_deg"][0] == -60.0
    assert config["observation"]["ray_angles_deg"][-1] == 60.0


def test_lidar_preset_is_what_the_profiles_use():
    for n in studio.STUDIO_QUBITS:
        profile = load_config() if n == 4 else load_config(f"q{n}")
        obs = studio.studio_observation(n, "lidar", profile["observation"])
        assert obs["ray_angles_deg"] == profile["observation"]["ray_angles_deg"]
        assert obs["features"] == ["rays", "speed"]


def test_options_are_validated():
    assert studio.option_problem(4, "lidar", 6) is not None  # 6 readouts need 6 qubits
    assert studio.option_problem(6, "lidar", 6) is None
    assert studio.option_problem(5, "lidar", 4) is not None
    assert studio.option_problem(4, "sonar", 4) is not None
    with pytest.raises(ValueError):
        studio.studio_config(load_config(), 4, "lidar", 8)


def test_studio_config_leaves_the_profile_alone():
    profile = load_config("q6")
    before = repr(profile)
    config = studio.studio_config(profile, 6, "corner", 6)
    assert repr(profile) == before
    assert config["circuit"]["n_actions"] == 6
    assert config["observation"]["features"] == ["rays", "corner_speed_ratio"]


def test_only_cold_lidar_four_action_runs_are_studied():
    assert studio.is_studied("lidar", 4, False)
    assert not studio.is_studied("lidar", 4, True)
    assert not studio.is_studied("corner", 4, False)
    assert not studio.is_studied("lidar", 6, False)


# --------------------------------------------------------------- stats


def test_studio_stats_are_up_to_date():
    assert export_studio.main(["--check"]) == 0


def test_catalog_has_every_combination_and_honest_estimates():
    stats = studio.load_stats()
    cat = studio.catalog(stats, {"oval_q4": True}, 300)
    assert set(cat["combos"]) == {f"{t}_q{n}" for t in studio.STUDIO_TRACKS
                                  for n in studio.STUDIO_QUBITS}
    oval4 = cat["combos"]["oval_q4"]
    assert oval4["warm"] and oval4["study"]["runs"] >= 8 and oval4["fits"]
    assert cat["combos"]["oval_q10"]["fits"] is False  # ~9 min at the measured speed
    assert cat["combos"]["combo_q10"]["study"] is None  # nobody studied it


def test_compare_ranks_against_the_study_runs():
    stats = {"cells": {"oval_q4": {"first_laps": [200, 250, 300, 350],
                                   "best_mean_laps": [13.0, 14.0]}}}
    out = studio.compare(stats, "oval", 4, 260, {"mean_lap": 13.5})
    assert out["first_lap_faster_than"] == 2 and out["first_lap_runs"] == 4
    assert out["mean_lap_faster_than"] == 1 and out["mean_lap_runs"] == 2
    assert studio.compare(stats, "gp", 10, 100, None) is None


# --------------------------------------------------------------- board


def _entry(name, lapped, mean_lap):
    return {"name": name, "qubits": 4, "sensors": "lidar", "actions": 4, "warm": False,
            "episodes": 400, "first_lap": 200, "lapped": lapped, "eval_episodes": 12,
            "mean_lap": mean_lap, "best_lap": mean_lap, "seconds": 60.0}


def test_board_ranks_reliability_then_pace_and_round_trips(tmp_path):
    entries, rank = studio.add_entry([], _entry("a", 10, 12.5))
    entries, rank = studio.add_entry(entries, _entry("b", 12, 14.0))
    assert rank == 1 and [e["name"] for e in entries] == ["b", "a"]
    entries, rank = studio.add_entry(entries, _entry("c", 12, 13.0))
    assert rank == 1 and [e["name"] for e in entries] == ["c", "b", "a"]
    studio.save_board("oval", entries, tmp_path)
    assert studio.load_board("oval", tmp_path) == entries
    assert studio.load_board("gp", tmp_path) == []


def test_board_is_capped():
    entries = []
    for i in range(studio.BOARD_MAX_ENTRIES + 5):
        entries, _ = studio.add_entry(entries, _entry(f"p{i}", 12, 12.0 + i))
    assert len(entries) == studio.BOARD_MAX_ENTRIES
    _, rank = studio.add_entry(entries, _entry("slow", 1, 40.0))
    assert rank is None


# ------------------------------------------------------------ protocol


def test_protocol_round_trip():
    msg = P.parse_client({"type": "studio", "action": "start", "track": "chicane",
                          "qubits": 6, "sensors": "corner", "actions": 6, "warm": False})
    assert msg == P.Studio(action="start", track="chicane", qubits=6, sensors="corner",
                           actions=6, warm=False)
    assert P.serialize(P.Studio(action="stop")) == {
        "type": "studio", "action": "stop", "qubits": 4, "sensors": "lidar",
        "actions": 4, "warm": False}
    for bad in ({"action": "fly"}, {"action": "start", "sensors": "sonar"},
                {"action": "start", "qubits": "six"}, {"action": "start", "extra": 1}):
        with pytest.raises(P.ProtocolError):
            P.parse_client({"type": "studio", **bad})
    status = P.parse_server({"type": "studio", "phase": "done", "time_limit_s": 300,
                             "result": {"episodes": 10}, "rank": 2})
    assert status.phase == "done" and status.rank == 2
    assert "studio" in P.MODES


# ------------------------------------------------------- session runs


def _session(tmp_path, limit=2.0):
    config = load_config()
    config["studio"] = {"time_limit_s": limit}
    return DemoSession(config, ghosts_dir=tmp_path)


def _run_until_done(session, timeout=60.0):
    t0 = time.monotonic()
    statuses = []
    while time.monotonic() - t0 < timeout:
        session.tick()
        for msg in session.drain_outbox():
            if msg["type"] == "studio":
                statuses.append(msg)
            assert msg["type"] != "error", msg
        if session.studio.phase == "done":
            return statuses
        time.sleep(0.005)
    raise AssertionError("studio run did not finish")


def test_studio_run_trains_files_and_races(tmp_path):
    s = _session(tmp_path)
    s.handle_message(P.SetName(name="Ada"))
    s.handle_message(P.SetMode(mode="studio"))
    assert s.drain_outbox()[-1]["catalog"]["time_limit_s"] == 2.0
    s.handle_message(P.Studio(action="start", track="chicane", qubits=4))
    assert s.track_name == "chicane" and s.studio.phase == "training"
    statuses = _run_until_done(s)
    done = statuses[-1]
    assert done["phase"] == "done"
    result = done["result"]
    assert result["stop_reason"] == "time" and result["episodes"] > 0
    assert result["best_eval"]["eval_episodes"] > 0
    assert done["spec"]["n_params"] == 56 and done["spec"]["studied"]
    # a 2-second run rarely laps; when it does, it is on the board
    board = studio.load_board("chicane", tmp_path / "leaderboard")
    assert bool(board) == bool(result["best_eval"]["lapped_episodes"])

    s.handle_message(P.Studio(action="race"))
    assert [c.id for c in s.cars] == ["human", "studio"] and s.studio.phase == "race"
    for _ in range(30):
        s.tick()
    s.handle_message(P.Studio(action="result"))
    assert s.cars == [] and s.studio.phase == "done"


def test_experimental_run_uses_its_sensors_and_gives_the_profile_back(tmp_path):
    s = _session(tmp_path, limit=1.0)
    s.handle_message(P.Qubits(n=6))
    s.handle_message(P.Studio(action="start", track="oval", qubits=6, sensors="corner",
                              actions=6))
    assert s.config["observation"]["features"] == ["rays", "corner_speed_ratio"]
    assert s.config["circuit"]["n_actions"] == 6
    assert s.studio.spec["n_params"] == 3 * 4 * 6 + 2 * 6 and not s.studio.spec["studied"]
    done = _run_until_done(s)[-1]
    assert done["result"]["comparison"] is None  # nobody studied this

    s.handle_message(P.SetMode(mode="attract"))  # e.g. the idle timer
    assert s.config["observation"]["features"] == ["rays", "speed"]
    assert not s.studio.config_changed

    s.handle_message(P.Studio(action="watch"))  # back: the model drives under its own config
    assert s.mode == "studio" and s.config["circuit"]["n_actions"] == 6
    assert s.config["observation"]["features"] == ["rays", "corner_speed_ratio"]
    for _ in range(12):
        s.tick()


def test_studio_refuses_impossible_choices(tmp_path):
    s = _session(tmp_path)
    s.handle_message(P.Studio(action="start", track="oval", qubits=4, actions=6))
    s.handle_message(P.Studio(action="start", track="oval", qubits=4, sensors="corner",
                              warm=True))
    s.handle_message(P.Studio(action="race"))
    errors = [m["message"] for m in s.drain_outbox() if m["type"] == "error"]
    assert len(errors) == 3
    assert "6 actions need" in errors[0] and "warm start" in errors[1]
    assert s.studio.phase == "setup" and not s.jobs


def test_stop_ends_the_run_with_the_best_so_far(tmp_path):
    s = _session(tmp_path, limit=60.0)
    s.handle_message(P.Studio(action="start", track="oval", qubits=4))
    for _ in range(30):
        s.tick()
    s.handle_message(P.Studio(action="stop"))
    done = _run_until_done(s)[-1]
    assert done["result"]["stop_reason"] == "stopped"
    assert s.studio.model is not None


class _Trainer:
    def __init__(self):
        self.last_eval = None


class _Job:
    def __init__(self):
        self.trainer = _Trainer()


def test_converges_once_every_test_laps_and_nothing_improves(tmp_path):
    s = _session(tmp_path)
    ctl = StudioController(s)
    job = _Job()
    stops = []
    s.stop_training = lambda: stops.append(True)
    laps = [(6, 15.0), (12, 14.0)] + [(12, 14.5)] * 6
    for i, (lapped, mean) in enumerate(laps, start=1):
        job.trainer.last_eval = {"episode": 50 * i, "lapped_episodes": lapped,
                                 "eval_episodes": 12, "mean_lap": mean}
        ctl._watch_tests(job)
        ctl._watch_tests(job)  # the same test seen twice counts once
    assert ctl.best_test["mean_lap"] == 14.0
    assert ctl.stop_reason == "converged" and stops == [True]


def test_estimates_use_this_machines_measured_speed(tmp_path):
    """The setup screen's time estimate comes from the reference laptop until
    a studio run on this machine has measured its own training speed."""
    from racetraq import studio as st

    stats = {"cells": {"oval_q4": {"first_lap": {"median": 200}}},
             "speed": {"4": {"s_per_episode": 0.1}}}
    laptop = st.catalog(stats, {}, 300.0)["combos"]["oval_q4"]
    assert laptop["estimate_s"] == 20.0 and laptop["estimate_here"] is False
    st.record_local_speed(tmp_path, 4, episodes=10, seconds=5.0)  # too short to count
    assert st.load_local_speed(tmp_path) == {}
    st.record_local_speed(tmp_path, 4, episodes=400, seconds=120.0)
    here = st.catalog(stats, {}, 300.0, st.load_local_speed(tmp_path))["combos"]["oval_q4"]
    assert here["estimate_s"] == 60.0 and here["estimate_here"] is True
