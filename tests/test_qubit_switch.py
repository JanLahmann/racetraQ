"""Live qubit-count switching: the {type:"qubits", n} message and its session
rebuild (packaged q{n} profile overlay, attract reset, welcome re-broadcast),
plus the obs_labels welcome field and strict protocol parsing."""

import shutil
import threading

import numpy as np
import pytest

from racetraq.agents.quantum.qdqn import QuantumQFunction
from racetraq.config import load_config
from racetraq.server import protocol as P
from racetraq.server import session as session_mod
from racetraq.server.runtime import WEIGHTS_DIR
from racetraq.server.session import DemoSession, quantum_weights_path


def make_session(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.drain_outbox()
    return session


def by_type(msgs, type_tag):
    return [m for m in msgs if m["type"] == type_tag]


# ------------------------------------------------------------------- protocol


def test_qubits_round_trip():
    msg = P.Qubits(n=6)
    wire = P.serialize(msg)
    assert wire == {"type": "qubits", "n": 6}
    assert P.parse_client(wire) == msg


@pytest.mark.parametrize("data", [
    {"type": "qubits"},                    # missing n
    {"type": "qubits", "n": "6"},          # wrong type
    {"type": "qubits", "n": True},         # bool is not an int
    {"type": "qubits", "n": 6.0},          # float is not an int
    {"type": "qubits", "n": 0},            # out of range
    {"type": "qubits", "n": -4},           # out of range
    {"type": "qubits", "n": 6, "x": 1},    # unknown field
], ids=lambda d: repr(d)[:60])
def test_qubits_parse_rejects_garbage(data):
    with pytest.raises(P.ProtocolError):
        P.parse_client(data)


def test_welcome_obs_labels_round_trip():
    msg = P.Welcome(mode="attract", track={"name": "oval"}, tracks=["oval"],
                    circuit_spec={"n_qubits": 4}, ui={"kiosk": False},
                    obs_labels=["ray -60°", "ray 0°", "ray +60°", "speed"])
    wire = P.serialize(msg)
    assert wire["obs_labels"] == ["ray -60°", "ray 0°", "ray +60°", "speed"]
    assert P.parse_server(wire) == msg
    # absent obs_labels stays off the wire (pre-switch welcome shape)
    wire = P.serialize(P.Welcome(mode="attract", track={}, tracks=[],
                                 circuit_spec={}, ui={}))
    assert "obs_labels" not in wire


# ------------------------------------------------------------- session switch


def test_switch_4_to_6_rebuilds_and_drives(tmp_path):
    session = make_session(tmp_path)
    session.handle_message(P.Qubits(n=6))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")

    welcome = by_type(msgs, "welcome")
    assert len(welcome) == 1
    P.parse_server(welcome[0])  # validates the extended wire shape
    assert welcome[0]["mode"] == "attract"
    assert welcome[0]["circuit_spec"]["n_qubits"] == 6
    assert welcome[0]["circuit_spec"]["n_params"]["total"] == 80
    labels = welcome[0]["obs_labels"]
    assert len(labels) == 6
    assert labels[-1] == "speed"
    assert "-60" in labels[0]

    # attract mode drives the bundled quantum_oval_q6.npz weights
    assert session.mode == "attract"
    assert [c.kind for c in session.cars] == ["quantum"]
    car = session.cars[0]
    assert car.qfunc.n_features == 6
    bundled = np.load(quantum_weights_path("oval", 6))["params"]
    assert np.array_equal(car.qfunc.get_params(), bundled)

    for _ in range(120):  # 2 s of sim time = 20 agent decisions
        session.tick()
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    states = by_type(msgs, "state")
    assert states
    P.parse_server(states[-1])
    xs = [c["x"] for m in states for c in m["cars"]]
    ys = [c["y"] for m in states for c in m["cars"]]
    assert (max(xs) - min(xs)) + (max(ys) - min(ys)) > 1.0  # the car moves
    rays = next(c["rays"] for m in reversed(states) for c in m["cars"] if "rays" in c)
    assert len(rays) == 5  # 6-feature obs: 5 rays + speed
    quantum = by_type(msgs, "quantum")
    assert quantum  # gauges show all 6 qubits; actions stay pinned to 4
    assert all(len(q["expectations"]) == 6 for q in quantum)
    assert all(len(q["q_values"]) == 4 for q in quantum)


def test_switch_back_to_4_is_bit_identical_default(tmp_path):
    switched = make_session(tmp_path)
    switched.handle_message(P.Qubits(n=6))
    switched.drain_outbox()
    switched.handle_message(P.Qubits(n=4))
    msgs = switched.drain_outbox()
    assert not by_type(msgs, "error")
    welcome = by_type(msgs, "welcome")[0]
    assert welcome["circuit_spec"]["n_qubits"] == 4
    assert welcome["circuit_spec"]["n_params"]["total"] == 56
    assert welcome["obs_labels"] == ["ray -60°", "ray 0°", "ray +60°", "speed"]

    # the rebuilt session matches a fresh default session exactly
    fresh = make_session(tmp_path)
    assert welcome == fresh.welcome_payload()
    bundled = np.load(WEIGHTS_DIR / "quantum_oval.npz")["params"]
    assert np.array_equal(switched.cars[0].qfunc.get_params(), bundled)
    for _ in range(60):
        switched.tick()
        fresh.tick()
    assert np.array_equal(switched.cars[0].state, fresh.cars[0].state)
    switched_q = by_type(switched.drain_outbox(), "quantum")
    fresh_q = by_type(fresh.drain_outbox(), "quantum")
    assert [q["q_values"] for q in switched_q] == [q["q_values"] for q in fresh_q]


def test_switch_to_untrained_8_degrades_gracefully(tmp_path, monkeypatch):
    # Resolve weights against a dir holding ONLY the 4-qubit oval training, so
    # the test keeps meaning "switch to an untrained size" even after q8
    # weights ship in the real bundle.
    weights = tmp_path / "weights"
    weights.mkdir()
    shutil.copy(WEIGHTS_DIR / "quantum_oval.npz", weights / "quantum_oval.npz")
    monkeypatch.setattr(session_mod, "WEIGHTS_DIR", weights)
    session = make_session(tmp_path)
    session.handle_message(P.Qubits(n=8))
    msgs = session.drain_outbox()

    # rebuilt at 8 qubits, but no bundled q8 weights: car-less attract + error
    errors = by_type(msgs, "error")
    assert errors and "quantum_oval_q8.npz" in errors[0]["message"]
    P.parse_server(errors[0])  # existing wire shape; protocol not extended
    welcome = by_type(msgs, "welcome")[0]
    assert welcome["circuit_spec"]["n_qubits"] == 8
    assert len(welcome["obs_labels"]) == 8
    assert session.mode == "attract"
    assert session.cars == []
    assert session.n_qubits == 8

    # the degraded session still ticks and broadcasts (car-less) states
    for _ in range(6):
        session.tick()
    states = by_type(session.drain_outbox(), "state")
    assert states and states[-1]["cars"] == []
    P.parse_server(states[-1])

    # weight-needing mode switches stay rejected via the existing path
    for switch in (P.SetMode(mode="race"), P.SetMode(mode="evolution"),
                   P.SetMode(mode="hardware")):
        session.handle_message(switch)
        assert session.mode == "attract"
        errors = by_type(session.drain_outbox(), "error")
        assert len(errors) == 1 and f"'{switch.mode}'" in errors[0]["message"]
    session.handle_message(P.Race(action="start", opponent="quantum"))
    assert session.mode == "attract"
    errors = by_type(session.drain_outbox(), "error")
    assert errors and "quantum_oval_q8.npz" in errors[0]["message"]


def test_switch_keeps_the_sections_the_server_was_started_with(tmp_path):
    # a q{n} profile is about [circuit] and [observation] only: the exhibition
    # kiosk must stay a kiosk, and a [hardware] overlay must survive the switch
    config = load_config("exhibition")
    config["hardware"]["prune_light_cone"] = False
    config["hardware"]["fake_name"] = "fake_fez"
    session = DemoSession(config, ghosts_dir=tmp_path)
    session.drain_outbox()
    ui = dict(config["ui"])
    assert ui["kiosk"] is True and ui["attract_idle_seconds"] == 20

    for n in (6, 4):
        session.handle_message(P.Qubits(n=n))
        msgs = session.drain_outbox()
        assert not by_type(msgs, "error")
        welcome = by_type(msgs, "welcome")[-1]
        assert welcome["circuit_spec"]["n_qubits"] == n
        assert welcome["ui"] == ui
        assert welcome["circuit_spec"]["pruned_on_hardware"] is False
        assert session.config["hardware"]["fake_name"] == "fake_fez"
        assert session.config["server"] == config["server"]
    # the size itself still comes from the packaged profile
    assert session.config["observation"]["ray_angles_deg"] == [-60.0, 0.0, 60.0]


def test_weights_of_another_depth_are_adopted_not_refused(tmp_path, monkeypatch):
    # a driver brings its own depth, like its observation and action count:
    # 4-block weights without a sidecar (like the files bundled so far) under
    # [circuit] n_layers = 5 load and drive as the 4-block circuit they are,
    # and the welcome's circuit_spec says so; the configured 5 blocks stay
    # what a fresh training run builds
    weights = tmp_path / "weights"
    weights.mkdir()
    shutil.copy(WEIGHTS_DIR / "quantum_oval.npz", weights / "quantum_oval.npz")
    params = QuantumQFunction({"n_qubits": 8, "n_layers": 4}, seed=3).get_params()
    np.savez(weights / "quantum_oval_q8.npz", params=params)
    monkeypatch.setattr(session_mod, "WEIGHTS_DIR", weights)
    config = load_config("q8")
    config["circuit"]["n_layers"] = 5
    session = DemoSession(config, ghosts_dir=tmp_path)
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    spec = session.welcome_payload()["circuit_spec"]
    assert spec["n_layers"] == 4 and spec["n_params"]["total"] == 104
    assert by_type(msgs, "welcome")[-1]["circuit_spec"] == spec  # re-broadcast
    assert len(session.cars) == 1 and session.cars[0].qfunc.n_layers == 4
    assert np.array_equal(session.cars[0].qfunc.get_params(), params)
    for switch in (P.SetMode(mode="race"), P.SetMode(mode="hardware"),
                   P.SetMode(mode="attract")):
        session.handle_message(switch)
        assert session.mode == switch.mode
        assert not by_type(session.drain_outbox(), "error")
    for _ in range(12):
        session.tick()
    assert not by_type(session.drain_outbox(), "error")
    session._sync_to_weights(use_weights=False)  # what a train start does
    assert session.welcome_payload()["circuit_spec"]["n_layers"] == 5
    session.handle_message(P.Qubits(n=4))  # the default size still drives
    assert not by_type(session.drain_outbox(), "error") and len(session.cars) == 1
    # (a parameter count that fits NO depth is still refused:
    # tests/test_depth_aware_weights.py)


@pytest.mark.parametrize("n", [3, 5, 12])
def test_invalid_qubit_count_leaves_state_unchanged(tmp_path, n):
    session = make_session(tmp_path)
    config, track, cars = session.config, session.track, session.cars
    session.handle_message(P.Qubits(n=n))
    msgs = session.drain_outbox()
    errors = by_type(msgs, "error")
    assert len(errors) == 1 and f"unknown qubit count {n}" in errors[0]["message"]
    assert not by_type(msgs, "welcome")
    assert session.mode == "attract"
    assert session.n_qubits == 4
    assert session.config is config and session.track is track
    assert session.cars is cars


def _alive_dummy_thread():
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    return thread, release


def test_switch_rejected_while_hardware_job_runs(tmp_path):
    session = make_session(tmp_path)
    job = session_mod.HardwareJob(kind="lap", stop_event=threading.Event())
    job.thread, release = _alive_dummy_thread()
    session.hw_job = job
    try:
        session.handle_message(P.Qubits(n=6))
        errors = by_type(session.drain_outbox(), "error")
        assert len(errors) == 1 and "hardware job" in errors[0]["message"]
        assert session.n_qubits == 4  # state unchanged
    finally:
        release.set()
        job.thread.join(timeout=5.0)
        session.hw_job = None


def test_switch_rejected_while_training_runs(tmp_path):
    session = make_session(tmp_path)
    job = session_mod.TrainingJob(agent="mlp", env=None, trainer=None,
                                  stop_event=threading.Event(), epsilon=1.0)
    job.thread, release = _alive_dummy_thread()
    session.jobs["mlp"] = job
    try:
        session.handle_message(P.Qubits(n=6))
        errors = by_type(session.drain_outbox(), "error")
        assert len(errors) == 1 and "training" in errors[0]["message"]
        assert session.n_qubits == 4  # state unchanged
    finally:
        release.set()
        job.thread.join(timeout=5.0)
        session.jobs.clear()


def test_qubit_options_only_list_trained_sizes(tmp_path):
    """The qubit picker offers only sizes with a driver for the current track
    (an untrained size would leave Watch without a car)."""
    from racetraq.server.session import QUBIT_SIZES

    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    welcome = session.welcome_payload()
    expected = [n for n in QUBIT_SIZES
                if quantum_weights_path(session.track_name, n).is_file()]
    assert welcome["qubit_options"] == expected
    assert 4 in expected
    P.parse_server(welcome)
    session.drain_outbox()
    session.handle_message(P.parse_client({"type": "set_track", "track": "combo"}))
    (track_msg,) = [m for m in session.drain_outbox() if m["type"] == "track"]
    assert track_msg["qubit_options"] == [
        n for n in QUBIT_SIZES if quantum_weights_path("combo", n).is_file()]
    P.parse_server(track_msg)
