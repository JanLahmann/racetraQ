"""Depth-aware weight resolution: a driver brings its own circuit depth.

``runtime.weights_circuit`` resolves the shape a quantum weights file needs —
from its sidecar's ``circuit`` block, else from its parameter count — and the
demo session adopts it per driver, like the observation and the action count
(attract, race, evolution, driver / track / qubit switches; training reverts
to the profile depth).  Everything here runs on synthetic weights in a tmp
weights dir; the hardware side is in test_session_hardware.py / test_hardware.py.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import racetraq.server.runtime as runtime_mod
import racetraq.server.session as session_mod
from racetraq.agents.quantum.circuit import circuit_spec
from racetraq.agents.quantum.qdqn import QuantumQFunction
from racetraq.config import load_config
from racetraq.records import _quantum_config
from racetraq.server import protocol as P
from racetraq.server.runtime import (
    WEIGHTS_DIR,
    load_agent,
    observation_note,
    weights_actions,
    weights_circuit,
    weights_observation,
    with_weights_circuit,
    with_weights_config,
)
from racetraq.server.session import DemoSession

BUNDLED = WEIGHTS_DIR  # the real bundle (read-only here), before any monkeypatch


def by_type(msgs, tag):
    return [m for m in msgs if m["type"] == tag]


def write_driver(directory, name, n_qubits, n_layers, *, n_actions=None, sidecar=None,
                 seed=0):
    """Synthetic quantum weights ``<directory>/<name>.npz`` of the given shape
    (plus ``sidecar`` as its .meta.json when not None); returns the params."""
    cfg = {"n_qubits": n_qubits, "n_layers": n_layers}
    if n_actions is not None:
        cfg["n_actions"] = n_actions
    qfunc = QuantumQFunction(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    params = qfunc.get_params() + rng.normal(0.0, 0.3, size=qfunc.n_params)
    np.savez(directory / f"{name}.npz", params=params)
    if sidecar is not None:
        (directory / f"{name}.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return params


def direct_q(params, n_qubits, n_layers, obs, n_actions=4):
    """Q-values of a QuantumQFunction built directly at the right depth."""
    qfunc = QuantumQFunction({"n_qubits": n_qubits, "n_layers": n_layers,
                              "n_actions": n_actions})
    qfunc.set_params(params)
    return qfunc.q_values(obs)


def expected_spec(profile, **circuit):
    """circuit_spec of ``profile`` with the given [circuit] keys replaced."""
    config = load_config(profile)
    config["circuit"].update(circuit)
    return circuit_spec(config)


# The depth of the SHIPPED q8 profile (5 blocks since October 2026, 4 before).
# Tests that reach the profile through the code under test (records, the qubit
# switch, the hardware tool) take their "other" depth relative to it.
Q8_DEPTH = int(load_config("q8")["circuit"]["n_layers"])


def q8_config(n_layers=4):
    """The 8-qubit profile at an EXPLICIT circuit depth.  These tests are about
    drivers whose depth differs from the profile's (and about the blind spots
    of 4 blocks at 8 qubits), so they build that profile themselves instead of
    relying on the depth the shipped q8.toml happens to have."""
    config = load_config("q8")
    config["circuit"]["n_layers"] = n_layers
    return config


@pytest.fixture()
def weights_dir(tmp_path, monkeypatch):
    """An empty weights dir every loader resolves against."""
    directory = tmp_path / "weights"
    directory.mkdir()
    monkeypatch.setattr(session_mod, "WEIGHTS_DIR", directory)
    monkeypatch.setattr(runtime_mod, "WEIGHTS_DIR", directory)
    return directory


OBS8 = np.random.default_rng(11).uniform(0.0, 1.0, size=(3, 8))
OBS4 = np.random.default_rng(12).uniform(0.0, 1.0, size=(3, 4))


# ------------------------------------------------------------------ resolver


def test_resolver_reads_the_sidecar_circuit_block(tmp_path):
    write_driver(tmp_path, "driver", 8, 5,
                 sidecar={"circuit": {"n_qubits": 8, "n_layers": 5, "n_actions": 4}})
    shape = {"n_qubits": 8, "n_layers": 5, "n_actions": 4}
    # the block names the size too: no caller size, no filename tag needed
    assert weights_circuit(tmp_path / "driver.npz") == shape
    assert weights_circuit(tmp_path / "driver.npz", 8) == shape
    # a block of another size wins over the caller's; with_weights_circuit refuses it
    assert weights_circuit(tmp_path / "driver.npz", 10)["n_qubits"] == 8
    with pytest.raises(ValueError, match="8-qubit circuit"):
        with_weights_circuit(load_config("q10"), tmp_path / "driver.npz")


def test_resolver_infers_the_depth_without_a_circuit_block(tmp_path):
    for layers in (1, 4, 5, 6):
        write_driver(tmp_path, f"quantum_x{layers}_q8", 8, layers)
        # qubit count from the caller, or implied by the _q<n> filename tag
        assert weights_circuit(tmp_path / f"quantum_x{layers}_q8.npz", 8)["n_layers"] == layers
        assert weights_circuit(tmp_path / f"quantum_x{layers}_q8.npz") == {
            "n_qubits": 8, "n_layers": layers, "n_actions": 4}
    write_driver(tmp_path, "quantum_plain", 4, 6)  # no tag: 4 qubits
    assert weights_circuit(tmp_path / "quantum_plain.npz") == {
        "n_qubits": 4, "n_layers": 6, "n_actions": 4}
    # ... or by the recorded observation width (7 rays + speed = 8 scalars)
    write_driver(tmp_path, "untagged", 8, 5, sidecar={"observation": {
        "ray_angles_deg": [-60, -40, -20, 0, 20, 40, 60], "features": ["rays", "speed"]}})
    assert weights_circuit(tmp_path / "untagged.npz") == {
        "n_qubits": 8, "n_layers": 5, "n_actions": 4}


def test_resolver_honours_the_recorded_action_count(tmp_path):
    # 6 actions, recorded the old way (actions block): P = 3*5*6 + 12
    write_driver(tmp_path, "quantum_a_q6", 6, 5, n_actions=6,
                 sidecar={"actions": {"n_actions": 6}})
    assert weights_circuit(tmp_path / "quantum_a_q6.npz", 6) == {
        "n_qubits": 6, "n_layers": 5, "n_actions": 6}
    # ... or only in the circuit block (weights_actions reads it from there too)
    write_driver(tmp_path, "quantum_b_q6", 6, 5, n_actions=6,
                 sidecar={"circuit": {"n_qubits": 6, "n_layers": 5, "n_actions": 6}})
    assert weights_actions(tmp_path / "quantum_b_q6.npz") == 6
    assert weights_circuit(tmp_path / "quantum_b_q6.npz", 6)["n_actions"] == 6
    # no recorded count: the caller's fallback, else min(4, n)
    write_driver(tmp_path, "quantum_c_q6", 6, 4, n_actions=6)
    assert weights_circuit(tmp_path / "quantum_c_q6.npz", 6, 6) == {
        "n_qubits": 6, "n_layers": 4, "n_actions": 6}
    with pytest.raises(ValueError, match="84 parameters"):
        weights_circuit(tmp_path / "quantum_c_q6.npz", 6)
    # more action readouts than qubits cannot be a circuit
    write_driver(tmp_path, "quantum_d", 4, 4, sidecar={"actions": {"n_actions": 6}})
    with pytest.raises(ValueError, match="6 action readouts"):
        weights_circuit(tmp_path / "quantum_d.npz", 4)


def test_resolver_trusts_the_parameter_count_over_a_wrong_block(tmp_path):
    # the sidecar says 5 blocks, the file holds 4: what loads is what counts
    write_driver(tmp_path, "quantum_w_q8", 8, 4,
                 sidecar={"circuit": {"n_qubits": 8, "n_layers": 5, "n_actions": 4}})
    assert weights_circuit(tmp_path / "quantum_w_q8.npz", 8)["n_layers"] == 4


def test_resolver_refuses_what_fits_no_depth(tmp_path):
    np.savez(tmp_path / "quantum_bad_q8.npz", params=np.zeros(105))
    with pytest.raises(ValueError, match="105 parameters.*fits no depth"):
        weights_circuit(tmp_path / "quantum_bad_q8.npz", 8)
    np.savez(tmp_path / "quantum_tiny.npz", params=np.zeros(8))  # head only: L = 0
    with pytest.raises(ValueError, match="fits no depth"):
        weights_circuit(tmp_path / "quantum_tiny.npz", 4)
    np.savez(tmp_path / "quantum_other.npz", other=np.zeros(56))  # no "params"
    with pytest.raises(ValueError, match="cannot read weights"):
        weights_circuit(tmp_path / "quantum_other.npz", 4)
    (tmp_path / "quantum_junk.npz").write_text("not an archive", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read weights"):
        weights_circuit(tmp_path / "quantum_junk.npz", 4)
    with pytest.raises(FileNotFoundError):
        weights_circuit(tmp_path / "quantum_missing.npz", 4)


def test_with_weights_circuit_overlays_a_copy(tmp_path):
    write_driver(tmp_path, "quantum_oval_q8", 8, 5)
    config = q8_config(4)
    resolved = with_weights_circuit(config, tmp_path / "quantum_oval_q8.npz")
    assert resolved["circuit"]["n_layers"] == 5 and resolved["circuit"]["n_actions"] == 4
    assert config["circuit"]["n_layers"] == 4 and "n_actions" not in config["circuit"]
    assert resolved["observation"] is config["observation"]  # only [circuit] is new


def test_every_bundled_driver_resolves_and_keeps_its_q_values():
    """Every bundled quantum file resolves to a loadable shape at the qubit
    count its name implies. Default behaviour is unchanged: a file at its
    profile's own depth (every file bundled before sidecars recorded one)
    yields exactly the Q-values of a Q-function built from the config."""
    files = sorted(BUNDLED.glob("quantum_*.npz"))
    assert files
    rng = np.random.default_rng(5)
    for path in files:
        tag = path.name.split(".")[0].rsplit("_q", 1)
        n = int(tag[1]) if len(tag) == 2 and tag[1].isdigit() else 4
        config = load_config() if n == 4 else load_config(f"q{n}")
        shape = weights_circuit(path, n)
        assert shape == weights_circuit(path) and shape["n_qubits"] == n, path.name
        params = np.load(path)["params"]
        resolved = with_weights_config(config, path)
        if weights_observation(path) is None:  # nothing recorded: the profile's own
            assert resolved["observation"] is config["observation"], path.name
        after = QuantumQFunction(resolved["circuit"])
        after.set_params(params)  # loads, whatever its depth
        assert (after.n_layers, after.n_actions) == (shape["n_layers"], shape["n_actions"])
        if (after.n_layers, after.n_actions) == (int(config["circuit"]["n_layers"]), 4):
            before = QuantumQFunction(config["circuit"])
            before.set_params(params)
            obs = rng.uniform(0.0, 1.0, size=(4, n))
            assert np.array_equal(before.q_values(obs), after.q_values(obs)), path.name


def test_load_agent_and_records_use_the_weights_depth(weights_dir):
    params = write_driver(weights_dir, "quantum_oval_q8", 8, 5)
    agent = load_agent("quantum", "oval", config=q8_config(4))
    assert agent.n_layers == 5
    assert np.array_equal(agent.q_values(OBS8), direct_q(params, 8, 5, OBS8))
    # records resolves the shipped q8 profile itself: a driver one block deeper
    deeper = Q8_DEPTH + 1
    write_driver(weights_dir, "quantum_gp_q8", 8, deeper)
    config = _quantum_config(8, weights_dir / "quantum_gp_q8.npz")
    assert config["circuit"]["n_layers"] == deeper
    # weights at the profile's own depth (no sidecar): the plain profile, untouched
    write_driver(weights_dir, "quantum_chicane_q8", 8, Q8_DEPTH)
    assert _quantum_config(8, weights_dir / "quantum_chicane_q8.npz") == load_config("q8")


def test_with_weights_config_overlays_observation_and_shape(tmp_path):
    recorded = {"ray_angles_deg": [-60.0, 0.0, 60.0],
                "features": ["rays", "speed", "curvature_ahead", "lateral_offset",
                             "heading_error", "corner_speed_ratio"]}
    write_driver(tmp_path, "quantum_gp_q8", 8, 5, sidecar={"observation": recorded})
    config = q8_config(4)
    resolved = with_weights_config(config, tmp_path / "quantum_gp_q8.npz")
    assert resolved["observation"] == {**config["observation"], **recorded}
    assert resolved["observation"]["ray_max_dist"] == config["observation"]["ray_max_dist"]
    assert resolved["circuit"]["n_layers"] == 5
    assert len(config["observation"]["ray_angles_deg"]) == 7  # the caller's: untouched
    assert observation_note(config, resolved) == (
        "observation from the weights' sidecar: 3 rays, features ['rays', 'speed', "
        "'curvature_ahead', 'lateral_offset', 'heading_error', 'corner_speed_ratio'] "
        "(profile: 7 rays, features ['rays', 'speed'])")
    # a sidecar recording the profile's own observation: nothing to report
    write_driver(tmp_path, "quantum_oval_q8", 8, 4, sidecar={"observation": {
        "ray_angles_deg": config["observation"]["ray_angles_deg"],
        "features": ["rays", "speed"]}})
    same = with_weights_config(config, tmp_path / "quantum_oval_q8.npz")
    assert observation_note(config, same) is None
    # the bundled engineered-feature driver is the case this exists for
    if weights_observation(BUNDLED / "quantum_gp_q10.npz"):
        q10 = load_config("q10")
        note = observation_note(q10, with_weights_config(q10, BUNDLED / "quantum_gp_q10.npz"))
        assert note is not None and "curvature_ahead" in note


def _hw_reliability_tool():
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "hw_reliability.py"
    spec = importlib.util.spec_from_file_location("racetraq_hw_reliability_depth", tool_path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_hw_reliability_tool_reports_a_sidecar_observation(tmp_path, capsys):
    tool = _hw_reliability_tool()
    write_driver(tmp_path, "quantum_feat", 4, 4, sidecar={"observation": {
        "ray_angles_deg": [-30.0, 30.0], "features": ["rays", "speed", "curvature_ahead"]}})
    write_driver(tmp_path, "quantum_plain", 4, 4)
    args = ["--track", "oval", "--no-device", "--shots", "256", "--rescale", "off",
            "--episodes", "2", "--max-decisions", "5"]
    assert tool.main(["--weights", str(tmp_path / "quantum_feat.npz"), *args]) == 0
    assert ("observation from the weights' sidecar: 2 rays, features "
            "['rays', 'speed', 'curvature_ahead']") in capsys.readouterr().err
    assert tool.main(["--weights", str(tmp_path / "quantum_plain.npz"), *args]) == 0
    assert "observation from" not in capsys.readouterr().err


def test_hw_reliability_tool_resolves_the_depth(tmp_path):
    tool = _hw_reliability_tool()

    # no sidecar: profile size, inferred depth (one block more than the profile's)
    write_driver(tmp_path, "quantum_oval_q8", 8, Q8_DEPTH + 1)
    config = tool.weights_config(tmp_path / "quantum_oval_q8.npz", "q8")
    assert config["circuit"] == {**load_config("q8")["circuit"], "n_layers": Q8_DEPTH + 1,
                                 "n_actions": 4}
    # a full sidecar still wins over the (4-qubit) default profile, size included
    features = ["rays", "speed", "curvature_ahead", "corner_speed_ratio"]
    write_driver(tmp_path, "driver", 6, 5, n_actions=6, sidecar={
        "circuit": {"n_qubits": 6, "n_layers": 5, "n_actions": 6},
        "observation": {"ray_angles_deg": [-60.0, 0.0, 60.0], "features": features},
        "actions": {"n_actions": 6}})
    config = tool.weights_config(tmp_path / "driver.npz", None)
    assert config["circuit"] == {**config["circuit"], "n_qubits": 6, "n_layers": 5,
                                 "n_actions": 6}
    assert config["observation"]["features"] == features
    np.savez(tmp_path / "quantum_bad_q8.npz", params=np.zeros(105))
    with pytest.raises(ValueError, match="fits no depth"):
        tool.weights_config(tmp_path / "quantum_bad_q8.npz", "q8")


# ------------------------------------------------- train_headless --init

FAST = ["reward.max_decisions=25", "training.eval_episodes=5", "training.eval_every=4"]


def test_train_headless_init_brings_its_depth_and_actions(tmp_path, capsys):
    from racetraq.train_headless import train

    init = tmp_path / "init" / "quantum_oval_q6.npz"
    init.parent.mkdir()
    write_driver(init.parent, "quantum_oval_q6", 6, 5, n_actions=6,
                 sidecar={"actions": {"n_actions": 6}})
    out = tmp_path / "out"

    def run(**kwargs):
        return train("quantum", "oval", episodes=4, seed=1, profile="q6",
                     out_dir=str(out), init=str(init), **{"overrides": FAST, **kwargs})

    # the q6 profile says 4 blocks, 4 actions; the init weights are 5 blocks, 6 actions
    run()
    printed = capsys.readouterr().out
    assert ("circuit shape from --init quantum_oval_q6.npz: 5 blocks, 6 actions "
            "(config: 4 blocks, 4 actions)") in printed
    assert "warm-started from" in printed
    meta = json.loads((out / "quantum_oval_q6.meta.json").read_text(encoding="utf-8"))
    assert meta["circuit"] == {"n_qubits": 6, "n_layers": 5, "n_actions": 6}
    assert meta["actions"] == {"n_actions": 6}
    assert np.load(out / "quantum_oval_q6.npz")["params"].shape == (3 * 5 * 6 + 12,)

    # asking explicitly for what the weights are: fine, and nothing to announce
    run(overrides=[*FAST, "circuit.n_layers=5"], actions=6)
    assert "circuit shape from --init" not in capsys.readouterr().out

    # asking explicitly for something else: a clear error, not a set_params crash
    with pytest.raises(ValueError, match=r"have 5 blocks, but circuit\.n_layers = 4 was"):
        run(overrides=[*FAST, "circuit.n_layers=4"])
    with pytest.raises(ValueError, match="use 6 actions, but 4 were requested"):
        run(actions=4)
    with pytest.raises(ValueError, match="use 6 actions, but 4 were requested"):
        run(overrides=[*FAST, "circuit.n_actions=4"])
    # weights whose sidecar names another qubit count than the profile's
    write_driver(tmp_path, "quantum_six", 6, 4, sidecar={
        "circuit": {"n_qubits": 6, "n_layers": 4, "n_actions": 4}})
    with pytest.raises(ValueError, match="use the matching --profile"):
        train("quantum", "oval", episodes=4, seed=1, profile="q8", out_dir=str(out),
              init=str(tmp_path / "quantum_six.npz"), overrides=FAST)


def test_train_headless_init_at_the_config_shape_is_unchanged(tmp_path, capsys):
    """An init file of the configured shape (every bundled warm start): no
    announcement, no change to the resolved config."""
    from racetraq.train_headless import train

    write_driver(tmp_path, "start", 4, 4)
    np.savez(tmp_path / "bad.npz", params=np.zeros(57))
    cold = train("quantum", "oval", episodes=4, seed=1, profile=None,
                 out_dir=str(tmp_path / "cold"), overrides=FAST)
    warm = train("quantum", "oval", episodes=4, seed=1, profile=None,
                 out_dir=str(tmp_path / "warm"), init=str(tmp_path / "start.npz"),
                 overrides=FAST)
    assert "circuit shape from --init" not in capsys.readouterr().out
    metas = [json.loads((tmp_path / d / "quantum_oval.meta.json").read_text(encoding="utf-8"))
             for d in ("cold", "warm")]
    assert metas[0]["config_hash"] == metas[1]["config_hash"]
    assert metas[1]["circuit"] == {"n_qubits": 4, "n_layers": 4, "n_actions": 4}
    assert cold["weights_path"] != warm["weights_path"]
    with pytest.raises(ValueError, match="57 parameters, which fits no depth"):
        train("quantum", "oval", episodes=4, seed=1, profile=None,
              out_dir=str(tmp_path / "bad"), init=str(tmp_path / "bad.npz"), overrides=FAST)


# --------------------------------------------------------- session adoption


@pytest.mark.parametrize("with_block", [True, False], ids=["sidecar", "inferred"])
def test_session_loads_and_drives_a_five_block_driver(tmp_path, weights_dir, with_block):
    sidecar = ({"circuit": {"n_qubits": 8, "n_layers": 5, "n_actions": 4}}
               if with_block else None)
    params = write_driver(weights_dir, "quantum_oval_q8", 8, 5, sidecar=sidecar)

    session = DemoSession(q8_config(4), ghosts_dir=tmp_path)
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")

    # the welcome describes the ACTIVE driver's circuit: 5 blocks, at which
    # every action sees every feature (this profile's 4 blocks leave blind spots)
    spec = session.welcome_payload()["circuit_spec"]
    assert spec == expected_spec("q8", n_layers=5)
    assert spec["n_layers"] == 5 and spec["n_params"]["total"] == 128
    assert len(spec["gates"]) == 4 * 5 * 8
    assert spec["min_layers_full_visibility"] == 5
    assert all(all(row) for row in spec["visibility"])
    blind = expected_spec("q8", n_layers=4)["visibility"]  # 4 blocks: blind
    assert not all(all(row) for row in blind)
    assert by_type(msgs, "welcome")[-1]["circuit_spec"] == spec  # re-broadcast

    assert [c.kind for c in session.cars] == ["quantum"]
    qfunc = session.cars[0].qfunc
    assert (qfunc.n_qubits, qfunc.n_layers, qfunc.n_actions) == (8, 5, 4)
    assert np.array_equal(qfunc.get_params(), params)
    assert np.array_equal(qfunc.q_values(OBS8), direct_q(params, 8, 5, OBS8))

    for _ in range(120):  # 2 s of sim time = 20 agent decisions
        session.tick()
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    quantum = by_type(msgs, "quantum")
    assert quantum and all(len(q["q_values"]) == 4 for q in quantum)
    assert all(len(q["expectations"]) == 8 for q in quantum)
    for msg in by_type(msgs, "state")[-1:] + quantum[-1:]:
        P.parse_server(msg)


def test_drivers_of_different_depth_switch_both_ways(tmp_path, weights_dir):
    # same qubit count, three drivers: 4 blocks (the profile's depth, no
    # sidecar), 5 blocks with a sidecar block, 6 blocks without one
    oval = write_driver(weights_dir, "quantum_oval_q8", 8, 4)
    chicane = write_driver(weights_dir, "quantum_chicane_q8", 8, 5, seed=1, sidecar={
        "circuit": {"n_qubits": 8, "n_layers": 5, "n_actions": 4}})
    gp = write_driver(weights_dir, "quantum_gp_q8", 8, 6, seed=2)
    drivers = {"oval": (oval, 4), "chicane": (chicane, 5), "gp": (gp, 6)}

    session = DemoSession(q8_config(4), ghosts_dir=tmp_path)
    assert not by_type(session.drain_outbox(), "error")
    assert session.welcome_payload()["circuit_spec"]["n_layers"] == 4

    def check(name):
        params, layers = drivers[name]
        msgs = session.drain_outbox()
        assert not by_type(msgs, "error")
        welcome = by_type(msgs, "welcome")[-1]
        assert welcome["circuit_spec"] == expected_spec("q8", n_layers=layers)
        assert welcome["circuit_spec"] == session.welcome_payload()["circuit_spec"]
        car = next(c for c in session.cars if c.kind == "quantum")
        assert car.qfunc.n_layers == layers
        assert np.array_equal(car.qfunc.q_values(OBS8), direct_q(params, 8, layers, OBS8))
        for _ in range(30):
            session.tick()
        msgs = session.drain_outbox()
        assert not by_type(msgs, "error") and by_type(msgs, "quantum")

    # driver switch (attract), 4 -> 5 -> 6 -> 4 -> 6 -> 5 -> auto (oval: 4)
    for name in ("chicane", "gp", "oval", "gp", "chicane"):
        session.handle_message(P.SetDriver(driver=name))
        check(name)
    session.handle_message(P.SetDriver(driver="auto"))
    check("oval")

    # track switch: the track's specialist brings its depth, and back
    session.handle_message(P.SetTrack(track="chicane"))
    check("chicane")
    session.handle_message(P.SetTrack(track="oval"))
    check("oval")

    # race: the opponent is the picked driver, at its depth
    session.handle_message(P.SetDriver(driver="gp"))
    session.drain_outbox()
    session.handle_message(P.Race(action="start", opponent="quantum"))
    assert session.mode == "race"
    assert session.welcome_payload()["circuit_spec"]["n_layers"] == 6
    car = next(c for c in session.cars if c.kind == "quantum")
    assert np.array_equal(car.qfunc.q_values(OBS8), direct_q(gp, 8, 6, OBS8))
    for _ in range(30):
        session.tick()
    assert not by_type(session.drain_outbox(), "error")


def test_training_uses_the_profile_depth(tmp_path, weights_dir):
    write_driver(weights_dir, "quantum_oval_q8", 8, 5)
    write_driver(weights_dir, "quantum_oval_warmstart_q8", 8, 5, seed=3)
    config = q8_config(4)  # the profile trains 4 blocks, the bundled driver has 5
    config["reward"] = dict(config["reward"], max_decisions=50)
    config["training"] = dict(config["training"], n_parallel_envs=2, replay_size=500,
                              batch_size=8)
    session = DemoSession(config, ghosts_dir=tmp_path)
    session.drain_outbox()
    assert session.welcome_payload()["circuit_spec"]["n_layers"] == 5
    try:
        # a fresh agent is the PROFILE's circuit, and the spec says so
        session.handle_message(P.Train(action="start", agent="quantum", episodes=100_000))
        job = session.jobs["quantum"]
        assert job.trainer.qfunc.n_layers == 4
        msgs = session.drain_outbox()
        assert not by_type(msgs, "error")
        assert by_type(msgs, "welcome")[-1]["circuit_spec"] == expected_spec("q8", n_layers=4)
        session.handle_message(P.Train(action="stop", agent="quantum"))
        job.thread.join(timeout=30.0)
        assert not job.thread.is_alive()

        # warm weights of another depth cannot seed it: cold start, with a reason
        session.handle_message(
            P.Train(action="start", agent="quantum", warm=True, episodes=100_000))
        job = session.jobs["quantum"]
        assert job.trainer.qfunc.n_layers == 4
        errors = by_type(session.drain_outbox(), "error")
        assert len(errors) == 1 and "5 blocks (profile: 4)" in errors[0]["message"]
        assert "from scratch" in errors[0]["message"]
        session.handle_message(P.Train(action="stop", agent="quantum"))
        job.thread.join(timeout=30.0)
        assert not job.thread.is_alive()

        # back to watching: the driver's depth again
        session.handle_message(P.SetMode(mode="attract"))
        msgs = session.drain_outbox()
        assert not by_type(msgs, "error")
        assert by_type(msgs, "welcome")[-1]["circuit_spec"]["n_layers"] == 5
        assert session.cars[0].qfunc.n_layers == 5
    finally:
        session.shutdown()


def test_qubit_switch_adopts_the_depth_at_the_new_size(tmp_path, weights_dir):
    write_driver(weights_dir, "quantum_oval", 4, 4, seed=4)
    # the switch loads the shipped q8 profile: a driver one block deeper than it
    depth = Q8_DEPTH + 1
    params = write_driver(weights_dir, "quantum_oval_q8", 8, depth)
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    default_welcome = session.welcome_payload()
    session.drain_outbox()

    session.handle_message(P.Qubits(n=8))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    welcome = by_type(msgs, "welcome")[-1]
    assert welcome["circuit_spec"] == expected_spec("q8", n_layers=depth)
    assert len(welcome["obs_labels"]) == 8
    car = session.cars[0]
    assert car.qfunc.n_layers == depth
    assert np.array_equal(car.qfunc.q_values(OBS8), direct_q(params, 8, depth, OBS8))
    for _ in range(30):
        session.tick()
    assert not by_type(session.drain_outbox(), "error")

    session.handle_message(P.Qubits(n=4))  # back: the default circuit, exactly
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert by_type(msgs, "welcome")[-1] == default_welcome
    assert session.cars[0].qfunc.n_layers == 4


def test_evolution_cars_each_drive_at_their_own_depth(tmp_path, weights_dir):
    depths = {1: 4, 2: 4, 3: 5, 4: 5}
    stages = {i: write_driver(weights_dir, f"quantum_oval_stage{i}", 4, depth, seed=i)
              for i, depth in depths.items()}
    best = write_driver(weights_dir, "quantum_oval", 4, 6, seed=9)
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.drain_outbox()
    session.handle_message(P.SetMode(mode="evolution"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error") and session.mode == "evolution"

    # stages 1-3 at their own depth; the last car is the shipped driver, which
    # is also the one the circuit spec describes
    assert [c.qfunc.n_layers for c in session.cars] == [4, 4, 5, 6]
    for i, car in enumerate(session.cars[:3], start=1):
        assert np.array_equal(car.qfunc.q_values(OBS4),
                              direct_q(stages[i], 4, depths[i], OBS4))
    assert np.array_equal(session.cars[3].qfunc.q_values(OBS4), direct_q(best, 4, 6, OBS4))
    assert session.welcome_payload()["circuit_spec"]["n_layers"] == 6
    for _ in range(60):
        session.tick()
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert {q["car_id"] for q in by_type(msgs, "quantum")} == {
        "stage1", "stage2", "stage3", "stage4"}

    # a stage file that fits no depth blocks the mode instead of crashing it
    session.handle_message(P.SetMode(mode="attract"))
    session.drain_outbox()
    np.savez(weights_dir / "quantum_oval_stage2.npz", params=np.zeros(57))
    session._agent_cache.clear()
    session.handle_message(P.SetMode(mode="evolution"))
    errors = by_type(session.drain_outbox(), "error")
    assert len(errors) == 1 and "quantum_oval_stage2.npz" in errors[0]["message"]
    assert "57 parameters" in errors[0]["message"]
    assert session.mode == "attract"


def test_parameter_count_that_fits_no_depth_is_refused_cleanly(tmp_path, weights_dir):
    np.savez(weights_dir / "quantum_oval_q8.npz", params=np.zeros(105))
    chicane = write_driver(weights_dir, "quantum_chicane_q8", 8, 5, seed=1)
    session = DemoSession(q8_config(4), ghosts_dir=tmp_path)
    msgs = session.drain_outbox()
    errors = by_type(msgs, "error")
    assert len(errors) == 1
    P.parse_server(errors[0])
    assert "quantum_oval_q8.npz" in errors[0]["message"]
    assert "105 parameters" in errors[0]["message"]
    assert "fits no depth" in errors[0]["message"]
    assert session.cars == [] and session.mode == "attract"
    # no driver loaded: the spec shows the profile's circuit
    assert session.welcome_payload()["circuit_spec"] == expected_spec("q8", n_layers=4)

    for switch in (P.SetMode(mode="race"), P.SetMode(mode="hardware")):
        session.handle_message(switch)
        assert session.mode == "attract"
        errors = by_type(session.drain_outbox(), "error")
        assert len(errors) == 1 and "105 parameters" in errors[0]["message"]
    session.handle_message(P.HardwareMsg(action="lap", backend="fake", max_decisions=1))
    statuses = by_type(session.drain_outbox(), "hardware_status")
    assert statuses[-1]["phase"] == "error" and "105 parameters" in statuses[-1]["message"]
    assert session.hw_job is None
    for _ in range(6):
        session.tick()
    states = by_type(session.drain_outbox(), "state")
    assert states and states[-1]["cars"] == []

    # a usable driver at the same size still loads, at ITS depth
    session.handle_message(P.SetDriver(driver="chicane"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert by_type(msgs, "welcome")[-1]["circuit_spec"]["n_layers"] == 5
    assert np.array_equal(session.cars[0].qfunc.q_values(OBS8),
                          direct_q(chicane, 8, 5, OBS8))


def test_weights_recorded_for_another_qubit_count_are_refused(tmp_path, weights_dir):
    # named for 8 qubits, but its sidecar says it is a 10-qubit circuit
    write_driver(weights_dir, "quantum_oval_q8", 10, 4, sidecar={
        "circuit": {"n_qubits": 10, "n_layers": 4, "n_actions": 4}})
    session = DemoSession(load_config("q8"), ghosts_dir=tmp_path)
    errors = by_type(session.drain_outbox(), "error")
    assert len(errors) == 1 and "10-qubit circuit" in errors[0]["message"]
    assert session.cars == []
    for _ in range(6):
        session.tick()


def test_reference_driver_pick_still_syncs_to_the_quantum_car(tmp_path, weights_dir):
    """Hero / pro only replace the ATTRACT car. In race, evolution and
    hardware mode the track's quantum specialist drives, so observation,
    action count, depth and the welcome follow it there — and revert to the
    profile when the reference car is back on its own."""
    recorded = {"ray_angles_deg": [-60.0, 0.0, 60.0],
                "features": ["rays", "speed", "curvature_ahead", "lateral_offset",
                             "heading_error", "corner_speed_ratio"]}
    sidecar = {"circuit": {"n_qubits": 8, "n_layers": 5, "n_actions": 6},
               "actions": {"n_actions": 6}, "observation": recorded}
    best = write_driver(weights_dir, "quantum_oval_q8", 8, 5, n_actions=6, sidecar=sidecar)
    write_driver(weights_dir, "quantum_oval_warmstart_q8", 8, 5, n_actions=6,
                 sidecar=sidecar, seed=1)
    profile = q8_config(4)
    profile_spec = circuit_spec(profile)
    profile_labels = [f"ray {a:+g}°" if a else "ray 0°"
                      for a in profile["observation"]["ray_angles_deg"]] + ["speed"]

    session = DemoSession(q8_config(4), ghosts_dir=tmp_path)
    # a stand-in controller: building the real racing line takes seconds and
    # is not what this test is about
    session._agent_cache[("hero", session.track_name, "")] = lambda state: (0.0, 0.5, 0.0)
    session.handle_message(P.SetDriver(driver="hero"))
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")

    def assert_profile():  # the hero car alone: nothing quantum to describe
        welcome = session.welcome_payload()
        assert [c.kind for c in session.cars] == ["hero"]
        assert welcome["circuit_spec"] == profile_spec
        assert welcome["obs_labels"] == profile_labels

    def assert_quantum(mode, n_quantum):
        msgs = session.drain_outbox()
        assert not by_type(msgs, "error") and session.mode == mode
        welcome = by_type(msgs, "welcome")[-1]  # re-broadcast on the switch
        assert welcome == session.welcome_payload()
        spec = welcome["circuit_spec"]
        assert (spec["n_layers"], spec["n_actions"]) == (5, 6)
        assert spec == expected_spec("q8", n_layers=5, n_actions=6)
        assert welcome["obs_labels"][:4] == ["ray -60°", "ray 0°", "ray +60°", "speed"]
        assert "curvature ahead" in welcome["obs_labels"] and len(welcome["obs_labels"]) == 8
        cars = [c for c in session.cars if c.kind == "quantum"]
        assert len(cars) == n_quantum
        car = cars[-1]  # the shipped driver
        assert (car.qfunc.n_layers, car.qfunc.n_actions) == (5, 6)
        assert np.array_equal(car.qfunc.q_values(OBS8), direct_q(best, 8, 5, OBS8, 6))
        obs = session._car_obs(car)
        assert obs.shape == (1, 8) and len(car.rays) == 3  # the recorded 3 rays
        return msgs

    assert_profile()

    session.handle_message(P.Race(action="start", opponent="quantum"))
    assert_quantum("race", 1)
    for _ in range(30):
        session.tick()
    msgs = session.drain_outbox()
    assert not by_type(msgs, "error")
    assert all(len(q["q_values"]) == 6 for q in by_type(msgs, "quantum"))

    session.handle_message(P.SetMode(mode="attract"))
    assert by_type(session.drain_outbox(), "welcome")  # reverted, and said so
    assert_profile()

    session.handle_message(P.SetMode(mode="evolution"))  # warm-start + shipped driver
    assert_quantum("evolution", 2)
    for _ in range(30):
        session.tick()
    assert not by_type(session.drain_outbox(), "error")

    session.handle_message(P.SetMode(mode="attract"))
    session.drain_outbox()
    assert_profile()

    session.handle_message(P.SetMode(mode="hardware"))  # idle: no backend involved
    msgs = assert_quantum("hardware", 1)
    assert by_type(msgs, "hardware_status")[-1]["phase"] == "idle"

    session.handle_message(P.SetMode(mode="attract"))
    session.drain_outbox()
    assert_profile()
    for _ in range(30):
        session.tick()
    assert not by_type(session.drain_outbox(), "error")
