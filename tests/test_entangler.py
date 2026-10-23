"""``[circuit] entangler``: the CZ ring can be switched off (entanglement ablation).

``"none"`` drops every CZ gate and changes nothing else — same encoding,
variational sublayers, readout and parameter layout — in every place the
circuit is built or simulated: fastsim, adjoint gradients, the Qiskit circuit
and its spec, the light-cone analysis, the surrogate spectra, and the weights'
``.meta.json`` sidecar that loaders rebuild the circuit from. ``"cz_ring"``
(the default) must stay bit-for-bit what it was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from racetraq.agents.quantum import adjoint, lightcone, surrogate
from racetraq.agents.quantum.circuit import build_circuit, circuit_spec, split_parameters
from racetraq.agents.quantum.fastsim import (
    DEFAULT_ENTANGLER,
    ENTANGLERS,
    FastStatevectorSim,
    check_entangler,
    cz_ring_diagonal,
)
from racetraq.agents.quantum.qdqn import QuantumQFunction
from racetraq.config import load_config
from racetraq.server import runtime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import export_browser  # noqa: E402
import study  # noqa: E402

NONE = {"n_qubits": 4, "n_layers": 4, "entangler": "none"}


def _random_circuit(rng, n=4, layers=4, batch=5):
    s = rng.uniform(0.0, 1.0, size=(batch, n))
    lam = rng.uniform(0.0, 2.0 * np.pi, size=(layers, n))
    theta = rng.uniform(-np.pi, np.pi, size=(layers, n, 2))
    return s, lam, theta


def _perturbed(cfg, seed=0):
    """QuantumQFunction off its symmetric init (generic gradients everywhere)."""
    qf = QuantumQFunction(cfg, seed=7)
    rng = np.random.default_rng(seed)
    qf.set_params(qf.get_params() + rng.uniform(-0.7, 0.7, size=qf.n_params))
    return qf


# ------------------------------------------------------------------- config


def test_default_config_keeps_the_cz_ring():
    assert load_config()["circuit"]["entangler"] == DEFAULT_ENTANGLER == "cz_ring"
    assert ENTANGLERS == ("cz_ring", "none")
    for profile in ("q6", "q8", "q10"):
        assert load_config(profile)["circuit"]["entangler"] == "cz_ring"


@pytest.mark.parametrize("bad", ["cz", "None", "", None, 0, "CZ_RING"])
def test_unknown_entangler_is_a_clear_value_error(bad):
    with pytest.raises(ValueError, match=r"\[circuit\] entangler must be one of"):
        check_entangler(bad)
    with pytest.raises(ValueError, match="entangler"):
        QuantumQFunction({"n_qubits": 4, "entangler": bad})
    with pytest.raises(ValueError, match="entangler"):
        FastStatevectorSim(4, 4, entangler=bad)
    with pytest.raises(ValueError, match="entangler"):
        circuit_spec({"circuit": {"n_qubits": 4, "entangler": bad}})
    with pytest.raises(ValueError, match="entangler"):
        lightcone.live_gates(4, 4, entangler=bad)


# ------------------------------------------------- default path: unchanged


def test_default_path_is_bit_for_bit_unchanged():
    """No key, the explicit default and the bundled driver all give the same
    numbers; the ring diagonal is today's."""
    sim = FastStatevectorSim(4, 4)
    assert sim.entangler == "cz_ring"
    np.testing.assert_array_equal(sim.ring_diagonal, cz_ring_diagonal(4))

    params = np.load(ROOT / "racetraq" / "weights" / "quantum_oval.npz")["params"]
    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(16, 4))
    q = {}
    for name, cfg in (("implicit", {"n_qubits": 4, "n_layers": 4}),
                      ("explicit", {"n_qubits": 4, "n_layers": 4, "entangler": "cz_ring"}),
                      ("loaded", None)):
        qf = (runtime.load_agent("quantum", "oval") if cfg is None
              else QuantumQFunction(cfg))
        qf.set_params(params)
        assert qf.entangler == "cz_ring"
        q[name] = (qf.q_values(obs), qf.grad_selected(obs, np.arange(16) % 4, np.ones(16)))
    for name in ("explicit", "loaded"):
        np.testing.assert_array_equal(q[name][0], q["implicit"][0])
        np.testing.assert_array_equal(q[name][1], q["implicit"][1])
    # the bundled sidecars record no entangler: they are CZ-ring drivers
    assert runtime.weights_entangler(ROOT / "racetraq" / "weights" / "quantum_oval.npz") \
        == "cz_ring"


def test_none_changes_the_numbers_but_not_the_parameter_layout():
    ring, none = QuantumQFunction({"n_qubits": 4}), QuantumQFunction(NONE)
    assert none.n_params == ring.n_params == 56
    assert none.param_groups() == ring.param_groups()
    np.testing.assert_array_equal(none.get_params(), ring.get_params())  # same init stream
    assert FastStatevectorSim(4, 4, entangler="none").ring_diagonal is None
    s, lam, theta = _random_circuit(np.random.default_rng(5))
    e_ring = FastStatevectorSim(4, 4).forward(s, lam, theta)
    e_none = FastStatevectorSim(4, 4, entangler="none").forward(s, lam, theta)
    assert np.max(np.abs(e_ring - e_none)) > 1e-2


# ------------------------------------------------------------- simulation


def test_none_is_a_product_of_single_qubit_circuits():
    """Without CZ every qubit evolves alone: <Z_a> of the n-qubit "none"
    circuit equals the 1-qubit circuit run on feature a only."""
    rng = np.random.default_rng(17)
    n, layers = 5, 3
    s, lam, theta = _random_circuit(rng, n, layers)
    full = FastStatevectorSim(n, layers, entangler="none").forward(s, lam, theta)
    for a in range(n):
        alone = FastStatevectorSim(1, layers, entangler="none").forward(
            s[:, [a]], lam[:, [a]], theta[:, [a], :])
        np.testing.assert_allclose(full[:, a], alone[:, 0], atol=1e-12)


def test_fastsim_matches_qiskit_statevector_without_entangler():
    """fastsim "none" vs the Qiskit circuit build_circuit(..., "none") through
    the Aer exact path (QiskitQFunction), <= 1e-9 — mirrors
    test_fastsim_vs_qiskit for the default ring."""
    from racetraq.agents.quantum import make_qfunction

    qf_exact = make_qfunction("aer_statevector", {**NONE, "seed": 7})
    assert qf_exact.entangler == "none"
    rng = np.random.default_rng(11)
    sim = FastStatevectorSim(4, 4, entangler="none")
    for _ in range(3):
        s, lam, theta = _random_circuit(rng, batch=3)
        e_fast = sim.forward(s, lam, theta)
        qf_exact.set_params(np.concatenate([lam.ravel(), theta.ravel(), np.ones(4),
                                            np.zeros(4)]))
        assert np.max(np.abs(e_fast - qf_exact.q_values(s))) <= 1e-9


def test_qiskit_circuit_without_entangler_has_no_cz_and_the_same_parameters():
    from qiskit.quantum_info import Statevector

    ring, none = build_circuit(4, 4), build_circuit(4, 4, "none")
    assert none.count_ops().get("cz", 0) == 0
    assert ring.count_ops()["cz"] == 16
    assert {k: v for k, v in ring.count_ops().items() if k != "cz"} == dict(none.count_ops())
    x_ring, w_ring = split_parameters(ring)
    x_none, w_none = split_parameters(none)
    assert [p.name for p in x_none] == [p.name for p in x_ring]
    assert [p.name for p in w_none] == [p.name for p in w_ring]

    # and its statevector is fastsim's (direct Statevector, no Aer)
    rng = np.random.default_rng(29)
    s, lam, theta = _random_circuit(rng, batch=1)
    x = (lam * s[0]).ravel()
    bound = none.assign_parameters(dict(zip(x_none + w_none, [*x, *theta.ravel()],
                                            strict=True)))
    _, psi = FastStatevectorSim(4, 4, entangler="none").forward(s, lam, theta,
                                                                return_state=True)
    np.testing.assert_allclose(Statevector(bound).data, psi[0], atol=1e-12)


def test_adjoint_gradients_match_finite_differences_without_entangler():
    qf = _perturbed(NONE, seed=3)
    rng = np.random.default_rng(41)
    obs = rng.uniform(0.0, 1.0, size=(6, 4))
    actions = rng.integers(0, 4, size=6)
    upstream = rng.normal(size=6)
    analytic = qf.grad_selected(obs, actions, upstream)

    def loss(params):
        qf.set_params(params)
        return float(np.sum(upstream * qf.q_values(obs)[np.arange(6), actions]))

    base, eps = qf.get_params(), 1e-6
    numeric = np.array([(loss(base + eps * e) - loss(base - eps * e)) / (2 * eps)
                        for e in np.eye(base.size)])
    qf.set_params(base)
    assert np.linalg.norm(numeric[:48]) > 1e-3  # the circuit part has teeth
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
    # the ring-on gradients are different: the option reaches the backward pass
    ring = QuantumQFunction({"n_qubits": 4, "n_layers": 4})
    ring.set_params(base)
    assert np.max(np.abs(ring.grad_selected(obs, actions, upstream) - analytic)) > 1e-3


def test_adjoint_grad_without_entangler_matches_parameter_shift():
    """adjoint.grad(entangler="none") vs the exact parameter-shift rule
    (shift pi/2 per gate angle) on fastsim."""
    rng = np.random.default_rng(43)
    s, lam, theta = _random_circuit(rng, batch=4)
    obs_idx = rng.integers(0, 4, size=4)
    upstream = rng.normal(size=4)
    dlam, dtheta = adjoint.grad(s, lam, theta, obs_idx, upstream, entangler="none")
    sim = FastStatevectorSim(4, 4, entangler="none")

    def value(lam_, theta_):
        e = sim.forward(s, lam_, theta_)
        return float(np.dot(upstream, e[np.arange(4), obs_idx]))

    shift = np.pi / 2
    for idx in np.ndindex(theta.shape):
        plus, minus = theta.copy(), theta.copy()
        plus[idx] += shift
        minus[idx] -= shift
        assert dtheta[idx] == pytest.approx((value(lam, plus) - value(lam, minus)) / 2,
                                            abs=1e-10)
    # lam enters as angle lam * s per sample: shift the angle, not lam
    for layer, i in np.ndindex(lam.shape):
        total = 0.0
        for b in range(4):
            def single(delta, b=b, layer=layer, i=i):
                lam_b = lam.copy()
                lam_b[layer, i] += delta / s[b, i]
                e = sim.forward(s[[b]], lam_b, theta)
                return upstream[b] * e[0, obs_idx[b]]
            total += s[b, i] * (single(shift) - single(-shift)) / 2
        assert dlam[layer, i] == pytest.approx(total, abs=1e-9)


# -------------------------------------------------------------- light cone


@pytest.mark.parametrize("n,layers,actions", [(4, 4, None), (6, 4, 4), (8, 5, 6), (10, 6, 4),
                                              (3, 1, 3)])
def test_light_cone_without_entangler_sees_only_its_own_qubit(n, layers, actions):
    a_count = min(4, n) if actions is None else actions
    visible = lightcone.feature_visibility(n, layers, actions, entangler="none")
    np.testing.assert_array_equal(visible, np.eye(a_count, n, dtype=bool))
    for a in range(a_count):
        live = lightcone.readout_live_gates(n, layers, a, entangler="none")
        own = np.zeros((layers, n), dtype=bool)
        own[:, a] = True
        np.testing.assert_array_equal(live["enc"], own)
        np.testing.assert_array_equal(live["ry"], own)
        own_rz = own.copy()
        own_rz[-1, a] = False  # the last RZ before a Z readout commutes with it
        np.testing.assert_array_equal(live["rz"], own_rz)
        assert not live["cz"].any()
    with pytest.raises(ValueError, match="no circuit depth"):
        lightcone.min_layers_full_visibility(n, actions, entangler="none")
    assert lightcone.cz_gates_per_block(n, "none") == 0


def test_dead_parameters_without_entangler_have_exactly_zero_gradient():
    """6 qubits, 4 readouts, no ring: qubits 4 and 5 and the last RZs are dead —
    their adjoint gradient is 0 (to rounding) for random inputs; the live
    ones are not."""
    cfg = {"n_qubits": 6, "n_layers": 3, "n_actions": 4, "entangler": "none"}
    qf = _perturbed(cfg, seed=9)
    rng = np.random.default_rng(13)
    obs = rng.uniform(0.0, 1.0, size=(32, 6))
    mask = lightcone.live_parameter_mask(6, 3, 4, entangler="none")
    live = np.concatenate([mask["lam"].ravel(), mask["theta"].ravel()])
    assert int((~live).sum()) == 3 * 3 * 2 + 4  # 2 idle qubits x 9 params + 4 last RZ
    grads = np.stack([qf.grad_selected(obs, np.full(32, a), np.ones(32))[:live.size]
                      for a in range(4)])
    assert np.max(np.abs(grads[:, ~live])) < 1e-12
    assert np.all(np.max(np.abs(grads[:, live]), axis=0) > 1e-6)


def test_pruned_circuit_without_entangler_keeps_the_readouts():
    from qiskit.quantum_info import Statevector

    n, layers = 6, 2
    pruned = lightcone.pruned_circuit(n, layers, 4, entangler="none")
    assert pruned.circuit.count_ops().get("cz", 0) == 0
    assert set(np.unique(pruned.input_index % n)) == {0, 1, 2, 3}
    rng = np.random.default_rng(31)
    s, lam, theta = _random_circuit(rng, n, layers, batch=1)
    x, w = (lam * s[0]).ravel(), theta.ravel()
    params = split_parameters(pruned.circuit)
    bound = pruned.circuit.assign_parameters(dict(zip(
        params[0] + params[1], [*x[pruned.input_index], *w[pruned.weight_index]],
        strict=True)))
    probs = np.abs(Statevector(bound).data) ** 2
    exact = FastStatevectorSim(n, layers, entangler="none").forward(s, lam, theta)[0]
    for a in range(4):
        z = 1.0 - 2.0 * ((np.arange(1 << n) >> a) & 1)
        assert float(probs @ z) == pytest.approx(exact[a], abs=1e-12)


def test_blind_spot_warning_names_the_missing_entangler():
    names = ["ray -60", "ray 0", "ray +60", "speed"]
    assert lightcone.blind_spot_warning(names, 4) == []
    lines = lightcone.blind_spot_warning(names, 4, entangler="none")
    assert 'entangler = "none"' in lines[0]
    assert lines[1].strip() == "Right (Z_0) cannot see: ray 0, ray +60, speed"
    assert "no entangler" in lightcone.report(4, 4, entangler="none")
    assert "Live two-qubit gates: 0/0 CZ" in lightcone.report(4, 4, entangler="none")


def test_circuit_spec_without_entangler():
    ring = circuit_spec({"circuit": {"n_qubits": 4}})
    none = circuit_spec({"circuit": {"n_qubits": 4, "entangler": "none"}})
    assert ring["entangler"] == "cz_ring" and none["entangler"] == "none"
    assert not any(g["type"] == "cz" for g in none["gates"])
    assert [g for g in ring["gates"] if g["type"] != "cz"] == none["gates"]
    assert none["counts"] == {"ry_enc": 16, "ry": 16, "rz": 16, "cz": 0, "total": 48}
    assert ring["counts"]["total"] == 64
    assert none["n_params"] == ring["n_params"]
    assert none["visibility"] == np.eye(4, dtype=int).tolist()
    assert none["min_layers_full_visibility"] is None
    assert none["dead_gates"] == {"ry_enc": 0, "ry": 0, "rz": 4, "cz": 0, "total": 4}
    # a wider register: the idle qubits' gates are dead, and no CZ is counted
    wide = circuit_spec({"circuit": {"n_qubits": 6, "entangler": "none"}})
    assert wide["dead_gates"] == {"ry_enc": 8, "ry": 8, "rz": 12, "cz": 0, "total": 28}
    assert wide["dead_params"] == 28


def test_surrogate_spectrum_follows_the_entangler():
    lam = np.random.default_rng(2).uniform(1.0, 4.0, size=(4, 4))
    ring = surrogate.spectrum_size(lam, 0)
    none = surrogate.spectrum_size(lam, 0, entangler="none")
    assert none["live_gates"] == 4 < ring["live_gates"]
    assert none["per_feature"][1:] == [1, 1, 1]  # only feature 0 carries frequencies
    np.testing.assert_array_equal(surrogate.encoding_mask(4, 4, 2, "none"),
                                  np.eye(4, dtype=bool)[[2] * 4])
    # the white-box coefficients: exact Fourier series of the "none" readout
    theta = np.random.default_rng(4).uniform(-np.pi, np.pi, size=(4, 4, 2))
    gates, k, coef = surrogate.gate_coefficients(theta, 1, entangler="none")
    assert gates.tolist() == [[layer, 1] for layer in range(4)]
    s = np.random.default_rng(6).uniform(0.0, 1.0, size=(3, 4))
    phi = lam[gates[:, 0], gates[:, 1]][None, :] * s[:, [1]]  # (3, G)
    series = (np.exp(1j * phi @ k.T) @ coef).real
    exact = FastStatevectorSim(4, 4, entangler="none").forward(s, lam, theta)[:, 1]
    np.testing.assert_allclose(series, exact, atol=1e-10)
    # a fitted surrogate takes the model's entangler
    qf = _perturbed(NONE)
    fitted = surrogate.fit_surrogate(qf, np.random.default_rng(8).uniform(size=(200, 4)))
    assert fitted.entangler == "none"
    assert [int(m.sum()) for m in fitted.masks] == [4, 4, 4, 4]


# ------------------------------------------------------ weights sidecar


def test_sidecar_round_trip_without_entangler(tmp_path):
    """save_weights records entangler "none"; every loader rebuilds that circuit."""
    from racetraq.train_headless import save_weights

    config = load_config()
    config["circuit"]["entangler"] = "none"
    qf = _perturbed(config["circuit"], seed=5)
    path = save_weights(qf, "quantum", "oval", config, 1, out_dir=tmp_path)
    meta = json.loads(path.with_suffix("").with_suffix(".meta.json").read_text("utf-8"))
    assert meta["circuit"] == {"n_qubits": 4, "n_layers": 4, "n_actions": 4,
                               "entangler": "none"}
    assert runtime.weights_entangler(path) == "none"
    assert runtime.weights_circuit(path) == {"n_qubits": 4, "n_layers": 4, "n_actions": 4}

    obs = np.random.default_rng(1).uniform(0.0, 1.0, size=(8, 4))
    for resolved in (runtime.with_weights_circuit(load_config(), path),
                     runtime.with_weights_config(load_config(), path)):
        assert resolved["circuit"]["entangler"] == "none"
        loaded = QuantumQFunction(resolved["circuit"])
        loaded.set_params(np.load(path)["params"])
        np.testing.assert_array_equal(loaded.q_values(obs), qf.q_values(obs))
    # the profile's circuit table is not touched
    assert load_config()["circuit"]["entangler"] == "cz_ring"
    # loaded under the CZ ring the same params would drive differently
    wrong = QuantumQFunction(load_config()["circuit"])
    wrong.set_params(np.load(path)["params"])
    assert np.max(np.abs(wrong.q_values(obs) - qf.q_values(obs))) > 1e-3

    # a default driver records no entangler and resets an ablation config
    ring_cfg = load_config()
    ring_path = save_weights(QuantumQFunction(ring_cfg["circuit"]), "quantum", "chicane",
                             ring_cfg, 1, out_dir=tmp_path)
    ring_meta = json.loads(ring_path.with_suffix("").with_suffix(".meta.json").read_text())
    assert "entangler" not in ring_meta["circuit"]
    assert runtime.with_weights_circuit(config, ring_path)["circuit"]["entangler"] == "cz_ring"


def test_unknown_recorded_entangler_is_refused(tmp_path):
    np.savez(tmp_path / "quantum_oval.npz", params=QuantumQFunction({}).get_params())
    (tmp_path / "quantum_oval.meta.json").write_text(json.dumps(
        {"circuit": {"n_qubits": 4, "n_layers": 4, "n_actions": 4, "entangler": "cx"}}))
    with pytest.raises(ValueError, match="circuit.entangler = 'cx'"):
        runtime.weights_entangler(tmp_path / "quantum_oval.npz")
    with pytest.raises(ValueError, match="entangler"):
        runtime.with_weights_config(load_config(), tmp_path / "quantum_oval.npz")


def test_training_without_entangler_records_it_and_init_adopts_it(tmp_path, capsys):
    """A few seconds of training with --set circuit.entangler=none; the run's
    weights carry it, --init continues on it, and an explicit contradiction
    is refused."""
    from racetraq.train_headless import train

    fast = ["reward.max_decisions=25", "training.eval_episodes=4", "training.eval_every=4"]
    out = train("quantum", "oval", episodes=4, seed=1, profile=None,
                out_dir=str(tmp_path / "none"), overrides=[*fast, "circuit.entangler=none"])
    assert "entangler = \"none\"" in capsys.readouterr().out  # the light-cone warning
    weights = Path(out["weights_path"])
    assert runtime.weights_entangler(weights) == "none"

    out2 = train("quantum", "oval", episodes=4, seed=2, profile=None,
                 out_dir=str(tmp_path / "warm"), init=str(weights), overrides=fast)
    assert "circuit entangler from --init" in capsys.readouterr().out
    assert runtime.weights_entangler(Path(out2["weights_path"])) == "none"
    with pytest.raises(ValueError, match="trained with circuit.entangler"):
        train("quantum", "oval", episodes=4, seed=2, profile=None,
              out_dir=str(tmp_path / "bad"), init=str(weights),
              overrides=[*fast, "circuit.entangler=cz_ring"])


def test_study_weights_config_and_browser_export_honour_the_sidecar(tmp_path):
    from racetraq.train_headless import save_weights

    config = load_config()
    config["circuit"]["entangler"] = "none"
    path = save_weights(QuantumQFunction(config["circuit"]), "quantum", "oval", config, 1,
                        out_dir=tmp_path)
    spec = {"profile": None, "overrides": [], "actions": None}
    assert study.weights_config(spec, path)["circuit"]["entangler"] == "none"
    # a study variant can sweep it: the override parses to the string "none"
    assert study.new_config_keys({"profile": None,
                                  "overrides": ["circuit.entangler=none"]}) == []
    with pytest.raises(ValueError, match="browser edition only runs the CZ-ring circuit"):
        export_browser.load_driver(path, load_config())
