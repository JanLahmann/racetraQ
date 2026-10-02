"""Light-cone analysis: the structural prediction must equal the numerics.

``lightcone`` derives, from the circuit structure alone, which features each
readout <Z_a> can see and which gates/parameters are dead. Here every claim is
checked against the simulators it never touches: fastsim (resample a feature,
does <Z_a> move?), adjoint gradients (is d<Z_a>/dparam zero for every input?)
and Aer (does the pruned circuit reproduce the full one exactly?).
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from traqmania.agents.quantum import adjoint, lightcone
from traqmania.agents.quantum.fastsim import (
    FastStatevectorSim,
    apply_ry,
    apply_rz,
    z_diagonals,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

SIZES = (4, 5, 6, 7, 8, 10)
DEPTHS = (2, 3, 4, 5, 6)
GRID = [(n, layers) for n in SIZES for layers in DEPTHS]
CASES = [(n, layers, a) for n, layers in GRID for a in (4, 6, 8) if a <= n]

ZERO = 1e-12  # "exactly zero" for a float64 statevector simulation
BATCH = 16  # feature-resampling batch
GRAD_BATCH = 4  # samples for the per-sample gradient maximum


def _generic_params(n: int, layers: int, rng: np.random.Generator):
    """Random generic (lam, theta): no special angles, so nothing cancels by accident."""
    return rng.uniform(1.0, 4.0, size=(layers, n)), rng.uniform(-3.0, 3.0, size=(layers, n, 2))


@functools.cache
def _numerics(n: int, layers: int):
    """Numerical ground truth for every readout qubit a < min(8, n).

    Returns ``(delta (R, n), glam (R, L, n), gtheta (R, L, n, 2))``:
    ``delta[a, j]`` the largest change of <Z_a> when feature j is resampled,
    ``glam`` / ``gtheta`` the largest |d<Z_a>/dparam| over a random batch.
    """
    rng = np.random.default_rng(1000 * n + layers)
    sim = FastStatevectorSim(n, layers)
    lam, theta = _generic_params(n, layers, rng)
    s = rng.uniform(0.0, 1.0, size=(BATCH, n))
    readouts = min(8, n)

    e0 = sim.forward(s, lam, theta)
    delta = np.zeros((readouts, n))
    for j in range(n):
        s2 = s.copy()
        s2[:, j] = rng.uniform(0.0, 1.0, size=BATCH)
        delta[:, j] = np.abs(sim.forward(s2, lam, theta) - e0)[:, :readouts].max(axis=0)

    glam = np.zeros((readouts, layers, n))
    gtheta = np.zeros((readouts, layers, n, 2))
    for a in range(readouts):
        for b in range(GRAD_BATCH):
            dlam, dtheta = adjoint.grad(s[b : b + 1], lam, theta, np.array([a]), np.array([1.0]))
            glam[a] = np.maximum(glam[a], np.abs(dlam))
            gtheta[a] = np.maximum(gtheta[a], np.abs(dtheta))
    return delta, glam, gtheta


def _ring_distance(n: int) -> np.ndarray:
    """(n, n) ring distance between qubits."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return np.minimum(d, n - d)


# ------------------------------------------------- structure == numerics


@pytest.mark.parametrize(("n", "layers", "n_actions"), CASES)
def test_feature_visibility_matches_numerics(n, layers, n_actions):
    delta, _, _ = _numerics(n, layers)
    visible = lightcone.feature_visibility(n, layers, n_actions)
    assert visible.shape == (n_actions, n)
    assert visible.dtype == bool
    np.testing.assert_array_equal(visible, delta[:n_actions] > ZERO)


@pytest.mark.parametrize(("n", "layers", "n_actions"), CASES)
def test_dead_parameter_mask_matches_gradients(n, layers, n_actions):
    """Dead = max |d<Z_a>/dparam| over a random batch and all readouts < 1e-12."""
    _, glam, gtheta = _numerics(n, layers)
    mask = lightcone.live_parameter_mask(n, layers, n_actions)
    assert mask["lam"].shape == (layers, n)
    assert mask["theta"].shape == (layers, n, 2)
    np.testing.assert_array_equal(mask["lam"], glam[:n_actions].max(axis=0) >= ZERO)
    np.testing.assert_array_equal(mask["theta"], gtheta[:n_actions].max(axis=0) >= ZERO)


@pytest.mark.parametrize(("n", "layers"), GRID)
def test_single_readout_live_gates_match_gradients(n, layers):
    """The per-readout cones (not just their union) match, with a wide margin."""
    _, glam, gtheta = _numerics(n, layers)
    for a in range(min(8, n)):
        live = lightcone.readout_live_gates(n, layers, a)
        predicted = np.stack([live["ry"], live["rz"]], axis=-1)
        np.testing.assert_array_equal(live["enc"], glam[a] >= ZERO)
        np.testing.assert_array_equal(predicted, gtheta[a] >= ZERO)
        # generic parameters saturate the bound: live gradients are far from zero
        assert glam[a][live["enc"]].min() > 1e-6
        assert gtheta[a][predicted].min() > 1e-6


@pytest.mark.parametrize(("n", "layers", "n_actions"), CASES)
def test_live_gates_is_the_union_over_readouts(n, layers, n_actions):
    live = lightcone.live_gates(n, layers, n_actions)
    assert set(live) == {"enc", "ry", "rz", "cz"}
    for kind, mask in live.items():
        assert mask.shape == (layers, n)
        assert mask.dtype == bool
        union = np.any(
            [lightcone.readout_live_gates(n, layers, a)[kind] for a in range(n_actions)], axis=0
        )
        np.testing.assert_array_equal(mask, union)
    params = lightcone.live_parameter_mask(n, layers, n_actions)
    np.testing.assert_array_equal(params["lam"], live["enc"])
    np.testing.assert_array_equal(params["theta"][..., 0], live["ry"])
    np.testing.assert_array_equal(params["theta"][..., 1], live["rz"])
    # the encoding RY and the variational RY sit back to back: live or dead together
    np.testing.assert_array_equal(live["enc"], live["ry"])


@pytest.mark.parametrize(("n", "layers", "n_actions"), CASES)
def test_visibility_is_ring_distance_below_depth(n, layers, n_actions):
    """<Z_a> sees exactly the features within ring distance L - 1 of qubit a."""
    expected = _ring_distance(n)[:n_actions] <= layers - 1
    np.testing.assert_array_equal(lightcone.feature_visibility(n, layers, n_actions), expected)


def _forward_without_cz(n, layers, s, lam, theta, dropped=frozenset()):
    """<Z_q> (B, n) of the canonical circuit minus the CZs ``dropped`` = {(block, i)}.

    (block, i) names CZ(i, (i+1) % n) of that block, as in ``live_gates()["cz"]``.
    """
    idx = np.arange(1 << n)
    psi = np.zeros((s.shape[0], 1 << n), dtype=np.complex128)
    psi[:, 0] = 1.0
    for layer in range(layers):
        for i in range(n):
            apply_ry(psi, i, lam[layer, i] * s[:, i], n)
        for i in range(n):
            apply_ry(psi, i, theta[layer, i, 0], n)
            apply_rz(psi, i, theta[layer, i, 1], n)
        for i in range(n):
            if (layer, i) not in dropped:
                psi *= 1.0 - 2.0 * ((idx >> i) & 1) * ((idx >> ((i + 1) % n)) & 1)
    return (psi.real**2 + psi.imag**2) @ z_diagonals(n).T


@pytest.mark.parametrize(("n", "layers"), [(3, 2), (3, 4), (4, 4), (5, 3), (6, 4), (7, 5), (8, 4),
                                           (10, 4)])
def test_cz_live_mask_matches_numerics(n, layers):
    """A CZ is live for <Z_a> iff deleting that single gate moves <Z_a> — both ways.

    (n = 2 is excluded: its two CZs are one and the same gate applied twice,
    so only the pair can be removed; see test_two_qubit_ring_is_the_identity.)
    """
    rng = np.random.default_rng(500 * n + layers)
    lam, theta = _generic_params(n, layers, rng)
    s = rng.uniform(0.0, 1.0, size=(6, n))
    e0 = _forward_without_cz(n, layers, s, lam, theta)
    assert np.max(np.abs(e0 - FastStatevectorSim(n, layers).forward(s, lam, theta))) < ZERO
    moved = np.zeros((n, layers, n))  # [a, block, i]: largest change of <Z_a>
    for layer in range(layers):
        for i in range(n):
            e = _forward_without_cz(n, layers, s, lam, theta, {(layer, i)})
            moved[:, layer, i] = np.abs(e - e0).max(axis=0)
    for a in range(n):
        live = lightcone.readout_live_gates(n, layers, a)["cz"]
        np.testing.assert_array_equal(live, moved[a] > ZERO)
        assert moved[a][live].min(initial=1.0) > 1e-6
    # ... and all CZs that are dead for the readouts together can go at once
    for n_actions in (1, min(4, n), n):
        dead = {(int(layer), int(i))
                for layer, i in np.argwhere(~lightcone.live_gates(n, layers, n_actions)["cz"])}
        e = _forward_without_cz(n, layers, s, lam, theta, dead)
        assert np.max(np.abs(e - e0)[:, :n_actions]) < ZERO


# Outside the main grid: the degenerate 2-ring, the 3-ring, a single block, L > n.
EDGE_GRID = [(2, 1), (2, 4), (3, 1), (3, 2), (3, 5), (4, 1), (4, 8), (5, 7), (6, 1), (9, 4)]


@pytest.mark.parametrize(("n", "layers"), EDGE_GRID)
def test_edge_shapes_match_numerics(n, layers):
    """Every readout qubit, with lam of either sign and far from pi, large theta."""
    rng = np.random.default_rng(77 * n + layers)
    sim = FastStatevectorSim(n, layers)
    lam = rng.uniform(0.4, 3.0, size=(layers, n)) * rng.choice([-1.0, 1.0], size=(layers, n))
    theta = rng.uniform(-6.0, 6.0, size=(layers, n, 2))
    s = rng.uniform(0.05, 1.0, size=(BATCH, n))
    for a in range(n):
        # one batched call; random positive weights so live gradients cannot cancel
        weights = rng.uniform(0.5, 1.5, size=BATCH)
        dlam, dtheta = adjoint.grad(s, lam, theta, np.full(BATCH, a), weights)
        live = lightcone.readout_live_gates(n, layers, a)
        np.testing.assert_array_equal(live["enc"], np.abs(dlam) >= ZERO)
        np.testing.assert_array_equal(
            np.stack([live["ry"], live["rz"]], axis=-1), np.abs(dtheta) >= ZERO
        )
    visible = lightcone.feature_visibility(n, layers, n)
    e0 = sim.forward(s, lam, theta)
    for j in range(n):
        s2 = s.copy()
        s2[:, j] = rng.uniform(0.05, 1.0, size=BATCH)
        delta = np.abs(sim.forward(s2, lam, theta) - e0).max(axis=0)
        np.testing.assert_array_equal(visible[:, j], delta > ZERO)


def test_default_n_actions_is_min_4_n():
    for n in (2, 3, 4, 6, 10):
        assert lightcone.feature_visibility(n, 4).shape == (min(4, n), n)
        for kind, mask in lightcone.live_gates(n, 4).items():
            np.testing.assert_array_equal(mask, lightcone.live_gates(n, 4, min(4, n))[kind])


# ------------------------------------------------------ the audit numbers


@pytest.mark.parametrize(("n", "dead", "total"), [(4, 4, 48), (6, 12, 72), (8, 26, 96),
                                                  (10, 46, 120)])
def test_audit_dead_parameter_counts(n, dead, total):
    """L = 4, default 4 readout qubits: 4/48, 12/72, 26/96, 46/120 dead circuit params."""
    mask = lightcone.live_parameter_mask(n, 4)
    flat = np.concatenate([mask["lam"].ravel(), mask["theta"].ravel()])
    assert flat.size == total
    assert int((~flat).sum()) == dead
    # ... and that is what the gradients say, too
    _, glam, gtheta = _numerics(n, 4)
    numeric_dead = (glam[:4].max(axis=0) < ZERO).sum() + (gtheta[:4].max(axis=0) < ZERO).sum()
    assert int(numeric_dead) == dead


def test_audit_blindness_at_four_blocks():
    """L = 4: complete at 4/6 qubits, one blind feature at 8, three at 10."""
    for n, blind in ((4, 0), (6, 0), (8, 1), (10, 3)):
        visible = lightcone.feature_visibility(n, 4)
        np.testing.assert_array_equal((~visible).sum(axis=1), np.full(4, blind))
    np.testing.assert_array_equal(np.flatnonzero(~lightcone.feature_visibility(8, 4)[3]), [7])
    np.testing.assert_array_equal(
        np.flatnonzero(~lightcone.feature_visibility(10, 4)[0]), [4, 5, 6]
    )


@pytest.mark.parametrize(("n", "layers", "n_actions"), CASES)
def test_final_cz_ring_and_final_rz_are_dead(n, layers, n_actions):
    live = lightcone.live_gates(n, layers, n_actions)
    assert not live["cz"][-1].any()
    assert not live["rz"][-1].any()


@pytest.mark.parametrize("n", SIZES)
def test_final_cz_ring_never_affects_the_readout(n):
    rng = np.random.default_rng(n)
    sim = FastStatevectorSim(n, 4)
    lam, theta = _generic_params(n, 4, rng)
    s = rng.uniform(0.0, 1.0, size=(BATCH, n))
    e_full, psi = sim.forward(s, lam, theta, return_state=True)
    psi_no_ring = psi * sim.ring_diagonal  # the ring is self-inverse: undo the last one
    e_no_ring = (psi_no_ring.real**2 + psi_no_ring.imag**2) @ sim.z_diags.T
    assert np.max(np.abs(e_no_ring - e_full)) < ZERO


# ------------------------------------------------- depth for full visibility


@pytest.mark.parametrize(("n", "expected"), [(4, 3), (5, 3), (6, 4), (7, 4), (8, 5), (10, 6)])
def test_min_layers_full_visibility(n, expected):
    needed = lightcone.min_layers_full_visibility(n)
    assert needed == expected == n // 2 + 1
    # derived, not assumed: the numerics agree that `needed` suffices and one less does not
    delta_enough, _, _ = _numerics(n, needed)
    delta_short, _, _ = _numerics(n, needed - 1)
    assert (delta_enough[:4] > ZERO).all()
    assert not (delta_short[:4] > ZERO).all()


def test_min_layers_full_visibility_other_shapes():
    for n in (3, 9, 11, 12):
        assert lightcone.min_layers_full_visibility(n) == n // 2 + 1
    for n, n_actions in ((6, 6), (8, 6), (8, 8), (10, 8)):
        needed = lightcone.min_layers_full_visibility(n, n_actions)
        assert lightcone.feature_visibility(n, needed, n_actions).all()
        assert not lightcone.feature_visibility(n, needed - 1, n_actions).all()
    # the shipped depth L = 4 is enough up to 7 qubits and not beyond
    assert all(lightcone.min_layers_full_visibility(n) <= 4 for n in (4, 5, 6, 7))
    assert all(lightcone.min_layers_full_visibility(n) > 4 for n in (8, 10))


def test_two_qubit_ring_is_the_identity():
    """CZ(0,1) CZ(1,0) = 1: no depth ever shows Z_0 the feature on qubit 1."""
    np.testing.assert_array_equal(lightcone.feature_visibility(2, 5), np.eye(2, dtype=bool))
    assert not lightcone.live_gates(2, 5)["cz"].any()
    with pytest.raises(ValueError, match="no circuit depth"):
        lightcone.min_layers_full_visibility(2)
    rng = np.random.default_rng(2)
    lam, theta = _generic_params(2, 5, rng)
    s = rng.uniform(0.0, 1.0, size=(BATCH, 2))
    s2 = s.copy()
    s2[:, 1] = rng.uniform(0.0, 1.0, size=BATCH)
    sim = FastStatevectorSim(2, 5)
    assert np.max(np.abs(sim.forward(s, lam, theta)[:, 0] - sim.forward(s2, lam, theta)[:, 0])) \
        < ZERO


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="n_qubits"):
        lightcone.feature_visibility(1, 4)
    with pytest.raises(ValueError, match="n_layers"):
        lightcone.live_gates(4, 0)
    with pytest.raises(ValueError, match="n_actions"):
        lightcone.live_parameter_mask(4, 4, 6)
    with pytest.raises(ValueError, match="readout qubit"):
        lightcone.readout_live_gates(4, 4, 4)


# ---------------------------------------------------------------- blind spots

Q8_FEATURES = ["ray -60°", "ray -40°", "ray -20°", "ray 0°", "ray +20°", "ray +40°", "ray +60°",
               "speed"]
Q10_FEATURES = ["ray -60°", "ray -30°", "ray 0°", "ray +30°", "ray +60°", "speed",
                "curvature ahead", "lateral offset", "heading error", "corner speed"]


def test_blind_spots_empty_when_everything_is_visible():
    assert lightcone.blind_spots(["ray -60°", "ray 0°", "ray +60°", "speed"], 4) == []
    assert lightcone.blind_spots(Q8_FEATURES[:5] + ["speed"], 4) == []
    assert lightcone.blind_spots(Q8_FEATURES, 5) == []


def test_blind_spots_names_actions_and_features():
    assert lightcone.blind_spots(Q8_FEATURES, 4) == [
        "Right (Z_0) cannot see: ray +20°",
        "Straight (Z_1) cannot see: ray +40°",
        "Left (Z_2) cannot see: ray +60°",
        "Brake (Z_3) cannot see: speed",
    ]
    lines = lightcone.blind_spots(Q10_FEATURES, 4)
    assert lines[0] == "Right (Z_0) cannot see: ray +60°, speed, curvature ahead"
    assert lines[3] == "Brake (Z_3) cannot see: lateral offset, heading error, corner speed"


def test_blind_spots_labels():
    lines = lightcone.blind_spots(Q8_FEATURES, 4, n_actions=6)
    assert len(lines) == 6
    assert lines[4] == "Brake right (Z_4) cannot see: ray -60°"
    custom = lightcone.blind_spots(Q8_FEATURES, 4, action_labels=["R", "S", "L", "B"])
    assert custom[3] == "B (Z_3) cannot see: speed"
    # no action set of that size: bare readout names
    assert lightcone.blind_spots(Q8_FEATURES, 4, n_actions=1) == ["Z_0 cannot see: ray +20°"]


def test_blind_spot_warning_is_empty_without_blind_spots():
    assert lightcone.blind_spot_warning(["ray -60°", "ray 0°", "ray +60°", "speed"], 4) == []
    assert lightcone.blind_spot_warning(Q8_FEATURES, 5) == []


def test_blind_spot_warning_names_the_fix_and_the_blind_spots():
    lines = lightcone.blind_spot_warning(Q8_FEATURES, 4)
    assert "n_layers = 4 is too shallow for 8 qubits" in lines[0]
    assert "full visibility needs n_layers >= 5" in lines[0]
    assert [line.strip() for line in lines[1:]] == lightcone.blind_spots(Q8_FEATURES, 4)
    assert "needs n_layers >= 6" in lightcone.blind_spot_warning(Q10_FEATURES, 4)[0]
    # n = 2: the identity ring never reaches full visibility
    assert "no depth gives full visibility" in lightcone.blind_spot_warning(["a", "b"], 4)[0]


# ------------------------------------------------------------- circuit_spec


@pytest.mark.parametrize("n, layers, n_actions",
                         [(4, 4, 4), (6, 4, 4), (8, 4, 4), (10, 4, 4), (8, 5, 6), (10, 2, 8)])
def test_circuit_spec_exposes_the_light_cone(n, layers, n_actions):
    import json

    from traqmania.agents.quantum.circuit import circuit_spec

    spec = circuit_spec({"circuit": {"n_qubits": n, "n_layers": layers,
                                     "n_actions": n_actions}})
    visible = lightcone.feature_visibility(n, layers, n_actions)
    assert spec["visibility"] == visible.astype(int).tolist()
    assert all(v in (0, 1) and type(v) is int for row in spec["visibility"] for v in row)
    mask = lightcone.live_parameter_mask(n, layers, n_actions)
    assert spec["dead_params"] == int((~mask["lam"]).sum() + (~mask["theta"]).sum())
    assert spec["min_layers_full_visibility"] == n // 2 + 1
    json.dumps(spec)  # still JSON-serializable (it rides in the welcome message)


def test_circuit_spec_light_cone_at_the_shipped_sizes():
    from traqmania.agents.quantum.circuit import circuit_spec

    dead = {n: circuit_spec({"n_qubits": n})["dead_params"] for n in (4, 6, 8, 10)}
    assert dead == {4: 4, 6: 12, 8: 26, 10: 46}
    assert circuit_spec({"n_qubits": 6})["visibility"] == [[1] * 6] * 4
    assert circuit_spec({"n_qubits": 8})["visibility"][3] == [1, 1, 1, 1, 1, 1, 1, 0]
    # a shape the analysis rejects (more readouts than qubits) keeps the spec usable
    odd = circuit_spec({"n_qubits": 4, "n_actions": 6})
    assert odd["visibility"] is None and odd["dead_params"] is None
    assert odd["min_layers_full_visibility"] is None and odd["n_actions"] == 6
    assert odd["dead_gates"] is None and all(g["live"] is None for g in odd["gates"])
    # 4 qubits: the last block's RZs and CZ ring never reach a readout (12 of 16 CZ live)
    assert circuit_spec({"n_qubits": 4})["dead_gates"] == {
        "ry_enc": 0, "ry": 0, "rz": 4, "cz": 4, "total": 8}
    assert circuit_spec({"n_qubits": 10})["dead_gates"] == {
        "ry_enc": 12, "ry": 12, "rz": 22, "cz": 19, "total": 65}


_SPEC_KIND = {"ry_enc": "enc", "ry": "ry", "rz": "rz", "cz": "cz"}  # spec gate type -> lightcone


@pytest.mark.parametrize("n, layers, n_actions",
                         [(4, 4, 4), (6, 4, 4), (8, 4, 4), (10, 4, 4), (8, 5, 6), (10, 2, 8)])
def test_circuit_spec_flags_every_gate_live_or_dead(n, layers, n_actions):
    from traqmania.agents.quantum.circuit import circuit_spec

    spec = circuit_spec({"circuit": {"n_qubits": n, "n_layers": layers,
                                     "n_actions": n_actions}})
    live = lightcone.live_gates(n, layers, n_actions)
    assert len(spec["gates"]) == spec["counts"]["total"]
    for gate in spec["gates"]:
        qubit = gate["q0"] if gate["type"] == "cz" else gate["qubit"]
        assert gate["live"] is bool(live[_SPEC_KIND[gate["type"]]][gate["layer"], qubit])
    dead = spec["dead_gates"]
    assert set(dead) == set(spec["counts"])  # same keys as the per-type gate counts
    for key, kind in _SPEC_KIND.items():
        assert dead[key] == int((~live[kind]).sum())
        assert dead[key] == sum(g["type"] == key and not g["live"] for g in spec["gates"])
    assert dead["total"] == sum(not g["live"] for g in spec["gates"])
    # every dead parameter sits in a dead rotation gate, and vice versa
    assert spec["dead_params"] == dead["ry_enc"] + dead["ry"] + dead["rz"]


@pytest.mark.parametrize("n, layers, n_actions", [(4, 4, 4), (10, 4, 4), (8, 5, 6)])
def test_circuit_spec_live_gates_are_the_pruned_hardware_circuit(n, layers, n_actions):
    """What the diagram leaves undimmed is, gate for gate, what hardware runs."""
    from traqmania.agents.quantum.circuit import circuit_spec

    spec = circuit_spec({"circuit": {"n_qubits": n, "n_layers": layers,
                                     "n_actions": n_actions}})
    drawn = [
        ("cz", (g["q0"], g["q1"])) if g["type"] == "cz"
        else ("rz" if g["type"] == "rz" else "ry", (g["qubit"],))
        for g in spec["gates"] if g["live"]
    ]
    pruned = lightcone.pruned_circuit(n, layers, n_actions).circuit
    assert [op[:2] for op in _ops(pruned)] == drawn


def test_circuit_spec_keeps_the_keys_older_clients_read():
    from traqmania.agents.quantum.circuit import circuit_spec

    spec = circuit_spec({"circuit": {"n_qubits": 6, "n_layers": 4}})
    assert spec["n_qubits"] == 6 and spec["n_layers"] == 4 and spec["n_actions"] == 4
    assert spec["counts"] == {"ry_enc": 24, "ry": 24, "rz": 24, "cz": 24, "total": 96}
    assert spec["n_params"] == {"lam": 24, "theta": 48, "w": 4, "b": 4, "total": 80}
    assert spec["param_layout"] == ["lam", "theta", "w", "b"]
    assert spec["readout"] == ["Z_0", "Z_1", "Z_2", "Z_3"]
    assert spec["action_labels"] == ["Right", "Straight", "Left", "Brake"]
    # the gate list minus the new flag is the pre-light-cone list, in the same order
    legacy = [{k: v for k, v in g.items() if k != "live"} for g in spec["gates"]]
    assert legacy[0] == {"type": "ry_enc", "qubit": 0, "layer": 0}
    assert legacy[6:8] == [{"type": "ry", "qubit": 0, "layer": 0},
                           {"type": "rz", "qubit": 0, "layer": 0}]
    assert legacy[23] == {"type": "cz", "q0": 5, "q1": 0, "layer": 0}
    assert [g["type"] for g in legacy[:24]] == ["ry_enc"] * 6 + ["ry", "rz"] * 6 + ["cz"] * 6
    assert [g["layer"] for g in legacy] == [layer for layer in range(4) for _ in range(24)]


def test_circuit_spec_says_whether_hardware_prunes_the_dead_gates():
    """"Skipped on hardware" in the page is only true while the hardware path
    prunes: the spec carries ``[hardware] prune_light_cone``."""
    from traqmania.agents.quantum.circuit import circuit_spec
    from traqmania.config import load_config

    assert circuit_spec({"n_qubits": 4})["pruned_on_hardware"] is True  # [circuit] alone
    config = load_config()
    assert circuit_spec(config)["pruned_on_hardware"] is True  # the shipped default
    config["hardware"]["prune_light_cone"] = False
    assert circuit_spec(config)["pruned_on_hardware"] is False
    assert circuit_spec(config)["dead_gates"]["total"] == 8  # the analysis itself is unchanged


def _run_node(probe: str, js_file: str, payload) -> list:
    """Run a web/js module's builders under node (skip without node): ``probe``
    gets the module path as argv[1] and the JSON payload on stdin."""
    import json
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = subprocess.run(
        [node, "--input-type=module", "-e", probe,
         str(REPO_ROOT / "traqmania" / "web" / "js" / js_file)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


# One diagram / legend / "who sees what" element for every case, like the page:
# each welcome re-renders into the same nodes, so anything stale would show.
_WEB_PROBE = """
import fs from "node:fs";
const src = fs.readFileSync(process.argv[1]);
const C = await import("data:text/javascript;base64," + src.toString("base64"));
const diagram = { innerHTML: "" }, legend = { innerHTML: "" };
const section = { innerHTML: "", hidden: true };
const out = JSON.parse(fs.readFileSync(0, "utf8")).map(({ spec, labels }) => {
  C.renderCircuit(spec, diagram, legend);
  C.renderVisibility(spec, labels, section);
  return {
    svg: C.circuitSvg(spec),
    caption: C.parameterCaption(spec),
    blind: C.blindSpots(spec, labels),
    note: C.visibilityNote(spec, labels),
    table: C.visibilityMatrixHtml(spec, labels),
    diagram: diagram.innerHTML,
    legend: legend.innerHTML,
    section: section.innerHTML,
    sectionHidden: section.hidden,
  };
});
process.stdout.write(JSON.stringify(out));
"""


def _matrix_cells(table: str) -> tuple[list[str], list[tuple[str, list[dict]]]]:
    """Parse the matrix the way a browser would: ``(column heads, [(row head,
    [cell attributes])])``. Fails on any element or attribute the builder does
    not write itself — a label that escaped its text/attribute would add one."""
    from html.parser import HTMLParser

    heads: list[str] = []
    rows: list[tuple[str, list[dict]]] = []

    class Matrix(HTMLParser):
        where = None  # "col" / "row": the header cell whose text is being read

        def handle_starttag(self, tag, attrs):
            names = [name for name, _ in attrs]
            assert tag in ("table", "thead", "tbody", "tr", "th", "td", "span"), tag
            if tag == "th":
                assert names == ["scope"]
                self.where = dict(attrs)["scope"]
                if self.where == "row":
                    rows.append(("", []))
                else:
                    heads.append("")
            elif tag == "td" and rows:  # (the corner cell of the head row has no attributes)
                assert names == ["class", "data-tip", "aria-label"]
                rows[-1][1].append(dict(attrs))
            else:
                assert names in ([], ["class"]), (tag, names)

        def handle_endtag(self, tag):
            if tag == "th":
                self.where = None

        def handle_data(self, data):
            if self.where == "col":
                heads[-1] += data
            elif self.where == "row":
                rows[-1] = (rows[-1][0] + data, rows[-1][1])

    Matrix().feed(table)
    return heads, rows


def test_web_diagram_and_matrix_render_the_spec():
    """The browser's builders (web/js/circuit.js, run under node) against the
    structural analysis: the diagram dims exactly the dead gates, and the
    "who sees what" matrix and its one-line note say what ``blind_spots`` says
    — an action/feature transposition in the page would fail here."""
    import collections
    import re
    import xml.etree.ElementTree as ET

    from traqmania.agents.quantum.circuit import circuit_spec

    gp10 = ["ray -60°", "ray -30°", "ray 0°", "ray +30°", "ray +60°", "speed",
            "curvature ahead", "lateral offset", "heading error", "corner speed"]
    hostile = ['<img src=x onerror=1>', '"><b>', "a&b", "it's", 'x" onmouseover="y', "f", "g", "h"]
    cases = [
        ({"n_qubits": 4}, ["ray -60°", "ray 0°", "ray +60°", "speed"]),
        ({"n_qubits": 10}, gp10),  # the bundled gp driver's observation at 10 qubits
        ({"n_qubits": 8, "n_actions": 6}, hostile),
        ({"n_qubits": 10, "n_layers": 6, "n_actions": 8}, None),  # no labels: wire names
        ({"n_qubits": 4, "n_actions": 6}, ["a", "b", "c", "d"]),  # rejected by the analysis
        ({"n_qubits": 8}, ["stale", "labels", "of another size"]),  # mismatch: wire names
    ]
    specs = [circuit_spec({"circuit": cfg}) for cfg, _ in cases]
    specs[2]["action_labels"] = ["<i>R</i>", 'S" onclick="z', "L&R", "B's", "E", "F"]
    # same circuit as the first case, hardware pruning switched off in the config
    specs.append(circuit_spec({"circuit": {"n_qubits": 4},
                               "hardware": {"prune_light_cone": False}}))
    cases.append(({}, cases[0][1]))
    payload = [{"spec": spec, "labels": labels}
               for spec, (_, labels) in zip(specs, cases, strict=True)]
    rendered = _run_node(_WEB_PROBE, "circuit.js", payload)

    for spec, (_, labels), page in zip(specs, cases, rendered, strict=True):
        n, layers, n_actions = spec["n_qubits"], spec["n_layers"], spec["n_actions"]
        assert page["diagram"] == page["svg"]
        svg = ET.fromstring(page["svg"])  # well-formed, whatever the labels
        marked = list(svg.iter("{http://www.w3.org/2000/svg}g"))
        groups = collections.Counter(g.get("class") for g in marked)
        # grey <Z> boxes: the qubits no action reads
        assert groups["meas-gauge-only"] == max(n - n_actions, 0)
        dimmed = collections.Counter(
            "ry_enc" if "λx" in body else "rz" if ">RZ<" in body
            else "ry" if ">RY<" in body else "cz"
            for body in re.findall(r'<g class="gate-dead"[^>]*>(.*?)</g>', page["svg"]))
        if spec["visibility"] is None:  # nothing to show: plain diagram, no matrix
            assert not dimmed and page["blind"] is None and page["note"] is None
            assert page["table"] == ""
            assert page["caption"] == f"{spec['n_params']['total']} trainable parameters"
            # ... and the matrix of the previous welcome is gone, not left standing
            assert page["sectionHidden"] is True and page["section"] == ""
            assert "Dimmed" not in page["legend"]
            continue
        dead = spec["dead_gates"]
        assert dimmed == {k: v for k, v in dead.items() if k != "total" and v}
        assert groups["gate-dead"] == dead["total"]
        assert page["caption"] == (f"{spec['n_params']['total']} trainable parameters, "
                                   f"{spec['dead_params']} structurally dead")
        assert page["caption"] in page["legend"] and "Dimmed gate" in page["legend"]
        # "skipped on hardware" only while the hardware path really prunes
        tips = {g.get("data-tip") for g in marked if g.get("class") == "gate-dead"}
        assert len(tips) == 1 and "cannot influence any action" in next(iter(tips))
        claims = [text.count("skipped on hardware") for text in (*tips, page["legend"])]
        assert claims == [int(spec["pruned_on_hardware"])] * 2

        usable = labels is not None and len(labels) == n
        names = labels if usable else [f"q{j}" for j in range(n)]
        expected = lightcone.blind_spots(names, layers, n_actions, spec["action_labels"])
        spoken = [f"{b['action']} cannot see: {', '.join(b['hidden'])}" for b in page["blind"]]
        assert spoken == [re.sub(r" \(Z_\d+\)", "", line) for line in expected]
        assert page["sectionHidden"] is False and page["table"] in page["section"]
        if expected:
            assert page["note"]["kind"] == "warn" and spoken[-1] in page["note"]["text"]
            assert (f"full visibility needs {spec['min_layers_full_visibility']} layers"
                    in page["note"]["text"])
        else:
            assert page["note"]["kind"] == "ok"
        # amber note and the "empty cell" legend row only when something is hidden
        assert ("cone-warn" in page["section"]) == bool(expected)
        assert ("vis-mark-off" in page["section"]) == bool(expected)

        # one row per feature (qubit order), one cell per action (readout order);
        # labels come from config/weights sidecars and stay text, never markup
        heads, rows = _matrix_cells(page["table"])
        assert heads == spec["action_labels"]
        assert [head for head, _ in rows] == [
            f"q{j} {names[j]}" if usable else f"q{j}" for j in range(n)]
        for j, (_, cells) in enumerate(rows):
            assert [cell["class"] for cell in cells] == [
                "vis-on" if row[j] else "vis-off" for row in spec["visibility"]]
            for a, cell in enumerate(cells):
                tip, action = cell["data-tip"], spec["action_labels"][a]
                assert cell["aria-label"] == tip
                if spec["visibility"][a][j]:
                    assert tip == f"{action} can see {names[j]}."
                    continue
                # a blind cell says why: the feature sits too far around the ring
                steps = min(abs(a - j), n - abs(a - j))
                assert steps > layers - 1
                assert tip.startswith(f"{action} cannot see {names[j]}: ")
                assert f"qubit {j}, {steps} steps around the ring from readout qubit {a}" in tip
                assert f"{layers} layers only reach {layers - 1}" in tip


# explain.js imports the documentation browser (fetch + DOM); the probe swaps
# that import for a stub and gives initExplain the few DOM calls it makes.
_EXPLAIN_PROBE = """
import fs from "node:fs";
const src = fs.readFileSync(process.argv[1], "utf8")
  .replace('import { initDocs } from "./docs.js";', "const initDocs = () => {};");
class El {
  constructor() {
    this.children = []; this.dataset = {}; this.innerHTML = ""; this.active = false;
    this.classList = { toggle: (name, on) => { this.active = on; } };
  }
  addEventListener(type, fn) { this.click = fn; }
  append(...kids) { this.children.push(...kids); }
  replaceChildren(...kids) { this.children = kids; }
  querySelectorAll() { return this.children; }
}
globalThis.document = { createElement: () => new El() };
const E = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));
const root = new El();
const look = () => {
  const [nav, body] = root.children;
  return { active: nav.children.filter((b) => b.active).map((b) => b.dataset.section),
           body: body.innerHTML.replace(/\\s+/g, " ") };
};
const out = [];
E.initExplain(root);  // boot: no spec yet
out.push(look());
root.children[0].children.find((b) => b.dataset.section === "circuit").click();
out.push(look());
for (const spec of JSON.parse(fs.readFileSync(0, "utf8"))) {
  E.initExplain(root, spec);  // a welcome: qubit or driver switch
  out.push(look());
}
process.stdout.write(JSON.stringify(out));
"""


def test_web_explain_light_cone_paragraph_follows_the_spec():
    """Explain -> "The quantum circuit" (web/js/explain.js under node): the
    light-cone paragraph states what the spec says about this circuit size,
    and a welcome re-templates the copy without closing the open sub-tab."""
    from traqmania.agents.quantum.circuit import circuit_spec

    specs = [circuit_spec({"n_qubits": 4}), circuit_spec({"n_qubits": 10}),
             circuit_spec({"n_qubits": 10, "n_layers": 6}),
             circuit_spec({"n_qubits": 4, "n_actions": 6})]  # rejected: no verdict
    boot, opened, q4, q10, q10_deep, rejected = _run_node(_EXPLAIN_PROBE, "explain.js", specs)

    full, short = "every action can see every input", "the cones are too short"
    assert boot["active"] == ["what"] and "light cone" not in boot["body"]
    assert opened["active"] == ["circuit"] and "<strong>light cone</strong>" in opened["body"]
    assert full not in opened["body"] and short not in opened["body"]  # no spec, no verdict
    for page in (q4, q10, q10_deep, rejected):
        assert page["active"] == ["circuit"]  # the open sub-tab survives the rebuild
        assert "<strong>light cone</strong>" in page["body"]
    assert f"With 4 qubits and 4 layers {full}." in q4["body"] and short not in q4["body"]
    assert f"With 10 qubits and 4 layers {short}" in q10["body"] and full not in q10["body"]
    assert "(that would take 6 layers)" in q10["body"] and "Who sees what" in q10["body"]
    assert f"With 10 qubits and 6 layers {full}." in q10_deep["body"]
    assert full not in rejected["body"] and short not in rejected["body"]


# ------------------------------------------------------------- pruned circuit


def _aer_expectations(qc, index_x, index_theta, x, theta_flat, n_actions):
    """Exact <Z_a> (B, A) of ``qc`` on Aer, binding x[:, index_x] and theta[index_theta]."""
    from qiskit_aer.primitives import EstimatorV2

    from traqmania.agents.quantum.circuit import observables, split_parameters

    input_params, weight_params = split_parameters(qc)
    values = {
        tuple(input_params): x[:, index_x],
        tuple(weight_params): np.broadcast_to(theta_flat[index_theta],
                                              (x.shape[0], len(index_theta))),
    }
    obs = [[o] for o in observables(qc.num_qubits)[:n_actions]]
    estimator = EstimatorV2(options={"default_precision": 0.0})  # exact expectation values
    return np.asarray(estimator.run([(qc, obs, values)]).result()[0].data.evs).T


def _ops(qc) -> list[tuple]:
    """(gate name, qubit indices, parameter names) per instruction, in circuit order."""
    return [
        (
            inst.operation.name,
            tuple(qc.find_bit(q).index for q in inst.qubits),
            tuple(str(p) for p in inst.operation.params),
        )
        for inst in qc.data
    ]


@pytest.mark.parametrize(("n", "layers", "n_actions"),
                         [(4, 4, None), (6, 4, None), (10, 4, None), (8, 5, 6), (10, 4, 8),
                          (2, 3, None), (3, 5, None), (5, 1, None), (7, 2, 1)])
def test_pruned_circuit_equals_full_circuit_on_aer(n, layers, n_actions):
    from traqmania.agents.quantum.circuit import build_circuit

    actions = min(4, n) if n_actions is None else n_actions
    rng = np.random.default_rng(100 * n + layers)
    lam, theta = _generic_params(n, layers, rng)
    s = rng.uniform(0.0, 1.0, size=(3, n))
    x = (lam[None, :, :] * s[:, None, :]).reshape(3, layers * n)  # x[l*n + i] = lam[l,i]*s[i]

    full = build_circuit(n, layers)
    pruned = lightcone.pruned_circuit(n, layers, n_actions)
    e_full = _aer_expectations(
        full, np.arange(layers * n), np.arange(2 * layers * n), x, theta.ravel(), actions
    )
    e_pruned = _aer_expectations(
        pruned.circuit, pruned.input_index, pruned.weight_index, x, theta.ravel(), actions
    )
    assert e_full.shape == e_pruned.shape == (3, actions)
    assert np.max(np.abs(e_pruned - e_full)) <= 1e-10
    e_fast = FastStatevectorSim(n, layers).forward(s, lam, theta)[:, :actions]
    assert np.max(np.abs(e_pruned - e_fast)) <= 1e-10


@pytest.mark.parametrize(("n", "layers", "n_actions"), [(4, 4, None), (6, 4, None), (10, 4, None),
                                                        (8, 5, 6), (2, 3, None)])
def test_pruned_circuit_keeps_names_order_and_indices(n, layers, n_actions):
    from traqmania.agents.quantum.circuit import build_circuit, split_parameters

    full = build_circuit(n, layers)
    pruned = lightcone.pruned_circuit(n, layers, n_actions)
    assert isinstance(pruned, lightcone.PrunedCircuit)
    qc, input_index, weight_index = pruned
    assert qc.num_qubits == n

    # same Parameter names as build_circuit, and the index arrays say which remain
    input_params, weight_params = split_parameters(qc)
    assert {p.name for p in qc.parameters} <= {p.name for p in full.parameters}
    assert [p.index for p in input_params] == input_index.tolist()
    assert [p.index for p in weight_params] == weight_index.tolist()
    mask = lightcone.live_parameter_mask(n, layers, n_actions)
    np.testing.assert_array_equal(input_index, np.flatnonzero(mask["lam"].ravel()))
    np.testing.assert_array_equal(weight_index, np.flatnonzero(mask["theta"].ravel()))

    # only live gates, in the original order: a subsequence of the full circuit
    remaining = iter(_ops(full))
    assert all(op in remaining for op in _ops(qc))
    live = lightcone.live_gates(n, layers, n_actions)
    counts = qc.count_ops()
    assert counts.get("cz", 0) == int(live["cz"].sum())
    assert counts.get("rz", 0) == int(live["rz"].sum())
    assert counts.get("ry", 0) == int(live["enc"].sum() + live["ry"].sum())


def test_pruned_circuit_drops_two_qubit_gates():
    from traqmania.agents.quantum.circuit import build_circuit

    assert build_circuit(4, 4).count_ops()["cz"] == 16
    assert lightcone.pruned_circuit(4, 4).circuit.count_ops()["cz"] == 12
    # beyond n = 4 whole stretches of the ring fall outside the cone, not just the last block
    for n, cz in ((6, 17), (8, 20), (10, 21)):
        assert lightcone.pruned_circuit(n, 4).circuit.count_ops()["cz"] == cz


@pytest.mark.parametrize(("n", "layers"), [(4, 4), (6, 3), (8, 4)])
def test_pruned_circuit_is_exact_at_special_angles(n, layers):
    """Dead gates commute for EVERY angle — also at the 0 / pi/2 / pi corners."""
    from traqmania.agents.quantum.circuit import build_circuit

    rng = np.random.default_rng(n + layers)
    x = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2], size=(4, layers * n))
    theta = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2, 2 * np.pi], size=2 * layers * n)
    pruned = lightcone.pruned_circuit(n, layers)
    e_full = _aer_expectations(
        build_circuit(n, layers), np.arange(layers * n), np.arange(2 * layers * n), x, theta, 4
    )
    e_pruned = _aer_expectations(
        pruned.circuit, pruned.input_index, pruned.weight_index, x, theta, 4
    )
    assert np.max(np.abs(e_pruned - e_full)) <= 1e-10


# ------------------------------------------------- command line / packaging


def test_report_lists_the_audit_findings(capsys):
    assert lightcone.main(["--qubits", "10", "--layers", "4"]) == 0
    out = capsys.readouterr().out
    assert "10 qubits, 4 blocks, readout Z_0..Z_3" in out
    assert "Right (Z_0)     XXXX...XXX  blind to: 4, 5, 6" in out
    assert "Brake (Z_3)     XXXXXXX...  blind to: 7, 8, 9" in out
    assert "46/120" in out
    assert "Full visibility needs L >= 6 blocks (NOT reached at L = 4)" in out

    assert lightcone.main(["--qubits", "4", "--layers", "4", "--actions", "4"]) == 0
    out = capsys.readouterr().out
    assert "Brake (Z_3)     XXXX  sees all" in out
    assert "4/48" in out
    assert "dead theta[l, i, k]: (3,0,1) (3,1,1) (3,2,1) (3,3,1)" in out
    assert "Live two-qubit gates: 12/16 CZ" in out
    assert "Full visibility needs L >= 3 blocks (reached)" in out


def test_module_runs_as_script_without_qiskit():
    """``python -m traqmania.agents.quantum.lightcone`` works, warning-free, numpy-only."""
    result = subprocess.run(
        [sys.executable, "-W", "error", "-m", "traqmania.agents.quantum.lightcone",
         "--qubits", "8", "--layers", "4", "--actions", "6"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Brake left (Z_5)   X.XXXXXX  blind to: 1" in result.stdout
    assert "14/96" in result.stdout

    check = (
        "import sys, traqmania.agents.quantum as q; "
        "assert 'traqmania.agents.quantum.lightcone' not in sys.modules; "
        "q.lightcone.live_gates(6, 4); q.lightcone.blind_spots(['f'] * 8, 4); "
        "assert 'qiskit' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
