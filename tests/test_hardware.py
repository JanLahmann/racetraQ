"""Hardware execution path, exercised entirely on LOCAL fake backends.

No network, no IBM account: ``get_backend(use_fake=True)`` returns a
``FakeBackendV2`` that the hardware path simulates on Aer (as an n-qubit
device patch for the 120-156 qubit fakes), and runtime Sessions enter local
testing mode. Account-side behaviour (Open Plan accounts cannot open a
Session) is covered with stubs. Skipped wholesale when qiskit-ibm-runtime
(the [hardware] extra) is not installed.
"""

from __future__ import annotations

import json
import sys
import types
import warnings

import numpy as np
import pytest

qiskit_ibm_runtime = pytest.importorskip("qiskit_ibm_runtime")

from traqmania import hardware  # noqa: E402
from traqmania.agents.quantum import lightcone  # noqa: E402
from traqmania.agents.quantum.qdqn import QuantumQFunction  # noqa: E402
from traqmania.agents.training import spsa  # noqa: E402
from traqmania.config import load_config  # noqa: E402
from traqmania.env.racing_env import RacingEnv  # noqa: E402
from traqmania.env.track import Track  # noqa: E402

CIRCUIT_CFG = {"n_qubits": 4, "n_layers": 4, "seed": 7}
OVAL_WEIGHTS = hardware.WEIGHTS_DIR / "quantum_oval.npz"

# Zero-SWAP two-qubit gate counts of the CZ ring on the Nighthawk square
# lattice at L = 4: {n_qubits: (light-cone pruned, full circuit)}.
NIGHTHAWK_CZ = {4: (12, 16), 6: (17, 24), 10: (21, 40)}


@pytest.fixture(scope="module")
def fake_backend():
    return hardware.get_backend(use_fake=True)


@pytest.fixture(scope="module")
def manila():
    return hardware.get_backend(use_fake=True, fake_name="fake_manila")


# ------------------------------------------------------------- fake backends


def test_default_fake_is_the_nighthawk_twin(fake_backend):
    assert hardware.DEFAULT_FAKE == "fake_miami"
    assert fake_backend.name == "fake_miami"
    assert fake_backend.num_qubits >= 5
    assert "cz" in fake_backend.operation_names  # current IBM QPUs are CZ-based


@pytest.mark.parametrize(
    ("name", "cls_name"),
    [
        ("fake_fez", "FakeFez"),
        ("FakeFez", "FakeFez"),
        ("ibm_fez", "FakeFez"),
        ("fake_miami", "FakeMiami"),
        ("fake_manila", "FakeManilaV2"),
        ("FakeManilaV2", "FakeManilaV2"),
        ("fake_manila_v2", "FakeManilaV2"),
    ],
)
def test_fake_name_resolution_accepts_every_spelling(name, cls_name):
    assert hardware._fake_class(name).__name__ == cls_name


def test_every_listed_fake_resolves():
    names = hardware.available_fakes()
    assert {"fake_miami", "fake_fez", "fake_manila"} <= set(names)
    for name in names:
        assert hardware._fake_class(name).backend_name == name


def test_unknown_fake_name_raises_with_the_available_ones():
    """No silent fallback to some other device (the old path fell back to manila)."""
    with pytest.raises(ValueError, match="unknown fake backend 'fake_bogus'") as excinfo:
        hardware.get_backend(use_fake=True, fake_name="fake_bogus")
    assert "fake_miami" in str(excinfo.value) and "fake_fez" in str(excinfo.value)


def test_too_small_fake_falls_back_and_says_so():
    with pytest.warns(UserWarning, match="fake_manila has only 5 qubits"):
        backend = hardware.get_backend(use_fake=True, fake_name="fake_manila", min_qubits=6)
    assert backend.name == "fake_lagos"  # smallest retired Falcon fake that fits
    assert backend.num_qubits >= 6


# ------------------------------------------------------------ device patches


@pytest.mark.parametrize("n_qubits", sorted(NIGHTHAWK_CZ))
def test_nighthawk_patch_embeds_the_ring_without_swaps(fake_backend, n_qubits):
    cfg = {"n_qubits": n_qubits, "n_layers": 4, "seed": 7}
    pruned_cz, full_cz = NIGHTHAWK_CZ[n_qubits]

    hw = hardware.HardwareQFunction(cfg, fake_backend)
    assert hw.pruned
    assert hw.two_qubit_gates == pruned_cz
    assert hw.backend.num_qubits == n_qubits  # the patch, not the 120-qubit device
    assert hw.backend_name.startswith(f"fake_miami ({n_qubits}-qubit patch")
    assert hw.depth > 0

    full = hardware.HardwareQFunction(cfg, fake_backend, prune_light_cone=False)
    assert not full.pruned
    assert full.two_qubit_gates == full_cz == 4 * n_qubits
    assert full.backend.num_qubits == n_qubits


def test_patch_keeps_the_device_properties_and_is_cached(fake_backend):
    sim = hardware.local_simulator(fake_backend, 4, 4)
    assert hardware.local_simulator(fake_backend, 4, 4) is sim  # noise model built once
    assert hardware.local_simulator(fake_backend, 4, 4, prune=False) is not sim
    assert hardware.execution_backend(fake_backend, CIRCUIT_CFG) is sim
    assert hardware.execution_backend(sim, CIRCUIT_CFG) is sim  # simulators pass through
    assert sim.options.noise_model is not None

    device, qubits = hardware._SIM_INFO[id(sim)]
    assert device == "fake_miami" and len(qubits) == 4
    # every patch edge carries the gate error of the physical pair it stands for
    edges = [qargs for qargs in sim.target["cz"] if qargs is not None]
    assert len(edges) == 8  # the 4-ring, both directions
    for a, b in edges:
        expected = fake_backend.target["cz"][(qubits[a], qubits[b])].error
        assert sim.target["cz"][(a, b)].error == expected
    for i, qubit in enumerate(qubits):
        expected = fake_backend.target["measure"][(qubit,)].error
        assert sim.target["measure"][(i,)].error == expected


def test_patch_simulation_is_noisy_and_reproducible_under_a_seed(fake_backend):
    """The patch twin really APPLIES the device noise (not just carries it in
    its Target): a high-shot estimate is pulled away from the exact value by
    far more than its shot noise and shrinks |<Z_a>|, while a noise-free
    simulator in the same basis is not; a fixed simulator seed reproduces the
    estimate bit for bit."""
    from qiskit_aer import AerSimulator

    params = np.load(OVAL_WEIGHTS)["params"]
    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    fast.set_params(params)
    obs = np.random.default_rng(0).uniform(0.0, 1.0, size=(8, 4))
    exact = (fast.q_values(obs) - fast.b) / fast.w  # the raw <Z_a>
    shots = 100_000  # shot noise 1/sqrt(shots) ~ 0.003 per entry

    def estimate(backend, seed):
        hw = hardware.HardwareQFunction(CIRCUIT_CFG, backend, shots=shots)
        hw.set_params(params)
        hw._estimator.options.simulator.seed_simulator = seed
        return hw.expectations(obs)

    noisy = estimate(fake_backend, 5)
    np.testing.assert_array_equal(estimate(fake_backend, 5), noisy)
    assert not np.array_equal(estimate(fake_backend, 6), noisy)

    ideal = estimate(AerSimulator(basis_gates=["rz", "sx", "x", "cz"]), 5)
    assert np.mean(np.abs(ideal - exact)) < 0.006  # shot noise only
    # 12 CZ at ~2e-3 each plus ~2.5e-3 readout error: a few percent, not zero
    assert 0.008 < np.mean(np.abs(noisy - exact)) < 0.1
    assert np.mean(np.abs(noisy)) < np.mean(np.abs(exact)) - 0.005


def test_local_simulator_builds_once_under_concurrency(fake_backend, monkeypatch):
    """The server asks for the twin from worker threads: concurrent first
    requests share ONE build (and one noise model), later ones hit the cache."""
    import threading

    monkeypatch.setattr(hardware, "_SIMULATORS", {})
    monkeypatch.setattr(hardware, "_SIM_INFO", {})
    built: list[tuple] = []
    real_patch = hardware._patch_backend

    def counting_patch(device, qubits):
        built.append(tuple(qubits))
        return real_patch(device, qubits)

    monkeypatch.setattr(hardware, "_patch_backend", counting_patch)
    sims: list = []
    errors: list[Exception] = []

    def request():
        try:
            sims.append(hardware.local_simulator(fake_backend, 4, 4))
        except Exception as exc:  # reported by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=request) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120.0)
    assert not errors
    assert len(sims) == 4 and len({id(sim) for sim in sims}) == 1
    assert len(built) == 1 and len(built[0]) == 4
    assert hardware.backend_label(sims[0]).startswith("fake_miami (4-qubit patch")


def test_patch_keeps_a_home_for_idle_circuit_qubits(fake_backend):
    """At 12 qubits two of them lie outside every readout's light cone: the
    pruned circuit never touches them, the patch must still seat them."""
    cfg = {"n_qubits": 12, "n_layers": 4, "seed": 7}
    assert not lightcone.live_gates(12, 4)["enc"].any(axis=0).all()
    hw = hardware.HardwareQFunction(cfg, fake_backend)
    assert hw.backend.num_qubits == 12
    assert hw.two_qubit_gates == int(lightcone.live_gates(12, 4)["cz"].sum())


def test_small_fakes_are_simulated_whole(manila):
    sim = hardware.execution_backend(manila, CIRCUIT_CFG)
    assert sim.num_qubits == manila.num_qubits == 5
    assert hardware.backend_label(sim) == "fake_manila"
    assert hardware.execution_backend(manila, CIRCUIT_CFG, prune=False) is sim


def test_mid_size_fakes_get_a_patch_too():
    """The whole 16-qubit fake_guadalupe costs ~14 s per 10-qubit evaluation."""
    guadalupe = hardware.get_backend(use_fake=True, fake_name="fake_guadalupe", min_qubits=6)
    assert guadalupe.num_qubits == 16 > hardware.PATCH_MIN_QUBITS
    hw = hardware.HardwareQFunction({"n_qubits": 6, "n_layers": 4, "seed": 7}, guadalupe)
    assert 6 <= hw.backend.num_qubits < 16
    assert hw.backend_name.startswith(f"fake_guadalupe ({hw.backend.num_qubits}-qubit patch")


# ------------------------------------------------------ light-cone pruning


@pytest.mark.parametrize(
    ("n_qubits", "n_actions"), [(4, None), (6, None), (10, None), (6, 6), (10, 6)]
)
def test_pruned_and_full_circuits_agree_exactly(fake_backend, n_qubits, n_actions):
    """On an exact (noise-free, shot-free) estimator the pruned ISA circuit,
    the full ISA circuit and fastsim give the same <Z_a> — with the default
    four readouts and with six (the light cone widens with the readout set)."""
    from qiskit.primitives import StatevectorEstimator

    cfg = {"n_qubits": n_qubits, "n_layers": 4, "seed": 7}
    if n_actions is not None:
        cfg["n_actions"] = n_actions
    fast = QuantumQFunction(cfg, seed=7)
    params = fast.get_params()
    rng = np.random.default_rng(11)
    params[: fast.lam.size + fast.theta.size] += rng.normal(0.0, 0.7, fast.lam.size
                                                            + fast.theta.size)
    fast.set_params(params)
    obs = rng.uniform(0.0, 1.0, size=(5, n_qubits))
    exact = fast.q_values(obs)  # w = 1, b = 0: the raw expectations

    results = {}
    for prune in (True, False):
        hw = hardware.HardwareQFunction(cfg, fake_backend, prune_light_cone=prune)
        hw.set_params(params)
        hw._estimator = StatevectorEstimator()
        results[prune] = hw.expectations(obs)
        np.testing.assert_allclose(results[prune], exact, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(results[True], results[False], rtol=0.0, atol=1e-9)

    pruned = hardware.HardwareQFunction(cfg, fake_backend)
    assert results[True].shape == (5, pruned.n_actions)
    index = lightcone.pruned_circuit(n_qubits, 4, n_actions)
    assert len(pruned._isa_params) == index.input_index.size + index.weight_index.size
    assert pruned.two_qubit_gates == int(lightcone.live_gates(n_qubits, 4, n_actions)["cz"].sum())
    assert pruned.two_qubit_gates < 4 * n_qubits


def test_pruning_survives_routing_through_swaps(manila):
    """Same equality on a device that needs SWAPs and has a spare qubit."""
    from qiskit.primitives import StatevectorEstimator

    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(4, 4))
    counts = {}
    for prune in (True, False):
        hw = hardware.HardwareQFunction(CIRCUIT_CFG, manila, prune_light_cone=prune)
        hw.set_params(fast.get_params())
        hw._estimator = StatevectorEstimator()
        np.testing.assert_allclose(hw.expectations(obs), fast.q_values(obs),
                                   rtol=0.0, atol=1e-9)
        counts[prune] = hw.two_qubit_gates
    assert counts[True] < counts[False]  # fewer CX on the line-shaped Falcon, too


# -------------------------------------------------------------- Q-function


def test_hardware_qvalues_match_fastsim(fake_backend):
    """Fake-backend Q-values track the exact simulator on the same parameters.

    Tolerances are deliberately loose: the fake backend simulates a device
    noise model, which shrinks |<Z_a>| — we ask for argmax agreement on most
    observations and absolute agreement within 0.5. The Aer shot noise is
    seeded so the statistical margins cannot flake.
    """
    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, shots=4096)
    hw.set_params(fast.get_params())
    hw._estimator.options.simulator.seed_simulator = 1234  # pin the shot noise

    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(8, 4))
    q_fast = fast.q_values(obs)
    q_hw = hw.q_values(obs)

    assert q_hw.shape == (8, 4)
    agreement = float(np.mean(np.argmax(q_hw, axis=1) == np.argmax(q_fast, axis=1)))
    assert agreement >= 0.6, f"argmax agreement {agreement:.2f} < 0.6\n{q_hw}\nvs\n{q_fast}"
    assert np.max(np.abs(q_hw - q_fast)) < 0.5


def test_hardware_qfunction_param_roundtrip_and_layout(fake_backend):
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend)
    assert hw.n_params == 56
    params = np.arange(56, dtype=np.float64)
    hw.set_params(params)
    np.testing.assert_allclose(hw.get_params(), params)
    # Same flat layout as the fastsim twin: [lam(16), theta(32), w(4), b(4)].
    np.testing.assert_allclose(hw.lam.ravel(), params[:16])
    np.testing.assert_allclose(hw.theta.ravel(), params[16:48])
    np.testing.assert_allclose(hw.w, params[48:52])
    np.testing.assert_allclose(hw.b, params[52:])


def test_grad_selected_is_not_implemented(fake_backend):
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend)
    obs = np.zeros((2, 4))
    with pytest.raises(NotImplementedError, match="spsa_sprint"):
        hw.grad_selected(obs, np.zeros(2, dtype=np.int64), np.ones(2))


# --------------------------------------------- estimator + error mitigation


def test_uses_the_client_side_estimator_when_available(fake_backend):
    pytest.importorskip("qiskit_ibm_runtime.executor_estimator")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # EstimatorV2 would warn
        hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, shots=256)
        q = hw.q_values(np.full((1, 4), 0.3))
    assert type(hw._estimator).__module__.startswith("qiskit_ibm_runtime.executor_estimator")
    assert hw._estimator.options.default_shots == 256
    assert np.all(np.isfinite(q))


def test_falls_back_to_estimator_v2_on_older_runtimes(fake_backend, monkeypatch):
    """Without the client-side Estimator (runtime < 0.48) the server-side
    primitive is used; in local mode it is not asked to mitigate at level 0."""
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime.executor_estimator", None)
    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    obs = np.random.default_rng(3).uniform(0.0, 1.0, size=(2, 4))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, shots=2048)
        hw.set_params(fast.get_params())
        q = hw.q_values(obs)
    assert type(hw._estimator).__name__ == "EstimatorV2"
    assert hw._estimator.options.default_shots == 2048
    assert np.max(np.abs(q - fast.q_values(obs))) < 0.5
    assert not [w for w in caught if "resilience_level" in str(w.message)]


def test_resilience_level_reaches_the_estimator(fake_backend):
    assert hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend).resilience_level == 0
    for level in sorted(hardware.RESILIENCE_LEVELS):
        hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, resilience_level=level)
        assert hw.resilience_level == level
        assert hw._estimator.options.resilience_level == level
    with pytest.raises(ValueError, match="resilience_level"):
        hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, resilience_level=3)


def test_trex_runs_on_the_fake_backend(fake_backend):
    """resilience_level = 1 really executes locally (measurement twirling)."""
    pytest.importorskip("qiskit_ibm_runtime.executor_estimator")
    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, shots=2048,
                                    resilience_level=1)
    hw.set_params(fast.get_params())
    hw._estimator.options.simulator.seed_simulator = 99
    obs = np.random.default_rng(5).uniform(0.0, 1.0, size=(2, 4))
    assert np.max(np.abs(hw.q_values(obs) - fast.q_values(obs))) < 0.5


def test_hardware_config_reaches_lap_and_sprint(fake_backend, monkeypatch):
    """[hardware] resilience_level / prune_light_cone are read from the config,
    explicit arguments win, and both land on the HardwareQFunction."""
    built: list[hardware.HardwareQFunction] = []
    real_init = hardware.HardwareQFunction.__init__

    def recording_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(hardware.HardwareQFunction, "__init__", recording_init)
    config = load_config()
    config["hardware"] = {**config["hardware"], "resilience_level": 1,
                          "prune_light_cone": False}

    started: list[dict] = []
    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=1, config=config,
                                       on_start=started.append)
    assert (built[-1].resilience_level, built[-1].pruned) == (1, False)
    assert built[-1]._estimator.options.resilience_level == 1
    assert (result["resilience_level"], result["pruned"]) == (1, False)
    assert result["two_qubit_gates"] == NIGHTHAWK_CZ[4][1]
    assert len(started) == 1
    assert started[0] == {k: result[k] for k in started[0]}

    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, iterations=1,
                                  shots=64, batch=4, config=config,
                                  resilience_level=0, prune_light_cone=True)
    assert (built[-1].resilience_level, built[-1].pruned) == (0, True)
    assert (result["resilience_level"], result["pruned"]) == (0, True)
    assert result["two_qubit_gates"] == NIGHTHAWK_CZ[4][0]


# ------------------------------------------------------ attenuation rescale


class _PlantedEstimator:
    """Exact expectations pushed through a planted per-readout slope and bias."""

    def __init__(self, slope, bias=0.0, n_actions: int = 4):
        from qiskit.primitives import StatevectorEstimator

        self._exact = StatevectorEstimator()
        self.slope = np.broadcast_to(np.asarray(slope, dtype=np.float64), (n_actions,))
        self.bias = np.broadcast_to(np.asarray(bias, dtype=np.float64), (n_actions,))
        self.options = types.SimpleNamespace(default_shots=None)
        self.job_shots: list = []  # default_shots at the time of each job

    def run(self, pubs):
        self.job_shots.append(self.options.default_shots)
        evs = np.asarray(self._exact.run(pubs).result()[0].data.evs, dtype=np.float64)
        evs = evs.reshape(self.slope.size, -1) * self.slope[:, None] + self.bias[:, None]
        pub = types.SimpleNamespace(data=types.SimpleNamespace(evs=evs))
        return types.SimpleNamespace(result=lambda: [pub])


def _planted(fake_backend, slope, bias=0.0, shots: int = 1024, **kwargs):
    """(HardwareQFunction on a planted-noise estimator, its exact fastsim twin),
    both loaded with the bundled oval driver."""
    params = np.load(OVAL_WEIGHTS)["params"]
    fast = QuantumQFunction(CIRCUIT_CFG, seed=7)
    fast.set_params(params)
    hw = hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, shots=shots, **kwargs)
    hw.set_params(params)
    hw._estimator = _PlantedEstimator(slope, bias)
    hw._estimator.options.default_shots = shots
    return hw, fast


def test_rescale_is_off_by_default(fake_backend):
    hw, fast = _planted(fake_backend, 0.8)
    obs = np.random.default_rng(0).uniform(0.0, 1.0, size=(5, 4))
    assert hw.rescale is None and hw.correction is None
    np.testing.assert_allclose(hw.q_values(obs), 0.8 * fast.expectations(obs) * fast.w + fast.b,
                               atol=1e-8)
    assert hw._estimator.job_shots == [1024]  # no calibration job
    assert hw.calibration is None


def test_global_rescale_recovers_a_planted_attenuation(fake_backend):
    hw, fast = _planted(fake_backend, 0.8, rescale=True)
    assert hw.rescale == "global"
    obs = np.random.default_rng(0).uniform(0.0, 1.0, size=(5, 4))
    np.testing.assert_allclose(hw.q_values(obs), fast.q_values(obs), atol=1e-7)
    assert hw.calibration["attenuation"] == pytest.approx(0.8, abs=1e-9)
    assert hw.correction.slope == pytest.approx(0.8, abs=1e-9)
    # ONE calibration job, at the calibration's shots, then the configured ones
    hw.q_values(obs)
    assert hw._estimator.job_shots == [hardware.CALIBRATION_MIN_SHOTS, 1024, 1024]
    assert hw._estimator.options.default_shots == 1024
    np.testing.assert_allclose(hw.expectations(obs), 0.8 * fast.expectations(obs), atol=1e-9)

    # more shots per decision than the calibration minimum: the job follows
    hw, _ = _planted(fake_backend, 0.8, shots=20_000, rescale="global")
    hw.calibrate(samples=16)
    assert hw._estimator.job_shots == [20_000]


def test_readout_rescale_corrects_each_readouts_slope_and_bias(fake_backend):
    slope, bias = [0.95, 0.90, 0.97, 0.80], [0.01, -0.02, 0.0, 0.03]
    obs = np.random.default_rng(1).uniform(0.0, 1.0, size=(6, 4))

    hw, fast = _planted(fake_backend, slope, bias, rescale="readout")
    np.testing.assert_allclose(hw.q_values(obs), fast.q_values(obs), atol=1e-6)
    np.testing.assert_allclose(hw.correction.slope, slope, atol=1e-9)
    np.testing.assert_allclose(hw.correction.bias, bias, atol=1e-9)

    # one global f cannot undo four different readouts
    hw, fast = _planted(fake_backend, slope, bias, rescale="global")
    error = np.max(np.abs(hw.q_values(obs) - fast.q_values(obs)))
    assert error > 1.0
    assert 0.80 < hw.calibration["attenuation"] < 0.97

    with pytest.raises(ValueError, match="rescale"):
        hardware.HardwareQFunction(CIRCUIT_CFG, fake_backend, rescale="per-qubit")


def test_rescale_is_refused_when_the_signal_is_gone(fake_backend):
    hw, fast = _planted(fake_backend, 0.1, rescale="global")
    obs = np.random.default_rng(2).uniform(0.0, 1.0, size=(3, 4))
    q = hw.q_values(obs)
    assert hw.rescale is None and hw.correction is None
    assert "calibrated attenuation 0.10 is below" in hw.rescale_note
    np.testing.assert_allclose(q, 0.1 * fast.expectations(obs) * fast.w + fast.b, atol=1e-8)
    assert hw.calibration["attenuation"] == pytest.approx(0.1, abs=1e-9)  # still reported


def test_rescale_reaches_lap_and_sprint_results(fake_backend):
    """Default: off, nothing calibrated. [hardware] rescale / the argument turn
    it on; the run info says what is applied and with which attenuation."""
    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=1)
    assert (result["rescale"], result["attenuation"], result["shots"]) == (None, None, 128)
    assert hardware.mitigation_text(result) == hardware.RESILIENCE_LEVELS[0]

    config = load_config()
    config["hardware"] = {**config["hardware"], "rescale": True, "calibration_samples": 16,
                          "calibration_shots": 2048}
    started: list[dict] = []
    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=1, config=config,
                                       on_start=started.append, seed_simulator=4)
    assert result["rescale"] == "global" and 0.85 < result["attenuation"] < 1.0
    assert started[0]["attenuation"] == result["attenuation"]
    assert "global attenuation rescale (calibrated f = 0.9" in hardware.mitigation_text(result)

    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=1, config=config, rescale="off")
    assert result["rescale"] is None  # the argument beats the config

    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, iterations=1,
                                  shots=64, batch=4, rescale="readout", seed_simulator=4)
    assert result["rescale"] == "readout" and 0.85 < result["attenuation"] < 1.0
    assert "per-readout attenuation rescale" in hardware.mitigation_text(result)


def test_seed_simulator_makes_a_local_lap_reproducible(fake_backend):
    def lap(seed):
        seen: list[list[float]] = []
        hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=256,
                                  max_decisions=3, seed_simulator=seed,
                                  on_decision=lambda _i, info: seen.append(info["q_values"]))
        return seen

    assert lap(11) == lap(11)
    assert lap(11) != lap(12)


# ----------------------------------------------------------- execution modes


class _Refused(Exception):
    pass


def _stub_mode(opened: list, refuse: str | None = None):
    """Session/Batch stand-in: records the backend, or refuses like the API."""

    class Mode:
        def __init__(self, backend):
            if refuse is not None:
                raise _Refused(refuse)
            self.backend = backend
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    return Mode


OPEN_PLAN = ('\'400 Client Error: Bad Request for url: .../sessions. {"errors":[{"code":1352,'
             '"message":"You are not authorized to run a session when using the open plan."')


def test_execution_mode_prefers_a_session(monkeypatch):
    opened: list = []
    monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode(opened))
    monkeypatch.setattr(qiskit_ibm_runtime, "Batch", _stub_mode(opened, refuse="unused"))
    backend = object()
    mode, name, note = hardware.open_execution_mode(backend)
    assert (name, note) == ("session", None)
    assert opened == [mode] and mode.backend is backend


def test_execution_mode_falls_back_to_batch_on_the_open_plan(monkeypatch):
    opened: list = []
    monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode(opened, refuse=OPEN_PLAN))
    monkeypatch.setattr(qiskit_ibm_runtime, "Batch", _stub_mode(opened))
    mode, name, note = hardware.open_execution_mode(object())
    assert name == "batch" and opened == [mode]
    assert note.startswith("Session unavailable on")
    assert "1352" in note and note.endswith("using a Batch")


def test_execution_mode_falls_back_to_job_mode(monkeypatch):
    opened: list = []
    monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode(opened, refuse=OPEN_PLAN))
    monkeypatch.setattr(qiskit_ibm_runtime, "Batch",
                        _stub_mode(opened, refuse="batch\nrefused too"))
    mode, name, note = hardware.open_execution_mode(object())
    assert mode is None and name == "job" and opened == []
    assert "Session unavailable" in note and "Batch unavailable" in note
    assert "refused too" not in note  # one line per reason
    assert note.endswith("using job mode (each job queues on its own)")
    assert name in hardware.EXECUTION_MODES


def test_lap_runs_in_batch_mode_when_sessions_are_refused(fake_backend, monkeypatch):
    """End to end on the fake: Session refused -> a real (local) Batch carries
    the jobs, and the result says which mode ran and why."""
    monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode([], refuse=OPEN_PLAN))
    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=2)
    assert result["decisions"] == 2
    assert result["mode"] == "batch"
    assert "Session unavailable" in result["note"] and "1352" in result["note"]


def test_lap_runs_in_job_mode_and_closes_nothing(fake_backend, monkeypatch):
    monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode([], refuse=OPEN_PLAN))
    monkeypatch.setattr(qiskit_ibm_runtime, "Batch", _stub_mode([], refuse="no batch"))
    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=2)
    assert result["decisions"] == 2
    assert result["mode"] == "job"
    assert "using job mode" in result["note"]


@pytest.mark.parametrize("kind", ["Session", "Batch"])
def test_mode_is_closed_after_every_run(fake_backend, monkeypatch, kind):
    """A real (local) Session / Batch is closed exactly once per run: after a
    finished lap, after a rollout that raised, after an aborted lap and after
    a sprint."""
    import threading

    closed: list[str] = []

    class Recording(getattr(qiskit_ibm_runtime, kind)):
        def close(self):
            closed.append(kind)
            super().close()

    monkeypatch.setattr(qiskit_ibm_runtime, kind, Recording)
    if kind == "Batch":
        monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode([], refuse=OPEN_PLAN))

    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=2)
    assert result["mode"] == kind.lower() and closed == [kind]

    def failing(_i, _info):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                  max_decisions=2, on_decision=failing)
    assert closed == [kind] * 2

    stop = threading.Event()
    stop.set()
    result = hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, shots=128,
                                       max_decisions=2, stop_event=stop)
    assert result["aborted"] and result["decisions"] == 0
    assert closed == [kind] * 3

    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, iterations=1,
                                  shots=64, batch=4)
    assert result["mode"] == kind.lower() and closed == [kind] * 4


def test_mode_is_closed_when_the_qfunction_cannot_be_built(fake_backend, monkeypatch):
    opened: list = []
    monkeypatch.setattr(qiskit_ibm_runtime, "Session", _stub_mode(opened))
    with pytest.raises(ValueError, match="resilience_level"):
        hardware.run_hardware_lap("oval", OVAL_WEIGHTS, fake_backend, max_decisions=1,
                                  resilience_level=7)
    assert len(opened) == 1 and opened[0].closed


# ------------------------------------------------------------------- SPSA


def test_spsa_minimize_converges_on_quadratic():
    target = np.array([0.3, -0.2, 0.5, 0.1, -0.4])

    def f(x):
        return float(np.sum((x - target) ** 2))

    x0 = np.zeros(5)
    seen: list[int] = []
    result = spsa.minimize(f, x0, iterations=300, seed=1, callback=lambda k, _info: seen.append(k))

    assert len(result["loss_history"]) == 300
    assert seen == list(range(300))
    assert f(result["x"]) < 0.01 * f(x0)
    np.testing.assert_allclose(result["x"], target, atol=0.05)


def test_spsa_minimize_is_deterministic():
    def f(x):
        return float(np.sum(x**2))

    r1 = spsa.minimize(f, np.ones(4), iterations=20, seed=42)
    r2 = spsa.minimize(f, np.ones(4), iterations=20, seed=42)
    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["loss_history"] == r2["loss_history"]


def test_spsa_defaults_are_the_plain_two_evaluation_loop():
    """No safeguard requested: bit for bit the textbook loop."""
    target = np.array([0.3, -0.2, 0.5, 0.1])

    def f(x):
        return float(np.sum((x - target) ** 2) + 0.1 * np.sum(x**4))

    x, losses = np.ones(4), []
    rng = np.random.default_rng(3)
    for k in range(25):
        a_k = 0.15 / (k + 1 + 2.5) ** 0.602
        c_k = 0.1 / (k + 1) ** 0.101
        delta = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
        f_plus, f_minus = f(x + c_k * delta), f(x - c_k * delta)
        g_hat = (f_plus - f_minus) / (2.0 * c_k) * delta
        x = x - a_k * g_hat
        losses.append(0.5 * (f_plus + f_minus))

    result = spsa.minimize(f, np.ones(4), iterations=25, seed=3)
    np.testing.assert_array_equal(result["x"], x)
    assert result["loss_history"] == losses
    assert result["evaluations"] == 50 and result["accepted"] == [True] * 25
    assert result["vetoed"] == [False] * 25
    assert "loss_start" not in result


def test_spsa_scale_sets_each_parameters_unit_and_freezes_zeros():
    probes: list[np.ndarray] = []

    def f(x):
        probes.append(x.copy())
        return float(np.sum(x**2))

    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    scale = np.array([1.0, 0.5, 0.0, 10.0])
    result = spsa.minimize(f, x0, iterations=1, a=0.05, c=0.1, seed=0, scale=scale)
    np.testing.assert_allclose(np.abs(probes[0] - x0), 0.1 * scale)  # probe = c * scale
    moved = np.abs(result["x"] - x0)
    assert moved[2] == 0.0 and result["x"][2] == 1.0  # scale 0: frozen
    np.testing.assert_allclose(moved / moved[0], scale)  # steps in the same units


def test_spsa_max_step_is_a_trust_region():
    def f(x):
        return float(1e4 * np.sum(x**2))  # steep: an unclipped step would be huge

    free = spsa.minimize(f, np.ones(3), iterations=1, a=0.15, seed=1)
    assert np.max(np.abs(free["x"] - 1.0)) > 10.0
    scale = np.array([1.0, 1.0, 5.0])
    held = spsa.minimize(f, np.ones(3), iterations=1, a=0.15, seed=1, max_step=0.02,
                         scale=scale)
    np.testing.assert_allclose(np.abs(held["x"] - 1.0), 0.02 * scale)  # in units of scale


def test_spsa_blocking_only_takes_steps_that_do_not_raise_the_loss():
    target = np.array([0.3, -0.2, 0.5, 0.1, -0.4])

    def f(x):
        return float(np.sum((x - target) ** 2))

    seen: list[dict] = []
    # a gain this large overshoots: without blocking the loss explodes
    wild = spsa.minimize(f, np.zeros(5), iterations=20, a=1.0, seed=2)
    assert f(wild["x"]) > 5.0 * f(np.zeros(5))
    result = spsa.minimize(f, np.zeros(5), iterations=20, a=1.0, seed=2, blocking=True,
                           callback=lambda _k, info: seen.append(info))
    assert result["loss_start"] == f(np.zeros(5))
    history = [result["loss_start"], *result["loss_history"]]
    assert all(b <= a for a, b in zip(history, history[1:], strict=False))
    assert result["loss_history"][-1] == f(result["x"])  # the loss AT the iterate
    assert 0 < sum(result["accepted"]) < 20  # some steps taken, some refused
    assert result["evaluations"] == 1 + 3 * 20
    for before, after in zip(seen, seen[1:], strict=False):
        if not after["accepted"]:
            np.testing.assert_array_equal(after["x"], before["x"])  # refused: stay put

    # allowed_increase lets through what strict blocking refuses
    loose = spsa.minimize(f, np.zeros(5), iterations=20, a=1.0, seed=2, blocking=True,
                          allowed_increase=1e9)
    assert loose["accepted"] == [True] * 20
    np.testing.assert_array_equal(loose["x"], wild["x"])


def test_spsa_accept_vetoes_steps_without_evaluating_them():
    """``accept`` refuses proposed points: the iterate stays put and, with
    blocking, the refused point is never evaluated."""
    target = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
    evaluated: list[np.ndarray] = []

    def f(x):
        evaluated.append(x.copy())
        return float(np.sum((x - target) ** 2))

    def inside(x):  # a box the iterate must not leave
        return bool(np.all(np.abs(x) <= 0.25))

    seen: list[dict] = []
    result = spsa.minimize(f, np.zeros(5), iterations=30, a=0.3, seed=2, blocking=True,
                           accept=inside, callback=lambda _k, info: seen.append(info))
    vetoed, accepted = result["vetoed"], result["accepted"]
    assert any(vetoed) and any(accepted)
    assert not any(v and a for v, a in zip(vetoed, accepted, strict=True))
    assert inside(result["x"]) and all(inside(info["x"]) for info in seen)
    assert [info["vetoed"] for info in seen] == vetoed
    # start + 2 probes per iteration + one blocking evaluation per step not vetoed
    assert result["evaluations"] == len(evaluated) == 1 + 2 * 30 + (30 - sum(vetoed))
    assert f(result["x"]) < f(np.zeros(5))  # it still descended inside the box

    # without blocking a vetoed step is simply not taken
    free = spsa.minimize(f, np.zeros(5), iterations=30, a=0.3, seed=2, accept=inside)
    assert any(free["vetoed"]) and inside(free["x"])
    assert free["accepted"] == [not v for v in free["vetoed"]]
    assert free["evaluations"] == 60


def test_spsa_resamplings_average_the_gradient_estimate():
    calls: list[int] = []
    g = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.2])

    def f(x):
        calls.append(1)
        return float(g @ x)

    def direction_error(resamplings: int) -> float:
        errors = []
        for seed in range(30):
            result = spsa.minimize(f, np.zeros(6), iterations=1, a=1.0, A=0.0, seed=seed,
                                   resamplings=resamplings)
            errors.append(np.linalg.norm(-result["x"] - g))  # x = -a_k * g_hat, a_k = 1
        return float(np.mean(errors))

    one = direction_error(1)
    assert len(calls) == 30 * 2
    calls.clear()
    eight = direction_error(8)
    assert len(calls) == 30 * 2 * 8
    assert eight < 0.5 * one  # averaging directions gives a much better gradient


def test_calibrate_gain_averages_several_probe_pairs():
    g = np.array([1.0, -2.0, 0.5, 3.0])
    calls: list[int] = []

    def f(x):
        calls.append(1)
        return float(g @ x)

    a, magnitude = spsa.calibrate_gain(f, np.zeros(4), c=0.1, target_step=0.01, A=3.0,
                                       pairs=6, seed=4)
    assert len(calls) == 12
    rng = np.random.default_rng(4)
    expected = np.mean([abs(g @ rng.choice(np.array([-1.0, 1.0]), size=4)) for _ in range(6)])
    assert magnitude == pytest.approx(expected)
    assert a == pytest.approx(0.01 * 4.0**0.602 / expected)
    # in units of scale: a frozen coordinate does not contribute to the magnitude
    scale = np.array([1.0, 1.0, 1.0, 0.0])
    _a, frozen = spsa.calibrate_gain(f, np.zeros(4), c=0.1, target_step=0.01, A=3.0,
                                     pairs=6, seed=4, scale=scale)
    rng = np.random.default_rng(4)
    expected = np.mean([abs(g[:3] @ rng.choice(np.array([-1.0, 1.0]), size=4)[:3])
                        for _ in range(6)])
    assert frozen == pytest.approx(expected)


# ------------------------------------------------------------ lap + sprint


def test_run_hardware_lap_smoke(fake_backend):
    decisions_seen: list[int] = []
    result = hardware.run_hardware_lap(
        "oval",
        OVAL_WEIGHTS,
        fake_backend,
        shots=512,
        max_decisions=40,
        on_decision=lambda i, _info: decisions_seen.append(i),
    )

    assert isinstance(result["lapped"], bool)
    assert 0 < result["decisions"] <= 40
    # ~0.1 s on the device patch; the raw 120-qubit fake took 4-30 s per decision
    assert 0.0 < result["seconds_per_decision"] < 3.0
    assert len(decisions_seen) == result["decisions"]
    trajectory = result["trajectory"]
    assert len(trajectory) >= 1
    assert all(len(state) == 4 for state in trajectory)  # [x, y, theta, v]
    if result["lapped"]:
        assert result["best_lap_s"] > 0.0
    else:
        assert result["best_lap_s"] is None
    # how it ran: local Session, raw noise, pruned circuit on the device patch
    assert result["mode"] == "session" and result["note"] is None
    assert result["backend_name"].startswith("fake_miami (4-qubit patch")
    assert result["two_qubit_gates"] == NIGHTHAWK_CZ[4][0]
    assert result["depth"] > 0
    assert (result["resilience_level"], result["pruned"]) == (0, True)


def test_spsa_sprint_keeps_the_policy(fake_backend):
    """The regression this sprint was rebuilt for: ten iterations on the
    default fake used to take the bundled oval driver from a lapping return
    to a crash (a random step over all 56 parameters). Now only the head
    moves and the guard refuses every step that would cost the simulator
    policy more than a tenth of its return. Seeded simulator, so the run is
    reproducible."""
    iters_seen: list[int] = []
    started: list[dict] = []
    result = hardware.spsa_sprint(
        "oval",
        OVAL_WEIGHTS,
        fake_backend,
        iterations=10,
        shots=1024,
        batch=16,
        on_iter=lambda k, _info: iters_seen.append(k),
        on_start=started.append,
        seed_simulator=7,
    )

    assert len(result["loss_history"]) == 10
    assert iters_seen == list(range(10))
    assert all(np.isfinite(loss) for loss in result["loss_history"])
    assert result["mode"] == "session" and result["note"] is None
    assert result["two_qubit_gates"] == NIGHTHAWK_CZ[4][0]
    assert len(started) == 1 and started[0]["mode"] == "session"

    # the policy survives: the driver still laps in most episodes
    assert result["return_before"] > 500.0
    assert result["return_after"] >= hardware.SPRINT_GUARD * result["return_before"]
    # the defaults: output head only, blocking, the simulator guard
    assert (result["groups"], result["blocking"]) == (["head"], True)
    assert result["guard"] == hardware.SPRINT_GUARD
    assert result["params"].shape == (56,)
    start = np.load(OVAL_WEIGHTS)["params"]
    moved = result["params"] - start
    assert np.all(moved[:48] == 0.0)  # circuit angles untouched
    # the head stays inside ten trust regions (in practice far inside: ~1 % of |w|)
    probe = hardware.SPRINT_HEAD_PROBE * np.mean(np.abs(start[48:52]))
    assert np.max(np.abs(moved[48:])) <= 10 * 2.0 * hardware.SPRINT_STEP_TARGET * probe + 1e-9
    accepted, vetoed = result["accepted"], result["vetoed"]
    assert len(accepted) == len(vetoed) == 10
    assert not any(a and v for a, v in zip(accepted, vetoed, strict=True))
    if not any(accepted):
        assert result["loss_after"] == result["loss_before"]
    # gain calibration, the starting point, two probes per iteration, one
    # blocking evaluation per step the guard let through, the fresh losses
    fresh = hardware.SPRINT_LOSS_EVALUATIONS * (2 if any(accepted) else 1)
    assert result["jobs"] == (2 * hardware.SPRINT_CALIBRATION_PAIRS + 1 + 2 * 10
                              + (10 - sum(vetoed)) + fresh)
    assert hardware.sprint_steps_text(result).startswith(f"{sum(accepted)}/10")


def test_sprint_targets_follow_the_recipe_the_weights_were_trained_with(tmp_path):
    """The TD target of a sprint is the trainer's: discount, reward scale and
    advantage-learning term come from the weights' sidecar, else the config."""
    import json

    config = load_config()
    # no sidecar: the config's [training], which is the plain double-DQN target
    plain_recipe = {"gamma": config["training"]["gamma"], "action_gap": 0.0,
                    "reward_scale": 1.0}
    assert hardware._td_recipe(tmp_path / "quantum_oval.npz", config) == plain_recipe
    # the bundled oval driver brings the recipe its sidecar records (since the
    # October 2026 re-bundling: advantage learning, action_gap 0.8)
    recorded = json.loads(OVAL_WEIGHTS.with_suffix("").with_suffix(".meta.json")
                          .read_text(encoding="utf-8"))["training"]
    assert hardware._td_recipe(OVAL_WEIGHTS, config) == {
        "gamma": recorded["gamma"], "action_gap": recorded.get("action_gap", 0.0),
        "reward_scale": recorded.get("reward_scale", 1.0)}
    config["training"]["action_gap"] = 0.5
    assert hardware._td_recipe(tmp_path / "quantum_oval.npz", config)["action_gap"] == 0.5
    weights = tmp_path / "quantum_oval.npz"
    weights.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(
        {"training": {"gamma": 0.9, "action_gap": 0.8, "reward_scale": 0.1}}))
    recipe = hardware._td_recipe(weights, config)
    assert recipe == {"gamma": 0.9, "action_gap": 0.8, "reward_scale": 0.1}

    config = load_config()
    fast = QuantumQFunction(config["circuit"], seed=42)
    fast.set_params(np.load(OVAL_WEIGHTS)["params"])
    obs, act, plain = hardware._collect_batch(fast, "oval", config, 16, 42)
    same = hardware._collect_batch(fast, "oval", config, 16, 42, recipe=plain_recipe)
    np.testing.assert_array_equal(same[2], plain)  # the default recipe changes nothing
    obs_g, act_g, gapped = hardware._collect_batch(
        fast, "oval", config, 16, 42,
        recipe={"gamma": config["training"]["gamma"], "action_gap": 0.8})
    np.testing.assert_array_equal(obs_g, obs)
    np.testing.assert_array_equal(act_g, act)
    q = fast.q_values(obs)
    lost = q.max(axis=1) - q[np.arange(16), act]
    assert np.any(lost > 0.0)  # the batch holds exploratory actions
    np.testing.assert_allclose(gapped, plain - 0.8 * lost)


def test_spsa_sprint_guard_refuses_steps_that_cost_return(fake_backend, monkeypatch):
    """The guard asks the exact simulator whether the driver still drives —
    and a refused step costs no job."""
    start = np.load(OVAL_WEIGHTS)["params"]
    seeds: list[int] = []

    def fragile_return(qfunc, _track, _config, seed, max_steps=600, episodes=1):
        seeds.append(seed)
        return 1000.0 if np.array_equal(qfunc.get_params(), start) else 500.0

    monkeypatch.setattr(hardware, "_greedy_return", fragile_return)
    kwargs = {"iterations": 2, "shots": 64, "batch": 4, "seed_simulator": 3}
    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, **kwargs)
    assert result["vetoed"] == [True, True] and result["accepted"] == [False, False]
    np.testing.assert_array_equal(result["params"], start)
    assert (result["return_before"], result["return_after"]) == (1000.0, 1000.0)
    assert result["loss_after"] == result["loss_before"]
    assert result["jobs"] == (2 * hardware.SPRINT_CALIBRATION_PAIRS + 1 + 2 * 2
                              + hardware.SPRINT_LOSS_EVALUATIONS)
    assert "2 refused by the simulator guard" in hardware.sprint_steps_text(result)
    # before, one call per proposed step, after: all on the same episodes
    assert seeds == [42] * 4

    # a guard that tolerates the loss of half the return lets the steps through
    seeds.clear()
    config = load_config()
    config["hardware"] = {**config["hardware"], "spsa_guard": 0.5, "spsa_blocking": False}
    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, config=config, **kwargs)
    assert result["guard"] == 0.5 and result["vetoed"] == [False, False]
    assert result["accepted"] == [True, True] and result["return_after"] == 500.0
    assert hardware.sprint_steps_text(result) == "2/2"

    # guard = 0: the simulator is not asked at all during the sprint
    seeds.clear()
    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, guard=0.0, **kwargs)
    assert result["guard"] == 0.0 and result["vetoed"] == [False, False]
    assert seeds == [42, 42]
    with pytest.raises(ValueError, match="guard"):
        hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, guard=1.5, **kwargs)


def test_spsa_sprint_descends_a_planted_device_error(fake_backend, monkeypatch):
    """The safeguards must not turn the sprint into a no-op: a sprint that
    never moved would satisfy every "keeps the policy" check above. On a
    device WITHOUT shot noise whose only error is a planted slope and bias per
    readout — exactly what the output head can absorb — the loss is
    deterministic, so the sprint (head only, calibrated gain, trust region,
    blocking) has to take steps, move the head and lower the loss. The guard
    is off here: it may only veto steps, and it is tested on its own above."""
    slope, bias = [0.8, 0.9, 0.85, 0.95], [0.02, -0.03, 0.0, 0.01]
    monkeypatch.setattr(hardware, "_make_estimator",
                        lambda *_args, **_kwargs: _PlantedEstimator(slope, bias))
    start = np.load(OVAL_WEIGHTS)["params"]
    result = hardware.spsa_sprint("oval", OVAL_WEIGHTS, fake_backend, iterations=10,
                                  shots=64, batch=16, guard=0.0)

    assert result["vetoed"] == [False] * 10
    assert sum(result["accepted"]) >= 5  # all 10 with qiskit 2.5 / numpy 2.5
    assert result["loss_after"] < 0.9 * result["loss_before"]  # 67.9 -> 51.2
    # no noise: the blocking rule sees the true loss, which can only go down
    history = [result["loss_before"], *result["loss_history"]]
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:], strict=False))
    assert result["loss_history"][-1] == pytest.approx(result["loss_after"])
    moved = result["params"] - start
    assert np.all(moved[:48] == 0.0)  # head only
    probe = hardware.SPRINT_HEAD_PROBE * np.mean(np.abs(start[48:52]))
    assert np.max(np.abs(moved[48:])) > 0.5 * probe  # it really stepped (1.4 probes)
    # start + calibration + 2 probes and 1 blocking evaluation per iteration + fresh losses
    assert result["jobs"] == (2 * hardware.SPRINT_CALIBRATION_PAIRS + 1 + 3 * 10
                              + 2 * hardware.SPRINT_LOSS_EVALUATIONS)


# -------------------------------------------------------------------- CLI


@pytest.mark.parametrize("command", ["lap", "sprint"])
def test_cli_help_renders(command, capsys):
    """argparse %-formats every help string: a literal percent sign in one of
    them (the guard's "90 %") crashed ``sprint --help`` with a TypeError."""
    with pytest.raises(SystemExit) as exit_info:
        hardware.main([command, "--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--resilience" in out and "--rescale" in out
    if command == "sprint":
        assert "--no-guard" in out and "--no-blocking" in out and "--groups" in out
        assert f"{100 * hardware.SPRINT_GUARD:.0f} %" in out


def test_cli_lap_reports_mode_circuit_and_mitigation(capsys):
    hardware.main(["lap", "--track", "oval", "--fake", "--max-decisions", "2",
                   "--shots", "128", "--resilience", "1"])
    out = capsys.readouterr().out
    assert "backend: fake_miami (fake, local simulation)" in out
    assert "runs on: fake_miami (4-qubit patch" in out
    assert "execution mode: session" in out
    assert "circuit: 12 two-qubit gates" in out and "(light-cone pruned)" in out
    assert "resilience level 1 (TREX" in out
    assert "decisions: 2" in out


def test_cli_no_prune_and_unknown_fake_name(capsys):
    hardware.main(["lap", "--fake", "--max-decisions", "1", "--shots", "128", "--no-prune"])
    out = capsys.readouterr().out
    assert "circuit: 16 two-qubit gates" in out and "pruned" not in out
    assert "resilience level 0 (raw device noise" in out

    with pytest.raises(SystemExit):
        hardware.main(["lap", "--fake", "--fake-name", "fake_bogus"])
    assert "unknown fake backend 'fake_bogus'" in capsys.readouterr().err


def test_lap_sprint_and_cli_run_the_weights_own_depth(tmp_path, fake_backend, capsys):
    """A driver brings its own depth: 6-block weights under the default
    4-block config are transpiled (and pruned) as the 6-block circuit — by the
    lap, the sprint and the CLI — instead of failing in set_params."""
    weights = tmp_path / "quantum_deep.npz"  # no sidecar: depth from the parameter count
    params = QuantumQFunction({"n_qubits": 4, "n_layers": 6}, seed=1).get_params()
    np.savez(weights, params=params)
    cz = int(lightcone.live_gates(4, 6, 4)["cz"].sum())
    assert cz == 20  # all but the last ring; 12 at the profile's 4 blocks

    config = load_config()
    result = hardware.run_hardware_lap("oval", weights, fake_backend, shots=64,
                                       max_decisions=1, config=config)
    assert result["decisions"] == 1 and result["pruned"] is True
    assert result["two_qubit_gates"] == cz
    assert config["circuit"]["n_layers"] == 4  # the caller's config is not touched

    result = hardware.spsa_sprint("oval", weights, fake_backend, iterations=1, shots=64,
                                  batch=4)
    assert result["two_qubit_gates"] == cz and result["params"].shape == params.shape

    hardware.main(["lap", "--fake", "--weights", str(weights), "--max-decisions", "1",
                   "--shots", "64"])
    out = capsys.readouterr().out
    assert ("circuit shape from the weights: 6 blocks, 4 actions "
            "(profile: 4 blocks, 4 actions)") in out
    assert f"circuit: {cz} two-qubit gates" in out and "decisions: 1" in out

    # a parameter count that fits no depth: a usage error, before any backend
    bad = tmp_path / "quantum_bad.npz"
    np.savez(bad, params=np.zeros(57))
    with pytest.raises(SystemExit):
        hardware.main(["lap", "--fake", "--weights", str(bad)])
    assert "57 parameters, which fits no depth" in capsys.readouterr().err
    with pytest.raises(ValueError, match="fits no depth"):
        hardware.run_hardware_lap("oval", bad, fake_backend, max_decisions=1)


def _record_envs_and_observations(monkeypatch):
    """Capture every env the hardware path builds and every observation batch
    it evaluates on the backend."""
    envs, seen = [], []
    build_env, q_values = hardware._build_env, hardware.HardwareQFunction.q_values

    def recording_build(*args, **kwargs):
        envs.append(build_env(*args, **kwargs))
        return envs[-1]

    def recording_q(self, obs):
        seen.append(np.array(obs))
        return q_values(self, obs)

    monkeypatch.setattr(hardware, "_build_env", recording_build)
    monkeypatch.setattr(hardware.HardwareQFunction, "q_values", recording_q)
    return envs, seen


def test_lap_and_cli_adopt_the_weights_recorded_observation(tmp_path, fake_backend, capsys,
                                                            monkeypatch):
    """A driver trained on engineered features is fed those features on the
    hardware path too — the observation its sidecar records, not the
    profile's rays (the rule the server and records apply)."""
    weights = tmp_path / "quantum_feat.npz"
    np.savez(weights, params=QuantumQFunction(CIRCUIT_CFG, seed=2).get_params())
    recorded = {"ray_angles_deg": [-30.0, 30.0],
                "features": ["rays", "speed", "curvature_ahead"]}
    (tmp_path / "quantum_feat.meta.json").write_text(
        json.dumps({"observation": recorded}), encoding="utf-8")
    names = ["ray -30°", "ray +30°", "speed", "curvature ahead"]
    envs, seen = _record_envs_and_observations(monkeypatch)

    config = load_config()
    result = hardware.run_hardware_lap("oval", weights, fake_backend, shots=64,
                                       max_decisions=3, config=config)
    assert result["decisions"] == 3
    assert envs[-1].feature_names == names
    assert len(seen) == 3 and all(obs.shape == (1, 4) for obs in seen)
    # exactly what an env under the recorded observation shows at the start
    # line — and not what the profile's three rays + speed would
    track = Track.load("oval", config["track"]["resample_spacing"])
    seed = int(config["training"]["seed"])
    recorded_cfg = {**config, "observation": {**config["observation"], **recorded}}
    assert np.array_equal(seen[0], RacingEnv(track, recorded_cfg, n_envs=1, seed=seed).reset())
    assert not np.array_equal(seen[0], RacingEnv(track, config, n_envs=1, seed=seed).reset())
    assert config["observation"]["ray_angles_deg"] == [-60.0, 0.0, 60.0]  # caller's: untouched

    hardware.spsa_sprint("oval", weights, fake_backend, iterations=1, shots=64, batch=4)
    assert envs[-1].feature_names == names

    seen.clear()
    hardware.main(["lap", "--fake", "--weights", str(weights), "--max-decisions", "3",
                   "--shots", "64"])
    out = capsys.readouterr().out
    assert ("observation from the weights' sidecar: 2 rays, features "
            "['rays', 'speed', 'curvature_ahead'] (profile: 3 rays, features "
            "['rays', 'speed'])") in out
    assert "decisions: 3" in out and "circuit shape from the weights" not in out
    assert envs[-1].feature_names == names and len(seen) == 3


def test_weights_without_a_recorded_observation_run_as_before(tmp_path, fake_backend,
                                                             capsys, monkeypatch):
    """No sidecar, or one that records no observation: the config's own
    observation, and bit-identical decisions to running the config as it is
    (what the hardware path did before it looked at the sidecar)."""
    weights = tmp_path / "quantum_plain.npz"
    np.savez(weights, params=QuantumQFunction(CIRCUIT_CFG, seed=2).get_params())
    config = load_config()
    assert hardware._weights_config(config, weights)["observation"] is config["observation"]
    for path in sorted(hardware.WEIGHTS_DIR.glob("quantum_*.npz")):  # the bundle
        meta = path.with_suffix("").with_suffix(".meta.json")
        if meta.is_file() and "observation" in json.loads(meta.read_text(encoding="utf-8")):
            continue
        tag = path.stem.rsplit("_q", 1)
        profile = load_config(f"q{tag[1]}" if len(tag) == 2 and tag[1].isdigit() else None)
        assert hardware._weights_config(profile, path)["observation"] \
            is profile["observation"], path.name

    def lap():
        decisions = []
        result = hardware.run_hardware_lap(
            "oval", weights, fake_backend, shots=64, max_decisions=3, config=config,
            seed_simulator=5, on_decision=lambda i, info: decisions.append(
                (info["action"], info["q_values"], info["state"])))
        return decisions, result["trajectory"]

    envs, _seen = _record_envs_and_observations(monkeypatch)
    now = lap()
    assert envs[-1].feature_names == ["ray -60°", "ray 0°", "ray +60°", "speed"]
    with monkeypatch.context() as patch:
        patch.setattr(hardware, "_weights_config", lambda config, path: config)
        assert lap() == now

    hardware.main(["lap", "--fake", "--weights", str(weights), "--max-decisions", "1",
                   "--shots", "64"])
    assert "observation from the weights' sidecar" not in capsys.readouterr().out


def test_cli_fake_name_implies_fake(capsys, monkeypatch):
    """Naming a fake without --fake must not reach for (and bill) a real device."""

    def no_real_backend(*_args, **_kwargs):
        raise AssertionError("the CLI contacted IBM Quantum")

    monkeypatch.setattr(hardware, "_real_backend", no_real_backend)
    hardware.main(["lap", "--fake-name", "FakeFez", "--max-decisions", "1", "--shots", "128"])
    out = capsys.readouterr().out
    assert "backend: fake_fez (fake, local simulation)" in out
    assert "runs on: fake_fez (4-qubit patch" in out
    assert "decisions: 1" in out
