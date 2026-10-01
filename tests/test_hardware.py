"""Hardware execution path, exercised entirely on LOCAL fake backends.

No network, no IBM account: ``get_backend(use_fake=True)`` returns a
``FakeBackendV2`` that the hardware path simulates on Aer (as an n-qubit
device patch for the 120-156 qubit fakes), and runtime Sessions enter local
testing mode. Account-side behaviour (Open Plan accounts cannot open a
Session) is covered with stubs. Skipped wholesale when qiskit-ibm-runtime
(the [hardware] extra) is not installed.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

qiskit_ibm_runtime = pytest.importorskip("qiskit_ibm_runtime")

from traqmania import hardware  # noqa: E402
from traqmania.agents.quantum import lightcone  # noqa: E402
from traqmania.agents.quantum.qdqn import QuantumQFunction  # noqa: E402
from traqmania.agents.training import spsa  # noqa: E402
from traqmania.config import load_config  # noqa: E402

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
    assert hardware.DEFAULT_FAKE == "fake_nighthawk"
    assert fake_backend.name == "fake_nighthawk"
    assert fake_backend.num_qubits >= 5
    assert "cz" in fake_backend.operation_names  # current IBM QPUs are CZ-based


@pytest.mark.parametrize(
    ("name", "cls_name"),
    [
        ("fake_fez", "FakeFez"),
        ("FakeFez", "FakeFez"),
        ("ibm_fez", "FakeFez"),
        ("fake_nighthawk", "FakeNighthawk"),
        ("fake_manila", "FakeManilaV2"),
        ("FakeManilaV2", "FakeManilaV2"),
        ("fake_manila_v2", "FakeManilaV2"),
    ],
)
def test_fake_name_resolution_accepts_every_spelling(name, cls_name):
    assert hardware._fake_class(name).__name__ == cls_name


def test_every_listed_fake_resolves():
    names = hardware.available_fakes()
    assert {"fake_nighthawk", "fake_fez", "fake_manila"} <= set(names)
    for name in names:
        assert hardware._fake_class(name).backend_name == name


def test_unknown_fake_name_raises_with_the_available_ones():
    """No silent fallback to some other device (the old path fell back to manila)."""
    with pytest.raises(ValueError, match="unknown fake backend 'fake_bogus'") as excinfo:
        hardware.get_backend(use_fake=True, fake_name="fake_bogus")
    assert "fake_nighthawk" in str(excinfo.value) and "fake_fez" in str(excinfo.value)


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
    assert hw.backend_name.startswith(f"fake_nighthawk ({n_qubits}-qubit patch")
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
    assert device == "fake_nighthawk" and len(qubits) == 4
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
    assert hardware.backend_label(sims[0]).startswith("fake_nighthawk (4-qubit patch")


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
    assert result["backend_name"].startswith("fake_nighthawk (4-qubit patch")
    assert result["two_qubit_gates"] == NIGHTHAWK_CZ[4][0]
    assert result["depth"] > 0
    assert (result["resilience_level"], result["pruned"]) == (0, True)


def test_spsa_sprint_smoke(fake_backend):
    iters_seen: list[int] = []
    started: list[dict] = []
    result = hardware.spsa_sprint(
        "oval",
        OVAL_WEIGHTS,
        fake_backend,
        iterations=3,
        shots=256,
        batch=8,
        on_iter=lambda k, _info: iters_seen.append(k),
        on_start=started.append,
    )

    assert len(result["loss_history"]) == 3
    assert iters_seen == [0, 1, 2]
    assert all(np.isfinite(loss) for loss in result["loss_history"])
    assert np.isfinite(result["return_before"])
    assert np.isfinite(result["return_after"])
    assert result["params"].shape == (56,)
    assert result["mode"] == "session" and result["note"] is None
    assert result["two_qubit_gates"] == NIGHTHAWK_CZ[4][0]
    assert len(started) == 1 and started[0]["mode"] == "session"


# -------------------------------------------------------------------- CLI


def test_cli_lap_reports_mode_circuit_and_mitigation(capsys):
    hardware.main(["lap", "--track", "oval", "--fake", "--max-decisions", "2",
                   "--shots", "128", "--resilience", "1"])
    out = capsys.readouterr().out
    assert "backend: fake_nighthawk (fake, local simulation)" in out
    assert "runs on: fake_nighthawk (4-qubit patch" in out
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
