"""IBM Quantum hardware execution path: real/fake backends, inference, SPSA sprints.

This module is the bridge from traQmania's numpy fast path to actual quantum
processors (or their local fake twins) via ``qiskit-ibm-runtime``:

- :func:`get_backend` — a real IBMBackend through ``QiskitRuntimeService`` or a
  local noise-model twin from the fake provider (default: ``fake_miami``, a
  calibration snapshot of a 120-qubit square-lattice, CZ-based Nighthawk
  processor — the family the circuit's CZ ring embeds into with zero SWAPs).
- :func:`local_simulator` — the Aer twin a fake device is actually simulated
  on, noise model built ONCE: the whole device for the 5-7 qubit fakes, an
  n-qubit "device patch" (only the physical qubits the routed circuit touches)
  for everything bigger — above all the 120-156 qubit ones, whose full noise
  model is far too big to simulate per job.
- :class:`HardwareQFunction` — the same ``QFunction`` contract (and the same
  flat ``[lam, theta, w, b]`` layout, 56 parameters at 4 qubits) as
  ``QuantumQFunction``, but INFERENCE ONLY: expectation values come from
  Estimator PUBs on an ISA-transpiled circuit (by default the light-cone
  pruned one: same expectation values, fewer two-qubit gates). Gradients are
  deliberately not implemented — a param-shift gradient would cost 2 circuit
  evaluations per circuit parameter per batch, which is why hardware
  fine-tuning uses SPSA (two evaluations, period).
- :func:`run_hardware_lap` — greedy rollout of one car with every steering
  decision made by the quantum backend, inside a runtime ``Session`` where the
  account allows one (else a ``Batch``, else plain job mode).
- :func:`spsa_sprint` — a short TD-loss fine-tune: replay batch and double-DQN
  targets are computed ONCE in the exact fastsim simulator, then SPSA descends
  the hardware-evaluated MSE loss — by default on the output head only, with
  a blocking rule that refuses steps the hardware loss does not confirm and
  a guard that refuses steps which would cost the policy its return on the
  exact simulator.

Error mitigation is explicit: ``resilience_level`` 0 (the default here) shows
the raw device noise, 1 adds TREX readout mitigation, 2 adds ZNE with gate
twirling on top of TREX — the runtime's own default would be 1. ``rescale`` (default off)
divides the measured expectations by an attenuation calibrated against the
exact simulator with one extra job, globally or per readout
(``traqmania.agents.quantum.noise`` has the model behind it).

Import hygiene: importing this module must NOT import qiskit — all qiskit /
qiskit-ibm-runtime imports live inside functions. Run the CLI with
``python -m traqmania.hardware lap|sprint --track oval [--fake] ...``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from traqmania.agents.training import spsa

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Default local twin: fake_miami, a calibration snapshot of a real Nighthawk
# processor (square lattice, CZ, 120 qubits; qiskit-ibm-runtime >= 0.47, as is
# fake_berlin). fake_nighthawk (>= 0.44) has the same lattice but IBM's
# placeholder error values ("not intended to represent typical Nighthawk error
# values"); fake_fez / fake_marrakesh / fake_kingston / ... are heavy-hex Herons.
DEFAULT_FAKE = "fake_miami"

# Fakes with more qubits than this are simulated as a device patch (see
# local_simulator). Simulating the whole device costs 4-30 s per evaluation on
# the 120-156 qubit fakes and ~14 s for a 10-qubit circuit on the 16-qubit
# fake_guadalupe; the patch ~0.1 s at 4 qubits, ~1 s at 10.
PATCH_MIN_QUBITS = 7

# Retired CX-era Falcon fakes (V2 API), smallest first: where a NAMED fake that
# is too small for the circuit falls through to — 5 qubits, then 7, then 16.
_FAKE_FALLBACKS = (
    "FakeManilaV2", "FakeLimaV2", "FakeBelemV2", "FakeQuitoV2",  # 5 qubits
    "FakeLagosV2", "FakeNairobiV2", "FakeJakartaV2",  # 7 qubits
    "FakeGuadalupeV2",  # 16 qubits
)

# Execution modes, most to least exclusive; Open Plan accounts get no Session.
EXECUTION_MODES = ("session", "batch", "job")

# Estimator resilience levels (the same numbering as qiskit-ibm-runtime).
RESILIENCE_LEVELS = {
    0: "raw device noise, no error mitigation",
    1: "TREX readout-error mitigation",
    2: "TREX + ZNE with gate twirling",
}

# Attenuation calibration (HardwareQFunction.calibrate): one extra job of this
# many random inputs, at no fewer shots than this.
CALIBRATION_SAMPLES = 64
CALIBRATION_MIN_SHOTS = 8192
# Below this fitted attenuation the rescale is refused: most of the signal is
# gone and dividing by f would mainly amplify shot noise.
MIN_ATTENUATION = 0.3

# SPSA sprint (spsa_sprint): the parameter groups it moves by default, the
# probe size per group (circuit angles in rad; the head as a fraction of mean
# |w|), the first-step size in probe sizes, the probe pairs the gain is
# calibrated on, the greedy episodes behind return_before / return_after (and
# behind the guard), the share of its exact-simulator return a step must keep
# (the guard; 0 = off) and the fresh evaluations averaged into loss_before /
# loss_after.
SPRINT_GROUPS = ("head",)
SPRINT_ANGLE_PROBE = 0.05
SPRINT_HEAD_PROBE = 0.02
SPRINT_STEP_TARGET = 0.3
SPRINT_CALIBRATION_PAIRS = 4
SPRINT_EVAL_EPISODES = 12
SPRINT_GUARD = 0.9
SPRINT_LOSS_EVALUATIONS = 3

# Layout seeds tried per transpilation; the fewest two-qubit gates wins.
_TRANSPILE_SEEDS = tuple(range(8))

_SERVICE_HELP = (
    "Could not reach IBM Quantum. To run on real hardware:\n"
    "  1. Create an account at https://quantum.cloud.ibm.com and copy your API token.\n"
    "  2. Either export QISKIT_IBM_TOKEN=<token>, or save it once with\n"
    "     QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=...).\n"
    "  3. Re-run. Or pass --fake to use a local noise-model twin instead."
)

# local_simulator cache: {key: simulator} plus, per simulator, the device it
# twins and (for a device patch) the physical qubits it spans.
_SIM_LOCK = threading.Lock()
_SIMULATORS: dict[tuple, Any] = {}
_SIM_INFO: dict[int, tuple[str, tuple[int, ...] | None]] = {}


# --------------------------------------------------------------------- backends


def ensure_qiskit_imported() -> None:
    """Import qiskit from the calling thread — which must be a LONG-LIVED one.

    qiskit's compiled extension pins process-wide lazy state to the thread
    that first imports it; if that thread exits, the next hardware job on any
    other thread segfaults (observed deterministically with qiskit 2.5.0 and
    re-checked on qiskit 2.5.2 / qiskit-ibm-runtime 0.50.0, client-side
    Estimator included: SIGSEGV on the SECOND hardware job of a process whose
    first job ran on a since-exited thread; the crash site varies —
    ``SparseObservable.to_sparse_list``, Aer's ``NoiseModel.from_backend``,
    ``get_standard_gate_name_mapping`` while a fake backend loads). The demo
    server runs each hardware job on a short-lived worker thread, so it calls
    this first from the session thread, which outlives every worker.
    Idempotent; the first call pays the one-off qiskit import cost (~1 s).
    """
    import qiskit  # noqa: F401
    import qiskit_ibm_runtime  # noqa: F401


def _fake_classes() -> dict[str, type]:
    """{class name: class} of every fake device in the installed fake provider."""
    from qiskit_ibm_runtime import fake_provider
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

    classes = {}
    for name in sorted(dir(fake_provider)):
        cls = getattr(fake_provider, name)
        if isinstance(cls, type) and issubclass(cls, FakeBackendV2) and cls is not FakeBackendV2:
            classes[name] = cls
    return classes


def available_fakes() -> list[str]:
    """Names of the fake devices this qiskit-ibm-runtime ships ('fake_fez', ...)."""
    return sorted(str(cls.backend_name) for cls in _fake_classes().values())


def _fake_key(name: str) -> str:
    """Spelling-insensitive key: 'fake_fez' / 'FakeFez' / 'ibm_fez' -> 'fakefez'."""
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    key = key.removeprefix("ibm").removeprefix("fake")
    return "fake" + key


def _fake_class(fake_name: str) -> type:
    """Resolve a fake backend by name; unknown names raise (no silent fallback).

    Accepts the class name ('FakeFez', 'FakeManilaV2') and the snake-case
    device name ('fake_fez', 'fake_manila') alike — the retired Falcon-era
    fakes carry a ``V2`` class suffix, the current Heron / Nighthawk ones do
    not, so both spellings are tried.
    """
    classes = _fake_classes()
    if fake_name in classes:
        return classes[fake_name]
    by_key: dict[str, type] = {}
    for cls_name, cls in classes.items():
        by_key.setdefault(_fake_key(cls_name), cls)
        by_key.setdefault(_fake_key(str(cls.backend_name)), cls)
    key = _fake_key(fake_name)
    for candidate in (key, key + "v2", key.removesuffix("v2")):
        if candidate in by_key:
            return by_key[candidate]
    raise ValueError(
        f"unknown fake backend {fake_name!r}; this qiskit-ibm-runtime ships: "
        f"{', '.join(available_fakes())}"
    )


def _fake_backend(fake_name: str | None = None, min_qubits: int = 5):
    """The named fake device (default: DEFAULT_FAKE), big enough for the circuit.

    Only when the named device has fewer than ``min_qubits`` qubits does this
    fall through — to the smallest retired Falcon fake that fits, else the
    default fake — and it says so in a ``UserWarning``.
    """
    backend = _fake_class(fake_name or DEFAULT_FAKE)()
    if backend.num_qubits >= min_qubits:
        return backend
    tried = [backend.name]
    for name in (*_FAKE_FALLBACKS, DEFAULT_FAKE):
        try:
            candidate = _fake_class(name)()
        except Exception:  # noqa: BLE001 - a fallback missing/broken in this runtime: keep trying
            continue
        tried.append(candidate.name)
        if candidate.num_qubits >= min_qubits:
            warnings.warn(
                f"fake backend {backend.name} has only {backend.num_qubits} qubits, the "
                f"circuit needs {min_qubits}; using {candidate.name} "
                f"({candidate.num_qubits} qubits) instead",
                stacklevel=3,
            )
            return candidate
    raise RuntimeError(
        f"no {min_qubits}+ qubit fake backend found in qiskit_ibm_runtime.fake_provider "
        f"(tried {tried})"
    )


def _real_backend(name: str | None, min_qubits: int = 5):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    try:
        service = QiskitRuntimeService(token=token) if token else QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(f"{_SERVICE_HELP}\n(underlying error: {exc})") from exc
    try:
        if name:
            return service.backend(name)
        return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
    except Exception as exc:
        raise RuntimeError(
            f"could not get a backend from IBM Quantum "
            f"({'name=' + name if name else 'least busy'}): {exc}\n{_SERVICE_HELP}"
        ) from exc


def get_backend(
    name: str | None = None,
    use_fake: bool = False,
    fake_name: str = DEFAULT_FAKE,
    min_qubits: int = 5,
):
    """A qiskit backend: real via ``QiskitRuntimeService`` or a local fake twin.

    ``use_fake=True`` returns a ``FakeBackendV2`` (noise model + coupling map
    of an IBM device, simulated locally — no account needed). ``fake_name``
    takes any spelling ('fake_fez', 'FakeFez', 'fake_manila', 'FakeManilaV2');
    an unknown name raises ``ValueError`` listing the available fakes. Current
    devices are CZ-based: 'fake_miami' (default; a Nighthawk, square lattice,
    120 qubits — the CZ ring needs no SWAPs) and the heavy-hex Herons ('fake_fez',
    'fake_marrakesh', 'fake_torino', ...); 'fake_manila', 'fake_lagos' and the
    other ``...V2`` classes are retired CX-era Falcon devices. A named fake
    with fewer than ``min_qubits`` qubits is replaced by the smallest
    known-good fake that is big enough, with a ``UserWarning``. Otherwise the
    runtime service is reached with a token from ``QISKIT_IBM_TOKEN`` or the
    saved account; ``name`` picks a device, empty means least busy. Raises
    ``RuntimeError`` with setup instructions when the service is unavailable.
    """
    if use_fake:
        return _fake_backend(fake_name, min_qubits=min_qubits)
    return _real_backend(name, min_qubits=min_qubits)


def _two_qubit_gates(circuit) -> int:
    """Number of two-qubit gates in a circuit (what dominates the device error)."""
    return sum(
        1
        for inst in circuit.data
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier"
    )


def _transpile_best(circuit, backend, seeds: Sequence[int] = _TRANSPILE_SEEDS):
    """ISA circuit for ``backend`` with the fewest two-qubit gates, then lowest depth.

    Routing a ring onto a heavy-hex device needs SWAPs and their number varies
    a lot with the layout seed (37-44 CZ at 4 qubits, 64-149 at 10), so a few
    seeds are tried; on the square lattice the first already is SWAP-free and
    the search stops there.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    swap_free = _two_qubit_gates(circuit)
    best, best_cost = None, None
    for seed in seeds:
        pass_manager = generate_preset_pass_manager(
            optimization_level=2, backend=backend, seed_transpiler=seed
        )
        isa = pass_manager.run(circuit)
        cost = (_two_qubit_gates(isa), isa.depth())
        if best_cost is None or cost < best_cost:
            best, best_cost = isa, cost
        if cost[0] <= swap_free:
            break
    return best


def _build_circuit(n_qubits: int, n_layers: int, n_actions: int, prune: bool):
    """The canonical circuit, or its light-cone-pruned twin (same parameter names)."""
    if prune:
        from traqmania.agents.quantum import lightcone

        return lightcone.pruned_circuit(n_qubits, n_layers, n_actions).circuit
    from traqmania.agents.quantum import circuit as circuit_mod

    return circuit_mod.build_circuit(n_qubits, n_layers)


def _circuit_shape(circuit_cfg: dict) -> tuple[int, int, int]:
    """(n_qubits, n_layers, n_actions) of a [circuit] table (or a full config)."""
    cfg = circuit_cfg.get("circuit", circuit_cfg)
    n_qubits = int(cfg.get("n_qubits", 4))
    return n_qubits, int(cfg.get("n_layers", 4)), int(cfg.get("n_actions", min(4, n_qubits)))


def _patch_qubits(device, circuit) -> tuple[int, ...]:
    """Physical qubits of ``device`` the routed ``circuit`` lands on, sorted.

    Every qubit the best ISA circuit touches (SWAP ancillas included) plus the
    home of every circuit qubit, so idle circuit qubits keep a place too.
    """
    isa = _transpile_best(circuit, device)
    touched = {isa.find_bit(qubit).index for inst in isa.data for qubit in inst.qubits}
    touched.update(isa.layout.initial_index_layout(filter_ancillas=True))
    return tuple(sorted(touched))


def _patch_backend(device, qubits: Sequence[int]):
    """A ``BackendV2`` describing only ``qubits`` of ``device`` (renumbered 0..k-1).

    Its Target keeps those qubits' gate/readout errors, durations and T1/T2
    and drops everything else, so ``AerSimulator.from_backend`` builds a
    k-qubit noise model instead of the whole processor's.
    """
    from qiskit.providers import BackendV2, Options
    from qiskit.transpiler import Target

    index = {qubit: i for i, qubit in enumerate(qubits)}
    properties = device.target.qubit_properties
    target = Target(
        num_qubits=len(qubits),
        dt=device.target.dt,
        qubit_properties=[properties[q] for q in qubits] if properties else None,
    )
    for name in device.target.operation_names:
        kept = {
            tuple(index[q] for q in qargs): props
            for qargs, props in device.target[name].items()
            if qargs is not None and all(q in index for q in qargs)
        }
        if kept:
            target.add_instruction(device.target.operation_from_name(name), kept)

    class DevicePatch(BackendV2):
        """The sub-device spanned by a few physical qubits of a larger one."""

        @property
        def target(self):
            return target

        @property
        def max_circuits(self):
            return None

        @classmethod
        def _default_options(cls):
            return Options()

        def run(self, run_input, **options):
            raise NotImplementedError(
                "a device patch only describes hardware; simulate it through "
                "AerSimulator.from_backend (see local_simulator)"
            )

    return DevicePatch(
        name=f"{device.name}_patch{len(qubits)}",
        backend_version=getattr(device, "backend_version", None),
    )


def local_simulator(
    device,
    n_qubits: int = 4,
    n_layers: int = 4,
    n_actions: int | None = None,
    prune: bool = True,
):
    """Aer twin of a fake device with its noise model built ONCE (cached).

    Devices of up to ``PATCH_MIN_QUBITS`` qubits (the 5- and 7-qubit Falcon
    fakes) are simulated whole. Bigger ones — up to the 120-156 qubit Heron /
    Nighthawk fakes — as a DEVICE PATCH: the circuit — canonical, or
    light-cone pruned with ``prune`` — is routed onto the full device with a
    few layout seeds, the layout with the fewest two-qubit gates is kept, and
    the simulator models exactly the physical qubits it touches (same gate
    and readout errors, T1/T2, coupling). The circuit then transpiles onto
    the patch as it would onto the device, at a fraction of the simulation
    cost. Cached per (device, circuit shape).
    """
    from qiskit_aer import AerSimulator

    if n_actions is None:
        n_actions = min(4, int(n_qubits))
    key: tuple = (type(device).__name__, device.name, getattr(device, "backend_version", None))
    patch = device.num_qubits > PATCH_MIN_QUBITS
    if patch:
        key += (int(n_qubits), int(n_layers), int(n_actions), bool(prune))
    with _SIM_LOCK:
        simulator = _SIMULATORS.get(key)
        if simulator is None:
            qubits = None
            source = device
            if patch:
                circuit = _build_circuit(int(n_qubits), int(n_layers), int(n_actions), prune)
                qubits = _patch_qubits(device, circuit)
                source = _patch_backend(device, qubits)
            simulator = AerSimulator.from_backend(source)
            _SIMULATORS[key] = simulator
            _SIM_INFO[id(simulator)] = (str(device.name), qubits)
    return simulator


def execution_backend(backend, circuit_cfg: dict, prune: bool = True):
    """The backend a circuit actually runs on.

    A fake device becomes its cached Aer twin (:func:`local_simulator`); real
    backends — and anything that already is a simulator — pass through.
    """
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

    if not isinstance(backend, FakeBackendV2):
        return backend
    n_qubits, n_layers, n_actions = _circuit_shape(circuit_cfg)
    return local_simulator(backend, n_qubits, n_layers, n_actions, prune)


def backend_label(backend) -> str:
    """Display name: the device, plus the physical qubits of a device patch."""
    device, qubits = _SIM_INFO.get(id(backend), (None, None))
    if device is None:
        return str(getattr(backend, "name", backend))
    if qubits is None:
        return device
    return f"{device} ({len(qubits)}-qubit patch: physical qubits {', '.join(map(str, qubits))})"


def _brief(exc: BaseException, limit: int = 200) -> str:
    """First line of an exception message, for one-line notes."""
    lines = str(exc).strip().splitlines()
    text = lines[0].strip() if lines else type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def open_execution_mode(backend) -> tuple[Any, str, str | None]:
    """Open the most exclusive execution mode ``backend`` grants.

    Tries a runtime ``Session`` (dedicated access; local testing mode on fake
    backends), then a ``Batch``, then plain job mode. Open Plan accounts
    cannot open a Session (error 1352, raised when the Session is
    constructed) but may use Batch and job mode. Returns ``(mode, name,
    note)``: the Session/Batch to hand to the Estimator (``None`` in job mode,
    where the Estimator takes the backend and every job queues on its own),
    its name from ``EXECUTION_MODES``, and — only after a fallback — a note
    saying which modes were refused and why. Close a returned Session/Batch
    when done.
    """
    from qiskit_ibm_runtime import Batch, Session

    name = getattr(backend, "name", backend)
    refused: list[str] = []
    for label, opener in (("session", Session), ("batch", Batch)):
        try:
            mode = opener(backend=backend)
        except Exception as exc:  # noqa: BLE001 - plan/backend restrictions surface as many types
            refused.append(f"{label.capitalize()} unavailable on {name} ({_brief(exc)})")
            continue
        note = "; ".join(refused) + f"; using a {label.capitalize()}" if refused else None
        return mode, label, note
    return None, "job", "; ".join(refused) + "; using job mode (each job queues on its own)"


def _mode_closed_error(exc: BaseException) -> bool:
    """Does this runtime error say the Session/Batch was closed by the service
    (IBM error 1217: maximum lifetime or interactive timeout exceeded)?"""
    text = str(exc)
    return "1217" in text or "has been closed" in text


def _is_ibm_backend(backend) -> bool:
    from qiskit_ibm_runtime import IBMBackend

    return isinstance(backend, IBMBackend)


def _make_estimator(mode, backend, shots: int, resilience_level: int):
    """Estimator primitive for ``mode`` (a Session/Batch, or the backend itself).

    The client-side ``executor_estimator.Estimator`` (added in
    qiskit-ibm-runtime 0.50.0; it honours ``resilience_level`` on fake
    backends too) when available, else the server-side ``EstimatorV2`` that
    0.50 deprecates.
    """
    try:
        from qiskit_ibm_runtime.executor_estimator import Estimator

        legacy = False
    except ImportError:  # older runtime: only the server-side primitive exists
        from qiskit_ibm_runtime import EstimatorV2 as Estimator

        legacy = True
    estimator = Estimator(mode=mode)
    estimator.options.default_shots = int(shots)
    # The legacy primitive cannot mitigate in local testing mode and warns on
    # every job when asked to; only bother it there when mitigation was asked for.
    if not legacy or resilience_level != 0 or _is_ibm_backend(backend):
        estimator.options.resilience_level = int(resilience_level)
    return estimator


# ---------------------------------------------------------------- Q-function


class HardwareQFunction:
    """Inference-only ``QFunction`` on an IBM (or fake) backend via the Estimator.

    Same flat parameter layout as ``QuantumQFunction``:
    ``[lam (L*n), theta (L*n*2), w (A), b (A)]`` with ``A = [circuit]
    n_actions`` (default min(4, n)) and ``Q_a = w[a] * <Z_a> + b[a]``. The
    circuit is transpiled to ISA form ONCE at construction; every
    ``q_values`` call is a single Estimator job whose PUB batches all
    observation rows (parameter bindings of shape ``(B, ...)``) against all
    ``A`` ``Z_a`` observables.

    ``prune_light_cone`` (default on) transpiles the light-cone-pruned circuit
    (``lightcone.pruned_circuit``): gates that cannot influence any readout
    are dropped, the expectation values are mathematically identical and the
    device executes fewer two-qubit gates (12 instead of 16 CZ at 4 qubits).
    ``resilience_level`` is the Estimator's error mitigation: 0 (default) raw
    device noise, 1 TREX, 2 TREX + ZNE with gate twirling. ``rescale`` (default off)
    corrects the measured expectations before the output head with an
    attenuation fitted by :meth:`calibrate`: ``"global"`` (or true) reads
    ``Q_a = w[a] * <Z_a> / f + b[a]`` with one f for all readouts — the
    global-depolarizing picture — and ``"readout"`` ``Q_a = w[a] * (<Z_a> -
    bias_a) / slope_a + b[a]`` with each readout's own fit (readout error
    differs from qubit to qubit, and it is that difference which re-ranks
    actions). Without a ``session`` a fake
    device is swapped for its Aer twin (:func:`execution_backend`) and jobs
    run in job mode; with one (a runtime Session or Batch), ``backend`` must
    be the backend it was opened on. ``two_qubit_gates`` / ``depth`` describe
    the transpiled circuit, ``backend_name`` where it runs. A hand-made
    simulator works as ``backend`` only with an IBM basis (e.g.
    ``AerSimulator(basis_gates=["rz", "sx", "x", "cz"])``): the client-side
    Estimator rejects circuits in other gates.
    """

    def __init__(
        self,
        circuit_cfg: dict,
        backend,
        shots: int = 1024,
        session=None,
        resilience_level: int = 0,
        prune_light_cone: bool = True,
        rescale: bool | str | None = False,
    ):
        cfg = circuit_cfg.get("circuit", circuit_cfg)
        self.n_qubits = int(cfg.get("n_qubits", 4))
        self.n_layers = int(cfg.get("n_layers", 4))
        self.seed = int(cfg.get("seed", 7))
        self.shots = int(shots)
        from traqmania.agents.quantum import noise as noise_mod

        # Attenuation mitigation: None (off) | "global" | "readout". The
        # correction itself is fitted by calibrate() — one extra job.
        self.rescale = noise_mod.rescale_mode(rescale)
        self.correction = None  # noise.ReadoutCorrection once calibrated
        self.calibration: dict | None = None  # the fit behind it
        self.rescale_note: str | None = None  # why a requested rescale is off
        self.jobs = 0  # Estimator jobs run (since the last seed_simulator call)
        self._simulator_seed: int | None = None
        self.resilience_level = int(resilience_level)
        if self.resilience_level not in RESILIENCE_LEVELS:
            raise ValueError(
                f"resilience_level must be one of {sorted(RESILIENCE_LEVELS)}, "
                f"got {resilience_level!r}"
            )
        self.pruned = bool(prune_light_cone)

        self.n_features = self.n_qubits
        # Z_a readout on the first n_actions qubits (default: the first 4)
        self.n_actions = int(cfg.get("n_actions", min(4, self.n_qubits)))
        if self.n_actions > self.n_qubits:
            raise ValueError(
                f"[circuit] n_actions = {self.n_actions} needs at least as many "
                f"qubits, got n_qubits = {self.n_qubits}"
            )

        # Same initialization (and rng stream) as the numpy fast path.
        rng = np.random.default_rng(self.seed)
        self.lam = np.full((self.n_layers, self.n_qubits), np.pi, dtype=np.float64)
        self.theta = rng.uniform(-0.1, 0.1, size=(self.n_layers, self.n_qubits, 2))
        self.w = np.ones(self.n_actions, dtype=np.float64)
        self.b = np.zeros(self.n_actions, dtype=np.float64)

        from traqmania.agents.quantum import circuit as circuit_mod

        if session is None:
            backend = execution_backend(backend, cfg, self.pruned)
        self.backend = backend
        self.backend_name = backend_label(backend)

        qc = _build_circuit(self.n_qubits, self.n_layers, self.n_actions, self.pruned)
        self._isa_circuit = _transpile_best(qc, backend)
        self._isa_observables = [
            obs.apply_layout(self._isa_circuit.layout)
            for obs in circuit_mod.observables(self.n_qubits)[: self.n_actions]
        ]
        # Bind exactly the parameters the ISA circuit kept: column k of the
        # row [x (L*n), theta (L*n*2)] for each, found by vector name + index.
        self._isa_params = tuple(self._isa_circuit.parameters)
        self._param_columns = np.array(
            [
                p.index + (0 if p.vector.name == "x" else self.lam.size)
                for p in self._isa_params
            ],
            dtype=np.int64,
        )
        self.two_qubit_gates = _two_qubit_gates(self._isa_circuit)
        self.depth = int(self._isa_circuit.depth())
        self._mode = session  # the Session/Batch the Estimator runs in (None: job mode)
        self.mode_reopens = 0
        self._estimator = _make_estimator(
            session if session is not None else backend, backend, self.shots,
            self.resilience_level,
        )

    @property
    def mode(self):
        """The Session/Batch currently in use (re-opened ones included), or None."""
        return self._mode

    def close_mode(self) -> None:
        """Close the Session/Batch this Q-function runs in, if any."""
        if self._mode is not None:
            try:
                self._mode.close()
            finally:
                self._mode = None

    def reopen_mode(self) -> None:
        """Replace a Session/Batch the service has closed by a fresh one of the
        same kind and rebuild the Estimator in it.

        A Batch or Session has a maximum lifetime (10 minutes on the Open
        Plan) and an interactive timeout; a rollout of 140 sequential
        decisions outlives both, and the service then refuses further jobs
        with error 1217 ("Session has been closed"). Re-opening keeps the
        rollout going at the price of one more queue wait.
        """
        if self._mode is None:
            raise RuntimeError("no Session/Batch to re-open (job mode)")
        kind = type(self._mode)
        with contextlib.suppress(Exception):  # already closed server-side
            self._mode.close()
        self._mode = kind(backend=self.backend)
        self.mode_reopens += 1
        self._estimator = _make_estimator(self._mode, self.backend, self.shots,
                                          self.resilience_level)

    def _pub(self, obs: np.ndarray) -> tuple:
        """Estimator PUB for a batch of observations: (circuit, observables, bindings)."""
        batch = obs.shape[0]
        # x[b, l*n + i] = lam[l, i] * obs[b, i] — encoding angles bound per row.
        x = (self.lam[None, :, :] * obs[:, None, :]).reshape(batch, -1)
        theta = np.repeat(self.theta.reshape(1, -1), batch, axis=0)
        values = np.concatenate([x, theta], axis=1)[:, self._param_columns]
        # Observables shaped (A, 1) broadcast against (B,) bindings -> evs (A, B).
        return (
            self._isa_circuit,
            [[o] for o in self._isa_observables],
            {self._isa_params: values},
        )

    def expectations(self, obs: np.ndarray) -> np.ndarray:
        """Raw readout expectations <Z_a> from the backend: (B, F) -> (B, A)."""
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        batch = obs.shape[0]
        if self._simulator_seed is not None:
            simulator = getattr(self._estimator.options, "simulator", None)
            if simulator is not None:
                simulator.seed_simulator = self._simulator_seed + self.jobs
        self.jobs += 1
        try:
            result = self._estimator.run([self._pub(obs)]).result()[0]
        except Exception as exc:  # noqa: BLE001 - runtime error types vary by version
            if self._mode is None or not _mode_closed_error(exc):
                raise
            self.reopen_mode()
            result = self._estimator.run([self._pub(obs)]).result()[0]
        evs = np.asarray(result.data.evs, dtype=np.float64).reshape(self.n_actions, batch)
        return evs.T

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for a batch of observations: (B, F) -> (B, A).

        With ``rescale`` the expectations pass through the calibrated
        correction first (calibrating on the first call if need be).
        """
        if self.rescale is not None and self.correction is None:
            self.calibrate()  # may refuse and turn the rescale off
        expectations = self.expectations(obs)
        if self.correction is not None:
            expectations = self.correction.apply(expectations)
        return expectations * self.w + self.b

    def seed_simulator(self, seed: int | None) -> None:
        """Make a LOCAL simulation reproducible (no effect on a real device).

        Job ``k`` from now on samples with ``seed + k``: one fixed seed would
        hand every job the same random numbers and so correlate the shot
        noise of consecutive, similar decisions. ``None`` leaves the
        simulator unseeded again.
        """
        self._simulator_seed = None if seed is None else int(seed)
        self.jobs = 0

    def calibrate(
        self, samples: int = CALIBRATION_SAMPLES, shots: int | None = None, seed: int = 0
    ) -> dict:
        """Fit the device's attenuation with ONE extra job; returns the fit.

        Runs ``samples`` random observations through the backend with the
        CURRENT parameters and compares them with the exact fastsim values
        (``noise.fit_attenuation``): ``<Z>_device ~ f * <Z>_exact`` over all
        readouts — a global-depolarizing summary of gate and readout error —
        and a slope and bias per readout. The job uses ``shots`` shots per
        input (default: the Q-function's, but at least
        ``CALIBRATION_MIN_SHOTS``). Stores the fit as ``calibration`` and,
        when ``rescale`` is on, the ``correction`` that ``q_values`` applies:
        ``<Z> / f`` ("global") or ``(<Z> - bias_a) / slope_a`` ("readout").
        A fitted slope below ``MIN_ATTENUATION`` (the signal is mostly gone;
        dividing by it would only amplify noise) turns ``rescale`` off and
        says so in ``rescale_note``. The attenuation is a property of
        (backend, circuit), so one calibration serves a whole lap or sprint.
        It needs the classical simulator — a tool for circuits small enough
        to simulate, which every traQmania circuit is.
        """
        from traqmania.agents.quantum import noise
        from traqmania.agents.quantum.qdqn import QuantumQFunction

        fast = QuantumQFunction(
            {"n_qubits": self.n_qubits, "n_layers": self.n_layers, "n_actions": self.n_actions},
            seed=self.seed,
        )
        fast.set_params(self.get_params())
        obs = noise.calibration_inputs(self.n_features, samples, seed)
        shots = max(self.shots, CALIBRATION_MIN_SHOTS) if shots is None else int(shots)
        options = self._estimator.options
        options.default_shots = shots
        try:
            measured = self.expectations(obs)
        finally:
            options.default_shots = self.shots
        fit = noise.fit_attenuation(measured, fast.expectations(obs), shots=shots)
        self.calibration = fit
        if self.rescale is not None:
            try:
                correction = noise.ReadoutCorrection.from_fit(fit, self.rescale)
            except ValueError:  # a non-positive slope: nothing left to rescale
                correction = None
            if correction is None or correction.min_slope < MIN_ATTENUATION:
                self.rescale_note = (
                    f"rescale off: calibrated attenuation {fit['attenuation']:.2f} is below "
                    f"{MIN_ATTENUATION} (too little signal left to rescale)"
                )
                self.rescale, self.correction = None, None
            else:
                self.correction = correction
        return fit

    def grad_selected(
        self, obs: np.ndarray, action_idx: np.ndarray, upstream: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError(
            "HardwareQFunction is inference-only: a parameter-shift gradient needs "
            "2 evaluations per circuit parameter = "
            f"{2 * (self.lam.size + self.theta.size)} extra Estimator "
            "jobs per batch on real hardware (minutes of QPU time per DQN update). "
            "Fine-tune on hardware with traqmania.hardware.spsa_sprint instead, which "
            "needs exactly 2 jobs per iteration regardless of parameter count."
        )

    @property
    def n_params(self) -> int:
        return self.lam.size + self.theta.size + self.w.size + self.b.size

    def get_params(self) -> np.ndarray:
        """Copy of the flat parameter vector, shape (P,)."""
        return np.concatenate([self.lam.ravel(), self.theta.ravel(), self.w, self.b])

    def set_params(self, params: np.ndarray) -> None:
        """Load a flat parameter vector, shape (P,)."""
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"params must have shape ({self.n_params},), got {params.shape}")
        n_lam = self.lam.size
        n_theta = self.theta.size
        n_w = self.w.size
        self.lam = params[:n_lam].reshape(self.lam.shape).copy()
        self.theta = params[n_lam : n_lam + n_theta].reshape(self.theta.shape).copy()
        self.w = params[n_lam + n_theta : n_lam + n_theta + n_w].copy()
        self.b = params[n_lam + n_theta + n_w :].copy()


# -------------------------------------------------------------------- rollout


def _build_env(track_name: str, config: dict, n_envs: int, seed: int):
    from traqmania.env.racing_env import RacingEnv
    from traqmania.env.track import Track

    track = Track.load(track_name, config["track"]["resample_spacing"])
    return RacingEnv(track, config, n_envs=n_envs, seed=seed)


def _start_execution(
    config: dict,
    backend,
    shots: int,
    resilience_level: int | None,
    prune_light_cone: bool | None,
    rescale: bool | str | None = None,
    params: np.ndarray | None = None,
    seed_simulator: int | None = None,
) -> tuple[HardwareQFunction, Any, dict]:
    """Open an execution mode on ``backend`` and build the Q-function in it.

    ``resilience_level`` / ``prune_light_cone`` / ``rescale`` default to
    ``[hardware] resilience_level`` (0), ``prune_light_cone`` (true) and
    ``rescale`` (false) of ``config``. ``params`` are loaded into the
    Q-function; with ``rescale`` on, the attenuation is then calibrated — one
    extra job of ``[hardware] calibration_samples`` inputs at no fewer than
    ``calibration_shots`` shots. ``seed_simulator`` pins a local simulation's
    shot noise. Returns ``(qfunc, mode, info)``: ``mode`` is the
    Session/Batch to close afterwards (``None`` in job mode), ``info`` the
    facts a caller reports — ``{backend_name, mode, note, two_qubit_gates,
    depth, resilience_level, pruned, shots, rescale, attenuation}`` with
    ``rescale`` None | "global" | "readout" (what is actually applied) and
    ``attenuation`` the calibrated f (None without a calibration).
    """
    from traqmania.agents.quantum import noise as noise_mod

    hw_cfg = config.get("hardware", {})
    if resilience_level is None:
        resilience_level = int(hw_cfg.get("resilience_level", 0))
    if prune_light_cone is None:
        prune_light_cone = bool(hw_cfg.get("prune_light_cone", True))
    if rescale is None:
        rescale = hw_cfg.get("rescale", False)
    rescale = noise_mod.rescale_mode(rescale)
    backend = execution_backend(backend, config["circuit"], prune_light_cone)
    mode, mode_name, note = open_execution_mode(backend)
    try:
        qfunc = HardwareQFunction(
            config["circuit"], backend, shots=shots, session=mode,
            resilience_level=resilience_level, prune_light_cone=prune_light_cone,
            rescale=rescale,
        )
        if seed_simulator is not None:
            qfunc.seed_simulator(seed_simulator)
        if params is not None:
            qfunc.set_params(params)
        if qfunc.rescale is not None:
            qfunc.calibrate(
                samples=int(hw_cfg.get("calibration_samples", CALIBRATION_SAMPLES)),
                shots=max(qfunc.shots,
                          int(hw_cfg.get("calibration_shots", CALIBRATION_MIN_SHOTS))),
            )
            if qfunc.rescale_note:
                note = f"{note}; {qfunc.rescale_note}" if note else qfunc.rescale_note
    except BaseException:
        if mode is not None:
            mode.close()
        raise
    info = {
        "backend_name": qfunc.backend_name,
        "mode": mode_name,
        "note": note,
        "two_qubit_gates": qfunc.two_qubit_gates,
        "depth": qfunc.depth,
        "resilience_level": qfunc.resilience_level,
        "pruned": qfunc.pruned,
        "shots": qfunc.shots,
        "rescale": qfunc.rescale,
        "attenuation": None if qfunc.calibration is None
        else float(qfunc.calibration["attenuation"]),
    }
    return qfunc, mode, info


def _weights_config(config: dict, weights_path: str | Path) -> dict:
    """``config`` as the weights file needs it (``runtime.with_weights_config``):
    the observation its sidecar records overlaid on the config's, and its own
    circuit depth and action count — a driver brings all three along, so the
    same profile runs 4-block and 6-block weights, plain-ray and
    engineered-feature drivers alike. Raises ``ValueError`` for weights that
    fit no depth or another qubit count."""
    from traqmania.server.runtime import with_weights_config

    return with_weights_config(config, Path(weights_path))


def mitigation_text(info: dict) -> str:
    """One line on the error mitigation of a run, from its ``on_start`` info:
    the Estimator's resilience level plus the attenuation rescale, if any."""
    text = RESILIENCE_LEVELS[info["resilience_level"]]
    if info.get("rescale"):
        text = text.removesuffix(", no error mitigation")  # the rescale is one
        kind = "per-readout" if info["rescale"] == "readout" else "global"
        text += f" + {kind} attenuation rescale (calibrated f = {info['attenuation']:.3f})"
    return text


def run_hardware_lap(
    track_name: str,
    weights_path: str | Path,
    backend,
    shots: int = 1024,
    max_decisions: int | None = None,
    on_decision: Callable[[int, dict], None] | None = None,
    stop_event: Any = None,
    config: dict | None = None,
    resilience_level: int | None = None,
    prune_light_cone: bool | None = None,
    on_start: Callable[[dict], None] | None = None,
    rescale: bool | str | None = None,
    seed_simulator: int | None = None,
) -> dict:
    """Drive ONE car greedily with every decision evaluated on ``backend``.

    Opens a runtime ``Session`` (local testing mode on fake backends), falling
    back to a ``Batch`` and then to plain job mode where the account or
    backend refuses one — the mode used is returned as ``mode``, the reason
    for any fallback as ``note`` — and rolls out until the first completed
    lap, the episode ending, or ``max_decisions``. ``on_start(info)`` fires
    once the circuit is transpiled (and, with ``rescale``, calibrated), before
    the first decision, with ``{backend_name, mode, note, two_qubit_gates,
    depth, resilience_level, pruned, shots, rescale, attenuation}``.
    ``on_decision(i, info)`` fires after each decision with the
    action taken, the Q-values, the car state and per-decision latency.
    ``stop_event`` (optional): ``threading.Event``-like; when set, the rollout
    stops between decisions (cooperative cancellation) and ``aborted`` is
    True. ``config`` (optional) is the fully-resolved config the weights were
    trained under (circuit size + observation geometry); defaults to
    ``load_config()``. What the weights file brings along wins over
    ``config`` (``runtime.with_weights_config``): the observation its sidecar
    records, and its own circuit DEPTH and action count (the sidecar's
    ``circuit`` block, else its parameter count).
    ``resilience_level`` (0 raw noise | 1 TREX | 2 TREX +
    ZNE with gate twirling) and ``prune_light_cone`` default to the config's
    ``[hardware]`` values (0 / true). ``rescale`` (default ``[hardware]
    rescale``, false) divides the measured expectations by an attenuation
    calibrated with one extra job before the first decision: ``"global"`` (or
    true) by the one fitted f, ``"readout"`` by each readout's own slope after
    removing its bias — see ``HardwareQFunction.calibrate``. ``shots`` is the
    number of shots PER DECISION. ``seed_simulator`` pins the shot noise of a
    local simulation. Returns ``{lapped, best_lap_s, decisions,
    seconds_per_decision, trajectory, aborted}`` plus the ``on_start`` fields.
    """
    if config is None:
        from traqmania.config import load_config

        config = load_config()
    config = _weights_config(config, weights_path)
    seed = int(config["training"]["seed"])
    env = _build_env(track_name, config, n_envs=1, seed=seed)
    if max_decisions is None:
        max_decisions = env.max_decisions
    params = np.load(weights_path)["params"]

    qfunc, mode, run_info = _start_execution(
        config, backend, shots, resilience_level, prune_light_cone, rescale, params,
        seed_simulator,
    )
    try:
        if on_start is not None:
            on_start(dict(run_info))

        obs = env.reset()
        trajectory: list[list[float]] = [env.state[0].tolist()]
        lapped = False
        best_lap_s: float | None = None
        decisions = 0
        aborted = False
        t0 = time.perf_counter()

        for i in range(int(max_decisions)):
            if stop_event is not None and stop_event.is_set():
                aborted = True
                break
            t_dec = time.perf_counter()
            q = qfunc.q_values(obs)
            action = int(np.argmax(q[0]))
            obs, _reward, done, info = env.step(np.array([action]))
            decisions += 1

            if info["lap"][0] >= 1:
                lapped = True
                best_lap_s = float(info["last_lap_time"][0])
            if not done[0]:  # after done the env auto-respawns; don't record that pose
                trajectory.append(env.state[0].tolist())
            if on_decision is not None:
                on_decision(
                    i,
                    {
                        "action": action,
                        "q_values": q[0].tolist(),
                        "state": trajectory[-1],
                        "off_track": bool(info["off_track"][0]),
                        "lap": int(info["lap"][0]),
                        "seconds": time.perf_counter() - t_dec,
                    },
                )
            if lapped or done[0]:
                break

        elapsed = time.perf_counter() - t0
    finally:
        if mode is not None:
            qfunc.close_mode()

    return {
        "lapped": lapped,
        "best_lap_s": best_lap_s,
        "decisions": decisions,
        "seconds_per_decision": elapsed / max(1, decisions),
        "trajectory": trajectory,
        "aborted": aborted,
        "mode_reopens": qfunc.mode_reopens,
        **run_info,
    }


# ---------------------------------------------------------------- SPSA sprint


def _greedy_return(
    qfunc, track_name: str, config: dict, seed: int, max_steps: int = 600, episodes: int = 1
) -> float:
    """Mean total reward of ``episodes`` distinct greedy episodes under ``qfunc``
    (one env per episode, each from its own spawn jitter, fixed seed)."""
    env = _build_env(track_name, config, n_envs=episodes, seed=seed)
    obs = env.reset()
    total = np.zeros(episodes)
    alive = np.ones(episodes, dtype=bool)
    for _ in range(max_steps):
        action = np.argmax(qfunc.q_values(obs), axis=1)
        obs, reward, done, _info = env.step(action)
        total[alive] += np.asarray(reward)[alive]
        alive &= ~np.asarray(done, dtype=bool)
        if not alive.any():
            break
    return float(total.mean())


def _td_recipe(weights_path: str | Path, config: dict) -> dict:
    """What defines the TD target the weights were trained on: ``{gamma,
    action_gap, reward_scale}`` from the ``training`` table of the weights'
    ``.meta.json`` sidecar (``train_headless`` records the resolved recipe
    there), else from ``config["training"]``. A sprint that fine-tuned an
    advantage-learning driver (``action_gap``) on the plain double-DQN target
    would spend its steps closing the action gaps the training opened."""
    import json

    training = dict(config["training"])
    meta_path = Path(weights_path).with_suffix("").with_suffix(".meta.json")
    if meta_path.is_file():
        recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("training")
        if isinstance(recorded, dict):
            training.update(recorded)
    return {
        "gamma": float(training["gamma"]),
        "action_gap": float(training.get("action_gap", 0.0)),
        "reward_scale": float(training.get("reward_scale", 1.0)),
    }


def _collect_batch(
    qfunc, track_name: str, config: dict, batch: int, seed: int, epsilon: float = 0.2,
    recipe: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay batch from a fastsim rollout: (obs, action, double-DQN TD target).

    States come from a greedy rollout with epsilon-greedy noise mixed in (so the
    batch covers more than the on-policy tube); targets use the same parameter
    snapshot as both online and target network — fine for a short sprint.
    ``recipe`` (:func:`_td_recipe`; default: ``[training] gamma`` alone) makes
    them the targets of the trainer that produced the weights: its discount,
    its reward scale and its advantage-learning term.
    """
    n_envs = 4
    env = _build_env(track_name, config, n_envs=n_envs, seed=seed)
    rng = np.random.default_rng(seed)
    if recipe is None:
        recipe = {"gamma": float(config["training"]["gamma"])}
    gamma = recipe["gamma"]
    action_gap = recipe.get("action_gap", 0.0)
    reward_scale = recipe.get("reward_scale", 1.0)

    obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
    obs = env.reset()
    while len(obs_list) * n_envs < 4 * batch:
        greedy = np.argmax(qfunc.q_values(obs), axis=1)
        random_a = rng.integers(qfunc.n_actions, size=n_envs)
        explore = rng.random(n_envs) < epsilon
        actions = np.where(explore, random_a, greedy)
        next_obs, reward, done, _info = env.step(actions)
        obs_list.append(obs)
        act_list.append(actions)
        rew_list.append(reward)
        next_list.append(next_obs)
        done_list.append(done)
        obs = next_obs

    all_obs = np.concatenate(obs_list)
    all_act = np.concatenate(act_list)
    all_rew = np.concatenate(rew_list)
    all_next = np.concatenate(next_list)
    all_done = np.concatenate(done_list).astype(np.float64)

    idx = rng.choice(all_obs.shape[0], size=batch, replace=False)
    obs_b, act_b = all_obs[idx], all_act[idx]

    # Double-DQN targets, computed ONCE with the exact simulator.
    rows = np.arange(batch)
    a_star = np.argmax(qfunc.q_values(all_next[idx]), axis=1)
    q_next = qfunc.q_values(all_next[idx])[rows, a_star]
    reward = all_rew[idx] if reward_scale == 1.0 else all_rew[idx] * reward_scale
    target_b = reward + gamma * (1.0 - all_done[idx]) * q_next
    if action_gap > 0.0:  # advantage learning, as in DQNTrainer._update
        q_now = qfunc.q_values(obs_b)
        target_b = target_b - action_gap * (q_now.max(axis=1) - q_now[rows, act_b])
    return obs_b, act_b, target_b


def _sprint_scale(fast, groups: Sequence[str]) -> np.ndarray:
    """Per-parameter probe size of the sprint: ``SPRINT_ANGLE_PROBE`` rad for
    the circuit groups, ``SPRINT_HEAD_PROBE * mean|w|`` for the output head,
    0 (frozen) for every group not in ``groups``."""
    slices = fast.param_groups()
    unknown = sorted(set(groups) - set(slices))
    if unknown or not groups:
        raise ValueError(
            f"spsa groups must be a non-empty subset of {sorted(slices)}, got {list(groups)}"
        )
    scale = np.zeros(fast.n_params)
    for name in groups:
        if name == "head":
            scale[slices[name]] = SPRINT_HEAD_PROBE * max(float(np.mean(np.abs(fast.w))), 1.0)
        else:
            scale[slices[name]] = SPRINT_ANGLE_PROBE
    return scale


def spsa_sprint(
    track_name: str,
    init_weights_path: str | Path,
    backend,
    iterations: int = 30,
    shots: int = 1024,
    batch: int = 16,
    on_iter: Callable[[int, dict], None] | None = None,
    step_target: float = SPRINT_STEP_TARGET,
    stop_event: Any = None,
    config: dict | None = None,
    resilience_level: int | None = None,
    prune_light_cone: bool | None = None,
    on_start: Callable[[dict], None] | None = None,
    rescale: bool | str | None = None,
    groups: Sequence[str] | None = None,
    blocking: bool | None = None,
    seed_simulator: int | None = None,
    guard: float | None = None,
) -> dict:
    """Short TD-loss SPSA fine-tune of trained weights ON the backend.

    The expensive-but-exact parts run ONCE in fastsim (replay batch collection
    and double-DQN targets — with the discount, reward scale and
    advantage-learning term the weights' ``.meta.json`` sidecar records, else
    the config's); the hardware only evaluates the MSE TD loss, one Estimator
    job of ``batch`` parameter bindings x A observables per evaluation.

    The first version of this sprint — plain SPSA over all 56 parameters, the
    gain calibrated on one probe pair — wrecked the bundled oval driver: of 16
    seeded ten-iteration runs on the default fake, 10 lost more than a fifth
    of their simulator return, one of them after a mis-calibrated gain moved
    the circuit angles by radians. Two things come together. One evaluation
    of the loss (16 rows x 1024 shots) scatters by 10-17 %, about as much as
    the two probes differ, so a step is largely a random walk. And a driver
    trained without noise can sit on action gaps so thin that a random step
    of a hundredth of a radian reshuffles which episodes it finishes. What
    keeps a short sprint from wrecking it:

    - ``groups`` (default ``[hardware] spsa_groups``, else ``("head",)``):
      the parameter groups SPSA moves, out of ``lam`` | ``theta`` | ``head``
      (``QuantumQFunction.param_groups``). The output head ``w, b`` is the
      default because a device's error is, to first order, a slope and a bias
      per readout — exactly what the head can absorb — and 2A parameters are
      few enough for two-evaluation gradients to mean something. Probes are
      sized per group: ``SPRINT_HEAD_PROBE`` of mean |w| for the head,
      ``SPRINT_ANGLE_PROBE`` rad for circuit angles.
    - gain calibration over ``SPRINT_CALIBRATION_PAIRS`` probe pairs (one
      pair can land on a flat direction and inflate every later step), so the
      first step moves each parameter by about ``step_target`` probe sizes,
      plus a trust region of twice that.
    - ``blocking`` (default ``[hardware] spsa_blocking``, else true): a step
      is only taken if the hardware loss at the proposed parameters is not
      worse than at the current ones — one more job per iteration.
    - ``guard`` (default ``[hardware] spsa_guard``, else ``SPRINT_GUARD``; 0
      turns it off): a step is refused — before any job is spent on it — if
      it would take the policy's greedy return on the EXACT simulator below
      ``guard`` times its starting value. The loss only sees a few replay
      rows; whether the car still drives is a different question, and for
      circuits this small the simulator answers it for the price of a few
      rollouts. The guard evaluates the very episodes ``return_before`` /
      ``return_after`` report, so ``return_after >= guard * return_before``
      holds BY CONSTRUCTION: it is a safety rule, not a measurement of how
      the sprint generalizes to other episodes.

    Jobs: 2 per calibration pair, 2 per iteration, with blocking 1 more per
    iteration that passed the guard and 1 for the starting point, then
    ``SPRINT_LOSS_EVALUATIONS`` each for the before/after loss — plus the
    rescale calibration, if on (``jobs`` in the result counts them).
    ``loss_history[k]`` is the hardware loss at the iterate after iteration
    ``k`` (without blocking: the mean of that iteration's two probes);
    ``loss_before`` / ``loss_after`` are means of FRESH evaluations at the
    initial and final parameters, so their difference is not biased by the
    blocking rule having selected on a noisy measurement (they are one and
    the same number when no step was taken).

    ``return_before`` / ``return_after`` are exact-simulator greedy returns,
    the mean over ``SPRINT_EVAL_EPISODES`` distinct episodes (a single episode
    of a marginal driver flips between a full run and an early crash on the
    smallest parameter change). Note the deliberate asymmetry: the loss is
    evaluated on the NOISY backend while the returns use the exact simulator,
    so a sprint that compensates hardware noise can trade away fastsim return
    — the guard bounds how much.

    ``stop_event`` (optional): ``threading.Event``-like; when set, the SPSA
    loop stops between iterations (cooperative cancellation) and the result
    reflects the iterations completed so far. ``config`` (optional) is the
    fully-resolved config the weights were trained under; defaults to
    ``load_config()`` — the recorded observation, circuit depth and action
    count are the weights file's own, as in :func:`run_hardware_lap`.
    Execution mode (Session, else
    Batch, else job mode),
    ``resilience_level``, ``prune_light_cone``, ``rescale``,
    ``seed_simulator`` and ``on_start`` work as in :func:`run_hardware_lap`.
    Returns ``{loss_history, loss_before, loss_after, accepted, vetoed,
    params, return_before, return_after, iterations, seconds, jobs, groups,
    blocking, guard}`` plus the ``on_start`` fields; ``accepted[k]`` says
    whether iteration ``k``'s step was taken, ``vetoed[k]`` whether the guard
    refused it.
    """
    if config is None:
        from traqmania.config import load_config

        config = load_config()
    config = _weights_config(config, init_weights_path)
    hw_cfg = config.get("hardware", {})
    if groups is None:
        groups = hw_cfg.get("spsa_groups", SPRINT_GROUPS)
    groups = (groups,) if isinstance(groups, str) else tuple(groups)
    if blocking is None:
        blocking = bool(hw_cfg.get("spsa_blocking", True))
    if guard is None:
        guard = hw_cfg.get("spsa_guard", SPRINT_GUARD)
    if isinstance(guard, (bool, str)) or not 0.0 <= float(guard) <= 1.0:
        raise ValueError(f"spsa guard must be a number in [0, 1] (0 = off), got {guard!r}")
    guard = float(guard)
    seed = int(config["training"]["seed"])
    params0 = np.load(init_weights_path)["params"]

    from traqmania.agents.quantum.qdqn import QuantumQFunction

    fast = QuantumQFunction(config["circuit"], seed=seed)
    fast.set_params(params0)
    scale = _sprint_scale(fast, groups)

    obs_b, act_b, target_b = _collect_batch(
        fast, track_name, config, batch, seed,
        recipe=_td_recipe(init_weights_path, config),
    )

    def exact_return(params: np.ndarray) -> float:
        fast.set_params(params)
        return _greedy_return(fast, track_name, config, seed, episodes=SPRINT_EVAL_EPISODES)

    return_before = exact_return(params0)
    keeps_driving = None
    if guard > 0.0:
        floor = return_before - (1.0 - guard) * abs(return_before)

        def keeps_driving(candidate: np.ndarray) -> bool:
            return exact_return(candidate) >= floor

    rows = np.arange(batch)
    hw, mode, run_info = _start_execution(
        config, backend, shots, resilience_level, prune_light_cone, rescale, params0,
        seed_simulator,
    )
    t0 = time.perf_counter()
    try:
        if on_start is not None:
            on_start(dict(run_info))

        def loss(theta: np.ndarray) -> float:
            hw.set_params(theta)
            q_sel = hw.q_values(obs_b)[rows, act_b]
            return float(np.mean((q_sel - target_b) ** 2))

        # Gain calibration (Spall): a few probe pairs estimate the gradient
        # magnitude along random directions; choose `a` so the FIRST step
        # moves each parameter by ~step_target probe sizes whatever the loss
        # scale is (it varies over orders of magnitude between weight files).
        stability = iterations / 10.0
        a, _magnitude = spsa.calibrate_gain(
            loss, params0, c=1.0, target_step=step_target, A=stability,
            pairs=SPRINT_CALIBRATION_PAIRS, seed=seed, scale=scale,
        )
        result = spsa.minimize(
            loss,
            params0,
            iterations=iterations,
            a=a,
            c=1.0,
            A=stability,
            seed=seed,
            callback=on_iter,
            stop_event=stop_event,
            scale=scale,
            max_step=2.0 * step_target,
            blocking=blocking,
            accept=keeps_driving,
        )
        history = result["loss_history"]
        if stop_event is not None and stop_event.is_set():  # aborted: no further jobs
            loss_before = result.get("loss_start", history[0] if history else float("nan"))
            loss_after = history[-1] if history else loss_before
        elif not any(result["accepted"]):  # nothing moved: one point, one number
            loss_before = loss_after = float(
                np.mean([loss(params0) for _ in range(SPRINT_LOSS_EVALUATIONS)]))
        else:
            # interleaved, so a drifting device biases neither side
            fresh = [(loss(params0), loss(result["x"]))
                     for _ in range(SPRINT_LOSS_EVALUATIONS)]
            loss_before, loss_after = (float(v) for v in np.mean(fresh, axis=0))
    finally:
        if mode is not None:
            hw.close_mode()
    seconds = time.perf_counter() - t0

    return_after = exact_return(result["x"])

    return {
        "loss_history": result["loss_history"],
        "loss_before": loss_before,
        "loss_after": loss_after,
        "accepted": result["accepted"],
        "vetoed": result["vetoed"],
        "params": result["x"],
        "return_before": return_before,
        "return_after": return_after,
        "iterations": int(iterations),
        "seconds": seconds,
        "jobs": hw.jobs,
        "groups": list(groups),
        "blocking": blocking,
        "guard": guard,
        **run_info,
    }


def sprint_steps_text(result: dict) -> str:
    """One line on what a sprint did with its steps, from its result: how many
    were taken and why the others were not."""
    accepted, vetoed = result["accepted"], result["vetoed"]
    text = f"{sum(map(bool, accepted))}/{len(accepted)}"
    reasons = []
    if any(vetoed):
        reasons.append(f"{sum(map(bool, vetoed))} refused by the simulator guard (the policy "
                       f"would keep less than {result['guard']:.0%} of its return)")
    blocked = len(accepted) - sum(map(bool, accepted)) - sum(map(bool, vetoed))
    if blocked:
        reasons.append(f"{blocked} did not lower the hardware loss")
    return text + (f" ({'; '.join(reasons)})" if reasons else "")


# ------------------------------------------------------------------------ CLI


def _default_weights(track: str, n_qubits: int = 4) -> Path:
    suffix = "" if n_qubits == 4 else f"_q{n_qubits}"
    return WEIGHTS_DIR / f"quantum_{track}{suffix}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m traqmania.hardware",
        description="Run traQmania's quantum policy on IBM hardware (or a local fake twin).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("lap", "greedy rollout of one car with decisions made on the backend"),
        ("sprint", "SPSA TD-loss fine-tune on the backend"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--track", default="oval",
                       help="track name (oval | chicane | gp | combo)")
        p.add_argument("--profile", default=None,
                       help="config profile overlay (e.g. q6; picks circuit size, "
                            "observation geometry and the default weights file; an "
                            "observation recorded with the weights wins)")
        p.add_argument("--fake", action="store_true", help="use a local fake backend")
        p.add_argument("--fake-name", default=None,
                       help="fake backend, any spelling; implies --fake (default: [hardware] "
                            f"fake_name, else {DEFAULT_FAKE}; e.g. fake_fez for a heavy-hex "
                            "Heron, fake_manila for a retired CX-era Falcon)")
        p.add_argument("--backend", default=None,
                       help="real backend name (default: [hardware] backend_name, else "
                            "least busy)")
        p.add_argument("--weights", default=None, help="weights .npz (default: bundled)")
        p.add_argument("--shots", type=int, default=None,
                       help="shots per Estimator job (default: [hardware] "
                            + ("decision_shots, else shots" if name == "lap" else "shots")
                            + ", else 1024)")
        p.add_argument("--rescale", choices=["off", "global", "readout"], default=None,
                       help="attenuation rescale, calibrated with one extra job: global "
                            "divides <Z> by the fitted attenuation f, readout corrects each "
                            "readout's own slope and bias (default: [hardware] rescale, "
                            "else off)")
        p.add_argument("--resilience", type=int, choices=sorted(RESILIENCE_LEVELS), default=None,
                       help="Estimator error mitigation: 0 raw device noise, 1 TREX, 2 TREX + "
                            "ZNE with gate twirling (default: [hardware] resilience_level, "
                            "else 0)")
        p.add_argument("--no-prune", action="store_true",
                       help="run the full circuit instead of the light-cone-pruned one "
                            "(same expectation values, more two-qubit gates)")
        if name == "lap":
            p.add_argument("--max-decisions", type=int, default=None,
                           help="stop the rollout after N decisions")
        else:
            p.add_argument("--iterations", type=int, default=30, help="SPSA iterations")
            p.add_argument("--batch", type=int, default=16, help="replay batch size")
            p.add_argument("--groups", default=None,
                           help="comma-separated parameter groups SPSA moves, out of "
                                "lam,theta,head (default: [hardware] spsa_groups, else "
                                f"{','.join(SPRINT_GROUPS)})")
            p.add_argument("--no-blocking", action="store_true",
                           help="take every SPSA step (default: only steps the hardware "
                                "loss does not get worse on)")
            p.add_argument("--no-guard", action="store_true",
                           help="do not refuse steps that cost the policy its return on "
                                "the exact simulator (default: [hardware] spsa_guard, else "
                                f"keep {100 * SPRINT_GUARD:.0f} %% of it)")
    args = parser.parse_args(argv)

    from traqmania.config import load_config

    config = load_config(args.profile)
    hw_cfg = config.get("hardware", {})
    n_qubits = int(config["circuit"]["n_qubits"])
    weights = Path(args.weights) if args.weights else _default_weights(args.track, n_qubits)
    if not weights.exists():
        parser.error(f"weights file not found: {weights} (train first or pass --weights)")
    profile_config, profile_shape = config, _circuit_shape(config)
    try:  # the weights' own observation, depth and action count, not the profile's
        config = _weights_config(config, weights)
    except ValueError as exc:
        parser.error(str(exc))
    # Naming a fake asks for one: never contact (or bill) a real device instead.
    use_fake = args.fake or args.fake_name is not None
    try:
        backend = get_backend(
            args.backend or hw_cfg.get("backend_name") or None,
            use_fake=use_fake,
            fake_name=args.fake_name or hw_cfg.get("fake_name") or DEFAULT_FAKE,
            min_qubits=max(5, n_qubits),
        )
    except ValueError as exc:  # unknown --fake-name: list what exists instead of a traceback
        parser.error(str(exc))
    print(f"backend: {getattr(backend, 'name', backend)}"
          f"{' (fake, local simulation)' if use_fake else ''}")
    print(f"weights: {weights}")
    if _circuit_shape(config) != profile_shape:
        _n, n_layers, n_actions = _circuit_shape(config)
        print(f"circuit shape from the weights: {n_layers} blocks, {n_actions} actions "
              f"(profile: {profile_shape[1]} blocks, {profile_shape[2]} actions)")
    from traqmania.server.runtime import observation_note

    note = observation_note(profile_config, config)
    if note:
        print(note)
    # None leaves the choice to the config's [hardware] table (see _start_execution).
    prune = False if args.no_prune else None
    shots = args.shots
    if shots is None:
        shots = int(hw_cfg.get("shots", 1024))
        if args.command == "lap":
            shots = int(hw_cfg.get("decision_shots", shots))

    def on_start(info: dict) -> None:
        level = info["resilience_level"]
        print(f"runs on: {info['backend_name']}")
        print(f"execution mode: {info['mode']}")
        if info["note"]:
            print(f"note: {info['note']}")
        print(f"circuit: {info['two_qubit_gates']} two-qubit gates, depth {info['depth']}"
              f"{' (light-cone pruned)' if info['pruned'] else ''}")
        print(f"shots: {info['shots']} per "
              f"{'decision' if args.command == 'lap' else 'loss evaluation row'}")
        print(f"error mitigation: resilience level {level} ({mitigation_text(info)})")

    if args.command == "lap":
        def on_decision(i: int, info: dict) -> None:
            q = " ".join(f"{v:+.2f}" for v in info["q_values"])
            print(f"decision {i + 1:>3}  action={info['action']}  Q=[{q}]  "
                  f"lap={info['lap']}  {info['seconds']:.2f}s")

        result = run_hardware_lap(args.track, weights, backend, shots=shots,
                                  max_decisions=args.max_decisions, on_decision=on_decision,
                                  config=config, resilience_level=args.resilience,
                                  prune_light_cone=prune, on_start=on_start,
                                  rescale=args.rescale)
        lap_txt = f"{result['best_lap_s']:.2f}s" if result["lapped"] else "no (rollout ended)"
        print(f"\nlap completed: {lap_txt}")
        print(f"decisions: {result['decisions']}  "
              f"({result['seconds_per_decision']:.2f}s per decision, {result['mode']} mode)")
    else:
        def on_iter(k: int, info: dict) -> None:
            verdict = "" if info["accepted"] else (
                "  step refused by the guard" if info["vetoed"] else "  step rejected")
            print(f"iter {k + 1:>3}/{args.iterations}  loss={info['loss']:.4f}  "
                  f"(f+={info['f_plus']:.4f} f-={info['f_minus']:.4f}){verdict}")

        groups = args.groups.split(",") if args.groups else None
        try:
            result = spsa_sprint(args.track, weights, backend, iterations=args.iterations,
                                 shots=shots, batch=args.batch, on_iter=on_iter,
                                 config=config, resilience_level=args.resilience,
                                 prune_light_cone=prune, on_start=on_start,
                                 rescale=args.rescale, groups=groups,
                                 blocking=False if args.no_blocking else None,
                                 guard=0.0 if args.no_guard else None)
        except ValueError as exc:  # e.g. an unknown --groups name
            parser.error(str(exc))
        print(f"\nSPSA sprint done in {result['seconds']:.1f}s ({args.iterations} iterations "
              f"on {'+'.join(result['groups'])}, {result['jobs']} Estimator jobs, "
              f"{result['mode']} mode)")
        print(f"steps taken: {sprint_steps_text(result)}")
        print(f"hardware loss (mean of {SPRINT_LOSS_EVALUATIONS} fresh evaluations): "
              f"{result['loss_before']:.4f} -> {result['loss_after']:.4f}")
        print(f"greedy return (fastsim, mean of {SPRINT_EVAL_EPISODES} episodes): "
              f"{result['return_before']:.1f} -> {result['return_after']:.1f}")


if __name__ == "__main__":
    main()
