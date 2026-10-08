"""Quantum Q-function backends.

Importing this package must NOT import qiskit: the numpy fast path
(``fastsim``, ``adjoint``, ``qdqn``) is re-exported eagerly, while the
qiskit-facing submodules (``circuit``, ``qnn``) are loaded lazily on first
attribute access. ``lightcone`` (structural light-cone analysis; numpy-only
until its ``pruned_circuit`` is called) is lazy too, so
``python -m racetraq.agents.quantum.lightcone`` runs it exactly once, and so
is ``surrogate`` (classical Fourier surrogates of a trained circuit; numpy-only).
"""

from __future__ import annotations

from racetraq.agents.quantum.fastsim import FastStatevectorSim
from racetraq.agents.quantum.qdqn import QuantumQFunction

__all__ = ["FastStatevectorSim", "QuantumQFunction", "make_qfunction"]

_LAZY_SUBMODULES = ("circuit", "qnn", "lightcone", "surrogate")


def make_qfunction(kind: str, circuit_cfg: dict, **kw):
    """Factory for Q-function backends.

    kind: 'fastsim' (numpy fast path) or one of the qiskit execution paths
    'aer_statevector' | 'aer_shots' | 'aer_noisy' (loaded lazily so this
    package never imports qiskit unless a qiskit backend is requested).
    Extra keyword arguments are forwarded to the backend constructor.
    """
    if kind == "fastsim":
        return QuantumQFunction(circuit_cfg, **kw)
    from racetraq.agents.quantum.qnn import BACKENDS, QiskitQFunction

    if kind not in BACKENDS:
        raise ValueError(f"kind must be 'fastsim' or one of {BACKENDS}, got {kind!r}")
    return QiskitQFunction(circuit_cfg, backend=kind, **kw)


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"racetraq.agents.quantum.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_SUBMODULES))
