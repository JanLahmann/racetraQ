"""Structural light-cone analysis of the canonical circuit (numpy only).

Which observation features can a readout ``<Z_a>`` depend on, which gates and
parameters can influence it at all, and how deep must the circuit be before
every action sees every feature?  All of it follows from the circuit's
structure alone (see ``circuit.py``), without simulating a single amplitude.

Method. Back-propagate the readout ``Z_a`` through the circuit in the
Heisenberg picture, last gate first, tracking per qubit only the KIND of
support the operator has there: none, Z-only (diagonal: I or Z), or general
(some X/Y component). Backward rules, per gate:

    CZ(i, j)  a general component on i puts a Z on j (none -> Z-only; Z-only
              and general stay as they are), and vice versa; Z-only or no
              support does not spread. The gate is live iff i or j is general
              — otherwise it commutes with the operator and can be dropped.
    RZ on i   commutes with none / Z-only support (dead for this readout);
              keeps general support general (live).
    RY on i   dead without support; otherwise live, and the support becomes
              general. The encoding RY(lam*s) and the variational RY act
              consecutively on the same qubit, so they are live or dead
              together.

A gate that is dead for a readout commutes with the back-propagated operator,
so removing it — or changing its angle — leaves ``<Z_a>`` exactly unchanged:
its parameter has zero gradient for EVERY input. The analysis is an upper
bound on what can matter that generic parameter values saturate
(``tests/test_lightcone.py`` checks it against fastsim and adjoint gradients).

Consequences for the nearest-neighbour CZ ring with L blocks (n >= 3 qubits;
at n = 2 the "ring" CZ(0,1) CZ(1,0) is the identity, so nothing ever spreads
and both CZs are reported dead — removable as a pair, not one at a time):

- the last block's CZ ring only ever meets the diagonal ``Z_a`` — it never
  affects a readout;
- in block l the encoding/RY gates are live on qubits within ring distance
  ``L - 1 - l`` of a readout qubit and the RZ gates within ``L - 2 - l``, so
  ``<Z_a>`` sees exactly the features within ring distance ``L - 1`` of qubit
  a, and full visibility needs ``L >= n // 2 + 1`` blocks.

numpy/stdlib only at import time — qiskit is imported lazily inside
:func:`pruned_circuit`.

Command line::

    python -m traqmania.agents.quantum.lightcone --qubits 10 --layers 4 [--actions 4]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from traqmania.agents.base import action_labels as default_action_labels

# Kind of support the back-propagated operator has on one qubit.
_NONE, _DIAG, _FULL = 0, 1, 2

GATE_KINDS = ("enc", "ry", "rz", "cz")


class PrunedCircuit(NamedTuple):
    """The canonical circuit with every structurally dead gate removed.

    ``circuit`` keeps build_circuit's Parameter names (``x[k]``, ``theta[k]``);
    ``input_index`` / ``weight_index`` are the sorted flat indices of the
    ``x`` / ``theta`` parameters that remain, so a caller binds
    ``x[:, input_index]`` and ``theta[weight_index]``.
    """

    circuit: Any
    input_index: np.ndarray
    weight_index: np.ndarray


def _check_shape(n_qubits: int, n_layers: int, n_actions: int | None) -> tuple[int, int, int]:
    """Validated ``(n, L, A)`` with the default ``A = min(4, n)`` filled in."""
    n, layers = int(n_qubits), int(n_layers)
    if n < 2:
        raise ValueError(f"the CZ ring needs n_qubits >= 2, got {n}")
    if layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {layers}")
    actions = min(4, n) if n_actions is None else int(n_actions)
    if not 1 <= actions <= n:
        raise ValueError(f"n_actions must be in 1..n_qubits = {n}, got {actions}")
    return n, layers, actions


def readout_live_gates(n_qubits: int, n_layers: int, qubit: int) -> dict[str, np.ndarray]:
    """Live gates for the single readout ``<Z_qubit>``.

    Returns bool arrays ``{"enc", "ry", "rz", "cz"}``, each (L, n), True where
    the gate can influence the readout; ``cz[l, i]`` is CZ(i, (i+1) % n) of
    block l. The building block of every other function in this module.
    """
    n, layers, _ = _check_shape(n_qubits, n_layers, None)
    qubit = int(qubit)
    if not 0 <= qubit < n:
        raise ValueError(f"readout qubit must be in 0..{n - 1}, got {qubit}")

    # At n = 2 the "ring" is CZ(0,1) CZ(1,0) = identity: nothing entangles.
    ring = [(i, (i + 1) % n) for i in range(n)] if n > 2 else []

    live = {kind: np.zeros((layers, n), dtype=bool) for kind in GATE_KINDS}
    support = np.full(n, _NONE, dtype=np.int8)
    support[qubit] = _DIAG  # the observable Z_qubit itself

    for layer in reversed(range(layers)):
        # CZ ring. A CZ never changes which qubits are general, so the ring's
        # (commuting) gates can be processed in any order.
        general = support == _FULL
        for i, j in ring:
            live["cz"][layer, i] = general[i] or general[j]
            if general[i]:
                support[j] = max(support[j], _DIAG)
            if general[j]:
                support[i] = max(support[i], _DIAG)
        # Variational RZ, then variational RY and encoding RY (backward order).
        live["rz"][layer] = support == _FULL
        touched = support != _NONE
        live["ry"][layer] = touched
        live["enc"][layer] = touched
        support[touched] = _FULL
    return live


def live_gates(
    n_qubits: int, n_layers: int, n_actions: int | None = None
) -> dict[str, np.ndarray]:
    """Gates that can influence at least one of the readouts ``<Z_0..Z_{A-1}>``.

    Returns bool arrays ``{"enc": (L, n), "ry": (L, n), "rz": (L, n),
    "cz": (L, n)}`` — the union over the A readouts (A = n_actions, default
    min(4, n)); ``cz[l, i]`` is CZ(i, (i+1) % n) of block l.
    """
    n, layers, actions = _check_shape(n_qubits, n_layers, n_actions)
    live = {kind: np.zeros((layers, n), dtype=bool) for kind in GATE_KINDS}
    for a in range(actions):
        for kind, mask in readout_live_gates(n, layers, a).items():
            live[kind] |= mask
    return live


def feature_visibility(n_qubits: int, n_layers: int, n_actions: int | None = None) -> np.ndarray:
    """Which features each readout can depend on: bool (A, n).

    ``[a, j]`` is True iff ``<Z_a>`` can depend on the feature encoded on
    qubit j, i.e. iff at least one of its encoding gates is live for Z_a.
    """
    n, layers, actions = _check_shape(n_qubits, n_layers, n_actions)
    return np.stack(
        [readout_live_gates(n, layers, a)["enc"].any(axis=0) for a in range(actions)]
    )


def live_parameter_mask(
    n_qubits: int, n_layers: int, n_actions: int | None = None
) -> dict[str, np.ndarray]:
    """Circuit parameters that can have a non-zero gradient for some readout.

    Returns ``{"lam": bool (L, n), "theta": bool (L, n, 2)}`` in the layout of
    ``QuantumQFunction`` (``theta[..., 0]`` the RY angle, ``[..., 1]`` the RZ
    angle); ``np.concatenate([lam.ravel(), theta.ravel()])`` masks the circuit
    part of the flat parameter vector. False = structurally dead: the gradient
    is exactly zero for every input and every readout.
    """
    live = live_gates(n_qubits, n_layers, n_actions)
    return {"lam": live["enc"].copy(), "theta": np.stack([live["ry"], live["rz"]], axis=-1)}


def min_layers_full_visibility(n_qubits: int, n_actions: int | None = None) -> int:
    """Smallest number of blocks L at which every readout sees every feature.

    Found by growing L until :func:`feature_visibility` is all True (for the
    CZ ring this lands on ``n_qubits // 2 + 1``). Raises ValueError when no
    depth achieves it (n = 2, whose two-CZ "ring" is the identity).
    """
    n, _, actions = _check_shape(n_qubits, 1, n_actions)
    previous = None
    layers = 1
    while True:
        visible = feature_visibility(n, layers, actions)
        if visible.all():
            return layers
        if previous is not None and np.array_equal(visible, previous):
            raise ValueError(
                f"no circuit depth gives every readout every feature at n_qubits = {n}"
            )
        previous = visible
        layers += 1


def _readout_name(a: int, labels: Sequence[str]) -> str:
    """'Brake (Z_3)' when the readout has an action label, else 'Z_3'."""
    return f"{labels[a]} (Z_{a})" if a < len(labels) else f"Z_{a}"


def _default_labels(n_actions: int) -> tuple[str, ...]:
    """Action labels for a known action-set size, else none (bare Z_a names)."""
    try:
        return tuple(default_action_labels(n_actions))
    except ValueError:
        return ()


def blind_spots(
    feature_names: Sequence[str],
    n_layers: int,
    n_actions: int | None = None,
    action_labels: Sequence[str] | None = None,
) -> list[str]:
    """Human-readable list of what each action's readout cannot see.

    One line per readout with hidden features, e.g.
    ``"Brake (Z_3) cannot see: speed, ray +45°"``; an empty list when every
    readout sees every feature. ``n_qubits = len(feature_names)`` (feature j
    is encoded on qubit j); ``action_labels`` defaults to the action set's
    labels (``traqmania.agents.base.action_labels``).
    """
    names = [str(name) for name in feature_names]
    visible = feature_visibility(len(names), n_layers, n_actions)
    labels = _default_labels(visible.shape[0]) if action_labels is None else list(action_labels)
    lines = []
    for a, row in enumerate(visible):
        hidden = [names[j] for j in np.flatnonzero(~row)]
        if hidden:
            lines.append(f"{_readout_name(a, labels)} cannot see: {', '.join(hidden)}")
    return lines


def blind_spot_warning(
    feature_names: Sequence[str],
    n_layers: int,
    n_actions: int | None = None,
    action_labels: Sequence[str] | None = None,
) -> list[str]:
    """Warning lines for a circuit whose light cone hides features from actions.

    A headline (the depth that would give full visibility) followed by the
    :func:`blind_spots` lines; an empty list when every readout sees every
    feature. What training entry points print before a run — a warning, not
    an error: the circuit still trains, each action just decides without the
    listed features.
    """
    lines = blind_spots(feature_names, n_layers, n_actions, action_labels)
    if not lines:
        return []
    n = len(feature_names)
    try:
        fix = f"full visibility needs n_layers >= {min_layers_full_visibility(n, n_actions)}"
    except ValueError:
        fix = "no depth gives full visibility at this size"
    head = (f"light cone: n_layers = {int(n_layers)} is too shallow for {n} qubits, "
            f"some actions cannot see every feature ({fix})")
    return [head, *(f"  {line}" for line in lines)]


def pruned_circuit(n_qubits: int, n_layers: int, n_actions: int | None = None) -> PrunedCircuit:
    """Build the canonical circuit with only its live gates, in the original order.

    The Parameters carry the same names as ``circuit.build_circuit``'s
    (``x[l*n + i]``, ``theta[(l*n + i)*2 + k]``), so ``split_parameters``
    works unchanged; ``input_index`` / ``weight_index`` list the flat indices
    that remain. Its ``<Z_a>`` (a < n_actions) equals the full circuit's
    exactly for all parameter values — with fewer gates to run (at n = 4,
    L = 4: 12 CZ instead of 16).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n, layers, actions = _check_shape(n_qubits, n_layers, n_actions)
    live = live_gates(n, layers, actions)

    x = ParameterVector("x", layers * n)
    theta = ParameterVector("theta", layers * n * 2)

    qc = QuantumCircuit(n, name="traqmania_qcircuit_pruned")
    for layer in range(layers):
        for i in range(n):
            if live["enc"][layer, i]:
                qc.ry(x[layer * n + i], i)
        for i in range(n):
            if live["ry"][layer, i]:
                qc.ry(theta[(layer * n + i) * 2], i)
            if live["rz"][layer, i]:
                qc.rz(theta[(layer * n + i) * 2 + 1], i)
        for i in range(n):
            if live["cz"][layer, i]:
                qc.cz(i, (i + 1) % n)

    mask = live_parameter_mask(n, layers, actions)
    return PrunedCircuit(
        circuit=qc,
        input_index=np.flatnonzero(mask["lam"].ravel()),
        weight_index=np.flatnonzero(mask["theta"].ravel()),
    )


def report(n_qubits: int, n_layers: int, n_actions: int | None = None) -> str:
    """Plain-text light-cone report (what the command line prints)."""
    n, layers, actions = _check_shape(n_qubits, n_layers, n_actions)
    visible = feature_visibility(n, layers, actions)
    live = live_gates(n, layers, actions)
    mask = live_parameter_mask(n, layers, actions)
    labels = _default_labels(actions)
    names = [_readout_name(a, labels) for a in range(actions)]
    width = max(len(name) for name in names)

    lines = [
        f"Light cone of the canonical circuit: {n} qubits, {layers} blocks, "
        f"readout Z_0..Z_{actions - 1}",
        "",
        "Feature visibility (row = readout, column = feature/qubit; X sees, . blind)",
        f"  {'':<{width}}  " + "".join(str(j % 10) for j in range(n)),
    ]
    for a, row in enumerate(visible):
        hidden = np.flatnonzero(~row)
        blind = "blind to: " + ", ".join(str(j) for j in hidden) if hidden.size else "sees all"
        lines.append(
            f"  {names[a]:<{width}}  " + "".join("X" if v else "." for v in row) + f"  {blind}"
        )

    n_lam, n_ry, n_rz = (int(m.size - m.sum()) for m in (live["enc"], live["ry"], live["rz"]))
    per = layers * n
    lines += [
        "",
        f"Dead circuit parameters (zero gradient for every input): "
        f"{n_lam + n_ry + n_rz}/{3 * per}",
        f"  lam (encoding RY)  {n_lam}/{per}",
        f"  theta RY           {n_ry}/{per}",
        f"  theta RZ           {n_rz}/{per}",
    ]
    for key, part in (("lam[l, i]", mask["lam"]), ("theta[l, i, k]", mask["theta"])):
        dead = ["(" + ",".join(str(v) for v in idx) + ")" for idx in np.argwhere(~part)]
        if dead:
            lines.append(f"  dead {key}: " + " ".join(dead))
    lines.append(f"Live two-qubit gates: {int(live['cz'].sum())}/{per} CZ")

    lines.append("")
    try:
        needed = min_layers_full_visibility(n, actions)
    except ValueError:
        lines.append("Full visibility: not reachable at any depth")
    else:
        verdict = "reached" if layers >= needed else f"NOT reached at L = {layers}"
        lines.append(f"Full visibility needs L >= {needed} blocks ({verdict})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m traqmania.agents.quantum.lightcone",
        description="Structural light-cone analysis of the canonical traQmania circuit.",
    )
    parser.add_argument("--qubits", type=int, default=4, help="n_qubits (default 4)")
    parser.add_argument("--layers", type=int, default=4, help="re-uploading blocks L (default 4)")
    parser.add_argument(
        "--actions", type=int, default=None,
        help="readout qubits Z_0..Z_{A-1} (default min(4, qubits))",
    )
    args = parser.parse_args(argv)
    try:
        print(report(args.qubits, args.layers, args.actions))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
