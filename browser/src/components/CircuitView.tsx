/**
 * The live circuit of one decision, shown in QAMPoser's circuit editor
 * (controlled and read-only: edits are ignored). Gates that act in parallel
 * share a column so a block reads: encode, RY, RZ, CZ ring.
 */
import { useMemo } from 'react';
import { CircuitEditor, QamposerProvider, type Circuit, type Gate } from '@qamposer/react';
import type { SimulationGate } from '@qamposer/react';

/** Column layout: single-qubit gates go to the first column after the
 *  qubit's last gate; a CZ occupies every wire between its two qubits. */
export function layoutCircuit(nQubits: number, gates: SimulationGate[]): Circuit {
  const free = new Array(nQubits).fill(0);
  const placed: Gate[] = gates.map((g, k) => {
    const wires =
      g.type === 'CZ' || g.type === 'CNOT'
        ? range(Math.min(g.control!, g.target!), Math.max(g.control!, g.target!))
        : [g.qubit!];
    const column = Math.max(...wires.map((w) => free[w]));
    for (const w of wires) free[w] = column + 1;
    return { ...g, id: `g${k}`, position: column } as Gate;
  });
  return { qubits: nQubits, gates: placed };
}

function range(lo: number, hi: number): number[] {
  return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
}

const ignore = () => {};

export function CircuitView({
  nQubits,
  gates,
  zoom = 1,
}: {
  nQubits: number;
  gates: SimulationGate[];
  /** CSS zoom of QAMPoser's editor (its lanes are 80 px apart) */
  zoom?: number;
}) {
  const circuit = useMemo(() => layoutCircuit(nQubits, gates), [nQubits, gates]);
  const config = useMemo(() => ({ maxQubits: Math.max(nQubits, 5), maxGates: 1000 }), [nQubits]);
  return (
    <div
      className="circuit-view"
      style={{ zoom }}
      aria-label={`${nQubits}-qubit circuit of the current decision`}
    >
      <QamposerProvider circuit={circuit} onCircuitChange={ignore} config={config}>
        <CircuitEditor />
      </QamposerProvider>
    </div>
  );
}
