/** The IBM Quantum Composer handoff exports the exact bound circuit. */
import { readFileSync, writeFileSync } from 'node:fs';
import { circuitToQasm } from '@qamposer/react';
import LZString from 'lz-string';
import { describe, expect, it } from 'vitest';
import { layoutCircuit } from '../src/components/CircuitView';
import { composerUrl } from '../src/components/QuantumBrain';
import { QuantumDriver } from '../src/sim/agents';

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));

describe('composer handoff', () => {
  const fixture = read('./fixtures/parity.json');
  const driver = new QuantumDriver(read('../public/data/drivers/quantum_oval.json'));
  const obs: number[] = fixture.q_values.quantum_oval.obs[0];
  const gates = driver.boundGates(obs);
  const qasm = circuitToQasm(layoutCircuit(driver.nQubits, gates));

  it('keeps every gate, in an order with the same state', () => {
    expect(qasm.match(/^ry\(/gm)).toHaveLength(2 * driver.nLayers * driver.nQubits);
    expect(qasm.match(/^rz\(/gm)).toHaveLength(driver.nLayers * driver.nQubits);
    expect(qasm.match(/^cz /gm)).toHaveLength(driver.nLayers * driver.nQubits);
    // the column layout reorders only commuting gates
    const laidOut = layoutCircuit(driver.nQubits, gates).gates;
    const reordered = driver.expectations(obs, [...laidOut]);
    const original = driver.expectations(obs, gates);
    reordered.forEach((z, i) => expect(z).toBeCloseTo(original[i], 12));
    if (process.env.TRAQMANIA_QASM_OUT) writeFileSync(process.env.TRAQMANIA_QASM_OUT, qasm);
  });

  it('packs the circuit into the ?initial= payload', () => {
    const url = composerUrl(qasm, 'test');
    const payload = new URL(url).searchParams.get('initial')!;
    const decoded = JSON.parse(LZString.decompressFromEncodedURIComponent(payload)!);
    expect(decoded).toEqual({ title: 'test', description: '', qasm });
  });
});
