/**
 * The two Q-function families of traQmania, inference only.
 *
 * QuantumDriver runs the canonical data re-uploading circuit
 * (traqmania/agents/quantum/circuit.py) on QAMPoser's in-browser state-vector
 * simulator: per block l, RY(lam[l,i] * s_i) encodes feature i on qubit i,
 * RY(theta[l,i,0]) RZ(theta[l,i,1]) is the trainable layer, then a CZ ring.
 * Readout: Q_a = w[a] * <Z_a> + b[a] on the first n_actions qubits.
 *
 * MlpDriver is the classical baseline: tanh hidden layer, linear output
 * (traqmania/agents/classical/mlp.py).
 */
import { expectationZ, simulateStatevector, type SimulationGate } from '@qamposer/react';
import type { ObservationConfig } from './observer';

export interface CircuitShape {
  n_qubits: number;
  n_layers: number;
  n_actions: number;
}

export interface DriverFile {
  id: string;
  agent: 'quantum' | 'mlp';
  n_actions: number;
  observation: ObservationConfig;
  circuit?: CircuitShape;
  hidden?: number;
  params: number[];
}

export interface Decision {
  q: Float64Array;
  action: number;
  /** <Z_i> of every qubit (quantum drivers only) */
  expectations?: Float64Array;
  /** the bound circuit that produced this decision (quantum drivers only) */
  gates?: SimulationGate[];
}

export interface Driver {
  readonly id: string;
  readonly agent: 'quantum' | 'mlp';
  readonly nFeatures: number;
  readonly nActions: number;
  readonly nParams: number;
  decide(obs: ArrayLike<number>): Decision;
}

/** First index of the maximum (numpy.argmax). */
export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export class QuantumDriver implements Driver {
  readonly agent = 'quantum' as const;
  readonly id: string;
  readonly nQubits: number;
  readonly nLayers: number;
  readonly nActions: number;
  readonly nFeatures: number;
  readonly nParams: number;
  /** lam[l * n + i] */
  readonly lam: Float64Array;
  /** theta[(l * n + i) * 2 + k], k = 0 RY, k = 1 RZ */
  readonly theta: Float64Array;
  readonly w: Float64Array;
  readonly b: Float64Array;

  constructor(file: DriverFile) {
    if (!file.circuit) throw new Error(`driver ${file.id}: no circuit shape`);
    const { n_qubits: n, n_layers: layers, n_actions: actions } = file.circuit;
    const expected = 3 * layers * n + 2 * actions;
    if (file.params.length !== expected) {
      throw new Error(`driver ${file.id}: ${file.params.length} params, expected ${expected}`);
    }
    this.id = file.id;
    this.nQubits = n;
    this.nLayers = layers;
    this.nActions = actions;
    this.nFeatures = n;
    this.nParams = expected;
    const p = Float64Array.from(file.params);
    const nLam = layers * n;
    const nTheta = layers * n * 2;
    this.lam = p.slice(0, nLam);
    this.theta = p.slice(nLam, nLam + nTheta);
    this.w = p.slice(nLam + nTheta, nLam + nTheta + actions);
    this.b = p.slice(nLam + nTheta + actions);
  }

  /** The circuit with every angle bound for one observation, in QAMPoser's
   *  gate format (one column per gate, so the order is unambiguous). */
  boundGates(obs: ArrayLike<number>): SimulationGate[] {
    const n = this.nQubits;
    const gates: SimulationGate[] = [];
    let position = 0;
    for (let l = 0; l < this.nLayers; l++) {
      for (let i = 0; i < n; i++) {
        gates.push({ type: 'RY', qubit: i, parameter: this.lam[l * n + i] * obs[i], position: position++ });
      }
      for (let i = 0; i < n; i++) {
        const t = (l * n + i) * 2;
        gates.push({ type: 'RY', qubit: i, parameter: this.theta[t], position: position++ });
        gates.push({ type: 'RZ', qubit: i, parameter: this.theta[t + 1], position: position++ });
      }
      for (let i = 0; i < n; i++) {
        gates.push({ type: 'CZ', control: i, target: (i + 1) % n, position: position++ });
      }
    }
    return gates;
  }

  /** <Z_i> for every qubit, from QAMPoser's exact state vector. */
  expectations(obs: ArrayLike<number>, gates = this.boundGates(obs)): Float64Array {
    const state = simulateStatevector(this.nQubits, gates);
    return Float64Array.from({ length: this.nQubits }, (_, i) => expectationZ(state, i));
  }

  decide(obs: ArrayLike<number>): Decision {
    const gates = this.boundGates(obs);
    const expectations = this.expectations(obs, gates);
    const q = new Float64Array(this.nActions);
    for (let a = 0; a < this.nActions; a++) q[a] = expectations[a] * this.w[a] + this.b[a];
    return { q, action: argmax(q), expectations, gates };
  }
}

export class MlpDriver implements Driver {
  readonly agent = 'mlp' as const;
  readonly id: string;
  readonly nFeatures: number;
  readonly hidden: number;
  readonly nActions: number;
  readonly nParams: number;
  private readonly w1: Float64Array; // (F, H) row-major
  private readonly b1: Float64Array;
  private readonly w2: Float64Array; // (H, A) row-major
  private readonly b2: Float64Array;

  constructor(file: DriverFile) {
    const a = file.n_actions;
    const h = file.hidden ?? 8;
    const f = (file.params.length - a - h * a - h) / h;
    if (!Number.isInteger(f) || f < 1) throw new Error(`driver ${file.id}: bad MLP shape`);
    this.id = file.id;
    this.nFeatures = f;
    this.hidden = h;
    this.nActions = a;
    this.nParams = file.params.length;
    const p = Float64Array.from(file.params);
    let o = 0;
    this.w1 = p.slice(o, (o += f * h));
    this.b1 = p.slice(o, (o += h));
    this.w2 = p.slice(o, (o += h * a));
    this.b2 = p.slice(o, (o += a));
  }

  decide(obs: ArrayLike<number>): Decision {
    const { nFeatures: f, hidden: h, nActions: a } = this;
    const hiddenOut = new Float64Array(h);
    for (let j = 0; j < h; j++) {
      let z = 0;
      for (let i = 0; i < f; i++) z += obs[i] * this.w1[i * h + j];
      hiddenOut[j] = Math.tanh(z + this.b1[j]);
    }
    const q = new Float64Array(a);
    for (let k = 0; k < a; k++) {
      let z = 0;
      for (let j = 0; j < h; j++) z += hiddenOut[j] * this.w2[j * a + k];
      q[k] = z + this.b2[k];
    }
    return { q, action: argmax(q) };
  }
}

export function makeDriver(file: DriverFile): Driver {
  return file.agent === 'quantum' ? new QuantumDriver(file) : new MlpDriver(file);
}
