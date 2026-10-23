/**
 * "What if?" for one paused decision, inference only: run the SAME driver on
 * a copy of the observation with some sensors covered (set to 0), or a copy
 * of a quantum driver with one trainable rotation nudged. Nothing here touches
 * the car, the world, the RNG or the driver it is given.
 *
 * Also the structural light cone of the canonical circuit (a port of
 * racetraq/agents/quantum/lightcone.py, readout_live_gates / live_gates): which
 * trainable RY/RZ gates can influence any readout <Z_a> at all. A dead gate
 * commutes with every back-propagated readout, so nudging it cannot move the
 * Q-values for any input.
 */
import { argmax, QuantumDriver, type Decision, type Driver } from './agents';

/** One trainable rotation of a quantum driver: theta[(block * n + qubit) * 2 + k]. */
export interface AngleRef {
  block: number;
  qubit: number;
  gate: 'RY' | 'RZ';
}

/** A copy of the observation with the given inputs set to 0. */
export function coveredObservation(obs: ArrayLike<number>, covered: Iterable<number>): Float64Array {
  const copy = Float64Array.from(obs);
  for (const i of covered) {
    if (!Number.isInteger(i) || i < 0 || i >= copy.length) throw new Error(`no sensor ${i}`);
    copy[i] = 0;
  }
  return copy;
}

/** The decision the same driver takes with some sensors covered. */
export function coverSensors(driver: Driver, obs: ArrayLike<number>, covered: Iterable<number>): Decision {
  return driver.decide(coveredObservation(obs, covered));
}

/** Flat index of a rotation in QuantumDriver.theta. */
export function thetaIndex(driver: QuantumDriver, ref: AngleRef): number {
  const { block, qubit, gate } = ref;
  if (!Number.isInteger(block) || block < 0 || block >= driver.nLayers) throw new Error(`no block ${block}`);
  if (!Number.isInteger(qubit) || qubit < 0 || qubit >= driver.nQubits) throw new Error(`no qubit ${qubit}`);
  return (block * driver.nQubits + qubit) * 2 + (gate === 'RZ' ? 1 : 0);
}

/** The trained angle of a rotation (radians). */
export function trainedAngle(driver: QuantumDriver, ref: AngleRef): number {
  return driver.theta[thetaIndex(driver, ref)];
}

/** A new driver with every parameter copied and one rotation shifted by `offset`. */
export function nudgedDriver(driver: QuantumDriver, ref: AngleRef, offset: number): QuantumDriver {
  const theta = Float64Array.from(driver.theta);
  theta[thetaIndex(driver, ref)] += offset;
  return new QuantumDriver({
    id: driver.id,
    agent: 'quantum',
    n_actions: driver.nActions,
    // the circuit never reads the observation config; the observer does
    observation: { ray_angles_deg: [], ray_max_dist: 1, features: [], lookahead_m: 0 },
    circuit: { n_qubits: driver.nQubits, n_layers: driver.nLayers, n_actions: driver.nActions },
    params: [...driver.lam, ...theta, ...driver.w, ...driver.b],
  });
}

/** The decision of the same observation with one rotation shifted by `offset`. */
export function nudgeAngle(driver: QuantumDriver, obs: ArrayLike<number>, ref: AngleRef, offset: number): Decision {
  return nudgedDriver(driver, ref, offset).decide(Float64Array.from(obs));
}

/** How a what-if decision compares with the original one. */
export interface Comparison {
  /** what-if Q minus original Q, per action */
  delta: Float64Array;
  /** the driver's choice changed */
  changed: boolean;
}

export function compare(original: Decision, whatIf: Decision): Comparison {
  const delta = Float64Array.from(whatIf.q, (q, a) => q - original.q[a]);
  return { delta, changed: whatIf.action !== original.action };
}

/** How a guessed action relates to the decision: right or wrong, and by how much. */
export interface GuessResult {
  right: boolean;
  /** Q of the best action minus Q of the guess (0 when right) */
  gap: number;
  /** when right: Q of the best minus the runner-up (the winning margin) */
  margin: number;
}

export function judgeGuess(q: ArrayLike<number>, guess: number): GuessResult {
  if (!Number.isInteger(guess) || guess < 0 || guess >= q.length) throw new Error(`no action ${guess}`);
  const best = argmax(q);
  let runnerUp = -Infinity;
  for (let a = 0; a < q.length; a++) if (a !== best && q[a] > runnerUp) runnerUp = q[a];
  return {
    right: guess === best,
    gap: q[best] - q[guess],
    margin: q.length > 1 ? q[best] - runnerUp : 0,
  };
}

// ------------------------------------------------------------ light cone

const NONE = 0;
const DIAG = 1;
const FULL = 2;

/** Trainable gates that can influence at least one readout <Z_0..Z_{A-1}>:
 *  `ry[l][i]` / `rz[l][i]` for block l, qubit i (lightcone.live_gates). */
export function liveRotations(nQubits: number, nLayers: number, nActions: number): { ry: boolean[][]; rz: boolean[][] } {
  const grid = () => Array.from({ length: nLayers }, () => new Array<boolean>(nQubits).fill(false));
  const ry = grid();
  const rz = grid();
  // at n = 2 the "ring" CZ(0,1) CZ(1,0) is the identity: nothing spreads
  const ring: [number, number][] = nQubits > 2 ? Array.from({ length: nQubits }, (_, i) => [i, (i + 1) % nQubits]) : [];
  for (let a = 0; a < Math.min(nActions, nQubits); a++) {
    const support = new Array<number>(nQubits).fill(NONE);
    support[a] = DIAG;
    for (let l = nLayers - 1; l >= 0; l--) {
      const general = support.map((s) => s === FULL);
      for (const [i, j] of ring) {
        if (general[i]) support[j] = Math.max(support[j], DIAG);
        if (general[j]) support[i] = Math.max(support[i], DIAG);
      }
      for (let i = 0; i < nQubits; i++) {
        if (support[i] === FULL) rz[l][i] = true;
        if (support[i] !== NONE) {
          ry[l][i] = true;
          support[i] = FULL;
        }
      }
    }
  }
  return { ry, rz };
}

/** Whether nudging this rotation can move any Q-value (structurally). */
export function isLive(driver: QuantumDriver, ref: AngleRef): boolean {
  thetaIndex(driver, ref); // validates
  const live = liveRotations(driver.nQubits, driver.nLayers, driver.nActions);
  return (ref.gate === 'RY' ? live.ry : live.rz)[ref.block][ref.qubit];
}
