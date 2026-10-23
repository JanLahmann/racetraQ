/**
 * A toy that trains in the browser: one or two qubits learn "steer toward
 * the longer ray" on a frozen batch of real sensor readings, by gradient
 * descent with the parameter-shift rule. Pure functions and one small
 * trainer class; no React, no DOM, no Math.random (everything follows a seed).
 *
 * This is NOT how the race drivers were trained (double DQN, thousands of
 * episodes, offline in Python): it is a supervised classifier on 64 fixed
 * samples, small enough to train live, so the page shows learning itself.
 *
 * Data: positions and headings sampled on a bundled track; the existing
 * Observer measures the two side rays (+60° = left, -60° = right, the
 * drivers' outer rays, normalised to [0, 1]). Label: Left if left > right.
 *
 * Models (the circuits run on QAMPoser's state-vector simulator, the same
 * one QuantumDriver uses):
 * - 2 qubits, per block: RY(λ·left) on q0, RY(λ·right) on q1, trainable
 *   RY RZ on each qubit, CZ(0,1). Margin m = <Z0> - <Z1> (score Left minus
 *   score Right). Without the CZ each score would see one ray only (so
 *   with one block, where the CZ comes last, it changes nothing).
 * - 1 qubit, data re-uploading: RY(λ·left), trainable RY RZ, RY(λ·right),
 *   trainable RY RZ. Margin m = <Z>.
 * Both: P(Left) = sigmoid(β·m), i.e. a softmax over the scaled scores
 * β<Z0>, β<Z1> for two qubits, logistic loss for one; loss = mean cross-entropy.
 *
 * Gradients: parameter-shift, exact for RY/RZ (generator with eigenvalues
 * ±1/2): dm/dθ = (m(θ + π/2) - m(θ - π/2)) / 2, then the chain rule
 * dL/dθ = mean over samples of β·(P(Left) - [label is Left])·dm/dθ.
 */
import { expectationZ, simulateStatevector, type SimulationGate } from '@qamposer/react';
import type { CarState, PhysicsConfig } from './car';
import { Observer, type ObservationConfig } from './observer';
import type { Track } from './track';

export type ToyQubits = 1 | 2;
/** 0 = steer Left (left ray longer), 1 = steer Right */
export type ToyLabel = 0 | 1;

export interface ToySample {
  /** left ray (+60°), in [0, 1] */
  left: number;
  /** right ray (-60°), in [0, 1] */
  right: number;
  label: ToyLabel;
}

export interface ToyBatch {
  seed: number;
  /** the track the readings were taken on */
  track: string;
  train: ToySample[];
  /** held out: never used for a gradient, only to report accuracy */
  test: ToySample[];
}

/** Encoding scale: RY(λ·x) maps a reading in [0, 1] to [0, π]. */
export const LAMBDA = Math.PI;
/** Readout scale: P(Left) = sigmoid(β·m). */
export const BETA = 3;
export const N_TRAIN = 64;
export const N_TEST = 32;
/** The bundled track the readings are taken on. */
export const TOY_TRACK = 'oval';
/** The drivers' outer rays: +60° looks left, -60° right (theta grows to the left). */
export const TOY_OBSERVATION: ObservationConfig = {
  ray_angles_deg: [60, -60],
  ray_max_dist: 30,
  features: ['rays'],
  lookahead_m: 15,
};
const SHIFT = Math.PI / 2;

// ---------------------------------------------------------------- random

/** mulberry32: a small seeded PRNG, uniform in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ------------------------------------------------------------------ data

export interface SampleOptions {
  /** max |lateral offset| as a fraction of the half width */
  lateral?: number;
  /** max |heading error| against the track tangent (rad) */
  heading?: number;
}

/** `count` readings at random poses on the track: a random centerline point,
 *  a random sideways offset, a heading near the tangent. */
export function sampleReadings(
  track: Track,
  physics: PhysicsConfig,
  count: number,
  random: () => number,
  { lateral = 0.8, heading = 0.5 }: SampleOptions = {},
): ToySample[] {
  const observer = new Observer(track, physics, TOY_OBSERVATION);
  const out: ToySample[] = [];
  while (out.length < count) {
    const i = Math.floor(random() * track.n);
    const u = (2 * random() - 1) * lateral * track.halfWidth;
    const d = (2 * random() - 1) * heading;
    const state: CarState = [
      track.cx[i] + u * track.nx[i],
      track.cy[i] + u * track.ny[i],
      Math.atan2(track.ty[i], track.tx[i]) + d,
      0,
    ];
    const [left, right] = observer.observe(state);
    if (left === right) continue; // a tie has no right answer
    out.push({ left, right, label: left > right ? 0 : 1 });
  }
  return out;
}

/** The frozen batch for a seed: N_TRAIN training and N_TEST held-out readings. */
export function makeBatch(track: Track, physics: PhysicsConfig, seed: number, nTrain = N_TRAIN, nTest = N_TEST): ToyBatch {
  const random = rng(seed * 2 + 1);
  const all = sampleReadings(track, physics, nTrain + nTest, random);
  return { seed, track: track.name, train: all.slice(0, nTrain), test: all.slice(nTrain) };
}

// ----------------------------------------------------------------- model

export interface ToyModel {
  qubits: ToyQubits;
  /** encode + train blocks (2 qubits); always 1 for the re-uploading qubit */
  blocks: number;
  nParams: number;
}

export function toyModel(qubits: ToyQubits, blocks = 2): ToyModel {
  if (qubits === 1) return { qubits, blocks: 1, nParams: 4 };
  if (!Number.isInteger(blocks) || blocks < 1) throw new Error(`bad block count ${blocks}`);
  return { qubits, blocks, nParams: 4 * blocks };
}

/** Which gate a parameter is: params[k] is the `gate` of `qubit` in `block`
 *  (for one qubit, "block" 0 follows the left ray, 1 the right ray). */
export function paramRef(model: ToyModel, k: number): { block: number; qubit: number; gate: 'RY' | 'RZ' } {
  const gate = k % 2 ? 'RZ' : 'RY';
  if (model.qubits === 1) return { block: k >> 1, qubit: 0, gate };
  return { block: Math.floor(k / 4), qubit: (k >> 1) % 2, gate };
}

/** The bound circuit for one reading, in QAMPoser's gate format. */
export function toyGates(model: ToyModel, params: ArrayLike<number>, left: number, right: number): SimulationGate[] {
  const gates: SimulationGate[] = [];
  let position = 0;
  const add = (g: Omit<SimulationGate, 'position'>) => gates.push({ ...g, position: position++ } as SimulationGate);
  if (model.qubits === 1) {
    add({ type: 'RY', qubit: 0, parameter: LAMBDA * left });
    add({ type: 'RY', qubit: 0, parameter: params[0] });
    add({ type: 'RZ', qubit: 0, parameter: params[1] });
    add({ type: 'RY', qubit: 0, parameter: LAMBDA * right });
    add({ type: 'RY', qubit: 0, parameter: params[2] });
    add({ type: 'RZ', qubit: 0, parameter: params[3] });
    return gates;
  }
  for (let b = 0; b < model.blocks; b++) {
    add({ type: 'RY', qubit: 0, parameter: LAMBDA * left });
    add({ type: 'RY', qubit: 1, parameter: LAMBDA * right });
    for (let q = 0; q < 2; q++) {
      add({ type: 'RY', qubit: q, parameter: params[b * 4 + q * 2] });
      add({ type: 'RZ', qubit: q, parameter: params[b * 4 + q * 2 + 1] });
    }
    add({ type: 'CZ', control: 0, target: 1 });
  }
  return gates;
}

/** <Z_i> of every qubit for one reading (exact state vector). */
export function toyExpectations(model: ToyModel, params: ArrayLike<number>, left: number, right: number): Float64Array {
  const state = simulateStatevector(model.qubits, toyGates(model, params, left, right));
  return Float64Array.from({ length: model.qubits }, (_, i) => expectationZ(state, i));
}

/** Score Left minus score Right: <Z0> - <Z1>, or <Z> for one qubit. In [-2, 2] / [-1, 1]. */
export function margin(model: ToyModel, params: ArrayLike<number>, left: number, right: number): number {
  const z = toyExpectations(model, params, left, right);
  return model.qubits === 1 ? z[0] : z[0] - z[1];
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/** P(steer Left) for one reading. */
export function probLeft(model: ToyModel, params: ArrayLike<number>, left: number, right: number): number {
  return sigmoid(BETA * margin(model, params, left, right));
}

/** -log P(label) from the margin, stable for large |β·m|. */
function crossEntropy(m: number, label: ToyLabel): number {
  const z = label === 0 ? BETA * m : -BETA * m; // logit of the true class
  return z > 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z));
}

export interface Evaluation {
  loss: number;
  accuracy: number;
}

/** Mean cross-entropy and accuracy (P(Left) > 0.5 means Left) on samples. */
export function evaluate(model: ToyModel, params: ArrayLike<number>, samples: readonly ToySample[]): Evaluation {
  let loss = 0;
  let right = 0;
  for (const s of samples) {
    const m = margin(model, params, s.left, s.right);
    loss += crossEntropy(m, s.label);
    if ((m > 0 ? 0 : 1) === s.label) right++;
  }
  const n = Math.max(samples.length, 1);
  return { loss: loss / n, accuracy: right / n };
}

/** The mean loss and its exact gradient by the parameter-shift rule
 *  (2 extra circuits per parameter and sample). */
export function lossAndGradient(
  model: ToyModel,
  params: ArrayLike<number>,
  samples: readonly ToySample[],
): Evaluation & { grad: Float64Array } {
  const grad = new Float64Array(model.nParams);
  const shifted = Float64Array.from(params);
  let loss = 0;
  let right = 0;
  for (const s of samples) {
    const m = margin(model, params, s.left, s.right);
    loss += crossEntropy(m, s.label);
    if ((m > 0 ? 0 : 1) === s.label) right++;
    const dLdm = BETA * (sigmoid(BETA * m) - (s.label === 0 ? 1 : 0));
    for (let k = 0; k < model.nParams; k++) {
      const theta = shifted[k];
      shifted[k] = theta + SHIFT;
      const plus = margin(model, shifted, s.left, s.right);
      shifted[k] = theta - SHIFT;
      const minus = margin(model, shifted, s.left, s.right);
      shifted[k] = theta;
      grad[k] += dLdm * (plus - minus) / 2;
    }
  }
  const n = Math.max(samples.length, 1);
  for (let k = 0; k < grad.length; k++) grad[k] /= n;
  return { loss: loss / n, accuracy: right / n, grad };
}

/** Initial angles for a seed: small random angles, uniform in [-spread, spread] rad. */
export function initParams(model: ToyModel, seed: number, spread = 1): Float64Array {
  const random = rng(seed * 2 + 2);
  return Float64Array.from({ length: model.nParams }, () => (2 * random() - 1) * spread);
}

// --------------------------------------------------------------- trainer

export interface ToyPoint {
  step: number;
  trainLoss: number;
  trainAccuracy: number;
  testLoss: number;
  testAccuracy: number;
}

export const LEARNING_RATES = [0.02, 0.08, 0.3] as const;
export const DEFAULT_LEARNING_RATE = 0.08;
/** The panel's pace: slow enough to watch the decision regions move. */
export const STEPS_PER_SECOND = 15;
/** The trainer stops itself here: the curves have long flattened out. */
export const MAX_STEPS = 300;

const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-8;

/**
 * Full-batch training with Adam: every step is one epoch over the 64
 * training readings (one gradient by parameter shift, one update). The
 * history holds train and held-out loss and accuracy before step 1 and
 * after every step.
 */
export class ToyTrainer {
  readonly model: ToyModel;
  readonly batch: ToyBatch;
  readonly params: Float64Array;
  readonly history: ToyPoint[] = [];
  learningRate: number;
  private readonly m: Float64Array;
  private readonly v: Float64Array;
  private t = 0;

  constructor(batch: ToyBatch, qubits: ToyQubits, { learningRate = DEFAULT_LEARNING_RATE, blocks = 2 } = {}) {
    this.batch = batch;
    this.model = toyModel(qubits, blocks);
    this.params = initParams(this.model, batch.seed);
    this.learningRate = learningRate;
    this.m = new Float64Array(this.model.nParams);
    this.v = new Float64Array(this.model.nParams);
    this.record();
  }

  get steps(): number {
    return this.t;
  }

  get done(): boolean {
    return this.t >= MAX_STEPS;
  }

  get last(): ToyPoint {
    return this.history[this.history.length - 1];
  }

  /** One Adam step on the full training batch. */
  step(): ToyPoint {
    const { grad } = lossAndGradient(this.model, this.params, this.batch.train);
    this.t++;
    const c1 = 1 - ADAM_B1 ** this.t;
    const c2 = 1 - ADAM_B2 ** this.t;
    for (let k = 0; k < grad.length; k++) {
      this.m[k] = ADAM_B1 * this.m[k] + (1 - ADAM_B1) * grad[k];
      this.v[k] = ADAM_B2 * this.v[k] + (1 - ADAM_B2) * grad[k] * grad[k];
      this.params[k] -= (this.learningRate * this.m[k]) / c1 / (Math.sqrt(this.v[k] / c2) + ADAM_EPS);
    }
    return this.record();
  }

  /** P(Left) on a res × res grid over (left, right) in [0, 1]²: row r is
   *  right = (r + 0.5) / res from the bottom up, column c is left = (c + 0.5) / res. */
  decisionGrid(res: number): Float64Array {
    const out = new Float64Array(res * res);
    for (let r = 0; r < res; r++) {
      for (let c = 0; c < res; c++) {
        out[r * res + c] = probLeft(this.model, this.params, (c + 0.5) / res, (r + 0.5) / res);
      }
    }
    return out;
  }

  private record(): ToyPoint {
    const train = evaluate(this.model, this.params, this.batch.train);
    const test = evaluate(this.model, this.params, this.batch.test);
    const point = {
      step: this.t,
      trainLoss: train.loss,
      trainAccuracy: train.accuracy,
      testLoss: test.loss,
      testAccuracy: test.accuracy,
    };
    this.history.push(point);
    return point;
  }
}
