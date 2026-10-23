/** The toy that trains in the browser: data, circuits, parameter-shift gradients, training. */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { PhysicsConfig } from '../src/sim/car';
import { Observer } from '../src/sim/observer';
import { Track } from '../src/sim/track';
import {
  BETA,
  evaluate,
  initParams,
  LAMBDA,
  lossAndGradient,
  makeBatch,
  margin,
  N_TEST,
  N_TRAIN,
  paramRef,
  probLeft,
  rng,
  toyExpectations,
  toyGates,
  toyModel,
  ToyTrainer,
  TOY_OBSERVATION,
  TOY_TRACK,
  type ToyModel,
} from '../src/sim/toy';

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const manifest = read('../public/data/manifest.json');
const physics: PhysicsConfig = manifest.physics;
const track = new Track(read(`../public/data/tracks/${TOY_TRACK}.json`), manifest.resample_spacing);
const batch = makeBatch(track, physics, 1);

// ---------------------------------------------- a direct state vector (no QAMPoser)

type C = [number, number];
const mul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const add = (a: C, b: C): C => [a[0] + b[0], a[1] + b[1]];
const ry = (t: number): C[][] => [
  [[Math.cos(t / 2), 0], [-Math.sin(t / 2), 0]],
  [[Math.sin(t / 2), 0], [Math.cos(t / 2), 0]],
];
const rz = (t: number): C[][] => [
  [[Math.cos(t / 2), -Math.sin(t / 2)], [0, 0]],
  [[0, 0], [Math.cos(t / 2), Math.sin(t / 2)]],
];

/** Apply a 2×2 matrix to qubit q of an n-qubit state (index bit q = qubit q). */
function apply(state: C[], m: C[][], q: number): C[] {
  const out = state.map((a) => [...a] as C);
  for (let i = 0; i < state.length; i++) {
    if (i & (1 << q)) continue;
    const j = i | (1 << q);
    out[i] = add(mul(m[0][0], state[i]), mul(m[0][1], state[j]));
    out[j] = add(mul(m[1][0], state[i]), mul(m[1][1], state[j]));
  }
  return out;
}

/** <Z_i> per qubit, written out gate by gate from the model description. */
function directExpectations(model: ToyModel, p: ArrayLike<number>, left: number, right: number): number[] {
  let s: C[] = Array.from({ length: 1 << model.qubits }, (_, i) => (i === 0 ? [1, 0] : [0, 0]));
  if (model.qubits === 1) {
    s = apply(s, ry(LAMBDA * left), 0);
    s = apply(s, ry(p[0]), 0);
    s = apply(s, rz(p[1]), 0);
    s = apply(s, ry(LAMBDA * right), 0);
    s = apply(s, ry(p[2]), 0);
    s = apply(s, rz(p[3]), 0);
  } else {
    for (let b = 0; b < model.blocks; b++) {
      s = apply(s, ry(LAMBDA * left), 0);
      s = apply(s, ry(LAMBDA * right), 1);
      for (let q = 0; q < 2; q++) {
        s = apply(s, ry(p[b * 4 + q * 2]), q);
        s = apply(s, rz(p[b * 4 + q * 2 + 1]), q);
      }
      s[3] = [-s[3][0], -s[3][1]]; // CZ(0, 1)
    }
  }
  return Array.from({ length: model.qubits }, (_, q) =>
    s.reduce((z, a, i) => z + (i & (1 << q) ? -1 : 1) * (a[0] ** 2 + a[1] ** 2), 0),
  );
}

const MODELS: [string, ToyModel][] = [
  ['1 qubit', toyModel(1)],
  ['2 qubits, 1 block', toyModel(2, 1)],
  ['2 qubits, 2 blocks', toyModel(2, 2)],
];

describe('frozen batch', () => {
  it('has 64 training and 32 held-out readings in [0, 1], labelled by the longer ray', () => {
    expect(batch.train).toHaveLength(N_TRAIN);
    expect(batch.test).toHaveLength(N_TEST);
    for (const s of [...batch.train, ...batch.test]) {
      expect(s.left).toBeGreaterThanOrEqual(0);
      expect(s.left).toBeLessThanOrEqual(1);
      expect(s.right).toBeGreaterThanOrEqual(0);
      expect(s.right).toBeLessThanOrEqual(1);
      expect(s.label).toBe(s.left > s.right ? 0 : 1);
    }
    const lefts = batch.train.filter((s) => s.label === 0).length;
    expect(lefts).toBeGreaterThan(N_TRAIN / 4); // both classes well represented
    expect(lefts).toBeLessThan((3 * N_TRAIN) / 4);
  });

  it('is the same for the same seed and differs for another', () => {
    expect(makeBatch(track, physics, 1)).toEqual(batch);
    expect(makeBatch(track, physics, 2).train).not.toEqual(batch.train);
  });

  it('reads the rays with the observer: +60° looks left', () => {
    // a car left of the centreline, heading along it, sees the left wall closer
    const observer = new Observer(track, physics, TOY_OBSERVATION);
    const [x, y, theta] = track.startPose();
    const shift = 0.5 * track.halfWidth;
    const [left, right] = observer.observe([x + shift * track.nx[0], y + shift * track.ny[0], theta, 0]);
    expect(left).toBeLessThan(right);
    expect(left).toBeCloseTo(track.raycast(x + shift * track.nx[0], y + shift * track.ny[0], theta + Math.PI / 3, 30) / 30, 12);
  });

  it('seeds a reproducible PRNG', () => {
    const a = rng(7);
    const b = rng(7);
    const xs = Array.from({ length: 5 }, a);
    expect(Array.from({ length: 5 }, b)).toEqual(xs);
    for (const v of xs) expect(v >= 0 && v < 1).toBe(true);
  });
});

describe('circuits', () => {
  it.each(MODELS)('%s: expectations match a direct state-vector computation', (_, model) => {
    const random = rng(11);
    for (let trial = 0; trial < 6; trial++) {
      const p = Float64Array.from({ length: model.nParams }, () => (2 * random() - 1) * Math.PI);
      const left = random();
      const right = random();
      const z = toyExpectations(model, p, left, right);
      const direct = directExpectations(model, p, left, right);
      for (let q = 0; q < model.qubits; q++) expect(z[q]).toBeCloseTo(direct[q], 12);
    }
  });

  it('builds the documented gate list', () => {
    const two = toyGates(toyModel(2), new Float64Array(8), 0.5, 0.25);
    expect(two.map((g) => g.type).join(' ')).toBe('RY RY RY RZ RY RZ CZ RY RY RY RZ RY RZ CZ');
    expect(two[0]).toMatchObject({ qubit: 0, parameter: LAMBDA * 0.5 });
    expect(two[1]).toMatchObject({ qubit: 1, parameter: LAMBDA * 0.25 });
    const one = toyGates(toyModel(1), new Float64Array(4), 0.5, 0.25);
    expect(one.map((g) => g.type).join(' ')).toBe('RY RY RZ RY RY RZ');
    expect(one[3]).toMatchObject({ qubit: 0, parameter: LAMBDA * 0.25 });
    expect(paramRef(toyModel(2), 6)).toEqual({ block: 1, qubit: 1, gate: 'RY' });
    expect(paramRef(toyModel(1), 3)).toEqual({ block: 1, qubit: 0, gate: 'RZ' });
  });

  it('without the CZ (one block: it comes last) each score sees only its own ray', () => {
    const model = toyModel(2, 1);
    const p = initParams(model, 3);
    const a = toyExpectations(model, p, 0.3, 0.6);
    const b = toyExpectations(model, p, 0.3, 0.9); // right ray changed
    expect(b[0]).toBeCloseTo(a[0], 12); // score Left unchanged
    // with two blocks the first CZ carries the right ray into <Z0>
    const two = toyModel(2, 2);
    const q = initParams(two, 3);
    expect(toyExpectations(two, q, 0.3, 0.9)[0]).not.toBeCloseTo(toyExpectations(two, q, 0.3, 0.6)[0], 6);
  });

  it('turns the margin into P(Left) = sigmoid(β·m)', () => {
    const model = toyModel(2);
    const p = initParams(model, 1);
    const m = margin(model, p, 0.4, 0.7);
    expect(probLeft(model, p, 0.4, 0.7)).toBeCloseTo(1 / (1 + Math.exp(-BETA * m)), 14);
  });
});

describe('parameter-shift gradient', () => {
  it.each(MODELS)('%s: matches central finite differences', (_, model) => {
    const p = initParams(model, 5, Math.PI);
    const samples = batch.train.slice(0, 16);
    const { loss, grad } = lossAndGradient(model, p, samples);
    expect(loss).toBeCloseTo(evaluate(model, p, samples).loss, 14);
    const h = 1e-5;
    for (let k = 0; k < model.nParams; k++) {
      const plus = Float64Array.from(p);
      const minus = Float64Array.from(p);
      plus[k] += h;
      minus[k] -= h;
      const fd = (evaluate(model, plus, samples).loss - evaluate(model, minus, samples).loss) / (2 * h);
      expect(Math.abs(grad[k] - fd), `param ${k}`).toBeLessThan(1e-6);
    }
  });

  it('is exactly zero for the last RZ on each wire (it commutes with the Z readout)', () => {
    const model = toyModel(2);
    const { grad } = lossAndGradient(model, initParams(model, 2), batch.train);
    expect(grad[5]).toBeCloseTo(0, 14); // block 2, q0, RZ
    expect(grad[7]).toBeCloseTo(0, 14); // block 2, q1, RZ
    expect(Math.abs(grad[4])).toBeGreaterThan(1e-4);
  });
});

describe('training', () => {
  it('2 qubits, seed 1: held-out accuracy ≥ 0.9 within 40 steps, and the loss falls', () => {
    const trainer = new ToyTrainer(batch, 2);
    const start = trainer.last;
    let reached = -1;
    while (trainer.steps < 40) {
      const p = trainer.step();
      if (reached < 0 && p.testAccuracy >= 0.9) reached = p.step;
    }
    expect(reached).toBeGreaterThan(0);
    expect(reached).toBeLessThanOrEqual(40); // measured: step 13
    expect(trainer.last.trainLoss).toBeLessThan(0.5 * start.trainLoss);
    expect(trainer.history).toHaveLength(41);
    // after 100 steps it stays there
    while (trainer.steps < 100) trainer.step();
    expect(trainer.last.testAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(trainer.last.trainAccuracy).toBeGreaterThanOrEqual(0.9);
  });

  it('1 qubit, seed 2: held-out accuracy ≥ 0.9 within 60 steps', () => {
    const trainer = new ToyTrainer(makeBatch(track, physics, 2), 1);
    while (trainer.steps < 60 && trainer.last.testAccuracy < 0.9) trainer.step();
    expect(trainer.last.testAccuracy).toBeGreaterThanOrEqual(0.9); // measured: step 11
  });

  it('is reproducible: same seed, same trajectory; another seed, another one', () => {
    const a = new ToyTrainer(makeBatch(track, physics, 4), 2);
    const b = new ToyTrainer(makeBatch(track, physics, 4), 2);
    const c = new ToyTrainer(makeBatch(track, physics, 5), 2);
    for (let i = 0; i < 25; i++) {
      a.step();
      b.step();
      c.step();
    }
    expect(b.history).toEqual(a.history);
    expect(Array.from(b.params)).toEqual(Array.from(a.params));
    expect(Array.from(c.params)).not.toEqual(Array.from(a.params));
  });

  it('a step stays cheap at 2 qubits (a guard against gross slowdowns)', () => {
    // about 3 ms on a laptop; shared CI runners under parallel test load measured 33 ms,
    // so this takes the fastest of several short bursts and only catches a 10x regression
    const trainer = new ToyTrainer(batch, 2);
    trainer.step(); // warm up
    let best = Infinity;
    for (let burst = 0; burst < 5; burst++) {
      const t0 = performance.now();
      for (let i = 0; i < 4; i++) trainer.step();
      best = Math.min(best, (performance.now() - t0) / 4);
    }
    expect(best).toBeLessThan(60);
  });

  it('maps P(Left) over the (left, right) square', () => {
    const trainer = new ToyTrainer(batch, 2);
    while (trainer.steps < 60) trainer.step();
    const res = 8;
    const grid = trainer.decisionGrid(res);
    expect(grid).toHaveLength(res * res);
    // row r is right = (r + 0.5) / res, column c is left = (c + 0.5) / res
    for (const [r, c] of [[0, 0], [2, 5], [7, 1]]) {
      expect(grid[r * res + c]).toBe(probLeft(trainer.model, trainer.params, (c + 0.5) / res, (r + 0.5) / res));
    }
    // inside the data (rays of 9–15 m): the longer ray wins
    expect(probLeft(trainer.model, trainer.params, 0.5, 0.3)).toBeGreaterThan(0.5);
    expect(probLeft(trainer.model, trainer.params, 0.3, 0.5)).toBeLessThan(0.5);
  });
});
