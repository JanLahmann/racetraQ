/** "Predict, then check": covering a sensor, nudging an angle, judging a guess — inference only. */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { makeDriver, MlpDriver, QuantumDriver, type DriverFile } from '../src/sim/agents';
import type { PhysicsConfig } from '../src/sim/car';
import { Observer } from '../src/sim/observer';
import { Track } from '../src/sim/track';
import { World } from '../src/sim/world';
import {
  compare,
  coveredObservation,
  coverSensors,
  isLive,
  judgeGuess,
  liveRotations,
  nudgeAngle,
  nudgedDriver,
  thetaIndex,
  trainedAngle,
  type AngleRef,
} from '../src/sim/whatif';

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const manifest = read('../public/data/manifest.json');
const physics: PhysicsConfig = manifest.physics;
const driverFile = (id: string): DriverFile => read(`../public/data/drivers/${id}.json`);

/** A deterministic observation in [0, 1] (no RNG involved). */
const sampleObs = (n: number, k = 0) => Float64Array.from({ length: n }, (_, i) => (Math.sin(1.7 * i + 0.9 * k + 0.3) + 1) / 2);

const QUANTUM = ['quantum_oval', 'quantum_oval_q6', 'quantum_oval_q8', 'quantum_gp_q10'];

describe('cover a sensor', () => {
  it.each([...QUANTUM, 'mlp_oval', 'mlp_pro'])('%s: equals the driver on the modified observation', (id) => {
    const driver = makeDriver(driverFile(id));
    const obs = sampleObs(driver.nFeatures);
    const before = Float64Array.from(obs);
    for (let i = 0; i < driver.nFeatures; i++) {
      const manual = Float64Array.from(obs);
      manual[i] = 0;
      const expected = driver.decide(manual);
      const covered = coverSensors(driver, obs, [i]);
      expect(Array.from(covered.q)).toEqual(Array.from(expected.q));
      expect(covered.action).toBe(expected.action);
    }
    expect(Array.from(obs)).toEqual(Array.from(before)); // the observation itself is untouched
  });

  it('covers several sensors at once and rejects unknown ones', () => {
    const obs = [0.5, 0.25, 0.75];
    expect(Array.from(coveredObservation(obs, [0, 2]))).toEqual([0, 0.25, 0]);
    expect(Array.from(coveredObservation(obs, []))).toEqual(obs);
    expect(() => coveredObservation(obs, [3])).toThrow();
    expect(() => coveredObservation(obs, [-1])).toThrow();
  });

  it('works for the MLP driver too', () => {
    const driver = makeDriver(driverFile('mlp_oval'));
    expect(driver).toBeInstanceOf(MlpDriver);
    const obs = sampleObs(driver.nFeatures, 3);
    const original = driver.decide(obs);
    const changed = Array.from({ length: driver.nFeatures }, (_, i) => compare(original, coverSensors(driver, obs, [i])));
    // covering an input with a non-zero value moves at least one score
    expect(changed.some((c) => c.delta.some((d) => d !== 0))).toBe(true);
  });
});

describe('nudge an angle', () => {
  it.each(QUANTUM)('%s: a nudge of 0 reproduces the Q-values exactly', (id) => {
    const driver = makeDriver(driverFile(id)) as QuantumDriver;
    const obs = sampleObs(driver.nFeatures, 1);
    const original = driver.decide(obs);
    for (const ref of [
      { block: 0, qubit: 0, gate: 'RY' },
      { block: driver.nLayers - 1, qubit: driver.nQubits - 1, gate: 'RZ' },
    ] as AngleRef[]) {
      const again = nudgeAngle(driver, obs, ref, 0);
      expect(Array.from(again.q)).toEqual(Array.from(original.q));
      expect(Array.from(again.expectations!)).toEqual(Array.from(original.expectations!));
      expect(again.action).toBe(original.action);
    }
  });

  it('changes exactly one parameter, on a copy', () => {
    const driver = makeDriver(driverFile('quantum_oval_q8')) as QuantumDriver;
    const theta = Float64Array.from(driver.theta);
    const ref: AngleRef = { block: 2, qubit: 3, gate: 'RZ' };
    const copy = nudgedDriver(driver, ref, 0.5);
    const k = thetaIndex(driver, ref);
    expect(k).toBe((2 * 8 + 3) * 2 + 1);
    expect(copy.theta[k]).toBe(theta[k] + 0.5);
    expect(trainedAngle(driver, ref)).toBe(theta[k]);
    for (let i = 0; i < theta.length; i++) if (i !== k) expect(copy.theta[i]).toBe(theta[i]);
    expect(Array.from(copy.lam)).toEqual(Array.from(driver.lam));
    expect(Array.from(copy.w)).toEqual(Array.from(driver.w));
    expect(Array.from(copy.b)).toEqual(Array.from(driver.b));
    expect(Array.from(driver.theta)).toEqual(Array.from(theta)); // the original is untouched
  });

  it('moves the Q-values through a live gate', () => {
    const driver = makeDriver(driverFile('quantum_oval')) as QuantumDriver;
    const obs = sampleObs(driver.nFeatures, 2);
    const original = driver.decide(obs);
    const ref: AngleRef = { block: 0, qubit: 0, gate: 'RY' };
    expect(isLive(driver, ref)).toBe(true);
    const { delta } = compare(original, nudgeAngle(driver, obs, ref, 1));
    expect(Math.max(...delta.map(Math.abs))).toBeGreaterThan(1e-6);
  });

  it('rejects rotations the circuit does not have', () => {
    const driver = makeDriver(driverFile('quantum_oval')) as QuantumDriver;
    expect(() => thetaIndex(driver, { block: 4, qubit: 0, gate: 'RY' })).toThrow();
    expect(() => thetaIndex(driver, { block: 0, qubit: 4, gate: 'RZ' })).toThrow();
  });
});

describe('light cone', () => {
  // dead-gate counts from racetraq.agents.quantum.lightcone.live_gates
  it.each([
    [4, 4, 4, 0, 4],
    [6, 4, 4, 2, 8],
    [8, 5, 4, 6, 14],
    [10, 6, 4, 12, 22],
    [10, 6, 6, 6, 16],
  ])('n=%i L=%i A=%i: %i dead RY, %i dead RZ like lightcone.py', (n, layers, actions, deadRy, deadRz) => {
    const { ry, rz } = liveRotations(n, layers, actions);
    const dead = (g: boolean[][]) => g.flat().filter((v) => !v).length;
    expect(dead(ry)).toBe(deadRy);
    expect(dead(rz)).toBe(deadRz);
  });

  it('matches lightcone.py gate by gate (8 qubits, 5 blocks, RZ)', () => {
    const { rz } = liveRotations(8, 5, 4);
    expect(rz.map((row) => row.map(Number))).toEqual([
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 0, 0, 1],
      [1, 1, 1, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
    ]);
  });

  it.each(['quantum_oval_q8', 'quantum_oval_q10'])('%s: a dead gate cannot move any Q-value', (id) => {
    const driver = makeDriver(driverFile(id)) as QuantumDriver;
    const { ry, rz } = liveRotations(driver.nQubits, driver.nLayers, driver.nActions);
    let deadChecked = 0;
    for (let l = 0; l < driver.nLayers; l++) {
      for (let i = 0; i < driver.nQubits; i++) {
        for (const [gate, live] of [['RY', ry[l][i]], ['RZ', rz[l][i]]] as const) {
          if (live) continue;
          deadChecked++;
          for (let k = 0; k < 2; k++) {
            const obs = sampleObs(driver.nFeatures, k);
            const { delta } = compare(driver.decide(obs), nudgeAngle(driver, obs, { block: l, qubit: i, gate }, 2.1));
            expect(Math.max(...delta.map(Math.abs))).toBeLessThan(1e-9);
          }
        }
      }
    }
    expect(deadChecked).toBeGreaterThan(0);
  });
});

describe('judge a guess', () => {
  it('scores right and wrong picks', () => {
    const q = [0.1, 0.9, 0.4, -0.2];
    expect(judgeGuess(q, 1)).toEqual({ right: true, gap: 0, margin: 0.9 - 0.4 });
    const wrong = judgeGuess(q, 3);
    expect(wrong.right).toBe(false);
    expect(wrong.gap).toBeCloseTo(1.1, 12);
    expect(() => judgeGuess(q, 4)).toThrow();
  });

  it('takes the first maximum on a tie, like the driver', () => {
    expect(judgeGuess([0.5, 0.5], 0).right).toBe(true);
    expect(judgeGuess([0.5, 0.5], 1)).toMatchObject({ right: false, gap: 0 });
  });
});

describe('the live race is untouched', () => {
  it('what-ifs leave the car, its decision and the world as they were', () => {
    const file = driverFile('quantum_oval_q8');
    const track = new Track(read('../public/data/tracks/oval.json'), manifest.resample_spacing);
    const world = new World(track, physics);
    const driver = makeDriver(file) as QuantumDriver;
    const car = world.addCar({
      id: 'q',
      kind: 'quantum',
      label: 'quantum',
      color: '#7a5cff',
      driver,
      observer: new Observer(track, physics, file.observation),
    });
    for (let k = 0; k < 5 * physics.substeps_per_decision; k++) world.step();
    const snapshot = JSON.stringify({
      state: car.state,
      obs: Array.from(car.obs!),
      q: Array.from(car.decision!.q),
      action: car.action,
      decisions: car.decisions,
      substep: world.substep,
      theta: Array.from(driver.theta),
    });
    const decision = car.decision!;

    coverSensors(driver, car.obs!, [0, 3]);
    nudgeAngle(driver, car.obs!, { block: 1, qubit: 2, gate: 'RY' }, -1.3);
    judgeGuess(decision.q, 0);

    expect(car.decision).toBe(decision);
    expect(
      JSON.stringify({
        state: car.state,
        obs: Array.from(car.obs!),
        q: Array.from(car.decision!.q),
        action: car.action,
        decisions: car.decisions,
        substep: world.substep,
        theta: Array.from(driver.theta),
      }),
    ).toBe(snapshot);

    // and the race goes on exactly as a twin race without what-ifs
    const twin = new World(track, physics);
    const twinCar = twin.addCar({
      id: 'q',
      kind: 'quantum',
      label: 'quantum',
      color: '#7a5cff',
      driver: makeDriver(file),
      observer: new Observer(track, physics, file.observation),
    });
    for (let k = 0; k < 10 * physics.substeps_per_decision; k++) {
      if (k >= 5 * physics.substeps_per_decision) world.step();
      twin.step();
    }
    expect(car.state).toEqual(twinCar.state);
    expect(car.action).toBe(twinCar.action);
  });
});
