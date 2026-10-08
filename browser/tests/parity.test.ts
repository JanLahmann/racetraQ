/**
 * Parity of the TypeScript port with the numpy implementation, against the
 * reference values tools/export_browser.py writes to fixtures/parity.json.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { actionSet } from '../src/sim/actions';
import { makeDriver, QuantumDriver, type DriverFile } from '../src/sim/agents';
import { stepCar, type CarState, type PhysicsConfig } from '../src/sim/car';
import { Observer, type ObservationConfig } from '../src/sim/observer';
import { rollout } from '../src/sim/rollout';
import { Track, type TrackJson } from '../src/sim/track';

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const fixture = read('./fixtures/parity.json');
const manifest = read('../public/data/manifest.json');
const physics: PhysicsConfig = manifest.physics;
const trackJson = (id: string): TrackJson => read(`../public/data/tracks/${id}.json`);
const driverFile = (id: string): DriverFile => read(`../public/data/drivers/${id}.json`);
const tracks = new Map<string, Track>(
  Object.keys(fixture.tracks).map((id) => [id, new Track(trackJson(id), manifest.resample_spacing)]),
);

const TOL = 1e-9;
function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tol = TOL) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const scale = Math.max(1, Math.abs(expected[i]));
    if (Math.abs(actual[i] - expected[i]) > tol * scale) {
      throw new Error(`index ${i}: ${actual[i]} vs ${expected[i]}`);
    }
  }
}

describe.each(Object.keys(fixture.tracks))('track %s', (id) => {
  const ref = fixture.tracks[id];
  const track = tracks.get(id)!;

  it('resamples like Track._resample', () => {
    expect(track.n).toBe(ref.n_points);
    expectClose([track.totalLength, track.maxAbsCurvature], [ref.total_length, ref.max_abs_curvature]);
    expectClose(track.startPose(), ref.start_pose);
    ref.samples.index.forEach((i: number, k: number) => {
      expectClose([track.cx[i], track.cy[i]], ref.samples.centerline[k]);
      expectClose([track.nx[i], track.ny[i]], ref.samples.normals[k]);
      expectClose([track.curvature[i], track.s[i]], [ref.samples.curvature[k], ref.samples.s[k]]);
    });
  });

  it('projects points like Track.project', () => {
    ref.project.points.forEach(([x, y]: [number, number], k: number) => {
      const { s, lateral } = track.project(x, y);
      expectClose([s, lateral], [ref.project.s[k], ref.project.lateral[k]]);
    });
  });

  it('raycasts like Track.raycast', () => {
    ref.raycast.origins.forEach(([x, y]: [number, number], k: number) => {
      const d = track.raycast(x, y, ref.raycast.angles[k], ref.raycast.max_dist);
      expectClose([d], [ref.raycast.dist[k]]);
    });
  });

  it('looks ahead like curvature_ahead and tangent_angle', () => {
    const { s, lookahead, kappa } = ref.curvature_ahead;
    expectClose(s.map((v: number) => track.curvatureAhead(v, lookahead)), kappa);
    expectClose(ref.tangent_angle.s.map((v: number) => track.tangentAngle(v)), ref.tangent_angle.angle);
  });
});

describe('car physics', () => {
  it('steps like CarPhysics.step', () => {
    const { states, controls, next } = fixture.physics;
    states.forEach((state: CarState, k: number) => {
      const [steer, throttle, brake] = controls[k];
      expectClose(stepCar(physics, state, steer, throttle, brake), next[k]);
    });
  });
});

describe('observer', () => {
  const config: ObservationConfig = fixture.observer.config;
  it.each(Object.keys(fixture.observer.cases))('observes engineered features on %s', (id) => {
    const observer = new Observer(tracks.get(id)!, physics, config);
    expect(observer.nFeatures).toBe(14);
    const { states, obs } = fixture.observer.cases[id];
    states.forEach((state: CarState, k: number) => expectClose(observer.observe(state), obs[k]));
  });
});

describe('Q-functions', () => {
  it.each(Object.keys(fixture.q_values))('%s matches numpy', (id) => {
    const driver = makeDriver(driverFile(id));
    const ref = fixture.q_values[id];
    ref.obs.forEach((obs: number[], k: number) => {
      const decision = driver.decide(obs);
      expectClose(decision.q, ref.q[k]);
      if (ref.expectations) {
        expectClose((driver as QuantumDriver).expectations(obs), ref.expectations[k]);
      }
    });
  });
});

describe('greedy rollouts', () => {
  it.each(fixture.rollouts.map((r: { driver: string; track: string }) => [r.driver, r.track]) as [string, string][])(
    '%s on %s',
    (driverId: string, trackId: string) => {
      const ref = fixture.rollouts.find(
        (r: { driver: string; track: string }) => r.driver === driverId && r.track === trackId,
      );
      const file = driverFile(driverId);
      const track = tracks.get(trackId)!;
      const driver = makeDriver(file);
      const observer = new Observer(track, physics, file.observation);

      // teacher-forced: the numpy state of every decision gives the same
      // observation, Q-values and action
      ref.states.forEach((state: CarState, k: number) => {
        const obs = observer.observe(state);
        expectClose(obs, ref.obs[k]);
        const decision = driver.decide(obs);
        expectClose(decision.q, ref.q[k]);
        expect(decision.action).toBe(ref.actions[k]);
      });

      // closed loop: the browser drives the same lap
      const ours = rollout(track, physics, observer, driver, 600);
      expect(ours.result).toBe(ref.result);
      expect(ours.substeps).toBe(ref.substeps);
      expect(ours.actions).toEqual(ref.actions);
      ours.states.forEach((state, k) => expectClose(state, ref.states[k], 1e-7));
      expect(actionSet(driver.nActions).length).toBe(driver.nActions);
    },
  );
});

describe('live world', () => {
  it('drives the reference lap with the same timing as the rollout', async () => {
    const { World } = await import('../src/sim/world');
    const ref = fixture.rollouts.find((r: { driver: string }) => r.driver === 'quantum_gp');
    const file = driverFile('quantum_gp');
    const track = tracks.get('gp')!;
    const world = new World(track, physics);
    const laps: number[] = [];
    world.on((e) => {
      if (e.kind === 'lap') laps.push(e.lapTime);
    });
    world.addCar({
      id: 'q',
      kind: 'quantum',
      label: 'quantum',
      color: '#7a5cff',
      driver: makeDriver(file),
      observer: new Observer(track, physics, file.observation),
    });
    for (let k = 0; k < ref.substeps; k++) world.step();
    expect(laps).toHaveLength(1);
    expect(laps[0]).toBeCloseTo(ref.lap_time, 9);
    expect(world.cars[0].decisions).toBe(ref.actions.length);
  });
});
