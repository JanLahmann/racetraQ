/**
 * Greedy closed-loop drive of one agent car, the way the demo server drives
 * it: a decision every substeps_per_decision substeps (the first at t = 0),
 * projection after every substep, stop at the first lap or the first
 * off-track substep. The parity tests replay tools/export_browser.py's numpy
 * rollouts against this.
 */
import { actionSet } from './actions';
import type { Driver } from './agents';
import { stepCar, type CarState, type PhysicsConfig } from './car';
import type { Observer } from './observer';
import { pyMod, type Track } from './track';

export interface RolloutResult {
  states: CarState[];
  actions: number[];
  q: Float64Array[];
  obs: Float64Array[];
  result: 'lap' | 'crash' | 'timeout';
  lapTime: number | null;
  substeps: number | null;
}

export function rollout(
  track: Track,
  physics: PhysicsConfig,
  observer: Observer,
  driver: Driver,
  maxDecisions: number,
): RolloutResult {
  const table = actionSet(driver.nActions);
  const [x0, y0, h0] = track.startPose();
  let state: CarState = [x0, y0, h0, 0];
  let sPrev = track.project(state[0], state[1]).s;
  const total = track.totalLength;
  let progress = 0;
  let substep = 0;
  const out: RolloutResult = {
    states: [],
    actions: [],
    q: [],
    obs: [],
    result: 'timeout',
    lapTime: null,
    substeps: null,
  };
  for (let d = 0; d < maxDecisions; d++) {
    const obs = observer.observe(state);
    const { q, action } = driver.decide(obs);
    out.states.push(state);
    out.obs.push(obs);
    out.q.push(q);
    out.actions.push(action);
    const [steer, throttle, brake] = table[action];
    for (let k = 0; k < physics.substeps_per_decision; k++) {
      state = stepCar(physics, state, steer, throttle, brake);
      substep++;
      const { s, lateral } = track.project(state[0], state[1]);
      progress += pyMod(s - sPrev + 0.5 * total, total) - 0.5 * total;
      sPrev = s;
      if (progress >= total) {
        return { ...out, result: 'lap', lapTime: substep * physics.dt, substeps: substep };
      }
      if (Math.abs(lateral) > track.halfWidth) {
        return { ...out, result: 'crash', substeps: substep };
      }
    }
  }
  return out;
}
