/** Car physics — a port of racetraq/env/car.py (one car, one substep). */

export interface PhysicsConfig {
  dt: number;
  accel: number;
  brake: number;
  drag: number;
  v_max: number;
  v_turn: number;
  k_steer: number;
  substeps_per_decision: number;
}

/** [x, y, theta, v] */
export type CarState = [number, number, number, number];

/** Steering authority in [0, 1]; peaks at v = v_turn, zero at rest. */
export function steerFalloff(p: PhysicsConfig, v: number): number {
  return (2.0 * v * p.v_turn) / (p.v_turn ** 2 + v ** 2);
}

/** Advance ONE substep (semi-implicit Euler: speed first, then heading and
 *  position with the new speed). steer in [-1, 1], +1 turns left. */
export function stepCar(
  p: PhysicsConfig,
  state: CarState,
  steer: number,
  throttle: number,
  brake: number,
): CarState {
  const [x, y, theta, v] = state;
  const dv = (throttle * p.accel - brake * p.brake - p.drag * v) * p.dt;
  const vNew = Math.min(Math.max(v + dv, 0.0), p.v_max);
  const thetaNew = theta + steer * p.k_steer * steerFalloff(p, vNew) * p.dt;
  return [
    x + vNew * Math.cos(thetaNew) * p.dt,
    y + vNew * Math.sin(thetaNew) * p.dt,
    thetaNew,
    vNew,
  ];
}
