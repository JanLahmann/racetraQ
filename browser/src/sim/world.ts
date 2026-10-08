/**
 * The live race: agent cars, an optional human car and an optional ghost on
 * one track, advanced in 60 Hz physics substeps the way the demo server's
 * session does (traqmania/server/session.py): agents decide every
 * substeps_per_decision substeps (the first decision before the first
 * substep), every car is projected after every substep, a lap is one track
 * length of net progress, an agent that leaves the track respawns at once,
 * a human is frozen for a second first.
 */
import { actionSet } from './actions';
import type { Decision, Driver } from './agents';
import { stepCar, type CarState, type PhysicsConfig } from './car';
import type { Observer } from './observer';
import { pyMod, type Track } from './track';

export type CarKind = 'quantum' | 'mlp' | 'human';

export const HUMAN_RESPAWN_DELAY_S = 1.0;

export interface Controls {
  steer: number;
  throttle: number;
  brake: number;
}

export interface WorldCar {
  id: string;
  kind: CarKind;
  label: string;
  color: string;
  driver?: Driver;
  observer?: Observer;
  state: CarState;
  /** speed one substep ago (for the brake light) */
  prevV: number;
  s: number;
  progress: number;
  lap: number;
  lapStartSubstep: number;
  lastLap: number | null;
  bestLap: number | null;
  /** off track at least once this lap */
  lapDirty: boolean;
  offTrack: boolean;
  /** substep at which a frozen (crashed) human respawns */
  frozenUntil: number | null;
  controls: Controls;
  action: number;
  decisions: number;
  /** decisions since the car's current lap (or spawn) began */
  decisionsThisLap: number;
  obs?: Float64Array;
  rays?: number[];
  decision?: Decision;
}

export interface GhostLap {
  track: string;
  lap_time: number;
  kind?: string;
  driver?: string;
  /** [x, y, theta] at decision rate */
  points: [number, number, number][];
}

export type WorldEvent =
  | { kind: 'lap'; carId: string; lapTime: number; clean: boolean }
  | { kind: 'crash'; carId: string }
  | { kind: 'timeout'; carId: string }
  | { kind: 'decision'; carId: string };

export interface CarSpec {
  id: string;
  kind: CarKind;
  label: string;
  color: string;
  driver?: Driver;
  observer?: Observer;
}

export class World {
  substep = 0;
  cars: WorldCar[] = [];
  ghost: GhostLap | null = null;
  /** substep the ghost lap started at */
  ghostStart = 0;
  /** car whose laps restart the ghost */
  ghostFollows: string | null = null;
  humanControls: Controls = { steer: 0, throttle: 0, brake: 0 };
  /** cars hold still until this substep (race countdown) */
  holdUntil = 0;
  /** An agent car that has not finished a lap after this many decisions
   *  respawns — the training env's episode cap ([reward] max_decisions, 60 s),
   *  so a policy that parks (some early snapshots brake to a stop) does not
   *  stay stuck forever. 0 disables it. */
  stallDecisions = 0;
  private listeners = new Set<(event: WorldEvent) => void>();

  constructor(
    readonly track: Track,
    readonly physics: PhysicsConfig,
  ) {}

  get t(): number {
    return this.substep * this.physics.dt;
  }

  on(listener: (event: WorldEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(event: WorldEvent) {
    for (const listener of this.listeners) listener(event);
  }

  addCar(spec: CarSpec): WorldCar {
    const car: WorldCar = {
      ...spec,
      state: [0, 0, 0, 0],
      prevV: 0,
      s: 0,
      progress: 0,
      lap: 0,
      lapStartSubstep: 0,
      lastLap: null,
      bestLap: null,
      lapDirty: false,
      offTrack: false,
      frozenUntil: null,
      controls: { steer: 0, throttle: 0, brake: 0 },
      action: 0,
      decisions: 0,
      decisionsThisLap: 0,
    };
    this.respawn(car);
    this.cars.push(car);
    return car;
  }

  respawn(car: WorldCar) {
    const [x, y, theta] = this.track.startPose();
    car.state = [x, y, theta, 0];
    car.prevV = 0;
    car.s = this.track.project(x, y).s;
    car.progress = 0;
    car.lap = 0;
    car.lapStartSubstep = Math.max(this.substep, this.holdUntil);
    car.lapDirty = false;
    car.offTrack = false;
    car.frozenUntil = null;
    car.controls = { steer: 0, throttle: 0, brake: 0 };
    car.action = 0;
    car.decisionsThisLap = 0;
    if (car.id === this.ghostFollows) this.ghostStart = car.lapStartSubstep;
  }

  setGhost(ghost: GhostLap | null, follows: string | null) {
    this.ghost = ghost;
    this.ghostFollows = follows;
    const car = this.cars.find((c) => c.id === follows);
    this.ghostStart = car ? car.lapStartSubstep : this.substep;
  }

  /** Ghost pose at the current time, interpolated between its samples. */
  ghostPose(): [number, number, number] | null {
    const ghost = this.ghost;
    if (!ghost || ghost.points.length < 2) return null;
    const elapsed = (this.substep - this.ghostStart) * this.physics.dt;
    if (elapsed < 0) return ghost.points[0];
    const period = ghost.lap_time;
    const local = pyMod(elapsed, period);
    const decisionDt = this.physics.dt * this.physics.substeps_per_decision;
    const f = local / decisionDt;
    const i = Math.min(Math.floor(f), ghost.points.length - 1);
    const j = Math.min(i + 1, ghost.points.length - 1);
    const t = f - Math.floor(f);
    const [x0, y0, h0] = ghost.points[i];
    const [x1, y1, h1] = ghost.points[j];
    const dh = pyMod(h1 - h0 + Math.PI, 2 * Math.PI) - Math.PI;
    return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, h0 + dh * t];
  }

  private decide(car: WorldCar) {
    const { driver, observer } = car;
    if (!driver || !observer) return;
    const rays = observer.rays(car.state);
    const obs = observer.observe(car.state, rays);
    const decision = driver.decide(obs);
    const [steer, throttle, brake] = actionSet(driver.nActions)[decision.action];
    car.rays = rays;
    car.obs = obs;
    car.decision = decision;
    car.action = decision.action;
    car.controls = { steer, throttle, brake };
    car.decisions++;
    this.emit({ kind: 'decision', carId: car.id });
  }

  /** Advance one physics substep. */
  step() {
    if (this.substep < this.holdUntil) {
      this.substep++;
      return;
    }
    const decisionTick = (this.substep - this.holdUntil) % this.physics.substeps_per_decision === 0;
    this.substep++;
    const total = this.track.totalLength;
    for (const car of this.cars) {
      if (car.frozenUntil !== null) {
        if (this.substep >= car.frozenUntil) this.respawn(car);
        else continue;
      }
      if (car.kind === 'human') car.controls = { ...this.humanControls };
      else if (decisionTick) {
        if (this.stallDecisions > 0 && car.decisionsThisLap >= this.stallDecisions) {
          car.lapDirty = true;
          this.emit({ kind: 'timeout', carId: car.id });
          this.respawn(car);
        }
        this.decide(car);
        car.decisionsThisLap++;
      }
      const { steer, throttle, brake } = car.controls;
      car.prevV = car.state[3];
      car.state = stepCar(this.physics, car.state, steer, throttle, brake);
      const { s, lateral } = this.track.project(car.state[0], car.state[1]);
      car.progress += pyMod(s - car.s + 0.5 * total, total) - 0.5 * total;
      car.s = s;

      const lapsNow = Math.floor(car.progress / total);
      if (lapsNow > car.lap) {
        const lapTime = (this.substep - car.lapStartSubstep) * this.physics.dt;
        const clean = !car.lapDirty;
        car.lap = lapsNow;
        car.lastLap = lapTime;
        if (clean && (car.bestLap === null || lapTime < car.bestLap)) car.bestLap = lapTime;
        car.lapStartSubstep = this.substep;
        car.decisionsThisLap = 0;
        car.lapDirty = false;
        if (car.id === this.ghostFollows) this.ghostStart = this.substep;
        this.emit({ kind: 'lap', carId: car.id, lapTime, clean });
      }

      car.offTrack = Math.abs(lateral) > this.track.halfWidth;
      if (car.offTrack) {
        car.lapDirty = true;
        this.emit({ kind: 'crash', carId: car.id });
        if (car.kind === 'human') {
          car.frozenUntil = this.substep + Math.round(HUMAN_RESPAWN_DELAY_S / this.physics.dt);
        } else {
          this.respawn(car);
        }
      }
    }
  }

  /** Current lap time of a car (seconds). */
  lapClock(car: WorldCar): number {
    return Math.max(0, (this.substep - car.lapStartSubstep) * this.physics.dt);
  }
}
