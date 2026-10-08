/**
 * The [observation] feature pipeline — a port of CarObserver in
 * traqmania/env/racing_env.py. Every scalar is normalized to [0, 1].
 */
import type { PhysicsConfig, CarState } from './car';
import { Track, pyMod } from './track';

export interface ObservationConfig {
  ray_angles_deg: number[];
  ray_max_dist: number;
  features: string[];
  lookahead_m: number;
}

const FEATURE_LABELS: Record<string, string> = {
  speed: 'speed',
  curvature_ahead: 'curvature ahead',
  lateral_offset: 'lateral offset',
  heading_error: 'heading error',
  corner_speed_ratio: 'corner speed',
};
const CURVATURE_KINDS = new Set(['curvature_ahead', 'corner_speed_ratio']);

const clip = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

function rayLabel(deg: number): string {
  return deg === 0 ? 'ray 0°' : `ray ${deg > 0 ? '+' : ''}${deg}°`;
}

function parseFeature(kind: string): [string, number | null] {
  const [base, arg] = kind.split(':');
  if (arg === undefined) return [base, null];
  if (!CURVATURE_KINDS.has(base)) throw new Error(`feature '${kind}' takes no lookahead`);
  const metres = Number(arg);
  if (!(metres > 0)) throw new Error(`bad lookahead in feature '${kind}'`);
  return [base, metres];
}

export class Observer {
  readonly rayAngles: number[];
  readonly featureNames: string[] = [];
  readonly nFeatures: number;
  /** [start, end) of the rays block inside the observation, or null */
  readonly raysSlice: [number, number] | null = null;
  private readonly parsed: [string, number | null][];

  constructor(
    readonly track: Track,
    readonly physics: PhysicsConfig,
    readonly config: ObservationConfig,
  ) {
    this.rayAngles = config.ray_angles_deg.map((d) => (d * Math.PI) / 180);
    this.parsed = config.features.map(parseFeature);
    for (const [base, metres] of this.parsed) {
      if (base === 'rays') {
        const start = this.featureNames.length;
        this.raysSlice = [start, start + this.rayAngles.length];
        this.featureNames.push(...config.ray_angles_deg.map(rayLabel));
      } else if (FEATURE_LABELS[base] === undefined) {
        throw new Error(`unknown observation feature '${base}'`);
      } else {
        this.featureNames.push(
          metres === null ? FEATURE_LABELS[base] : `${FEATURE_LABELS[base]} ${metres}m`,
        );
      }
    }
    this.nFeatures = this.featureNames.length;
  }

  /** Raw ray distances (world units, capped at ray_max_dist). */
  rays(state: CarState): number[] {
    const [x, y, theta] = state;
    return this.rayAngles.map((a) => this.track.raycast(x, y, theta + a, this.config.ray_max_dist));
  }

  observe(state: CarState, rawRays?: number[]): Float64Array {
    const { track, physics, config } = this;
    const [x, y, theta, v] = state;
    const out = new Float64Array(this.nFeatures);
    let proj: { s: number; lateral: number } | null = null;
    const projection = () => (proj ??= track.project(x, y));
    let k = 0;
    for (const [base, metres] of this.parsed) {
      if (base === 'rays') {
        const dist = rawRays ?? this.rays(state);
        for (const d of dist) out[k++] = clip(d / config.ray_max_dist, 0, 1);
        continue;
      }
      let feat: number;
      if (base === 'speed') {
        feat = clip(v / physics.v_max, 0, 1);
      } else if (base === 'lateral_offset') {
        feat = clip((projection().lateral / track.halfWidth + 1) / 2, 0, 1);
      } else if (base === 'heading_error') {
        const err = pyMod(theta - track.tangentAngle(projection().s) + Math.PI, 2 * Math.PI) - Math.PI;
        feat = clip((err / Math.PI + 1) / 2, 0, 1);
      } else {
        const kappa = track.curvatureAhead(projection().s, metres ?? config.lookahead_m);
        if (base === 'curvature_ahead') {
          feat = clip(kappa / Math.max(track.maxAbsCurvature, 1e-9), 0, 1);
        } else {
          const radius = 1 / Math.max(kappa, 1e-6);
          const vSafe = Math.sqrt(
            Math.max(0, 2 * physics.k_steer * physics.v_turn * radius - physics.v_turn ** 2),
          );
          feat = clip(v / Math.max(vSafe, 1e-6), 0, 2) / 2;
        }
      }
      out[k++] = feat;
    }
    return out;
  }
}
