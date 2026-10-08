/**
 * Track geometry — a line-by-line port of traqmania/env/track.py.
 *
 * The raw centerline is resampled to ~uniform arc-length spacing; tangents,
 * left normals, unsigned curvature and the two boundary polylines are
 * precomputed. Queries (projection, raycasts, curvature lookahead) use a
 * full search over all segments instead of Python's spatial hash: the hash
 * only prunes candidates, so both return the same nearest segment / first
 * hit (ties resolve to the lowest segment index in both).
 */

export interface TrackTheme {
  surface?: string;
  edge?: string;
}

export interface TrackJson {
  name?: string;
  id?: string;
  centerline: [number, number][];
  half_width: number;
  checkpoints?: number[];
  theme?: TrackTheme;
}

export const MIN_CORNER_RADIUS = 6.0;
export const MIN_HALF_WIDTH = 3.0;

/** Python's round(): halves go to the even neighbour. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Python float modulo: result has the sign of the divisor. */
export function pyMod(a: number, b: number): number {
  const r = a % b;
  return r !== 0 && r < 0 !== b < 0 ? r + b : r;
}

/** numpy.interp for increasing xp (x inside [xp[0], xp[-1]]). */
function interp(x: number, xp: Float64Array, fp: Float64Array): number {
  const last = xp.length - 1;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[last]) return fp[last];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

const norm2 = (x: number, y: number) => Math.sqrt(x * x + y * y);

export class Track {
  readonly name: string;
  readonly halfWidth: number;
  readonly checkpoints: number[];
  readonly theme: TrackTheme;
  /** resampled centerline, interleaved [x0, y0, x1, y1, ...] */
  readonly cx: Float64Array;
  readonly cy: Float64Array;
  readonly s: Float64Array;
  readonly totalLength: number;
  readonly n: number;
  readonly tx: Float64Array;
  readonly ty: Float64Array;
  /** left normals */
  readonly nx: Float64Array;
  readonly ny: Float64Array;
  readonly curvature: Float64Array;
  readonly maxAbsCurvature: number;
  // centerline segments i -> i+1
  private readonly segDx: Float64Array;
  private readonly segDy: Float64Array;
  private readonly segLen: Float64Array;
  private readonly segLen2: Float64Array;
  private readonly segUx: Float64Array;
  private readonly segUy: Float64Array;
  // boundary segments: left ring then right ring
  readonly bndAx: Float64Array;
  readonly bndAy: Float64Array;
  private readonly bndDx: Float64Array;
  private readonly bndDy: Float64Array;

  constructor(data: TrackJson, resampleSpacing = 1.5) {
    this.name = String(data.name ?? data.id ?? 'track');
    this.halfWidth = Number(data.half_width);
    this.checkpoints = (data.checkpoints ?? []).map(Number);
    this.theme = data.theme ?? {};
    if (this.halfWidth < MIN_HALF_WIDTH) {
      throw new Error(`track '${this.name}': half_width ${this.halfWidth} < ${MIN_HALF_WIDTH}`);
    }

    // ---- resample (Track._resample)
    let pts = data.centerline.map(([x, y]) => [Number(x), Number(y)] as [number, number]);
    if (pts.length < 3) throw new Error(`track '${this.name}': centerline needs >= 3 points`);
    const first = pts[0];
    const lastPt = pts[pts.length - 1];
    if (norm2(first[0] - lastPt[0], first[1] - lastPt[1]) < 1e-9) pts = pts.slice(0, -1);
    const closed = [...pts, pts[0]];
    const m = closed.length;
    const segLen = new Float64Array(m - 1);
    let total = 0;
    for (let i = 0; i < m - 1; i++) {
      segLen[i] = norm2(closed[i + 1][0] - closed[i][0], closed[i + 1][1] - closed[i][1]);
      total += segLen[i];
    }
    const gap = norm2(pts[0][0] - pts[pts.length - 1][0], pts[0][1] - pts[pts.length - 1][1]);
    if (total <= 0 || gap > 0.25 * total) {
      throw new Error(`track '${this.name}': centerline loop is not closed`);
    }
    const cum = new Float64Array(m);
    for (let i = 1; i < m; i++) cum[i] = cum[i - 1] + segLen[i - 1];
    const xs = Float64Array.from(closed, (p) => p[0]);
    const ys = Float64Array.from(closed, (p) => p[1]);
    const n = Math.max(roundHalfEven(total / resampleSpacing), 8);
    const step = total / n;
    this.n = n;
    this.cx = new Float64Array(n);
    this.cy = new Float64Array(n);
    this.s = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const si = i * step;
      this.s[i] = si;
      this.cx[i] = interp(si, cum, xs);
      this.cy[i] = interp(si, cum, ys);
    }
    this.totalLength = total;

    // ---- tangents, normals, curvature (Track._precompute)
    const { cx, cy } = this;
    this.tx = new Float64Array(n);
    this.ty = new Float64Array(n);
    this.nx = new Float64Array(n);
    this.ny = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const nxt = (i + 1) % n;
      const prv = (i - 1 + n) % n;
      const dx = cx[nxt] - cx[prv];
      const dy = cy[nxt] - cy[prv];
      const len = norm2(dx, dy);
      this.tx[i] = dx / len;
      this.ty[i] = dy / len;
      this.nx[i] = -this.ty[i];
      this.ny[i] = this.tx[i];
    }
    const kappa = new Float64Array(n);
    let minRadius = Infinity;
    for (let i = 0; i < n; i++) {
      const i1 = (i + 1) % n;
      const i2 = (i + 2) % n;
      const a = norm2(cx[i1] - cx[i], cy[i1] - cy[i]);
      const b = norm2(cx[i2] - cx[i1], cy[i2] - cy[i1]);
      const c = norm2(cx[i2] - cx[i], cy[i2] - cy[i]);
      const area2 = Math.abs(
        (cx[i1] - cx[i]) * (cy[i2] - cy[i]) - (cy[i1] - cy[i]) * (cx[i2] - cx[i]),
      );
      kappa[i] = area2 > 1e-12 ? (2.0 * area2) / (a * b * c) : 0.0;
      const radius = area2 > 1e-12 ? (a * b * c) / (2.0 * area2) : Infinity;
      if (radius < minRadius) minRadius = radius;
    }
    if (minRadius < MIN_CORNER_RADIUS) {
      throw new Error(
        `track '${this.name}': min corner radius ${minRadius.toFixed(2)} < ${MIN_CORNER_RADIUS}`,
      );
    }
    this.curvature = new Float64Array(n);
    let maxK = 0;
    for (let i = 0; i < n; i++) {
      this.curvature[i] = kappa[(i - 1 + n) % n];
      if (this.curvature[i] > maxK) maxK = this.curvature[i];
    }
    this.maxAbsCurvature = maxK;

    // ---- centerline segments
    this.segDx = new Float64Array(n);
    this.segDy = new Float64Array(n);
    this.segLen = new Float64Array(n);
    this.segLen2 = new Float64Array(n);
    this.segUx = new Float64Array(n);
    this.segUy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      this.segDx[i] = cx[j] - cx[i];
      this.segDy[i] = cy[j] - cy[i];
      this.segLen[i] = norm2(this.segDx[i], this.segDy[i]);
      this.segLen2[i] = Math.max(this.segLen[i] ** 2, 1e-12);
      this.segUx[i] = this.segDx[i] / this.segLen[i];
      this.segUy[i] = this.segDy[i] / this.segLen[i];
    }

    // ---- boundary polylines and their segments
    const hw = this.halfWidth;
    this.bndAx = new Float64Array(2 * n);
    this.bndAy = new Float64Array(2 * n);
    this.bndDx = new Float64Array(2 * n);
    this.bndDy = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      this.bndAx[i] = cx[i] + this.nx[i] * hw;
      this.bndAy[i] = cy[i] + this.ny[i] * hw;
      this.bndAx[n + i] = cx[i] - this.nx[i] * hw;
      this.bndAy[n + i] = cy[i] - this.ny[i] * hw;
    }
    for (let ring = 0; ring < 2; ring++) {
      for (let i = 0; i < n; i++) {
        const k = ring * n + i;
        const k1 = ring * n + ((i + 1) % n);
        this.bndDx[k] = this.bndAx[k1] - this.bndAx[k];
        this.bndDy[k] = this.bndAy[k1] - this.bndAy[k];
      }
    }
  }

  /** Height / width of the boundary bounding box (for layout). */
  aspect(): number {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let k = 0; k < this.bndAx.length; k++) {
      minX = Math.min(minX, this.bndAx[k]);
      maxX = Math.max(maxX, this.bndAx[k]);
      minY = Math.min(minY, this.bndAy[k]);
      maxY = Math.max(maxY, this.bndAy[k]);
    }
    return (maxY - minY) / Math.max(maxX - minX, 1e-9);
  }

  /** (x, y, heading) at s = 0, heading along the track tangent. */
  startPose(): [number, number, number] {
    return [this.cx[0], this.cy[0], Math.atan2(this.ty[0], this.tx[0])];
  }

  /** Nearest centerline point: arc length s (wrapped) and signed lateral
   *  offset (positive = left of the direction of travel). */
  project(x: number, y: number): { s: number; lateral: number } {
    const { cx, cy, segDx, segDy, segLen2, n } = this;
    let best = Infinity;
    let bestSeg = 0;
    let bestT = 0;
    let bestCx = 0;
    let bestCy = 0;
    for (let i = 0; i < n; i++) {
      const apx = x - cx[i];
      const apy = y - cy[i];
      let t = (apx * segDx[i] + apy * segDy[i]) / segLen2[i];
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = cx[i] + t * segDx[i];
      const qy = cy[i] + t * segDy[i];
      const dx = x - qx;
      const dy = y - qy;
      const d2 = dx * dx + dy * dy;
      if (d2 < best) {
        best = d2;
        bestSeg = i;
        bestT = t;
        bestCx = qx;
        bestCy = qy;
      }
    }
    const s = pyMod(this.s[bestSeg] + bestT * this.segLen[bestSeg], this.totalLength);
    const lateral = this.segUx[bestSeg] * (y - bestCy) - this.segUy[bestSeg] * (x - bestCx);
    return { s, lateral };
  }

  isInside(x: number, y: number): boolean {
    return Math.abs(this.project(x, y).lateral) <= this.halfWidth;
  }

  /** Distance from (ox, oy) along heading `angle` to the first boundary
   *  crossing, capped at maxDist. */
  raycast(ox: number, oy: number, angle: number, maxDist: number): number {
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    const { bndAx, bndAy, bndDx, bndDy } = this;
    let best = Infinity;
    for (let k = 0; k < bndAx.length; k++) {
      const ex = bndDx[k];
      const ey = bndDy[k];
      const denom = dx * ey - dy * ex;
      if (!(Math.abs(denom) > 1e-12)) continue;
      const aox = bndAx[k] - ox;
      const aoy = bndAy[k] - oy;
      const t = (aox * ey - aoy * ex) / denom;
      if (!(t > 1e-9) || t >= best) continue;
      const u = (aox * dy - aoy * dx) / denom;
      if (u >= -1e-9 && u <= 1.0 + 1e-9) best = t;
    }
    return Math.min(best, maxDist);
  }

  /** Max |kappa| over the arc-length window [s, s + lookahead] (wraps). */
  curvatureAhead(s: number, lookahead: number): number {
    const ds = this.totalLength / this.n;
    const first = Math.floor(s / ds);
    const count = Math.ceil(lookahead / ds) + 1;
    let best = -Infinity;
    for (let k = 0; k < count; k++) {
      const v = this.curvature[pyMod(first + k, this.n)];
      if (v > best) best = v;
    }
    return best;
  }

  /** Heading of the centerline tangent at the nearest resampled point. */
  tangentAngle(s: number): number {
    const ds = this.totalLength / this.n;
    const idx = pyMod(Math.floor(s / ds + 0.5), this.n);
    return Math.atan2(this.ty[idx], this.tx[idx]);
  }

}
