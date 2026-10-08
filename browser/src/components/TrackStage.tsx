/**
 * The race canvas. The simulation runs here in fixed 60 Hz substeps; the
 * drawing is the server demo's own renderer (traqmania/web/js/race.js), fed
 * the same car messages the demo server broadcasts, at its 20 Hz rate.
 */
import { useEffect, useRef } from 'react';
import { RaceRenderer, type RenderCar, type TrackPayload } from '@demo/race.js';
import type { Track } from '../sim/track';
import type { World } from '../sim/world';

interface Props {
  world: World;
  track: Track;
  /** renderer mode: 'attract' (watch), 'race' or 'evolution' */
  mode: string;
  paused: boolean;
  speed: number;
  showRays: boolean;
  children?: React.ReactNode;
}

/** Longest real-time gap the loop catches up on (tab switches etc.). */
const MAX_CATCHUP_S = 0.25;
/** server [server] broadcast_hz = 20 at 60 Hz physics */
const BROADCAST_EVERY = 3;

/** runtime.track_payload for a browser Track. */
export function trackPayload(track: Track): TrackPayload {
  const [x, y, theta] = track.startPose();
  const n = track.n;
  const pts = (xs: ArrayLike<number>, ys: ArrayLike<number>, offset = 0) =>
    Array.from({ length: n }, (_, i) => [xs[offset + i], ys[offset + i]] as [number, number]);
  return {
    name: track.name,
    half_width: track.halfWidth,
    total_length: track.totalLength,
    checkpoints: track.checkpoints,
    theme: track.theme,
    start: { x, y, theta },
    centerline: pts(track.cx, track.cy),
    left: pts(track.bndAx, track.bndAy, 0),
    right: pts(track.bndAx, track.bndAy, n),
  };
}

/** The state message the demo server would broadcast for this world. */
export function stateMessage(world: World): { cars: RenderCar[] } {
  const cars: RenderCar[] = world.cars.map((car) => {
    const [x, y, theta, v] = car.state;
    const rays = car.rays && car.observer
      ? car.rays.map((d) => Math.min(Math.max(d / car.observer!.config.ray_max_dist, 0), 1))
      : undefined;
    return {
      id: car.id,
      kind: car.id === 'pro' ? 'pro' : car.kind,
      label: car.label,
      x,
      y,
      theta,
      v,
      rays: car.frozenUntil === null ? rays : undefined,
      off_track: car.frozenUntil !== null,
    };
  });
  const ghost = world.ghostPose();
  if (ghost) cars.push({ id: 'ghost', kind: 'quantum', ghost: true, x: ghost[0], y: ghost[1], theta: ghost[2], v: 0 });
  return { cars };
}

export function TrackStage({ world, track, mode, paused, speed, showRays, children }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<RaceRenderer | null>(null);
  const live = useRef({ paused, speed });
  live.current = { paused, speed };

  useEffect(() => {
    const renderer = new RaceRenderer(canvasRef.current!);
    renderer.start();
    rendererRef.current = renderer;
    return () => {
      renderer.running = false;
    };
  }, []);

  useEffect(() => {
    rendererRef.current!.setTrack(trackPayload(track));
  }, [track]);

  useEffect(() => {
    rendererRef.current!.setMode(mode);
  }, [mode]);

  useEffect(() => {
    rendererRef.current!.showRays = showRays;
  }, [showRays]);

  useEffect(() => {
    const renderer = rendererRef.current!;
    renderer.pushState(stateMessage(world));
    const off = world.on((event) => {
      if (event.kind === 'lap') renderer.addEffect(event.clean ? 'clean_lap' : 'lap', event.carId);
      else if (event.kind === 'crash') renderer.addEffect('crash', event.carId);
    });
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const dt = world.physics.dt;
    const frame = (now: number) => {
      const elapsed = Math.min((now - last) / 1000, MAX_CATCHUP_S);
      last = now;
      if (!live.current.paused) {
        acc += elapsed * live.current.speed;
        while (acc >= dt) {
          world.step();
          acc -= dt;
          if (world.substep % BROADCAST_EVERY === 0) renderer.pushState(stateMessage(world));
        }
      } else {
        acc = 0;
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      off();
    };
  }, [world]);

  return (
    <div className="stage" style={{ '--track-aspect': track.aspect().toFixed(3) } as React.CSSProperties}>
      <canvas ref={canvasRef} aria-label="Race track" />
      {children}
    </div>
  );
}
