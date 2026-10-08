/** Static data written by tools/export_browser.py into public/data/. */
import type { CircuitShape, DriverFile } from './sim/agents';
import type { PhysicsConfig } from './sim/car';
import type { TrackJson } from './sim/track';
import type { GhostLap } from './sim/world';

export interface FreshEval {
  episodes?: number;
  lapped_fraction?: number;
  mean_lap?: number;
  best_lap?: number;
}

export interface DriverInfo {
  id: string;
  agent: 'quantum' | 'mlp';
  /** track it was trained on, or 'multi' */
  track: string;
  kind: 'driver' | 'stage' | 'warmstart';
  stage: number | null;
  episodes: number | null;
  n_params: number;
  n_actions: number;
  circuit?: CircuitShape;
  hidden?: number;
  features: string[];
  n_rays: number;
  provenance: string | null;
  fresh_eval: FreshEval | null;
}

export interface Manifest {
  version: number;
  physics: PhysicsConfig;
  resample_spacing: number;
  max_decisions: number;
  action_set: [number, number, number][];
  action_labels: string[];
  tracks: { id: string; name: string }[];
  drivers: DriverInfo[];
  ghosts: { track: string; lap_time: number; kind: string; driver: string }[];
}

const BASE = `${import.meta.env.BASE_URL}data/`;
const cache = new Map<string, Promise<unknown>>();

function fetchJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(BASE + path).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

export const loadManifest = () => fetchJson<Manifest>('manifest.json');
export const loadTrack = (id: string) => fetchJson<TrackJson>(`tracks/${id}.json`);
export const loadDriver = (id: string) => fetchJson<DriverFile>(`drivers/${id}.json`);
export const loadGhost = (track: string) => fetchJson<GhostLap>(`ghosts/${track}.json`);

/** The quantum drivers of a track by qubit count (main drivers only). */
export function quantumDrivers(manifest: Manifest, track: string): DriverInfo[] {
  return manifest.drivers
    .filter((d) => d.agent === 'quantum' && d.kind === 'driver' && d.track === track)
    .sort((a, b) => (a.circuit?.n_qubits ?? 0) - (b.circuit?.n_qubits ?? 0));
}

export function findDriver(manifest: Manifest, id: string): DriverInfo | undefined {
  return manifest.drivers.find((d) => d.id === id);
}

/** Evolution cars: the bundled stage snapshots, the last replaced by the
 *  shipped best-snapshot driver (runtime.evolution_stage_specs). */
export function evolutionStages(manifest: Manifest, track: string): { label: string; id: string }[] {
  const stages = manifest.drivers
    .filter(
      (d) => d.agent === 'quantum' && d.kind === 'stage' && d.track === track && d.circuit?.n_qubits === 4,
    )
    .sort((a, b) => (a.stage ?? 0) - (b.stage ?? 0))
    .map((d) => ({ label: d.episodes ? `ep ${d.episodes}` : `stage ${d.stage}`, id: d.id }));
  const best = findDriver(manifest, `quantum_${track}`);
  if (stages.length && best) {
    stages[stages.length - 1] = {
      label: best.episodes ? `best (of ${best.episodes} ep run)` : 'best',
      id: best.id,
    };
  }
  return stages;
}
