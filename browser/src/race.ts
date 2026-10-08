/** Builds a World for a track + mode from the static data. */
import { KIND_COLORS, STAGE_COLORS } from '@demo/race.js';
import {
  evolutionStages,
  findDriver,
  loadDriver,
  loadGhost,
  loadTrack,
  type Manifest,
} from './data';
import { makeDriver } from './sim/agents';
import { Observer } from './sim/observer';
import { Track } from './sim/track';
import { World, type CarSpec } from './sim/world';

export type Mode = 'watch' | 'race' | 'evolution';
export type Rival = 'none' | 'mlp' | 'pro';

// the demo renderer's palette, so the lap board matches the cars
export const COLORS = {
  quantum: KIND_COLORS.quantum,
  mlp: KIND_COLORS.mlp,
  pro: KIND_COLORS.pro,
  human: KIND_COLORS.human,
};
export { STAGE_COLORS };
export const COUNTDOWN_S = 3;

export interface RaceSetup {
  track: string;
  mode: Mode;
  /** quantum driver id (watch and race) */
  driver: string;
  rival: Rival;
  ghost: boolean;
}

export interface BuiltRace {
  world: World;
  track: Track;
  focusId: string;
}

async function agentSpec(
  manifest: Manifest,
  track: Track,
  driverId: string,
  id: string,
  label: string,
  color: string,
): Promise<CarSpec> {
  const file = await loadDriver(driverId);
  return {
    id,
    kind: file.agent,
    label,
    color,
    driver: makeDriver(file),
    observer: new Observer(track, manifest.physics, file.observation),
  };
}

export function quantumLabel(manifest: Manifest, driverId: string): string {
  const info = findDriver(manifest, driverId);
  const n = info?.circuit?.n_qubits ?? 4;
  return info?.track === 'multi' ? `Quantum · ${n} qubits · all-round` : `Quantum · ${n} qubits`;
}

export async function buildRace(manifest: Manifest, setup: RaceSetup): Promise<BuiltRace> {
  const track = new Track(await loadTrack(setup.track), manifest.resample_spacing);
  const world = new World(track, manifest.physics);
  world.stallDecisions = manifest.max_decisions;
  let focusId = 'quantum';
  const specs: CarSpec[] = [];

  if (setup.mode === 'evolution') {
    const stages = evolutionStages(manifest, setup.track);
    for (const [i, stage] of stages.entries()) {
      specs.push(
        await agentSpec(manifest, track, stage.id, `stage${i + 1}`, stage.label, STAGE_COLORS[i % STAGE_COLORS.length]),
      );
    }
    focusId = specs[specs.length - 1]?.id ?? '';
  } else {
    if (setup.mode === 'race') {
      specs.push({ id: 'human', kind: 'human', label: 'You', color: COLORS.human });
    }
    specs.push(
      await agentSpec(manifest, track, setup.driver, 'quantum', quantumLabel(manifest, setup.driver), COLORS.quantum),
    );
    if (setup.mode === 'watch' && setup.rival !== 'none') {
      const rivalId = setup.rival === 'pro' ? 'mlp_pro' : `mlp_${setup.track}`;
      const info = findDriver(manifest, rivalId);
      if (info) {
        const label =
          setup.rival === 'pro'
            ? `Classical pro · ${info.n_params.toLocaleString('en')} params`
            : `Classical MLP · ${info.n_params} params`;
        specs.push(await agentSpec(manifest, track, rivalId, setup.rival, label, setup.rival === 'pro' ? COLORS.pro : COLORS.mlp));
      }
    }
  }
  for (const spec of specs) world.addCar(spec);

  if (setup.mode === 'race') {
    world.holdUntil = Math.round(COUNTDOWN_S / manifest.physics.dt);
    for (const car of world.cars) world.respawn(car);
  }
  // the ghost is the track's 4-qubit driver's standing-start lap: in Watch it
  // only adds something when a different driver races it
  const ghostIsTheDriver = setup.mode === 'watch' && setup.driver === `quantum_${setup.track}`;
  if (
    setup.ghost &&
    setup.mode !== 'evolution' &&
    !ghostIsTheDriver &&
    manifest.ghosts.some((g) => g.track === setup.track)
  ) {
    world.setGhost(await loadGhost(setup.track), setup.mode === 'race' ? 'human' : 'quantum');
  }
  return { world, track, focusId };
}
