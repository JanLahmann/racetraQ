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
import { fmtInt, t } from './i18n';

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
  /** the setup it was built for (labelCars) */
  setup: RaceSetup;
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
  return t(info?.track === 'multi' ? 'car.quantum_multi' : 'car.quantum', { n });
}

const rivalId = (setup: RaceSetup) => (setup.rival === 'pro' ? 'mlp_pro' : `mlp_${setup.track}`);

function rivalLabel(manifest: Manifest, setup: RaceSetup): string {
  const info = findDriver(manifest, rivalId(setup));
  const params = fmtInt(info?.n_params ?? 0);
  if (setup.rival === 'pro') return t('car.pro', { params });
  // the MLP is matched to the 4-qubit circuit (same 3 rays + speed), not to the larger ones
  const larger = (findDriver(manifest, setup.driver)?.circuit?.n_qubits ?? 4) > 4;
  return larger ? t('car.mlp_unmatched', { params, rays: info?.n_rays ?? 0 }) : t('car.mlp', { params });
}

/** (Re)label a race's cars in the active language: at build time and when
 *  the visitor switches language mid-race. */
export function labelCars(manifest: Manifest, race: BuiltRace): void {
  const { setup, world } = race;
  const stages = setup.mode === 'evolution' ? evolutionStages(manifest, setup.track) : [];
  for (const car of world.cars) {
    const stage = /^stage(\d+)$/.exec(car.id);
    if (stage) car.label = stages[Number(stage[1]) - 1]?.label ?? car.label;
    else if (car.id === 'human') car.label = t('car.you');
    else if (car.id === 'quantum') car.label = quantumLabel(manifest, setup.driver);
    else if (car.id === 'mlp' || car.id === 'pro') car.label = rivalLabel(manifest, setup);
  }
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
      specs.push({ id: 'human', kind: 'human', label: t('car.you'), color: COLORS.human });
    }
    specs.push(
      await agentSpec(manifest, track, setup.driver, 'quantum', quantumLabel(manifest, setup.driver), COLORS.quantum),
    );
    if (setup.mode === 'watch' && setup.rival !== 'none') {
      const id = rivalId(setup);
      if (findDriver(manifest, id)) {
        const color = setup.rival === 'pro' ? COLORS.pro : COLORS.mlp;
        specs.push(await agentSpec(manifest, track, id, setup.rival, rivalLabel(manifest, setup), color));
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
  return { world, track, focusId, setup };
}
