/** The brain panel's "predict, then check" flow in jsdom: guess, reveal, cover, step, analytics. */
import { readFileSync } from 'node:fs';
import { createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { act } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QuantumBrain } from '../src/components/QuantumBrain';
import { makeDriver, QuantumDriver, type DriverFile } from '../src/sim/agents';
import type { PhysicsConfig } from '../src/sim/car';
import { Observer } from '../src/sim/observer';
import { Track } from '../src/sim/track';
import { World, type WorldCar } from '../src/sim/world';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const manifest = read('../public/data/manifest.json');
const physics: PhysicsConfig = manifest.physics;

function raceOn(id: string): { world: World; car: WorldCar } {
  const file: DriverFile = read(`../public/data/drivers/${id}.json`);
  const track = new Track(read('../public/data/tracks/oval.json'), manifest.resample_spacing);
  const world = new World(track, physics);
  const car = world.addCar({
    id: 'q',
    kind: 'quantum',
    label: 'quantum',
    color: '#7a5cff',
    driver: makeDriver(file),
    observer: new Observer(track, physics, file.observation),
  });
  stepDecision(world, 3);
  return { world, car };
}

function stepDecision(world: World, n = 1) {
  for (let k = 0; k < n * physics.substeps_per_decision; k++) world.step();
}

let host: HTMLDivElement;
let root: Root;
let umami: ReturnType<typeof vi.fn>;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  umami = vi.fn();
  (window as unknown as { umami: unknown }).umami = { track: umami };
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  delete (window as unknown as { umami?: unknown }).umami;
});

function render(car: WorldCar, paused: boolean, guessMode: boolean) {
  act(() =>
    root.render(
      createElement(QuantumBrain, {
        car,
        wideCircuit: true, // keeps QAMPoser's editor out of jsdom
        onToggleWide: () => {},
        paused,
        guessMode,
        onGuessMode: () => {},
      }),
    ),
  );
}

const $$ = (sel: string) => Array.from(host.querySelectorAll<HTMLElement>(sel));
const click = (el: HTMLElement) => act(() => el.click());
const events = (name: string) => umami.mock.calls.filter(([n]) => n === name);

describe('predict, then check', () => {
  it('a running race shows the scores and no what-if tools', () => {
    const { car } = raceOn('quantum_oval');
    render(car, false, true);
    expect($$('.actions li')).toHaveLength(4);
    expect($$('.guess-grid button')).toHaveLength(0);
    expect($$('.features .mini')).toHaveLength(0);
    expect(host.querySelector('.nudge')).toBeNull();
  });

  it('hides the scores until a guess, then judges it; Step hides them again', () => {
    const { world, car } = raceOn('quantum_oval');
    render(car, true, true);
    expect($$('.actions')).toHaveLength(0);
    expect($$('.qubits .num').every((el) => el.textContent === '?')).toBe(true);
    const buttons = $$('.guess-grid button');
    expect(buttons).toHaveLength(4);

    click(buttons[car.decision!.action]);
    expect(host.querySelector('.verdict.right')).not.toBeNull();
    expect($$('.actions li.guessed.chosen')).toHaveLength(1);
    expect(events('racetraQ: predict')).toEqual([['racetraQ: predict', { result: 'right', qubits: '4' }]]);

    stepDecision(world);
    render(car, true, true);
    expect($$('.guess-grid button')).toHaveLength(4);
    click($$('.guess-grid button')[(car.decision!.action + 1) % 4]);
    expect(host.querySelector('.verdict.wrong')).not.toBeNull();
    expect(events('racetraQ: predict')[1][1]).toEqual({ result: 'wrong', qubits: '4' });
  });

  it('covering sensors shows the what-if scores and reports once per decision', () => {
    const { world, car } = raceOn('quantum_oval');
    const before = JSON.stringify({ state: car.state, obs: Array.from(car.obs!), q: Array.from(car.decision!.q) });
    render(car, true, false);
    const covers = $$('.features .mini');
    expect(covers).toHaveLength(car.observer!.nFeatures);

    click(covers[0]);
    expect($$('.actions.with-whatif')).toHaveLength(1);
    const covered = car.driver!.decide(Float64Array.from(car.obs!, (v, i) => (i === 0 ? 0 : v)));
    const shown = $$('.actions li:not(.head) .num.whatif').map((el) => el.textContent);
    expect(shown).toEqual(Array.from(covered.q, (q) => q.toFixed(2)));
    click($$('.features .mini')[1]);
    expect(events('racetraQ: what-if')).toHaveLength(1);

    click($$('.features .mini')[0]); // uncover
    click($$('.features .mini')[1]);
    expect($$('.actions.with-whatif')).toHaveLength(0);
    expect(JSON.stringify({ state: car.state, obs: Array.from(car.obs!), q: Array.from(car.decision!.q) })).toBe(before);

    stepDecision(world);
    render(car, true, false);
    click($$('.features .mini')[2]);
    expect(events('racetraQ: what-if')).toHaveLength(2);
    expect(events('racetraQ: what-if')[1][1]).toEqual({ kind: 'sensor' });
  });

  it('nudging an angle re-evaluates on a copy and flags dead gates', () => {
    const { car } = raceOn('quantum_oval_q8');
    const driver = car.driver as QuantumDriver;
    const theta = Array.from(driver.theta);
    render(car, true, false);
    click(host.querySelector<HTMLElement>('.nudge .chip')!);
    const [block, , gate] = $$('.nudge-pick select') as unknown as HTMLSelectElement[];
    const slider = host.querySelector<HTMLInputElement>('.nudge-slider input')!;

    // React listens for 'input' on range inputs; set the value the way a drag would
    const setValue = (el: HTMLInputElement | HTMLSelectElement, value: string, type: string) =>
      act(() => {
        const proto = el instanceof HTMLInputElement ? HTMLInputElement.prototype : HTMLSelectElement.prototype;
        Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(el, value);
        el.dispatchEvent(new Event(type, { bubbles: true }));
      });

    setValue(slider, '1.5', 'input');
    expect($$('.actions.with-whatif')).toHaveLength(1);
    setValue(slider, '-0.7', 'input');
    expect(events('racetraQ: what-if')).toEqual([['racetraQ: what-if', { kind: 'angle' }]]);
    expect(host.querySelector('.hint.dead')).toBeNull(); // block 1 RY q0 is live

    // the last block's RZ gates are outside every readout's light cone
    setValue(block, String(driver.nLayers - 1), 'change');
    setValue(gate, 'RZ', 'change');
    expect(host.querySelector('.hint.dead')).not.toBeNull();
    setValue(host.querySelector<HTMLInputElement>('.nudge-slider input')!, '2', 'input');
    const now = $$('.actions li:not(.head) > .num:not(.whatif)').map((el) => el.textContent);
    const whatIf = $$('.actions li:not(.head) .num.whatif').map((el) => el.textContent);
    expect(whatIf).toHaveLength(4);
    expect(whatIf).toEqual(now);
    expect(host.querySelector('.whatif-line.changed')).toBeNull();
    expect(Array.from(driver.theta)).toEqual(theta);
  });
});
