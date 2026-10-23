/** "Train a tiny circuit" in jsdom: opens, steps, trains, pauses, stops on close, one analytics event per run, stuck runs. */
import { readFileSync } from 'node:fs';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { About } from '../src/components/About';
import { isStuck, ToyPanel } from '../src/components/ToyPanel';
import type { Manifest } from '../src/data';
import { makeBatch, ToyTrainer } from '../src/sim/toy';
import { Track } from '../src/sim/track';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const read = (path: string) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const manifest: Manifest = read('../public/data/manifest.json');
const oval = read('../public/data/tracks/oval.json');

let host: HTMLDivElement;
let root: Root;
let umami: ReturnType<typeof vi.fn>;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  umami = vi.fn();
  (window as unknown as { umami: unknown }).umami = { track: umami };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (!String(url).endsWith('tracks/oval.json')) throw new Error(`unexpected fetch ${url}`);
      return { ok: true, json: async () => oval };
    }),
  );
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  delete (window as unknown as { umami?: unknown }).umami;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const $ = (sel: string) => host.querySelector<HTMLElement>(sel);
const button = (text: string) => {
  const b = Array.from(host.querySelectorAll('button')).find((el) => el.textContent?.includes(text));
  if (!b) throw new Error(`no button "${text}"`);
  return b;
};
const click = (el: HTMLElement) => act(() => el.click());
const status = () => $('.toy-status')?.textContent ?? '';
const step = () => Number(/Step (\d+)\//.exec(status())?.[1] ?? NaN);
const toyEvents = () => umami.mock.calls.filter(([name]) => name === 'racetraQ: toy train');
const wait = (ms: number) => act(() => new Promise((resolve) => setTimeout(resolve, ms)));

async function open(onClose = () => {}) {
  await act(async () => root.render(createElement(ToyPanel, { manifest, onClose })));
  await wait(0); // the track arrives
  expect($('.toy-status'), 'panel loaded').not.toBeNull();
}

describe('Train a tiny circuit', () => {
  it('opens clearly labelled, with the honest note and links to the real thing', async () => {
    await open();
    const dialog = $('[role="dialog"]')!;
    expect(dialog.getAttribute('aria-label')).toBe('Train a tiny circuit (a toy)');
    expect($('h2')?.textContent).toContain('a toy');
    expect($('.toy-honest')?.textContent).toContain('frozen batch by gradient descent');
    expect($('.toy-honest')?.textContent).toContain('parameter-shift');
    expect(dialog.textContent).toContain('half the difference of the two results is the exact slope');
    expect($('.toy-honest')?.textContent).toContain('double deep Q-learning');
    expect($('.toy-honest')?.textContent).toContain('quantum advantage');
    const links = Array.from(host.querySelectorAll('a')).map((a) => a.getAttribute('href'));
    expect(links.some((h) => h?.endsWith('docs/LEARN.md'))).toBe(true);
    expect(step()).toBe(0);
    expect(host.querySelectorAll('.toy-scatter circle')).toHaveLength(96); // 64 + 32 held out
    expect(host.querySelectorAll('.toy-scatter rect').length).toBeGreaterThan(100); // the decision map
    expect(toyEvents()).toHaveLength(0); // opening is not a run
  });

  it('Step trains one step and updates the trained angles', async () => {
    await open();
    const before = Array.from(host.querySelectorAll('.toy-circuit .rot .val')).map((e) => e.textContent);
    expect(before).toHaveLength(8);
    click(button('Step'));
    expect(step()).toBe(1);
    const after = Array.from(host.querySelectorAll('.toy-circuit .rot .val')).map((e) => e.textContent);
    expect(after).not.toEqual(before);
    expect(toyEvents()).toEqual([['racetraQ: toy train', { qubits: '2' }]]);
  });

  it('Train runs on its own, Pause stops it, one event per run', async () => {
    await open();
    click(button('Train'));
    expect(step()).toBe(1); // the click answers at once
    await wait(400);
    const running = step();
    expect(running).toBeGreaterThan(2);
    click(button('Pause'));
    const paused = step();
    await wait(200);
    expect(step()).toBe(paused);
    click(button('Train'));
    click(button('Pause'));
    expect(toyEvents()).toHaveLength(1); // still the same run
    // another circuit is another run
    click(button('1 qubit'));
    expect(step()).toBe(0);
    expect(host.querySelectorAll('.toy-circuit .rot .val')).toHaveLength(4);
    click(button('Step'));
    expect(toyEvents().map(([, data]) => data)).toEqual([{ qubits: '2' }, { qubits: '1' }]);
    // Reset draws a new seed and starts over
    click(button('Reset'));
    expect(step()).toBe(0);
    expect(status()).toContain('seed 2');
  });

  it('pauses when the tab is hidden', async () => {
    await open();
    click(button('Train'));
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(true);
    act(() => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    expect(button('Train')).toBeTruthy();
    const frozen = step();
    await wait(200);
    expect(step()).toBe(frozen);
    hidden.mockRestore();
  });

  it('closes (✕, Escape, backdrop) and stops training', async () => {
    const onClose = vi.fn();
    await open(onClose);
    click(button('Train'));
    click($('.close')!);
    act(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    click($('.modal-backdrop')!);
    expect(onClose).toHaveBeenCalledTimes(3);
    const steps = vi.spyOn(ToyTrainer.prototype, 'step');
    await wait(200);
    expect(steps).toHaveBeenCalled(); // still training while open
    steps.mockClear();
    act(() => root.render(createElement('div'))); // the parent closes the panel
    await wait(200);
    expect(steps).not.toHaveBeenCalled();
  });

  it('is offered in About', async () => {
    const onOpenToy = vi.fn();
    await act(async () => root.render(createElement(About, { manifest, onClose: () => {}, onOpenToy })));
    click(button('Train a tiny circuit'));
    expect(onOpenToy).toHaveBeenCalledOnce();
  });
});

describe('stuck', () => {
  it('flags a flat loss without the rule (1 qubit, seed 1: a local minimum), not a good run', () => {
    const oval_ = new Track(oval, manifest.resample_spacing);
    const stuck = new ToyTrainer(makeBatch(oval_, manifest.physics, 1), 1);
    const good = new ToyTrainer(makeBatch(oval_, manifest.physics, 1), 2);
    for (let i = 0; i < 100; i++) {
      stuck.step();
      good.step();
    }
    expect(isStuck(stuck.history)).toBe(true);
    expect(isStuck(good.history)).toBe(false);
    expect(isStuck(stuck.history.slice(0, 30))).toBe(false); // too early to say
  });
});
