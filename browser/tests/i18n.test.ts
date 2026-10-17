/** The EN/DE interface: complete tables, matching placeholders, language order, formatting. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { de } from '../src/i18n/de';
import { en } from '../src/i18n/en';
import {
  fmtInt,
  fmtNum,
  fmtSeconds,
  getLang,
  initLang,
  placeholders,
  resolveLang,
  setLang,
  t,
  TABLES,
  type Key,
} from '../src/i18n';

describe('string tables', () => {
  it('have the same keys', () => {
    expect(Object.keys(de).sort()).toEqual(Object.keys(en).sort());
  });

  it('have the same placeholders per key', () => {
    for (const key of Object.keys(en) as Key[]) {
      expect(placeholders(de[key]), key).toEqual(placeholders(en[key]));
    }
  });

  it('have no empty strings', () => {
    for (const [key, s] of [...Object.entries(en), ...Object.entries(de)]) expect(s.trim(), key).not.toBe('');
  });
});

describe('t', () => {
  it('fills placeholders', () => {
    expect(t('chip.qubits', { n: 6 }, 'en')).toBe('6 qubits');
    expect(t('chip.qubits', { n: 6 }, 'de')).toBe('6 Qubits');
  });

  it('falls back to English per key', () => {
    const tables = TABLES as unknown as Record<string, Record<string, string>>;
    const saved = tables.de['header.about'];
    delete tables.de['header.about'];
    try {
      expect(t('header.about', undefined, 'de')).toBe(en['header.about']);
      expect(t('header.edition', undefined, 'de')).toBe('Browser-Edition');
    } finally {
      tables.de['header.about'] = saved;
    }
  });
});

describe('language resolution', () => {
  it('lets the URL parameter win', () => {
    expect(resolveLang('?lang=de', 'en', 'en-US')).toBe('de');
    expect(resolveLang('?lang=en', 'de', 'de-DE')).toBe('en');
    expect(resolveLang('?track=gp&lang=DE', null, 'en')).toBe('de');
  });

  it('then the stored choice', () => {
    expect(resolveLang('', 'de', 'en-US')).toBe('de');
    expect(resolveLang('?lang=fr', 'en', 'de-AT')).toBe('en');
  });

  it('then the browser language', () => {
    expect(resolveLang('', null, 'de-CH')).toBe('de');
    expect(resolveLang('', 'xx', 'de')).toBe('de');
    expect(resolveLang('', null, 'fr-FR')).toBe('en');
    expect(resolveLang('', null, undefined)).toBe('en');
  });

  it('sets the page language and remembers a switch', () => {
    localStorage.clear();
    initLang();
    setLang('de');
    expect(getLang()).toBe('de');
    expect(document.documentElement.lang).toBe('de');
    expect(localStorage.getItem('racetraq-lang')).toBe('de');
    expect(initLang()).toBe('de');
    setLang('en');
    expect(document.documentElement.lang).toBe('en');
  });
});

describe('numbers', () => {
  it('formats lap times per language', () => {
    expect(fmtSeconds(12.2, 'en')).toBe('12.20 s');
    expect(fmtSeconds(12.2, 'de')).toBe('12,20\u00a0s');
  });

  it('formats values and counts', () => {
    expect(fmtNum(0.426, 2, { lang: 'de' })).toBe('0,43');
    expect(fmtNum(-0.5, 2, { lang: 'en', sign: true })).toBe('-0.50');
    expect(fmtNum(0.5, 2, { lang: 'de', sign: true })).toBe('+0,50');
    expect(fmtInt(12345, 'en')).toBe('12,345');
    expect(fmtInt(12345, 'de')).toBe('12.345');
  });
});

describe('the EN/DE switch', () => {
  it('is a labelled group of pressed/unpressed buttons that switches the language', async () => {
    const { act, createElement } = await import('react');
    const { createRoot } = await import('react-dom/client');
    const { LangSwitch } = await import('../src/components/LangSwitch');
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    setLang('en');
    const host = document.createElement('div');
    document.body.appendChild(host);
    const root = createRoot(host);
    await act(async () => root.render(createElement(LangSwitch)));
    const group = host.querySelector('[role="group"]')!;
    expect(group.getAttribute('aria-label')).toBe('Language');
    const [enBtn, deBtn] = [...host.querySelectorAll('button')];
    expect(enBtn.getAttribute('aria-pressed')).toBe('true');
    expect(deBtn.getAttribute('aria-pressed')).toBe('false');
    await act(async () => deBtn.click());
    expect(getLang()).toBe('de');
    expect(deBtn.getAttribute('aria-pressed')).toBe('true');
    expect(group.getAttribute('aria-label')).toBe('Sprache');
    await act(async () => root.unmount());
    setLang('en');
  });
});

// a variable path: Vite rewrites new URL('literal', import.meta.url) as an asset
const MANIFEST = '../public/data/manifest.json';

describe('car labels', () => {
  it('follow a language switch, evolution stages included', async () => {
    const { labelCars } = await import('../src/race');
    const manifest = JSON.parse(readFileSync(fileURLToPath(new URL(MANIFEST, import.meta.url)), 'utf8'));
    const car = (id: string) => ({ id, label: '' });
    const watch = {
      setup: { track: 'oval', mode: 'watch', driver: 'quantum_oval', rival: 'mlp', ghost: true },
      world: { cars: [car('quantum'), car('mlp')] },
    };
    const evolution = {
      setup: { ...watch.setup, mode: 'evolution' },
      world: { cars: [car('stage1'), car('stage4')] },
    };
    const labels = (race: typeof watch) => race.world.cars.map((c) => c.label);
    setLang('de');
    labelCars(manifest, watch as never);
    labelCars(manifest, evolution as never);
    expect(labels(watch)).toEqual(['Quanten-KI · 4 Qubits', 'Klassisches MLP · 76 Parameter']);
    expect(labels(evolution)).toEqual(['Ep. 150', 'beste (aus Lauf mit 800 Ep.)']);
    setLang('en');
    labelCars(manifest, watch as never);
    labelCars(manifest, evolution as never);
    expect(labels(watch)).toEqual(['Quantum · 4 qubits', 'Classical MLP · 76 params']);
    expect(labels(evolution)).toEqual(['ep 150', 'best (of 800 ep run)']);
  });
});
