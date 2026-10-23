// @vitest-environment jsdom
/** Accessibility checks from the family quality scan: names that contain the
 *  visible label, and text colours that meet WCAG AA on the dark surfaces. */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it } from 'vitest';
import { LangSwitch } from '../src/components/LangSwitch';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
const contrast = (a: string, b: string) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

describe('label in name', () => {
  it('each language button is named starting with its visible text', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(createElement(LangSwitch)));
    const buttons = [...host.querySelectorAll('button')];
    expect(buttons).toHaveLength(2);
    for (const b of buttons) expect(b.getAttribute('aria-label')!.startsWith(b.textContent!)).toBe(true);
    act(() => root.unmount());
  });
});

describe('colour tokens', () => {
  const css = readFileSync(resolve(__dirname, '../src/styles.css'), 'utf8');
  const token = (name: string) => css.match(new RegExp(`--${name}: (#[0-9a-f]{6})`, 'i'))![1];

  it('purple text meets 4.5:1 on the page and the panels', () => {
    for (const surface of ['bg', 'bg-panel', 'bg-inset']) {
      expect(contrast(token('quantum-text'), token(surface))).toBeGreaterThanOrEqual(4.5);
    }
  });

  it('white text on the purple fill meets 4.5:1', () => {
    expect(contrast('#ffffff', token('quantum-fill'))).toBeGreaterThanOrEqual(4.5);
  });
});
