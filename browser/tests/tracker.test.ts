// The family's standard Umami snippet in index.html: the before-send hook is
// defined before the tracker loads, and the tracker stays on racetraq.org.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const html = readFileSync(resolve(__dirname, '..', 'index.html'), 'utf8');

describe('umami tracker', () => {
  const tracker = html.indexOf('src="https://cloud.umami.is/script.js"');
  const hook = html.indexOf('window.fwqUmamiBeforeSend=');

  it('defines the bot filter and ?notrack before the tracker loads', () => {
    expect(hook).toBeGreaterThan(-1);
    expect(tracker).toBeGreaterThan(hook);
    expect(html).toContain("localStorage.setItem('umami.disabled','1')");
  });

  it('uses the hook and stays restricted to racetraq.org', () => {
    const tag = html.slice(html.lastIndexOf('<script', tracker), html.indexOf('</script>', tracker));
    expect(tag).toContain('data-before-send="fwqUmamiBeforeSend"');
    expect(tag).toContain('data-domains="racetraq.org"');
  });
});
