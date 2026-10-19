// Search-engine metadata in index.html: the JSON-LD parses, carries the
// concept DOI from CITATION.cff, and the hreflang links match the sitemap.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(__dirname, '..');
const html = readFileSync(resolve(root, 'index.html'), 'utf8');
const sitemap = readFileSync(resolve(root, 'public/sitemap.xml'), 'utf8');
const citation = readFileSync(resolve(root, '../CITATION.cff'), 'utf8');

function jsonLd(): { '@graph': Array<Record<string, unknown>> } {
  const match = html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/);
  expect(match).not.toBeNull();
  return JSON.parse(match![1]);
}

describe('structured data', () => {
  it('parses and names the app, the code, the studies and the report', () => {
    const types = jsonLd()['@graph'].map((node) => node['@type']);
    expect(types).toEqual(
      expect.arrayContaining(['SoftwareApplication', 'SoftwareSourceCode', 'Dataset', 'Report']),
    );
  });

  it('carries the concept DOI of CITATION.cff', () => {
    const doi = citation.match(/^doi:\s*(\S+)/m)![1];
    const code = jsonLd()['@graph'].find((node) => node['@type'] === 'SoftwareSourceCode')!;
    expect(code.identifier).toBe(`https://doi.org/${doi}`);
  });

  it('resolves every @id reference inside the graph', () => {
    const graph = jsonLd()['@graph'];
    const ids = new Set(graph.map((node) => node['@id']));
    const refs = JSON.stringify(graph).match(/\{"@id":"[^"]+"\}/g) ?? [];
    for (const ref of refs) expect(ids.has(JSON.parse(ref)['@id'])).toBe(true);
  });
});

describe('hreflang', () => {
  it('offers en, de and x-default in the page and in the sitemap', () => {
    for (const lang of ['en', 'de', 'x-default']) {
      const page = new RegExp(`<link rel="alternate" hreflang="${lang}" href="([^"]+)"`);
      const href = html.match(page)![1];
      expect(sitemap).toContain(`hreflang="${lang}" href="${href}"`);
    }
  });
});

describe('crawlable page', () => {
  const root = html.match(/<div id="root">([\s\S]*?)<\/div>\s*<script/)![1];

  it('prerenders an h1, an intro and links into #root', () => {
    expect(root.match(/<h1>[^<]+<\/h1>/g)).toHaveLength(1);
    expect(root).toMatch(/<p>[\s\S]{200,}<\/p>/);
    expect(root).toContain('href="https://github.com/JanLahmann/racetraQ"');
    expect(root).toContain('href="https://fun-with-quantum.org/"');
  });

  it('has a canonical URL and a descriptive title of at most 60 characters', () => {
    expect(html).toContain('<link rel="canonical" href="https://racetraq.org/" />');
    const title = html.match(/<title>([^<]+)<\/title>/)![1];
    expect(title.length).toBeGreaterThan(30);
    expect(title.length).toBeLessThanOrEqual(60);
  });

  it('lists each page once in the sitemap, with a lastmod', () => {
    const locs = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
    expect(locs).toEqual(['https://racetraq.org/', 'https://racetraq.org/?lang=de']);
    expect(sitemap.match(/<lastmod>\d{4}-\d{2}-\d{2}<\/lastmod>/g)).toHaveLength(locs.length);
  });
});
