/**
 * The browser edition's EN/DE interface: flat string tables (en.ts, de.ts),
 * t(key, vars) with a per-key fallback to English, a tiny store that React
 * components subscribe to with useLang(), and number formatting in the
 * active language.
 *
 * Language order: a `?lang=de` / `?lang=en` URL parameter, then the choice
 * stored by the EN/DE switch, then the browser language ("de…" -> German),
 * else English.
 */
import { createElement, Fragment, useSyncExternalStore, type ReactNode } from 'react';
import { de } from './de';
import { en, type Key } from './en';

export type { Key };
export type Lang = 'en' | 'de';
export const LANGS: readonly Lang[] = ['en', 'de'];
export const TABLES: Record<Lang, Partial<Record<Key, string>>> = { en, de };

const STORAGE_KEY = 'racetraq-lang';

const isLang = (v: string | null | undefined): v is Lang => v === 'en' || v === 'de';

/** The language for a URL query string, a stored choice and the browser language. */
export function resolveLang(search: string, stored: string | null, browserLang: string | undefined): Lang {
  let param: string | null = null;
  try {
    param = new URLSearchParams(search).get('lang');
  } catch {
    // malformed query: ignore it
  }
  const fromParam = param?.toLowerCase();
  if (isLang(fromParam)) return fromParam;
  if (isLang(stored)) return stored;
  return browserLang?.toLowerCase().startsWith('de') ? 'de' : 'en';
}

function readStored(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // storage blocked
  }
}

let current: Lang = 'en';
const listeners = new Set<() => void>();

export const getLang = (): Lang => current;

function apply(): void {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = current;
  document.title = t('app.title');
  // ?lang=de is the German page in its own right (hreflang); every other URL is the English one
  const canonical = document.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (canonical) canonical.href = current === 'de' && /[?&]lang=de\b/i.test(location.search) ? 'https://racetraq.org/?lang=de' : 'https://racetraq.org/';
}

/** Pick the language at startup (before the first render). */
export function initLang(): Lang {
  const search = typeof location === 'undefined' ? '' : location.search;
  const browser = typeof navigator === 'undefined' ? undefined : navigator.language;
  current = resolveLang(search, readStored(), browser);
  apply();
  return current;
}

/** Switch the language (the EN/DE toggle): remembered in this browser, and a
 *  `lang` parameter already in the URL follows, so a reload keeps the choice. */
export function setLang(lang: Lang): void {
  if (lang === current) return;
  current = lang;
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // storage blocked: the choice lasts for this page only
  }
  try {
    const url = new URL(location.href);
    if (url.searchParams.has('lang')) {
      url.searchParams.set('lang', lang);
      history.replaceState(history.state, '', url);
    }
  } catch {
    // no history API: nothing to update
  }
  apply();
  for (const fn of listeners) fn();
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** The active language; the calling component re-renders when it changes. */
export function useLang(): Lang {
  return useSyncExternalStore(subscribe, getLang, getLang);
}

export type Vars = Record<string, string | number>;

function lookup(key: Key, lang: Lang): string {
  return TABLES[lang][key] ?? en[key] ?? key;
}

/** The string for a key in the active language, `{name}` placeholders filled. */
export function t(key: Key, vars?: Vars, lang: Lang = current): string {
  const s = lookup(key, lang);
  return vars ? s.replace(/\{(\w+)\}/g, (m, name: string) => (name in vars ? String(vars[name]) : m)) : s;
}

/** Whether a (dynamic) key exists. */
export const hasKey = (key: string): key is Key => key in en;

/** A string with React nodes (links, emphasis) in its placeholders. */
export function rich(key: Key, nodes: Record<string, ReactNode>): ReactNode {
  const parts = lookup(key, current).split(/\{(\w+)\}/);
  // split with a capture group: odd indices are placeholder names
  const children = parts.map((part, i) => (i % 2 ? (part in nodes ? nodes[part] : `{${part}}`) : part));
  return createElement(Fragment, null, ...children);
}

/** The placeholder names of a string (tests: en and de must agree). */
export function placeholders(s: string): string[] {
  return [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
}

// ------------------------------------------------------------- numbers

const formats = new Map<string, Intl.NumberFormat>();

function numberFormat(lang: Lang, digits: number | null, sign: boolean): Intl.NumberFormat {
  const id = `${lang}|${digits}|${sign}`;
  let f = formats.get(id);
  if (!f) {
    f = new Intl.NumberFormat(
      lang,
      digits === null
        ? { maximumFractionDigits: 0 }
        : {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
            useGrouping: false,
            signDisplay: sign ? 'always' : 'auto',
          },
    );
    formats.set(id, f);
  }
  return f;
}

/** A value with fixed decimals: "0.42" / "0,42" (sign: "+0.42"). */
export function fmtNum(v: number, digits = 2, opts: { sign?: boolean; lang?: Lang } = {}): string {
  return numberFormat(opts.lang ?? current, digits, !!opts.sign).format(v);
}

/** A whole number with grouping: "1,234" / "1.234". */
export function fmtInt(v: number, lang: Lang = current): string {
  return numberFormat(lang, null, false).format(v);
}

/** Seconds with two decimals: "12.20 s" / "12,20 s" (no-break space in German). */
export function fmtSeconds(v: number, lang: Lang = current): string {
  return `${fmtNum(v, 2, { lang })}${lang === 'de' ? '\u00a0' : ' '}s`;
}
