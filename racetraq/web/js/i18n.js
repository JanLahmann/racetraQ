// Interface language: English (default) and German. The strings live in flat
// tables, web/i18n/<lang>.json, with `{name}` placeholders; t() looks a key up
// in the active language, falls back to English per missing key, and fills
// the placeholders. applyDom() fills the page's data-i18n* attributes.
//
// The booth's default comes from `[ui] language` (welcome.ui.language); a
// visitor's own choice (the EN/DE toggle, kept in localStorage) wins.
//
// Runs in the browser (loadI18n() fetches both tables before the first
// render) and under node for the tests: there the tables are read from disk at
// import, and nothing touches document / window / localStorage.

export const LANGS = ["en", "de"];
const STORE_KEY = "racetraq-lang";

const tables = { en: {}, de: {} };
let current = "en";
const warned = new Set();
const listeners = new Set();

const IS_NODE = typeof process !== "undefined" && Boolean(process.versions && process.versions.node);

if (IS_NODE) {
  // tests: no server to fetch from — read the tables next to this module
  const { readFileSync } = await import("node:fs");
  for (const l of LANGS) {
    try {
      tables[l] = JSON.parse(readFileSync(new URL(`../i18n/${l}.json`, import.meta.url), "utf8"));
    } catch {
      // a missing table: t() returns the keys
    }
  }
}

/** Fetch both string tables (relative to this module, i.e. web/i18n/).
 *  Call once at boot, before the first render. */
export async function loadI18n() {
  await Promise.all(
    LANGS.map(async (l) => {
      try {
        const res = await fetch(new URL(`../i18n/${l}.json`, import.meta.url));
        if (res.ok) tables[l] = await res.json();
      } catch {
        // offline / blocked: English keys show, the page still works
      }
    }),
  );
}

/** The active language ("en" | "de"). */
export function lang() {
  return current;
}

/** Whether `key` has a string (in the active language or English). */
export function has(key) {
  return key in tables[current] || key in tables.en;
}

function warnMissing(key) {
  const id = `${current}:${key}`;
  if (warned.has(id)) return;
  warned.add(id);
  if (!IS_NODE && typeof console !== "undefined") console.warn(`racetraQ i18n: missing "${key}" (${current})`);
}

/** The string for `key` in the active language (English when missing), with
 *  `{name}` placeholders filled from `vars`; the key itself if no table has it. */
export function t(key, vars) {
  let s = tables[current][key];
  if (s === undefined) {
    warnMissing(key);
    s = tables.en[key];
  }
  if (s === undefined) return key;
  if (!vars) return s;
  return s.replace(/\{(\w+)\}/g, (m, name) => (name in vars ? String(vars[name]) : m));
}

/** The visitor's stored choice, or null. */
export function storedLang() {
  try {
    const v = typeof localStorage !== "undefined" ? localStorage.getItem(STORE_KEY) : null;
    return LANGS.includes(v) ? v : null;
  } catch {
    return null; // storage blocked
  }
}

/** Forget the stored choice (a kiosk's next visitor gets the booth default). */
export function clearStoredLang() {
  try {
    if (typeof localStorage !== "undefined") localStorage.removeItem(STORE_KEY);
  } catch {
    // storage blocked: nothing was stored
  }
}

/** Switch the language; `persist` remembers it as the visitor's choice.
 *  Listeners re-render everything built from state. */
export function setLang(next, { persist = true } = {}) {
  if (!LANGS.includes(next)) return;
  if (persist) {
    try {
      if (typeof localStorage !== "undefined") localStorage.setItem(STORE_KEY, next);
    } catch {
      // storage blocked: the choice lasts until the page reloads
    }
  }
  if (next === current) return;
  current = next;
  if (typeof document !== "undefined" && document.documentElement) {
    document.documentElement.lang = next;
  }
  for (const fn of listeners) fn(next);
}

/** Call `fn(lang)` after every language switch. */
export function onLangChange(fn) {
  listeners.add(fn);
}

const DOM_ATTRS = [
  ["data-i18n", (el, s) => { el.textContent = s; }],
  ["data-i18n-html", (el, s) => { el.innerHTML = s; }], // authored markup only
  ["data-i18n-tip", (el, s) => { el.dataset.tip = s; }],
  ["data-i18n-title", (el, s) => { el.title = s; }],
  ["data-i18n-placeholder", (el, s) => { el.placeholder = s; }],
  ["data-i18n-aria", (el, s) => { el.setAttribute("aria-label", s); }],
];

/** Fill every data-i18n* element under `root` in the active language. */
export function applyDom(root = typeof document !== "undefined" ? document : null) {
  if (!root || !root.querySelectorAll) return;
  for (const [attr, apply] of DOM_ATTRS) {
    for (const el of root.querySelectorAll(`[${attr}]`)) apply(el, t(el.getAttribute(attr)));
  }
}

// -- server-sent names ---------------------------------------------------------
// Feature, action and number names arrive in English from the server; these
// translate the known ones and pass anything else through unchanged (so the
// English text is the server's exactly).

const FEATURE_KEYS = {
  speed: "feature.speed",
  "curvature ahead": "feature.curvature_ahead",
  "lateral offset": "feature.lateral_offset",
  "heading error": "feature.heading_error",
  "corner speed": "feature.corner_speed",
};

/** "ray -60°" / "curvature ahead 30m" (env.feature_names) in the active language. */
export function featureLabel(name) {
  const s = String(name);
  const ray = /^ray (\S+°)$/.exec(s);
  if (ray) return t("feature.ray", { deg: ray[1] });
  const m = /^(.+?)( \d+(?:\.\d+)?m)?$/.exec(s);
  const key = m && FEATURE_KEYS[m[1]];
  return key ? t(key) + (m[2] || "") : s;
}

const ACTION_KEYS = {
  Right: "action.right",
  Straight: "action.straight",
  Left: "action.left",
  Brake: "action.brake",
  "Brake right": "action.brake_right",
  "Brake left": "action.brake_left",
  "Half right": "action.half_right",
  "Half left": "action.half_left",
};

/** An action label (circuit_spec.action_labels) in the active language. */
export function actionLabel(name) {
  const s = String(name);
  return ACTION_KEYS[s] ? t(ACTION_KEYS[s]) : s;
}

/** "three" for 3 (the authored number words; digits beyond them);
 *  `cap` capitalizes the first letter for a sentence start. */
export function numWord(n, cap = false) {
  const key = `num.${n}`;
  const word = Number.isInteger(n) && key in tables.en ? t(key) : String(n);
  return cap ? word.charAt(0).toUpperCase() + word.slice(1) : word;
}

/** "a, b and c" in the active language. */
export function listJoin(parts) {
  if (parts.length <= 1) return parts.join("");
  return `${parts.slice(0, -1).join(", ")} ${t("list.and")} ${parts[parts.length - 1]}`;
}
