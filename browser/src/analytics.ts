/**
 * Umami events, Fun with Quantum family taxonomy v2 (Fun-with-Quantum/family/EVENTS.md): every
 * family site reports to one shared Umami website, so each event is named `<Site>: <what happened>`
 * (lower case after the colon); `<Site>` is the member's manifest label, here its name "racetraQ".
 * Best-effort: no-ops when the tracker isn't loaded (dev, previews, self-hosted copies, blockers).
 *
 * Events sent by the browser edition:
 * - `racetraQ: mode change`    — mode (watch | race | evolution), track
 * - `racetraQ: track change`   — track, mode
 * - `racetraQ: driver change`  — driver (weights id), qubits
 * - `racetraQ: rival change`   — rival (none | mlp | pro)
 * - `racetraQ: race lap`       — track, lap_time (s, 1 decimal), clean (yes | no), opponent
 * - `racetraQ: composer open`  — qubits, track (decision circuit sent to IBM Quantum Composer)
 * - `racetraQ: about open`
 * - `racetraQ: outbound click` — host (a `data-umami-event` link)
 */

type Umami = { track: (name: string, data?: Record<string, string>) => void };

export const SITE_LABEL = 'racetraQ';

export function eventName(what: string): string {
  return `${SITE_LABEL}: ${what}`;
}

export type TraqEvent =
  | 'mode change'
  | 'track change'
  | 'driver change'
  | 'rival change'
  | 'race lap'
  | 'composer open'
  | 'about open';

export function track(what: TraqEvent, data: Record<string, string | number | undefined> = {}): void {
  try {
    const clean: Record<string, string> = {};
    for (const [k, v] of Object.entries(data)) if (v !== undefined) clean[k] = String(v);
    (window as unknown as { umami?: Umami }).umami?.track(eventName(what), clean);
  } catch {
    /* analytics is best-effort */
  }
}

/** Attributes that make Umami record a click on an external link (no JS needed). */
export function outbound(href: string): Record<string, string> {
  return {
    'data-umami-event': eventName('outbound click'),
    'data-umami-event-host': new URL(href).host,
  };
}
