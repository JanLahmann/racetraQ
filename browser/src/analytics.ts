/**
 * Umami events, Fun with Quantum family taxonomy v2 (Fun-with-Quantum/family/EVENTS.md): every
 * family site reports to one shared Umami website, so each event is named `<Site>: <what happened>`
 * (lower case after the colon); `<Site>` is the member's manifest label, here its name "traQmania".
 * Best-effort: no-ops when the tracker isn't loaded (dev, previews, self-hosted copies, blockers).
 *
 * Events sent by the browser edition:
 * - `traQmania: mode change`    — mode (watch | race | evolution), track
 * - `traQmania: track change`   — track, mode
 * - `traQmania: driver change`  — driver (weights id), qubits
 * - `traQmania: rival change`   — rival (none | mlp | pro)
 * - `traQmania: race lap`       — track, lap_time (s, 1 decimal), clean (yes | no), opponent
 * - `traQmania: composer open`  — qubits, track (decision circuit sent to IBM Quantum Composer)
 * - `traQmania: about open`
 * - `traQmania: outbound click` — host (a `data-umami-event` link)
 */

type Umami = { track: (name: string, data?: Record<string, string>) => void };

export const SITE_LABEL = 'traQmania';

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
