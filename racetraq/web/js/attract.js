// Attract-mode caption rotation + client-side idle timer that returns the
// exhibit to attract mode after `ui.attract_idle_seconds` without interaction.
// Only the browser holding the wheel runs the timer (a watching phone never
// resets the booth; the server covers a wheel nobody holds). Running demos
// hold it off: training, a hardware job, the evolution show ("hold": never)
// and a visitor reading their studio result ("linger": longer).

import { t, featureLabel, listJoin, numWord } from "./i18n.js";

// Plain-language captions for passers-by (the Explain panel and the docs
// carry the physics), from the string tables ("attract.*"). Four of them
// follow the welcome's circuit_spec / obs_labels (setCircuitSpec): the qubit
// count, the number of learned parameters, who votes and the sensor list.
// Without a spec the authored text describes the default 4-qubit circuit.

/** "Three distance sensors and speed" / "Five distance sensors, speed,
 *  curvature ahead and corner speed": the inputs from the server's obs_labels
 *  (rays are labelled "ray …°"); `n - 1` rays and speed when no labels came.
 *  Pure function (given the language). */
export function sensorPhrase(n = 4, labels = null) {
  let rays = n - 1;
  let others = ["speed"];
  if (Array.isArray(labels) && labels.length) {
    rays = labels.filter((l) => /^ray\b/.test(String(l))).length;
    others = labels.filter((l) => !/^ray\b/.test(String(l))).map(String);
  }
  const rayText = rays > 0
    ? t(rays === 1 ? "attract.rays_one" : "attract.rays", { count: numWord(rays, true) })
    : "";
  const parts = others.map(featureLabel);
  if (rayText) parts.unshift(rayText);
  if (parts.length === 0) return t("attract.no_sensors");
  return listJoin(parts);
}

/** "52 of its 56 trainable parameters can steer the car" from the spec's
 *  n_params.total and dead_params; just the total without light-cone data;
 *  "" without a usable total. Pure function (the expert wording, kept for
 *  the Explain panel's readers). */
export function parameterPhrase(spec) {
  const total = spec && spec.n_params && spec.n_params.total;
  if (!Number.isInteger(total) || total <= 0) return "";
  const dead = spec.dead_params;
  if (!Number.isInteger(dead) || dead <= 0) return t("attract.param_total", { total });
  return t("attract.param_live", { live: total - dead, total });
}

/** The rotating captions in the active language for a welcome's `spec` and
 *  `obsLabels`; the stock 4-qubit copy without a spec. Pure function. */
export function attractCaptions(spec = null, obsLabels = null) {
  const n = (spec && spec.n_qubits) || 4;
  const total = spec && spec.n_params && spec.n_params.total;
  return [
    t("attract.circuit", { n }),
    spec && Number.isInteger(total) && total > 0
      ? t("attract.params", { total })
      : t("attract.params_stock"),
    t("attract.measure"),
    n === 4 ? t("attract.votes") : t("attract.votes_wide"),
    t("attract.reupload"),
    t("attract.colors"),
    t("attract.learned"),
    spec ? t("attract.senses", { sensors: sensorPhrase(n, obsLabels) }) : t("attract.senses_stock"),
    t("attract.race"),
    t("attract.simulator"),
    t("attract.honest"),
  ];
}

const ROTATE_MS = 6000;

export class AttractManager {
  constructor({ captionEl, onIdle }) {
    this.captionEl = captionEl;
    this.onIdle = onIdle;
    this.idleSeconds = 45;
    this.mode = "attract";
    this.captionIdx = 0;
    this.rotateId = null;
    this.idleId = null;
    this.holds = new Map(); // reason -> "hold" | "linger"
    this.driving = false;
    this.spec = null; // welcome circuit_spec / obs_labels the captions describe
    this.obsLabels = null;

    const bump = () => this.notifyActivity();
    for (const ev of ["pointerdown", "pointermove", "keydown", "wheel", "touchstart"]) {
      window.addEventListener(ev, bump, { passive: true });
    }
  }

  /** Hold off the idle return for `reason` (a studio run, training, a
   *  hardware job …): level "hold" never idles — watching the run IS the
   *  interaction — "linger" waits max(3 × idle, 90 s); null releases it. */
  setHold(reason, level) {
    if (level) this.holds.set(reason, level);
    else this.holds.delete(reason);
    this._armIdle();
  }

  /** Only the browser that holds the wheel runs the idle timer. */
  setDriving(driving) {
    this.driving = Boolean(driving);
    this._armIdle();
  }

  /** Seconds until the idle return under the current holds (0: never). */
  effectiveIdleSeconds() {
    const levels = new Set(this.holds.values());
    if (!this.idleSeconds || levels.has("hold")) return 0;
    return levels.has("linger") ? Math.max(3 * this.idleSeconds, 90) : this.idleSeconds;
  }

  setIdleSeconds(s) {
    this.idleSeconds = s;
    this._armIdle();
  }

  /** Describe the welcome `circuit_spec` and `obs_labels` in the captions
   *  (every welcome: qubit and driver switches change both). Without a spec
   *  the stock copy is back. */
  setCircuitSpec(spec, obsLabels = null) {
    this.spec = spec || null;
    this.obsLabels = spec ? obsLabels : null;
  }

  /** The captions as they read now (language and circuit). */
  captions() {
    return attractCaptions(this.spec, this.obsLabels);
  }

  /** Re-show the current caption (a language switch). */
  refresh() {
    if (this.mode === "attract") this._showCaption();
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === "attract") this._startCaptions();
    else this._stopCaptions();
    this._armIdle();
  }

  notifyActivity() {
    this._armIdle();
  }

  _armIdle() {
    if (this.idleId) clearTimeout(this.idleId);
    this.idleId = null;
    const seconds = this.effectiveIdleSeconds();
    if (this.mode === "attract" || !seconds || !this.driving) return;
    this.idleId = setTimeout(() => {
      if (this.mode !== "attract" && this.onIdle) this.onIdle();
    }, seconds * 1000);
  }

  _startCaptions() {
    this.captionEl.hidden = false;
    this._showCaption();
    if (!this.rotateId) {
      this.rotateId = setInterval(() => {
        this.captionIdx = (this.captionIdx + 1) % this.captions().length;
        this._showCaption();
      }, ROTATE_MS);
    }
  }

  _stopCaptions() {
    this.captionEl.hidden = true;
    if (this.rotateId) {
      clearInterval(this.rotateId);
      this.rotateId = null;
    }
  }

  _showCaption() {
    const el = this.captionEl;
    el.classList.remove("caption-in");
    el.textContent = this.captions()[this.captionIdx];
    // retrigger the fade-in animation
    void el.offsetWidth;
    el.classList.add("caption-in");
  }
}
