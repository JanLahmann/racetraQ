// Attract-mode caption rotation + client-side idle timer that returns the
// exhibit to attract mode after `ui.attract_idle_seconds` without interaction.
// Only the browser holding the wheel runs the timer (a watching phone never
// resets the booth; the server covers a wheel nobody holds). Running demos
// hold it off: training, a hardware job, the evolution show ("hold": never)
// and a visitor reading their studio result ("linger": longer).

// Plain-language captions for passers-by (the Explain panel and the docs
// carry the physics). Rewritten by setCircuitSpec() from the welcome's
// circuit_spec / obs_labels: indexes 0, 1, 3 and 7 carry the qubit count,
// the number of learned parameters and the sensor list. The authored text is
// the default 4-qubit circuit.
const CAPTIONS = [
  "A 4-qubit quantum circuit is driving this car.",
  "It drives with 56 learned numbers — about as many as a tiny classical network.",
  "Ten times a second, the car measures the qubits and picks a move.",
  "Each qubit votes for one move: steer right, go straight, steer left or brake.",
  "The car's sensor readings are fed into the circuit again and again, layer by layer.",
  "Purple is the quantum driver, blue a small classical network. Same game, same rules.",
  "Nobody programmed the driving: both learned by trial and error, lap after lap.",
  "Three distance sensors and its speed — that's all the car can sense.",
  "Press an arrow key or a controller button to race it yourself.",
  "This runs on a quantum simulator — the same circuit can run on a real quantum computer.",
  "Honest result: the classical network learns faster and more reliably here.",
];

const RAY_WORDS = { 3: "Three", 5: "Five", 7: "Seven", 9: "Nine" };

/** "Three distance sensors and speed" / "five distance sensors, speed,
 *  curvature ahead and corner speed": the inputs from the server's obs_labels
 *  (rays are labelled "ray …°"); `n - 1` rays and speed when no labels came.
 *  Pure function. */
export function sensorPhrase(n = 4, labels = null) {
  let rays = n - 1;
  let others = ["speed"];
  if (Array.isArray(labels) && labels.length) {
    rays = labels.filter((l) => /^ray\b/.test(String(l))).length;
    others = labels.filter((l) => !/^ray\b/.test(String(l))).map(String);
  }
  const rayText = rays > 0
    ? `${RAY_WORDS[rays] || rays} distance sensor${rays === 1 ? "" : "s"}` : "";
  const parts = rayText ? [rayText, ...others] : [...others];
  if (parts.length === 0) return "No sensors";
  if (parts.length === 1) return parts[0];
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** "52 of its 56 trainable parameters can steer the car" from the spec's
 *  n_params.total and dead_params; just the total without light-cone data;
 *  "" without a usable total. Pure function (the expert wording, kept for
 *  the Explain panel's readers). */
export function parameterPhrase(spec) {
  const total = spec && spec.n_params && spec.n_params.total;
  if (!Number.isInteger(total) || total <= 0) return "";
  const dead = spec.dead_params;
  if (!Number.isInteger(dead) || dead <= 0) return `${total} trainable parameters steer the car`;
  return `${total - dead} of its ${total} trainable parameters can steer the car`;
}

// Authored copy of the captions setCircuitSpec() rewrites, so a welcome that
// carries no spec restores the stock text.
const STOCK_CAPTIONS = [...CAPTIONS];

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

  /** Rewrite the circuit-size captions from the welcome `circuit_spec` and
   *  `obs_labels` (every welcome: qubit and driver switches change both).
   *  Without a spec the stock copy is restored. */
  setCircuitSpec(spec, obsLabels = null) {
    if (!spec) {
      for (const i of [0, 1, 3, 7]) CAPTIONS[i] = STOCK_CAPTIONS[i];
      return;
    }
    const n = spec.n_qubits || 4;
    CAPTIONS[0] = `A ${n}-qubit quantum circuit is driving this car.`;
    const total = spec.n_params && spec.n_params.total;
    CAPTIONS[1] = Number.isInteger(total) && total > 0
      ? `It drives with ${total} learned numbers — about as many as a small classical network.`
      : STOCK_CAPTIONS[1];
    CAPTIONS[3] =
      n === 4
        ? STOCK_CAPTIONS[3]
        : "The first four qubits vote for the moves; the others carry more sensor readings.";
    CAPTIONS[7] = `${sensorPhrase(n, obsLabels)} — that's all the car can sense.`;
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
        this.captionIdx = (this.captionIdx + 1) % CAPTIONS.length;
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
    el.textContent = CAPTIONS[this.captionIdx];
    // retrigger the fade-in animation
    void el.offsetWidth;
    el.classList.add("caption-in");
  }
}
