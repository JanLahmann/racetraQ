// Attract-mode caption rotation + client-side idle timer that returns the
// exhibit to attract mode after `ui.attract_idle_seconds` without interaction.

// Rewritten by setCircuitSpec() from the welcome's circuit_spec / obs_labels:
// indexes 0, 1, 3 and 7 carry the qubit count, the parameter count (live
// against total — a share of the circuit's angles is structurally dead) and
// the sensor list. The authored text is the default 4-qubit circuit.
const CAPTIONS = [
  "A 4-qubit quantum circuit is driving this car.",
  "52 of its 56 trainable parameters can steer the car — the rival classical MLP is comparably small.",
  "Watch the panel on the right: live qubit measurements steer the car.",
  "Each action is one qubit: its ⟨Z⟩ expectation value becomes a Q-value.",
  "The circuit re-reads the car's sensors in every layer — data re-uploading.",
  "Purple = quantum agent, green = classical MLP. Same game, same rewards.",
  "Both agents learned by trial and error with double DQN.",
  "Three lidar rays and speed — that's all the car can sense.",
  "Press Race to grab the wheel yourself (arrow keys or WASD).",
  "This runs a quantum simulator — the same circuit can run on real hardware.",
  "The classical MLP learns faster and more reliably here — no quantum advantage is claimed.",
];

// Authored copy of the captions setCircuitSpec() rewrites, so a welcome that
// carries no spec restores the stock text.
const STOCK_CAPTIONS = [...CAPTIONS];

const RAY_WORDS = { 3: "Three", 5: "Five", 7: "Seven", 9: "Nine" };

/** "Three lidar rays and speed" / "five lidar rays, speed, curvature ahead and
 *  corner speed": the inputs from the server's obs_labels (rays are labelled
 *  "ray …°"); `n - 1` rays and speed when no labels came. Pure function. */
export function sensorPhrase(n = 4, labels = null) {
  let rays = n - 1;
  let others = ["speed"];
  if (Array.isArray(labels) && labels.length) {
    rays = labels.filter((l) => /^ray\b/.test(String(l))).length;
    others = labels.filter((l) => !/^ray\b/.test(String(l))).map(String);
  }
  const rayText = rays > 0 ? `${RAY_WORDS[rays] || rays} lidar ray${rays === 1 ? "" : "s"}` : "";
  const parts = rayText ? [rayText, ...others] : [...others];
  if (parts.length === 0) return "No sensors";
  if (parts.length === 1) return parts[0];
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** "52 of its 56 trainable parameters can steer the car" from the spec's
 *  n_params.total and dead_params; just the total without light-cone data;
 *  "" without a usable total. Pure function. */
export function parameterPhrase(spec) {
  const total = spec && spec.n_params && spec.n_params.total;
  if (!Number.isInteger(total) || total <= 0) return "";
  const dead = spec.dead_params;
  if (!Number.isInteger(dead) || dead <= 0) return `${total} trainable parameters steer the car`;
  return `${total - dead} of its ${total} trainable parameters can steer the car`;
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

    const bump = () => this.notifyActivity();
    for (const ev of ["pointerdown", "pointermove", "keydown", "wheel", "touchstart"]) {
      window.addEventListener(ev, bump, { passive: true });
    }
  }

  /** While busy (a studio training run) the exhibit never idles back to
   *  attract mode — watching training IS the interaction. */
  setBusy(busy) {
    this.busy = Boolean(busy);
    this._armIdle();
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
    const params = parameterPhrase(spec);
    CAPTIONS[1] = params
      ? `${params} — the rival classical MLP is comparably small.`
      : STOCK_CAPTIONS[1];
    CAPTIONS[3] =
      n === 4
        ? STOCK_CAPTIONS[3]
        : "The first four qubits are the actions: each ⟨Z⟩ expectation value becomes a Q-value.";
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
    if (this.mode === "attract" || !this.idleSeconds || this.busy) return;
    this.idleId = setTimeout(() => {
      if (this.mode !== "attract" && this.onIdle) this.onIdle();
    }, this.idleSeconds * 1000);
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
