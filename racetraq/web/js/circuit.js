// Static SVG rendering of the circuit_spec from the welcome message, plus the
// light-cone views built from the same spec: dimmed dead gates in the diagram
// and the "who sees what" matrix (actions x observation features).
// spec: {n_qubits, n_layers, n_actions, action_labels,
//        gates:[{type:'ry_enc'|'ry'|'rz'|'cz', qubit|q0/q1, layer, live}],
//        visibility: A x n of 0/1, dead_params, dead_gates, min_layers_full_visibility,
//        pruned_on_hardware}
// Everything light-cone related is optional: a spec without it (older server,
// or a shape the analysis rejects) renders the plain diagram and no matrix.

const WIRE_GAP = 44;
const COL_W = 34;
const LEFT_PAD = 46;
const TOP_PAD = 26;
const BOX_W = 26;
const BOX_H = 22;
const DEAD_OPACITY = 0.3;

const COLORS = {
  wire: "#3a4152",
  box: "#232836",
  boxStroke: "#4a5268",
  enc: "#3b2d73",
  encStroke: "#7a5cff",
  text: "#dfe3ea",
  encText: "#c9baff",
  cz: "#8fa0c9",
  meas: "#1e3a2f",
  measStroke: "#2fbf71",
  measText: "#9fe8c4",
  mutedText: "#8a91a0",
};

const DEAD_TIP = "Outside every readout's light cone — cannot influence any action";
const GAUGE_ONLY_TIP =
  "Not an action readout: this qubit's ⟨Z⟩ only feeds its gauge above.";

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Greedy column packing: each gate goes in the first free column for the
 *  wires it spans (cz spans the full min..max qubit range). */
function layoutGates(spec) {
  const nextFree = new Array(spec.n_qubits).fill(0);
  const placed = [];
  for (const g of spec.gates) {
    let qs;
    if (g.type === "cz") {
      const lo = Math.min(g.q0, g.q1);
      const hi = Math.max(g.q0, g.q1);
      qs = [];
      for (let q = lo; q <= hi; q++) qs.push(q);
    } else {
      qs = [g.qubit];
    }
    const col = Math.max(...qs.map((q) => nextFree[q]));
    for (const q of qs) nextFree[q] = col + 1;
    placed.push({ gate: g, col, qs });
  }
  return { placed, nCols: Math.max(...nextFree) };
}

function wireY(q) {
  return TOP_PAD + q * WIRE_GAP;
}

function colX(c) {
  return LEFT_PAD + c * COL_W + COL_W / 2;
}

function gateBox(x, y, fill, stroke, label, sub, subColor) {
  const parts = [
    `<rect x="${x - BOX_W / 2}" y="${y - BOX_H / 2}" width="${BOX_W}" height="${BOX_H}"` +
      ` rx="4" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`,
    `<text x="${x}" y="${y + (sub ? -1 : 4)}" text-anchor="middle" font-size="9"` +
      ` fill="${subColor || COLORS.text}" font-weight="600">${esc(label)}</text>`,
  ];
  if (sub) {
    parts.push(
      `<text x="${x}" y="${y + 8.5}" text-anchor="middle" font-size="7"` +
        ` fill="${subColor || COLORS.text}">${esc(sub)}</text>`,
    );
  }
  return parts.join("");
}

/** Readout width: the first n_actions qubits carry the actions (all of them
 *  when the spec does not say). */
function readoutCount(spec) {
  return Number.isInteger(spec.n_actions) ? spec.n_actions : spec.n_qubits;
}

/** Whether the hardware path drops the dead gates ([hardware] prune_light_cone,
 *  on unless the spec says otherwise). */
function prunedOnHardware(spec) {
  return spec.pruned_on_hardware !== false;
}

/** Build the SVG markup for a circuit_spec. Pure function, no DOM needed.
 *  Gates flagged `live: false` are drawn dimmed, with a data-tip tooltip. */
export function circuitSvg(spec) {
  const n = spec.n_qubits;
  const nReadout = readoutCount(spec);
  const deadTip = esc(`${DEAD_TIP}${prunedOnHardware(spec) ? "; skipped on hardware" : ""}.`);
  const { placed, nCols } = layoutGates(spec);
  const width = LEFT_PAD + nCols * COL_W + 58;
  const height = TOP_PAD + (n - 1) * WIRE_GAP + 30;
  const out = [];

  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"` +
      ` viewBox="0 0 ${width} ${height}" role="img" aria-label="Quantum circuit diagram">`,
  );

  // wires + qubit labels + terminal <Z> boxes (green = an action reads it)
  for (let q = 0; q < n; q++) {
    const y = wireY(q);
    out.push(
      `<text x="8" y="${y + 3}" font-size="10" fill="${COLORS.text}">q${q}</text>`,
      `<line x1="${LEFT_PAD - 8}" y1="${y}" x2="${width - 44}" y2="${y}"` +
        ` stroke="${COLORS.wire}" stroke-width="1.2"/>`,
    );
    if (q < nReadout) {
      out.push(gateBox(width - 26, y, COLORS.meas, COLORS.measStroke, `⟨Z${q}⟩`, null,
        COLORS.measText));
    } else {
      out.push(
        `<g class="meas-gauge-only" data-tip="${esc(GAUGE_ONLY_TIP)}">` +
          gateBox(width - 26, y, COLORS.box, COLORS.boxStroke, `⟨Z${q}⟩`, null,
            COLORS.mutedText) +
          "</g>",
      );
    }
  }

  // layer separators
  const colsPerLayer = nCols / spec.n_layers;
  for (let l = 1; l < spec.n_layers; l++) {
    const x = LEFT_PAD + l * colsPerLayer * COL_W - 2;
    out.push(
      `<line x1="${x}" y1="${TOP_PAD - 16}" x2="${x}" y2="${height - 12}"` +
        ` stroke="#2a3040" stroke-width="1" stroke-dasharray="3 4"/>`,
    );
  }

  for (const { gate, col } of placed) {
    const x = colX(col);
    let markup;
    if (gate.type === "cz") {
      const y0 = wireY(gate.q0);
      const y1 = wireY(gate.q1);
      const top = Math.min(y0, y1);
      const bottom = Math.max(y0, y1);
      markup =
        `<line x1="${x}" y1="${top}" x2="${x}" y2="${bottom}"` +
        ` stroke="${COLORS.cz}" stroke-width="1.5"/>` +
        `<circle cx="${x}" cy="${y0}" r="3.4" fill="${COLORS.cz}"/>` +
        `<circle cx="${x}" cy="${y1}" r="3.4" fill="${COLORS.cz}"/>`;
      if (gate.live === false) {
        // invisible, wider hover target: the 1.5px line is hard to hit
        markup +=
          `<line x1="${x}" y1="${top}" x2="${x}" y2="${bottom}"` +
          ` stroke="#000" stroke-opacity="0" stroke-width="12"/>`;
      }
    } else if (gate.type === "ry_enc") {
      markup = gateBox(x, wireY(gate.qubit), COLORS.enc, COLORS.encStroke, "RY", "λx",
        COLORS.encText);
    } else {
      markup = gateBox(x, wireY(gate.qubit), COLORS.box, COLORS.boxStroke,
        gate.type.toUpperCase(), "θ");
    }
    out.push(
      gate.live === false
        ? `<g class="gate-dead" opacity="${DEAD_OPACITY}" data-tip="${deadTip}">` +
            `${markup}</g>`
        : markup,
    );
  }

  out.push("</svg>");
  return out.join("");
}

/** "56 trainable parameters, 4 structurally dead" — live vs total, from the
 *  spec; just the total when the spec carries no light-cone analysis. */
export function parameterCaption(spec) {
  const total = spec.n_params && spec.n_params.total;
  if (!Number.isInteger(total) || total <= 0) return ""; // numbers only: this lands in innerHTML
  const dead = spec.dead_params;
  if (!Number.isInteger(dead)) return `${total} trainable parameters`;
  return `${total} trainable parameters, ${dead > 0 ? dead : "none"} structurally dead`;
}

/** Render circuit + legend into the given containers (once, static). */
export function renderCircuit(spec, diagramEl, legendEl) {
  diagramEl.innerHTML = circuitSvg(spec);
  if (!legendEl) return;
  const caption = parameterCaption(spec);
  const hasDead = Boolean(spec.dead_gates && spec.dead_gates.total > 0);
  legendEl.innerHTML = `
    <ul class="circuit-legend">
      <li><span class="lg lg-enc"></span> Encoding gate RY(λ·x): writes an observation feature onto a qubit (re-uploaded every layer)</li>
      <li><span class="lg lg-var"></span> Trainable rotation RY/RZ(θ): the "weights" the agent learns</li>
      <li><span class="lg lg-cz"></span> CZ entangler ring: lets qubits influence each other</li>
      <li><span class="lg lg-meas"></span> ⟨Z⟩ readout: one expectation value per action</li>${
        hasDead
          ? `
      <li><span class="lg lg-dead"></span> Dimmed gate: outside every readout's light cone — it cannot influence any action${
        prunedOnHardware(spec) ? " and is skipped on hardware" : ""
      }</li>`
          : ""
      }
    </ul>
    <p class="hint">${esc(spec.n_qubits)} qubits × ${esc(spec.n_layers)} data re-uploading layers${
      caption ? ` — ${caption}` : ""
    }${
      caption && spec.dead_params > 0
        ? " (zero gradient for every input: they cannot change any Q-value)"
        : ""
    }.</p>`;
}

// -- who sees what --------------------------------------------------------------

/** The spec's A x n visibility matrix, or null when it is missing/malformed. */
function visibilityRows(spec) {
  const rows = spec && spec.visibility;
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const n = spec.n_qubits;
  return rows.every((row) => Array.isArray(row) && row.length === n) ? rows : null;
}

function actionName(spec, a) {
  const labels = Array.isArray(spec.action_labels) ? spec.action_labels : [];
  return String(labels[a] ?? `Z${a}`);
}

/** Feature j is encoded on qubit j; fall back to the wire name when the
 *  server sent no (or a mismatched) obs_labels list. */
function featureName(spec, obsLabels, j) {
  const ok = Array.isArray(obsLabels) && obsLabels.length === spec.n_qubits;
  return ok ? String(obsLabels[j]) : `q${j}`;
}

/** Actions with hidden features: [{action, hidden: [feature names]}], in
 *  action order; [] at full visibility, null without light-cone data. */
export function blindSpots(spec, obsLabels) {
  const rows = visibilityRows(spec);
  if (!rows) return null;
  const out = [];
  rows.forEach((row, a) => {
    const hidden = [];
    row.forEach((v, j) => {
      if (!v) hidden.push(featureName(spec, obsLabels, j));
    });
    if (hidden.length) out.push({ action: actionName(spec, a), hidden });
  });
  return out;
}

/** One plain-language line about the light cone: {kind: "warn" | "ok", text},
 *  or null without light-cone data. The warning names the last blind action
 *  in full (Brake, in the 4-action set) and counts the others — the matrix
 *  next to it shows every cell. */
export function visibilityNote(spec, obsLabels) {
  const blind = blindSpots(spec, obsLabels);
  if (!blind) return null;
  const n = spec.n_qubits;
  const layers = spec.n_layers;
  if (blind.length === 0) {
    return {
      kind: "ok",
      text: `Every action can see every input: ${layers} layers are deep enough at ${n} qubits.`,
    };
  }
  const example = blind[blind.length - 1];
  const others = blind.length - 1;
  const more =
    others > 0 ? ` (${others} more action${others > 1 ? "s" : ""} also miss inputs)` : "";
  const needed = spec.min_layers_full_visibility;
  const fix = Number.isInteger(needed) ? ` — full visibility needs ${needed} layers` : "";
  return {
    kind: "warn",
    text:
      `At ${n} qubits and ${layers} layers, ${example.action} cannot see: ` +
      `${example.hidden.join(", ")}${more}${fix}.`,
  };
}

function cellTip(spec, obsLabels, a, j, visible) {
  const action = actionName(spec, a);
  const feature = featureName(spec, obsLabels, j);
  if (visible) return `${action} can see ${feature}.`;
  // On the CZ ring an input reaches a readout qubit one neighbour per layer.
  const n = spec.n_qubits;
  const steps = Math.min(Math.abs(a - j), n - Math.abs(a - j));
  const reach = spec.n_layers - 1;
  return steps > reach
    ? `${action} cannot see ${feature}: it enters on qubit ${j}, ${steps} steps around the ` +
        `ring from readout qubit ${a} — ${spec.n_layers} layers only reach ${reach}.`
    : `${action} cannot see ${feature}: outside the light cone of readout qubit ${a}.`;
}

/** The matrix as an HTML table: one row per observation feature (in qubit
 *  order, like the diagram's wires), one column per action (like the Q-value
 *  bars). Filled cell = that action's Q-value can depend on that feature.
 *  Pure function; "" without light-cone data. */
export function visibilityMatrixHtml(spec, obsLabels) {
  const rows = visibilityRows(spec);
  if (!rows) return "";
  const n = spec.n_qubits;
  const nActions = rows.length;
  const head = rows
    .map((_, a) => `<th scope="col"><span>${esc(actionName(spec, a))}</span></th>`)
    .join("");
  const body = [];
  for (let j = 0; j < n; j++) {
    const cells = rows
      .map((row, a) => {
        const visible = Boolean(row[j]);
        const tip = esc(cellTip(spec, obsLabels, a, j, visible));
        return (
          `<td class="${visible ? "vis-on" : "vis-off"}" data-tip="${tip}" aria-label="${tip}">` +
          '<span class="vis-mark"></span></td>'
        );
      })
      .join("");
    // "q3 speed": the wire name ties the row to the diagram; without labels
    // from the server the wire name is all there is
    const name = featureName(spec, obsLabels, j);
    const label =
      name === `q${j}` ? name : `<span class="vis-qubit">q${j}</span> ${esc(name)}`;
    body.push(`<tr><th scope="row">${label}</th>${cells}</tr>`);
  }
  // many actions: turn the column headers sideways so narrow sidebars still fit
  return (
    `<table class="vis-matrix${nActions > 4 ? " vis-many" : ""}">` +
    `<thead><tr><td></td>${head}</tr></thead><tbody>${body.join("")}</tbody></table>`
  );
}

/** Render the "who sees what" section (heading, one-line note, matrix) into
 *  `el`; hides `el` when the spec carries no light-cone data. Re-run on every
 *  welcome, so qubit-count and driver switches update it. */
export function renderVisibility(spec, obsLabels, el) {
  if (!el) return;
  const note = spec ? visibilityNote(spec, obsLabels) : null;
  if (!note) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  const noteHtml =
    note.kind === "warn"
      ? `<p class="cone-note cone-warn"><span aria-hidden="true">⚠</span> ${esc(note.text)}</p>`
      : `<p class="cone-note">${esc(note.text)}</p>`;
  el.innerHTML = `
    <h2>Who sees what</h2>
    ${noteHtml}
    <div class="vis-wrap">${visibilityMatrixHtml(spec, obsLabels)}</div>
    <ul class="circuit-legend">
      <li><span class="lg vis-mark-on"></span> Filled: this action's Q-value can depend on this input</li>${
        note.kind === "warn"
          ? `
      <li><span class="lg vis-mark-off"></span> Empty: it cannot — the input lies outside that action's light cone</li>`
          : ""
      }
    </ul>`;
  el.hidden = false;
}
