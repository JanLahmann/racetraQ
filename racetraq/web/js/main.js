// Boot + wiring: websocket handlers, mode/track UI, lap board, panels.

import * as net from "./net.js";
import { RaceRenderer, KIND_COLORS } from "./race.js";
import { initInput, setInputActive } from "./input.js";
import { QuantumPanel } from "./quantum-panel.js";
import { renderCircuit, renderVisibility } from "./circuit.js";
import { TrainingChart, LapChart } from "./charts.js";
import { AttractManager } from "./attract.js";
import { initDraw } from "./draw.js";
import { initExplain } from "./explain.js";
import { initHardwarePanel } from "./hardware-panel.js";
import { initTooltips } from "./tooltip.js";
import { initStudioPanel } from "./studio-panel.js";

const $ = (sel) => document.querySelector(sel);

const state = {
  mode: "attract",
  tracks: [],
  trackName: null,
  bestLaps: new Map(), // car id -> best lap seconds
  lastState: null,
  training: false,
};

// -- components --------------------------------------------------------------

const renderer = new RaceRenderer($("#race-canvas"));
renderer.start();

const quantumPanel = new QuantumPanel({
  gaugesEl: $("#qgauges"),
  barsEl: $("#qbars"),
  actionEl: $("#qaction"),
});

const chart = new TrainingChart($("#training-chart"));
const lapChart = new LapChart($("#lap-chart"));

const attract = new AttractManager({
  captionEl: $("#attract-caption"),
  onIdle: () => {
    net.idleReset(); // server: attract mode, name cleared, booth defaults
    resetBoothUI();
    applyMode("attract");
  },
});

initExplain($("#panel-explain"));
const studioPanel = initStudioPanel({
  root: $("#panel-studio"),
  send: net.studioCmd,
  moderate: (opts) => net.boardCmd("remove", { board: "studio", ...opts }),
  stage: $("#studio-stage"),
  onTrack: (name) => {
    if (name !== state.trackName) net.setTrack(name); // the stage shows the chosen track
  },
  onFirstLap: (episode) => showBanner(`FIRST LAP · episode ${episode}`),
  laps: () => ({ human: state.bestLaps.get("human"), model: state.bestLaps.get("studio") }),
  active: () => state.mode === "studio",
  setName: (name) => {
    net.setName(name);
    $("#race-name").value = name;
  },
  onStart: () => {
    chart.reset();
    lapChart.reset();
    episodeByAgent.clear();
    renderEpisodeOverlay();
  },
  onPhase: (phase) => {
    attract.setHold("studio", phase === "training" ? "hold" : phase === "done" ? "linger" : null);
    const driving = state.mode === "race" || (state.mode === "studio" && phase === "race");
    setInputActive(driving);
    $("#race-camera").hidden = !driving;
    renderEpisodeOverlay();
  },
});
const hardwarePanel = initHardwarePanel();
initInput(() => attract.notifyActivity(), {
  onGamepadChange: (connected) => {
    $("#gamepad-pill").hidden = !connected;
  },
  // attract mode: a driving key or a pad button starts a race ("press to race")
  onWake: () => {
    if (state.mode !== "attract") return;
    startRace();
    applyMode("race"); // optimistic; the server's state confirms
  },
});

// -- helpers -----------------------------------------------------------------

// the names racetraq.org uses too (track ids stay the short file names)
const TRACK_NAMES = { oval: "Oval", chicane: "Chicane", gp: "Grand Prix", combo: "Combo" };

const KIND_NAMES = { quantum: "Quantum", mlp: "MLP", human: "You", hero: "Hero", pro: "Pro" };

function fmtLap(t) {
  if (typeof t !== "number" || !isFinite(t)) return "—";
  return `${t.toFixed(2)}s`;
}

function toast(message, isError = true) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("toast-error", isError);
  el.hidden = false;
  clearTimeout(toast._id);
  toast._id = setTimeout(() => {
    el.hidden = true;
  }, 4000);
}

function setStatus(text, cls) {
  const pill = $("#status-pill");
  pill.textContent = text;
  pill.className = `pill ${cls}`;
}

function applyMode(mode) {
  state.mode = mode;
  for (const btn of document.querySelectorAll(".mode-btn")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  $("#race-controls").hidden = mode !== "race";
  $("#race-camera").hidden = !humanDriving();
  $("#driver-picker").hidden = mode !== "attract";
  setInputActive(mode === "race" || (mode === "studio" && studioPanel.phase === "race"));
  studioPanel.syncStage();
  attract.setHold("evolution", mode === "evolution" ? "hold" : null);
  if (mode !== "train") attract.setHold("train", null);
  if (mode !== "hardware") attract.setHold("hardware", null);
  if (mode === "attract") $("#race-name").value = ""; // the next visitor starts anonymous
  attract.setMode(mode);
  renderer.setMode(mode);
  $("#evo-caption").hidden = mode !== "evolution";
  $("#attract-headline").hidden = mode !== "attract";
  if (mode !== "race") hideRaceOverlays();
  $("#evo-legend").hidden = true; // refilled from the next state broadcast
  evoLegendKey = "";
  renderEpisodeOverlay();
  if (mode === "train") selectTab("training");
  else if (mode === "studio") selectTab("studio");
  else if (mode === "hardware") selectTab("hardware");
  else if (mode === "attract" || mode === "evolution") selectTab("quantum");
}

function selectTab(name) {
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  }
}

function applyTrack(payload) {
  state.trackName = payload.name;
  studioPanel.setTrack(payload.name);
  state.bestLaps.clear();
  renderer.setTrack(payload);
  const sel = $("#track-select");
  // Generated tracks are named "random #<seed>"; they map onto the picker's
  // "random" entry, whose label shows the seed so the track is reproducible.
  const isRandom = payload.name.startsWith("random #");
  const randomOpt = sel.querySelector('option[value="random"]');
  if (randomOpt) randomOpt.textContent = isRandom ? `🎲 ${payload.name}` : "🎲 random";
  // Drawn tracks ("drawn #N") live in a transient picker entry of their own.
  const isDrawn = payload.name.startsWith("drawn #");
  let drawnOpt = sel.querySelector('option[value="drawn"]');
  if (isDrawn && !drawnOpt) {
    drawnOpt = document.createElement("option");
    drawnOpt.value = "drawn";
    sel.append(drawnOpt);
  }
  if (drawnOpt) {
    if (isDrawn) drawnOpt.textContent = `✏️ ${payload.name}`;
    else drawnOpt.remove();
  }
  const value = isRandom ? "random" : isDrawn ? "drawn" : payload.name;
  if (sel.value !== value) sel.value = value;
  $("#track-reroll").hidden = !isRandom;
  const seedInput = $("#track-seed");
  seedInput.hidden = !isRandom;
  $("#track-length").hidden = !isRandom;
  if (isRandom) {
    // show the active seed as the placeholder (copy it to save the track);
    // clear any typed value so the next 🎲 press rolls a fresh one
    const m = payload.name.match(/^random #(\d+)/);
    if (m) seedInput.placeholder = m[1];
    seedInput.value = "";
  }
}

// Expert mode (open the page with #expert) reveals the hidden hero driver:
// a model-based racing-line controller, the demo's "perfect drive" ceiling.
const EXPERT = window.location.hash.includes("expert");

/** Populate the Watch-mode driver picker from welcome.drivers/driver. */
function applyDrivers(drivers, current) {
  const sel = $("#driver-select");
  const label = (d) =>
    d === "auto" ? "auto (this track)"
    : d === "hero" ? "hero — racing line"
    : d === "pro" ? "pro — big classical DQN"
    : `${d}-trained`;
  sel.replaceChildren(
    ...(drivers || ["auto"])
      .filter((d) => EXPERT || (d !== "hero" && d !== "pro"))
      .map((d) => {
        const opt = document.createElement("option");
        opt.value = d;
        opt.textContent = label(d);
        return opt;
      }),
  );
  sel.value =
    current && (EXPERT || (current !== "hero" && current !== "pro"))
      ? current : "auto";
}

/** Ask for a generated track: typed seed (empty -> fresh roll) + length. */
function requestRandomTrack() {
  const raw = $("#track-seed").value.trim();
  const seed = /^\d+$/.test(raw) ? parseInt(raw, 10) : undefined;
  const length = $("#track-length").value;
  net.setTrack("random", seed, length === "medium" ? undefined : length);
}

// Circuit-size copy (q6/q8/q10 profiles, a live qubit switch or a driver with
// its own observation): fix up the captions, the readout hint, the header
// dropdown and the Explain panel from the welcome's circuit_spec and
// obs_labels. At the default 4 qubits the authored text is reproduced, so
// switching back from a larger circuit restores it.
function applyCircuitSize(spec, obsLabels) {
  attract.setCircuitSpec(spec, obsLabels);
  const n = spec.n_qubits || 4;
  const sel = $("#qubit-select");
  if (sel && sel.value !== String(n)) sel.value = String(n);
  const hint = $("#qubit-hint");
  if (hint) {
    hint.innerHTML =
      n === 4
        ? "Pauli-Z expectation values &lt;Z<sub>a</sub>&gt; of the 4 qubits — one per action."
        : `Pauli-Z expectation values &lt;Z<sub>i</sub>&gt; of all ${n} qubits — ` +
          "the highlighted first four are the action readout.";
  }
  initExplain($("#panel-explain"), spec, obsLabels);
}

// Observation feature names (welcome.obs_labels): what each encoded input is.
function applyObsLabels(labels) {
  const el = $("#obs-labels");
  if (!el) return;
  el.textContent = Array.isArray(labels) && labels.length ? `Inputs: ${labels.join(" · ")}` : "";
}

// -- lap board ---------------------------------------------------------------

let lapboardAt = 0;

function renderLapboard(cars) {
  const now = performance.now();
  if (now - lapboardAt < 250) return;
  lapboardAt = now;
  const board = $("#lapboard");
  const rows = cars.map((car) => {
    const best = state.bestLaps.get(car.id);
    if (car.ghost) {
      // ghosts stay off the board except for a dim "Ghost (best …)" entry
      const t = typeof best === "number" ? best : car.last_lap_time;
      // car.label carries the record's provenance ("best 14.2s · universal")
      const text =
        typeof t === "number" && isFinite(t)
          ? `Ghost (${car.label || `best ${fmtLap(t)}`})`
          : car.label
            ? `Ghost — ${car.label}`
            : "Ghost";
      return `<div class="lap-row ghost">
        <span class="dot ghost-dot"></span>
        <span class="lap-kind">${text}</span>
      </div>`;
    }
    const evo = state.mode === "evolution" && car.label;
    const color = evo ? renderer.stageColor(car.label) : KIND_COLORS[car.kind] || "#ccc";
    const name = evo ? car.label : KIND_NAMES[car.kind] || car.kind;
    return `<div class="lap-row${car.off_track ? " off" : ""}">
      <span class="dot" style="background:${color}"></span>
      <span class="lap-kind">${name}</span>
      <span class="lap-cell">Lap <b>${car.lap}</b></span>
      <span class="lap-cell">Last <b>${fmtLap(car.last_lap_time)}</b></span>
      <span class="lap-cell">Best <b>${fmtLap(best)}</b></span>
    </div>`;
  });
  board.innerHTML = rows.join("");
}

// -- car legend ----------------------------------------------------------------
// Corner legend for every labelled car (evolution stages, hero/pro driver
// descriptions, ghosts): the description lives here, colors identify the
// moving cars — no text is attached to the cars themselves.

let evoLegendKey = "";

function updateCarLegend(cars) {
  const labeled = cars.filter((c) => c.label);
  const key = state.mode + "|" +
    labeled.map((c) => `${c.label}${c.ghost ? "*" : ""}`).join("|");
  if (key === evoLegendKey) return;
  evoLegendKey = key;
  const el = $("#evo-legend");
  el.hidden = labeled.length === 0;
  el.innerHTML = labeled
    .map((c) => {
      if (c.ghost) {
        return `<div class="legend-row ghost"><span class="dot ghost-dot"></span>${c.label}</div>`;
      }
      const evo = state.mode === "evolution";
      const color = evo ? renderer.stageColor(c.label) : KIND_COLORS[c.kind] || "#ccc";
      const num = evo ? `<b style="color:${color}">${renderer.stageNumber(c.label)}</b> ` : "";
      return `<div class="legend-row"><span class="dot" style="background:${color}"></span>${num}${c.label}</div>`;
    })
    .join("");
}

// -- episode counter + best-lap banner -----------------------------------------

const episodeByAgent = new Map(); // agent -> latest episode

function renderEpisodeOverlay() {
  const el = $("#episode-overlay");
  // (the studio's own ticker on the stage carries the episode count)
  const training = state.mode === "train";
  if (!training || episodeByAgent.size === 0) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.innerHTML = [...episodeByAgent.entries()]
    .map(
      ([agent, ep]) =>
        `<div class="ep-line" style="color:${KIND_COLORS[agent] || "#e6e9ef"}">ep ${ep}</div>`,
    )
    .join("");
}

function showBestBanner(lapTime) {
  showBanner(`NEW BEST LAP ${lapTime.toFixed(2)}s`);
}

function showBanner(text) {
  const el = $("#best-banner");
  el.textContent = text;
  el.hidden = false;
  el.classList.remove("banner-in");
  void el.offsetWidth; // retrigger the animation
  el.classList.add("banner-in");
  clearTimeout(showBestBanner._id);
  showBestBanner._id = setTimeout(() => {
    el.hidden = true;
  }, 1600);
}

// -- websocket handlers ------------------------------------------------------

// Lost server: after a short grace a stage overlay says so (a 12 px pill is
// easy to miss from a booth's distance); it clears on reconnect.
let disconnectTimer = null;
net.on("_open", () => {
  setStatus("connected", "pill-ok");
  clearTimeout(disconnectTimer);
  $("#disconnect-overlay").hidden = true;
});
net.on("_close", () => {
  setStatus("reconnecting…", "pill-off");
  clearTimeout(disconnectTimer);
  disconnectTimer = setTimeout(() => {
    $("#disconnect-overlay").hidden = false;
  }, 2500);
});

net.on("welcome", (msg) => {
  state.tracks = msg.tracks || [];
  const sel = $("#track-select");
  const randomOpt = document.createElement("option");
  randomOpt.value = "random";
  randomOpt.textContent = "🎲 random";
  sel.replaceChildren(
    ...state.tracks.map((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = TRACK_NAMES[name] || name;
      return opt;
    }),
    randomOpt,
  );
  if (msg.circuit_spec) {
    renderCircuit(msg.circuit_spec, $("#circuit-diagram"), $("#circuit-legend"));
    // light cone: which inputs each action can see (labels change with the driver)
    renderVisibility(msg.circuit_spec, msg.obs_labels, $("#light-cone"));
    applyCircuitSize(msg.circuit_spec, msg.obs_labels);
    quantumPanel.setCircuit(msg.circuit_spec);
  }
  applyObsLabels(msg.obs_labels);
  applyDrivers(msg.drivers, msg.driver);
  applyQubitOptions(msg.qubit_options);
  if (msg.ui) {
    attract.setIdleSeconds(msg.ui.attract_idle_seconds || 45);
    document.body.classList.toggle("kiosk", Boolean(msg.ui.kiosk));
  }
  if (msg.track) applyTrack(msg.track);
  applyMode(msg.mode || "attract");
});

net.on("track", (msg) => {
  applyTrack(msg.track);
  applyQubitOptions(msg.qubit_options);
});

/** Grey out circuit sizes without a trained driver for this track (an
 *  untrained size would leave Watch with no car). */
function applyQubitOptions(options) {
  if (!Array.isArray(options)) return;
  for (const opt of $("#qubit-select").options) {
    const ok = options.includes(Number(opt.value));
    opt.disabled = !ok;
    opt.textContent = ok ? opt.value : `${opt.value} (no driver for this track)`;
  }
}

// Leaderboard: ranked named human laps + unranked AI reference rows.
const isOperator = () => document.body.classList.contains("operator");

net.on("leaderboard", (msg) => {
  $("#board-track").textContent = TRACK_NAMES[msg.track] || msg.track;
  state.boardTrack = msg.track;
  const entries = $("#board-entries");
  // #operator: a ✕ per entry and the clear buttons (the server takes them
  // from the booth machine only)
  const remove = (e) => isOperator()
    ? `<button type="button" class="board-remove" data-name="${escapeHtml(e.name)}"
        data-lap="${e.lap_s}" title="Remove this entry" aria-label="Remove ${escapeHtml(e.name)}">✕</button>`
    : "";
  entries.innerHTML = msg.entries
    .map((e) => `<li><span class="board-name">${escapeHtml(e.name)}</span>
      <b>${e.lap_s.toFixed(2)}s</b><span class="board-date">${e.date || ""}</span>${remove(e)}</li>`)
    .join("");
  $("#board-moderation").hidden = !isOperator();
  $("#board-empty").hidden = msg.entries.length > 0;
  $("#board-references").innerHTML = msg.references
    .map((r) => `<div class="board-ref">
      <span class="dot" style="background:${KIND_COLORS[r.kind] || "#ccc"}"></span>
      <span>${KIND_NAMES[r.kind] || r.kind}</span>
      <span class="board-driver">${escapeHtml(r.driver)}</span>
      <b>${r.lap_s.toFixed(2)}s</b>
    </div>`)
    .join("") || '<p class="hint">No reference laps on this track yet.</p>';
});

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Driver lock + turn queue: one client controls the shared session at a
// time; interacting while someone else drives joins the waiting line, and
// turns are time-limited only while the line is non-empty.
net.on("control", (msg) => {
  attract.setDriving(msg.driving);
  const pill = $("#control-pill");
  let text = null;
  if (msg.driving) {
    if (msg.turn_ends_in_s != null) {
      text = `🏎️ your turn · ${msg.turn_ends_in_s}s left · ${msg.waiting} waiting`;
    } else if (msg.watchers > 0) {
      text = `🏎️ you have the wheel · ${msg.watchers} watching`;
    }
  } else if (msg.queue_pos != null) {
    const ahead = msg.queue_pos - 1;
    text = ahead === 0
      ? `⏳ you're next · your turn in ~${msg.turn_ends_in_s ?? "?"}s`
      : `⏳ in line — ${ahead} ahead of you`;
  } else if (msg.locked) {
    text = "👀 spectating — press any control to get in line";
  }
  pill.hidden = text === null;
  if (text !== null) pill.textContent = text;
});

// -- race feedback: countdown, HUD, off-track, lap result ------------------------

let goShownUntil = 0;

function hideRaceOverlays() {
  for (const id of ["#race-hud", "#race-countdown", "#offtrack-banner", "#lap-result"]) {
    $(id).hidden = true;
  }
}

function renderRaceOverlays(msg) {
  if (msg.mode !== "race") return;
  const cd = $("#race-countdown");
  const now = performance.now();
  if (typeof msg.countdown === "number") {
    cd.hidden = false;
    cd.classList.remove("go");
    cd.textContent = String(Math.max(1, Math.ceil(msg.countdown)));
    goShownUntil = now + 700; // "GO!" right after the last count
  } else if (now < goShownUntil) {
    cd.hidden = false;
    cd.classList.add("go");
    cd.textContent = "GO!";
  } else {
    cd.hidden = true;
  }
  const me = (msg.cars || []).find((c) => c.kind === "human" && !c.ghost);
  const hud = $("#race-hud");
  hud.hidden = !me;
  if (!me) return;
  const best = state.bestLaps.get(me.id);
  hud.innerHTML = `<div class="hud-time">${fmtLap(me.lap_t ?? 0)}</div>
    <div class="hud-meta">Lap ${me.lap + 1} · last ${fmtLap(me.last_lap_time)} · best ${fmtLap(best)}</div>`;
  $("#offtrack-banner").hidden = !me.off_track;
}

function showLapResult(msg) {
  const el = $("#lap-result");
  let rank = "";
  let note = "";
  if (!msg.clean) {
    note = "You left the track this lap, so it doesn't count. Next one!";
  } else if (msg.named && msg.rank) {
    rank = `#${msg.rank} on the board`;
  } else if (msg.rank) {
    rank = `That's #${msg.rank} on the board`;
    note = "Type your name next to Start to get on it.";
  } else {
    note = msg.named ? "Not on the board this time — keep going!" : "";
  }
  el.innerHTML = `<div class="lr-note">Your lap</div>
    <div class="lr-time">${fmtLap(msg.lap_time)}</div>
    ${rank ? `<div class="lr-rank">${rank}</div>` : ""}
    ${note ? `<div class="lr-note">${note}</div>` : ""}`;
  el.hidden = false;
  clearTimeout(showLapResult._id);
  showLapResult._id = setTimeout(() => {
    el.hidden = true;
  }, 5000);
}

net.on("state", (msg) => {
  state.lastState = msg;
  renderRaceOverlays(msg);
  renderer.pushState(msg);
  renderLapboard(msg.cars || []);
  updateCarLegend(msg.cars || []);
  if (msg.mode && msg.mode !== state.mode) applyMode(msg.mode); // server-driven mode change
});

net.on("quantum", (msg) => quantumPanel.update(msg));

// a hardware job (and the replay of its lap) holds off the idle return
const HARDWARE_HOLD_PHASES = new Set(["connecting", "transpiling", "running", "replay"]);
net.on("hardware_status", (msg) => {
  attract.setHold("hardware", HARDWARE_HOLD_PHASES.has(msg.phase) ? "hold" : null);
  hardwarePanel.handleStatus(msg);
});

net.on("studio", (msg) => studioPanel.handleStatus(msg));

net.on("telemetry", (msg) => {
  if (state.mode === "train") attract.setHold("train", "hold"); // a run is live
  chart.addPoint(msg.agent, msg.episode, msg.mean_return, msg.epsilon);
  if (msg.lap_times !== undefined || msg.best_lap_s != null) {
    lapChart.setAgentData(msg.agent, msg.lap_times, msg.best_lap_s);
  }
  if (typeof msg.episode === "number") {
    episodeByAgent.set(msg.agent, msg.episode);
    renderEpisodeOverlay();
  }
  updateTrainStats(msg);
});

net.on("event", (msg) => {
  switch (msg.kind) {
    case "lap":
    case "clean_lap":
      if (msg.car_id && state.mode === "studio") setTimeout(() => studioPanel.refresh(), 0);
      if (msg.car_id) {
        renderer.addEffect("lap", msg.car_id);
        if (typeof msg.lap_time === "number") {
          const best = state.bestLaps.get(msg.car_id);
          if (best === undefined || msg.lap_time < best) {
            state.bestLaps.set(msg.car_id, msg.lap_time);
          }
        }
      }
      break;
    case "lap_result":
      showLapResult(msg);
      break;
    case "crash":
      if (msg.car_id) renderer.addEffect("crash", msg.car_id);
      break;
    case "new_best_lap":
      if (typeof msg.lap_time === "number") {
        showBestBanner(msg.lap_time);
        if (msg.car_id) {
          const best = state.bestLaps.get(msg.car_id);
          if (best === undefined || msg.lap_time < best) {
            state.bestLaps.set(msg.car_id, msg.lap_time);
          }
        }
      }
      break;
    case "training_done":
      state.training = false;
      attract.setHold("train", null);
      $("#train-start").disabled = false;
      toast(`Training done${msg.agent ? ` (${msg.agent})` : ""}`, false);
      break;
  }
});

// Server errors are written for operators ("bundled mlp weights use …"): a
// kiosk shows visitors a plain line and keeps the detail in the console
// (#operator shows it).
net.on("error", (msg) => {
  const detail = msg.message || "server error";
  if (msg.field === "name") {
    $("#race-name").value = "";
    studioPanel.clearName();
  }
  if (msg.visitor) {
    toast(detail); // written for visitors
  } else if (document.body.classList.contains("kiosk") && !isOperator()) {
    console.warn("racetraQ:", detail);
    toast("That isn't available right now — try another track or mode.");
  } else {
    toast(detail);
  }
});

// -- UI wiring ---------------------------------------------------------------

const MODE_FOR_BUTTON = {
  attract: "attract",
  train: "train",
  evolution: "evolution",
  race: "race",
  hardware: "hardware",
  studio: "studio",
};

for (const btn of document.querySelectorAll(".mode-btn")) {
  btn.addEventListener("click", () => {
    const mode = MODE_FOR_BUTTON[btn.dataset.mode];
    net.setMode(mode);
    applyMode(mode); // optimistic; server `state.mode` confirms
  });
}

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => selectTab(btn.dataset.tab));
}

// Picking "random" (and each reroll click) asks the server for a generated
// track — a fresh roll, or a specific one when a seed is typed; the answering
// track payload carries the seed in its name. Changing length regenerates.
$("#track-select").addEventListener("change", (ev) => {
  if (ev.target.value === "random") requestRandomTrack();
  else if (ev.target.value !== "drawn") net.setTrack(ev.target.value); // "drawn" IS the current track
});
$("#track-reroll").addEventListener("click", requestRandomTrack);
$("#track-seed").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") requestRandomTrack();
});
$("#track-length").addEventListener("change", requestRandomTrack);

$("#qubit-select").addEventListener("change", (ev) => {
  net.setQubits(parseInt(ev.target.value, 10)); // server answers with a fresh welcome
});

$("#driver-select").addEventListener("change", (ev) => {
  net.setDriver(ev.target.value); // server rebuilds the attract car + re-welcomes
});

function startRace() {
  net.setName($("#race-name").value.trim()); // leaderboard name for this stint
  net.raceCmd("start", $("#race-opponent").value, state.trackName || undefined);
}
$("#race-start").addEventListener("click", (ev) => {
  startRace();
  ev.currentTarget.blur(); // Space and Enter are for driving now, not a restart
});
$("#race-name").addEventListener("keydown", (ev) => {
  if (ev.key !== "Enter") return;
  startRace();
  ev.currentTarget.blur(); // hand the keys to the car
});

/** Booth defaults for the next visitor (the server resets its side on
 *  idle_reset): no name, the stock opponent and camera, a fresh studio. */
function resetBoothUI() {
  $("#race-name").value = "";
  $("#race-opponent").value = "quantum";
  setCamera("top");
  studioPanel.reset();
}
$("#race-reset").addEventListener("click", () => {
  state.bestLaps.clear();
  net.raceCmd("reset", $("#race-opponent").value);
});

// Camera for the human driver: button or C cycles top / chase / cockpit; the
// choice is remembered in this browser.
const CAMERA_LABELS = { top: "Top", chase: "Chase", cockpit: "Cockpit" };
function setCamera(view) {
  renderer.setCamera(view);
  $("#race-camera").textContent = `📷 ${CAMERA_LABELS[renderer.camera]}`;
  try {
    localStorage.setItem("racetraq-camera", renderer.camera);
  } catch {
    // storage blocked: the choice just isn't remembered
  }
}
/** A visitor drives: Race mode, or racing their own model in the studio. */
function humanDriving() {
  return state.mode === "race" || (state.mode === "studio" && studioPanel.phase === "race");
}
function cycleCamera() {
  setCamera(renderer.cycleCamera());
}
try {
  setCamera(localStorage.getItem("racetraq-camera") || "top");
} catch {
  setCamera("top");
}
$("#race-camera").addEventListener("click", (ev) => {
  cycleCamera();
  ev.currentTarget.blur(); // keep Space/Enter for the race, not this button
});
window.addEventListener("keydown", (ev) => {
  if (ev.code !== "KeyC" || ev.repeat || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (ev.target.closest && ev.target.closest("input, select, textarea")) return;
  if (humanDriving()) cycleCamera();
});

$("#train-start").addEventListener("click", () => {
  const agent = $("#train-agent").value;
  const episodes = parseInt($("#train-episodes").value, 10);
  chart.reset();
  lapChart.reset();
  episodeByAgent.clear();
  renderEpisodeOverlay();
  state.training = true;
  attract.setHold("train", "hold");
  $("#train-start").disabled = true;
  net.trainCmd("start", agent, {
    track: state.trackName || undefined,
    warm: $("#train-warm").checked,
    episodes: Number.isFinite(episodes) ? episodes : undefined,
  });
});
$("#train-stop").addEventListener("click", () => {
  net.trainCmd("stop", $("#train-agent").value);
  state.training = false;
  attract.setHold("train", null);
  $("#train-start").disabled = false;
});

const trainStats = new Map(); // agent -> latest telemetry

function updateTrainStats(msg) {
  trainStats.set(msg.agent, msg);
  const rows = [...trainStats.values()].map(
    (m) => `<div class="stat-row">
      <span class="dot" style="background:${KIND_COLORS[m.agent] || "#ccc"}"></span>
      <span>${KIND_NAMES[m.agent] || m.agent}</span>
      <span>ep <b>${m.episode}</b></span>
      <span>ret <b>${m.mean_return.toFixed(1)}</b></span>
      <span>ε <b>${m.epsilon.toFixed(2)}</b></span>
      <span>loss <b>${m.loss == null ? "—" : m.loss.toFixed(3)}</b></span>
    </div>`,
  );
  $("#train-stats").innerHTML = rows.join("");
}

// -- draw-a-track ----------------------------------------------------------------

initDraw({
  button: $("#track-draw"),
  stage: $("#stage"),
  getTransform: () => renderer.transform,
  submit: (points) => net.drawTrack(points),
});

// -- sidebar resize ------------------------------------------------------------

{
  const resizer = $("#sidebar-resizer");
  const setWidth = (px) => {
    const w = Math.round(Math.min(Math.max(px, 300), window.innerWidth * 0.7));
    document.documentElement.style.setProperty("--sidebar-w", `${w}px`);
    return w;
  };
  const saved = Number(localStorage.getItem("traq-sidebar-w"));
  if (saved) setWidth(saved);
  resizer.addEventListener("pointerdown", (ev) => {
    ev.preventDefault();
    resizer.setPointerCapture(ev.pointerId);
    resizer.classList.add("dragging");
  });
  resizer.addEventListener("pointermove", (ev) => {
    if (!resizer.hasPointerCapture(ev.pointerId)) return;
    setWidth(window.innerWidth - ev.clientX);
  });
  resizer.addEventListener("pointerup", (ev) => {
    resizer.releasePointerCapture(ev.pointerId);
    resizer.classList.remove("dragging");
    const w = setWidth(window.innerWidth - ev.clientX);
    localStorage.setItem("traq-sidebar-w", String(w));
  });
  resizer.addEventListener("dblclick", () => {
    document.documentElement.style.removeProperty("--sidebar-w");
    localStorage.removeItem("traq-sidebar-w");
  });
}

// -- board moderation (#operator) -----------------------------------------------

$("#board-entries").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".board-remove");
  if (!btn) return;
  net.boardCmd("remove", { track: state.boardTrack, name: btn.dataset.name,
    lap_s: Number(btn.dataset.lap) });
});
$("#board-clear-today").addEventListener("click", () => {
  net.boardCmd("clear_today", { track: state.boardTrack });
});
// two clicks to clear the whole board (the page cannot show a confirm dialog)
$("#board-clear").addEventListener("click", (ev) => {
  const btn = ev.currentTarget;
  if (btn.dataset.armed === "1") {
    net.boardCmd("clear", { track: state.boardTrack });
    btn.dataset.armed = "";
    btn.textContent = "Clear the board";
    return;
  }
  btn.dataset.armed = "1";
  btn.textContent = "Click again to clear";
  setTimeout(() => {
    btn.dataset.armed = "";
    btn.textContent = "Clear the board";
  }, 3000);
});

// -- boot --------------------------------------------------------------------

// #operator: a kiosk shows its operator controls (Train, Hardware, Qubits,
// Driver) again — for setting up the booth, not for visitors.
document.body.classList.toggle("operator", window.location.hash.includes("operator"));

initTooltips();
setStatus("connecting…", "pill-off");
applyMode("attract");
net.connect();
