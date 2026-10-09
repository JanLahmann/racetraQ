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
import {
  t, has, lang, setLang, loadI18n, applyDom, onLangChange, storedLang, clearStoredLang,
  featureLabel,
} from "./i18n.js";

const $ = (sel) => document.querySelector(sel);

// Both string tables load before anything renders; until the welcome says
// otherwise the page speaks the visitor's stored choice (or English).
await loadI18n();
setLang(storedLang() || "en", { persist: false });
applyDom();

const state = {
  mode: "attract",
  tracks: [],
  trackName: null,
  bestLaps: new Map(), // car id -> best lap seconds
  lastState: null,
  training: false,
  // the last of each message the page renders from: a language switch
  // re-renders them
  welcome: null,
  leaderboard: null,
  control: null,
  status: ["status.connecting", "pill-off"],
  boothLang: "en", // [ui] language: the default when the visitor chose none
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
  onFirstLap: (episode) => showBanner(t("banner.first_lap", { episode })),
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

/** A track's display name: the bundled ones by name, generated and drawn
 *  ones ("random #42", "drawn #3") in the active language. */
function trackLabel(name) {
  if (TRACK_NAMES[name]) return TRACK_NAMES[name];
  const m = /^(random|drawn) #(\d+)/.exec(name || "");
  return m ? t(`track.${m[1]}_n`, { n: m[2] }) : name;
}

/** Driver kind ("quantum", "mlp", "human", …) in the active language. */
const kindName = (kind) => (has(`kind.${kind}`) ? t(`kind.${kind}`) : kind);

// Car labels come from the server in English ("ep 250", "best 14.2s · gp",
// "driver: gp-trained", "your circuit · Ada"); the known shapes are shown in
// the active language, anything else as sent.
const CAR_LABELS = [
  [/^ep (\d+)$/, (m) => t("car.ep", { n: m[1] })],
  [/^stage (\d+)$/, (m) => t("car.stage", { n: m[1] })],
  [/^best \(of (\d+) ep run\)$/, (m) => t("car.best_of", { n: m[1] })],
  [/^best$/, () => t("car.best")],
  [/^best (\d+(?:\.\d+)?s)(?: · (.+))?$/, (m) =>
    t("car.ghost_best", { time: m[1] }) + (m[2] ? ` · ${carDriver(m[2])}` : "")],
  [/^driver: (.+)$/, (m) => t("car.driver", { driver: carDriver(m[1]) })],
  [/^your circuit(?: · (.+))?$/, (m) => t("car.your_circuit") + (m[1] ? ` · ${m[1]}` : "")],
  [/^hardware lap$/, () => t("car.hardware_lap")],
];
function carDriver(text) {
  if (text === "racing line (model-based, not learned)") return t("car.driver_hero");
  if (text === "pro (classical DQN, big MLP)") return t("car.driver_pro");
  if (text === "gp-trained generalist") return t("car.driver_generalist");
  const m = /^(.+)-trained$/.exec(text);
  return m ? t("car.trained", { track: m[1] }) : text;
}
function carLabel(label) {
  for (const [re, fn] of CAR_LABELS) {
    const m = re.exec(label);
    if (m) return fn(m);
  }
  return label;
}

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

function setStatus(key, cls) {
  state.status = [key, cls];
  const pill = $("#status-pill");
  pill.textContent = t(key);
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
  // Drawn tracks ("drawn #N") live in a transient picker entry of their own.
  const isDrawn = payload.name.startsWith("drawn #");
  let drawnOpt = sel.querySelector('option[value="drawn"]');
  if (isDrawn && !drawnOpt) {
    drawnOpt = document.createElement("option");
    drawnOpt.value = "drawn";
    sel.append(drawnOpt);
  }
  if (drawnOpt && !isDrawn) drawnOpt.remove();
  applyTrackLabels();
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

/** The picker's 🎲 / ✏️ entries name the current generated or drawn track
 *  ("🎲 random #42"; plain "🎲 random" otherwise). */
function applyTrackLabels() {
  const sel = $("#track-select");
  const name = state.trackName || "";
  const randomOpt = sel.querySelector('option[value="random"]');
  if (randomOpt) {
    randomOpt.textContent = `🎲 ${name.startsWith("random #") ? trackLabel(name) : t("track.random")}`;
  }
  const drawnOpt = sel.querySelector('option[value="drawn"]');
  if (drawnOpt && name.startsWith("drawn #")) drawnOpt.textContent = `✏️ ${trackLabel(name)}`;
}

/** Populate the Watch-mode driver picker from welcome.drivers/driver. */
function applyDrivers(drivers, current) {
  const sel = $("#driver-select");
  const label = (d) =>
    d === "auto" ? t("driver.auto")
    : d === "hero" ? t("driver.hero")
    : d === "pro" ? t("driver.pro")
    : t("driver.trained", { track: d });
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
  if (hint) hint.innerHTML = n === 4 ? t("quantum.hint4") : t("quantum.hint_n", { n });
  initExplain($("#panel-explain"), spec, obsLabels);
}

// Observation feature names (welcome.obs_labels): what each encoded input is.
function applyObsLabels(labels) {
  const el = $("#obs-labels");
  if (!el) return;
  el.textContent = Array.isArray(labels) && labels.length
    ? t("quantum.inputs", { inputs: labels.map(featureLabel).join(" · ") }) : "";
}

// -- lap board ---------------------------------------------------------------

let lapboardAt = 0;

function renderLapboard(cars, force = false) {
  const now = performance.now();
  if (!force && now - lapboardAt < 250) return;
  lapboardAt = now;
  const board = $("#lapboard");
  const rows = cars.map((car) => {
    const best = state.bestLaps.get(car.id);
    if (car.ghost) {
      // ghosts stay off the board except for a dim "Ghost (best …)" entry
      const lapT = typeof best === "number" ? best : car.last_lap_time;
      // car.label carries the record's provenance ("best 14.2s · universal")
      const text =
        typeof lapT === "number" && isFinite(lapT)
          ? t("lap.ghost_paren", {
            label: car.label ? carLabel(car.label) : t("car.ghost_best", { time: fmtLap(lapT) }),
          })
          : car.label
            ? t("lap.ghost_dash", { label: carLabel(car.label) })
            : t("lap.ghost");
      return `<div class="lap-row ghost">
        <span class="dot ghost-dot"></span>
        <span class="lap-kind">${text}</span>
      </div>`;
    }
    const evo = state.mode === "evolution" && car.label;
    const color = evo ? renderer.stageColor(car.label) : KIND_COLORS[car.kind] || "#ccc";
    const name = evo ? carLabel(car.label) : kindName(car.kind);
    return `<div class="lap-row${car.off_track ? " off" : ""}">
      <span class="dot" style="background:${color}"></span>
      <span class="lap-kind">${name}</span>
      <span class="lap-cell">${t("lap.lap")} <b>${car.lap}</b></span>
      <span class="lap-cell">${t("lap.last")} <b>${fmtLap(car.last_lap_time)}</b></span>
      <span class="lap-cell">${t("lap.best")} <b>${fmtLap(best)}</b></span>
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
        return `<div class="legend-row ghost"><span class="dot ghost-dot"></span>${carLabel(c.label)}</div>`;
      }
      const evo = state.mode === "evolution";
      const color = evo ? renderer.stageColor(c.label) : KIND_COLORS[c.kind] || "#ccc";
      const num = evo ? `<b style="color:${color}">${renderer.stageNumber(c.label)}</b> ` : "";
      return `<div class="legend-row"><span class="dot" style="background:${color}"></span>${num}${carLabel(c.label)}</div>`;
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
        `<div class="ep-line" style="color:${KIND_COLORS[agent] || "#e6e9ef"}">${t("train.ep_n", { n: ep })}</div>`,
    )
    .join("");
}

function showBestBanner(lapTime) {
  showBanner(t("banner.best_lap", { time: `${lapTime.toFixed(2)}s` }));
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
  setStatus("status.connected", "pill-ok");
  clearTimeout(disconnectTimer);
  $("#disconnect-overlay").hidden = true;
});
net.on("_close", () => {
  setStatus("status.reconnecting", "pill-off");
  clearTimeout(disconnectTimer);
  disconnectTimer = setTimeout(() => {
    $("#disconnect-overlay").hidden = false;
  }, 2500);
});

net.on("welcome", (msg) => {
  state.welcome = msg;
  if (msg.ui) {
    // the booth's default language; a visitor's own choice wins
    state.boothLang = msg.ui.language === "de" ? "de" : "en";
    if (!storedLang()) setLang(state.boothLang, { persist: false });
  }
  state.tracks = msg.tracks || [];
  const sel = $("#track-select");
  const randomOpt = document.createElement("option");
  randomOpt.value = "random";
  randomOpt.textContent = `🎲 ${t("track.random")}`;
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
  state.qubitOptions = options;
  for (const opt of $("#qubit-select").options) {
    const ok = options.includes(Number(opt.value));
    opt.disabled = !ok;
    opt.textContent = ok ? opt.value : t("qubits.no_driver", { n: opt.value });
  }
}

// Leaderboard: ranked named human laps + unranked AI reference rows.
const isOperator = () => document.body.classList.contains("operator");

net.on("leaderboard", renderLeaderboard);

function renderLeaderboard(msg) {
  state.leaderboard = msg;
  $("#board-track").textContent = trackLabel(msg.track);
  state.boardTrack = msg.track;
  const entries = $("#board-entries");
  // #operator: a ✕ per entry and the clear buttons (the server takes them
  // from the booth machine only)
  const remove = (e) => isOperator()
    ? `<button type="button" class="board-remove" data-name="${escapeHtml(e.name)}"
        data-lap="${e.lap_s}" title="${escapeHtml(t("board.remove_title"))}"
        aria-label="${escapeHtml(t("board.remove_aria", { name: e.name }))}">✕</button>`
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
      <span>${kindName(r.kind)}</span>
      <span class="board-driver">${escapeHtml(r.driver)}</span>
      <b>${r.lap_s.toFixed(2)}s</b>
    </div>`)
    .join("") || `<p class="hint">${t("board.no_refs")}</p>`;
}

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
  renderControl(msg);
});

function renderControl(msg) {
  state.control = msg;
  const pill = $("#control-pill");
  let text = null;
  if (msg.driving) {
    if (msg.turn_ends_in_s != null) {
      text = t("control.your_turn", { s: msg.turn_ends_in_s, waiting: msg.waiting });
    } else if (msg.watchers > 0) {
      text = t("control.wheel", { watchers: msg.watchers });
    }
  } else if (msg.queue_pos != null) {
    const ahead = msg.queue_pos - 1;
    text = ahead === 0
      ? t("control.next", { s: msg.turn_ends_in_s ?? "?" })
      : t("control.in_line", { ahead });
  } else if (msg.locked) {
    text = t("control.spectating");
  }
  pill.hidden = text === null;
  if (text !== null) pill.textContent = text;
}

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
    cd.textContent = t("race.go");
  } else {
    cd.hidden = true;
  }
  const me = (msg.cars || []).find((c) => c.kind === "human" && !c.ghost);
  const hud = $("#race-hud");
  hud.hidden = !me;
  if (!me) return;
  const best = state.bestLaps.get(me.id);
  hud.innerHTML = `<div class="hud-time">${fmtLap(me.lap_t ?? 0)}</div>
    <div class="hud-meta">${t("race.hud", { lap: me.lap + 1, last: fmtLap(me.last_lap_time), best: fmtLap(best) })}</div>`;
  $("#offtrack-banner").hidden = !me.off_track;
}

function showLapResult(msg) {
  const el = $("#lap-result");
  let rank = "";
  let note = "";
  if (!msg.clean) {
    note = t("result.off_track");
  } else if (msg.named && msg.rank) {
    rank = t("result.rank", { rank: msg.rank });
  } else if (msg.rank) {
    rank = t("result.rank_anon", { rank: msg.rank });
    note = t("result.name_hint");
  } else {
    note = msg.named ? t("result.not_on_board") : "";
  }
  el.innerHTML = `<div class="lr-note">${t("result.your_lap")}</div>
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
      toast(msg.agent ? t("toast.training_done_agent", { agent: kindName(msg.agent) })
        : t("toast.training_done"), false);
      break;
  }
});

// Server errors are written for operators ("bundled mlp weights use …"): a
// kiosk shows visitors a plain line and keeps the detail in the console
// (#operator shows it).
net.on("error", (msg) => {
  const detail = msg.message || t("toast.server_error");
  if (msg.field === "name") {
    $("#race-name").value = "";
    studioPanel.clearName();
  }
  if (msg.visitor) {
    // written for visitors; `key` names the same text in the string tables
    toast(msg.key && has(msg.key) ? t(msg.key) : detail);
  } else if (document.body.classList.contains("kiosk") && !isOperator()) {
    console.warn("racetraQ:", detail);
    toast(t("toast.unavailable"));
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
  // a kiosk's next visitor reads the booth's language again
  if (document.body.classList.contains("kiosk")) {
    clearStoredLang();
    setLang(state.boothLang, { persist: false });
  }
}
$("#race-reset").addEventListener("click", () => {
  state.bestLaps.clear();
  net.raceCmd("reset", $("#race-opponent").value);
});

// Camera for the human driver: button or C cycles top / chase / cockpit; the
// choice is remembered in this browser.
function setCamera(view) {
  renderer.setCamera(view);
  $("#race-camera").textContent = `📷 ${t(`camera.${renderer.camera}`)}`;
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
  $("#train-stats").innerHTML = trainStatsHtml();
}

function trainStatsHtml() {
  const rows = [...trainStats.values()].map(
    (m) => `<div class="stat-row">
      <span class="dot" style="background:${KIND_COLORS[m.agent] || "#ccc"}"></span>
      <span>${kindName(m.agent)}</span>
      <span>${t("train.stat_ep")} <b>${m.episode}</b></span>
      <span>${t("train.stat_ret")} <b>${m.mean_return.toFixed(1)}</b></span>
      <span>ε <b>${m.epsilon.toFixed(2)}</b></span>
      <span>${t("train.stat_loss")} <b>${m.loss == null ? "—" : m.loss.toFixed(3)}</b></span>
    </div>`,
  );
  return rows.join("");
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
    btn.textContent = t("board.clear");
    return;
  }
  btn.dataset.armed = "1";
  btn.textContent = t("board.clear_armed");
  setTimeout(() => {
    btn.dataset.armed = "";
    btn.textContent = t("board.clear");
  }, 3000);
});

// -- language ------------------------------------------------------------------

// EN/DE toggle in the header (kiosk too): the visitor's choice is kept in
// this browser and wins over the booth default.
function renderLangToggle() {
  for (const el of document.querySelectorAll("#lang-toggle [data-lang]")) {
    el.classList.toggle("active", el.dataset.lang === lang());
  }
}
$("#lang-toggle").addEventListener("click", (ev) => {
  setLang(lang() === "de" ? "en" : "de");
  ev.currentTarget.blur(); // keep Space/Enter for the race
});

/** Everything built from state, again in the new language. */
function rerenderAll() {
  applyDom();
  renderLangToggle();
  setStatus(...state.status);
  setCamera(renderer.camera);
  const w = state.welcome;
  if (w) {
    for (const opt of $("#track-select").options) {
      if (opt.value in TRACK_NAMES) opt.textContent = TRACK_NAMES[opt.value];
    }
    if (w.circuit_spec) {
      renderCircuit(w.circuit_spec, $("#circuit-diagram"), $("#circuit-legend"));
      renderVisibility(w.circuit_spec, w.obs_labels, $("#light-cone"));
      applyCircuitSize(w.circuit_spec, w.obs_labels);
    } else {
      initExplain($("#panel-explain"));
    }
    applyObsLabels(w.obs_labels);
    applyDrivers(w.drivers, $("#driver-select").value || w.driver);
  } else {
    initExplain($("#panel-explain"));
  }
  if (state.qubitOptions) applyQubitOptions(state.qubitOptions);
  applyTrackLabels();
  quantumPanel.relabel();
  attract.refresh();
  if (state.leaderboard) renderLeaderboard(state.leaderboard);
  if (state.control) renderControl(state.control);
  studioPanel.rerender();
  hardwarePanel.rerender();
  chart.draw();
  lapChart.draw();
  evoLegendKey = "";
  if (state.lastState) {
    renderLapboard(state.lastState.cars || [], true);
    updateCarLegend(state.lastState.cars || []);
  }
  renderEpisodeOverlay();
  if (trainStats.size) $("#train-stats").innerHTML = trainStatsHtml();
}
onLangChange(rerenderAll);

// -- boot --------------------------------------------------------------------

// #operator: a kiosk shows its operator controls (Train, Hardware, Qubits,
// Driver) again — for setting up the booth, not for visitors.
document.body.classList.toggle("operator", window.location.hash.includes("operator"));

initTooltips();
renderLangToggle();
setStatus("status.connecting", "pill-off");
applyMode("attract");
net.connect();
