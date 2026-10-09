// Hardware tab: run the trained circuit on an IBM Quantum backend (real
// device, or a local noisy "fake" simulation of one). Sends C->S "hardware"
// commands and renders S->C "hardware_status" updates: phase pill, live
// counters (including how the job runs: execution mode, transpiled two-qubit
// gate count and depth, shots per job and the attenuation rescale when one is
// active), the note explaining an execution-mode fallback, the sprint loss
// chart and the before/after eval-return comparison.

import { hardwareCmd } from "./net.js";
import { LossChart } from "./charts.js";
import { t } from "./i18n.js";

const BUSY_PHASES = new Set(["connecting", "transpiling", "running"]);

// How the Estimator jobs are scheduled on the backend (hardware_status
// execution_mode): a Session is dedicated access, Open Plan accounts fall back
// to a Batch or to plain job mode. (Values: string-table keys.)
const MODE_LABEL = {
  session: "hw.mode.session",
  batch: "hw.mode.batch",
  job: "hw.mode.job",
};

// Attenuation rescale (hardware_status rescale): the measured expectations are
// divided by an attenuation calibrated with one extra job before the run.
const RESCALE_LABEL = {
  global: "hw.rescale.global",
  readout: "hw.rescale.readout",
};

const PHASE_CLASS = {
  idle: "pill-idle",
  connecting: "pill-busy",
  transpiling: "pill-busy",
  running: "pill-busy",
  replay: "pill-ok",
  done: "pill-ok",
  error: "pill-err",
};

export function initHardwarePanel() {
  const $ = (sel) => document.querySelector(sel);
  const els = {
    backend: $("#hw-backend"),
    shots: $("#hw-shots"),
    iterations: $("#hw-iterations"),
    lap: $("#hw-lap"),
    sprint: $("#hw-sprint"),
    abort: $("#hw-abort"),
    phase: $("#hw-phase"),
    backendName: $("#hw-backend-name"),
    message: $("#hw-message"),
    counters: $("#hw-counters"),
    eval: $("#hw-eval"),
    replayCaption: $("#hw-replay-caption"),
  };
  const lossChart = new LossChart($("#hw-loss-chart"));

  // Fallback note ("Session unavailable on ... ; using a Batch"): its own line
  // under the status message, which later statuses keep overwriting.
  const noteEl = document.createElement("div");
  noteEl.id = "hw-note";
  noteEl.className = "hint";
  noteEl.hidden = true;
  els.message.after(noteEl);

  // label key -> () => formatted value (re-read on a language switch), insertion-ordered
  const counters = new Map();
  let evalBefore = null;
  let evalAfter = null;
  let phase = "idle";

  function intVal(el, fallback) {
    const v = parseInt(el.value, 10);
    return Number.isFinite(v) ? v : fallback;
  }

  function setBusy(busy) {
    els.lap.disabled = busy;
    els.sprint.disabled = busy;
    els.abort.disabled = !busy;
  }

  function renderCounters() {
    els.counters.innerHTML = [...counters.entries()]
      .map(
        ([label, value]) =>
          `<div class="stat-row"><span>${t(label)}</span><span><b>${value()}</b></span></div>`,
      )
      .join("");
  }

  function renderEval() {
    if (evalBefore === null && evalAfter === null) {
      els.eval.innerHTML = "";
      return;
    }
    const fmt = (v) => (typeof v === "number" ? v.toFixed(1) : "…");
    let delta = "";
    if (typeof evalBefore === "number" && typeof evalAfter === "number") {
      const d = evalAfter - evalBefore;
      const cls = d >= 0 ? "eval-up" : "eval-down";
      delta = ` <span class="${cls}">(${d >= 0 ? "+" : ""}${d.toFixed(1)})</span>`;
    }
    els.eval.innerHTML = `<div class="stat-row"><span>${t("hw.eval_return")}</span>
      <span>${t("hw.eval_before_after", {
        before: `<b>${fmt(evalBefore)}</b>`, after: `<b>${fmt(evalAfter)}</b>`,
      })}${delta}</span></div>`;
  }

  function resetRun() {
    counters.clear();
    renderCounters();
    els.message.textContent = "";
    noteEl.textContent = "";
    noteEl.hidden = true;
  }

  els.lap.addEventListener("click", () => {
    resetRun();
    hardwareCmd("lap", {
      backend: els.backend.value,
      shots: intVal(els.shots, 1024),
    });
    setBusy(true); // optimistic; hardware_status confirms
  });

  els.sprint.addEventListener("click", () => {
    resetRun();
    lossChart.reset();
    evalBefore = null;
    evalAfter = null;
    renderEval();
    hardwareCmd("sprint", {
      backend: els.backend.value,
      iterations: intVal(els.iterations, 10),
      shots: intVal(els.shots, 1024),
    });
    setBusy(true);
  });

  els.abort.addEventListener("click", () => hardwareCmd("abort"));

  function renderPhase() {
    els.phase.textContent = t(`hw.phase.${phase}`);
  }

  /** Handle an S->C hardware_status message. */
  function handleStatus(msg) {
    phase = msg.phase || "idle";
    renderPhase();
    els.phase.className = `pill ${PHASE_CLASS[phase] || "pill-idle"}`;
    if (typeof msg.backend_name === "string") els.backendName.textContent = msg.backend_name;
    if (typeof msg.message === "string") els.message.textContent = msg.message;
    els.message.classList.toggle("hw-error", phase === "error");
    setBusy(BUSY_PHASES.has(phase));
    els.replayCaption.hidden = phase !== "replay";

    if (typeof msg.note === "string") {
      noteEl.textContent = msg.note;
      noteEl.hidden = msg.note === "";
    }
    if (Object.hasOwn(MODE_LABEL, msg.execution_mode)) {
      const key = MODE_LABEL[msg.execution_mode];
      counters.set("hw.counter.mode", () => t(key));
    }
    const plain = (label, text) => counters.set(label, () => text);
    if (typeof msg.two_qubit_gates === "number") {
      plain("hw.counter.two_qubit_gates", String(msg.two_qubit_gates));
    }
    if (typeof msg.circuit_depth === "number") plain("hw.counter.depth", String(msg.circuit_depth));
    if (typeof msg.shots === "number") plain("hw.counter.shots", String(msg.shots));
    if (Object.hasOwn(RESCALE_LABEL, msg.rescale)) {
      const key = RESCALE_LABEL[msg.rescale];
      const f = typeof msg.attenuation === "number" ? `, f = ${msg.attenuation.toFixed(3)}` : "";
      counters.set("hw.counter.rescale", () => `${t(key)}${f}`);
    }
    if (typeof msg.decision === "number") plain("hw.counter.decision", String(msg.decision));
    if (typeof msg.seconds_per_decision === "number") {
      plain("hw.counter.s_per_decision", msg.seconds_per_decision.toFixed(2));
    }
    if (typeof msg.iteration === "number") plain("hw.counter.iteration", String(msg.iteration));
    if (typeof msg.loss === "number") {
      plain("hw.counter.loss", msg.loss.toFixed(4));
      if (typeof msg.iteration === "number") lossChart.addPoint(msg.iteration, msg.loss);
    }
    if (typeof msg.lap_time === "number") {
      plain("hw.counter.lap", `${msg.lap_time.toFixed(2)}s`);
    }
    renderCounters();

    if (typeof msg.eval_return_before === "number") evalBefore = msg.eval_return_before;
    if (typeof msg.eval_return_after === "number") evalAfter = msg.eval_return_after;
    renderEval();
  }

  /** Re-render the labels in the active language. */
  function rerender() {
    renderPhase();
    renderCounters();
    renderEval();
    lossChart.draw();
  }

  setBusy(false);
  renderPhase();
  renderCounters();
  return { handleStatus, rerender };
}
