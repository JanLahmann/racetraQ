// Training studio (#29): pick track, qubits, sensors and action set, train a
// quantum driver live within the time limit, compare it with the multi-seed
// studies, race it, and see the booth board. Renders the server's `studio`
// status messages; sends `studio` commands.

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const fmtS = (s) => {
  if (typeof s !== "number" || !isFinite(s)) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, "0")} s`;
};
const fmtLap = (t) => (typeof t === "number" && isFinite(t) ? `${t.toFixed(2)} s` : "—");
const SENSOR_SHORT = { lidar: "lidar", corner: "corner" };

/** "sooner than 3 of 8 study runs" with the ends said plainly. Pure. */
export function rankPhrase(better, runs, betterWord, worseWord) {
  if (better >= runs) return `${betterWord} than every one of the ${runs} study runs`;
  if (better <= 0) return `${worseWord} than all ${runs} study runs`;
  return `${betterWord} than ${better} of ${runs} study runs`;
}

// circuit depth per size: the packaged q<n> profiles (racetraq/config)
const LAYERS = { 4: 4, 6: 4, 8: 5, 10: 6 };

/** Trainable numbers of the canonical circuit: 3·L·n + 2·A. Pure. */
export function circuitParams(n, actions) {
  const layers = LAYERS[n] || 4;
  return 3 * layers * n + 2 * actions;
}

/** One line on what the studies say for a (track, qubits) combo. Pure. */
export function studyLine(combo, limitS) {
  if (!combo || !combo.study) return "No study runs for this combination — an experiment.";
  const st = combo.study;
  const parts = [];
  if (st.first_lap) {
    parts.push(
      `first lap after ~${Math.round(st.first_lap.median)} episodes` +
        ` (${st.lapped_runs}/${st.runs} study runs lapped)`,
    );
  } else {
    parts.push(`no study run lapped (${st.runs} runs)`);
  }
  if (typeof combo.estimate_s === "number") {
    parts.push(`≈ ${fmtS(combo.estimate_s)} here`);
  }
  let line = `Studies: ${parts.join(" · ")}.`;
  if (combo.fits === false) {
    line += ` ⚠ Longer than the ${fmtS(limitS)} limit — expect no lap${combo.warm ? ", or try a warm start" : ""}.`;
  }
  return line;
}

export function initStudioPanel({ root, send, setName, onPhase, onStart }) {
  const sel = { track: null, qubits: 4, sensors: "lidar", actions: 4, warm: false };
  let catalog = null;
  let last = null;
  let phase = "setup";

  const comboFor = () => catalog && catalog.combos[`${sel.track}_q${sel.qubits}`];

  function chip(group, value, label, { active, disabled, title } = {}) {
    return `<button type="button" class="chip${active ? " active" : ""}" data-group="${group}"
      data-value="${esc(value)}"${disabled ? " disabled" : ""}${title ? ` title="${esc(title)}"` : ""}>${esc(label)}</button>`;
  }

  function setupHtml() {
    if (!catalog) return '<p class="hint">Loading the studio…</p>';
    if (!sel.track) sel.track = catalog.tracks[0].id;
    if (sel.actions > sel.qubits) sel.actions = 4;
    const combo = comboFor();
    const studied = sel.sensors === "lidar" && sel.actions === 4;
    const warmOk = Boolean(combo && combo.warm) && studied;
    if (!warmOk) sel.warm = false;
    const track = catalog.tracks.find((t) => t.id === sel.track);
    const sensor = catalog.sensors.find((s) => s.id === sel.sensors);
    const action = catalog.actions.find((a) => a.n === sel.actions);
    return `
      <p class="hint">Build a quantum driver and train it live — at most
      ${fmtS(catalog.time_limit_s)}. The best version found during training becomes yours.</p>
      <label class="studio-name"><span>Your name</span>
        <input type="text" id="studio-name" maxlength="24" placeholder="for the booth board"
               value="${esc((last && last.name) || "")}"></label>
      <div class="studio-step"><h3>1 · Track</h3>
        <div class="chips">${catalog.tracks.map((t) => chip("track", t.id, t.id, { active: t.id === sel.track })).join("")}</div>
        <p class="hint">${esc(track ? track.note : "")}</p></div>
      <div class="studio-step"><h3>2 · Qubits</h3>
        <div class="chips">${catalog.qubits.map((n) => chip("qubits", n, `${n}`, { active: n === sel.qubits })).join("")}</div>
        <p class="hint">${esc(studyLine(combo, catalog.time_limit_s))}</p></div>
      <div class="studio-step"><h3>3 · Sensors</h3>
        <div class="chips">${catalog.sensors.map((s) => chip("sensors", s.id, s.label, { active: s.id === sel.sensors })).join("")}</div>
        <p class="hint">${sel.qubits - 1} ${esc(sensor ? sensor.blurb : "")} — one sensor per qubit.</p></div>
      <div class="studio-step"><h3>4 · Actions</h3>
        <div class="chips">${catalog.actions
          .map((a) =>
            chip("actions", a.n, `${a.n}`, {
              active: a.n === sel.actions,
              disabled: a.n > sel.qubits,
              title: a.n > sel.qubits ? `needs at least ${a.n} qubits (one readout each)` : a.labels.join(", "),
            }),
          )
          .join("")}</div>
        <p class="hint">${esc(action ? action.blurb : "")}</p></div>
      <div class="studio-step"><h3>5 · Head start</h3>
        <label class="check"><input type="checkbox" id="studio-warm"${sel.warm ? " checked" : ""}${warmOk ? "" : " disabled"}>
        <span>Warm start from an early checkpoint of the bundled driver's training (one that does not lap yet)</span></label>
        ${warmOk ? "" : '<p class="hint">Available with lidar sensors, 4 actions and a bundled checkpoint for this track and size.</p>'}</div>
      <div class="studio-summary">Your circuit: <b>${sel.qubits} qubits × ${LAYERS[sel.qubits] || 4} blocks</b>,
        <b>${circuitParams(sel.qubits, sel.actions)}</b> trainable numbers${studied && !sel.warm ? "" : " · <i>experiment: no study to compare with</i>"}</div>
      <button type="button" class="accent-btn" id="studio-start">Start training</button>`;
  }

  const testText = (ev) =>
    `${ev.lapped_episodes}/${ev.eval_episodes} drives lapped${ev.mean_lap ? `, mean lap ${fmtLap(ev.mean_lap)}` : ""}`;

  function liveHtml(msg) {
    const spec = msg.spec || {};
    const live = msg.live || {};
    const elapsed = msg.elapsed_s || 0;
    const frac = Math.min(1, elapsed / (msg.time_limit_s || 1));
    const combo = catalog && catalog.combos[`${spec.track}_q${spec.qubits}`];
    const median = combo && combo.study && combo.study.first_lap ? Math.round(combo.study.first_lap.median) : null;
    const ev = live.last_eval;
    return `
      <h3>Training · ${esc(spec.track)} · ${spec.qubits} qubits · ${SENSOR_SHORT[spec.sensors] || spec.sensors} · ${spec.actions} actions${spec.warm ? " · warm" : ""}</h3>
      <div class="studio-timer"><div class="studio-timer-fill" style="width:${(frac * 100).toFixed(1)}%"></div></div>
      <p class="studio-big">${fmtS(elapsed)} <span class="hint">of ${fmtS(msg.time_limit_s)}</span></p>
      <dl class="studio-facts">
        <dt>Episodes</dt><dd>${live.episode ?? 0}</dd>
        <dt>First lap</dt><dd>${live.first_lap ? `episode ${live.first_lap}` : "not yet"}${median ? ` <span class="hint">(studies: ~${median})</span>` : ""}</dd>
        <dt>Latest test</dt><dd>${ev ? testText(ev) : "after 50 episodes"}</dd>
        <dt>Best test</dt><dd>${live.best_test ? testText(live.best_test) : "—"}</dd>
      </dl>
      <p class="hint">Eight cars practise at once; every 50 episodes the current circuit takes a test
      (12 drives without exploration) and the best test so far is kept. Training ends at the time
      limit — or sooner, once every test drive laps and six more tests bring no improvement.
      Learning curve: Training tab.</p>
      <button type="button" id="studio-stop">Stop now — keep the best so far</button>`;
  }

  function resultHtml(msg) {
    const spec = msg.spec || {};
    const r = msg.result || {};
    const best = r.best_eval;
    const cmp = r.comparison;
    const lapped = best && best.lapped_episodes > 0;
    const reasons = {
      time: "time limit reached",
      converged: "every test drive lapped, no further improvement",
      budget: "episode budget done",
      stopped: "stopped early",
      error: "training failed",
    };
    let compare = "";
    if (cmp) {
      const bits = [];
      if (cmp.first_lap_runs) {
        bits.push(`your first lap came ${rankPhrase(cmp.first_lap_faster_than, cmp.first_lap_runs, "sooner", "later")}`);
      }
      if (cmp.mean_lap_runs) {
        bits.push(`your best test's mean lap is ${rankPhrase(cmp.mean_lap_faster_than, cmp.mean_lap_runs, "faster", "slower")}`);
      }
      const st = cmp.study;
      const median = st.first_lap ? ` (their median first lap: episode ${Math.round(st.first_lap.median)}${st.best_mean_lap ? `; median best-test lap ${fmtLap(st.best_mean_lap)}` : ""})` : "";
      if (bits.length) compare = `<p>Next to the studies: ${bits.join(", and ")}${median}. <span class="hint">Source: ${st.runs} study runs of ${st.episodes} episodes each (${esc(st.source)}).</span></p>`;
    } else if (!spec.studied) {
      compare = '<p class="hint">An experiment: no study runs to compare with.</p>';
    }
    const rank = msg.rank ? `<p class="studio-rank">#${msg.rank} on the ${esc(spec.track)} board</p>` : "";
    const named = msg.name ? "" : lapped ? '<p class="hint">Enter a name before training to make the board.</p>' : "";
    return `
      <h3>Your circuit · ${esc(spec.track)} · ${spec.qubits} qubits · ${spec.n_params} trainable numbers</h3>
      <dl class="studio-facts">
        <dt>Trained</dt><dd>${r.episodes} episodes in ${fmtS(r.seconds)} <span class="hint">(${reasons[r.stop_reason] || r.stop_reason})</span></dd>
        <dt>First lap</dt><dd>${r.first_lap ? `episode ${r.first_lap}` : "none"}</dd>
        <dt>Best test</dt><dd>${best ? `${best.lapped_episodes}/${best.eval_episodes} drives lapped${best.mean_lap ? `, mean lap ${fmtLap(best.mean_lap)}` : ""}` : "—"}</dd>
      </dl>
      ${r.error ? `<p class="hint">Error: ${esc(r.error)}</p>` : ""}
      ${compare}${rank}${named}
      ${lapped ? "" : '<p class="hint">No lap yet — try fewer qubits, an easier track, or a warm start.</p>'}
      <div class="btn-row">
        <button type="button" class="accent-btn" data-cmd="race"${r.error ? " disabled" : ""}>Race your model</button>
        <button type="button" data-cmd="watch"${r.error ? " disabled" : ""}>Watch it drive</button>
        <button type="button" data-cmd="setup">Train another</button>
      </div>`;
  }

  function drivingHtml(msg) {
    const race = msg.phase === "race";
    return `
      <h3>${race ? "You vs your circuit" : "Your circuit drives"}</h3>
      <p class="hint">${race ? "Arrow keys / WASD or a gamepad. Your circuit decides 10 times a second." : "The live qubit readout of your circuit is in the Quantum tab."}</p>
      <div class="btn-row">
        <button type="button" data-cmd="${race ? "watch" : "race"}">${race ? "Watch it drive" : "Race it"}</button>
        <button type="button" data-cmd="result">Back to the result</button>
      </div>`;
  }

  function boardHtml(board) {
    if (!board) return "";
    const rows = (board.entries || [])
      .map(
        (e, i) => `<tr><td>${i + 1}</td><td>${esc(e.name)}</td>
          <td>${e.qubits}q · ${SENSOR_SHORT[e.sensors] || e.sensors} · ${e.actions}a${e.warm ? " · warm" : ""}</td>
          <td>${e.lapped}/${e.eval_episodes}</td><td>${fmtLap(e.mean_lap)}</td></tr>`,
      )
      .join("");
    return `<h2>Studio board — ${esc(board.track)}</h2>
      ${rows ? `<table class="studio-board"><thead><tr><th>#</th><th>Name</th><th>Circuit</th><th>Test</th><th>Mean lap</th></tr></thead><tbody>${rows}</tbody></table>`
        : '<p class="hint">No studio drivers yet — be the first.</p>'}
      <p class="hint">Ranked by the best test: share of the 12 test drives that lapped, then mean lap time.</p>`;
  }

  function render() {
    if (!last) {
      root.innerHTML = '<h2>Training studio</h2><p class="hint">Loading…</p>';
      return;
    }
    let body;
    if (phase === "training") body = liveHtml(last);
    else if (phase === "done") body = resultHtml(last);
    else if (phase === "race" || phase === "watch") body = drivingHtml(last);
    else body = setupHtml();
    root.innerHTML = `<h2>Training studio</h2>${body}${boardHtml(last.board)}`;
  }

  root.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn || btn.disabled) return;
    if (btn.dataset.group) {
      const v = btn.dataset.value;
      sel[btn.dataset.group] = btn.dataset.group === "track" || btn.dataset.group === "sensors" ? v : Number(v);
      render();
    } else if (btn.id === "studio-start") {
      const name = root.querySelector("#studio-name");
      setName(name ? name.value.trim() : "");
      onStart && onStart();
      send("start", { ...sel });
    } else if (btn.id === "studio-stop") {
      send("stop");
    } else if (btn.dataset.cmd) {
      send(btn.dataset.cmd);
    }
  });
  root.addEventListener("change", (ev) => {
    if (ev.target.id === "studio-warm") sel.warm = ev.target.checked;
  });

  render();
  return {
    get phase() {
      return phase;
    },
    setTrack(name) {
      if (catalog && catalog.tracks.some((t) => t.id === name) && phase === "setup") {
        sel.track = name;
        render();
      }
    },
    handleStatus(msg) {
      if (msg.catalog) catalog = msg.catalog;
      if (msg.spec && msg.phase === "training" && !sel.track) sel.track = msg.spec.track;
      const changed = msg.phase !== phase;
      phase = msg.phase;
      last = msg;
      // keep the name field the visitor is typing in: only re-render the
      // setup form when something about it changed
      const typing = document.activeElement && document.activeElement.id === "studio-name";
      if (!(phase === "setup" && typing && !changed)) render();
      if (changed && onPhase) onPhase(phase);
    },
  };
}
