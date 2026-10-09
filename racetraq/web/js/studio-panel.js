// Training studio (#29): pick track, qubits, sensors and action set, train a
// quantum driver live within the time limit, compare it with the multi-seed
// studies, race it, and see the booth board. Renders the server's `studio`
// status messages; sends `studio` commands. The text comes from the string
// tables ("studio.*"); the server's catalog labels and blurbs are English and
// are shown translated by their ids (sensor preset, action count, track).

import { t, has, actionLabel } from "./i18n.js";

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const fmtS = (s) => {
  if (typeof s !== "number" || !isFinite(s)) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, "0")} s`;
};
const fmtLap = (t) => (typeof t === "number" && isFinite(t) ? `${t.toFixed(2)} s` : "—");
const TRACK_LABELS = { oval: "Oval", chicane: "Chicane", gp: "Grand Prix", combo: "Combo" };

/** A string-table entry when there is one, else the server's own text. */
const tOr = (key, fallback) => (has(key) ? t(key) : fallback);
const sensorShort = (id) => tOr(`studio.sensor_short.${id}`, id);

/** "sooner than 3 of 8 study runs" with the ends said plainly. `betterWord`
 *  and `worseWord` are already in the active language. Pure. */
export function rankPhrase(better, runs, betterWord, worseWord) {
  if (better >= runs) return t("studio.rank.all", { word: betterWord, runs });
  if (better <= 0) return t("studio.rank.none", { word: worseWord, runs });
  return t("studio.rank.some", { word: betterWord, count: better, runs });
}

// circuit depth per size: the packaged q<n> profiles (racetraq/config)
const LAYERS = { 4: 4, 6: 4, 8: 5, 10: 6 };

/** Trainable numbers of the canonical circuit: 3·L·n + 2·A. Pure. */
export function circuitParams(n, actions) {
  const layers = LAYERS[n] || 4;
  return 3 * layers * n + 2 * actions;
}

/** One line on what the studies say for a (track, qubits) combo; the time
 *  estimate says where its speed was measured (this machine after its first
 *  studio run at that size, else the reference laptop). Pure (given the
 *  language). */
export function studyLine(combo, limitS, machine = null) {
  if (!combo || !combo.study) return t("studio.study.none");
  const st = combo.study;
  const parts = [];
  if (st.first_lap) {
    parts.push(t("studio.study.first_lap", {
      median: Math.round(st.first_lap.median), lapped: st.lapped_runs, runs: st.runs,
    }));
  } else {
    parts.push(t("studio.study.no_lap", { runs: st.runs }));
  }
  if (typeof combo.estimate_s === "number") {
    const where = combo.estimate_here
      ? t("studio.study.here")
      : t("studio.study.reference", { machine: machine ? ` (${machine})` : "" });
    parts.push(`≈ ${fmtS(combo.estimate_s)} ${where}`);
  }
  let line = t("studio.study.line", { parts: parts.join(" · ") });
  if (combo.fits === false) {
    line += t("studio.study.too_long", {
      limit: fmtS(limitS), warm: combo.warm ? t("studio.study.try_warm") : "",
    });
  }
  return line;
}

/** A small bar chart of the in-training tests: drives lapped (of 12) per
 *  test, oldest left. `tests` is [[episode, lapped, of], ...]. Pure. */
export function testsSvg(tests) {
  if (!tests.length) return "";
  const w = 280;
  const h = 56;
  const n = Math.max(tests.length, 12);
  const bw = w / n;
  const bars = tests
    .map(([ep, lapped, of], i) => {
      const bh = Math.max(2, (lapped / Math.max(1, of)) * (h - 4));
      return `<rect x="${(i * bw + 1).toFixed(1)}" y="${(h - bh).toFixed(1)}" width="${Math.max(1, bw - 2).toFixed(1)}"
        height="${bh.toFixed(1)}" rx="1.5" class="${lapped > 0 ? "t-lap" : "t-none"}"><title>${esc(t("studio.tests.bar", { episode: ep, lapped, of }))}</title></rect>`;
    })
    .join("");
  return `<svg class="studio-tests" viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img"
    aria-label="${esc(t("studio.tests.aria"))}">${bars}</svg>`;
}

export function initStudioPanel({ root, send, setName, onPhase, onStart, moderate, stage,
                                  onFirstLap, onTrack, laps, active }) {
  const DEFAULT_SEL = { track: null, qubits: 4, sensors: "lidar", actions: 4, warm: false };
  const sel = { ...DEFAULT_SEL };
  let catalog = null;
  let last = null;
  let phase = "setup";
  let tests = []; // in-training tests of the current run: [episode, lapped, of]
  let firstLapSeen = null;

  const comboFor = () => catalog && catalog.combos[`${sel.track}_q${sel.qubits}`];

  function chip(group, value, label, { active, disabled, title } = {}) {
    return `<button type="button" class="chip${active ? " active" : ""}" data-group="${group}"
      data-value="${esc(value)}"${disabled ? " disabled" : ""}${title ? ` title="${esc(title)}"` : ""}>${esc(label)}</button>`;
  }

  function setupHtml() {
    if (!catalog) return `<p class="hint">${t("studio.loading_setup")}</p>`;
    if (!sel.track) sel.track = catalog.tracks[0].id;
    if (sel.actions > sel.qubits) sel.actions = 4;
    const combo = comboFor();
    const studied = sel.sensors === "lidar" && sel.actions === 4;
    const warmOk = Boolean(combo && combo.warm) && studied;
    if (!warmOk) sel.warm = false;
    const track = catalog.tracks.find((tr) => tr.id === sel.track);
    const sensor = catalog.sensors.find((s) => s.id === sel.sensors);
    const action = catalog.actions.find((a) => a.n === sel.actions);
    return `
      <p class="hint">${t("studio.intro", { limit: fmtS(catalog.time_limit_s) })}</p>
      <label class="studio-name"><span>${t("studio.your_name")}</span>
        <input type="text" id="studio-name" maxlength="24" placeholder="${esc(t("studio.name_placeholder"))}"
               value="${esc((last && last.name) || "")}"></label>
      <div class="studio-step"><h3>${t("studio.step.track")}</h3>
        <div class="chips">${catalog.tracks.map((tr) => chip("track", tr.id, TRACK_LABELS[tr.id] || tr.id, { active: tr.id === sel.track })).join("")}</div>
        <p class="hint">${esc(track ? tOr(`studio.track_note.${track.id}`, track.note) : "")}</p></div>
      <div class="studio-step"><h3>${t("studio.step.qubits")}</h3>
        <div class="chips">${catalog.qubits.map((n) => chip("qubits", n, `${n}`, { active: n === sel.qubits })).join("")}</div>
        <p class="hint">${esc(studyLine(combo, catalog.time_limit_s, catalog.speed_machine))}</p></div>
      <div class="studio-step"><h3>${t("studio.step.sensors")}</h3>
        <div class="chips">${catalog.sensors.map((s) => chip("sensors", s.id, tOr(`studio.sensor.${s.id}.label`, s.label), { active: s.id === sel.sensors })).join("")}</div>
        <p class="hint">${esc(t("studio.sensor_hint", {
          count: sel.qubits - 1,
          blurb: sensor ? tOr(`studio.sensor.${sensor.id}.blurb`, sensor.blurb) : "",
        }))}</p></div>
      <div class="studio-step"><h3>${t("studio.step.actions")}</h3>
        <div class="chips">${catalog.actions
          .map((a) =>
            chip("actions", a.n, `${a.n}`, {
              active: a.n === sel.actions,
              disabled: a.n > sel.qubits,
              title: a.n > sel.qubits ? t("studio.actions_need", { n: a.n }) : a.labels.map(actionLabel).join(", "),
            }),
          )
          .join("")}</div>
        <p class="hint">${esc(action ? tOr(`studio.actions_blurb.${action.n}`, action.blurb) : "")}</p></div>
      <div class="studio-step"><h3>${t("studio.step.warm")}</h3>
        <label class="check"><input type="checkbox" id="studio-warm"${sel.warm ? " checked" : ""}${warmOk ? "" : " disabled"}>
        <span>${t("studio.warm")}</span></label>
        ${warmOk ? "" : `<p class="hint">${t("studio.warm_unavailable")}</p>`}</div>
      <div class="studio-summary">${t("studio.summary", {
        size: `<b>${t("studio.size", { n: sel.qubits, layers: LAYERS[sel.qubits] || 4 })}</b>`,
        params: `<b>${circuitParams(sel.qubits, sel.actions)}</b>`,
      })}${studied && !sel.warm ? "" : ` · <i>${t("studio.experiment_note")}</i>`}</div>
      <button type="button" class="accent-btn" id="studio-start">${t("studio.start")}</button>`;
  }

  const testText = (ev) =>
    t("studio.test_text", { lapped: ev.lapped_episodes, of: ev.eval_episodes }) +
    (ev.mean_lap ? t("studio.test_mean", { lap: fmtLap(ev.mean_lap) }) : "");

  function liveHtml(msg) {
    const spec = msg.spec || {};
    const live = msg.live || {};
    const elapsed = msg.elapsed_s || 0;
    const frac = Math.min(1, elapsed / (msg.time_limit_s || 1));
    const combo = catalog && catalog.combos[`${spec.track}_q${spec.qubits}`];
    const median = combo && combo.study && combo.study.first_lap ? Math.round(combo.study.first_lap.median) : null;
    const ev = live.last_eval;
    return `
      <h3>${t("studio.live.heading", {
        track: esc(spec.track), n: spec.qubits, sensors: esc(sensorShort(spec.sensors)), actions: spec.actions,
      })}${spec.warm ? ` · ${t("studio.warm_short")}` : ""}</h3>
      <div class="studio-timer"><div class="studio-timer-fill" style="width:${(frac * 100).toFixed(1)}%"></div></div>
      <p class="studio-big">${fmtS(elapsed)} <span class="hint">${t("studio.live.of", { limit: fmtS(msg.time_limit_s) })}</span></p>
      <dl class="studio-facts">
        <dt>${t("studio.fact.episodes")}</dt><dd>${live.episode ?? 0}</dd>
        <dt>${t("studio.fact.first_lap")}</dt><dd>${live.first_lap ? t("studio.episode_n", { n: live.first_lap }) : t("studio.not_yet")}${median ? ` <span class="hint">${t("studio.live.studies_median", { n: median })}</span>` : ""}</dd>
        <dt>${t("studio.fact.latest_test")}</dt><dd>${ev ? testText(ev) : t("studio.live.after_50")}</dd>
        <dt>${t("studio.fact.best_test")}</dt><dd>${live.best_test ? testText(live.best_test) : "—"}</dd>
      </dl>
      ${tests.length ? `<h3>${t("studio.tests.heading")}</h3>${testsSvg(tests)}
        <p class="hint">${t("studio.tests.hint")}</p>` : ""}
      <p class="hint">${t("studio.live.how")}</p>
      <button type="button" id="studio-stop">${t("studio.stop")}</button>`;
  }

  function resultHtml(msg) {
    const spec = msg.spec || {};
    const r = msg.result || {};
    const best = r.best_eval;
    const cmp = r.comparison;
    const lapped = best && best.lapped_episodes > 0;
    const reason = (key) => tOr(`studio.reason.${key}`, key);
    let compare = "";
    if (cmp) {
      const bits = [];
      if (cmp.first_lap_runs) {
        bits.push(t("studio.cmp.first_lap", {
          rank: rankPhrase(cmp.first_lap_faster_than, cmp.first_lap_runs,
            t("studio.word.sooner"), t("studio.word.later")),
        }));
      }
      if (cmp.mean_lap_runs) {
        bits.push(t("studio.cmp.mean_lap", {
          rank: rankPhrase(cmp.mean_lap_faster_than, cmp.mean_lap_runs,
            t("studio.word.faster"), t("studio.word.slower")),
        }));
      }
      const st = cmp.study;
      const median = st.first_lap
        ? t("studio.cmp.median", {
          n: Math.round(st.first_lap.median),
          best: st.best_mean_lap ? t("studio.cmp.median_best", { lap: fmtLap(st.best_mean_lap) }) : "",
        })
        : "";
      if (bits.length) {
        compare = `<p>${t("studio.cmp.line", { bits: bits.join(t("studio.cmp.and")), median })}
          <span class="hint">${t("studio.cmp.source", { runs: st.runs, episodes: st.episodes, source: esc(st.source) })}</span></p>`;
      }
    } else if (!spec.studied) {
      compare = `<p class="hint">${t("studio.cmp.experiment")}</p>`;
    }
    const rank = msg.rank
      ? `<p class="studio-rank">${t("studio.rank_on_board", { rank: msg.rank, track: esc(spec.track) })}</p>` : "";
    const named = msg.name ? "" : lapped ? `<p class="hint">${t("studio.name_for_board")}</p>` : "";
    return `
      <h3>${t("studio.result.heading", { track: esc(spec.track), n: spec.qubits, params: spec.n_params })}</h3>
      <dl class="studio-facts">
        <dt>${t("studio.fact.trained")}</dt><dd>${t("studio.result.trained", { episodes: r.episodes, time: fmtS(r.seconds) })} <span class="hint">(${esc(reason(r.stop_reason))})</span></dd>
        <dt>${t("studio.fact.first_lap")}</dt><dd>${r.first_lap ? t("studio.episode_n", { n: r.first_lap }) : t("studio.none")}</dd>
        <dt>${t("studio.fact.best_test")}</dt><dd>${best ? testText(best) : "—"}</dd>
      </dl>
      ${r.error ? `<p class="hint">${t("studio.error", { error: esc(r.error) })}</p>` : ""}
      ${compare}${rank}${named}
      ${lapped ? "" : `<p class="hint">${t("studio.no_lap_tip")}</p>`}
      <div class="btn-row">
        <button type="button" class="accent-btn" data-cmd="race"${r.error ? " disabled" : ""}>${t("studio.btn.race")}</button>
        <button type="button" data-cmd="watch"${r.error ? " disabled" : ""}>${t("studio.btn.watch")}</button>
        <button type="button" data-cmd="setup">${t("studio.btn.again")}</button>
      </div>`;
  }

  function drivingHtml(msg) {
    const race = msg.phase === "race";
    const best = laps ? laps() : {};
    const duel = race
      ? `<dl class="studio-facts">
          <dt>${t("studio.duel.you")}</dt><dd>${fmtLap(best.human)}</dd>
          <dt>${t("studio.duel.circuit")}</dt><dd>${fmtLap(best.model)}</dd>
        </dl>${typeof best.human === "number" && typeof best.model === "number"
          ? `<p class="studio-rank">${best.human < best.model ? t("studio.duel.won") : t("studio.duel.behind")}</p>` : ""}`
      : "";
    return `
      <h3>${race ? t("studio.drive.race_heading") : t("studio.drive.watch_heading")}</h3>${duel}
      <p class="hint">${race ? t("studio.drive.race_hint") : t("studio.drive.watch_hint")}</p>
      <div class="btn-row">
        <button type="button" data-cmd="${race ? "watch" : "race"}">${race ? t("studio.btn.watch") : t("studio.btn.race_it")}</button>
        <button type="button" data-cmd="result">${t("studio.btn.result")}</button>
      </div>`;
  }

  // #operator: a ✕ per studio board entry (the server takes it from the booth machine only)
  const operator = () => document.body.classList.contains("operator");

  function boardHtml(board) {
    if (!board) return "";
    const rows = (board.entries || [])
      .map(
        (e, i) => `<tr><td>${i + 1}</td><td>${esc(e.name)}${operator()
          ? ` <button type="button" class="board-remove" data-studio-remove="${i}"
              data-name="${esc(e.name)}" aria-label="${esc(t("board.remove_aria", { name: e.name }))}">✕</button>` : ""}</td>
          <td>${e.qubits}q · ${esc(sensorShort(e.sensors))} · ${e.actions}a${e.warm ? ` · ${t("studio.warm_short")}` : ""}</td>
          <td>${e.lapped}/${e.eval_episodes}</td><td>${fmtLap(e.mean_lap)}</td></tr>`,
      )
      .join("");
    return `<h2>${t("studio.board.heading", { track: esc(board.track) })}</h2>
      ${rows ? `<table class="studio-board"><thead><tr><th>#</th><th>${t("studio.board.name")}</th><th>${t("studio.board.circuit")}</th><th>${t("studio.board.test")}</th><th>${t("studio.board.mean_lap")}</th></tr></thead><tbody>${rows}</tbody></table>`
        : `<p class="hint">${t("studio.board.empty")}</p>`}
      <p class="hint">${t("studio.board.hint")}</p>`;
  }

  /** The stage while the studio is open: the setup's summary with a big
   *  Start (the sidebar form can be long), or a training ticker. */
  function renderStage() {
    if (!stage) return;
    if (active && !active()) {
      stage.hidden = true; // e.g. an idle reset's status arrives in Watch
      return;
    }
    const spec = (last && last.spec) || {};
    if (phase === "setup" && catalog) {
      const track = TRACK_LABELS[sel.track] || sel.track;
      stage.innerHTML = `<div class="studio-stage-card">
        <div class="eyebrow">${t("studio.title")}</div>
        <div class="headline">${t("studio.stage.headline", {
          track: esc(track), n: sel.qubits, params: circuitParams(sel.qubits, sel.actions),
        })}</div>
        <p>${t("studio.stage.pick", { limit: fmtS(catalog.time_limit_s) })}</p>
        <button type="button" class="accent-btn" data-stage-start="1">${t("studio.start")}</button>
      </div>`;
      stage.hidden = false;
    } else if (phase === "training" && last) {
      const live = last.live || {};
      stage.innerHTML = `<div class="studio-stage-ticker">
        <span class="big">${fmtS(last.elapsed_s || 0)}</span>
        <span>${t("studio.stage.episode", { n: live.episode ?? 0 })}</span>
        <span>${live.first_lap ? t("studio.stage.first_lap", { n: live.first_lap }) : t("studio.stage.no_lap")}</span>
        <span>${esc(TRACK_LABELS[spec.track] || spec.track || "")} · ${t("studio.stage.qubits", { n: spec.qubits ?? "" })}</span>
      </div>`;
      stage.hidden = false;
    } else {
      stage.hidden = true;
    }
  }

  function render() {
    renderStage();
    if (!last) {
      root.innerHTML = `<h2>${t("studio.title")}</h2><p class="hint">${t("studio.loading")}</p>`;
      return;
    }
    let body;
    if (phase === "training") body = liveHtml(last);
    else if (phase === "done") body = resultHtml(last);
    else if (phase === "race" || phase === "watch") body = drivingHtml(last);
    else body = setupHtml();
    root.innerHTML = `<h2>${t("studio.title")}</h2>${body}${boardHtml(last.board)}`;
  }

  root.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn || btn.disabled) return;
    if (btn.dataset.group) {
      const v = btn.dataset.value;
      sel[btn.dataset.group] = btn.dataset.group === "track" || btn.dataset.group === "sensors" ? v : Number(v);
      if (btn.dataset.group === "track" && onTrack) onTrack(v); // show it on the stage
      render();
    } else if (btn.id === "studio-start") {
      start();
    } else if (btn.dataset.studioRemove !== undefined) {
      if (moderate && last && last.board) {
        moderate({ track: last.board.track, name: btn.dataset.name,
          index: Number(btn.dataset.studioRemove) });
      }
    } else if (btn.id === "studio-stop") {
      send("stop");
    } else if (btn.dataset.cmd) {
      send(btn.dataset.cmd);
    }
  });
  function start() {
    const name = root.querySelector("#studio-name");
    setName(name ? name.value.trim() : "");
    tests = [];
    firstLapSeen = null;
    onStart && onStart();
    send("start", { ...sel });
  }
  if (stage) {
    stage.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-stage-start]")) start();
    });
  }

  root.addEventListener("change", (ev) => {
    if (ev.target.id === "studio-warm") sel.warm = ev.target.checked;
  });

  render();
  return {
    get phase() {
      return phase;
    },
    /** Show or hide the stage card after a mode switch. */
    syncStage() {
      renderStage();
    },
    /** Re-render (a lap in "you vs your circuit" changed the best laps). */
    refresh() {
      if (phase === "race") render();
    },
    /** Re-render in the active language, keeping a half-typed name. */
    rerender() {
      const name = root.querySelector("#studio-name");
      const typed = name ? name.value : null;
      render();
      const again = root.querySelector("#studio-name");
      if (again && typed !== null) again.value = typed;
    },
    /** The server refused the name: empty the studio's name field. */
    clearName() {
      const name = root.querySelector("#studio-name");
      if (name) name.value = "";
      if (last) last = { ...last, name: null };
    },
    /** Booth idle: the next visitor gets the default choices and no name. */
    reset() {
      Object.assign(sel, DEFAULT_SEL);
      if (last) last = { ...last, name: null };
      const name = root.querySelector("#studio-name");
      if (name) name.value = "";
      render();
    },
    setTrack(name) {
      // before the catalog arrives (the welcome comes first) the studio's
      // own tracks are the ones with a label
      const known = catalog ? catalog.tracks.some((tr) => tr.id === name) : name in TRACK_LABELS;
      if (known && phase === "setup") {
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
      const live = msg.live || {};
      if (phase === "training") {
        const ev = live.last_eval;
        if (ev && typeof ev.episode === "number" && !tests.some((x) => x[0] === ev.episode)) {
          tests.push([ev.episode, ev.lapped_episodes, ev.eval_episodes]);
        }
        if (live.first_lap && firstLapSeen === null) {
          firstLapSeen = live.first_lap;
          if (onFirstLap) onFirstLap(live.first_lap);
        }
      }
      // keep the name field the visitor is typing in: only re-render the
      // setup form when something about it changed
      const typing = document.activeElement && document.activeElement.id === "studio-name";
      if (!(phase === "setup" && typing && !changed)) render();
      if (changed && onPhase) onPhase(phase);
    },
  };
}
