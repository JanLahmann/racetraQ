// Static explainer content for the Explain panel: sub-tabs with concise,
// accurate copy about the exhibit. The copy is templated on the circuit spec
// (welcome.circuit_spec) so q6/q8/q10 profiles state the right sizes; the
// defaults reproduce the 4-qubit text verbatim. The text itself lives in the
// string tables (web/i18n/*.json, keys "explain.*"); a language switch
// rebuilds the panel. The final sub-tab embeds the full repo documentation
// (docs.js; the documents themselves are English).

import { initDocs } from "./docs.js";
import { t, featureLabel, listJoin, numWord } from "./i18n.js";

/** "three lidar rays and speed" / "five lidar rays, speed, curvature ahead,
 *  lateral offset, heading error and corner speed": the circuit's inputs from
 *  the welcome's obs_labels (rays are labelled "ray …°"), `n - 1` rays and
 *  speed when no labels came. Pure function (given the language). */
export function sensorList(n = 4, labels = null) {
  let rays = n - 1;
  let others = ["speed"];
  if (Array.isArray(labels) && labels.length) {
    rays = labels.filter((l) => /^ray\b/.test(String(l))).length;
    others = labels.filter((l) => !/^ray\b/.test(String(l))).map(String);
  }
  const parts = rays > 0
    ? [t(rays === 1 ? "explain.rays_one" : "explain.rays", { count: numWord(rays) })]
    : [];
  parts.push(...others.map(featureLabel));
  if (parts.length === 0) return t("explain.no_sensors");
  return listJoin(parts);
}

/** The light-cone paragraph of "The quantum circuit". `visibility` (actions x
 *  features, 0/1) and `needed` (layers for full visibility) come from the
 *  circuit spec: the middle sentence says whether this circuit size leaves
 *  blind spots, and is dropped when the spec carries no light-cone analysis.
 *  Pure function (given the language). */
export function lightConeHtml(n = 4, layers = 4, visibility = null, needed = null) {
  let verdict = "";
  if (Array.isArray(visibility) && visibility.length) {
    const full = visibility.every((row) => row.every(Boolean));
    verdict = full
      ? t("explain.cone.full", { n, layers })
      : t("explain.cone.short", {
        n,
        layers,
        needed: Number.isInteger(needed) ? t("explain.cone.needed", { needed }) : "",
      });
  }
  return `
      <p>${t("explain.cone.html", { verdict })}</p>`;
}

const sections = ({
  n_qubits: n = 4,
  n_layers: layers = 4,
  n_params: np = {},
  dead_params: dead,
  visibility,
  min_layers_full_visibility: minLayers,
} = {}, obsLabels = null) => [
  {
    id: "what",
    title: t("explain.what.title"),
    html: t("explain.what.html", { n, sensors: sensorList(n, obsLabels) }),
  },
  {
    id: "learn",
    title: t("explain.learn.title"),
    html: t("explain.learn.html"),
  },
  {
    id: "circuit",
    title: t("explain.circuit.title"),
    html:
      t("explain.circuit.html", {
        n,
        layers,
        readout: n === 4 ? t("explain.circuit.each_qubit") : t("explain.circuit.first_four"),
      }) + lightConeHtml(n, layers, visibility, minLayers),
  },
  {
    id: "compare",
    title: t("explain.compare.title"),
    html: t("explain.compare.html", {
      params: np.total ?? 3 * layers * n + 8,
      dead: Number.isInteger(dead) && dead > 0 ? t("explain.compare.dead", { dead }) : "",
    }),
  },
  {
    id: "try",
    title: t("explain.try.title"),
    html: t("explain.try.html"),
  },
  {
    id: "docs",
    title: t("explain.docs.title"),
    html: '<div class="docs-root"></div>',
    mount: (body) => initDocs(body.querySelector(".docs-root")),
  },
];

// The open sub-tab survives a rebuild: every welcome (qubit or driver switch)
// and every language switch re-templates the copy, and the reader should see
// the paragraph change rather than be sent back to the first section.
let activeId = null;

/** Build the explain panel (sub-tab nav + sections) inside `root`.
 *  `spec` is the welcome `circuit_spec` (optional: defaults to 4 qubits),
 *  `obsLabels` the welcome's `obs_labels` (optional: n - 1 rays and speed). */
export function initExplain(root, spec, obsLabels = null) {
  const SECTIONS = sections(spec, obsLabels);
  const nav = document.createElement("nav");
  nav.className = "explain-nav";
  const body = document.createElement("div");
  body.className = "explain-body";

  const show = (id) => {
    activeId = id;
    for (const btn of nav.querySelectorAll("button")) {
      btn.classList.toggle("active", btn.dataset.section === id);
    }
    const section = SECTIONS.find((s) => s.id === id);
    body.innerHTML = `<h2>${section.title}</h2>${section.html}`;
    if (section.mount) section.mount(body);
  };

  for (const s of SECTIONS) {
    const btn = document.createElement("button");
    btn.dataset.section = s.id;
    btn.textContent = s.title;
    btn.addEventListener("click", () => show(s.id));
    nav.append(btn);
  }

  root.replaceChildren(nav, body);
  show(SECTIONS.some((s) => s.id === activeId) ? activeId : SECTIONS[0].id);
}
