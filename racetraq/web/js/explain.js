// Static explainer content for the Explain panel: sub-tabs with concise,
// accurate copy about the exhibit. The copy is templated on the circuit spec
// (welcome.circuit_spec) so q6/q8/q10 profiles state the right sizes; the
// defaults reproduce the 4-qubit text verbatim. The final sub-tab embeds the
// full repo documentation (docs.js).

import { initDocs } from "./docs.js";

const RAY_WORDS = { 3: "three", 5: "five", 7: "seven", 9: "nine" };

/** "three lidar rays and speed" / "five lidar rays, speed, curvature ahead,
 *  lateral offset, heading error and corner speed": the circuit's inputs from
 *  the welcome's obs_labels (rays are labelled "ray …°"), `n - 1` rays and
 *  speed when no labels came. Pure function. */
export function sensorList(n = 4, labels = null) {
  let rays = n - 1;
  let others = ["speed"];
  if (Array.isArray(labels) && labels.length) {
    rays = labels.filter((l) => /^ray\b/.test(String(l))).length;
    others = labels.filter((l) => !/^ray\b/.test(String(l))).map(String);
  }
  const parts = rays > 0 ? [`${RAY_WORDS[rays] || rays} lidar ray${rays === 1 ? "" : "s"}`] : [];
  parts.push(...others);
  if (parts.length === 0) return "no sensors";
  if (parts.length === 1) return parts[0];
  return `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
}

/** The light-cone paragraph of "The quantum circuit". `visibility` (actions x
 *  features, 0/1) and `needed` (layers for full visibility) come from the
 *  circuit spec: the middle sentence says whether this circuit size leaves
 *  blind spots, and is dropped when the spec carries no light-cone analysis.
 *  Pure function. */
export function lightConeHtml(n = 4, layers = 4, visibility = null, needed = null) {
  let verdict = "";
  if (Array.isArray(visibility) && visibility.length) {
    const full = visibility.every((row) => row.every(Boolean));
    verdict = full
      ? `With ${n} qubits and ${layers} layers every action can see every input.`
      : `With ${n} qubits and ${layers} layers the cones are too short: some actions cannot
      see every input${Number.isInteger(needed) ? ` (that would take ${needed} layers)` : ""}
      — the <em>Who sees what</em> grid in the Quantum tab shows which.`;
  }
  return `
      <p>Influence spreads through the CZ ring only one neighbour per layer,
      so every readout has a <strong>light cone</strong>: the inputs and
      gates close enough to reach it. Whatever lies outside cannot change
      that action's Q-value, however long the agent trains. ${verdict}
      Gates outside every light cone are drawn dimmed in the circuit
      diagram.</p>`;
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
    title: "What is this?",
    html: `
      <p><strong>racetraQ</strong> is a racing game where the driver is a
      <em>quantum circuit</em>. A tiny ${n}-qubit parameterized circuit reads the
      car's sensors — ${sensorList(n, obsLabels)} —
      and its measurement
      results decide whether to steer right, go straight, steer left, or
      brake.</p>
      <p>A classical neural network (MLP) of similar size trains on exactly the
      same game, so you can compare the two approaches head to head — or race
      against either of them yourself. Compared over many training runs, the
      classical net is the better learner here (<em>Classical vs
      quantum</em>).</p>`,
  },
  {
    id: "learn",
    title: "How does it learn?",
    html: `
      <p>Both agents learn with <strong>double DQN</strong>, a reinforcement
      learning algorithm. The agent tries actions, receives rewards (progress
      along the track, lap bonuses, penalties for going off track) and slowly
      learns a <em>Q-function</em>: an estimate of how much future reward each
      action is worth in the current situation.</p>
      <p>Exploration is <strong>ε-greedy</strong>: early in training the agent
      picks mostly random actions (ε near 1), and as ε decays it increasingly
      trusts its own Q-values. "Double" DQN means one network chooses the best
      next action while a periodically synced target network evaluates it —
      this reduces the over-optimism that plain DQN suffers from.</p>
      <p>Watch the Training tab: the learning curve shows the mean return per
      episode climbing as the agent figures out the track.</p>
      <p>Learning is not a one-way street. On the harder tracks the agent's
      greedy policy comes and goes during training — the circuit drives best
      while it is still exploring and often loses the skill again later — so
      the bundled drivers are the best <em>snapshot</em> of a training run,
      picked by greedy test episodes along the way, not the run's final
      weights.</p>`,
  },
  {
    id: "circuit",
    title: "The quantum circuit",
    html: `
      <p>The Q-function of the quantum agent is a
      <strong>variational quantum circuit</strong> with ${n} qubits and ${layers} layers.
      Each layer first <em>encodes</em> the observation with RY(λ·x) rotations,
      then applies trainable RY/RZ rotations, then entangles neighbouring
      qubits with a ring of CZ gates.</p>
      <p>Re-encoding the input in every layer is called
      <strong>data re-uploading</strong> — it lets even a small circuit
      represent rich, non-linear functions of the input.</p>
      <p>The output is read as the <strong>⟨Z⟩ expectation value</strong> of
      ${n === 4 ? "each qubit" : "each of the first four qubits"}: four numbers
      in [-1, 1], scaled to become the four Q-values.
      The gauges in the Quantum tab show them live.</p>${lightConeHtml(n, layers, visibility, minLayers)}`,
  },
  {
    id: "compare",
    title: "Classical vs quantum",
    html: `
      <p>The classical baseline is a small <strong>MLP</strong> (multi-layer
      perceptron) trained with the same double DQN algorithm, the same rewards
      and the same observations. The quantum circuit has only
      <strong>${np.total ?? 3 * layers * n + 8} trainable parameters</strong>${
        Number.isInteger(dead) && dead > 0
          ? ` (${dead} of them structurally dead — the dimmed gates in the circuit diagram)`
          : ""
      }; the MLP is kept comparably small.</p>
      <p>To be honest: the classical net wins this comparison. Measured over
      many training seeds, it learns the easy tracks in about half the
      episodes, holds on to what it learned far more reliably, and drives the
      hard tracks faster. The circuit's one point is on the gp track, where it
      reaches a (slower) lapping policy earlier — and then does not keep
      it. The quantum agent has <em>no advantage</em> here, and none is
      claimed: a circuit this small is simulated exactly on a laptop. What this
      exhibit shows is that a genuinely quantum model <em>can</em> learn a
      control task end to end, and lets you inspect every moving part while it
      does. The numbers behind this paragraph, with seeds and intervals, are in
      <em>Full documentation</em> → The science behind racetraQ.</p>`,
  },
  {
    id: "try",
    title: "Try it",
    html: `
      <ul>
        <li><strong>Watch</strong> — attract mode: trained agents drive laps on
        their own.</li>
        <li><strong>Train</strong> — start a fresh (or warm-started) training
        run and watch the learning curve grow in the Training tab.</li>
        <li><strong>Studio</strong> — build your own quantum driver: pick the
        track, the number of qubits, its sensors and actions, train it live (at
        most 5 minutes), see how it compares with our studies, then race
        it.</li>
        <li><strong>Race</strong> — drive yourself with the arrow keys or WASD
        (↑/W throttle, ↓/S brake, ←/→ steer) against the quantum or classical
        agent. A game controller works too: plug one in and the left stick
        steers while the triggers give analog throttle and brake.</li>
        <li><strong>Hardware</strong> — run the trained circuit on an IBM
        Quantum backend: either a local noisy simulation of a real device, or
        an actual quantum computer (queue times apply). Watch the hardware lap
        replay as a ghost next to a simulator car, or run a short SPSA
        training sprint directly on the backend.</li>
      </ul>
      <p>Everything else runs a fast <em>simulator</em> of the quantum circuit
      — that is standard practice for training, since today's real quantum
      hardware is too slow and too noisy for millions of training steps. The
      identical circuit executes on real quantum hardware in Hardware
      mode.</p>`,
  },
  {
    id: "docs",
    title: "Full documentation",
    html: '<div class="docs-root"></div>',
    mount: (body) => initDocs(body.querySelector(".docs-root")),
  },
];

// The open sub-tab survives a rebuild: every welcome (qubit or driver switch)
// re-templates the copy, and the reader should see the paragraph change
// rather than be sent back to the first section.
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
