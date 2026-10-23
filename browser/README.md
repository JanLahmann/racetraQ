# racetraQ — browser edition

**Live: https://racetraq.org/**

The trained racetraQ drivers racing entirely in the browser: no Python, no
server, no account. A static site you can host anywhere (GitHub Pages,
qamposer.org, a Raspberry Pi at a workshop) or open from a USB stick through
any local web server.

**Inference only** for the race: no driver is trained here (the one exception
is a labelled toy, below). Every 0.1 s the quantum driver
turns its sensor readings into rotation angles, the circuit runs on
[QAMPoser](https://qamposer.org)'s in-browser state-vector simulator
(`@qamposer/react`), and the car takes the action with the highest
Q = w·⟨Z⟩ + b. Simulation is exact and noiseless; there is no Aer, no noise
model and no hardware path.

## What it shows

- **Watch** — one quantum driver (4, 6, 8 or 10 qubits where bundled, or the
  universal 4-qubit driver) against an optional classical rival (the 76-parameter
  MLP or the 2,436-parameter "pro"), and a ghost: the track's 4-qubit driver's
  lap from a standing start (in Race mode too, as a target). The side panel
  follows every decision: the sensor values, the live circuit with this
  decision's angles (QAMPoser's circuit editor, read-only), ⟨Z⟩ of every qubit
  and the Q-values. Pause, step one decision at a time, or open the exact circuit
  of a decision in IBM Quantum Composer.
  **Predict, then check** (paused only, inference on copies, `src/sim/whatif.ts`):
  with "Guess before you look" on, the scores (and ⟨Z⟩) stay hidden until you
  pick the move you expect; "cover" a sensor to re-run the same decision with
  that input at 0, or nudge one trained RY/RZ angle by up to ±π, and compare the
  Q-values side by side (gates outside every readout's light cone are flagged).
- **Race** — you (arrow keys / WASD / on-screen pedals) against the quantum car;
  **C** or the 📷 button switches between top-down, chase and cockpit cameras.
- **Evolution** — four snapshots of one training run racing each other (the same mode as at the booth).

## Train a tiny circuit — a toy

So the page shows learning, not only its result (Learning Audit item 14):
"Watch a tiny circuit learn" in the line under the header, or the link in
"What is this?", opens a panel where one or two qubits learn **steer toward
the longer ray** live, clearly labelled a toy. The core is pure and tested
(`src/sim/toy.ts`, `tests/toy.test.ts`); the panel is
`src/components/ToyPanel.tsx`.

- **Data:** a frozen, seeded batch of real readings: 64 training and 32
  held-out poses sampled on the oval (random centreline point, sideways offset
  up to 0.8 half widths, heading within 0.5 rad of the tangent), the two side
  rays (+60° left, −60° right, the drivers' outer rays) measured by the same
  `Observer` as the race. Label: Left if the left ray is longer. The readings
  lie close to one curve (left + right is roughly the track width), plus a few
  capped at 30 m.
- **Circuits** (QAMPoser's state vector, as in the race): 2 qubits, two blocks
  of RY(π·left) on q0, RY(π·right) on q1, trained RY RZ per qubit, CZ(0,1);
  score Left ⟨Z0⟩, score Right ⟨Z1⟩. Or 1 qubit, data re-uploading:
  RY(π·left), RY RZ, RY(π·right), RY RZ; ⟨Z⟩ > 0 means Left. P(Left) =
  sigmoid(3·margin), i.e. a softmax over 3·⟨Z0⟩, 3·⟨Z1⟩; mean cross-entropy.
  The last RZ on each wire commutes with the readout: its gradient is 0 and
  the panel greys it out.
- **Training:** full-batch gradient descent (Adam; learning rate 0.02, 0.08
  or 0.3), gradients by the parameter-shift rule (exact for RY/RZ, checked
  against finite differences in the tests), initial angles uniform in
  [−1, 1] rad from the seed, 15 steps a second in `requestAnimationFrame`
  chunks, at most 300 steps. A step at 2 qubits is about 3 ms (64 samples ×
  17 circuits). Closing the panel stops it; a hidden tab pauses it.
- **What it reaches:** seed 1 at 2 qubits passes 90 % held-out accuracy at
  step 13 (the test allows 40); at 1 qubit seed 1 gets stuck in a local
  minimum (loss ≈ 0.89, the panel says so), seed 2 learns in 11 steps.
- **Honest framing, in the panel:** a toy on a frozen batch; the race drivers
  were trained with double DQN over thousands of episodes, offline in Python;
  one line of classical code (left > right) solves the task, so nothing here
  claims a quantum advantage. Without the CZ each score would see only its own
  ray, which suffices for this rule, as on the oval in the entanglement
  ablation.

## Languages

English and German: the **EN/DE** switch in the header (at phone width at
the end of the modes row). The first visit follows the browser language
(`de…` → German, anything else English); a choice made with the switch is
remembered in this browser, and `?lang=de` / `?lang=en` in the URL wins over
both, e.g. https://racetraq.org/?lang=de for a German class. Lap times and
values follow the language (12,20 s in German). The linked documentation and
the notebooks stay English.

The strings are flat tables in `src/i18n/en.ts` and `src/i18n/de.ts`, with
`{name}` placeholders and the desktop app's German vocabulary
(`racetraq/web/i18n/de.json`); a key missing in German falls back to English,
and `tests/i18n.test.ts` checks that both tables have the same keys and
placeholders.

## How it relates to the Python project

| | server demo (`racetraq/`) | browser edition (`browser/`) |
|---|---|---|
| track geometry, car physics, observation | numpy | TypeScript port (`src/sim/`) |
| quantum circuit | numpy fastsim / Aer / IBM runtime | QAMPoser `simulateStatevector` |
| drawing | `racetraq/web/js/race.js` | the same file, imported unchanged |
| training, hardware, random/drawn tracks | yes | no (only the labelled toy trains) |

The port is checked against numpy, not just eyeballed: `tests/parity.test.ts`
compares track resampling, projection, raycasts, curvature look-ahead, physics
substeps, all 14 observation features, Q-values of 4- to 10-qubit circuits and
MLPs, and seven full greedy laps (action for action, same substep count)
against `tests/fixtures/parity.json`, which `tools/export_browser.py` writes
from the Python implementation.

One behaviour the server demo does not have: an agent car that has not
finished a lap after 600 decisions (60 s, the training env's episode cap)
respawns, so a policy that brakes to a standstill does not stay parked.

## Develop

```bash
cd browser
npm install
npm run dev        # http://localhost:5173
npm test           # parity tests
npm run build      # static site in dist/ (relative paths, host it anywhere)
```

The data in `public/data/` (tracks, every bundled driver, ghosts, manifest)
and the parity fixture are generated — after retraining or re-bundling a
driver, regenerate them from the repository root:

```bash
python tools/export_browser.py          # writes browser/public/data + tests/fixtures
python tools/export_browser.py --check  # what CI runs (tests/test_browser_export.py)
```

`@qamposer/react` comes from the `entangible` integration branch of
[JanLahmann/qamposer-react](https://github.com/JanLahmann/qamposer-react)
(first-class CZ gates and the exported `simulateStatevector` /
`expectationZ`), the same branch [Entangible](https://entangible.org) uses.
It is vendored as a built tarball, `vendor/qamposer-react-0.2.0-a50cc94.tgz`
(commit a50cc94 of that branch, packed from its `npm install` output): since
2026-10-16 npm can no longer prepare the git dependency (`Cannot read
properties of null (reading 'edgesOut')` in the fork's build step), which
broke CI and the Pages deploy. To update it, build the fork
(`npm install && npm run build` in a checkout), `npm pack`, replace the
tarball and point `package.json` at the new file.

## Hosting and analytics

`.github/workflows/pages.yml` builds and publishes the site to GitHub Pages on
every push to `main` that touches `browser/` (or `race.js`). The build uses
relative paths, so `dist/` also works from any other web server.

The page loads the Fun with Quantum family's cookie-free
[Umami](https://umami.is) tracker, restricted to `racetraq.org` by
`data-domains` (local dev, previews and self-hosted copies send nothing).
It uses the family's standard snippet: a `data-before-send` hook drops
headless browsers and bots, and `?notrack` turns tracking off in this browser
(`?track` turns it back on).
Events follow the family taxonomy (`racetraQ: <what happened>`); the list is
in `src/analytics.ts`. The "predict, then check" tools in the brain panel add
two: `racetraQ: predict` (`result` right | wrong, `qubits`) when a visitor
guesses a paused decision, and `racetraQ: what-if` (`kind` sensor | angle) when
a paused decision is re-run with a covered sensor or a nudged angle, at most
once per decision and kind. The toy adds `racetraQ: toy train` (`qubits` 1 | 2)
when a training run starts (Train or Step), once per run (seed and circuit),
never per step.

