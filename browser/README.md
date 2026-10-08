# traQmania — browser edition

**Live: https://janlahmann.github.io/traQmania/**

The trained traQmania drivers racing entirely in the browser: no Python, no
server, no account. A static site you can host anywhere (GitHub Pages,
qamposer.org, a Raspberry Pi at a workshop) or open from a USB stick through
any local web server.

**Inference only.** Nothing is trained here. Every 0.1 s the quantum driver
turns its sensor readings into rotation angles, the circuit runs on
[QAMPoser](https://qamposer.org)'s in-browser state-vector simulator
(`@qamposer/react`), and the car takes the action with the highest
Q = w·⟨Z⟩ + b. Simulation is exact and noiseless; there is no Aer, no noise
model and no hardware path.

## What it shows

- **Watch** — one quantum driver (4, 6, 8 or 10 qubits where bundled, or the
  universal 4-qubit driver) against an optional classical rival (the 76-parameter
  MLP or the 2,436-parameter "pro"), with the recorded ghost lap. The side panel
  follows every decision: the sensor values, the live circuit with this
  decision's angles (QAMPoser's circuit editor, read-only), ⟨Z⟩ of every qubit
  and the Q-values. Pause, step one decision at a time, or open the exact circuit
  of a decision in IBM Quantum Composer.
- **Race** — you (arrow keys / WASD / on-screen pedals) against the quantum car.
- **Learning** — the evolution mode: four snapshots of one training run.

## How it relates to the Python project

| | server demo (`traqmania/`) | browser edition (`browser/`) |
|---|---|---|
| track geometry, car physics, observation | numpy | TypeScript port (`src/sim/`) |
| quantum circuit | numpy fastsim / Aer / IBM runtime | QAMPoser `simulateStatevector` |
| drawing | `traqmania/web/js/race.js` | the same file, imported unchanged |
| training, hardware, random/drawn tracks | yes | no |

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

## Hosting and analytics

`.github/workflows/pages.yml` builds and publishes the site to GitHub Pages on
every push to `main` that touches `browser/` (or `race.js`). The build uses
relative paths, so `dist/` also works from any other web server.

The page loads the Fun with Quantum family's cookie-free
[Umami](https://umami.is) tracker, restricted to `janlahmann.github.io` by
`data-domains` (local dev, previews and self-hosted copies send nothing).
Events follow the family taxonomy (`traQmania: <what happened>`); the list is
in `src/analytics.ts`.

