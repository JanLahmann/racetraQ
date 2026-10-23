# Changelog

## Unreleased

- **Train a tiny circuit — a toy** in the browser edition (Learning Audit
  item 14), so racetraq.org shows learning, not only its result: one or two
  qubits learn "steer toward the longer ray" live, on a frozen, seeded batch
  of 64 real side-ray readings from the oval (32 more held out), by full-batch
  gradient descent with parameter-shift gradients. Live decision map over the
  batch, loss and accuracy curves (training and held out), the circuit with
  its angles moving; Train/Pause, Step, Reset (new seed), three learning
  rates, 1 or 2 qubits; flags a run stuck in a local minimum. Clearly labelled
  a toy: the race drivers were trained with double DQN, offline in Python, and
  a classical one-liner solves the task. Seed 1 at 2 qubits passes 90 %
  held-out accuracy at step 13. Opened from the line under the header and from
  "What is this?"; EN/DE; pure tested core (`browser/src/sim/toy.ts`); one new
  Umami event, `racetraQ: toy train` (`qubits`), once per run.
- **Notebook 08 — Does it drive tracks it never saw?** Generalisation and
  overfitting: four quantum and two MLP drivers on eight freshly generated
  tracks (trackgen seeds 800–807, none used for training or selection) — the
  gp-trained and universal drivers lap them all, the oval and chicane
  specialists almost never — and the universal study's top-ranked seed that
  laps its training tracks and none of the generated ones. Selection bias:
  over 645 committed training runs, the in-training eval a best snapshot was
  picked on reads +0.028 [+0.021, +0.036] above 36 fresh episodes, against
  +0.005 [−0.002, +0.011] for the unpicked final parameters. No training; runs
  in about 45 s and in CI. Linked from the README, the learning path (tracks
  2–5), the glossary (overfitting, generalisation, selection bias) and the
  teaching answer key.
- **Entanglement ablation** (SCIENCE.md, REPORT §4.7; `data/studies/entangler_oval_4q`,
  `entangler_chicane_4q`): the shipped 4-qubit recipe without the CZ ring, 8 seeds. On the
  oval nothing is lost (best snapshots lap 276 of 288 episodes without the ring, 278
  with it); on the chicane no seed becomes reliable (26 without, 280 with). Without the ring each action is scored from one sensor, so
  this measures the sensors-to-actions channel, not entanglement alone.
- `[circuit] entangler = "none"` builds the circuit without its CZ rings (the
  entanglement ablation of Learning Audit item 13; default `"cz_ring"`,
  unchanged). fastsim, the adjoint gradients, the Qiskit/hardware circuits,
  the light-cone analysis and the surrogate spectra all follow it; weights
  trained that way record it in their `.meta.json` and every loader rebuilds
  the same circuit. The browser export refuses such drivers.
- **Predict, then check** in the browser edition (Learning Audit item 5): at a
  paused decision, "Guess before you look" hides the scores until you pick the
  move you expect, then says whether the driver agrees and how close it was;
  "cover" a sensor re-runs the same decision with that input at 0, and (quantum
  drivers) a slider nudges one trained RY/RZ angle by up to ±π, with the
  Q-values side by side and a note when the gate is outside every readout's
  light cone. Inference only, on copies (`browser/src/sim/whatif.ts`, tested);
  two new Umami events, `racetraQ: predict` and `racetraQ: what-if`, at most
  once per decision.
- Booth: Explain → **Take it home** (DE: Zum Mitnehmen) shows four QR codes:
  racetraq.org, the learning path, the notebooks and the technical report, so
  visitors leave with the next step. The codes are committed SVGs
  (`tools/make_qr.py`), so the booth shows them offline.
- racetraq.org pings IndexNow (Bing, Yandex & co.) after each deploy, through
  the Fun with Quantum family workflow; the sitemap's `lastmod` is the deploy
  date.
- racetraq.org for search engines: the page carries a heading, a short intro
  and links in its raw HTML (replaced by the app on mount), a canonical URL
  (`?lang=de` keeps its own), a descriptive title, and a sitemap without the
  duplicate `?lang=en` entry, with `lastmod`. The header logo is now the
  page's h1.
- racetraq.org uses the Fun with Quantum family's standard Umami snippet:
  headless browsers and bots send nothing, and `?notrack` turns tracking off
  in this browser (`?track` turns it back on).
- racetraq.org: structured data for search engines (JSON-LD: the app, the
  source code with its DOI, the study data and the report) and hreflang
  en/de in the page and the sitemap.
- `rqb-demo.json` pins the Docker image by digest (v0.3.3), with the tag as a
  fallback, so a RasQberry Pi runs the reviewed image, not whatever `:latest` is.
- **racetraq.org in German**: an EN/DE switch in the header of the browser
  edition (also `?lang=de`, and German by default in a German browser), with
  the desktop app's vocabulary and German number formats; the notebooks stay
  English.
- The sixteen audit studies now name the code they ran: a
  `git_commits_reconstructed` block in each `report.json` (`eb1182f`, and
  `778e5a8` for the three `robust_*` studies) and a table in REPORT §7. Both
  matches are exact, by file hashes.

## 0.3.3 — 2026-10-17

- **Claims audit**: about 2,600 numbers and statements in SCIENCE.md,
  REPORT.md, the README, the other docs, the notebooks and `data/records.md`
  were checked against the study data; 89 were corrected. Among them: the
  chicane 4-versus-6-block result at 10 qubits (6 blocks lap supportedly
  faster), `quantum_gp_q10` as the fastest learned gp driver *after* the
  `pro` MLP, and statements left over from before the October re-bundles.
- **Studio and app** (from persona walkthroughs in English and German):
  working links in the in-app docs, a readable lap board, track names
  instead of ids, German number formatting, feedback on Stop, axis titles
  on the training charts, "blocks" as in the docs, clearer comparison
  text, and an accurate claim instead of "genuinely quantum".
- **racetraq.org**: the rival is labelled when it is not a matched
  comparison, About is accurate for the larger circuits and cites the DOI,
  About pauses a race, and the help text fits touch screens.
- **Docs**: realistic teaching and learning plans (the Studio needs the
  local app; racetraq.org is English-only), five glossary entries, a
  command that reproduces one published row, no private paths in
  `data/studies`, and the hardware log names its weights' sha256.
- The browser edition vendors the built `@qamposer/react`: npm can no
  longer prepare it as a git dependency.

## 0.3.2 — 2026-10-14

- **New 10-qubit gp driver**: `quantum_gp_q10` is now a 6-block circuit
  with 6 actions, chosen by rule from an 8-seed study. It laps 72 of 72
  fresh gp episodes at 18.9 s (the July file: 49 of 72 at 22.3 s), has no
  light-cone blind spots, and also laps oval and chicane.
- **10-qubit gp follow-ups** in SCIENCE.md and REPORT: plain rays, a pace
  phase and wider action sets at 8 seeds each (`data/studies/gp_q10feat_*`);
  `data/records` re-run for the new driver.
- `tools/study.py`: a cell no longer fails when the study already runs at
  the lowest priority.
- The browser edition serves `sitemap.xml` and `robots.txt`.
- The image build no longer pulls anything from Docker Hub (its anonymous
  rate limit had failed the 0.3.1 publish).
- `CITATION.cff` and the README carry the Zenodo DOI
  ([10.5281/zenodo.23270600](https://doi.org/10.5281/zenodo.23270600), all
  versions).

## 0.3.1 — 2026-10-09

- `train_headless` without `--out` writes a new `runs/<agent>_<track>_<time>/`
  folder instead of replacing the shipped weights; `--out bundled` does that
  on purpose.
- First release archived on Zenodo, which gives racetraQ a DOI for citing.

## 0.3.0 — 2026-10-09

- **Studio** (#15): the stage during setup and training, a first-lap
  moment, test results as a chart, you against your circuit, time
  estimates measured on this machine, plainer words.
- **Board moderation** (#15): a name filter and operator controls
  (`#operator`).
- **German UI** (#18): an EN/DE switch in the header; `[ui] language =
  "de"` makes German the booth default (the visitor's own choice wins).
- **Learning path** (#19): docs/LEARN.md with five routes (visitor to researcher), a
  new notebook 00 (qubits for drivers), exercises and checkpoints in
  notebooks 01–04, docs/GLOSSARY.md and docs/TEACHING.md, all reachable
  from the booth's docs menu.
- **Science follow-ups** (#20): which device-noise lever matters (both,
  on oval and chicane; neither on gp), a pace phase with a pace selection
  rule (`snapshot_rank = "pace"`), 10 qubits on gp under the July recipe
  (8 seeds per depth), and the universal driver re-selected by a rule that
  includes ten generated tracks (`bundle_driver.py --unseen`; same seed).
  New trainer option: widening a 4-action init to 6 or 8 actions.

## 0.2.2 — 2026-10-09

- Container: the publish workflow's Trivy gate now reads a `.trivyignore`
  for four findings that are artefacts of the QuBins base image (versions
  recorded in a base layer that a later layer replaces; the image has the
  fixed versions). This is the first `ghcr.io/janlahmann/racetraq` image:
  the v0.2.0 and v0.2.1 publish runs stopped at that gate.

## 0.2.1 — 2026-10-09

- Container: floors msgpack, setuptools and urllib3 in the Dockerfile. It
  did not clear the scan: the image already had the fixed versions (see
  0.2.2).

## 0.2.0 — 2026-10-09

The project is now **racetraQ** (formerly traQmania): package `racetraq`,
repository `JanLahmann/racetraQ`, website https://racetraq.org, container
`ghcr.io/janlahmann/racetraq`.

- **October 2026 audit** (#4): light-cone analysis of what each action can
  see, every bundled driver retrained and re-selected from multi-seed
  studies (`data/studies/`), and the IBM hardware path brought up to date.
  One full lap on `ibm_marrakesh` (2026-10-03). The headline changed: with
  8–10 seeds the matched classical baseline is ahead of the circuit on most
  metrics (docs/SCIENCE.md, "Honest claims").
- **Browser edition** (#5): the trained drivers race in any browser with no
  server, at https://racetraq.org.
- **Training studio** (#6): visitors build, train, compare and race their
  own quantum driver.
- **Rename** to racetraQ (#7).
- New hero GIF and records matrix (#8).
- **Chase and cockpit cameras** for the human driver (#9).
- **Booth fixes** (#10): the race name field accepts W, A, S, D; idle resets
  the booth for the next visitor without cutting off running demos; a
  watching phone can no longer reset the booth; offline-safe `run.sh`; a
  pre-event checklist; link previews and a first-visit explainer on
  racetraq.org.
- **Race feedback, attract headline, kiosk mode** (#12): 3-2-1-GO, a YOU
  tag, a lap clock, a lap result with board rank, recovery where the car
  left the track; any arrow key or controller button starts a race from
  Watch; the kiosk hides operator controls (`#operator` shows them).
- **README for first-time readers, shared vocabulary, accessibility**
  (#14): CONTRIBUTING.md and CITATION.cff; chase camera by default on
  phones; qubit sizes without a driver greyed out; colour-blind-safe car
  colours.

## 0.1.0 — 2026-07-12

First release as traQmania: quantum DQN race cars (4 and 6 qubits), Watch,
Train, Evolution, Race and Hardware modes, notebooks, Raspberry Pi profiles
and a multi-arch container.
