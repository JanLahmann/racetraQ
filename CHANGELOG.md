# Changelog

## Unreleased

- **racetraq.org in German**: an EN/DE switch in the header of the browser
  edition (also `?lang=de`, and German by default in a German browser), with
  the desktop app's vocabulary and German number formats; the notebooks stay
  English.

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
