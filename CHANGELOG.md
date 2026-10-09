# Changelog

## 0.2.1 — 2026-10-09

- Container: upgrades msgpack, setuptools and urllib3 from the base image,
  whose versions failed the publish workflow's security scan, so the v0.2.0
  image was never pushed. This is the first `ghcr.io/janlahmann/racetraq`
  image.

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

## 0.1.0 — 2026-07-12

First release as traQmania: quantum DQN race cars (4 and 6 qubits), Watch,
Train, Evolution, Race and Hardware modes, notebooks, Raspberry Pi profiles
and a multi-arch container.
