# Contributing to racetraQ

Issues and pull requests are welcome: bugs, booth experiences, teaching
material, new tracks, and measurements that confirm or contradict
[docs/SCIENCE.md](docs/SCIENCE.md).

## Setup

Python ≥ 3.11:

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,hardware]"      # add ,notebooks for the notebooks
python -m racetraq                    # the server at http://127.0.0.1:8000
```

## Checks (CI runs the same)

```sh
ruff check .
pytest                                # a few tests drive the web JS through node
```

The notebooks are executed in CI; run one locally with
`jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb`.

## The browser edition (`browser/`)

Use **Node 20** (npm 11 on Node 24+ cannot prepare the QAMPoser git
dependency):

```sh
cd browser
npm ci
npm test
npm run dev        # http://localhost:5173
npm run build
```

It shares the track and car renderer with the server's web UI
(`racetraq/web/js/race.js`, imported as `@demo/race.js`), so check both when
you change it. `python tools/export_browser.py` regenerates
`browser/public/data/` from the bundled weights and tracks.

## Training and results

- `python -m racetraq.train_headless ...` writes a new
  `runs/<agent>_<track>_<time>/` folder unless `--out` names one; only
  `--out bundled` replaces the shipped weights in `racetraq/weights/`.
- One run is an anecdote. A claim about a recipe needs a study
  (`tools/study.py`, several seeds, interval statistics), and a bundled
  driver changes only through `tools/bundle_driver.py`, which records the
  selection rule in the weights' `.meta.json`.
- Numbers in the README and docs come from `data/records.json` and
  `data/studies/`; update them together.

## Style

Code reads like the code around it. Docs say what was measured, on how many
seeds, and what was not shown. The protocol is documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); a new message goes there too.
