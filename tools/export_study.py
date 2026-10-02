"""Export a tools/study.py study into a compact, committable summary.

A study directory holds weights and full logs for every (variant, seed) cell —
too big for the repo.  This writes what the docs and notebooks need to
``<out>/<name>/``:

- ``report.md`` / ``report.json`` — the statistical report (``study.py report``)
- ``cells.json`` — one record per finished cell: variant, seed, agent, track,
  profile, overrides, parameter count, first clean lap, the in-training eval
  log, the 36-episode reliability evals of the best snapshot and of the final
  params, mean return per 100 episodes, wall time

Usage: python tools/export_study.py STUDY_DIR --name gp_4q [--out data/studies]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import study  # noqa: E402

CELL_KEYS = ("variant", "seed", "agent", "track", "profile", "episodes", "overrides",
             "n_params", "first_clean_episode", "eval_log", "best_eval", "final_eval",
             "best_snapshot_eval", "final_params_eval", "mean_return_last200",
             "returns_by_100", "wall_time_s")


def export(study_dir: Path, name: str, out: Path, baseline: str | None = None) -> Path:
    """Write the summary of ``study_dir`` to ``out/name``; returns that directory."""
    target = out / name
    target.mkdir(parents=True, exist_ok=True)
    report = study.build_report(study_dir, baseline=baseline)
    (target / "report.md").write_text(study.render_markdown(report), encoding="utf-8")
    (target / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    cells = []
    for path in sorted((study_dir / "cells").glob("*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cells.append({key: result.get(key) for key in CELL_KEYS})
    (target / "cells.json").write_text(json.dumps(cells) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", type=Path, help="study directory (tools/study.py --out)")
    parser.add_argument("--name", required=True, help="summary directory name")
    parser.add_argument("--out", type=Path, default=Path("data/studies"))
    parser.add_argument("--baseline", default=None, help="baseline variant of the report")
    args = parser.parse_args(argv)
    target = export(args.study, args.name, args.out, args.baseline)
    n_cells = len(json.loads((target / "cells.json").read_text(encoding="utf-8")))
    print(f"{args.study} -> {target} ({n_cells} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
