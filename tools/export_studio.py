"""Build the training studio's comparison data, racetraq/data/studio_stats.json.

For every (track, qubits) the studio offers, picks the multi-seed study runs
closest to a studio run (cold start, lidar sensors, 4 actions, the profile's
depth, the live recipe where a study matches it) from data/studies/*/cells.json
and records: runs, runs that lapped, first-clean-lap episodes, and the best
in-training test (12-episode greedy eval) of each run. ``speed`` holds the
live training speed per circuit size measured through the server's own
training path (``--calibrate``), which turns episode counts into time
estimates.

Usage:
  python tools/export_studio.py                 # rebuild from data/studies + calibration
  python tools/export_studio.py --calibrate 60  # first re-measure live speed (60 s per size)
  python tools/export_studio.py --check         # what tests/test_studio.py runs
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / "data" / "studies"
CALIBRATION = STUDIES / "studio_calibration.json"
OUT = ROOT / "racetraq" / "data" / "studio_stats.json"

# (track, qubits) -> (study, variant): the closest study runs to a studio run.
# oval/chicane 4q and oval 6q use the live recipe (action gap + acting noise,
# robust_* studies); the rest bootstrap truncation (+ epsilon_end 0.30 on the
# hard tracks), at the profile's depth (q8: 5 blocks, q10: 6).
SOURCES = {
    ("oval", 4): ("robust_oval_4q", "gap8an"),
    ("chicane", 4): ("robust_chicane_4q", "gap8an"),
    ("oval", 6): ("robust_oval_q6", "gap8an"),
    ("chicane", 6): ("chicane_q6", "quantum"),
    ("oval", 8): ("oval_q8", "L5"),
    ("chicane", 8): ("chicane_q8", "L5"),
    ("oval", 10): ("oval_q10", "L6"),
    ("chicane", 10): ("chicane_q10", "L6"),
    ("gp", 4): ("gp_4q", "eps30_trunc"),
    ("gp", 6): ("gp_4q", "q6"),
    ("combo", 4): ("combo_4q", "quantum"),
}


def _quartiles(values: list[float]) -> dict:
    values = sorted(values)
    if len(values) == 1:
        return {"median": values[0], "q25": values[0], "q75": values[0]}
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "q25": q[0], "q75": q[2]}


def cell_stats(study: str, variant: str) -> dict:
    cells = [c for c in json.loads((STUDIES / study / "cells.json").read_text())
             if c["variant"] == variant and c["agent"] == "quantum"]
    if not cells:
        raise ValueError(f"no quantum runs of variant {variant} in {study}")
    firsts = sorted(int(c["first_clean_episode"]) for c in cells
                    if c.get("first_clean_episode") is not None)
    best = [c["best_eval"] for c in cells if c.get("best_eval")]
    shares = [b["lapped_episodes"] / max(1, b["eval_episodes"]) for b in best]
    mean_laps = sorted(round(float(b["mean_lap"]), 3) for b in best
                       if b.get("mean_lap") is not None)
    return {
        "source": f"{study} / {variant}",
        "episodes": int(cells[0]["episodes"]),
        "runs": len(cells),
        "lapped_runs": len(firsts),
        "first_laps": firsts,
        "first_lap": _quartiles(firsts) if firsts else None,
        "best_lapped_share": round(statistics.median(shares), 3) if shares else None,
        "best_mean_laps": mean_laps,
        "best_mean_lap": round(statistics.median(mean_laps), 3) if mean_laps else None,
    }


def calibrate(seconds: float) -> dict:
    """Live training speed per circuit size on the oval, through the demo
    server's own training path (DemoSession + train start)."""
    from racetraq.config import load_config
    from racetraq.server import protocol
    from racetraq.server.session import DemoSession

    speed = {}
    for n in (4, 6, 8, 10):
        config = load_config() if n == 4 else load_config(f"q{n}")
        session = DemoSession(config, ghosts_dir=Path(tempfile.mkdtemp()))
        session.handle_message(protocol.Train(action="start", agent="quantum",
                                              track="oval", episodes=100_000))
        job = session.jobs["quantum"]
        time.sleep(seconds)
        with job.lock:
            episodes = job.episode + 1
        session.stop_training()
        job.thread.join(timeout=120)
        speed[str(n)] = {"s_per_episode": round(seconds / episodes, 4),
                         "episodes": episodes, "seconds": seconds}
        print(f"{n} qubits: {episodes} episodes in {seconds:.0f} s", flush=True)
    return speed


def machine() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        chip = ""
    return chip or platform.processor() or platform.machine()


def build() -> dict:
    calibration = json.loads(CALIBRATION.read_text()) if CALIBRATION.is_file() else {}
    return {
        "note": "generated by tools/export_studio.py from data/studies and "
                "data/studies/studio_calibration.json — do not edit by hand",
        "machine": calibration.get("machine"),
        "speed": calibration.get("speed", {}),
        "cells": {f"{track}_q{n}": cell_stats(*src) for (track, n), src in SOURCES.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibrate", type=float, metavar="SECONDS",
                        help="re-measure live training speed first (seconds per size)")
    parser.add_argument("--check", action="store_true",
                        help="fail when studio_stats.json is out of date")
    args = parser.parse_args(argv)
    if args.calibrate:
        CALIBRATION.write_text(json.dumps(
            {"machine": machine(), "date": time.strftime("%Y-%m-%d"),
             "track": "oval", "speed": calibrate(args.calibrate)}, indent=1) + "\n")
    text = json.dumps(build(), indent=1) + "\n"
    if args.check:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is out of date — run python tools/export_studio.py",
                  file=sys.stderr)
            return 1
        print("studio stats up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
