"""The browser edition's static data (browser/public/data) and its numpy parity
fixture must match what tools/export_browser.py exports from the bundled
weights, tracks and ghosts — re-run the tool after retraining or re-bundling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import export_browser  # noqa: E402

BROWSER = ROOT / "browser"


@pytest.mark.skipif(not (BROWSER / "public" / "data" / "manifest.json").is_file(),
                    reason="browser export not present")
def test_browser_export_is_up_to_date():
    assert export_browser.check(BROWSER) == []


def test_every_bundled_driver_is_exported():
    manifest = json.loads((BROWSER / "public" / "data" / "manifest.json").read_text())
    exported = {d["id"] for d in manifest["drivers"]}
    bundled = {p.stem for p in export_browser.driver_files()}
    assert exported == bundled


def test_reference_rollouts_complete_a_lap():
    fixture = json.loads((BROWSER / "tests" / "fixtures" / "parity.json").read_text())
    assert fixture["rollouts"]
    assert all(r["result"] == "lap" for r in fixture["rollouts"])
