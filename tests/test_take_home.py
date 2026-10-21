"""Explain -> "Take it home": every QR code the tab shows is committed, and
tools/make_qr.py (which wrote it) knows its URL."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "racetraq" / "web"


def _take_home_ids() -> list[str]:
    src = (WEB / "js" / "explain.js").read_text()
    match = re.search(r"export const TAKE_HOME = (\[[^\]]*\]);", src)
    assert match, "TAKE_HOME not found in explain.js"
    return json.loads(match.group(1))


def _qr_links() -> dict[str, str]:
    """``LINKS`` from tools/make_qr.py, read without importing segno."""
    tree = ast.parse((ROOT / "tools" / "make_qr.py").read_text())
    consts = {"REPO": "https://github.com/JanLahmann/racetraQ"}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "LINKS":
            value = node.value
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "LINKS":
            value = node.value
        else:
            continue
        out = {}
        for k, v in zip(value.keys, value.values):
            if isinstance(v, ast.JoinedStr):
                v = "".join(consts[p.value.id] if isinstance(p, ast.FormattedValue) else p.value for p in v.values)
            else:
                v = v.value
            out[k.value] = v
        return out
    raise AssertionError("LINKS not found in tools/make_qr.py")


def test_every_take_home_code_is_committed_and_generated():
    ids = _take_home_ids()
    links = _qr_links()
    assert ids == list(links)
    for key in ids:
        svg = WEB / "img" / f"qr-{key}.svg"
        assert svg.is_file(), f"{svg} missing: run python tools/make_qr.py"
        assert svg.read_text().startswith("<svg")
        assert links[key].startswith("https://")


def test_take_home_strings_exist_in_both_languages():
    for lang in ("en", "de"):
        strings = json.loads((WEB / "i18n" / f"{lang}.json").read_text())
        for key in _take_home_ids():
            for part in ("title", "text", "alt"):
                assert strings.get(f"explain.home.{key}.{part}"), (lang, key, part)
        assert strings.get("explain.home.title") and strings.get("explain.home.intro")


def test_qr_svgs_are_packaged():
    assert '"web/img/*.svg"' in (ROOT / "pyproject.toml").read_text()
