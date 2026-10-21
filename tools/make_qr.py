"""Generate the QR codes of the booth's Explain -> "Take it home" tab.

Run as ``python tools/make_qr.py`` (needs ``pip install segno``, a one-off
dev tool, not a runtime dependency).  Writes one SVG per link to
``racetraq/web/img/qr-<id>.svg``: the codes are committed, so the booth shows
them offline, with no network call and no QR library in the browser.

The ids match ``TAKE_HOME`` in ``racetraq/web/js/explain.js``; a test checks
that every id there has its SVG here and that each code's URL is listed in
``LINKS`` below.
"""

from __future__ import annotations

from pathlib import Path

import segno

OUT = Path(__file__).resolve().parent.parent / "racetraq" / "web" / "img"

REPO = "https://github.com/JanLahmann/racetraQ"

#: id -> URL, one per learner level (docs/LEARN.md's tracks)
LINKS = {
    "play": "https://racetraq.org/",
    "learn": f"{REPO}/blob/main/docs/LEARN.md",
    "notebooks": f"{REPO}#notebooks",
    "research": f"{REPO}/blob/main/docs/REPORT.md",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, url in LINKS.items():
        path = OUT / f"qr-{key}.svg"
        # dark modules on a white quiet zone: scanners need the contrast, even on the dark booth theme
        segno.make(url, error="m").save(path, kind="svg", scale=4, border=2, dark="#0f1117", light="#ffffff", xmldecl=False, svgns=True)
        print(f"{path.relative_to(OUT.parent.parent.parent)}  {url}")


if __name__ == "__main__":
    main()
