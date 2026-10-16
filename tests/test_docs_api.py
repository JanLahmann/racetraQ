"""The /api/docs endpoints that back the in-UI documentation browser."""

import re
from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from racetraq.config import load_config  # noqa: E402
from racetraq.server.app import create_app, discover_docs  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return fastapi_testclient.TestClient(create_app(load_config()))


def test_docs_index_lists_repo_docs(client):
    docs = client.get("/api/docs").json()["docs"]
    ids = [d["id"] for d in docs]
    # in a source checkout the README (racetraQ) leads; every entry carries
    # its curated menu title, and the parked TM2020 concept stays off the menu
    if not discover_docs():
        assert docs == []  # bare install: feature reports empty, UI hides it
        return
    assert ids == ["README", "EXPLAINER", "LEARN", "GLOSSARY", "REPORT", "EXHIBITION",
                   "TEACHING", "SCIENCE", "ARCHITECTURE"]
    assert all(d["title"] for d in docs)


def test_doc_content_and_404(client):
    if not discover_docs():
        pytest.skip("no repo docs next to the package")
    doc = client.get("/api/docs/SCIENCE").json()
    assert doc["id"] == "SCIENCE"
    assert doc["markdown"].lstrip().startswith("#")
    assert client.get("/api/docs/nope").status_code == 404
    # path traversal never reaches the filesystem: ids are whitelist-matched
    assert client.get("/api/docs/..%2FREADME").status_code == 404


def test_relative_doc_links_point_at_files():
    """Every relative link and image in docs/*.md names a file or folder in
    the repo (the in-app viewer sends them to GitHub, resolved like there)."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    if not docs.is_dir():
        pytest.skip("no repo docs")
    link = re.compile(r"\]\(([^)\s]+)\)")
    broken = []
    for md in sorted(docs.glob("*.md")):
        text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)
        for href in link.findall(text):
            if re.match(r"[a-z]+:", href) or href.startswith("#"):
                continue
            target = (md.parent / href.split("#")[0]).resolve()
            if not target.exists():
                broken.append(f"{md.name}: {href}")
    assert broken == []
