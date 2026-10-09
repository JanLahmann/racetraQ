"""German UI: the string tables (racetraq/web/i18n/en.json, de.json) match
each other, cover every key the page uses, and stay in sync with the English
text the server sends (the page translates catalog entries, features and
action labels by id). The i18n module itself runs under node."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1] / "racetraq" / "web"
LANGS = ("en", "de")
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _table(lang: str) -> dict[str, str]:
    return json.loads((WEB / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables() -> dict[str, dict[str, str]]:
    return {lang: _table(lang) for lang in LANGS}


def test_tables_have_the_same_keys(tables):
    en, de = tables["en"], tables["de"]
    assert sorted(set(en) - set(de)) == [], "missing in de.json"
    assert sorted(set(de) - set(en)) == [], "missing in en.json"


def test_tables_have_no_empty_strings(tables):
    for lang, table in tables.items():
        assert all(isinstance(v, str) for v in table.values()), lang
        empty = [k for k, v in table.items() if not v.strip()]
        assert empty == [], f"empty strings in {lang}.json"


def test_placeholders_match_per_key(tables):
    en, de = tables["en"], tables["de"]
    diff = {key: (sorted(set(PLACEHOLDER.findall(en[key]))),
                  sorted(set(PLACEHOLDER.findall(de[key]))))
            for key in en if key in de}
    assert {k: v for k, v in diff.items() if v[0] != v[1]} == {}


def _html_keys() -> set[str]:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    return set(re.findall(r'data-i18n(?:-html|-tip|-title|-placeholder|-aria)?="([^"]+)"', html))


def _js_keys() -> dict[str, set[str]]:
    """Literal keys of t("…") / tr("…") calls, including both arms of a
    `t(cond ? "a" : "b")`, per web/js file."""
    call = re.compile(r'\b(?:t|tr)\(\s*"([^"]+)"')
    ternary = re.compile(r'\b(?:t|tr)\([^()"]*\?\s*"([^"]+)"\s*:\s*"([^"]+)"')
    out = {}
    for path in sorted((WEB / "js").glob("*.js")):
        src = path.read_text(encoding="utf-8")
        keys = set(call.findall(src))
        for a, b in ternary.findall(src):
            keys |= {a, b}
        out[path.name] = keys
    return out


def test_every_key_the_page_uses_exists(tables):
    en = tables["en"]
    html = _html_keys()
    assert len(html) > 50  # the scan found the attributes
    assert sorted(html - set(en)) == [], "index.html"
    js = _js_keys()
    assert sum(len(k) for k in js.values()) > 200
    for name, keys in js.items():
        assert sorted(keys - set(en)) == [], name


def test_dynamic_keys_exist(tables):
    """Keys the page builds at runtime (camera views, hardware phases, driver
    kinds, generated-track names, number words) are all in the tables."""
    from racetraq.server.protocol import HARDWARE_PHASES

    en = tables["en"]
    dynamic = (
        [f"camera.{v}" for v in ("top", "chase", "cockpit")]
        + [f"hw.phase.{p}" for p in HARDWARE_PHASES]
        + [f"kind.{k}" for k in ("quantum", "mlp", "human", "hero", "pro")]
        + ["track.random_n", "track.drawn_n"]
        + [f"num.{n}" for n in (3, 5, 7, 9)]
        + [f"hw.mode.{m}" for m in ("session", "batch", "job")]
        + [f"hw.rescale.{r}" for r in ("global", "readout")]
        + [f"studio.reason.{r}" for r in ("time", "converged", "budget", "stopped", "error")]
    )
    assert [k for k in dynamic if k not in en] == []


def test_server_text_matches_the_english_table(tables):
    """The page shows the server's English catalog, feature and action names
    translated by id; the English entries must say what the server says, or
    the two languages drift apart."""
    from racetraq.agents.base import ACTION_SIZES, action_labels
    from racetraq.env.racing_env import FEATURE_LABELS
    from racetraq.studio import ACTION_BLURBS, SENSOR_PRESETS, STUDIO_TRACKS, TRACK_NOTES

    en = tables["en"]
    for sid, preset in SENSOR_PRESETS.items():
        assert en[f"studio.sensor.{sid}.label"] == preset["label"]
        assert en[f"studio.sensor.{sid}.blurb"] == preset["blurb"]
        assert f"studio.sensor_short.{sid}" in en
    for n in ACTION_SIZES:
        assert en[f"studio.actions_blurb.{n}"] == ACTION_BLURBS[n]
    for track in STUDIO_TRACKS:
        assert en[f"studio.track_note.{track}"] == TRACK_NOTES[track]
    for kind, label in FEATURE_LABELS.items():
        key = "feature.corner_speed" if kind == "corner_speed_ratio" else f"feature.{kind}"
        assert en[key] == label
    for label in action_labels(max(ACTION_SIZES)):
        assert en["action." + label.lower().replace(" ", "_")] == label


def test_visitor_error_key_matches_its_message(tables):
    from racetraq.config import load_config
    from racetraq.server import protocol as P
    from racetraq.server.session import DemoSession

    session = DemoSession(load_config())
    session.drain_outbox()
    session.handle_message(P.parse_client({"type": "set_name", "name": "sh1t head"}))
    (err,) = [m for m in session.drain_outbox() if m["type"] == "error"]
    assert tables["en"][err["key"]] == err["message"]
    assert tables["de"][err["key"]] != err["message"]


def test_default_language_is_english():
    from racetraq.config import load_config

    assert load_config()["ui"]["language"] == "en"


# ------------------------------------------------------------------ under node

_PROBE = """
import { pathToFileURL } from "node:url";
const dir = process.argv[1];
const I = await import(pathToFileURL(dir + "/i18n.js").href);
const A = await import(pathToFileURL(dir + "/attract.js").href);
const C = await import(pathToFileURL(dir + "/circuit.js").href);
const out = {};
out.lang = I.lang();
out.en = I.t("race.hud", { lap: 2, last: "—", best: "12.30s" });
out.enFeature = I.featureLabel("ray -60°") + "|" + I.featureLabel("curvature ahead 30m");
out.enAction = I.actionLabel("Brake right") + "|" + I.actionLabel("<b>odd</b>");
out.enSensors = A.sensorPhrase(10, ["ray -60°", "ray 0°", "ray +60°", "speed", "heading error"]);
out.enCaptions = A.attractCaptions({ n_qubits: 6, n_params: { total: 80 } }, null);
I.setLang("de");  // node: no localStorage, no document — must not throw
out.de = I.t("race.hud", { lap: 2, last: "—", best: "12.30s" });
out.deFeature = I.featureLabel("ray -60°") + "|" + I.featureLabel("curvature ahead 30m");
out.deAction = I.actionLabel("Brake right") + "|" + I.actionLabel("<b>odd</b>");
out.deSensors = A.sensorPhrase(10, ["ray -60°", "ray 0°", "ray +60°", "speed", "heading error"]);
out.deStock = A.attractCaptions();
out.deParams = C.parameterCaption({ n_params: { total: 56 }, dead_params: 0 });
out.missing = I.t("no.such.key");
out.unfilled = I.t("race.hud", { lap: 1 });
I.setLang("xx");
out.stillDe = I.lang();
console.log(JSON.stringify(out));
"""


def test_i18n_module_under_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = subprocess.run([node, "--input-type=module", "-e", _PROBE, str(WEB / "js")],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["lang"] == "en"
    assert out["en"] == "Lap 2 · last — · best 12.30s"
    assert out["enFeature"] == "ray -60°|curvature ahead 30m"  # English: the server's text
    assert out["enAction"] == "Brake right|<b>odd</b>"  # unknown labels pass through
    assert out["enSensors"] == "Three distance sensors, speed and heading error"
    assert out["enCaptions"][0] == "A 6-qubit quantum circuit is driving this car."
    assert "80 learned numbers" in out["enCaptions"][1]
    assert out["de"] == "Runde 2 · letzte — · beste 12.30s"
    assert out["deFeature"] == "Strahl -60°|Krümmung voraus 30m"
    assert out["deAction"] == "Bremsen rechts|<b>odd</b>"
    assert out["deSensors"] == "Drei Abstandssensoren, Geschwindigkeit und Richtungsfehler"
    assert out["deStock"][0] == "Ein Quantenschaltkreis mit 4 Qubits fährt dieses Auto."
    assert out["deParams"] == "56 trainierbare Parameter, davon keiner strukturell tot"
    assert out["missing"] == "no.such.key"
    assert out["unfilled"].startswith("Runde 1 · letzte {last}")
    assert out["stillDe"] == "de"
