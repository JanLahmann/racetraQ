"""Booth idle: the next visitor starts from the booth defaults (no inherited
leaderboard name), running demos are never cut off, and only the driving
browser — or the server, when nobody holds the wheel — sends the booth back
to attract mode."""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from racetraq.config import load_config
from racetraq.server import protocol as P
from racetraq.server.session import DemoSession
from racetraq.server.ws import control_ticker

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_session(tmp_path, **ui):
    config = load_config()
    config["ui"] = {**config["ui"], **ui}
    session = DemoSession(config, ghosts_dir=tmp_path)
    session.drain_outbox()
    return session


def test_idle_reset_message_parses_strictly():
    assert isinstance(P.parse_client({"type": "idle_reset"}), P.IdleReset)
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "idle_reset", "mode": "race"})


def test_idle_reset_clears_the_name_and_returns_to_attract(tmp_path):
    session = make_session(tmp_path)
    session.handle_message(P.parse_client({"type": "set_name", "name": "Sarah"}))
    session.handle_message(P.parse_client({"type": "race", "action": "start",
                                           "opponent": "quantum"}))
    assert session.mode == "race" and session.racer_name == "Sarah"
    session.handle_message(P.parse_client({"type": "idle_reset"}))
    assert session.mode == "attract"
    assert session.racer_name == ""
    assert session.cars, "attract mode has its quantum car back"


def test_watch_mode_clears_the_name(tmp_path):
    session = make_session(tmp_path)
    session.racer_name = "Sarah"
    session.handle_message(P.parse_client({"type": "set_mode", "mode": "attract"}))
    assert session.racer_name == ""


def test_kiosk_idle_restores_the_startup_track_and_driver(tmp_path):
    session = make_session(tmp_path, kiosk=True)
    startup = session.track_name
    other = "oval" if startup != "oval" else "chicane"
    session.handle_message(P.parse_client({"type": "set_track", "track": other}))
    session.handle_message(P.parse_client({"type": "set_driver", "driver": "universal"}))
    session.drain_outbox()
    session.idle_reset()
    assert session.track_name == startup
    assert session.driver == "auto"
    types = [m["type"] for m in session.drain_outbox()]
    assert "track" in types and "welcome" in types


def test_without_kiosk_idle_keeps_the_operator_track(tmp_path):
    session = make_session(tmp_path, kiosk=False)
    session.handle_message(P.parse_client({"type": "set_track", "track": "chicane"}))
    session.idle_reset()
    assert session.track_name == "chicane"


def test_running_demos_hold_the_idle_return(tmp_path):
    session = make_session(tmp_path, attract_idle_seconds=20)
    session.handle_message(P.parse_client({"type": "set_mode", "mode": "race"}))
    assert session.idle_hold() is None
    assert session.idle_seconds() == 20

    session.handle_message(P.parse_client({"type": "set_mode", "mode": "evolution"}))
    assert session.mode == "evolution"
    assert session.idle_hold() == "hold" and session.idle_seconds() == 0
    session.idle_reset()
    assert session.mode == "evolution", "the evolution show is not cut off"

    session.mode = "studio"
    session.studio.phase = "done"  # a visitor reads their result
    assert session.idle_hold() == "linger"
    assert session.idle_seconds() == 90  # max(3 x 20 s, 90 s)


def test_idle_reset_forgets_the_studio_model(tmp_path):
    session = make_session(tmp_path)
    session.mode = "studio"
    session.studio.phase = "done"
    session.studio.model = object()
    session.studio.result = {"lapped": 3}
    session.idle_reset()
    assert session.studio.phase == "setup"
    assert session.studio.model is None and session.studio.result is None


class _Lock:
    def __init__(self, locked):
        self.locked = locked

    def tick(self):
        return False


class _Hub:
    def __init__(self, locked):
        self.lock = _Lock(locked)

    async def send_control_states(self):
        pass


class _Session:
    def __init__(self, mode="race", idle_s=0.03):
        self.mode = mode
        self._idle_s = idle_s
        self.resets = 0

    def set_input(self, keys):
        pass

    def idle_seconds(self):
        return self._idle_s

    def idle_reset(self):
        self.resets += 1
        self.mode = "attract"


async def _tick_for(hub, session, seconds):
    task = asyncio.create_task(control_ticker(hub, session, interval=0.01))
    await asyncio.sleep(seconds)
    task.cancel()


def test_server_resets_the_booth_when_nobody_holds_the_wheel():
    session = _Session()
    asyncio.run(_tick_for(_Hub(locked=False), session, 0.2))
    assert session.resets == 1 and session.mode == "attract"


def test_server_leaves_a_held_wheel_to_the_driving_browser():
    session = _Session()
    asyncio.run(_tick_for(_Hub(locked=True), session, 0.2))
    assert session.resets == 0


def test_server_never_resets_a_held_demo():
    session = _Session(idle_s=0.0)  # idle_seconds() is 0 while a demo holds it
    asyncio.run(_tick_for(_Hub(locked=False), session, 0.2))
    assert session.resets == 0


# --------------------------------------------------------------- browser side

_ATTRACT_PROBE = """
globalThis.window = { addEventListener() {} };
const { AttractManager } = await import(process.argv[1] + "/attract.js");
const { isTypingTarget } = await import(process.argv[1] + "/input.js");
const el = { classList: { add() {}, remove() {} }, hidden: false, textContent: "" };
const out = {};
const m = new AttractManager({ captionEl: el, onIdle() {} });
m.setIdleSeconds(20);
m.setMode("race");
out.watcherArmed = m.idleId !== null;
m.setDriving(true);
out.driverArmed = m.idleId !== null;
out.plain = m.effectiveIdleSeconds();
m.setHold("studio", "linger");
out.linger = m.effectiveIdleSeconds();
m.setHold("train", "hold");
out.hold = m.effectiveIdleSeconds();
out.holdArmed = m.idleId !== null;
m.setHold("train", null);
m.setHold("studio", null);
out.released = m.effectiveIdleSeconds();
m.setDriving(false);
const matches = (sel, tag) => sel.split(",").some((s) => s.trim() === tag);
const field = (tag) => ({ closest: (sel) => (matches(sel, tag) ? {} : null) });
out.typingInput = isTypingTarget(field("input"));
out.typingCanvas = isTypingTarget(field("canvas"));
out.typingNone = isTypingTarget(null);
console.log(JSON.stringify(out));
process.exit(0); // the caption rotation interval would keep node alive
"""


def test_only_the_driving_browser_arms_the_idle_timer():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = subprocess.run(
        [node, "--input-type=module", "-e", _ATTRACT_PROBE,
         str(REPO_ROOT / "racetraq" / "web" / "js")],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["watcherArmed"] is False
    assert out["driverArmed"] is True
    assert (out["plain"], out["linger"], out["hold"], out["released"]) == (20, 90, 0, 20)
    assert out["holdArmed"] is False
    assert out["typingInput"] is True
    assert out["typingCanvas"] is False and out["typingNone"] is False


def test_frontend_files_revalidate():
    """A booth browser must not keep running last week's JS after an update."""
    from fastapi.testclient import TestClient

    from racetraq.server.app import create_app

    client = TestClient(create_app(load_config()))
    response = client.get("/js/input.js")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert "isTypingTarget" in response.text
