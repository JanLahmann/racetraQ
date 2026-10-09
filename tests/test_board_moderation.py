"""Board moderation: blocked names are refused, and the operator (on the
booth machine only) removes entries or clears a board."""

import time

import pytest

from racetraq import studio as studio_mod
from racetraq.config import load_config
from racetraq.moderation import name_allowed
from racetraq.server import protocol as P
from racetraq.server import ws as ws_mod
from racetraq.server.session import DemoSession


@pytest.mark.parametrize("name", ["Sarah", "Cassie", "Scunthorpe", "Dickens", "Assia",
                                  "Müller", "Jan L.", "Anna_2009"])
def test_real_names_pass(name):
    assert name_allowed(name)


@pytest.mark.parametrize("name", ["fuck", "Fu.ck", "f u c k", "sh1t", "shitty driver",
                                  "Bitches", "Fotze123", "WICHSER"])
def test_blocked_words_are_caught(name):
    assert not name_allowed(name)


def test_booth_words_extend_the_list():
    assert name_allowed("Rival Corp")
    assert not name_allowed("Rival Corp", ["rival"])


@pytest.fixture()
def session(tmp_path):
    s = DemoSession(load_config(), ghosts_dir=tmp_path)
    s.drain_outbox()
    return s


def test_blocked_name_is_refused(session):
    session.handle_message(P.parse_client({"type": "set_name", "name": "Ada"}))
    assert session.racer_name == "Ada"
    session.handle_message(P.parse_client({"type": "set_name", "name": "sh1t head"}))
    assert session.racer_name == ""
    (err,) = [m for m in session.drain_outbox() if m["type"] == "error"]
    assert err["visitor"] is True and err["field"] == "name"
    assert err["key"] == "error.name_blocked"  # the page shows it in its own language
    P.parse_server(err)


def board(*entries):
    return [{"name": n, "lap_s": t, "date": d} for n, t, d in entries]


def test_remove_one_race_entry(session):
    today = time.strftime("%Y-%m-%d")
    session._board["entries"] = board(("Ada", 12.0, today), ("Bo", 13.0, today),
                                      ("Ada", 14.0, today))
    session.handle_message(P.parse_client({"type": "board", "action": "remove",
                                           "name": "Ada", "lap_s": 14.0}))
    assert [(e["name"], e["lap_s"]) for e in session._board["entries"]] == [
        ("Ada", 12.0), ("Bo", 13.0)]
    msgs = [m for m in session.drain_outbox() if m["type"] == "leaderboard"]
    assert msgs and len(msgs[-1]["entries"]) == 2


def test_clear_today_keeps_older_laps_and_references(session):
    today = time.strftime("%Y-%m-%d")
    session._board["entries"] = board(("Ada", 12.0, today), ("Old", 11.0, "2026-01-01"))
    session._board["references"] = {"quantum": {"driver": "x", "lap_s": 12.7}}
    session.handle_message(P.parse_client({"type": "board", "action": "clear_today"}))
    assert [e["name"] for e in session._board["entries"]] == ["Old"]
    session.handle_message(P.parse_client({"type": "board", "action": "clear"}))
    assert session._board["entries"] == []
    assert session._board["references"], "AI reference laps stay"


def test_remove_a_studio_entry(session, tmp_path):
    board_dir = session.studio.board_dir
    entries = [{"name": "Ada", "lapped": 12, "eval_episodes": 12, "mean_lap": 13.0},
               {"name": "Bo", "lapped": 10, "eval_episodes": 12, "mean_lap": 14.0}]
    studio_mod.save_board("oval", entries, board_dir)
    session.handle_message(P.parse_client({"type": "board", "action": "remove",
                                           "board": "studio", "track": "oval",
                                           "name": "Bo", "index": 1}))
    assert [e["name"] for e in studio_mod.load_board("oval", board_dir)] == ["Ada"]


def test_board_messages_parse_strictly():
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "board", "action": "remove"})  # which entry?
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "board", "action": "wipe"})


def test_moderation_only_from_the_booth_machine(monkeypatch):
    from fastapi.testclient import TestClient

    from racetraq.server.app import create_app

    monkeypatch.setattr(ws_mod, "LOCAL_HOSTS", set())  # pretend: a phone on the LAN
    client = TestClient(create_app(load_config()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "board", "action": "clear"})
        for _ in range(20):
            msg = ws.receive_json()
            if msg["type"] == "error":
                assert "booth machine" in msg["message"]
                break
        else:
            raise AssertionError("no refusal")
