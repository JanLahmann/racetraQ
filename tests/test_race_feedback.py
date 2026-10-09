"""Race feedback for visitors: a "3, 2, 1, GO" start with the cars side by
side and the ghost waiting on the line, recovery near the spot where a car
left the track (the lap goes on, dirty), and a lap result with board rank."""

import numpy as np
import pytest

from racetraq.config import load_config
from racetraq.server import protocol as P
from racetraq.server.session import RACE_COUNTDOWN_S, DemoSession


@pytest.fixture()
def session(tmp_path):
    s = DemoSession(load_config(), ghosts_dir=tmp_path)
    s.drain_outbox()
    return s


def start_race(session, opponent="quantum"):
    session.handle_message(P.Race(action="start", opponent=opponent))
    assert session.mode == "race"


def skip_countdown(session):
    while session.race_countdown() is not None:
        session.tick()


def human(session):
    return next(c for c in session.cars if c.kind == "human")


def events(session, kind):
    return [m for m in session.drain_outbox() if m["type"] == "event" and m["kind"] == kind]


def test_countdown_holds_the_grid_then_go(session):
    start_race(session)
    assert session.race_countdown() == pytest.approx(RACE_COUNTDOWN_S)
    session.handle_message(P.Input(keys=P.KEY_THROTTLE))
    for _ in range(30):
        session.tick()
    assert human(session).state[3] == 0.0, "nobody moves before GO"
    states = [m for m in session.drain_outbox() if m["type"] == "state"]
    assert states and "countdown" in states[-1]
    P.parse_server(states[-1])
    skip_countdown(session)
    for _ in range(12):
        session.tick()
    car = human(session)
    assert car.state[3] > 0.0
    payload = session._car_payload(car)
    assert 0.0 < payload["lap_t"] < 1.0, "the lap clock starts at GO"
    last = [m for m in session.drain_outbox() if m["type"] == "state"][-1]
    assert "countdown" not in last


def test_cars_start_side_by_side(session):
    start_race(session)
    a, b = (c.state[:2] for c in session.cars)
    gap = float(np.linalg.norm(a - b))
    assert gap == pytest.approx(0.8 * session.track.half_width, rel=1e-6)


def test_race_reset_counts_down_again(session):
    start_race(session)
    skip_countdown(session)
    session.handle_message(P.Race(action="reset", opponent="quantum"))
    assert session.race_countdown() == pytest.approx(RACE_COUNTDOWN_S)


def test_off_track_human_recovers_nearby_and_keeps_the_lap(session):
    start_race(session, "mlp")
    skip_countdown(session)
    car = human(session)
    # drive a stretch, then shove the car off the track sideways
    session.handle_message(P.Input(keys=P.KEY_THROTTLE))
    for _ in range(120):
        session.tick()
    session.handle_message(P.Input(keys=0))
    progress = car.progress
    assert progress > 5.0
    nx, ny = session.track.normals[int(car.s / session.track.total_length
                                       * len(session.track.centerline))]
    car.state[:2] += np.array([nx, ny]) * session.track.half_width * 3
    while car.respawn_at is None:
        session.tick()
    while car.respawn_at is not None:
        session.tick()
    assert not car.off_track
    assert car.lap_dirty, "a recovered lap is not a clean lap"
    assert progress - 10.0 < car.progress <= progress, "back near where it left"
    assert car.state[3] < 1.0, "it resumes from (near) standstill"


def test_lap_result_ranks_named_and_unnamed_laps(session):
    start_race(session, "mlp")
    skip_countdown(session)
    session._board["entries"] = [{"name": "Ada", "lap_s": 12.0, "date": ""},
                                 {"name": "Bo", "lap_s": 15.0, "date": ""}]
    car = human(session)

    session.racer_name = ""
    session._lap_result(car, 13.5)
    (ev,) = events(session, "lap_result")
    P.parse_server(ev)
    assert (ev["clean"], ev["rank"], ev["named"], ev["board_size"]) == (True, 2, False, 2)

    session.racer_name = "Cy"
    session._record_leaderboard(car, 11.0)
    session._lap_result(car, 11.0)
    (ev,) = events(session, "lap_result")
    assert (ev["rank"], ev["named"]) == (1, True)

    car.lap_dirty = True
    session._lap_result(car, 10.0)
    (ev,) = events(session, "lap_result")
    assert ev["clean"] is False and ev["rank"] is None


def test_ghost_waits_on_the_line_until_go(session):
    session._ghost = {"lap_time": 10.0, "kind": "quantum", "driver": None,
                      "points": [[float(i), 0.0, 0.0] for i in range(50)]}
    start_race(session)
    for _ in range(20):
        session.tick()
    assert session._ghost_payload()["x"] == 0.0
    skip_countdown(session)
    for _ in range(30):
        session.tick()
    assert session._ghost_payload()["x"] > 0.0
