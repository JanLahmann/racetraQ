"""Training studio controller (#29): the session side of studio mode.

Phases: ``setup`` (the visitor picks track, qubits, sensors, action set and
warm start) -> ``training`` (a live quantum training job, stopped at the time
limit, the recipe's episode budget or the visitor's "stop") -> ``done`` (the
best snapshot is the visitor's model: test result, study comparison, booth
board) -> ``race`` (you vs your model) / ``watch`` (your model drives alone,
live qubit gauges). The pure parts live in :mod:`racetraq.studio`.
"""

from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from racetraq import studio
from racetraq.server import protocol
from racetraq.server.runtime import LEADERBOARD_DIR

if TYPE_CHECKING:
    from racetraq.server.session import DemoSession

DEFAULT_TIME_LIMIT_S = 300.0
STATUS_EVERY_S = 0.5
# Training runs until the time limit unless the model is done sooner: its
# best test laps every test drive and this many further tests (50 episodes
# apart) brought no improvement.
CONVERGED_AFTER_EVALS = 6
MAX_EPISODES = 100_000  # effectively unbounded: the time limit ends the run


class StudioController:
    def __init__(self, session: DemoSession) -> None:
        self.s = session
        self.phase = "setup"
        self.spec: dict | None = None  # the visitor's choices for the current run
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.stop_reason: str | None = None
        self.model: Any = None  # trained Q-function (best snapshot)
        self.result: dict | None = None
        self.rank: int | None = None
        self.config_changed = False  # session config is a studio config
        self._last_status = -math.inf
        self.best_test: dict | None = None  # best greedy test so far (live)
        self._evals_seen = 0
        self._stale_evals = 0
        self.stats = studio.load_stats()

    # ------------------------------------------------------------ helpers

    @property
    def time_limit_s(self) -> float:
        return float(self.s.config.get("studio", {}).get("time_limit_s", DEFAULT_TIME_LIMIT_S))

    @property
    def board_dir(self):
        return self.s._leaderboard_dir if self.s._leaderboard_dir is not None else LEADERBOARD_DIR

    def _warm_available(self) -> dict:
        from racetraq.server.session import quantum_weights_path

        return {studio.stats_key(t, n): quantum_weights_path(t, n, "_warmstart").is_file()
                for t in studio.STUDIO_TRACKS for n in studio.STUDIO_QUBITS}

    def _job(self):
        return self.s.jobs.get("quantum")

    # ------------------------------------------------------------- entry

    def enter(self) -> None:
        """Mode switched to studio: show the setup (or the finished model)."""
        self.s.mode = "studio"
        self.s.cars = []
        if self.phase in ("race", "watch"):
            self.phase = "done"
        if self.phase not in ("done",):
            self.phase = "setup"
        self.emit(full=True)

    def leave(self) -> None:
        """Mode switched away: stop a running job, give the session its
        profile config back (studio sensors / action sets are studio-only)."""
        if self.phase == "training":
            self.s.stop_training()
            self.phase = "setup"
        elif self.phase in ("race", "watch"):
            self.phase = "done"
        if self.config_changed:
            self.s._apply_profile(self.s.n_qubits)
            self.config_changed = False

    def reset(self) -> None:
        """Forget the last visitor's model and result (the booth went idle):
        the next visitor opening the studio starts at the setup."""
        if self.phase == "training":
            self.s.stop_training()
        self.phase = "setup"
        self.spec = None
        self.model = None
        self.result = None
        self.rank = None
        self.best_test = None
        self.stop_reason = None
        self.started_at = self.stopped_at = None
        self.emit(full=True)

    def handle(self, msg: protocol.Studio) -> None:
        action = msg.action
        if action == "start":
            self._start(msg)
        elif action == "stop":
            if self.phase == "training":
                self.stop_reason = "stopped"
                self.s.stop_training()
        elif action in ("race", "watch"):
            if self.model is None:
                self.s._error("train a model first")
                return
            self._drive(action)
        elif action == "result":
            if self.model is not None and self.phase in ("race", "watch", "done"):
                self.phase = "done"
                self.s.cars = []
                self.emit(full=True)
        elif action == "setup":
            if self.phase == "training":
                self.s.stop_training()
            self.phase = "setup"
            self.s.cars = []
            self.emit(full=True)

    # ----------------------------------------------------------- training

    def _start(self, msg: protocol.Studio) -> None:
        s = self.s
        if self.phase == "training" or s._training_alive():
            s._error("a training run is already in progress")
            return
        track = msg.track or s.track_name
        if track not in studio.STUDIO_TRACKS:
            s._error(f"the studio trains on {', '.join(studio.STUDIO_TRACKS)}")
            return
        n, sensors, n_actions = msg.qubits, msg.sensors, msg.actions
        problem = studio.option_problem(n, sensors, n_actions)
        if problem is not None:
            s._error(problem)
            return
        warm = bool(msg.warm)
        if warm and not (studio.is_studied(sensors, n_actions, False)
                         and self._warm_available().get(studio.stats_key(track, n))):
            s._error("warm start needs lidar sensors, 4 actions and a bundled "
                     "warm-start checkpoint for this track and size")
            return
        if not s._set_track(track):
            return
        s.stop_training()
        self._use_config(n, sensors, n_actions, broadcast=False)
        s.driver = "auto"  # warm start resolves quantum_<track>_warmstart[_q<n>]
        s.mode = "studio"
        s.cars = []
        s.jobs = {}
        s._start_job("quantum", warm=warm, episodes=MAX_EPISODES,
                     seed_offset=random.randrange(1, 1_000_000))
        job = self._job()
        self.spec = {"track": track, "qubits": n, "sensors": sensors, "actions": n_actions,
                     "warm": warm, "name": s.racer_name or None,
                     "n_params": int(job.trainer.qfunc.get_params().size),
                     "n_layers": int(s.config["circuit"].get("n_layers", 4)),
                     "studied": studio.is_studied(sensors, n_actions, warm)}
        self.phase = "training"
        self.started_at = time.monotonic()
        self.stopped_at = None
        self.stop_reason = None
        self.model = None
        self.result = None
        self.rank = None
        self.best_test = None
        self._evals_seen = 0
        self._stale_evals = 0
        s._outbox.append(s.welcome_payload())  # the circuit panel shows the studio circuit
        self.emit(full=True)

    def _use_config(self, n: int, sensors: str, n_actions: int, broadcast: bool = True) -> None:
        """Put the session on the studio config of a run (q<n> profile with
        the run's sensors and action set)."""
        s = self.s
        s._apply_profile(n, broadcast=False)
        s._apply_config(studio.studio_config(s.config, n, sensors, n_actions))
        s._agent_cache.clear()
        self.config_changed = True
        if broadcast:
            s._outbox.append(s.welcome_payload())

    def _use_model_config(self) -> bool:
        """The finished model drives under its own track and config (the
        session may have moved on while the visitor looked elsewhere)."""
        spec = self.spec
        s = self.s
        if spec["track"] != s.track_name and not s._set_track(spec["track"]):
            return False
        if not self.config_changed:
            self._use_config(spec["qubits"], spec["sensors"], spec["actions"])
        return True

    def tick(self) -> None:
        s = self.s
        if self.phase == "training":
            s._tick_train()
            job = self._job()
            now = time.monotonic()
            if job is not None and self.stop_reason is None:
                self._watch_tests(job)
            if (self.stop_reason is None and self.started_at is not None
                    and now - self.started_at >= self.time_limit_s):
                self.stop_reason = "time"
                s.stop_training()
            if job is not None and job.announced:
                self._finish(job)
            elif now - self._last_status >= STATUS_EVERY_S:
                self.emit()
        elif self.phase in ("race", "watch"):
            if (s._substep - 1) % s.substeps_per_decision == 0:
                s._decide()
            s._step_cars()

    def _watch_tests(self, job) -> None:
        """Follow the trainer's periodic greedy tests: keep the best (the
        trainer's own ranking: test drives lapped, then mean lap) and stop
        once it laps every drive and no longer improves."""
        trainer = job.trainer
        last = getattr(trainer, "last_eval", None)
        if last is None or last.get("episode", 0) <= self._evals_seen:
            return
        self._evals_seen = int(last["episode"])

        def score(ev: dict) -> tuple:
            mean_lap = ev.get("mean_lap")
            return (ev["lapped_episodes"], -(mean_lap if mean_lap is not None else math.inf))

        if self.best_test is None or score(last) > score(self.best_test):
            self.best_test = dict(last)
            self._stale_evals = 0
        else:
            self._stale_evals += 1
        best = self.best_test
        if (best["lapped_episodes"] >= best["eval_episodes"]
                and self._stale_evals >= CONVERGED_AFTER_EVALS):
            self.stop_reason = "converged"
            self.s.stop_training()

    def _finish(self, job) -> None:
        s = self.s
        self.stopped_at = time.monotonic()
        history = job.history or {}
        if self.stop_reason is None:
            self.stop_reason = "error" if job.error else "budget"
        with job.lock:
            first_lap = job.first_lap_episode
            episodes = job.episode + 1
        best = history.get("best_eval")
        self.model = job.trainer.qfunc if job.error is None else None
        self.result = {
            "episodes": int(episodes),
            "seconds": round(self.stopped_at - (self.started_at or self.stopped_at), 1),
            "first_lap": first_lap,
            "best_eval": best,
            "stop_reason": self.stop_reason,
            "error": job.error,
            "comparison": (studio.compare(self.stats, self.spec["track"], self.spec["qubits"],
                                          first_lap, best)
                           if self.spec["studied"] else None),
        }
        self.rank = None
        name = s.racer_name or None
        if name and best and best.get("lapped_episodes"):
            self.spec["name"] = name
            entries = studio.load_board(self.spec["track"], self.board_dir)
            entry = {"name": name, "qubits": self.spec["qubits"],
                     "sensors": self.spec["sensors"], "actions": self.spec["actions"],
                     "warm": self.spec["warm"], "episodes": int(episodes),
                     "first_lap": first_lap, "lapped": int(best["lapped_episodes"]),
                     "eval_episodes": int(best["eval_episodes"]),
                     "mean_lap": None if best.get("mean_lap") is None
                     else round(float(best["mean_lap"]), 3),
                     "best_lap": None if best.get("best_lap") is None
                     else round(float(best["best_lap"]), 3),
                     "seconds": self.result["seconds"],
                     "date": time.strftime("%Y-%m-%d")}
            entries, self.rank = studio.add_entry(entries, entry)
            studio.save_board(self.spec["track"], entries, self.board_dir)
        self.phase = "done"
        s.jobs = {}
        self.emit(full=True)

    # ------------------------------------------------------------ driving

    def _drive(self, action: str) -> None:
        from racetraq.server.session import _Car

        s = self.s
        if not self._use_model_config():
            return
        s.mode = "studio"
        x, y, theta = s.track.start_pose()
        label = f"your circuit{' · ' + self.spec['name'] if self.spec.get('name') else ''}"
        car = _Car(id="studio", kind="quantum", state=np.array([x, y, theta, 0.0]),
                   qfunc=self.model, label=label)
        s._respawn(car)
        cars = [car]
        if action == "race":
            human = _Car(id="human", kind="human", state=np.array([x, y, theta, 0.0]))
            s._respawn(human)
            cars = [human, car]
        s.cars = cars
        self.phase = action
        self.emit(full=True)

    # ------------------------------------------------------------- status

    def status_payload(self, full: bool = False) -> dict:
        job = self._job()
        live: dict[str, Any] = {}
        if self.phase == "training" and job is not None:
            with job.lock:
                live = {"episode": int(job.episode + 1) if job.returns else 0,
                        "first_lap": job.first_lap_episode,
                        "best_lap_s": job.best_lap_s}
            last_eval = getattr(job.trainer, "last_eval", None)
            if last_eval is not None:
                live["last_eval"] = last_eval
            if self.best_test is not None:
                live["best_test"] = self.best_test
        payload: dict[str, Any] = {
            "type": "studio",
            "phase": self.phase,
            "spec": self.spec,
            "time_limit_s": self.time_limit_s,
            "elapsed_s": None if self.started_at is None else round(
                (self.stopped_at or time.monotonic()) - self.started_at, 1),
            "live": live or None,
            "result": self.result,
            "rank": self.rank,
            "name": self.s.racer_name or None,
        }
        track = (self.spec or {}).get("track") or self.s.track_name
        if track in studio.STUDIO_TRACKS:
            payload["board"] = {"track": track,
                                "entries": studio.load_board(track, self.board_dir)[:10]}
        if full:
            payload["catalog"] = studio.catalog(self.stats, self._warm_available(),
                                                self.time_limit_s)
        return payload

    def emit(self, full: bool = False) -> None:
        self._last_status = time.monotonic()
        self.s._outbox.append(self.status_payload(full))
