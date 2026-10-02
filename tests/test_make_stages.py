"""tools/make_stages.py: evolution stages are snapshots of the bundled driver's
own training run.

The tool replays the run a driver's sidecar records through
``train_headless.train`` (the code path that trained it), keeps the parameters
of every snapshot eval, checks that the replay's best snapshot IS the driver
and writes four eval-ordered stages plus the pre-first-lap warm-start
checkpoint.  The stage selection is tested on made-up scores; the replay and
the files on tiny real runs (25-decision episodes — nothing laps there, and
the untrained policy often scores best, so those tests pick their snapshots
with a stub instead of the improving-chain rule).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from traqmania import train_headless
from traqmania.agents.training import DQNTrainer
from traqmania.server.runtime import WEIGHTS_DIR, weights_circuit

REPO_ROOT = Path(__file__).resolve().parents[1]
INF = float("inf")
# tiny runs: 25-decision episodes, a 5-episode snapshot eval every 4 episodes
SHORT = "reward.max_decisions=25"
FAST = [SHORT, "training.eval_episodes=5", "training.eval_every=4"]
# 40 episodes: at least 5 snapshot evals even when all 8 sub-envs time out together
EPISODES, SEED = 40, 3


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("traqmania_make_stages",
                                                  REPO_ROOT / "tools" / "make_stages.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def meta_of(npz: Path) -> dict:
    return json.loads(npz.with_suffix("").with_suffix(".meta.json").read_text(encoding="utf-8"))


def train_driver(weights: Path, seed: int = SEED, episodes: int = EPISODES) -> dict:
    """A tiny quantum oval driver in ``weights``, trained as the studies train
    one: ``train_headless.train`` (its sidecar records the resolved recipe)."""
    weights.mkdir(exist_ok=True)
    return train_headless.train("quantum", "oval", episodes, seed, None,
                                out_dir=str(weights), overrides=FAST)


@pytest.fixture()
def plumbing(tool, monkeypatch):
    """Snapshot pickers that do not depend on how a tiny run happens to
    learn: the first three snapshots that are not the best one, then the
    best; the first snapshot as the warm start."""

    def pick(scores, exact=None):
        best = tool.best_index(scores)
        return [*[i for i in range(len(scores)) if i != best][:3], best]

    monkeypatch.setattr(tool, "select_stages", pick)
    monkeypatch.setattr(tool, "warmstart_choice", lambda lapped: (0, 1))
    return pick


# ------------------------------------------------------------ the selection


def test_eval_score_orders_like_the_trainer(tool):
    lapping = {"lapped_episodes": 3, "mean_lap": 14.0, "mean_return": 100.0}
    faster = {"lapped_episodes": 3, "mean_lap": 13.0, "mean_return": 90.0}
    reliable = {"lapped_episodes": 4, "mean_lap": 20.0, "mean_return": 10.0}
    no_lap = {"lapped_episodes": 0, "mean_lap": None, "mean_return": 50.0}
    further = {"lapped_episodes": 0, "mean_lap": None, "mean_return": 60.0}
    scores = [tool.eval_score(e) for e in (no_lap, further, lapping, faster, reliable)]
    assert scores == sorted(scores) and len(set(scores)) == 5
    assert tool.eval_score(no_lap) == (0, -INF, 50.0)
    # the earliest of equal scores is the best, as the trainer keeps it
    assert tool.best_index([(0, -INF, 1.0), (2, -13.0, 5.0), (2, -13.0, 5.0)]) == 1


def test_select_stages_takes_an_improving_chain_ending_at_the_best(tool):
    # 0 laps (improving return), first lap at index 3, churn, best at index 8
    scores = [(0, -INF, 10.0), (0, -INF, 30.0), (0, -INF, 20.0), (2, -15.0, 300.0),
              (0, -INF, 40.0), (7, -14.0, 500.0), (12, -13.5, 900.0), (5, -13.0, 400.0),
              (12, -12.7, 950.0), (1, -14.0, 100.0)]
    picked = tool.select_stages(scores)
    assert len(picked) == 4 and picked == sorted(set(picked))
    assert picked[-1] == tool.best_index(scores) == 8
    assert [scores[i] for i in picked] == sorted(scores[i] for i in picked)
    # four lapping chain members exist (3, 5, 6, 8): the story starts at the first lap
    assert picked == [3, 5, 6, 8]

    # only two lapping snapshots on the chain: earlier, lap-less ones fill in
    scores = [(0, -INF, 10.0), (0, -INF, 30.0), (0, -INF, 50.0), (0, -INF, 40.0),
              (3, -25.0, 300.0), (12, -23.0, 900.0), (0, -INF, 5.0)]
    picked = tool.select_stages(scores)
    assert picked[-2:] == [4, 5] and picked[0] == 0 and len(set(picked)) == 4
    assert [scores[i] for i in picked] == sorted(scores[i] for i in picked)

    # the best snapshot comes too early for four improving stages: say so
    with pytest.raises(RuntimeError, match="only 2 strictly improving snapshots"):
        tool.select_stages([(0, -INF, 1.0), (5, -13.0, 9.0), (0, -INF, 2.0), (1, -14.0, 3.0)])


def test_select_stages_also_respects_the_exact_simulator_eval(tool):
    """The case this exists for (the bundled chicane run): the trainer's
    12-episode eval ranks a snapshot above an earlier one that the 36-episode
    exact eval puts well below it — evolution mode would show the "later" car
    crashing more often than the "earlier" one."""
    scores = [(0, -INF, 90.0), (0, -INF, 235.0), (8, -13.12, 742.0), (9, -13.05, 812.0),
              (9, -12.71, 1218.0), (12, -12.64, 1924.0), (11, -12.71, 1472.0)]
    exact = [(0, -INF), (0, -INF), (35, -12.98), (20, -12.83), (35, -12.58), (36, -12.63),
             (36, -12.70)]
    assert tool.select_stages(scores) == [2, 3, 4, 5]  # the trainer's evals alone
    picked = tool.select_stages(scores, exact)
    # index 3 (20/36 after 35/36) cannot follow index 2: the chain is
    # 0, 1, 2, 4, 5, of which four are taken evenly
    assert picked == [0, 1, 4, 5]
    assert [exact[i] for i in picked] == sorted(exact[i] for i in picked)
    assert [scores[i] for i in picked] == sorted(scores[i] for i in picked)
    # equal exact scores may follow each other (two snapshots that never lap);
    # a worse one may not
    assert tool.select_stages(scores[:2] + scores[4:6] + scores[:0],
                              [(0, -INF), (0, -INF), (35, -12.58), (36, -12.63)]) \
        == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="only 3 strictly improving.*exact-simulator"):
        tool.select_stages(scores[:2] + scores[4:6],
                           [(0, -INF), (3, -14.0), (2, -12.58), (36, -12.63)])
    assert tool.exact_score({"lapped": 35, "mean_lap": 12.98}) == (35, -12.98)
    assert tool.exact_score({"lapped": 0, "mean_lap": None}) == (0, -INF)


def test_warmstart_is_the_last_lapless_snapshot_before_the_breakthrough(tool):
    # lapped share per snapshot (the larger one of its two evals)
    assert tool.BREAKTHROUGH == 0.5
    # the plain case: nothing laps, then the run gets it — a pre-first-lap checkpoint
    assert tool.warmstart_choice([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]) == (2, 3)
    # a few early laps that are lost again are not the breakthrough (the
    # bundled combo run: 2/12 at episode 600, reliable from 1100 on)
    assert tool.warmstart_choice([0.0, 0.17, 0.0, 0.08, 0.28, 0.0, 0.67, 0.0, 1.0]) == (5, 6)
    # the snapshot right before the breakthrough may lap a little: step back
    # to the last one that does not lap at all
    assert tool.warmstart_choice([0.0, 0.0, 0.25, 0.5]) == (1, 3)
    assert tool.warmstart_choice([0.5, 0.0, 1.0]) is None  # lapped at once
    assert tool.warmstart_choice([0.1, 0.2, 0.9]) is None  # no lap-less snapshot before
    assert tool.warmstart_choice([0.0, 0.0, 0.4]) is None  # never broke through
    assert tool.warmstart_choice([0.0, 0.3, 0.4], breakthrough=0.25) == (0, 1)


# ---------------------------------------------------------------- the replay


def test_recording_trainer_keeps_every_eval_and_changes_nothing(tool, tmp_path):
    plain = train_driver(tmp_path / "plain")
    record: list[dict] = []
    with tool.recorded_evals(record):
        recorded = train_driver(tmp_path / "recorded")
    assert train_headless.DQNTrainer is DQNTrainer  # the swap is undone

    # the same run, bit for bit — recording draws nothing and evaluates nothing extra
    best = np.load(recorded["weights_path"])["params"]
    np.testing.assert_array_equal(best, np.load(plain["weights_path"])["params"])
    assert recorded["episode_returns"] == plain["episode_returns"]
    # one record per eval-log entry, with the parameters it scored
    assert [{k: v for k, v in entry.items() if k != "params"} for entry in record] \
        == recorded["eval_log"]
    assert len(record) >= 5
    scores = [tool.eval_score(entry) for entry in record]
    index = tool.best_index(scores)
    assert record[index]["episode"] == recorded["best_eval"]["episode"]
    np.testing.assert_array_equal(record[index]["params"], best)  # what the run saved


def test_driver_recipe_comes_from_the_sidecar(tool, tmp_path):
    weights = tmp_path / "weights"
    train_driver(weights)
    recipe = tool.driver_recipe(weights / "quantum_oval.npz")
    training = meta_of(weights / "quantum_oval.npz")["training"]
    assert (recipe["seed"], recipe["episodes"]) == (SEED, EPISODES)
    assert {k: v for k, v in recipe["overrides"].items() if k.startswith("training.")} \
        == {f"training.{k}": v for k, v in training.items()}
    assert recipe["overrides"]["circuit.n_layers"] == 4
    assert "circuit.n_actions" not in recipe["overrides"]  # the default: not forced
    assert recipe["overrides"]["observation.ray_angles_deg"] == [-60.0, 0.0, 60.0]
    assert recipe["overrides"]["observation.features"] == ["rays", "speed"]

    # no sidecar, or one without a training table: nothing to replay
    np.savez(tmp_path / "quantum_bare.npz", params=np.zeros(56))
    assert tool.driver_recipe(tmp_path / "quantum_bare.npz") is None
    (tmp_path / "quantum_bare.meta.json").write_text(json.dumps({"episodes": 800}))
    assert tool.driver_recipe(tmp_path / "quantum_bare.npz") is None


def test_every_bundled_stage_family_has_a_replayable_driver(tool):
    """The stage families that ship: each driver's sidecar must carry what the
    replay needs (a pre-October-2026 sidecar did not)."""
    for name, profile in (("oval", None), ("chicane", None), ("gp", None), ("combo", None),
                          ("oval_q6", "q6")):
        driver = WEIGHTS_DIR / f"quantum_{name}.npz"
        recipe = tool.driver_recipe(driver, profile)
        assert recipe is not None, driver.name
        meta = meta_of(driver)
        assert recipe["episodes"] == meta["episodes"] == meta["training"]["episodes"]
        if "selection" in meta:
            assert recipe["seed"] == meta["selection"]["chosen_seed"], driver.name
        assert recipe["overrides"]["circuit.n_layers"] == weights_circuit(driver)["n_layers"]


def test_replay_reproduces_the_driver_and_ends_on_it(tool, plumbing, tmp_path, capsys):
    weights = tmp_path / "weights"
    summary = train_driver(weights)
    driver = weights / "quantum_oval.npz"
    driver_bytes = driver.read_bytes()
    driver_meta = meta_of(driver)

    # everything but the short episodes comes from the driver's sidecar
    paths = tool.make_stages("oval", weights_dir=weights, overrides=[SHORT],
                             exact_episodes=2)
    out = capsys.readouterr().out
    assert f"replaying the run behind quantum_oval.npz: seed {SEED}, {EPISODES} episodes" in out
    assert "== quantum_oval.npz, parameter for parameter" in out
    assert [p.name for p in paths] == [f"quantum_oval_stage{i}.npz" for i in range(1, 5)]
    assert driver.read_bytes() == driver_bytes  # the driver itself is not rewritten

    log = summary["eval_log"]
    picked = plumbing([tool.eval_score(entry) for entry in log])
    assert log[picked[-1]]["episode"] == summary["best_eval"]["episode"]
    # the last stage IS the driver: the same file, byte for byte
    assert paths[-1].read_bytes() == driver_bytes
    for stage, (path, index) in enumerate(zip(paths, picked, strict=True), start=1):
        meta = meta_of(path)
        entry = log[index]
        assert (meta["agent"], meta["track"], meta["stage"]) == ("quantum", "oval", stage)
        assert meta["episodes"] == entry["episode"]  # the "ep N" car label
        assert meta["eval_episodes"] == 5
        assert meta["eval_lapped"] == entry["lapped_episodes"]
        assert meta["eval_mean_return"] == pytest.approx(entry["mean_return"], abs=1e-3)
        assert meta["eval_mean_lap"] is None and meta["eval_acting_noise"] is not None
        assert meta["run"] == {
            "seed": SEED, "episodes": EPISODES,
            "best_episode": summary["best_eval"]["episode"], "driver": "quantum_oval.npz",
            "driver_sha256": tool._sha256(driver), "reproduces_driver": True}
        # the blocks of a train_headless sidecar, as the driver has them
        for block in ("circuit", "observation", "actions", "training"):
            assert meta[block] == driver_meta[block]
        assert meta["exact_eval"]["episodes"] == 2
        assert f"snapshot at episode {entry['episode']} of the training run that produced " \
               "quantum_oval.npz" in meta["provenance"]
        assert weights_circuit(path) == weights_circuit(driver)
    assert "a copy of quantum_oval.npz" in meta_of(paths[-1])["provenance"]

    # the warm-start checkpoint: a snapshot of the same run, with the same blocks
    warm = weights / "quantum_oval_warmstart.npz"
    meta = meta_of(warm)
    assert meta["episodes"] == log[0]["episode"] and "stage" not in meta
    assert meta["run"]["reproduces_driver"] is True
    assert meta["circuit"] == driver_meta["circuit"]
    assert meta["warmstart"] == {"breakthrough_episode": log[1]["episode"],
                                 "breakthrough_fraction": 0.5,
                                 "earlier_lapping_episodes": []}
    assert "a pre-first-lap checkpoint" in meta["provenance"]
    assert np.load(warm)["params"].shape == np.load(driver)["params"].shape
    assert sorted(p.name for p in weights.glob("*.npz")) == sorted(
        ["quantum_oval.npz", "quantum_oval_warmstart.npz", *(p.name for p in paths)])


def test_replay_that_misses_the_driver_writes_nothing(tool, plumbing, tmp_path, capsys):
    weights = tmp_path / "weights"
    train_driver(weights)
    driver = weights / "quantum_oval.npz"
    before = sorted(p.name for p in weights.iterdir())

    # another seed is another run: refuse, and leave the directory alone
    with pytest.raises(RuntimeError, match="is NOT quantum_oval.npz.*nothing written"):
        tool.make_stages("oval", seed=SEED + 1, weights_dir=weights, overrides=[SHORT],
                         exact_episodes=0)
    assert sorted(p.name for p in weights.iterdir()) == before

    # --allow-mismatch: stages of the other run, ending on a copy of the driver,
    # and every sidecar says that this is not the driver's run
    capsys.readouterr()
    paths = tool.make_stages("oval", seed=SEED + 1, weights_dir=weights, overrides=[SHORT],
                             exact_episodes=0, allow_mismatch=True)
    assert "WARNING: the replay's best snapshot" in capsys.readouterr().out
    assert paths[-1].read_bytes() == driver.read_bytes()
    metas = [meta_of(path) for path in paths]
    assert all(meta["run"]["reproduces_driver"] is False for meta in metas)
    assert all(meta["run"]["seed"] == SEED + 1 for meta in metas)
    assert all("did NOT reproduce quantum_oval.npz" in meta["provenance"] for meta in metas)
    assert "NOT a snapshot of the run the earlier stages come from" in metas[-1]["provenance"]
    assert metas[-1]["episodes"] == meta_of(driver)["episodes"]
    assert "eval_lapped" not in metas[-1]  # no snapshot eval of this run to report


def test_fresh_run_ends_on_its_own_best_snapshot(tool, plumbing, tmp_path, capsys):
    weights = tmp_path / "weights"
    weights.mkdir()
    # no driver in the directory: the config's recipe, from the given seed
    paths = tool.make_stages("oval", seed=SEED, episodes=EPISODES, weights_dir=weights,
                             overrides=FAST, exact_episodes=0)
    assert "fresh run (no replayable quantum_oval.npz)" in capsys.readouterr().out
    # ... which is the run train_headless makes of that recipe, bit for bit
    reference = train_driver(tmp_path / "reference")
    np.testing.assert_array_equal(np.load(paths[-1])["params"],
                                  np.load(reference["weights_path"])["params"])
    meta = meta_of(paths[-1])
    assert meta["run"] == {"seed": SEED, "episodes": EPISODES,
                           "best_episode": reference["best_eval"]["episode"]}
    assert meta["provenance"].startswith(
        f"snapshot at episode {meta['episodes']} of a tools/make_stages.py run")

    # --fresh ignores a driver that is there
    train_driver(weights, seed=SEED + 1)
    paths = tool.make_stages("oval", seed=SEED, episodes=EPISODES, weights_dir=weights,
                             overrides=FAST, exact_episodes=0, fresh=True)
    assert "fresh run (--fresh)" in capsys.readouterr().out
    assert "driver" not in meta_of(paths[-1])["run"]
    np.testing.assert_array_equal(np.load(paths[-1])["params"],
                                  np.load(reference["weights_path"])["params"])
