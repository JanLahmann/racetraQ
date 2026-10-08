"""tools/bundle_driver.py: selecting a bundled driver from a multi-seed study.

The studies here are synthetic: every cell has real weights and a real sidecar
(``train_headless.save_weights``) but made-up study numbers, and no training
runs.  Most tests script the rollouts (the first parameters of a cell's
weights say what its evals return), so ranking, the fresh re-evaluation and
the refusal rules are checked exactly; one test drives real (tiny) envs.
"""

import ast
import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL_PATH = Path(__file__).resolve().parent.parent / "tools" / "bundle_driver.py"
_spec = importlib.util.spec_from_file_location("racetraq_bundle_driver_tool", TOOL_PATH)
bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bundle)
study = bundle.study

STUDY_EPISODES, STUDY_SEED = 36, 20_000
TRACKS = ("oval", "chicane", "gp", "combo")
SHORT = "reward.max_decisions=25"  # real rollouts stay tiny


def test_tool_imports_only_numpy_and_stdlib_at_module_level():
    tree = ast.parse(TOOL_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots - set(sys.stdlib_module_names) == {"numpy"}


# ------------------------------------------------------------ synthetic study


def _log(*fractions, every=50, n=12):
    return [{"episode": every * (i + 1), "lapped_episodes": round(f * n), "eval_episodes": n}
            for i, f in enumerate(fractions)]


def _reliability(lapped, lap):
    return {"episodes": STUDY_EPISODES, "lapped_episodes": lapped, "laps": lapped,
            "mean_lap": lap if lapped else None, "best_lap": lap - 0.5 if lapped else None}


def make_cell(root, variant, seed, *, agent="mlp", track="oval", profile=None,
              overrides=(SHORT,), study_lapped=36, study_lap=14.0, final_lapped=18,
              fractions=(0, 1.0, 1.0), fresh=1.0, fresh_lap=14.0, device=0.0, init=None):
    """One finished cell.  The weights are a fresh Q-function of the cell's
    shape whose first parameters carry the script the ``scripted`` fixture
    plays back: [seed, fresh lapped fraction, fresh mean lap, study lapped,
    study mean lap, per-track fresh fractions (4), device lapped fraction]."""
    from racetraq.config import apply_overrides, load_config
    from racetraq.train_headless import build_qfunc, save_weights

    cell = study.cell_dir(root, variant, seed)
    cell.mkdir(parents=True)
    spec = {"agent": agent, "track": track, "profile": profile, "episodes": 40,
            "preset": "auto", "actions": None, "pace": False, "init": init,
            "variant": variant, "seed": seed, "overrides": list(overrides),
            "eval_episodes": STUDY_EPISODES, "eval_seed": STUDY_SEED}
    (cell / study.SPEC_NAME).write_text(json.dumps(spec), encoding="utf-8")

    config = apply_overrides(load_config(profile=profile), spec["overrides"])
    qfunc = build_qfunc(agent, int(config["circuit"]["n_qubits"]), seed, config)
    params = qfunc.get_params().copy()
    per_track = [fresh[name] for name in TRACKS] if isinstance(fresh, dict) else [0.0] * 4
    params[:10] = [seed, -1.0 if isinstance(fresh, dict) else fresh, fresh_lap,
                   study_lapped, study_lap, *per_track, device]
    qfunc.set_params(params)
    weights = save_weights(qfunc, agent, track, config, 40, out_dir=cell,
                           training_cfg={"gamma": 0.98, "episodes": 40, "seed": seed})
    final = weights.with_suffix("").with_suffix(".final.npz")
    np.savez(final, params=params)

    result = {"schema": 1, "variant": variant, "seed": seed, "agent": agent, "track": track,
              "profile": profile, "init": init, "overrides": {},
              "n_params": int(params.size),
              "wall_time_s": 1.0, "first_clean_episode": 60 if study_lapped else None,
              "eval_log": _log(*fractions),
              "best_snapshot_eval": _reliability(study_lapped, study_lap),
              "final_params_eval": _reliability(final_lapped, study_lap),
              "weights": weights.name, "final_weights": final.name}
    (cell / study.RESULT_NAME).write_text(json.dumps(result), encoding="utf-8")
    return weights


def write_manifest(root, variants):
    (root / study.MANIFEST_NAME).write_text(json.dumps({
        "schema": 1, "variants": {name: {"name": name, "overrides": [SHORT]}
                                  for name in variants},
        "seeds": [0, 1, 2, 3, 4], "eval": {"episodes": STUDY_EPISODES, "seed": STUDY_SEED},
        "runs": [{"git_commit": "abc1234", "command": "study.py run"}],
    }), encoding="utf-8")


@pytest.fixture
def scripted(monkeypatch):
    """Replace the rollouts by the script in each cell's weights; returns the
    list of evals asked for.  Envs and Q-functions are still built for real."""
    calls = []
    real_env = study._eval_env

    def tagged_env(spec, config):
        env = real_env(spec, config)
        env.test_tag = (spec["track"], spec["eval_seed"])
        return env

    def fake_eval(qfunc, env, max_steps):
        track, seed = env.test_tag
        p = qfunc.get_params()
        n = env.n_envs
        calls.append({"cell": int(p[0]), "track": track, "seed": seed, "episodes": n})
        if seed == STUDY_SEED:  # the study's own eval, replayed
            lapped, lap = int(round(p[3])), float(p[4])
        else:
            frac = p[5 + TRACKS.index(track)] if p[1] < 0 else p[1]
            lapped, lap = int(round(frac * n)), float(p[2])
        return {"episodes": n, "lapped_episodes": lapped, "laps": lapped,
                "mean_lap": lap if lapped else None, "best_lap": lap - 0.5 if lapped else None}

    monkeypatch.setattr(study, "_eval_env", tagged_env)
    monkeypatch.setattr(study, "greedy_eval", fake_eval)
    return calls


@pytest.fixture
def oval_study(tmp_path):
    """Five MLP seeds.  By the study's eval: 0 > 1 > 2 > 3 > 4.  On fresh
    episodes seed 0 halves, seeds 1-3 lap everything, seed 2 faster than 1."""
    root = tmp_path / "study_synth"
    make_cell(root, "v", 0, study_lapped=36, study_lap=13.0, fresh=0.5, fresh_lap=13.0,
              final_lapped=36, fractions=(0, 1.0, 1.0, 1.0))
    make_cell(root, "v", 1, study_lapped=36, study_lap=14.0, fresh=1.0, fresh_lap=14.5,
              final_lapped=18, fractions=(0, 1.0, 0.5, 0.0))
    make_cell(root, "v", 2, study_lapped=35, study_lap=12.0, fresh=1.0, fresh_lap=12.5,
              final_lapped=9, fractions=(0, 0, 0.5, 1.0))
    make_cell(root, "v", 3, study_lapped=18, study_lap=11.0, fresh=1.0, fresh_lap=11.0,
              final_lapped=0, fractions=(0, 0.5, 0, 0))
    make_cell(root, "v", 4, study_lapped=0, fresh=0.0, final_lapped=0, fractions=(0, 0, 0))
    make_cell(root, "other", 0, study_lapped=9, fresh=0.25, final_lapped=0)
    write_manifest(root, ["v", "other"])
    return root


def run(root, *extra, variant="v", name="mlp_oval"):
    return bundle.main(["--study", str(root), "--variant", variant, "--name", name,
                        "--eval-episodes", "20", *extra])


def sidecar(out_dir, name="mlp_oval"):
    return json.loads((out_dir / f"{name}.meta.json").read_text(encoding="utf-8"))


# -------------------------------------------------------------------- ranking


def test_study_key_orders_by_lapped_fraction_then_mean_lap():
    def result(seed, lapped, lap):
        return {"seed": seed, "best_snapshot_eval": _reliability(lapped, lap)}

    results = [result(0, 30, 12.0), result(1, 36, 15.0), result(2, 36, 14.0),
               result(3, 0, 0.0), result(4, 36, 14.0), result(5, 0, 0.0)]
    results[3]["best_snapshot_eval"]["mean_lap"] = float("nan")  # json's NaN: "no lap"
    order = [r["seed"] for r in sorted(results, key=bundle.study_key)]
    # all-lapping seeds first, the faster ahead (ties by seed); a fast 30/36 after them
    assert order == [2, 4, 1, 0, 3, 5]


def test_full_per_seed_table_is_printed_in_study_rank_order(oval_study, scripted, capsys):
    assert run(oval_study, "--dry-run") == 0
    out = capsys.readouterr().out
    table = out.split("### Seeds ranked")[1].split("Seed spread")[0]
    rows = [[c.strip() for c in line.split("|")[1:-1]] for line in table.splitlines()
            if line.startswith("| ") and not line.startswith("| rank")]
    assert [(row[0], row[1], row[2], row[3]) for row in rows] == [
        ("1", "0", "36/36", "13.0"), ("2", "1", "36/36", "14.0"), ("3", "2", "35/36", "12.0"),
        ("4", "3", "18/36", "11.0"), ("5", "4", "0/36", "—")]
    assert [row[-1] for row in rows] == ["candidate"] * 3 + ["", ""]
    assert "study_synth / v: mlp on oval, 5 seeds" in out


# ------------------------------------------------- fresh eval decides + sidecar


def test_fresh_eval_chooses_the_driver_and_is_what_the_sidecar_reports(
        oval_study, scripted, tmp_path, capsys):
    out_dir = tmp_path / "out"
    assert run(oval_study, "--out-dir", str(out_dir), "--note", "audit trial") == 0
    out = capsys.readouterr().out
    meta = sidecar(out_dir)
    sel = meta["selection"]

    # the study's number one (seed 0) laps half of the fresh episodes; seeds 1
    # and 2 lap all of them and seed 2 is the faster: chosen by the fresh eval
    assert sel["chosen_seed"] == 2
    assert "### Chosen: seed 2 -> mlp_oval" in out
    assert sel["fresh_eval"] == {"episodes": 20, "eval_seed": bundle.EVAL_SEED, "lapped": 20,
                                 "lapped_fraction": 1.0, "laps": 20, "mean_lap": 12.5,
                                 "best_lap": 12.0}
    # ...never the numbers it was shortlisted by (35/36 at 12.0 s)
    assert sel["candidates"] == [
        {"seed": 0, "study_lapped": 36, "study_episodes": 36, "study_mean_lap": 13.0,
         "study_eval_reproduced": True, "fresh_lapped": 10, "fresh_episodes": 20,
         "fresh_mean_lap": 13.0, "fresh_best_lap": 12.5},
        {"seed": 1, "study_lapped": 36, "study_episodes": 36, "study_mean_lap": 14.0,
         "study_eval_reproduced": True, "fresh_lapped": 20, "fresh_episodes": 20,
         "fresh_mean_lap": 14.5, "fresh_best_lap": 14.0},
        {"seed": 2, "study_lapped": 35, "study_episodes": 36, "study_mean_lap": 12.0,
         "study_eval_reproduced": True, "fresh_lapped": 20, "fresh_episodes": 20,
         "fresh_mean_lap": 12.5, "fresh_best_lap": 12.0},
    ]
    # only the top 3 were evaluated: the study's eval replayed, then fresh episodes
    assert [(c["cell"], c["seed"], c["episodes"]) for c in scripted] == [
        (0, STUDY_SEED, 36), (0, bundle.EVAL_SEED, 20), (1, STUDY_SEED, 36),
        (1, bundle.EVAL_SEED, 20), (2, STUDY_SEED, 36), (2, bundle.EVAL_SEED, 20)]

    # the bundled weights are the chosen cell's, byte for byte
    source = study.cell_dir(oval_study, "v", 2) / "mlp_oval.npz"
    assert (out_dir / "mlp_oval.npz").read_bytes() == source.read_bytes()
    assert sel["source"] == "cells/v/seed2/mlp_oval.npz"
    assert sel["weights_sha256"] == bundle._sha256(out_dir / "mlp_oval.npz")

    assert (sel["study"], sel["variant"], sel["n_seeds"]) == ("study_synth", "v", 5)
    assert str(tmp_path) not in json.dumps(meta)  # no absolute paths on record
    assert sel["overrides"] == [SHORT] and sel["profile"] is None
    assert sel["study_eval"] == {"episodes": 36, "eval_seed": STUDY_SEED}
    assert sel["study_commits"] == ["abc1234"]
    assert sel["note"] == "audit trial"
    assert "top 3 re-evaluated on 20 fresh distinct greedy episodes" in sel["rule"]
    assert "selection effect" in sel["rule"]

    provenance = meta["provenance"]
    assert isinstance(provenance, str) and "\n" not in provenance
    for text in ("best snapshot of seed 2", "5-seed study study_synth variant v",
                 "mlp on oval, 40 episodes", "laps in 20/20 episodes, best 12.0 s, mean 12.5 s",
                 "4/5 seeds lap in at least half", "audit trial"):
        assert text in provenance
    assert "35/36" not in provenance  # the shortlist number is not the claim
    assert provenance in out


def test_more_candidates_can_change_the_choice_and_one_candidate_has_no_choice(
        oval_study, scripted, tmp_path, capsys):
    # seed 3 is 4th by the study's eval but the fastest all-lapping one on fresh episodes
    assert run(oval_study, "--top", "4", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "### Chosen: seed 3 -> mlp_oval" in out
    printed = json.loads(out[out.index("{\n"):])
    assert printed["selection"]["chosen_seed"] == 3
    assert len(printed["selection"]["candidates"]) == 4
    # --top 1: the study's number one, whose fresh eval (10/20) misses --min-lapped
    scripted.clear()
    assert run(oval_study, "--top", "1", "--out-dir", str(tmp_path / "out")) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert {c["cell"] for c in scripted} == {0}
    assert not (tmp_path / "out").exists()


def test_seed_spread_statistics_in_the_sidecar(oval_study, scripted, tmp_path):
    out_dir = tmp_path / "out"
    assert run(oval_study, "--out-dir", str(out_dir)) == 0
    spread = sidecar(out_dir)["selection"]["seed_spread"]
    best = [1.0, 1.0, 35 / 36, 0.5, 0.0]
    final = [1.0, 0.5, 0.25, 0.0, 0.0]
    # stability = mean lapped fraction of the in-training evals after the first lapping
    # one: (1 + 1) / 2, (0.5 + 0) / 2, 1 / 1, (0 + 0) / 2, and 0 for the seed that never laps
    stability = [1.0, 0.25, 1.0, 0.0, 0.0]
    assert spread["n_seeds"] == 5
    assert spread["seeds_lapping_half"] == 4
    for key, values in (("best_snapshot_lapped", best), ("final_params_lapped", final),
                        ("stability", stability)):
        point, low, high = study.bootstrap_ci(values, statistic=study.iqm)
        assert spread[key] == {"iqm": round(point, 4), "ci_low": round(low, 4),
                               "ci_high": round(high, 4),
                               "median": round(float(np.median(values)), 4), "n": 5}
        assert spread[key]["ci_low"] <= spread[key]["iqm"] <= spread[key]["ci_high"]
    assert "bootstrap" in spread["interval"]


def test_sidecar_carries_the_cell_fields_a_real_date_and_feeds_the_runtime_readers(
        oval_study, scripted, tmp_path):
    from racetraq.server import runtime

    out_dir = tmp_path / "out"
    assert run(oval_study, "--out-dir", str(out_dir)) == 0
    meta = sidecar(out_dir)
    cell_meta = json.loads((study.cell_dir(oval_study, "v", 2) / "mlp_oval.meta.json")
                           .read_text(encoding="utf-8"))
    assert cell_meta["date"] == "DATE"  # the trainer's placeholder...
    assert datetime.date.fromisoformat(meta["date"]) == datetime.date.today()  # ...is gone
    for key in ("agent", "track", "config_hash", "episodes", "observation", "actions",
                "circuit", "training"):
        assert meta[key] == cell_meta[key]
    assert meta["training"] == {"gamma": 0.98, "episodes": 40, "seed": 2}
    assert set(meta) == set(cell_meta) | {"provenance", "selection"}

    weights = out_dir / "mlp_oval.npz"
    assert runtime.weights_observation(weights) == cell_meta["observation"]
    assert runtime.weights_actions(weights) == 4
    assert runtime.best_stage_label(weights) == "best (of 40 ep run)"
    # the study tool's own loader rule reads it too
    config = study.weights_config({"profile": None, "overrides": [], "actions": None}, weights)
    assert config["circuit"]["n_layers"] == cell_meta["circuit"]["n_layers"]
    json.loads(json.dumps(meta), parse_constant=pytest.fail)  # strict JSON: no NaN


# -------------------------------------------------------------------- refusals


def test_min_lapped_refuses_without_writing_unless_forced(oval_study, scripted, tmp_path,
                                                         capsys):
    out_dir = tmp_path / "out"
    # every candidate's fresh eval tops out at 20/20: an unreachable bar is not possible,
    # so shortlist the one seed that laps half of them
    args = ("--top", "1", "--out-dir", str(out_dir))
    assert run(oval_study, *args) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out and "10/20" in out and "--min-lapped 0.9" in out
    assert not out_dir.exists()
    # a dry run reports the same refusal
    assert run(oval_study, *args, "--dry-run") == 1
    assert not out_dir.exists()
    # a lower bar passes; --force passes the default one and says so
    assert run(oval_study, *args, "--min-lapped", "0.5") == 0
    assert sidecar(out_dir)["selection"]["fresh_eval"]["lapped"] == 10
    capsys.readouterr()
    assert run(oval_study, *args, "--force", "--overwrite") == 0
    assert "below --min-lapped 0.9; bundling it anyway (--force)" in capsys.readouterr().out
    assert sidecar(out_dir)["selection"]["chosen_seed"] == 0


def test_existing_target_is_kept_unless_overwrite(oval_study, scripted, tmp_path, capsys):
    out_dir = tmp_path / "out"
    assert run(oval_study, "--top", "1", "--force", "--out-dir", str(out_dir),
               "--note", "first bundle") == 0
    before = {p.name: p.read_bytes() for p in out_dir.iterdir()}
    assert sorted(before) == ["mlp_oval.meta.json", "mlp_oval.npz"]
    capsys.readouterr()
    scripted.clear()

    assert run(oval_study, "--out-dir", str(out_dir)) == 2
    captured = capsys.readouterr()
    assert "exists; pass --overwrite" in captured.err
    assert scripted == []  # refused before any evaluation
    assert {p.name: p.read_bytes() for p in out_dir.iterdir()} == before
    # a sidecar alone blocks too (it would describe other weights)
    (out_dir / "mlp_oval.npz").unlink()
    assert run(oval_study, "--out-dir", str(out_dir)) == 2
    (out_dir / "mlp_oval.npz").write_bytes(before["mlp_oval.npz"])
    capsys.readouterr()

    assert run(oval_study, "--out-dir", str(out_dir), "--overwrite") == 0
    out = capsys.readouterr().out
    assert f"replacing {out_dir / 'mlp_oval.npz'}" in out and "different weights" in out
    assert f"replacing {out_dir / 'mlp_oval.meta.json'}" in out
    assert "study seed 0" in out and "first bundle" in out  # what is being replaced
    assert sidecar(out_dir)["selection"]["chosen_seed"] == 2
    assert sorted(p.name for p in out_dir.iterdir()) == ["mlp_oval.meta.json", "mlp_oval.npz"]


def test_dry_run_writes_nothing(oval_study, scripted, tmp_path, capsys):
    out_dir = tmp_path / "out"
    assert run(oval_study, "--out-dir", str(out_dir), "--dry-run") == 0
    out = capsys.readouterr().out
    assert not out_dir.exists()
    assert "dry run: nothing written" in out
    printed = json.loads(out[out.index("{\n"):])  # the sidecar it would write
    assert printed["selection"]["chosen_seed"] == 2
    # an existing target does not stop a dry run, and is left alone
    out_dir.mkdir()
    (out_dir / "mlp_oval.npz").write_bytes(b"old")
    assert run(oval_study, "--out-dir", str(out_dir), "--dry-run") == 0
    assert "a real run needs --overwrite" in capsys.readouterr().out
    assert [(p.name, p.read_bytes()) for p in out_dir.iterdir()] == [("mlp_oval.npz", b"old")]
    # nothing was written into the study either
    assert not list(oval_study.rglob("*.tmp"))


def test_eval_seed_must_be_fresh_and_bad_arguments_are_refused(oval_study, scripted, capsys):
    for seed in (STUDY_SEED, STUDY_SEED + 3, 10_001, 2):
        # the study's eval (a multi study uses seed..seed+3), seed 1's in-training
        # snapshot evals (1 + 10_000), seed 2's training env
        assert run(oval_study, "--dry-run", "--eval-seed", str(seed)) == 2
        assert "is not fresh" in capsys.readouterr().err
    assert scripted == []
    assert run(oval_study, "--dry-run", "--eval-seed", str(STUDY_SEED + 4)) == 0
    capsys.readouterr()

    assert run(oval_study, "--dry-run", variant="nope") == 2
    assert "variant 'nope' has no finished cells" in capsys.readouterr().err
    assert run(oval_study, "--dry-run", name="sub/mlp_oval") == 2
    assert bundle.main(["--study", str(oval_study), "--variant", "v"]) == 2  # no --name
    assert run(oval_study / "cells", "--dry-run") == 2  # not a study directory
    assert run(oval_study, "--dry-run", "--top", "0") == 2
    assert run(oval_study, "--dry-run", "--rank-by", "device") == 2
    assert run(oval_study, "--dry-run", "--device-episodes", "2") == 2
    assert "needs a quantum driver" in capsys.readouterr().err


def test_candidate_without_weights_is_refused(oval_study, scripted, capsys):
    (study.cell_dir(oval_study, "v", 1) / "mlp_oval.meta.json").unlink()
    assert run(oval_study, "--dry-run") == 2
    assert "seed 1" in capsys.readouterr().err
    assert run(oval_study, "--dry-run", "--top", "1") in (0, 1)  # seed 1 is not asked for


# ----------------------------------------------------------------- multi-track


def test_multi_track_study_is_evaluated_per_track(tmp_path, scripted, capsys):
    root = tmp_path / "study_multi"
    kw = {"variant": "u", "agent": "quantum", "track": "multi", "study_lapped": 36,
          "init": "/somewhere/else/quantum_universal_v2.npz"}  # a fine-tune recipe
    # seed 0: three tracks perfect, combo below the bar -> more episodes lapped in total
    make_cell(root, seed=0, study_lap=18.0,
              fresh={"oval": 1.0, "chicane": 1.0, "gp": 1.0, "combo": 0.8}, **kw)
    # seed 1: all four tracks at 90% -> fewer in total, but reliable everywhere
    make_cell(root, seed=1, study_lap=19.0,
              fresh={"oval": 0.9, "chicane": 0.9, "gp": 0.9, "combo": 0.9}, **kw)
    make_cell(root, seed=2, study_lap=20.0,
              fresh={"oval": 1.0, "chicane": 0.5, "gp": 0.0, "combo": 0.0}, **kw)
    out_dir = tmp_path / "out"

    assert run(root, "--out-dir", str(out_dir), variant="u", name="quantum_universal") == 0
    out = capsys.readouterr().out
    meta = sidecar(out_dir, "quantum_universal")
    sel = meta["selection"]
    assert sel["chosen_seed"] == 1  # tracks lapped reliably first, then total lapped
    fresh = sel["fresh_eval"]
    assert (fresh["episodes"], fresh["episodes_per_track"], fresh["lapped"]) == (80, 20, 72)
    assert (fresh["tracks_reliable"], fresh["reliable_fraction"]) == (4, 0.9)
    assert list(fresh["tracks"]) == list(TRACKS)
    assert fresh["tracks"]["gp"] == {"episodes": 20, "lapped": 18, "lapped_fraction": 0.9,
                                     "laps": 18, "mean_lap": 14.0, "best_lap": 13.5}
    assert [(c["seed"], c["fresh_lapped"], c["fresh_tracks_reliable"], c["fresh_tracks"])
            for c in sel["candidates"]] == [
        (0, 76, 3, {"oval": 20, "chicane": 20, "gp": 20, "combo": 16}),
        (1, 72, 4, {"oval": 18, "chicane": 18, "gp": 18, "combo": 18}),
        (2, 30, 1, {"oval": 20, "chicane": 10, "gp": 0, "combo": 0})]
    # the study's own (mixed) eval was replayed on the mixture, the fresh one per track
    assert [(c["track"], c["seed"], c["episodes"]) for c in scripted if c["cell"] == 0] == [
        ("multi", STUDY_SEED, 36), *((name, bundle.EVAL_SEED, 20) for name in TRACKS)]
    assert "tracks lapped in >= 90% of the episodes, then episodes lapped" in sel["rule"]
    assert "gp laps in 18/20 episodes" in meta["provenance"]
    assert "| fresh oval | fresh chicane | fresh gp | fresh combo |" in out
    assert meta["track"] == "multi"
    # the fine-tune's starting point is on record by file name, not by absolute path
    assert sel["init"] == "quantum_universal_v2.npz"
    assert "warm-started from quantum_universal_v2.npz" in meta["provenance"]
    assert "/somewhere/else" not in json.dumps(meta)
    assert bundle.main(["--study", str(root), "--list"]) == 0
    assert "`init=quantum_universal_v2.npz`" in capsys.readouterr().out

    # the bar holds on EVERY track: seed 0 alone (combo at 80%) is refused
    assert run(root, "--top", "1", "--out-dir", str(tmp_path / "o2"), variant="u",
               name="quantum_universal") == 1
    assert "3/4 tracks lap in >= 90%" in capsys.readouterr().out
    assert not (tmp_path / "o2").exists()


def test_unseen_tracks_rank_with_the_bundled_ones(tmp_path, scripted, monkeypatch, capsys):
    """The universal lesson: the seed that is best on its four training tracks
    but laps no generated track loses to one that laps both."""
    root = tmp_path / "study_multi"
    kw = {"variant": "u", "agent": "quantum", "track": "multi", "study_lapped": 36}
    make_cell(root, seed=0, study_lap=13.0,
              fresh={"oval": 1.0, "chicane": 1.0, "gp": 1.0, "combo": 1.0}, **kw)
    make_cell(root, seed=1, study_lap=27.0,
              fresh={"oval": 0.95, "chicane": 0.95, "gp": 0.95, "combo": 0.95}, **kw)
    unseen_frac = {0: 0.0, 1: 1.0}
    asked = []

    def fake_unseen(spec, weights, args):
        asked.append((spec["seed"], args.unseen, args.unseen_seed, args.unseen_difficulty))
        n, frac = args.unseen_episodes, unseen_frac[spec["seed"]]
        return {f"random #{s}": {"episodes": n, "lapped_episodes": round(frac * n),
                                 "laps": round(frac * n), "mean_lap": 30.0 if frac else None,
                                 "best_lap": 29.0 if frac else None}
                for s in range(args.unseen_seed, args.unseen_seed + args.unseen)}

    monkeypatch.setattr(bundle, "evaluate_unseen", fake_unseen)
    assert run(root, "--top", "2", "--dry-run", variant="u", name="quantum_universal") == 0
    assert "Chosen: seed 0" in capsys.readouterr().out  # training tracks alone pick seed 0
    assert asked == []

    out_dir = tmp_path / "out"
    assert run(root, "--top", "2", "--unseen", "3", "--out-dir", str(out_dir),
               variant="u", name="quantum_universal") == 0
    out = capsys.readouterr().out
    assert [a[0] for a in asked] == [0, 1] and asked[0][1:] == (3, 100, 0.65)
    sel = sidecar(out_dir, "quantum_universal")["selection"]
    assert sel["chosen_seed"] == 1
    assert sel["unseen_eval"]["trackgen_seeds"] == [100, 102]
    assert (sel["unseen_eval"]["lapped"], sel["unseen_eval"]["episodes"],
            sel["unseen_eval"]["tracks_reliable"]) == (36, 36, 3)
    assert [(c["seed"], c["unseen_lapped"]) for c in sel["candidates"]] == [(0, 0), (1, 36)]
    assert "and 3 generated tracks (trackgen seeds 100-102, difficulty 0.65" in sel["rule"]
    assert "| unseen lapped | unseen tracks reliable |" in out


# ------------------------------------------------------ real rollouts, profiles


def test_real_rollouts_under_the_cells_own_profile_and_overrides(tmp_path, capsys):
    from racetraq.server import runtime

    root = tmp_path / "study_real"
    features = ["rays", "speed", "curvature_ahead", "corner_speed_ratio"]
    overrides = (SHORT, "circuit.n_layers=2", "observation.ray_angles_deg=[-60.0, 0.0, 60.0]",
                 f"observation.features={json.dumps(features)}")
    # untrained weights: nothing laps in 25 decisions.  Seed 0's stored study eval says
    # so; seed 1's claims 36/36, which this checkout cannot reproduce.
    q0 = make_cell(root, "q6L2", 0, agent="quantum", profile="q6", overrides=overrides,
                   study_lapped=0, final_lapped=0, fractions=(0, 0))
    make_cell(root, "q6L2", 1, agent="quantum", profile="q6", overrides=overrides,
              study_lapped=36, study_lap=14.0)
    make_cell(root, "mlp12", 0, agent="mlp", overrides=(SHORT, "mlp.hidden=12"),
              study_lapped=0, final_lapped=0, fractions=(0, 0))
    assert q0.name == "quantum_oval_q6.npz"
    out_dir = tmp_path / "out"
    common = ["--study", str(root), "--eval-episodes", "6", "--out-dir", str(out_dir)]

    assert bundle.main([*common, "--variant", "q6L2", "--name", "quantum_oval"]) == 1
    out = capsys.readouterr().out
    assert "REFUSED: the fresh eval laps in 0/6 episodes" in out
    assert "no '_q6' tag" in out  # the loaders' filename rule
    assert "note: seed 1: re-running the study's own eval here gives 0/36 lapped" in out
    assert not out_dir.exists()

    assert bundle.main([*common, "--variant", "q6L2", "--name", "quantum_oval_q6",
                        "--min-lapped", "0"]) == 0
    assert "no '_q6' tag" not in capsys.readouterr().out
    meta = sidecar(out_dir, "quantum_oval_q6")
    sel = meta["selection"]
    assert meta["circuit"] == {"n_qubits": 6, "n_layers": 2, "n_actions": 4}
    assert meta["observation"]["features"] == features
    assert (sel["profile"], sel["overrides"]) == ("q6", list(overrides))
    assert sel["fresh_eval"] == {"episodes": 6, "eval_seed": bundle.EVAL_SEED, "lapped": 0,
                                 "lapped_fraction": 0.0, "laps": 0, "mean_lap": None,
                                 "best_lap": None}
    # 0/6 each: the tie falls back to the study rank (seed 1 claimed 36/36)
    assert sel["chosen_seed"] == 1
    assert [(c["seed"], c["study_eval_reproduced"]) for c in sel["candidates"]] == [
        (1, False), (0, True)]
    weights = out_dir / "quantum_oval_q6.npz"
    assert np.load(weights)["params"].size == 3 * 2 * 6 + 2 * 4
    assert runtime.weights_observation(weights)["features"] == features
    assert runtime.weights_actions(weights) == 4
    weights_circuit = getattr(runtime, "weights_circuit", None)
    if weights_circuit is not None:  # loaders that take the depth from the sidecar
        assert weights_circuit(weights)["n_layers"] == 2

    # an MLP of a non-default width, without the replay check
    assert bundle.main([*common, "--variant", "mlp12", "--name", "mlp_oval", "--force",
                        "--skip-replay"]) == 0
    meta = sidecar(out_dir, "mlp_oval")
    assert meta["agent"] == "mlp" and meta["selection"]["overrides"][1] == "mlp.hidden=12"
    assert np.load(out_dir / "mlp_oval.npz")["params"].size == 4 * 12 + 12 + 12 * 4 + 4
    assert meta["selection"]["candidates"][0]["study_eval_reproduced"] is None
    assert meta["selection"]["fresh_eval"]["lapped"] == 0


# ------------------------------------------------------------------ device path


@pytest.fixture
def scripted_device(monkeypatch):
    """hw_reliability's device_row replaced by the script (params[9] = lapped
    fraction on the device); its argument parser is the real one."""
    import racetraq.hardware as hardware

    hw = bundle.hw_tool()
    calls = []

    def fake_row(config, params, args, backend, shots, rescale, resilience, row_seed):
        calls.append({"cell": int(params[0]), "track": args.track, "shots": shots,
                      "rescale": rescale, "resilience": resilience, "backend": backend,
                      "episodes": args.device_episodes, "cap": args.device_max_decisions,
                      "eval_seed": args.eval_seed, "n_layers": config["circuit"]["n_layers"],
                      "calibration": (args.calibration_samples, args.calibration_shots)})
        lapped = int(round(params[9] * args.device_episodes))
        return {"path": "device", "episodes": args.device_episodes, "lapped": lapped,
                "crashed": args.device_episodes - lapped, "mean_lap": 15.0 if lapped else None,
                "backend": "aer(fake_test)", "two_qubit_gates": 12}

    monkeypatch.setattr(hw, "device_row", fake_row)
    monkeypatch.setattr(hardware, "get_backend",
                        lambda **kwargs: ("backend", kwargs["fake_name"], kwargs["min_qubits"]))
    return calls


@pytest.fixture
def quantum_study(tmp_path):
    root = tmp_path / "study_hw"
    kw = {"variant": "q", "agent": "quantum", "study_lapped": 36}
    make_cell(root, seed=0, study_lap=13.0, fresh=1.0, fresh_lap=13.0, device=0.0, **kw)
    make_cell(root, seed=1, study_lap=14.0, fresh=1.0, fresh_lap=14.0, device=0.75, **kw)
    make_cell(root, seed=2, study_lap=15.0, fresh=0.5, fresh_lap=12.0, device=1.0, **kw)
    return root


def test_device_check_is_reported_and_ranks_only_on_request(
        quantum_study, scripted, scripted_device, tmp_path, capsys):
    from racetraq import hardware

    out_dir = tmp_path / "out"
    args = ("--device-episodes", "4", "--shots", "2048", "--fake", "fake_test",
            "--device-max-decisions", "150", "--out-dir", str(out_dir))
    # default: the exact fresh eval decides, the device result is on record only
    assert run(quantum_study, *args, variant="q", name="quantum_oval") == 0
    out = capsys.readouterr().out
    meta = sidecar(out_dir, "quantum_oval")
    sel = meta["selection"]
    assert sel["chosen_seed"] == 0 and sel["rank_by"] == "fresh"
    assert [(c["seed"], c["device_lapped"], c["device_episodes"]) for c in sel["candidates"]] \
        == [(0, 0, 4), (1, 3, 4), (2, 4, 4)]
    assert sel["device_eval"] == {
        "fake": "fake_test", "backend": "aer(fake_test)", "shots": 2048, "rescale": "off",
        "resilience": 0, "episodes": 4, "lapped": 0, "crashed": 4, "max_decisions": 150,
        "eval_seed": bundle.EVAL_SEED, "simulator_seed": 0, "two_qubit_gates": 12,
        "mean_lap": None}
    assert "| device lapped |" in out
    assert "simulated device (aer(fake_test), 2048 shots, rescale off, resilience 0): " \
           "laps in 0/4 episodes" in meta["provenance"]
    # device_row got the cell's config, this run's settings and hw_reliability's defaults
    assert [c["cell"] for c in scripted_device] == [0, 1, 2]
    first = scripted_device[0]
    assert first == {
        "cell": 0, "track": "oval", "shots": 2048, "rescale": "off", "resilience": 0,
        "backend": ("backend", "fake_test", 5), "episodes": 4, "cap": 150,
        "eval_seed": bundle.EVAL_SEED, "n_layers": 4,
        "calibration": (hardware.CALIBRATION_SAMPLES, hardware.CALIBRATION_MIN_SHOTS)}

    # --rank-by device: device laps first (seed 2: 4/4), although its fresh eval is
    # 10/20 — which --min-lapped then refuses
    assert run(quantum_study, *args, "--rank-by", "device", "--overwrite", variant="q",
               name="quantum_oval") == 1
    assert "### Chosen: seed 2" in capsys.readouterr().out
    assert sidecar(out_dir, "quantum_oval")["selection"]["chosen_seed"] == 0  # untouched
    assert run(quantum_study, *args, "--rank-by", "device", "--overwrite", "--top", "2",
               variant="q", name="quantum_oval") == 0
    sel = sidecar(out_dir, "quantum_oval")["selection"]
    assert sel["chosen_seed"] == 1 and sel["rank_by"] == "device"
    assert sel["device_eval"]["lapped"] == 3 and sel["device_eval"]["mean_lap"] == 15.0
    assert "episodes lapped on the simulated device, then lapped fraction" in sel["rule"]
    assert "chosen by the simulated-device laps, then that result" in \
        sidecar(out_dir, "quantum_oval")["provenance"]


def test_call_known_tolerates_signature_changes():
    def now(config, params, backend, shots, extra=1):
        return (config, params, backend, shots, extra)

    offered = {"config": "c", "params": "p", "args": "a", "backend": "b", "shots": 5}
    assert bundle._call_known(now, **offered) == ("c", "p", "b", 5, 1)  # "args" dropped

    def needs_more(config, calibration):
        return None

    with pytest.raises(ValueError, match="now needs 'calibration'"):
        bundle._call_known(needs_more, **offered)


# ------------------------------------------------------------------------ list


def test_list_prints_every_variant_with_its_seed_spread(oval_study, capsys):
    assert bundle.main(["--study", str(oval_study), "--list"]) == 0
    out = capsys.readouterr().out
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[1:-1]]
            for line in out.splitlines() if line.startswith("| ")}
    assert list(rows) == ["variant", "v", "other"]
    v = rows["v"]
    point, low, high = study.bootstrap_ci([1.0, 1.0, 35 / 36, 0.5, 0.0], statistic=study.iqm)
    assert v[1:5] == ["mlp", "oval", "—", "5"]
    assert v[5] == f"{point:.2f} [{low:.2f}, {high:.2f}]"
    assert v[6] == "0.97"  # median of the best-snapshot lapped fractions
    assert v[9] == "4/5"
    assert v[10] == "seed 0: 36/36, 13.0 s"
    assert v[11] == f"`{SHORT}`"
    assert rows["other"][4] == "1" and rows["other"][5] == "0.25 [—]"  # one seed: no interval
    # restricted to one variant; nothing is evaluated or written by --list
    assert bundle.main(["--study", str(oval_study), "--list", "--variant", "other"]) == 0
    assert "| v |" not in capsys.readouterr().out
    assert bundle.main(["--study", str(oval_study), "--list", "--variant", "nope"]) == 2
    assert bundle.main(["--study", str(oval_study / "nowhere"), "--list"]) == 2
