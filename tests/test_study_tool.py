"""tools/study.py: the multi-seed study harness.

The statistics (IQM, stratified bootstrap CIs, probability of improvement,
sample complexity, stability) are pinned against hand-computed values; a tiny
end-to-end study (MLP on the oval, 2 variants x 2 seeds, 30 episodes) checks
the run -> resume -> report pipeline, including a failing cell.
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

STUDY_PATH = Path(__file__).resolve().parent.parent / "tools" / "study.py"
_spec = importlib.util.spec_from_file_location("traqmania_study_tool", STUDY_PATH)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def test_tool_imports_only_numpy_and_stdlib_at_module_level():
    # traqmania is loaded inside the cell worker only
    tree = ast.parse(STUDY_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots - set(sys.stdlib_module_names) == {"numpy"}


# --------------------------------------------------------------- statistics


def test_iqm_of_known_vectors():
    # n = 8: drop 2 at each end -> mean(3, 4, 5, 6)
    assert study.iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)
    # order does not matter, outliers at both ends are ignored
    assert study.iqm([1000, 3, 5, -1000, 4, 6, 2, 7]) == pytest.approx(4.5)
    # n = 10: 2.5 values go at each end -> the 3rd and 8th count half:
    # (0.5 * 0 + 0 + 1 + 1 + 1 + 0.5 * 1) / 5
    assert study.iqm([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) == pytest.approx(0.7)
    # n = 5: 1.25 go at each end -> (0.75 * 2 + 3 + 0.75 * 10) / 2.5
    assert study.iqm([1, 2, 3, 10, 100]) == pytest.approx(4.8)
    # n = 3: 0.75 go at each end -> (0.25 * 1 + 2 + 0.25 * 6) / 1.5
    assert study.iqm([1, 2, 6]) == pytest.approx(2.5)
    assert study.iqm([1, 3]) == pytest.approx(2.0)
    assert study.iqm([7.5]) == 7.5
    assert np.isnan(study.iqm([]))


@pytest.mark.parametrize("n", range(1, 14))
def test_iqm_is_the_mean_of_the_middle_half_at_any_sample_size(n):
    # Four copies of every value make each quartile a whole number of entries,
    # so the middle half can be sliced out directly.
    values = np.random.default_rng(n).normal(size=n)
    middle_half = np.repeat(np.sort(values), 4)[n:3 * n]
    assert study.iqm(values) == pytest.approx(middle_half.mean())


def test_median_of_known_vectors():
    assert study.median([3, 1, 2]) == 2.0
    assert study.median([4, 1, 2, 3]) == 2.5
    assert np.isnan(study.median([]))


def test_prob_improvement_constructed_cases():
    assert study.prob_improvement([5, 6, 7], [1, 2, 3]) == 1.0
    assert study.prob_improvement([1, 2, 3], [5, 6, 7]) == 0.0
    # identical samples, and all-ties: exactly one half
    assert study.prob_improvement([1, 2, 3], [1, 2, 3]) == 0.5
    assert study.prob_improvement([0, 0, 0, 0], [0, 0]) == 0.5
    # hand count: pairs (1,1) tie, (1,2) loss, (3,1) win, (3,2) win -> 2.5 / 4
    assert study.prob_improvement([1, 3], [1, 2]) == pytest.approx(0.625)
    assert np.isnan(study.prob_improvement([], [1.0]))


def test_bootstrap_ci_contains_truth_and_is_deterministic():
    rng = np.random.default_rng(123)
    sample = rng.normal(loc=3.0, scale=1.0, size=200)
    for statistic in (study.iqm, study.median):
        point, low, high = study.bootstrap_ci(sample, statistic=statistic)
        assert low < 3.0 < high
        assert low <= point <= high
        assert high - low < 0.6  # ~ +-2 standard errors, not a vacuous interval
        assert point == statistic(sample)
        # fixed RNG seed: the same call gives the same interval, bit for bit
        assert study.bootstrap_ci(sample, statistic=statistic) == (point, low, high)
    # another seed resamples differently
    assert study.bootstrap_ci(sample, seed=1) != study.bootstrap_ci(sample, seed=0)
    # the default interval is the 95% one: for a mean, +-1.96 standard errors
    standard_error = sample.std(ddof=1) / np.sqrt(sample.size)
    point, low, high = study.bootstrap_ci(sample, statistic=np.mean)
    assert point == pytest.approx(sample.mean())
    assert high - low == pytest.approx(2 * 1.96 * standard_error, rel=0.1)
    _, low, high = study.bootstrap_ci(sample, statistic=np.mean, ci=0.5)
    assert high - low == pytest.approx(2 * 0.674 * standard_error, rel=0.15)


def test_bootstrap_ci_degenerate_inputs():
    assert study.bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0, 2.0)  # no spread, no width
    assert all(np.isnan(v) for v in study.bootstrap_ci([]))
    # one seed: a point estimate, but no interval (not a zero-width "certain" one)
    point, low, high = study.bootstrap_ci([4.0])
    assert point == 4.0 and np.isnan(low) and np.isnan(high)
    point, low, high = study.bootstrap_ci([1.0, 2.0, 3.0], [0.0],
                                          statistic=study.prob_improvement)
    assert point == 1.0 and np.isnan(low) and np.isnan(high)


def test_bootstrap_ci_resamples_each_stratum_on_its_own():
    seen = set()

    def shapes(a, b):
        seen.add((a.size, b.size))
        # every resample of a stratum is drawn from that stratum only
        assert set(a) <= {1.0, 2.0, 3.0} and set(b) <= {10.0, 20.0}
        return a.mean() - b.mean()

    point, low, high = study.bootstrap_ci([1, 2, 3], [10, 20], statistic=shapes, n_boot=200)
    assert seen == {(3, 2)}  # strata keep their sizes
    assert point == pytest.approx(2.0 - 15.0)
    assert low <= point <= high


def test_bootstrap_ci_of_prob_improvement():
    better = [0.9, 1.0, 0.8, 1.0, 0.95, 0.9, 1.0, 0.85]
    worse = [0.1, 0.0, 0.2, 0.0, 0.3, 0.1, 0.0, 0.05]
    assert study.bootstrap_ci(better, worse, statistic=study.prob_improvement) \
        == (1.0, 1.0, 1.0)
    assert study.bootstrap_ci(worse, better, statistic=study.prob_improvement) \
        == (0.0, 0.0, 0.0)
    point, low, high = study.bootstrap_ci(better, better, statistic=study.prob_improvement)
    assert point == 0.5 and low < 0.5 < high


def _log(*fractions, every=50, n=12):
    return [{"episode": every * (i + 1), "lapped_episodes": round(f * n), "eval_episodes": n}
            for i, f in enumerate(fractions)]


def test_sample_complexity():
    log = _log(0, 0, 0.25, 0.5, 0, 1.0, 0)
    assert study.sample_complexity(log, 0.5) == 200  # first eval at >= 50%
    assert study.sample_complexity(log, 0.9) == 300  # running best: the later dip is ignored
    assert study.sample_complexity(log, 0.25) == 150
    assert study.sample_complexity(_log(0, 0.25, 0.25), 0.5) is None
    assert study.sample_complexity([], 0.5) is None
    # 11/12 = 0.917 clears 90%, 10/12 = 0.833 does not
    assert study.sample_complexity(_log(10 / 12, 11 / 12), 0.9) == 100


def test_episodes_for_fraction():
    # 3 of 4 seeds get there: half of all seeds (2) are there by the 2nd fastest
    assert study.episodes_for_fraction([400, None, 100, 250], 0.5) == 250
    assert study.episodes_for_fraction([400, None, 100, 250], 0.75) == 400
    assert study.episodes_for_fraction([400, None, 100, 250], 1.0) is None
    assert study.episodes_for_fraction([None, None, 100], 0.5) is None  # 1 of 3 < half
    assert study.episodes_for_fraction([], 0.5) is None


def test_stability():
    # first lap at the 2nd eval; afterwards 1.0, 0, 0.5 -> mean 0.5
    assert study.stability(_log(0, 0.25, 1.0, 0, 0.5)) == pytest.approx(0.5)
    assert study.stability(_log(0, 1.0, 1.0, 1.0)) == 1.0
    assert study.stability(_log(0, 1.0, 0, 0, 0)) == 0.0  # lapped once, never again
    assert study.stability(_log(0, 0, 0)) == 0.0  # never lapped
    assert study.stability(_log(0, 0, 1.0)) is None  # nothing after the first lap
    assert study.stability([]) == 0.0


# ------------------------------------------------------- variants and seeds


def test_parse_seeds():
    assert study.parse_seeds("0-3") == [0, 1, 2, 3]
    assert study.parse_seeds("42,0,1") == [42, 0, 1]
    assert study.parse_seeds("0-2, 42, 1") == [0, 1, 2, 42]
    for bad in ("", "a", "3-1", "1.5", "0-"):
        with pytest.raises(ValueError):
            study.parse_seeds(bad)


def test_parse_variant():
    assert study.parse_variant("base") == ("base", {}, [])
    name, pseudo, overrides = study.parse_variant(
        'huber:training.loss="huber",training.huber_delta=10')
    assert (name, pseudo) == ("huber", {})
    assert overrides == ['training.loss="huber"', "training.huber_delta=10"]
    # commas inside tables, arrays and strings belong to the value
    name, pseudo, overrides = study.parse_variant(
        'mix:training.lr_groups={ head = 0.1, lam = 0.001 },agent=mlp, '
        'observation.ray_angles_deg=[-60, 0, 60],profile=q10,pace=true,'
        'init="a,b/{seed}.npz"')
    assert overrides == ["training.lr_groups={ head = 0.1, lam = 0.001 }",
                         "observation.ray_angles_deg=[-60, 0, 60]"]
    assert pseudo == {"agent": "mlp", "profile": "q10", "pace": "true",
                      "init": '"a,b/{seed}.npz"'}
    for bad in ("bad name:x.y=1", ":x.y=1", "v:nodot=1", "v:training.lr", "v:training.lr="):
        with pytest.raises(ValueError):
            study.parse_variant(bad)


def test_resolve_variant():
    defaults = {"agent": "quantum", "track": "gp", "profile": "q10", "episodes": None,
                "preset": "auto", "actions": None, "pace": False, "init": None}
    base = study.resolve_variant("base", defaults, ["training.eval_every=25"])
    assert base == {"name": "base", **defaults, "overrides": ["training.eval_every=25"]}
    recipe = study.resolve_variant(
        "m:agent=mlp,profile=none,episodes=500,actions=6,pace=true,init=w/{seed}.npz,"
        "circuit.n_layers=6", defaults, ["training.eval_every=25"])
    assert recipe == {
        "name": "m", "agent": "mlp", "track": "gp", "profile": None, "episodes": 500,
        "preset": "auto", "actions": 6, "pace": True, "init": "w/{seed}.npz",
        # study-wide overrides first, the variant's own after them (they win)
        "overrides": ["training.eval_every=25", "circuit.n_layers=6"],
    }
    with pytest.raises(ValueError):
        study.resolve_variant("m:episodes=many", defaults)
    with pytest.raises(ValueError):
        study.resolve_variant("m:pace=maybe", defaults)


def test_new_config_keys_flags_a_mistyped_override():
    defaults = {"agent": "quantum", "track": "oval", "profile": None, "episodes": None,
                "preset": "auto", "actions": None, "pace": False, "init": None}

    def new(spec):
        return study.new_config_keys(study.resolve_variant(spec, defaults))

    assert new('ok:circuit.n_layers=6,reward.lap_bonus=0,training.loss="huber"') == []
    # the typo would train the unmodified recipe under the name "L6"
    assert new("L6:circuit.n_layer=6,circuit.n_layers=6") == ["circuit.n_layer"]
    assert new("x:nosuchsection.key=1,profile=q6") == ["nosuchsection.key"]
    # unknown [training] keys are the trainer's to reject (it does, per cell)
    assert new("t:training.no_such_option=1") == []
    with pytest.raises(ValueError):
        new("bad:circuit.n_layers=[6")


def test_run_refuses_variant_names_that_differ_only_by_case(tmp_path, capsys):
    # one directory on a case-folding file system: both would train into it
    args = ["run", "--out", str(tmp_path / "s"), "--agent", "mlp", "--track", "oval",
            "--episodes", "8", "--seeds", "0"]
    assert study.main([*args, "--variant", "l6", "--variant", "L6:training.lr=0.5"]) == 2
    assert "given twice" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()  # refused before anything was written

    recipe = {"name": "l6", "overrides": []}
    evals = {"episodes": 36, "seed": 20_000}
    study.update_manifest(tmp_path, [recipe], [0], evals, "cmd", 1)
    with pytest.raises(ValueError, match="upper/lower case"):
        study.update_manifest(tmp_path, [{"name": "L6", "overrides": []}], [0], evals,
                              "cmd", 1)
    study.update_manifest(tmp_path, [recipe], [1], evals, "cmd", 1)  # same name: a resume
    manifest = json.loads((tmp_path / study.MANIFEST_NAME).read_text())
    assert list(manifest["variants"]) == ["l6"] and manifest["seeds"] == [0, 1]


# ------------------------------------------------------- reliability eval


def test_weights_config_follows_the_weights_sidecar(tmp_path):
    features = ["rays", "speed", "curvature_ahead", "corner_speed_ratio"]
    (tmp_path / "quantum_oval_q6.meta.json").write_text(json.dumps({
        "circuit": {"n_qubits": 6, "n_layers": 5, "n_actions": 6},
        "observation": {"ray_angles_deg": [-60.0, 0.0, 60.0], "features": features},
        "actions": {"n_actions": 6},
    }), encoding="utf-8")
    spec = {"profile": None, "actions": None, "overrides": ["reward.max_decisions=100"],
            "track": "oval", "seed": 0, "eval_episodes": 5, "eval_seed": 1}
    config = study.weights_config(spec, tmp_path / "quantum_oval_q6.npz")
    # the sidecar's shapes win over the (4-qubit, 3 rays + speed) default profile...
    assert config["circuit"] == {**config["circuit"], "n_qubits": 6, "n_layers": 5,
                                 "n_actions": 6}
    assert config["observation"]["features"] == features
    assert config["reward"]["max_decisions"] == 100  # ...on top of the cell's overrides
    env = study._eval_env(spec, config)
    assert (env.n_envs, env.n_features, env.n_actions) == (5, 6, 6)


def test_greedy_eval_matches_records_on_distinct_episodes():
    from traqmania import records
    from traqmania.train_headless import WEIGHTS_DIR, build_qfunc

    weights = WEIGHTS_DIR / "quantum_oval.npz"
    spec = {"profile": None, "actions": None, "overrides": ["reward.max_decisions=200"],
            "track": "oval", "seed": 0, "eval_episodes": study.EVAL_EPISODES,
            "eval_seed": study.EVAL_SEED}
    config = study.weights_config(spec, weights)
    env = study._eval_env(spec, config)
    start = env.reset()
    assert len({row.tobytes() for row in start}) == 36  # 36 spawns, no episode repeated

    qfunc = build_qfunc("quantum", env.n_features, 0, config, n_actions=env.n_actions)
    qfunc.set_params(np.load(weights)["params"])
    mine = study.greedy_eval(qfunc, study._eval_env(spec, config), 201)
    driver = records.Driver("quantum_oval", "quantum", "oval", 4, weights, config)
    reference = records.evaluate(driver, "oval", study.EVAL_EPISODES, seed=study.EVAL_SEED)
    assert mine["episodes"] == 36
    assert mine["lapped_episodes"] > 0  # the bundled driver laps: not a 0 == 0 check
    assert (mine["lapped_episodes"], mine["laps"]) == (
        reference["lapped_episodes"], reference["laps"])
    assert round(mine["mean_lap"], 2) == reference["mean_s"]
    assert round(mine["best_lap"], 2) == reference["best_s"]


def test_cell_evaluates_best_snapshot_and_final_params_separately(tmp_path, monkeypatch):
    import traqmania.train_headless as train_headless

    def fake_train(agent, track, episodes, seed, profile, out_dir, history_path, **kwargs):
        # "best snapshot" = the bundled lapping driver, "final params" = all zeros
        out = Path(out_dir)
        lapping = np.load(train_headless.WEIGHTS_DIR / "quantum_oval.npz")["params"]
        np.savez(out / "quantum_oval.npz", params=lapping)
        (out / "quantum_oval.meta.json").write_text(
            (train_headless.WEIGHTS_DIR / "quantum_oval.meta.json").read_text())
        np.savez(out / "quantum_oval.final.npz", params=np.zeros_like(lapping))
        Path(history_path).write_text(json.dumps({"episodes": episodes, "training": {}}))
        return {"weights_path": str(out / "quantum_oval.npz"),
                "final_weights_path": str(out / "quantum_oval.final.npz"),
                "episode_returns": [1.0], "wall_time_s": 0.1, "first_clean_episode": None,
                "eval_log": [], "best_eval": None, "final_eval": None}

    monkeypatch.setattr(train_headless, "train", fake_train)
    monkeypatch.setattr(study.os, "nice", lambda increment: 0)  # keep pytest's priority
    spec = {"variant": "v", "agent": "quantum", "track": "oval", "episodes": 5, "seed": 0,
            "profile": None, "preset": "auto", "actions": None, "pace": False, "init": None,
            "overrides": ["reward.max_decisions=200"], "eval_episodes": 8,
            "eval_seed": study.EVAL_SEED}
    (tmp_path / study.SPEC_NAME).write_text(json.dumps(spec))
    study.run_cell(tmp_path)
    result = json.loads((tmp_path / study.RESULT_NAME).read_text())
    assert result["best_snapshot_eval"]["episodes"] == 8
    assert result["best_snapshot_eval"]["lapped_episodes"] > 0
    assert result["best_snapshot_eval"]["mean_lap"] is not None
    assert result["final_params_eval"] == {
        "episodes": 8, "lapped_episodes": 0, "laps": 0, "mean_lap": None, "best_lap": None}
    assert (result["weights"], result["final_weights"]) == (
        "quantum_oval.npz", "quantum_oval.final.npz")


# --------------------------------------------------- report on made-up cells


def _fake_result(variant, seed, best, final, fractions, first_clean=None, mean_lap=None):
    def reliability(lapped):
        return {"episodes": 36, "lapped_episodes": lapped, "laps": lapped,
                "mean_lap": mean_lap if lapped else None,
                "best_lap": mean_lap if lapped else None}

    return {"variant": variant, "seed": seed, "n_params": 56, "wall_time_s": 1.0,
            "first_clean_episode": first_clean, "eval_log": _log(*fractions),
            "best_snapshot_eval": reliability(best), "final_params_eval": reliability(final)}


def _write_fake_study(directory, cells):
    for result in cells:
        cell = study.cell_dir(directory, result["variant"], result["seed"])
        cell.mkdir(parents=True)
        (cell / study.RESULT_NAME).write_text(json.dumps(result), encoding="utf-8")


def test_report_aggregates_hand_made_cells(tmp_path):
    cells = [
        # baseline: laps once, then forgets (stability 0, final params lap nothing)
        _fake_result("base", 0, 36, 0, [0, 1.0, 0, 0], first_clean=120, mean_lap=30.0),
        _fake_result("base", 1, 18, 0, [0, 0, 0.5, 0], first_clean=160, mean_lap=32.0),
        _fake_result("base", 2, 0, 0, [0, 0, 0, 0]),
        _fake_result("base", 3, 9, 0, [0, 0.25, 0, 0], first_clean=110, mean_lap=34.0),
        # variant: every seed laps and keeps lapping
        _fake_result("stable", 0, 36, 36, [0, 1.0, 1.0, 1.0], first_clean=90, mean_lap=28.0),
        _fake_result("stable", 1, 36, 27, [0.5, 1.0, 1.0, 0.75], first_clean=40,
                     mean_lap=29.0),
        _fake_result("stable", 2, 36, 36, [0, 0, 1.0, 1.0], first_clean=140, mean_lap=30.0),
        _fake_result("stable", 3, 27, 18, [0, 0.5, 0.5, 0.5], first_clean=95, mean_lap=31.0),
    ]
    _write_fake_study(tmp_path, cells)
    failed = study.cell_dir(tmp_path, "stable", 4)
    failed.mkdir(parents=True)
    (failed / study.ERROR_NAME).write_text("exit code 1\n", encoding="utf-8")

    report = study.build_report(tmp_path)
    assert report["baseline"] == "base"
    assert list(report["variants"]) == ["base", "stable"]
    assert report["failed_cells"] == [
        {"variant": "stable", "seed": 4, "error_file": str(failed / study.ERROR_NAME)}]
    base, stable = report["variants"]["base"], report["variants"]["stable"]
    assert (base["n_seeds"], stable["n_seeds"]) == (4, 4)
    assert stable["failed_seeds"] == [4]
    assert (base["reliable_seeds"], stable["reliable_seeds"]) == (2, 4)  # best >= 18/36

    # best-snapshot lapped fractions, base: 1, 0.5, 0, 0.25 -> IQM = mean(0.25, 0.5)
    best = base["metrics"]["best_lapped_frac"]
    assert best["n"] == 4
    assert best["iqm"]["value"] == pytest.approx(0.375)
    assert best["median"]["value"] == pytest.approx(0.375)
    assert best["iqm"]["ci_low"] <= 0.375 <= best["iqm"]["ci_high"]
    assert base["metrics"]["final_lapped_frac"]["iqm"] == {
        "value": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    # mean lap only over the 3 base seeds that lap: 30, 32, 34
    assert base["metrics"]["best_mean_lap"]["n"] == 3
    assert base["metrics"]["best_mean_lap"]["median"]["value"] == pytest.approx(32.0)
    assert base["metrics"]["first_clean_episode"]["n"] == 3
    assert base["metrics"]["first_clean_episode"]["median"]["value"] == pytest.approx(120.0)
    # stability: base 0, 0, 0 (never lapped), 0; stable 1, 11/12, 1, 0.5
    assert [r["stability"] for r in base["per_seed"]] == [0.0, 0.0, 0.0, 0.0]
    assert [r["stability"] for r in stable["per_seed"]] == pytest.approx(
        [1.0, (1.0 + 1.0 + 0.75) / 3, 1.0, 0.5])

    # sample complexity (evals every 50 episodes)
    sc50, sc90 = base["sample_complexity"]["episodes_to_50"], \
        base["sample_complexity"]["episodes_to_90"]
    assert [r["episodes_to_50"] for r in base["per_seed"]] == [100, 150, None, None]
    assert (sc50["reached"], sc50["fraction_reached"]) == (2, 0.5)
    assert sc50["episodes_half_of_seeds"] == 150
    assert sc50["median"]["value"] == pytest.approx(125.0)
    assert (sc90["reached"], sc90["episodes_half_of_seeds"]) == (1, None)
    assert [r["episodes_to_50"] for r in stable["per_seed"]] == [100, 50, 150, 100]
    assert stable["sample_complexity"]["episodes_to_50"]["episodes_half_of_seeds"] == 100
    assert stable["sample_complexity"]["episodes_to_90"]["reached"] == 3

    # versus baseline: every stable seed beats every base seed on both metrics
    assert "vs_baseline" not in base
    for key in ("final_lapped_frac", "stability"):
        assert stable["vs_baseline"][key] == {
            "p_improvement": 1.0, "ci_low": 1.0, "ci_high": 1.0, "n": 4, "n_baseline": 4}
    # ...and the comparison flips with the baseline
    flipped = study.build_report(tmp_path, baseline="stable", n_boot=200)
    assert flipped["variants"]["base"]["vs_baseline"]["stability"]["p_improvement"] == 0.0

    assert study.build_report(tmp_path) == report  # deterministic
    markdown = study.render_markdown(report)
    for heading in ("## Variants", "## IQM over seeds", "## Median over seeds",
                    "## Sample complexity", "## Versus baseline `base`",
                    "## Per-seed results", "## Failed cells"):
        assert heading in markdown
    assert "| stable | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |" in markdown

    with pytest.raises(ValueError):
        study.build_report(tmp_path, baseline="nope")
    with pytest.raises(ValueError):
        study.build_report(tmp_path / "empty")


def test_report_with_a_single_seed_has_no_intervals(tmp_path):
    _write_fake_study(tmp_path, [
        _fake_result("base", 0, 0, 0, [0, 0, 0]),
        _fake_result("better", 0, 36, 36, [0, 1.0, 1.0], first_clean=80, mean_lap=30.0),
    ])
    report = study.build_report(tmp_path)
    better = report["variants"]["better"]
    assert better["metrics"]["final_lapped_frac"]["iqm"] == {
        "value": 1.0, "ci_low": None, "ci_high": None}
    # one seed against one seed is an anecdote: P = 1, but no interval claims it
    assert better["vs_baseline"]["final_lapped_frac"] == {
        "p_improvement": 1.0, "ci_low": None, "ci_high": None, "n": 1, "n_baseline": 1}
    markdown = study.render_markdown(report)
    assert "| better | 1.00 [—] | 1.00 [—] |" in markdown
    assert "[1.00, 1.00]" not in markdown


def test_report_with_unequal_seed_counts_and_seeds_without_values(tmp_path):
    nan_lap = _fake_result("base", 4, 9, 0, [0, 0.25, 0, 0], mean_lap=30.0)
    nan_lap["best_snapshot_eval"]["mean_lap"] = float("nan")  # as json's NaN token
    lapped_last = _fake_result("few", 1, 36, 36, [0, 0, 0, 1.0], first_clean=190,
                               mean_lap=29.0)
    _write_fake_study(tmp_path, [
        _fake_result("base", 0, 0, 0, [0, 0, 0, 0]),  # never laps
        _fake_result("base", 1, 0, 0, [0, 0, 0, 0]),
        _fake_result("base", 2, 36, 0, [0, 1.0, 0, 0], first_clean=70, mean_lap=31.0),
        _fake_result("base", 3, 18, 9, [0, 0.5, 0.5, 0], first_clean=90, mean_lap=33.0),
        nan_lap,
        _fake_result("few", 0, 36, 36, [0, 1.0, 1.0, 1.0], first_clean=60, mean_lap=28.0),
        lapped_last,
    ])
    report = study.build_report(tmp_path)
    base, few = report["variants"]["base"], report["variants"]["few"]
    assert (base["n_seeds"], few["n_seeds"]) == (5, 2)
    # seeds that never lap count in the lapped fractions and score stability 0...
    assert [r["stability"] for r in base["per_seed"]] == pytest.approx(
        [0.0, 0.0, 0.0, 0.25, 0.0])
    # final fractions 0, 0, 0, 0.25, 0 -> IQM (0.75 * 0 + 0 + 0.75 * 0) / 2.5
    assert base["metrics"]["final_lapped_frac"]["iqm"]["value"] == 0.0
    assert base["metrics"]["final_lapped_frac"]["n"] == 5
    # ...but have no lap time; neither has the NaN one: mean lap over 31 and 33 only
    lap = base["metrics"]["best_mean_lap"]
    assert lap["n"] == 2 and lap["median"]["value"] == pytest.approx(32.0)
    assert base["per_seed"][4]["best_mean_lap"] is None
    # a seed whose first lapping eval is the last one has no stability value
    assert [r["stability"] for r in few["per_seed"]] == [1.0, None]
    assert few["metrics"]["stability"]["n"] == 1
    assert few["metrics"]["best_mean_lap"]["median"]["value"] == pytest.approx(28.5)
    # P(few > base) on final fractions (1, 1) vs (0, 0, 0, 0.25, 0): all 10 pairs won;
    # stability: the one valued seed (1.0) beats all five
    final = few["vs_baseline"]["final_lapped_frac"]
    assert (final["p_improvement"], final["n"], final["n_baseline"]) == (1.0, 2, 5)
    stable = few["vs_baseline"]["stability"]
    assert (stable["p_improvement"], stable["n"], stable["n_baseline"]) == (1.0, 1, 5)
    assert stable["ci_low"] is None  # ...a single seed: no interval
    json.dumps(report, allow_nan=False)  # report.json stays strict JSON
    assert "| few | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [—] (n=1) |" \
        in study.render_markdown(report)


# ------------------------------------------------------- end-to-end study


RUN_ARGS = ["--agent", "mlp", "--track", "oval", "--episodes", "30", "--seeds", "0-1",
            "--variant", "base",
            # a shape-changing override too: it must reach the trainer AND the evals
            "--variant", "fast:training.eval_every=10,training.lr=0.02,mlp.hidden=12",
            "--jobs", "2"]


@pytest.fixture(scope="module")
def tiny_study(tmp_path_factory):
    out = tmp_path_factory.mktemp("study")
    assert study.main(["run", "--out", str(out), *RUN_ARGS]) == 0
    return out


def test_study_run_writes_results_weights_and_manifest(tiny_study):
    assert (tiny_study / study.DONE_NAME).read_text().strip() == "4 cells: 4 ok, 0 failed"
    fast_overrides = {"training.eval_every": 10, "training.lr": 0.02, "mlp.hidden": 12}
    # MLP 4-h-4: 4h + h + 4h + 4 params
    for variant, overrides, n_params in (("base", {}, 76), ("fast", fast_overrides, 112)):
        for seed in (0, 1):
            cell = study.cell_dir(tiny_study, variant, seed)
            assert not (cell / study.ERROR_NAME).exists()
            result = json.loads((cell / study.RESULT_NAME).read_text())
            assert (result["variant"], result["seed"]) == (variant, seed)
            assert (result["agent"], result["track"], result["episodes"]) == ("mlp", "oval", 30)
            assert result["overrides"] == overrides
            assert result["training"]["seed"] == seed
            assert result["n_params"] == n_params
            assert result["wall_time_s"] > 0
            assert "first_clean_episode" in result
            assert result["best_eval"]["eval_episodes"] == 12
            assert result["final_eval"] == result["eval_log"][-1]
            # two reliability evals of 36 distinct greedy episodes each
            for key in ("best_snapshot_eval", "final_params_eval"):
                assert result[key]["episodes"] == 36
                assert 0 <= result[key]["lapped_episodes"] <= 36
                assert (result[key]["mean_lap"] is None) == (result[key]["lapped_episodes"] == 0)
            # weights: best snapshot (+ sidecar) and final params
            best = np.load(cell / result["weights"])["params"]
            final = np.load(cell / result["final_weights"])["params"]
            assert best.shape == final.shape == (n_params,)
            assert (cell / "mlp_oval.meta.json").is_file()
            assert (cell / study.LOG_NAME).read_text().startswith("training agent=mlp")
    # the variant's overrides reached the trainer: an eval every 10 episodes
    fast = json.loads((study.cell_dir(tiny_study, "fast", 0) / study.RESULT_NAME).read_text())
    base = json.loads((study.cell_dir(tiny_study, "base", 0) / study.RESULT_NAME).read_text())
    assert (fast["training"]["lr"], fast["training"]["eval_every"]) == (0.02, 10)
    assert len(fast["eval_log"]) >= 3 and fast["eval_log"][0]["episode"] < 20
    assert len(fast["eval_log"]) > len(base["eval_log"])
    assert base["training"]["lr"] != 0.02

    manifest = json.loads((tiny_study / study.MANIFEST_NAME).read_text())
    assert list(manifest["variants"]) == ["base", "fast"]
    assert manifest["variants"]["fast"]["overrides"] == [
        "training.eval_every=10", "training.lr=0.02", "mlp.hidden=12"]
    assert manifest["seeds"] == [0, 1]
    assert manifest["eval"] == {"episodes": 36, "seed": 20_000}
    run = manifest["runs"][0]
    assert "study.py run --out" in run["command"] and "--variant base" in run["command"]
    assert "git_commit" in run and "git_dirty" in run


def test_study_second_run_is_a_noop_and_report_works(tiny_study, capsys):
    result_files = sorted(tiny_study.glob(f"cells/*/*/{study.RESULT_NAME}"))
    assert len(result_files) == 4
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in result_files}
    logs = {path: path.stat().st_mtime_ns for path in tiny_study.glob("cells/*/*/train.log")}

    capsys.readouterr()
    assert study.main(["run", "--out", str(tiny_study), *RUN_ARGS]) == 0
    out = capsys.readouterr().out
    assert "4 already done, 0 to run" in out and "DONE 4 cells: 4 ok, 0 failed" in out
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in result_files} == before
    assert {p: p.stat().st_mtime_ns for p in tiny_study.glob("cells/*/*/train.log")} == logs
    manifest = json.loads((tiny_study / study.MANIFEST_NAME).read_text())
    assert len(manifest["runs"]) == 2  # the resume is on record

    # a variant name cannot be re-used for another recipe
    assert study.main(["run", "--out", str(tiny_study), "--agent", "mlp", "--track", "oval",
                       "--episodes", "30", "--seeds", "0",
                       "--variant", "base:training.lr=0.5"]) == 2
    assert "already exists" in capsys.readouterr().err

    assert study.main(["report", str(tiny_study)]) == 0
    markdown = capsys.readouterr().out
    assert (tiny_study / "report.md").read_text() == markdown
    for text in ("## IQM over seeds [95% CI]", "## Sample complexity",
                 "## Versus baseline `base`", "| fast |", "`training.lr=0.02`"):
        assert text in markdown
    report = json.loads((tiny_study / "report.json").read_text())
    assert report["baseline"] == "base"
    assert report["bootstrap"]["resamples"] >= 2000
    assert [(name, v["n_seeds"]) for name, v in report["variants"].items()] == [
        ("base", 2), ("fast", 2)]
    assert report["failed_cells"] == []
    versus = report["variants"]["fast"]["vs_baseline"]
    assert set(versus) == {"final_lapped_frac", "stability"}
    stat = versus["final_lapped_frac"]
    assert (stat["n"], stat["n_baseline"]) == (2, 2)
    assert 0.0 <= stat["ci_low"] <= stat["p_improvement"] <= stat["ci_high"] <= 1.0

    assert study.main(["report", str(tiny_study), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_failed_cell_writes_error_and_does_not_stop_the_study(tmp_path, capsys):
    code = study.main(["run", "--out", str(tmp_path), "--agent", "mlp", "--track", "oval",
                       "--episodes", "8", "--seeds", "0", "--eval-episodes", "4",
                       "--variant", "bad:training.no_such_option=1",
                       "--variant", "ok", "--jobs", "2"])
    out = capsys.readouterr().out
    assert code == 1
    bad, ok = study.cell_dir(tmp_path, "bad", 0), study.cell_dir(tmp_path, "ok", 0)
    assert "no_such_option" in (bad / study.ERROR_NAME).read_text()
    assert not (bad / study.RESULT_NAME).exists()
    assert json.loads((ok / study.RESULT_NAME).read_text())["best_snapshot_eval"][
        "episodes"] == 4
    assert "bad seed 0: FAILED" in out and "DONE 2 cells: 1 ok, 1 failed" in out
    assert (tmp_path / study.DONE_NAME).read_text().strip() == "2 cells: 1 ok, 1 failed"
