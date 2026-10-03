"""Classical Fourier surrogates: the spectrum derivation must be exact, and the
sampled surrogates must imitate the bundled driver.

Three layers, cheapest first. (1) Structure: the per-feature frequency sets
have the predicted sizes and respect the light cone. (2) Exactness: on tiny
circuits the FULL spectrum basis reproduces fastsim to ~1e-10 — and a wrong
spectrum does not — which is what validates "frequencies = sums of
+-lam over live encoding gates". (3) Practice: random-Fourier-feature and
kernel surrogates fitted from a fixed budget of driving samples reproduce the
bundled 4-qubit oval driver's Q-values, greedy actions and laps.

The thresholds in (3) were set with margin on the bundled weights — last
re-measured 2026-10-03 for the oval driver re-bundled on 2026-10-02 (study
robust_oval, seed 0; file sha256 1cd3f7a390b9...), at 1000 samples on 1500
held-out driving states, sample/frequency seeds 0-4 with one held-out log and
10-14 with another, both methods (20 fits): nrmse 0.019-0.080, <Z> RMSE
0.0043-0.0196, greedy agreement >= 0.998, regret <= 0.004 of the mean gap;
over 36 greedy episodes at three episode seeds every surrogate lapped 36/36,
exactly as the quantum driver does, with mean laps within 0.01 s of its
12.66 s.  (The July 2026 driver fitted more easily: nrmse 0.005-0.024, <Z>
RMSE 0.0008-0.0042; the advantage-learning driver has a sharper Q-landscape.)
They are statements about those weights, so re-measure if a retrained driver
trips them (auditA/scratch/w3-testsreview/measure_surrogate.py did this
round). They do NOT test the spectrum: a kernel with the wrong frequencies or
a generic Gaussian kernel also stays inside these thresholds on this driver
(notebook 07 shows such controls) — the spectrum is tested by layers (1) and
(2).
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from traqmania.agents.quantum import lightcone, surrogate
from traqmania.agents.quantum.qdqn import QuantumQFunction
from traqmania.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "traqmania" / "weights" / "quantum_oval.npz"


def _generic(n: int, layers: int, seed: int = 0) -> QuantumQFunction:
    """A circuit with generic (non-special) parameters: nothing cancels by accident."""
    rng = np.random.default_rng(100 * n + 10 * layers + seed)
    qfunc = QuantumQFunction({"n_qubits": n, "n_layers": layers})
    qfunc.lam = rng.uniform(1.0, 4.0, size=qfunc.lam.shape)
    qfunc.theta = rng.uniform(-3.0, 3.0, size=qfunc.theta.shape)
    qfunc.w = rng.uniform(20.0, 60.0, size=qfunc.w.shape)
    qfunc.b = rng.uniform(-10.0, 10.0, size=qfunc.b.shape)
    return qfunc


# ------------------------------------------------------------------ spectrum


def test_spectrum_sizes_without_light_cone():
    """3**L distinct frequencies per feature for generic lam, 2L+1 when all lam agree."""
    lam = _generic(4, 4).lam
    spectrum = surrogate.frequency_spectrum(lam)
    assert [f.size for f in spectrum] == [81] * 4
    for i, freqs in enumerate(spectrum):
        assert np.all(np.diff(freqs) > 0)
        np.testing.assert_allclose(freqs, -freqs[::-1], atol=1e-12)  # symmetric
        assert 0.0 in freqs
        assert freqs.max() == pytest.approx(np.abs(lam[:, i]).sum())
    size = surrogate.spectrum_size(lam)
    assert size == {"per_feature": [81] * 4, "total": 3**16, "live_gates": 16, "bound": 3**16}

    at_init = np.full((4, 4), np.pi)  # every lam = pi: integer multiples of pi only
    for freqs in surrogate.frequency_spectrum(at_init):
        np.testing.assert_allclose(freqs, np.pi * np.arange(-4, 5), atol=1e-9)
    assert surrogate.spectrum_size(at_init)["total"] == 9**4


@pytest.mark.parametrize(("n", "layers"), [(4, 4), (6, 4), (8, 4), (10, 4), (5, 2), (8, 6)])
def test_spectrum_respects_the_light_cone(n, layers):
    """Per readout: 3**(live encoding gates on that qubit) frequencies per feature."""
    lam = _generic(n, layers).lam
    visible = lightcone.feature_visibility(n, layers, min(4, n))
    for a in range(min(4, n)):
        live = lightcone.readout_live_gates(n, layers, a)["enc"]
        np.testing.assert_array_equal(surrogate.encoding_mask(n, layers, a), live)
        sizes = [f.size for f in surrogate.frequency_spectrum(lam, a)]
        assert sizes == [3 ** int(k) for k in live.sum(axis=0)]
        # a feature has a non-trivial spectrum exactly when the readout can see it
        np.testing.assert_array_equal(np.array(sizes) > 1, visible[a])
        info = surrogate.spectrum_size(lam, a)
        assert info["live_gates"] == int(live.sum())
        assert info["total"] == info["bound"] == 3 ** int(live.sum())


def test_default_circuit_spectrum_numbers():
    """The numbers the notebook quotes for 4 qubits, 4 blocks."""
    lam = _generic(4, 4).lam
    assert surrogate.spectrum_size(lam, 0)["per_feature"] == [81, 27, 9, 27]
    assert surrogate.spectrum_size(lam, 0)["total"] == 3**12
    # 8 qubits: readout 0 is blind to the feature on the opposite qubit
    blind = surrogate.frequency_spectrum(_generic(8, 4).lam, 0)[4]
    np.testing.assert_array_equal(blind, [0.0])


def test_product_kernel_is_the_uniform_sum_over_the_spectrum():
    """prod (1 + 2 cos)/3 == 3**-G * sum over all index vectors of cos(omega . delta)."""
    rng = np.random.default_rng(3)
    lam = rng.uniform(1.0, 4.0, size=(2, 3))
    mask = surrogate.encoding_mask(3, 2, 0)
    x, y = rng.uniform(0, 1, size=(5, 3)), rng.uniform(0, 1, size=(4, 3))
    gates = np.argwhere(mask)
    index = np.stack(np.meshgrid(*[[-1, 0, 1]] * len(gates), indexing="ij"), -1)
    index = index.reshape(-1, len(gates))
    omega = np.zeros((index.shape[0], 3))
    for g, (layer, i) in enumerate(gates):
        omega[:, i] += index[:, g] * lam[layer, i]
    delta = x[:, None, :] - y[None, :, :]
    explicit = np.cos(delta @ omega.T).mean(axis=-1)
    np.testing.assert_allclose(surrogate.product_kernel(x, y, lam, mask), explicit, atol=1e-12)


# ----------------------------------------------------------------- exactness


@pytest.mark.parametrize(("n", "layers"), [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (4, 2)])
def test_full_spectrum_basis_reproduces_the_circuit(n, layers):
    """Least squares on the full (light-cone restricted) basis is exact."""
    qfunc = _generic(n, layers)
    rng = np.random.default_rng(n + layers)
    train = rng.uniform(0.0, 3.0, size=(1500, n))
    test = rng.uniform(0.0, 3.0, size=(300, n))
    fitted = surrogate.fit_surrogate(qfunc, train, method="full", ridge=0.0)
    expected_basis = sum(
        surrogate.spectrum_size(qfunc.lam, a)["total"] for a in range(qfunc.n_actions)
    )
    assert fitted.n_coefficients == expected_basis
    np.testing.assert_allclose(
        fitted.expectations(test), qfunc.expectations(test), atol=1e-8)
    np.testing.assert_allclose(fitted.q_values(test), qfunc.q_values(test), atol=1e-6)


def test_light_cone_restriction_loses_nothing():
    """Same exactness with 81 basis functions per readout as with all 729."""
    qfunc = _generic(3, 2)
    rng = np.random.default_rng(1)
    train = rng.uniform(0.0, 3.0, size=(2500, 3))
    test = rng.uniform(0.0, 3.0, size=(200, 3))
    cone = surrogate.fit_surrogate(qfunc, train[:1500], method="full", ridge=0.0)
    everything = surrogate.fit_surrogate(
        qfunc, train, method="full", ridge=0.0, light_cone=False, n_actions=1)
    assert cone.n_coefficients == 3 * 81
    assert everything.n_coefficients == 3**6
    np.testing.assert_allclose(cone.expectations(test), qfunc.expectations(test), atol=1e-8)
    np.testing.assert_allclose(
        everything.expectations(test)[:, 0], qfunc.expectations(test)[:, 0], atol=1e-7)


def test_wrong_spectrum_does_not_reproduce_the_circuit():
    """Control: the exact fit is not an artefact of a flexible basis."""
    qfunc = _generic(3, 2)
    rng = np.random.default_rng(2)
    train = rng.uniform(0.0, 3.0, size=(1500, 3))
    test = rng.uniform(0.0, 3.0, size=(300, 3))
    targets = qfunc.expectations(train)
    wrong = surrogate.FourierSurrogate(qfunc.lam * 1.3, 3, method="full", ridge=0.0)
    wrong.fit(train, targets)
    assert np.abs(wrong.expectations(test) - qfunc.expectations(test)).max() > 1e-2


def test_full_basis_refuses_big_circuits():
    qfunc = _generic(4, 4)
    obs = np.random.default_rng(0).uniform(size=(50, 4))
    with pytest.raises(ValueError, match="531441"):
        surrogate.fit_surrogate(qfunc, obs, method="full", ridge=0.0)


def test_slice_coefficients_are_exact_and_show_the_light_cone():
    qfunc = _generic(4, 4)
    base = np.array([0.3, 0.7, 0.1, 0.9])
    x = np.linspace(-2.0, 3.0, 101)
    for readout, feature in ((1, 1), (1, 3), (0, 2), (3, 0)):
        k, omega, coef = surrogate.slice_coefficients(
            qfunc.lam, qfunc.theta, base, feature, readout)
        assert k.shape == (81, 4) and omega.shape == coef.shape == (81,)
        obs = np.tile(base, (x.size, 1))
        obs[:, feature] = x
        series = np.exp(1j * np.outer(x, omega)) @ coef
        np.testing.assert_allclose(series.imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(series.real, qfunc.expectations(obs)[:, readout], atol=1e-12)
        # every frequency used is in the light-cone spectrum ...
        spectrum = surrogate.frequency_spectrum(qfunc.lam, readout)[feature]
        used = omega[np.abs(coef) > 1e-12]
        assert np.abs(used[:, None] - spectrum[None, :]).min(axis=1).max() < 1e-9
        # ... because a dead encoding gate carries no amplitude
        dead = ~surrogate.encoding_mask(4, 4, readout)[:, feature]
        assert np.abs(coef[(k[:, dead] != 0).any(axis=1)]).max(initial=0.0) < 1e-12


def test_gate_coefficients_are_exact_and_count_the_real_spectrum():
    """The product spectrum is an upper bound: the gate-wise transform reproduces
    the circuit exactly and shows which index vectors the structure rules out."""
    qfunc, other = _generic(4, 3), _generic(4, 3, seed=1)
    obs = np.random.default_rng(4).uniform(-1.0, 2.0, size=(40, 4))
    for readout in range(4):
        gates, k, coef = surrogate.gate_coefficients(qfunc.theta, readout)
        np.testing.assert_array_equal(gates, np.argwhere(surrogate.encoding_mask(4, 3, readout)))
        assert k.shape == (3**8, 8) and coef.shape == (3**8,) and not k[0].any()
        # the series IS the circuit: omega_i = sum over the gates on qubit i of k * lam
        omega = np.zeros((k.shape[0], 4))
        for g, (layer, i) in enumerate(gates):
            omega[:, i] += k[:, g] * qfunc.lam[layer, i]
        series = np.exp(1j * obs @ omega.T) @ coef
        np.testing.assert_allclose(series, qfunc.expectations(obs)[:, readout], atol=1e-12)
        # every frequency vector it uses is in the product spectrum ...
        spectrum = surrogate.frequency_spectrum(qfunc.lam, readout)
        used = np.abs(coef) > 1e-12
        for i in range(4):
            assert np.abs(omega[used, i][:, None] - spectrum[i][None, :]).min(axis=1).max() < 1e-9
        # ... but most of the product spectrum is never used, whatever the angles:
        # the zeros are structural (same index vectors for unrelated parameters)
        other_used = np.abs(surrogate.gate_coefficients(other.theta, readout)[2]) > 1e-12
        np.testing.assert_array_equal(used, other_used)
        assert used.sum() == 792 < surrogate.spectrum_size(qfunc.lam, readout)["total"] == 3**8
        # e.g. the measured qubit's last encoding gate never has k = 0
        last = np.flatnonzero((gates[:, 0] == 2) & (gates[:, 1] == readout))[0]
        assert not used[k[:, last] == 0].any()
    with pytest.raises(ValueError, match="simulator runs"):
        surrogate.gate_coefficients(np.zeros((4, 6, 2)), 0)  # 15 live gates
    with pytest.raises(ValueError, match="theta"):
        surrogate.gate_coefficients(np.zeros((4, 6)), 0)


def test_random_features_estimate_the_product_kernel():
    """The RFF and kernel surrogates are one model: feature inner products are the
    Monte Carlo estimate of ``product_kernel`` (+ 1 for the constant feature)."""
    qfunc = _generic(4, 4)
    rng = np.random.default_rng(9)
    x, y = rng.uniform(size=(6, 4)), rng.uniform(size=(5, 4))
    model = surrogate.FourierSurrogate(qfunc.lam, method="rff")
    for mask in model.masks[:2]:
        exact = surrogate.product_kernel(x, y, qfunc.lam, mask) + 1.0
        errors = []
        for count in (250, 4000):
            freqs = surrogate._sample_frequencies(
                qfunc.lam, mask, count, np.random.default_rng(0), 1e-9)
            assert freqs.shape == (count, 4)
            assert np.all(freqs[:, ~mask.any(axis=0)] == 0.0)  # nothing outside the cone
            estimate = model._features(x, freqs) @ model._features(y, freqs).T
            errors.append(np.abs(estimate - exact).max())
        assert errors[1] < 0.06  # ~ a few times 1 / sqrt(2 * 4000)
        assert errors[1] < errors[0]


# ---------------------------------------------------- interface and plumbing


def test_surrogate_validates_its_inputs():
    lam = np.full((2, 3), np.pi)
    with pytest.raises(ValueError, match="method"):
        surrogate.FourierSurrogate(lam, method="magic")
    with pytest.raises(ValueError, match="ridge"):
        surrogate.FourierSurrogate(lam, method="kernel", ridge=0.0)
    with pytest.raises(ValueError, match="n_actions"):
        surrogate.FourierSurrogate(lam, n_actions=4)
    with pytest.raises(ValueError, match="shape"):
        surrogate.FourierSurrogate(np.ones(3))
    model = surrogate.FourierSurrogate(lam)
    with pytest.raises(RuntimeError, match="fit"):
        model.q_values(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="obs"):
        model.fit(np.zeros((5, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="targets"):
        model.fit(np.zeros((5, 3)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="lam"):
        surrogate.fit_surrogate(lambda obs: obs, np.zeros((5, 3)))
    with pytest.raises(TypeError):
        surrogate.fit_surrogate(object(), np.zeros((5, 3)), lam=lam)


@pytest.mark.parametrize("method", ["kernel", "rff"])
def test_callable_models_and_direct_q_fits(method):
    """Any callable works; with a head it fits <Z>, without one it fits Q itself."""
    qfunc = _generic(3, 2)
    rng = np.random.default_rng(5)
    train, test = rng.uniform(size=(600, 3)), rng.uniform(size=(200, 3))

    with_head = surrogate.fit_surrogate(
        qfunc.expectations, train, lam=qfunc.lam, head=(qfunc.w, qfunc.b),
        method=method, n_frequencies=200)
    assert (with_head.n_features, with_head.n_actions) == (3, 3)
    assert with_head.q_values(test).shape == (200, 3)
    # Q-values span tens of units; the ridge (1e-6) leaves errors of order 1e-2
    np.testing.assert_allclose(with_head.q_values(test), qfunc.q_values(test), atol=0.05)

    q_direct = surrogate.fit_surrogate(
        qfunc.q_values, train, lam=qfunc.lam, method=method, n_frequencies=200)
    np.testing.assert_allclose(q_direct.q_values(test), qfunc.q_values(test), atol=0.05)
    np.testing.assert_array_equal(q_direct.q_values(test), q_direct.expectations(test))
    assert np.isnan(surrogate.compare(q_direct, qfunc, test)["rmse_expectation"])
    assert surrogate.compare(with_head, qfunc, test)["rmse_expectation"] < 1e-3


def test_rff_is_deterministic_per_seed_and_handles_more_features_than_samples():
    qfunc = _generic(3, 2)
    rng = np.random.default_rng(6)
    train, test = rng.uniform(size=(40, 3)), rng.uniform(size=(20, 3))
    fits = [
        surrogate.fit_surrogate(qfunc, train, method="rff", n_frequencies=30, seed=seed)
        for seed in (1, 1, 2)
    ]
    np.testing.assert_array_equal(fits[0].q_values(test), fits[1].q_values(test))
    assert not np.array_equal(fits[0].q_values(test), fits[2].q_values(test))
    assert fits[0].n_coefficients == 3 * (1 + 2 * 30)  # 61 features > 40 samples: dual solve
    assert np.isfinite(fits[0].q_values(test)).all()
    # a spectrum smaller than the request just returns every frequency it has
    tiny = surrogate.fit_surrogate(_generic(2, 2), rng.uniform(size=(50, 2)), method="rff",
                                   n_frequencies=500)
    assert tiny.n_coefficients == 2 * 9


def test_kernel_prediction_matches_the_plain_kernel_formula(monkeypatch):
    """The shared-factor, chunked prediction path equals K(obs, X) @ coef."""
    qfunc = _generic(6, 4)  # readouts with different light cones
    rng = np.random.default_rng(8)
    train, test = rng.uniform(size=(120, 6)), rng.uniform(size=(50, 6))
    fitted = surrogate.fit_surrogate(qfunc, train, method="kernel")
    assert fitted.n_coefficients == 4 * 120
    targets = qfunc.expectations(train)
    mean, scale = targets.mean(axis=0), targets.std(axis=0)
    plain = np.empty((50, 4))
    for a, mask in enumerate(fitted.masks):
        gram = surrogate.product_kernel(train, train, qfunc.lam, mask) + 1.0
        coef = np.linalg.solve(gram + fitted.ridge * np.eye(120), (targets[:, a] - mean[a])
                               / scale[a])
        cross = surrogate.product_kernel(test, train, qfunc.lam, mask) + 1.0
        plain[:, a] = (cross @ coef) * scale[a] + mean[a]
    np.testing.assert_allclose(fitted.expectations(test), plain, atol=1e-9)
    monkeypatch.setattr(surrogate, "_KERNEL_CHUNK", 1)  # one observation per chunk
    np.testing.assert_allclose(fitted.expectations(test), plain, atol=1e-9)


class _TablePolicy:
    """q_values looked up from a fixed table, one row per observation index."""

    def __init__(self, table):
        self.table = np.asarray(table, dtype=float)

    def q_values(self, obs):
        return self.table[np.asarray(obs)[:, 0].astype(int)]


def test_compare_metrics_on_a_hand_made_case():
    obs = np.arange(4.0)[:, None]
    reference = _TablePolicy([[5, 1, 0], [0, 4, 3], [2, 2.5, 0], [9, 0, 1]])
    imitator = _TablePolicy([[5, 1, 0], [0, 3, 4], [3, 2.5, 0], [9, 0, 1]])
    result = surrogate.compare(imitator, reference, obs)
    # greedy actions: reference 0,1,1,0 / imitator 0,2,0,0; gaps 4, 1, 0.5, 8
    assert result["n"] == 4
    assert result["agreement"] == pytest.approx(0.5)
    assert result["gap_weighted_agreement"] == pytest.approx(12.0 / 13.5)
    assert result["regret"] == pytest.approx((1.0 + 0.5) / 4)
    assert result["mean_gap"] == pytest.approx(13.5 / 4)
    assert result["rmse"] == pytest.approx(np.sqrt(3.0 / 12))
    assert np.isnan(result["rmse_expectation"])
    same = surrogate.compare(reference, reference, obs)
    assert same["agreement"] == same["gap_weighted_agreement"] == 1.0
    assert same["rmse"] == same["regret"] == 0.0


def test_package_exports_the_module_lazily_and_qiskit_free():
    check = (
        "import sys, traqmania.agents.quantum as q; "
        "assert 'traqmania.agents.quantum.surrogate' not in sys.modules; "
        "q.surrogate.FourierSurrogate; "
        "import traqmania.agents.quantum.surrogate as s; "
        "s.collect_observations; s.drive_laps; "
        "assert 'qiskit' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", check], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ----------------------------------------------- the bundled 4-qubit driver


@pytest.fixture(scope="module")
def oval():
    """The bundled oval driver, its config, and held-out driving states."""
    config = load_config()
    qfunc = QuantumQFunction(config["circuit"])
    qfunc.set_params(np.load(WEIGHTS)["params"])
    held_out = surrogate.collect_observations(qfunc, "oval", config, 1500, seed=999)
    return qfunc, config, held_out


@pytest.fixture(scope="module")
def short_config():
    """Default config with 12 s episodes: keeps the sampling-plumbing test quick."""
    config = copy.deepcopy(load_config())
    config["reward"]["max_decisions"] = 120
    return config


def test_sampling_distributions(oval, short_config):
    qfunc, config = oval[0], short_config
    driven = surrogate.sample_observations(
        300, 4, track="oval", config=config, policy=qfunc, seed=1)
    assert driven.shape == (300, 4)
    assert driven.min() >= 0.0 and driven.max() <= 1.0
    np.testing.assert_array_equal(
        driven, surrogate.collect_observations(qfunc, "oval", config, 300, seed=1))
    # epsilon does something: without the random detours the cars visit other states
    greedy = surrogate.collect_observations(qfunc, "oval", config, 300, epsilon=0.0, seed=1)
    assert not np.array_equal(greedy, driven)
    # the log covers whole episodes by default; `decisions` shortens it
    early = surrogate.collect_observations(qfunc, "oval", config, 64, decisions=4, seed=1)
    assert early.shape == (64, 4)
    assert early[:, 3].max() < 0.5 < driven[:, 3].max()  # still accelerating off the line
    cube = surrogate.sample_observations(200, 4, seed=1)
    assert cube.shape == (200, 4)
    # driving states sit on a thin set; the cube does not
    assert np.linalg.det(np.cov(driven.T)) < 0.5 * np.linalg.det(np.cov(cube.T))
    mixed = surrogate.sample_observations(
        200, 4, track="oval", config=config, policy=qfunc, cube_fraction=0.25, seed=1)
    assert mixed.shape == (200, 4)
    np.testing.assert_array_equal(mixed[150:], cube[:50])
    with pytest.raises(ValueError, match="policy"):
        surrogate.sample_observations(10, 4, track="oval", config=config)


@pytest.fixture(scope="module", params=["kernel", "rff"])
def fitted(request, oval):
    """A surrogate of the bundled driver from 1000 driving samples, fixed seeds."""
    qfunc, config, _ = oval
    return surrogate.fit_surrogate(
        qfunc, track="oval", config=config, n_samples=1000, sample_seed=0,
        method=request.param, n_frequencies=256, ridge=1e-6, seed=0)


def test_surrogate_matches_the_bundled_driver_on_its_own_states(oval, fitted):
    qfunc, _, held_out = oval
    assert fitted.n_samples == 1000
    assert (fitted.n_features, fitted.n_actions) == (qfunc.n_features, qfunc.n_actions)
    result = surrogate.compare(fitted, qfunc, held_out)
    # about twice the worst of 20 fits (module docstring): the fits here are
    # seed 0 — kernel 0.075 / 0.018, rff 0.064 / 0.015
    assert result["nrmse"] < 0.15  # measured 0.019-0.080 over 10 seeds x 2 methods
    assert result["rmse_expectation"] < 0.04  # <Z> units; measured 0.0043-0.0196
    assert result["agreement"] > 0.97  # measured >= 0.998
    assert result["gap_weighted_agreement"] > 0.97  # measured >= 0.999
    assert result["regret"] < 0.1 * result["mean_gap"]  # measured <= 0.0034


def test_surrogate_drives_the_oval_like_the_quantum_driver(oval, fitted):
    """Asserted because it held for 10/10 sample seeds x both methods (see the
    module docstring): every surrogate lapped exactly as often as the quantum
    driver (36/36, and 12/12 here); the margin here is 3 episodes below."""
    qfunc, config, _ = oval
    quantum = surrogate.drive_laps(qfunc, "oval", config, episodes=12)
    classical = surrogate.drive_laps(fitted, "oval", config, episodes=12)
    assert quantum["lapped_episodes"] >= 1, "bundled oval driver no longer laps"
    assert classical["lapped_episodes"] >= max(1, quantum["lapped_episodes"] - 3)
    assert abs(classical["mean_s"] - quantum["mean_s"]) < 0.5
    assert classical["episodes"] == 12
    assert len(classical["lap_times"]) == classical["laps"]
    assert classical["best_s"] == pytest.approx(min(classical["lap_times"]))


def test_drive_laps_follows_the_records_protocol(oval):
    """Same seed, same episodes: the numbers of ``traqmania.records``."""
    from traqmania import records

    qfunc, config, _ = oval
    driver = next(d for d in records.discover_drivers() if d.id == "quantum_oval")
    official = records.evaluate(driver, "oval", episodes=6)
    ours = surrogate.drive_laps(qfunc, "oval", config, episodes=6)
    for key in ("episodes", "lapped_episodes", "crashed_before_lap", "laps"):
        assert ours[key] == official[key]
    assert round(ours["best_s"], 2) == official["best_s"]
    assert round(ours["mean_s"], 2) == official["mean_s"]
    assert ours["crashed"] >= ours["crashed_before_lap"]


def test_surrogate_from_shot_noise_samples_beats_one_noisy_evaluation(oval):
    """Regression averages the shot noise of its training samples away."""
    qfunc, config, held_out = oval
    noisy = surrogate.ShotNoisePolicy(qfunc, shots=256, seed=1)
    assert (noisy.n_features, noisy.n_actions) == (4, 4)
    train = surrogate.collect_observations(qfunc, "oval", config, 2000, decisions=200, seed=0)
    fitted = surrogate.fit_surrogate(noisy, train, method="rff", n_frequencies=256, ridge=1e-2)
    raw = surrogate.compare(noisy, qfunc, held_out)
    smoothed = surrogate.compare(fitted, qfunc, held_out)
    assert 0.005 < raw["rmse_expectation"] < 256**-0.5  # sqrt(mean(1 - E^2) / shots)
    assert smoothed["rmse_expectation"] < 0.6 * raw["rmse_expectation"]
    assert smoothed["agreement"] > raw["agreement"]
