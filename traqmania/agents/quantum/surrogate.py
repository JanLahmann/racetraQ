"""Classical Fourier surrogates of the trained circuit (numpy only).

The dequantization teaching experiment: once the quantum driver is trained,
how much of it can a purely classical model reproduce from input/output
samples alone?

The Fourier picture. Feature ``s_i`` enters the circuit only through the
encoding gates ``RY(lam[l, i] * s_i)``, one per block l. Conjugating an
operator by ``RY(phi)`` produces terms constant in phi and terms in
``cos(phi)`` / ``sin(phi)`` — nothing else — so every readout is a
multilinear polynomial in ``(1, cos(lam[l, i] s_i), sin(lam[l, i] s_i))``
over the encoding gates:

    <Z_a>(s) = sum_omega  c_omega * exp(i omega . s),
    omega_i  = sum_l  k[l, i] * lam[l, i],      k[l, i] in {-1, 0, +1}

(Schuld, Sweke & Meyer, Phys. Rev. A 103, 032430 (2021): a re-uploading model
is a truncated Fourier series whose frequencies are fixed by the encoding.)
At most ``3**L`` frequencies per feature, and fewer inside the light cone: an
encoding gate outside readout a's light cone (``lightcone.readout_live_gates``)
contributes nothing, so it is left out of the sum. This product set is an
UPPER BOUND — what the encoding gates allow; the rest of the circuit forces
many of the coefficients to zero whatever its parameters (at 4 qubits and 4
blocks 45072 of the 531441 index vectors k survive: :func:`gate_coefficients`).
A basis that contains the spectrum is all the fits below need. ``lam`` is trainable — the
spectrum is LEARNED and generally incommensurate — but once training stops it
is a fixed, known set, and the model is a linear combination of known basis
functions with unknown coefficients. That is a classical regression problem.

Three ways to fit it from black-box samples ``(s, <Z>(s))`` — or ``(s, Q(s))``:

``"full"``    every frequency of the product spectrum, plain least squares.
              Exact (to ~1e-8) with enough samples, but the basis has up to
              ``3**(live encoding gates)`` functions — the tiny-circuit check
              that validates the spectrum derivation, nothing more.
``"rff"``     random Fourier features: D frequency vectors drawn from the
              product spectrum (each ``k[l, i]`` independently) plus ridge
              regression (Landman et al., arXiv:2210.13200; Sweke et al.,
              Quantum 9, 1640 (2025)).
``"kernel"``  kernel ridge regression with the trigonometric product kernel the
              random features approximate — it has a closed form,

                  k_a(s, s') = prod_{live (l, i)}
                      (1 + 2 cos(lam[l, i] (s_i - s'_i))) / 3,

              so no frequency is ever enumerated.

A fitted :class:`FourierSurrogate` implements the inference half of the
``QFunction`` contract (``q_values``, ``n_features``, ``n_actions``), so it
drops into the same greedy driving loop as the quantum driver —
:func:`drive_laps` — and :func:`compare` scores it against the model it
imitates. What this does and does not show is spelled out in notebook 07: the
surrogate needs query access to an already TRAINED model, so it replaces
inference, not training (classical surrogates: Schreiber, Eisert & Meyer,
Phys. Rev. Lett. 131, 100803 (2023)).

numpy/stdlib only — no qiskit imports in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from traqmania.agents.quantum import lightcone

METHODS = ("full", "rff", "kernel")

# "full" builds one basis function per frequency vector; beyond this it is
# the wrong tool (the 4-qubit, 4-block driver would need 3**12 = 531441).
MAX_FULL_BASIS = 20_000

# Kernel predictions hold (L * n, batch chunk, training samples) factors at once;
# the chunk is sized to keep that below this many float64 values (~32 MB).
_KERNEL_CHUNK = 4_000_000

# gate_coefficients runs the simulator 3**G times (G live encoding gates of one
# readout): 12 gates = 531441 runs is a few seconds at 4 qubits; this is the cap.
MAX_DFT_GATES = 13
# ... in batches of about this many statevector amplitudes (~16 MB complex128)
_DFT_CHUNK = 1 << 20

# Two frequencies closer than this are one frequency (lam values that coincide,
# e.g. every lam = pi at initialisation, collapse the spectrum).
_FREQ_TOL = 1e-9


# ------------------------------------------------------------------ spectrum


def _as_lam(lam) -> np.ndarray:
    lam = np.asarray(lam, dtype=np.float64)
    if lam.ndim != 2:
        raise ValueError(f"lam must have shape (n_layers, n_qubits), got {lam.shape}")
    return lam


def encoding_mask(n_qubits: int, n_layers: int, readout: int | None = None) -> np.ndarray:
    """Encoding gates that can contribute frequencies: bool (L, n).

    ``readout=None`` ignores the light cone (every gate of every block: the
    a-priori bound); ``readout=a`` keeps only the gates inside the light cone
    of ``<Z_a>``. A single qubit has no ring and nothing to prune: every
    encoding gate acts on the measured qubit.
    """
    if readout is None or int(n_qubits) == 1:
        return np.ones((int(n_layers), int(n_qubits)), dtype=bool)
    return lightcone.readout_live_gates(n_qubits, n_layers, readout)["enc"]


def _distinct(values: np.ndarray, tol: float) -> np.ndarray:
    """Sorted values with runs closer than ``tol`` merged into their first member."""
    values = np.sort(values)
    keep = np.concatenate([[True], np.diff(values) > tol])
    return values[keep]


def frequency_spectrum(
    lam, readout: int | None = None, *, tol: float = _FREQ_TOL
) -> list[np.ndarray]:
    """Frequencies each feature can carry in a readout, one sorted array per feature.

    ``lam`` (L, n) are the input scalings (the number of blocks is read off its
    shape). Feature i's frequencies are all sums ``sum_l k_l * lam[l, i]`` with
    ``k_l in {-1, 0, 1}`` over the blocks whose encoding gate on qubit i counts:
    every block for ``readout=None``, only the gates in the light cone of
    ``<Z_readout>`` otherwise — an upper bound: every frequency the readout
    has is in the set, not every member need occur (:func:`gate_coefficients`
    counts the ones that do). Each array is symmetric about 0 and contains 0;
    a feature the readout cannot see has the single frequency 0. Sums closer
    than ``tol`` are one frequency, so coinciding ``lam`` values (all pi at
    initialisation) give ``2 L + 1`` frequencies rather than ``3**L``.
    """
    lam = _as_lam(lam)
    layers, n = lam.shape
    mask = encoding_mask(n, layers, readout)
    spectrum = []
    for i in range(n):
        freqs = np.zeros(1)
        for layer in np.flatnonzero(mask[:, i]):
            step = lam[layer, i]
            freqs = _distinct(np.concatenate([freqs - step, freqs, freqs + step]), tol)
        # an exact, symmetric zero: sums that cancel only up to rounding are 0
        freqs[np.abs(freqs) <= tol] = 0.0
        spectrum.append(freqs)
    return spectrum


def spectrum_size(lam, readout: int | None = None, *, tol: float = _FREQ_TOL) -> dict[str, Any]:
    """How big the Fourier basis of a readout is (the upper bound, see above).

    Returns ``{"per_feature": [...], "total": int, "live_gates": G,
    "bound": 3**G}``: distinct frequencies per feature, the size of the product
    spectrum (= the number of real basis functions cos/sin/constant that are
    enough to span the readout), the number of encoding gates that count, and the generic
    size ``3**G`` reached when no two frequency sums coincide. Python ints —
    these overflow int64 quickly.
    """
    lam = _as_lam(lam)
    per_feature = [int(f.size) for f in frequency_spectrum(lam, readout, tol=tol)]
    total = 1
    for count in per_feature:
        total *= count
    gates = int(encoding_mask(lam.shape[1], lam.shape[0], readout).sum())
    return {"per_feature": per_feature, "total": total, "live_gates": gates, "bound": 3**gates}


def _canonical_half(freqs: np.ndarray, tol: float) -> np.ndarray:
    """One representative of every +-omega pair, the zero vector dropped: (K, n).

    cos/sin of omega and of -omega span the same two functions, so the real
    basis only needs the vectors whose first non-zero component is positive.
    """
    nonzero = np.abs(freqs) > tol
    has_any = nonzero.any(axis=1)
    first = np.argmax(nonzero, axis=1)
    lead = freqs[np.arange(freqs.shape[0]), first]
    flipped = np.where((lead < 0)[:, None], -freqs, freqs)
    return flipped[has_any]


def _full_frequencies(spectrum: Sequence[np.ndarray], tol: float) -> np.ndarray:
    """Every frequency vector of the product spectrum, up to sign: (K, n)."""
    total = 1
    for freqs in spectrum:
        total *= freqs.size
    if total > MAX_FULL_BASIS:
        raise ValueError(
            f"the full Fourier basis has {total} functions (limit {MAX_FULL_BASIS}); "
            "use method='rff' or 'kernel' for circuits of this size"
        )
    grid = np.stack(np.meshgrid(*spectrum, indexing="ij"), axis=-1).reshape(-1, len(spectrum))
    half = _canonical_half(grid, tol)
    # +omega and -omega both map to the same canonical vector: keep one of each
    return _unique_rows(half, tol) if half.shape[0] else half


def _unique_rows(rows: np.ndarray, tol: float) -> np.ndarray:
    """Distinct rows (to ``tol``), in first-appearance order."""
    keys = np.round(rows / max(tol, 1e-12)).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return rows[np.sort(index)]


def _sample_frequencies(
    lam: np.ndarray, mask: np.ndarray, n_frequencies: int,
    rng: np.random.Generator, tol: float,
) -> np.ndarray:
    """Up to ``n_frequencies`` distinct random frequency vectors: (D, n).

    Each live encoding gate draws ``k in {-1, 0, +1}`` uniformly and
    independently; the gate sums give omega. That is the spectral measure of
    the product kernel in :func:`product_kernel`, so these are random Fourier
    features for exactly that kernel. Duplicates, sign twins and the zero
    vector are dropped (a tiny spectrum can run dry before reaching
    ``n_frequencies``).
    """
    found = np.zeros((0, lam.shape[1]))
    for _ in range(12):
        need = n_frequencies - found.shape[0]
        if need <= 0:
            break
        k = rng.integers(-1, 2, size=(2 * need + 8, *lam.shape))
        omega = (k * (lam * mask)[None]).sum(axis=1)
        found = _unique_rows(np.concatenate([found, _canonical_half(omega, tol)]), tol)
    return found[:n_frequencies]


def product_kernel(x, y, lam, mask: np.ndarray | None = None) -> np.ndarray:
    """The trigonometric product kernel of the circuit's encoding: (M, M').

        k(x, y) = prod_{(l, i) in mask} (1 + 2 cos(lam[l, i] (x_i - y_i))) / 3

    Expanding the product gives ``3**-G sum_k cos(omega_k . (x - y))`` over
    all ``3**G`` index vectors k — the kernel whose feature space is exactly
    the circuit's frequency spectrum (every index vector weighted equally),
    evaluated in O(G) without enumerating it. ``mask`` (L, n) selects the
    encoding gates (default: all).
    """
    lam = _as_lam(lam)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.ones(lam.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    kernel = np.ones((x.shape[0], y.shape[0]))
    for layer, i in np.argwhere(mask):
        ax, ay = lam[layer, i] * x[:, i], lam[layer, i] * y[:, i]
        # cos(a - b) = cos a cos b + sin a sin b: outer products, no (M, M') cos
        cos_diff = np.outer(np.cos(ax), np.cos(ay)) + np.outer(np.sin(ax), np.sin(ay))
        kernel *= (1.0 + 2.0 * cos_diff) / 3.0
    return kernel


def slice_coefficients(
    lam, theta, obs, feature: int, readout: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Fourier coefficients of ``<Z_readout>`` along one feature.

    With every other feature held at ``obs`` (n,), the readout as a function
    of ``s = obs[feature]`` is

        f(s) = sum_k c[k] * exp(i * omega[k] * s),   omega[k] = k . lam[:, feature]

    over the ``3**L`` index vectors ``k in {-1, 0, 1}^L``. Returns
    ``(k (3**L, L) int, omega (3**L,), c (3**L,) complex)``. The coefficients
    are computed, not fitted: the circuit is of degree one in the angle of
    each encoding gate separately, so simulating it on the three-point grid
    ``{0, 2 pi / 3, 4 pi / 3}`` per gate (``3**L`` runs, the gates' angles made
    independent) and taking the discrete Fourier transform is exact for any
    ``lam`` — no conditioning problem when frequencies nearly coincide. A gate
    outside the readout's light cone shows up as ``c[k] = 0`` whenever its
    ``k_l != 0``. White-box (it needs the simulator); the surrogates above
    never use it.
    """
    from traqmania.agents.quantum.fastsim import FastStatevectorSim

    lam = _as_lam(lam)
    layers, n = lam.shape
    if layers > 9:
        raise ValueError(f"3**L simulator runs: n_layers = {layers} is too deep for this")
    obs = np.asarray(obs, dtype=np.float64).reshape(n)
    index = np.stack(
        np.meshgrid(*[np.arange(3)] * layers, indexing="ij"), axis=-1
    ).reshape(-1, layers)  # (3**L, L), entries 0..2
    grid = index * (2.0 * np.pi / 3.0)  # one angle per block for the swept feature
    # fastsim takes angles lam * s: feed s = 1 and put the angles into "lam"
    angles = np.broadcast_to(lam * obs, (grid.shape[0], layers, n)).copy()
    angles[:, :, feature] = grid
    sim = FastStatevectorSim(n_qubits=n, n_layers=layers)
    ones = np.ones((1, n))
    values = np.array([sim.forward(ones, a, theta)[0, readout] for a in angles])
    k = index.copy()
    k[k == 2] = -1  # DFT index 2 on a 3-point grid is frequency -1
    coef = np.exp(-1j * (k @ grid.T)) @ values / grid.shape[0]
    return k, k @ lam[:, feature], coef


def gate_coefficients(theta, readout: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Fourier coefficients of ``<Z_readout>`` over ALL its live encoding gates.

    The angle of every encoding gate inside the readout's light cone is made
    an independent variable ``phi_g`` (G gates); the readout is then

        f(phi) = sum_k c[k] * exp(i * k . phi),      k in {-1, 0, 1}^G,

    and the circuit's ``<Z_readout>(s)`` is that series at
    ``phi_g = lam[l_g, i_g] * s[i_g]`` — the term k has the frequency vector
    ``omega_i = sum_{g on qubit i} k_g * lam[l_g, i_g]``. Returns
    ``(gates (G, 2) int rows [layer, qubit], k (3**G, G) int, c (3**G,)
    complex)``; ``k[0]`` is the zero vector (the constant term). ``theta``
    (L, n, 2) are the variational angles: the coefficients do not depend on
    ``lam`` at all (it only places the frequencies).

    This is what turns the product spectrum from an upper bound into a count:
    ``3**G`` index vectors are ALLOWED by the encoding gates, but the fixed
    structure of the circuit (|0..0> input, Z readout, the gate set) forces
    many ``c[k]`` to zero for every ``theta`` — e.g. the measured qubit's last
    encoding gate never has ``k = 0``, because ``RY(phi)`` turns Z into
    ``cos(phi) Z - sin(phi) X`` with no constant part. Computed like
    :func:`slice_coefficients` (three-point grid per gate, discrete Fourier
    transform; exact), which costs ``3**G`` simulator runs: fine for the
    4-qubit, 4-block circuit (G = 12), refused beyond ``MAX_DFT_GATES``.
    White-box; the surrogates never use it.
    """
    from traqmania.agents.quantum.fastsim import (
        apply_ry,
        apply_rz,
        cz_ring_diagonal,
        z_diagonals,
    )

    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 3 or theta.shape[2] != 2:
        raise ValueError(f"theta must have shape (n_layers, n_qubits, 2), got {theta.shape}")
    layers, n = theta.shape[:2]
    gates = np.argwhere(encoding_mask(n, layers, readout))
    n_gates = gates.shape[0]
    if n_gates > MAX_DFT_GATES:
        raise ValueError(
            f"{n_gates} live encoding gates need 3**{n_gates} simulator runs "
            f"(limit: {MAX_DFT_GATES} gates)"
        )
    index = np.stack(
        np.meshgrid(*[np.arange(3)] * n_gates, indexing="ij"), axis=-1
    ).reshape(-1, n_gates)  # (3**G, G), entries 0..2, row 0 all zero
    column = np.full((layers, n), -1)
    column[gates[:, 0], gates[:, 1]] = np.arange(n_gates)
    ring, z_diag = cz_ring_diagonal(n), z_diagonals(n)[int(readout)]
    values = np.empty(index.shape[0])
    step = max(1, _DFT_CHUNK >> n)
    for start in range(0, index.shape[0], step):
        angles = index[start:start + step] * (2.0 * np.pi / 3.0)
        psi = np.zeros((angles.shape[0], 1 << n), dtype=np.complex128)
        psi[:, 0] = 1.0
        for layer in range(layers):
            for i in range(n):  # dead encoding gates cannot matter: left out
                if column[layer, i] >= 0:
                    apply_ry(psi, i, angles[:, column[layer, i]], n)
            for i in range(n):
                apply_ry(psi, i, theta[layer, i, 0], n)
                apply_rz(psi, i, theta[layer, i, 1], n)
            psi *= ring
        values[start:start + step] = (psi.real**2 + psi.imag**2) @ z_diag
    coef = np.fft.fftn(values.reshape((3,) * n_gates)).reshape(-1) / index.shape[0]
    k = index.copy()
    k[k == 2] = -1  # DFT index 2 on a 3-point grid is frequency -1
    return gates, k, coef


# ----------------------------------------------------------------- surrogate


class FourierSurrogate:
    """A classical model of the circuit's readouts, fitted from samples.

    One independent regression per readout a (each has its own light cone,
    hence its own spectrum), all on the same sample inputs. ``lam`` (L, n) is
    the only thing taken from the circuit: it fixes the frequencies. Fit with
    :meth:`fit` on black-box samples; afterwards the object drives like a
    Q-function (``q_values(obs) -> (B, A)``, ``n_features``, ``n_actions``).

    method            ``"kernel"`` | ``"rff"`` | ``"full"`` (see the module docstring)
    n_frequencies     random frequency vectors per readout (``"rff"`` only;
                      each contributes a cos and a sin feature)
    ridge             regularisation strength, in units of the standardised
                      targets (0 is allowed for ``"full"``: plain least
                      squares). Exact samples want it tiny; noisy samples
                      (shots, hardware) want it larger, or the fit chases the noise
    light_cone        restrict each readout's spectrum to its light cone
                      (False: use every encoding gate — the a-priori basis)
    head              optional ``(w, b)``: when the fitted targets are raw
                      expectations ``<Z_a>``, ``q_values`` returns
                      ``w * fit + b``; without it ``q_values`` returns the fit
                      itself (targets were Q-values)
    """

    def __init__(
        self,
        lam,
        n_actions: int | None = None,
        *,
        method: str = "kernel",
        n_frequencies: int = 1024,
        ridge: float = 1e-6,
        light_cone: bool = True,
        head: tuple[Any, Any] | None = None,
        seed: int = 0,
    ):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if ridge < 0 or (ridge == 0 and method != "full"):
            raise ValueError("ridge must be > 0 (0 is allowed only for method='full')")
        self.lam = _as_lam(lam).copy()
        self.n_layers, self.n_features = self.lam.shape
        self.n_actions = min(4, self.n_features) if n_actions is None else int(n_actions)
        if not 1 <= self.n_actions <= self.n_features:
            raise ValueError(
                f"n_actions must be in 1..n_qubits = {self.n_features}, got {self.n_actions}"
            )
        self.method = method
        self.n_frequencies = int(n_frequencies)
        self.ridge = float(ridge)
        self.light_cone = bool(light_cone)
        self.seed = int(seed)
        self.w = self.b = None
        if head is not None:
            self.w = np.asarray(head[0], dtype=np.float64).copy()
            self.b = np.asarray(head[1], dtype=np.float64).copy()
        self.masks = [
            encoding_mask(self.n_features, self.n_layers, a if self.light_cone else None)
            for a in range(self.n_actions)
        ]
        self.n_samples = 0
        self._models: list[dict[str, np.ndarray]] = []
        self._x = self._cos = self._sin = None  # kernel method: the training inputs
        self._mean = np.zeros(self.n_actions)
        self._scale = np.ones(self.n_actions)

    # -------------------------------------------------------------- fitting

    def _features(self, obs: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """[1, cos(omega . s) / sqrt(D), sin(omega . s) / sqrt(D)]: (B, 1 + 2 D).

        With the 1/sqrt(D) scale the inner product of the trigonometric
        features is the Monte Carlo estimate of the product kernel, so
        ``ridge`` means the same for ``"rff"`` and ``"kernel"``. The constant
        column is the intercept (the zero frequency), kept at full weight.
        """
        phase = obs @ freqs.T
        scale = 1.0 / np.sqrt(max(freqs.shape[0], 1))
        ones = np.ones((obs.shape[0], 1))
        return np.concatenate([ones, np.cos(phase) * scale, np.sin(phase) * scale], axis=1)

    def _kernel(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Product kernel plus 1 — the constant feature of :meth:`_features`."""
        return product_kernel(x, y, self.lam, mask) + 1.0

    def _kernel_predict(self, obs: np.ndarray) -> np.ndarray:
        """Kernel-method predictions for all readouts: (B, A), standardised units.

        Every readout's kernel is a product over a subset of the same L * n
        per-gate factors, so the factors are computed once per batch (the
        training side's cos/sin are cached at fit time) and each readout
        multiplies its own subset — what makes driving with the kernel
        surrogate cheap. Chunked over the batch to bound memory.
        """
        n_train = self._x.shape[0]
        out = np.empty((obs.shape[0], self.n_actions))
        step = max(1, _KERNEL_CHUNK // (self.lam.size * n_train))
        for start in range(0, obs.shape[0], step):
            angle = self.lam[:, :, None] * obs[start:start + step].T[None, :, :]  # (L, n, b)
            factors = (1.0 + 2.0 * (
                np.cos(angle)[..., None] * self._cos[:, :, None, :]
                + np.sin(angle)[..., None] * self._sin[:, :, None, :]
            )) / 3.0  # (L, n, b, M)
            for a, (mask, model) in enumerate(zip(self.masks, self._models, strict=True)):
                kernel = np.ones(factors.shape[2:])
                for layer, i in np.argwhere(mask):
                    kernel *= factors[layer, i]
                out[start:start + step, a] = (kernel + 1.0) @ model["coef"]
        return out

    def fit(self, obs, targets) -> FourierSurrogate:
        """Fit to samples: ``obs`` (M, n_features), ``targets`` (M, n_actions).

        ``targets[:, a]`` are the values of readout a at the sample inputs —
        expectations ``<Z_a>`` (exact, shot-sampled or from hardware) when a
        ``head`` was given, Q-values otherwise. Each column is centred and
        scaled before the regression, so ``ridge`` is unit-free.
        """
        obs = np.asarray(obs, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != self.n_features:
            raise ValueError(f"obs must have shape (M, {self.n_features}), got {obs.shape}")
        if targets.shape != (obs.shape[0], self.n_actions):
            raise ValueError(
                f"targets must have shape ({obs.shape[0]}, {self.n_actions}), got {targets.shape}"
            )
        self.n_samples = obs.shape[0]
        self._mean = targets.mean(axis=0)
        spread = targets.std(axis=0)
        self._scale = np.where(spread > 0, spread, 1.0)
        y = (targets - self._mean) / self._scale

        rng = np.random.default_rng(self.seed)
        self._models = []
        if self.method == "kernel":
            self._x = obs.copy()
            angle = self.lam[:, :, None] * obs.T[None, :, :]  # (L, n, M)
            self._cos, self._sin = np.cos(angle), np.sin(angle)
        for a, mask in enumerate(self.masks):
            if self.method == "kernel":
                gram = self._kernel(obs, obs, mask)
                gram[np.diag_indices_from(gram)] += self.ridge
                self._models.append({"coef": np.linalg.solve(gram, y[:, a])})
                continue
            if self.method == "full":
                spectrum = frequency_spectrum(self.lam, a if self.light_cone else None)
                freqs = _full_frequencies(spectrum, _FREQ_TOL)
            else:
                freqs = _sample_frequencies(self.lam, mask, self.n_frequencies, rng, _FREQ_TOL)
            phi = self._features(obs, freqs)
            if self.ridge == 0.0:
                coef = np.linalg.lstsq(phi, y[:, a], rcond=None)[0]
            elif phi.shape[1] <= phi.shape[0]:  # primal ridge
                normal = phi.T @ phi
                normal[np.diag_indices_from(normal)] += self.ridge
                coef = np.linalg.solve(normal, phi.T @ y[:, a])
            else:  # more features than samples: the dual form is the smaller system
                gram = phi @ phi.T
                gram[np.diag_indices_from(gram)] += self.ridge
                coef = phi.T @ np.linalg.solve(gram, y[:, a])
            self._models.append({"freqs": freqs, "coef": coef})
        return self

    # ------------------------------------------------------------ inference

    @property
    def n_coefficients(self) -> int:
        """Fitted coefficients over all readouts (one per sample for
        ``"kernel"``, intercept + cos + sin per frequency otherwise)."""
        return sum(m["coef"].size for m in self._models)

    def expectations(self, obs) -> np.ndarray:
        """The fitted readouts, before any output head: (B, F) -> (B, A)."""
        if not self._models:
            raise RuntimeError("the surrogate has not been fitted yet (call fit)")
        obs = np.asarray(obs, dtype=np.float64)
        if self.method == "kernel":
            return self._kernel_predict(obs) * self._scale + self._mean
        out = np.empty((obs.shape[0], self.n_actions))
        for a, model in enumerate(self._models):
            out[:, a] = self._features(obs, model["freqs"]) @ model["coef"]
        return out * self._scale + self._mean

    def q_values(self, obs) -> np.ndarray:
        """Q-values for a batch of observations: (B, F) -> (B, A)."""
        values = self.expectations(obs)
        if self.w is None:
            return values
        return values * self.w + self.b


def _resolve_model(model, lam, head, n_actions):
    """(query function, lam, head, n_actions) for a model.

    A circuit Q-function (anything with ``expectations``, ``lam``, ``w``, ``b``)
    is queried for raw expectations and lends its ``lam`` and output head;
    any other callable is queried as-is and needs ``lam`` passed explicitly.
    """
    if hasattr(model, "expectations") and hasattr(model, "lam"):
        lam = model.lam if lam is None else lam
        if head is None and getattr(model, "w", None) is not None:
            head = (model.w, model.b)
        n_actions = getattr(model, "n_actions", n_actions) if n_actions is None else n_actions
        return model.expectations, lam, head, n_actions
    if not callable(model):
        raise TypeError("model must be a circuit Q-function or a callable obs -> (B, A) values")
    if lam is None:
        raise ValueError("a callable model needs lam=(n_layers, n_qubits) input scalings")
    return model, lam, head, n_actions


def fit_surrogate(
    model,
    obs=None,
    *,
    lam=None,
    head: tuple[Any, Any] | None = None,
    n_actions: int | None = None,
    n_samples: int = 1000,
    track=None,
    config: dict | None = None,
    policy=None,
    epsilon: float = 0.1,
    cube_fraction: float = 0.0,
    sample_seed: int = 0,
    **surrogate_kwargs,
) -> FourierSurrogate:
    """Query ``model`` at sample inputs and fit a :class:`FourierSurrogate` to it.

    ``model`` is a trained ``QuantumQFunction`` (queried through
    ``expectations``; its ``lam`` and output head are reused) or ANY callable
    ``obs (B, n) -> (B, A)`` readout values — a shot-sampled or hardware
    Estimator works the same way, each sample costing one circuit evaluation;
    for a callable pass ``lam`` (and ``head=(w, b)`` if it returns raw
    expectations and ``q_values`` should apply the output head).

    Where the samples come from: pass ``obs`` (M, n) directly, or leave it
    None and give a sample budget ``n_samples`` plus a distribution —
    ``track`` + ``config`` for states visited by epsilon-greedy driving
    (:func:`collect_observations`; the driver is ``policy``, default
    ``model``), ``cube_fraction`` of the budget from the uniform cube
    ``[0, 1]^n`` instead (1.0, or no track: all of it). Remaining keyword
    arguments go to :class:`FourierSurrogate` (``method``, ``n_frequencies``,
    ``ridge``, ``light_cone``, ``seed``).
    """
    query, lam, head, n_actions = _resolve_model(model, lam, head, n_actions)
    lam = _as_lam(lam)
    if obs is None:
        obs = sample_observations(
            n_samples, lam.shape[1], track=track, config=config,
            policy=model if policy is None else policy,
            epsilon=epsilon, cube_fraction=cube_fraction, seed=sample_seed,
        )
    obs = np.asarray(obs, dtype=np.float64)
    targets = np.asarray(query(obs), dtype=np.float64)
    if n_actions is None:
        n_actions = targets.shape[1]
    surrogate = FourierSurrogate(lam, n_actions, head=head, **surrogate_kwargs)
    return surrogate.fit(obs, targets[:, : surrogate.n_actions])


# ------------------------------------------------------------------ sampling


def _load_track(track, config: dict):
    """A Track from a bundled name / path (or the Track itself)."""
    if isinstance(track, str):
        from traqmania.env.track import Track

        return Track.load(track, config["track"]["resample_spacing"])
    return track


def collect_observations(
    policy, track, config: dict, n_samples: int, *,
    epsilon: float = 0.1, n_envs: int = 16, decisions: int | None = None, seed: int = 0,
) -> np.ndarray:
    """Observations visited by epsilon-greedy driving: (n_samples, n_features).

    ``policy`` is anything with ``q_values``; ``n_envs`` cars drive in
    parallel for ``decisions`` decisions each (default: one full episode
    budget, ``[reward] max_decisions`` — several laps, so the log covers the
    whole track and not just the first seconds after the start; more when
    ``n_samples`` needs it), each decision greedy with probability
    ``1 - epsilon`` and uniformly random otherwise, crashed cars respawning at
    the start line. ``n_samples`` rows are then drawn at random from that
    driving log (without replacement, in random order). The small random
    detours matter: they cover the neighbourhood of the racing line, where an
    imperfect imitator ends up.

    Cost accounting: ``n_samples`` is the number of states handed on for
    labelling; the driving itself evaluates ``policy`` once per decision. For
    a quantum driver that log is a by-product of any laps it has driven.
    """
    from traqmania.env.racing_env import RacingEnv

    n_samples = int(n_samples)
    env = RacingEnv(_load_track(track, config), config, n_envs=n_envs, seed=seed)
    if decisions is None:
        decisions = env.max_decisions
    decisions = max(int(decisions), -(-n_samples // n_envs))
    rng = np.random.default_rng(seed)
    obs = env.reset()
    visited = [obs]
    for _ in range(decisions - 1):
        actions = np.argmax(policy.q_values(obs), axis=1)
        explore = rng.random(n_envs) < epsilon
        actions = np.where(explore, rng.integers(0, env.n_actions, size=n_envs), actions)
        obs, _, _, _ = env.step(actions)
        visited.append(obs)
    return rng.permutation(np.concatenate(visited))[:n_samples]


def sample_observations(
    n_samples: int, n_features: int, *, track=None, config: dict | None = None,
    policy=None, epsilon: float = 0.1, cube_fraction: float = 0.0, seed: int = 0,
) -> np.ndarray:
    """A sample budget drawn from the driving distribution and/or the uniform cube.

    ``round(cube_fraction * n_samples)`` points are uniform on ``[0, 1]^n``;
    the rest are states visited by epsilon-greedy driving of ``policy`` on
    ``track`` (:func:`collect_observations`). Without a track everything comes
    from the cube.
    """
    n_samples = int(n_samples)
    n_cube = n_samples if track is None else int(round(cube_fraction * n_samples))
    n_cube = min(max(n_cube, 0), n_samples)
    parts = []
    if n_samples - n_cube:
        if config is None or not hasattr(policy, "q_values"):
            raise ValueError(
                "sampling driving states needs track, config and a policy with q_values"
            )
        parts.append(collect_observations(
            policy, track, config, n_samples - n_cube, epsilon=epsilon, seed=seed))
    if n_cube:
        rng = np.random.default_rng(seed + 1)
        parts.append(rng.uniform(0.0, 1.0, size=(n_cube, int(n_features))))
    return np.concatenate(parts)


# ------------------------------------------------------------------- metrics


def compare(surrogate, reference, obs) -> dict[str, float]:
    """Score ``surrogate`` against ``reference`` on the observations ``obs``.

    Both expose ``q_values``. Returned (all on this sample, so use held-out
    observations from the distribution you care about):

    rmse                    RMS error of the Q-values, all actions pooled
    nrmse                   rmse / RMS deviation of the reference Q-values from
                            their per-action means (1.0 = no better than a constant)
    rmse_expectation        RMS error of the raw readouts ``<Z_a>`` (only when
                            both sides expose ``expectations`` behind an
                            output head ``w``, ``b``; NaN otherwise)
    agreement               fraction of states with the same greedy action
    gap_weighted_agreement  the same, each state weighted by the reference's
                            action gap (best minus second-best Q): disagreeing
                            where the reference barely prefers its action is cheap
    regret                  mean reference Q-value given up by following the
                            surrogate's greedy action instead of the reference's
    mean_gap                mean action gap of the reference (the scale of regret)
    """
    obs = np.asarray(obs, dtype=np.float64)
    q_ref = np.asarray(reference.q_values(obs), dtype=np.float64)
    q_sur = np.asarray(surrogate.q_values(obs), dtype=np.float64)
    rows = np.arange(obs.shape[0])
    a_ref, a_sur = q_ref.argmax(axis=1), q_sur.argmax(axis=1)
    agree = a_ref == a_sur
    ordered = np.sort(q_ref, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    rmse = float(np.sqrt(np.mean((q_sur - q_ref) ** 2)))
    spread = float(np.sqrt(np.mean((q_ref - q_ref.mean(axis=0)) ** 2)))
    rmse_e = float("nan")
    if all(getattr(m, "w", None) is not None and hasattr(m, "expectations")
           for m in (surrogate, reference)):
        diff = np.asarray(surrogate.expectations(obs)) - np.asarray(reference.expectations(obs))
        rmse_e = float(np.sqrt(np.mean(diff**2)))
    return {
        "n": int(obs.shape[0]),
        "rmse": rmse,
        "nrmse": rmse / spread if spread > 0 else float("nan"),
        "rmse_expectation": rmse_e,
        "agreement": float(agree.mean()),
        "gap_weighted_agreement": float((gap * agree).sum() / gap.sum()) if gap.sum() > 0
        else float("nan"),
        "regret": float(np.mean(q_ref[rows, a_ref] - q_ref[rows, a_sur])),
        "mean_gap": float(gap.mean()),
    }


# ------------------------------------------------------------------- driving


def drive_laps(
    policy, track, config: dict, episodes: int = 36, seed: int = 20_000,
) -> dict[str, Any]:
    """Greedy standing-start episodes of any ``q_values`` policy on a track.

    The protocol of ``traqmania.records`` (same default seed, so a quantum
    driver scores exactly as in the records table): ``episodes`` parallel cars
    with distinct seeded start jitter, each driving greedily until its first
    crash or the decision budget; laps completed on the way are timed.
    Returns ``episodes``, ``lapped_episodes`` (at least one lap),
    ``crashed_before_lap``, ``crashed`` (ended off track at any point),
    ``laps``, ``best_s`` / ``mean_s`` (None without a lap) and ``lap_times``.
    """
    from traqmania.env.racing_env import RacingEnv

    episodes = int(episodes)
    env = RacingEnv(_load_track(track, config), config, n_envs=episodes, seed=seed)
    obs = env.reset()
    finished = np.zeros(episodes, dtype=bool)
    crashed = np.zeros(episodes, dtype=bool)
    laps = np.zeros(episodes, dtype=int)
    prev_lap = np.zeros(episodes, dtype=int)
    lap_times: list[float] = []
    actions = np.zeros(episodes, dtype=np.intp)
    for _ in range(env.max_decisions + 1):
        # finished cars keep rolling (auto-reset) but are ignored: no need to ask the policy
        actions[~finished] = np.argmax(policy.q_values(obs[~finished]), axis=1)
        obs, _, done, info = env.step(actions)
        cur_lap = np.asarray(info["lap"], dtype=int)
        last = np.asarray(info["last_lap_time"], dtype=np.float64)
        event = (cur_lap > prev_lap) & ~finished
        lap_times.extend(last[event].tolist())
        laps[event] += 1
        done = np.asarray(done, dtype=bool)
        crashed |= done & ~finished & np.asarray(info["off_track"], dtype=bool)
        finished |= done  # frozen at the first done: auto-reset starts no second episode
        prev_lap = np.where(done, 0, cur_lap)
        if finished.all():
            break
    times = np.asarray(lap_times, dtype=np.float64)
    return {
        "episodes": episodes,
        "lapped_episodes": int(np.sum(laps >= 1)),
        "crashed_before_lap": int(np.sum(finished & (laps == 0))),
        "crashed": int(crashed.sum()),
        "laps": int(times.size),
        "best_s": float(times.min()) if times.size else None,
        "mean_s": float(times.mean()) if times.size else None,
        "lap_times": times.tolist(),
    }


class ShotNoisePolicy:
    """Wrap a circuit Q-function so its readouts are finite-shot estimates.

    Each ``<Z_a>`` is replaced by the mean of ``shots`` simulated +-1
    outcomes (binomial sampling of the exact expectation) — what a sampling
    backend returns. Lets :func:`fit_surrogate` be tried on noisy samples
    without leaving numpy: pass ``ShotNoisePolicy(q, shots).expectations`` as
    the model (with ``lam`` and ``head``), or the object itself.
    """

    def __init__(self, qfunc, shots: int = 1024, seed: int = 0):
        self.qfunc = qfunc
        self.shots = int(shots)
        self.lam, self.w, self.b = qfunc.lam, qfunc.w, qfunc.b
        self.n_features, self.n_actions = qfunc.n_features, qfunc.n_actions
        self._rng = np.random.default_rng(seed)

    def expectations(self, obs) -> np.ndarray:
        exact = np.clip(self.qfunc.expectations(obs), -1.0, 1.0)
        ones = self._rng.binomial(self.shots, (1.0 + exact) / 2.0)
        return 2.0 * ones / self.shots - 1.0

    def q_values(self, obs) -> np.ndarray:
        return self.expectations(obs) * self.w + self.b
