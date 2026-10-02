"""Simultaneous Perturbation Stochastic Approximation (SPSA), pure numpy.

SPSA estimates the gradient of a scalar loss ``f(x)`` from exactly TWO
evaluations per iteration, independent of ``dim(x)``: both evaluations
perturb ALL coordinates at once along a random Rademacher direction
``delta`` (each entry independently +1 or -1),

    g_hat = (f(x + c_k * delta) - f(x - c_k * delta)) / (2 * c_k) * delta,

which is an unbiased-to-first-order estimate of the true gradient.  That
two-evaluations property is exactly what makes it the standard optimizer for
quantum hardware, where every loss evaluation is a batch of real circuit
executions and parameter-shift gradients would cost 2 * P evaluations.

Gain sequences follow Spall's standard recommendation:

    a_k = a / (k + 1 + A) ** 0.602        (step size, decaying)
    c_k = c / (k + 1) ** 0.101            (perturbation size, decaying)

with the stability offset ``A`` defaulting to ``iterations / 10``.

One probe pair is a very noisy gradient: the step points along a single
random direction, so most of it is a random walk across the loss surface.
Five optional safeguards (all off by default) keep a short run on a noisy
loss from walking away from a good starting point:

- ``scale`` — per-parameter units: perturbations and steps are multiplied by
  it, so parameter groups of very different magnitude (circuit angles vs an
  output head with |w| ~ 50) move comparably, and ``scale = 0`` freezes a
  parameter;
- ``max_step`` — a trust region: the per-coordinate step is clipped to it;
- ``resamplings`` — average the gradient estimate over several directions;
- ``blocking`` — evaluate the loss at the proposed point and only accept the
  step if it did not get worse by more than ``allowed_increase`` (the rule of
  the same name in Qiskit's SPSA optimizer); costs one more evaluation per
  iteration plus one for the starting point;
- ``accept`` — a veto: a callable that sees each proposed point and can
  refuse it without any evaluation of the loss (a constraint the loss does
  not know about, e.g. "the policy must still drive on the simulator").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

ALPHA = 0.602  # step-size decay exponent
GAMMA = 0.101  # perturbation decay exponent


def calibrate_gain(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    c: float,
    target_step: float,
    A: float,
    pairs: int = 1,
    seed: int | None = 0,
    scale: np.ndarray | None = None,
) -> tuple[float, float]:
    """Step-size gain ``a`` such that the FIRST step moves each parameter by
    about ``target_step`` (in units of ``scale``); ``2 * pairs`` evaluations.

    Averages the gradient magnitude ``|f(x0 + c*delta) - f(x0 - c*delta)| /
    (2c)`` over ``pairs`` random directions and returns ``(a, magnitude)``
    with ``a = target_step * (A + 1) ** 0.602 / magnitude``.  A single pair
    can land on a direction along which the loss barely changes, which
    inflates ``a`` (and every later step) by orders of magnitude — average a
    few.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    unit = np.ones_like(x0) if scale is None else np.asarray(scale, dtype=np.float64)
    rng = np.random.default_rng(seed)
    magnitudes = []
    for _ in range(max(1, int(pairs))):
        delta = rng.choice(np.array([-1.0, 1.0]), size=x0.shape) * unit
        magnitudes.append(abs(float(f(x0 + c * delta)) - float(f(x0 - c * delta))) / (2.0 * c))
    magnitude = float(np.mean(magnitudes))
    return target_step * (A + 1.0) ** ALPHA / max(magnitude, 1e-12), magnitude


def minimize(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    iterations: int,
    a: float = 0.15,
    c: float = 0.1,
    A: float | None = None,
    seed: int | None = 0,
    callback: Callable[[int, dict], None] | None = None,
    stop_event: Any = None,
    scale: np.ndarray | None = None,
    max_step: float | None = None,
    resamplings: int = 1,
    blocking: bool = False,
    allowed_increase: float = 0.0,
    accept: Callable[[np.ndarray], bool] | None = None,
) -> dict:
    """Minimize ``f`` with SPSA.

    By default exactly ``2 * iterations`` evaluations of ``f``; ``resamplings
    = R`` makes it ``2 * R`` per iteration and ``blocking`` adds one per
    iteration plus one up front (see the module docstring for ``scale``,
    ``max_step``, ``resamplings``, ``blocking`` / ``allowed_increase``).
    ``accept(candidate) -> bool`` can veto a proposed point; a vetoed step
    is not taken and — with blocking — not evaluated either.

    Returns ``{"x", "loss_history", "accepted", "vetoed", "evaluations"}``.
    Without blocking ``loss_history[k]`` is the mean of the probe evaluations
    of iteration ``k`` (a smoothed estimate of ``f`` near the iterate); with
    blocking it is the loss measured AT the iterate after iteration ``k``,
    ``loss_start`` the one measured at ``x0``.  ``accepted[k]`` says whether
    the step was taken, ``vetoed[k]`` whether ``accept`` refused it.
    ``callback(k, info)`` receives ``x``, ``loss``, ``f_plus``, ``f_minus``
    (of the last probe pair), ``a_k``, ``c_k``, ``accepted`` and ``vetoed``
    after each iteration.  ``stop_event``
    (optional): ``threading.Event``-like; when set, the loop returns early
    between iterations (cooperative cancellation).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    if A is None:
        A = iterations / 10.0
    rng = np.random.default_rng(seed)
    unit = None if scale is None else np.asarray(scale, dtype=np.float64)
    resamplings = max(1, int(resamplings))

    evaluations = 0
    fx = None
    if blocking:
        fx = float(f(x))
        evaluations += 1
    loss_start = fx

    loss_history: list[float] = []
    accepted: list[bool] = []
    vetoed: list[bool] = []
    for k in range(int(iterations)):
        if stop_event is not None and stop_event.is_set():
            break
        a_k = a / (k + 1 + A) ** ALPHA
        c_k = c / (k + 1) ** GAMMA

        g_hat = np.zeros_like(x)
        probes: list[float] = []
        for _ in range(resamplings):
            delta = rng.choice(np.array([-1.0, 1.0]), size=x.shape)
            step = c_k * delta if unit is None else c_k * delta * unit
            f_plus = float(f(x + step))
            f_minus = float(f(x - step))
            evaluations += 2
            probes += [f_plus, f_minus]
            # 1/delta == delta for Rademacher entries, so this IS the SPSA estimate.
            g_hat += (f_plus - f_minus) / (2.0 * c_k) * delta
        if resamplings > 1:
            g_hat /= resamplings
        update = a_k * g_hat
        if max_step is not None:
            update = np.clip(update, -max_step, max_step)
        if unit is not None:
            update = update * unit
        candidate = x - update

        veto = accept is not None and not accept(candidate)
        took = not veto
        if blocking:
            if took:
                f_next = float(f(candidate))
                evaluations += 1
                took = f_next <= fx + allowed_increase
            if took:
                x, fx = candidate, f_next
            loss = fx
        else:
            if took:
                x = candidate
            loss = 0.5 * (f_plus + f_minus) if resamplings == 1 else float(np.mean(probes))
        loss_history.append(loss)
        accepted.append(took)
        vetoed.append(veto)
        if callback is not None:
            callback(
                k,
                {
                    "x": x.copy(),
                    "loss": loss,
                    "f_plus": f_plus,
                    "f_minus": f_minus,
                    "a_k": a_k,
                    "c_k": c_k,
                    "accepted": took,
                    "vetoed": veto,
                },
            )

    result = {"x": x, "loss_history": loss_history, "accepted": accepted,
              "vetoed": vetoed, "evaluations": evaluations}
    if blocking:
        result["loss_start"] = loss_start
    return result
