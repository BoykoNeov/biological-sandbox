"""Classic Lotka-Volterra predator-prey — two exact invariants, far from equilibrium.

    dx/dt = x (alpha - beta y)          prey
    dy/dt = y (delta x - gamma)         predator

Every other deterministic model in this project is validated *at* or *near* a
fixed point: Hodgkin-Huxley's resting state, Gray-Scott's homogeneous state,
gLV's interior equilibrium. This one is validated **on the orbit**, and that is
the reason it earns its own file rather than being a two-species ``glv`` case
(which it is not, anyway — the pure predator-prey system has no self-limitation,
so its interior fixed point is a *center*, not an attractor, and ``glv``'s
stability guard would rightly refuse it).

**The interior fixed point is a center**, ``(x*, y*) = (gamma/delta,
alpha/beta)``, with Jacobian eigenvalues ``+- i sqrt(alpha gamma)`` — purely
imaginary. Nothing decays, so there is no relaxation rate to measure and no
attractor to converge to. What there is instead is stronger: two quantities that
are exactly conserved along *every* orbit, at *any* amplitude.

**1. The conserved quantity.**

    V = delta x - gamma ln x + beta y - alpha ln y

is constant along trajectories (``dV/dt = 0`` identically). This is the
``analytic_predictions`` claim, and it is a genuine category-A check that lives
far from any fixed point. Its honest limit, stated rather than glossed: a params
typo made *consistently* in both the RHS and ``V`` would be invisible here, so
the hand-transcribed RHS in ``tests/test_lotka_volterra.py`` is load-bearing, not
decorative — the same division of labour as Hodgkin-Huxley's textbook RHS.

The drift is **strongly amplitude-dependent** — measured ``9.3e-15`` at
``amp = 0.4`` against ``1.6e-11`` at ``amp = 4.0``, a factor of 1670 at identical
``dt``. A ``sem_floor`` measured at one amplitude would be wrong at the other, so
it is derived per configuration by Richardson in ``dt``.

**2. The time-average identity.** Over one full cycle, ``<x> = x*`` and
``<y> = y*`` exactly, for any amplitude — integrate ``d(ln x)/dt = alpha - beta y``
around a closed orbit and the left side telescopes to zero. Checked in the tests,
where the error floor is the **cycle-endpoint detection**, not the integrator.

**3. The small-oscillation period is a LIMIT, not an identity.** Linearizing
about the center gives ``2 pi / sqrt(alpha gamma)``, but the true period *grows*
with amplitude, so this must be checked by extrapolation and never as a tolerance
at one amplitude. Measured excess over the linear value, with ``amp`` the initial
displacement in ``x``:

===========  ==================  ================
``amp``      excess              excess/``amp^2``
===========  ==================  ================
0.05         8.340e-05           0.033360
0.10         3.309e-04           0.033087
0.20         1.302e-03           0.032558
0.40         5.049e-03           0.031555
0.80         1.904e-02           0.029748
4.00         3.331e-01           0.020816
===========  ==================  ================

so the excess is ``O(amp^2)`` and the coefficient is only converging *below*
``amp ~ 0.4``. Richardson in the amplitude at order 2 — ``(4 P(a/2) - P(a))/3`` —
then extrapolates to the closed form with error ``5.35e-5 / 7.06e-6 / 9.08e-7``
for ``a = 0.4 / 0.2 / 0.1``, a residual ratio of ~7.8. This is **Richardson in
the amplitude for the fifth time in this project** (HH's transient, Gray-Scott's
linearization, gLV's relaxation twice, now a small-oscillation limit).

**The amplitude scale here is this file's own, and the plan's numbers do not
transfer to it.** ``amp`` is the initial displacement in ``x``, with ``y`` started
at ``y*``. The planning slice recorded ``9.491 / 9.805 / 11.271`` at what it
called amplitude ``1.2 / 2.0 / 4.0``; on *this* scale ``9.4913`` occurs at
``amp = 0.8`` and ``9.8053`` at ``amp = 4.0``, so the two scales are not related
by any constant factor and the slice did not record its convention. Every number
above was re-measured here. The definition is fixed in params precisely so that
"amplitude" is a well-defined small parameter for the extrapolation.

**No ``analytic_predictions`` for the period or the average.** Both are
properties of a *cycle*, and the protocol's contract is the replicate mean of an
observable's **final** value. Bending either into a final-value observable would
mean carrying a running integral whose window does not align with ``t_max``;
they are checked in the tests, where the instrument can be stated honestly.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import rk4_step
from sandbox.core.registry import register

STATE_KEYS: tuple[str, str] = ("x", "y")


@dataclass(frozen=True)
class LVParams:
    """Lotka-Volterra parameters (all plain, JSON-serializable numbers).

    ``amp`` is the initial displacement of the prey from ``x*``, with the predator
    started exactly at ``y*``. Parameterizing the initial condition by a single
    amplitude rather than by a free ``(x0, y0)`` is deliberate: the period claim
    is a **limit as the amplitude vanishes**, so the small parameter has to be a
    param, not a derived quantity. The defaults are the planning slice's case.
    """

    alpha: float = 1.1
    beta: float = 0.4
    gamma: float = 0.4
    delta: float = 0.1
    amp: float = 0.4
    t_max: float = 100.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "delta"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive, got {getattr(self, name)}; with a "
                    "non-positive rate the interior fixed point leaves the positive "
                    "quadrant and there are no closed orbits to validate on"
                )
        if self.gamma / self.delta + self.amp <= 0:
            raise ValueError(
                f"amp ({self.amp}) puts the prey at "
                f"{self.gamma / self.delta + self.amp:g} <= 0; the orbit must start "
                "inside the positive quadrant"
            )
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        steps = round(self.t_max / self.dt)
        if steps < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        if abs(steps * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{steps} steps land at {steps * self.dt}"
            )


@dataclass(frozen=True)
class LVState:
    """``y = [prey, predator]``, the step counter, and the embedded params + RHS."""

    y: np.ndarray
    step_index: int
    t: float
    params: LVParams
    rhs: Callable[[np.ndarray], np.ndarray]


def n_lv_steps(params: LVParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


def lv_rhs(params: LVParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous RHS ``f(y) -> dy/dt`` for ``y = [prey, predator]``."""
    alpha, beta, gamma, delta = params.alpha, params.beta, params.gamma, params.delta

    def rhs(y: np.ndarray) -> np.ndarray:
        return np.array([y[0] * (alpha - beta * y[1]), y[1] * (delta * y[0] - gamma)])

    return rhs


def fixed_point(params: LVParams) -> np.ndarray:
    """The interior fixed point ``(x*, y*) = (gamma/delta, alpha/beta)``.

    A **center**, not an attractor — see :func:`center_eigenvalues`. Nothing
    converges to it, which is why this model's claims are about the orbit.
    """
    return np.array([params.gamma / params.delta, params.alpha / params.beta])


def center_eigenvalues(params: LVParams) -> np.ndarray:
    """Jacobian eigenvalues at the fixed point: ``+- i sqrt(alpha gamma)``.

    Computed from the Jacobian rather than returned as the closed form, so the
    test can compare the two and catch drift between them.
    """
    x_star, y_star = fixed_point(params)
    jacobian = np.array(
        [
            [params.alpha - params.beta * y_star, -params.beta * x_star],
            [params.delta * y_star, params.delta * x_star - params.gamma],
        ]
    )
    return np.linalg.eigvals(jacobian)


def small_oscillation_period(params: LVParams) -> float:
    """``2 pi / sqrt(alpha gamma)`` — the period in the **vanishing-amplitude limit**.

    Not the period of any particular orbit: the true period grows with amplitude
    (see the module docstring). It is checked by extrapolating measured periods to
    zero amplitude, never by a tolerance at one amplitude.
    """
    return 2.0 * math.pi / math.sqrt(params.alpha * params.gamma)


def conserved_v(y: np.ndarray, params: LVParams) -> float:
    """``V = delta x - gamma ln x + beta y - alpha ln y``, constant along orbits.

    ``nan`` outside the positive quadrant, where the logarithms are undefined.
    A closed orbit never leaves it, so a ``nan`` here means the trajectory has
    escaped and the invariant no longer describes it — which is a report worth
    making rather than an exception worth raising.
    """
    x, predator = float(y[0]), float(y[1])
    if x <= 0.0 or predator <= 0.0:
        return float("nan")
    return (
        params.delta * x
        - params.gamma * math.log(x)
        + params.beta * predator
        - params.alpha * math.log(predator)
    )


def initial_point(params: LVParams) -> np.ndarray:
    """``(x* + amp, y*)`` — the orbit's starting point."""
    y = fixed_point(params)
    return np.array([y[0] + params.amp, y[1]])


class LotkaVolterra:
    """Stateless classic Lotka-Volterra model. Register one shared instance."""

    def initial_state(self, params: LVParams, rng: Generator) -> LVState:
        # Deterministic: rng is unused.
        return LVState(
            y=initial_point(params),
            step_index=0,
            t=0.0,
            params=params,
            rhs=lv_rhs(params),
        )

    def step(self, state: LVState, rng: Generator) -> LVState:
        params = state.params
        index = state.step_index + 1
        return LVState(
            y=rk4_step(state.rhs, state.y, params.dt),
            step_index=index,
            t=index * params.dt,  # counted, not accumulated
            params=params,
            rhs=state.rhs,
        )

    def observables(self, state: LVState) -> dict[str, float]:
        return {
            "x": float(state.y[0]),
            "y": float(state.y[1]),
            "V": conserved_v(state.y, state.params),
        }

    def is_terminal(self, state: LVState) -> bool:
        return state.step_index >= n_lv_steps(state.params)

    def analytic_predictions(self, params: LVParams) -> dict[str, float]:
        """``V`` at ``t_max`` equals ``V`` at ``t = 0`` — the invariant, in closed form.

        The one prediction this model makes through the protocol, and the only one
        it *can*: the period and the time-average are properties of a cycle rather
        than of the final state (module docstring). Deliberately evaluated far from
        the fixed point, which is what distinguishes it from every other
        deterministic check in the project.
        """
        return {"V": conserved_v(initial_point(params), params)}


# The single shared, stateless instance used throughout the sandbox.
MODEL = LotkaVolterra()
register("lotka_volterra", MODEL)
