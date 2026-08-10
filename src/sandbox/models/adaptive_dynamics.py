"""Adaptive dynamics — the trait-substitution sequence and its canonical limit.

A single resident trait ``x`` sits at its carrying capacity in the Gaussian
competition model of Dieckmann & Doebeli (1999):

    K(x) = K0 exp(-x^2 / 2 sK^2)        resource availability at trait ``x``
    a(y, x) = exp(-(y - x)^2 / 2 sa^2)  how much a ``y``-individual feels an ``x``
    s_x(y) = r (1 - a(y, x) K(x) / K(y))   invasion fitness of a rare mutant ``y``

Mutations arrive as a Poisson process of rate ``mu K(x)`` (rate per birth times
the resident's abundance), the mutant trait is ``y = x + delta`` with
``delta ~ N(0, sm^2)``, and it fixes with probability ``max(0, s_x(y)) / r`` —
the branching-process survival probability of a rare favourable type. On
fixation the resident is replaced outright. That jump process is the
**trait-substitution sequence**, and it is what :class:`AdaptiveDynamics`
simulates.

**Its deterministic limit is the canonical equation.** Taking ``sm -> 0``,

    E[dx/dt] = mu K(x) E[delta max(0, s_x(x + delta))] / r

and linearizing ``s_x(x + delta) ~ D(x) delta`` with
``E[delta^2 1{D delta > 0}] = sm^2 / 2`` gives

    dx/dt = (1/2) mu sm^2 K(x) D(x),      D(x) = -r x / sK^2

exactly (:func:`canonical_rhs`). This is the phase's second stochastic-to-
deterministic thread after ``glv_stochastic``, and the limit parameter is the
**mutation step ``sm``** rather than a system size.

**There are deliberately no ``analytic_predictions``.** The canonical value is a
*limit*, not an identity, and the planning slice measured what happens if you
treat it as one: at ``sm = 0.0125`` five independent replicate groups all landed
*above* the prediction, a systematic ``O(sm)`` offset. A tolerance derived from
the replicate SE therefore tightens onto a real bias, and the check would start
failing a **correct** implementation as replicates grow — ``validate()``'s
tightening tolerance arriving from the wrong direction. The honest claim is
category B: the discrepancy vanishes **linearly in ``sm``**, asserted as a
two-sided band on the log-log slope in ``tests/test_adaptive_dynamics.py``.
``repressilator`` takes the same stance for the same reason.

**Why the band must be two-sided, measured rather than assumed.** A *wrong*
canonical equation makes the discrepancy an ``O(1)`` constant, and a constant
fits a near-perfect line — the three teeth read slopes of ``+0.018``, ``+0.005``
and ``-0.003`` with standard errors 30-200x *smaller* than the correct point's,
so every one of them is tens of sigma from zero. **"Significantly nonzero"
passes all three.** This is the Phase-3b trap verbatim (the wrong-``R``
random-matrix draw that cleared zero at 6 sigma) and the reason the shipped
assertion is ``0.6 <= slope <= 1.4``.

**Two implementations of one process, and they are pinned bit-for-bit.**
:meth:`AdaptiveDynamics.step` advances one replicate by one mutation event, which
is what the protocol wants and what the demo plots. The ``O(sm)`` sweep needs
~40 000 replicates at the smallest ``sm``, so it runs through
:func:`run_cohort`, which advances the whole *still-living* cohort by one event
per iteration in NumPy. The two draw from the generator in the same order
(waiting time, then step, then fixation coin — **including the waiting time of
the event that overshoots the horizon**), so at ``n_rep = 1`` they produce
bit-identical trait sequences, and a test asserts exactly that.

**Convention, recovered rather than inherited.** No slice code survived, so the
recorded canonical target ``1.849492`` had to be reproduced from the plan's
tables alone. Holding ``mu sm^2 t_max`` fixed pins one scalar,
``U = (1/2) mu sm^2 K0 t_max``, because the canonical equation collapses under
``u = (1/2) mu sm^2 K0 t`` to the parameter-free ``dx/du = -x exp(-x^2/2)``.
``U = 1/2`` reproduces the recorded value on all six digits, and three further
recorded numbers the reconstruction was *not* fitted to (the teeth targets)
agree to ``3e-7``. ``K0`` and ``mu`` are **not separately identifiable and do not
need to be**: ``K0`` cancels in the invasion fitness (a ratio of ``K``\\ s) and
the event count is ``mu K0 t_max = 2U / sm^2``.

**What the record does *not* pin**, and this is the load-bearing half: ``sa`` and
the mutation-step distribution are invisible to every recorded number, yet both
enter the ``O(sm)`` coefficient. ``sigma_a = 0.7`` and a Gaussian step are
therefore *stated choices*. **Consequence: the slice's ``+0.004`` offset and its
teeth ``z`` values may not be cited against this convention** — the offset
measured here is ``+2.39e-3``. A recorded number travels only with the estimator
that produced it, and here the estimator was only *partly* recoverable; separate
what a record pins from what it merely used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import integrate_rk4
from sandbox.core.registry import register

#: Scaled canonical distance of one run, ``U = (1/2) mu sm^2 K0 t_max``. The
#: design constant recovered in ``docs/plans/phase3-tasks.md``; every ``sm``
#: travels the same canonical distance, so the prediction does not move.
U_TARGET = 0.5

#: ``x(U_TARGET)`` from ``x0 = 2`` under the canonical equation with
#: ``r = sK = 1`` — the recorded target, reproduced to all six recorded digits.
CANONICAL_TRAIT = 1.849492240597

#: Step for the canonical integration, **chosen by measuring where the answer
#: stops moving** rather than typed. ``x(0.5)`` is identical to twelve digits at
#: ``1e-2`` through ``1e-6``; against ``du = 1e-4``, ``du = 1e-3`` moves ``x(U)``
#: by ``3e-15`` and the most demanding tooth target by ``2e-13`` relative.
#:
#: The choice matters because **the expensive part of a stochastic test can be
#: its deterministic helper**. The ``sm``-for-``sm^2`` tooth integrates to
#: ``u = U / sm``, so the five sweep points together travel ``u = 78`` — eight
#: million pure-Python RK4 steps at ``1e-5``, 27.7 s at ``1e-4``, and **1.7 s
#: here**, against 2.3 s for the whole stochastic sweep it exists to check. So
#: the cheap step is the default rather than a caller's opt-in: a helper that has
#: to be *remembered* to be cheap will drift back.
CANONICAL_DU = 1e-3


@dataclass(frozen=True)
class AdaptiveDynamicsParams:
    """Gaussian competition model plus the mutation process.

    ``sigma_m`` is the limit parameter: the canonical equation is what the
    process becomes as it goes to zero. ``t_max`` is *derived* from
    :data:`U_TARGET` rather than given, so every ``sigma_m`` travels the same
    canonical distance and the prediction is one number for the whole sweep.

    ``sigma_a`` and the Gaussian mutation step are **choices**, not values
    recovered from the record — see the module docstring for why that matters
    when citing the slice's numbers.
    """

    sigma_m: float = 0.05
    sigma_a: float = 0.7
    sigma_k: float = 1.0
    r_growth: float = 1.0
    k0: float = 1.0
    mutation_rate: float = 1.0  # mu, per birth
    x_init: float = 2.0
    u_target: float = U_TARGET

    def __post_init__(self) -> None:
        for name in ("sigma_m", "sigma_a", "sigma_k", "r_growth", "k0", "mutation_rate"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.u_target <= 0.0:
            raise ValueError(f"u_target must be positive, got {self.u_target}")

    @property
    def t_max(self) -> float:
        """``2 U / (mu K0 sm^2)`` — the horizon that travels ``u_target``."""
        return 2.0 * self.u_target / (self.mutation_rate * self.k0 * self.sigma_m**2)


# --------------------------------------------------------------------------
# Trait-space primitives. Everything else in the 3e arc is built from these.
# --------------------------------------------------------------------------


def carrying_capacity(x: float | np.ndarray, params: AdaptiveDynamicsParams) -> float | np.ndarray:
    """``K(x) = K0 exp(-x^2 / 2 sK^2)`` — the resource landscape."""
    return params.k0 * np.exp(-(np.asarray(x) ** 2) / (2.0 * params.sigma_k**2))


def competition(
    y: float | np.ndarray, x: float | np.ndarray, params: AdaptiveDynamicsParams
) -> float | np.ndarray:
    """``a(y, x) = exp(-(y - x)^2 / 2 sa^2)`` — symmetric, and Gaussian on purpose.

    A Gaussian kernel is positive definite, which is what makes the post-branching
    two-morph state structurally degenerate (Gyllenberg & Meszena). That is a
    literature-anchored, **category C** statement about the continuum model and is
    reported, never asserted — see ``models/trait_branching.py`` for what actually
    produces the two morphs on a grid.
    """
    return np.exp(-((np.asarray(y) - np.asarray(x)) ** 2) / (2.0 * params.sigma_a**2))


def invasion_fitness(
    y: float | np.ndarray, x: float | np.ndarray, params: AdaptiveDynamicsParams
) -> float | np.ndarray:
    """``s_x(y) = r (1 - a(y, x) K(x) / K(y))`` — growth rate of a rare ``y``.

    Written as one exponential rather than as a ratio of two ``K``\\ s so that
    ``K0`` visibly cancels: the exponent is
    ``-(y - x)^2 / 2 sa^2 + (y^2 - x^2) / 2 sK^2``.

    With ``r > 0`` this is bounded above by ``r``, so ``max(0, s) / r`` is already
    a probability and needs no clip at 1. :func:`run_cohort` reports how close it
    gets (``frac_saturated``), because the linearization the canonical equation
    rests on is worst exactly where it is large.
    """
    y_arr, x_arr = np.asarray(y), np.asarray(x)
    exponent = -((y_arr - x_arr) ** 2) / (2.0 * params.sigma_a**2) + (y_arr**2 - x_arr**2) / (
        2.0 * params.sigma_k**2
    )
    return params.r_growth * (1.0 - np.exp(exponent))


def selection_gradient(x: float | np.ndarray, params: AdaptiveDynamicsParams) -> float | np.ndarray:
    """``D(x) = ds_x(y)/dy`` at ``y = x``, in closed form: ``-r x / sK^2``.

    Zero only at ``x = 0``: the **singular point**, and the trait the canonical
    equation runs toward. Checked signed against a central difference of
    :func:`invasion_fitness` in the tests.
    """
    return -params.r_growth * np.asarray(x) / params.sigma_k**2


def gradient_slope(params: AdaptiveDynamicsParams) -> float:
    """``dD/dx = -r / sK^2`` — negative everywhere, so the singular point is
    always **convergence stable**: selection carries the resident toward it from
    either side, whatever ``sa`` does. This is the derivative the recorded
    ``-1.000000000000`` pinned ``r`` and ``sK`` with.
    """
    return -params.r_growth / params.sigma_k**2


def mutant_curvature(x: float | np.ndarray, params: AdaptiveDynamicsParams) -> float | np.ndarray:
    """``d2 s_x(y) / dy^2`` at ``y = x``: ``r (1/sa^2 - 1/sK^2 - x^2/sK^4)``.

    At the singular point this is :func:`splitting_rate`. Its **sign is the
    branching criterion**: positive means the resident sits in a fitness
    *minimum*, so both directions can invade and the population splits.
    """
    x_arr = np.asarray(x)
    return params.r_growth * (
        1.0 / params.sigma_a**2 - 1.0 / params.sigma_k**2 - x_arr**2 / params.sigma_k**4
    )


def resident_curvature(x: float | np.ndarray, params: AdaptiveDynamicsParams) -> float | np.ndarray:
    """``d2 s_x(y) / dx dy`` at ``y = x``: ``r (x^2/sK^4 - 1/sa^2)``.

    Carried because of the identity it closes with: the two second derivatives
    must sum to :func:`gradient_slope`, since ``D(x) = ds_x(y)/dy|_{y=x}`` is a
    derivative *along the diagonal*. That chain-rule identity is a check on both
    closed forms at once and is independent of the finite-difference comparison —
    a sign error in either one breaks it.
    """
    x_arr = np.asarray(x)
    return params.r_growth * (x_arr**2 / params.sigma_k**4 - 1.0 / params.sigma_a**2)


def splitting_rate(params: AdaptiveDynamicsParams) -> float:
    """``r (1/sa^2 - 1/sK^2)`` — the curvature at the singular point.

    **Positive iff ``sa < sK``**: competition narrower than the resource
    distribution. That sign change is the branch/no-branch criterion, and its
    magnitude is the rate the branching time diverges as the reciprocal of
    (``models/trait_branching.py``).
    """
    return float(mutant_curvature(0.0, params))


# --------------------------------------------------------------------------
# The deterministic limit
# --------------------------------------------------------------------------


def canonical_rhs(params: AdaptiveDynamicsParams) -> Callable[[np.ndarray], np.ndarray]:
    """``dx/du = K(x) D(x) / K0`` — the canonical equation in **scaled** time.

    Scaled time ``u = (1/2) mu sm^2 K0 t`` removes ``mu``, ``sm`` and ``K0``
    together, which is exactly why they are not separately identifiable from the
    record. With ``r = sK = 1`` this is the parameter-free
    ``dx/du = -x exp(-x^2/2)``.

    Module-level and single-definition, for the reason ``daisyworld_rhs`` is:
    the demo, the tests and :func:`canonical_trait` all integrate *this* function,
    so none of them can drift from the equation the ``O(sm)`` claim is about.
    """

    def rhs(y: np.ndarray) -> np.ndarray:
        return carrying_capacity(y, params) * selection_gradient(y, params) / params.k0

    return rhs


def canonical_trait(
    params: AdaptiveDynamicsParams, u: float | None = None, du: float = CANONICAL_DU
) -> float:
    """The canonical equation's trait after scaled time ``u`` (default
    ``params.u_target``), started from ``params.x_init``.

    This is the quantity the stochastic mean is claimed to approach as
    ``sm -> 0``. It does **not** depend on ``sm``: holding ``u`` fixed is what
    makes the sweep a limit measurement rather than five noisy estimates of five
    different numbers.
    """
    if u is None:
        u = params.u_target
    if u == 0.0:
        return float(params.x_init)
    _, ys = integrate_rk4(canonical_rhs(params), float(params.x_init), float(u), du)
    return float(ys[-1, 0])


# --------------------------------------------------------------------------
# The stochastic process
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptiveDynamicsState:
    """Resident trait, clock, event counters, and the embedded params."""

    x: float
    t: float
    n_events: int
    n_fixed: int
    params: AdaptiveDynamicsParams


@dataclass
class CohortResult:
    """What :func:`run_cohort` returns — the ensemble plus its own diagnostics."""

    x_final: np.ndarray  # (n_rep,) trait at the horizon, one per replicate
    n_events: int  # mutations offered, summed over replicates
    n_fixed: int  # of those, how many fixed
    max_fixation_probability: float
    frac_saturated: float  # fraction of offered mutants with s/r > 1/2
    n_iterations: int  # loop trips = events of the longest-lived replicate

    @property
    def mean(self) -> float:
        return float(self.x_final.mean())

    @property
    def sem(self) -> float:
        """Standard error of :attr:`mean` over replicates."""
        return float(self.x_final.std(ddof=1)) / np.sqrt(self.x_final.size)


def run_cohort(params: AdaptiveDynamicsParams, rng: Generator, n_rep: int) -> CohortResult:
    """``n_rep`` independent trait-substitution sequences, vectorized.

    Each iteration advances the whole *still-running* cohort by one mutation
    event; replicates whose clock passes ``t_max`` drop out, so the loop runs as
    many trips as the longest-lived replicate has events. That is what makes the
    ``O(sm)`` sweep cost 2.3 s instead of minutes — at the smallest ``sm`` it is
    40 000 replicates of 6400 events each.

    The draw order is **waiting time, then mutation step, then fixation coin**,
    and a replicate whose waiting time carries it past the horizon consumes the
    waiting time and nothing else. :meth:`AdaptiveDynamics.step` does the same,
    which is what makes the two bit-identical at ``n_rep = 1``.
    """
    if n_rep < 1:
        raise ValueError(f"n_rep must be at least 1, got {n_rep}")

    t_max = params.t_max
    x = np.full(n_rep, float(params.x_init))
    t = np.zeros(n_rep)
    live = np.arange(n_rep)

    n_events = n_fixed = n_iterations = n_saturated = 0
    max_p = 0.0

    while live.size:
        n_iterations += 1
        rate = params.mutation_rate * carrying_capacity(x[live], params)
        t[live] += rng.standard_exponential(live.size) / rate

        live = live[t[live] < t_max]
        if not live.size:
            break

        n_events += live.size
        delta = rng.standard_normal(live.size) * params.sigma_m
        # The clip is documentary, not operative, and mutation testing proved it:
        # deleting it is the one mutant of 33 that survives. `rng.random()` lives
        # in [0, 1), so `random() < p` is False for every p <= 0 whether p is zero
        # or negative -- same outcome, same draws, bit-identical trajectory. It
        # stays because `p_fix` is reported as a probability (`max_fixation_
        # probability`, `frac_saturated`) and a negative one would be a lie about
        # the quantity, not merely a harmless value.
        p_fix = np.maximum(
            0.0, invasion_fitness(x[live] + delta, x[live], params) / params.r_growth
        )

        max_p = max(max_p, float(p_fix.max()))
        n_saturated += int(np.count_nonzero(p_fix > 0.5))

        fixed = rng.random(live.size) < p_fix
        n_fixed += int(np.count_nonzero(fixed))
        x[live[fixed]] += delta[fixed]

    return CohortResult(
        x_final=x,
        n_events=n_events,
        n_fixed=n_fixed,
        max_fixation_probability=max_p,
        frac_saturated=n_saturated / max(n_events, 1),
        n_iterations=n_iterations,
    )


class AdaptiveDynamics:
    """Stateless trait-substitution sequence — one replicate, one event per step.

    No ``analytic_predictions``: the canonical value is a limit, not an identity
    (module docstring). The model is here because the process *is* a dynamical
    process with a clock, so it belongs on the protocol — the demo plots its
    trajectories, and it is the reference :func:`run_cohort` is pinned against.
    """

    def initial_state(
        self, params: AdaptiveDynamicsParams, rng: Generator
    ) -> AdaptiveDynamicsState:
        return AdaptiveDynamicsState(
            x=float(params.x_init), t=0.0, n_events=0, n_fixed=0, params=params
        )

    def step(self, state: AdaptiveDynamicsState, rng: Generator) -> AdaptiveDynamicsState:
        """One mutation event: wait, then offer a mutant, then flip for fixation.

        The waiting time is drawn **before** the horizon is tested, and a step
        that overshoots consumes only that draw. Testing terminality first would
        be the natural way to write this and would silently desynchronize the
        stream from :func:`run_cohort` on the final event of every replicate.
        """
        params = state.params
        rate = params.mutation_rate * float(carrying_capacity(state.x, params))
        t = state.t + float(rng.standard_exponential()) / rate

        if t >= params.t_max:
            return AdaptiveDynamicsState(
                x=state.x,
                t=t,
                n_events=state.n_events,
                n_fixed=state.n_fixed,
                params=params,
            )

        delta = float(rng.standard_normal()) * params.sigma_m
        p_fix = max(
            0.0,
            float(invasion_fitness(state.x + delta, state.x, params)) / params.r_growth,
        )
        fixed = float(rng.random()) < p_fix
        return AdaptiveDynamicsState(
            x=state.x + delta if fixed else state.x,
            t=t,
            n_events=state.n_events + 1,
            n_fixed=state.n_fixed + int(fixed),
            params=params,
        )

    def observables(self, state: AdaptiveDynamicsState) -> dict[str, float]:
        return {
            "x": float(state.x),
            "fitness_gradient": float(selection_gradient(state.x, state.params)),
            "n_events": float(state.n_events),
            "n_fixed": float(state.n_fixed),
        }

    def is_terminal(self, state: AdaptiveDynamicsState) -> bool:
        return state.t >= state.params.t_max


#: The single shared, stateless instance used throughout the sandbox.
MODEL = AdaptiveDynamics()
register("adaptive_dynamics", MODEL)
