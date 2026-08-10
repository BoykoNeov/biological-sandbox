"""Daisyworld — Watson & Lovelock 1983, and the phase's one closed-form surprise.

Two daisy species compete for bare ground on a planet whose albedo they set:

    da_w/dt = a_w (x beta(T_w) - gamma)
    da_b/dt = a_b (x beta(T_b) - gamma)

with bare fraction ``x = 1 - a_w - a_b``, planetary albedo
``A = a_w A_w + a_b A_b + x A_g``, a global energy balance
``T_e^4 = S L (1 - A) / sigma``, and a *local* temperature for each species,

    T_i^4 = q (A - A_i) + T_e^4

so a white patch (high ``A_i``) sits colder than the planet and a black one
hotter. Growth ``beta(T) = max(0, 1 - k (T_opt - T)^2)`` is a downward parabola
about ``T_opt``, clipped at zero.

**The planning slice set out to root-find ``T*(L)`` and did not have to.** At an
interior equilibrium both species satisfy ``x beta(T_i) = gamma``, so
``beta(T_w) = beta(T_b)``; ``beta`` is symmetric about ``T_opt``, so ``T_w`` and
``T_b`` are its two roots ``T_opt -+ delta``. Substituting into the local
temperature law, the ``T_e^4`` and ``A`` terms cancel and leave

    T_b^4 - T_w^4 = q (A_w - A_b)

— **one equation in ``delta`` alone**, a depressed cubic
``delta^3 + T_opt^2 delta - q(A_w - A_b)/(8 T_opt) = 0`` solved by Cardano in
:func:`delta_offset`. Everything downstream follows:

======================  ==========================  ================
quantity                value (W&L constants)       depends on ``L``?
======================  ==========================  ================
``delta``               4.988282516540778 K         **no**
``T_w*``                290.511717483459222 K       **no**
``T_b*``                300.488282516540778 K       **no**
``beta(T_w*)``          0.918757127552342           **no**
bare fraction ``x*``    0.326528079079211           **no**
======================  ==========================  ================

Luminosity only sets how the *fixed* daisy cover ``1 - x*`` splits between white
and black, through a **linear** energy balance (:func:`interior_equilibrium`).
That is Daisyworld's homeostasis in exact form rather than as "the curve looks
flat", and it is why this model's headline is category A and not a fit.

**Two things the closed form buys, and how each is checked.**

* ``dT_w/dL = dT_b/dL = 0`` *exactly*. Asserting that the closed form is
  ``L``-free would be a tautology about this source file — the expression
  contains no ``L``. What the tests assert instead is (a) that the **local
  temperature law evaluated at the ``L``-dependent equilibrium cover** returns
  the same ``T_w`` at every ``L`` (a real check of :func:`interior_equilibrium`'s
  albedo solve, since ``a_w*`` and ``a_b*`` both move), and (b) that **simulated**
  endpoints from one common far start agree across ``L`` to within the residual
  transient. Only the second can fail for a model whose algebra is ``L``-free but
  whose dynamics drift.
* The planetary temperature does not merely flatten, it **overcompensates**:
  ``dT_e/dL`` is *negative* across the whole band while a bare planet's is
  strongly positive. The tests assert the **sign**; the magnitudes
  (``-13.57 / -10.77 / -8.75`` at ``L = 0.9 / 1.0 / 1.1``) are reported in the
  demo and never bounded, because a recorded magnitude travels only with the
  estimator that produced it.

**Where it refuses to answer**, in the stance ``hodgkin_huxley`` takes past the
Hopf and ``glv`` takes on an infeasible ``x*``:

* outside the **regulating band** ``L in [0.738722418247, 1.359472371265]``,
  where ``a_w*`` or ``a_b*`` goes non-positive and the attractor is a boundary
  state (one species only, or a dead planet) rather than the interior one;
* with no albedo contrast (``A_w <= A_b``), where ``delta`` is zero or negative
  and the white/black labelling inverts;
* when ``beta(T_w*) <= gamma``, so the implied bare fraction leaves no room for
  daisies.

**The interior state is stable but not globally attracting, and where it is not
is also closed-form.** A dead planet can be started only by a species that can
invade it, ``beta(T_i) > gamma`` at zero cover, and that gives a second, quite
different luminosity range — :func:`invasion_luminosities`. With the W&L
constants white invades for ``L in [0.8332, 1.2079]`` and black for
``[0.7058, 1.0805]``, against a regulating band of ``[0.7387, 1.3595]``. The two
ranges do not coincide, and the gap is the interesting part: for
``L in (1.2079, 1.3595)`` the interior state exists *and* the dead planet is
stable, so the planet's fate depends on its history. At the cold end there is no
such window — invadability reaches below the band — so the bistability is
one-sided.

Reachability from a particular start is a basin question, not a prediction
question, so ``analytic_predictions`` does **not** refuse on it. But tests that
integrate from a bare start must stay inside the invadable range: measured, from
``(0.01, 0.01)`` the planet reaches the interior equilibrium for
``L in [0.8, 1.2]`` and goes extinct for ``L >= 1.25``.

**The RHS is only C^0 and it is not academic.** ``beta``'s clip at zero puts a
kink in the vector field, and RK4's order-4 claim needs smoothness. Measured at
``L = 1.0`` from ``(0.01, 0.01)`` the clip never bites (``beta`` stays above
``0.733``) and clipped and smooth integrations are **bit-identical**, so the order
test is honest there — ratios ``15.26 / 15.56 / 15.76``. At ``L = 0.8``, where the
same trajectory does drive ``beta`` negative, the same measurement reads
``58.68 / 0.49 / 2.84`` with non-monotone errors. **Which luminosity the order
claim is made at is part of the claim**, the Phase-2 Laplacian lesson with a new
mechanism.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import rk4_step
from sandbox.core.registry import register

INITIAL_CONDITIONS = ("state", "equilibrium")

#: Keys :meth:`Daisyworld.analytic_predictions` returns, in reporting order.
PREDICTED_KEYS = ("a_w", "a_b", "x_bare", "albedo", "T_e", "T_w", "T_b")


@dataclass(frozen=True)
class DaisyworldParams:
    """Watson & Lovelock 1983 constants, with ``luminosity`` the control knob.

    The constants are params rather than module globals so a test can mutate one
    and watch the recorded literals move — that is the only thing that catches a
    value wrong in *both* the closed form and the RHS, which stays a genuine
    fixed point and passes every self-consistency check (see
    ``tests/test_daisyworld.py``).

    ``initial`` selects the claim: ``"state"`` integrates from ``a_init``,
    ``"equilibrium"`` starts at the closed-form interior state itself.
    """

    luminosity: float = 1.0
    solar_flux: float = 917.0  # S, W / m^2
    stefan_boltzmann: float = 5.67032e-8  # sigma
    q: float = 2.06e9  # K^4, local heat-transfer coefficient
    gamma: float = 0.3  # per-capita death rate
    albedo_white: float = 0.75
    albedo_black: float = 0.25
    albedo_ground: float = 0.5
    t_opt: float = 295.5  # K, peak of the growth parabola
    beta_k: float = 0.003265  # K^-2, its curvature
    initial: str = "state"
    a_init: tuple[float, float] = (0.01, 0.01)
    t_max: float = 100.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "a_init", tuple(float(v) for v in self.a_init))

        if len(self.a_init) != 2:
            raise ValueError(f"a_init must be (a_w, a_b), got {self.a_init}")
        if self.initial not in INITIAL_CONDITIONS:
            raise ValueError(f"initial must be one of {INITIAL_CONDITIONS}, got {self.initial!r}")
        if self.albedo_white <= self.albedo_black:
            raise ValueError(
                f"albedo_white ({self.albedo_white}) must exceed albedo_black "
                f"({self.albedo_black}): with no contrast delta is zero and the two "
                "species are indistinguishable, and with the contrast reversed the "
                "white/black labelling inverts rather than the model becoming wrong"
            )
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.beta_k <= 0.0:
            raise ValueError(f"beta_k must be positive, got {self.beta_k}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        steps = round(self.t_max / self.dt)
        if steps < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        # The prediction is read at exactly t_max and time is counted as
        # step_index * dt rather than accumulated, so dt must divide t_max.
        if abs(steps * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{steps} steps land at {steps * self.dt}"
            )


@dataclass(frozen=True)
class DaisyworldState:
    """``y = (a_w, a_b)``, the step counter, and the embedded params + RHS."""

    y: np.ndarray
    step_index: int
    t: float
    params: DaisyworldParams
    rhs: Callable[[np.ndarray], np.ndarray]


def n_daisy_steps(params: DaisyworldParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


def growth_rate(temperature: float | np.ndarray, params: DaisyworldParams) -> np.ndarray:
    """``beta(T) = max(0, 1 - k (T_opt - T)^2)`` — the clipped growth parabola.

    The clip is what makes the vector field only C^0; see the module docstring
    for where that costs the order-4 claim and where it is measured not to.
    """
    return np.maximum(0.0, 1.0 - params.beta_k * (params.t_opt - temperature) ** 2)


def growth_rate_unclipped(temperature: float | np.ndarray, params: DaisyworldParams) -> np.ndarray:
    """The same parabola **without** the clip — the smooth comparison branch.

    Exists only so a test can integrate both and show they are bit-identical on a
    trajectory where the clip never bites. Never used by the model itself.
    """
    return 1.0 - params.beta_k * (params.t_opt - temperature) ** 2


def temperatures(
    a_w: float, a_b: float, params: DaisyworldParams
) -> tuple[float, float, float, float]:
    """``(albedo, T_e, T_w, T_b)`` for a given daisy cover.

    ``T_e`` is the planetary black-body temperature from the global energy
    balance; ``T_w`` and ``T_b`` are the *local* temperatures the two species
    actually experience, offset from ``T_e`` by the heat-transfer term
    ``q (A - A_i)``. The offsets are what couple life to climate: raising
    ``a_w`` raises ``A``, which cools the planet *and* warms each patch relative
    to it.
    """
    bare = 1.0 - a_w - a_b
    albedo = a_w * params.albedo_white + a_b * params.albedo_black + bare * params.albedo_ground
    t_e4 = params.solar_flux * params.luminosity * (1.0 - albedo) / params.stefan_boltzmann
    t_w = (params.q * (albedo - params.albedo_white) + t_e4) ** 0.25
    t_b = (params.q * (albedo - params.albedo_black) + t_e4) ** 0.25
    return albedo, t_e4**0.25, t_w, t_b


def daisyworld_rhs(params: DaisyworldParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous RHS ``f((a_w, a_b)) -> d/dt``.

    The single definition of the vector field in the project. Module-level for
    the same reason ``glv_rhs`` and ``hh_rhs`` are: anything that needs the
    dynamics (the demo's luminosity ramps, a Jacobian, a hand-check) uses this
    one and cannot drift from what ``step`` integrates.
    """

    def rhs(y: np.ndarray) -> np.ndarray:
        a_w, a_b = y[0], y[1]
        bare = 1.0 - a_w - a_b
        _, _, t_w, t_b = temperatures(a_w, a_b, params)
        return np.array(
            [
                a_w * (bare * growth_rate(t_w, params) - params.gamma),
                a_b * (bare * growth_rate(t_b, params) - params.gamma),
            ]
        )

    return rhs


def delta_offset(params: DaisyworldParams) -> float:
    """``delta``, the half-separation of the two daisy temperatures about ``T_opt``.

    Solves ``(T_opt + d)^4 - (T_opt - d)^4 = q (A_w - A_b)``. Expanding cancels
    the even powers and leaves the depressed cubic

        d^3 + T_opt^2 d - q (A_w - A_b) / (8 T_opt) = 0

    whose discriminant is positive here (``p^3/27 ~ 2.5e13`` dwarfs
    ``q_c^2/4 ~ 4.7e10``), so Cardano's single real root applies. **One of the two
    cube-root arguments is negative** (``+5.19e6`` and ``-4.75e6``), which is why
    this uses :func:`numpy.cbrt` rather than ``x ** (1/3)`` — the latter returns
    ``nan``. The two roots are ``+173.12`` and ``-168.13`` and cancel down to
    ``4.988``, so about 1.5 digits are lost: the value agrees with a 200-iteration
    bisection to ``5.7e-15`` relative, not to the last bit.
    """
    contrast = params.q * (params.albedo_white - params.albedo_black)
    p = params.t_opt**2
    q_c = -contrast / (8.0 * params.t_opt)
    root = np.sqrt(q_c * q_c / 4.0 + p**3 / 27.0)
    return float(np.cbrt(-q_c / 2.0 + root) + np.cbrt(-q_c / 2.0 - root))


def interior_temperatures(params: DaisyworldParams) -> tuple[float, float]:
    """``(T_w*, T_b*) = T_opt -+ delta`` — the pinned daisy temperatures.

    Independent of luminosity, which is the whole homeostasis claim. See the
    module docstring for why the *tests* do not assert that independence here.
    """
    delta = delta_offset(params)
    return params.t_opt - delta, params.t_opt + delta


def bare_fraction(params: DaisyworldParams) -> float:
    """``x* = gamma / beta(T_w*)`` — the bare-ground fraction at equilibrium.

    Also independent of luminosity. Raises when ``beta(T_w*) <= gamma``, where the
    implied bare fraction is at least the whole planet and no interior state with
    living daisies exists.
    """
    t_w, _ = interior_temperatures(params)
    beta_star = float(growth_rate(t_w, params))
    if beta_star <= params.gamma:
        raise ValueError(
            f"beta(T_w*) = {beta_star:.6g} does not exceed gamma = {params.gamma:.6g}, so "
            f"the implied bare fraction is {'infinite' if beta_star <= 0 else 'at least 1'} "
            "and no interior state with living daisies exists"
        )
    return params.gamma / beta_star


def equilibrium_albedo(params: DaisyworldParams) -> float:
    """``A*`` from the linear energy balance at the pinned ``T_w*``.

    ``T_w*^4 = q (A - A_w) + S L (1 - A) / sigma`` is linear in ``A`` once ``T_w*``
    is known, which is the reason no root-find is needed anywhere in this model.
    """
    t_w, _ = interior_temperatures(params)
    k = params.solar_flux * params.luminosity / params.stefan_boltzmann
    denominator = k - params.q
    if denominator == 0.0:
        raise ValueError(
            "the energy balance is degenerate: S L / sigma equals q, so the albedo "
            "drops out of the local temperature law and A* is not determined"
        )
    return (k - t_w**4 - params.q * params.albedo_white) / denominator


def _luminosity_at_albedo(albedo: float, params: DaisyworldParams) -> float:
    """Invert :func:`equilibrium_albedo`: the ``L`` at which ``A* = albedo``."""
    t_w, _ = interior_temperatures(params)
    k = (t_w**4 + params.q * (params.albedo_white - albedo)) / (1.0 - albedo)
    return k * params.stefan_boltzmann / params.solar_flux


def regulating_band(params: DaisyworldParams) -> tuple[float, float]:
    """``(L_lo, L_hi)`` — the luminosities between which the interior state exists.

    ``A*(L)`` rises monotonically with ``L`` while the *total* cover ``1 - x*``
    stays fixed, so the interior state runs out of room exactly when the albedo it
    requires falls outside what the fixed cover can produce: all black at
    ``A_lo = (1 - x*) A_b + x* A_g`` and all white at
    ``A_hi = (1 - x*) A_w + x* A_g``. Inverting the (linear) albedo balance at
    those two values gives the endpoints in closed form — the planning slice
    bisected for them and got the same numbers, ``0.738722418247`` and
    ``1.359472371265``.
    """
    bare = bare_fraction(params)
    cover = 1.0 - bare
    albedo_lo = cover * params.albedo_black + bare * params.albedo_ground
    albedo_hi = cover * params.albedo_white + bare * params.albedo_ground
    return _luminosity_at_albedo(albedo_lo, params), _luminosity_at_albedo(albedo_hi, params)


def invasion_luminosities(params: DaisyworldParams) -> dict[str, tuple[float, float]]:
    """Per species, the ``L`` range in which it can invade a **bare** planet.

    A rare species grows when ``x beta(T_i) > gamma``; at zero cover ``x = 1``, so
    the condition is ``beta(T_i) > gamma``, i.e. ``T_i`` inside
    ``T_opt +- sqrt((1 - gamma) / k)``. On a bare planet the albedo is ``A_g``
    whatever the luminosity, so ``T_i^4 = q (A_g - A_i) + S L (1 - A_g) / sigma``
    is *linear* in ``L`` and inverts directly.

    This is a different question from :func:`regulating_band`, and the difference
    is the point. The band says where the interior state **exists**; this says
    where a dead planet can be **started**. Measured with the W&L constants:
    white invades for ``L in [0.8332, 1.2079]`` and black for
    ``[0.7058, 1.0805]``, against a band of ``[0.7387, 1.3595]``. So between
    ``1.2079`` and the band's top edge both the interior state and the dead planet
    are attractors — genuine bistability, width ``0.1516`` — while at the cold end
    invadability reaches *below* the band, and there is **no** bistable window
    (the arithmetic gives a negative width, ``-0.0329``).

    Raises when ``gamma >= 1``: growth then never exceeds death at any temperature
    and no species can invade anything, so there is no window to report.
    """
    if params.gamma >= 1.0:
        raise ValueError(
            f"gamma = {params.gamma:.6g} is at least the peak growth rate of 1, so no "
            "species can invade a bare planet at any luminosity and the window is empty"
        )
    half_width = math.sqrt((1.0 - params.gamma) / params.beta_k)
    scale = params.solar_flux * (1.0 - params.albedo_ground)
    out: dict[str, tuple[float, float]] = {}
    for name, albedo_i in (
        ("white", params.albedo_white),
        ("black", params.albedo_black),
    ):
        offset = params.q * (params.albedo_ground - albedo_i)
        edges = tuple(
            ((params.t_opt + sign * half_width) ** 4 - offset) * params.stefan_boltzmann / scale
            for sign in (-1.0, 1.0)
        )
        out[name] = (float(edges[0]), float(edges[1]))
    return out


def interior_equilibrium(params: DaisyworldParams) -> np.ndarray:
    """``(a_w*, a_b*)`` — the interior equilibrium cover.

    The daisy temperatures and the total cover are pinned; ``L`` only sets the
    split, which follows from ``A* = a_w A_w + a_b A_b + x* A_g`` with
    ``a_w + a_b = 1 - x*``.

    Raises outside the regulating band, where one of the two components is
    non-positive. Predicting it anyway would be the Gray-Scott error of Phase 2 —
    validating against a state the system does not have. Note this refusal is
    about *existence*, not reachability: inside the band the interior state is
    stable but not globally attracting (module docstring).
    """
    bare = bare_fraction(params)
    albedo = equilibrium_albedo(params)
    a_w = (albedo - (1.0 - bare) * params.albedo_black - bare * params.albedo_ground) / (
        params.albedo_white - params.albedo_black
    )
    a_b = (1.0 - bare) - a_w
    if a_w <= 0.0 or a_b <= 0.0:
        low, high = regulating_band(params)
        raise ValueError(
            f"no interior equilibrium at L = {params.luminosity:.6g}: the closed form gives "
            f"(a_w, a_b) = ({a_w:.6g}, {a_b:.6g}), which is not strictly interior. The "
            f"regulating band is L in [{low:.10f}, {high:.10f}]; outside it the attractor is "
            "a boundary state -- a single-species planet or a dead one -- so predicting the "
            "interior point would be a wrong number that still looks green"
        )
    return np.array([a_w, a_b])


def jacobian(params: DaisyworldParams, y: np.ndarray | None = None) -> np.ndarray:
    """Central-difference Jacobian of the RHS, at ``y`` (default: the equilibrium).

    Finite-differenced rather than hand-derived: unlike gLV's ``diag(x*) A`` there
    is no clean closed form here (``beta`` composes with a quartic root of a linear
    albedo), and the only consumer is a relaxation *time scale* used to size a
    tolerance, not a validated prediction. A wrong Jacobian would loosen or tighten
    that bound, and the test that uses it checks the bound against the measured
    residual to a ratio near 1, which is what would catch it.
    """
    if y is None:
        y = interior_equilibrium(params)
    rhs = daisyworld_rhs(params)
    step = 1e-7
    out = np.empty((2, 2))
    for j in range(2):
        offset = np.zeros(2)
        offset[j] = step
        out[:, j] = (rhs(y + offset) - rhs(y - offset)) / (2.0 * step)
    return out


def slowest_rate(params: DaisyworldParams) -> float:
    """``|Re lambda|`` of the slowest Jacobian mode — the relaxation rate at ``y*``.

    Raises if the equilibrium is not stable. Measured across the band the pair is
    always real and negative, from ``{-0.0986, -0.7031}`` at ``L = 0.8`` to
    ``{-0.9213, -0.0420}`` at ``L = 1.3``; the slow rate goes to zero at both band
    edges, where one species is on its way out (``tau = 151`` at ``L = 1.35``).
    """
    eigenvalues = np.linalg.eigvals(jacobian(params))
    if np.any(eigenvalues.real >= 0.0):
        raise ValueError(
            f"the interior equilibrium is unstable (max Re eig = "
            f"{eigenvalues.real.max():+.4g}); it is not what a long run converges to"
        )
    return float(np.abs(eigenvalues.real).min())


class Daisyworld:
    """Stateless Daisyworld model. Register one shared instance."""

    def initial_state(self, params: DaisyworldParams, rng: Generator) -> DaisyworldState:
        # Deterministic: rng is unused.
        if params.initial == "equilibrium":
            y = interior_equilibrium(params)
        else:
            y = np.asarray(params.a_init, dtype=float)
        return DaisyworldState(
            y=y,
            step_index=0,
            t=0.0,
            params=params,
            rhs=daisyworld_rhs(params),
        )

    def step(self, state: DaisyworldState, rng: Generator) -> DaisyworldState:
        params = state.params
        index = state.step_index + 1
        return DaisyworldState(
            y=rk4_step(state.rhs, state.y, params.dt),
            step_index=index,
            t=index * params.dt,  # counted, not accumulated
            params=params,
            rhs=state.rhs,
        )

    def observables(self, state: DaisyworldState) -> dict[str, float]:
        """Cover, albedo, and the three temperatures.

        ``x_bare`` rather than ``x``: ``x`` means an abundance everywhere else in
        this codebase, and here it is a *fraction of the surface*.
        """
        a_w, a_b = float(state.y[0]), float(state.y[1])
        albedo, t_e, t_w, t_b = temperatures(a_w, a_b, state.params)
        return {
            "a_w": a_w,
            "a_b": a_b,
            "x_bare": 1.0 - a_w - a_b,
            "albedo": float(albedo),
            "T_e": float(t_e),
            "T_w": float(t_w),
            "T_b": float(t_b),
        }

    def is_terminal(self, state: DaisyworldState) -> bool:
        return state.step_index >= n_daisy_steps(state.params)

    def analytic_predictions(self, params: DaisyworldParams) -> dict[str, float]:
        """The full interior state — cover, albedo, and all three temperatures.

        Raises outside the regulating band and whenever the interior state fails
        to exist (see :func:`interior_equilibrium`, :func:`bare_fraction`).

        ``dT_w/dL = 0`` is deliberately **not** a key here: ``validate()`` matches
        each predicted key to an observable's final value, and a derivative with
        respect to a *parameter* is not an observable of any single run. It is
        checked by the two luminosity-invariance tests instead.

        **Every key is taken from the closed form's own route, never from the
        cover.** ``T_w`` and ``T_b`` come from Cardano (``T_opt -+ delta``), not
        from ``temperatures(y*)``; ``albedo`` comes from the linear energy balance,
        not from summing the equilibrium cover. The *observables* compute all four
        the other way — from ``(a_w, a_b)`` through the albedo sum and the local
        temperature law.

        That is the right way round, but be precise about what it buys, because the
        mutation run measured it: **switching this method to the cover's own
        temperatures is an invisible mutant.** ``validate()`` cannot see the
        difference, since the two routes are already pinned to each other at every
        luminosity by
        ``test_the_temperature_law_returns_the_pinned_temperatures_at_every_luminosity``
        — which is where the cross-route agreement is actually checked. Reading
        from Cardano here keeps the prediction independent of the cover solve as a
        matter of design; it is not what catches a discrepancy between them.
        """
        y_star = interior_equilibrium(params)
        albedo = equilibrium_albedo(params)
        t_e4 = params.solar_flux * params.luminosity * (1.0 - albedo) / params.stefan_boltzmann
        t_w, t_b = interior_temperatures(params)
        return {
            "a_w": float(y_star[0]),
            "a_b": float(y_star[1]),
            "x_bare": bare_fraction(params),
            "albedo": float(albedo),
            "T_e": float(t_e4**0.25),
            "T_w": float(t_w),
            "T_b": float(t_b),
        }


# The single shared, stateless instance used throughout the sandbox.
MODEL = Daisyworld()
register("daisyworld", MODEL)
