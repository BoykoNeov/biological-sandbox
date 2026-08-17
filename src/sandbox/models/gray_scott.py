"""Gray-Scott reaction-diffusion — and validated linear stability analysis.

    du/dt = Du lap(u) - u v^2 + F (1 - u)
    dv/dt = Dv lap(v) + u v^2 - (F + k) v

on ``[0, L]^2`` with periodic boundaries. This is the project's first PDE, and its
checkable claim is not a large-system limit but **linear stability analysis**: the
growth rate ``lambda(q)`` of a small single-mode perturbation about the homogeneous
steady state.

**HANDOFF's promise had to be reframed, and the reason is measured.** HANDOFF §5
asks to "validate pattern wavelength against linear stability analysis". At
Pearson's classic ``(F, k)`` — the parameters behind every famous Gray-Scott
image — the non-trivial homogeneous steady state **does not exist as a real
solution** (``F < 4(F+k)^2``); the only homogeneous state is the trivial
``(1, 0)``. Those spots and labyrinths are excitable/subcritical structures, not
Turing patterns, and linear theory sets no wavelength for them. So the promise
splits in two: ``lambda(q)`` is validated here (sharper — exact, cheap, works at
any ``(F, k)``, and spans a **sign change**), and the emergent Pearson pattern is
kept as a labelled-qualitative demo with no wavelength assertion.

Genuine Turing points — real non-trivial state, stable *without* diffusion, and an
interior maximum ``lambda(q*) > 0`` at ``q* > 0`` — occupy a razor sliver of the
plane: ``F in [0.049, 0.117]``, ``k in [0.054, 0.062]``, with the ``k`` window at
fixed ``F`` only ~0.5% wide. The defaults sit at ``(0.074, 0.062)``.

**Those three conditions all hold here and are still not enough to give this model a
wavelength**, which is measured rather than argued
(``docs/plans/phase2c-gray-scott-selection-measurement.md``). The bifurcation is
**subcritical**: the saturated amplitude is already ``0.580`` at ``d/d_c = 1.02``,
where the growth rate is ``9.8e-04`` and Schnakenberg's amplitude is ``0.379`` and
falling; a pattern formed above onset **survives at** ``d/d_c = 0.95``, where the
homogeneous state is linearly stable and white noise decays to exactly zero; and the
emergent state spreads its power over twelve comparable modes (peak fraction
``0.05-0.14`` against Schnakenberg's ``0.53-1.00``). So the pattern here is a
large-amplitude localized structure rather than a perturbation of the homogeneous
state, and linear stability analysis sets no wavelength for it — at ``d/d_c = 1.001``
the fastest linear mode is 43 and what appears is 6 to 10.

This is Phase 2's finding one layer in. There, Pearson's parameters had no Turing
state at all; here the Turing state is genuine and the *pattern* is still not one
linear theory predicts. **A Turing instability is necessary but not sufficient —
supercriticality is the load-bearing precondition**, which is why wavelength selection
lives in ``schnakenberg`` (supercritical, verified) and why this model's ``initial``
offers no random-noise condition: it cannot be asked the selection question, so there
is no refusal here to look for.

**The prediction uses the DISCRETE Laplacian's eigenvalue.** What is integrated is
the 5-point stencil, whose eigenvalue ``-(4D/h^2) sin^2(q h/2)`` differs from the
continuum ``-D q^2`` by 408% at ``n = 64, j = 12`` — and the two even *disagree
about the sign* at ``j = 13``, so the discrete band extends one mode further than
the continuum one. Predicting with the continuum form would not be a slightly
looser check; it would be a different answer. Both the eigenvalue **and the
eigenvector used to seed the perturbation** come from the discrete operator, or a
second eigenmode survives in the initial condition as a transient.

**Three things it refuses to predict**, each because the alternative is a wrong
number that still looks green — the stance ``hodgkin_huxley`` takes past the Hopf:

* parameters with no real non-trivial homogeneous state (Pearson's regime);
* a mode whose eigenvalue pair is **complex**, where the amplitude oscillates as it
  decays and ``log(a(T)/a(0))/T`` depends on where the horizon lands;
* a mode whose eigenvalue is **too close to zero** to measure — at ``j = 13``,
  ``lambda = 1.06e-4`` needs ``t_max = 18822`` for two e-folds and returns ``nan``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.laplacian import cfl_limit, grid_axis, laplacian
from sandbox.core.ode import rk4_step
from sandbox.core.registry import register

# Below this the growth rate is unmeasurable: two e-folds would need a horizon of
# 2/|lambda| > 2e4 time units, which is 1e5 steps and returns nan in practice.
MIN_MEASURABLE_RATE = 1e-3

INITIAL_CONDITIONS = ("mode", "pearson")


@dataclass(frozen=True)
class GrayScottParams:
    """Gray-Scott parameters (all plain, JSON-serializable numbers).

    ``mode_j`` selects the seeded wavenumber ``q = 2 pi j / L`` — restricted to
    integers because only those are periodic on the box, and to ``j <= n/2``
    because beyond Nyquist a "wavenumber" names a different field than the caller
    means. ``eps`` is the perturbation amplitude and ``initial`` chooses between
    the validated single-mode seeding and the exploratory Pearson blob.
    """

    feed: float = 0.074
    kill: float = 0.062
    du: float = 2.0e-5
    dv: float = 1.0e-5
    n: int = 64
    length: float = 1.0
    mode_j: int = 7
    eps: float = 1.35e-5
    t_max: float = 133.6
    dt: float = 0.2
    initial: str = "mode"
    noise: float = 0.0

    def __post_init__(self) -> None:
        if self.n < 4 or self.n % 2:
            raise ValueError(f"n must be an even integer >= 4, got {self.n}")
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        for name in ("du", "dv"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.initial not in INITIAL_CONDITIONS:
            raise ValueError(f"initial must be one of {INITIAL_CONDITIONS}, got {self.initial!r}")
        if not 1 <= self.mode_j <= self.n // 2:
            raise ValueError(
                f"mode_j must be between 1 and n/2 = {self.n // 2}, got {self.mode_j}; "
                "j=0 is the uniform mode and j>n/2 aliases onto a lower one"
            )
        limit = cfl_limit(self.length / self.n, max(self.du, self.dv), ndim=2)
        if self.dt > limit:
            raise ValueError(
                f"dt ({self.dt}) exceeds the CFL limit ({limit:.4g}) for explicit "
                f"diffusion at h = L/n = {self.length / self.n:.4g}; the grid-scale "
                "mode would grow instead of decaying and the run would blow up"
            )
        steps = round(self.t_max / self.dt)
        if steps < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        if abs(steps * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{steps} steps land at {steps * self.dt}"
            )


@dataclass(frozen=True)
class GrayScottState:
    """``y`` of shape ``(2, n, n)`` — ``u`` and ``v`` — plus what observables need.

    ``shape`` (the ``cos(q x)`` template) and ``a0`` (the seeded amplitude) are
    embedded rather than recomputed because ``observables`` is called once per
    recorded step, and ``a0`` in particular must be the value from ``t = 0`` — a
    growth rate is a ratio, so it needs its own origin carried along.
    """

    y: np.ndarray
    step_index: int
    t: float
    params: GrayScottParams
    shape: np.ndarray
    a0: float
    rhs: Callable[[np.ndarray], np.ndarray]


def n_gs_steps(params: GrayScottParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


def homogeneous_state(params: GrayScottParams) -> tuple[float, float]:
    """The upper non-trivial homogeneous steady state ``(u*, v*)``.

    Solving ``u v^2 = (F+k) v`` and ``F(1-u) = u v^2`` gives
    ``(F+k) v^2 - F v + F(F+k) = 0``, real only when ``F >= 4 (F+k)^2``. The upper
    root is taken: it is the one that is stable without diffusion in the Turing
    sliver, which is the state linear stability analysis is *about*.

    Raises where no real non-trivial state exists — Pearson's famous parameters —
    because the alternative is casting a complex root to a float and validating a
    perturbation about a state the system does not have.
    """
    feed, kill = params.feed, params.kill
    disc = feed * feed - 4.0 * feed * (feed + kill) ** 2
    if disc < 0.0:
        raise ValueError(
            f"no real non-trivial homogeneous state at F={feed:g}, k={kill:g}: "
            f"F = {feed:g} < 4(F+k)^2 = {4 * (feed + kill) ** 2:g}, so the only "
            "homogeneous state is the trivial (1, 0). Patterns here are excitable "
            "structures, not Turing patterns, and linear stability analysis about a "
            "non-existent state predicts nothing about them."
        )
    v = (feed + math.sqrt(disc)) / (2.0 * (feed + kill))
    return (feed + kill) / v, v


def reaction_jacobian(u: float, v: float, feed: float, kill: float) -> np.ndarray:
    """Jacobian of the reaction terms alone (no diffusion), at ``(u, v)``."""
    return np.array(
        [
            [-(v**2) - feed, -2.0 * u * v],
            [v**2, 2.0 * u * v - (feed + kill)],
        ]
    )


def dispersion(params: GrayScottParams, mode_j: int) -> tuple[float, np.ndarray, bool]:
    """``(growth rate, eigenvector, is_real)`` for the mode ``q = 2 pi j / L``.

    The diffusion term uses the **discrete** operator: ``q_eff^2 = (4/h^2)
    sin^2(q h / 2)``, i.e. exactly ``-stencil_eigenvalue / D``. Using ``q^2`` here
    would predict a different number — see the module docstring.

    When the pair is complex the returned "rate" is ``tr/2`` (the common real
    part) and ``is_real`` is ``False``; callers that need a measurable exponential
    must reject it rather than use it.
    """
    u_star, v_star = homogeneous_state(params)
    h = params.length / params.n
    q = 2.0 * math.pi * mode_j / params.length
    q_eff_squared = (4.0 / (h * h)) * math.sin(q * h / 2.0) ** 2
    m = reaction_jacobian(u_star, v_star, params.feed, params.kill) - q_eff_squared * np.diag(
        [params.du, params.dv]
    )

    trace = m[0, 0] + m[1, 1]
    determinant = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if trace * trace - 4.0 * determinant < 0.0:
        values, vectors = np.linalg.eig(m)
        return 0.5 * trace, np.real(vectors[:, 0]), False

    values, vectors = np.linalg.eig(m)
    dominant = int(np.argmax(values.real))
    vector = np.real(vectors[:, dominant])
    vector = vector / np.linalg.norm(vector)
    if vector[0] < 0.0:
        vector = -vector
    return float(values[dominant].real), vector, True


def gray_scott_rhs(params: GrayScottParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous RHS ``f(y) -> dy/dt`` for ``y`` of shape ``(2, n, n)``.

    The single definition of the Gray-Scott vector field. ``rk4_step`` operates on
    whatever array shape it is handed, so the fields never need flattening.
    """
    h = params.length / params.n
    du, dv, feed, kill = params.du, params.dv, params.feed, params.kill

    def rhs(y: np.ndarray) -> np.ndarray:
        u, v = y[0], y[1]
        reaction = u * v * v
        return np.stack(
            [
                du * laplacian(u, h) - reaction + feed * (1.0 - u),
                dv * laplacian(v, h) + reaction - (feed + kill) * v,
            ]
        )

    return rhs


def mode_template(params: GrayScottParams) -> np.ndarray:
    """``cos(q x)`` on the grid, constant along ``y`` — the seeded and measured mode."""
    x = grid_axis(params.n, params.length)
    q = 2.0 * math.pi * params.mode_j / params.length
    return np.repeat(np.cos(q * x)[:, None], params.n, axis=1)


def mode_amplitude(field: np.ndarray, template: np.ndarray) -> float:
    """Projection of ``field`` onto ``cos(q x)``: ``2 <field, cos> / <1, 1>``.

    The factor 2 is ``1/<cos, cos>`` for a full-period cosine, so the result is the
    coefficient itself. The uniform background contributes exactly nothing, because
    ``cos`` sums to zero over whole periods — which is why this measures the
    *perturbation* without needing to subtract the steady state.
    """
    return float(2.0 * (field * template).mean())


class GrayScott:
    """Stateless Gray-Scott model. Register one shared instance."""

    def initial_state(self, params: GrayScottParams, rng: Generator) -> GrayScottState:
        template = mode_template(params)
        if params.initial == "mode":
            u_star, v_star = homogeneous_state(params)
            _, eigenvector, _ = dispersion(params, params.mode_j)
            # Seeded along the DISCRETE operator's eigenvector: any other direction
            # is a mixture of both eigenmodes, and the sub-dominant one decays as a
            # transient that contaminates the measured rate at early times.
            y = np.stack(
                [
                    u_star + params.eps * float(eigenvector[0]) * template,
                    v_star + params.eps * float(eigenvector[1]) * template,
                ]
            )
        else:
            y = _pearson_blob(params, rng)

        return GrayScottState(
            y=y,
            step_index=0,
            t=0.0,
            params=params,
            shape=template,
            a0=mode_amplitude(y[0], template),
            rhs=gray_scott_rhs(params),
        )

    def step(self, state: GrayScottState, rng: Generator) -> GrayScottState:
        params = state.params
        index = state.step_index + 1
        return GrayScottState(
            y=rk4_step(state.rhs, state.y, params.dt),
            step_index=index,
            t=index * params.dt,  # counted, not accumulated
            params=params,
            shape=state.shape,
            a0=state.a0,
            rhs=state.rhs,
        )

    def observables(self, state: GrayScottState) -> dict[str, float]:
        """Scalars only — which is why the Recorder needed no change for a PDE.

        ``growth_rate`` is ``log(a(t)/a(0))/t``, i.e. the measured exponent that
        ``analytic_predictions`` predicts. It is ``nan`` at ``t = 0`` (genuinely
        ``0/0``) and whenever the ratio is non-positive; reporting a convenient
        ``0.0`` there would be inventing a measurement.
        """
        amplitude = mode_amplitude(state.y[0], state.shape)
        ratio = amplitude / state.a0 if state.a0 != 0.0 else float("nan")
        rate = math.log(ratio) / state.t if state.t > 0.0 and ratio > 0.0 else float("nan")
        return {
            "a_q": amplitude,
            "growth_rate": rate,
            "u_mean": float(state.y[0].mean()),
            "v_mean": float(state.y[1].mean()),
        }

    def is_terminal(self, state: GrayScottState) -> bool:
        return state.step_index >= n_gs_steps(state.params)

    def fields(self, state: GrayScottState) -> dict[str, np.ndarray]:
        """The two concentration fields — the ``FieldModel`` extension, for viz only.

        Copies rather than views, because the state is meant to be immutable and a
        renderer that normalised in place would silently corrupt the trajectory.
        No validation reads this: the dispersion check goes through the scalar
        ``a_q`` observable instead.
        """
        return {"u": state.y[0].copy(), "v": state.y[1].copy()}

    def analytic_predictions(self, params: GrayScottParams) -> dict[str, float]:
        """The dispersion relation's growth rate — where it is a measurable one.

        Raises rather than returning a number it does not believe: on the
        exploratory initial condition (no seeded mode to grow), on a complex
        eigenvalue pair (the amplitude oscillates, so the endpoint ratio is not the
        exponent), and on a rate too near zero to resolve over any affordable
        horizon.
        """
        if params.initial != "mode":
            raise ValueError(
                f"initial={params.initial!r} seeds no single mode, so there is no "
                "growth rate to predict. The Pearson-regime pattern is exploratory "
                "and is reported qualitatively in the demo, never asserted."
            )
        rate, _, is_real = dispersion(params, params.mode_j)
        if not is_real:
            raise ValueError(
                f"the eigenvalue pair at mode_j={params.mode_j} is complex "
                f"(common real part {rate:+.6g}); the perturbation amplitude "
                "oscillates while it decays, so log(a(T)/a(0))/T depends on where "
                "the horizon lands in the oscillation and is not the growth rate"
            )
        if abs(rate) < MIN_MEASURABLE_RATE:
            raise ValueError(
                f"the growth rate at mode_j={params.mode_j} is {rate:+.6g}, too "
                f"close to zero to measure (|rate| < {MIN_MEASURABLE_RATE:g}): two "
                f"e-folds would need t_max = {2.0 / abs(rate):.0f}, and the measured "
                "amplitude ratio is swamped by the nonlinear correction long before "
                "the mode does anything"
            )
        return {"growth_rate": rate}


def _pearson_blob(params: GrayScottParams, rng: Generator) -> np.ndarray:
    """Exploratory initial condition: the trivial state with a perturbed central square.

    The standard recipe behind the familiar Gray-Scott images. It is **not** a
    perturbation of a Turing state — at Pearson's parameters no such state exists —
    so nothing here is validated, and ``analytic_predictions`` refuses it.
    """
    n = params.n
    u = np.ones((n, n))
    v = np.zeros((n, n))
    lo, hi = int(0.4 * n), int(0.6 * n)
    u[lo:hi, lo:hi] = 0.5
    v[lo:hi, lo:hi] = 0.25
    if params.noise > 0.0:
        u = u + params.noise * rng.standard_normal((n, n))
        v = v + params.noise * rng.standard_normal((n, n))
    return np.stack([u, v])


# The single shared, stateless instance used throughout the sandbox.
MODEL = GrayScott()
register("gray_scott", MODEL, GrayScottParams)
