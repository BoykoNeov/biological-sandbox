"""Schnakenberg reaction-diffusion — and validated wavelength *selection*.

    du/dt = Du lap(u) + a - u + u^2 v
    dv/dt = Dv lap(v) + b - u^2 v

on a periodic box with 1-D or 2-D fields. This is the project's second PDE, and it
exists for the claim Phase 2 could not make.

**What Gray-Scott validates and what this validates are different claims.**
Gray-Scott checks ``lambda(q)``: the growth rate of a mode *you seed by hand*.
HANDOFF §5 asked to "validate pattern wavelength against linear stability
analysis", which is a claim about the mode that appears when you seed **nothing** —
and Gray-Scott cannot make it, because at Pearson's parameters there is no real
non-trivial homogeneous state, so the famous spots are excitable structures rather
than Turing patterns and linear theory sets no wavelength for them. Schnakenberg's
non-trivial state exists for *every* positive ``a, b`` and its Turing bifurcation is
supercritical (measured: amplitude continuous from zero, monotone, no jump), so the
near-onset selection argument actually applies here.

**Everything about the onset is closed form**, and ``det J = u*^2`` *exactly* is why:

* ``u* = a + b``, ``v* = b/(a+b)^2`` (solve ``u^2 v = b`` then ``a - u + b = 0``);
* ``f_u = (b-a)/(a+b)``, ``g_v = -(a+b)^2``, and ``det J = u*^2`` identically, so the
  Turing determinant condition holds for every positive ``a, b`` and the whole onset
  question reduces to one inequality in ``f_u``;
* hence ``q_c^2 = u*/sqrt(Du Dv)`` and ``d_c = [u*(1 + sqrt(1+f_u))/f_u]^2``.

**The prediction uses the DISCRETE Laplacian's eigenvalue**, as Gray-Scott's does,
and here it decides the *wavelength* rather than only a rate: at about 4.3 cells per
wavelength the stencil and the continuum disagree by 2.6 modes about which mode is
fastest, and the emergent pattern went with the stencil: across the three grids where
the two answers are 2-3 modes apart, the measured peak matched the continuum's answer
**0 times in 24 runs** (once in 32, on a fourth grid where the answers are adjacent).

**Three things learned by measuring rather than by reading**
(`docs/plans/phase2c-schnakenberg-measurement.md`):

1. **A grid can be too coarse to have the claim at all.** Coarsen far enough and the
   stencil's fastest mode is Nyquist — a checkerboard, and the measured pattern
   disagrees with it. Coarser still and the instability *vanishes*, because the
   largest representable ``q_eff^2 = 4/h^2`` falls below the unstable band. A grid
   too coarse to carry the pattern models a different problem, not this one badly.
2. **The emergent mode is quantized, and its ensemble mean has no exact target.**
   Comparing against the fastest *integer* mode looked like a resolution-dependent
   bias; most of it was the wrong target, since a mean over integers need not be one.
   But neither the integer nor the continuous maximiser survives at 4 SE on both
   grids, and the deviation does **not** shrink as replicates grow. Hence
   :func:`selection_report` asserts a *discrimination margin*, and
   :meth:`Schnakenberg.analytic_predictions` **refuses** the random initial
   condition — this project's first refusal on statistical rather than algebraic
   grounds.
3. **Four seeds were not enough.** The selected mode matched the prediction at 4/4
   seeds and at 6/8. The claim is about a distribution, so it needs one.

Refusals, each because the alternative is a wrong number that still looks green:
a mode whose eigenvalue pair is **complex** (the amplitude oscillates as it decays,
so the endpoint log-ratio is not the exponent — measured 2.8% off at ``mode_j=12``);
a rate too near zero to resolve; a homogeneous state that is **already unstable
without diffusion** (then the pattern is not a Turing pattern); and the random
initial condition, for the statistical reason above.
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

# Below this the growth rate is unmeasurable over any affordable horizon: two
# e-folds need 2/|lambda|, and the nonlinear correction arrives first. Gray-Scott's
# threshold, for the same reason.
MIN_MEASURABLE_RATE = 1e-3

# The pattern's wavenumber is still drifting below this many e-folds of the fastest
# mode. Swept at lambda* t = 12, 16, 20, 24, 30 across grids, and then checked far out:
# the selected mode is IDENTICAL at 22.7, 45.3 and 113.3 e-folds at both shipped
# resolutions, so past this threshold the answer is horizon-independent rather than
# merely slow-moving. (What does keep changing out there is how much power sits in
# harmonics -- see peak_power_fraction, whose first docstring confused the two.)
MIN_SELECTION_EFOLDS = 20.0

INITIAL_CONDITIONS = ("mode", "noise")


@dataclass(frozen=True)
class SchnakenbergParams:
    """Schnakenberg parameters (all plain, JSON-serializable numbers).

    ``initial`` picks the claim: ``"mode"`` seeds a single Fourier mode along the
    discrete operator's eigenvector and has an exact predicted growth rate;
    ``"noise"`` seeds a small random perturbation and is the wavelength-selection
    configuration, checked by :func:`selection_report` rather than by
    ``analytic_predictions``.

    ``ndim`` is 1 or 2. The validated claims are 1-D — in 2-D the emergent pattern
    also selected the predicted mode, but at two seeds and about a minute a run,
    which belongs in a demo.
    """

    a: float = 0.05
    b: float = 1.0
    du: float = 1.0e-3
    dv: float = 9.155730e-3  # d/d_c = 1.2 at the default (a, b); see critical_ratio
    n: int = 256
    length: float = 8.0
    ndim: int = 1
    mode_j: int = 24
    eps: float = 1.0e-5
    noise: float = 1.0e-3
    initial: str = "noise"
    t_max: float = 300.0
    dt: float = 0.02  # 0.375 of the CFL limit at the default grid, and divides t_max

    def __post_init__(self) -> None:
        if self.n < 4 or self.n % 2:
            raise ValueError(f"n must be an even integer >= 4, got {self.n}")
        if self.ndim not in (1, 2):
            raise ValueError(f"ndim must be 1 or 2, got {self.ndim}")
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        for name in ("a", "b", "du", "dv"):
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
        limit = cfl_limit(self.length / self.n, max(self.du, self.dv), ndim=self.ndim)
        if self.dt > limit:
            raise ValueError(
                f"dt ({self.dt}) exceeds the CFL limit ({limit:.4g}) for explicit "
                f"diffusion at h = L/n = {self.length / self.n:.4g} in {self.ndim}-D; "
                "the grid-scale mode would grow instead of decaying"
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
class SchnakenbergState:
    """``y`` of shape ``(2, n)`` or ``(2, n, n)`` — ``u`` and ``v`` — plus origins.

    ``shape``/``a0`` carry the seeded mode's template and its ``t = 0`` amplitude,
    because a growth rate is a ratio and needs its own origin. ``initial_mode`` is
    the mode the random perturbation loaded most at ``t = 0``, carried along so a
    test can assert the *emergent* mode does not track it — measured across four
    seeds whose initial modes were 112, 32, 24 and 17 and whose final modes were all
    24.
    """

    y: np.ndarray
    step_index: int
    t: float
    params: SchnakenbergParams
    shape: np.ndarray
    a0: float
    initial_mode: int
    rhs: Callable[[np.ndarray], np.ndarray]


def n_steps(params: SchnakenbergParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


def homogeneous_state(params: SchnakenbergParams) -> tuple[float, float]:
    """The homogeneous steady state ``(u*, v*) = (a+b, b/(a+b)^2)``.

    Unlike Gray-Scott's, this state exists for every positive ``a, b`` and needs no
    discriminant check: ``u^2 v = b`` and ``a - u + u^2 v = 0`` give it directly.
    """
    return params.a + params.b, params.b / (params.a + params.b) ** 2


def reaction_jacobian(u: float, v: float) -> np.ndarray:
    """Jacobian of the reaction terms alone, at ``(u, v)``.

    ``f = a - u + u^2 v``, ``g = b - u^2 v``, so the Jacobian is independent of
    ``a`` and ``b`` except through the point it is evaluated at.
    """
    return np.array([[-1.0 + 2.0 * u * v, u * u], [-2.0 * u * v, -(u * u)]])


def reaction_determinant(params: SchnakenbergParams) -> float:
    """``det J`` at the steady state, which is ``u*^2`` exactly.

    Kept as its own function computed *from the Jacobian* so a test can assert the
    identity rather than restate it: returning ``u*^2`` here would make the check a
    tautology about this source file.
    """
    u_star, v_star = homogeneous_state(params)
    j = reaction_jacobian(u_star, v_star)
    return float(j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0])


def critical_ratio(params: SchnakenbergParams) -> float:
    """The diffusion ratio ``d = Dv/Du`` at Turing onset.

    From ``Dv f_u + Du g_v = 2 sqrt(Du Dv det)`` with ``det = u*^2``: in
    ``s = sqrt(d)`` that is ``f_u s^2 - 2 u* s + g_v = 0``, so
    ``s_c = u*(1 + sqrt(1 + f_u))/f_u`` (the root above 1).

    Raises when ``f_u <= 0``: with no self-activation of ``u`` there is no Turing
    instability at any ratio, and returning a large number would suggest one exists
    just out of reach.
    """
    u_star, v_star = homogeneous_state(params)
    f_u = float(reaction_jacobian(u_star, v_star)[0, 0])
    if f_u <= 0.0:
        raise ValueError(
            f"f_u = {f_u:+.6g} <= 0 at a={params.a:g}, b={params.b:g}: the activator "
            "does not self-activate, so no diffusion ratio makes this state Turing "
            "unstable and there is no critical ratio to return"
        )
    s = u_star * (1.0 + math.sqrt(1.0 + f_u)) / f_u
    return s * s


def critical_wavenumber(params: SchnakenbergParams) -> float:
    """``q_c = sqrt(u*/sqrt(Du Dv))`` — the wavenumber that goes unstable first.

    Exact at onset only; away from onset the fastest wavenumber moves and
    :func:`fastest_mode` is the quantity to use. Verified against an argmax of the
    growth rate at onset: ``19.496952`` vs ``19.497001``.
    """
    u_star, _ = homogeneous_state(params)
    return math.sqrt(u_star / math.sqrt(params.du * params.dv))


def is_diffusionless_stable(params: SchnakenbergParams) -> bool:
    """Is the homogeneous state stable with the diffusion switched off?

    A Turing claim is *about* a state that is stable on its own and destabilized by
    diffusion. ``det J = u*^2 > 0`` always, so this reduces to ``trace < 0``.
    """
    u_star, v_star = homogeneous_state(params)
    j = reaction_jacobian(u_star, v_star)
    return bool(j[0, 0] + j[1, 1] < 0.0)


def q_effective_squared(q: float, h: float, ndim: int = 1) -> float:
    """``(4/h^2) sin^2(q h/2)`` per populated axis — minus the stencil's eigenvalue.

    A mode along one axis populates one axis whatever ``ndim`` is, so this does not
    multiply by ``ndim``: the template is ``cos(q x)``, constant along the others.
    """
    return (4.0 / (h * h)) * math.sin(q * h / 2.0) ** 2


def dispersion(params: SchnakenbergParams, mode_j: float) -> tuple[float, np.ndarray, bool]:
    """``(growth rate, eigenvector, is_real)`` for the mode ``q = 2 pi j / L``.

    Uses the **discrete** operator, so this is the rate of the mode in the
    simulation rather than in the equation on paper. Accepts a non-integer ``j``,
    because the maximiser over continuous ``j`` is a quantity the measurement needed
    (see :func:`fastest_mode`) even though only integers live on the grid.

    When the pair is complex the returned rate is ``tr/2`` (the common real part) and
    ``is_real`` is ``False``; a caller needing a measurable exponential must reject
    it rather than use it.
    """
    u_star, v_star = homogeneous_state(params)
    h = params.length / params.n
    q = 2.0 * math.pi * mode_j / params.length
    m = reaction_jacobian(u_star, v_star) - q_effective_squared(q, h) * np.diag(
        [params.du, params.dv]
    )

    trace = m[0, 0] + m[1, 1]
    determinant = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    values, vectors = np.linalg.eig(m)
    if trace * trace - 4.0 * determinant < 0.0:
        return 0.5 * trace, np.real(vectors[:, 0]), False

    dominant = int(np.argmax(values.real))
    vector = np.real(vectors[:, dominant])
    vector = vector / np.linalg.norm(vector)
    if vector[0] < 0.0:
        vector = -vector
    return float(values[dominant].real), vector, True


def continuum_dispersion(params: SchnakenbergParams, mode_j: float) -> float:
    """The same rate computed with ``-D q^2`` — the equation on paper, for contrast.

    Not used by any prediction. It exists so tests and demos can quote the number a
    continuum calculation would have given (39% off at a growing mode, 184% at
    Nyquist, and 2.6 modes off about *which* mode is fastest at 4.3 cells per
    wavelength) instead of asserting the distinction abstractly.
    """
    u_star, v_star = homogeneous_state(params)
    q = 2.0 * math.pi * mode_j / params.length
    m = reaction_jacobian(u_star, v_star) - q * q * np.diag([params.du, params.dv])
    trace = m[0, 0] + m[1, 1]
    determinant = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if trace * trace - 4.0 * determinant < 0.0:
        return 0.5 * trace
    return 0.5 * (trace + math.sqrt(trace * trace - 4.0 * determinant))


def unstable_band(params: SchnakenbergParams) -> np.ndarray:
    """The integer modes ``1 <= j <= n/2`` whose discrete growth rate is positive.

    The band's *integer content* is what decides whether selection is a measurement
    or a tautology: with one unstable mode, "the fastest mode grew" says only that
    the only mode that could grow, grew. The box length is the lever — at
    ``d/d_c = 1.2`` the count runs 1, 3, 6, 12, 25 for ``L = 1, 2, 4, 8, 16``.
    """
    js = np.arange(1, params.n // 2 + 1)
    rates = np.array([dispersion(params, int(j))[0] for j in js])
    return js[rates > 0.0]


def fastest_mode(params: SchnakenbergParams, *, continuous: bool = False) -> float:
    """The mode with the largest discrete growth rate.

    ``continuous=False`` returns the best **integer** mode, which is what a single
    run can produce. ``continuous=True`` maximises over real ``j`` by grid-and-zoom,
    which is what an *ensemble mean over integers* should be compared against — the
    two differ by up to 0.3 modes, and confusing them accounted for most of an
    apparent bias. Neither is exact to 4 SE on every grid; see
    :func:`selection_report`.
    """
    js = np.arange(1, params.n // 2 + 1)
    rates = np.array([dispersion(params, int(j))[0] for j in js])
    best = int(js[int(np.argmax(rates))])
    if not continuous:
        return float(best)

    lo, hi = max(0.5, best - 1.5), min(params.n / 2.0, best + 1.5)
    for _ in range(4):
        grid = np.linspace(lo, hi, 2001)
        values = np.array([dispersion(params, float(j))[0] for j in grid])
        k = int(np.argmax(values))
        span = (hi - lo) / 2000.0
        lo, hi = grid[k] - 2.0 * span, grid[k] + 2.0 * span
    return 0.5 * (lo + hi)


def schnakenberg_rhs(params: SchnakenbergParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous RHS ``f(y) -> dy/dt``; ``y`` is ``(2, n)`` or ``(2, n, n)``.

    The single definition of the vector field. ``laplacian`` is dimension-agnostic
    and ``rk4_step`` works on whatever shape it is handed, so 1-D and 2-D need no
    separate code path.
    """
    h = params.length / params.n
    a, b, du, dv = params.a, params.b, params.du, params.dv

    def rhs(y: np.ndarray) -> np.ndarray:
        u, v = y[0], y[1]
        reaction = u * u * v
        return np.stack(
            [du * laplacian(u, h) + a - u + reaction, dv * laplacian(v, h) + b - reaction]
        )

    return rhs


def mode_template(params: SchnakenbergParams) -> np.ndarray:
    """``cos(q x)`` on the grid, constant along any further axis."""
    x = grid_axis(params.n, params.length)
    q = 2.0 * math.pi * params.mode_j / params.length
    line = np.cos(q * x)
    if params.ndim == 1:
        return line
    return np.repeat(line[:, None], params.n, axis=1)


def mode_amplitude(field: np.ndarray, template: np.ndarray) -> float:
    """Projection of ``field`` onto ``cos(q x)``: ``2 <field, cos>``.

    The factor 2 is ``1/<cos, cos>`` for a whole number of periods, so the result is
    the coefficient. The uniform background contributes exactly nothing because
    ``cos`` sums to zero over whole periods, which is why this measures the
    perturbation without subtracting the steady state.
    """
    return float(2.0 * (field * template).mean())


def power_by_mode(field: np.ndarray) -> np.ndarray:
    """Power at each integer mode ``0 .. n/2``, with the uniform mode zeroed.

    One definition for both dimensionalities and for both quantities read off it, so
    the dominant mode and its power share can never be computed from two different
    spectra. In 1-D this is the ``rfft`` magnitude squared. In 2-D it is the
    **radially binned** 2-D spectrum, because a pattern of wavenumber ``q`` spreads
    its power around a ring rather than onto a single ``(jx, jy)`` — a stripe pattern
    and a spot pattern of the same wavelength must report the same wavenumber.

    The uniform component is removed by **zeroing mode 0 and nothing else** — the
    field is *not* mean-subtracted first. Both would work, and having both is worse
    than having one: with the mean already subtracted, mode 0 comes out as exact
    floating-point cancellation on a symmetric field, so deleting the zeroing changes
    nothing and no test can fail. That was found by mutation, and it is this project's
    "a threshold nothing can fail is not a check" arriving in a spectral estimator.
    One mechanism, and removing it is catastrophic and therefore visible.
    """
    if field.ndim == 1:
        power = np.abs(np.fft.rfft(field)) ** 2
        power[0] = 0.0
        return power

    n = field.shape[0]
    full = np.abs(np.fft.fft2(field)) ** 2
    axis = np.fft.fftfreq(n, d=1.0 / n)
    radius = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    radial = np.bincount(
        np.rint(radius).astype(int).ravel(), weights=full.ravel(), minlength=n // 2 + 1
    )[: n // 2 + 1]
    radial[0] = 0.0
    return radial


def dominant_mode(field: np.ndarray) -> int:
    """The integer mode carrying the most power, excluding the uniform mode.

    Deliberately an integer and deliberately the *peak*. A sub-mode estimator
    (parabolic interpolation, or a power-weighted centroid) was measured and
    rejected: the pattern really is an integer mode on a periodic box, so a
    continuous estimator dresses up the quantization instead of removing it — and
    peak and centroid were measured to diverge by more than the mode spacing once
    the spectrum broadens.
    """
    return int(np.argmax(power_by_mode(field)))


def peak_power_fraction(field: np.ndarray) -> float:
    """Share of the non-uniform power sitting in the dominant mode.

    **Strongly horizon-dependent, and an earlier version of this docstring quoted one
    horizon's number as if it were the model's.** Measured at three seeds, ``d/d_c =
    1.2``, over horizons of 22.7, 45.3 and 113.3 e-folds of the fastest mode:

    * 10.6 cells per wavelength: ``0.53-0.90``, then ``0.86-0.97``, then ``0.997-0.999``;
    * 4.3 cells per wavelength: ``0.45-0.70``, then ``0.39-0.78``, and **frozen there** —
      the coarse grid never cleans up, because the harmonics it carries have nowhere
      to go.

    So this is a description of how *sinusoidal* the saturated pattern is, not a
    settling criterion: at ``1.2 d_c`` the modulation is about 50% of the background,
    which is not weakly nonlinear, and roughly half the power sits in harmonics until
    a long horizon drains it. The **selected mode**, by contrast, is identical at all
    three horizons on both grids, which is what :data:`MIN_SELECTION_EFOLDS` is about.
    """
    power = power_by_mode(field)
    total = power.sum()
    return float(power.max() / total) if total > 0.0 else float("nan")


class Schnakenberg:
    """Stateless Schnakenberg model. Register one shared instance."""

    def initial_state(self, params: SchnakenbergParams, rng: Generator) -> SchnakenbergState:
        u_star, v_star = homogeneous_state(params)
        template = mode_template(params)
        shape = (params.n,) if params.ndim == 1 else (params.n, params.n)

        if params.initial == "mode":
            _, eigenvector, _ = dispersion(params, params.mode_j)
            # Seeded along the DISCRETE operator's eigenvector: any other direction
            # mixes in the sub-dominant eigenmode, which decays as a transient and
            # contaminates the measured rate at early times.
            y = np.stack(
                [
                    u_star + params.eps * float(eigenvector[0]) * template,
                    v_star + params.eps * float(eigenvector[1]) * template,
                ]
            )
        else:
            # Only u is perturbed: the claim is that the growth rate picks the mode,
            # so the initial condition should carry no wavenumber information beyond
            # a flat random spectrum.
            y = np.stack(
                [
                    np.full(shape, u_star) + params.noise * rng.standard_normal(shape),
                    np.full(shape, v_star),
                ]
            )

        return SchnakenbergState(
            y=y,
            step_index=0,
            t=0.0,
            params=params,
            shape=template,
            a0=mode_amplitude(y[0], template),
            initial_mode=dominant_mode(y[0]),
            rhs=schnakenberg_rhs(params),
        )

    def step(self, state: SchnakenbergState, rng: Generator) -> SchnakenbergState:
        params = state.params
        index = state.step_index + 1
        return SchnakenbergState(
            y=rk4_step(state.rhs, state.y, params.dt),
            step_index=index,
            t=index * params.dt,  # counted, not accumulated
            params=params,
            shape=state.shape,
            a0=state.a0,
            initial_mode=state.initial_mode,
            rhs=state.rhs,
        )

    def observables(self, state: SchnakenbergState) -> dict[str, float]:
        """Scalars only, so the Recorder needed no change for a second PDE.

        ``growth_rate`` is ``log(a(t)/a(0))/t`` — the exponent
        ``analytic_predictions`` predicts — and is ``nan`` at ``t = 0`` (genuinely
        ``0/0``) and whenever the ratio is non-positive. ``dominant_mode`` is ``nan``
        at ``t = 0`` too: the field there is the homogeneous state plus noise, so
        there is no pattern to have a wavenumber. The mode the noise *loaded* is
        reported separately and constantly as ``initial_mode``, which is what makes
        "the emergent mode does not track the initial condition" checkable.
        """
        amplitude = mode_amplitude(state.y[0], state.shape)
        ratio = amplitude / state.a0 if state.a0 != 0.0 else float("nan")
        rate = math.log(ratio) / state.t if state.t > 0.0 and ratio > 0.0 else float("nan")
        return {
            "a_q": amplitude,
            "growth_rate": rate,
            "dominant_mode": float(dominant_mode(state.y[0])) if state.t > 0.0 else float("nan"),
            "initial_mode": float(state.initial_mode),
            "peak_power_fraction": peak_power_fraction(state.y[0]),
            "pattern_amplitude": float(0.5 * (state.y[0].max() - state.y[0].min())),
            "u_mean": float(state.y[0].mean()),
            "v_mean": float(state.y[1].mean()),
        }

    def is_terminal(self, state: SchnakenbergState) -> bool:
        return state.step_index >= n_steps(state.params)

    def fields(self, state: SchnakenbergState) -> dict[str, np.ndarray]:
        """The two concentration fields — the ``FieldModel`` extension, viz only.

        Shape is ``(n,)`` in 1-D and ``(n, n)`` in 2-D; Phase 4's front-end already
        reports ``field_shapes`` because ``trait_branching``'s field is 1-D, so a
        non-2-D field needs nothing new there. Copies, not views: the state is meant
        to be immutable and a renderer normalising in place would corrupt the run.
        """
        return {"u": state.y[0].copy(), "v": state.y[1].copy()}

    def analytic_predictions(self, params: SchnakenbergParams) -> dict[str, float]:
        """The seeded mode's growth rate — where it is a measurable one.

        **Refuses the random initial condition**, and the reason is statistical
        rather than algebraic, which is a first for this project. ``validate()``
        asserts ``|mean - predicted| <= z * SE``; the emergent wavenumber is
        quantized by the box, and measurement shows no single scalar matches its
        ensemble mean to 4 SE — at 6.4 cells per wavelength the deviation from the
        continuous maximiser *grows* with replicate count (1.08, 0.98, 1.81, 2.75 SE
        at R = 8, 16, 32, 48) because the mean stays put while the error bar shrinks,
        and the better target flips between grids. A prediction here would be a check
        that passes because ``R`` is small. :func:`selection_report` asserts the
        discrimination instead.
        """
        if params.initial != "mode":
            raise ValueError(
                f"initial={params.initial!r} seeds no single mode, so there is no "
                "growth rate to predict. The emergent wavenumber IS predicted, but "
                "as a discrimination rather than a scalar an ensemble mean matches: "
                "use selection_report(). See phase2c-schnakenberg-measurement.md §7."
            )
        if not is_diffusionless_stable(params):
            u_star, v_star = homogeneous_state(params)
            j = reaction_jacobian(u_star, v_star)
            raise ValueError(
                f"the homogeneous state at a={params.a:g}, b={params.b:g} is already "
                f"unstable without diffusion (trace {j[0, 0] + j[1, 1]:+.6g} > 0), so "
                "a growing mode here is not a diffusion-driven instability and "
                "calling it a Turing pattern would be wrong"
            )
        rate, _, is_real = dispersion(params, params.mode_j)
        if not is_real:
            raise ValueError(
                f"the eigenvalue pair at mode_j={params.mode_j} is complex (common "
                f"real part {rate:+.6g}); the perturbation amplitude oscillates while "
                "it decays, so log(a(T)/a(0))/T depends on where the horizon lands — "
                "measured 2.8% off at this model's own mode_j=12"
            )
        if abs(rate) < MIN_MEASURABLE_RATE:
            raise ValueError(
                f"the growth rate at mode_j={params.mode_j} is {rate:+.6g}, too close "
                f"to zero to measure (|rate| < {MIN_MEASURABLE_RATE:g}): two e-folds "
                f"would need t_max = {2.0 / abs(rate):.0f}, and the nonlinear "
                "correction arrives long before the mode does anything"
            )
        return {"growth_rate": rate}


# The single shared, stateless instance used throughout the sandbox.
MODEL = Schnakenberg()
register("schnakenberg", MODEL, SchnakenbergParams)
