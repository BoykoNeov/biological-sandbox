"""Generalized Lotka-Volterra — Phase 3's ``birth_death``.

    dx_i/dt = x_i (r_i + sum_j A_ij x_j)

``A_ij`` is the per-capita effect of species *j* on species *i* (row-major; the
reference matrix below is deliberately asymmetric so a transposed convention is
detectable, and it was checked against the planning slice's recorded ``x*`` and
eigenvalues before this file was written).

**Why this model exists here.** Phase 3's headline finding is that HANDOFF's
"May's criterion becomes a checkable prediction about your own simulated webs"
*cannot be run as written* — the random-gLV ensemble is empty at the ``S`` where
the criterion means anything, and conditioning on feasibility moves the spectrum
(``docs/plans/phase3-plan.md``). So the phase splits the promise: the random
matrix law is validated directly as linear algebra (3b), and gLV is validated on
**small hand-built systems where ``x*`` is exact and feasible by construction**.
This file is the second half. It claims nothing about random webs.

**Two claims, two initial conditions**, mirroring Gray-Scott's ``initial`` switch:

* ``initial="equilibrium"`` starts *at* ``x* = -A^{-1} r`` and predicts each
  ``x_i`` — so the only residual is integration error, which Richardson in ``dt``
  measures. Predicting the components individually rather than a norm is
  deliberate: a predicted ``||x - x*|| = 0`` is a one-sided check that a
  non-negative measurement can only miss in one direction.
* ``initial="relax"`` starts at ``x* + eps v``, ``v`` the eigenvector of the
  **slowest** mode of the community matrix ``M = diag(x*) A``, and predicts
  ``relaxation_rate = log(|d(T)|/|d(0)|)/T`` to equal that eigenvalue. That is a
  *linearization*, so its error is ``O(eps)`` — measured here at
  ``4.27e-3 / 4.24e-4 / 4.24e-5 / 4.23e-6`` relative for
  ``eps = 1e-2 ... 1e-5`` at ``T = 10``, i.e. ratios ``10.08, 10.01, 10.00``.
  The honest instrument is therefore **Richardson in the amplitude**, with the
  first-order factor ``2 |m(eps) - m(eps/2)|``, not a typed tolerance.

  Worth recording because it nearly became a wrong number: the planning slice
  measured ``3.03e-4`` at ``eps = 1e-2`` by *fitting* ``log|x - x*|`` over a
  window. This model uses a single **endpoint** log-ratio, a different estimator,
  and its constant is 14x larger (and drifts with ``T``: ``4.27e-3 / 2.28e-3 /
  1.52e-3`` at ``T = 10 / 20 / 30``). The ``O(eps)`` *scaling* transferred; the
  constant did not. Deriving the tolerance at runtime is what makes that safe.

**Three things it refuses to predict**, each because the alternative is a wrong
number that still looks green — the stance ``hodgkin_huxley`` takes past the Hopf
and ``gray_scott`` takes outside the Turing sliver:

* a **singular** ``A``, where no isolated interior equilibrium exists;
* an **infeasible** ``x*`` (some ``x*_i <= 0``), where the system does not go
  there at all — species go extinct and the attractor is on a boundary face.
  This is not a corner case in this phase: feasibility collapsing with ``S`` is
  precisely the measurement that forced the May reframe;
* an **unstable** ``x*`` (``max Re eig M >= 0``), and, for the relaxation claim
  only, a **complex** slowest pair — the perturbation then spirals as it decays
  and the endpoint log-ratio depends on where ``T`` lands in the oscillation.

**Observables stay scalar, and per-species keys do not scale.** ``S = 3`` here,
so ``x0 .. x_{S-1}`` are emitted unconditionally — no ``S <= k`` cap, because an
untested branch is exactly what this project keeps catching. A future large-``S``
gLV must switch to summaries (survivor count, total biomass, ``||x - x*||``) and
pass explicit ``observable_keys`` to the convergence pathway, which weights
observables equally and would otherwise let mixed scales silently reweight the
check. ``n_survivors`` is deliberately *not* emitted here: it needs an extinction
threshold, and this model has no scale to tie one to. The stochastic gLV does
(``1/Omega``), so the threshold belongs there, next to the thing that sets it.

**No ``deterministic_rhs`` yet.** :func:`glv_rhs` is module-level so the Phase-3c
stochastic model reuses this exact vector field structurally — the reason
``hh_rhs`` is module-level. The ``DeterministicLimitModel`` methods are added
when the convergence pathway actually needs them, not in advance; adding protocol
surface on speculation is what the ``FieldModel`` docstring argues against.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import rk4_step
from sandbox.core.registry import register

INITIAL_CONDITIONS = ("state", "equilibrium", "relax")


def species_keys(n_species: int) -> tuple[str, ...]:
    """``("x0", "x1", ...)`` — the observable keys naming each species' abundance."""
    return tuple(f"x{i}" for i in range(n_species))


@dataclass(frozen=True)
class GLVParams:
    """gLV parameters. ``r`` and ``A`` are nested plain numbers, not arrays.

    Lists survive a JSON round-trip and tuples do not, so ``Experiment.params``
    must carry **lists** while this frozen dataclass normalizes them to nested
    **tuples** — a list field would be both unhashable and aliased into the
    "immutable" params. Constructing ``GLVParams`` from either shape works; the
    normalization happens in ``__post_init__``.

    ``initial`` selects the claim being made (see the module docstring):
    ``"state"`` integrates from ``x_init``, ``"equilibrium"`` starts at ``x*``
    itself, and ``"relax"`` starts at ``x* + eps v`` along the slowest mode.
    """

    r: tuple[float, ...] = (1.0, 0.8, 1.2)
    A: tuple[tuple[float, ...], ...] = (
        (-1.0, -0.3, -0.2),
        (-0.4, -1.0, -0.1),
        (-0.2, -0.5, -1.0),
    )
    x_init: tuple[float, ...] = (0.1, 0.1, 0.1)
    initial: str = "state"
    eps: float = 1e-3
    t_max: float = 20.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", tuple(float(v) for v in self.r))
        object.__setattr__(self, "A", tuple(tuple(float(v) for v in row) for row in self.A))
        object.__setattr__(self, "x_init", tuple(float(v) for v in self.x_init))

        n = len(self.r)
        if n == 0:
            raise ValueError("r must name at least one species")
        if len(self.A) != n or any(len(row) != n for row in self.A):
            raise ValueError(
                f"A must be {n}x{n} to match r (len {n}), got "
                f"{len(self.A)}x{tuple(len(row) for row in self.A)}"
            )
        if self.initial not in INITIAL_CONDITIONS:
            raise ValueError(f"initial must be one of {INITIAL_CONDITIONS}, got {self.initial!r}")
        if self.initial == "state" and len(self.x_init) != n:
            raise ValueError(f"x_init must have {n} entries to match r, got {len(self.x_init)}")
        if self.initial == "relax" and self.eps <= 0:
            raise ValueError(f"eps must be positive for initial='relax', got {self.eps}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        steps = round(self.t_max / self.dt)
        if steps < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        # The prediction is evaluated at exactly t_max, and time is counted as
        # step_index * dt rather than accumulated -- so dt must divide t_max.
        if abs(steps * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{steps} steps land at {steps * self.dt}"
            )

    @property
    def n_species(self) -> int:
        return len(self.r)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(r, A)`` as float arrays."""
        return np.asarray(self.r, dtype=float), np.asarray(self.A, dtype=float)


@dataclass(frozen=True)
class GLVState:
    """``x`` (abundances), the step counter, and what the observables need.

    ``x_star`` and ``d0`` are embedded rather than recomputed because
    ``observables`` runs once per recorded step, and ``d0`` in particular must be
    the displacement at ``t = 0`` — a decay rate is a ratio, so it needs its own
    origin carried along (the same reason Gray-Scott's state carries ``a0``).
    They are ``None`` / ``nan`` unless the run is a relaxation run, because for
    arbitrary params the equilibrium may not exist at all.
    """

    x: np.ndarray
    step_index: int
    t: float
    params: GLVParams
    rhs: Callable[[np.ndarray], np.ndarray]
    x_star: np.ndarray | None
    d0: float


def n_glv_steps(params: GLVParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


def glv_rhs(params: GLVParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous RHS ``f(x) -> dx/dt``.

    The single definition of the gLV vector field in the project; Phase 3c's
    stochastic model derives its mass-action limit from this same function, so
    the two cannot drift apart.
    """
    r, a = params.arrays()

    def rhs(x: np.ndarray) -> np.ndarray:
        return x * (r + a @ x)

    return rhs


def equilibrium(params: GLVParams) -> np.ndarray:
    """The interior equilibrium ``x* = -A^{-1} r``.

    Raises on a singular ``A`` (no isolated interior equilibrium) and on an
    infeasible one (any ``x*_i <= 0``). An infeasible "equilibrium" is a root of
    the linear system that the *positive orthant* does not contain, so the
    dynamics never approach it — returning it would be the Gray-Scott error of
    Phase 2, validating against a state the system does not have.
    """
    r, a = params.arrays()
    try:
        x_star = -np.linalg.solve(a, r)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"A is singular (det = {np.linalg.det(a):.3g}), so there is no isolated "
            "interior equilibrium to predict; the system has a line or plane of "
            "equilibria and where it ends up depends on the initial condition"
        ) from exc
    if np.any(x_star <= 0.0):
        dead = [i for i, v in enumerate(x_star) if v <= 0.0]
        raise ValueError(
            f"the interior equilibrium is infeasible: x*[{dead}] = "
            f"{np.array2string(x_star[dead], precision=6)} <= 0. The attractor lies "
            "on a boundary face (those species go extinct), so the interior root is "
            "not where the system goes and predicting it would be a wrong number "
            "that still looks green. Feasibility collapsing like this is the "
            "measurement behind the phase's May reframe -- see docs/plans/phase3-plan.md"
        )
    return x_star


def community_matrix(params: GLVParams) -> np.ndarray:
    """``M = diag(x*) A`` — the Jacobian of the gLV RHS at the interior equilibrium.

    ``d/dx_j [x_i (r + A x)_i] = delta_ij (r + A x)_i + x_i A_ij``, and the first
    term vanishes at ``x*``. This is the matrix May's criterion is *about*, and
    the reason the phase validates the random-matrix law on ``M`` directly rather
    than on a gLV whose ``x*`` came from a random draw.
    """
    _, a = params.arrays()
    return np.diag(equilibrium(params)) @ a


def community_eigenvalues(params: GLVParams) -> np.ndarray:
    """Eigenvalues of :func:`community_matrix` (complex array, unsorted)."""
    return np.linalg.eigvals(community_matrix(params))


def slow_mode(params: GLVParams) -> tuple[float, np.ndarray]:
    """``(rate, unit eigenvector)`` of the **slowest** mode of the community matrix.

    The slowest mode is the one with the largest (least negative) real part: it is
    what a decaying perturbation ends up aligned with, so it is what an endpoint
    log-ratio measures.

    Raises if that eigenvalue is not real. A complex pair means the perturbation
    spirals while it decays, so ``log(|d(T)|/|d(0)|)/T`` depends on where ``T``
    lands in the oscillation and is not the decay rate — the same refusal
    ``gray_scott.analytic_predictions`` makes for a complex dispersion pair.
    """
    values, vectors = np.linalg.eig(community_matrix(params))
    index = int(np.argmax(values.real))
    value = values[index]
    if abs(value.imag) > 1e-12 * max(1.0, abs(value.real)):
        raise ValueError(
            f"the slowest mode of the community matrix is complex "
            f"({value.real:+.6g}{value.imag:+.6g}j); the perturbation spirals as it "
            "decays, so log(|d(T)|/|d(0)|)/T depends on where T lands in the "
            "oscillation and is not the relaxation rate"
        )
    vector = np.real(vectors[:, index])
    vector = vector / np.linalg.norm(vector)
    if vector[0] < 0.0:
        vector = -vector  # fix the sign so the seeded state is reproducible
    return float(value.real), vector


class GLV:
    """Stateless generalized Lotka-Volterra model. Register one shared instance."""

    def initial_state(self, params: GLVParams, rng: Generator) -> GLVState:
        # Deterministic: rng is unused.
        x_star: np.ndarray | None = None
        if params.initial == "state":
            x = np.asarray(params.x_init, dtype=float)
        elif params.initial == "equilibrium":
            x = equilibrium(params)
        else:  # "relax"
            x_star = equilibrium(params)
            _, vector = slow_mode(params)
            x = x_star + params.eps * vector

        d0 = float(np.linalg.norm(x - x_star)) if x_star is not None else float("nan")
        return GLVState(
            x=x,
            step_index=0,
            t=0.0,
            params=params,
            rhs=glv_rhs(params),
            x_star=x_star,
            d0=d0,
        )

    def step(self, state: GLVState, rng: Generator) -> GLVState:
        params = state.params
        index = state.step_index + 1
        return GLVState(
            x=rk4_step(state.rhs, state.x, params.dt),
            step_index=index,
            t=index * params.dt,  # counted, not accumulated
            params=params,
            rhs=state.rhs,
            x_star=state.x_star,
            d0=state.d0,
        )

    def observables(self, state: GLVState) -> dict[str, float]:
        """Per-species abundances, total biomass, and the relaxation rate.

        ``relaxation_rate`` is ``log(|d(t)|/|d(0)|)/t`` — the measured exponent
        that ``analytic_predictions`` predicts for a relaxation run. It is ``nan``
        outside such a run, at ``t = 0`` (genuinely ``0/0``), and if the
        displacement has collapsed to zero; reporting a convenient ``0.0`` there
        would be inventing a measurement.
        """
        out = {key: float(v) for key, v in zip(species_keys(state.x.size), state.x, strict=True)}
        out["total_biomass"] = float(state.x.sum())

        rate = float("nan")
        if state.x_star is not None and state.t > 0.0 and state.d0 > 0.0:
            displacement = float(np.linalg.norm(state.x - state.x_star))
            if displacement > 0.0:
                rate = math.log(displacement / state.d0) / state.t
        out["relaxation_rate"] = rate
        return out

    def is_terminal(self, state: GLVState) -> bool:
        return state.step_index >= n_glv_steps(state.params)

    def analytic_predictions(self, params: GLVParams) -> dict[str, float]:
        """The interior equilibrium, or the relaxation rate — never both.

        A relaxation run ends near ``x*`` but not *at* it (that is the whole point
        of measuring the approach), so predicting the components there would be
        asserting a residual the run is designed to still carry. Each initial
        condition gets exactly the claim it can support.

        Raises unless ``x*`` exists, is feasible **and** is stable: an unstable
        equilibrium is not what a long run converges to.
        """
        x_star = equilibrium(params)
        eigenvalues = np.linalg.eigvals(np.diag(x_star) @ params.arrays()[1])
        if np.any(eigenvalues.real >= 0.0):
            raise ValueError(
                f"the interior equilibrium is unstable (max Re eig = "
                f"{eigenvalues.real.max():+.4g}); the attractor is elsewhere -- a "
                "limit cycle, a boundary face, or unbounded growth -- so predicting "
                "the fixed point would be a wrong number that still looks green"
            )

        if params.initial == "relax":
            rate, _ = slow_mode(params)
            return {"relaxation_rate": rate}
        return dict(zip(species_keys(params.n_species), x_star.tolist(), strict=True))


# The single shared, stateless instance used throughout the sandbox.
MODEL = GLV()
register("glv", MODEL)
