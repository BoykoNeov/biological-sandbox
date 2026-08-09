"""Deterministic Hodgkin-Huxley — the stiff, threshold ODE of Phase 2.

The classical 1952 four-variable membrane model in the modern convention
(``V`` in mV, ``t`` in ms, ``C_m`` in uF/cm^2, conductances in mS/cm^2, currents
in uA/cm^2):

    C_m dV/dt = I_ext - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L)
    dx/dt     = alpha_x(V) (1 - x) - beta_x(V) x        for x in {m, h, n}

This is the deterministic limit that the channel-noise model converges to as the
number of ion channels grows — the ``N -> infinity`` end of Phase 2's thread. It
is also the project's first genuinely *threshold* system, which changes what
"converges" can even mean (see the Phase-2 plan: at low channel counts the
failure mode is a changed spike **count**, which obeys no ``N^{-1/2}`` law).

**Explicit RK4 is enough — measured, not assumed.** The fastest time constant
anywhere in ``[-90, 60] mV`` is ``tau_m = 0.0622 ms``, so ``dt = 0.01 ms`` sits
6.2x inside it and classical RK4 is comfortably stable and accurate: against a
``dt = 5e-4`` reference over 20 ms at ``I = 10``, the error runs
``6.45e-7 / 4.43e-8 / 2.89e-9 / 1.84e-10`` for ``dt = 0.02 / 0.01 / 0.005 /
0.0025`` — ratios 14.6, 15.3, 15.7, converging on ``2^4``. So ``core/ode.py`` is
reused as-is and **scipy stays out of the project** (non-negotiable #4).

**What ``analytic_predictions`` claims, and what it cannot.** It returns the
resting fixed point, obtained by *root-finding* the algebraic steady-state
equations — a genuinely different code path from time-integration, so making the
two agree catches drift between them. It does **not** validate the biological
constants: a wrong ``g_Na``, or ``m^2`` in place of ``m^3``, would move the root
and the trajectory *consistently* and the fixed point would still be a fixed
point. Those are caught elsewhere — by the hand-transcribed textbook RHS in
``tests/test_hodgkin_huxley.py`` (written from the paper, not from this file) and,
coarsely, by the literature-anchored ``V_rest ~ -65 mV``. The three-category
framing in ``docs/plans/phase2-plan.md`` exists precisely to keep that boundary
visible rather than letting self-consistency pass for validation.

**It refuses to predict an unstable fixed point.** Past the subcritical Hopf
(measured: still stable at ``I = 8``, unstable by ``I = 10``) the attractor is a
limit cycle and the fixed point is not where the system ends up. Returning it
anyway would be a wrong number that still looks green, so
``analytic_predictions`` raises — the same stance as ``validate()``'s
``require_termination`` guard.

**Params are flat and duplicated, deliberately.** The channel-noise model needs
the same seven membrane constants. They are repeated there rather than shared
through a nested params object, because ``convergence_report`` builds params via
``params_factory({**base, "n_channels": N})`` and that dict must stay flat and
JSON-serializable (non-negotiable #3). The invariant that actually matters —
that both models integrate the *same* RHS — is guaranteed structurally instead:
both call :func:`hh_rhs`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import rk4_step
from sandbox.core.registry import register
from sandbox.models.hh_rates import alphas_betas, steady_state

# The state vector's component order, shared by the ODE, the observables and every
# downstream column index.
STATE_KEYS: tuple[str, str, str, str] = ("V", "m", "h", "n")


@dataclass(frozen=True)
class HHParams:
    """Hodgkin-Huxley parameters (all plain, JSON-serializable numbers).

    ``i_ext`` is the injected current density in uA/cm^2, ``v0`` the potential the
    cell is initialised at (its gates start at ``x_inf(v0)``), ``t_max`` the run
    length in ms and ``dt`` the RK4 step, which must divide ``t_max`` exactly.
    The remaining seven are the membrane constants; the defaults are the standard
    1952 squid-axon values.
    """

    i_ext: float = 0.0
    t_max: float = 50.0
    dt: float = 0.01
    v0: float = -65.0
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.387

    def __post_init__(self) -> None:
        if self.c_m <= 0:
            raise ValueError(f"c_m must be positive, got {self.c_m}")
        for name in ("g_na", "g_k", "g_l"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        n = round(self.t_max / self.dt)
        if n < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        # Same reason as the voltage clamp: the prediction is evaluated at exactly
        # t_max, and time is counted as step_index * dt rather than accumulated.
        if abs(n * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{n} steps land at {n * self.dt}"
            )


@dataclass(frozen=True)
class HHState:
    """``y = [V, m, h, n]``, the step counter, and the embedded params + RHS."""

    y: np.ndarray
    step_index: int
    t: float
    params: HHParams
    rhs: Callable[[np.ndarray], np.ndarray]


def n_hh_steps(params: HHParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


def hh_rhs(params: HHParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous RHS ``f(y) -> dy/dt`` for ``y = [V, m, h, n]``.

    The single definition of the Hodgkin-Huxley vector field in the project. The
    channel-noise model's ``deterministic_rhs`` returns this same function, so the
    stochastic model and its deterministic limit cannot drift apart — the same
    reason the Gillespie engine derives its ODE from the very ``rates`` the SSA
    samples.
    """
    c_m, i_ext = params.c_m, params.i_ext
    g_na, g_k, g_l = params.g_na, params.g_k, params.g_l
    e_na, e_k, e_l = params.e_na, params.e_k, params.e_l

    def rhs(y: np.ndarray) -> np.ndarray:
        v = y[0]
        m, h, n = y[1], y[2], y[3]
        i_ionic = g_na * m**3 * h * (v - e_na) + g_k * n**4 * (v - e_k) + g_l * (v - e_l)
        alpha, beta = alphas_betas(v)
        out = np.empty(4, dtype=float)
        out[0] = (i_ext - i_ionic) / c_m
        out[1:] = alpha * (1.0 - y[1:]) - beta * y[1:]
        return out

    return rhs


def steady_state_current(v: np.ndarray | float, params: HHParams) -> np.ndarray:
    """Net current at voltage ``v`` with every gate at ``x_inf(v)``.

    Its roots are the fixed points: the gates equilibrate to ``x_inf(V)``, so the
    whole 4-D steady-state problem collapses onto this one scalar equation.
    """
    x = steady_state(v)
    m, h, n = x[..., 0], x[..., 1], x[..., 2]
    v = np.asarray(v, dtype=float)
    return (
        params.i_ext
        - params.g_na * m**3 * h * (v - params.e_na)
        - params.g_k * n**4 * (v - params.e_k)
        - params.g_l * (v - params.e_l)
    )


def resting_state(params: HHParams, *, n_scan: int = 20_001, iterations: int = 200) -> np.ndarray:
    """The fixed point ``[V*, m*, h*, n*]``, by root-finding — not by integrating.

    Scans ``[E_K - 10, E_Na + 10]`` for sign changes of
    :func:`steady_state_current`, then bisects. Bisection rather than Newton
    because it cannot diverge and needs no derivative; 200 halvings take the
    bracket to well below machine precision, and the whole thing costs
    microseconds.

    Raises unless the scan finds exactly one root, and **says what it did find**.

    At the standard 1952 conductances the root is provably unique for *any*
    ``i_ext``: the steady-state I-V curve ``I_ss(V)`` is monotonically increasing
    across the whole scanned window (measured, and asserted in
    ``tests/test_hodgkin_huxley.py``), so ``i_ext - I_ss(V)`` crosses zero exactly
    once. Hodgkin-Huxley's famous bistability is between a stable fixed point and
    a stable *limit cycle*, not between two fixed points — which is why the guard
    never fires at the defaults.

    It is kept anyway because the conductances are params: turn ``g_na`` up and
    ``I_ss`` can acquire the N-shape that gives **three** fixed points. If that
    happens the honest report is not "your params are exotic" but "there are three
    equilibria here and I am not going to pick one for you", so the message lists
    the brackets it found. Two further limits worth knowing: a grid scan cannot see
    a *tangency* (two roots merging without a sign change), nor a pair of roots
    inside one cell — at 20k points neither is likely, but the guard is not proof
    of uniqueness. The monotonicity test is.
    """
    lo_v, hi_v = params.e_k - 10.0, params.e_na + 10.0
    grid = np.linspace(lo_v, hi_v, n_scan)
    values = steady_state_current(grid, params)
    sign_changes = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    if sign_changes.size != 1:
        brackets = (
            ", ".join(f"[{grid[i]:.4f}, {grid[i + 1]:.4f}]" for i in sign_changes[:6]) or "none"
        )
        raise ValueError(
            f"expected exactly one resting potential in [{lo_v:g}, {hi_v:g}] for "
            f"i_ext={params.i_ext:g}, found {sign_changes.size} (brackets: {brackets}"
            f"{', ...' if sign_changes.size > 6 else ''}). Three roots means the "
            "steady-state I-V curve is N-shaped at these conductances and the system "
            "has three equilibria; this helper will not pick one for you. Zero means "
            "i_ext lies outside the range the membrane can balance."
        )
    lo, hi = float(grid[sign_changes[0]]), float(grid[sign_changes[0] + 1])
    f_lo = float(steady_state_current(lo, params))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = float(steady_state_current(mid, params))
        if f_lo * f_mid <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    v_star = 0.5 * (lo + hi)
    return np.concatenate([[v_star], np.asarray(steady_state(v_star), dtype=float)])


def jacobian(params: HHParams, y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian of :func:`hh_rhs` at ``y``, by central differences.

    Central rather than forward differences: the ``O(eps^2)`` truncation keeps the
    eigenvalues accurate enough to decide stability near the Hopf point, where a
    forward difference's ``O(eps)`` error can straddle zero. The perturbation is
    scaled per component because ``V`` is O(100) while the gates are O(1).
    """
    y = np.asarray(y, dtype=float)
    rhs = hh_rhs(params)
    out = np.empty((4, 4), dtype=float)
    for j in range(4):
        step = eps * max(1.0, abs(float(y[j])))
        up, down = y.copy(), y.copy()
        up[j] += step
        down[j] -= step
        out[:, j] = (rhs(up) - rhs(down)) / (2.0 * step)
    return out


def fixed_point_eigenvalues(params: HHParams) -> np.ndarray:
    """Eigenvalues of the Jacobian at the resting fixed point."""
    return np.linalg.eigvals(jacobian(params, resting_state(params)))


class HodgkinHuxley:
    """Stateless deterministic Hodgkin-Huxley model. Register one shared instance."""

    def initial_state(self, params: HHParams, rng: Generator) -> HHState:
        # Deterministic: rng is unused. The cell starts rested at v0, i.e. with each
        # gate already at its steady state for that voltage.
        y = np.concatenate([[params.v0], np.asarray(steady_state(params.v0), dtype=float)])
        return HHState(y=y, step_index=0, t=0.0, params=params, rhs=hh_rhs(params))

    def step(self, state: HHState, rng: Generator) -> HHState:
        params = state.params
        y = rk4_step(state.rhs, state.y, params.dt)
        index = state.step_index + 1
        return HHState(
            y=y,
            step_index=index,
            t=index * params.dt,  # counted, not accumulated
            params=params,
            rhs=state.rhs,
        )

    def observables(self, state: HHState) -> dict[str, float]:
        return dict(zip(STATE_KEYS, state.y.tolist(), strict=True))

    def is_terminal(self, state: HHState) -> bool:
        return state.step_index >= n_hh_steps(state.params)

    def analytic_predictions(self, params: HHParams) -> dict[str, float]:
        """The resting fixed point — provided it is actually stable.

        Raises past the subcritical Hopf, where the attractor is a limit cycle and
        the fixed point is emphatically *not* where a long run ends up. A silent
        wrong answer there would be worse than no answer.
        """
        y_star = resting_state(params)
        eigenvalues = np.linalg.eigvals(jacobian(params, y_star))
        if np.any(eigenvalues.real >= 0.0):
            raise ValueError(
                f"the fixed point at i_ext={params.i_ext:g} is unstable "
                f"(max Re eig = {eigenvalues.real.max():+.4g}); past the subcritical "
                "Hopf the attractor is a limit cycle, so the fixed point is not what "
                "a long run converges to and predicting it would be a wrong number "
                "that still looks green"
            )
        return dict(zip(STATE_KEYS, y_star.tolist(), strict=True))

    def deterministic_rhs(self, params: HHParams) -> Callable[[np.ndarray], np.ndarray]:
        """The same vector field this model integrates — see :func:`hh_rhs`."""
        return hh_rhs(params)

    def initial_concentrations(self, params: HHParams) -> np.ndarray:
        """``[V, m, h, n]`` at ``t = 0``; the ``DeterministicLimitModel`` entry point."""
        return np.concatenate([[params.v0], np.asarray(steady_state(params.v0), dtype=float)])


# The single shared, stateless instance used throughout the sandbox.
MODEL = HodgkinHuxley()
register("hodgkin_huxley", MODEL)
