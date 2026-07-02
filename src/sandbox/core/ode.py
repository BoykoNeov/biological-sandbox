"""Fixed-step RK4 — the deterministic limit's integrator.

Phase 1's organizing move is "stochastic simulation vs. the deterministic limit
it collapses into." The stochastic side is the Gillespie SSA; this module is the
deterministic side — a small, dependency-light integrator for the mass-action
ODE ``dc/dt = f(c)``.

Two decisions, both deliberate:

**Hand-rolled RK4, not scipy.** Non-negotiable #4 says stay in NumPy until
profiling forces otherwise. The Phase-1 RHS functions (birth-death, isomerization,
the repressilator's smooth Hill form) are non-stiff, so classical fixed-step RK4
at a modest ``dt`` is ample. ``scipy.integrate.solve_ivp`` is the noted escape
hatch if a later model's RHS turns stiff.

**Autonomous RHS ``f(y) -> dy/dt`` — no explicit ``t``.** Every deterministic
limit in the project is autonomous (mass-action rates depend on concentrations,
not wall-clock time), and the ``DeterministicLimitModel`` protocol declares
``deterministic_rhs`` to return exactly this shape. A ``t`` argument would
gratuitously diverge from that contract.

**Integration step is decoupled from any later sampling grid.** ``integrate_rk4``
returns a *dense uniform* trajectory at spacing ``dt``. Downstream code (the
convergence pathway) interpolates that onto whatever sample times it needs with
``np.interp``. This is what makes the Richardson accuracy check honest: halving
``dt`` refines the integration without touching the comparison grid, so a change
in the result is attributable to integration error alone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def integrate_rk4(
    rhs: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray | list[float] | float,
    t_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``dy/dt = rhs(y)`` from ``t = 0`` to ``t_max`` with classical RK4.

    Parameters
    ----------
    rhs:
        Autonomous right-hand side ``f(y) -> dy/dt``. Given a state vector,
        returns its time derivative of the same shape.
    y0:
        Initial state. A scalar or list is promoted to a 1-D float array, so
        callers may pass either a single concentration or a vector.
    t_max:
        Final time (inclusive up to the last whole ``dt`` step; see below).
    dt:
        Fixed integration step. Must be positive.

    Returns
    -------
    ``(t, y)`` where ``t`` has shape ``(n,)`` and ``y`` has shape ``(n, dim)``.
    ``t[0] == 0`` and ``t[-1]`` lands on ``t_max`` to within floating-point
    rounding: the number of steps is ``round(t_max / dt)``, so the horizon
    endpoint is preserved exactly regardless of ``dt`` (the Richardson check
    depends on this — halving ``dt`` must not move the endpoints).

    Notes
    -----
    NumPy-only, no adaptive stepping, no dense-output object. The RHS is assumed
    non-stiff; if that ever fails, reach for ``scipy.integrate.solve_ivp``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if t_max < 0:
        raise ValueError(f"t_max must be non-negative, got {t_max}")

    y = np.asarray(y0, dtype=float).reshape(-1)

    n_steps = round(t_max / dt)
    t = np.arange(n_steps + 1, dtype=float) * dt
    # Pin the endpoint exactly on t_max so halving dt never nudges the horizon.
    if n_steps > 0:
        t[-1] = t_max

    out = np.empty((n_steps + 1, y.size), dtype=float)
    out[0] = y

    for i in range(n_steps):
        # The last step may be a hair shorter than dt so the trajectory ends
        # exactly at t_max; every other step is a full dt.
        h = t[i + 1] - t[i]
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = y

    return t, out
