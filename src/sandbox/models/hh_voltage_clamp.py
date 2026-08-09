"""Voltage-clamped Hodgkin-Huxley gating — Phase 2's exact analytic anchor.

Hold the membrane potential at a fixed ``V`` and the Hodgkin-Huxley system stops
being a coupled nonlinear mess: the voltage feedback is cut, so each gate obeys
its own *scalar linear* ODE with constant coefficients,

    dx/dt = (x_inf(V) - x) / tau(V),

solved exactly by ``x(t) = x_inf + (x_0 - x_inf) exp(-t/tau)``. That is a genuine
closed form for a whole **trajectory**, not merely a stationary scalar, and it
costs milliseconds to check — which is why this is the lead category-A anchor of
the Phase-2 Hodgkin-Huxley track, the ``birth_death`` of this phase. The
headline model (channel-noise convergence) is validated by a *scaling law*; this
model is what keeps the track's correctness from resting on that alone.

It is also the honest experimental protocol rather than a contrivance: hold at
``v_hold``, step to ``v_clamp``, watch the gates relax. That is what a
voltage-clamp rig does.

**What it validates, and what it does not.** The closed form is assembled from
the same ``x_inf`` / ``tau`` that ``step`` integrates, so an error *inside* a rate
function cancels on both sides. What this model proves is that the integrator,
the decoupled gating structure, and the per-gate plumbing reproduce the exact
solution of the ODE they claim to solve. The rate functions themselves are pinned
independently in ``tests/test_hh_rates.py`` — published values at rest, the
activation/inactivation directions, the asymptotic limits, and the removable
singularities at ``V = -40`` / ``V = -55``. Neither file closes the loop alone;
together they do. Stating that boundary is the point, not a caveat.

**Time is counted in steps, never accumulated.** ``state.t`` is ``step_index *
dt``. Adding ``dt`` a thousand times drifts by enough that a ``t >= t_max``
terminal test misses its mark and the run overshoots by a whole step — which
would sample the trajectory at the wrong time for the closed-form comparison.
``dt`` is also required to divide ``t_max`` exactly, for the same reason.

**A fixed numerical ``dt`` that lives in the params** is the third time-advance
discipline in the project (Wright-Fisher advances one generation, Gillespie a
sampled ``tau``). It is precisely the case the no-``dt``-argument ``step``
signature was designed to accommodate: the model owns its increment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import rk4_step
from sandbox.core.registry import register
from sandbox.models.hh_rates import GATES, steady_state, time_constant


@dataclass(frozen=True)
class HHVoltageClampParams:
    """Clamp protocol parameters (all plain, JSON-serializable numbers).

    ``v_hold`` is the potential the cell rests at before the step (its gates start
    at ``x_inf(v_hold)``); ``v_clamp`` is the potential it is stepped to and held
    at. ``t_max`` is the clamp duration in ms and ``dt`` the RK4 step, which must
    divide it exactly.
    """

    v_clamp: float
    v_hold: float = -65.0
    t_max: float = 10.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        n = round(self.t_max / self.dt)
        if n < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        # Not fussiness: the analytic prediction is evaluated at exactly t_max, so a
        # dt the horizon is not a whole multiple of would compare the simulation at
        # one time against the closed form at another, and the gap would look like
        # integration error.
        if abs(n * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{n} steps land at {n * self.dt}"
            )


@dataclass(frozen=True)
class HHVoltageClampState:
    """Gate values in ``GATES`` order, the step counter, and the embedded protocol.

    ``x_inf`` and ``tau`` are evaluated once at ``v_clamp`` and carried here rather
    than recomputed inside ``step``: the clamp holds ``V`` fixed, so they are
    constants of the run. This is the same pattern as the repressilator embedding
    its ``ReactionNetwork`` — ``initial_state`` embeds everything ``step`` needs,
    keeping ``step`` a pure function of ``(state, rng)``.
    """

    gates: np.ndarray
    step_index: int
    t: float
    params: HHVoltageClampParams
    x_inf: np.ndarray
    tau: np.ndarray


def n_clamp_steps(params: HHVoltageClampParams) -> int:
    """Number of ``dt`` steps in the clamp — the run's terminal step index."""
    return round(params.t_max / params.dt)


class HHVoltageClamp:
    """Stateless voltage-clamp model. Register one shared instance."""

    @staticmethod
    def _rhs(x_inf: np.ndarray, tau: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def rhs(x: np.ndarray) -> np.ndarray:
            return (x_inf - x) / tau

        return rhs

    def initial_state(self, params: HHVoltageClampParams, rng: Generator) -> HHVoltageClampState:
        # Deterministic: rng is unused. Two replicates with different streams must
        # agree exactly, which is what makes validate()'s SE legitimately zero.
        return HHVoltageClampState(
            gates=np.asarray(steady_state(params.v_hold), dtype=float),
            step_index=0,
            t=0.0,
            params=params,
            x_inf=np.asarray(steady_state(params.v_clamp), dtype=float),
            tau=np.asarray(time_constant(params.v_clamp), dtype=float),
        )

    def step(self, state: HHVoltageClampState, rng: Generator) -> HHVoltageClampState:
        params = state.params
        gates = rk4_step(self._rhs(state.x_inf, state.tau), state.gates, params.dt)
        index = state.step_index + 1
        return HHVoltageClampState(
            gates=gates,
            step_index=index,
            t=index * params.dt,  # counted, not accumulated -- see the module docstring
            params=params,
            x_inf=state.x_inf,
            tau=state.tau,
        )

    def observables(self, state: HHVoltageClampState) -> dict[str, float]:
        return dict(zip(GATES, state.gates.tolist(), strict=True))

    def is_terminal(self, state: HHVoltageClampState) -> bool:
        # On the step counter, not on `t >= t_max`: the float comparison is what
        # drifts, and overshooting by one step moves the final sample off t_max.
        return state.step_index >= n_clamp_steps(state.params)

    def analytic_predictions(self, params: HHVoltageClampParams) -> dict[str, float]:
        """Exact gate values at ``t_max``: ``x_inf + (x_0 - x_inf) e^{-t_max/tau}``."""
        x0 = steady_state(params.v_hold)
        x_inf = steady_state(params.v_clamp)
        tau = time_constant(params.v_clamp)
        final = x_inf + (x0 - x_inf) * np.exp(-params.t_max / tau)
        return dict(zip(GATES, np.asarray(final, dtype=float).tolist(), strict=True))


# The single shared, stateless instance used throughout the sandbox.
MODEL = HHVoltageClamp()
register("hh_voltage_clamp", MODEL)
