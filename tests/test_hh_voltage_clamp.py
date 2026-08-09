"""Voltage clamp: Phase 2's exact analytic anchor (category A).

Written before ``models/hh_voltage_clamp.py`` exists (workflow rule: confirm the
test can fail first).

**Why this model exists.** Hold the membrane potential fixed and the Hodgkin-
Huxley gating equations *decouple*: each gate obeys its own scalar linear ODE
``dx/dt = (x_inf(V) - x)/tau(V)`` with constant coefficients, solved exactly by

    x(t) = x_inf + (x_0 - x_inf) exp(-t / tau).

That is a genuine closed form for a *trajectory*, not just a stationary scalar,
and it costs milliseconds to check. It is the ``birth_death`` of Phase 2 — the
anchor that keeps the Hodgkin-Huxley track's correctness from resting on the
fuzzier convergence check alone.

**What it does and does not validate — stated, not glossed.** The closed form is
built from the same ``x_inf`` / ``tau`` the model integrates, so an error *inside*
a rate function would cancel on both sides. What this file proves is that the
integrator, the decoupled gating structure and the per-gate plumbing reproduce
the exact solution of the ODE they claim to solve. The **rate functions
themselves** are pinned independently in ``test_hh_rates.py`` (published values at
rest, activation/inactivation direction, asymptotic limits, and the removable
singularities). Together those two files close the loop; neither does it alone.

**Tolerance is derived, never hardcoded.** The model is deterministic, so the
replicate standard error is exactly zero and ``validate()``'s statistical
tolerance degenerates. The honest tolerance here is not statistical but
*numerical* — the RK4 truncation error — so it is **measured by Richardson
extrapolation** (run at ``dt`` and ``dt/2``, take the difference) and fed to
``validate()`` as ``sem_floor``. A number that is measured each run, rather than
typed in, keeps this consistent with the ValidationSuite's philosophy.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.protocol import Experiment
from sandbox.core.recorder import run_replicate
from sandbox.core.validation import validate
from sandbox.models.hh_rates import GATES, steady_state, time_constant
from sandbox.models.hh_voltage_clamp import (
    MODEL,
    HHVoltageClampParams,
    n_clamp_steps,
)


def PARAMS_FACTORY(d: dict) -> HHVoltageClampParams:  # noqa: N802 (reads as a constant)
    """The core stays params-agnostic; callers supply the concrete constructor."""
    return HHVoltageClampParams(**d)


def exact_gates(v_hold: float, v_clamp: float, t: np.ndarray | float) -> np.ndarray:
    """``x(t) = x_inf + (x_0 - x_inf) e^{-t/tau}`` for all three gates.

    The reference. Shaped ``(..., 3)`` in ``GATES`` order, matching ``steady_state``.
    """
    x0 = steady_state(v_hold)
    x_inf = steady_state(v_clamp)
    tau = time_constant(v_clamp)
    t = np.asarray(t, dtype=float)[..., None]
    return x_inf + (x0 - x_inf) * np.exp(-t / tau)


def run_clamp(v_clamp: float, *, v_hold: float = -65.0, t_max: float = 10.0, dt: float = 0.01):
    """One replicate; returns ``(times, gates)`` with ``gates`` shaped ``(n, 3)``.

    Asserts the run actually reached its terminal state. ``validate()`` has the
    ``require_termination`` guard for this, but these helpers call ``run_replicate``
    directly and would otherwise **silently truncate** at ``max_steps`` — every
    comparison here is against the closed form evaluated at ``t_max``, so a short
    trajectory would be compared at the wrong horizon and read as integration error.
    """
    params = HHVoltageClampParams(v_clamp=v_clamp, v_hold=v_hold, t_max=t_max, dt=dt)
    traj = run_replicate(
        MODEL, params, np.random.default_rng(0), max_steps=n_clamp_steps(params) + 10
    )
    assert traj.terminated, (
        f"clamp run at v_clamp={v_clamp}, dt={dt} hit max_steps without terminating; "
        "the trajectory would be silently short of t_max"
    )
    times, series = traj.as_arrays()
    return times, np.stack([series[g] for g in GATES], axis=1)


def max_trajectory_error(
    v_clamp: float, dt: float, *, v_hold: float = -65.0, t_max: float = 10.0
) -> float:
    """Largest ``|simulated - exact|`` over the whole clamp, all three gates.

    Deliberately the max over *time*, not the value at ``t_max``. By 10 ms every
    gate has relaxed onto ``x_inf`` and both the exact and the numerical solution
    agree there to machine precision (~1e-14) whatever ``dt`` was — so a check
    that looks only at the endpoint measures floating-point noise and would report
    a meaningless convergence order. All of RK4's error lives in the transient.
    """
    times, gates = run_clamp(v_clamp, v_hold=v_hold, t_max=t_max, dt=dt)
    return float(np.abs(gates - exact_gates(v_hold, v_clamp, times)).max())


def richardson_bound(
    v_clamp: float, dt: float, *, v_hold: float = -65.0, t_max: float = 10.0
) -> float:
    """Measured bound on the ``dt`` run's error: ``max|x(dt) - x(dt/2)|``.

    For a 4th-order method the fine run's error is ~1/16 of the coarse one, so
    this difference is ~15/16 of the coarse error — a tight, *measured* bound
    rather than a typed-in epsilon. Compared on the shared grid: the coarse times
    are exactly every second fine time, so ``fine[::2]`` aligns with ``coarse``.
    """
    _, coarse = run_clamp(v_clamp, v_hold=v_hold, t_max=t_max, dt=dt)
    _, fine = run_clamp(v_clamp, v_hold=v_hold, t_max=t_max, dt=dt / 2.0)
    return float(np.abs(coarse - fine[::2]).max())


def richardson_bound_at_t_max(
    v_clamp: float, dt: float, *, v_hold: float = -65.0, t_max: float = 10.0
) -> float:
    """The same bound restricted to ``t_max`` — what ``validate()`` actually compares."""
    _, coarse = run_clamp(v_clamp, v_hold=v_hold, t_max=t_max, dt=dt)
    _, fine = run_clamp(v_clamp, v_hold=v_hold, t_max=t_max, dt=dt / 2.0)
    return float(np.abs(coarse[-1] - fine[-1]).max())


@pytest.mark.parametrize("v_clamp", [-20.0, 0.0, 20.0, -80.0])
def test_trajectory_matches_the_exact_exponential(v_clamp):
    # The whole trajectory, not just its final value: a wrong tau that happened to
    # land near the right endpoint would still bend the wrong way in between.
    #
    # The tolerance is MEASURED, not typed: Richardson bounds this run's own
    # integration error, and the comparison is against an independent closed form,
    # so converging to the wrong answer still fails (the bound would be small while
    # the error stayed large). A first draft used atol=1e-9 and failed honestly --
    # the real max error at dt=0.01 reaches 4.1e-8 at v_clamp=+20, where tau_m is
    # fastest (0.165 ms) and the transient is steepest.
    error = max_trajectory_error(v_clamp, 0.01)
    bound = richardson_bound(v_clamp, 0.01)
    assert error <= 4.0 * bound, f"error {error:.3e} exceeds 4x Richardson bound {bound:.3e}"
    # Guard against a degenerate pass: a bound this large would mean dt is simply
    # too coarse for the claim to mean anything, whatever the ratio says.
    assert bound < 1e-5


@pytest.mark.parametrize("v_clamp", [-40.0, -55.0])
def test_clamping_to_the_singular_voltages_works(v_clamp):
    # V = -40 and V = -55 are exactly where alpha_m / alpha_n are 0/0. A clamp
    # protocol steps to round numbers, so this is the realistic way that trap is
    # hit -- the rate-function fix is what makes this test possible at all.
    times, gates = run_clamp(v_clamp)
    assert np.all(np.isfinite(gates))
    error = max_trajectory_error(v_clamp, 0.01)
    assert error <= 4.0 * richardson_bound(v_clamp, 0.01)
    assert np.all(np.isfinite(exact_gates(-65.0, v_clamp, times)))


def test_validate_reproduces_analytic_predictions():
    # Routed through the ValidationSuite (non-negotiable #2), with the tolerance
    # MEASURED by Richardson rather than typed in. Two replicates, not one:
    # validate() cannot form a standard error from a single sample (it returns
    # inf, and the check would pass vacuously).
    v_clamp = 0.0
    params = HHVoltageClampParams(v_clamp=v_clamp)
    # validate() compares the FINAL observable, so the bound must be the final-time
    # one. By t_max = 10 ms the gates have fully relaxed onto x_inf, which is why
    # this number sits at machine precision (~1e-13) while the trajectory's peak
    # error is five orders of magnitude larger.
    bound = richardson_bound_at_t_max(v_clamp, 0.01)
    assert bound > 0.0, "Richardson bound collapsed to zero; it would be a hardcoded epsilon"

    experiment = Experiment(
        model="hh_voltage_clamp",
        params={"v_clamp": v_clamp, "v_hold": -65.0, "t_max": 10.0, "dt": 0.01},
        replicates=2,
        observables=GATES,
        seed=0,
        max_steps=n_clamp_steps(params) + 10,
    )
    report = validate(experiment, PARAMS_FACTORY, z=4.0, sem_floor=bound)
    assert report.passed, str(report)
    assert {c.name for c in report.checks} == set(GATES)
    # Deterministic model, distinct RNG streams -> the replicates must agree exactly.
    assert all(c.sem == 0.0 for c in report.checks), str(report)


@pytest.mark.parametrize("v_clamp", [-80.0, 0.0, 20.0])
def test_error_is_fourth_order_in_dt(v_clamp):
    # Category B. This is what licenses the Richardson bound above: if the method
    # were secretly lower order, |x(dt) - x(dt/2)| would not bound the error of the
    # coarse run the way a 4th-order method makes it.
    #
    # Measured on the max-over-trajectory error, for the reason in
    # max_trajectory_error's docstring -- on the endpoint alone this ratio is a
    # ratio of two machine-noise numbers and means nothing. Measured ratios at
    # dt = 0.01 -> 0.005 -> 0.0025: 16.6 / 16.3 at v=-80, 16.3 / 16.1 at v=0,
    # 16.4 / 16.2 at v=+20.
    coarse = max_trajectory_error(v_clamp, 0.01)
    fine = max_trajectory_error(v_clamp, 0.005)
    assert coarse > 0.0 and fine > 0.0
    ratio = coarse / fine
    assert 12.0 < ratio < 20.0, f"expected ~16 for RK4, got {ratio:.2f}"


def test_clamping_to_the_holding_potential_leaves_the_gates_still():
    # x_0 == x_inf, so every derivative is exactly zero: a fixed point. This
    # catches a sign error in the RHS that a decaying exponential can mask.
    times, gates = run_clamp(-65.0, v_hold=-65.0)
    assert np.allclose(gates, gates[0], rtol=0, atol=1e-14)
    assert times[-1] == pytest.approx(10.0)


def test_gates_stay_in_the_unit_interval():
    for v_clamp in (-90.0, -65.0, -30.0, 0.0, 40.0):
        _, gates = run_clamp(v_clamp)
        assert np.all(gates >= 0.0) and np.all(gates <= 1.0)


def test_the_run_lands_exactly_on_t_max():
    # t is computed as step_index * dt, never accumulated -- 1000 additions of 0.01
    # drift enough that `t >= t_max` misses and the run overshoots by a whole step,
    # which would put the final sample at the wrong time for the closed form.
    times, _ = run_clamp(0.0, t_max=10.0, dt=0.01)
    assert times[-1] == pytest.approx(10.0, abs=1e-12)
    assert times.size == 1001


def test_predictions_cover_every_observable():
    params = HHVoltageClampParams(v_clamp=0.0)
    assert set(MODEL.analytic_predictions(params)) == set(
        MODEL.observables(MODEL.initial_state(params, np.random.default_rng(0)))
    )


def test_analytic_predictions_match_the_closed_form():
    params = HHVoltageClampParams(v_clamp=-20.0, v_hold=-70.0, t_max=3.0, dt=0.01)
    predicted = MODEL.analytic_predictions(params)
    expected = exact_gates(-70.0, -20.0, 3.0)
    for i, gate in enumerate(GATES):
        assert predicted[gate] == pytest.approx(float(expected[i]), rel=1e-14)


def test_experiment_round_trips():
    experiment = Experiment(
        model="hh_voltage_clamp",
        params={"v_clamp": 0.0, "v_hold": -65.0, "t_max": 2.0, "dt": 0.01},
        replicates=2,
        observables=GATES,
        seed=3,
        max_steps=500,
    )
    assert Experiment.from_json(experiment.to_json()) == experiment


def test_rejects_a_dt_that_does_not_divide_t_max():
    # The closed-form comparison samples at exactly t_max; a dt that cannot land
    # there would silently compare at a slightly different time.
    with pytest.raises(ValueError, match="divide"):
        HHVoltageClampParams(v_clamp=0.0, t_max=1.0, dt=0.3)
