"""RK4 integrator: exact linear checks + 4th-order convergence.

Written before ``core/ode.py`` exists (workflow rule: confirm the test can fail
first). These checks are what make the Phase-1 convergence plan's "high-Omega
reference floor" reasoning trustworthy — if RK4 were secretly lower order (a
coefficient bug), the deterministic reference would floor the discrepancy and we
would wrongly blame the SSA. So we pin both the closed-form match *and* the order.
"""

from __future__ import annotations

import numpy as np

from sandbox.core.ode import integrate_rk4


def test_exponential_decay_matches_closed_form():
    # dy/dt = -gamma y  =>  y(t) = y0 exp(-gamma t).
    gamma = 1.5
    y0 = np.array([2.0])
    t, y = integrate_rk4(lambda c: -gamma * c, y0, t_max=4.0, dt=0.01)
    expected = y0 * np.exp(-gamma * t)[:, None]
    assert np.allclose(y, expected, rtol=0, atol=1e-8)


def test_harmonic_oscillator_matches_closed_form():
    # (y1, y2)' = (y2, -y1), y0 = (1, 0)  =>  (cos t, -sin t). Coupled/vector RHS.
    def rhs(c):
        return np.array([c[1], -c[0]])

    y0 = np.array([1.0, 0.0])
    t, y = integrate_rk4(rhs, y0, t_max=2.0 * np.pi, dt=0.001)
    expected = np.stack([np.cos(t), -np.sin(t)], axis=1)
    assert np.allclose(y, expected, rtol=0, atol=1e-6)


def test_global_error_is_fourth_order():
    # Halving dt must cut the global error by ~2^4 = 16x. This is the check that
    # would go red for Euler or a wrong Butcher coefficient.
    gamma = 1.3
    y0 = np.array([1.0])

    def final_error(dt: float) -> float:
        t, y = integrate_rk4(lambda c: -gamma * c, y0, t_max=3.0, dt=dt)
        exact = float(y0[0]) * np.exp(-gamma * t[-1])
        return abs(y[-1, 0] - exact)

    coarse = final_error(0.1)
    fine = final_error(0.05)
    ratio = coarse / fine
    # 4th order => ratio ~ 16. Allow a generous window (leading-order regime).
    assert 12.0 < ratio < 20.0


def test_halving_dt_preserves_start_and_end_times():
    # The Richardson check in convergence.py relies on this: changing dt must not
    # move the horizon endpoints (only the internal step density).
    y0 = np.array([1.0])
    t_coarse, _ = integrate_rk4(lambda c: -c, y0, t_max=5.0, dt=0.1)
    t_fine, _ = integrate_rk4(lambda c: -c, y0, t_max=5.0, dt=0.05)
    assert t_coarse[0] == 0.0 and t_fine[0] == 0.0
    assert np.isclose(t_coarse[-1], 5.0) and np.isclose(t_fine[-1], 5.0)


def test_scalar_and_vector_initial_conditions_both_work():
    # np.asarray(y0, dtype=float) should make a plain float y0 behave like a 1-vector.
    t_s, y_s = integrate_rk4(lambda c: -c, 1.0, t_max=1.0, dt=0.01)
    t_v, y_v = integrate_rk4(lambda c: -c, [1.0], t_max=1.0, dt=0.01)
    assert y_s.shape == y_v.shape
    assert np.allclose(y_s, y_v)


def test_returned_arrays_have_consistent_lengths():
    t, y = integrate_rk4(lambda c: -c, np.array([1.0, 2.0]), t_max=2.0, dt=0.1)
    assert t.ndim == 1
    assert y.shape == (t.size, 2)
