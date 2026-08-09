"""Hodgkin-Huxley rate functions: the removable singularities, and the 1952 fit.

Written before ``models/hh_rates.py`` is correct (workflow rule: confirm the test
can fail first). The specific failure this file exists to catch is **not** a typo
in a coefficient — it is that ``alpha_m`` and ``alpha_n`` are ``x/(1-exp(-x/k))``
forms with removable ``0/0`` singularities at exactly ``V = -40`` and ``V = -55``.
Those are round numbers, so they are precisely the voltages a caller probes, a
demo sweeps across, or a root-finder lands on. A naive transcription returns
``nan`` there and poisons everything downstream — silently, because a ``nan``
membrane potential looks like a numerical instability, not a rate-function bug.

Everything else in Phase 2's Hodgkin-Huxley track is built on these six
functions: the voltage-clamp exact anchor validates them individually, the
deterministic model integrates them, and the stochastic model draws channel
transitions from them. So they get pinned first.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.models.hh_rates import (
    GATES,
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    rate_pair,
    steady_state,
    time_constant,
)

# The two removable singularities, with their analytic limits.
# alpha_m(V) = 0.1 (V+40)/(1-exp(-(V+40)/10))  -> 0.1 * 10 = 1.0   at V = -40
# alpha_n(V) = 0.01(V+55)/(1-exp(-(V+55)/10))  -> 0.01 * 10 = 0.1  at V = -55
SINGULARITIES = [(alpha_m, -40.0, 1.0), (alpha_n, -55.0, 0.1)]


@pytest.mark.parametrize(("fn", "v_sing", "limit"), SINGULARITIES)
def test_removable_singularity_returns_the_analytic_limit(fn, v_sing, limit):
    value = float(fn(v_sing))
    assert np.isfinite(value), f"{fn.__name__}({v_sing}) is not finite -- the 0/0 was not handled"
    assert value == pytest.approx(limit, rel=1e-12)


@pytest.mark.parametrize(("fn", "v_sing", "limit"), SINGULARITIES)
def test_rate_is_continuous_across_the_singularity(fn, v_sing, limit):
    # Approach from both sides over 10 decades. A guard implemented with too wide
    # a threshold (or a plain series with no exp branch) shows up as a kink here.
    for delta in np.logspace(-10, -1, 10):
        for sign in (-1.0, +1.0):
            value = float(fn(v_sing + sign * delta))
            assert np.isfinite(value)
            # The linoid is smooth with slope 1/2 in x, so the value must sit within
            # O(delta) of the limit -- a generous but non-vacuous bound.
            assert abs(value - limit) < 0.51 * delta + 1e-12


def _linoid_series(x: float, k: float) -> float:
    """``x/(1-exp(-x/k))`` via its Bernoulli series -- the reference, not the code.

    ``u/(1-e^-u) = 1 + u/2 + u^2/12 - u^4/720 + O(u^6)``. For ``|u| <= 1e-2`` the
    dropped ``u^6/30240`` term is ``~3e-17`` relative, i.e. below double precision,
    so this is an *independent* closed form to check the implementation against
    rather than a restatement of it.
    """
    u = x / k
    return k * (1.0 + u / 2.0 + u * u / 12.0 - u**4 / 720.0)


@pytest.mark.parametrize("k", [10.0])
def test_no_catastrophic_cancellation_in_the_transition_band(k):
    # The naive `1 - exp(-x/k)` loses precision long before it returns nan: it is a
    # difference of two nearly-equal numbers. Measured relative error of the naive
    # form against expm1 at k=10: 4.9e-10 at x=1e-6, 1.6e-11 at x=1e-5. So the band
    # just ABOVE the series guard (|x| >= 1e-7 k) is where the implementation choice
    # actually shows, and that is what this probes.
    #
    # Note this calls _linoid directly rather than going through alpha_m(V). Probing
    # via `V = -40 + delta` cannot reach here: recovering x as `(-40 + delta) + 40`
    # carries ~7e-15 of absolute rounding (one ulp at 40), which is enough to flip
    # which side of the guard threshold a nominal 1e-6 lands on -- so those probes
    # silently take the series branch and the exp branch goes untested.
    from sandbox.models.hh_rates import _linoid

    for exponent in np.arange(-6.0, -0.99, 0.25):  # x from 1e-6 up to 0.1
        for sign in (-1.0, +1.0):
            x = sign * 10.0**exponent
            assert float(_linoid(x, k)) == pytest.approx(_linoid_series(x, k), rel=1e-14)


@pytest.mark.parametrize(("fn", "v_sing", "limit"), SINGULARITIES)
def test_values_just_off_the_singularity_match_the_series(fn, v_sing, limit):
    # The same claim as above, but through the public API and across the guard
    # threshold, so the two branches are checked to agree with each other.
    for delta in (1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4):
        for sign in (-1.0, +1.0):
            x = sign * delta
            value = float(fn(v_sing + x))
            # Both rates are `coeff * _linoid(V - v_sing, 10)` with coeff = limit/10,
            # since _linoid(0, 10) = 10 and the function's value at v_sing is `limit`.
            expected = (limit / 10.0) * _linoid_series(x, 10.0)
            assert value == pytest.approx(expected, rel=1e-11)


def test_dense_sweep_is_finite_everywhere():
    # A sweep that deliberately lands ON the singular voltages (they are exact
    # multiples of the step), which is how a real f-I or clamp sweep hits them.
    V = np.arange(-100.0, 100.0 + 1e-9, 0.5)
    assert -40.0 in V and -55.0 in V
    for fn in (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n):
        values = np.asarray(fn(V), dtype=float)
        assert np.all(np.isfinite(values)), f"{fn.__name__} is non-finite somewhere on the sweep"
        assert np.all(values > 0.0), f"{fn.__name__} must be a strictly positive rate"


def test_vectorized_matches_scalar():
    V = np.array([-100.0, -65.0, -55.0, -40.0, 0.0, 50.0])
    for fn in (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n):
        vectorized = np.asarray(fn(V), dtype=float)
        scalar = np.array([float(fn(float(v))) for v in V])
        assert np.array_equal(vectorized, scalar), f"{fn.__name__} differs scalar vs vectorized"


def test_steady_state_and_time_constant_match_their_definitions():
    # x_inf = a/(a+b), tau = 1/(a+b). Circular by construction -- its job is to pin
    # the *plumbing* (gate ordering, no transposition), not the physics.
    V = np.array([-80.0, -65.0, -40.0, 0.0, 20.0])
    for i, gate in enumerate(GATES):
        a, b = rate_pair(gate, V)
        assert np.allclose(steady_state(V)[..., i], a / (a + b), rtol=0, atol=1e-15)
        assert np.allclose(time_constant(V)[..., i], 1.0 / (a + b), rtol=0, atol=1e-15)


def test_gate_order_is_m_h_n():
    # Everything downstream (the 4-vector [V, m, h, n], the channel-state model,
    # the observable keys) indexes on this order. A silent permutation would give
    # a wrong-but-plausible spike shape.
    assert GATES == ("m", "h", "n")


def test_resting_gate_values_match_the_published_fit():
    # LITERATURE-ANCHORED (category C), and labelled as such: these are the widely
    # published HH-1952 values at V_rest ~ -65 mV. They catch a transcribed
    # coefficient, which no self-consistency check can. They are evidence that the
    # fit was copied correctly -- not a derivation.
    x = steady_state(-65.0)
    assert x[0] == pytest.approx(0.0529, abs=5e-4)  # m_inf
    assert x[1] == pytest.approx(0.5961, abs=5e-4)  # h_inf
    assert x[2] == pytest.approx(0.3177, abs=5e-4)  # n_inf


def test_activation_and_inactivation_run_the_right_way():
    # m and n are ACTIVATION gates (open on depolarization); h is INACTIVATION
    # (closes). A swapped alpha/beta in any one of them passes every smoothness
    # check above and fails here.
    V = np.linspace(-90.0, 40.0, 400)
    x = steady_state(V)
    assert np.all(np.diff(x[:, 0]) > 0.0), "m_inf must increase with V"
    assert np.all(np.diff(x[:, 1]) < 0.0), "h_inf must decrease with V"
    assert np.all(np.diff(x[:, 2]) > 0.0), "n_inf must increase with V"


def test_asymptotic_gate_limits():
    # Bounds, not tolerances: the gates approach 0/1 monotonically but at very
    # different speeds, so a single shared `abs=` would be either vacuous for m
    # or wrong for n. beta_n decays with an 80 mV length constant -- the slowest
    # of the six -- so n_inf is only 0.9982 even at a wildly unphysical +200 mV.
    x_hyper = steady_state(-200.0)
    x_depol = steady_state(200.0)
    assert x_hyper[0] < 1e-3  # m closed when hyperpolarized
    assert x_hyper[1] > 1.0 - 1e-6  # h open
    assert x_hyper[2] < 1e-5  # n closed
    assert x_depol[0] > 1.0 - 1e-6  # m open when depolarized
    assert x_depol[1] < 1e-6  # h shut
    assert x_depol[2] > 0.99  # n open (slowest of the six; 0.99822 here)


def test_time_constants_are_positive_and_m_is_the_fastest():
    # tau_m is the stiffness that sets the integrator step. Phase-2 planning
    # measured min tau_m = 0.0622 ms over [-90, 60] mV, which is why explicit RK4
    # at dt = 0.01 ms is enough and no stiff solver is needed. Pin that.
    V = np.linspace(-90.0, 60.0, 3001)
    tau = time_constant(V)
    assert np.all(tau > 0.0)
    assert np.all(tau[:, 0] < tau[:, 1]), "tau_m must be faster than tau_h everywhere"
    assert np.all(tau[:, 0] < tau[:, 2]), "tau_m must be faster than tau_n everywhere"
    assert tau[:, 0].min() == pytest.approx(0.0622, abs=5e-4)
