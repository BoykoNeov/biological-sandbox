"""Channel-noise Hodgkin-Huxley: the exact propagator, and what it is anchored to.

The load-bearing claim of ``models/hh_stochastic.py`` is that with ``V`` frozen
over a step, the 8-state Na chain and the 5-state K chain are advanced **exactly**
-- no ``rate * dt`` discretization at all. That claim needs an *independent*
reference, not a rearrangement of the same algebra, so this file hand-transcribes
the published kinetic scheme into a generator matrix ``Q`` and exponentiates it
with its own scaling-and-squaring routine. Neither ``Q`` nor ``expm`` is imported
from the model; the model never builds a generator.

That independence is the Phase-2 lesson applied again: the deterministic HH model
learned that root-finding-vs-integrating agreement cannot catch a wrong ``g_na``,
because both paths move consistently. Here, a factorized propagator checked
against a factorized reference would be self-consistency wearing a lab coat.
``Q`` below is written from the transition diagram -- ``m`` has three identical
subunits with per-subunit rates ``alpha_m``/``beta_m``, so the ``i -> i+1``
transition carries ``(3-i) alpha_m`` -- and the comparison is a genuine check.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import DeterministicLimitModel, Model
from sandbox.core.registry import get_model
from sandbox.models.hh_rates import alphas_betas, steady_state
from sandbox.models.hh_stochastic import (
    K_CONDUCTING,
    K_STATES,
    NA_CONDUCTING,
    NA_STATES,
    HHStochastic,
    HHStochasticParams,
    count_propagator,
    initial_occupancies,
    k_propagator,
    na_propagator,
    subunit_propagator,
)
from sandbox.models.hodgkin_huxley import HHParams, hh_rhs

PROBE_VOLTAGES = (-90.0, -65.0, -55.0, -40.0, 0.0, 40.0)
PROBE_DTS = (0.001, 0.01, 0.025, 0.1)


# --------------------------------------------------------------------------
# the independent reference: a hand-transcribed generator and our own expm
# --------------------------------------------------------------------------


def reference_expm(a: np.ndarray) -> np.ndarray:
    """``exp(A)`` by scaling-and-squaring with a Taylor series.

    Deliberately naive and deliberately local: 8x8 at most, so 30 Taylor terms
    after scaling the 1-norm below 1/2 is far past double precision. Written here
    rather than imported so the reference shares no code with the thing it checks.
    """
    norm = float(np.abs(a).sum(axis=1).max())
    s = int(np.ceil(np.log2(norm / 0.5))) if norm > 0.5 else 0
    b = a / (2.0**s)
    out = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for k in range(1, 30):
        term = term @ b / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def reference_na_generator(v: float) -> np.ndarray:
    """The 8-state Na generator, written from the kinetic diagram.

    State ``(i, j)`` -> index ``i * 2 + j``: ``i`` is the number of open ``m``
    subunits (0..3) and ``j`` is the ``h`` gate (1 = not inactivated). Conducting
    state is ``(3, 1)``. Three identical, independent ``m`` subunits means the
    ``i -> i+1`` transition happens at ``(3 - i) alpha_m`` (any of the closed
    ones) and ``i -> i-1`` at ``i beta_m``.
    """
    alpha, beta = alphas_betas(v)
    a_m, b_m = float(alpha[0]), float(beta[0])
    a_h, b_h = float(alpha[1]), float(beta[1])
    q = np.zeros((8, 8))
    for i in range(4):
        for j in range(2):
            s = i * 2 + j
            if i < 3:
                q[s, (i + 1) * 2 + j] += (3 - i) * a_m
            if i > 0:
                q[s, (i - 1) * 2 + j] += i * b_m
            q[s, i * 2 + (1 - j)] += a_h if j == 0 else b_h
    q[np.diag_indices(8)] = -q.sum(axis=1)
    return q


def reference_k_generator(v: float) -> np.ndarray:
    """The 5-state K generator: four identical, independent ``n`` subunits."""
    alpha, beta = alphas_betas(v)
    a_n, b_n = float(alpha[2]), float(beta[2])
    q = np.zeros((5, 5))
    for i in range(5):
        if i < 4:
            q[i, i + 1] += (4 - i) * a_n
        if i > 0:
            q[i, i - 1] += i * b_n
    q[np.diag_indices(5)] = -q.sum(axis=1)
    return q


# --------------------------------------------------------------------------
# the exact-propagator anchor (category A)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("v", PROBE_VOLTAGES)
@pytest.mark.parametrize("dt", PROBE_DTS)
def test_na_propagator_matches_matrix_exponential(v: float, dt: float) -> None:
    """``kron(P_m, P_h)`` IS ``expm(Q dt)`` -- to machine precision, at any dt.

    This is what separates the model from a ``rate * dt`` scheme, and it is not a
    tolerance that loosens with ``dt``: the factorization is exact, so ``dt = 0.1``
    must agree as well as ``dt = 0.001``. A scheme with ``O(dt)`` bias would pass
    at the small ``dt`` and fail at the large one -- which is exactly why the
    parametrization spans two decades of ``dt``.
    """
    measured = na_propagator(v, dt)
    reference = reference_expm(reference_na_generator(v) * dt)
    assert np.abs(measured - reference).max() < 1e-12


@pytest.mark.parametrize("v", PROBE_VOLTAGES)
@pytest.mark.parametrize("dt", PROBE_DTS)
def test_k_propagator_matches_matrix_exponential(v: float, dt: float) -> None:
    measured = k_propagator(v, dt)
    reference = reference_expm(reference_k_generator(v) * dt)
    assert np.abs(measured - reference).max() < 1e-12


@pytest.mark.parametrize("v", PROBE_VOLTAGES)
def test_propagator_rows_are_probability_distributions(v: float) -> None:
    for p in (na_propagator(v, 0.025), k_propagator(v, 0.025)):
        assert np.all(p >= 0.0)
        assert np.abs(p.sum(axis=1) - 1.0).max() < 1e-14


def test_count_propagator_is_a_binomial_convolution() -> None:
    """``P[i, i']`` is the law of ``a + b``, ``a ~ Bin(i, p_oo)``, ``b ~ Bin(k-i, p_co)``.

    Checked against an explicit enumeration over ``(a, b)`` with ``math.comb`` --
    a different expression of the same combinatorics, so a transposed or
    off-by-one convolution index shows up here rather than in a downstream slope.
    """
    import math

    k, p_oo, p_co = 4, 0.31, 0.07
    p = count_propagator(k, p_oo, p_co)
    for i in range(k + 1):
        for target in range(k + 1):
            expected = sum(
                math.comb(i, a)
                * p_oo**a
                * (1 - p_oo) ** (i - a)
                * math.comb(k - i, target - a)
                * p_co ** (target - a)
                * (1 - p_co) ** (k - i - (target - a))
                for a in range(max(0, target - (k - i)), min(i, target) + 1)
            )
            assert p[i, target] == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("v", PROBE_VOLTAGES)
def test_subunit_propagator_is_the_voltage_clamp_closed_form(v: float) -> None:
    """The 2-state factor is the very solution ``hh_voltage_clamp`` validates.

    ``p_closed->open = x_inf (1 - e^{-dt/tau})`` and ``p_open->open = x_inf +
    (1 - x_inf) e^{-dt/tau}`` are ``x(dt)`` from ``x(0) = 0`` and ``x(0) = 1`` of
    ``dx/dt = (x_inf - x)/tau``. Stating that link is the point: the channel model
    inherits the clamp anchor's rate functions rather than re-deriving them.
    """
    dt = 0.025
    alpha, beta = alphas_betas(v)
    for gate in range(3):
        a, b = float(alpha[gate]), float(beta[gate])
        x_inf, tau = a / (a + b), 1.0 / (a + b)
        p_oo, p_co = subunit_propagator(x_inf, a + b, dt)
        assert p_co == pytest.approx(x_inf * (1.0 - np.exp(-dt / tau)), rel=1e-14)
        assert p_oo == pytest.approx(x_inf + (1.0 - x_inf) * np.exp(-dt / tau), rel=1e-14)


@pytest.mark.parametrize("v", PROBE_VOLTAGES)
def test_propagator_mean_reproduces_the_gate_ode(v: float) -> None:
    """The mean number of open subunits follows the deterministic gate exactly.

    ``E[i' | i] = i p_oo + (k - i) p_co``, and dividing by ``k`` must give
    ``x_inf + (x - x_inf) e^{-dt/tau}`` with ``x = i/k``. This is the bridge the
    ``N -> infinity`` limit rides on, so it is asserted rather than assumed: if it
    failed, the stochastic model would converge to something that is not the
    Hodgkin-Huxley ODE and the whole convergence track would be measuring a
    different limit.
    """
    dt = 0.025
    alpha, beta = alphas_betas(v)
    for gate, k in ((0, 3), (2, 4)):
        a, b = float(alpha[gate]), float(beta[gate])
        x_inf = a / (a + b)
        decay = np.exp(-dt * (a + b))
        p = count_propagator(k, *subunit_propagator(x_inf, a + b, dt))
        counts = np.arange(k + 1)
        for i in counts:
            mean_next = float((p[i] * counts).sum()) / k
            x = i / k
            assert mean_next == pytest.approx(x_inf + (x - x_inf) * decay, rel=1e-13)


def test_conducting_state_indices_name_the_right_states() -> None:
    """``NA_CONDUCTING`` is ``(3 open m, h open)`` and ``K_CONDUCTING`` is ``n4``.

    Asserted via the *distribution*, not the integer: at ``v0`` the initial
    occupancy of the conducting Na state must be ``m_inf^3 h_inf`` and of the
    conducting K state ``n_inf^4``. Picking the wrong index (say ``(3, 0)``, the
    inactivated neighbour, whose weight is ``m^3 (1 - h)``) changes the number by
    a factor of 1.5 at rest and would otherwise only surface as a wrong resting
    conductance far downstream.
    """
    v0 = -65.0
    p_na, p_k = initial_occupancies(v0)
    m, h, n = (float(x) for x in steady_state(v0))
    assert p_na.shape == (NA_STATES,)
    assert p_k.shape == (K_STATES,)
    assert p_na.sum() == pytest.approx(1.0, abs=1e-15)
    assert p_k.sum() == pytest.approx(1.0, abs=1e-15)
    assert p_na[NA_CONDUCTING] == pytest.approx(m**3 * h, rel=1e-14)
    assert p_k[K_CONDUCTING] == pytest.approx(n**4, rel=1e-14)


# --------------------------------------------------------------------------
# the model's protocol behaviour
# --------------------------------------------------------------------------


def _params(**kwargs: object) -> HHStochasticParams:
    base: dict[str, object] = {"i_ext": 0.0, "t_max": 1.0, "dt": 0.025, "n_channels": 5000.0}
    base.update(kwargs)
    return HHStochasticParams(**base)  # type: ignore[arg-type]


def test_registered_and_implements_the_protocols() -> None:
    model = get_model("hh_stochastic")
    assert isinstance(model, Model)
    assert isinstance(model, DeterministicLimitModel)


@pytest.mark.parametrize("n_channels", [5000.0, 10_000.0, 12_345.0])
def test_channel_counts_are_conserved_exactly(n_channels: float) -> None:
    """Every step is a redistribution, never a creation. Integer-exact, not approximate.

    **Swept over ``n_channels``, and that is not decoration.** Written first at a
    single ``N = 5000`` this test could not see an initial state built by
    *rounding* ``N p`` instead of sampling it -- at exactly 5000 the rounded
    occupancies happen to sum to 5000 for both populations. At 10000 the K
    population sums to 10001. One lucky size made a conservation check that
    conserved nothing, so the size is now part of the test rather than a constant
    inside it. The initial state is checked too, for the same reason: the failure
    was born there and every later step preserves whatever total it is handed.
    """
    model = HHStochastic()
    params = _params(t_max=5.0, n_channels=n_channels)
    rng = np.random.default_rng(7)
    state = model.initial_state(params, rng)
    n = int(params.n_channels)
    for _ in range(201):
        assert state.na_counts.sum() == n
        assert state.k_counts.sum() == n
        assert np.all(state.na_counts >= 0)
        assert np.all(state.k_counts >= 0)
        state = model.step(state, rng)


def test_time_is_counted_not_accumulated() -> None:
    model = HHStochastic()
    params = _params(t_max=1.0)
    rng = np.random.default_rng(0)
    state = model.initial_state(params, rng)
    for i in range(1, 41):
        state = model.step(state, rng)
        assert state.step_index == i
        assert state.t == i * params.dt
    assert model.is_terminal(state)


def test_same_seed_reproduces_the_trajectory() -> None:
    model = HHStochastic()
    params = _params(t_max=2.0)
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(3)
        state = model.initial_state(params, rng)
        vs = []
        for _ in range(80):
            state = model.step(state, rng)
            vs.append(state.v)
        runs.append(vs)
    assert runs[0] == runs[1]


def test_deterministic_rhs_is_the_hodgkin_huxley_vector_field() -> None:
    """The limit is the *same* function object-shape as the deterministic model's.

    Not "an equivalent RHS" -- literally ``hh_rhs`` applied to these params, which
    duck-types because the stochastic params carry the same eight membrane
    constants. Compared on a scatter of states so a divergent constant cannot hide.
    """
    model = HHStochastic()
    params = _params(i_ext=3.0, g_na=115.0, e_k=-80.0)
    rhs = model.deterministic_rhs(params)
    reference = hh_rhs(
        HHParams(
            i_ext=3.0,
            t_max=params.t_max,
            dt=params.dt,
            v0=params.v0,
            c_m=params.c_m,
            g_na=115.0,
            g_k=params.g_k,
            g_l=params.g_l,
            e_na=params.e_na,
            e_k=-80.0,
            e_l=params.e_l,
        )
    )
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = np.concatenate([rng.uniform(-90, 50, 1), rng.uniform(0, 1, 3)])
        assert np.array_equal(rhs(y), reference(y))


def test_observables_expose_only_what_a_channel_model_actually_has() -> None:
    """``V`` plus the two conducting fractions -- and deliberately no ``m``/``h``/``n``.

    A channel-state model has no gating *variables*: it has occupancy counts whose
    ``N -> infinity`` limits are ``m^3 h`` and ``n^4``, not ``m``, ``h``, ``n``
    individually. Inventing them (say as ``(na_open)^{1/3}``) would manufacture a
    number with no microscopic referent. This is why the convergence check compares
    ``V`` alone rather than blending four species.
    """
    model = HHStochastic()
    params = _params()
    state = model.initial_state(params, np.random.default_rng(0))
    assert set(model.observables(state)) == {"V", "na_open", "k_open"}


def test_has_no_analytic_predictions_and_says_why() -> None:
    """Not an oversight: the stationary mean of ``V`` is NOT the ODE fixed point.

    Channel noise in a nonlinear system shifts the mean by ``O(1/N)`` (the
    noise-induced drift), so ``analytic_predictions -> {"V": V*}`` would be a wrong
    number that passes at loose tolerance and fails as replicates grow. The honest
    check is the ``N^{-1/2}`` convergence law, exactly as for the repressilator.
    """
    model = get_model("hh_stochastic")
    assert not hasattr(model, "analytic_predictions")


def test_initial_voltage_is_exact_and_counts_are_sampled_not_rounded() -> None:
    """``V(0)`` is exactly ``v0``; the counts are a draw, not a rounded mean.

    Rounding ``N * p`` would be a deterministic ``O(1/N)`` artifact sitting on top
    of the very ``N^{-1/2}`` law being measured.

    **The scatter is compared to the multinomial standard error, not to zero.**
    The first version of this test asserted ``fractions.std() > 0.0`` and a
    rounded implementation *passed* it: ``np.std`` of sixty bit-identical values
    returns ``2.7e-20``, not ``0.0``, because the mean subtraction is not exact.
    A threshold nothing can fail is not a check. The spread must now match
    ``sqrt(p(1-p)/N)`` to within a factor of two -- which sampling does and
    rounding misses by eighteen orders of magnitude.
    """
    params = _params(n_channels=10_000.0)
    model = HHStochastic()
    n = int(params.n_channels)
    p_na, p_k = initial_occupancies(params.v0)
    seeds = 200
    na_fractions, k_fractions = [], []
    for seed in range(seeds):
        state = model.initial_state(params, np.random.default_rng(seed))
        assert state.v == params.v0
        assert state.t == 0.0
        na_fractions.append(state.na_counts[NA_CONDUCTING] / n)
        k_fractions.append(state.k_counts[K_CONDUCTING] / n)

    for fractions, p_conducting in (
        (na_fractions, float(p_na[NA_CONDUCTING])),
        (k_fractions, float(p_k[K_CONDUCTING])),
    ):
        arr = np.asarray(fractions)
        se = np.sqrt(p_conducting * (1.0 - p_conducting) / n)
        assert 0.5 * se < arr.std(ddof=1) < 2.0 * se, (
            "occupancy scatter does not match the multinomial standard error: "
            f"measured {arr.std(ddof=1):.4g}, expected ~{se:.4g} "
            "(a rounded initial condition would give ~0)"
        )
        assert abs(arr.mean() - p_conducting) < 4.0 * se / np.sqrt(seeds)


def _discrepancy_against_the_ode(n_channels: float, seed: int, replicates: int) -> float:
    """Mean over replicates of the time-averaged ``|V_stoch(t) - V_ode(t)|``.

    A miniature of what ``convergence_report`` does, kept local and cheap. The
    stochastic times are ``step_index * dt`` and ``integrate_rk4`` returns the same
    uniform grid, so the two line up index-for-index with no interpolation at all.
    """
    params = _params(i_ext=0.5, t_max=20.0, v0=-65.0, n_channels=n_channels)
    model = HHStochastic()
    _, y_ode = integrate_rk4(
        model.deterministic_rhs(params),
        model.initial_concentrations(params),
        params.t_max,
        params.dt,
    )
    v_ode = y_ode[:, 0]
    per_replicate = []
    for child in np.random.SeedSequence(seed).spawn(replicates):
        rng = np.random.default_rng(child)
        state = model.initial_state(params, rng)
        trace = [state.v]
        while not model.is_terminal(state):
            state = model.step(state, rng)
            trace.append(state.v)
        per_replicate.append(float(np.abs(np.asarray(trace) - v_ode).mean()))
    return float(np.mean(per_replicate))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_discrepancy_shrinks_when_n_grows(seed: int) -> None:
    """Sixteen times as many channels must halve the discrepancy twice over.

    **This test exists because a mutant that never moves a single channel passed
    everything else.** Aggregating the multinomial draw along the wrong axis
    returns the input counts unchanged -- the conductances freeze at their ``t=0``
    values forever. At the resting fixed point a frozen membrane sits still in
    exactly the way a correct one does, so conservation, reproducibility, counting
    and the propagator checks all stayed green.

    Two things had to change to see it. The regime is a *transient*, so the
    deterministic limit actually moves and frozen channels cannot follow it; and
    the assertion is a **ratio between two system sizes**, not a tolerance --
    because most of a frozen-channel error is ``N``-independent, and no tolerance
    distinguishes "small" from "not shrinking".

    **The transient had to be chosen, not assumed, and the first two choices were
    wrong.** At ``i_ext = 2`` from ``V = -65`` the deterministic overshoot is 5 mV
    *toward* threshold, and channel noise turns that into real spikes: at
    ``N = 2e4`` seeds 1 and 3 gave ``D = 6.1`` instead of ``0.7``, and the ratio
    read 16 and 27 -- passing loudly for entirely the wrong reason. Raising ``N``
    to ``2e5`` did not fix it (three of six seeds still spiked, and one clean seed
    gave 2.216, *below* the bound). ``i_ext = 0.5`` is the regime that works: the
    limit runs ``-65 -> -64.558`` with a 0.9 mV peak, ~10 mV clear of threshold,
    and no seed spikes.

    Measured there, ``N = 1e5`` against ``1.6e6`` (theoretical ratio 4):

    * correct: **4.516, 4.570, 5.650, 3.854** (seeds 0-3)
    * frozen-channel mutant: **1.829, 2.055, 1.686, 1.624**

    The mutant is not at 1.0 because its *initial* counts are still sampled, so a
    genuine ``N^{-1/2}`` component survives underneath the frozen dynamics -- which
    is exactly why the bound sits at 2.5 rather than anywhere near 1.
    """
    small = _discrepancy_against_the_ode(100_000.0, seed, replicates=4)
    big = _discrepancy_against_the_ode(1_600_000.0, seed, replicates=4)
    assert small / big > 2.5, f"D(1e5)={small:.4g}, D(1.6e6)={big:.4g}, ratio={small / big:.3f}"


def test_the_voltage_half_step_is_exact_not_merely_close() -> None:
    """At frozen conductance, one step IS the linear ODE's solution -- to 1e-12.

    The reference is a *finely integrated* one: ``integrate_rk4`` at ``dt/500`` on
    the membrane current balance written out here from the post-transition
    occupancies. That is a genuinely different algorithm from a closed-form
    exponential, so agreement is a check rather than a restatement -- the same
    stance ``textbook_rhs`` takes in the deterministic tests.

    **Two mutants motivated this, and the second is why it is not enough to test
    the direction of travel.**

    * ``V <- V_inf - (V - V_inf)e^{-dt g/C}`` (sign flipped) makes the potential
      ping-pong across ``V_inf`` at almost undiminished amplitude
      (``e^{-dt g/C} = 0.983`` at rest). It survived every statistical test,
      because that amplitude is *driven by channel noise* and so still shrinks as
      ``N^{-1/2}``: the discrepancy ratio stayed near 4 and everything stayed
      green.
    * Forward Euler for ``V`` then survived a *structural* replacement test
      (``|dV| <= |dt dV/dt|``, from ``1 - e^{-x} <= x``) because Euler satisfies
      it with equality. Measured, its error is only ``(dt g/C)^2 / 2 = 0.014%``
      of ``|V - V_inf|`` -- and since ``|V - V_inf|`` is itself set by the noise,
      that bias scales *with* ``N^{-1/2}`` and never floors the slope. It is
      therefore not a threat to the convergence claim; it is a threat to the
      module's stated claim of exactness, which is what this test pins.

    Started at ``V = -70`` so ``|V - V_inf|`` is volts, not millivolts: the Euler
    error is then ~7e-4 mV per step against a 1e-12 bound.
    """
    model = HHStochastic()
    params = _params(i_ext=0.5, t_max=5.0, v0=-70.0, n_channels=50_000.0)
    n = float(params.n_channels)
    rng = np.random.default_rng(5)
    state = model.initial_state(params, rng)
    for _ in range(60):
        nxt = model.step(state, rng)
        g_na = params.g_na * float(nxt.na_counts[NA_CONDUCTING]) / n
        g_k = params.g_k * float(nxt.k_counts[K_CONDUCTING]) / n

        def frozen_conductance_rhs(
            y: np.ndarray, g_na: float = g_na, g_k: float = g_k
        ) -> np.ndarray:
            return np.array(
                [
                    (
                        params.i_ext
                        - g_na * (y[0] - params.e_na)
                        - g_k * (y[0] - params.e_k)
                        - params.g_l * (y[0] - params.e_l)
                    )
                    / params.c_m
                ]
            )

        _, y_fine = integrate_rk4(
            frozen_conductance_rhs, np.array([state.v]), params.dt, params.dt / 500.0
        )
        assert nxt.v == pytest.approx(float(y_fine[-1, 0]), abs=1e-12)
        # Direction of travel, asserted separately because it is free and it says
        # in one line what the tolerance says in twelve digits.
        assert (nxt.v - state.v) * float(frozen_conductance_rhs(np.array([state.v]))[0]) >= 0.0
        state = nxt


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.03}, "divide"),
        ({"dt": -1.0}, "positive"),
        ({"t_max": 0.0}, "positive"),
        ({"n_channels": 0.0}, "at least 1"),
        ({"n_channels": 100.5}, "whole number"),
    ],
)
def test_params_reject_configurations_that_would_bias_silently(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _params(**kwargs)
