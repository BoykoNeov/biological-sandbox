"""Generalized Lotka-Volterra: the equilibrium, the community matrix, relaxation.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` steps 1, 2 and 4.

**What is anchored to what** — the three-category framing Phase 2 established:

* **Category A, independent.** :func:`textbook_rhs` below is a hand-written double
  loop, transcribed from the equation rather than from the implementation, and the
  **two-species symmetric closed form** ``x* = 1/(1+a)``, ``eig = x*(-1 -+ a)``,
  which does *not* come from ``-A^{-1} r``. Those are the teeth. The reference
  matrix is deliberately asymmetric so that a transposed ``A`` convention shows up
  as a disagreement rather than passing silently.
* **Category A, self-consistency.** ``analytic_predictions`` solves a linear
  system; the simulation integrates an ODE. Making them agree catches drift
  between the two code paths and a non-attracting "equilibrium" — but a
  transposition made *consistently* in both would still produce a genuine fixed
  point of the wrong RHS, which is why the loop above is load-bearing.
* **Category B.** RK4's order 4 on the transient, and the relaxation rate as an
  ``O(eps)`` limit measured by Richardson in the amplitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.protocol import Experiment
from sandbox.core.recorder import run_replicate
from sandbox.core.validation import validate
from sandbox.models.glv import (
    MODEL,
    GLVParams,
    community_matrix,
    equilibrium,
    glv_rhs,
    n_glv_steps,
    slow_mode,
    species_keys,
)

# The planning slice's three-species reference case. Asymmetric on purpose.
R3 = [1.0, 0.8, 1.2]
A3 = [[-1.0, -0.3, -0.2], [-0.4, -1.0, -0.1], [-0.2, -0.5, -1.0]]
KEYS3 = species_keys(3)


def params_factory(d: dict) -> GLVParams:
    return GLVParams(**d)


def reference(**overrides) -> GLVParams:
    return GLVParams(**{"r": R3, "A": A3, **overrides})


def textbook_rhs(x: np.ndarray, p: GLVParams) -> np.ndarray:
    """``dx_i/dt = x_i (r_i + sum_j A_ij x_j)``, hand-written as a double loop.

    Deliberately shares no code with the model — no ``@``, no broadcasting — so a
    transposed ``A``, a missing ``x_i`` prefactor or an off-by-one in the sum shows
    up as a disagreement. This is the independent anchor; the fixed-point check is
    only self-consistency.
    """
    n = len(p.r)
    out = np.empty(n, dtype=float)
    for i in range(n):
        total = p.r[i]
        for j in range(n):
            total += p.A[i][j] * x[j]
        out[i] = x[i] * total
    return out


def run_glv(params: GLVParams) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One replicate; returns ``(times, series)``."""
    traj = run_replicate(
        MODEL, params, np.random.default_rng(0), max_steps=n_glv_steps(params) + 10
    )
    assert traj.terminated, "gLV run hit max_steps without terminating"
    return traj.as_arrays()


def final_x(params: GLVParams) -> np.ndarray:
    _, series = run_glv(params)
    return np.array([series[k][-1] for k in species_keys(params.n_species)])


# --------------------------------------------------------------------------
# Category A, independent: the RHS itself
# --------------------------------------------------------------------------


def test_rhs_matches_the_hand_written_loop():
    p = reference()
    rhs = glv_rhs(p)
    rng = np.random.default_rng(11)
    for _ in range(25):
        x = rng.uniform(0.0, 3.0, size=3)
        assert np.allclose(rhs(x), textbook_rhs(x, p), rtol=1e-13, atol=0.0)


def test_a_is_indexed_as_the_effect_of_j_on_i():
    # THE transpose detector. A[0][1] = -0.3 and A[1][0] = -0.4 differ, so putting
    # only species 1 in the system pins which index the RHS reads. A transposed
    # implementation gives -0.4 here and the assertion fails.
    p = reference()
    x = np.array([1.0, 1.0, 0.0])
    dx = glv_rhs(p)(x)
    assert dx[0] == pytest.approx(1.0 * (1.0 - 1.0 - 0.3), rel=1e-14)
    assert dx[1] == pytest.approx(1.0 * (0.8 - 0.4 - 1.0), rel=1e-14)


def test_the_per_capita_prefactor_is_present():
    # Dropping the x_i factor turns gLV into an affine system: an absent species
    # would spontaneously appear. It must not.
    p = reference()
    dx = glv_rhs(p)(np.array([0.0, 0.5, 0.5]))
    assert dx[0] == 0.0, "an absent species grew from nothing -- the x_i prefactor is missing"


# --------------------------------------------------------------------------
# Category A, independent: the two-species symmetric closed form
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a", [0.0, 0.25, 0.5, 0.8])
def test_two_species_equilibrium_and_eigenvalues_match_the_scalar_closed_form(a):
    # r = 1, A_ii = -1, A_ij = -a has x* = 1/(1+a) and eig(diag(x*) A) = x*(-1 -+ a)
    # -- algebra done by hand, NOT by inverting A. This is what makes the check
    # independent rather than self-consistent, and it is the deterministic
    # analogue of "verify a tooth at 3-4 seeds": a second configuration, swept.
    p = GLVParams(r=[1.0, 1.0], A=[[-1.0, -a], [-a, -1.0]], x_init=[0.1, 0.1])
    x_star = equilibrium(p)
    expected = 1.0 / (1.0 + a)
    assert x_star == pytest.approx([expected, expected], rel=1e-14)

    eigenvalues = np.sort(np.linalg.eigvals(community_matrix(p)).real)
    assert eigenvalues == pytest.approx(
        sorted([expected * (-1 - a), expected * (-1 + a)]), abs=1e-14
    )


# --------------------------------------------------------------------------
# Category A, self-consistency: linear solve vs time integration
# --------------------------------------------------------------------------


def test_equilibrium_is_a_root_of_the_rhs():
    p = reference()
    x_star = equilibrium(p)
    assert np.abs(textbook_rhs(x_star, p)).max() < 1e-14
    assert x_star == pytest.approx([0.7009569378, 0.4354066986, 0.8421052632], rel=1e-9)


def test_community_matrix_is_the_jacobian_at_the_equilibrium():
    # SIGNED comparison against central differences. The adaptive-dynamics slice
    # found closed forms that matched a finite difference in magnitude and
    # disagreed in sign at every probe -- an |a| vs |b| test would have passed.
    p = reference()
    x_star = equilibrium(p)
    rhs = glv_rhs(p)
    eps = 1e-6
    numerical = np.empty((3, 3))
    for j in range(3):
        up, down = x_star.copy(), x_star.copy()
        up[j] += eps
        down[j] -= eps
        numerical[:, j] = (rhs(up) - rhs(down)) / (2.0 * eps)
    assert np.allclose(community_matrix(p), numerical, rtol=0.0, atol=1e-9)


def test_validate_reproduces_the_equilibrium():
    # Routed through the ValidationSuite (non-negotiable #2). Started AT the
    # equilibrium, so the only residual is integration error -- which Richardson in
    # dt measures. Two replicates, because one gives sem = inf and passes vacuously.
    #
    # Measured: the difference between dt and dt/2 is EXACTLY 0.0 here. That is not
    # a broken probe -- |rhs(x*)| = 1.9e-16 and x* ~ 0.7, so a step moves the state
    # by ~2e-18, below the ULP, and RK4 never leaves. The floor therefore falls back
    # to 1e-13, and what this check turns on is structural: that the linear solve's
    # root really is stationary under the integrated vector field. Precision lives
    # in the attraction test below, which integrates from far away.
    base = {"r": R3, "A": A3, "initial": "equilibrium", "t_max": 20.0, "dt": 0.01}
    bound = float(
        np.abs(final_x(GLVParams(**base)) - final_x(GLVParams(**{**base, "dt": 0.005}))).max()
    )
    experiment = Experiment(
        model="glv",
        params=base,
        replicates=2,
        observables=KEYS3,
        seed=0,
        max_steps=n_glv_steps(GLVParams(**base)) + 10,
    )
    report = validate(experiment, params_factory, z=4.0, sem_floor=max(bound, 1e-13))
    assert report.passed, str(report)
    assert {c.name for c in report.checks} == set(KEYS3)


def test_starting_at_the_equilibrium_does_not_drift():
    _, series = run_glv(reference(initial="equilibrium", t_max=50.0, dt=0.01))
    for key in KEYS3:
        values = np.asarray(series[key])
        assert np.abs(values - values[0]).max() < 1e-13


def test_the_equilibrium_attracts_from_a_displaced_start():
    # The two-code-path check the validate() test above cannot be: this trajectory
    # starts far from x* and is compared against the LINEAR SOLVE's answer, so
    # integration and algebra have to agree without either being handed the other's
    # result. That the root is stationary does not make it an attractor.
    #
    # The tolerance is derived, not typed, exactly as in test_hodgkin_huxley.py: for
    # a residual decaying as A e^{-t/tau}, |x(t_max) - x(t_max/2)| is dominated by
    # r(t_max/2), so r(t_max) ~ still_decaying * exp(-t_max / (2 tau)), with tau
    # taken from the community matrix rather than from a constant.
    p = reference(initial="state", x_init=[0.1, 0.1, 0.1], t_max=60.0, dt=0.01)
    half = reference(initial="state", x_init=[0.1, 0.1, 0.1], t_max=30.0, dt=0.01)
    x_star = equilibrium(p)
    x_full, x_half = final_x(p), final_x(half)

    still_decaying = float(np.abs(x_full - x_half).max())
    residual = float(np.abs(x_full - x_star).max())
    tau_slow = 1.0 / abs(slow_mode(p)[0])
    predicted = still_decaying * float(np.exp(-p.t_max / (2.0 * tau_slow)))

    assert residual <= 10.0 * predicted, (
        f"residual {residual:.3e} exceeds 10x the decay-law prediction {predicted:.3e} "
        f"(still_decaying {still_decaying:.3e}, tau {tau_slow:.4f})"
    )
    # A companion absolute bound so the check cannot pass vacuously if the two
    # horizons ever disagree wildly and inflate `predicted`.
    assert residual < 1e-7, "not converged enough for the claim to mean anything"


# --------------------------------------------------------------------------
# Category B: the relaxation rate as an O(eps) limit
# --------------------------------------------------------------------------

RELAX = {"r": R3, "A": A3, "initial": "relax", "t_max": 10.0, "dt": 0.005}


def measured_rate(eps: float) -> float:
    _, series = run_glv(GLVParams(**{**RELAX, "eps": eps}))
    return float(series["relaxation_rate"][-1])


def test_richardson_in_the_amplitude_predicts_the_relaxation_error():
    # THE instrument, and the sharpest statement in this file. The endpoint
    # log-ratio measures the LINEARIZED decay rate, so its error is O(eps) -- not
    # O(dt), which Richardson in dt would see instead and which is negligible here.
    # First order, so the predicted error at eps is 2|m(eps) - m(eps/2)|; Gray-Scott
    # needed 4/3 because its amplitude error was second order.
    #
    # Planning measured ratios 0.9954 / 0.9977 / 0.9991 / 0.9995 at
    # eps = 1e-2 / 5e-3 / 2e-3 / 1e-3. Note the planning SLICE's constant does not
    # transfer: it fitted log|x - x*| over a window and read 3.03e-4 at eps = 1e-2,
    # where this endpoint estimator reads 4.27e-3. The O(eps) SCALING transferred;
    # the constant did not, which is precisely why the tolerance is derived at
    # runtime instead of copied from the plan.
    lam = slow_mode(GLVParams(**RELAX))[0]
    for eps in (1e-2, 1e-3):
        coarse, fine = measured_rate(eps), measured_rate(eps / 2)
        predicted = 2.0 * abs(coarse - fine)
        actual = abs(coarse - lam)
        assert predicted > 0.0
        ratio = actual / predicted
        assert 0.9 < ratio < 1.1, (
            f"at eps={eps:g} Richardson predicted {predicted:.4e} and the true error "
            f"is {actual:.4e} (ratio {ratio:.4f}); the error is not first order in eps"
        )


def test_the_error_shrinks_linearly_with_the_amplitude():
    # The other half of the O(eps) claim: a ratio near 1 above says Richardson
    # models the error at each eps, this says the error actually goes away.
    lam = slow_mode(GLVParams(**RELAX))[0]
    errors = [abs(measured_rate(eps) - lam) for eps in (1e-2, 1e-3, 1e-4)]
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert 9.0 < coarse / fine < 11.0, f"errors not linear in eps: {errors}"


def test_validate_reproduces_the_slow_eigenvalue():
    # Through the ValidationSuite, with the sem_floor DERIVED by Richardson in the
    # amplitude rather than typed. Two replicates (one gives sem = inf).
    eps = 1e-3
    floor = 2.0 * abs(measured_rate(eps) - measured_rate(eps / 2))
    experiment = Experiment(
        model="glv",
        params={**RELAX, "eps": eps},
        replicates=2,
        observables=("relaxation_rate",),
        seed=0,
        max_steps=n_glv_steps(GLVParams(**RELAX)) + 10,
    )
    report = validate(experiment, params_factory, z=4.0, sem_floor=max(floor, 1e-13))
    assert report.passed, str(report)
    assert [c.name for c in report.checks] == ["relaxation_rate"]


def test_the_seeded_perturbation_lies_along_the_slow_eigenvector():
    p = GLVParams(**{**RELAX, "eps": 1e-3})
    state = MODEL.initial_state(p, np.random.default_rng(0))
    rate, vector = slow_mode(p)
    assert np.allclose(state.x - equilibrium(p), p.eps * vector, rtol=1e-12, atol=0.0)
    # It must be the SLOWEST mode -- seeding the fastest one measures -1.0368 and
    # the run would decay away from the rate it claims to measure.
    assert rate == pytest.approx(np.linalg.eigvals(community_matrix(p)).real.max(), rel=1e-12)


# --------------------------------------------------------------------------
# Category B: integrator order, inside the prescribed window
# --------------------------------------------------------------------------


def test_error_is_fourth_order_on_the_transient():
    # PRESCRIBED WINDOW: t_max = 5, dt in [0.125, 0.03125]. Measured here
    # 16.42 / 16.17 / 16.08 for dt = 0.25 -> 0.03125, converging on 16 from above.
    #
    # Recorded honestly: the plan warns that t_max = 20 reads 65.89/5.76/12.65/15.09
    # -- noise from an error collapsed to roundoff. That did NOT reproduce at this
    # configuration; from x_init = (0.1, 0.1, 0.1) with a max-norm error against a
    # dt = 1e-3 reference, t_max = 20 reads 15.18/15.59/15.80. The trap depends on
    # the initial condition and error norm, neither of which the slice recorded. The
    # t_max = 5 window is used regardless: its errors are three orders of magnitude
    # larger (3.8e-6 vs 2.2e-9) and its ratios sit closer to 16, so it is
    # unambiguously inside the asymptotic regime -- and which grids are inside it is
    # part of the claim (the Phase-2 Laplacian lesson).
    def error(dt: float, ref: np.ndarray) -> float:
        p = reference(initial="state", x_init=[0.1, 0.1, 0.1], t_max=5.0, dt=dt)
        return float(np.abs(final_x(p) - ref).max())

    ref = final_x(reference(initial="state", x_init=[0.1, 0.1, 0.1], t_max=5.0, dt=0.001))
    errors = [error(dt, ref) for dt in (0.125, 0.0625, 0.03125)]
    assert all(e > 0.0 for e in errors)
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        ratio = coarse / fine
        assert 12.0 < ratio < 20.0, f"expected ~16 for RK4, got {ratio:.2f} (errors {errors})"


# --------------------------------------------------------------------------
# Refusing to answer
# --------------------------------------------------------------------------


def test_equilibrium_refuses_an_infeasible_system():
    # Strong asymmetric competition: x* = (-0.05, 0.35). The negative component is
    # not a rare pathology in this phase -- feasibility collapsing with S is the
    # measurement that forced the May reframe.
    p = GLVParams(r=[1.0, 0.2], A=[[-1.0, -3.0], [-3.0, -1.0]], x_init=[0.1, 0.1])
    with pytest.raises(ValueError, match="infeasible"):
        equilibrium(p)
    with pytest.raises(ValueError, match="infeasible"):
        MODEL.analytic_predictions(p)


def test_equilibrium_refuses_a_singular_matrix():
    p = GLVParams(r=[1.0, 1.0], A=[[-1.0, -1.0], [-1.0, -1.0]], x_init=[0.1, 0.1])
    with pytest.raises(ValueError, match="singular"):
        equilibrium(p)


def test_analytic_predictions_refuse_an_unstable_equilibrium():
    # Bistable competition: x* = (0.4, 0.4) is feasible but a saddle,
    # eig = {-1, +0.2}. The attractor is one of the two single-species boundary
    # states, so predicting the interior point would be a wrong number that still
    # looks green -- hodgkin_huxley's stance past the Hopf.
    p = GLVParams(r=[1.0, 1.0], A=[[-1.0, -1.5], [-1.5, -1.0]], x_init=[0.1, 0.1])
    assert equilibrium(p) == pytest.approx([0.4, 0.4], rel=1e-12)
    with pytest.raises(ValueError, match="unstable"):
        MODEL.analytic_predictions(p)


def test_the_relaxation_rate_refuses_a_complex_slow_pair():
    # Predator-prey with weak self-limitation: eig = -0.0767 +- 0.7473j. The
    # equilibrium is stable and perfectly predictable; the RELAXATION RATE is not,
    # because the perturbation spirals as it decays and the endpoint log-ratio
    # depends on where t_max lands in the oscillation. Exactly the discrimination
    # gray_scott makes for a complex dispersion pair -- and the two refusals must
    # not be conflated, which is why the equilibrium claim still goes through.
    stable = GLVParams(r=[1.0, -0.5], A=[[-0.1, -1.0], [1.0, -0.1]], x_init=[0.1, 0.1])
    assert set(MODEL.analytic_predictions(stable)) == set(species_keys(2))
    with pytest.raises(ValueError, match="complex"):
        slow_mode(stable)
    with pytest.raises(ValueError, match="complex"):
        MODEL.analytic_predictions(
            GLVParams(
                r=[1.0, -0.5], A=[[-0.1, -1.0], [1.0, -0.1]], initial="relax", x_init=[0.1, 0.1]
            )
        )


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_observables_cover_the_species_plus_the_summaries():
    obs = MODEL.observables(MODEL.initial_state(reference(), np.random.default_rng(0)))
    assert set(obs) == {*KEYS3, "total_biomass", "relaxation_rate"}
    assert obs["total_biomass"] == pytest.approx(0.3)
    # nan outside a relaxation run -- reporting 0.0 would be inventing a measurement.
    assert np.isnan(obs["relaxation_rate"])


def test_params_normalize_lists_to_nested_tuples():
    # Experiment.params must carry lists (tuples do not survive a JSON round-trip)
    # while the frozen dataclass must hold tuples (a list field is unhashable and
    # aliased into the supposedly immutable params). gLV is the first model in the
    # project with list-valued params, so this contract is new here.
    p = GLVParams(r=[1.0, 2.0], A=[[-1.0, 0.0], [0.0, -1.0]], x_init=[0.5, 0.5])
    assert p.r == (1.0, 2.0)
    assert p.A == ((-1.0, 0.0), (0.0, -1.0))
    assert isinstance(p.x_init, tuple)
    assert p == GLVParams(r=(1.0, 2.0), A=((-1.0, 0.0), (0.0, -1.0)), x_init=(0.5, 0.5))


def test_experiment_round_trips():
    experiment = Experiment(
        model="glv",
        params={"r": R3, "A": A3, "initial": "state", "x_init": [0.1, 0.1, 0.1], "t_max": 5.0},
        replicates=1,
        observables=KEYS3,
        seed=1,
        max_steps=600,
    )
    assert Experiment.from_json(experiment.to_json()) == experiment


def test_rejects_a_mismatched_matrix():
    with pytest.raises(ValueError, match="3x3"):
        GLVParams(r=[1.0, 1.0, 1.0], A=[[-1.0, 0.0], [0.0, -1.0]], x_init=[0.1, 0.1, 0.1])


def test_rejects_a_dt_that_does_not_divide_t_max():
    with pytest.raises(ValueError, match="divide"):
        reference(t_max=1.0, dt=0.3)


def test_the_run_lands_exactly_on_t_max():
    times, _ = run_glv(reference(t_max=5.0, dt=0.01))
    assert times[-1] == pytest.approx(5.0, abs=1e-12)
    assert times.size == 501
