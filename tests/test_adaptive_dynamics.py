"""Adaptive dynamics: the trait-space closed forms, and the ``O(sm)`` limit.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` steps 14 and 16.

**What is anchored to what** — the three-category framing Phase 2 established:

* **Category A, independent.** :func:`longhand_invasion_fitness` below is written
  out from the model definition and shares no code with the model; the selection
  gradient and both second derivatives are checked **signed** against central
  differences of it; and the two second derivatives must sum to ``dD/dx`` by the
  chain rule, an identity that holds at every trait and is independent of the
  finite-difference comparison.
* **Category A, external.** The **recorded literals** — the canonical target
  ``1.849492`` and the three teeth targets ``1.662326 / 1.213061 / 0.000000``.
  The last three are what confirm the *convention*: they are properties of the
  deterministic side alone, so they are independent of ``sigma_a`` and of the
  mutation-step law — the two things nothing in the record could pin. The
  reconstruction was not fitted to them and reproduces all three.
* **Category A, exact.** :func:`run_cohort` and :meth:`AdaptiveDynamics.step` are
  two implementations of one jump process, and the test below asserts they
  consume the generator **draw for draw** — not merely that they agree on a final
  number.
* **Category B.** The headline: the canonical equation's discrepancy vanishes
  **linearly in ``sm``**, asserted as a two-sided band on the log-log slope.
* **Category C — never asserted against a bound.** The interpretation of the
  singular point as a branching point belongs to ``test_trait_branching.py``;
  nothing here reads the literature for a number.

**Why the headline is a slope and not a tolerance.** A ``z``-test at one ``sm``
was measured to tighten onto a real ``O(sm)`` bias: five independent replicate
groups all landed *above* the prediction. Such a check fails a **correct**
implementation as replicates grow, which is ``validate()``'s tightening tolerance
arriving from the wrong direction.

**And why the band is two-sided.** Each tooth is a *wrong canonical equation*,
which makes the discrepancy an ``O(1)`` constant — and a constant fits a
near-perfect line. The three teeth land at slopes of ``+0.018``, ``+0.005`` and
``-0.003`` with standard errors 30-200x *smaller* than the correct point's, so
all three are tens of sigma from zero. **"Significantly nonzero" passes every
one of them.** This is Phase 3b's wrong-``R`` trap verbatim.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import SeedSequence

from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import Experiment, Model, TerminableModel, ValidatableModel
from sandbox.core.recorder import run_replicate
from sandbox.core.registry import get_model
from sandbox.core.rng import make_rng, spawn_rngs
from sandbox.core.validation import validate
from sandbox.models.adaptive_dynamics import (
    CANONICAL_DU,
    CANONICAL_TRAIT,
    MODEL,
    U_TARGET,
    AdaptiveDynamicsParams,
    canonical_rhs,
    canonical_trait,
    carrying_capacity,
    competition,
    gradient_slope,
    invasion_fitness,
    mutant_curvature,
    resident_curvature,
    run_cohort,
    selection_gradient,
    splitting_rate,
)

#: Traits the derivative checks are made at. Spans both signs and includes the
#: singular point, where ``D`` vanishes and a sign error is *invisible* — so the
#: off-centre traits are the ones carrying the signed part of the claim.
PROBE_TRAITS = (-2.0, -1.3, -0.4, 0.0, 0.7, 2.0)

#: The shipped ``O(sm)`` sweep — config B. **These replicate counts are the ones
#: the assertion band was scored against**, at four seed-sets, and tidying them
#: would invalidate the measured 5.7-SE margin. ``R ~ 1945 / (4 sm)``, derived
#: from a pilot (``err ~ 0.17 sm``, ``SD ~ 0.5 sqrt(sm)``) rather than typed;
#: "1200 replicates" is a number from a different estimator and does not travel.
SWEEP: tuple[tuple[float, int], ...] = (
    (0.15, 3_300),
    (0.10, 5_000),
    (0.05, 10_000),
    (0.025, 20_000),
    (0.0125, 40_000),
)

#: Replicate streams per sweep point. Buys a free self-check (group scatter must
#: agree with the pooled within-group SE) and keeps the trajectories independent
#: of ``R``.
N_GROUPS = 5

#: Two-sided band on the log-log slope, placed in a **measured** gap at four
#: seed-sets: the correct slope lands in it 4/4 (``0.9558 / 0.9802 / 1.0113 /
#: 1.0409``) and every tooth 0/4 (worst ``+0.0342``). The lower edge sits 5.7
#: correct-SE below the worst correct seed and 431 tooth-SE above the worst
#: tooth; the upper edge 5.8 correct-SE above the best.
SLOPE_BAND = (0.6, 1.4)


def params_factory(d: dict) -> AdaptiveDynamicsParams:
    return AdaptiveDynamicsParams(**d)


def reference(**overrides) -> AdaptiveDynamicsParams:
    return AdaptiveDynamicsParams(**overrides)


# --------------------------------------------------------------------------
# Hand-written reference -- shares no code with the model
# --------------------------------------------------------------------------


def longhand_invasion_fitness(y: float, x: float, p: AdaptiveDynamicsParams) -> float:
    """``s_x(y) = r (1 - a(y,x) K(x) / K(y))``, built from three separate pieces.

    Deliberately forms ``a``, ``K(x)`` and ``K(y)`` independently and divides,
    rather than combining the exponents the way the model does. A sign error in
    the ``(y^2 - x^2)`` term, a swapped ``sa``/``sK`` or a dropped factor of two
    shows up as a disagreement instead of propagating consistently through both
    sides.
    """
    a = np.exp(-((y - x) ** 2) / (2.0 * p.sigma_a**2))
    k_x = p.k0 * np.exp(-(x**2) / (2.0 * p.sigma_k**2))
    k_y = p.k0 * np.exp(-(y**2) / (2.0 * p.sigma_k**2))
    return float(p.r_growth * (1.0 - a * k_x / k_y))


def refined(estimate, h: float = 1e-3) -> tuple[float, float]:
    """A finite-difference estimate at ``h/2``, **with its own error bound**.

    Every stencil below is second order, so ``estimate(h) - estimate(h/2)`` is
    ``(3/4) C h^2`` and the finer estimate's remaining error is a third of that.
    Returning the pair means the derivative tests compare against a tolerance
    that was *measured on the spot* rather than typed — Richardson in a
    discretization parameter, the Phase-2 stencil instrument. Measured, the
    mixed-derivative stencil's error runs ``6.8e-4 / 6.8e-6 / 6.1e-8`` at
    ``h = 1e-2 / 1e-3 / 1e-4``, confirming the order the bound assumes.
    """
    coarse, fine = estimate(h), estimate(0.5 * h)
    return fine, abs(coarse - fine) / 3.0


def central_difference(f, x: float):
    return refined(lambda h: (f(x + h) - f(x - h)) / (2.0 * h))


def second_difference(f, x: float):
    return refined(lambda h: (f(x + h) - 2.0 * f(x) + f(x - h)) / h**2)


def mixed_difference(f, x: float, y: float):
    """``d2 f / dx dy`` with resident and mutant varied **independently**.

    The stencil that walks the diagonal instead — ``f(x+h, y+h)`` paired with
    ``f(x-h, y-h)`` — is a different and also-correct quantity, ``dD/dx``, and it
    returned exactly ``-1.000000`` where the mixed partial is ``+1.959``. A
    finite-difference check can be green against the wrong derivative.
    """
    return refined(
        lambda h: (
            (f(y + h, x + h) - f(y - h, x + h) - f(y + h, x - h) + f(y - h, x - h)) / (4.0 * h * h)
        )
    )


def target_after(rhs, u: float) -> float:
    """Integrate a canonical-equation *variant* to scaled time ``u`` from ``x0 = 2``.

    Shared by the correct target and all three teeth, so a tooth differs from the
    truth **only** in the equation or the distance — never in the integrator.
    """
    if u == 0.0:
        return 2.0
    _, ys = integrate_rk4(rhs, 2.0, u, CANONICAL_DU)
    return float(ys[-1, 0])


def teeth_targets(p: AdaptiveDynamicsParams) -> dict[str, callable]:
    """The correct canonical target and three wrong ones, as functions of ``sm``.

    Each tooth is a *different equation*, written here rather than obtained by
    calling the model with odd arguments:

    * **drop the 1/2** — ``dx/dt = mu sm^2 K D``, twice the canonical speed, so
      the same horizon travels twice the scaled distance.
    * **omit K(x)** — ``dx/dt = (1/2) mu sm^2 D(x)``, leaving ``dx/du = -x``.
    * **sm for sm^2** — one power of ``sm`` short, so the scaled distance becomes
      ``U / sm`` and *moves with the sweep*.

    Only the last has any ``sm`` dependence at all, which is the point: at a fixed
    canonical distance the correct target is one number for the whole sweep, so a
    wrong equation shows up as a **constant** offset.
    """
    correct = canonical_rhs(p)
    return {
        "correct": lambda _sm: target_after(correct, U_TARGET),
        "drop the 1/2": lambda _sm: target_after(correct, 2.0 * U_TARGET),
        "omit K(x)": lambda _sm: target_after(lambda y: selection_gradient(y, p), U_TARGET),
        "sm for sm^2": lambda sm: target_after(correct, U_TARGET / sm),
    }


def weighted_loglog_slope(
    sm: np.ndarray, err: np.ndarray, sem: np.ndarray
) -> tuple[float, float, float]:
    """Slope of ``log|err|`` on ``log sm``, weighted by each point's **own** error.

    Returns ``(slope, slope_se, chi2_per_dof)``. The ``log`` error is ``sem/|err|``
    by propagation. A residual-only standard error on a handful of points is how
    3c nearly shipped a "4.5-sigma subleading correction" that was 1.1 sigma from
    its predicted value; ``chi2/dof`` is the tell, and it is returned so a test
    can assert the scatter is consistent with the error bars rather than assuming.
    """
    log_sm = np.log(sm)
    log_err = np.log(np.abs(err))
    log_sem = sem / np.abs(err)
    w = 1.0 / log_sem**2

    total = w.sum()
    mean_x = (w * log_sm).sum() / total
    mean_y = (w * log_err).sum() / total
    sxx = (w * (log_sm - mean_x) ** 2).sum()
    slope = (w * (log_sm - mean_x) * (log_err - mean_y)).sum() / sxx
    intercept = mean_y - slope * mean_x

    residual = log_err - (intercept + slope * log_sm)
    chi2 = (w * residual**2).sum() / max(len(log_sm) - 2, 1)
    return float(slope), float(1.0 / np.sqrt(sxx)), float(chi2)


def measure_sweep(seed: int) -> dict[str, np.ndarray]:
    """Run the shipped sweep once and return everything any assertion needs.

    One measurement, many claims. Splitting the slope, the teeth, the signs and
    the saturation into separate test functions would make xdist rebuild the
    whole 2.3 s sweep once per function — 3c's lesson — and, worse, would score
    the teeth against *differently noisy* means than the correct target, turning
    an exact comparison into a noisy one.
    """
    branches = SeedSequence(seed).spawn(len(SWEEP))
    sigma_m, mean, sem, saturation = [], [], [], []

    for (sm, n_rep), branch in zip(SWEEP, branches, strict=True):
        per_group = n_rep // N_GROUPS
        results = [
            run_cohort(reference(sigma_m=sm), rng, per_group)
            for rng in spawn_rngs(branch, N_GROUPS)
        ]
        finals = np.concatenate([r.x_final for r in results])
        events = sum(r.n_events for r in results)

        sigma_m.append(sm)
        mean.append(float(finals.mean()))
        sem.append(float(finals.std(ddof=1)) / np.sqrt(finals.size))
        saturation.append(sum(r.frac_saturated * r.n_events for r in results) / events)

    return {
        "sigma_m": np.array(sigma_m),
        "mean": np.array(mean),
        "sem": np.array(sem),
        "saturation": np.array(saturation),
    }


# --------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"sigma_m": 0.0},
        {"sigma_m": -0.1},
        {"sigma_a": 0.0},
        {"sigma_k": -1.0},
        {"r_growth": 0.0},
        {"k0": -2.0},
        {"mutation_rate": 0.0},
        {"u_target": 0.0},
    ],
)
def test_nonpositive_params_are_rejected(overrides):
    with pytest.raises(ValueError):
        reference(**overrides)


@pytest.mark.parametrize("sigma_m", [0.2, 0.05, 0.0125])
@pytest.mark.parametrize("mutation_rate", [1.0, 2.5])
def test_the_horizon_travels_one_canonical_distance(sigma_m, mutation_rate):
    """``(1/2) mu sm^2 K0 t_max = U`` — the design constant, at every ``sm``.

    This is what makes the sweep a *limit* measurement: the canonical prediction
    is one number for every point, so the discrepancy is the only thing moving.
    A sweep at fixed ``t_max`` would move the target too, testing a strictly
    stronger claim than the phase can afford.
    """
    p = reference(sigma_m=sigma_m, mutation_rate=mutation_rate)
    scaled = 0.5 * p.mutation_rate * p.sigma_m**2 * p.k0 * p.t_max
    assert scaled == pytest.approx(U_TARGET, rel=1e-12)


# --------------------------------------------------------------------------
# Trait-space closed forms, signed against finite differences
# --------------------------------------------------------------------------


@pytest.mark.parametrize("x", PROBE_TRAITS)
@pytest.mark.parametrize("r_growth", [1.0, 2.5])
@pytest.mark.parametrize("k0", [1.0, 0.4])
@pytest.mark.parametrize("sigma_k", [1.0, 1.7])
def test_invasion_fitness_matches_the_longhand_form(x, r_growth, k0, sigma_k):
    """The scale constants are swept, not left at their defaults.

    ``r_growth``, ``k0`` and ``sigma_k`` all default to exactly ``1.0``, where a
    dropped factor, a lost square and a missing multiplication are all invisible.
    "Sweep the constant the probe happens to land on" is this project's most
    repeated lesson — it has caught a conservation test at ``N = 5000``, an
    albedo mutant that was a no-op at ``A_g = 0.5``, and a white/black swap that
    left the answer bit-identical.
    """
    p = reference(r_growth=r_growth, k0=k0, sigma_k=sigma_k)
    for y in (x - 0.3, x, x + 0.45):
        assert float(invasion_fitness(y, x, p)) == pytest.approx(
            longhand_invasion_fitness(y, x, p), abs=1e-14
        )


@pytest.mark.parametrize("x", PROBE_TRAITS)
def test_a_resident_cannot_invade_itself(x):
    """``s_x(x) = 0`` exactly — the one value that is a *definition*, not a limit."""
    p = reference()
    assert float(invasion_fitness(x, x, p)) == 0.0


@pytest.mark.parametrize("x", PROBE_TRAITS)
@pytest.mark.parametrize("r_growth", [1.0, 2.5])
@pytest.mark.parametrize("sigma_k", [1.0, 1.7])
def test_the_selection_gradient_is_the_signed_derivative(x, r_growth, sigma_k):
    """``D(x) = ds_x(y)/dy`` at ``y = x``, sign included.

    Asserted against ``longhand_invasion_fitness``, so the closed form is checked
    against an *independently written* fitness rather than against itself.

    ``r_growth`` and ``sigma_k`` are swept because at their defaults of ``1.0``
    the formula ``-r x / sK^2`` is indistinguishable from ``-x / sK``, from
    ``-x / sK^2``, and from ``-r x / sK``. **Both mutants survived the first
    mutation run** for exactly that reason — the constant the probe happened to
    land on, for the fourth time in this project.
    """
    p = reference(r_growth=r_growth, sigma_k=sigma_k)
    numerical, bound = central_difference(lambda y: longhand_invasion_fitness(y, x, p), x)
    assert abs(float(selection_gradient(x, p)) - numerical) <= max(3.0 * bound, 1e-11)


def test_the_gradient_vanishes_only_at_the_singular_point():
    p = reference()
    assert float(selection_gradient(0.0, p)) == 0.0
    for x in PROBE_TRAITS:
        if x != 0.0:
            assert float(selection_gradient(x, p)) != 0.0
            # ... and it points back toward zero, which is convergence stability.
            assert float(selection_gradient(x, p)) * x < 0.0


@pytest.mark.parametrize("x", PROBE_TRAITS)
def test_the_mutant_curvature_is_the_signed_second_derivative(x):
    p = reference()
    numerical, bound = second_difference(lambda y: longhand_invasion_fitness(y, x, p), x)
    assert abs(float(mutant_curvature(x, p)) - numerical) <= max(3.0 * bound, 1e-9)


@pytest.mark.parametrize("x", PROBE_TRAITS)
def test_the_resident_curvature_is_the_signed_mixed_derivative(x):
    """``d2 s / dx dy`` at ``y = x``, resident and mutant varied independently.

    The natural way to write this stencil walks the diagonal and silently
    measures ``dD/dx`` instead — see :func:`mixed_difference`. It returned
    ``-1.000000`` at every probe, a perfectly correct number for a different
    derivative, and only disagreed with the closed form because the closed form
    was right.
    """
    p = reference()
    numerical, bound = mixed_difference(lambda y, xx: longhand_invasion_fitness(y, xx, p), x, x)
    assert abs(float(resident_curvature(x, p)) - numerical) <= max(3.0 * bound, 1e-9)


@pytest.mark.parametrize("x", PROBE_TRAITS)
@pytest.mark.parametrize("sigma_a", [0.4, 0.7, 1.0, 1.6])
@pytest.mark.parametrize("sigma_k", [0.8, 1.0, 1.5])
def test_the_two_curvatures_sum_to_the_gradient_slope(x, sigma_a, sigma_k):
    """``d2s/dy2 + d2s/dxdy = dD/dx``, exactly, everywhere.

    ``D(x)`` is a derivative taken *along the diagonal* ``y = x``, so
    differentiating it in ``x`` picks up both partials by the chain rule. This
    checks both closed forms at once and is independent of every
    finite-difference comparison above — a sign error in either one breaks it
    even where the finite differences are least sensitive.

    ``sigma_a`` and ``sigma_k`` are swept because the identity must not depend on
    them, and because the reference values ``0.7`` and ``1.0`` are exactly the
    kind of constant a probe can accidentally land on.
    """
    p = reference(sigma_a=sigma_a, sigma_k=sigma_k)
    total = float(mutant_curvature(x, p)) + float(resident_curvature(x, p))
    assert total == pytest.approx(gradient_slope(p), rel=1e-13)


@pytest.mark.parametrize("sigma_a", [0.4, 0.7, 0.95, 0.999, 1.001, 1.05, 1.5, 3.0])
def test_the_splitting_rate_changes_sign_at_sigma_a_equals_sigma_k(sigma_a):
    """Positive iff competition is narrower than the resource distribution.

    The branch/no-branch criterion, stated on the trait-space side. Whether the
    *dynamics* actually honour it is a separate, simulated claim and lives in
    ``test_trait_branching.py`` — a criterion that agrees with itself is not
    evidence.
    """
    p = reference(sigma_a=sigma_a, sigma_k=1.0)
    rate = splitting_rate(p)
    assert (rate > 0.0) == (sigma_a < p.sigma_k)
    assert rate == pytest.approx(float(mutant_curvature(0.0, p)), rel=1e-15)


@pytest.mark.parametrize("k0", [1.0, 0.4, 3.0])
@pytest.mark.parametrize("sigma_k", [1.0, 1.7])
def test_competition_and_capacity_have_the_shapes_the_model_claims(k0, sigma_k):
    p = reference(k0=k0, sigma_k=sigma_k)
    x = np.linspace(-3.0, 3.0, 13)
    a = np.asarray(competition(x[:, None], x[None, :], p))
    assert np.allclose(np.diag(a), 1.0, atol=0.0)  # a(x, x) = 1 exactly
    assert np.array_equal(a, a.T)  # symmetric, bit-for-bit
    k = np.asarray(carrying_capacity(x, p))
    assert k.argmax() == len(x) // 2  # peaks at the singular point
    assert float(carrying_capacity(0.0, p)) == p.k0
    # The width is sigma_k, checked at a trait where the two candidate widths
    # differ -- K(sigma_k) / K(0) = e^{-1/2} whatever the scale.
    assert float(carrying_capacity(sigma_k, p)) / p.k0 == pytest.approx(np.exp(-0.5), rel=1e-14)


# --------------------------------------------------------------------------
# The deterministic limit
# --------------------------------------------------------------------------


def test_the_canonical_target_reproduces_the_recorded_literal():
    """``x(U) = 1.849492`` — a **category-A external** anchor.

    No slice code survived, so this literal is the only thing that can catch a
    convention that was reconstructed wrongly but self-consistently. The teeth
    test below adds three more the reconstruction was not fitted to.
    """
    assert canonical_trait(reference()) == pytest.approx(CANONICAL_TRAIT, abs=1e-11)


@pytest.mark.parametrize("du", [1e-2, 1e-3, 1e-4])
def test_the_canonical_target_does_not_move_with_the_integration_step(du):
    """The step was chosen by measuring where the answer stops moving.

    Guards the cheap default: if a future change makes ``CANONICAL_DU`` matter,
    this fails rather than silently shifting every recorded discrepancy.
    """
    assert canonical_trait(reference(), du=du) == pytest.approx(CANONICAL_TRAIT, abs=1e-10)


def test_the_canonical_equation_is_free_of_sigma_m_and_the_rates():
    """Scaled time removes ``mu``, ``sm`` and ``K0`` together.

    This is *why* ``K0`` and ``mu`` are not separately identifiable from the
    record — and why they did not need to be.
    """
    base = canonical_trait(reference())
    for overrides in (
        {"sigma_m": 0.2},
        {"sigma_m": 0.001},
        {"mutation_rate": 7.0},
        {"k0": 0.3},
    ):
        assert canonical_trait(reference(**overrides)) == pytest.approx(base, rel=1e-12)


def test_the_canonical_flow_points_toward_the_singular_point():
    p = reference()
    rhs = canonical_rhs(p)
    assert float(rhs(np.array([0.0]))[0]) == 0.0
    for x in (-2.0, -0.5, 0.5, 2.0):
        assert float(rhs(np.array([x]))[0]) * x < 0.0


# --------------------------------------------------------------------------
# The two implementations are one process
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_m", [0.15, 0.05, 0.025])
def test_one_replicate_of_the_cohort_is_the_protocol_model_draw_for_draw(sigma_m):
    """The vectorized sweep and ``step`` consume the generator identically.

    Asserted on the **generator state**, not only on the final trait: a
    final-value match can hide a compensating reordering of draws, and the whole
    reason ``run_cohort`` exists is that the sweep cannot afford ``step``.

    The subtle half is the horizon. ``run_cohort`` advances the clock *before*
    testing it, so a replicate carried past ``t_max`` consumes its waiting time
    and nothing else; ``step`` must do the same. Testing terminality first is the
    natural way to write it and would desynchronize the stream on the last event
    of every replicate — invisible in the mean, fatal to this pin.
    """
    p = reference(sigma_m=sigma_m)

    cohort_rng = make_rng(11)
    cohort = run_cohort(p, cohort_rng, 1)

    step_rng = make_rng(11)
    state = MODEL.initial_state(p, step_rng)
    while not MODEL.is_terminal(state):
        state = MODEL.step(state, step_rng)

    assert cohort.x_final[0] == state.x
    assert cohort.n_events == state.n_events
    assert cohort.n_fixed == state.n_fixed
    assert cohort_rng.bit_generator.state == step_rng.bit_generator.state
    # Non-vacuity: at sigma_m = 0.15 from x0 = 2 the horizon is short enough that
    # a replicate can fix nothing at all, and two frozen residents agree trivially.
    if sigma_m <= 0.05:
        assert state.n_fixed > 0


def test_a_cohort_is_not_a_single_replicate_repeated():
    """Replicates must differ — the vectorization is per-replicate, not broadcast.

    A cohort drawing one shared mutation per iteration would reproduce the mean
    and every marginal, and would show up only here.
    """
    result = run_cohort(reference(sigma_m=0.05), make_rng(3), 200)
    assert len(np.unique(result.x_final)) > 100
    assert result.x_final.std(ddof=1) > 0.0


def test_the_cohort_rejects_an_empty_ensemble():
    with pytest.raises(ValueError):
        run_cohort(reference(), make_rng(0), 0)


@pytest.mark.parametrize("sigma_m", [0.1, 0.03])
def test_the_cohort_is_reproducible_from_its_seed(sigma_m):
    p = reference(sigma_m=sigma_m)
    first = run_cohort(p, make_rng(5), 64)
    second = run_cohort(p, make_rng(5), 64)
    assert np.array_equal(first.x_final, second.x_final)
    assert not np.array_equal(run_cohort(p, make_rng(6), 64).x_final, first.x_final)


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_the_model_is_registered_and_satisfies_the_protocol():
    assert get_model("adaptive_dynamics") is MODEL
    assert isinstance(MODEL, Model)
    assert isinstance(MODEL, TerminableModel)


def test_the_model_declines_to_be_validated():
    """No ``analytic_predictions``, deliberately — and the suite must say so.

    The canonical value is a limit, not an identity. A ``validate()`` tolerance
    would tighten onto the real ``O(sm)`` bias and start failing a **correct**
    implementation as replicates grow. ``repressilator`` takes the same stance.
    """
    assert not isinstance(MODEL, ValidatableModel)
    experiment = Experiment(
        model="adaptive_dynamics", params={"sigma_m": 0.1}, replicates=2, seed=0
    )
    with pytest.raises(TypeError, match="analytic_predictions"):
        validate(experiment, params_factory)


def test_the_recorder_sees_a_trajectory_that_ends_at_the_horizon():
    p = reference(sigma_m=0.1)
    traj = run_replicate(MODEL, p, make_rng(0), max_steps=10_000)
    assert traj.terminated
    assert traj.times[-1] >= p.t_max
    assert set(traj.series) == {"x", "fitness_gradient", "n_events", "n_fixed"}
    assert traj.series["x"][0] == p.x_init
    # The gradient observable must track the trait, not sit at its initial value.
    assert traj.series["fitness_gradient"][-1] == pytest.approx(
        float(selection_gradient(traj.series["x"][-1], p)), rel=1e-15
    )


# --------------------------------------------------------------------------
# The headline: the discrepancy is O(sm)
# --------------------------------------------------------------------------


def test_the_canonical_discrepancy_vanishes_linearly_in_the_mutation_step():
    """**The 3e headline.** ``mean - canonical`` scales as ``sm^1``, not ``sm^0``.

    One sweep, four claims, and the teeth scored against the **same measured
    means** — a tooth changes only the *target*, so re-simulating for each would
    add noise to a comparison that is otherwise exact.

    The band is two-sided because "significantly nonzero" cannot fail: a wrong
    canonical equation makes the discrepancy an ``O(1)`` constant, and a constant
    fits a near-perfect line with standard errors 30-200x smaller than the
    correct point's. Every tooth is then tens of sigma from zero. Measured at
    four seed-sets, the correct slope lands in ``[0.6, 1.4]`` 4/4 and every tooth
    0/4, worst tooth ``+0.0342``.
    """
    p = reference()
    measured = measure_sweep(seed=0)
    sigma_m, mean, sem = measured["sigma_m"], measured["mean"], measured["sem"]
    targets = teeth_targets(p)

    correct = mean - np.array([targets["correct"](sm) for sm in sigma_m])
    slope, slope_se, chi2 = weighted_loglog_slope(sigma_m, correct, sem)

    assert SLOPE_BAND[0] <= slope <= SLOPE_BAND[1], (
        f"canonical discrepancy scales as sm^{slope:.4f} +- {slope_se:.4f}, "
        f"outside the band {SLOPE_BAND}"
    )

    # The sign is an independent refutation route: O(sm) predicts a consistent
    # sign, and a flip across the sweep would refute the reading whatever the fit
    # said. Measured: all six discrepancies positive at full statistics.
    assert np.all(correct > 0.0), f"discrepancies changed sign: {correct}"

    # chi2/dof near 1 says the scatter matches the propagated per-point errors.
    # Far below 1 would mean the SEs are overstated -- which is exactly how a
    # residual-only fit manufactures significance.
    assert 0.1 < chi2 < 6.0, f"chi2/dof = {chi2:.2f}"

    # Every point must be resolved, or the fit is being run through noise.
    assert np.all(np.abs(correct) / sem > 4.0)

    # ...and the teeth, on the same means.
    for name in ("drop the 1/2", "omit K(x)", "sm for sm^2"):
        wrong = mean - np.array([targets[name](sm) for sm in sigma_m])
        tooth_slope, tooth_se, _ = weighted_loglog_slope(sigma_m, wrong, sem)
        assert not SLOPE_BAND[0] <= tooth_slope <= SLOPE_BAND[1], (
            f"tooth {name!r} passed: slope {tooth_slope:+.4f} +- {tooth_se:.4f}"
        )
        # The trap itself, asserted rather than described: the tooth is *more*
        # significantly nonzero than the correct point is, so any one-sided
        # "is it nonzero" check would pass it.
        assert abs(tooth_slope) / tooth_se > 4.0
        assert tooth_se < slope_se


def test_the_recorded_teeth_targets_confirm_the_reconstructed_convention():
    """Three recorded numbers the reconstruction was **not** fitted to.

    ``sigma_a`` and the mutation-step law are invisible to everything the plan
    recorded, so the convention was pinned by ``U`` alone. These three targets
    are properties of the *deterministic* side only — independent of both — so
    reproducing them is genuine external confirmation rather than a round trip.

    They also mark the boundary of what may be cited: the slice's ``+0.004``
    offset and its teeth ``z`` values depend on the parts of the estimator that
    could **not** be recovered, and are not asserted anywhere.
    """
    targets = teeth_targets(reference())
    recorded = {
        "drop the 1/2": 1.662326,
        "omit K(x)": 1.213061,
        "sm for sm^2": 0.000000,
    }
    for name, value in recorded.items():
        assert targets[name](0.0125) == pytest.approx(value, abs=1e-6)


def test_the_sweep_stays_inside_the_regime_the_linearization_assumes():
    """Non-vacuity guard on the sweep's leftmost point.

    The canonical equation linearizes ``s_x(x + delta)``, so it is worst where
    the fixation probability is large. ``sm = 0.2`` puts **4.57%** of offered
    mutants above ``s/r = 1/2`` and is the highest-leverage point on the fit;
    ``0.15`` puts ``1.41%`` there at the same cost and centres the slope on
    ``1.00`` rather than ``0.955``. Which probes are inside the regime is part of
    the claim, so it is asserted rather than left to the config.
    """
    measured = measure_sweep(seed=0)
    assert measured["saturation"].max() < 0.02
    # ...and the guard must not be vacuous: saturation has to be *measurable* at
    # the largest step, or this would pass for a cohort that offers no mutants.
    assert measured["saturation"][0] > 0.0
    assert measured["saturation"][0] == measured["saturation"].max()
