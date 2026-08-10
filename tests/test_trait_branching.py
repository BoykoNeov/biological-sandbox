"""Evolutionary branching: the sign change, and how fast the split diverges.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` steps 15 and 17.

**What is anchored to what** — the three-category framing Phase 2 established:

* **Category A, independent.** :func:`longhand_rhs` below writes the gLV vector
  field as a double loop with explicit indices and shares no code with the model,
  so dividing by the wrong species' carrying capacity — which a symmetric
  interaction matrix hides from every self-consistency check — shows up as a
  disagreement. The two-species coexistence state is checked against a linear
  solve of ``A n = K`` as well as against its own closed form.
* **Category A, exact.** ``n_i = 0`` is a per-bin invariant of pure gLV, so the
  158 bins nobody seeded must be **bit-identically** ``0.0`` at the branch time.
  That is the assertion that fails loudly if a mutation/diffusion term is ever
  reintroduced, and it is what makes ``dt = 0.5`` safe on this domain.
* **Category B.** The branch/no-branch **sign change** across ``sa = sK``, and
  the divergence ``t_branch ~ rate^-1``, asserted as the **exponent**. The
  prefactor moves with the detection threshold and the seed amplitude — measured
  here, not assumed — so it is never a bound.
* **Category C — never asserted against a bound.** That the two morphs are the
  structurally degenerate Gaussian pair of Gyllenberg & Meszena is a
  literature-anchored claim about the *continuum* model. What produces the
  ``+-h`` spacing *here* is not that argument at all but the seeding, and the
  tests assert the seeding.

**The absence claim states its horizon in its own name.** "Does not branch" was
first checked to ``t = 20 000`` while the nearest presence claim takes
``149 822`` — an absence asserted 7.5x short of the phenomenon is a statement
about the budget, not a sign change.

**The scaling test drops ``sa = 0.95``, deliberately.** Its branch time alone is
more than the other four combined (``149 822`` against ``121 816``), and the
four-point fit is both cheaper and *closer* to the prediction:
``-1.00072 +- 0.00050`` over a 7.6x rate range against ``-1.00196 +- 0.00070``
over 16.5x. The coarser trait grid was rejected as the cheaper lever for the
opposite reason — it is the *least* converged, with the product spread running
``2.75 / 0.62 / 0.15 %`` at ``h = 0.1 / 0.05 / 0.025``.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from sandbox.core.protocol import (
    Experiment,
    FieldModel,
    Model,
    TerminableModel,
    ValidatableModel,
)
from sandbox.core.recorder import run_replicate
from sandbox.core.registry import get_model
from sandbox.core.rng import make_rng
from sandbox.core.validation import validate
from sandbox.models.adaptive_dynamics import splitting_rate
from sandbox.models.trait_branching import (
    MODEL,
    PREDICTED_KEYS,
    TraitBranchingParams,
    capacities,
    centre_index,
    find_branch_time,
    has_gap,
    initial_abundances,
    interaction_matrix,
    n_branching_steps,
    neighbour_fitness,
    pair_equilibrium,
    pair_indices,
    predicted_product,
    trait_branching_rhs,
    trait_grid,
)

#: The four ``sigma_a`` the divergence exponent is fitted over. ``0.95`` is
#: dropped: its branch time exceeds the other four put together, and the
#: remaining fit is both cheaper and closer to ``-1``.
SCALING_SIGMA_A = (0.60, 0.70, 0.80, 0.90)

#: Two ``sigma_a`` past the criterion. ``1.05`` is the near side — a tooth that
#: narrows the competition kernel makes it branch spuriously at ``t = 19 759`` —
#: and ``1.5`` is the far side, where the same tooth does *not* bite but the
#: stiffest bins live (``lambda dt = -42`` if anything ever seeds them).
NO_BRANCH_SIGMA_A = (1.05, 1.5)

#: Two-sided band on the divergence exponent, placed in a **measured** gap. The
#: correct model reads ``-1.00072 +- 0.00050``; a rate formula missing its
#: ``-1/sK^2`` term reads ``-2.45``, and a competition kernel missing the factor
#: of two in its exponent reads ``-0.55``. Both sit more than 4 tooth-SE outside.
#: One-sided "the exponent is negative" would pass both.
EXPONENT_BAND = (-1.15, -0.85)


def params_factory(d: dict) -> TraitBranchingParams:
    return TraitBranchingParams(**d)


def reference(**overrides) -> TraitBranchingParams:
    return TraitBranchingParams(**overrides)


@cache
def branch_time(params: TraitBranchingParams) -> float | None:
    """Cached branch time. Several tests share the same handful of long runs."""
    return find_branch_time(params).t_refined


def ols_loglog(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope of ``log y`` on ``log x``, with the residual scatter as its error.

    These runs are deterministic, so there is no per-point error to propagate and
    the residual *is* the uncertainty — the one situation in which a residual-only
    standard error is the honest choice rather than the trap it was in 3c.
    """
    log_x, log_y = np.log(x), np.log(y)
    n = len(log_x)
    sxx = ((log_x - log_x.mean()) ** 2).sum()
    slope = ((log_x - log_x.mean()) * (log_y - log_y.mean())).sum() / sxx
    residual = log_y - (log_y.mean() + slope * (log_x - log_x.mean()))
    return float(slope), float(np.sqrt((residual**2).sum() / (n - 2) / sxx))


def scaling_fit(
    sigma_a_values: tuple[float, ...] = SCALING_SIGMA_A, **overrides
) -> tuple[float, float, np.ndarray]:
    """``(exponent, se, products)`` of the divergence law at one configuration."""
    rates, times = [], []
    for sigma_a in sigma_a_values:
        p = reference(sigma_a=sigma_a, t_max=120_000.0, **overrides)
        t_branch = branch_time(p)
        assert t_branch is not None, f"sigma_a = {sigma_a} did not branch"
        rates.append(splitting_rate(p.trait_params))
        times.append(t_branch)
    rates, times = np.array(rates), np.array(times)
    slope, se = ols_loglog(rates, times)
    return slope, se, rates * times


# --------------------------------------------------------------------------
# Hand-written reference -- shares no code with the model
# --------------------------------------------------------------------------


def longhand_rhs(n: np.ndarray, p: TraitBranchingParams) -> np.ndarray:
    """``dn_i/dt = r n_i (1 - sum_j a(x_i,x_j) n_j / K(x_i))``, as a double loop.

    Written with explicit indices because the matrix form hides the one error a
    symmetric kernel makes invisible: dividing by ``K_j`` instead of ``K_i``.
    Transposing ``A`` — 3a's mutant that survived ``validate()`` — cannot be
    caught here either, since this kernel is symmetric by construction; the test
    that catches *that* is the one asserting the symmetry directly.
    """
    x = np.linspace(-p.half_width, p.half_width, p.n_grid)
    out = np.zeros(p.n_grid)
    for i in range(p.n_grid):
        load = 0.0
        for j in range(p.n_grid):
            load += np.exp(-((x[i] - x[j]) ** 2) / (2.0 * p.sigma_a**2)) * n[j]
        k_i = p.k0 * np.exp(-(x[i] ** 2) / (2.0 * p.sigma_k**2))
        out[i] = p.r_growth * n[i] * (1.0 - load / k_i)
    return out


# --------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"sigma_a": 0.0},
        {"sigma_k": -1.0},
        {"r_growth": 0.0},
        {"dt": 0.0},
        {"t_max": -5.0},
        {"half_width": 0.0},
        {"n_grid": 160},  # even: no centre bin
        {"n_grid": 1},
        {"initial": "branching?"},
        {"check_interval": 0},
        {"seed_amplitude": 1e-2},  # already above the detection threshold
        {"seed_amplitude": 0.0},
        {"pair_traits": (0.5,)},
        {"pair_init": (0.0, 0.2)},
    ],
)
def test_bad_params_are_rejected(overrides):
    with pytest.raises(ValueError):
        reference(**overrides)


@pytest.mark.parametrize("n_grid", [81, 161, 321])
def test_the_grid_spacing_is_the_only_way_n_grid_enters(n_grid):
    p = reference(n_grid=n_grid)
    x = trait_grid(p)
    assert p.spacing == pytest.approx(2.0 * p.half_width / (n_grid - 1), rel=1e-15)
    assert np.diff(x) == pytest.approx(p.spacing, rel=1e-12)
    assert x[centre_index(p)] == 0.0  # the singular point is a grid point exactly


# --------------------------------------------------------------------------
# The vector field
# --------------------------------------------------------------------------


def test_the_interaction_matrix_is_symmetric_with_a_unit_diagonal():
    p = reference(n_grid=41)
    a = interaction_matrix(p)
    assert np.array_equal(a, a.T)  # bit-for-bit, not merely close
    assert np.array_equal(np.diag(a), np.ones(p.n_grid))
    k = capacities(p)
    assert k.argmax() == centre_index(p)
    assert k[centre_index(p)] == p.k0


@pytest.mark.parametrize("n_grid", [9, 41, 161])
def test_the_gaussian_kernel_is_positive_definite_but_numerically_singular(n_grid):
    """Positive definite in exact arithmetic — and at the roundoff floor on any
    grid fine enough to matter.

    A Gaussian competition kernel is PD, which is the premise of the
    Gyllenberg-Meszena degeneracy argument (category C, continuum model). On the
    grid the smallest eigenvalue collapses to roundoff as soon as neighbouring
    bins are similar: ``-4.9e-16`` at ``n_grid = 41`` against a largest eigenvalue
    of ``8.5``. So "positive definite" is asserted the only way it can honestly
    be — as *not meaningfully negative*, against a floor scaled by the matrix's
    own size and largest eigenvalue.

    Worth being explicit that this near-singularity is not a numerical nuisance
    to be tolerated: it **is** the structural degeneracy the literature argument
    is about, showing up in the arithmetic.
    """
    a = interaction_matrix(reference(n_grid=n_grid))
    eigenvalues = np.linalg.eigvalsh(a)
    floor = n_grid * np.finfo(float).eps * eigenvalues.max()
    assert eigenvalues.min() > -floor
    if n_grid == 9:  # coarse enough that PD is still resolvable
        assert eigenvalues.min() > 1e-6


def test_the_rhs_matches_the_longhand_double_loop():
    p = reference(n_grid=41)
    rhs = trait_branching_rhs(p)
    rng = make_rng(0)
    for _ in range(3):
        n = rng.random(p.n_grid) * 0.4
        assert rhs(n) == pytest.approx(longhand_rhs(n, p), rel=1e-12, abs=1e-14)


def test_an_empty_bin_stays_empty_under_the_vector_field():
    """``n_i = 0`` is a per-bin invariant — the property the whole formulation
    rests on. It is what lets ``dt = 0.5`` run on a domain whose outer bins would
    otherwise have local eigenvalues near ``-84``."""
    p = reference(n_grid=41)
    rhs = trait_branching_rhs(p)
    n = np.zeros(p.n_grid)
    n[20] = 0.7
    assert np.count_nonzero(rhs(n)) == 1


# --------------------------------------------------------------------------
# The exactly solvable case
# --------------------------------------------------------------------------


def test_the_pair_equilibrium_solves_the_two_species_system():
    """Closed form against a linear solve of ``A n = K`` on the live sub-system."""
    p = reference(initial="pair")
    i, j = pair_indices(p)
    a = interaction_matrix(p)[np.ix_([i, j], [i, j])]
    solved = np.linalg.solve(a, capacities(p)[[i, j]])
    assert pair_equilibrium(p) == pytest.approx(solved, rel=1e-13)


def test_the_pair_traits_are_asymmetric_enough_to_discriminate():
    """Non-vacuity guard on the anchor itself.

    At symmetric traits both components collapse to ``K / (1 + alpha)``, and a
    wrong ``K(x)`` becomes indistinguishable from a wrong ``a(x, y)`` — 3a's
    lesson that a self-consistent system is not an independent check. The shipped
    traits must therefore give *unequal* abundances by a wide margin.
    """
    n_star = pair_equilibrium(reference(initial="pair"))
    assert n_star.min() > 0.0
    assert n_star.max() / n_star.min() > 1.5


def test_the_pair_case_converges_to_its_closed_form():
    p = reference(initial="pair", t_max=50.0, dt=0.1)
    i, j = pair_indices(p)
    traj_state = MODEL.initial_state(p, make_rng(0))
    while not MODEL.is_terminal(traj_state):
        traj_state = MODEL.step(traj_state, make_rng(0))
    n = traj_state.n
    assert n[[i, j]] == pytest.approx(pair_equilibrium(p), abs=1e-13)
    # ...and nothing else ever became populated.
    assert np.count_nonzero(n) == 2


def test_validate_reproduces_the_pair_equilibrium():
    """``validate()`` end to end: the closed form against an integrated run.

    Deterministic, so the statistical standard error degenerates to zero and a
    numerical ``sem_floor`` has to be supplied — derived here from a step
    refinement rather than typed, and two replicates are used because one gives
    ``sem = inf`` and passes vacuously.
    """
    base = {"initial": "pair", "t_max": 50.0, "dt": 0.1}

    def finals(dt: float) -> dict[str, float]:
        p = TraitBranchingParams(**{**base, "dt": dt})
        traj = run_replicate(MODEL, p, make_rng(0), max_steps=n_branching_steps(p) + 5)
        return {k: v[-1] for k, v in traj.series.items()}

    coarse, fine = finals(0.1), finals(0.05)
    bound = max(abs(coarse[k] - fine[k]) for k in PREDICTED_KEYS)

    experiment = Experiment(
        model="trait_branching",
        params=base,
        replicates=2,
        observables=PREDICTED_KEYS,
        seed=0,
        max_steps=n_branching_steps(TraitBranchingParams(**base)) + 5,
    )
    report = validate(experiment, params_factory, z=4.0, sem_floor=max(bound, 1e-13))
    assert report.passed, str(report)
    assert {c.name for c in report.checks} == set(PREDICTED_KEYS)


def test_the_model_refuses_to_predict_the_branching_outcome():
    """Where a split population settles is a sign change and an exponent, not a
    closed form. Returning the seeded state's arithmetic would be a wrong number
    that still looks green — Phase 2's Gray-Scott error."""
    with pytest.raises(ValueError, match="no closed-form prediction"):
        MODEL.analytic_predictions(reference(initial="branching"))


def test_the_pair_equilibrium_refuses_when_a_species_is_excluded():
    """Wide competition drives one component non-positive: the attractor is a
    single-species boundary state, so the interior point is not what a long run
    converges to."""
    with pytest.raises(ValueError, match="no coexistence equilibrium"):
        pair_equilibrium(reference(initial="pair", sigma_a=6.0))


@pytest.mark.parametrize("n_grid", [81, 321])
def test_the_prefactor_law_refuses_off_the_grid_its_offset_was_fitted_at(n_grid):
    """Half of :func:`predicted_product` is derived and half is a fitted constant.

    The ``2/h^2`` slope comes from the mechanism and transfers; the offset was
    fitted at ``h = 0.05`` and does not. Evaluating it at another spacing would
    return a number wrong in a way nothing downstream could detect — so it raises,
    the stance ``hodgkin_huxley`` takes past the Hopf and this model takes on a
    branching run. Nothing in the suite or the demo calls it off-default, which is
    exactly what makes it worth closing: a landmine, not a live bug.
    """
    with pytest.raises(ValueError, match="was fitted at"):
        predicted_product(reference(n_grid=n_grid))
    assert predicted_product(reference()) > 0.0  # ...and the fitted grid still works


def test_two_traits_that_snap_to_one_bin_are_rejected():
    with pytest.raises(ValueError, match="same bin"):
        # h = 0.2 at n_grid = 41, so both of these snap to the bin at x = 0.4.
        pair_indices(reference(initial="pair", pair_traits=(0.31, 0.34), n_grid=41))


# --------------------------------------------------------------------------
# The gap criterion
# --------------------------------------------------------------------------


def test_the_gap_criterion_counts_separated_clusters_not_peaks():
    """Counting local maxima reported "2 peaks = branching" for every
    ``sa < sK`` in the slice, because the peaks were the seed's own immediate
    neighbours. Replacing it moved ``sa = 0.7`` from ``t = 4000`` to ``15 455``."""
    threshold = 1e-3
    assert not has_gap(np.zeros(9), threshold)
    assert not has_gap(np.array([0, 0, 1.0, 1.0, 1.0, 0, 0, 0, 0]), threshold)
    # A dip that never goes below threshold is one cluster, not two.
    assert not has_gap(np.array([0, 1.0, 0.5, 0.9, 0, 0, 0, 0, 0]), threshold)
    assert has_gap(np.array([0, 1.0, 0, 0, 1.0, 0, 0, 0, 0]), threshold)
    # Clusters touching the domain edges still count.
    assert has_gap(np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 1.0]), threshold)


def test_refinement_locates_the_branch_more_precisely_than_the_checkpoint():
    """Checkpoint-and-refine buys ``+-dt/2`` for ~``check_interval`` extra steps.

    It also *changes the answer*: coarse detection rounds up, and that inflation
    is largest where ``t_branch`` is smallest — which is where the true product is
    lowest — so quantization partly cancels the real drift and made the products
    look **more** constant than they are (``0.381%`` at quantum 100 against
    ``0.622%`` true). A tolerance read off the coarse measurement would have been
    too tight and would have failed a correct model.
    """
    p = reference(sigma_a=0.6, t_max=120_000.0)
    result = find_branch_time(p)
    quantum = p.check_interval * p.dt
    assert result.branched
    assert result.t_coarse % quantum == 0.0
    assert result.t_coarse - result.t_refined < quantum
    assert result.refine_steps < p.check_interval

    # Refinement must have *moved* the answer, not merely been available. A
    # mutant that returned the coarse checkpoint unchanged passed the three
    # assertions above -- `t_refined == t_coarse` satisfies "within one quantum"
    # and "fewer than check_interval steps" perfectly. A threshold nothing can
    # fail is not a check, and this one was caught by mutation testing rather
    # than by reading. At sigma_a = 0.6 the gap opens 47 time units before the
    # checkpoint, so the refined answer is strictly inside the interval and off
    # the coarse grid; a branch that happened to open exactly on a checkpoint
    # would legitimately give t_refined == t_coarse, which is why this is
    # asserted at a measured sigma_a rather than swept.
    assert result.t_refined < result.t_coarse
    assert result.refine_steps > 0
    assert result.t_refined % quantum != 0.0


# --------------------------------------------------------------------------
# The structure the branching produces
# --------------------------------------------------------------------------


def test_only_the_seeded_bins_ever_carry_mass():
    """158 of 161 bins are **bit-identically** ``0.0`` at the branch time.

    So the two morphs sitting one grid spacing either side of the seed are not a
    resolution artifact to be apologized for: those are the only bins that ever
    existed. **This is a 3-species gLV in which ``n_grid`` enters only through
    ``h``** — a stronger statement than "grid-dependent", and the assertion fails
    loudly the moment a mutation or diffusion term is reintroduced.
    """
    p = reference(sigma_a=0.6, t_max=120_000.0)
    result = find_branch_time(p)
    assert result.branched
    mid = centre_index(p)
    seeded = {mid - 1, mid, mid + 1}
    untouched = [i for i in range(p.n_grid) if i not in seeded]
    assert np.array_equal(result.n_final[untouched], np.zeros(len(untouched)))
    assert np.count_nonzero(result.n_final) == 3


@pytest.mark.parametrize("n_grid", [81, 161])
def test_the_morphs_sit_one_grid_spacing_either_side_of_the_seed(n_grid):
    p = reference(sigma_a=0.6, n_grid=n_grid, t_max=120_000.0)
    result = find_branch_time(p)
    assert result.branched
    x = trait_grid(p)
    alive = x[result.n_final > p.threshold]
    assert alive == pytest.approx([-p.spacing, p.spacing], rel=1e-12)


def test_branching_does_not_depend_on_the_centre_starting_at_its_capacity():
    """The split is driven by the neighbours' invasion fitness, not by the
    resident's exact abundance — so starting the centre 20% low must not change
    the outcome, and barely moves the timing."""
    base = branch_time(reference(sigma_a=0.6, t_max=120_000.0))
    perturbed = branch_time(reference(sigma_a=0.6, t_max=120_000.0, centre_fraction=0.8))
    assert perturbed is not None
    assert perturbed == pytest.approx(base, rel=0.02)


# --------------------------------------------------------------------------
# The sign change
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_a", SCALING_SIGMA_A)
def test_a_narrow_competition_kernel_branches(sigma_a):
    p = reference(sigma_a=sigma_a, t_max=120_000.0)
    assert splitting_rate(p.trait_params) > 0.0
    assert branch_time(p) is not None


@pytest.mark.parametrize("sigma_a", NO_BRANCH_SIGMA_A)
def test_a_wide_competition_kernel_does_not_branch_by_t_200000(sigma_a):
    """The absence half of the sign change, at a horizon **past every branch time
    this model exhibits** (``sa = 0.95`` takes ``149 822``).

    The horizon is in the test's name because an absence claim is only as strong
    as its budget: the same check at ``t = 20 000`` was 7.5x short and would have
    been a statement about patience rather than about ``sa``. Also asserted: the
    run stays finite and the decayed bins reach **exactly** ``+0.0``, since a
    diffusion term would both destroy that and make ``dt = 0.5`` unstable here.
    """
    p = reference(sigma_a=sigma_a)
    assert splitting_rate(p.trait_params) < 0.0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = find_branch_time(p)
    assert not result.branched
    assert np.isfinite(result.n_final).all()
    mid = centre_index(p)
    assert result.n_final[mid] == pytest.approx(p.k0, abs=1e-9)
    # The neighbours decayed rather than merely failing to reach the threshold --
    # and by exactly as much as their (negative) invasion fitness says they should.
    # Against a resident pinned at K0 the neighbour's per-capita rate IS
    # `neighbour_fitness`, so `seed * exp(s_0 h) ` is a prediction, not a bound: it
    # lands within 0.2% over decay factors of 1e-10 and 1e-60. Asymptotic
    # extinction means they stay strictly nonzero, so the seeded three are still
    # the only three bins that are.
    decay = float(np.log(result.n_final[mid + 1] / p.seed_amplitude))
    assert decay == pytest.approx(neighbour_fitness(p) * p.t_max, rel=0.02)
    assert result.n_final[mid - 1] == result.n_final[mid + 1]  # symmetric, exactly
    assert np.count_nonzero(result.n_final) == 3
    assert result.n_final.min() == 0.0


@pytest.mark.parametrize("sigma_a", SCALING_SIGMA_A + NO_BRANCH_SIGMA_A)
def test_the_neighbour_fitness_departs_from_its_limit_by_exactly_the_next_order(sigma_a):
    """The grid's actual invasion fitness at ``+-h``, against its small-``h``
    limit ``rate h^2 / 2``.

    Keeping both is what lets the residual drift in ``t_branch * rate`` be
    *attributed* to the next order in ``h^2`` rather than written off as scatter.
    So the gap is not merely bounded, it is **predicted**: expanding
    ``1 - e^{-z}`` at ``z = rate h^2 / 2`` gives a relative departure of exactly
    ``-z/2 = -rate h^2 / 4``, which is ``-1.111e-3`` at ``sigma_a = 0.6`` and
    matches to three digits. A loose ``rel=1e-3`` tolerance was tried first and
    failed by 11% — the departure is real, and worth predicting rather than
    tolerating.
    """
    p = reference(sigma_a=sigma_a)
    rate = splitting_rate(p.trait_params)
    limit = rate * p.spacing**2 / 2.0
    exact = neighbour_fitness(p)
    assert np.sign(exact) == np.sign(rate)
    assert (exact - limit) / limit == pytest.approx(-rate * p.spacing**2 / 4.0, rel=0.05)


# --------------------------------------------------------------------------
# The divergence law
# --------------------------------------------------------------------------


def test_the_branch_time_diverges_as_the_reciprocal_of_the_splitting_rate():
    """**The 3e branching headline.** ``t_branch * rate`` is constant, asserted as
    the exponent.

    Two-sided for the reason every band in this phase is: a wrong rate formula
    reads ``-2.45`` and a competition kernel missing the factor of two in its
    exponent reads ``-0.55``, and "the exponent is significantly negative" would
    pass both. Measured correct value ``-1.00072 +- 0.00050``, which is 300
    standard errors inside either edge of ``[-1.15, -0.85]``.
    """
    exponent, se, products = scaling_fit()
    assert EXPONENT_BAND[0] <= exponent <= EXPONENT_BAND[1], (
        f"t_branch scales as rate^{exponent:.5f} +- {se:.5f}, outside {EXPONENT_BAND}"
    )
    # Non-vacuity: the fit must span a real range in rate, or a flat line through
    # four nearly identical points would satisfy any exponent.
    rates = np.array([splitting_rate(reference(sigma_a=s).trait_params) for s in SCALING_SIGMA_A])
    assert rates.max() / rates.min() > 5.0
    # The product's residual spread is O(h^2) and is NOT asserted as a tolerance;
    # this only pins it to the order of magnitude the h-refinement measured.
    assert (products.max() - products.min()) / products.min() < 0.02


def test_the_threshold_and_the_seed_move_the_prefactor_but_not_the_exponent():
    """The actual justification for asserting the exponent and not the constant.

    Measured rather than asserted by fiat: raising the detection threshold or
    lowering the seed amplitude shifts the product by up to 23% while the
    exponent stays at ``-0.9996`` to four decimals. The prefactor law
    :func:`predicted_product` tracks the shift with its **slope fixed** by the
    mechanism and only its offset fitted — a good predictor, and explicitly not a
    closed form, since the centre bin's fall is underived and leaves ``1.39``
    unexplained in the ``h -> 0`` limit.
    """
    # Three sigma_a rather than four: sigma_a = 0.9 costs more than the other
    # three together and this is a *comparison* between configurations, so the
    # long lever arm buys nothing that the shared baseline does not already have.
    # 21.7 s -> 9.5 s, and the exponents agree to four decimals either way.
    cheap = SCALING_SIGMA_A[:3]
    baseline, _, base_products = scaling_fit(cheap)
    for overrides in ({"threshold": 1e-4}, {"seed_amplitude": 1e-8}):
        exponent, _, products = scaling_fit(cheap, **overrides)
        assert EXPONENT_BAND[0] <= exponent <= EXPONENT_BAND[1]
        assert exponent == pytest.approx(baseline, abs=0.01)
        # The prefactor genuinely moved -- otherwise this test proves nothing.
        assert products.mean() > 1.05 * base_products.mean()
        # ...and the fitted law tracks where it moved to.
        assert products.mean() == pytest.approx(predicted_product(reference(**overrides)), rel=0.02)


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_the_model_is_registered_and_satisfies_the_protocols():
    assert get_model("trait_branching") is MODEL
    assert isinstance(MODEL, Model)
    assert isinstance(MODEL, TerminableModel)
    assert isinstance(MODEL, ValidatableModel)
    assert isinstance(MODEL, FieldModel)


def test_observables_summarize_a_whole_trait_distribution():
    p = reference(initial="pair", t_max=10.0, dt=0.1)
    state = MODEL.initial_state(p, make_rng(0))
    obs = MODEL.observables(state)
    i, j = pair_indices(p)
    x = trait_grid(p)
    n = initial_abundances(p)
    assert obs["total"] == pytest.approx(n.sum(), rel=1e-15)
    assert obs["mean_trait"] == pytest.approx((x[i] * n[i] + x[j] * n[j]) / n.sum())
    assert obs["n_clusters"] == 2.0
    assert MODEL.fields(state)["abundance"] is state.n


def test_the_mean_trait_is_nan_on_an_empty_grid():
    """Rather than a silent ``0.0``, which would read as "the mean trait is the
    singular point" — the one value this model spends its whole time near."""
    p = reference(initial="pair")
    state = MODEL.initial_state(p, make_rng(0))
    empty = type(state)(n=np.zeros(p.n_grid), step_index=0, t=0.0, params=p, rhs=state.rhs)
    assert np.isnan(MODEL.observables(empty)["mean_trait"])
