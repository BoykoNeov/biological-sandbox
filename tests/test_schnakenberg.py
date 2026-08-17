"""Schnakenberg: the onset closed forms, the seeded-mode rate, and selection.

Three claims, tested at three sharpnesses, and the third one is why the model
exists — the *wavelength* half of HANDOFF §5's "validate pattern wavelength against
linear stability analysis", which Phase 2 could not deliver.

Every number asserted here was measured first through this code, at the step size
this code ships with, and recorded in
``docs/plans/phase2c-schnakenberg-measurement.md``. Two of the measurement
document's own conclusions did not survive being re-taken through the shipped step
size, which is noted where it matters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment, FieldModel, TerminableModel, ValidatableModel
from sandbox.core.registry import get_model, register
from sandbox.core.selection import selection_report
from sandbox.core.sweep import run_experiment
from sandbox.core.validation import validate
from sandbox.models.schnakenberg import (
    MODEL,
    Schnakenberg,
    SchnakenbergParams,
    continuum_dispersion,
    critical_ratio,
    critical_wavenumber,
    dispersion,
    dominant_mode,
    fastest_mode,
    homogeneous_state,
    is_diffusionless_stable,
    mode_template,
    n_steps,
    peak_power_fraction,
    power_by_mode,
    reaction_determinant,
    reaction_jacobian,
    unstable_band,
)

# The parameter point every dynamical number below sits at. `dv` is 1.2 x the
# closed-form onset ratio: near enough to onset that the bifurcation is
# supercritical (measured), far enough that the band holds 13-17 integer modes so
# "the fastest mode won" is not a statement about the only mode that could.
BASE = dict(a=0.05, b=1.0, du=1.0e-3, length=8.0, ndim=1, initial="noise", noise=1.0e-3)

# The two shipped grids. COARSE is where the stencil and the continuum disagree by
# 2.27 modes about which mode is fastest, so selection can tell them apart; FINE is
# where they agree to 0.35 modes and it cannot. Both step sizes are ~0.36 of the CFL
# limit and divide t_max exactly.
COARSE = {**BASE, "n": 112, "dt": 0.1, "t_max": 300.0}
FINE = {**BASE, "n": 160, "dt": 0.05, "t_max": 300.0}

# R=32 at 4 seed-sets gave a weakest margin of 6.68 SE against z=4 (1.67x of
# headroom). R=16 was REJECTED for being seed-lucky: it cleared the band-centre
# competitor by 0.11 SE on one seed-set of four.
COARSE_REPLICATES = 32
FINE_REPLICATES = 8

PARAM_POINTS = [(0.05, 1.0), (0.1, 0.9), (0.02, 1.4), (0.2, 0.5)]


def factory(d: dict) -> SchnakenbergParams:
    return SchnakenbergParams(**d)


def params_at(**overrides) -> SchnakenbergParams:
    return SchnakenbergParams(**{**COARSE, **overrides})


def continuum_fastest_mode(params: SchnakenbergParams, *, continuous: bool = False) -> float:
    """The same argmax as :func:`fastest_mode`, on the continuum operator.

    Written here rather than shipped in the model: it is the hypothesis the model's
    prediction must beat, not a prediction the model makes.
    """
    js = np.arange(1, params.n // 2 + 1)
    best = float(js[int(np.argmax([continuum_dispersion(params, int(j)) for j in js]))])
    if not continuous:
        return best
    lo, hi = best - 1.5, best + 1.5
    for _ in range(4):
        grid = np.linspace(lo, hi, 2001)
        k = int(np.argmax([continuum_dispersion(params, float(j)) for j in grid]))
        span = (hi - lo) / 2000.0
        lo, hi = grid[k] - 2.0 * span, grid[k] + 2.0 * span
    return 0.5 * (lo + hi)


def competing_hypotheses(params: SchnakenbergParams) -> tuple[dict[str, float], tuple[int, int]]:
    """The alternatives the prediction has to beat, and the unstable band."""
    band = unstable_band(params)
    return (
        {
            "continuum operator": continuum_fastest_mode(params, continuous=True),
            "band centre": 0.5 * float(band[0] + band[-1]),
            "lowest unstable": float(band[0]),
            "highest unstable": float(band[-1]),
        },
        (int(band[0]), int(band[-1])),
    )


def run_selection(config: dict, replicates: int, seed: int = 0):
    params = factory(config)
    competitors, band = competing_hypotheses(params)
    lam, _, _ = dispersion(params, int(fastest_mode(params)))
    return selection_report(
        "schnakenberg",
        config,
        factory,
        predicted_mode=fastest_mode(params, continuous=True),
        competitors=competitors,
        band=band,
        replicates=replicates,
        seed=seed,
        z=4.0,
        max_steps=n_steps(params) + 10,
        # Ends only. The Recorder always keeps the first and last states, and an FFT
        # per intermediate step would be pure waste for a report that reads finals.
        record_every=n_steps(params),
        efolds=lam * params.t_max,
    )


def run_selection_with_band(config: dict, replicates: int, band: tuple[int, int]):
    """As :func:`run_selection`, but with the band supplied — so it can be wrong."""
    params = factory(config)
    competitors, _ = competing_hypotheses(params)
    lam, _, _ = dispersion(params, int(fastest_mode(params)))
    return selection_report(
        "schnakenberg",
        config,
        factory,
        predicted_mode=fastest_mode(params, continuous=True),
        competitors=competitors,
        band=band,
        replicates=replicates,
        seed=0,
        z=4.0,
        max_steps=n_steps(params) + 10,
        record_every=n_steps(params),
        efolds=lam * params.t_max,
    )


@pytest.fixture(scope="module")
def coarse_report():
    """The headline measurement. ~14 s, so it is shared across every test using it."""
    return run_selection(COARSE, COARSE_REPLICATES)


@pytest.fixture(scope="module")
def fine_report():
    """The same claim where the two operators agree — and it stops discriminating."""
    return run_selection(FINE, FINE_REPLICATES)


# --------------------------------------------------------------- the closed forms


@pytest.mark.parametrize(("a", "b"), PARAM_POINTS)
def test_homogeneous_state_is_a_fixed_point_of_the_reaction(a: float, b: float) -> None:
    """``u* = a+b``, ``v* = b/(a+b)^2``, checked by substitution.

    Unlike Gray-Scott's, this state needs no discriminant: it exists for every
    positive ``a, b``, which is exactly why Schnakenberg can carry a selection claim
    where Gray-Scott's famous parameters cannot carry any Turing claim at all.
    """
    params = params_at(a=a, b=b)
    u_star, v_star = homogeneous_state(params)
    reaction = np.array([a - u_star + u_star**2 * v_star, b - u_star**2 * v_star])
    assert np.abs(reaction).max() < 1e-15


@pytest.mark.parametrize(("a", "b"), PARAM_POINTS)
def test_reaction_jacobian_matches_central_differences(a: float, b: float) -> None:
    """The hand-written Jacobian against a finite difference of the RHS itself."""
    params = params_at(a=a, b=b)
    y0 = np.array(homogeneous_state(params))

    def reaction(y: np.ndarray) -> np.ndarray:
        u, v = y
        return np.array([a - u + u * u * v, b - u * u * v])

    eps = 1e-6
    numerical = np.zeros((2, 2))
    for column in range(2):
        step = np.zeros(2)
        step[column] = eps
        numerical[:, column] = (reaction(y0 + step) - reaction(y0 - step)) / (2.0 * eps)

    analytic = reaction_jacobian(*homogeneous_state(params))
    assert np.abs(analytic - numerical).max() < 1e-8


@pytest.mark.parametrize(("a", "b"), PARAM_POINTS)
def test_reaction_determinant_is_exactly_u_star_squared(a: float, b: float) -> None:
    """``det J = u*^2`` identically — and it is why ``d_c`` has a closed form.

    ``reaction_determinant`` computes it *from the Jacobian entries*, so this is a
    check on the identity rather than a restatement of the source.
    """
    params = params_at(a=a, b=b)
    u_star, _ = homogeneous_state(params)
    assert reaction_determinant(params) == pytest.approx(u_star**2, abs=1e-15)


@pytest.mark.parametrize(("a", "b"), PARAM_POINTS[:3])
def test_critical_ratio_matches_a_bisection(a: float, b: float) -> None:
    """``d_c`` against a bisection on ``max_q Re lambda(q)`` — two unrelated routes.

    The bisection uses the **continuum** operator over unbounded ``q``, because that
    is the limit the closed form is derived in. On a finite grid the onset ratio
    differs slightly, which is a fact about the discretization rather than an error
    in the algebra.
    """
    params = params_at(a=a, b=b)
    qs = np.linspace(1e-3, 400.0, 40001)
    jacobian = reaction_jacobian(*homogeneous_state(params))

    def worst(ratio: float) -> float:
        """``max_q`` of the dominant real part, written out rather than reusing the
        model's ``dispersion``: the bisection walks ratios up to 400, where a
        ``SchnakenbergParams`` would (rightly) refuse the step size, and a check
        against an independent route should not borrow the code it checks anyway."""
        dv = ratio * params.du
        m00 = jacobian[0, 0] - params.du * qs**2
        m11 = jacobian[1, 1] - dv * qs**2
        trace = m00 + m11
        determinant = m00 * m11 - jacobian[0, 1] * jacobian[1, 0]
        disc = np.maximum(trace * trace - 4.0 * determinant, 0.0)
        return float(np.max(0.5 * (trace + np.sqrt(disc))))

    low, high = 1.0, 400.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if worst(mid) > 0.0:
            high = mid
        else:
            low = mid

    assert critical_ratio(params) == pytest.approx(0.5 * (low + high), rel=1e-6)


def test_critical_wavenumber_is_the_argmax_at_onset() -> None:
    """``q_c^2 = u*/sqrt(Du Dv)`` is the first mode to go unstable, at onset only.

    Away from onset the fastest wavenumber moves — at ``1.2 d_c`` the closed form
    reads 18.628 while the fastest mode is near 19.0 — so this is checked exactly
    where the claim is made and :func:`fastest_mode` is used everywhere else.
    """
    reference = params_at()
    at_onset = params_at(dv=critical_ratio(reference) * reference.du)
    qs = np.linspace(1.0, 60.0, 200001)
    rates = np.array(
        [continuum_dispersion(at_onset, q * at_onset.length / (2.0 * math.pi)) for q in qs[::200]]
    )
    coarse_peak = qs[::200][int(np.argmax(rates))]
    assert critical_wavenumber(at_onset) == pytest.approx(coarse_peak, abs=0.05)
    assert critical_wavenumber(at_onset) == pytest.approx(19.496952, rel=1e-6)


def test_critical_ratio_refuses_without_self_activation() -> None:
    """``f_u <= 0`` means no diffusion ratio works, so no critical ratio exists.

    Returning a huge number would suggest an instability sits just out of reach.
    """
    with pytest.raises(ValueError, match="does not self-activate"):
        critical_ratio(params_at(a=1.0, b=0.05))


def test_the_shipped_parameters_sit_just_above_onset() -> None:
    """The default ``dv`` is 1.2x the closed-form onset ratio, and the state is stable.

    Both halves matter: a Turing claim is about a state that is stable on its own and
    destabilized by diffusion.
    """
    params = params_at()
    assert critical_ratio(params) == pytest.approx(7.629775, rel=1e-6)
    assert params.dv / params.du == pytest.approx(1.2 * critical_ratio(params), rel=1e-6)
    assert is_diffusionless_stable(params)


# ------------------------------------------------- the band, and what resolves it


@pytest.mark.parametrize(
    ("length", "n", "expected_count"),
    [(1.0, 32, 1), (2.0, 64, 3), (4.0, 128, 6), (8.0, 256, 13), (16.0, 512, 25)],
)
def test_band_integer_content_grows_with_the_box(
    length: float, n: int, expected_count: int
) -> None:
    """How many integer modes are unstable — the question selection is vacuous without.

    With one unstable mode, "the fastest-growing mode appeared" says only that the
    only mode which could grow, grew. The box length is the lever that buys the claim
    content, and these counts are why ``L = 8`` ships.
    """
    # Grid SPACING held fixed with n proportional to L, which is the comparison that
    # isolates the box from the resolution: at fixed n a longer box is also a coarser
    # one, and the coarseness moves the band on its own.
    params = SchnakenbergParams(
        **{**BASE, "n": n, "dt": 0.02, "t_max": 300.0, "length": length, "mode_j": 3}
    )
    assert params.length / params.n == pytest.approx(0.03125)
    assert unstable_band(params).size == expected_count


def test_stencil_and_continuum_disagree_at_coarse_resolution_and_agree_at_fine() -> None:
    """The gap that gives the selection claim its teeth, in arithmetic alone.

    At 4.3 cells per wavelength the two operators differ by 2.27 modes about *which*
    mode grows fastest; at 10.6 they differ by 0.35. So the discrimination lives on
    the coarse grid, and the fine grid can only confirm agreement.
    """
    coarse = factory(COARSE)
    fine = SchnakenbergParams(**{**BASE, "n": 256, "dt": 0.02, "t_max": 300.0})

    assert fastest_mode(coarse) == 26
    assert continuum_fastest_mode(coarse) == 24
    assert fastest_mode(coarse, continuous=True) == pytest.approx(26.0978, abs=1e-3)
    assert continuum_fastest_mode(coarse, continuous=True) == pytest.approx(23.8285, abs=1e-3)
    assert (
        abs(fastest_mode(coarse, continuous=True) - continuum_fastest_mode(coarse, continuous=True))
        > 2.0
    )

    assert fastest_mode(fine) == continuum_fastest_mode(fine) == 24
    assert (
        abs(fastest_mode(fine, continuous=True) - continuum_fastest_mode(fine, continuous=True))
        < 0.5
    )


def test_a_grid_too_coarse_to_carry_the_pattern_has_no_instability() -> None:
    """Coarsen far enough and the instability does not blur — it *vanishes*.

    The largest ``q_eff^2`` a stencil can represent is ``4/h^2``. Below the unstable
    band, every representable mode is stable, and a grid too coarse to carry the
    pattern is modelling a different problem rather than this one badly.
    """
    params = params_at(n=48, dt=0.1, mode_j=20)
    assert unstable_band(params).size == 0
    h = params.length / params.n
    assert 4.0 / (h * h) < critical_wavenumber(params) ** 2


def test_the_fastest_mode_becomes_nyquist_when_the_grid_is_barely_adequate() -> None:
    """The trap the first design walked into: a prediction of a checkerboard.

    At two cells per wavelength the stencil's fastest mode is ``n/2`` — ``pi j/n``
    exactly ``pi/2`` — and the measured pattern does not match it. A configuration
    whose prediction is Nyquist is not a wavelength-selection configuration, and this
    test exists so that cannot be re-entered silently.
    """
    params = params_at(n=64, dt=0.1, mode_j=20)
    assert fastest_mode(params) == params.n // 2
    q = 2.0 * math.pi * fastest_mode(params) / params.length
    assert q * (params.length / params.n) / 2.0 == pytest.approx(math.pi / 2.0)


def test_fastest_mode_continuous_lies_within_half_a_mode_of_the_best_integer() -> None:
    """Two different quantities, and confusing them accounted for an apparent bias.

    Only integers live on the grid, so a single run selects one of those; an ensemble
    *mean* over integers need not be an integer, and the continuous maximiser is what
    it should be compared against.
    """
    for config in (COARSE, FINE):
        params = factory(config)
        assert abs(fastest_mode(params, continuous=True) - fastest_mode(params)) <= 0.5


def test_the_determinant_check_cannot_catch_a_transposed_jacobian() -> None:
    """Which test catches what, asserted rather than assumed.

    ``det(J^T) = det(J)``, so the determinant identity is blind to a transpose, and
    only the central-difference comparison sees it. Phase 3a found the same shape:
    a consistently transposed system still has a genuine fixed point.
    """
    params = params_at()
    jacobian = reaction_jacobian(*homogeneous_state(params))
    transposed = jacobian.T

    def determinant(m: np.ndarray) -> float:
        return float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])

    assert determinant(transposed) == pytest.approx(determinant(jacobian), rel=1e-15)
    assert np.abs(transposed - jacobian).max() > 1.0  # but they are far from equal


# ----------------------------------------------------------------- params guards


def test_params_reject_a_cfl_violation() -> None:
    with pytest.raises(ValueError, match="CFL limit"):
        params_at(dt=1.0)


def test_params_require_dt_to_divide_t_max_exactly() -> None:
    """The guard Gray-Scott already carries, and the slice's own instrument lacked.

    A step that does not divide the horizon lands the last step somewhere else, and a
    snapshot requested at ``t_max`` is then silently missed — which happened to this
    phase's own measurement script.
    """
    with pytest.raises(ValueError, match="must divide"):
        params_at(dt=0.07)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n": 63}, "even integer"),
        ({"ndim": 3}, "ndim must be 1 or 2"),
        ({"initial": "blob"}, "initial must be one of"),
        ({"mode_j": 0}, "mode_j must be between"),
        ({"mode_j": 57}, "mode_j must be between"),
        ({"a": 0.0}, "a must be positive"),
        ({"du": -1.0}, "du must be positive"),
        ({"length": 0.0}, "length must be positive"),
        ({"t_max": 0.0}, "t_max must be positive"),
    ],
)
def test_params_reject_nonsense(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        params_at(**overrides)


# ------------------------------------------------------- the spectral instrument


@pytest.mark.parametrize("planted", [3, 7, 24, 40])
def test_dominant_mode_recovers_a_planted_wave(planted: int) -> None:
    """The estimator, tested on a field whose answer is known by construction.

    A measurement instrument gets its own test here for the same reason the figures
    do: the first plot this project rendered in a browser was a valid PNG with no
    data in it.
    """
    params = params_at(mode_j=planted)
    field = 3.5 + 0.01 * mode_template(params)
    assert dominant_mode(field) == planted
    assert peak_power_fraction(field) == pytest.approx(1.0, abs=1e-12)


def test_dominant_mode_ignores_the_uniform_component() -> None:
    """A large mean must not become "mode 0", which is not a wavelength.

    The offset here is a million times the pattern, so mode 0 dominates the raw
    spectrum by twelve orders of magnitude and only the explicit zeroing can hide it.
    An earlier version of this test could not fail: the estimator subtracted the mean
    *and* zeroed mode 0, and on a symmetric field the subtraction cancelled exactly,
    so deleting the zeroing changed nothing. Found by mutation.
    """
    params = params_at(mode_j=11)
    field = 1000.0 + 0.001 * mode_template(params)
    raw = np.abs(np.fft.rfft(field)) ** 2
    assert raw[0] > 1e12 * raw[11]  # the offset really does swamp the pattern
    assert power_by_mode(field)[0] == 0.0
    assert dominant_mode(field) == 11
    assert peak_power_fraction(field) == pytest.approx(1.0, abs=1e-12)


def test_dominant_mode_in_two_dimensions_bins_radially() -> None:
    """A stripe and a diagonal pattern of the same wavelength report the same mode.

    Without radial binning a pattern at 45 degrees would report its axis projections
    instead of its wavenumber.
    """
    n, length, j = 64, 8.0, 6
    axis = np.arange(n) * (length / n)
    q = 2.0 * math.pi * j / length
    stripe = 1.0 + 0.01 * np.repeat(np.cos(q * axis)[:, None], n, axis=1)
    diagonal = 1.0 + 0.01 * np.cos(q * axis[:, None]) * np.cos(q * axis[None, :])

    assert dominant_mode(stripe) == j
    assert dominant_mode(diagonal) == int(round(j * math.sqrt(2)))


# --------------------------------------------------------- protocol conformance


def test_registered_with_every_protocol_it_implements() -> None:
    model = get_model("schnakenberg")
    assert model is MODEL
    assert isinstance(model, ValidatableModel)
    assert isinstance(model, TerminableModel)
    assert isinstance(model, FieldModel)


def test_dominant_mode_is_nan_before_a_pattern_exists() -> None:
    """At ``t = 0`` the field is the flat state plus noise: no pattern, no wavenumber.

    Reporting the noise's loudest mode there would be a number that is not a
    measurement — the stance Gray-Scott's ``growth_rate`` takes at ``t = 0``.
    """
    params = factory(COARSE)
    rng = np.random.default_rng(0)
    state = MODEL.initial_state(params, rng)
    assert math.isnan(MODEL.observables(state)["dominant_mode"])
    assert math.isnan(MODEL.observables(state)["growth_rate"])

    after = MODEL.step(state, rng)
    assert math.isfinite(MODEL.observables(after)["dominant_mode"])


def test_initial_mode_is_carried_unchanged_through_the_run() -> None:
    """The mode the noise loaded, kept constant so independence stays checkable."""
    params = factory(COARSE)
    rng = np.random.default_rng(0)
    state = MODEL.initial_state(params, rng)
    loaded = MODEL.observables(state)["initial_mode"]
    for _ in range(5):
        state = MODEL.step(state, rng)
        assert MODEL.observables(state)["initial_mode"] == loaded


def test_fields_are_copies_not_views() -> None:
    params = factory(COARSE)
    state = MODEL.initial_state(params, np.random.default_rng(0))
    fields = MODEL.fields(state)
    fields["u"][:] = -99.0
    assert state.y[0].max() > 0.0


@pytest.mark.parametrize("ndim", [1, 2])
def test_field_shape_follows_ndim(ndim: int) -> None:
    """Phase 4's front-end reports ``field_shapes`` because a field need not be 2-D."""
    params = params_at(n=32, ndim=ndim, dt=0.02, mode_j=8)
    state = MODEL.initial_state(params, np.random.default_rng(0))
    expected = (32,) if ndim == 1 else (32, 32)
    assert MODEL.fields(state)["u"].shape == expected


def test_the_same_seed_reproduces_the_same_run() -> None:
    experiment = Experiment(
        model="schnakenberg",
        params={**COARSE, "t_max": 20.0},
        replicates=2,
        seed=7,
        max_steps=1000,
        record_every=200,
    )
    first = run_experiment(experiment, factory)
    second = run_experiment(experiment, factory)
    for a, b in zip(first.final_observables[0], second.final_observables[0], strict=True):
        assert a == b


# ------------------------------------- claim A1: the seeded mode's growth rate


SEEDED = {**BASE, "n": 256, "dt": 0.02, "t_max": 40.0, "initial": "mode", "eps": 1.0e-5}


def seeded_experiment(mode_j: int, **overrides) -> tuple[Experiment, SchnakenbergParams]:
    config = {**SEEDED, "mode_j": mode_j, **overrides}
    params = factory(config)
    return (
        Experiment(
            model=config.pop("model", "schnakenberg"),
            params=config,
            replicates=2,  # one replicate gives sem = inf and passes vacuously
            seed=0,
            max_steps=n_steps(params) + 10,
            record_every=n_steps(params),
        ),
        params,
    )


def relative_floor(params: SchnakenbergParams, mode_j: int, relative: float = 1e-6) -> float:
    """A numerical floor proportional to the rate being predicted.

    A deterministic model's statistical SE is zero, so ``validate()`` needs a floor —
    and an *absolute* one cannot serve rates spanning 0.012 to 0.234. The claim is a
    relative precision, so the floor is too. At ``1e-6`` it still rejects the
    continuum formula at every probe, by 8x at the closest one.
    """
    return relative * abs(dispersion(params, mode_j)[0])


@pytest.mark.parametrize(
    ("mode_j", "expected_rate", "sign"), [(19, 0.0123618427, +1), (40, -0.2336330693, -1)]
)
def test_validate_reproduces_the_seeded_growth_rate(
    mode_j: int, expected_rate: float, sign: int
) -> None:
    """The category-A anchor, spanning a sign change: mode 19 grows, mode 40 decays.

    A one-sided check cannot see a dispersion relation that has the right shape and
    the wrong sign, which is why both probes ship.
    """
    experiment, params = seeded_experiment(mode_j)
    report = validate(experiment, factory, z=4.0, sem_floor=relative_floor(params, mode_j))
    assert report.passed, str(report)
    (check,) = report.checks
    assert check.predicted == pytest.approx(expected_rate, rel=1e-8)
    assert math.copysign(1.0, check.measured) == sign
    assert abs(check.measured / check.predicted - 1.0) < 1e-6


@pytest.mark.parametrize("mode_j", [19, 40])
def test_a_continuum_prediction_fails_the_seeded_rate_check(mode_j: int) -> None:
    """Tooth: predict with ``-D q^2`` instead of the stencil and the check must fail.

    Off by 39% at mode 19 and 26% at mode 40 — this is Phase 2b's lesson, and here it
    is the same tooth biting a second model.
    """
    experiment, params = seeded_experiment(mode_j)
    broken = f"_test_schnakenberg_continuum_{mode_j}"
    report = validate(
        Experiment(**{**experiment.to_dict(), "model": broken}),
        factory,
        z=4.0,
        sem_floor=relative_floor(params, mode_j),
    )
    assert not report.passed, str(report)
    (check,) = report.checks
    assert abs(check.predicted / dispersion(params, mode_j)[0] - 1.0) > 0.2


def test_the_seeded_residual_falls_with_the_amplitude() -> None:
    """What identifies the residual as the nonlinear correction rather than the stencil.

    Measured relative error 2.07e-06, 2.07e-08, 7.13e-11 at ``eps = 1e-4, 1e-5,
    1e-6`` — a hundredfold fall for a tenfold amplitude, so the correction is
    *quadratic* in the seeded amplitude, as a ``u^2 v`` reaction implies. A residual
    that did not shrink would be a claim about the operator instead.
    """
    errors = []
    for eps in (1e-4, 1e-5):
        experiment, params = seeded_experiment(24, eps=eps)
        report = validate(experiment, factory, z=4.0, sem_floor=1.0)  # floor: report, not assert
        (check,) = report.checks
        errors.append(abs(check.measured / check.predicted - 1.0))
    assert errors[0] == pytest.approx(2.07e-6, rel=0.2)
    assert errors[0] / errors[1] > 20.0


def test_refuses_a_complex_eigenvalue_pair() -> None:
    """Gray-Scott's refusal, reproduced with the error it produces when ignored.

    At ``mode_j = 12`` the pair is complex and the endpoint log-ratio reads 2.8% off
    the common real part, because the amplitude oscillates while it decays.
    """
    params = factory({**SEEDED, "mode_j": 12})
    _, _, is_real = dispersion(params, 12)
    assert not is_real
    with pytest.raises(ValueError, match="complex"):
        MODEL.analytic_predictions(params)


def test_refuses_a_rate_too_near_zero_to_measure() -> None:
    """Just above onset, where the fastest mode itself grows too slowly to measure.

    At ``1.001 d_c`` the fastest mode's rate is ``2.15e-04``, so two e-folds would
    need ``t_max`` near 9300 — and the nonlinear correction arrives long before the
    mode does anything. Reached by moving the *ratio* to onset rather than by walking
    out to a band edge: past the edge the rate is large and negative, which is
    perfectly measurable and refuses nothing.
    """
    reference = factory(COARSE)
    near_onset = {
        **COARSE,
        "initial": "mode",
        "dv": 1.001 * critical_ratio(reference) * reference.du,
        "mode_j": 27,
    }
    assert abs(dispersion(factory(near_onset), 27)[0]) < 1e-3
    with pytest.raises(ValueError, match="too close"):
        MODEL.analytic_predictions(factory(near_onset))


def test_refuses_the_random_initial_condition_and_says_where_to_look() -> None:
    """The project's first refusal on *statistical* grounds.

    The emergent wavenumber is genuinely predicted — but as a discrimination, not as
    a scalar an ensemble mean matches to 4 SE, because the mean of a quantized
    quantity carries a small offset that does not shrink with replicate count.
    """
    with pytest.raises(ValueError, match="selection_report"):
        MODEL.analytic_predictions(factory(COARSE))


def test_refuses_a_state_that_is_unstable_without_diffusion() -> None:
    """Then a growing mode is not a diffusion-driven instability at all."""
    # trace > 0 needs (a+b)^3 < b-a, so both a and b have to be small; a=0.02, b=4.0
    # looks unstable and is not (trace -15.2), which is why this is asserted not assumed.
    params = factory({**COARSE, "initial": "mode", "a": 0.01, "b": 0.5, "mode_j": 20})
    assert not is_diffusionless_stable(params)
    with pytest.raises(ValueError, match="without diffusion"):
        MODEL.analytic_predictions(params)


# ----------------------------------- claim B: which wavelength a run selects


def test_selection_excludes_every_competing_hypothesis(coarse_report) -> None:
    """The headline. Recorded: mean 26.4688 +- 0.1738 against a prediction of 26.0978.

    Margins at ``R = 32``: continuum operator 13.06 SE, band centre 6.68, lowest
    unstable 35.08, highest 52.70 — all against ``z = 4``, and verified at four
    seed-sets whose weakest margin was 6.68.
    """
    assert coarse_report.passed, str(coarse_report)
    margins = {check.name: check.margin_se for check in coarse_report.checks}
    assert margins["continuum operator"] > 10.0
    assert margins["band centre"] > 5.0
    assert margins["lowest unstable"] > 25.0
    assert margins["highest unstable"] > 40.0
    assert coarse_report.measured_mean == pytest.approx(26.4688, abs=0.01)


def test_the_prediction_is_the_closest_hypothesis_not_an_exact_one(coarse_report) -> None:
    """Both halves of the honest claim, asserted together.

    The prediction is nearer the measurement than any alternative — and it is *not*
    exact: the gap grows from 0.61 to 2.24 SE as replicates go from 8 to 48, because
    a real residual of about a third of a mode sits under a shrinking error bar. That
    is why this claim is a margin and not an equality, and why it is not in
    ``analytic_predictions``.
    """
    gaps = [check.gap for check in coarse_report.checks]
    assert coarse_report.gap_to_prediction < min(gaps)
    assert 1.0 < coarse_report.gap_in_se < 4.0
    assert 0.2 < coarse_report.gap_to_prediction < 0.5


def test_the_margin_is_a_difference_of_gaps_not_a_gap(coarse_report) -> None:
    """The report's arithmetic, restated independently of the report.

    A margin must be ``(gap_competitor - gap_prediction) / SE``: a bare
    ``gap_competitor / SE`` would grade a hypothesis on how far it is from the
    measurement without asking whether the prediction does any better, which is not a
    discrimination at all. Mutating the formula survived every other test here, so it
    gets its own.
    """
    for check in coarse_report.checks:
        expected = (check.gap - coarse_report.gap_to_prediction) / coarse_report.sem
        assert check.margin_se == pytest.approx(expected, rel=1e-12)
        assert check.gap == pytest.approx(abs(coarse_report.measured_mean - check.mode), rel=1e-12)


def test_the_band_guard_can_fail(coarse_report) -> None:
    """The containment guard, made falsifiable.

    Every replicate lands inside the true band, so hard-coding ``all_inside_band =
    True`` is invisible — another check nothing could fail. Handing the report a band
    that excludes the answer must flip it, and must sink the whole report.
    """
    wrong = run_selection_with_band(COARSE, replicates=2, band=(1, 3))
    assert not wrong.all_inside_band
    assert not wrong.passed
    assert coarse_report.all_inside_band  # and the real band still contains them


def test_every_replicate_selects_inside_the_unstable_band(coarse_report) -> None:
    """Outside the band nothing can grow, so a mode there would falsify the setup."""
    assert coarse_report.all_inside_band
    assert coarse_report.band == (20, 36)
    assert len(set(coarse_report.selected_modes)) > 1  # and it is a distribution


def test_selection_does_not_track_the_initial_condition(coarse_report) -> None:
    """The trap that would make the agreement a coincidence.

    If the winner were whichever mode the noise loaded most, the prediction would be
    irrelevant. Measured: the initially-loaded modes scatter with a spread of ~15
    modes and most of them fall *outside* the unstable band, while the selected modes
    spread by 0.98 — a ratio of 0.057.
    """
    assert coarse_report.spread_ratio < 0.15
    assert coarse_report.initial_spread > 5.0
    assert coarse_report.initials_outside_band >= COARSE_REPLICATES // 4


def test_predicting_the_continuum_mode_would_fail_the_same_measurement(coarse_report) -> None:
    """Tooth, and it is directional: swap prediction and competitor, and it must break.

    Computed from the same runs rather than by re-measuring, so the assertion costs
    nothing and restates the margin independently of the report's own arithmetic.
    """
    continuum = continuum_fastest_mode(factory(COARSE), continuous=True)
    stencil = fastest_mode(factory(COARSE), continuous=True)
    mean, sem = coarse_report.measured_mean, coarse_report.sem
    swapped_margin = (abs(mean - stencil) - abs(mean - continuum)) / sem
    assert swapped_margin < -10.0


def test_discrimination_needs_the_two_operators_to_disagree(fine_report) -> None:
    """Where the claim runs out, asserted rather than left as a caveat.

    On the fine grid the prediction agrees *better* (0.76 SE against the coarse
    grid's 2.13) and yet the report **fails**, because the band centre is only 1.11 SE
    away and cannot be excluded. Better agreement and weaker discrimination at once,
    which is why the shipped configuration is the coarse one. Verified at four
    seed-sets: the band-centre margin never exceeded 2.81.
    """
    assert fine_report.gap_in_se < 2.0
    assert not fine_report.passed
    failed = [check.name for check in fine_report.checks if not check.passed]
    assert "band centre" in failed
    assert fine_report.all_inside_band


# ------------------------------------------------- the report's own refusals


def test_selection_report_refuses_a_horizon_that_has_not_settled() -> None:
    """Below 20 e-folds of the fastest mode the wavenumber is still drifting.

    Swept at 12, 16, 20, 24 and 30 e-folds; what would be reported at 12 is a
    transient rather than a selection.
    """
    config = {**COARSE, "t_max": 100.0}
    params = factory(config)
    with pytest.raises(ValueError, match="e-folds"):
        selection_report(
            "schnakenberg",
            config,
            factory,
            predicted_mode=fastest_mode(params, continuous=True),
            competitors={"band centre": 28.0},
            band=(20, 36),
            replicates=4,
            max_steps=n_steps(params) + 10,
            efolds=dispersion(params, 26)[0] * params.t_max,
        )


def test_selection_report_refuses_without_a_competing_hypothesis() -> None:
    """A discrimination with nothing to discriminate against passes vacuously."""
    with pytest.raises(ValueError, match="at least one competing hypothesis"):
        selection_report(
            "schnakenberg",
            COARSE,
            factory,
            predicted_mode=26.0,
            competitors={},
            band=(20, 36),
            replicates=4,
        )


def test_selection_report_refuses_a_single_replicate() -> None:
    """One run has no standard error, so every margin would be infinite."""
    config = {**COARSE, "t_max": 300.0}
    params = factory(config)
    with pytest.raises(ValueError, match="at least 2 replicates"):
        selection_report(
            "schnakenberg",
            config,
            factory,
            predicted_mode=26.1,
            competitors={"band centre": 28.0},
            band=(20, 36),
            replicates=1,
            max_steps=n_steps(params) + 10,
            record_every=n_steps(params),
            efolds=None,
        )


def test_selection_report_refuses_a_non_finite_selected_mode() -> None:
    """``dominant_mode`` is ``nan`` before a pattern exists; averaging it is not a
    measurement. A zero-step run is the honest way to reach that state."""
    config = {**COARSE, "t_max": 0.1, "dt": 0.1}
    with pytest.raises(ValueError, match="non-finite"):
        selection_report(
            "schnakenberg",
            config,
            factory,
            predicted_mode=26.1,
            competitors={"band centre": 28.0},
            band=(20, 36),
            replicates=2,
            max_steps=0,  # never steps, so the final observables are the t = 0 ones
            efolds=None,
        )


def test_selection_report_refuses_a_model_without_the_observables() -> None:
    with pytest.raises(KeyError, match="does not report"):
        selection_report(
            "gray_scott",
            {"n": 32, "mode_j": 4, "t_max": 1.0, "dt": 0.05},
            lambda d: __import__(
                "sandbox.models.gray_scott", fromlist=["GrayScottParams"]
            ).GrayScottParams(**d),
            predicted_mode=4.0,
            competitors={"other": 6.0},
            band=(1, 16),
            replicates=2,
            max_steps=100,
            efolds=None,
        )


# ------------------------------------------------------------------------ teeth


class _ContinuumSchnakenberg19(Schnakenberg):
    """Predicts the seeded rate from ``-D q^2``: the equation on paper, not the code."""

    def analytic_predictions(self, params: SchnakenbergParams) -> dict[str, float]:
        return {"growth_rate": continuum_dispersion(params, params.mode_j)}


register("_test_schnakenberg_continuum_19", _ContinuumSchnakenberg19())
register("_test_schnakenberg_continuum_40", _ContinuumSchnakenberg19())
