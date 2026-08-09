"""Gray-Scott: validated linear stability analysis, across a sign change.

HANDOFF promised "validate pattern wavelength against linear stability analysis".
That check is **not available** at the parameters that make Gray-Scott famous, and
the reframing is the most important thing in this file. At Pearson's classic
``(F, k)`` there is no real non-trivial homogeneous steady state at all
(``F < 4(F+k)^2``), so the spots and labyrinths there are not Turing patterns and
linear theory makes no claim about their wavelength. Comparing a measured
wavelength to that prediction would be comparing a number to a prediction that
does not exist.

So Phase 2 validates **``lambda(q)`` itself**, which is sharper: it is exact, it
is cheap, it works at any ``(F, k)``, and it spans a **sign change** — probes that
grow and probes that decay, so a wrong sign cannot hide inside a tolerance.

**The reference is the DISCRETE operator's eigenvalue, and that is not a detail.**
What is simulated is the 5-point stencil, not the continuum Laplacian, and at
``n = 64`` they disagree by 408% at ``j = 12``; worse, the two disagree about
*where the band ends* — ``j = 13`` grows under the discrete operator and decays
under the continuum one. Predicting with ``-D q^2`` and widening the tolerance
until it passed would turn an exact check into a vague one.

**Where the tolerance comes from.** The claim is a *linearization*, so the
discrepancy is not a discretization error and Richardson in ``dt`` cannot see it:
it is ``O(a^2)`` in the perturbation amplitude. Richardson in the **amplitude**
can, and does — measured, ``(4/3)|m(a) - m(a/2)|`` predicts ``|m(a) - lambda|`` to
a ratio of **1.000** at all eight probes, and the extrapolant lands on ``lambda``
to 1e-11..1e-14. That is the tolerance: measured per probe, not typed in.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.laplacian import cfl_limit, stencil_eigenvalue
from sandbox.core.protocol import Experiment, Model, ValidatableModel
from sandbox.core.registry import get_model
from sandbox.core.validation import validate
from sandbox.models.gray_scott import (
    GrayScott,
    GrayScottParams,
    dispersion,
    homogeneous_state,
    reaction_jacobian,
)

# The validated Turing point. Genuine Turing points occupy a razor sliver --
# F in [0.049, 0.117], k in [0.054, 0.062], with the k window at fixed F only
# ~0.5% wide -- so these are not numbers to nudge casually.
_FEED, _KILL = 0.074, 0.062
_N, _DT = 64, 0.2

# Per-probe horizons and seed amplitudes, pinned as literals rather than derived
# from analytic_predictions at run time, so the Experiment is reproducible from
# its params alone and a wrong prediction cannot reshape its own test.
#
# Each horizon is ~2 e-folds, and each `eps` is chosen so every probe ENDS at the
# same amplitude (1e-4) rather than starting there. That matters: the linearization
# error is set by the *final* amplitude, so seeding them all equally made j=3's
# relative error 7.5e-3 against j=7's 1.5e-5 -- a 500x spread that no single
# tolerance could describe honestly. Equalizing the endpoint brings the worst case
# to 1.4e-4 and most probes to ~1e-6.
_GROWING_EPS = 1.35e-5
_DECAYING_EPS = 7.4e-4
_PROBES = [
    (3, 545.4, _GROWING_EPS),
    (5, 161.2, _GROWING_EPS),
    (7, 133.6, _GROWING_EPS),
    (10, 181.4, _GROWING_EPS),
    (12, 457.8, _GROWING_EPS),
    (14, 429.6, _DECAYING_EPS),
    (16, 130.6, _DECAYING_EPS),
    (20, 51.8, _DECAYING_EPS),
]


def _params(**kwargs: object) -> GrayScottParams:
    base: dict[str, object] = {
        "feed": _FEED,
        "kill": _KILL,
        "n": _N,
        "dt": _DT,
        "mode_j": 7,
        "eps": _GROWING_EPS,
        "t_max": 133.6,
    }
    base.update(kwargs)
    return GrayScottParams(**base)  # type: ignore[arg-type]


def _measure(params: GrayScottParams) -> float:
    model = GrayScott()
    rng = np.random.default_rng(0)
    state = model.initial_state(params, rng)
    while not model.is_terminal(state):
        state = model.step(state, rng)
    return model.observables(state)["growth_rate"]


def _continuum_growth_rate(params: GrayScottParams, j: int) -> float:
    """``max Re eig(J - q^2 D)`` with the *continuum* Laplacian — the wrong reference.

    Written out here (not imported) precisely because it is what the model must
    NOT be doing.
    """
    u, v = homogeneous_state(params)
    q = 2.0 * np.pi * j / params.length
    m = reaction_jacobian(u, v, params.feed, params.kill) - q**2 * np.diag([params.du, params.dv])
    return float(np.max(np.linalg.eigvals(m).real))


# ---------------------------------------------------------------------------
# The homogeneous state and its linearization
# ---------------------------------------------------------------------------


def test_the_homogeneous_state_is_a_steady_state_of_the_full_rhs() -> None:
    """A uniform field at ``(u*, v*)`` does not move — checked through the PDE itself.

    The state is found algebraically; feeding it back through the same ``step``
    the simulation uses is a different code path, so agreement catches drift
    between the closed form and the discretization (the stance ``hodgkin_huxley``
    takes with its root-found fixed point).
    """
    params = _params()
    u_star, v_star = homogeneous_state(params)
    model = GrayScott()
    state = model.initial_state(_params(eps=0.0), np.random.default_rng(0))
    for _ in range(50):
        state = model.step(state, np.random.default_rng(0))
    assert np.abs(state.y[0] - u_star).max() < 1e-12
    assert np.abs(state.y[1] - v_star).max() < 1e-12


def test_the_chosen_point_is_a_genuine_turing_point() -> None:
    """Real non-trivial state, stable WITHOUT diffusion, unstable band at ``q > 0``.

    All three are required, and most of the ``(F, k)`` plane fails at least one —
    which is why the validated point sits where it does rather than at Pearson's
    famous parameters.
    """
    params = _params()
    u_star, v_star = homogeneous_state(params)
    assert u_star > 0 and v_star > 0
    without_diffusion = np.linalg.eigvals(
        reaction_jacobian(u_star, v_star, params.feed, params.kill)
    )
    assert np.all(without_diffusion.real < 0), "must be stable before diffusion is added"
    lam_zero, _, _ = dispersion(params, 0)
    assert lam_zero < 0, "an instability at q=0 would be bulk, not Turing"
    growing = [j for j in range(1, _N // 2) if dispersion(params, j)[0] > 0]
    assert growing, "no unstable band: nothing to validate"
    assert min(growing) > 1, "the band must start at q > 0"


def test_pearsons_famous_parameters_have_no_real_non_trivial_state() -> None:
    """The measured fact that forced HANDOFF's promise to be reframed.

    At every one of Pearson's classic pattern points the only homogeneous state is
    the trivial ``(1, 0)``, so those patterns are not Turing patterns and LSA says
    nothing about their wavelength. The model refuses rather than returning a
    complex root cast to a float.
    """
    for feed, kill in [(0.037, 0.060), (0.025, 0.055), (0.014, 0.054), (0.035, 0.065)]:
        assert feed < 4.0 * (feed + kill) ** 2
        with pytest.raises(ValueError, match="no real non-trivial"):
            homogeneous_state(_params(feed=feed, kill=kill))


# ---------------------------------------------------------------------------
# The discrete operator is the reference — asserted, not assumed
# ---------------------------------------------------------------------------


def test_the_prediction_uses_the_discrete_operator_not_the_continuum() -> None:
    """``-D q^2`` is not a slightly-worse reference here; it is a different answer.

    At ``n = 64`` the two disagree by more than 100% at ``j = 12``, and *disagree
    about the sign* at ``j = 13`` — the discrete band extends one mode further than
    the continuum one. So "predict with ``-D|k|^2`` and loosen the tolerance" is
    not available: no tolerance covers an opposite sign.

    This exists so that simplifying ``analytic_predictions`` to the continuum form
    fails loudly instead of quietly widening every other tolerance in the file.
    """
    params = _params()
    discrete_12, _, _ = dispersion(params, 12)
    continuum_12 = _continuum_growth_rate(params, 12)
    assert abs(discrete_12 / continuum_12 - 1.0) > 1.0

    discrete_13, _, _ = dispersion(params, 13)
    continuum_13 = _continuum_growth_rate(params, 13)
    assert discrete_13 > 0.0 > continuum_13

    # And the model's prediction really is the discrete one.
    predicted = GrayScott().analytic_predictions(_params(mode_j=12, t_max=457.8))
    assert predicted["growth_rate"] == pytest.approx(discrete_12, rel=1e-14)
    # ...which is exactly the stencil eigenvalue fed through the linearization.
    h = params.length / params.n
    assert stencil_eigenvalue((2.0 * np.pi * 12 / params.length, 0.0), h, params.du) < 0


# ---------------------------------------------------------------------------
# The headline: lambda(q) across the sign change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode_j", "t_max", "eps"), _PROBES)
def test_measured_growth_rate_matches_the_dispersion_relation(
    mode_j: int, t_max: float, eps: float
) -> None:
    """The category-A check, at a tolerance measured rather than chosen.

    Richardson in the **amplitude**: the linearization error is ``O(a^2)``, so
    halving the seed quarters it and ``(4/3)|m(a) - m(a/2)|`` estimates
    ``|m(a) - lambda|``. Measured, that estimate matches the real error to a ratio
    of **1.000** at every probe, and the extrapolated value lands on ``lambda`` to
    1e-11..1e-14 — so the extrapolant is asserted far more tightly than the raw
    measurement, which is where the sharpness actually lives.

    Richardson in ``dt`` would be the wrong instrument and would report a
    reassuringly tiny number: the error is not a discretization error, and both
    ``dt`` and ``dt/2`` carry the same ``O(a^2)`` term. That is the same trap the
    Hodgkin-Huxley attraction test hit from the other side.
    """
    params = _params(mode_j=mode_j, t_max=t_max, eps=eps)
    predicted = GrayScott().analytic_predictions(params)["growth_rate"]

    coarse = _measure(params)
    fine = _measure(_params(mode_j=mode_j, t_max=t_max, eps=eps / 2.0))
    estimate = (4.0 / 3.0) * abs(coarse - fine)
    extrapolated = (4.0 * fine - coarse) / 3.0

    assert abs(coarse - predicted) <= 3.0 * estimate, (
        f"j={mode_j}: measured {coarse:.8g} vs predicted {predicted:.8g}, "
        f"amplitude-Richardson estimate {estimate:.3g}"
    )
    assert abs(extrapolated - predicted) < 1e-9 * max(1.0, abs(predicted))
    # The sign is part of the claim: this probe grows, or this probe decays.
    assert np.sign(coarse) == np.sign(predicted)


def test_the_probes_actually_span_a_sign_change() -> None:
    """Guard on the parametrization itself: growing AND decaying probes are present.

    Without this, someone trimming ``_PROBES`` to speed the suite up could leave
    only growing modes, and the file would still be green while no longer checking
    the thing it was built to check.
    """
    params = _params()
    rates = [dispersion(params, j)[0] for j, _, _ in _PROBES]
    assert max(rates) > 0 and min(rates) < 0


def test_linearization_error_vanishes_as_the_square_of_the_amplitude() -> None:
    """The exponent is 2 — corroborating what the tolerance mechanism assumes.

    ``cos^2(qx)`` has no component at ``q``, so the quadratic nonlinearity feeds
    modes 0 and ``2q`` but not ``q``; the first feedback into the seeded mode is
    cubic, making the *relative* correction ``O(eps^2)``. Measured slope: 1.9978.
    A slope near 1 would mean the seeded field is contaminated with other modes,
    not that the theory is wrong.
    """
    epsilons = np.array([3.2e-3, 1.6e-3, 8.0e-4, 4.0e-4])
    predicted = GrayScott().analytic_predictions(_params())["growth_rate"]
    errors = np.array([abs(_measure(_params(eps=float(e))) - predicted) for e in epsilons])
    slope = float(np.polyfit(np.log(epsilons), np.log(errors), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.05)


def test_validate_reproduces_the_prediction_through_the_suite() -> None:
    """The protocol contract, exercised through ``validate()`` at one probe.

    The model is deterministic, so the replicate standard error is exactly zero
    and ``validate()``'s statistical tolerance degenerates — the same situation
    ``hh_voltage_clamp`` met. The honest floor is the numerical one, supplied here
    by the amplitude-Richardson estimate. Two replicates rather than one, because
    ``validate()`` reports ``sem = inf`` for a single sample and the check would
    pass vacuously; two identical results from two distinct RNG streams is also
    the determinism check.
    """
    params = _params()
    coarse = _measure(params)
    fine = _measure(_params(eps=_GROWING_EPS / 2.0))
    floor = (4.0 / 3.0) * abs(coarse - fine)

    experiment = Experiment(
        model="gray_scott",
        params={
            "feed": _FEED,
            "kill": _KILL,
            "n": _N,
            "dt": _DT,
            "mode_j": 7,
            "eps": _GROWING_EPS,
            "t_max": 133.6,
        },
        replicates=2,
        observables=("growth_rate",),
        seed=0,
        max_steps=round(133.6 / _DT) + 5,
    )
    report = validate(experiment, lambda d: GrayScottParams(**d), z=3.0, sem_floor=floor)
    assert report.passed, str(report)


# ---------------------------------------------------------------------------
# Refusals: the model declines to predict what it cannot
# ---------------------------------------------------------------------------


def test_a_complex_eigenvalue_pair_is_refused() -> None:
    """At low ``q`` the pair is complex, and ``log|a(T)/a(0)|/T`` is then not ``lambda``.

    Measured at the validated point: ``j = 1`` and ``j = 2`` are complex, ``j >= 3``
    real. The amplitude of a complex pair oscillates as it decays, so a growth rate
    read off two endpoints depends on where in the oscillation the horizon lands —
    a number that looks like a measurement and is not one.
    """
    for j in (1, 2):
        assert not dispersion(_params(), j)[2]
        with pytest.raises(ValueError, match="complex"):
            GrayScott().analytic_predictions(_params(mode_j=j))


def test_a_mode_too_close_to_the_band_edge_is_refused() -> None:
    """``j = 13`` sits at ``lambda = 1.06e-4`` — neither growing nor decaying usefully.

    Two e-folds there would need ``t_max = 18822`` (94,110 steps), and the measured
    result is ``nan``: the amplitude ratio is swamped long before the mode does
    anything. Returning that ``nan`` would propagate into a green-looking check, so
    the model raises instead — the same stance as ``hodgkin_huxley`` refusing to
    predict past the subcritical Hopf.
    """
    lam, _, _ = dispersion(_params(), 13)
    assert 0.0 < lam < 1e-3
    with pytest.raises(ValueError, match="too close to zero"):
        GrayScott().analytic_predictions(_params(mode_j=13, t_max=457.8))


def test_the_exploratory_initial_condition_is_refused_a_prediction() -> None:
    """A Pearson blob has no seeded mode, so there is no growth rate to predict.

    The speculative half of Phase 2b is kept runnable (it is the demo's whole
    point) but is structurally barred from ``analytic_predictions`` — the
    verifiable/exploratory boundary, enforced in code rather than in prose.
    """
    params = _params(feed=0.037, kill=0.060, initial="pearson")
    with pytest.raises(ValueError):
        GrayScott().analytic_predictions(params)


# ---------------------------------------------------------------------------
# Protocol and guards
# ---------------------------------------------------------------------------


def test_registered_and_validatable() -> None:
    model = get_model("gray_scott")
    assert isinstance(model, Model)
    assert isinstance(model, ValidatableModel)


def test_observables_are_scalars_so_the_recorder_needs_no_change() -> None:
    """A PDE with a 64x64x2 state still reports only numbers.

    The plan's claim that the validation track needs *zero* protocol change rests
    on this: the dispersion observable is a mode amplitude, which is a scalar.
    """
    model = GrayScott()
    state = model.initial_state(_params(), np.random.default_rng(0))
    obs = model.observables(state)
    assert set(obs) == {"a_q", "growth_rate", "u_mean", "v_mean"}
    for value in obs.values():
        assert isinstance(value, float)


def test_growth_rate_is_nan_at_t_zero_rather_than_a_convenient_zero() -> None:
    """``log(a/a0)/t`` is ``0/0`` at ``t = 0``. Reporting 0.0 would be a made-up number."""
    state = GrayScott().initial_state(_params(), np.random.default_rng(0))
    assert np.isnan(state.params.dt * 0 + GrayScott().observables(state)["growth_rate"])


def test_seeded_amplitude_matches_the_requested_epsilon() -> None:
    """``a_q(0)`` is ``eps`` times the eigenvector's ``u`` component, not something else.

    The projection ``2 <u, cos(qx)>`` must recover the seed exactly; the uniform
    background contributes nothing because ``cos`` integrates to zero over whole
    periods. If this drifted, every growth rate would still be right (it is a
    *ratio*) while ``a_q`` silently meant something else in the figures.
    """
    params = _params(eps=1e-3)
    state = GrayScott().initial_state(params, np.random.default_rng(0))
    _, evec, _ = dispersion(params, params.mode_j)
    assert GrayScott().observables(state)["a_q"] == pytest.approx(
        params.eps * float(evec[0]), rel=1e-12
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 5.0}, "CFL"),
        ({"dt": 0.3}, "divide"),
        ({"dt": -1.0}, "positive"),
        ({"n": 7}, "even"),
        ({"mode_j": 0}, "between"),
        ({"mode_j": 40}, "between"),
        ({"initial": "spirals"}, "initial"),
    ],
)
def test_params_reject_configurations_that_would_be_wrong_or_unstable(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _params(**kwargs)


def test_the_cfl_bound_is_the_one_the_stencil_dictates() -> None:
    """The guard uses :func:`cfl_limit` on the larger diffusivity, not a typed number."""
    params = _params()
    h = params.length / params.n
    limit = cfl_limit(h, max(params.du, params.dv), ndim=2)
    # t_max follows dt here so the "dt must divide t_max" guard cannot fire first
    # and make this look like a CFL pass when it is really a different rejection.
    _params(dt=limit / 2.0, t_max=limit * 5.0)  # just inside: accepted
    with pytest.raises(ValueError, match="CFL"):
        _params(dt=limit * 2.0, t_max=limit * 20.0)
