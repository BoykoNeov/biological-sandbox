"""Daisyworld: the closed-form interior state, and homeostasis as an assertion.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` step 11-13.

**What is anchored to what** — the three-category framing Phase 2 established:

* **Category A, independent.** :func:`textbook_rhs` and :func:`textbook_temperatures`
  below are hand-transcribed from Watson & Lovelock 1983 and share no code with
  the model; the two equilibrium conditions ``x beta(T_i) = gamma``; the quartic
  identity ``T_b^4 - T_w^4 = q (A_w - A_b)`` that the Cardano reduction encodes;
  a 200-iteration bisection for ``delta``; and a 2-D Newton root-find of the RHS,
  which reaches the same cover without ever touching the closed form.
* **Category A, external.** The **recorded literals** — ``delta``, ``T_w*``,
  ``T_b*``, ``beta*``, ``x*``, the band endpoints, and ``a_w*(L)``. These are the
  *only* thing that catches a constant wrong in **both** the closed form and the
  RHS: such a model still has a genuine fixed point, still passes every
  self-consistency check above, and still root-finds to its own answer. Measured:
  shifting ``t_opt`` by 1 K leaves ``|rhs(y*)| = 1.8e-16``.
* **Category A, self-consistency.** ``validate()`` at three luminosities: the
  prediction route (Cardano + the linear albedo balance) against the observable
  route (albedo sum + local temperature law) through an actual integration.
* **Category B.** RK4's order 4 on the transient, and the luminosity invariance
  of the *simulated* ``T_w`` bounded by a two-horizon decay law.
* **Category C — never asserted against a bound.** Hysteresis, dieback, the
  bare-planet comparison and the ``dT_e/dL`` magnitudes live in
  ``demos/daisyworld.py``. Only the *sign* of the overcompensation is asserted.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import pytest

from sandbox.core.ode import rk4_step
from sandbox.core.protocol import Experiment
from sandbox.core.recorder import run_replicate
from sandbox.core.validation import validate
from sandbox.models.daisyworld import (
    MODEL,
    PREDICTED_KEYS,
    DaisyworldParams,
    bare_fraction,
    daisyworld_rhs,
    delta_offset,
    equilibrium_albedo,
    growth_rate,
    growth_rate_unclipped,
    interior_equilibrium,
    interior_temperatures,
    invasion_luminosities,
    jacobian,
    n_daisy_steps,
    regulating_band,
    slowest_rate,
    temperatures,
)

BARE_START = (0.01, 0.01)


def params_factory(d: dict) -> DaisyworldParams:
    return DaisyworldParams(**d)


def reference(**overrides) -> DaisyworldParams:
    return DaisyworldParams(**overrides)


# --------------------------------------------------------------------------
# Hand-transcribed reference implementation -- shares no code with the model
# --------------------------------------------------------------------------


def textbook_temperatures(a_w: float, a_b: float, p: DaisyworldParams):
    """Watson & Lovelock's energy balance, written out longhand.

    Deliberately avoids the model's helpers so a swapped albedo, a dropped
    ``q (A - A_i)`` offset or a wrong Stefan-Boltzmann exponent shows up as a
    disagreement rather than propagating consistently through both sides.
    """
    bare = 1.0 - a_w - a_b
    albedo = 0.0
    for fraction, value in (
        (a_w, p.albedo_white),
        (a_b, p.albedo_black),
        (bare, p.albedo_ground),
    ):
        albedo += fraction * value
    t_e = math.pow(p.solar_flux * p.luminosity * (1.0 - albedo) / p.stefan_boltzmann, 0.25)
    t_w = math.pow(p.q * (albedo - p.albedo_white) + t_e**4, 0.25)
    t_b = math.pow(p.q * (albedo - p.albedo_black) + t_e**4, 0.25)
    return albedo, t_e, t_w, t_b


def textbook_rhs(y, p: DaisyworldParams) -> np.ndarray:
    a_w, a_b = float(y[0]), float(y[1])
    bare = 1.0 - a_w - a_b
    _, _, t_w, t_b = textbook_temperatures(a_w, a_b, p)
    beta_w = max(0.0, 1.0 - p.beta_k * (p.t_opt - t_w) ** 2)
    beta_b = max(0.0, 1.0 - p.beta_k * (p.t_opt - t_b) ** 2)
    return np.array([a_w * (bare * beta_w - p.gamma), a_b * (bare * beta_b - p.gamma)])


def test_rhs_matches_the_hand_written_transcription():
    # ABSOLUTE tolerance, and that is the measurement rather than a preference. The
    # RHS is `a_i (x beta - gamma)` and `x beta` sits near `gamma` over much of the
    # simplex, so a relative tolerance is meaningless near its zeros: over 4000
    # draws the worst relative disagreement is 1.6e-11 while the worst absolute one
    # is 1.554e-15. The source is the transcription's honest round trip -- it takes
    # T_e = X**0.25 and then squares back to T_e**4, costing 2 ULP of X ~ 8e9 -- and
    # keeping that round trip is the point of transcribing from the equations.
    #
    # 1e-13 sits 64x above the measured floor and ~1e10 below any real defect: a
    # swapped albedo or a dropped q(A - A_i) offset moves the RHS by O(0.01).
    p = reference()
    rhs = daisyworld_rhs(p)
    rng = np.random.default_rng(11)
    for _ in range(40):
        a_w, a_b = rng.uniform(0.0, 0.45, size=2)
        y = np.array([a_w, a_b])
        assert np.allclose(rhs(y), textbook_rhs(y, p), rtol=0.0, atol=1e-13)
        # The temperatures themselves have no cancellation, so they take a genuine
        # relative tolerance.
        assert temperatures(a_w, a_b, p) == pytest.approx(
            textbook_temperatures(a_w, a_b, p), rel=1e-14
        )


def test_the_local_temperature_law_carries_the_right_albedo_per_species():
    # THE swap detector for the temperature offsets: white sits COLDER than the
    # planet and black HOTTER, and the offsets are signed. Comparing |T_i - T_e|
    # would pass a model that swapped them.
    p = reference()
    a_w, a_b = 0.3, 0.2
    albedo, t_e, t_w, t_b = temperatures(a_w, a_b, p)
    assert t_w < t_e < t_b
    assert t_w**4 - t_e**4 == pytest.approx(p.q * (albedo - p.albedo_white), rel=1e-12)
    assert t_b**4 - t_e**4 == pytest.approx(p.q * (albedo - p.albedo_black), rel=1e-12)


def test_the_per_capita_prefactor_is_present():
    # Dropping the a_i factor would let an absent species spontaneously appear.
    p = reference()
    dy = daisyworld_rhs(p)(np.array([0.0, 0.4]))
    assert dy[0] == 0.0, "an absent species grew from nothing -- the a_i prefactor is missing"


def test_growth_is_a_parabola_clipped_at_zero_and_symmetric_about_t_opt():
    p = reference()
    for offset in (0.0, 3.0, 10.0, 17.0, 25.0):
        assert float(growth_rate(p.t_opt + offset, p)) == pytest.approx(
            float(growth_rate(p.t_opt - offset, p)), rel=0.0, abs=1e-15
        )
    assert float(growth_rate(p.t_opt, p)) == pytest.approx(1.0, rel=1e-15)
    # Outside T_opt +- 1/sqrt(k) = +- 17.500 K the raw parabola is negative and the
    # clip is what keeps beta a growth rate rather than a death rate.
    half_width = 1.0 / math.sqrt(p.beta_k)
    assert float(growth_rate_unclipped(p.t_opt + half_width + 1.0, p)) < 0.0
    assert float(growth_rate(p.t_opt + half_width + 1.0, p)) == 0.0


# --------------------------------------------------------------------------
# Category A, independent: the reduction that removes the root-find
# --------------------------------------------------------------------------


def test_delta_solves_the_quartic_identity_and_matches_a_bisection():
    # The Cardano root is a difference of two cube roots ~35x its own size
    # (+173.119 and -168.131), so about 1.5 digits are lost; rel=1e-12, not 1e-15.
    p = reference()
    delta = delta_offset(p)

    def residual(d: float) -> float:
        return (p.t_opt + d) ** 4 - (p.t_opt - d) ** 4 - p.q * (p.albedo_white - p.albedo_black)

    lo, hi = 0.0, 17.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    bisected = 0.5 * (lo + hi)

    assert delta == pytest.approx(bisected, rel=1e-12)
    target = p.q * (p.albedo_white - p.albedo_black)
    assert abs(residual(delta)) / target < 1e-13


def test_both_species_sit_at_the_same_growth_rate_and_it_balances_death():
    # The two equilibrium conditions the whole reduction rests on, checked at the
    # computed state rather than assumed: beta(T_w) = beta(T_b), and x beta = gamma
    # for BOTH species. Checking only one would miss a delta of the wrong sign.
    for luminosity in (0.8, 1.0, 1.2):
        p = reference(luminosity=luminosity)
        a_w, a_b = interior_equilibrium(p)
        bare = 1.0 - a_w - a_b
        _, _, t_w, t_b = temperatures(a_w, a_b, p)
        beta_w, beta_b = float(growth_rate(t_w, p)), float(growth_rate(t_b, p))
        assert beta_w == pytest.approx(beta_b, rel=1e-13)
        assert bare * beta_w == pytest.approx(p.gamma, rel=1e-12)
        assert bare * beta_b == pytest.approx(p.gamma, rel=1e-12)


def test_the_equilibrium_is_a_root_of_the_hand_written_rhs():
    for luminosity in (0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35):
        p = reference(luminosity=luminosity)
        y_star = interior_equilibrium(p)
        assert np.abs(textbook_rhs(y_star, p)).max() < 1e-15


def test_a_newton_root_find_of_the_rhs_reaches_the_closed_form():
    # The anchor that owes nothing to the Cardano algebra: Newton on the raw
    # vector field, started off the answer, converges to the same cover. It is
    # NOT independent of the constants -- it root-finds the same RHS -- which is
    # exactly why the recorded literals below are load-bearing too.
    for luminosity in (0.8, 0.9, 1.0, 1.1, 1.2):
        p = reference(luminosity=luminosity)
        y_star = interior_equilibrium(p)
        rhs = daisyworld_rhs(p)
        y = y_star + np.array([0.02, -0.015])
        for _ in range(60):
            jac = np.empty((2, 2))
            h = 1e-7
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                jac[:, j] = (rhs(y + e) - rhs(y - e)) / (2 * h)
            y = y - np.linalg.solve(jac, rhs(y))
        assert np.abs(y - y_star).max() < 1e-14, f"newton {y} vs closed form {y_star}"


# --------------------------------------------------------------------------
# Category A, EXTERNAL: the recorded literals
# --------------------------------------------------------------------------


def test_the_recorded_constants_reproduce():
    # The only check in this file that a value wrong in BOTH the closed form and
    # the RHS cannot survive. Measured: shifting t_opt to 296.5 gives
    # delta = 4.938018379, T_w* = 291.561982, x* = 0.325950, a_w*(1.0) = 0.385854
    # -- and |rhs(y*)| = 1.8e-16, i.e. a perfectly good fixed point of the wrong
    # planet. Every self-consistency test above stays green; only these numbers
    # move. The 3a analogue is transposing A everywhere.
    p = reference()
    t_w, t_b = interior_temperatures(p)
    assert delta_offset(p) == pytest.approx(4.988282516540778, rel=1e-13)
    assert t_w == pytest.approx(290.511717483459222, rel=1e-14)
    assert t_b == pytest.approx(300.488282516540778, rel=1e-14)
    assert float(growth_rate(t_w, p)) == pytest.approx(0.918757127552342, rel=1e-13)
    assert bare_fraction(p) == pytest.approx(0.326528079079211, rel=1e-13)


def test_the_regulating_band_reproduces_the_bisected_endpoints():
    # The planning slice bisected for these; the model inverts the linear albedo
    # balance instead and must land on the same numbers.
    low, high = regulating_band(reference())
    assert low == pytest.approx(0.738722418247, rel=1e-11)
    assert high == pytest.approx(1.359472371265, rel=1e-11)
    # ...and the endpoints are exactly where a component reaches zero.
    assert interior_equilibrium(reference(luminosity=low + 1e-9))[0] < 1e-8
    assert interior_equilibrium(reference(luminosity=high - 1e-9))[1] < 1e-8


# albedo_ground is SWEPT, and that is not decoration. The W&L value is 0.5, where
# `A_g` and `1 - A_g` are the same number -- so a mutant that swapped them in
# `invasion_luminosities` was a no-op, stayed green, and (because it changed
# nothing) went undetected long enough to corrupt a whole mutation run. 0.4 breaks
# the symmetry. This is Phase 2's "sweep the constant that the probe happens to
# land on" for the third time in the project, and the second time in this file
# after the white/black albedo swap.
INVASION_GROUNDS = (0.5, 0.4)


@pytest.mark.parametrize("albedo_ground", INVASION_GROUNDS)
def test_the_invasion_window_predicts_who_can_start_a_dead_planet(albedo_ground):
    # A SECOND closed form, and a different question from the band: the band says
    # where the interior state exists, this says where a bare planet can be
    # started. Checked against the dynamics rather than against itself -- a rare
    # species is seeded on an otherwise dead planet and must grow inside its own
    # window and shrink outside it.
    #
    # The probes stand off each edge by 0.01 deliberately: exactly AT the edge the
    # net growth rate is zero and neither outcome is a prediction. Which is also
    # the honest reading of the demo's ramp, where L = 1.20 sits 0.008 inside the
    # white window and has not converged after 400 time units.
    p0 = reference(albedo_ground=albedo_ground)
    windows = invasion_luminosities(p0)
    if albedo_ground == 0.5:
        assert windows["white"] == pytest.approx((0.8332000704, 1.2079168175), rel=1e-10)
        assert windows["black"] == pytest.approx((0.7058188359, 1.0805355830), rel=1e-10)

    seed, horizon = 1e-6, 40.0
    for index, (name, (low, high)) in enumerate(windows.items()):
        for luminosity, inside in (
            (low - 0.01, False),
            (low + 0.01, True),
            (high - 0.01, True),
            (high + 0.01, False),
        ):
            p = reference(
                albedo_ground=albedo_ground, luminosity=luminosity, t_max=horizon, dt=0.01
            )
            start = [0.0, 0.0]
            start[index] = seed
            y = np.asarray(start, dtype=float)
            rhs = daisyworld_rhs(p)
            for _ in range(n_daisy_steps(p)):
                y = rk4_step(rhs, y, p.dt)
            grew = bool(y[index] > seed)  # bool(), not np.bool_ -- `is` would always fail
            assert grew is inside, (
                f"{name} at A_g={albedo_ground}, L={luminosity:.4f}: seeded {seed:g}, ended "
                f"{y[index]:.3e}; the closed-form window {low:.6f}..{high:.6f} says "
                f"grew={inside}"
            )

    if albedo_ground != 0.5:
        return
    # The windows overlap, so a bare planet is invadable across their hull -- and
    # that hull is what act 5 of the demo differences against the regulating band.
    assert windows["black"][1] > windows["white"][0], "the two windows must overlap"
    band_low, band_high = regulating_band(p0)
    invadable_high = max(high for _, high in windows.values())
    invadable_low = min(low for low, _ in windows.values())
    # Bistable at the hot end (interior state exists, nothing can invade)...
    assert band_high - invadable_high == pytest.approx(0.151555553710, rel=1e-8)
    # ...and NOT at the cold end: invadability reaches below the band there, so the
    # arithmetic gives a negative width. Asserted so the asymmetry cannot silently
    # flip and turn the demo's one-sidedness claim into a wrong figure.
    assert invadable_low < band_low


@pytest.mark.parametrize(
    ("luminosity", "a_w_star", "a_b_star"),
    [
        (0.8, 0.121780644989, 0.551691275932),
        (1.0, 0.400242526271, 0.273229394650),
        (1.2, 0.574860659811, 0.098611261110),
    ],
)
def test_the_recorded_cover_reproduces(luminosity, a_w_star, a_b_star):
    # a_w* SPECIFICALLY, because the Watson & Lovelock albedos are symmetric
    # (A_w + A_b = 1 and 2 A_g = 1). Under a white/black swap delta flips sign,
    # T_w* and T_b* merely exchange, and x* is BIT-IDENTICAL -- so the literals
    # above are all blind to it. The swap shows up here: a_w*(1.0) becomes
    # 0.273229394650, which is the base model's a_b*. Same shape as 3c's "two of
    # the three wrong formulas are invisible on species 0".
    y_star = interior_equilibrium(reference(luminosity=luminosity))
    assert y_star == pytest.approx([a_w_star, a_b_star], rel=1e-11)


# --------------------------------------------------------------------------
# Category A, self-consistency: through the ValidationSuite
# --------------------------------------------------------------------------


def run_daisyworld(params: DaisyworldParams) -> dict[str, np.ndarray]:
    traj = run_replicate(
        MODEL, params, np.random.default_rng(0), max_steps=n_daisy_steps(params) + 10
    )
    assert traj.terminated, "daisyworld run hit max_steps without terminating"
    return traj.as_arrays()[1]


def final_observables(params: DaisyworldParams) -> dict[str, float]:
    series = run_daisyworld(params)
    return {key: float(series[key][-1]) for key in PREDICTED_KEYS}


@pytest.mark.parametrize("luminosity", [0.8, 1.0, 1.2])
def test_validate_reproduces_the_interior_state(luminosity):
    # Non-negotiable #2. Started AT the closed-form equilibrium, so the residual
    # is integration error only -- Richardson in dt measures it, and measures
    # EXACTLY 0.0: |rhs(y*)| ~ 1e-16 against a cover ~0.4, so a step moves the
    # state below the ULP and RK4 never leaves (gLV's finding verbatim). The floor
    # therefore falls back, and what this turns on is the CROSS-ROUTE agreement:
    # analytic_predictions builds T_w/T_b from Cardano and the albedo from the
    # linear energy balance, while the observables build all four from the cover
    # through the albedo sum and the local temperature law.
    #
    # The floor is absolute and shared across keys whose scales differ by 3 orders
    # (a_w ~ 0.4, T_b ~ 300), so it is sized for the temperatures: 1e-11 is ~180
    # ULP of 300 K, i.e. 3e-14 relative. Two replicates -- one gives sem = inf.
    base = {"luminosity": luminosity, "initial": "equilibrium", "t_max": 20.0, "dt": 0.01}
    coarse, fine = (
        final_observables(DaisyworldParams(**{**base, "dt": dt})) for dt in (0.01, 0.005)
    )
    bound = max(abs(coarse[k] - fine[k]) for k in PREDICTED_KEYS)
    experiment = Experiment(
        model="daisyworld",
        params=base,
        replicates=2,
        observables=PREDICTED_KEYS,
        seed=0,
        max_steps=n_daisy_steps(DaisyworldParams(**base)) + 10,
    )
    report = validate(experiment, params_factory, z=4.0, sem_floor=max(bound, 1e-11))
    assert report.passed, str(report)
    assert {c.name for c in report.checks} == set(PREDICTED_KEYS)


def test_starting_at_the_equilibrium_does_not_drift():
    series = run_daisyworld(reference(initial="equilibrium", t_max=200.0, dt=0.01))
    for key in ("a_w", "a_b"):
        values = np.asarray(series[key])
        assert np.abs(values - values[0]).max() < 1e-14


# --------------------------------------------------------------------------
# THE REGULATION CLAIM
# --------------------------------------------------------------------------

INVARIANCE_L = (0.9, 0.95, 1.0, 1.05, 1.1)
INVARIANCE_T = 100.0
INVARIANCE_DT = 0.01


@cache
def bare_start_endpoint(luminosity: float, t_max: float) -> tuple[float, float, float]:
    """``(T_w, T_b, T_e)`` after integrating from a bare planet. Cached per worker."""
    p = reference(luminosity=luminosity, a_init=BARE_START, t_max=t_max, dt=INVARIANCE_DT)
    series = run_daisyworld(p)
    return tuple(float(series[k][-1]) for k in ("T_w", "T_b", "T_e"))


def test_the_temperature_law_returns_the_pinned_temperatures_at_every_luminosity():
    # The CLOSED-FORM half of dT_w/dL = 0, and not a tautology: a_w* and a_b* both
    # move with L (0.024 -> 0.668 across the band), and the local temperature law
    # mixes them with L through both the albedo sum and T_e^4. That the result
    # comes back to T_opt - delta every time is a real check of the albedo solve.
    #
    # Asserting that `interior_temperatures` is L-free WOULD be a tautology -- the
    # expression contains no L -- which is why the route through the cover is the
    # one measured. Measured: bit-identical at every L below except a 1-ULP wobble
    # at the top of the band.
    #
    # This is also THE cross-route check, and the mutation run is what established
    # that. Making `analytic_predictions` read `temperatures(y*)` instead of Cardano
    # turns out to be an invisible mutant -- `validate()` cannot see the difference,
    # because this test has already pinned the two routes to each other at every L.
    # The albedo leg is included for the same reason: without it, the round trip
    # from the linear energy balance to the cover and back is nowhere asserted.
    p0 = reference()
    t_w_star, t_b_star = interior_temperatures(p0)
    ulp = float(np.spacing(t_b_star))
    seen_w, seen_b = [], []
    for luminosity in (0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35):
        p = reference(luminosity=luminosity)
        a_w, a_b = interior_equilibrium(p)
        albedo, _, t_w, t_b = temperatures(a_w, a_b, p)
        seen_w.append(t_w)
        seen_b.append(t_b)
        # The albedo the cover produces must be the one the balance asked for.
        # Measured agreement: <= 1 ULP at every L in the band. The slack matches
        # the temperature asserts below (4 ULP rather than 2) because the residual
        # is libm rounding in `**0.25`, which is not guaranteed identical across
        # platforms, and 4 ULP is still ~1e-16 on a quantity of order 0.5.
        assert abs(albedo - equilibrium_albedo(p)) <= 4.0 * float(np.spacing(albedo))
    for values, expected in ((seen_w, t_w_star), (seen_b, t_b_star)):
        spread = max(values) - min(values)
        assert spread <= 4.0 * ulp, f"daisy temperature moved {spread:.3e} K across L"
        assert abs(np.mean(values) - expected) <= 4.0 * ulp


def test_the_simulated_daisy_temperature_does_not_move_with_luminosity():
    # The DYNAMICAL half, and the one that can fail for a model whose algebra is
    # L-free but whose trajectories are not: one common bare start, five
    # luminosities, integrated to the same horizon, and the endpoint T_w compared
    # across L.
    #
    # The tolerance is derived, not typed. The residual here is LEFTOVER TRANSIENT,
    # not discretization error -- measured, |T_w(dt) - T_w(dt/2)| is exactly 0.0 at
    # L = 1.10 where the deviation from T_w* is 2.1e-6, so Richardson in dt is
    # blind to it (the Phase-2 lesson verbatim). The instrument that sees it is the
    # two-horizon decay law, as in test_glv / test_hodgkin_huxley.
    #
    # t_max = 100 is chosen because LONGER IS WORSE: at t_max = 150 and 200 the
    # true residual has hit the floating-point floor (~8e-13, ~14 ULP of T_w) while
    # the predicted bound keeps shrinking, and the ratio below runs to 3e7. The
    # horizon is part of the claim.
    predicted = []
    for luminosity in INVARIANCE_L:
        full = bare_start_endpoint(luminosity, INVARIANCE_T)[0]
        half = bare_start_endpoint(luminosity, INVARIANCE_T / 2.0)[0]
        tau = 1.0 / slowest_rate(reference(luminosity=luminosity))
        predicted.append(abs(full - half) * math.exp(-INVARIANCE_T / (2.0 * tau)))

    finals = [bare_start_endpoint(lum, INVARIANCE_T)[0] for lum in INVARIANCE_L]
    spread = max(finals) - min(finals)
    bound = 10.0 * max(predicted)
    assert spread <= bound, (
        f"simulated T_w moved {spread:.3e} K across L = {INVARIANCE_L}, above the "
        f"decay-law bound {bound:.3e} (per-L predictions {predicted})"
    )

    # The non-vacuity guard: over the SAME runs the planetary temperature must
    # actually move, or "T_w did not move" would be a threshold nothing can fail.
    # Measured, T_e spans 2.179 K here against T_w's 2.14e-6 -- a factor of 1.0e6.
    planetary = [bare_start_endpoint(lum, INVARIANCE_T)[2] for lum in INVARIANCE_L]
    assert max(planetary) - min(planetary) > 1.0, (
        "the planetary temperature barely moved either, so the invariance of T_w "
        "says nothing about regulation"
    )


@pytest.mark.parametrize("luminosity", INVARIANCE_L)
def test_the_decay_law_predicts_the_residual_transient(luminosity):
    # The instrument itself, verified rather than assumed -- the Richardson-in-the-
    # amplitude discipline applied to a transient. For a residual decaying as
    # A e^{-t/tau}, |T(t) - T(t/2)| is dominated by the value at t/2, so the
    # remaining error is that difference times exp(-t / (2 tau)).
    #
    # Measured ratios at t_max = 100: 1.000 / 1.000 / 1.000 / 1.000 / 1.003.
    full = bare_start_endpoint(luminosity, INVARIANCE_T)[0]
    half = bare_start_endpoint(luminosity, INVARIANCE_T / 2.0)[0]
    tau = 1.0 / slowest_rate(reference(luminosity=luminosity))
    predicted = abs(full - half) * math.exp(-INVARIANCE_T / (2.0 * tau))
    actual = abs(full - interior_temperatures(reference())[0])
    assert predicted > 0.0
    ratio = actual / predicted
    assert 0.5 < ratio < 2.0, (
        f"at L={luminosity} the decay law predicted {predicted:.4e} and the true "
        f"residual is {actual:.4e} (ratio {ratio:.4f})"
    )


def test_the_planet_overcompensates_while_a_bare_one_does_not():
    # The sharp form of regulation: T_e does not merely flatten, it runs BACKWARDS.
    # Only the SIGN is asserted -- the magnitudes (-13.57 / -10.77 / -8.75 at
    # L = 0.9 / 1.0 / 1.1) are category-C reporting and live in the demo, because a
    # recorded magnitude travels only with the estimator that produced it.
    simulated = [bare_start_endpoint(lum, INVARIANCE_T)[2] for lum in INVARIANCE_L]
    for cooler, warmer in zip(simulated[:-1], simulated[1:], strict=True):
        assert warmer < cooler, f"T_e did not fall with luminosity: {simulated}"

    p0 = reference()
    for luminosity in np.linspace(0.75, 1.35, 13):
        p = reference(luminosity=float(luminosity))
        albedo = equilibrium_albedo(p)
        t_e = (p.solar_flux * luminosity * (1.0 - albedo) / p.stefan_boltzmann) ** 0.25
        step = 1e-4
        neighbours = []
        for offset in (-step, step):
            q = reference(luminosity=float(luminosity) + offset)
            a = equilibrium_albedo(q)
            neighbours.append(
                (q.solar_flux * q.luminosity * (1.0 - a) / q.stefan_boltzmann) ** 0.25
            )
        slope_daisy = (neighbours[1] - neighbours[0]) / (2 * step)
        # A bare planet: A = A_g always, so T_e ~ L^{1/4} and dT_e/dL = T_e/(4L).
        t_e_bare = (
            p0.solar_flux * luminosity * (1.0 - p0.albedo_ground) / p0.stefan_boltzmann
        ) ** 0.25
        slope_bare = t_e_bare / (4.0 * luminosity)
        assert slope_daisy < 0.0 < slope_bare, (
            f"at L={luminosity:.3f} the daisy slope is {slope_daisy:+.4f} and the bare "
            f"slope {slope_bare:+.4f}; overcompensation requires opposite signs"
        )
        assert t_e > 0.0


# --------------------------------------------------------------------------
# Category B: integrator order, and the C^0 kink that can destroy it
# --------------------------------------------------------------------------

ORDER_DTS = (0.4, 0.2, 0.1, 0.05)


def transient_endpoint(luminosity: float, dt: float, t_max: float = 20.0) -> np.ndarray:
    p = reference(luminosity=luminosity, a_init=BARE_START, t_max=t_max, dt=dt)
    series = run_daisyworld(p)
    return np.array([series["a_w"][-1], series["a_b"][-1]])


def test_error_is_fourth_order_on_the_transient():
    # PRESCRIBED WINDOW: L = 1.0, y0 = (0.01, 0.01), t_max = 20, dt in [0.4, 0.05],
    # max-norm against a dt = 1e-3 reference. Measured 15.26 / 15.56 / 15.76.
    #
    # Both edges of the window are part of the claim. At t_max = 200 the trajectory
    # has reached its attractor and the "error" is the dt-INDEPENDENT distance to
    # the fixed point: measured ratio 1.00 at every dt (1.283e-13 -> 1.233e-13).
    # Only a transient can see discretization error. And at t_max = 40 the finest
    # steps are already into roundoff (7e-15), giving 12.69 / 14.38 / 15.11 / 14.56.
    reference_end = transient_endpoint(1.0, 1e-3)
    errors = [float(np.abs(transient_endpoint(1.0, dt) - reference_end).max()) for dt in ORDER_DTS]
    assert all(e > 0.0 for e in errors)
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        ratio = coarse / fine
        assert 12.0 < ratio < 20.0, f"expected ~16 for RK4, got {ratio:.2f} (errors {errors})"


def test_the_growth_clip_never_bites_where_the_order_claim_is_made():
    # What licenses the test above. beta's clip makes the RHS only C^0, and RK4's
    # order needs smoothness -- so the order claim is only honest on a trajectory
    # that stays inside the parabola's positive region. At L = 1.0 from (0.01, 0.01)
    # the raw beta never falls below +0.733 and clipped and smooth integrations are
    # BIT-IDENTICAL, which is the strongest form of "the clip is not in play".
    p = reference(luminosity=1.0, a_init=BARE_START, t_max=20.0, dt=0.01)

    def smooth_rhs(y: np.ndarray) -> np.ndarray:
        a_w, a_b = y[0], y[1]
        bare = 1.0 - a_w - a_b
        _, _, t_w, t_b = temperatures(a_w, a_b, p)
        return np.array(
            [
                a_w * (bare * float(growth_rate_unclipped(t_w, p)) - p.gamma),
                a_b * (bare * float(growth_rate_unclipped(t_b, p)) - p.gamma),
            ]
        )

    clipped_rhs = daisyworld_rhs(p)
    clipped = np.asarray(BARE_START, dtype=float)
    smooth = clipped.copy()
    worst = math.inf
    for _ in range(n_daisy_steps(p)):
        _, _, t_w, t_b = temperatures(clipped[0], clipped[1], p)
        worst = min(
            worst, float(growth_rate_unclipped(t_w, p)), float(growth_rate_unclipped(t_b, p))
        )
        clipped = rk4_step(clipped_rhs, clipped, p.dt)
        smooth = rk4_step(smooth_rhs, smooth, p.dt)

    assert worst > 0.7, f"the raw growth parabola dipped to {worst:+.4f} on this transient"
    assert np.array_equal(clipped, smooth), "clipped and smooth integrations diverged"


def test_the_clip_does_bite_elsewhere_and_the_order_collapses_when_it_does():
    # The other side of the same coin, and the reason the luminosity is written
    # into the claim rather than left implicit. At L = 0.8 the SAME start drives the
    # raw parabola to -0.025, and the order measurement reads 58.68 / 0.49 / 2.84
    # with non-monotone errors. Asserted structurally -- that the ratios are NOT all
    # near 16 -- rather than against those numbers, which are noise off a kink.
    p = reference(luminosity=0.8, a_init=BARE_START, t_max=20.0, dt=0.01)
    y, worst = np.asarray(BARE_START, dtype=float), math.inf
    rhs = daisyworld_rhs(p)
    for _ in range(n_daisy_steps(p)):
        _, _, t_w, t_b = temperatures(y[0], y[1], p)
        worst = min(
            worst, float(growth_rate_unclipped(t_w, p)), float(growth_rate_unclipped(t_b, p))
        )
        y = rk4_step(rhs, y, p.dt)
    assert worst < 0.0, "the clip was expected to bite at L = 0.8 and did not"

    reference_end = transient_endpoint(0.8, 1e-3)
    errors = [float(np.abs(transient_endpoint(0.8, dt) - reference_end).max()) for dt in ORDER_DTS]
    ratios = [c / f for c, f in zip(errors[:-1], errors[1:], strict=True)]
    assert not all(12.0 < r < 20.0 for r in ratios), (
        f"order 4 survived a C^0 kink at L = 0.8 (ratios {ratios}); if this ever "
        "passes, the clip stopped biting and the sibling test's window needs redoing"
    )


# --------------------------------------------------------------------------
# Refusing to answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("luminosity", [0.5, 0.70, 0.7387, 1.3595, 1.5, 2.0])
def test_predictions_refuse_outside_the_regulating_band(luminosity):
    p = reference(luminosity=luminosity)
    with pytest.raises(ValueError, match="no interior equilibrium"):
        interior_equilibrium(p)
    with pytest.raises(ValueError, match="no interior equilibrium"):
        MODEL.analytic_predictions(p)


def test_the_band_is_not_vacuous():
    # Guard against a band so wide that the refusal above could never fire, and so
    # narrow that the tests inside it are not really inside it.
    low, high = regulating_band(reference())
    assert 0.7 < low < 0.8 < 1.2 < high < 1.4
    for luminosity in (0.75, 1.0, 1.35):
        assert np.all(interior_equilibrium(reference(luminosity=luminosity)) > 0.0)


def test_params_refuse_an_inverted_albedo_contrast():
    with pytest.raises(ValueError, match="must exceed albedo_black"):
        DaisyworldParams(albedo_white=0.25, albedo_black=0.75)
    with pytest.raises(ValueError, match="must exceed albedo_black"):
        DaisyworldParams(albedo_white=0.5, albedo_black=0.5)


def test_bare_fraction_refuses_when_growth_cannot_outpace_death():
    # A weaker heat-transfer coefficient pulls the two daisy temperatures together
    # -- fine -- but a much stronger one pushes them out past the parabola's roots,
    # where beta(T_w*) is clipped to zero and the implied bare fraction is infinite.
    with pytest.raises(ValueError, match="does not exceed gamma"):
        bare_fraction(DaisyworldParams(q=4.0e10))
    # ...and a death rate above the achievable growth rate, which is the same
    # refusal reached from the other side.
    with pytest.raises(ValueError, match="does not exceed gamma"):
        bare_fraction(DaisyworldParams(gamma=0.95))


def test_the_equilibrium_is_stable_across_the_band():
    # Both eigenvalues real and negative everywhere inside the band. The slow rate
    # vanishes at BOTH edges -- one species is on its way out there -- so tau runs
    # from 4.33 mid-band to 51 at L = 0.75 and 151 at L = 1.35. That is why the
    # invariance test's luminosities stay in [0.9, 1.1]: at the edges the transient
    # would still be alive at any affordable horizon.
    slow = {}
    for luminosity in (0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35):
        p = reference(luminosity=luminosity)
        eigenvalues = np.linalg.eigvals(jacobian(p))
        assert np.all(eigenvalues.real < 0.0), f"unstable at L={luminosity}: {eigenvalues}"
        assert np.abs(eigenvalues.imag).max() == 0.0, f"complex pair at L={luminosity}"
        slow[luminosity] = 1.0 / slowest_rate(p)
    assert slow[1.0] < 5.0 < slow[0.75] < slow[1.35]


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_observables_cover_every_predicted_key():
    state = MODEL.initial_state(reference(), np.random.default_rng(0))
    obs = MODEL.observables(state)
    assert set(obs) == set(PREDICTED_KEYS)
    assert obs["a_w"] == pytest.approx(0.01)
    assert obs["x_bare"] == pytest.approx(0.98)


def test_params_normalize_a_init_to_a_tuple():
    p = DaisyworldParams(a_init=[0.2, 0.3])
    assert p.a_init == (0.2, 0.3)
    assert p == DaisyworldParams(a_init=(0.2, 0.3))


def test_experiment_round_trips():
    experiment = Experiment(
        model="daisyworld",
        params={"luminosity": 1.0, "a_init": [0.01, 0.01], "t_max": 5.0, "dt": 0.01},
        replicates=1,
        observables=PREDICTED_KEYS,
        seed=1,
        max_steps=600,
    )
    assert Experiment.from_json(experiment.to_json()) == experiment


def test_rejects_a_dt_that_does_not_divide_t_max():
    with pytest.raises(ValueError, match="divide"):
        reference(t_max=1.0, dt=0.3)


def test_the_run_lands_exactly_on_t_max():
    p = reference(t_max=5.0, dt=0.01)
    traj = run_replicate(MODEL, p, np.random.default_rng(0), max_steps=n_daisy_steps(p) + 10)
    assert traj.as_arrays()[0][-1] == pytest.approx(5.0, abs=1e-12)
