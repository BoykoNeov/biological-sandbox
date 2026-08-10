"""Lotka-Volterra: the conserved quantity, the time-average identity, the period.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` step 3.

This is the project's only deterministic model validated **on the orbit** rather
than at a fixed point, so the anchoring is worth stating:

* **Category A, independent.** :func:`textbook_rhs` is hand-transcribed from the
  published equations and shares no code with the model. It is what catches a
  swapped ``beta``/``delta`` or a sign slip — the conserved-``V`` check cannot,
  because a params typo made *consistently* in both the RHS and ``V`` would still
  produce something conserved.
* **Category A, on the orbit.** ``V`` constant, and ``<x> = x*`` over a cycle at
  *any* amplitude. Both hold far from the fixed point, which is what makes them
  stronger than an equilibrium check.
* **Category B.** The small-oscillation period as a **vanishing-amplitude limit**,
  reached by Richardson in the amplitude. Never a tolerance at one amplitude —
  the true period grows with amplitude, so a fixed tolerance would encode the
  amplitude it happened to be measured at.

The cycle-crossing and averaging helpers live here rather than in the model, for
the same reason ``count_spikes`` lives in ``tests/test_hodgkin_huxley.py``: they
are measurement instruments for a claim, not part of the model's contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sandbox.core.protocol import Experiment
from sandbox.core.recorder import run_replicate
from sandbox.core.validation import validate
from sandbox.models.lotka_volterra import (
    MODEL,
    STATE_KEYS,
    LVParams,
    center_eigenvalues,
    conserved_v,
    fixed_point,
    initial_point,
    lv_rhs,
    n_lv_steps,
    small_oscillation_period,
)


def params_factory(d: dict) -> LVParams:
    return LVParams(**d)


def textbook_rhs(y: np.ndarray, p: LVParams) -> np.ndarray:
    """The published Lotka-Volterra equations, hand-transcribed here on purpose.

    Written out longhand from ``dx/dt = alpha x - beta x y``,
    ``dy/dt = delta x y - gamma y`` — the *expanded* form, not the factored one the
    model uses — so a swapped coefficient or a factoring slip shows up as a
    disagreement. This is the independent anchor; conservation of ``V`` is not.
    """
    x, predator = float(y[0]), float(y[1])
    return np.array(
        [
            p.alpha * x - p.beta * x * predator,
            p.delta * x * predator - p.gamma * predator,
        ]
    )


def run_lv(params: LVParams) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    traj = run_replicate(MODEL, params, np.random.default_rng(0), max_steps=n_lv_steps(params) + 10)
    assert traj.terminated, "LV run hit max_steps without terminating"
    return traj.as_arrays()


def upward_crossings(times: np.ndarray, values: np.ndarray, level: float) -> np.ndarray:
    """Times at which ``values`` crosses ``level`` upward, linearly interpolated.

    Interpolating rather than snapping to the grid is the whole point: the
    quantity being measured (a period, a cycle average) has an error floor set by
    *where the cycle is judged to start and end*, not by the integrator, and a
    grid-snapped crossing would quantize it to ``dt``.
    """
    s = values - level
    idx = np.flatnonzero((s[:-1] < 0.0) & (s[1:] >= 0.0))
    if idx.size == 0:
        return np.empty(0, dtype=float)
    frac = -s[idx] / (s[idx + 1] - s[idx])
    return times[idx] + frac * (times[idx + 1] - times[idx])


def measured_period(params: LVParams, n_cycles: int = 8) -> float:
    """Mean period over ``n_cycles``, from interpolated crossings of ``y = y*``.

    Taking ``(t_N - t_0)/N`` rather than averaging N separate differences divides
    the crossing-interpolation error by ``N`` as well.
    """
    times, series = run_lv(params)
    crossings = upward_crossings(times, series["y"], float(fixed_point(params)[1]))
    assert crossings.size > n_cycles, (
        f"only {crossings.size} crossings in t_max = {params.t_max}; need > {n_cycles}"
    )
    return float((crossings[n_cycles] - crossings[0]) / n_cycles)


def cycle_average(params: LVParams) -> tuple[float, float]:
    """``(<x>, <y>)`` over one full interpolated cycle, by the trapezoid rule."""
    times, series = run_lv(params)
    predator = series["y"]
    crossings = upward_crossings(times, predator, float(fixed_point(params)[1]))
    assert crossings.size >= 2, "need at least two crossings to bound one cycle"
    start, end = float(crossings[0]), float(crossings[1])

    inside = (times >= start) & (times <= end)
    grid = np.concatenate([[start], times[inside], [end]])
    out = []
    for key in STATE_KEYS:
        values = series[key]
        edges = np.interp([start, end], times, values)
        column = np.concatenate([[edges[0]], values[inside], [edges[1]]])
        out.append(float(np.trapezoid(column, grid) / (end - start)))
    return out[0], out[1]


# --------------------------------------------------------------------------
# Category A, independent: the RHS and the centre
# --------------------------------------------------------------------------


def test_rhs_matches_the_hand_transcribed_textbook_equations():
    p = LVParams()
    rhs = lv_rhs(p)
    rng = np.random.default_rng(3)
    for _ in range(25):
        y = rng.uniform(0.05, 12.0, size=2)
        assert np.allclose(rhs(y), textbook_rhs(y, p), rtol=1e-13, atol=0.0)


def test_the_fixed_point_is_a_root_and_matches_the_closed_form():
    p = LVParams()
    y_star = fixed_point(p)
    assert y_star == pytest.approx([4.0, 2.75], rel=1e-14)
    assert np.abs(textbook_rhs(y_star, p)).max() < 1e-15


def test_the_fixed_point_is_a_centre_not_an_attractor():
    # Purely imaginary eigenvalues +- i sqrt(alpha gamma). This is *why* the model
    # is validated on the orbit: nothing decays, so there is no relaxation rate and
    # no attractor -- and why it is not a two-species `glv` case, whose stability
    # guard would (correctly) refuse a centre.
    p = LVParams()
    eigenvalues = center_eigenvalues(p)
    assert np.abs(eigenvalues.real).max() < 1e-14, "a centre must have zero real part"
    expected = math.sqrt(p.alpha * p.gamma)
    assert sorted(np.abs(eigenvalues.imag)) == pytest.approx([expected, expected], rel=1e-14)
    assert expected == pytest.approx(0.6633249581, rel=1e-9)


# --------------------------------------------------------------------------
# Category A, on the orbit: the conserved quantity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amp", [0.4, 4.0])
def test_validate_conserves_v(amp):
    # Routed through the ValidationSuite (non-negotiable #2), with the sem_floor
    # DERIVED per amplitude by Richardson in dt. That per-amplitude derivation is
    # not ceremony: the drift is 9.3e-15 at amp = 0.4 and 1.6e-11 at amp = 4.0 --
    # a factor of 1670 at identical dt -- so a floor measured at one amplitude
    # would be either vacuous or impossible at the other. Two replicates, because
    # one gives sem = inf and passes vacuously.
    base = {"amp": amp, "t_max": 100.0, "dt": 0.01}

    def final_v(dt: float) -> float:
        _, series = run_lv(LVParams(**{**base, "dt": dt}))
        return float(series["V"][-1])

    bound = abs(final_v(0.01) - final_v(0.005))
    experiment = Experiment(
        model="lotka_volterra",
        params=base,
        replicates=2,
        observables=("V",),
        seed=0,
        max_steps=n_lv_steps(LVParams(**base)) + 10,
    )
    report = validate(experiment, params_factory, z=4.0, sem_floor=max(bound, 1e-15))
    assert report.passed, str(report)
    assert [c.name for c in report.checks] == ["V"]


@pytest.mark.parametrize("amp", [0.4, 4.0])
def test_v_is_conserved_along_the_whole_orbit_not_just_at_the_endpoint(amp):
    # The endpoint check above could in principle be passed by a drift that wanders
    # off and happens to come back. This bounds the WHOLE series against the
    # endpoint dt-difference: both are the same O(dt^4) integration error, so their
    # ratio is O(1), and the claim is "V's wandering along the orbit is no worse
    # than the integration error the endpoint check already sees".
    #
    # The 50 is the one typed factor in 3a, so here is what it is worth. Measured
    # excursion/endpoint_bound = 2.97 / 12.63 / 1.56 / 1.10 at amp = 0.4 / 1.2 /
    # 4.0 / 8.0 -- so 50 carries 17x and 32x slack at the two amplitudes tested,
    # and 4x at the worst point across a 20x amplitude range. NOT monotone in
    # amplitude, which is why the tested pair does not bracket it and the range was
    # measured rather than assumed.
    #
    # The obvious "more derived" alternative was measured and REJECTED: bounding
    # the finer series' excursion against the coarse-fine difference gives ratios
    # of 0.05-0.07, because excursion(dt/2) is ~5% of excursion(dt) and the
    # difference is therefore dominated by the coarse term. That bound collapses
    # into "the finer run is not much worse than the coarser one" -- a threshold
    # almost nothing can fail. A derived-looking form is not automatically the
    # stronger one.
    def series_v(dt: float) -> np.ndarray:
        _, series = run_lv(LVParams(amp=amp, t_max=100.0, dt=dt))
        return np.asarray(series["V"])

    coarse = series_v(0.01)
    excursion = float(np.abs(coarse - coarse[0]).max())
    endpoint_bound = abs(float(coarse[-1]) - float(series_v(0.005)[-1]))
    assert excursion <= 50.0 * max(endpoint_bound, 1e-15), (
        f"V wanders by {excursion:.3e} along the orbit, far above the "
        f"{endpoint_bound:.3e} the endpoint check would see"
    )
    assert excursion < 1e-8, "V is not conserved to anything like the claimed precision"


# --------------------------------------------------------------------------
# Category A, on the orbit: the time-average identity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amp", [0.4, 1.2, 4.0])
def test_the_cycle_average_equals_the_fixed_point_at_any_amplitude(amp):
    # <x> = x* and <y> = y* hold for EVERY closed orbit, not just small ones -- so
    # this is checked across a 10x amplitude range. At amp = 4.0 the orbit runs slow
    # near the axes and the cycle is strongly non-sinusoidal, which is exactly where
    # a check that only worked near equilibrium would break.
    #
    # The tolerance is derived: the error floor is the cycle-ENDPOINT detection (a
    # linear crossing interpolation plus a trapezoid, both O(dt^2)), so the
    # difference between dt and dt/2 bounds what remains at dt/2. Measured at
    # dt = 5e-3 -> 2.5e-3: |<x> - x*| of 3.7e-9 / 3.7e-9 / 4.0e-8 against
    # differences of 4.0e-9 / 3.0e-8 / 6.2e-8.
    x_star, y_star = fixed_point(LVParams(amp=amp))
    coarse = cycle_average(LVParams(amp=amp, t_max=30.0, dt=0.005))
    fine = cycle_average(LVParams(amp=amp, t_max=30.0, dt=0.0025))

    for label, star, c, f in (("x", x_star, coarse[0], fine[0]), ("y", y_star, coarse[1], fine[1])):
        difference = abs(c - f)
        residual = abs(f - star)
        assert residual <= 3.0 * max(difference, 1e-14), (
            f"<{label}> = {f:.12f} sits {residual:.3e} from {star}, above 3x the "
            f"dt-difference {difference:.3e} that bounds the endpoint-detection error"
        )
    # Companion absolute bound, so the check cannot pass vacuously if the two dt
    # ever disagree wildly and inflate the derived bound.
    assert abs(fine[0] - x_star) < 1e-6 and abs(fine[1] - y_star) < 1e-6


# --------------------------------------------------------------------------
# Category B: the period as a vanishing-amplitude LIMIT
# --------------------------------------------------------------------------


def test_the_period_grows_with_amplitude():
    # The structural fact that forces the extrapolation. If the period were
    # amplitude-independent, a tolerance at one amplitude would be legitimate and
    # the whole Richardson apparatus below would be unnecessary ceremony. It is not:
    # measured 9.4726 / 9.4733 / 9.4773 / 9.4913 / 9.8053 at amp = 0.1 ... 4.0.
    periods = [measured_period(LVParams(amp=a, t_max=100.0, dt=0.01)) for a in (0.1, 0.4, 4.0)]
    assert periods == sorted(periods), f"period not monotone in amplitude: {periods}"
    assert periods[0] < periods[-1], "period is amplitude-independent -- extrapolation is moot"


def test_richardson_in_the_amplitude_extrapolates_to_the_closed_form_period():
    # THE claim of this file, and Richardson in the amplitude for the fifth time in
    # the project. The excess is O(amp^2) (measured excess/amp^2 = 0.033360 /
    # 0.033087 / 0.032558 / 0.031555 at amp = 0.05 / 0.1 / 0.2 / 0.4 -- converging,
    # unlike the plan's 1.2 / 2.0 / 4.0 probes, whose implied coefficients run
    # 0.0282 / 0.0255 / 0.0208 and are nowhere near constant). So the order-2
    # extrapolant is E(a) = (4 P(a/2) - P(a)) / 3.
    #
    # Everything below is derived from the three extrapolants themselves: the
    # residual's convergence RATIO is measured (~7.6, i.e. the leading leftover is
    # O(a^3)), and that ratio predicts the finest extrapolant's own error. Measured
    # E - P0 = 5.35e-5 / 7.06e-6 / 9.08e-7 for a = 0.4 / 0.2 / 0.1, and the
    # prediction lands within 3% of the truth.
    #
    # dt is NOT a confound here: the period estimator is dt-converged well before
    # this point -- it moves by under 6e-9 from dt = 2e-2 down to dt = 5e-4, three
    # orders below the smallest amplitude effect being measured (8.3e-5).
    p0 = small_oscillation_period(LVParams())
    periods = {
        a: measured_period(LVParams(amp=a, t_max=100.0, dt=0.01)) for a in (0.4, 0.2, 0.1, 0.05)
    }
    extrapolant = {a: (4.0 * periods[a / 2] - periods[a]) / 3.0 for a in (0.4, 0.2, 0.1)}

    # The extrapolants must themselves be converging, or the ratio below is noise.
    gap_coarse = abs(extrapolant[0.4] - extrapolant[0.2])
    gap_fine = abs(extrapolant[0.2] - extrapolant[0.1])
    ratio = gap_coarse / gap_fine
    assert ratio > 4.0, f"the extrapolants are not converging (successive gaps ratio {ratio:.2f})"

    predicted_error = gap_fine / (ratio - 1.0)
    actual_error = abs(extrapolant[0.1] - p0)
    assert actual_error <= 2.0 * predicted_error, (
        f"the amplitude-extrapolated period {extrapolant[0.1]:.12f} sits "
        f"{actual_error:.3e} from the closed form {p0:.12f}, above 2x the "
        f"{predicted_error:.3e} its own convergence predicts"
    )
    # And the extrapolation must actually buy something: the raw period at the
    # smallest amplitude is 10x further from the closed form than the extrapolant.
    assert actual_error < abs(periods[0.05] - p0) / 10.0


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_observables_cover_the_state_plus_the_invariant():
    obs = MODEL.observables(MODEL.initial_state(LVParams(), np.random.default_rng(0)))
    assert set(obs) == {"x", "y", "V"}
    assert obs["x"] == pytest.approx(4.4)  # x* + amp
    assert obs["y"] == pytest.approx(2.75)


def test_v_is_nan_outside_the_positive_quadrant():
    # Not an exception: a closed orbit never leaves the quadrant, so a nan here
    # reports that the trajectory has escaped rather than hiding it behind a
    # convenient number.
    assert math.isnan(conserved_v(np.array([0.0, 1.0]), LVParams()))
    assert math.isnan(conserved_v(np.array([1.0, -0.5]), LVParams()))


def test_the_initial_point_sits_on_the_predator_nullcline():
    # y0 = y* exactly, so the orbit starts on the crossing line the period and cycle
    # detectors use. That is deliberate -- it makes the first detected crossing a
    # full period later, with no partial cycle to trim.
    p = LVParams(amp=1.5)
    assert initial_point(p)[1] == pytest.approx(fixed_point(p)[1], rel=0, abs=0.0)


def test_rejects_a_non_positive_rate():
    with pytest.raises(ValueError, match="positive"):
        LVParams(beta=0.0)


def test_rejects_an_amplitude_that_leaves_the_quadrant():
    with pytest.raises(ValueError, match="positive quadrant"):
        LVParams(amp=-5.0)


def test_rejects_a_dt_that_does_not_divide_t_max():
    with pytest.raises(ValueError, match="divide"):
        LVParams(t_max=1.0, dt=0.3)


def test_experiment_round_trips():
    experiment = Experiment(
        model="lotka_volterra",
        params={"alpha": 1.1, "beta": 0.4, "gamma": 0.4, "delta": 0.1, "amp": 1.2, "t_max": 20.0},
        replicates=1,
        observables=("x", "y", "V"),
        seed=1,
        max_steps=2100,
    )
    assert Experiment.from_json(experiment.to_json()) == experiment


def test_the_run_lands_exactly_on_t_max():
    times, _ = run_lv(LVParams(t_max=5.0, dt=0.01))
    assert times[-1] == pytest.approx(5.0, abs=1e-12)
    assert times.size == 501
