"""Convergence pathway: the ``Omega^{-1/2}`` scaling check and its machinery.

This is the *second* validation track (the first being ``validate()`` against a
scalar closed form). It exists for models whose deterministic limit has no scalar
prediction — the repressilator's limit cycle — so it is exercised here on
``birth_death``, whose *linear* dynamics make the finite-size fluctuation
``Var(x) = (k/gamma)/Omega`` **exact**, giving a clean ``-1/2`` slope at modest
``Omega`` (cleaner than the repressilator will be). Building the machinery against
that exact case lets us pin it before step 6 hangs the headline on it.

The pathway's one load-bearing subtlety is that ``D(Omega)`` averages the
*per-replicate* discrepancy, **never** the error of the ensemble-mean trajectory
(the phase-diffusion trap). Because that bug is *slope-invariant* for a linear
model (it is just ``D_correct / sqrt(R)`` — same slope), the slope check alone
cannot catch it. So the teeth here are elsewhere and non-statistical:

* :func:`_per_replicate_discrepancy` is unit-tested on a hand-worked example where
  the per-replicate mean and the mean-first-then-error give *different* numbers;
* the ``D(Omega)*sqrt(Omega)`` **magnitude anchor** is checked against the exact
  ``sqrt(2/pi)*<sqrt(c(t))>`` — the mean-first bug would shrink it by ``~sqrt(R)``
  (``~11x`` at ``R=120``), far outside the tolerance;
* ``report.discrepancy == report.per_replicate.mean(axis=1)`` is asserted on the
  real run, proving the routine averages per replicate.

The broken-``Omega``-scaling slope teeth (a wrong propensity driving the slope off
``-1/2``) belong with the repressilator in step 6, mirroring the Phase-0
wrong-prediction guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import (
    _per_replicate_discrepancy,
    _sample_on_grid,
    check_grid_is_exact,
    convergence_report,
)
from sandbox.models.birth_death import BirthDeathParams
from sandbox.models.hh_stochastic import HHStochasticParams

# Birth-death convergence config. Omega spans exactly one decade (16 -> 256) with
# 5 points; T = 6 = 6/gamma so the run is well past relaxation and T << Omega
# across the whole sweep (no phase-saturation knee — birth-death has no phase to
# diffuse anyway, being a stable fixed point). record_every defaults to 1: a slope
# check must record every event (sub-sampling would sharpen interpolation *with*
# Omega and bias the slope). Config validated empirically: slope lands within a
# few 0.01 of -0.5 with slope_se ~ 0.011, so z=3 passes with a wide margin while
# still rejecting any slope outside ~[-0.53, -0.47].
_BASE = {"k": 2.0, "gamma": 1.0, "c0": 0.0}
_OMEGAS = [16.0, 32.0, 64.0, 128.0, 256.0]
_T_MAX = 6.0
_DT = 0.01
_N_GRID = 200
_REPLICATES = 120
_Z = 3.0


def _factory(d: dict) -> BirthDeathParams:
    return BirthDeathParams(**d)


# ---------------------------------------------------------------------------
# Pure machinery (deterministic, no statistics) — the real teeth
# ---------------------------------------------------------------------------


def test_sample_on_grid_holds_last_event_value():
    # A Gillespie trajectory is piecewise-constant: a value holds until the next
    # event. Sampling must return the last event at or before each grid time.
    times = np.array([0.0, 2.0, 5.0])
    values = np.array([10.0, 20.0, 30.0])
    grid = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    got = _sample_on_grid(times, values, grid)
    # t=1 still holds the t=0 value; t=2 fires to 20; t=3 holds 20; t=5 fires to 30.
    assert got.tolist() == [10.0, 10.0, 20.0, 20.0, 30.0]


def test_per_replicate_discrepancy_matches_hand_computation():
    # One replicate, one species, constant-zero ODE. A trajectory pinned at +1 has
    # |1 - 0| = 1 at every grid time, so the discrepancy is exactly 1.
    grid = np.array([0.0, 1.0])
    ode = np.zeros((2, 1))
    d = _per_replicate_discrepancy(
        np.array([0.0, 1.0]), {"x": np.array([1.0, 1.0])}, grid, ode, ("x",)
    )
    assert d == pytest.approx(1.0)


def test_per_replicate_averaging_differs_from_mean_first():
    """The trap made explicit: two anti-symmetric replicates (+1 and -1) each have
    discrepancy 1, so the *per-replicate* mean is 1 — but averaging the trajectories
    *first* cancels them to 0. The pathway must report 1, not 0."""
    grid = np.array([0.0, 1.0])
    ode = np.zeros((2, 1))
    up = _per_replicate_discrepancy(
        np.array([0.0, 1.0]), {"x": np.array([1.0, 1.0])}, grid, ode, ("x",)
    )
    down = _per_replicate_discrepancy(
        np.array([0.0, 1.0]), {"x": np.array([-1.0, -1.0])}, grid, ode, ("x",)
    )
    per_replicate_mean = 0.5 * (up + down)
    # Mean-first: average the two trajectories (+1, -1) -> 0, then |0 - 0| = 0.
    mean_first_trajectory = np.array([0.0, 0.0])
    mean_first = float(np.abs(mean_first_trajectory - ode[:, 0]).mean())
    assert per_replicate_mean == pytest.approx(1.0)
    assert mean_first == pytest.approx(0.0)
    assert per_replicate_mean != pytest.approx(mean_first)


def test_multi_species_discrepancy_averages_over_species_and_time():
    # Two species, constant ODE at [0, 0]; trajectory constant at [1, 3]; mean of
    # |[1,3] - [0,0]| over 2 species and 2 grid times = (1 + 3) / 2 = 2.
    grid = np.array([0.0, 1.0])
    ode = np.zeros((2, 2))
    series = {"a": np.array([1.0, 1.0]), "b": np.array([3.0, 3.0])}
    d = _per_replicate_discrepancy(np.array([0.0, 1.0]), series, grid, ode, ("a", "b"))
    assert d == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Grid alignment — the Phase-2 addition, and why it is checked not assumed
# ---------------------------------------------------------------------------


def test_check_grid_is_exact_accepts_a_subset_of_the_recorded_times():
    dt, n_steps, stride = 0.025, 1600, 8
    times = np.arange(n_steps + 1, dtype=float) * dt
    grid = np.arange(0, n_steps + 1, stride, dtype=float) * dt
    check_grid_is_exact(times, grid)  # must not raise
    # ...and the sampling really is a no-op: the sampled values ARE the recorded
    # ones, so the step-hold contributes no error at all rather than a small one.
    values = np.sin(times)
    assert np.array_equal(_sample_on_grid(times, values, grid), values[::stride])


def test_check_grid_is_exact_rejects_a_grid_one_ulp_low():
    """One bit of drift is a whole step of error, which is the point of checking.

    ``linspace(0, t_max, n_grid)`` looks like it lands on the recorded times and
    mostly does — but where it falls a single ulp short, ``searchsorted`` returns
    the *previous* recorded index. The resulting error does not shrink with
    ``Omega``, so it would floor ``D(Omega)`` and flatten the slope while looking
    like physics. Hence a check, not a comment.
    """
    times = np.arange(1601, dtype=float) * 0.025
    grid = np.arange(0, 1601, 8, dtype=float) * 0.025
    nudged = grid.copy()
    nudged[5] = np.nextafter(nudged[5], -np.inf)
    with pytest.raises(ValueError, match="not exactly recorded times"):
        check_grid_is_exact(times, nudged)
    # And the damage it would have done, made concrete: a whole step backwards.
    assert _sample_on_grid(times, times, nudged)[5] == times[8 * 5 - 1]


def test_linspace_alignment_is_a_coincidence_of_the_numbers():
    """A ``linspace`` grid lands on the recorded times *sometimes*, and that is worse
    than never.

    The first version of this test asserted that ``linspace`` misaligns, and it
    failed — at ``n_steps = 1600, stride = 8`` it aligns perfectly. The reason is
    the whole lesson: ``linspace`` computes ``i * (t_max / (n_grid - 1))``, so it
    matches ``(i * stride) * dt`` exactly when ``t_max / (n_grid - 1)`` is exactly
    ``stride * dt`` in binary. Dividing by a **power of two** is exact, so stride
    8 works and stride 3 does not. Swept over the plausible configurations, 373
    of them misalign.

    So the hazard is not "``linspace`` is wrong" but "``linspace`` is right until
    somebody changes ``n_grid`` from 201 to 121", and the symptom is a flattened
    slope rather than an error. That is precisely a case for a check instead of a
    convention.
    """
    dt, n_steps = 0.025, 120
    times = np.arange(n_steps + 1, dtype=float) * dt

    aligned = np.linspace(0.0, n_steps * dt, n_steps // 8 + 1)
    assert np.array_equal(aligned, times[::8])
    check_grid_is_exact(times, aligned)  # power-of-two stride: fine, by luck

    misaligned = np.linspace(0.0, n_steps * dt, n_steps // 3 + 1)
    assert int(np.count_nonzero(misaligned != times[::3])) == 25
    with pytest.raises(ValueError, match="not exactly recorded times"):
        check_grid_is_exact(times, misaligned)


# ---------------------------------------------------------------------------
# compare_keys — comparing a subset of the limit's components
# ---------------------------------------------------------------------------


def _hh_factory(d: dict) -> HHStochasticParams:
    return HHStochasticParams(**d)


_HH_KEYS = ("V", "m", "h", "n")
_HH_BASE = {"i_ext": 0.0, "dt": 0.05, "v0": -65.0}


def test_compare_keys_rejects_a_name_that_is_not_a_limit_component():
    with pytest.raises(ValueError, match="not in observable_keys"):
        convergence_report(
            "hh_stochastic",
            _HH_BASE,
            _hh_factory,
            omegas=[1e4, 2e4],
            t_max=0.5,
            dt=0.01,
            replicates=2,
            observable_keys=_HH_KEYS,
            compare_keys=("V", "na_open"),
            omega_key="n_channels",
            n_bootstrap=10,
        )


def test_compare_keys_lets_a_model_report_fewer_series_than_its_limit_has():
    """A channel-state model has ``V`` and two occupancy fractions — no ``m``/``h``/``n``.

    Its limit is nonetheless the full 4-D Hodgkin-Huxley ODE, so
    ``observable_keys`` must name four columns while the *comparison* uses one.
    Without ``compare_keys`` the pathway rejected this outright (the one-to-one
    length guard), which is why the Phase-2 plan's "just pass
    ``observable_keys=('V',)``" could not work as written.
    """
    report = convergence_report(
        "hh_stochastic",
        _HH_BASE,
        _hh_factory,
        omegas=[2e4, 8e4],
        t_max=1.0,
        dt=0.01,
        replicates=3,
        grid=np.arange(0, 21, 4, dtype=float) * 0.05,
        require_exact_grid=True,
        observable_keys=_HH_KEYS,
        compare_keys=("V",),
        omega_key="n_channels",
        n_bootstrap=10,
    )
    assert report.discrepancy.shape == (2,)
    assert np.all(np.isfinite(report.discrepancy))
    # Not requested, so it must be inert rather than a silent False.
    assert np.isnan(report.stochastic_delta) and report.stochastic_floor_ok


def test_stochastic_floor_check_runs_and_reports():
    """Plumbing for the symmetric floor check: halve the *model's own* step.

    Correctness of the criterion is exercised by the Hodgkin-Huxley slope test,
    which runs it in the regime it was designed for. Here it only has to happen
    and produce a finite number that reaches ``passed``.
    """
    report = convergence_report(
        "hh_stochastic",
        _HH_BASE,
        _hh_factory,
        omegas=[2e4, 8e4],
        t_max=1.0,
        dt=0.01,
        replicates=3,
        grid=np.arange(0, 21, 4, dtype=float) * 0.05,
        require_exact_grid=True,
        observable_keys=_HH_KEYS,
        compare_keys=("V",),
        omega_key="n_channels",
        stochastic_dt_key="dt",
        n_bootstrap=10,
    )
    assert np.isfinite(report.stochastic_delta)
    assert "stochastic_floor_ok" in str(report)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_non_deterministic_limit_model_is_rejected():
    # wright_fisher has no deterministic_rhs/initial_concentrations, so it has no
    # ODE limit to converge to; the pathway must refuse it loudly.
    with pytest.raises(TypeError):
        convergence_report(
            "wright_fisher",
            {"N": 100, "s": 0.0, "p0": 0.5},
            lambda d: None,
            omegas=[10.0, 20.0],
            t_max=1.0,
            dt=0.1,
            replicates=2,
        )


def test_truncated_horizon_is_rejected():
    # max_steps far too small: no replicate reaches t_max, so the time-average
    # would cover a truncated horizon. The anti-bias guard must raise (mirrors
    # validate()'s require_termination).
    with pytest.raises(ValueError, match="max_steps"):
        convergence_report(
            "birth_death",
            _BASE,
            _factory,
            omegas=[16.0, 32.0],
            t_max=_T_MAX,
            dt=_DT,
            replicates=4,
            max_steps=3,
            n_bootstrap=50,
        )


# ---------------------------------------------------------------------------
# The scaling law (the definition of "done" for the pathway)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2])
def test_birth_death_discrepancy_scales_as_omega_minus_half(seed):
    report = convergence_report(
        "birth_death",
        _BASE,
        _factory,
        omegas=_OMEGAS,
        t_max=_T_MAX,
        dt=_DT,
        replicates=_REPLICATES,
        n_grid=_N_GRID,
        seed=seed,
        z=_Z,
        n_bootstrap=300,
    )
    assert report.passed, str(report)
    # Slope consistent with -1/2 and significantly negative; reference not floored.
    assert report.consistent and report.significant and report.reference_ok
    assert abs(report.slope + 0.5) <= _Z * report.slope_se

    # Routine averages the discrepancy *per replicate*, not mean-first.
    assert np.allclose(report.discrepancy, report.per_replicate.mean(axis=1))

    # Magnitude anchor: for birth-death from empty, n(t) ~ Poisson(Omega*c(t)), so
    # E|x-c| = sqrt(2/(pi*Omega)) * sqrt(c(t)); time-averaged,
    #   D(Omega)*sqrt(Omega) ~= sqrt(2/pi) * <sqrt(c(t))>_t.
    # The mean-first bug would shrink this by ~sqrt(R) ~ 11x — far outside 25%.
    grid = np.linspace(0.0, _T_MAX, _N_GRID)
    c = (_BASE["k"] / _BASE["gamma"]) * (1.0 - np.exp(-_BASE["gamma"] * grid))
    expected_anchor = np.sqrt(2.0 / np.pi) * np.sqrt(c).mean()
    anchor = report.discrepancy * np.sqrt(report.omegas)
    assert np.all(np.abs(anchor / expected_anchor - 1.0) < 0.25), (
        f"anchor={anchor}, expected~{expected_anchor:.3f}"
    )
