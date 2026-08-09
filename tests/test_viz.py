"""Visualization backend — the two figures Phase 1 hands to the demo.

Plots are usually treated as untestable decoration. They are not: both helpers
here carry a claim that can be silently wrong *and still produce a plausible
picture*, which is the worst failure mode in a project whose whole point is that
the check is real.

* :func:`plot_replicates` overlays a deterministic limit on a stochastic cloud.
  The two must be in the **same units** (``observables()`` returns concentrations
  ``x = n/Omega``, and the ODE is integrated in concentration space) and the ODE
  column must be the one matching the plotted observable. A counts-vs-concentration
  mix-up is off by ``Omega``; a permuted protein column is the *right shape at the
  wrong phase*, which on a symmetric limit cycle looks entirely convincing. Both
  are checked here against a real replicate, with the tolerance measured across
  seeds rather than guessed.
* :func:`plot_convergence` draws ``D(Omega)`` against the ``Omega^{-1/2}`` law.
  Its fit must use only the masked points (the low-``Omega`` phase-saturation knee
  is excluded from the fit but still *drawn*, since it is evidence), and its guide
  line must be anchored to those fitted points — anchored at the knee it would
  visually miss the data and make a passing check look like a failing one. These
  are checked on **synthetic** arrays: the helper takes plain numbers precisely so
  its own test costs no SSA time.
"""

from __future__ import annotations

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import _sample_on_grid
from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import Experiment
from sandbox.core.sweep import run_experiment
from sandbox.models.repressilator import MODEL, OBSERVABLE_KEYS, RepressilatorParams

matplotlib = pytest.importorskip("matplotlib", reason="viz extra not installed")
matplotlib.use("Agg")  # headless: no display, no blocking show()

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from sandbox.viz.backends.matplotlib_backend import (  # noqa: E402
    EXCLUDED_POINTS_GID,
    FIT_GID,
    GUIDE_GID,
    plot_convergence,
    plot_replicates,
)

PERIOD = 16.095  # measured; see tests/test_repressilator.py
_BASE = {
    "alpha": 216.0,
    "alpha0": 0.216,
    "n_hill": 2.0,
    "beta": 1.0,
    "m0": 0.0,
    "p1_0": 0.0,
    "p2_0": 5.0,
    "p3_0": 15.0,
}


@pytest.fixture
def close_figures():
    """Close every figure the test opened (Agg still leaks them into pyplot's registry)."""
    yield
    plt.close("all")


def _line_by_gid(ax, gid: str):
    for line in ax.get_lines():
        if line.get_gid() == gid:
            return line
    raise AssertionError(f"no line with gid {gid!r}; got {[ln.get_gid() for ln in ax.get_lines()]}")


def _loglog_slope(line) -> float:
    """Slope of a drawn straight line, read back in log-log space."""
    xy = line.get_xydata()
    lx, ly = np.log(xy[:, 0]), np.log(xy[:, 1])
    return float((ly[-1] - ly[0]) / (lx[-1] - lx[0]))


# ---------------------------------------------------------------------------
# plot_replicates: the ODE overlay is in the same units and the same column
# ---------------------------------------------------------------------------


def test_ode_overlay_tracks_a_real_replicate_in_matching_units(close_figures):
    """One replicate at ``Omega=20`` must hug the ODE column it is plotted against.

    This is the whole content of "confirm the overlay's units match", made
    checkable. The data is read back **off the Axes**, so it also pins that
    ``plot_replicates`` passes both series through unrescaled.

    Two teeth ride along, computed from the same replicate (so they cost nothing):
    comparing against the *neighbouring* ODE column — a ~1/3-period phase shift on
    this cyclically symmetric limit cycle — and against counts instead of
    concentrations.

    Thresholds are measured, not guessed. Discrepancy over seeds 0-3 (as a fraction
    of the ODE amplitude): 0.097, 0.010, 0.023, 0.011 — phase-diffusion luck spans
    10x, so the pinned seed is **0**, the worst of the four, and the bound is 0.20
    (2x margin over that worst case). The wrong column sits at ~0.47 amplitude at
    every seed, i.e. 4.9x-45x the correct discrepancy; asserting 2.5x keeps 2x
    margin over the worst.
    """
    omega = 20.0
    observable = "x_p2"
    params = RepressilatorParams(Omega=omega, t_max=PERIOD, **_BASE)

    result = run_experiment(
        Experiment(
            model="repressilator",
            params={**_BASE, "Omega": omega, "t_max": PERIOD},
            replicates=1,
            observables=OBSERVABLE_KEYS,
            seed=0,
            max_steps=2_000_000,
            record_every=1,
        ),
        lambda d: RepressilatorParams(**d),
    )
    trajectory = result.trajectories[0][0]

    t_ode, y_ode = integrate_rk4(
        MODEL.deterministic_rhs(params), MODEL.initial_concentrations(params), PERIOD, 1e-2
    )
    # The index must come from the ordered keys, never a literal: this is the line
    # the demo has to get right.
    idx = OBSERVABLE_KEYS.index(observable)
    ax = plot_replicates([trajectory], observable, deterministic=(t_ode, y_ode[:, idx]), alpha=0.5)

    drawn = {line.get_label(): line.get_xydata() for line in ax.get_lines()}
    det = drawn["deterministic limit"]
    replicate = next(xy for label, xy in drawn.items() if label != "deterministic limit")
    assert replicate.shape[0] > 1000, "the replicate was not drawn (wrong observable key?)"

    grid = np.linspace(0.0, PERIOD, 400)
    # Step-interpolate the *drawn* SSA series (piecewise-constant between events).
    stoch = _sample_on_grid(replicate[:, 0], replicate[:, 1], grid)
    ode = np.interp(grid, det[:, 0], det[:, 1])
    amplitude = float(np.ptp(ode))

    discrepancy = float(np.abs(stoch - ode).mean())
    assert discrepancy < 0.20 * amplitude, (
        f"replicate does not track the overlaid ODE: {discrepancy:.3g} vs amplitude {amplitude:.3g}"
    )
    # Tracking alone catches a units mix-up only weakly: rescaling the overlay by
    # Omega inflates `amplitude` too, so the ratio above moves by just 2.4x. The
    # two *scales* do not cancel, so compare them head-on. Std, not peak-to-peak:
    # ptp is a max-statistic and a single SSA spike moves it (deviation across seeds
    # 0-3 is 0.32/0.07/0.10/0.08 for ptp against 0.19/0.04/0.02/0.04 for std), so
    # the 0.4 bound keeps a 2.1x margin here where ptp would leave 1.2x. An overlay
    # drawn in counts is off by a factor of Omega = 20.
    assert float(np.std(stoch)) == pytest.approx(float(np.std(ode)), rel=0.4), (
        "the overlaid ODE has a different scale than the replicate — a units mismatch"
    )

    wrong_column = float(np.abs(stoch - np.interp(grid, t_ode, y_ode[:, idx + 1])).mean())
    assert wrong_column > 2.5 * discrepancy, (
        f"a permuted ODE column is indistinguishable here ({wrong_column:.3g} vs {discrepancy:.3g})"
        " — the test cannot see a mis-indexed overlay"
    )

    counts = float(np.abs(stoch * omega - ode).mean())
    assert counts > 10.0 * discrepancy, "counts vs concentrations is not distinguishable"


def test_plot_replicates_draws_nothing_for_an_unknown_observable(close_figures):
    # Documented behaviour, and the reason the test above asserts the replicate was
    # actually drawn: a typo'd key yields a figure with only the deterministic line,
    # which looks like a perfectly clean run.
    params = RepressilatorParams(Omega=1.0, t_max=1.0, **_BASE)
    result = run_experiment(
        Experiment(
            model="repressilator",
            params={**_BASE, "Omega": 1.0, "t_max": 1.0},
            replicates=1,
            observables=("x_m1",),
            seed=0,
            max_steps=100_000,
        ),
        lambda d: RepressilatorParams(**d),
    )
    ax = plot_replicates(result.trajectories[0], "x_typo", deterministic=([0.0, 1.0], [1.0, 1.0]))
    assert len(ax.get_lines()) == 1  # the deterministic line only
    assert MODEL.initial_concentrations(params).size == 6  # params sanity, cheap


# ---------------------------------------------------------------------------
# plot_convergence: synthetic arrays only (no SSA — the suite is already 155 s)
# ---------------------------------------------------------------------------

_C = 26.0  # the repressilator's measured D*sqrt(Omega) plateau


def _law(omegas: np.ndarray) -> np.ndarray:
    return _C * omegas**-0.5


def test_plot_convergence_recovers_the_synthetic_slope(close_figures):
    omegas = np.array([2.0, 4.0, 8.0, 16.0])
    ax = plot_convergence(omegas, _law(omegas))
    assert _loglog_slope(_line_by_gid(ax, FIT_GID)) == pytest.approx(-0.5, abs=1e-6)
    assert _loglog_slope(_line_by_gid(ax, GUIDE_GID)) == pytest.approx(-0.5, abs=1e-6)


def test_plot_convergence_fits_only_the_masked_points_but_still_draws_the_knee(close_figures):
    """The knee must be excluded from the fit and visible in the figure.

    Synthetic version of the real repressilator sweep: the law holds for
    ``Omega >= 2`` and saturates below it (phase saturation caps ``D`` near the
    fully-dephased amplitude). Fitting through the knee biases the slope shallow —
    asserted here so the test fails if the mask is ignored.
    """
    omegas = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    d = _law(omegas)
    knee = omegas < 2.0
    d[knee] = _law(np.array([2.0]))[0] * 1.4  # flat, saturated plateau
    mask = ~knee

    ax = plot_convergence(omegas, d, fit_mask=mask, sem=0.02 * d)

    assert _loglog_slope(_line_by_gid(ax, FIT_GID)) == pytest.approx(-0.5, abs=1e-6)
    # ...and the unmasked fit really is different, so the assertion above has teeth.
    naive = float(np.polyfit(np.log(omegas), np.log(d), 1)[0])
    assert naive > -0.4, f"knee does not bias the naive fit ({naive}); the mask proves nothing here"

    excluded = _line_by_gid(ax, EXCLUDED_POINTS_GID).get_xydata()
    assert np.allclose(np.sort(excluded[:, 0]), omegas[knee]), (
        "excluded points must still be drawn — the knee is evidence about where the law stops"
    )


def test_plot_convergence_anchors_the_guide_line_to_the_fitted_points(close_figures):
    """The ``-1/2`` guide must pass through the fitted data, not through the knee."""
    omegas = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    d = _law(omegas)
    knee = omegas < 2.0
    d[knee] = _law(np.array([2.0]))[0] * 1.4
    mask = ~knee

    ax = plot_convergence(omegas, d, fit_mask=mask)
    guide = _line_by_gid(ax, GUIDE_GID).get_xydata()

    def guide_at(x: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.log(x), np.log(guide[:, 0]), np.log(guide[:, 1])))

    # The masked points lie exactly on the law, so a correctly anchored guide runs
    # straight through them.
    assert np.allclose(guide_at(omegas[mask]), d[mask], rtol=1e-6)
    # Anchoring at the first (saturated) point instead would put the guide a factor
    # ~2 off there; assert it is NOT anchored there.
    assert abs(np.log(guide_at(omegas[:1])[0] / d[0])) > 0.5


def test_plot_convergence_rejects_unplottable_input(close_figures):
    omegas = np.array([1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="differ in length"):
        plot_convergence(omegas, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        plot_convergence(omegas, np.array([1.0, 0.0, 4.0]))
    with pytest.raises(ValueError, match="every point"):
        plot_convergence(omegas, _law(omegas), fit_mask=[False, False, False])
