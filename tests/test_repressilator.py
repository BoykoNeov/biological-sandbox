"""Repressilator — Phase 1's headline model, validated by the convergence pathway.

This model deliberately has **no** ``analytic_predictions``: its deterministic
limit is a *limit cycle*, not a fixed point, so there is no stationary scalar for
:func:`~sandbox.core.validation.validate` to match an ensemble mean against. Its
checkable claim is Kurtz convergence — ``D(Omega) ~ Omega^{-1/2}`` — so "done" here
means :func:`~sandbox.core.convergence.convergence_report` passes.

Because the headline check is statistical and slow, most of the teeth in this file
are **not** the slope:

* the ODE genuinely sustains a limit cycle (amplitude does not decay) — the
  premise everything else rests on, and cheap to check;
* the cyclically symmetric initial condition is *rejected*, because on that
  invariant manifold the ODE does not oscillate at all;
* ``OBSERVABLE_KEYS`` column-matches ``initial_concentrations`` — a silent
  transposition would produce a wrong-but-plausible ``D`` that the slope check
  could not catch;
* ``deterministic_rhs`` reproduces the textbook RHS, pinning the network's
  ``rates``/stoichiometry (which the SSA also uses) to the published equations;
* two **deliberately broken Omega scalings** make the convergence check FAIL, in
  two different ways — see below.

Parameter choice (``alpha=216, alpha0=0.216, n_hill=2, beta=1``) was made from a
measured ODE sweep, not copied blindly. ``beta=5`` — a common textbook value —
*damps to the fixed point* here (m1 amplitude 136 -> 60 over 60 time units) and
does not oscillate; the Elowitz instability condition ``(beta+1)^2/beta < ...`` has
its left side minimized at ``beta = 1``. Measured period: **16.095** mRNA
lifetimes, amplitude ratio last/mid cycle over 19 periods = 1.00000.

Convergence config, and why:

* ``t_max`` = 2 periods. A time-averaged discrepancy over a single oscillation
  inherits phase-alignment noise; cost is linear in ``t_max`` and in replicates,
  but a longer horizon reduces that variance in a way replicates do not.
* ``fit_mask`` excludes ``Omega <= 1``. Those points are in the **phase-saturation
  knee**: replicates fully dephase, so ``D`` is capped near the fully-dephased
  ``O(amplitude)`` value (~45) and cannot keep growing as ``Omega`` falls. The
  signature is ``D*sqrt(Omega)`` dropping *below* the plateau — measured 22.3 and
  24.5 at ``Omega`` = 0.25, 0.5 against a plateau of ~26. Including them biases the
  slope shallow (-0.4477 instead of -0.4606). They are still *run and reported*
  (``convergence_report`` prints excluded points with a blank flag) so the knee
  stays visible as evidence rather than being hidden; they cost ~11% of runtime.
* The fitted ``Omega`` range is pushed up to 16 for **lever arm**, not for a nicer
  answer: the OLS slope SE goes as ``sigma_resid / (sqrt(K) * sd(log Omega))``, so
  widening the range buys precision more cheaply than adding replicates (measured:
  same ``R``, SE 0.0975 over ``Omega`` in [2, 8] vs 0.0734 out to 16).

Recorded anchors at this config (``seed=0``): slope **-0.4606 +/- 0.0734**,
CI[3 SE] = [-0.681, -0.240], ``se_ols`` 0.0677 / ``se_bootstrap`` 0.0734, PASS,
~245 s. Unlike the birth-death check this is run for a **single seed**: at ~4
minutes a parametrized second seed would dominate the suite. Seed 1 was verified
offline at the same config (slope -0.5191 +/- 0.1093, PASS), so the pass is not
seed-luck; the ``seed`` is pinned so the test is deterministic (non-negotiable #3).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import convergence_report
from sandbox.core.ode import integrate_rk4
from sandbox.core.registry import register
from sandbox.models.repressilator import (
    MODEL,
    OBSERVABLE_KEYS,
    Repressilator,
    RepressilatorParams,
)

# Measured in the ODE sanity sweep (see module docstring).
PERIOD = 16.095

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
_OMEGAS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
# Exclude the phase-saturation knee (Omega <= 1); see the module docstring.
_FIT_MASK = [False, False, False, True, True, True, True, True, True]
_T_MAX = 2.0 * PERIOD
_DT = 1e-3
_N_GRID = 400
_REPLICATES = 12
_Z = 3.0


def _factory(d: dict) -> RepressilatorParams:
    return RepressilatorParams(**d)


# ---------------------------------------------------------------------------
# Deliberately broken Omega scalings — the teeth for the convergence check
# ---------------------------------------------------------------------------
#
# Both override ONLY initial_state, substituting a wrong system size into the
# params that get embedded in the State. Because step / observables / is_terminal
# read that embedded copy, the whole SSA runs at the wrong system size — while
# deterministic_rhs and initial_concentrations never touch Omega, so the ODE
# reference is bit-identical to the real model. That isolates exactly one thing:
# how the fluctuations scale with Omega. Neither touches gillespie.py.

_FIXED_OMEGA = 4.0


class _FixedOmegaRepressilator(Repressilator):
    """Propensities pinned to a constant system size, ignoring the swept ``Omega``.

    The realistic bug of simply failing to thread ``Omega`` through. Fluctuations
    never shrink, so ``D(Omega)`` is flat and the slope is ~0: this fails the
    *significantly negative* leg, which no amount of replicate noise can rescue.

    Measured at the config below: slope **-0.1669 +/- 0.0722**, CI[3 SE] =
    [-0.384, +0.050] — ``consistent=False``, ``significant=False``, FAIL.
    """

    def initial_state(self, params: RepressilatorParams, rng):
        return super().initial_state(replace(params, Omega=_FIXED_OMEGA), rng)


class _SquaredOmegaRepressilator(Repressilator):
    """Propensities use ``Omega**2`` — fluctuations shrink *too fast*.

    Effective system size ``Omega^2`` gives ``D ~ (Omega^2)^{-1/2} = Omega^{-1}``,
    i.e. slope ~ -1. This is the complementary tooth: the slope is still
    *significantly negative*, so only the *consistent with -1/2* leg can reject
    it. Without this, a green check would only prove "noise decreases somehow".

    Measured at the config below: slope **-1.1315 +/- 0.1850**, CI[3 SE] =
    [-1.687, -0.576] — ``consistent=False`` but ``significant=True``, FAIL.
    """

    def initial_state(self, params: RepressilatorParams, rng):
        return super().initial_state(replace(params, Omega=params.Omega**2), rng)


register("_test_repressilator_fixed_omega", _FixedOmegaRepressilator())
register("_test_repressilator_squared_omega", _SquaredOmegaRepressilator())


# ---------------------------------------------------------------------------
# Deterministic premises (fast, non-statistical)
# ---------------------------------------------------------------------------


def test_ode_sustains_a_limit_cycle():
    """The whole convergence check is meaningless if the reference does not oscillate.

    Over ~10 periods the peak-to-trough amplitude of a *late* cycle must match a
    *mid* cycle: a damped spiral into the fixed point (what ``beta=5`` does) would
    show a clearly shrinking amplitude.
    """
    params = RepressilatorParams(Omega=1.0, t_max=170.0, **_BASE)
    t, y = integrate_rk4(
        MODEL.deterministic_rhs(params), MODEL.initial_concentrations(params), 170.0, 1e-2
    )
    m1 = y[:, 0]
    interior = (m1[1:-1] > m1[:-2]) & (m1[1:-1] > m1[2:])
    peaks = t[1:-1][interior]
    assert peaks.size >= 8, f"only {peaks.size} peaks — not oscillating"

    gaps = np.diff(peaks)
    assert gaps[-1] == pytest.approx(PERIOD, rel=0.02), f"period drifted: {gaps[-1]}"

    amps = np.array(
        [np.ptp(m1[(t >= a) & (t <= b)]) for a, b in zip(peaks[:-1], peaks[1:], strict=True)]
    )
    # Sustained, not damped: the last cycle is within 1% of the middle one.
    assert amps[-1] / amps[len(amps) // 2] == pytest.approx(1.0, rel=0.01), f"amplitudes {amps}"
    # And the oscillation is large, not a numerical wiggle (concentrations are O(100)).
    assert amps[-1] > 50.0


def test_cyclically_symmetric_initial_condition_is_rejected():
    # p1_0 == p2_0 == p3_0 lies on an invariant manifold whose internal dynamics
    # relax to the symmetric fixed point: no limit cycle at all. A silent flat
    # reference would make the convergence check meaningless, so refuse loudly.
    with pytest.raises(ValueError, match="symmetric"):
        RepressilatorParams(Omega=1.0, t_max=1.0, **{**_BASE, "p2_0": 0.0, "p3_0": 0.0})


def test_observable_keys_match_initial_concentrations_order():
    """``OBSERVABLE_KEYS`` must line up column-for-column with the ODE vector.

    ``convergence_report`` pairs ``observable_keys[s]`` with column ``s`` of the
    integrated ODE. If the two orders disagree the discrepancy compares species
    against the *wrong* species — a wrong-but-plausible ``D`` whose slope could
    still look fine. Distinct initial values per component make the check real.
    """
    params = RepressilatorParams(Omega=1000.0, t_max=1.0, **{**_BASE, "m0": 1.0})
    state = MODEL.initial_state(params, np.random.default_rng(0))
    observed = MODEL.observables(state)
    assert tuple(observed) == OBSERVABLE_KEYS
    got = np.array([observed[k] for k in OBSERVABLE_KEYS])
    # [m1, m2, m3, p1, p2, p3] = [1, 1, 1, 0, 5, 15] — all three proteins differ,
    # so any permutation of the protein block would be caught.
    assert np.allclose(got, MODEL.initial_concentrations(params), atol=1e-3)


def test_deterministic_rhs_matches_textbook_equations():
    """Pin the network to the published RHS.

    ``deterministic_rhs`` is ``stoich.T @ rates(c)`` over the *same* ``rates`` the
    SSA uses, so checking it against the hand-written Elowitz-Leibler equations
    also pins the stoichiometry and the repression wiring (``p_3 -| m_1``,
    ``p_1 -| m_2``, ``p_2 -| m_3``) that the stochastic side depends on.
    """
    params = RepressilatorParams(Omega=1.0, t_max=1.0, **_BASE)
    c = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])  # distinct, so mis-wiring shows
    a, a0, n, b = params.alpha, params.alpha0, params.n_hill, params.beta
    expected = np.array(
        [
            a / (1.0 + c[5] ** n) + a0 - c[0],  # m1 repressed by p3
            a / (1.0 + c[3] ** n) + a0 - c[1],  # m2 repressed by p1
            a / (1.0 + c[4] ** n) + a0 - c[2],  # m3 repressed by p2
            b * (c[0] - c[3]),
            b * (c[1] - c[4]),
            b * (c[2] - c[5]),
        ]
    )
    assert np.allclose(MODEL.deterministic_rhs(params)(c), expected)


def test_repressilator_has_no_analytic_predictions():
    # Deliberate: a limit cycle has no stationary scalar to match. Asserting it
    # keeps someone from "fixing" the omission with a meaningless closed form.
    assert not hasattr(MODEL, "analytic_predictions")


# ---------------------------------------------------------------------------
# The teeth: broken Omega scalings must FAIL the convergence check
# ---------------------------------------------------------------------------
#
# Deliberately cheap configs (fewer replicates, 1 period). Detecting a *wrong*
# slope needs far less precision than confirming the right one, and both failures
# below are structural rather than marginal.


def test_fixed_omega_propensities_fail_the_convergence_check():
    """Noise that never shrinks: slope ~0, so it cannot be significantly negative."""
    report = convergence_report(
        "_test_repressilator_fixed_omega",
        _BASE,
        _factory,
        omegas=[2.0, 4.0, 8.0],
        t_max=PERIOD,
        dt=_DT,
        replicates=8,
        n_grid=200,
        observable_keys=OBSERVABLE_KEYS,
        z=_Z,
        n_bootstrap=300,
    )
    assert not report.passed, str(report)
    assert not report.significant, str(report)


def test_squared_omega_propensities_fail_the_convergence_check():
    """Noise that shrinks too fast: slope ~-1, rejected by the -1/2 leg alone.

    ``significant`` stays True here, which is the point — this failure is caught
    *only* by the "consistent with -1/2" test, proving that leg has teeth.
    """
    report = convergence_report(
        "_test_repressilator_squared_omega",
        _BASE,
        _factory,
        omegas=[1.0, 1.5, 2.0, 2.5],
        t_max=PERIOD,
        dt=_DT,
        replicates=8,
        n_grid=200,
        observable_keys=OBSERVABLE_KEYS,
        z=_Z,
        n_bootstrap=300,
    )
    assert not report.passed, str(report)
    assert not report.consistent, str(report)


# ---------------------------------------------------------------------------
# The scaling law (the definition of "done" for the repressilator)
# ---------------------------------------------------------------------------


def test_repressilator_discrepancy_scales_as_omega_minus_half():
    report = convergence_report(
        "repressilator",
        _BASE,
        _factory,
        omegas=_OMEGAS,
        t_max=_T_MAX,
        dt=_DT,
        replicates=_REPLICATES,
        n_grid=_N_GRID,
        observable_keys=OBSERVABLE_KEYS,
        seed=0,
        z=_Z,
        fit_mask=_FIT_MASK,
        n_bootstrap=300,
        max_steps=2_000_000,
    )
    assert report.passed, str(report)
    assert report.consistent and report.significant and report.reference_ok
    assert abs(report.slope + 0.5) <= _Z * report.slope_se

    # Averaged per replicate, never mean-first (the phase-diffusion trap).
    assert np.allclose(report.discrepancy, report.per_replicate.mean(axis=1))

    # The excluded low-Omega points must actually BE saturated, otherwise the mask
    # is unjustified: D*sqrt(Omega) has to sit below the fitted plateau there.
    anchor = report.discrepancy * np.sqrt(report.omegas)
    plateau = float(anchor[report.fit_mask].mean())
    assert anchor[0] < plateau, (
        f"Omega={report.omegas[0]} is not in the knee: {anchor[0]} vs {plateau}"
    )
