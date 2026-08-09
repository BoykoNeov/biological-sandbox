"""Deterministic Hodgkin-Huxley: the fixed point, the integrator, and the RHS.

Written before ``models/hodgkin_huxley.py`` exists (workflow rule).

**Be honest about what is anchored to what.** Phase 2's plan names three
categories of checkable claim, and this model is the one where the boundary
between them actually bites:

* **Category A, self-consistency.** ``analytic_predictions`` returns the resting
  fixed point found by *root-finding* the algebraic steady-state equations; the
  simulation reaches it by *time-integration*. Those are two different code paths
  and making them agree is a real check — it catches drift between them, a
  non-attracting "equilibrium", and a fixed point that is not actually stationary.
  What it **cannot** catch is a wrong ``g_Na`` or a gate exponent of ``m^2``
  instead of ``m^3``: both paths would be wrong *consistently* and the root would
  still be a genuine root. Saying so is the point of the three-category framing.
* **Category A, independent.** :func:`textbook_rhs` below is a **separate
  hand-transcription** of the published Hodgkin-Huxley equations, living in the
  test file and written from the paper rather than from the implementation. That
  *is* what catches a wrong conductance or exponent, and it is the teeth this file
  really turns on.
* **Category C, literature-anchored.** ``V_rest ~ -65 mV``, the rheobase, the
  spike shape. Reported and coarsely sanity-bounded; **never** asserted against a
  precise literature number, because that would be asserting the resolution of
  whatever scan produced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.protocol import Experiment
from sandbox.core.recorder import run_replicate
from sandbox.core.validation import validate
from sandbox.models.hh_rates import (
    alpha_h,
    alpha_m,
    alpha_n,
    beta_h,
    beta_m,
    beta_n,
    steady_state,
)
from sandbox.models.hodgkin_huxley import (
    MODEL,
    STATE_KEYS,
    HHParams,
    hh_rhs,
    n_hh_steps,
    resting_state,
)


def params_factory(d: dict) -> HHParams:
    return HHParams(**d)


def textbook_rhs(y: np.ndarray, p: HHParams) -> np.ndarray:
    """The published Hodgkin-Huxley equations, hand-transcribed here on purpose.

    Deliberately written out longhand and NOT sharing code with the model, so a
    wrong conductance, a swapped reversal potential or an ``m^2`` where ``m^3``
    belongs shows up as a disagreement. This is the independent anchor; the
    fixed-point check is only self-consistency.
    """
    V, m, h, n = y
    i_na = p.g_na * m**3 * h * (V - p.e_na)
    i_k = p.g_k * n**4 * (V - p.e_k)
    i_l = p.g_l * (V - p.e_l)
    return np.array(
        [
            (p.i_ext - i_na - i_k - i_l) / p.c_m,
            alpha_m(V) * (1.0 - m) - beta_m(V) * m,
            alpha_h(V) * (1.0 - h) - beta_h(V) * h,
            alpha_n(V) * (1.0 - n) - beta_n(V) * n,
        ],
        dtype=float,
    )


def run_hh(params: HHParams) -> tuple[np.ndarray, np.ndarray]:
    """One replicate; returns ``(times, y)`` with ``y`` shaped ``(n, 4)``."""
    traj = run_replicate(MODEL, params, np.random.default_rng(0), max_steps=n_hh_steps(params) + 10)
    assert traj.terminated, "HH run hit max_steps without terminating"
    times, series = traj.as_arrays()
    return times, np.stack([series[k] for k in STATE_KEYS], axis=1)


def count_spikes(y: np.ndarray, threshold: float = 0.0) -> int:
    v = y[:, 0]
    return int(np.sum((v[:-1] < threshold) & (v[1:] >= threshold)))


# --------------------------------------------------------------------------
# Category A, independent: the RHS itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("i_ext", [0.0, 3.0, 10.0])
def test_rhs_matches_the_hand_transcribed_textbook_equations(i_ext):
    p = HHParams(i_ext=i_ext)
    rhs = hh_rhs(p)
    rng = np.random.default_rng(7)
    for _ in range(25):
        # Physiologically plausible states, plus the gate corners.
        y = np.array(
            [
                rng.uniform(-90.0, 50.0),
                rng.uniform(0.0, 1.0),
                rng.uniform(0.0, 1.0),
                rng.uniform(0.0, 1.0),
            ]
        )
        assert np.allclose(rhs(y), textbook_rhs(y, p), rtol=1e-13, atol=0.0)


def test_rhs_uses_the_documented_gate_exponents():
    # An m^2 or n^3 slip changes the current by a factor of the gate value, which
    # is O(1) -- so this is a coarse but decisive probe at a state where the gates
    # are far from 0 and 1 and every exponent is distinguishable.
    p = HHParams(i_ext=0.0)
    y = np.array([-40.0, 0.5, 0.5, 0.5])
    dv = hh_rhs(p)(y)[0]
    expected = (
        0.0
        - p.g_na * 0.5**3 * 0.5 * (-40.0 - p.e_na)
        - p.g_k * 0.5**4 * (-40.0 - p.e_k)
        - p.g_l * (-40.0 - p.e_l)
    ) / p.c_m
    assert dv == pytest.approx(expected, rel=1e-13)


def test_state_key_order_is_v_m_h_n():
    # The 4-vector, the observables and the ODE column order all index on this.
    assert STATE_KEYS == ("V", "m", "h", "n")


# --------------------------------------------------------------------------
# Category A, self-consistency: root-find vs time-integrate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("i_ext", [0.0, 2.0, 4.0, 6.0])
def test_resting_state_is_a_root_of_the_rhs(i_ext):
    p = HHParams(i_ext=i_ext)
    y_star = resting_state(p)
    assert np.abs(hh_rhs(p)(y_star)).max() < 1e-11
    # Gates must sit at their steady states for that voltage.
    assert np.allclose(y_star[1:], steady_state(y_star[0]), rtol=0, atol=1e-12)


def test_resting_potential_is_near_the_published_minus_65():
    # LITERATURE-ANCHORED (category C), used only as a COARSE bound. This is what
    # would catch a wrong conductance or reversal potential -- neither the
    # root-find-vs-integrate check nor the fixed-point test can, since both paths
    # would be wrong consistently. Not asserted to any precision it cannot support.
    v_rest = float(resting_state(HHParams(i_ext=0.0))[0])
    assert -66.0 < v_rest < -64.0


def test_starting_at_the_fixed_point_does_not_drift():
    # The sharpest category-A check here: if the algebraic root really is an
    # equilibrium of the ODE being integrated, nothing moves at all.
    p = HHParams(i_ext=0.0, v0=float(resting_state(HHParams(i_ext=0.0))[0]), t_max=20.0)
    _, y = run_hh(p)
    assert np.abs(y - y[0]).max() < 1e-11


def test_validate_reproduces_the_fixed_point():
    # Routed through the ValidationSuite (non-negotiable #2). Started AT the fixed
    # point, so the only residual is integration error -- which Richardson measures.
    # Relaxation-from-elsewhere is a separate claim, tested below, because its error
    # is dominated by the remaining transient rather than by dt and Richardson would
    # not see it (both dt and dt/2 carry the same leftover transient).
    v_star = float(resting_state(HHParams(i_ext=0.0))[0])
    base = {"i_ext": 0.0, "v0": v_star, "t_max": 20.0, "dt": 0.01}

    def final(dt: float) -> np.ndarray:
        _, y = run_hh(HHParams(**{**base, "dt": dt}))
        return y[-1]

    bound = float(np.abs(final(0.01) - final(0.005)).max())
    experiment = Experiment(
        model="hodgkin_huxley",
        params=base,
        replicates=2,
        observables=STATE_KEYS,
        seed=0,
        max_steps=n_hh_steps(HHParams(**base)) + 10,
    )
    report = validate(experiment, params_factory, z=4.0, sem_floor=max(bound, 1e-13))
    assert report.passed, str(report)
    assert {c.name for c in report.checks} == set(STATE_KEYS)


def test_the_fixed_point_attracts_from_a_displaced_start():
    # That the root is stationary does not make it an attractor. Start 5 mV away
    # and confirm the trajectory arrives. Tolerance is the MEASURED leftover
    # transient: |y(t_max) - y(t_max/2)| bounds what is still decaying, so it is a
    # number this run produces rather than one typed in. Planning measured the
    # residual at 2.9e-8 (100 ms), 6.9e-11 (150 ms), 1.8e-13 (200 ms) for tau=8.29 ms.
    v_star = resting_state(HHParams(i_ext=0.0))
    full = HHParams(i_ext=0.0, v0=-70.0, t_max=150.0)
    half = HHParams(i_ext=0.0, v0=-70.0, t_max=75.0)
    _, y_full = run_hh(full)
    _, y_half = run_hh(half)
    still_decaying = float(np.abs(y_full[-1] - y_half[-1]).max())
    residual = float(np.abs(y_full[-1] - v_star).max())
    assert residual <= still_decaying, (
        f"residual {residual:.3e} exceeds the measured leftover transient {still_decaying:.3e}"
    )
    assert residual < 1e-6, "not converged enough for the claim to mean anything"


def test_analytic_predictions_refuse_an_unstable_fixed_point():
    # At I = 10 the fixed point has lost stability (subcritical Hopf; planning
    # measured stable at I = 8, unstable by I = 10) and the attractor is a limit
    # cycle. Predicting the fixed point there would be a wrong number that still
    # looks green, so it must raise -- the same stance as validate()'s
    # require_termination guard.
    with pytest.raises(ValueError, match="stable|unstable|Hopf"):
        MODEL.analytic_predictions(HHParams(i_ext=10.0))


# --------------------------------------------------------------------------
# Category B: integrator order on the real, spiking RHS
# --------------------------------------------------------------------------


def test_error_is_fourth_order_on_a_spiking_trajectory():
    # The stiffest thing this model does. Planning measured, against a dt = 5e-4
    # reference over 20 ms at I = 10: 6.45e-7 / 4.43e-8 / 2.89e-9 / 1.84e-10 for
    # dt = 0.02 / 0.01 / 0.005 / 0.0025 -- ratios 14.6, 15.3, 15.7 -> 2^4.
    def final_v(dt: float) -> float:
        _, y = run_hh(HHParams(i_ext=10.0, v0=-65.0, t_max=20.0, dt=dt))
        return float(y[-1, 0])

    reference = final_v(0.000625)
    coarse = abs(final_v(0.01) - reference)
    fine = abs(final_v(0.005) - reference)
    assert coarse > 0.0 and fine > 0.0
    ratio = coarse / fine
    assert 12.0 < ratio < 20.0, f"expected ~16 for RK4, got {ratio:.2f}"


# --------------------------------------------------------------------------
# Category C: structural facts only -- no literature number as a bound
# --------------------------------------------------------------------------


def test_spike_count_is_monotone_non_decreasing_in_current():
    # Deliberately NOT "rheobase is between 6.2 and 6.5": that would assert the
    # 0.5-resolution of the planning scan and break on any change of horizon or
    # threshold. What is structural, and still falsifiable, is that more drive does
    # not produce fewer spikes, that a quiet regime exists, and that a firing one does.
    currents = [0.0, 4.0, 8.0, 20.0]
    counts = [count_spikes(run_hh(HHParams(i_ext=i, v0=-65.0, t_max=60.0))[1]) for i in currents]
    assert counts == sorted(counts), f"spike count not monotone in I: {counts}"
    assert counts[0] == 0, "the cell should be quiet with no injected current"
    assert counts[-1] > 0, "the cell should fire under strong drive"


def test_a_spike_overshoots_zero_and_repolarises():
    # Structural shape facts, not a published amplitude: an action potential goes
    # positive and comes back below its starting point (the after-hyperpolarisation).
    _, y = run_hh(HHParams(i_ext=20.0, v0=-65.0, t_max=30.0))
    v = y[:, 0]
    assert v.max() > 0.0, "no overshoot -- this is not an action potential"
    assert v.min() < -65.0, "no after-hyperpolarisation"


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_observables_cover_the_state_vector():
    p = HHParams()
    obs = MODEL.observables(MODEL.initial_state(p, np.random.default_rng(0)))
    assert set(obs) == set(STATE_KEYS)


def test_gates_start_at_their_steady_state_for_v0():
    p = HHParams(v0=-70.0)
    state = MODEL.initial_state(p, np.random.default_rng(0))
    obs = MODEL.observables(state)
    assert obs["V"] == pytest.approx(-70.0)
    expected = steady_state(-70.0)
    for i, gate in enumerate(("m", "h", "n")):
        assert obs[gate] == pytest.approx(float(expected[i]), rel=1e-14)


def test_the_run_lands_exactly_on_t_max():
    times, _ = run_hh(HHParams(t_max=5.0, dt=0.01))
    assert times[-1] == pytest.approx(5.0, abs=1e-12)
    assert times.size == 501


def test_experiment_round_trips():
    experiment = Experiment(
        model="hodgkin_huxley",
        params={"i_ext": 3.0, "v0": -65.0, "t_max": 5.0, "dt": 0.01},
        replicates=1,
        observables=STATE_KEYS,
        seed=1,
        max_steps=600,
    )
    assert Experiment.from_json(experiment.to_json()) == experiment


def test_rejects_a_dt_that_does_not_divide_t_max():
    with pytest.raises(ValueError, match="divide"):
        HHParams(t_max=1.0, dt=0.3)
