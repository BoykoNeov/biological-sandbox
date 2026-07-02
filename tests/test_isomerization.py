"""Isomerization ``A <-> B`` — the engine's second exact-closed-form check.

Written before ``models/isomerization.py`` exists (workflow rule: confirm the
test can fail first). Two reversible reactions, defined *macroscopically* (rates
as functions of concentration ``c = n / Omega``):

* ``A -> B``   rate ``f_1(c) = k1 * c_A``  (first order),
* ``B -> A``   rate ``f_2(c) = k2 * c_B``  (first order).

The total ``N = n_A + n_B`` is **conserved** (every reaction moves one molecule
between the two species). At stationarity each molecule is independently ``A``
with probability ``p = k2 / (k1 + k2)`` (detailed balance ``k1 <n_A> = k2 <n_B>``),
so ``n_A ~ Binomial(N, p)`` and the mean concentration is exact:

* ``<x_A> = (k2 / (k1 + k2)) * c_tot`` — the model's ``analytic_predictions``,
  Omega-independent (first-order reactions make ``a_j = Omega * f_j(n/Omega)``
  exact). Checked via the existing :func:`~sandbox.core.validation.validate`.

This is the second exact engine check (after birth-death): unlike birth-death it
exercises **multi-species stoichiometry and a conservation law**. The run relaxes
from all-``B`` (``cA0 = 0``) up to ``p * c_tot``, so the mean check is a real
relaxation test; the relaxation rate is ``k1 + k2``, so ``t_max`` is set to
``>> 1/(k1+k2)`` and the transient bias ``~ e^{-(k1+k2) t_max}`` is far below the
statistical tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment
from sandbox.core.validation import validate
from sandbox.models.isomerization import (
    Isomerization,
    IsomerizationParams,
    IsomerizationState,
)

# One point used by the mean-validation test. k1=1, k2=3 -> p = k2/(k1+k2) = 3/4,
# so <x_A> = 0.75 * c_tot = 1.5. Omega is modest on purpose: the mean is
# Omega-independent, and N = Omega*c_tot = 20 keeps the binomial near-normal while
# each replicate runs only ~t_max*(k1+k2)*N events -> the suite stays fast.
_PARAMS = {
    "k1": 1.0,
    "k2": 3.0,
    "Omega": 10.0,
    "t_max": 5.0,
    "c_tot": 2.0,
    "cA0": 0.0,
}
_REPLICATES = 1500
_PREDICTED_XA = (_PARAMS["k2"] / (_PARAMS["k1"] + _PARAMS["k2"])) * _PARAMS["c_tot"]


def _factory(d: dict) -> IsomerizationParams:
    """Turn a plain param dict into IsomerizationParams (core stays params-agnostic)."""
    return IsomerizationParams(**d)


def _experiment(seed: int, replicates: int = _REPLICATES) -> Experiment:
    return Experiment(
        model="isomerization",
        params=dict(_PARAMS),
        replicates=replicates,
        observables=("x_A", "x_B"),
        seed=seed,
        # Terminal is t >= t_max; runs stop there long before max_steps, but the
        # ceiling must exceed the ~t_max*(k1+k2)*N events any replicate needs or
        # validate() would raise on a non-terminated run.
        max_steps=20_000,
        # Only the final observable is read; avoid storing every event.
        record_every=20_000,
    )


@pytest.fixture
def model():
    return Isomerization()


# ---------------------------------------------------------------------------
# Unit behaviour (deterministic, independent of statistics)
# ---------------------------------------------------------------------------


def test_params_validation():
    with pytest.raises(ValueError):
        IsomerizationParams(k1=-1.0, k2=1.0, Omega=10.0, t_max=5.0, c_tot=2.0)
    with pytest.raises(ValueError):
        IsomerizationParams(k1=1.0, k2=0.0, Omega=10.0, t_max=5.0, c_tot=2.0)
    with pytest.raises(ValueError):
        IsomerizationParams(k1=1.0, k2=1.0, Omega=0.0, t_max=5.0, c_tot=2.0)
    with pytest.raises(ValueError):
        IsomerizationParams(k1=1.0, k2=1.0, Omega=10.0, t_max=0.0, c_tot=2.0)
    with pytest.raises(ValueError):
        IsomerizationParams(k1=1.0, k2=1.0, Omega=10.0, t_max=5.0, c_tot=0.0)
    # cA0 must lie within [0, c_tot].
    with pytest.raises(ValueError):
        IsomerizationParams(k1=1.0, k2=1.0, Omega=10.0, t_max=5.0, c_tot=2.0, cA0=-0.1)
    with pytest.raises(ValueError):
        IsomerizationParams(k1=1.0, k2=1.0, Omega=10.0, t_max=5.0, c_tot=2.0, cA0=2.5)


def test_initial_state_rounds_counts_and_conserves_total(model):
    # N = round(Omega*c_tot) = 30; n_A0 = round(Omega*cA0) = 6; n_B0 = N - n_A0 = 24.
    params = IsomerizationParams(k1=1.0, k2=1.0, Omega=10.0, t_max=5.0, c_tot=3.0, cA0=0.6)
    state = model.initial_state(params, np.random.default_rng(0))
    assert state.counts.tolist() == [6, 24]
    assert int(state.counts.sum()) == 30  # total = round(Omega*c_tot)
    assert state.t == 0.0
    assert state.params is params  # params embedded in the state


def test_step_conserves_total_and_moves_one_molecule(model):
    """The distinguishing feature of this model: every event conserves n_A + n_B
    and shifts exactly one molecule between the species (multi-species stoich)."""
    params = IsomerizationParams(k1=2.0, k2=2.0, Omega=10.0, t_max=100.0, c_tot=2.0, cA0=1.0)
    rng = np.random.default_rng(0)
    state = model.initial_state(params, rng)
    total0 = int(state.counts.sum())
    for _ in range(200):
        prev = state.counts.copy()
        state = model.step(state, rng)
        assert int(state.counts.sum()) == total0  # conservation, every step
        # exactly one A<->B flip: one species +1, the other -1
        delta = state.counts - prev
        assert delta.tolist() in ([1, -1], [-1, 1])


def test_observables_return_concentrations(model):
    params = IsomerizationParams(k1=1.0, k2=1.0, Omega=20.0, t_max=5.0, c_tot=5.0)
    state = IsomerizationState(
        counts=np.array([30, 70]), t=1.0, params=params, network=model._network(params)
    )
    obs = model.observables(state)
    assert obs["x_A"] == pytest.approx(30 / 20.0)
    assert obs["x_B"] == pytest.approx(70 / 20.0)


def test_is_terminal_at_time_horizon(model):
    params = IsomerizationParams(k1=1.0, k2=1.0, Omega=20.0, t_max=5.0, c_tot=2.0)
    net = model._network(params)
    assert not model.is_terminal(
        IsomerizationState(np.array([20, 20]), t=4.999, params=params, network=net)
    )
    assert model.is_terminal(
        IsomerizationState(np.array([20, 20]), t=5.0, params=params, network=net)
    )


def test_analytic_prediction_is_binomial_mean_and_omega_independent(model):
    # <x_A> = (k2/(k1+k2)) * c_tot; independent of Omega.
    small = IsomerizationParams(k1=1.0, k2=3.0, Omega=10.0, t_max=5.0, c_tot=2.0)
    large = IsomerizationParams(k1=1.0, k2=3.0, Omega=10_000.0, t_max=5.0, c_tot=2.0)
    assert model.analytic_predictions(small)["x_A"] == pytest.approx(1.5)
    assert model.analytic_predictions(large)["x_A"] == pytest.approx(1.5)  # same at any Omega


def test_deterministic_rhs_is_mass_action_and_conserves_total(model):
    # dc_A/dt = -k1 c_A + k2 c_B, dc_B/dt = +k1 c_A - k2 c_B (sum -> 0, conserved).
    params = IsomerizationParams(k1=1.0, k2=3.0, Omega=10.0, t_max=5.0, c_tot=2.0)
    rhs = model.deterministic_rhs(params)
    # Fixed point c_A* = (k2/(k1+k2)) c_tot = 1.5, c_B* = 0.5.
    d = rhs(np.array([1.5, 0.5]))
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(0.0)
    # Away from equilibrium the flux is antisymmetric (total is conserved).
    d = rhs(np.array([2.0, 0.0]))
    assert d[0] == pytest.approx(-2.0)  # -k1*2 + k2*0
    assert d[1] == pytest.approx(2.0)
    assert d.sum() == pytest.approx(0.0)


def test_initial_concentrations_split_matches_cA0(model):
    params = IsomerizationParams(k1=1.0, k2=3.0, Omega=10.0, t_max=5.0, c_tot=2.0, cA0=0.4)
    c0 = model.initial_concentrations(params)
    assert c0.tolist() == [0.4, 1.6]  # [cA0, c_tot - cA0]


# ---------------------------------------------------------------------------
# Validation (the definition of "done" for this model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [12345, 2024, 99])
def test_stationary_mean_matches_binomial_mean(seed):
    report = validate(_experiment(seed), _factory)
    assert report.passed, str(report)

    check = next(c for c in report.checks if c.name == "x_A")
    assert check.predicted == pytest.approx(_PREDICTED_XA)
    # Land within a few SE — guards against an accidentally-huge SE masking a miss.
    assert check.z_score < 4.0


def test_a_wrong_prediction_is_rejected(monkeypatch):
    """If the analytic prediction were wrong, validation must FAIL — proving the
    check has teeth and isn't trivially green. Uses the inverted split
    k1/(k1+k2)*c_tot = 0.5 (vs the correct 1.5), well outside z*SE."""
    from sandbox.models.isomerization import MODEL

    wrong = (_PARAMS["k1"] / (_PARAMS["k1"] + _PARAMS["k2"])) * _PARAMS["c_tot"]  # 0.5
    monkeypatch.setattr(MODEL, "analytic_predictions", lambda params: {"x_A": wrong})
    report = validate(_experiment(12345), _factory)
    assert not report.passed
