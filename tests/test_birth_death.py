"""Birth-death (immigration/death) — the Gillespie engine's exact-closed-form check.

Written before ``models/birth_death.py`` exists (workflow rule: confirm the test
can fail first). The reactions are ``0 -> X`` (rate ``k``) and ``X -> 0`` (rate
``gamma * c``) — the M/M/infinity queue, whose stationary law is
``n ~ Poisson(Omega * k / gamma)``. Two consequences are checked here:

* **mean concentration** ``<x> = k / gamma`` — exact and *Omega-independent*
  (because the reactions are zeroth/first order, ``a_j = Omega * f_j(n/Omega)`` is
  exact), so the same prediction validates at every system size. Checked via the
  existing :func:`~sandbox.core.validation.validate`.
* **Fano factor** ``Var(n) / <n> = 1`` — a stationary Poisson identity in *counts*,
  a stronger noise check than the mean alone. Checked in a dedicated test that
  reconstructs ``n = x * Omega`` (the ``Var(x)/<x> = 1/Omega`` version would be
  Omega-dependent; the counts version is the clean ``1``).

The model relaxes from ``c0 = 0`` up to ``k/gamma``, so the mean check is a real
relaxation test (a do-nothing ``step`` would keep ``<x> = 0`` and fail); ``t_max``
is set to ``~10/gamma`` so the transient bias ``~e^{-gamma t_max}`` is far below
the statistical tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment
from sandbox.core.sweep import run_experiment
from sandbox.core.validation import validate
from sandbox.models.birth_death import BirthDeath, BirthDeathParams, BirthDeathState

# One point used by both the mean-validation and Fano tests. Omega is modest on
# purpose: the mean is Omega-independent, and each replicate runs the *whole*
# horizon (no early absorption), so event count ~ t_max * 2 * k * Omega — keeping
# Omega small keeps the suite fast without weakening the checks. lambda =
# Omega*k/gamma = 20 is still large enough that Poisson ~ normal and the
# SE(Fano) ~ sqrt(2/(R-1)) approximation (excess kurtosis 1/lambda) holds.
_PARAMS = {"k": 2.0, "gamma": 1.0, "Omega": 10.0, "t_max": 10.0, "c0": 0.0}
_REPLICATES = 1500


def _factory(d: dict) -> BirthDeathParams:
    """Turn a plain param dict into BirthDeathParams (the core stays params-agnostic)."""
    return BirthDeathParams(**d)


def _experiment(seed: int, replicates: int = _REPLICATES) -> Experiment:
    return Experiment(
        model="birth_death",
        params=dict(_PARAMS),
        replicates=replicates,
        observables=("x",),
        seed=seed,
        # Terminal is t >= t_max; runs stop there long before max_steps, but the
        # ceiling must exceed the ~t_max*2k*Omega events any replicate needs or
        # validate() would raise on a non-terminated run. This is ~50x the mean.
        max_steps=20_000,
        # Only the final observable is read; avoid storing every event.
        record_every=20_000,
    )


@pytest.fixture
def model():
    return BirthDeath()


# ---------------------------------------------------------------------------
# Unit behaviour (deterministic, independent of statistics)
# ---------------------------------------------------------------------------


def test_params_validation():
    with pytest.raises(ValueError):
        BirthDeathParams(k=-1.0, gamma=1.0, Omega=10.0, t_max=5.0)
    with pytest.raises(ValueError):
        BirthDeathParams(k=1.0, gamma=0.0, Omega=10.0, t_max=5.0)
    with pytest.raises(ValueError):
        BirthDeathParams(k=1.0, gamma=1.0, Omega=0.0, t_max=5.0)
    with pytest.raises(ValueError):
        BirthDeathParams(k=1.0, gamma=1.0, Omega=10.0, t_max=0.0)
    with pytest.raises(ValueError):
        BirthDeathParams(k=1.0, gamma=1.0, Omega=10.0, t_max=5.0, c0=-0.1)


def test_initial_state_rounds_counts_and_embeds_params(model):
    params = BirthDeathParams(k=2.0, gamma=1.0, Omega=30.0, t_max=5.0, c0=0.5)
    state = model.initial_state(params, np.random.default_rng(0))
    assert state.counts.tolist() == [15]  # round(0.5 * 30)
    assert state.t == 0.0
    assert state.params is params  # params embedded in the state


def test_step_advances_time_and_changes_count_by_one(model):
    params = BirthDeathParams(k=5.0, gamma=1.0, Omega=10.0, t_max=5.0, c0=1.0)
    rng = np.random.default_rng(0)
    s0 = model.initial_state(params, rng)
    s1 = model.step(s0, rng)
    assert s1.t > s0.t  # a sampled exponential waiting time, strictly positive
    assert abs(int(s1.counts[0]) - int(s0.counts[0])) == 1  # exactly one event


def test_observables_return_concentration(model):
    params = BirthDeathParams(k=2.0, gamma=1.0, Omega=20.0, t_max=5.0)
    state = BirthDeathState(
        counts=np.array([50]), t=1.0, params=params, network=model._network(params)
    )
    assert model.observables(state)["x"] == pytest.approx(50 / 20.0)


def test_is_terminal_at_time_horizon(model):
    params = BirthDeathParams(k=2.0, gamma=1.0, Omega=20.0, t_max=5.0)
    net = model._network(params)
    assert not model.is_terminal(
        BirthDeathState(np.array([40]), t=4.999, params=params, network=net)
    )
    assert model.is_terminal(BirthDeathState(np.array([40]), t=5.0, params=params, network=net))


def test_analytic_prediction_is_k_over_gamma_and_omega_independent(model):
    small = BirthDeathParams(k=3.0, gamma=1.5, Omega=10.0, t_max=5.0)
    large = BirthDeathParams(k=3.0, gamma=1.5, Omega=10_000.0, t_max=5.0)
    assert model.analytic_predictions(small)["x"] == pytest.approx(2.0)
    assert model.analytic_predictions(large)["x"] == pytest.approx(2.0)  # same at any Omega


def test_deterministic_rhs_is_immigration_minus_death(model):
    # dc/dt = k - gamma * c, with the fixed point k/gamma = 2.
    params = BirthDeathParams(k=4.0, gamma=2.0, Omega=10.0, t_max=5.0)
    rhs = model.deterministic_rhs(params)
    assert rhs(np.array([0.0]))[0] == pytest.approx(4.0)  # pure immigration at c=0
    assert rhs(np.array([2.0]))[0] == pytest.approx(0.0)  # stationary at c=k/gamma


def test_initial_concentrations_is_c0(model):
    params = BirthDeathParams(k=4.0, gamma=2.0, Omega=10.0, t_max=5.0, c0=0.7)
    assert model.initial_concentrations(params).tolist() == [0.7]


# ---------------------------------------------------------------------------
# Validation (the definition of "done" for this model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [12345, 2024, 99])
def test_stationary_mean_matches_k_over_gamma(seed):
    report = validate(_experiment(seed), _factory)
    assert report.passed, str(report)

    check = report.checks[0]
    assert check.name == "x"
    assert check.predicted == pytest.approx(_PARAMS["k"] / _PARAMS["gamma"])
    # Land within a few SE — guards against an accidentally-huge SE masking a miss.
    assert check.z_score < 4.0


def test_a_wrong_prediction_is_rejected(monkeypatch):
    """If the analytic prediction were wrong, validation must FAIL — proving the
    check has teeth and isn't trivially green (mirrors the Wright-Fisher guard)."""
    from sandbox.models.birth_death import MODEL

    monkeypatch.setattr(MODEL, "analytic_predictions", lambda params: {"x": 5.0})
    report = validate(_experiment(12345), _factory)
    assert not report.passed


def test_fano_factor_is_one_in_counts():
    """Var(n)/<n> = 1 at stationarity (Poisson), computed across replicates in
    *counts* (reconstructing n = x * Omega). A sharper noise check than the mean:
    it fails for a step that gets the mean right but the fluctuations wrong."""
    result = run_experiment(_experiment(seed=7), _factory)
    finals = result.final_observables[0]  # single sweep point
    n = np.array([round(f["x"] * _PARAMS["Omega"]) for f in finals], dtype=float)

    mean_n = n.mean()
    var_n = n.var(ddof=1)
    fano = var_n / mean_n

    # SE(Fano) ~ sqrt(2/(R-1)) for Poisson (excess kurtosis 1/lambda is negligible
    # at lambda = Omega*k/gamma = 20); statistically-derived, not a hardcoded eps.
    se_fano = math.sqrt(2.0 / (len(n) - 1))
    assert abs(fano - 1.0) < 4.0 * se_fano, f"Fano={fano:.4f}, SE={se_fano:.4f}"
