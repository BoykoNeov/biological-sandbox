"""Wright-Fisher unit behaviour (deterministic checks, independent of statistics)."""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.models.wright_fisher import WFParams, WFState, WrightFisher


@pytest.fixture
def model():
    return WrightFisher()


def test_params_validation():
    with pytest.raises(ValueError):
        WFParams(N=0, p0=0.5)
    with pytest.raises(ValueError):
        WFParams(N=10, p0=1.5)
    with pytest.raises(ValueError):
        WFParams(N=10, p0=0.5, s=-1.0)


def test_initial_state_count_and_time(model):
    params = WFParams(N=200, p0=0.3)
    state = model.initial_state(params, np.random.default_rng(0))
    assert state.count == 60  # round(0.3 * 200)
    assert state.t == 0.0
    assert state.params is params  # params are embedded in the state


def test_step_advances_one_generation(model):
    params = WFParams(N=100, p0=0.5)
    rng = np.random.default_rng(0)
    s0 = model.initial_state(params, rng)
    s1 = model.step(s0, rng)
    assert s1.t == s0.t + 1.0
    assert 0 <= s1.count <= params.N


def test_absorbing_states_are_terminal(model):
    params = WFParams(N=50, p0=0.5)
    assert model.is_terminal(WFState(count=0, t=5.0, params=params))
    assert model.is_terminal(WFState(count=50, t=5.0, params=params))
    assert not model.is_terminal(WFState(count=25, t=5.0, params=params))


def test_fixed_state_never_changes(model):
    """A fixed population stays fixed under further steps (count == N is absorbing)."""
    params = WFParams(N=30, p0=1.0)
    rng = np.random.default_rng(123)
    state = model.initial_state(params, rng)
    assert state.count == 30
    for _ in range(20):
        state = model.step(state, rng)
        assert state.count == 30


def test_observables_report_fixation(model):
    params = WFParams(N=10, p0=0.5)
    assert model.observables(WFState(count=10, t=0.0, params=params))["fixed_A"] == 1.0
    assert model.observables(WFState(count=0, t=0.0, params=params))["lost_A"] == 1.0
    mid = model.observables(WFState(count=5, t=0.0, params=params))
    assert mid["freq"] == 0.5
    assert mid["fixed_A"] == 0.0 and mid["lost_A"] == 0.0


def test_neutral_analytic_prediction_is_p0(model):
    assert model.analytic_predictions(WFParams(N=500, p0=0.42))["fixed_A"] == 0.42


def test_selection_prediction_not_yet_supported(model):
    with pytest.raises(NotImplementedError):
        model.analytic_predictions(WFParams(N=500, p0=0.42, s=0.1))


def test_selection_adjusted_frequency():
    neutral = WFParams(N=10, p0=0.5, s=0.0)
    assert neutral.selection_adjusted_frequency(0.3) == 0.3
    favoured = WFParams(N=10, p0=0.5, s=1.0)
    # p' = 0.5*2 / (1 + 0.5) = 1/1.5 = 0.666...; selection raises A's frequency.
    assert favoured.selection_adjusted_frequency(0.5) == pytest.approx(2.0 / 3.0)
