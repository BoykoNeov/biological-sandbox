"""The headline check: the ValidationSuite reproduces Wright-Fisher's analytic
prediction. This is the test that defines "done" for the Phase-0 slice.

It is deterministic (fixed seed), and the tolerance is *derived from the
statistics of the measurement* (z * standard error), not hardcoded — so it
passes reliably for a correct model and would fail for a broken ``step``.
"""

from __future__ import annotations

import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment
from sandbox.core.validation import validate


def _experiment(p0: float, seed: int) -> Experiment:
    return Experiment(
        model="wright_fisher",
        params={"N": 200, "p0": p0, "s": 0.0},
        replicates=3000,
        observables=("fixed_A",),
        seed=seed,
        max_steps=50_000,
    )


@pytest.mark.parametrize(("p0", "seed"), [(0.3, 12345), (0.5, 2024), (0.7, 99)])
def test_neutral_fixation_probability_matches_p0(wf_params_factory, p0, seed):
    report = validate(_experiment(p0, seed), wf_params_factory)
    assert report.passed, str(report)

    check = report.checks[0]
    assert check.name == "fixed_A"
    assert check.predicted == p0
    # Sanity: the measurement should land within a few SE — a stronger statement
    # than merely "within tolerance", guarding against an accidentally-huge SE.
    assert check.z_score < 4.0


def test_a_wrong_prediction_is_rejected(wf_params_factory, monkeypatch):
    """If the analytic prediction were wrong, validation must FAIL — proving the
    check has teeth and isn't trivially green."""
    from sandbox.models.wright_fisher import MODEL

    monkeypatch.setattr(MODEL, "analytic_predictions", lambda params: {"fixed_A": 0.95})
    report = validate(_experiment(0.3, 12345), wf_params_factory)
    assert not report.passed


def test_validate_rejects_sweep(wf_params_factory):
    exp = Experiment(
        model="wright_fisher",
        params={"N": 200, "p0": 0.3, "s": 0.0},
        sweep={"N": [100, 200]},
    )
    with pytest.raises(ValueError):
        validate(exp, wf_params_factory)
