"""Protocol-level contracts: serialization and conformance."""

from __future__ import annotations

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment, Model, TerminableModel, ValidatableModel
from sandbox.core.registry import get_model
from sandbox.models.wright_fisher import WrightFisher


def test_experiment_json_round_trip():
    exp = Experiment(
        model="wright_fisher",
        params={"N": 100, "p0": 0.4, "s": 0.0},
        replicates=50,
        observables=("freq", "fixed_A"),
        seed=99,
        sweep={"N": [100, 200]},
    )
    restored = Experiment.from_json(exp.to_json())
    assert restored == exp
    # observables must survive as a tuple (not a list) so equality holds.
    assert isinstance(restored.observables, tuple)


def test_wright_fisher_satisfies_protocols():
    model = WrightFisher()
    assert isinstance(model, Model)
    assert isinstance(model, ValidatableModel)  # has analytic_predictions
    assert isinstance(model, TerminableModel)  # has is_terminal


def test_registry_resolves_by_name():
    assert isinstance(get_model("wright_fisher"), WrightFisher)
