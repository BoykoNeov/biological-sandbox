"""ValidationSuite — the heart of the project's credibility.

It runs many seeded replicates of a :class:`ValidatableModel`, measures the
mean over replicates of each predicted observable's *final* value, and asserts
that mean matches the model's ``analytic_predictions`` within a tolerance
**derived from the statistics of the measurement** — never a hardcoded epsilon.

Why statistical tolerance matters: the mean over ``R`` replicates of a random
observable has a standard error ``s / sqrt(R)`` (``s`` = sample std). A hardcoded
epsilon is either so loose it always passes (proving nothing) or so tight it
flakes. We assert ``|measured - predicted| <= z * SE``. With ``z = 4`` a correct
model fails by chance with probability < 1e-4, while a model that is *actually*
wrong fails as ``R`` grows because ``SE`` shrinks toward zero. That asymmetry —
tightening with evidence — is what makes this a real check rather than theater.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sandbox.core.protocol import Experiment, Result, TerminableModel, ValidatableModel
from sandbox.core.registry import get_model
from sandbox.core.sweep import run_experiment


@dataclass(frozen=True)
class Check:
    """One predicted-vs-measured comparison."""

    name: str
    predicted: float
    measured: float
    sem: float          # standard error of the measured mean
    tolerance: float    # z * sem
    z: float
    n: int              # replicates
    passed: bool

    @property
    def z_score(self) -> float:
        """How many standard errors the measurement sits from the prediction."""
        return abs(self.measured - self.predicted) / self.sem if self.sem > 0 else math.inf

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name}: measured {self.measured:.5g} vs "
            f"predicted {self.predicted:.5g} "
            f"(|diff| = {abs(self.measured - self.predicted):.3g}, "
            f"{self.z_score:.2f} SE, tol {self.tolerance:.3g}, n={self.n})"
        )


@dataclass(frozen=True)
class ValidationReport:
    model: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        head = f"Validation of {self.model!r}: {'PASS' if self.passed else 'FAIL'}"
        return "\n".join([head, *(f"  {c}" for c in self.checks)])


def _mean_and_sem(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    mean = math.fsum(values) / n
    if n == 1:
        return mean, math.inf  # cannot estimate a standard error from one sample
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)  # ddof=1
    return mean, math.sqrt(var / n)


def validate(
    experiment: Experiment,
    params_factory: Callable[[dict[str, Any]], Any],
    *,
    z: float = 4.0,
    sem_floor: float = 1e-12,
    require_termination: bool | None = None,
) -> ValidationReport:
    """Validate the model named by ``experiment`` against its analytic predictions.

    Runs the experiment (which must be a single param point — no sweep — so the
    predictions are well-defined), then compares each predicted observable to the
    replicate mean of its final value.

    ``sem_floor`` guards the degenerate case where every replicate returns an
    identical value (SE = 0, e.g. a prediction of probability exactly 1): without
    it the tolerance would be zero and an exactly-correct model could still
    "fail" on floating-point dust.

    ``require_termination`` guards against *silent* bias: a statistic that assumes
    absorption (e.g. a fixation probability) is dragged toward the truncation
    value when a replicate hits ``max_steps`` without reaching a terminal state.
    For a model with ``is_terminal`` this defaults to ``True``, so an under-budget
    run raises loudly (bump ``max_steps``) rather than reporting a wrong number
    that still looks green. Pass ``False`` for a terminable model whose prediction
    legitimately tolerates non-absorbed runs.
    """
    model = get_model(experiment.model)
    if not isinstance(model, ValidatableModel):
        raise TypeError(
            f"model {experiment.model!r} has no analytic_predictions; "
            "it cannot be validated (it belongs to the speculative arc)"
        )
    if experiment.sweep:
        raise ValueError("validate() expects a single param point, not a sweep")

    params = params_factory(dict(experiment.params))
    predictions = model.analytic_predictions(params)

    result: Result = run_experiment(experiment, params_factory)

    if require_termination is None:
        require_termination = isinstance(model, TerminableModel)
    if require_termination:
        trajectories = result.trajectories[0]  # single sweep point
        truncated = sum(1 for traj in trajectories if not traj.terminated)
        if truncated:
            raise ValueError(
                f"{truncated}/{len(trajectories)} replicates hit max_steps "
                f"(={experiment.max_steps}) without reaching a terminal state; the "
                "measured statistic would be biased. Increase max_steps, or pass "
                "require_termination=False if non-absorbed runs are acceptable here."
            )

    finals = result.final_observables[0]  # single sweep point

    checks: list[Check] = []
    for name, predicted in predictions.items():
        samples = [f[name] for f in finals if name in f]
        mean, sem = _mean_and_sem(samples)
        sem_eff = max(sem, sem_floor)
        tol = z * sem_eff
        checks.append(
            Check(
                name=name,
                predicted=float(predicted),
                measured=mean,
                sem=sem,
                tolerance=tol,
                z=z,
                n=len(samples),
                passed=abs(mean - predicted) <= tol,
            )
        )

    return ValidationReport(model=experiment.model, checks=checks)
