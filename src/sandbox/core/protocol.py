"""The Model protocol — the spine of the sandbox.

Every model (validated or speculative) implements this one small interface, and
the shared services (Recorder, ParameterSweep, ValidationSuite) are built *on
top of* it. A shared service consumes only the protocol surface; it never
reaches inside a model's concrete state.

Two design decisions are load-bearing and worth stating up front.

**1. ``step`` does not take a time step.** Models differ fundamentally in how
time advances:

* a Wright-Fisher step is exactly one generation;
* a Gillespie step advances by a *sampled* exponential waiting time that only
  the model can compute;
* a PDE / ODE step advances by a numerical increment that is a *parameter of
  the model*, not a per-call choice.

So the model owns its own increment and writes the resulting time into
``state.t``. Callers read ``state.t`` and must tolerate irregular sampling —
which Gillespie forces regardless. A signature that made the *caller* pass
``dt`` literally could not express Gillespie, so we do not have one.

**2. Model objects are stateless; ``initial_state`` embeds the params.** Because
``step`` receives only ``(state, rng)``, everything ``step`` /
``observables`` / ``is_terminal`` need must already live inside the ``State``.
``initial_state(params, rng)`` is therefore responsible for embedding the params
into the returned state. The model instance itself carries no configuration and
can be a shared singleton. ``analytic_predictions`` is the one method that takes
params directly, because it is a pure function of the params with no state.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from numpy.random import Generator

# Each model defines its own frozen-dataclass params type; the protocol is
# deliberately agnostic about its shape.
Params = Any


@runtime_checkable
class State(Protocol):
    """Marker protocol for a model's state.

    The only field shared services rely on is ``t`` — the current simulation
    time. Units are the model's own (generations for Wright-Fisher, seconds for
    Hodgkin-Huxley, ...). Concrete states are expected to be immutable
    (frozen dataclasses): ``step`` returns a *new* state rather than mutating.
    """

    t: float


@runtime_checkable
class Model(Protocol):
    """The minimal contract every model satisfies."""

    def initial_state(self, params: Params, rng: Generator) -> State:
        """Build the initial state, embedding everything ``step`` will need."""
        ...

    def step(self, state: State, rng: Generator) -> State:
        """Advance one increment. The model chooses the increment and updates
        ``state.t``. Must be pure given ``(state, rng)``."""
        ...

    def observables(self, state: State) -> dict[str, float]:
        """Scalar summaries of a state. The Recorder's only window into a model."""
        ...


@runtime_checkable
class ValidatableModel(Model, Protocol):
    """A model with a checkable analytic limit.

    Only models that implement this belong in the validated core. Speculative-arc
    models may omit it — and must be labelled exploratory where they appear.
    """

    def analytic_predictions(self, params: Params) -> dict[str, float]:
        """Closed-form values the simulation must reproduce.

        Each returned key names an observable. The ValidationSuite measures the
        mean over replicates of that observable's *final* value and asserts it
        matches the predicted value within a statistically-derived tolerance.
        """
        ...


@runtime_checkable
class TerminableModel(Model, Protocol):
    """A model whose runs can end early (e.g. Wright-Fisher reaching fixation)."""

    def is_terminal(self, state: State) -> bool: ...


@dataclass(frozen=True)
class Experiment:
    """A declarative, serializable specification of a run.

    This is the unit of reproducibility and sharing. A validation case is just
    an ``Experiment`` whose expected output is known. Two runs of the same
    ``Experiment`` produce the same ``Result``.
    """

    model: str
    params: dict[str, Any]
    replicates: int = 1
    observables: tuple[str, ...] = ()
    seed: int = 0
    max_steps: int = 10_000
    record_every: int = 1
    # Optional parameter grid: {param_name: [values, ...]}. The sweep runs the
    # full Cartesian product, with ``replicates`` seeded repeats at each point.
    sweep: dict[str, list[Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Experiment:
        d = dict(d)
        if d.get("observables") is not None:
            d["observables"] = tuple(d["observables"])
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> Experiment:
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class Result:
    """The reproducible output of running an ``Experiment``.

    ``trajectories`` is indexed by sweep point then replicate. Each entry is a
    :class:`~sandbox.core.recorder.Trajectory`. ``sweep_points`` lists the param
    override applied at each point (empty dict when there is no sweep).
    """

    experiment: Experiment
    sweep_points: list[dict[str, Any]]
    trajectories: list[list[Any]]  # list[list[Trajectory]]; Any avoids a cycle
    final_observables: list[list[dict[str, float]]] = field(default_factory=list)
