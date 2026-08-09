"""Recorder — captures observables over time.

A pure consumer of the protocol: it calls ``model.observables(state)`` and
``state.t`` and stores the results. It knows nothing about any model's concrete
state, and it tolerates irregular time sampling (Gillespie produces uneven
``t``; the Recorder never assumes a fixed step).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sandbox.core.protocol import Model, State


@dataclass
class Trajectory:
    """Times and per-observable value series from one replicate.

    ``terminated`` records *why* the run stopped: ``True`` if the model reached a
    terminal state, ``False`` if it was cut off at ``max_steps``. This matters for
    validity — a statistic that assumes absorption (e.g. fixation probability) is
    silently biased by truncated runs, so consumers can check this flag and fail
    loudly instead. Always ``False`` for models without ``is_terminal``.
    """

    times: list[float] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    terminated: bool = False

    def record(self, t: float, observables: dict[str, float]) -> None:
        self.times.append(float(t))
        for key, value in observables.items():
            self.series.setdefault(key, []).append(float(value))

    @property
    def final(self) -> dict[str, float]:
        """The last recorded value of each observable."""
        return {key: values[-1] for key, values in self.series.items() if values}

    def as_arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return (
            np.asarray(self.times, dtype=float),
            {key: np.asarray(values, dtype=float) for key, values in self.series.items()},
        )


def run_replicate(
    model: Model,
    params: object,
    rng: np.random.Generator,
    *,
    max_steps: int,
    record_every: int = 1,
) -> Trajectory:
    """Run a single replicate to completion, recording observables.

    Stops at ``max_steps`` or when the model reports a terminal state (if it
    implements ``is_terminal``). The initial and final states are always
    recorded, plus every ``record_every`` steps in between.

    ``is_terminal`` is evaluated **exactly once per state**: the result is reused
    for the loop condition, the "record the final state" decision, and the
    ``terminated`` flag. This is a hot path — for a Gillespie model ``is_terminal``
    ends in an absorbing-state check that re-evaluates the whole propensity
    vector, so testing the same state twice per step re-ran the model's ``rates``
    an extra time per event (measured at ~20% of repressilator SSA runtime). The
    initial state is tested *before* the loop because a model can be terminal at
    ``t = 0`` (Wright-Fisher initialized at fixation), which must record once and
    never step.
    """
    is_terminal = getattr(model, "is_terminal", None)

    state: State = model.initial_state(params, rng)
    traj = Trajectory()
    traj.record(state.t, model.observables(state))

    terminal_now = bool(is_terminal and is_terminal(state))
    step = 0
    while step < max_steps and not terminal_now:
        state = model.step(state, rng)
        step += 1
        terminal_now = bool(is_terminal and is_terminal(state))
        if step % record_every == 0 or terminal_now:
            traj.record(state.t, model.observables(state))

    traj.terminated = terminal_now
    return traj
