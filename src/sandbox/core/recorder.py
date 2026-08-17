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
        # Hot path: called once per SSA event. `setdefault(key, [])` builds a
        # throwaway list on *every* call (once per observable per event) even when
        # the key is present; the get-and-branch below allocates only on the first
        # record of each key.
        self.times.append(float(t))
        series = self.series
        for key, value in observables.items():
            values = series.get(key)
            if values is None:
                values = series[key] = []
            values.append(float(value))

    @property
    def final(self) -> dict[str, float]:
        """The last recorded value of each observable."""
        return {key: values[-1] for key, values in self.series.items() if values}

    def as_arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return (
            np.asarray(self.times, dtype=float),
            {key: np.asarray(values, dtype=float) for key, values in self.series.items()},
        )


class ReplicateRunner:
    """One replicate, advanced **incrementally** rather than in one call.

    :func:`run_replicate` runs a replicate to completion; a front-end cannot,
    because a run that blocks until it finishes cannot draw itself while it goes
    (Phase 4's whole premise). So the loop lives here and ``run_replicate``
    became a thin wrapper over it. That is deliberately *one* loop rather than
    two: the stepping semantics below are subtle enough that a second
    implementation would drift, and every recorded slope anchor in this project
    depends on the trajectory those semantics produce.

    Three properties of the loop are load-bearing, and all three survive being
    cut into chunks:

    * ``is_terminal`` is evaluated **exactly once per state**. The result is
      reused for the loop condition, the "record the final state" decision, and
      the ``terminated`` flag. This is a hot path — for a Gillespie model
      ``is_terminal`` ends in an absorbing-state check that re-evaluates the
      whole propensity vector, so testing the same state twice per step re-ran
      the model's ``rates`` an extra time per event (measured at ~20% of
      repressilator SSA runtime).
    * The initial state is tested **before** any stepping, because a model can be
      terminal at ``t = 0`` (Wright-Fisher initialized at fixation), which must
      record once and never step.
    * ``record_every`` is counted against the **cumulative** step index, not a
      per-chunk one. Advancing 20 000 steps in one call and in 4 096 calls of
      irregular size therefore record the same states — including when the chunk
      size is not a multiple of ``record_every``, which is the case a test that
      only ever chunks by multiples cannot see.

    The step loop below reads and writes locals and syncs back to the instance
    once per ``advance`` call, rather than touching ``self`` per event: the same
    reason ``Trajectory.record`` avoids ``setdefault``.
    """

    def __init__(
        self,
        model: Model,
        params: object,
        rng: np.random.Generator,
        *,
        max_steps: int,
        record_every: int = 1,
    ) -> None:
        if record_every < 1:
            raise ValueError(f"record_every must be >= 1, got {record_every}")
        if max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {max_steps}")

        self.model = model
        self.rng = rng
        self.max_steps = int(max_steps)
        self.record_every = int(record_every)
        self._is_terminal = getattr(model, "is_terminal", None)

        self.state: State = model.initial_state(params, rng)
        self.steps = 0
        self.trajectory = Trajectory()
        self.trajectory.record(self.state.t, model.observables(self.state))
        self.terminal = bool(self._is_terminal and self._is_terminal(self.state))
        self.trajectory.terminated = self.terminal

    @property
    def finished(self) -> bool:
        """True once the run is terminal or has spent its ``max_steps`` budget."""
        return self.terminal or self.steps >= self.max_steps

    def advance(self, n_steps: int) -> int:
        """Take up to ``n_steps`` steps; return how many were actually taken.

        Fewer than requested means the run finished — either the model went
        terminal or the ``max_steps`` budget ran out. Callers distinguish the two
        through :attr:`terminal`.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")

        model = self.model
        rng = self.rng
        is_terminal = self._is_terminal
        record_every = self.record_every
        traj = self.trajectory

        state = self.state
        step = self.steps
        terminal_now = self.terminal
        budget = min(step + n_steps, self.max_steps)
        started_at = step

        while step < budget and not terminal_now:
            state = model.step(state, rng)
            step += 1
            terminal_now = bool(is_terminal and is_terminal(state))
            if step % record_every == 0 or terminal_now:
                traj.record(state.t, model.observables(state))

        self.state = state
        self.steps = step
        self.terminal = terminal_now
        traj.terminated = terminal_now
        return step - started_at

    def run_to_completion(self) -> Trajectory:
        """Advance until terminal or out of budget, and return the trajectory."""
        self.advance(self.max_steps - self.steps)
        return self.trajectory


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
    recorded, plus every ``record_every`` steps in between. The loop itself lives
    in :class:`ReplicateRunner`; see there for why.
    """
    return ReplicateRunner(
        model,
        params,
        rng,
        max_steps=max_steps,
        record_every=record_every,
    ).run_to_completion()
