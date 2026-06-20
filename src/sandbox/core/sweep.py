"""ParameterSweep — run a model across a grid of params and/or many replicates.

Built purely on the protocol: it resolves the model by name, constructs each
params object, spawns independent seeded RNGs (one per replicate), and collects
trajectories. The same ``Experiment`` always yields the same ``Result``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from sandbox.core.protocol import Experiment, Result
from sandbox.core.recorder import Trajectory, run_replicate
from sandbox.core.registry import get_model
from sandbox.core.rng import spawn_rngs


def _expand_sweep(sweep: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    """Cartesian product of a sweep grid into a list of param overrides."""
    if not sweep:
        return [{}]
    keys = list(sweep)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*sweep.values())]


def run_experiment(
    experiment: Experiment,
    params_factory: Callable[[dict[str, Any]], Any],
) -> Result:
    """Execute an ``Experiment`` into a reproducible ``Result``.

    ``params_factory`` turns a plain dict of params into the model's concrete
    params object (e.g. ``lambda d: WFParams(**d)``). It is supplied by the
    caller so the core stays agnostic about params shapes.

    Reproducibility: replicate ``r`` at sweep point ``i`` uses an RNG spawned
    from ``seed`` offset by the point index, so every (point, replicate) pair has
    an independent yet fully determined stream.
    """
    model = get_model(experiment.model)
    sweep_points = _expand_sweep(experiment.sweep)

    trajectories: list[list[Trajectory]] = []
    final_observables: list[list[dict[str, float]]] = []

    for point_index, overrides in enumerate(sweep_points):
        merged = {**experiment.params, **overrides}
        params = params_factory(merged)
        # Distinct, reproducible seed per sweep point so points don't share streams.
        rngs = spawn_rngs(experiment.seed + point_index, experiment.replicates)

        point_trajectories: list[Trajectory] = []
        point_finals: list[dict[str, float]] = []
        for rng in rngs:
            traj = run_replicate(
                model,
                params,
                rng,
                max_steps=experiment.max_steps,
                record_every=experiment.record_every,
            )
            point_trajectories.append(traj)
            point_finals.append(traj.final)

        trajectories.append(point_trajectories)
        final_observables.append(point_finals)

    return Result(
        experiment=experiment,
        sweep_points=sweep_points,
        trajectories=trajectories,
        final_observables=final_observables,
    )
