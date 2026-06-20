"""Reproducibility is non-negotiable: same Experiment -> same Result."""

from __future__ import annotations

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment
from sandbox.core.sweep import run_experiment


def _experiment() -> Experiment:
    return Experiment(
        model="wright_fisher",
        params={"N": 150, "p0": 0.4, "s": 0.0},
        replicates=25,
        observables=("freq", "fixed_A"),
        seed=2024,
        max_steps=10_000,
        sweep={"N": [100, 150, 200]},
    )


def test_same_experiment_gives_identical_results(wf_params_factory):
    r1 = run_experiment(_experiment(), wf_params_factory)
    r2 = run_experiment(_experiment(), wf_params_factory)

    assert r1.sweep_points == r2.sweep_points
    assert len(r1.trajectories) == len(r2.trajectories)
    for point1, point2 in zip(r1.trajectories, r2.trajectories, strict=True):
        for traj1, traj2 in zip(point1, point2, strict=True):
            assert traj1.times == traj2.times
            assert traj1.series == traj2.series


def test_different_sweep_points_use_different_streams(wf_params_factory):
    """Each sweep point seeds independently, so the N=100 and N=200 points are
    not accidental copies (they would be if every point reused one stream)."""
    result = run_experiment(_experiment(), wf_params_factory)
    point_100 = result.trajectories[0]  # N=100
    point_200 = result.trajectories[2]  # N=200
    # Compare the first replicate's freq series; they must differ.
    assert point_100[0].series["freq"] != point_200[0].series["freq"]
