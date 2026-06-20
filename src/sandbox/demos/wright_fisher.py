"""Wright-Fisher demo — the full vertical slice, end to end.

Run it:  ``uv run python -m sandbox.demos.wright_fisher``

It (1) validates the neutral fixation-probability prediction over many seeded
replicates, printing the statistical check, and (2) if matplotlib is installed,
saves a figure overlaying replicate frequency trajectories on the deterministic
neutral limit (a flat line at p0).
"""

from __future__ import annotations

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.protocol import Experiment
from sandbox.core.sweep import run_experiment
from sandbox.core.validation import validate
from sandbox.models.wright_fisher import WFParams


def _params_factory(d: dict) -> WFParams:
    return WFParams(**d)


def main() -> None:
    p0 = 0.3
    experiment = Experiment(
        model="wright_fisher",
        params={"N": 200, "p0": p0, "s": 0.0},
        replicates=2000,
        observables=("freq", "fixed_A"),
        seed=12345,
        max_steps=20_000,
        record_every=1,
    )

    report = validate(experiment, _params_factory)
    print(report)
    print()
    print(
        "Interpretation: the neutral fixation probability of allele A equals its "
        f"initial frequency p0 = {p0}. The measured fixation fraction above should "
        "sit within a few standard errors of that value."
    )

    try:
        import matplotlib.pyplot as plt

        from sandbox.viz.backends.matplotlib_backend import plot_replicates
    except ModuleNotFoundError:
        print("\n(matplotlib not installed; skipping figure. `uv sync --extra viz` to enable.)")
        return

    # A handful of full frequency trajectories for the figure.
    fig_experiment = Experiment(
        model="wright_fisher",
        params={"N": 200, "p0": p0, "s": 0.0},
        replicates=40,
        observables=("freq",),
        seed=7,
        max_steps=20_000,
    )
    result = run_experiment(fig_experiment, _params_factory)
    trajectories = result.trajectories[0]

    max_t = max(t for traj in trajectories for t in traj.times)
    ax = plot_replicates(
        trajectories,
        "freq",
        deterministic=([0.0, max_t], [p0, p0]),
        title=f"Wright-Fisher drift (N={200}, p0={p0}) vs the deterministic limit",
    )
    ax.set_ylim(0.0, 1.0)
    out = "wright_fisher_drift.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
