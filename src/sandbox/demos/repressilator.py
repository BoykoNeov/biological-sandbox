"""Repressilator demo — Phase 1's vertical slice, end to end.

Run it:  ``uv run python -m sandbox.demos.repressilator``

Three acts, in the order the project's argument runs:

1. **The engine is right.** ``birth_death`` and ``isomerization`` have exact
   closed forms, so :func:`~sandbox.core.validation.validate` can check the
   Gillespie SSA against a *number*. Nothing downstream is worth looking at if
   these two are red.
2. **The limit is real.** Replicate trajectories of the repressilator at a small
   and a large system size, overlaid on the ODE limit cycle they collapse onto.
   This is the picture; act 3 is the measurement.
3. **The collapse obeys the law.** ``D(Omega) ~ Omega^{-1/2}`` — the
   Kurtz/van-Kampen scaling — checked by
   :func:`~sandbox.core.convergence.convergence_report` and plotted log-log.

**This demo runs a deliberately reduced convergence config** (fewer replicates
than the suite's, one oscillation period instead of two) so it finishes in about
a minute. It is a *demonstration*, not the check: the authoritative run is
``tests/test_repressilator.py``, whose pinned config measures slope
``-0.4606 +/- 0.0734`` in ~245 s and carries the two broken-``Omega`` teeth. The
reduced config was seed-checked (PASS at seeds 0, 1, 2 — slopes -0.399, -0.411,
-0.443) so the demo does not print a red check on a lucky-vs-unlucky draw, but
its error bars are honestly wider and it is printed as such.

Figures are written to the current directory (``*.png`` is gitignored), matching
``demos/wright_fisher.py``. All printed output is ASCII-only: this project is
developed on a Windows console (cp1252) where a stray Unicode glyph raises.
"""

from __future__ import annotations

import numpy as np

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import ConvergenceReport, convergence_report
from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import Experiment
from sandbox.core.recorder import Trajectory
from sandbox.core.sweep import run_experiment
from sandbox.core.validation import validate
from sandbox.models.birth_death import BirthDeathParams
from sandbox.models.isomerization import IsomerizationParams
from sandbox.models.repressilator import MODEL, OBSERVABLE_KEYS, RepressilatorParams

# Measured in the ODE sanity sweep; see tests/test_repressilator.py.
PERIOD = 16.095

_REPRESSILATOR_BASE = {
    "alpha": 216.0,
    "alpha0": 0.216,
    "n_hill": 2.0,
    "beta": 1.0,
    "m0": 0.0,
    "p1_0": 0.0,
    "p2_0": 5.0,
    "p3_0": 15.0,
}

# Reduced relative to the suite (which sweeps 9 sizes out to Omega=16 over 2
# periods at R=12). The fit mask still excludes Omega <= 1: those points sit in
# the phase-saturation knee, where replicates fully dephase and D stops falling.
# They are swept and printed anyway -- where the law stops applying is evidence.
_DEMO_OMEGAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
_DEMO_FIT_MASK = [False, False, True, True, True, True]
_DEMO_REPLICATES = 6
_DEMO_SEED = 0

# The overlay figure: one noisy panel, one tight panel, same axes.
_OVERLAY_OMEGAS = (1.0, 8.0)
_OVERLAY_REPLICATES = 6
_OVERLAY_OBSERVABLE = "x_p1"
# Event sub-sampling. The convergence sweep above must stay at record_every=1
# (sub-sampling sharpens the interpolation *with* Omega and biases the slope), but
# for a figure only the shape matters and Omega=8 emits ~110k events per replicate.
_OVERLAY_RECORD_EVERY = 20


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Act 1 -- the engine, against exact closed forms
# ---------------------------------------------------------------------------


def _validate_engine() -> bool:
    _rule("1. Gillespie engine vs exact closed forms")

    birth_death = validate(
        Experiment(
            model="birth_death",
            params={"k": 2.0, "gamma": 1.0, "Omega": 10.0, "t_max": 10.0, "c0": 0.0},
            replicates=600,
            observables=("x",),
            seed=1,
            max_steps=20_000,
            record_every=20_000,  # only the final value is read
        ),
        lambda d: BirthDeathParams(**d),
    )
    print("birth-death   0 -> X -> 0   (stationary mean <x> = k/gamma = 2.0)")
    print(birth_death)

    isomerization = validate(
        Experiment(
            model="isomerization",
            params={"k1": 1.0, "k2": 3.0, "Omega": 10.0, "t_max": 5.0, "c_tot": 2.0, "cA0": 0.0},
            replicates=600,
            observables=("x_A", "x_B"),
            seed=1,
            max_steps=20_000,
            record_every=20_000,
        ),
        lambda d: IsomerizationParams(**d),
    )
    print("\nisomerization A <-> B      (<x_A> = k2/(k1+k2) * c_tot = 1.5)")
    print(isomerization)

    ok = birth_death.passed and isomerization.passed
    print(
        "\nBoth are exact, Omega-independent predictions and the tolerance is "
        "z * standard error,\nnot a hardcoded epsilon. The repressilator below has "
        "no such closed form -- that is\nwhy the convergence pathway exists."
    )
    return ok


# ---------------------------------------------------------------------------
# Act 2 -- the picture: replicates collapsing onto the limit cycle
# ---------------------------------------------------------------------------


def _run_replicates(omega: float, t_max: float) -> list[Trajectory]:
    result = run_experiment(
        Experiment(
            model="repressilator",
            params={**_REPRESSILATOR_BASE, "Omega": omega, "t_max": t_max},
            replicates=_OVERLAY_REPLICATES,
            observables=OBSERVABLE_KEYS,
            seed=_DEMO_SEED,
            max_steps=4_000_000,
            record_every=_OVERLAY_RECORD_EVERY,
        ),
        lambda d: RepressilatorParams(**d),
    )
    return result.trajectories[0]


def _overlay_figure(plt, plot_replicates) -> str:
    _rule("2. Replicates collapsing onto the deterministic limit cycle")

    t_max = 2.0 * PERIOD
    params = RepressilatorParams(Omega=1.0, t_max=t_max, **_REPRESSILATOR_BASE)
    # The ODE lives in concentration space and carries no Omega, so one integration
    # serves both panels. observables() returns x = n/Omega, so the two overlay with
    # no rescaling -- plotting counts here would be wrong by a factor of Omega.
    t_ode, y_ode = integrate_rk4(
        MODEL.deterministic_rhs(params), MODEL.initial_concentrations(params), t_max, 1e-3
    )
    # Derive the column from the ordered keys. A hardcoded index would pick a
    # neighbouring protein, which on this cyclically symmetric limit cycle is the
    # right shape at the wrong phase -- convincing and wrong.
    column = y_ode[:, OBSERVABLE_KEYS.index(_OVERLAY_OBSERVABLE)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, omega in zip(axes, _OVERLAY_OMEGAS, strict=True):
        trajectories = _run_replicates(omega, t_max)
        events = sum(len(traj.times) for traj in trajectories) * _OVERLAY_RECORD_EVERY
        print(
            f"  Omega={omega:>5g}: {_OVERLAY_REPLICATES} replicates, "
            f"~{events:,} events, mean molecule count ~{omega * 100:.0f}"
        )
        plot_replicates(
            trajectories,
            _OVERLAY_OBSERVABLE,
            deterministic=(t_ode, column),
            title=f"Omega = {omega:g}",
            ax=ax,
            alpha=0.6,
        )
    fig.suptitle(
        f"Repressilator: single trajectories vs the deterministic limit ({_OVERLAY_OBSERVABLE})"
    )
    out = "repressilator_overlay.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(
        "\nSame ODE in both panels, same axes. The cloud tightens with Omega -- act 3\n"
        "measures how fast, which is the part that can actually be wrong.\n"
        "\nOne feature worth naming rather than glossing over: at small Omega the\n"
        "replicate peaks sit systematically ABOVE the ODE, not symmetrically around\n"
        "it. That is a finite-size correction, not a bug -- the Hill term is strongly\n"
        "nonlinear, so the mean of the stochastic system is not the solution of the\n"
        "mean-field equation. It vanishes in the limit: the mean second-cycle peak of\n"
        "x_p1 measures +29.2% at Omega=1, +10.9% at 4, and +1.0% at 16."
    )
    return out


# ---------------------------------------------------------------------------
# Act 3 -- the law: D(Omega) ~ Omega^{-1/2}
# ---------------------------------------------------------------------------


def _convergence(plt, plot_convergence) -> tuple[ConvergenceReport, str]:
    _rule("3. Kurtz convergence: D(Omega) ~ Omega^(-1/2)")
    print(
        "Reduced config (R=6, 1 period) so the demo runs in ~1 minute. The\n"
        "authoritative check is tests/test_repressilator.py: slope -0.4606 +/- 0.0734\n"
        "at R=12 over 2 periods, plus two broken-Omega teeth that must FAIL.\n"
    )

    report = convergence_report(
        "repressilator",
        _REPRESSILATOR_BASE,
        lambda d: RepressilatorParams(**d),
        omegas=_DEMO_OMEGAS,
        t_max=PERIOD,
        dt=1e-3,
        replicates=_DEMO_REPLICATES,
        n_grid=200,
        observable_keys=OBSERVABLE_KEYS,
        seed=_DEMO_SEED,
        z=3.0,
        fit_mask=_DEMO_FIT_MASK,
        n_bootstrap=300,
        max_steps=2_000_000,
    )
    print(report)

    # The compensated view: under the law D*sqrt(Omega) is a constant plateau, and
    # the excluded points fall BELOW it -- that dip is what identifies the knee, and
    # it is the same anchor the test asserts on.
    anchor = report.discrepancy * np.sqrt(report.omegas)
    plateau = float(anchor[report.fit_mask].mean())
    print(f"\n  D(Omega)*sqrt(Omega)  [flat under the law; plateau = {plateau:.2f}]")
    for om, a, used in zip(report.omegas, anchor, report.fit_mask, strict=True):
        note = "" if used else "   <- knee (excluded from the fit, still shown)"
        print(f"    Omega={om:>5g}  {a:7.2f}{note}")

    ax = plot_convergence(
        report.omegas,
        report.discrepancy,
        sem=report.discrepancy_sem,
        fit_mask=report.fit_mask,
        slope=report.slope,
        slope_se=report.slope_se,
        expected_slope=report.expected_slope,
        title="Repressilator: system-size convergence to the deterministic limit",
    )
    out = "repressilator_convergence.png"
    ax.figure.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    return report, out


def main() -> None:
    engine_ok = _validate_engine()

    try:
        import matplotlib.pyplot as plt

        from sandbox.viz.backends.matplotlib_backend import plot_convergence, plot_replicates
    except ModuleNotFoundError:
        print("\n(matplotlib not installed; skipping figures. `uv sync --extra viz` to enable.)")
        return

    overlay_png = _overlay_figure(plt, plot_replicates)
    report, convergence_png = _convergence(plt, plot_convergence)

    _rule("Summary")
    print(f"  engine (exact closed forms) : {'PASS' if engine_ok else 'FAIL'}")
    print(
        f"  repressilator scaling law   : {'PASS' if report.passed else 'FAIL'} "
        f"(slope {report.slope:+.4f} +/- {report.slope_se:.4f}, expected -0.50)"
    )
    print(f"\n  figures: {overlay_png}\n           {convergence_png}")


if __name__ == "__main__":
    main()
