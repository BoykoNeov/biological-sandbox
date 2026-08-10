"""Stochastic gLV demo — Phase 3's stochastic-vs-limit thread, end to end.

Run it:  ``uv run python -m sandbox.demos.glv_stochastic``

Four acts, in the order the argument runs:

1. **The two worlds are one vector field.** The reaction network's mass-action
   limit is checked, pointwise, against the deterministic gLV of 3a — the same
   ``r`` and ``A``, the same interior equilibrium. Nothing below means anything
   if these two have drifted apart.
2. **The limit is real.** Replicate trajectories at a small and a large system
   size, overlaid on the ODE they collapse onto.
3. **The collapse obeys the law.** ``D(Omega) ~ Omega^{-1/2}``, plotted log-log.
4. **What the model refuses, and the error it admits to.** Why there is no
   ``analytic_predictions`` — with the replicate count at which the tempting
   ``<x> = x*`` check would flip from green to red — and the ``O(1/Omega)``
   macroscopic-propensity bias, measured against its prediction.

All printed output is ASCII-only (Windows cp1252 console). Figures are written to
the current directory (``*.png`` is gitignored).
"""

from __future__ import annotations

import numpy as np

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import ConvergenceReport, convergence_report
from sandbox.core.ode import integrate_rk4
from sandbox.core.protocol import Experiment
from sandbox.core.sweep import run_experiment
from sandbox.models.glv import equilibrium, glv_rhs
from sandbox.models.glv_stochastic import (
    MODEL,
    GLVStochasticParams,
    observable_keys,
    propensity_bias,
    replicates_until_bias_dominates,
)

R3 = [1.0, 0.8, 1.2]
A3 = [[-1.0, -0.3, -0.2], [-0.4, -1.0, -0.1], [-0.2, -0.5, -1.0]]
BASE = {"r": R3, "A": A3, "x_init": [0.2, 0.2, 0.2]}
KEYS = observable_keys(3)
T_MAX = 20.0


def factory(d: dict) -> GLVStochasticParams:
    return GLVStochasticParams(**d)


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Act 1 — the two worlds are one vector field
# ---------------------------------------------------------------------------


def act1_same_vector_field() -> np.ndarray:
    _rule("1. One vector field, two worlds")
    params = GLVStochasticParams(Omega=100.0, t_max=T_MAX, **BASE)
    stochastic_limit = MODEL.deterministic_rhs(params)
    deterministic = glv_rhs(params.deterministic_params())

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = rng.uniform(0.0, 3.0, size=3)
        worst = max(worst, float(np.abs(stochastic_limit(x) - deterministic(x)).max()))
    print("  The SSA's propensities and the ODE's RHS are built from ONE rates(c),")
    print("  so the two worlds cannot silently disagree about the same network.")
    print(f"  max |network limit - glv_rhs| over 200 random points: {worst:.3g}")

    x_star = equilibrium(params.deterministic_params())
    print(f"  interior equilibrium x* = {np.array2string(x_star, precision=8)}")
    print(f"  |rhs(x*)|               = {np.abs(deterministic(x_star)).max():.3g}")
    print(f"  total propensity at x*  = {2 * float(np.dot(R3, x_star)):.4f} * Omega events/time")
    return x_star


# ---------------------------------------------------------------------------
# Act 2 — the picture
# ---------------------------------------------------------------------------


def act2_trajectories(x_star: np.ndarray) -> None:
    _rule("2. Demographic noise shrinking with the system size")
    params = GLVStochasticParams(Omega=1.0, t_max=T_MAX, **BASE)
    t_ode, y_ode = integrate_rk4(
        MODEL.deterministic_rhs(params), MODEL.initial_concentrations(params), T_MAX, 1e-3
    )

    panels = []
    for omega in (30.0, 1000.0):
        result = run_experiment(
            Experiment(
                model="glv_stochastic",
                params={**BASE, "Omega": omega, "t_max": T_MAX},
                replicates=6,
                observables=[*KEYS, "n_survivors"],
                seed=0,
                max_steps=4_000_000,
            ),
            factory,
        )
        trajectories = result.trajectories[0]
        spread = float(np.mean([abs(t.series["x0"][-1] - x_star[0]) for t in trajectories]))
        lost = sum(1 for t in trajectories if float(np.min(t.series["n_survivors"])) < 3)
        print(
            f"  Omega={omega:>7.0f}  n0(t=0)={round(omega * 0.2):>4}  "
            f"mean |x0(T) - x*_0| = {spread:.5f}   replicates losing a species: {lost}/6"
        )
        panels.append((omega, trajectories))

    _plot_trajectories(panels, t_ode, y_ode, x_star)


def _plot_trajectories(panels, t_ode, y_ode, x_star) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - viz is an optional extra
        print("  (matplotlib not installed -- skipping the figure; use --extra viz)")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4), sharey=True)
    colors = ("C0", "C1", "C2")
    for ax, (omega, trajectories) in zip(axes, panels, strict=True):
        for traj in trajectories:
            times, series = traj.as_arrays()
            for s, key in enumerate(KEYS):
                ax.step(times, series[key], where="post", color=colors[s], alpha=0.35, lw=0.8)
        for s, key in enumerate(KEYS):
            ax.plot(t_ode, y_ode[:, s], color=colors[s], lw=2.0, label=f"ODE {key}")
        ax.set_title(f"Omega = {omega:g}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("concentration")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Stochastic gLV: single scaled trajectories against their deterministic limit")
    fig.tight_layout()
    fig.savefig("glv_stochastic_trajectories.png", dpi=130)
    plt.close(fig)
    print("  wrote glv_stochastic_trajectories.png")


# ---------------------------------------------------------------------------
# Act 3 — the law
# ---------------------------------------------------------------------------


def act3_convergence() -> ConvergenceReport:
    _rule("3. D(Omega) ~ Omega^{-1/2}")
    print("  A REDUCED config (fewer replicates than the suite's) so the demo finishes")
    print("  quickly. The authoritative run is tests/test_glv_stochastic.py.")
    report = convergence_report(
        "glv_stochastic",
        BASE,
        factory,
        omegas=[100.0, 200.0, 400.0, 800.0],
        t_max=T_MAX,
        dt=1e-3,
        replicates=6,
        n_grid=200,
        observable_keys=KEYS,
        seed=0,
        z=3.0,
        n_bootstrap=300,
        max_steps=8_000_000,
    )
    print(report)
    _plot_convergence(report)
    return report


def _plot_convergence(report: ConvergenceReport) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("  (matplotlib not installed -- skipping the figure; use --extra viz)")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(
        report.omegas,
        report.discrepancy,
        yerr=report.discrepancy_sem,
        fmt="o",
        capsize=3,
        label="measured D(Omega)",
    )
    grid = np.array([report.omegas[0], report.omegas[-1]], dtype=float)
    anchor = float(report.discrepancy[0]) * (grid / grid[0]) ** -0.5
    ax.plot(grid, anchor, "k--", lw=1.2, label="Omega^-1/2 (predicted, not fitted)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Omega")
    ax.set_ylabel("D(Omega)")
    ax.set_title(f"fitted slope {report.slope:.4f} +/- {report.slope_se:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("glv_stochastic_convergence.png", dpi=130)
    plt.close(fig)
    print("  wrote glv_stochastic_convergence.png")


# ---------------------------------------------------------------------------
# Act 4 — the refusal, and the error the model admits to
# ---------------------------------------------------------------------------


def act4_refusal_and_bias() -> None:
    _rule("4. What the model refuses to predict, and the bias it admits to")

    print("  (a) No analytic_predictions. The tempting closed form is <x> = x*, and it")
    print("      is wrong at O(1/Omega): the van Kampen nonlinearity correction plus")
    print("      the macroscopic-propensity bias below. It would PASS today and FAIL")
    print("      as replicates grow, because validate()'s tolerance shrinks as")
    print("      1/sqrt(R) while the offset does not. Measured crossing point:")
    print()
    print(
        f"  {'Omega':>7} {'<x0>':>12} {'x*_0':>12} {'sd':>11} {'bias':>11} {'R at bias/SE=1':>16}"
    )
    x_star = equilibrium(GLVStochasticParams(Omega=1.0, t_max=1.0, **BASE).deterministic_params())
    for omega in (100.0, 400.0, 1600.0):
        result = run_experiment(
            Experiment(
                model="glv_stochastic",
                params={**BASE, "Omega": omega, "t_max": T_MAX},
                replicates=24,
                observables=KEYS,
                seed=0,
                max_steps=4_000_000,
            ),
            factory,
        )
        finals = np.array([t.series["x0"][-1] for t in result.trajectories[0]])
        params = GLVStochasticParams(Omega=omega, t_max=T_MAX, **BASE)
        sd = float(finals.std(ddof=1))
        bias = float(propensity_bias(params)[0])
        print(
            f"  {omega:>7.0f} {finals.mean():>12.6f} {x_star[0]:>12.6f} {sd:>11.5f} "
            f"{bias:>11.3e} {replicates_until_bias_dominates(params, sd):>16.0f}"
        )
    print()
    print("      A green check whose lifetime is a replicate count is not a check.")
    print("      (3b's fraction_report refuses a bias-limited configuration for the")
    print("      same reason; 3e flags it for the canonical equation.)")

    print()
    print("  (b) The macroscopic-propensity bias, stated rather than hidden. This engine")
    print("      uses |A_ii| n^2/Omega where the exact self-limitation propensity is")
    print("      |A_ii| n(n-1)/Omega -- a molecule cannot react with itself. The excess")
    print("      loss is an effective r' = r - |A_ii|/Omega, so:")
    print()
    print("      bias = (-A)^-1 diag|A_ii| 1 / Omega -- the (-A)^-1 solve is the whole")
    print("      claim: the excess loss is a per-species shift to r, and what that does")
    print("      to the equilibrium couples all three species.")
    print()
    print(f"  {'Omega':>7}   {'predicted':>32}   {'measured (T=500, R=8)':>34}   {'ratio':>22}")
    for omega, measured in (
        (50.0, None),
        (100.0, (7.43e-3, 7.44e-3, 4.24e-3)),
        (200.0, (3.75e-3, 3.64e-3, 2.49e-3)),
        (400.0, (1.71e-3, 1.84e-3, 1.34e-3)),
    ):
        bias = propensity_bias(GLVStochasticParams(Omega=omega, t_max=T_MAX, **BASE))
        if measured is None:
            print(
                f"  {omega:>7.0f}   {np.array2string(bias, precision=6):>32}   "
                f"{'UNRESOLVED (SE > signal)':>34}   {'--':>22}"
            )
            continue
        m = np.asarray(measured)
        print(
            f"  {omega:>7.0f}   {np.array2string(bias, precision=6):>32}   "
            f"{np.array2string(m, precision=6):>34}   "
            f"{np.array2string(m / bias, precision=3):>22}"
        )
    print()
    print("      Split-coupled (Anderson 2012): the two arms run as ONE chain, since the")
    print("      macro arm is the exact arm plus S extra loss channels. Independent arms")
    print("      cannot resolve this at any affordable cost -- SE 6.21e-3 independent,")
    print("      1.36e-3 common-random-number, 9.25e-4 coupled, at equal cost.")
    print()
    print("      Weighted log-log slopes over Omega = 100/200/400 (predicted -1):")
    print("        species 0  -1.0525 +/- 0.0624      species 1  -0.9945 +/- 0.0870")
    print("        species 2  -0.8522 +/- 0.1378")
    print("      All three are within 1.1 sigma of -1 and 2.6-8.9 sigma from -1/2, which")
    print("      is the point: the bias is SUBDOMINANT to act 3's signal and cannot floor")
    print("      that slope. Those SEs propagate each point's own error -- fitting only")
    print("      the residuals of 3 points quotes +/-0.013 for species 1, 7x too tight,")
    print("      and would have invented a physical effect on species 2.")
    print("      Omega = 50 stays unresolved even here: the bias cannot be measured where")
    print("      it is largest. The shipped assertion is in tests/test_glv_stochastic.py.")

    print()
    print("  (c) Why the assertion is on the VECTOR, and not on species 0.")
    print()
    candidates = _bias_candidates(_BIAS_DEMO_OMEGA)
    print(f"      {'formula':<40}" + "".join(f"{f'species {i}':>12}" for i in range(3)))
    for name, vec in candidates.items():
        rel = vec / candidates[_CORRECT_KEY] - 1.0
        print(f"      {name:<40}" + "".join(f"{100 * v:>+11.1f}%" for v in rel))
    print()
    print("      Dropping the (-A)^-1 solve is caught anywhere -- it is 44-90% off. The")
    print("      two errors that are STRUCTURALLY similar to the truth are not: a")
    print("      transposed A lands 1.7% away on species 0 and the plan's x*/Omega 1.0%")
    print("      away, while both are 35-60% off on species 1 and 2. Species 0 is the")
    print("      component every earlier measurement in this phase reported.")
    print()
    print("      Excluding the transpose on species 0 alone would need SE <= 2.4e-5,")
    print("      unreachable at any affordable cost; on species 1 and 2 it needs ~6e-4,")
    print("      which the shipped configuration reaches.")
    _plot_bias_candidates(candidates)


# The deepest recorded measurement (T = 500, R = 8, split-coupled, burn = 20).
# Recorded, not re-run here -- 490 s. Reproduce with the slice or the test.
_BIAS_DEMO_OMEGA = 100.0
_BIAS_DEMO_MEAN = np.array([7.430e-3, 7.440e-3, 4.240e-3])
_BIAS_DEMO_SEM = np.array([3.402e-4, 1.130e-3, 9.600e-4])
_CORRECT_KEY = "(-A)^-1 diag|A_ii| 1 / Omega (correct)"


def _bias_candidates(omega: float) -> dict[str, np.ndarray]:
    """The correct bias vector and the three formulas that nearly pass for it."""
    a = np.asarray(A3, dtype=float)
    diag = np.abs(np.diag(a))
    params = GLVStochasticParams(Omega=omega, t_max=T_MAX, **BASE)
    return {
        _CORRECT_KEY: propensity_bias(params),
        "|A_ii| / Omega (no linear-response solve)": diag / omega,
        "transposed A": np.linalg.solve(-a.T, diag) / omega,
        "x* / Omega (the plan's)": equilibrium(params.deterministic_params()) / omega,
    }


def _plot_bias_candidates(candidates: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("  (matplotlib not installed -- skipping the figure; use --extra viz)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    styles = (("C0", "-"), ("C3", "--"), ("C1", "-."), ("C2", ":"))
    for i, ax in enumerate(axes):
        ax.errorbar(
            [0.0],
            [_BIAS_DEMO_MEAN[i]],
            yerr=[_BIAS_DEMO_SEM[i]],
            fmt="ko",
            capsize=5,
            label="measured (recorded)" if i == 0 else None,
        )
        for (name, vec), (color, ls) in zip(candidates.items(), styles, strict=True):
            ax.axhline(vec[i], color=color, ls=ls, lw=1.6, label=name if i == 0 else None)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_title(f"species {i}")
    axes[0].set_ylabel(f"<x>_exact - <x>_macro  at Omega = {_BIAS_DEMO_OMEGA:g}")
    fig.legend(loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle(
        "On species 0 the transposed-A and x*/Omega formulas are indistinguishable\n"
        "from the correct one; species 1 and 2 separate them "
        "(recorded T = 500, R = 8; bar = 1 SE)"
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig("glv_stochastic_bias_candidates.png", dpi=130)
    plt.close(fig)
    print("  wrote glv_stochastic_bias_candidates.png")


def main() -> None:
    print("Stochastic generalized Lotka-Volterra -- Phase 3c")
    x_star = act1_same_vector_field()
    act2_trajectories(x_star)
    report = act3_convergence()
    act4_refusal_and_bias()

    _rule("Summary")
    print(
        f"  convergence check: {'PASS' if report.passed else 'FAIL'} "
        f"(slope {report.slope:.4f} +/- {report.slope_se:.4f}, expected -0.5)"
    )
    print("  the model's claim is the SCALING LAW, not a stationary number -- and the")
    print("  one systematic error it carries is printed above rather than argued away.")


if __name__ == "__main__":
    main()
