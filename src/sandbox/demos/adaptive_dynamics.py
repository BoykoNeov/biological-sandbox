"""Adaptive dynamics demo — a limit that is a slope, and a split that is a sign.

Run it:  ``uv run python -m sandbox.demos.adaptive_dynamics``

Four acts:

1. **The trait-space closed forms.** The selection gradient and both second
   derivatives, against central differences, signed. The branching criterion is
   the *sign* of one of them, and the two of them must sum to ``dD/dx`` — an
   identity that checks both at once.
2. **The canonical equation as a limit, and the trap in checking it.** The
   discrepancy between the stochastic mean and the canonical prediction is
   ``O(sm)``. Printed alongside three *wrong* canonical equations, because the
   interesting fact is not that the correct one fits — it is that all three wrong
   ones are **more** significantly nonzero than the correct one, so any one-sided
   "is the discrepancy there" check passes every one of them.
3. **Branching.** The sign change across ``sa = sK``, and the divergence of the
   branching time as the reciprocal of the splitting rate. Also the honest
   description of what is being simulated: with three seeded bins this is a
   **3-species gLV**, and the grid enters only through its spacing.
4. **What is refused, and what is labelled exploratory.** The model declines to
   predict a branching outcome; the post-branching morph structure is
   exploratory, with the *mechanism* as the stated reason rather than a bare
   grid-dependence measurement; and the divergence law's prefactor is a good
   predictor that is explicitly not a closed form.

All printed output is ASCII-only (Windows cp1252 console). Figures are written to
the current directory (``*.png`` is gitignored).
"""

from __future__ import annotations

import time

import numpy as np
from numpy.random import SeedSequence

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.ode import integrate_rk4, rk4_step
from sandbox.core.rng import spawn_rngs
from sandbox.models.adaptive_dynamics import (
    CANONICAL_DU,
    MODEL,
    U_TARGET,
    AdaptiveDynamicsParams,
    canonical_rhs,
    canonical_trait,
    gradient_slope,
    invasion_fitness,
    mutant_curvature,
    resident_curvature,
    run_cohort,
    selection_gradient,
    splitting_rate,
)
from sandbox.models.trait_branching import (
    MODEL as BRANCHING_MODEL,
)
from sandbox.models.trait_branching import (
    TraitBranchingParams,
    centre_index,
    find_branch_time,
    initial_abundances,
    predicted_product,
    trait_branching_rhs,
    trait_grid,
)

SWEEP: tuple[tuple[float, int], ...] = (
    (0.15, 3_300),
    (0.10, 5_000),
    (0.05, 10_000),
    (0.025, 20_000),
    (0.0125, 40_000),
)
N_GROUPS = 5
SCALING_SIGMA_A = (0.60, 0.70, 0.80, 0.90)
NO_BRANCH_SIGMA_A = (1.05, 1.5)


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Act 1 -- the closed forms
# --------------------------------------------------------------------------


def act1_the_closed_forms() -> None:
    _rule("Act 1 -- invasion fitness, its derivatives, and the branching criterion")
    p = AdaptiveDynamicsParams()
    print(
        f"  Gaussian competition model: sigma_a = {p.sigma_a}, "
        f"sigma_k = {p.sigma_k}, r = {p.r_growth}"
    )
    print("  s_x(y) = r (1 - a(y,x) K(x) / K(y)),  a Gaussian in (y - x), K a Gaussian in x")
    print()
    print("  Closed forms against central differences of s_x(y), signed:")
    print(
        f"    {'x':>6} {'D(x)':>12} {'d/dy (FD)':>12} {'d2s/dy2':>12} "
        f"{'FD':>12} {'d2s/dxdy':>12} {'FD':>12}"
    )

    def s(y: float, x: float) -> float:
        return float(invasion_fitness(y, x, p))

    for x in (-2.0, -0.4, 0.0, 0.7, 2.0):
        h = 5e-4
        d1 = (s(x + h, x) - s(x - h, x)) / (2 * h)
        d2 = (s(x + h, x) - 2 * s(x, x) + s(x - h, x)) / h**2
        dxy = (s(x + h, x + h) - s(x - h, x + h) - s(x + h, x - h) + s(x - h, x - h)) / (4 * h * h)
        print(
            f"    {x:>6.1f} {float(selection_gradient(x, p)):>12.6f} {d1:>12.6f} "
            f"{float(mutant_curvature(x, p)):>12.6f} {d2:>12.6f} "
            f"{float(resident_curvature(x, p)):>12.6f} {dxy:>12.6f}"
        )

    print()
    print("  The two second derivatives must sum to dD/dx, because D is a derivative")
    print("  ALONG the diagonal y = x.  That identity checks both closed forms at once:")
    for x in (-2.0, 0.0, 2.0):
        total = float(mutant_curvature(x, p)) + float(resident_curvature(x, p))
        print(f"    x = {x:>5.1f}:  {total:+.15f}  vs  dD/dx = {gradient_slope(p):+.15f}")

    print()
    print("  The branching criterion is the SIGN of d2s/dy2 at the singular point:")
    print(f"    {'sigma_a':>9} {'splitting rate':>15}   verdict")
    for sigma_a in (0.5, 0.7, 0.9, 0.999, 1.0, 1.001, 1.05, 1.5):
        q = AdaptiveDynamicsParams(sigma_a=sigma_a)
        rate = splitting_rate(q)
        verdict = "fitness MINIMUM -> branches" if rate > 0 else "fitness maximum -> ESS"
        if rate == 0.0:
            verdict = "degenerate (sa = sK exactly)"
        print(f"    {sigma_a:>9g} {rate:>+15.6f}   {verdict}")
    print()
    print("  Note the singular point is ALWAYS convergence stable: dD/dx = -r/sK^2 < 0")
    print("  whatever sigma_a does.  Evolution goes there either way; what sigma_a")
    print("  decides is whether it stays one species when it arrives.")


# --------------------------------------------------------------------------
# Act 2 -- the canonical equation
# --------------------------------------------------------------------------


def _weighted_loglog(sm, err, sem):
    log_sm, log_err, log_sem = np.log(sm), np.log(np.abs(err)), sem / np.abs(err)
    w = 1.0 / log_sem**2
    total = w.sum()
    mx, my = (w * log_sm).sum() / total, (w * log_err).sum() / total
    sxx = (w * (log_sm - mx) ** 2).sum()
    slope = (w * (log_sm - mx) * (log_err - my)).sum() / sxx
    return float(slope), float(1.0 / np.sqrt(sxx))


def _target_after(rhs, u: float) -> float:
    if u == 0.0:
        return 2.0
    _, ys = integrate_rk4(rhs, 2.0, u, CANONICAL_DU)
    return float(ys[-1, 0])


def act2_the_canonical_limit() -> dict:
    _rule("Act 2 -- the canonical equation, and why the check is a slope not a bound")
    p = AdaptiveDynamicsParams()
    print("  dx/dt = (1/2) mu sm^2 K(x) D(x)  is the sm -> 0 limit of the")
    print("  trait-substitution sequence.  Every run travels the same scaled")
    print(f"  distance U = {U_TARGET}, so the prediction is ONE number for the whole sweep:")
    print(f"    canonical x(U) = {canonical_trait(p):.12f}")
    print()
    print("  A tolerance at a single sm was measured to tighten onto a REAL O(sm)")
    print("  bias -- five independent replicate groups all landed above the")
    print("  prediction.  Such a check fails a correct model as replicates grow.")
    print()

    branches = SeedSequence(0).spawn(len(SWEEP))
    sigma_m, mean, sem, saturation = [], [], [], []
    t0 = time.perf_counter()
    for (sm, n_rep), branch in zip(SWEEP, branches, strict=True):
        results = [
            run_cohort(AdaptiveDynamicsParams(sigma_m=sm), rng, n_rep // N_GROUPS)
            for rng in spawn_rngs(branch, N_GROUPS)
        ]
        finals = np.concatenate([r.x_final for r in results])
        events = sum(r.n_events for r in results)
        sigma_m.append(sm)
        mean.append(float(finals.mean()))
        sem.append(float(finals.std(ddof=1)) / np.sqrt(finals.size))
        saturation.append(sum(r.frac_saturated * r.n_events for r in results) / events)
    elapsed = time.perf_counter() - t0
    sigma_m, mean, sem = np.array(sigma_m), np.array(mean), np.array(sem)
    saturation = np.array(saturation)

    print(
        f"    {'sm':>8} {'R':>7} {'mean':>11} {'discrepancy':>13} "
        f"{'SE':>10} {'z':>7} {'err/sm':>9} {'sat%':>6}"
    )
    for i, (sm, n_rep) in enumerate(SWEEP):
        err = mean[i] - canonical_trait(p)
        print(
            f"    {sm:>8g} {n_rep:>7d} {mean[i]:>11.6f} {err:>+13.3e} {sem[i]:>10.2e} "
            f"{err / sem[i]:>7.2f} {err / sm:>9.5f} {100 * saturation[i]:>6.2f}"
        )
    print(f"    ({elapsed:.1f} s)")

    correct = mean - canonical_trait(p)
    slope, slope_se = _weighted_loglog(sigma_m, correct, sem)
    print()
    print(f"  Weighted log-log slope: {slope:+.4f} +- {slope_se:.4f}   (the claim is 1)")
    print("  err/sm is flat, so no O(sm^2) correction is resolvable over this range.")

    print()
    print("  Now the same measured means against three WRONG canonical equations.")
    print("  This is the point of the act: read the last column.")
    rhs = canonical_rhs(p)
    teeth = {
        "correct": lambda _sm: _target_after(rhs, U_TARGET),
        "drop the 1/2": lambda _sm: _target_after(rhs, 2.0 * U_TARGET),
        "omit K(x)": lambda _sm: _target_after(lambda y: selection_gradient(y, p), U_TARGET),
        "sm for sm^2": lambda sm: _target_after(rhs, U_TARGET / sm),
    }
    print()
    print(f"    {'equation':<16} {'slope':>10} {'SE':>9} {'|slope|/SE':>11}  verdict")
    rows = {}
    for name, fn in teeth.items():
        err = mean - np.array([fn(sm) for sm in sigma_m])
        sl, se = _weighted_loglog(sigma_m, err, sem)
        rows[name] = (sl, se, err)
        inside = 0.6 <= sl <= 1.4
        print(
            f"    {name:<16} {sl:>+10.4f} {se:>9.4f} {abs(sl) / se:>11.1f}  "
            f"{'in the band [0.6, 1.4]' if inside else 'REJECTED by the band'}"
        )
    print()
    print("  Every wrong equation is MORE significantly nonzero than the correct one,")
    print("  with a standard error 30-200x smaller, because a constant offset fits a")
    print("  near-perfect line.  'The discrepancy is significantly nonzero' passes all")
    print("  three.  Only a two-sided band around 1 separates them.")
    return {"sigma_m": sigma_m, "sem": sem, "rows": rows}


# --------------------------------------------------------------------------
# Act 3 -- branching
# --------------------------------------------------------------------------


def act3_branching() -> dict:
    _rule("Act 3 -- the sign change, and how fast the split diverges")
    print("  On a trait grid the same model is a gLV:  dn_i/dt = r n_i (1 - (A n)_i / K_i).")
    print("  The centre bin is seeded at its capacity and its two neighbours at 1e-6;")
    print("  every other bin is left at exactly 0.0, and n_i = 0 is a gLV invariant.")
    print("  So this is a 3-SPECIES gLV, and the grid enters only through its spacing.")
    print()

    rows, trajectories = [], {}
    for sigma_a in SCALING_SIGMA_A:
        p = TraitBranchingParams(sigma_a=sigma_a, t_max=120_000.0)
        t0 = time.perf_counter()
        result = find_branch_time(p)
        rows.append((sigma_a, splitting_rate(p.trait_params), result.t_refined))
        print(
            f"    sigma_a = {sigma_a:<5g} rate = {splitting_rate(p.trait_params):>+8.4f}   "
            f"branches at t = {result.t_refined:>10.1f}   "
            f"product = {result.t_refined * splitting_rate(p.trait_params):>10.2f}   "
            f"({time.perf_counter() - t0:.1f} s)"
        )
    print()
    for sigma_a in NO_BRANCH_SIGMA_A:
        p = TraitBranchingParams(sigma_a=sigma_a)
        t0 = time.perf_counter()
        result = find_branch_time(p)
        mid = centre_index(p)
        print(
            f"    sigma_a = {sigma_a:<5g} rate = {splitting_rate(p.trait_params):>+8.4f}   "
            f"{'BRANCHES' if result.branched else 'no branch by t = 200 000':<25} "
            f"centre = {result.n_final[mid]:.6f}, neighbours = {result.n_final[mid + 1]:.3e}   "
            f"({time.perf_counter() - t0:.1f} s)"
        )
    print()
    print("  The absence claim is horizon-bounded and says so: t = 200 000 is past")
    print("  every branch time this model exhibits (sigma_a = 0.95 takes 149 822).")
    print("  Checked at 20 000, as it first was, it would be a statement about")
    print("  patience rather than about sigma_a.")

    rates = np.array([r for _, r, _ in rows])
    times = np.array([t for _, _, t in rows])
    log_x, log_y = np.log(rates), np.log(times)
    slope = ((log_x - log_x.mean()) * (log_y - log_y.mean())).sum() / (
        (log_x - log_x.mean()) ** 2
    ).sum()
    print()
    print(f"  t_branch ~ rate^{slope:.5f}   (the claim is -1; fitted over a 7.6x range in rate)")
    print(f"  Products: {'  '.join(f'{p:.1f}' for p in rates * times)}")
    print(f"  Prefactor law predicts {predicted_product(TraitBranchingParams()):.1f} -- a good")
    print("  predictor with its slope fixed by the mechanism and its offset FITTED.")
    print("  It is not a closed form: the centre bin's fall is underived and leaves")
    print("  1.39 unexplained in the h -> 0 limit.  The exponent is the claim.")

    # One trajectory for the figure, recorded as it runs.
    p = TraitBranchingParams(sigma_a=0.6, t_max=120_000.0)
    rhs = trait_branching_rhs(p)
    n = initial_abundances(p)
    mid = centre_index(p)
    keep = (mid - 1, mid, mid + 1)
    ts, series = [0.0], [[n[k] for k in keep]]
    for i in range(1, int(round(p.t_max / p.dt)) + 1):
        n = rk4_step(rhs, n, p.dt)
        if i % 200 == 0:
            ts.append(i * p.dt)
            series.append([n[k] for k in keep])
            if n[mid] < p.threshold and n[mid + 1] > p.threshold:
                break
    trajectories["t"] = np.array(ts)
    trajectories["n"] = np.array(series)
    trajectories["traits"] = trait_grid(p)[list(keep)]
    return {"rows": rows, "trajectory": trajectories}


# --------------------------------------------------------------------------
# Act 4 -- refusals and labels
# --------------------------------------------------------------------------


def act4_refusals_and_labels() -> None:
    _rule("Act 4 -- what is refused, and what is labelled exploratory")
    print("  The branching model declines to predict a branching outcome:")
    try:
        BRANCHING_MODEL.analytic_predictions(TraitBranchingParams(initial="branching"))
    except ValueError as exc:
        print(f"    ValueError: {exc}")
    print()
    print("  ...and the trait-substitution model has no analytic_predictions at all:")
    print(f"    hasattr(MODEL, 'analytic_predictions') = {hasattr(MODEL, 'analytic_predictions')}")
    print("    The canonical value is a LIMIT, not an identity.  A validate()")
    print("    tolerance would tighten onto the O(sm) bias act 2 measured.")
    print()
    print("  EXPLORATORY (category C) -- not asserted against any bound:")
    print("    * The number and positions of the post-branching morphs.  They sit at")
    print("      exactly +-h at every resolution, and the REASON is now mechanical:")
    print("      only three bins are ever seeded, and an empty bin stays empty in")
    print("      pure gLV.  That is stronger than 'grid-dependent' and it is a")
    print("      different claim from the Gyllenberg-Meszena degeneracy argument,")
    print("      which is about the CONTINUUM model and is the literature-anchored one.")
    print("    * The divergence law's prefactor, which moves with the detection")
    print("      threshold and the seed amplitude by up to 23%.")
    print("    * Anything read off the ecology of the two morphs after they split.")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _plot(sweep: dict, branching: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -- skipping the figures; use --extra viz)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    # (a) Pairwise invasibility plot at the branching sigma_a.
    ax = axes[0, 0]
    p = AdaptiveDynamicsParams(sigma_a=0.7)
    grid = np.linspace(-1.6, 1.6, 401)
    s = np.asarray(invasion_fitness(grid[None, :], grid[:, None], p))
    ax.contourf(grid, grid, (s > 0).T, levels=[-0.5, 0.5, 1.5], colors=["#f2f2f2", "#8ab4d6"])
    ax.plot(grid, grid, "k-", lw=0.8)
    ax.axvline(0.0, color="crimson", lw=1.0, ls="--")
    ax.set_xlabel("resident trait x")
    ax.set_ylabel("mutant trait y")
    ax.set_title(
        f"(a) Mutants that can invade (shaded), sigma_a = {p.sigma_a}\n"
        "at x = 0 BOTH directions invade: a fitness minimum",
        fontsize=9,
    )

    # (b) The O(sm) law, with the three wrong equations on the same axes.
    ax = axes[0, 1]
    sigma_m = sweep["sigma_m"]
    styles = {
        "correct": ("o-", "#1f4e79"),
        "drop the 1/2": ("s--", "#c0504d"),
        "omit K(x)": ("^--", "#9bbb59"),
        "sm for sm^2": ("v--", "#8064a2"),
    }
    for name, (style, colour) in styles.items():
        slope, _, err = sweep["rows"][name]
        ax.plot(sigma_m, np.abs(err), style, color=colour, label=f"{name} (slope {slope:+.2f})")
    ax.plot(sigma_m, 0.177 * sigma_m, "k:", lw=1.0, label="slope 1 reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mutation step sm")
    # NOT "canonical prediction" -- three of these four curves are measured
    # against a DIFFERENT equation's prediction, which is the whole point.
    ax.set_ylabel("|mean - that equation's prediction|")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(
        "(b) Only the correct equation's discrepancy VANISHES.\n"
        "The flat lines are all 'significantly nonzero'.",
        fontsize=9,
    )

    # (c) The branching trajectory -- three bins, and only three ever exist.
    #
    # The two neighbours are EXACTLY symmetric, so a naive plot draws one on top
    # of the other and the eye counts two curves under a title claiming three.
    # That is how a figure lies without stating anything false. They are drawn at
    # different widths and the coincidence is named in the legend, so the overlap
    # reads as a result rather than as a missing line.
    ax = axes[1, 0]
    traj = branching["trajectory"]
    left, centre, right = traj["n"][:, 0], traj["n"][:, 1], traj["n"][:, 2]
    coincide = np.array_equal(left, right)
    ax.semilogy(
        traj["t"],
        left,
        color="#1f4e79",
        lw=4.0,
        alpha=0.35,
        label=f"morph at trait {traj['traits'][0]:+.2f}",
    )
    ax.semilogy(
        traj["t"],
        right,
        color="#2e8b57",
        lw=1.4,
        label=f"morph at trait {traj['traits'][2]:+.2f}"
        + (" (bit-identical to the other)" if coincide else ""),
    )
    ax.semilogy(traj["t"], centre, color="#c0504d", lw=1.6, label="resident at trait +0.00")
    ax.axhline(1e-3, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("abundance")
    ax.legend(fontsize=7.5)
    ax.set_title(
        "(c) Branching at sigma_a = 0.6: the resident falls, both\n"
        "neighbours rise.  158 other bins are exactly 0.0 throughout.",
        fontsize=9,
    )

    # (d) The divergence law.
    ax = axes[1, 1]
    rates = np.array([r for _, r, _ in branching["rows"]])
    times = np.array([t for _, _, t in branching["rows"]])
    ax.loglog(rates, times, "o", color="#1f4e79", label="measured")
    reference_line = times[0] * (rates / rates[0]) ** -1.0
    ax.loglog(rates, reference_line, "k--", lw=1.0, label="exponent -1")
    for sigma_a, rate, t_branch in branching["rows"]:
        ax.annotate(
            f"sa={sigma_a:g}",
            (rate, t_branch),
            fontsize=7.5,
            textcoords="offset points",
            xytext=(5, -9),
        )
    ax.set_xlabel("splitting rate  r (1/sa^2 - 1/sK^2)")
    ax.set_ylabel("branch time")
    ax.legend(fontsize=8)
    ax.set_title(
        "(d) The branch time diverges as 1 / rate.\n"
        "The exponent is the claim; the prefactor moves with the detector.",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig("adaptive_dynamics.png", dpi=130)
    plt.close(fig)
    print("  wrote adaptive_dynamics.png")


def main() -> None:
    print("Adaptive dynamics -- Phase 3e")
    act1_the_closed_forms()
    sweep = act2_the_canonical_limit()
    branching = act3_branching()
    act4_refusals_and_labels()

    _rule("Figures")
    _plot(sweep, branching)

    _rule("Summary")
    print("  Two limits share this phase and they are limits in different senses.")
    print("  The canonical equation is what the jump process becomes as the mutation")
    print("  step goes to zero -- category B, asserted as a SLOPE, because a tolerance")
    print("  at one sm was measured to tighten onto a real bias.")
    print("  The branching criterion is a SIGN CHANGE, and the branching time diverges")
    print("  as the reciprocal of the splitting rate -- also category B, also asserted")
    print("  as an exponent, because the prefactor is a property of the detector.")
    print()
    print("  Both bands are two-sided, and both had to be: every wrong variant tried")
    print("  here is comfortably 'significant' against zero.")


if __name__ == "__main__":
    main()
