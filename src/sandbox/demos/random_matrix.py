"""May / Allesina-Tang demo — the matrix law, and the composition Phase 3 refuses.

Run it:  ``uv run python -m sandbox.demos.random_matrix``

HANDOFF S6 promises that "May's result (random community matrices: stability
decreases with complexity) becomes a real, checkable prediction about your own
simulated webs." **As literally written that check cannot be run**, and this demo
is mostly about showing why, honestly, with the measurements that establish it —
the Phase-3 analogue of Phase 2's Gray-Scott reframe.

Five acts:

1. **The ensemble and the elliptic law.** The spectrum of ``B`` fills an ellipse
   with semi-axes ``R(1 + rho)`` and ``R(1 - rho)``, ``R = sigma sqrt(S C)``. The
   enclosed-fraction check is *asserted* in ``tests/test_random_matrix.py``; here
   it is drawn, together with ``E[max Re]`` tracking ``R(1 + rho)``.
2. **The finite-``S`` bias**, the reason the fraction check has two tracks. The
   exponents are **reported here and not asserted**: they are probe-dependent
   (``-0.88 / -1.03 / -0.62``), nothing predicts them, and resolving the circular
   ones costs ~28 s each.
3. **``P(stable)`` and its sharpening with ``S``** — reported, never asserted. The
   transition is real and visible, but the measurement is grid-sensitive (at
   ``S = 400`` a 0.05-wide ``u`` grid straddling a ``0 -> 0.68 -> 0`` jump returned
   a meaningless width) and costs 45 s at ``S = 400`` alone.
4. **The refusal, with its evidence.** Feasibility of a random gLV interior
   equilibrium collapses to zero exactly where May's asymptotic criterion starts to
   mean anything, so the ensemble the promise needs is *empty*.
5. **And conditioning on feasibility moves the spectrum**, so the non-empty small-
   ``S`` corner does not rescue it either. The community matrix of a *feasible*
   random gLV is simply not distributed like a May matrix.

Acts 4 and 5 are why ``models/ecosystem/`` — the speculative quarantine — is still
empty at the end of Phase 3, and why that is the correct outcome rather than an
omission.

Figures are written to the current directory (``*.png`` is gitignored). All printed
output is ASCII-only (Windows cp1252 console).
"""

from __future__ import annotations

import time

import numpy as np

from sandbox.core.random_matrix import (
    bias_scaling_report,
    binomial_se,
    bulk_radius,
    disc_fraction,
    draw_community_matrix,
    ellipse_fraction,
    ensemble_eigenvalues,
)

RULE = "=" * 78


def act1_the_elliptic_law() -> dict[float, np.ndarray]:
    """Fractions and ``E[max Re]`` across the correlation range, plus a spectrum figure."""
    print(RULE)
    print("ACT 1 - the elliptic law: an ellipse of semi-axes R(1+rho), R(1-rho)")
    print(RULE)
    n_species, n_draws, probe = 400, 25, 0.5
    radius = bulk_radius(n_species)
    print(f"S = {n_species}, C = 1, sigma = 1  ->  R = sigma sqrt(S C) = {radius:g}")
    print(f"probe = {probe} of the mapped unit disc, so the predicted fraction is {probe**2}\n")
    print(f"{'rho':>6} {'fraction':>10} {'pred':>7} {'z':>7} {'E[max Re]':>11} {'R(1+rho)':>10}")

    spectra: dict[float, np.ndarray] = {}
    for correlation in (-0.8, -0.4, 0.0, 0.4, 0.8):
        rng = np.random.default_rng(1)
        eigs_per_draw = [
            np.linalg.eigvals(
                draw_community_matrix(rng, n_species=n_species, correlation=correlation)
            )
            for _ in range(n_draws)
        ]
        eigs = np.concatenate(eigs_per_draw)
        spectra[correlation] = eigs
        fraction = ellipse_fraction(eigs, radius=radius, correlation=correlation, probe=probe)
        se = binomial_se(probe**2, eigs.size)
        max_re = float(np.mean([e.real.max() for e in eigs_per_draw]))
        print(
            f"{correlation:>6.1f} {fraction:>10.6f} {probe**2:>7.2f} "
            f"{(fraction - probe**2) / se:>7.2f} {max_re:>11.4f} "
            f"{radius * (1 + correlation):>10.1f}"
        )

    print(
        "\nThe ecological content: rho < 0 is predator-prey structure and makes the web\n"
        "MORE stable (the ellipse is squashed along the real axis); rho > 0 is\n"
        "competition/mutualism and makes it less. May's criterion is the rho = 0 case."
    )
    _figure_spectra(spectra, radius, probe)
    return spectra


def act2_the_finite_size_bias() -> None:
    """The bias that forces two tracks — reported, with the exponents left unasserted."""
    print()
    print(RULE)
    print("ACT 2 - the finite-S bias, and why the fraction check needs two tracks")
    print(RULE)
    print(
        "The circular law is exact only as S -> infinity. At finite S there is a bias,\n"
        "and piling on eigenvalues shrinks the SE BELOW it -- so a CORRECT\n"
        "implementation starts failing. Precision at fixed S is therefore bounded.\n"
    )
    sizes = (25, 50, 100, 200)
    draws = (200, 200, 120, 80)
    print(f"{'S':>5} {'n eigs':>8} {'bias(0.2)':>11} {'bias(0.5)':>11} {'bias(0.9)':>11}")
    columns: dict[float, list[float]] = {0.2: [], 0.5: [], 0.9: []}
    counts: list[int] = []
    for n_species, n_draws in zip(sizes, draws, strict=True):
        rng = np.random.default_rng(10)
        eigs = ensemble_eigenvalues(rng, n_draws=n_draws, n_species=n_species)
        counts.append(eigs.size)
        row = []
        for probe in (0.2, 0.5, 0.9):
            bias = disc_fraction(eigs, radius=bulk_radius(n_species), probe=probe) - probe**2
            columns[probe].append(bias)
            row.append(f"{bias:>11.3e}")
        print(f"{n_species:>5} {eigs.size:>8} " + " ".join(row))

    print(
        f"\n{'probe':>6} {'exponent (this run)':>21} {'resolved':>9} {'per-point z':>26}"
        f"  {'reference (200k eigenvalues)':>28}"
    )
    reference = {0.2: "-0.8814 +/- 0.0250", 0.5: "-1.0274 +/- 0.0231", 0.9: "-0.6189 +/- 0.0673"}
    for probe, biases in columns.items():
        report = bias_scaling_report(sizes, biases, counts, predicted=probe**2)
        z_text = " ".join(f"{abs(z):5.2f}" for z in report.bias_z)
        print(
            f"{probe:>6} {report.exponent:>10.4f} +/- {report.exponent_se:<7.4f}"
            f"{str(report.resolved):>9} {z_text:>26}  {reference[probe]:>28}"
        )
    print(
        "\nREAD THE 'resolved' COLUMN BEFORE THE 'exponent' ONE. This act's own table is\n"
        "DELIBERATELY UNDER-RESOLVED: at S = 200 the bias sits at roughly 1-2 binomial\n"
        "SE, so those points cannot resolve the very quantity they are being fitted to,\n"
        "and the exponents printed above are not measurements. Resolving them properly\n"
        "costs ~28 s per circular probe, which is why the reference column -- measured\n"
        "at 200 000 eigenvalues, every point above z = 3 -- is the real result.\n"
        "\n"
        "This is not a shortcut, it is the demonstration. The plan's recorded bias law\n"
        "(0.6/S, slope -0.9279) was produced exactly this way: a fit whose two largest-S\n"
        "points sat below their own SE. Re-measured until every point resolved, the same\n"
        "probe gives 0.70/S and slope -1.0086. A fit will happily consume points that\n"
        "measure nothing and hand back a confident-looking slope, so bias_scaling_report\n"
        "carries a 'resolved' guard and the suite refuses a fit that trips it.\n"
        "\n"
        "The reference exponents themselves are REPORTED, NOT ASSERTED: they differ\n"
        "between probes by many sigma and no theory here predicts any of them. The\n"
        "tempting mechanism -- an O(1/S) bulk correction plus a slower one inside a\n"
        "Ginibre edge layer of width ~S^-1/2 -- predicts probe = 0.2, deep in the bulk,\n"
        "at -1. It measures -0.88, SHALLOWER than probe = 0.5, so the ordering is not\n"
        "monotone and the mechanism is wrong. The suite asserts only that the bias\n"
        "decays faster than S^-0.15, a rate no constant-offset bug reaches."
    )


def act3_the_stability_transition() -> None:
    """``P(stable)`` sharpening with ``S`` — reported, never asserted."""
    print()
    print(RULE)
    print("ACT 3 - P(stable) vs u = sigma sqrt(SC)/d, and its sharpening with S")
    print(RULE)
    print("May's criterion: the web is stable when u < 1. Sharp only as S -> infinity.\n")
    u_grid = np.linspace(0.6, 1.4, 9)
    print(f"{'u':>6} " + " ".join(f"{'S=' + str(s):>8}" for s in (25, 50, 100)))
    table: dict[int, list[float]] = {}
    for n_species in (25, 50, 100):
        rng = np.random.default_rng(20)
        row = []
        for u in u_grid:
            sigma = u / np.sqrt(n_species)  # C = 1, d = 1  ->  u = sigma sqrt(S)
            stable = 0
            n_draws = 60
            for _ in range(n_draws):
                matrix = draw_community_matrix(
                    rng, n_species=n_species, sigma=sigma, self_regulation=1.0
                )
                stable += int(np.linalg.eigvals(matrix).real.max() < 0.0)
            row.append(stable / n_draws)
        table[n_species] = row
    for i, u in enumerate(u_grid):
        print(f"{u:>6.2f} " + " ".join(f"{table[s][i]:>8.2f}" for s in (25, 50, 100)))
    print(
        "\nREPORTED, NOT ASSERTED. The transition is real and visibly sharpens, but the\n"
        "width is grid-sensitive -- at S = 400 a 0.05-wide u grid straddling a\n"
        "0 -> 0.68 -> 0 jump returned a meaningless 0.229 -- and costs 45 s at S = 400\n"
        "alone. A quantity whose measured value depends on the grid you happened to\n"
        "choose is not something to put a tightening tolerance on."
    )


def act4_the_ensemble_is_empty() -> None:
    """Feasibility collapses exactly where May's asymptotics start to bite."""
    print()
    print(RULE)
    print("ACT 4 - the refusal: the random-gLV ensemble is EMPTY at the S that matters")
    print(RULE)
    print(
        "HANDOFF asks for May's criterion as a prediction about simulated webs. That\n"
        "needs a random gLV with a feasible interior equilibrium x* = -A^-1 r > 0.\n"
        "Fraction of draws that have one (A = -I + B, off-diagonals nonzero w.p. 0.5):\n"
    )
    sigmas = (0.1, 0.25, 0.5)
    print(f"{'S':>5} " + " ".join(f"{'sigma=' + str(s):>12}" for s in sigmas))
    for n_species in (5, 10, 20, 40):
        row = []
        for sigma in sigmas:
            rng = np.random.default_rng(30)
            n_draws = 200
            feasible = 0
            for _ in range(n_draws):
                interactions = draw_community_matrix(
                    rng, n_species=n_species, connectance=0.5, sigma=sigma, self_regulation=1.0
                )
                growth = np.ones(n_species)
                try:
                    equilibrium = np.linalg.solve(interactions, -growth)
                except np.linalg.LinAlgError:  # pragma: no cover - singular draw
                    continue
                feasible += int(np.all(equilibrium > 0.0))
            row.append(f"{feasible / n_draws:>12.3f}")
        print(f"{n_species:>5} " + " ".join(row))
    print(
        "\nMay's criterion is asymptotic in S. By S = 40 the ensemble it would be tested\n"
        "on is empty at every sigma large enough for the criterion to discriminate.\n"
        "There is no sample to measure, so there is no check -- not a small one, none."
    )


def act5_feasibility_moves_the_spectrum() -> None:
    """The small-``S`` corner does not rescue the promise either."""
    print()
    print(RULE)
    print("ACT 5 - and conditioning on feasibility MOVES the spectrum")
    print(RULE)
    n_species, sigma, target = 20, 0.25, 150
    rng = np.random.default_rng(40)
    raw_max, community_max, spans, draws = [], [], [], 0
    while len(raw_max) < target and draws < 20_000:
        draws += 1
        interactions = draw_community_matrix(
            rng, n_species=n_species, connectance=0.5, sigma=sigma, self_regulation=1.0
        )
        try:
            equilibrium = np.linalg.solve(interactions, -np.ones(n_species))
        except np.linalg.LinAlgError:  # pragma: no cover - singular draw
            continue
        if not np.all(equilibrium > 0.0):
            continue
        raw_max.append(float(np.linalg.eigvals(interactions).real.max()))
        community = equilibrium[:, None] * interactions
        community_max.append(float(np.linalg.eigvals(community).real.max()))
        spans.append(float(equilibrium.max() / equilibrium.min()))

    print(f"S = {n_species}, sigma = {sigma}: {draws} draws to collect {len(raw_max)} feasible\n")
    print(f"  max Re eig(A)            = {np.mean(raw_max):+.4f}  (sd {np.std(raw_max):.4f})")
    print(
        f"  max Re eig(diag(x*) A)   = {np.mean(community_max):+.4f}  "
        f"(sd {np.std(community_max):.4f})"
    )
    # Median, not mean: max/min is heavy-tailed (a near-zero component sends one
    # draw's ratio to thousands), so its mean is an outlier report, not a summary.
    # The slice recorded 67 as a mean and this run's mean is 546 -- same ensemble,
    # different draw count. The median is stable across both.
    print(f"  x* components span a factor of {np.median(spans):.0f} (median) within a single web")
    print(
        "\nThe community matrix is diag(x*) A, not A -- and x* is wildly heterogeneous,\n"
        "so the row scaling is far from uniform. Conditioning on feasibility is a\n"
        "strong, spectrum-dependent selection: the two differ by more than a standard\n"
        "deviation. So Phase 3 validates the MATRIX LAW directly (acts 1-2) and claims\n"
        "nothing about feasibility-conditioned gLV. Asserting the composition would be\n"
        "exactly the Gray-Scott error of Phase 2: comparing a measurement against a\n"
        "prediction that makes no claim about it."
    )


def _figure_spectra(spectra: dict[float, np.ndarray], radius: float, probe: float) -> None:
    """Scatter the spectrum against its predicted ellipse, at three correlations.

    The claim the figure carries is the *boundary*: the drawn ellipse is
    ``R(1 +- rho)`` from theory, not a fit to the points. The inner dashed ellipse
    is the ``probe`` contour whose enclosed fraction the suite asserts, so the
    reader can see the statistic being measured rather than take it on trust.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the spectrum figure)")
        return

    shown = (-0.8, 0.0, 0.8)
    fig, axes = plt.subplots(1, len(shown), figsize=(13, 4.4), constrained_layout=True)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    for ax, correlation in zip(axes, shown, strict=True):
        eigs = spectra[correlation][:6000]
        semi_major, semi_minor = radius * (1 + correlation), radius * (1 - correlation)
        ax.scatter(eigs.real, eigs.imag, s=0.7, alpha=0.25, color="#2b6cb0", linewidths=0)
        ax.plot(
            semi_major * np.cos(theta),
            semi_minor * np.sin(theta),
            color="#c53030",
            lw=1.6,
            label="R(1+rho), R(1-rho)",
        )
        ax.plot(
            probe * semi_major * np.cos(theta),
            probe * semi_minor * np.sin(theta),
            color="#c53030",
            lw=1.2,
            ls="--",
            label=f"probe = {probe} (encloses {probe**2:g})",
        )
        ax.set_title(f"rho = {correlation:+.1f}")
        ax.set_xlabel("Re")
        ax.set_aspect("equal")
        ax.set_xlim(-1.15 * radius * 1.8, 1.15 * radius * 1.8)
        ax.set_ylim(-1.15 * radius * 1.8, 1.15 * radius * 1.8)
    axes[0].set_ylabel("Im")
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Allesina-Tang elliptic law: the boundary is predicted, not fitted "
        f"(S = 400, C = 1, sigma = 1, R = {radius:g})"
    )
    fig.savefig("random_matrix_elliptic_law.png", dpi=140)
    plt.close(fig)
    print("\nwrote random_matrix_elliptic_law.png")


def main() -> None:
    started = time.perf_counter()
    act1_the_elliptic_law()
    act2_the_finite_size_bias()
    act3_the_stability_transition()
    act4_the_ensemble_is_empty()
    act5_feasibility_moves_the_spectrum()
    print()
    print(RULE)
    print(f"done in {time.perf_counter() - started:.1f} s")
    print(
        "Asserted in the suite: the elliptic/circular enclosed fraction at a derived,\n"
        "bias-negligible configuration, and that the finite-S bias decays. Reported\n"
        "only: the exponents, P(stable), and everything in acts 4 and 5."
    )
    print(RULE)


if __name__ == "__main__":
    main()
