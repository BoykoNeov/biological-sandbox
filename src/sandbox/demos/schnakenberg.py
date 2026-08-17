"""Schnakenberg demo — the wavelength a pattern *chooses*, not just how fast it grows.

Run it:  ``uv run python -m sandbox.demos.schnakenberg``

Gray-Scott's demo validates ``lambda(q)``: the growth rate of a mode seeded by hand.
This one asks the question HANDOFF §5 actually asked and Phase 2 could not answer —
seed **nothing**, let a random perturbation grow, and see which wavelength appears.

Five acts:

1. **The onset, in closed form.** ``u* = a+b``, ``v* = b/(a+b)^2``, ``det J = u*^2``
   *exactly*, so ``q_c`` and the critical diffusion ratio ``d_c`` are both closed
   form. Each is printed against an independent route, not against a stored literal.
2. **The dispersion relation**, across a sign change, with the continuum ``-Dq^2``
   drawn alongside. The two are not close, and at this demo's coarse grid they
   disagree about *which mode is fastest* — which is what act 3 exploits.
3. **Selection.** Thirty-two random starts on the same grid, and the mode each one picks,
   against four competing hypotheses. This is the validated claim, and it is a
   *discrimination*: the measurement is far closer to the stencil's fastest mode than
   to the continuum's, the band centre, or either band edge.
4. **Supercriticality**, which is the precondition act 3's argument rests on: the
   amplitude has to grow continuously from zero rather than jump. It does. **Its
   log-log slope reads 0.4606 and that number is NOT the exponent** — see the panel's
   own caption, and §8 of the measurement document.
5. **Two dimensions**, labelled qualitative: it selected the predicted mode at two
   seeds, which is an observation and not a validated claim.

Figures are written to the current directory (``*.png`` is gitignored). All printed
output is ASCII-only (Windows cp1252 console).
"""

from __future__ import annotations

import math

import numpy as np

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.recorder import run_replicate
from sandbox.core.rng import spawn_rngs
from sandbox.core.selection import selection_report
from sandbox.models.schnakenberg import (
    MODEL,
    SchnakenbergParams,
    continuum_dispersion,
    critical_ratio,
    critical_wavenumber,
    dispersion,
    dominant_mode,
    fastest_mode,
    homogeneous_state,
    is_diffusionless_stable,
    n_steps,
    peak_power_fraction,
    reaction_determinant,
    reaction_jacobian,
    unstable_band,
)

_BASE = dict(a=0.05, b=1.0, du=1.0e-3, length=8.0, ndim=1, noise=1.0e-3)
# The coarse grid: 4.3 cells per wavelength, where the stencil and the continuum
# disagree by 2.27 modes about which mode grows fastest. That gap is what lets a
# nonlinear pattern discriminate between the two operators.
_COARSE = dict(_BASE, n=112, dt=0.1, t_max=300.0, initial="noise")
_REPLICATES = 32


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _params(**overrides) -> SchnakenbergParams:
    return SchnakenbergParams(**{**_COARSE, **overrides})


def _factory(d: dict) -> SchnakenbergParams:
    return SchnakenbergParams(**d)


def _continuum_fastest(params: SchnakenbergParams) -> float:
    js = np.arange(1, params.n // 2 + 1)
    best = float(js[int(np.argmax([continuum_dispersion(params, int(j)) for j in js]))])
    lo, hi = best - 1.5, best + 1.5
    for _ in range(4):
        grid = np.linspace(lo, hi, 2001)
        k = int(np.argmax([continuum_dispersion(params, float(j)) for j in grid]))
        span = (hi - lo) / 2000.0
        lo, hi = grid[k] - 2.0 * span, grid[k] + 2.0 * span
    return 0.5 * (lo + hi)


def _act1_closed_forms() -> bool:
    _rule("Act 1 - the onset is closed form, and every route is checked")
    params = _params()
    u_star, v_star = homogeneous_state(params)
    residual = np.array(
        [
            params.a - u_star + u_star**2 * v_star,
            params.b - u_star**2 * v_star,
        ]
    )
    jacobian = reaction_jacobian(u_star, v_star)
    print(f"u* = a + b            = {u_star:.12f}")
    print(f"v* = b/(a+b)^2        = {v_star:.12f}")
    print(f"|reaction at (u*,v*)| = {np.abs(residual).max():.3e}   (substitution, not algebra)")
    print(f"det J                 = {reaction_determinant(params):.12f}")
    print(f"u*^2                  = {u_star**2:.12f}   <- equal, and that is why d_c closes")
    print(
        f"trace J               = {jacobian[0, 0] + jacobian[1, 1]:+.6f}   stable without "
        f"diffusion: {is_diffusionless_stable(params)}"
    )

    d_c = critical_ratio(params)
    at_onset = _params(dv=d_c * params.du)
    qs = np.linspace(1.0, 60.0, 200001)
    rates = np.array(
        [continuum_dispersion(at_onset, q * at_onset.length / (2.0 * math.pi)) for q in qs[::100]]
    )
    argmax_q = qs[::100][int(np.argmax(rates))]
    print()
    print(f"d_c (closed form)     = {d_c:.6f}")
    print(
        f"this run's Dv/Du      = {params.dv / params.du:.6f} = "
        f"{params.dv / params.du / d_c:.3f} d_c"
    )
    print(f"q_c (closed form)     = {critical_wavenumber(at_onset):.6f}")
    print(f"argmax lambda at onset= {argmax_q:.6f}   (independent route)")

    ok = (
        np.abs(residual).max() < 1e-15
        and abs(reaction_determinant(params) - u_star**2) < 1e-15
        and abs(critical_wavenumber(at_onset) / argmax_q - 1.0) < 1e-3
    )
    print(f"\nclosed forms reproduce: {'PASS' if ok else 'FAIL'}")
    return ok


def _act2_dispersion() -> tuple[bool, list[tuple[int, float, float, float]]]:
    _rule("Act 2 - the seeded mode's growth rate, across a sign change")
    rows = []
    ok = True
    for mode_j, t_max in ((19, 40.0), (24, 40.0), (30, 40.0), (40, 40.0), (50, 20.0)):
        params = SchnakenbergParams(
            **{
                **_COARSE,
                "n": 256,
                "dt": 0.02,
                "t_max": t_max,
                "initial": "mode",
                "mode_j": mode_j,
                "eps": 1.0e-5,
            }
        )
        predicted, _, is_real = dispersion(params, mode_j)
        if not is_real:
            print(f"  j = {mode_j:>3}: REFUSED (complex pair) - the amplitude oscillates")
            continue
        (rng,) = spawn_rngs(0, 1)
        traj = run_replicate(
            MODEL, params, rng, max_steps=n_steps(params) + 10, record_every=n_steps(params)
        )
        measured = traj.final["growth_rate"]
        continuum = continuum_dispersion(params, mode_j)
        rows.append((mode_j, measured, predicted, continuum))
        relative = abs(measured / predicted - 1.0)
        ok = ok and relative < 1e-6
        print(
            f"  j = {mode_j:>3}: measured {measured:+.9f}  stencil {predicted:+.9f}  "
            f"rel {relative:.2e}   continuum {continuum:+.9f} "
            f"({abs(continuum / predicted - 1.0) * 100:6.2f}% off)"
        )
    print(
        f"\nsigns present: {sorted({int(math.copysign(1, m)) for _, m, _, _ in rows})} "
        f"(a one-sided check cannot see a sign error)"
    )
    print(f"category A anchor: {'PASS' if ok else 'FAIL'}")
    return ok, rows


def _act3_selection():
    _rule("Act 3 - which wavelength a random start selects (the validated claim)")
    params = _params()
    band = unstable_band(params)
    stencil = fastest_mode(params, continuous=True)
    continuum = _continuum_fastest(params)
    lam, _, _ = dispersion(params, int(fastest_mode(params)))
    print(
        f"grid: n = {params.n}, L = {params.length:g}, {params.n / stencil:.2f} cells per "
        f"wavelength, horizon {lam * params.t_max:.1f} e-folds"
    )
    print(
        f"unstable band: modes {band[0]}-{band[-1]} ({band.size} of them, so 'the fastest "
        f"one won' is not a statement about the only one that could)"
    )
    print(f"stencil's fastest mode   : {stencil:.4f}")
    print(f"continuum's fastest mode : {continuum:.4f}  <- 2.27 modes away, and wrong")

    report = selection_report(
        "schnakenberg",
        dict(_COARSE),
        _factory,
        predicted_mode=stencil,
        competitors={
            "continuum operator": continuum,
            "band centre": 0.5 * float(band[0] + band[-1]),
            "lowest unstable": float(band[0]),
            "highest unstable": float(band[-1]),
        },
        band=(int(band[0]), int(band[-1])),
        replicates=_REPLICATES,
        seed=0,
        z=4.0,
        max_steps=n_steps(params) + 10,
        record_every=n_steps(params),
        efolds=lam * params.t_max,
    )
    print()
    print(report)
    print()
    print("Read the gap-to-prediction line as a description, not a pass mark: it grows")
    print("from 0.61 to 2.24 SE as replicates go 8 -> 48, because a real residual of")
    print("about a third of a mode sits under a shrinking error bar. That is why this")
    print("claim is a margin and why analytic_predictions refuses the random start.")
    print()
    print("The selected mode is horizon-independent past the 20-e-fold threshold: at")
    print("22.7, 45.3 and 113.3 e-folds three seeds pick the same mode on both shipped")
    print("grids. What keeps changing out there is how much power sits in harmonics -")
    print("on the fine grid it drains away (0.53 -> 0.86 -> 0.997), on the coarse grid")
    print("it freezes (0.45 -> 0.39, and stays), because those harmonics have nowhere")
    print("to go. Panel (c) shows both, and an earlier version of this demo quoted the")
    print("fine grid's long-horizon 0.997 as if it described the shipped run.")
    return report


def _act4_supercritical() -> list[tuple[float, float, int]]:
    _rule("Act 4 - supercriticality, the precondition (and a slope that is not an exponent)")
    d_c = critical_ratio(_params())
    rows = []
    print(
        f"{'d/d_c':>7} {'d - d_c':>9} {'lambda*':>9} {'e-folds':>8} {'amplitude':>10} "
        f"{'amp/sqrt(eps)':>14} {'mode':>5}"
    )
    for ratio in (1.01, 1.02, 1.04, 1.08, 1.16, 1.32):
        dv = ratio * d_c * 1.0e-3
        probe = _params(dv=dv, t_max=300.0)
        lam, _, _ = dispersion(probe, int(fastest_mode(probe)))
        t_max = math.ceil(60.0 / lam / probe.dt) * probe.dt
        probe = _params(dv=dv, t_max=t_max)
        (rng,) = spawn_rngs(0, 1)
        traj = run_replicate(
            MODEL, probe, rng, max_steps=n_steps(probe) + 10, record_every=n_steps(probe)
        )
        eps = ratio * d_c - d_c
        amplitude = traj.final["pattern_amplitude"]
        rows.append((eps, amplitude, int(traj.final["dominant_mode"])))
        print(
            f"{ratio:>7.3f} {eps:>9.5f} {lam:>9.5f} {lam * t_max:>8.1f} {amplitude:>10.5f} "
            f"{amplitude / math.sqrt(eps):>14.5f} {int(traj.final['dominant_mode']):>5}"
        )

    eps = np.array([r[0] for r in rows])
    amp = np.array([r[1] for r in rows])
    slope = float(np.polyfit(np.log(eps), np.log(amp), 1)[0])
    print()
    print("Continuous from zero, monotone, no jump: supercritical, so the near-onset")
    print("selection argument applies here (it would not for a subcritical branch).")
    print(f"log-log slope over this range: {slope:.4f}.  This is NOT a measurement of the")
    print("exponent. The square-root law's O(eps) correction is still large here -- the")
    print("amp/sqrt(eps) column drifts by 14% -- and a Richardson extrapolation of that")
    print("column is itself still drifting. The honest claim is 'consistent with 1/2 with")
    print("corrections', and no affordable configuration sits inside the asymptotic regime.")
    return rows


def _act5_two_dimensions():
    _rule("Act 5 - two dimensions (QUALITATIVE: an observation, not a validated claim)")
    # 5.1 cells per wavelength. n=96 was tried first and rejected on looking at the
    # figure: at 3.6 cells per wavelength the picture shows the grid rather than the
    # wavelength, and a reader cannot see the claim the panel is illustrating.
    params = SchnakenbergParams(
        **{**_BASE, "n": 128, "ndim": 2, "dt": 0.04, "t_max": 300.0, "initial": "noise"}
    )
    predicted = fastest_mode(params)
    fields = []
    for seed in range(2):
        # The field is kept here rather than re-run for the figure: observables carry
        # scalars only, and re-integrating for the picture would double the phase's
        # most expensive runs to draw exactly what was already computed.
        field = _final_field(params, seed)
        peak = dominant_mode(field)
        print(
            f"  seed {seed}: radial peak mode {peak}, predicted {predicted:.0f}, "
            f"amplitude {0.5 * (field.max() - field.min()):.4f}, "
            f"{params.n / peak:.2f} cells per wavelength"
        )
        fields.append(field)
    print()
    print("Two seeds at about 25 s a run, so this is reported and asserted nowhere.")
    print("The 2-D spectrum is binned radially, because a wavelength spreads its power")
    print("around a ring: a stripe and a spot pattern of the same wavelength must report")
    print("the same wavenumber.")
    return params, fields


def _final_field(params: SchnakenbergParams, seed: int) -> np.ndarray:
    """Run one replicate keeping the final field, which observables cannot carry."""
    (rng,) = spawn_rngs(seed, 1)
    state = MODEL.initial_state(params, rng)
    for _ in range(n_steps(params)):
        state = MODEL.step(state, rng)
    return MODEL.fields(state)["u"]


def _figures(rows, report, amp_rows, two_d_params, two_d_fields) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping figures; uv sync --extra viz)")
        return

    params = _params()
    band = unstable_band(params)
    stencil = fastest_mode(params, continuous=True)
    continuum = _continuum_fastest(params)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) dispersion relation, both operators, with the measured points
    ax = axes[0, 0]
    fine = SchnakenbergParams(**{**_COARSE, "n": 256, "dt": 0.02, "initial": "mode", "mode_j": 24})
    js = np.linspace(1.0, 60.0, 400)
    discrete = np.array([dispersion(fine, float(j))[0] for j in js])
    real = np.array([dispersion(fine, float(j))[2] for j in js])
    cont = np.array([continuum_dispersion(fine, float(j)) for j in js])
    ax.plot(js, cont, "--", color="C7", lw=1.6, label="continuum $-Dq^2$ (NOT integrated)")
    ax.plot(js[real], discrete[real], "-", color="C0", lw=2.0, label="stencil (what is integrated)")
    # Drawn thick and in its own colour because it is otherwise invisible: at small j
    # diffusion barely matters, so the stencil and the continuum coincide there and this
    # segment hides under the dashed line. A legend entry for a curve the reader cannot
    # find is worse than no entry at all.
    ax.plot(
        js[~real],
        discrete[~real],
        "-",
        color="C1",
        lw=3.5,
        alpha=0.85,
        label="complex pair: Re only, NOT a measurable rate",
    )
    ax.plot(
        [j for j, _, _, _ in rows],
        [m for _, m, _, _ in rows],
        "o",
        color="C3",
        markersize=7,
        label="measured, seeded one mode",
    )
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("mode number j")
    ax.set_ylabel("growth rate")
    ax.set_title("(a) dispersion at n=256: measured points span a sign change")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) the selected modes, against every hypothesis
    ax = axes[0, 1]
    modes = np.array(report.selected_modes)
    edges = np.arange(modes.min() - 0.5, modes.max() + 1.5, 1.0)
    ax.hist(modes, bins=edges, color="C0", alpha=0.75, label=f"{modes.size} random starts")
    ax.axvline(
        report.predicted_mode, color="C3", lw=2.5, label=f"stencil {stencil:.2f} (predicted)"
    )
    ax.axvline(continuum, color="C7", ls="--", lw=2, label=f"continuum {continuum:.2f}")
    ax.axvline(
        0.5 * float(band[0] + band[-1]),
        color="C4",
        ls=":",
        lw=2,
        label=f"band centre {0.5 * (band[0] + band[-1]):.1f}",
    )
    ax.axvline(report.measured_mean, color="k", lw=1.2, label=f"mean {report.measured_mean:.3f}")
    # Widened so the continuum's prediction sits inside the axes rather than flush
    # against the spine, where a rejected claim reads as decoration.
    ax.set_xlim(min(continuum, modes.min()) - 0.8, modes.max() + 1.2)
    ax.set_xlabel("selected mode")
    ax.set_ylabel("runs")
    ax.set_title(
        "(b) selection excludes the alternatives, and is NOT exact\n"
        f"margins: continuum {report.checks[0].margin_se:.1f} SE, "
        f"centre {report.checks[1].margin_se:.1f} SE; "
        f"prediction itself {report.gap_in_se:.1f} SE off"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) the same run on both grids, because the trade-off IS the finding
    ax = axes[1, 0]
    fine_noise = SchnakenbergParams(**{**_COARSE, "n": 256, "dt": 0.02})
    coarse_field = _final_field(params, 0)
    fine_field = _final_field(fine_noise, 0)
    ax.plot(
        np.arange(fine_noise.n) * (fine_noise.length / fine_noise.n),
        fine_field,
        "-",
        color="C7",
        lw=1.2,
        label=(
            f"n=256, 10.6 cells/wavelength: mode {dominant_mode(fine_field)}, "
            f"{peak_power_fraction(fine_field) * 100:.0f}% of power in it"
        ),
    )
    ax.plot(
        np.arange(params.n) * (params.length / params.n),
        coarse_field,
        "-",
        color="C0",
        lw=1.4,
        label=(
            f"n=112, 4.3 cells/wavelength: mode {dominant_mode(coarse_field)}, "
            f"{peak_power_fraction(coarse_field) * 100:.0f}% of power in it"
        ),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(
        "(c) the two grids select DIFFERENT modes - that gap is what (b) measures.\n"
        "Both keep ~half the power in harmonics: at 1.2 d_c this is not weakly nonlinear"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (d) amplitude against the distance to onset
    ax = axes[1, 1]
    eps = np.array([r[0] for r in amp_rows])
    amp = np.array([r[1] for r in amp_rows])
    ax.plot(np.sqrt(eps), amp, "o-", color="C0", label="measured amplitude")
    reference = amp[0] / math.sqrt(eps[0])
    ax.plot(
        np.sqrt(eps),
        reference * np.sqrt(eps),
        "--",
        color="C7",
        label="pure $\\sqrt{d-d_c}$ through the first point",
    )
    ax.set_xlabel("$\\sqrt{d - d_c}$")
    ax.set_ylabel("pattern amplitude")
    slope = float(np.polyfit(np.log(eps), np.log(amp), 1)[0])
    ax.set_title(
        "(d) supercritical: continuous from zero, no jump\n"
        f"log-log slope reads {slope:.4f} and is NOT the exponent (O(eps) correction)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Schnakenberg: the wavelength a pattern chooses. "
        "(a)-(b) validated, (d) precondition, (c) illustrative.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("schnak_selection.png", dpi=120)
    plt.close(fig)
    print("wrote schnak_selection.png")

    # 2-D, on its own and labelled for what it is
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    extent = (0.0, two_d_params.length, 0.0, two_d_params.length)
    for ax, seed, field2d in zip(axes, (0, 1), two_d_fields, strict=True):
        # 'nearest' on purpose: at 5 cells per wavelength a smoothing interpolation
        # would draw a resolution this run does not have.
        image = ax.imshow(
            field2d, cmap="viridis", origin="lower", extent=extent, interpolation="nearest"
        )
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax.set_title(
            f"seed {seed}: radial peak mode {dominant_mode(field2d)}, "
            f"{two_d_params.n / dominant_mode(field2d):.1f} cells/wavelength"
        )
        ax.set_xlabel("x")
    fig.suptitle(
        f"QUALITATIVE - 2-D Schnakenberg, predicted fastest mode "
        f"{fastest_mode(two_d_params):.0f}. Two seeds; asserted nowhere.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("schnak_2d.png", dpi=120)
    plt.close(fig)
    print("wrote schnak_2d.png")


def main() -> int:
    closed_ok = _act1_closed_forms()
    anchor_ok, rows = _act2_dispersion()
    report = _act3_selection()
    amp_rows = _act4_supercritical()
    two_d_params, two_d_fields = _act5_two_dimensions()
    _figures(rows, report, amp_rows, two_d_params, two_d_fields)

    _rule("Summary")
    print(f"onset closed forms (independent routes)  : {'PASS' if closed_ok else 'FAIL'}")
    print(f"category A (seeded rate, sign change)    : {'PASS' if anchor_ok else 'FAIL'}")
    print(f"selection (discrimination, z = 4)        : {'PASS' if report.passed else 'FAIL'}")
    print("supercriticality                         : precondition, checked not asserted")
    print("2-D selection                            : QUALITATIVE, asserted nowhere")
    return 0 if (closed_ok and anchor_ok and report.passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
