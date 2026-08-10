"""Daisyworld demo — homeostasis as a closed form, and everything around it.

Run it:  ``uv run python -m sandbox.demos.daisyworld``

Five acts, in the order the argument runs:

1. **The reduction.** ``beta(T_w) = beta(T_b)`` collapses the interior
   equilibrium to a cubic in ``delta`` alone. Printed against a bisection, with
   the quartic identity's residual, so the algebra is visible rather than
   asserted.
2. **The regulation claim, category A.** ``T_w*`` and ``T_b*`` are the same
   number at every luminosity; the split between white and black is what moves.
3. **Overcompensation, category A for the sign and category C for the size.**
   ``dT_e/dL`` is *negative* across the band while a bare planet's is strongly
   positive. The magnitudes are printed and labelled, never bounded.
4. **What the model refuses**, and why each refusal is a refusal rather than a
   number: outside the band, without albedo contrast, and when growth cannot
   outpace death.
5. **The invasion criterion (category A), and hysteresis (category C).** A dead
   planet can only be started by a species that can invade it, which is a
   *different* luminosity range from the regulating band and is equally
   closed-form. Where the two disagree — a planet nothing can invade but on
   which the interior state exists — is where ramping the luminosity up and back
   down genuinely gives different planets at the same ``L``. The ramp itself is
   category C and is not asserted anywhere; the window that tells you which rows
   of it to believe is asserted.

All printed output is ASCII-only (Windows cp1252 console). Figures are written to
the current directory (``*.png`` is gitignored).
"""

from __future__ import annotations

import textwrap

import numpy as np

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.ode import rk4_step
from sandbox.models.daisyworld import (
    DaisyworldParams,
    bare_fraction,
    daisyworld_rhs,
    delta_offset,
    equilibrium_albedo,
    growth_rate,
    interior_equilibrium,
    interior_temperatures,
    invasion_luminosities,
    regulating_band,
    slowest_rate,
    temperatures,
)

BARE_START = (0.01, 0.01)
#: Propagule floor for the luminosity ramps -- see act 5 for why it exists.
SEED = 1e-6


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _relax(params: DaisyworldParams, y0, t_max: float, dt: float = 0.01) -> np.ndarray:
    rhs = daisyworld_rhs(params)
    y = np.asarray(y0, dtype=float)
    for _ in range(round(t_max / dt)):
        y = rk4_step(rhs, y, dt)
    return y


def _wrapped(message: str, indent: str = "    ") -> None:
    """Print a refusal message in full, wrapped -- never truncated.

    Splitting on "." would cut "L = 0.60" in half, which is how the first draft
    of this act printed `no interior equilibrium at L = 0`. The messages carry the
    band endpoints and the offending components, so they are worth reading whole.
    """
    for line in textwrap.wrap(f"ValueError: {message}", width=74 - len(indent)):
        print(f"{indent}{line}")


def planetary_temperature(params: DaisyworldParams, y) -> float:
    return float(temperatures(float(y[0]), float(y[1]), params)[1])


def bare_planet_temperature(params: DaisyworldParams) -> float:
    """``T_e`` with no daisies at all — albedo is the bare ground's, always."""
    flux = params.solar_flux * params.luminosity * (1.0 - params.albedo_ground)
    return float((flux / params.stefan_boltzmann) ** 0.25)


# ---------------------------------------------------------------------------
# Act 1 — the reduction that removes the root-find
# ---------------------------------------------------------------------------


def act1_the_reduction() -> None:
    _rule("1. The interior equilibrium is closed-form -- no root-find")
    p = DaisyworldParams()
    delta = delta_offset(p)
    t_w, t_b = interior_temperatures(p)
    contrast = p.q * (p.albedo_white - p.albedo_black)

    print("  At an interior equilibrium both species satisfy  x beta(T_i) = gamma,")
    print("  so beta(T_w) = beta(T_b).  beta is a downward parabola about T_opt, so")
    print("  T_w and T_b are its two roots, T_opt -+ delta.  The local temperature")
    print("  law then gives  T_b^4 - T_w^4 = q (A_w - A_b): one equation in delta.")
    print()

    def residual(d: float) -> float:
        return (p.t_opt + d) ** 4 - (p.t_opt - d) ** 4 - contrast

    lo, hi = 0.0, 17.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    bisected = 0.5 * (lo + hi)

    print(f"  delta (Cardano)             {delta:.15f} K")
    print(f"  delta (200x bisection)      {bisected:.15f} K")
    print(f"  agreement                   {abs(delta - bisected) / bisected:.3e} relative")
    print(f"  quartic residual            {abs(residual(delta)) / contrast:.3e} relative")
    print("    (Cardano subtracts two cube roots of +5.19e6 and -4.75e6, i.e. +173.12")
    print("     and -168.13, to land on 4.99 -- about 1.5 digits go in the cancellation.)")
    print()
    print(f"  T_w*                        {t_w:.12f} K")
    print(f"  T_b*                        {t_b:.12f} K")
    print(f"  beta(T_w*) = beta(T_b*)     {float(growth_rate(t_w, p)):.12f}")
    print(f"  bare fraction x*            {bare_fraction(p):.12f}")
    print(f"  daisy cover 1 - x*          {1.0 - bare_fraction(p):.12f}")
    print("  -- not one of those five contains L.")


# ---------------------------------------------------------------------------
# Act 2 — the regulation claim
# ---------------------------------------------------------------------------


def act2_regulation() -> list[tuple[float, float, float, float, float]]:
    _rule("2. Homeostasis: the daisy temperatures do not move with luminosity")
    low, high = regulating_band(DaisyworldParams())
    print(f"  regulating band: L in [{low:.10f}, {high:.10f}], width {high - low:.6f}")
    print("  Outside it one component of the cover goes non-positive and the model")
    print("  RAISES rather than returning the interior point (act 4).")
    print()
    print(f"  {'L':>6} {'a_w*':>13} {'a_b*':>13} {'A*':>10} {'T_w':>16} {'T_b':>16}")
    rows = []
    for luminosity in (0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35):
        p = DaisyworldParams(luminosity=luminosity)
        a_w, a_b = interior_equilibrium(p)
        _, t_e, t_w, t_b = temperatures(a_w, a_b, p)
        # t_w and t_b are carried per-L, COMPUTED at that L's cover, because the
        # figure plots them. Plotting a single value repeated would draw the
        # flatness the panel claims to show -- see _plot.
        rows.append((luminosity, float(a_w), float(a_b), t_e, t_w, t_b))
        print(
            f"  {luminosity:>6.2f} {a_w:>13.9f} {a_b:>13.9f} "
            f"{equilibrium_albedo(p):>10.6f} {t_w:>16.12f} {t_b:>16.12f}"
        )
    print("  -- the last two columns are constant to the printed digit at every L,")
    print("     while a_w* runs 0.024 -> 0.668.  L sets the SPLIT, not the temperature.")
    print()

    print("  The same thing measured on SIMULATED endpoints, from one common bare")
    print("  start (0.01, 0.01) integrated to t = 100 at every L:")
    simulated = []
    for luminosity in (0.9, 0.95, 1.0, 1.05, 1.1):
        p = DaisyworldParams(luminosity=luminosity)
        y = _relax(p, BARE_START, 100.0)
        _, t_e, t_w, _ = temperatures(float(y[0]), float(y[1]), p)
        simulated.append((luminosity, t_w, t_e))
        print(f"    L={luminosity:.2f}   T_w = {t_w:.12f}   T_e = {t_e:.9f}")
    spread_w = max(s[1] for s in simulated) - min(s[1] for s in simulated)
    spread_e = max(s[2] for s in simulated) - min(s[2] for s in simulated)
    print(
        f"  spread over L:  T_w {spread_w:.3e} K    T_e {spread_e:.6f} K"
        f"    ratio {spread_e / spread_w:.2e}"
    )
    print("  The residual in T_w is leftover TRANSIENT, not discretization error --")
    print("  |T_w(dt) - T_w(dt/2)| is exactly 0.0 here.  The test bounds it with a")
    print("  two-horizon decay law instead, which predicts it to a ratio of 1.000.")
    return rows


# ---------------------------------------------------------------------------
# Act 3 — overcompensation
# ---------------------------------------------------------------------------


def act3_overcompensation() -> list[tuple[float, float, float, float, float]]:
    _rule("3. The planet does not merely flatten -- it OVERCOMPENSATES")
    print("  dT_e/dL is NEGATIVE with daisies and strongly positive without them.")
    print("  CATEGORY C for the magnitudes: they are reported, never bounded by a")
    print("  test, because a recorded number travels only with its estimator.  Only")
    print("  the SIGN is asserted in tests/test_daisyworld.py.")
    print()
    print(
        f"  {'L':>6} {'T_e daisy':>12} {'T_e bare':>11} {'dTe/dL daisy':>14} "
        f"{'dTe/dL bare':>12} {'ratio':>8}"
    )
    rows = []
    step = 1e-4
    for luminosity in (0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.35):
        p = DaisyworldParams(luminosity=luminosity)
        neighbours = []
        for offset in (-step, step):
            q = DaisyworldParams(luminosity=luminosity + offset)
            albedo = equilibrium_albedo(q)
            neighbours.append(
                (q.solar_flux * q.luminosity * (1.0 - albedo) / q.stefan_boltzmann) ** 0.25
            )
        slope_daisy = (neighbours[1] - neighbours[0]) / (2 * step)
        t_e = float(
            (p.solar_flux * luminosity * (1.0 - equilibrium_albedo(p)) / p.stefan_boltzmann) ** 0.25
        )
        t_bare = bare_planet_temperature(p)
        slope_bare = t_bare / (4.0 * luminosity)
        rows.append((luminosity, t_e, t_bare, slope_daisy, slope_bare))
        print(
            f"  {luminosity:>6.2f} {t_e:>12.6f} {t_bare:>11.6f} {slope_daisy:>14.6f} "
            f"{slope_bare:>12.6f} {slope_daisy / slope_bare:>8.4f}"
        )
    print()
    print(
        f"  Across the whole band a bare planet warms by {rows[-1][2] - rows[0][2]:.2f} K while the"
    )
    print(f"  daisy planet COOLS by {rows[0][1] - rows[-1][1]:.2f} K.")
    return rows


# ---------------------------------------------------------------------------
# Act 4 — what the model refuses
# ---------------------------------------------------------------------------


def act4_refusals() -> None:
    _rule("4. What the model refuses to predict, and why")
    cases = [
        ("L = 0.60, below the band", DaisyworldParams(luminosity=0.60)),
        ("L = 1.50, above the band", DaisyworldParams(luminosity=1.50)),
        ("q = 4.0e10, daisy temperatures outside the growth parabola", DaisyworldParams(q=4.0e10)),
        ("gamma = 0.95, death outpaces the best achievable growth", DaisyworldParams(gamma=0.95)),
    ]
    for label, params in cases:
        try:
            interior_equilibrium(params)
        except ValueError as exc:
            print(f"  {label}:")
            _wrapped(str(exc))
        else:
            print(f"  {label}: NO REFUSAL -- unexpected")
    print()
    print("  and the params themselves refuse an inverted albedo contrast:")
    try:
        DaisyworldParams(albedo_white=0.25, albedo_black=0.75)
    except ValueError as exc:
        _wrapped(str(exc))
    print()
    print("  Note what is NOT a refusal: reachability.  Inside the band the interior")
    print("  state is stable but not globally attracting, and that is physics, not a")
    print("  defect -- see act 5.")


# ---------------------------------------------------------------------------
# Act 5 — category C: hysteresis, dieback, invasion
# ---------------------------------------------------------------------------


def act5_hysteresis() -> tuple[list, list]:
    _rule("5. CATEGORY C: hysteresis, dieback, and who can seed a bare planet")
    print("  The ramp table below is category C -- reported, never asserted.  The")
    print("  invasion window above it is NOT: it is closed-form and it is what makes")
    print("  the ramp readable.")
    print()
    print("  A bare planet has albedo A_g whatever the luminosity, so a rare species")
    print("  grows there iff beta(T_i) > gamma, i.e. T_i within T_opt +- sqrt((1-gamma)/k).")
    print("  T_i^4 is LINEAR in L on a bare planet, so that inverts directly:")
    p0 = DaisyworldParams()
    windows = invasion_luminosities(p0)
    band_lo, band_hi = regulating_band(p0)
    for name, (lo, hi) in windows.items():
        print(f"    {name:>5} invades a bare planet for  L in [{lo:.10f}, {hi:.10f}]")
    invade_hi = max(hi for _, hi in windows.values())
    invade_lo = min(lo for lo, _ in windows.values())
    print(f"    regulating band (interior exists)  L in [{band_lo:.10f}, {band_hi:.10f}]")
    print()
    print("  GENUINE bistability needs BOTH: an interior state that exists and a dead")
    print("  planet that nothing can invade.")
    print(
        f"    hot end:  L in ({invade_hi:.10f}, {band_hi:.10f})   width {band_hi - invade_hi:+.6f}"
    )
    print(
        f"    cold end: L in ({band_lo:.10f}, {invade_lo:.10f})   "
        f"width {invade_lo - band_lo:+.6f}  -- EMPTY"
    )
    print("  So the bistability is one-sided: invadability reaches below the band at")
    print("  the cold end, but stops short of it at the hot end.")
    print()
    print(
        f"  {'L':>6} {'T_w bare':>11} {'T_b bare':>11} {'beta(T_w)':>10} "
        f"{'beta(T_b)':>10} {'who seeds':>12}"
    )
    for luminosity in (0.70, 0.75, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30):
        p = DaisyworldParams(luminosity=luminosity)
        _, _, t_w0, t_b0 = temperatures(0.0, 0.0, p)
        beta_w, beta_b = float(growth_rate(t_w0, p)), float(growth_rate(t_b0, p))
        who = ", ".join(
            name for name, value in (("white", beta_w), ("black", beta_b)) if value > p.gamma
        )
        print(
            f"  {luminosity:>6.2f} {t_w0:>11.5f} {t_b0:>11.5f} {beta_w:>10.6f} "
            f"{beta_b:>10.6f} {who or 'nobody':>12}"
        )
    print()

    print("  Luminosity ramped up and back down, carrying the state forward, with a")
    print(f"  re-seed floor of {SEED:g} at every point.  The floor is not cosmetic:")
    print("  without it the FIRST DRAFT OF THIS PANEL WAS WRONG.  Extinction in this")
    print("  ODE is asymptotic -- a_i never reaches zero -- so an unfloored ramp left")
    print("  the state at a_b = 5.4e-144, still growing exponentially, and printed it")
    print("  as a dead planet.  Given t = 5000 instead of 1000 the same point recovers")
    print("  to a_b = 0.662.  That is a relaxation-time artefact wearing the costume")
    print("  of bistability, and it is the sort of thing only looking at the numbers")
    print("  catches.  With the floor, 'dead' means 'cannot grow from a seed'.")
    grid = np.round(np.arange(0.60, 1.81, 0.05), 2)
    ramps: dict[str, list] = {}
    for direction in ("up", "down"):
        order = grid if direction == "up" else grid[::-1]
        y = np.array([SEED, SEED], dtype=float)
        out = []
        for luminosity in order:
            p = DaisyworldParams(luminosity=float(luminosity))
            y = np.maximum(np.clip(_relax(p, y, 400.0, 0.05), 0.0, 1.0), SEED)
            out.append((float(luminosity), float(y[0]), float(y[1]), planetary_temperature(p, y)))
        ramps[direction] = out

    down_by_l = {row[0]: row for row in ramps["down"]}
    print(
        f"  {'L':>6} {'up a_w':>10} {'up a_b':>10} {'up T_e':>10} | "
        f"{'down a_w':>10} {'down a_b':>10} {'down T_e':>10}  {'verdict':>9}"
    )
    for row in ramps["up"][::2]:
        luminosity = row[0]
        other = down_by_l[luminosity]
        differs = abs(row[3] - other[3]) > 0.01
        # Read the verdict off the CLOSED FORM, not off the disagreement, so a slow
        # transient is never silently promoted to history-dependence -- and so the
        # two genuinely different kinds of bistability are not conflated.
        uninvadable = not (invade_lo <= luminosity <= invade_hi)
        interior = band_lo <= luminosity <= band_hi
        if not differs:
            verdict = ""
        elif uninvadable and interior:
            verdict = "BIST"
        elif uninvadable:
            verdict = "BIST-bdy"
        else:
            verdict = "(slow)"
        print(
            f"  {luminosity:>6.2f} {row[1]:>10.6f} {row[2]:>10.6f} {row[3]:>10.4f} | "
            f"{other[1]:>10.6f} {other[2]:>10.6f} {other[3]:>10.4f}  {verdict:>9}"
        )
    print("  -- BIST      both attractors exist: the INTERIOR state and a dead planet")
    print("               nothing can invade.  This is Daisyworld's bistability, and")
    print("               the closed form above says exactly where it lives.")
    print("  -- BIST-bdy  also genuine, but a different pair: outside the regulating")
    print("               band there is no interior state, so the up-ramp is holding a")
    print("               SINGLE-SPECIES boundary state against the dead planet.")
    print("  -- (slow)    the two halves disagree where the closed form says they")
    print("               cannot: the invader's net growth rate is near zero there and")
    print("               400 time units is not enough.  Reading this row as")
    print("               bistability would be the same mistake as the unfloored ramp.")
    return ramps["up"], ramps["down"]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _plot(regulation_rows, overcompensation_rows, ramp_up, ramp_down) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -- skipping the figures; use --extra viz)")
        return

    p0 = DaisyworldParams()
    t_w_star, t_b_star = interior_temperatures(p0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel 1: the temperatures, pinned, against the cover that is not.
    #
    # The two daisy series are the values COMPUTED at each L's own cover, not
    # `[t_w_star] * len(lums)`. The first draft did the latter, which draws the
    # flatness this panel exists to show -- the same defect as the Phase-2 Turing
    # panel whose stripes were a hand-seeded mode presented as a selected one. The
    # variation is ~1e-13 on a 290 K axis so the picture is identical either way;
    # the difference is whether the line demonstrates the claim or restates it.
    lums = [r[0] for r in regulation_rows]
    ax = axes[0]
    ax.plot(lums, [r[4] for r in regulation_rows], "o-", color="tab:blue", label="T_w (white)")
    ax.plot(lums, [r[5] for r in regulation_rows], "s-", color="tab:red", label="T_b (black)")
    ax.plot(
        [r[0] for r in overcompensation_rows],
        [r[1] for r in overcompensation_rows],
        "^-",
        color="tab:green",
        label="T_e (planet)",
    )
    ax.plot(
        [r[0] for r in overcompensation_rows],
        [r[2] for r in overcompensation_rows],
        "--",
        color="0.5",
        label="T_e, no daisies",
    )
    ax.set_xlabel("luminosity L")
    ax.set_ylabel("temperature (K)")
    ax.set_title("The daisy temperatures are flat;\nthe planet's runs BACKWARDS")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)

    # Panel 2: the cover, which is what L actually moves.
    ax = axes[1]
    ax.plot(lums, [r[1] for r in regulation_rows], "o-", color="tab:blue", label="a_w*")
    ax.plot(lums, [r[2] for r in regulation_rows], "s-", color="tab:red", label="a_b*")
    ax.plot(
        lums,
        [r[1] + r[2] for r in regulation_rows],
        "k--",
        label=f"total cover = {1 - bare_fraction(p0):.6f}",
    )
    low, high = regulating_band(p0)
    for edge in (low, high):
        ax.axvline(edge, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("luminosity L")
    ax.set_ylabel("fraction of surface")
    ax.set_title("L sets the SPLIT, not the total\n(dotted: the regulating band)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.3)

    # Panel 3: hysteresis -- category C, and labelled as such in the title.
    ax = axes[2]
    ax.plot(
        [r[0] for r in ramp_up],
        [r[3] for r in ramp_up],
        "o-",
        ms=3,
        color="tab:orange",
        label="L increasing",
    )
    ax.plot(
        [r[0] for r in ramp_down],
        [r[3] for r in ramp_down],
        "s--",
        ms=3,
        color="tab:purple",
        label="L decreasing",
    )
    bare = [bare_planet_temperature(DaisyworldParams(luminosity=r[0])) for r in ramp_up]
    ax.plot([r[0] for r in ramp_up], bare, ":", color="0.5", label="no daisies")

    # Shade the window where bistability is GENUINE, straight from the closed form.
    # Without it the panel says "every gap between the two ramps is history
    # dependence", which is the claim act 5's text spends a paragraph refusing:
    # the gap near L = 1.20 is a slow transient, because white can still invade
    # there. A figure gets looked at without its text, so the distinction has to
    # survive on the axes.
    windows = invasion_luminosities(p0)
    invade_hi = max(high for _, high in windows.values())
    ax.axvspan(
        invade_hi,
        high,
        color="tab:green",
        alpha=0.12,
        label=f"bistable: L in ({invade_hi:.3f}, {high:.3f})",
    )
    ax.axvline(invade_hi, color="tab:green", lw=0.8, ls="-.")
    ax.set_xlabel("luminosity L")
    ax.set_ylabel("T_e (K)")
    ax.set_title("CATEGORY C: hysteresis and dieback\n(shaded: both attractors exist)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Daisyworld -- the interior equilibrium is closed-form, so homeostasis is "
        "an exact statement:\n"
        f"T_w* = {t_w_star:.6f} K and T_b* = {t_b_star:.6f} K at every luminosity "
        "in the band, and dT_e/dL < 0",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig("daisyworld_regulation.png", dpi=130)
    plt.close(fig)
    print("  wrote daisyworld_regulation.png")


def main() -> None:
    print("Daisyworld -- Phase 3d")
    act1_the_reduction()
    regulation_rows = act2_regulation()
    overcompensation_rows = act3_overcompensation()
    act4_refusals()
    ramp_up, ramp_down = act5_hysteresis()

    _rule("Figures")
    _plot(regulation_rows, overcompensation_rows, ramp_up, ramp_down)

    _rule("Summary")
    p0 = DaisyworldParams()
    low, high = regulating_band(p0)
    t_w_star, t_b_star = interior_temperatures(p0)
    print(f"  T_w* = {t_w_star:.12f} K and T_b* = {t_b_star:.12f} K, independent of L.")
    print(f"  Bare fraction x* = {bare_fraction(p0):.12f}, also independent of L.")
    print(f"  Regulating band L in [{low:.10f}, {high:.10f}].")
    print(f"  Relaxation time at L = 1.0: {1.0 / slowest_rate(p0):.4f}")
    print("  The regulation claim is CATEGORY A -- a closed form, not a flat-looking")
    print("  curve.  Hysteresis, dieback and the dT_e/dL magnitudes are category C.")


if __name__ == "__main__":
    main()
