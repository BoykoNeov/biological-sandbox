"""Gray-Scott demo — validated linear stability, and the pattern that isn't one.

Run it:  ``uv run python -m sandbox.demos.gray_scott``

Four acts:

1. **The engine anchor.** A single Fourier mode is an *exact* eigenfunction of the
   periodic 5-point Laplacian, so pure diffusion of one seeded mode decays at a
   rate known in closed form. This is Phase 2b's ``birth_death``: it pins the
   stencil, the periodic wrap and the integrator together.
2. **The dispersion relation.** ``lambda(q)`` measured against
   ``max Re eig(J - q_eff^2 D)`` across a **sign change** — probes that grow and
   probes that decay. This is the honest version of HANDOFF's "validate pattern
   wavelength against LSA".
3. **The Turing pattern** at the validated point, run to saturation. Qualitative:
   the *linear* claim is act 2; the saturated amplitude is nonlinear and nothing
   here asserts anything about it.
4. **Pearson's famous pattern**, explicitly labelled exploratory — because at
   those parameters there is no real non-trivial homogeneous state at all, so the
   spots are not Turing patterns and linear theory sets no wavelength for them.

**The one thing worth taking away from act 2's figure.** The discrete operator and
the continuum ``-D q^2`` are plotted together, and they are not close: at
``n = 64`` they differ by 408% at ``j = 12`` and *disagree about the sign* at
``j = 13``. What is simulated is the stencil, so the stencil's eigenvalue is what
a prediction has to use.

Figures are written to the current directory (``*.png`` is gitignored). All
printed output is ASCII-only (Windows cp1252 console).
"""

from __future__ import annotations

import numpy as np

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.laplacian import (
    laplacian,
    mode_field,
    stencil_eigenvalue,
)
from sandbox.core.ode import integrate_rk4
from sandbox.core.recorder import run_replicate
from sandbox.models.gray_scott import (
    MODEL,
    GrayScottParams,
    dispersion,
    homogeneous_state,
    n_gs_steps,
    reaction_jacobian,
)

_N, _DT = 64, 0.2
_FEED, _KILL = 0.074, 0.062
_GROWING_EPS, _DECAYING_EPS = 1.35e-5, 7.4e-4
_PROBES = [
    (3, 545.4, _GROWING_EPS),
    (5, 161.2, _GROWING_EPS),
    (7, 133.6, _GROWING_EPS),
    (10, 181.4, _GROWING_EPS),
    (12, 457.8, _GROWING_EPS),
    (14, 429.6, _DECAYING_EPS),
    (16, 130.6, _DECAYING_EPS),
    (20, 51.8, _DECAYING_EPS),
]

# Pearson's classic spot-forming point. Kept for the picture only.
_PEARSON = {"feed": 0.037, "kill": 0.060}


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _params(**kwargs: object) -> GrayScottParams:
    base: dict[str, object] = {
        "feed": _FEED,
        "kill": _KILL,
        "n": _N,
        "dt": _DT,
        "mode_j": 7,
        "eps": _GROWING_EPS,
        "t_max": 133.6,
    }
    base.update(kwargs)
    return GrayScottParams(**base)  # type: ignore[arg-type]


def _final_state(params: GrayScottParams):
    rng = np.random.default_rng(0)
    state = MODEL.initial_state(params, rng)
    while not MODEL.is_terminal(state):
        state = MODEL.step(state, rng)
    return state


# ---------------------------------------------------------------------------
# Act 1 -- the exact engine anchor
# ---------------------------------------------------------------------------


def _diffusion_anchor() -> bool:
    _rule("1. The engine anchor: a Fourier mode is an EXACT eigenfunction")
    length, diffusivity, t_max, dt = 1.0, 2.0e-5, 130.0, 1.0
    h = length / _N
    k = (2.0 * np.pi * 5 / length, 2.0 * np.pi * 2 / length)
    lam = stencil_eigenvalue(k, h, diffusivity)
    u0 = mode_field(_N, length, k)

    def rhs(y: np.ndarray) -> np.ndarray:
        return diffusivity * laplacian(y.reshape(_N, _N), h).reshape(-1)

    _, y = integrate_rk4(rhs, u0.reshape(-1), t_max, dt)
    measured = float(np.abs(y[-1].reshape(_N, _N) - u0 * np.exp(lam * t_max)).max())
    print(f"pure diffusion of the mode (kx, ky) = 2pi(5, 2) on a {_N}x{_N} periodic grid")
    print(f"  closed-form rate lambda_h = -(4D/h^2) sum sin^2(k h/2) = {lam:+.8g}")
    print(f"  amplitude after {t_max:g} time units: exp(lambda_h t) = {np.exp(lam * t_max):.6g}")
    print(f"  max |simulated - exact| = {measured:.3g}")
    print("\nThis validates the stencil, the periodic wrap and the integrator at once:")
    print("a mode is an eigenfunction only if the wrap-around is right.")
    return measured < 1e-6


# ---------------------------------------------------------------------------
# Act 2 -- the dispersion relation, across the sign change
# ---------------------------------------------------------------------------


def _continuum_rate(j: int) -> float:
    u, v = homogeneous_state(_params())
    q = 2.0 * np.pi * j
    m = reaction_jacobian(u, v, _FEED, _KILL) - q**2 * np.diag([2.0e-5, 1.0e-5])
    return float(np.max(np.linalg.eigvals(m).real))


def _dispersion_check() -> tuple[bool, list[tuple[int, float, float]]]:
    _rule("2. Category A: lambda(q) measured against linear stability analysis")
    u_star, v_star = homogeneous_state(_params())
    print(f"homogeneous steady state at (F, k) = ({_FEED:g}, {_KILL:g}):")
    print(f"  u* = {u_star:.8f}, v* = {v_star:.8f}")
    eigen = np.linalg.eigvals(reaction_jacobian(u_star, v_star, _FEED, _KILL))
    print(f"  eig(J) without diffusion = {eigen[0]:.6g}, {eigen[1]:.6g}  (stable)")
    print("  -- so any instability below is diffusion-driven: a Turing instability.\n")

    print(f"  {'j':>3} {'predicted':>13} {'measured':>13} {'|err|':>10} {'tol':>10} {'':>4}")
    rows, ok = [], True
    for mode_j, t_max, eps in _PROBES:
        params = _params(mode_j=mode_j, t_max=t_max, eps=eps)
        predicted = MODEL.analytic_predictions(params)["growth_rate"]
        coarse = MODEL.observables(_final_state(params))["growth_rate"]
        fine = MODEL.observables(_final_state(_params(mode_j=mode_j, t_max=t_max, eps=eps / 2.0)))[
            "growth_rate"
        ]
        tolerance = 3.0 * (4.0 / 3.0) * abs(coarse - fine)
        passed = abs(coarse - predicted) <= tolerance
        ok = ok and passed
        rows.append((mode_j, predicted, coarse))
        print(
            f"  {mode_j:>3} {predicted:>+13.6g} {coarse:>+13.6g} "
            f"{abs(coarse - predicted):>10.2g} {tolerance:>10.2g} "
            f"{'ok' if passed else 'FAIL':>4}"
        )

    print("\nThe tolerance is Richardson in the AMPLITUDE, not in dt: the discrepancy is")
    print("a linearization error of order a^2, so halving the seed quarters it. Richardson")
    print("in dt would report a reassuringly tiny number about the wrong thing.")
    print("\nProbes at j = 3..12 GROW and j = 14..20 DECAY -- the check spans a sign")
    print("change, so a wrong sign cannot hide inside a tolerance.")
    print("\nDiscrete vs continuum, the reason the stencil eigenvalue is the reference:")
    for j in (7, 12, 13):
        discrete, _, _ = dispersion(_params(), j)
        continuum = _continuum_rate(j)
        print(
            f"  j={j:>3}: stencil {discrete:+.6g}  continuum {continuum:+.6g}  "
            f"({100 * abs(discrete / continuum - 1):.2g}% apart"
            f"{', OPPOSITE SIGN' if discrete * continuum < 0 else ''})"
        )
    return ok, rows


# ---------------------------------------------------------------------------
# Acts 3 and 4 -- the patterns
# ---------------------------------------------------------------------------


def _turing_pattern():
    _rule("3. The Turing pattern at the validated point (qualitative)")
    params = _params(mode_j=7, eps=1e-3, t_max=3000.0, dt=_DT, n=128)
    state = _final_state(params)
    fields = MODEL.fields(state)
    print(f"grown from the validated (F, k) over {params.t_max:g} time units, n = {params.n}")
    print(f"  u in [{fields['u'].min():.4f}, {fields['u'].max():.4f}]")
    print(f"  v in [{fields['v'].min():.4f}, {fields['v'].max():.4f}]")
    fastest = max(range(1, params.n // 2), key=lambda j: dispersion(params, j)[0])
    print(f"  fastest-growing mode of the dispersion relation at n = {params.n}: j = {fastest}")
    print("\nThe stripes are the SEEDED mode saturating -- j = 7 was put in by hand, not")
    print("chosen by the system. It is worth showing because 7 is also the fastest-growing")
    print("mode, so it is the one linear theory expects to dominate; but this figure is")
    print("NOT evidence of wavelength selection, and presenting it as such would be the")
    print("same overclaim as validating a wavelength against LSA.")
    print("\nThe LINEAR claim is act 2 and it is validated. The saturated amplitude is")
    print("nonlinear, and nothing here asserts anything about it.")
    return fields


def _pearson_pattern():
    _rule("4. Pearson's pattern -- EXPLORATORY, and here is why")
    feed, kill = _PEARSON["feed"], _PEARSON["kill"]
    print(
        f"at (F, k) = ({feed:g}, {kill:g}):  F = {feed:g}  <  4(F+k)^2 = {4 * (feed + kill) ** 2:g}"
    )
    try:
        homogeneous_state(_params(feed=feed, kill=kill))
    except ValueError as exc:
        print(f"\nhomogeneous_state raises, and this is the whole reframing:\n  {exc}")

    params = GrayScottParams(
        feed=feed,
        kill=kill,
        n=128,
        dt=0.5,
        t_max=8000.0,
        mode_j=7,
        eps=0.0,
        initial="pearson",
        noise=0.01,
    )
    traj = run_replicate(MODEL, params, np.random.default_rng(0), max_steps=n_gs_steps(params) + 10)
    state = _final_state(params)
    fields = MODEL.fields(state)
    print(f"\nran the classic blob initial condition for {params.t_max:g} time units")
    print(
        f"  v in [{fields['v'].min():.4f}, {fields['v'].max():.4f}], mean {fields['v'].mean():.4f}"
    )
    print(f"  ({len(traj.times)} recorded steps)")
    print("\nThe pattern below is real and famous. Its wavelength is NOT compared to")
    print("linear stability analysis anywhere in this project, because LSA about a")
    print("steady state that does not exist predicts nothing. Reporting a number here")
    print("and calling it validated is exactly the failure this phase was built to avoid.")
    return fields


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _figures(rows, turing, pearson) -> None:
    _rule("Figures")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping figures (uv sync --extra viz)")
        return

    from sandbox.viz.backends.matplotlib_backend import plot_field

    # 1. the dispersion relation: prediction curve, measurements, and the continuum
    fig, ax = plt.subplots(figsize=(8, 5.5))
    js = np.arange(1, 25)
    rates = [dispersion(_params(), int(j)) for j in js]
    discrete = np.array([r[0] for r in rates])
    real = np.array([r[2] for r in rates])
    continuum = np.array([_continuum_rate(int(j)) for j in js])
    q = 2.0 * np.pi * js
    # Split at the real/complex boundary. Where the pair is complex the plotted
    # value is tr/2, which is a decay envelope and NOT a measurable growth rate --
    # drawing it in the same style would make the kink there look like physics.
    ax.plot(q[real], discrete[real], "-", color="C0", label="prediction: stencil eigenvalue")
    ax.plot(
        q[~real],
        discrete[~real],
        ":",
        color="C0",
        alpha=0.5,
        label="complex pair: Re only, not a measurable rate",
    )
    ax.plot(q, continuum, "--", color="C7", label="continuum $-Dq^2$ (NOT what is simulated)")
    ax.plot(
        [2.0 * np.pi * j for j, _, _ in rows],
        [m for _, _, m in rows],
        "o",
        color="C3",
        markersize=7,
        label="measured growth rate",
    )
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("wavenumber q")
    ax.set_ylabel("growth rate lambda(q)")
    ax.set_title(f"Gray-Scott dispersion relation at (F, k) = ({_FEED:g}, {_KILL:g}), n = {_N}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("gs_dispersion.png", dpi=120)
    plt.close(fig)
    print("wrote gs_dispersion.png")

    # 2. the two patterns, side by side and labelled for what they are
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_field(
        turing["v"],
        title="VALIDATED point (0.074, 0.062): mode j=7 seeded, grown to saturation",
        ax=axes[0],
    )
    plot_field(
        pearson["v"],
        title="EXPLORATORY (0.037, 0.060): no Turing state exists here",
        cmap="magma",
        ax=axes[1],
    )
    fig.suptitle("Gray-Scott: v field. Left is where lambda(q) was validated; right is not.")
    fig.tight_layout()
    fig.savefig("gs_patterns.png", dpi=120)
    plt.close(fig)
    print("wrote gs_patterns.png")


def main() -> int:
    anchor_ok = _diffusion_anchor()
    dispersion_ok, rows = _dispersion_check()
    turing = _turing_pattern()
    pearson = _pearson_pattern()
    _figures(rows, turing, pearson)

    _rule("Summary")
    print(f"engine anchor (exact Fourier eigenvalue) : {'PASS' if anchor_ok else 'FAIL'}")
    print(f"category A (lambda(q), 8 probes)         : {'PASS' if dispersion_ok else 'FAIL'}")
    print("Turing pattern                           : qualitative, asserted nowhere")
    print("Pearson pattern                          : EXPLORATORY, no LSA claim made")
    return 0 if (anchor_ok and dispersion_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
