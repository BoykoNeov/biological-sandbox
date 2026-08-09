"""The periodic 5-point Laplacian — Phase 2b's exact engine anchor.

This is the ``birth_death`` of the Gray-Scott track: a check so exact and so cheap
that everything downstream can lean on it. A single Fourier mode is an **exact
eigenfunction** of the periodic discrete Laplacian,

    lap_h e^{i k.x} = -(4/h^2) sum_d sin^2(k_d h / 2) e^{i k.x}

with the eigenvalue known in closed form and no ``h -> 0`` limit involved. That
one identity pins three things at once that would otherwise be validated
separately and vaguely: the **stencil coefficients**, the **periodic wrap-around**
(a mode is an eigenfunction only if the wrap is right — get it wrong and the two
boundary rows are the only ones that break), and, when the mode is integrated in
time, the **integrator**.

The continuum eigenvalue ``-D|k|^2`` is deliberately *not* used as the reference
for exactness. It is the ``h -> 0`` limit of the discrete one, so comparing to it
would fold an ``O(h^2)`` truncation into a check that is otherwise exact to
machine precision. Instead that gap gets its own test, as a *category B* log-log
slope of 2 — which is a sharper statement than any tolerance on it would be.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.laplacian import (
    allowed_wavenumbers,
    cfl_limit,
    grid_axis,
    laplacian,
    mode_field,
    stencil_eigenvalue,
)
from sandbox.core.ode import integrate_rk4

L = 1.0


def _h(n: int) -> float:
    return L / n


# ---------------------------------------------------------------------------
# The exact eigenvalue identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("jx, jy", [(1, 0), (0, 1), (3, 0), (0, 5), (2, 3), (7, 7), (1, -4)])
def test_a_fourier_mode_is_an_exact_eigenfunction(n: int, jx: int, jy: int) -> None:
    """``lap_h u = lambda_h u`` to machine precision — not to ``O(h^2)``.

    The wavenumbers are restricted to ``2 pi j / L`` because only those are
    periodic on the box; anything else is not an eigenfunction at all and the
    identity simply does not hold.
    """
    h = _h(n)
    k = (2.0 * np.pi * jx / L, 2.0 * np.pi * jy / L)
    u = mode_field(n, L, k)
    expected = stencil_eigenvalue(k, h) * u
    got = laplacian(u, h)
    # Scaled by the eigenvalue magnitude: lambda_h reaches ~4e5 at n=128, so an
    # absolute bound would be comparing a large number to a small one.
    scale = max(1.0, abs(stencil_eigenvalue(k, h)))
    assert np.abs(got - expected).max() < 1e-11 * scale


@pytest.mark.parametrize("n", [16, 64])
def test_the_wrap_is_periodic_not_merely_the_interior(n: int) -> None:
    """The eigenfunction identity holds *on the boundary rows too*.

    Stated separately because it is the only thing distinguishing this stencil
    from a zero-padded or reflected one, and because a broken wrap breaks exactly
    two rows and two columns out of ``n`` — at ``n = 128`` that is 3% of the grid,
    which a mean-based check would comfortably hide.
    """
    h = _h(n)
    k = (2.0 * np.pi * 3 / L, 2.0 * np.pi * 2 / L)
    u = mode_field(n, L, k)
    residual = np.abs(laplacian(u, h) - stencil_eigenvalue(k, h) * u)
    edge = np.concatenate([residual[0], residual[-1], residual[:, 0], residual[:, -1]])
    assert edge.max() < 1e-11 * abs(stencil_eigenvalue(k, h))
    assert edge.max() <= 10.0 * residual.max()


@pytest.mark.parametrize("n", [32, 128])
def test_a_constant_field_has_zero_laplacian_to_cancellation_noise(n: int) -> None:
    """Zero, but not *exactly* zero — and the bound says which zero it is.

    ``4c - 4c`` is exact in isolation; accumulated as four separate additions and
    then divided by ``h^2`` it leaves about one ulp of ``4c``, amplified by
    ``1/h^2``. Measured at ``1.08 eps |c| / h^2``, and that ratio is the same at
    ``n = 32`` and ``n = 128``, confirming it is cancellation and not a stencil
    error (which would scale as a *fixed fraction* of ``c/h^2``, i.e. ~1e15 times
    larger).
    """
    c, h = 3.7, _h(n)
    residual = float(np.abs(laplacian(np.full((n, n), c), h)).max())
    assert residual <= 4.0 * np.finfo(float).eps * abs(c) / h**2


def test_the_nyquist_mode_is_the_most_negative_eigenvalue() -> None:
    """``|lambda_h|`` is bounded by ``4 * ndim / h^2``, attained at the grid scale.

    This is what sets the explicit-stepping stability limit, so it is asserted
    rather than assumed: the CFL helper divides by exactly this number.
    """
    n = 32
    h = _h(n)
    nyquist = 2.0 * np.pi * (n // 2) / L
    assert stencil_eigenvalue((nyquist, nyquist), h) == pytest.approx(-8.0 / h**2, rel=1e-12)
    worst = min(
        stencil_eigenvalue((2 * np.pi * jx / L, 2 * np.pi * jy / L), h)
        for jx in range(n // 2 + 1)
        for jy in range(n // 2 + 1)
    )
    assert worst == pytest.approx(-8.0 / h**2, rel=1e-12)


def test_allowed_wavenumbers_are_the_periodic_ones() -> None:
    n = 32
    ks = allowed_wavenumbers(n, L)
    assert ks[0] == 0.0
    assert ks[1] == pytest.approx(2.0 * np.pi / L)
    assert ks.size == n // 2 + 1
    # Every one of them is genuinely an eigenfunction; the next one up is aliased
    # onto a lower mode and is not an independent direction.
    h = _h(n)
    for k in ks:
        u = mode_field(n, L, (float(k), 0.0))
        scale = max(1.0, abs(stencil_eigenvalue((float(k), 0.0), h)))
        assert np.abs(laplacian(u, h) - stencil_eigenvalue((float(k), 0.0), h) * u).max() < (
            1e-11 * scale
        )


# ---------------------------------------------------------------------------
# Category B: second-order consistency with the continuum
# ---------------------------------------------------------------------------


def test_consistency_error_has_log_log_slope_two() -> None:
    """``|lambda_h - (-D|k|^2)|`` falls as ``h^2``. The claim is the exponent.

    The truncation is ``D|k|^4 h^2 / 12`` for a 1-D mode, so this also checks the
    *coefficient*: at the coarsest grid the ratio to that prediction must already
    be near 1, which a wrong stencil normalization (say ``h`` instead of ``h^2``)
    would miss even while producing some straight line on a log-log plot.

    **The grid range is chosen, not arbitrary.** Written first over
    ``n = 16..256`` this measured **1.9737**, and the shortfall is real rather
    than noise: the expansion is in ``k h / 2``, which at ``n = 16`` and
    ``j = 4`` is ``0.785`` — nowhere near small, so the ``h^4`` term is still
    contributing and the coarsest point sits 8% below the ``h^2`` line. On
    ``n = 64..1024`` the same mode gives **1.99835** with the coarsest point
    within 0.5% of the predicted coefficient. An asymptotic order is only
    measurable inside the asymptotic regime, and "which grids are inside it" is
    part of the claim.
    """
    k = (2.0 * np.pi * 4 / L, 0.0)
    diffusivity = 2.0e-5
    k2 = float(k[0] ** 2 + k[1] ** 2)
    ns = np.array([64, 128, 256, 512, 1024], dtype=float)
    hs = L / ns
    errors = np.array(
        [abs(stencil_eigenvalue(k, float(h), diffusivity) - (-diffusivity * k2)) for h in hs]
    )
    slope = float(np.polyfit(np.log(hs), np.log(errors), 1)[0])
    assert slope == pytest.approx(2.0, abs=0.01)

    predicted_coarse = diffusivity * k[0] ** 4 * hs[0] ** 2 / 12.0
    assert errors[0] == pytest.approx(predicted_coarse, rel=0.02)


# ---------------------------------------------------------------------------
# Stencil + periodic BCs + integrator, together
# ---------------------------------------------------------------------------


def test_pure_diffusion_of_one_mode_decays_at_the_stencil_rate() -> None:
    """``du/dt = D lap_h u`` on a seeded mode is exactly ``u(0) e^{lambda_h t}``.

    The whole point of anchoring on the *discrete* eigenvalue: this is an exact
    statement about the system actually being integrated, so the only error left
    is RK4's, and the tolerance is a numerical one rather than a modelling one.
    Richardson supplies it — halving ``dt`` must shrink the discrepancy by ~16,
    which is asserted so that "small" cannot quietly mean "small for the wrong
    reason".
    """
    # t_max chosen for ~3 e-folds at the measured lambda_h = -0.0224942: 3/lambda
    # is 133.4. The first draft used 400, which is nine e-folds -- the mode had
    # decayed to 1.2e-4 of its amplitude, so "the error is small" would have been
    # a statement about a field that was gone.
    n, diffusivity, t_max = 64, 2.0e-5, 130.0
    h = _h(n)
    k = (2.0 * np.pi * 5 / L, 2.0 * np.pi * 2 / L)
    lam = stencil_eigenvalue(k, h, diffusivity)
    u0 = mode_field(n, L, k)

    def rhs(y: np.ndarray) -> np.ndarray:
        return diffusivity * laplacian(y.reshape(n, n), h).reshape(-1)

    errors = []
    for dt in (2.0, 1.0):
        _, y = integrate_rk4(rhs, u0.reshape(-1), t_max, dt)
        errors.append(float(np.abs(y[-1].reshape(n, n) - u0 * np.exp(lam * t_max)).max()))
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.35)
    assert errors[1] < 1e-3 * abs(float(np.exp(lam * t_max)))
    # And it really decayed rather than sitting still: ~3 e-folds over the horizon.
    assert 0.02 < np.exp(lam * t_max) < 0.1


# ---------------------------------------------------------------------------
# CFL
# ---------------------------------------------------------------------------


def test_cfl_limit_is_the_explicit_stability_bound() -> None:
    """``dt <= h^2 / (2 ndim D)`` — derived from the Nyquist eigenvalue, not typed in.

    Deliberately the *forward-Euler* bound even though the models step with RK4,
    whose real-axis limit is ~1.39x more generous. A stability margin bought from
    the integrator's stability polygon is a margin that disappears the day the
    integrator changes.
    """
    h, diffusivity = 0.01, 2.0e-5
    assert cfl_limit(h, diffusivity, ndim=2) == pytest.approx(h**2 / (4.0 * diffusivity))
    assert cfl_limit(h, diffusivity, ndim=2) == pytest.approx(
        2.0 / abs(stencil_eigenvalue((np.pi / h, np.pi / h), h, diffusivity))
    )


def test_grid_axis_spans_the_box_without_duplicating_the_endpoint() -> None:
    """``L`` is the same point as ``0`` under periodicity, so it must not be sampled twice."""
    x = grid_axis(8, L)
    assert x.size == 8
    assert x[0] == 0.0
    assert x[-1] == pytest.approx(L - L / 8)
