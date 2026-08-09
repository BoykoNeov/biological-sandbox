"""Periodic 5-point Laplacian — the discrete operator Phase 2b is built on.

Gray-Scott is the project's first PDE, and a PDE brings a failure mode the ODE
models did not have: the thing being simulated is not the equation on paper but
its *discretization*, and the two differ by ``O(h^2)``. This module keeps that
distinction explicit rather than letting it blur.

**The exact identity.** On a periodic box, a Fourier mode is an exact
eigenfunction of the discrete Laplacian — not approximately, not to ``O(h^2)``,
exactly:

    lap_h e^{i k.x} = -(4/h^2) sum_d sin^2(k_d h / 2)  e^{i k.x}

for any ``k_d = 2 pi j_d / L``. So a seeded mode's diffusive decay has a
closed-form rate that the simulation must reproduce to *numerical* precision,
which validates the stencil coefficients, the periodic wrap and the integrator in
a single cheap check.

**Which eigenvalue a prediction should use.** The continuum operator gives
``-D|k|^2``, and ``-(4D/h^2) sum sin^2(k_d h/2)`` converges to it as ``h -> 0``.
At ``L = 1, n = 128`` and ``j = 20`` the two differ by 8%, which is enormous
compared to any statistical tolerance. So anything asserted about a *simulation*
must use :func:`stencil_eigenvalue` — the eigenvalue of the operator actually
being integrated — and the gap to the continuum gets its own **order-2** log-log
check. Predicting with ``-D|k|^2`` and widening the tolerance until it passes
would convert a sharp claim into a vague one.

**No FFT.** ``np.roll`` on a few arrays is the whole implementation, it is
``O(cells)`` with a small constant, and it keeps the operator local — which is
what a stencil *is*. A spectral Laplacian would be a different (and for
Gray-Scott's nonlinear reaction terms, less convenient) model, not a faster
version of this one.
"""

from __future__ import annotations

import numpy as np


def laplacian(field: np.ndarray, h: float) -> np.ndarray:
    """Discrete Laplacian of ``field`` with periodic boundaries and spacing ``h``.

    Dimension-agnostic: the stencil is "sum both neighbours along every axis,
    subtract ``2 * ndim`` times the centre". ``np.roll`` supplies the wrap-around,
    so the periodicity is structural rather than a special case bolted onto the
    edges — there is no boundary branch to get wrong, which is precisely why the
    eigenfunction test can check the boundary rows and the interior with the same
    assertion.
    """
    out = -2.0 * field.ndim * field
    for axis in range(field.ndim):
        out = out + np.roll(field, 1, axis) + np.roll(field, -1, axis)
    return out / (h * h)


def stencil_eigenvalue(
    wavenumbers: tuple[float, ...] | np.ndarray, h: float, diffusivity: float = 1.0
) -> float:
    """``lambda_h`` for the mode ``e^{i k.x}``: ``-(4D/h^2) sum_d sin^2(k_d h/2)``.

    Exact for the discrete operator. The continuum limit ``-D|k|^2`` is what this
    approaches as ``h -> 0``; see the module docstring for why predictions must
    use this one and not that one.
    """
    k = np.asarray(wavenumbers, dtype=float)
    return float(-(4.0 * diffusivity / (h * h)) * np.sum(np.sin(k * h / 2.0) ** 2))


def grid_axis(n: int, length: float) -> np.ndarray:
    """Cell coordinates ``0, h, ..., L - h`` along one axis.

    The endpoint is **not** included: under periodicity ``x = L`` is the same
    point as ``x = 0``, so sampling both would duplicate a cell and quietly make
    the spacing ``L/(n-1)`` instead of ``L/n``.
    """
    return np.arange(n, dtype=float) * (length / n)


def allowed_wavenumbers(n: int, length: float) -> np.ndarray:
    """The wavenumbers ``2 pi j / L`` that are periodic and unaliased on the grid.

    Runs to the Nyquist mode ``j = n/2``; beyond that a mode aliases onto a lower
    one and is not an independent direction, so a "wavenumber" there names a
    different field than the caller intends.
    """
    return 2.0 * np.pi * np.arange(n // 2 + 1, dtype=float) / length


def mode_field(n: int, length: float, wavenumbers: tuple[float, float]) -> np.ndarray:
    """``cos(k.x)`` sampled on an ``n x n`` periodic grid over ``[0, L]^2``."""
    x = grid_axis(n, length)
    kx, ky = wavenumbers
    return np.cos(kx * x[:, None] + ky * x[None, :])


def cfl_limit(h: float, diffusivity: float, ndim: int = 2) -> float:
    """Largest stable explicit step for pure diffusion: ``h^2 / (2 ndim D)``.

    Derived from the operator rather than quoted: the most negative eigenvalue is
    ``-4 D ndim / h^2`` (every ``sin^2`` at 1, i.e. the grid-scale mode), and
    forward Euler is stable while ``dt |lambda| <= 2``.

    **Deliberately the Euler bound, though the models step with RK4**, whose
    real-axis stability limit is ``2.785`` and would allow 1.39x more. A margin
    borrowed from the integrator's stability polygon is a margin that vanishes the
    day the integrator changes, and 1.39x is not worth that coupling.
    """
    return (h * h) / (2.0 * ndim * diffusivity)
