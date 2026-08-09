"""Hodgkin-Huxley voltage-dependent rate functions (the 1952 empirical fit).

The six rate functions ``alpha_x(V)``, ``beta_x(V)`` for the gates ``m`` (Na
activation), ``h`` (Na inactivation) and ``n`` (K activation), in the modern
convention: ``V`` in mV with rest near ``-65``, rates in ms^-1. Everything in the
Phase-2 Hodgkin-Huxley track sits on top of these — the voltage-clamp exact
anchor validates them individually, the deterministic model integrates them, and
the stochastic model draws channel transitions from them — so they are their own
module with their own test file.

**The one trap, and why it gets a helper instead of a comment.** Two of the six,
``alpha_m`` and ``alpha_n``, have the form ``x / (1 - exp(-x/k))``, which is
``0/0`` at ``x = 0`` — that is, at exactly ``V = -40`` and ``V = -55``. Those are
round numbers: they are precisely the voltages a clamp protocol steps to, an f-I
sweep lands on, or a root-finder converges to. A literal transcription returns
``nan`` there and the failure surfaces far downstream as a ``nan`` membrane
potential, which reads as a numerical instability rather than a rate-function bug.

:func:`_linoid` handles it. The removable singularity has limit ``k``
(L'Hopital), and the two branches are chosen for *accuracy*, not just finiteness:

* away from zero, ``1 - exp(-x/k)`` is evaluated as ``-expm1(-x/k)``. The naive
  difference cancels catastrophically for small ``x`` (at ``|x| ~ 1e-9`` it has
  lost most of its significant digits long before it returns ``nan``); ``expm1``
  is accurate to machine precision across the whole range.
* within ``|x| < 1e-7 k`` the series ``k + x/2`` is used, whose truncation error
  ``x^2/(12k)`` is already below machine epsilon relative to ``k`` there.
"""

from __future__ import annotations

import numpy as np

# Gate order, fixed once and shared by the 4-vector [V, m, h, n], the
# channel-state occupancy model and every observable key downstream.
GATES: tuple[str, str, str] = ("m", "h", "n")


def _linoid(x: np.ndarray | float, k: float) -> np.ndarray:
    """``x / (1 - exp(-x/k))``, with the removable ``0/0`` at ``x = 0`` handled.

    Returns ``k`` at ``x = 0`` (the L'Hopital limit) and stays accurate to machine
    precision on both sides — see the module docstring for why the naive form is
    wrong well before it is ``nan``.
    """
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-7 * k
    # The dummy value keeps expm1 away from its own zero; `small` entries are
    # discarded by the where, so the placeholder never reaches the result.
    safe = np.where(small, k, x)
    return np.where(small, k + 0.5 * x, safe / -np.expm1(-safe / k))


def alpha_m(V: np.ndarray | float) -> np.ndarray:
    return 0.1 * _linoid(np.asarray(V, dtype=float) + 40.0, 10.0)


def beta_m(V: np.ndarray | float) -> np.ndarray:
    return 4.0 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 18.0)


def alpha_h(V: np.ndarray | float) -> np.ndarray:
    return 0.07 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 20.0)


def beta_h(V: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(V, dtype=float) + 35.0) / 10.0))


def alpha_n(V: np.ndarray | float) -> np.ndarray:
    return 0.01 * _linoid(np.asarray(V, dtype=float) + 55.0, 10.0)


def beta_n(V: np.ndarray | float) -> np.ndarray:
    return 0.125 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 80.0)


_RATE_PAIRS = {
    "m": (alpha_m, beta_m),
    "h": (alpha_h, beta_h),
    "n": (alpha_n, beta_n),
}


def rate_pair(gate: str, V: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """``(alpha, beta)`` for one gate at voltage(s) ``V``."""
    try:
        a, b = _RATE_PAIRS[gate]
    except KeyError:
        raise KeyError(f"unknown gate {gate!r}; known: {', '.join(GATES)}") from None
    return a(V), b(V)


def alphas_betas(V: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Stacked ``(alpha, beta)``, each shaped ``(..., 3)`` in ``GATES`` order."""
    pairs = [rate_pair(g, V) for g in GATES]
    return (
        np.stack([p[0] for p in pairs], axis=-1),
        np.stack([p[1] for p in pairs], axis=-1),
    )


def steady_state(V: np.ndarray | float) -> np.ndarray:
    """``x_inf = alpha / (alpha + beta)``, shaped ``(..., 3)`` in ``GATES`` order."""
    a, b = alphas_betas(V)
    return a / (a + b)


def time_constant(V: np.ndarray | float) -> np.ndarray:
    """``tau = 1 / (alpha + beta)`` in ms, shaped ``(..., 3)`` in ``GATES`` order."""
    a, b = alphas_betas(V)
    return 1.0 / (a + b)
