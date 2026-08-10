"""The May / Allesina-Tang random community matrix — and it is deliberately NOT a Model.

A matrix ensemble has no ``step``, no ``observables`` and no ``state.t``. Contriving
a :class:`~sandbox.core.protocol.Model` whose ``step`` draws a matrix would be
exactly the protocol abuse non-negotiable #1 exists to prevent, so this module is
plain functions plus report dataclasses, checked by a plain pytest **outside** the
ValidationSuite path and **not** registered in ``models/__init__.py``. It lives in
``core/`` because it is a shared numerical fact rather than a model — the same
narrow ground the viz-only ``FieldModel`` stands on.

**What is claimed, and what is emphatically not.** May's criterion is a statement
about a random matrix *assumed* to be a community matrix. Phase 3 measured what
happens if you try to obtain that matrix from a random generalized Lotka-Volterra
system and found the promise cannot be run as written: the ensemble is *empty* at
the ``S`` where the asymptotic criterion means anything (feasibility 0.000 at
``S = 40, sigma = 0.25``), and conditioning on feasibility **moves the spectrum**
(``max Re eig(A) = -0.335`` against ``max Re eig(diag(x*) A) = -0.117``). So this
module validates the **matrix law directly** and claims nothing about
feasibility-conditioned gLV. The demo says so in as many words.

The ensemble
------------

``M = -d I + B``, where ``B`` has a **zero diagonal** (self-regulation is the
``-d I`` term, which is May's convention) and off-diagonal *pairs* ``(B_ij, B_ji)``
that are present with probability ``C`` and, when present, bivariate normal with
mean 0, variance ``sigma^2`` and correlation ``rho``. The bulk radius is
``R = sigma sqrt(S C)``; the spectrum fills an **ellipse** with semi-axes
``R(1 + rho)`` and ``R(1 - rho)``, centred at ``-d``. ``rho = 0`` recovers the
circular law. The ecologically meaningful reading is that predator-prey structure
(``rho < 0``) makes a web *more* stable and competition/mutualism (``rho > 0``)
less, which is strictly sharper than May's ``rho = 0`` special case at identical
cost.

Two tracks, and why the obvious single-track design does not work
-----------------------------------------------------------------

The natural check is the **fraction** of eigenvalues inside a scaled disc: for the
circular law it is exactly ``rho_probe^2``, giving ``S`` eigenvalues per draw and
a *binomial* SE. The trap — the Phase-2 floor lesson in a new guise — is that the
law is exact only as ``S -> infinity``. At finite ``S`` there is a **bias**, and
piling on eigenvalues shrinks the SE *below* it, so a correct implementation
starts failing. It is therefore not enough to pick ``S`` and pile on draws.

Phase 3 measured that bias properly and found it worse than the plan assumed, in
three ways that between them rule out the single-track design:

1. The bias constant is **not** ``0.6/S``; re-measured where it is resolvable it is
   ``0.70/S`` at ``rho_probe = 0.5``. The recorded ``0.6`` came from a fit whose two
   largest-``S`` points sat *below their own SE* — noise, fitted as if it were a
   bias.
2. It is **not one constant across probes**. ``bias * S`` runs ``0.48 ... 1.13``
   over ``rho_probe = 0.2 ... 0.9``, and the *exponent* differs too: ``-0.88``,
   ``-1.03``, ``-0.62`` at ``rho_probe = 0.2 / 0.5 / 0.9``. Only ``rho_probe = 0.5``
   is the clean ``1/S``.
3. The **elliptic** bias at ``rho = +0.8`` is 4-5x any circular one (``bias * S``
   about 3.5-3.8, decaying like ``S^-0.84``).

The binding quantity is ``bias / SE``, not ``bias``, and since SE varies with the
probe too, the binding probe maximizes ``c(rho_probe) / sqrt(p(1-p))``. Requiring
``bias <= SE/4`` then permits about **two draws** at ``rho_probe = 0.2``, **1.4** at
``0.9``, and **under one** at elliptic ``rho = +0.8``. No affordable ``S`` rescues
it: ``eigvals`` costs 0.018 s at ``S = 200`` and 0.15 s at ``S = 400`` (8x for 8x
the FLOPs, as expected) but then **2.0 s at ``S = 600`` — 13x the ``S = 400`` cost
for 3.4x the FLOPs**. ``S = 400`` is the last cheap size, and the ``S^3`` cost model
does not survive past it.

So the module offers two checks, and the *second* is the load-bearing one:

* :func:`fraction_report` — the direct probe, usable **only** where the bias really
  is negligible. It does not take that on trust: it computes the bias estimate from
  a caller-supplied measured constant and **refuses the configuration** whose bias
  is not small against its own SE, naming the binding probe. In practice the one
  satisfiable point is ``rho_probe = 0.5, S = 400, 9 draws`` (bias ``0.24 SE``,
  about 1.4 s), where a 10%-wrong ``R`` still shifts the fraction ``7.3 SE``.
* :func:`bias_scaling_report` — the asymptotic law, in the project's own idiom
  (log-log slope + CI, like ``D(Omega) ~ Omega^{-1/2}``). It needs ``bias >> SE``,
  which is the regime that *is* cheaply reachable, so it works at every probe
  including the elliptic ``rho = +-0.8`` that the direct check cannot touch.

**What makes the scaling check falsifiable**, since this is easy to get wrong:
fitting ``bias = c S^-p`` with ``p`` free and then "extrapolating to ``S ->
infinity``" cannot fail — the extrapolated limit is 0 by construction. The
falsifiable content is the frozen-channel lesson from Phase 2: *a wrong
implementation's discrepancy does not shrink with ``S`` at all*. A wrong ``R``
gives a constant offset; a Hermitian-ized draw gives a completely different
(real, semicircular) spectrum; a flipped ``rho`` sign at ``|rho| >= 0.4`` swaps the
semi-axes. All three produce an ``O(1)`` offset and an exponent of about **zero**.
So the assertion is that the exponent is **significantly negative**, and the fitted
value is reported rather than asserted — there is no theory here predicting
``-1``, and the exponents measurably differ between probes.

**And the second guard is the one the plan's own number needed.** ``resolved``
requires *every* fitted point to have ``|bias| > z * SE``. Without it the fit
happily consumes points that cannot resolve what they are measuring, which is
precisely how ``-0.9279`` was produced. A point that cannot fail is not a check.

A tempting mechanism that the measurement rejects
-------------------------------------------------

Most of the edge bias is the **zero diagonal** — filling it Ginibre-style drops
``bias * S`` at ``rho_probe = 0.9`` from 1.06 to 0.35 while barely moving
``rho_probe = 0.2``. Zeroing the diagonal pins ``trace = 0`` exactly, constraining
the eigenvalue sum and redistributing mass at the edge. That is a property of
May's ensemble, not a defect to correct: ``B`` *has* a zero diagonal.

It also suggests a bulk/edge split — an ``O(1/S)`` bulk correction plus a slower one
inside a Ginibre edge layer of width ``~S^-1/2`` — which would predict
``rho_probe = 0.2``, deep in the bulk, at ``-1``. **It measures ``-0.8814 +- 0.025``,
shallower than ``rho_probe = 0.5``.** The ordering is not monotone in the probe, so
the exponent is treated as probe-dependent and *unexplained*, and no predicted
exponent enters any assertion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def bulk_radius(n_species: int, connectance: float = 1.0, sigma: float = 1.0) -> float:
    """``R = sigma sqrt(S C)`` — the circular-law radius of the interaction block.

    With correlation ``rho`` the spectrum fills an ellipse of semi-axes
    ``R(1 + rho)`` and ``R(1 - rho)``; ``R`` itself is unchanged, which is why the
    same quantity serves both laws.
    """
    return float(sigma) * float(np.sqrt(n_species * connectance))


def draw_community_matrix(
    rng: np.random.Generator,
    *,
    n_species: int,
    connectance: float = 1.0,
    sigma: float = 1.0,
    correlation: float = 0.0,
    self_regulation: float = 0.0,
) -> np.ndarray:
    """One draw of ``M = -d I + B`` from the May / Allesina-Tang ensemble.

    Off-diagonal *pairs* ``(B_ij, B_ji)`` are drawn together: present with
    probability ``connectance`` and, when present, bivariate normal with mean 0,
    marginal variance ``sigma^2`` and correlation ``correlation``. Drawing the pair
    as a unit is what makes the correlation meaningful — zeroing one member without
    the other would destroy it — and it keeps the effective correlation equal to
    ``correlation`` over all pairs, since absence removes both moments together.

    The diagonal of ``B`` is **zero**: in May's decomposition self-interaction is
    entirely the ``-d I`` term. This is not cosmetic. It pins ``trace(B) = 0``
    exactly, which constrains the eigenvalue sum and is measurably the dominant
    source of the finite-``S`` bias near the spectral edge (see the module
    docstring). ``self_regulation`` only translates the spectrum along the real
    axis, so both fraction checks are insensitive to it; it defaults to 0 so that
    the returned matrix *is* ``B`` unless a caller wants the shifted community
    matrix.
    """
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must lie in [-1, 1], got {correlation!r}")
    if not 0.0 <= connectance <= 1.0:
        raise ValueError(f"connectance must lie in [0, 1], got {connectance!r}")

    upper_idx = np.triu_indices(n_species, k=1)
    n_pairs = upper_idx[0].size
    z = rng.standard_normal((n_pairs, 2))
    upper = sigma * z[:, 0]
    lower = sigma * (correlation * z[:, 0] + np.sqrt(1.0 - correlation**2) * z[:, 1])
    if connectance < 1.0:
        present = rng.random(n_pairs) < connectance
        upper = np.where(present, upper, 0.0)
        lower = np.where(present, lower, 0.0)

    matrix = np.zeros((n_species, n_species), dtype=float)
    matrix[upper_idx] = upper
    matrix[(upper_idx[1], upper_idx[0])] = lower
    if self_regulation:
        matrix[np.diag_indices(n_species)] = -float(self_regulation)
    return matrix


def ensemble_eigenvalues(
    rng: np.random.Generator, *, n_draws: int, n_species: int, **kwargs: float
) -> np.ndarray:
    """Eigenvalues of ``n_draws`` independent draws, concatenated.

    Returns a flat complex array of ``n_draws * n_species`` eigenvalues. Pooling
    across draws is legitimate for a *fraction* statistic — every eigenvalue is one
    Bernoulli trial for "inside the probe region" — which is what makes a binomial
    SE the right instrument and gives the check its enormous statistics. It would
    **not** be legitimate for an extreme statistic such as ``max |lambda|``, whose
    per-draw values are not exchangeable with each other; the spectral radius was
    measured and rejected as an anchor on separate grounds anyway (a 2-3.5% excess
    that barely moves across two decades of ``S``).
    """
    return np.concatenate(
        [
            np.linalg.eigvals(draw_community_matrix(rng, n_species=n_species, **kwargs))
            for _ in range(n_draws)
        ]
    )


def disc_fraction(
    eigenvalues: np.ndarray, *, radius: float, probe: float, center: float = 0.0
) -> float:
    """Fraction of eigenvalues within ``probe * radius`` of ``center``.

    The circular law makes this exactly ``probe^2`` in the limit. Note what the
    statistic is *insensitive* to: ``center`` (so the self-regulation ``d`` cannot
    affect it) and the overall scale of ``sigma`` and ``C`` (both absorbed into
    ``radius``). That insensitivity is the point — a failure to reproduce
    ``probe^2`` is diagnostic of the **draw**, not of the parameters.
    """
    return float(np.mean(np.abs(eigenvalues - center) < probe * radius))


def ellipse_fraction(
    eigenvalues: np.ndarray,
    *,
    radius: float,
    correlation: float,
    probe: float,
    center: float = 0.0,
) -> float:
    """Fraction inside the ``probe``-scaled ellipse, via the map to the unit disc.

    Dividing the real part by ``R(1 + rho)`` and the imaginary part by
    ``R(1 - rho)`` carries the elliptic law's support onto the unit disc, where the
    uniform density makes the enclosed fraction exactly ``probe^2`` again. Passing
    ``correlation = 0`` reduces this to :func:`disc_fraction` identically.
    """
    semi_major = radius * (1.0 + correlation)
    semi_minor = radius * (1.0 - correlation)
    if semi_major <= 0.0 or semi_minor <= 0.0:
        raise ValueError(
            f"correlation={correlation!r} collapses the ellipse to a segment "
            "(a semi-axis is zero), so the mapping to the unit disc is undefined; "
            "|correlation| must be < 1"
        )
    shifted = eigenvalues - center
    scaled = np.hypot(shifted.real / semi_major, shifted.imag / semi_minor)
    return float(np.mean(scaled < probe))


def binomial_se(p: float, n: int) -> float:
    """``sqrt(p (1-p) / n)`` — the SE of a fraction of ``n`` independent trials."""
    return float(np.sqrt(p * (1.0 - p) / n))


@dataclass(frozen=True)
class FractionReport:
    """One direct fraction probe, with its configuration guard.

    ``bias_estimate`` is ``bias_constant / n_species ** abs(bias_exponent)`` — the
    *caller's own measured* finite-``S`` bias law for this probe, never a default,
    because there is no single constant across probes. ``bias_ratio`` is
    ``bias_estimate / se``, and ``negligible`` is the guard: a configuration whose
    bias is not small against its own SE is refused rather than silently asserted
    against a tolerance it will eventually violate.
    """

    probe: float
    predicted: float
    measured: float
    n_eigenvalues: int
    se: float
    z: float
    z_max: float
    bias_estimate: float
    bias_ratio: float
    bias_ratio_max: float
    negligible: bool
    consistent: bool
    passed: bool

    def __str__(self) -> str:  # ASCII only (Windows cp1252 console)
        return (
            f"Fraction probe {'PASS' if self.passed else 'FAIL'}: "
            f"probe={self.probe:g}  measured={self.measured:.6f} "
            f"vs predicted={self.predicted:.6f}  "
            f"(n={self.n_eigenvalues}, SE={self.se:.3g}, z={self.z:+.2f}/{self.z_max:g})\n"
            f"  bias estimate {self.bias_estimate:.3g} = {self.bias_ratio:.2f} SE "
            f"(limit {self.bias_ratio_max:g}); negligible={self.negligible} "
            f"consistent={self.consistent}"
        )


def fraction_report(
    measured: float,
    *,
    probe: float,
    n_eigenvalues: int,
    n_species: int,
    bias_constant: float,
    bias_exponent: float = -1.0,
    z_max: float = 4.0,
    bias_ratio_max: float = 0.25,
) -> FractionReport:
    """Compare one measured fraction to ``probe^2``, refusing a bias-limited config.

    The tolerance is ``z_max`` binomial standard errors — statistical, never a typed
    epsilon. But a binomial SE alone is *not* a sufficient tolerance here, because
    adding eigenvalues shrinks it toward a bias that does not shrink at all. So the
    check has two legs and ``passed`` requires both:

    * ``consistent`` — the fraction sits within ``z_max`` SE of ``probe^2``;
    * ``negligible`` — the *configuration* is one in which that comparison is
      meaningful at all, i.e. the estimated finite-``S`` bias is at most
      ``bias_ratio_max`` of the SE.

    The second leg is what stops the caller buying spurious precision with draws.
    Because ``n_eigenvalues = n_species * n_draws``, raising the draw count lowers
    the SE while ``bias_estimate`` is fixed by ``n_species`` alone; precision at a
    fixed ``S`` is therefore bounded, and beyond that bound the only honest move is
    :func:`bias_scaling_report`. ``bias_constant`` and ``bias_exponent`` must come
    from a measurement of *this* probe — the constant varies by more than 2x across
    probes and the exponent between ``-0.62`` and ``-1.03``, so a shared default
    would be a fiction.
    """
    predicted = probe * probe
    se = binomial_se(predicted, n_eigenvalues)
    z = (measured - predicted) / se
    bias_estimate = bias_constant * n_species ** (-abs(bias_exponent))
    bias_ratio = bias_estimate / se
    negligible = bias_ratio <= bias_ratio_max
    consistent = abs(z) <= z_max
    return FractionReport(
        probe=probe,
        predicted=predicted,
        measured=measured,
        n_eigenvalues=n_eigenvalues,
        se=se,
        z=z,
        z_max=z_max,
        bias_estimate=bias_estimate,
        bias_ratio=bias_ratio,
        bias_ratio_max=bias_ratio_max,
        negligible=negligible,
        consistent=consistent,
        passed=consistent and negligible,
    )


def max_draws_for_negligible_bias(
    *,
    probe: float,
    n_species: int,
    bias_constant: float,
    bias_exponent: float = -1.0,
    bias_ratio_max: float = 0.25,
) -> int:
    """Largest draw count at which the finite-``S`` bias stays below the SE limit.

    Solves ``bias <= bias_ratio_max * sqrt(p(1-p) / (S * n_draws))`` for
    ``n_draws``. Use it to *derive* a configuration instead of picking one and
    hoping; a returned 0 means the probe is not assertable at this ``n_species`` by
    the direct check at all, which is the honest answer for ``rho_probe = 0.9`` and
    for the elliptic ``rho = +-0.8``.
    """
    p = probe * probe
    bias = bias_constant * n_species ** (-abs(bias_exponent))
    return int((bias_ratio_max**2) * p * (1.0 - p) / (bias**2 * n_species))


@dataclass(frozen=True)
class BiasScalingReport:
    """The asymptotic law: the finite-``S`` bias vanishes as a power of ``S``.

    ``exponent`` is the log-log slope of ``|bias|`` against ``n_species`` and is
    **reported, not asserted against a target** — no theory here predicts a value,
    and the measured exponents differ significantly between probes. What is
    asserted is ``significant`` (the exponent lies below ``-min_decay_rate`` by
    ``z`` standard errors) together with ``resolved``.

    **``significant`` demands an effect size, not merely a sign, and the teeth are
    why.** The first version of this check asserted only that the exponent was
    negative by ``z`` SE — and a *wrong radius* passed it. A constant offset fits a
    near-perfect straight line, so its exponent comes out tiny but with a tiny SE
    too: ``-0.0089 +/- 0.0015``, which clears zero at 6 sigma while the bias itself
    sits at ``0.33`` and does not move. Statistical significance is free when the
    residuals are small; the discriminating quantity is *how fast* the bias decays.
    """

    sizes: np.ndarray
    n_eigenvalues: np.ndarray
    bias: np.ndarray
    bias_se: np.ndarray
    bias_z: np.ndarray
    exponent: float
    exponent_se: float
    exponent_ci: tuple[float, float]
    z: float
    min_decay_rate: float
    resolved: bool
    significant: bool
    passed: bool

    def __str__(self) -> str:  # ASCII only (Windows cp1252 console)
        lines = [f"Bias scaling {'PASS' if self.passed else 'FAIL'}"]
        for s, n, b, se, bz in zip(
            self.sizes, self.n_eigenvalues, self.bias, self.bias_se, self.bias_z, strict=True
        ):
            lines.append(f"  S={s:>5d} n={n:>7d}  bias={b:>10.3e} +/- {se:.2g}  z={bz:>6.1f}")
        lines.append(
            f"  exponent = {self.exponent:.4f} +/- {self.exponent_se:.4f}; "
            f"CI[{self.z:g} SE] = [{self.exponent_ci[0]:.3f}, {self.exponent_ci[1]:.3f}]"
        )
        lines.append(
            f"  resolved={self.resolved}  "
            f"decays_faster_than_S^-{self.min_decay_rate:g}={self.significant}"
        )
        return "\n".join(lines)


def bias_scaling_report(
    sizes: Sequence[int],
    biases: Sequence[float],
    n_eigenvalues: Sequence[int],
    *,
    predicted: float,
    z: float = 3.0,
    min_decay_rate: float = 0.15,
) -> BiasScalingReport:
    """Fit ``|bias| ~ S^p`` and assert ``p`` decays at a rate a constant cannot fake.

    ``biases`` are the signed ``measured - predicted`` values at each ``n_species``,
    and ``n_eigenvalues`` the pool size behind each, which sets the binomial SE used
    for the ``resolved`` guard.

    ``resolved`` requires **every** fitted point to satisfy ``|bias| > z * SE``.
    This is not decoration. Reproducing the plan's own bias table showed its two
    largest-``S`` points sitting *below* their SE (``z = 0.69`` and ``0.90``), so the
    recorded ``-0.9279`` slope was partly a fit through noise; re-measured where
    every point resolves, the same probe gives ``-1.0086``. A point that cannot
    resolve what it measures must not be allowed to vote on the exponent.

    ``min_decay_rate`` is a **threshold placed in a measured gap**, and both of its
    margins are recorded rather than asserted by taste. Correct implementations
    measure exponents from ``-0.62 +- 0.067`` (circular ``probe = 0.9``, the
    shallowest of any probe) to ``-1.03 +- 0.023`` (circular ``probe = 0.5``); the
    three teeth measure about ``-0.01``. The default ``0.15`` therefore sits roughly
    90 SE above the teeth and 7 SE below the shallowest correct probe. It is the
    same species of choice as a convergence ``fit_mask`` window or Gray-Scott's
    asymptotic-grid bound: a claim about *which regime is being measured*, stated
    with the measurement that places it, not a tolerance absorbing an unknown error.

    The SE of the exponent is the ordinary-least-squares fit SE — the scatter of the
    points about the line — which needs at least four sizes to be defined at all.

    **The fitted range is part of the claim.** The exponent is not scale-free: the
    same circular ``probe = 0.9`` measures ``-0.6189 +- 0.0673`` over
    ``S = 25 ... 200`` but only ``-0.34 ... -0.50`` over ``S = 12 ... 96`` (three
    seeds). Shrinking the sizes to save compute — the cost of this check grows like
    ``S^3.2`` — silently changes the quantity being measured. Which sizes are inside
    the asymptotic regime is a measurement, never an assumption.
    """
    sizes_arr = np.asarray(list(sizes), dtype=int)
    bias_arr = np.asarray(list(biases), dtype=float)
    n_arr = np.asarray(list(n_eigenvalues), dtype=int)
    if sizes_arr.size < 4:
        raise ValueError(
            f"need at least 4 sizes to fit an exponent with a standard error, got "
            f"{sizes_arr.size}; a two-point slope was exactly how the S^-0.67 reading "
            "arose, and it carried a +-0.3 ratio uncertainty that could not "
            "distinguish -0.67 from -1.0"
        )
    if np.any(bias_arr == 0.0):
        raise ValueError("a bias of exactly zero has no logarithm; the fit is undefined")

    se_arr = np.array([binomial_se(predicted, int(n)) for n in n_arr], dtype=float)
    bias_z = bias_arr / se_arr
    resolved = bool(np.all(np.abs(bias_z) > z))

    coeffs, cov = np.polyfit(np.log(sizes_arr), np.log(np.abs(bias_arr)), 1, cov=True)
    exponent = float(coeffs[0])
    exponent_se = float(np.sqrt(cov[0, 0]))
    significant = exponent + z * exponent_se < -abs(min_decay_rate)

    return BiasScalingReport(
        sizes=sizes_arr,
        n_eigenvalues=n_arr,
        bias=bias_arr,
        bias_se=se_arr,
        bias_z=bias_z,
        exponent=exponent,
        exponent_se=exponent_se,
        exponent_ci=(exponent - z * exponent_se, exponent + z * exponent_se),
        z=z,
        min_decay_rate=abs(min_decay_rate),
        resolved=resolved,
        significant=significant,
        passed=resolved and significant,
    )
