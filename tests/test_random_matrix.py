"""The May / Allesina-Tang random community matrix — circular and elliptic laws.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` step 3b. This is a
**plain pytest outside the ValidationSuite path**: a matrix ensemble has no
``step``, no ``observables`` and no ``state.t``, so there is no model to register
and nothing for ``validate()`` to consume. See the module docstring of
``sandbox.core.random_matrix`` for why that is a deliberate boundary rather than an
omission.

**What is anchored to what.**

* **Category A, exact.** The circular law's enclosed fraction is exactly
  ``probe^2``, with a *binomial* SE from ``S`` eigenvalues per draw. Asserted at
  the one configuration where the finite-``S`` bias is genuinely negligible, and
  that configuration is **derived** by
  :func:`~sandbox.core.random_matrix.max_draws_for_negligible_bias`, not chosen.
* **Category B.** The finite-``S`` bias vanishes as a power of ``S``, asserted on
  the elliptic ``rho = +0.8`` probe. The exponent is *reported*; what is asserted
  is that it decays faster than ``S^-0.15``, a rate no constant-offset bug reaches.

**Why the asymptotic law is asserted on the elliptic probe and not a circular
one.** The two checks want opposite regimes — the direct probe needs
``bias << SE``, the scaling check needs ``bias >> SE`` — and the elliptic
``rho = +0.8`` bias is 4-5x any circular one. That makes it *unassertable* by the
direct check at any affordable ``S`` and *cheap* by the scaling check, while every
circular probe is the other way round. The circular scaling exponents were
measured (``-0.88 / -1.03 / -0.62`` at ``probe = 0.2 / 0.5 / 0.9``) but cost ~28 s
each to resolve, so they are reported in the demo rather than asserted here. The
cost of this check grows like ``S^3.2``, and it cannot be dodged by shrinking the
sizes: the same probe measures a *different* exponent over ``S = 12 ... 96``.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.core.random_matrix import (
    BiasScalingReport,
    bias_scaling_report,
    binomial_se,
    bulk_radius,
    disc_fraction,
    draw_community_matrix,
    ellipse_fraction,
    ensemble_eigenvalues,
    fraction_report,
    max_draws_for_negligible_bias,
)

# Measured finite-S bias law for probe = 0.5, from a four-size fit at 200 000
# eigenvalues where every point resolves (z = 27.5 / 13.9 / 7.6 / 3.3):
# exponent -1.0274 +/- 0.0231, c = bias * S flat at 0.692/0.722/0.696/0.658.
# The recorded slice value was 0.6/S with slope -0.9279; that fit included two
# points sitting below their own SE. These constants belong to THIS probe only.
BIAS_CONSTANT_PROBE_HALF = 0.70
BIAS_EXPONENT_PROBE_HALF = -1.0

# The sizes over which the exponents in the module docstring were measured. The
# range is part of the claim: see test_the_exponent_is_not_scale_free below.
SCALING_SIZES = (25, 50, 100, 200)


# --------------------------------------------------------------------------- draw


def test_bulk_radius_is_sigma_sqrt_sc() -> None:
    assert bulk_radius(400, 1.0, 1.0) == pytest.approx(20.0)
    assert bulk_radius(400, 0.25, 2.0) == pytest.approx(20.0)
    assert bulk_radius(100, 1.0, 1.0) == pytest.approx(10.0)


def test_draw_has_an_exactly_zero_diagonal() -> None:
    """May's ``B`` carries no self-interaction; it lives in the ``-d I`` term.

    Exact zero, not "small": the constraint ``trace(B) = 0`` is what makes the
    eigenvalue sum vanish identically, and it is measurably the dominant source of
    the finite-``S`` bias near the spectral edge (filling the diagonal
    Ginibre-style drops ``bias * S`` at ``probe = 0.9`` from 1.06 to 0.35).
    """
    matrix = draw_community_matrix(np.random.default_rng(0), n_species=40)
    assert np.all(np.diag(matrix) == 0.0)

    shifted = draw_community_matrix(np.random.default_rng(0), n_species=40, self_regulation=1.5)
    assert np.all(np.diag(shifted) == -1.5)
    # Only the diagonal differs: self-regulation is a pure translation.
    off = ~np.eye(40, dtype=bool)
    assert np.array_equal(matrix[off], shifted[off])


@pytest.mark.parametrize("correlation", [-0.8, -0.4, 0.0, 0.4, 0.8])
def test_pairs_carry_the_requested_correlation(correlation: float) -> None:
    """``(B_ij, B_ji)`` must be correlated at ``rho`` — the elliptic law's whole input."""
    rng = np.random.default_rng(1)
    upper_idx = np.triu_indices(50, k=1)
    pairs = []
    for _ in range(20):
        matrix = draw_community_matrix(rng, n_species=50, correlation=correlation)
        pairs.append(np.stack([matrix[upper_idx], matrix[(upper_idx[1], upper_idx[0])]], axis=1))
    sample = np.concatenate(pairs)
    measured = float(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])
    # SE of a correlation is about (1 - rho^2)/sqrt(n); 4 SE with n = 24 500 pairs.
    se = (1.0 - correlation**2) / np.sqrt(sample.shape[0])
    assert abs(measured - correlation) <= 4.0 * max(se, 1e-3)
    assert np.var(sample[:, 0]) == pytest.approx(1.0, rel=0.05)


def test_connectance_removes_both_members_of_a_pair_together() -> None:
    """A pair must be absent as a unit, or the correlation it carries is destroyed."""
    rng = np.random.default_rng(2)
    upper_idx = np.triu_indices(60, k=1)
    matrix = draw_community_matrix(rng, n_species=60, connectance=0.3, correlation=0.8)
    upper = matrix[upper_idx]
    lower = matrix[(upper_idx[1], upper_idx[0])]
    assert np.array_equal(upper == 0.0, lower == 0.0)
    fraction_present = float(np.mean(upper != 0.0))
    se = np.sqrt(0.3 * 0.7 / upper.size)
    assert abs(fraction_present - 0.3) <= 4.0 * se


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"correlation": 1.5}, "correlation"),
        ({"correlation": -1.5}, "correlation"),
        ({"connectance": 1.5}, "connectance"),
        ({"connectance": -0.1}, "connectance"),
    ],
)
def test_draw_rejects_out_of_range_parameters(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        draw_community_matrix(np.random.default_rng(0), n_species=10, **kwargs)


# ----------------------------------------------------------------------- fractions


def test_ellipse_fraction_reduces_to_disc_fraction_at_zero_correlation() -> None:
    """At ``rho = 0`` the two statistics are the same statistic, not merely close."""
    rng = np.random.default_rng(3)
    eigs = ensemble_eigenvalues(rng, n_draws=5, n_species=80)
    radius = bulk_radius(80)
    for probe in (0.2, 0.5, 0.9):
        assert ellipse_fraction(eigs, radius=radius, correlation=0.0, probe=probe) == disc_fraction(
            eigs, radius=radius, probe=probe
        )


def test_disc_fraction_ignores_the_self_regulation_shift() -> None:
    """The statistic must see the draw, not the parameters — that is what makes it diagnostic."""
    eigs_b = ensemble_eigenvalues(np.random.default_rng(4), n_draws=4, n_species=60)
    eigs_m = ensemble_eigenvalues(
        np.random.default_rng(4), n_draws=4, n_species=60, self_regulation=2.5
    )
    radius = bulk_radius(60)
    assert disc_fraction(eigs_m, radius=radius, probe=0.6, center=-2.5) == disc_fraction(
        eigs_b, radius=radius, probe=0.6
    )


def test_ellipse_fraction_refuses_a_degenerate_ellipse() -> None:
    """At ``|rho| = 1`` a semi-axis is zero and the map to the unit disc is undefined."""
    eigs = ensemble_eigenvalues(np.random.default_rng(5), n_draws=2, n_species=20)
    with pytest.raises(ValueError, match="collapses the ellipse"):
        ellipse_fraction(eigs, radius=bulk_radius(20), correlation=1.0, probe=0.5)


# ------------------------------------------------------------------ config guards


def test_fraction_report_refuses_a_bias_limited_configuration() -> None:
    """Buying precision with draws must not silently buy a failing test.

    Same measured fraction, same ``S``, two draw counts. The larger pool has the
    smaller SE and therefore the *worse* bias ratio — a correct implementation
    checked that way would eventually fail, so the report refuses it.
    """
    kwargs = {
        "probe": 0.5,
        "n_species": 400,
        "bias_constant": BIAS_CONSTANT_PROBE_HALF,
        "bias_exponent": BIAS_EXPONENT_PROBE_HALF,
    }
    ok = fraction_report(0.2515, n_eigenvalues=400 * 9, **kwargs)
    assert ok.negligible and ok.passed

    greedy = fraction_report(0.2515, n_eigenvalues=400 * 200, **kwargs)
    assert greedy.consistent, "the fraction itself is still fine"
    assert not greedy.negligible, "but the configuration is bias-limited"
    assert not greedy.passed
    assert greedy.bias_ratio > ok.bias_ratio


def test_max_draws_says_zero_where_the_probe_is_not_assertable() -> None:
    """The honest answer for the elliptic probe is 'you cannot do this check'."""
    # probe 0.5 at S = 400: the one satisfiable circular configuration.
    assert (
        max_draws_for_negligible_bias(
            probe=0.5,
            n_species=400,
            bias_constant=BIAS_CONSTANT_PROBE_HALF,
            bias_exponent=BIAS_EXPONENT_PROBE_HALF,
        )
        == 9
    )

    # The elliptic rho = +0.8 probe: bias * S ~ 3.8 and a shallower exponent. Zero
    # draws, i.e. not assertable by the direct check at all -- which is exactly the
    # probe the scaling check handles most cheaply.
    assert (
        max_draws_for_negligible_bias(
            probe=0.5, n_species=400, bias_constant=3.8, bias_exponent=-0.84
        )
        == 0
    )


def test_bias_scaling_report_requires_a_real_lever_arm() -> None:
    with pytest.raises(ValueError, match="at least 4 sizes"):
        bias_scaling_report([50, 100, 200], [1e-2, 5e-3, 2e-3], [4000] * 3, predicted=0.25)


def test_bias_scaling_report_refuses_points_that_cannot_resolve_their_bias() -> None:
    """The slice's own failure, reproduced as a unit test.

    Holding the eigenvalue count at 40 000, the recorded bias table's two
    largest-``S`` points sat below their own SE (``z = 0.69`` and ``0.90``) — so the
    ``-0.9279`` slope was partly fitted through noise. A fit is only allowed to
    consume points that resolve what they measure.
    """
    sizes = [50, 100, 200, 400]
    unresolved = [1.227e-2, 5.875e-3, 3.025e-3, 1.425e-3]
    report = bias_scaling_report(sizes, unresolved, [40_000] * 4, predicted=0.25)
    assert not report.resolved, "the S = 400 point is 0.66 SE — it resolves nothing"
    assert not report.passed
    # ...and the same numbers, measured until every point resolves, pass.
    resolved = bias_scaling_report(
        sizes, unresolved, [40_000, 200_000, 800_000, 4_000_000], predicted=0.25
    )
    assert resolved.resolved and resolved.passed


def test_significance_alone_would_have_passed_a_constant_offset() -> None:
    """Why ``significant`` demands a decay *rate* and not merely a negative sign.

    These are the measured biases of a Hermitian-ized draw — a spectrum collapsed
    onto the real line, about as wrong as the ensemble can be. The discrepancy sits
    at ``0.34`` and *does not move*: it shrinks 2% across an 8x range in ``S``. But
    a constant offset fits a near-perfect straight line, so the fitted exponent has
    a tiny SE and clears **zero** at 6 sigma. An earlier version of this check
    asserted exactly that, and this data passed it.
    """
    sizes = [25, 50, 100, 200]
    hermitian_bias = [0.3393, 0.3387, 0.3360, 0.3333]
    counts = [750, 1500, 3000, 6000]
    report = bias_scaling_report(sizes, hermitian_bias, counts, predicted=0.25)

    assert report.exponent + report.z * report.exponent_se < 0.0, (
        "a constant offset really is 'significantly negative' — that is the trap"
    )
    assert not report.significant, "but it does not decay, and that is what is asserted"
    assert not report.passed


# ----------------------------------------------------------- the direct probe (A)


def test_circular_law_fraction_at_the_one_bias_negligible_configuration() -> None:
    """``S = 400``, 9 draws, ``probe = 0.5`` — a *derived* configuration.

    The draw count is not chosen: it is
    :func:`max_draws_for_negligible_bias` for this probe and this ``S``, which is
    where the measured bias ``0.70/S`` sits at a quarter of the binomial SE. It is
    also the only affordable one — the cost of ``eigvals`` jumps 13x between
    ``S = 400`` and ``S = 600`` for 3.4x the FLOPs.
    """
    n_draws = max_draws_for_negligible_bias(
        probe=0.5,
        n_species=400,
        bias_constant=BIAS_CONSTANT_PROBE_HALF,
        bias_exponent=BIAS_EXPONENT_PROBE_HALF,
    )
    assert n_draws >= 5, "config derivation collapsed; the probe would be untestable"

    rng = np.random.default_rng(0)
    eigs = ensemble_eigenvalues(rng, n_draws=n_draws, n_species=400)
    measured = disc_fraction(eigs, radius=bulk_radius(400), probe=0.5)
    report = fraction_report(
        measured,
        probe=0.5,
        n_eigenvalues=eigs.size,
        n_species=400,
        bias_constant=BIAS_CONSTANT_PROBE_HALF,
        bias_exponent=BIAS_EXPONENT_PROBE_HALF,
    )
    print(report)
    assert report.passed, str(report)


def test_tooth_a_wrong_radius_moves_the_direct_probe_many_sigma() -> None:
    """The wrong-``R`` tooth, and the *only* check of the three that it can bite.

    It was tried on the scaling check first and does **not** bite there: a
    misscaled radius leaves an offset that *itself* decays, measured at
    ``S^-0.34 ... S^-0.42`` across four seeds, which no threshold can separate from
    the shallowest correct probe (``-0.62``). That is the repressilator's
    ``Omega^2`` tooth all over again — a tooth must bite the *right* check, and
    reusing one because it is available produces a test that is green for a reason
    unrelated to the claim.

    On the direct probe it bites hard, because there the statistic is the fraction
    itself rather than its trend.
    """
    rng = np.random.default_rng(0)
    eigs = ensemble_eigenvalues(rng, n_draws=9, n_species=400)
    honest = disc_fraction(eigs, radius=bulk_radius(400), probe=0.5)
    wrong = disc_fraction(eigs, radius=1.1 * bulk_radius(400), probe=0.5)
    se = binomial_se(0.25, eigs.size)
    assert abs(wrong - 0.25) / se > 5.0
    assert abs(honest - 0.25) / se < 4.0


# ------------------------------------------------------- the asymptotic law (B)


def _elliptic_biases(
    seed: int,
    draws: tuple[int, ...],
    *,
    sizes: tuple[int, ...] = SCALING_SIZES,
    correlation: float = 0.8,
    probe: float = 0.5,
    map_correlation: float | None = None,
    hermitian: bool = False,
) -> tuple[list[float], list[int]]:
    """Elliptic biases per size, with hooks for the two teeth (flipped sign, Hermitian)."""
    biases, counts = [], []
    for n_species, n_draws in zip(sizes, draws, strict=True):
        rng = np.random.default_rng(seed)
        matrices = [
            draw_community_matrix(rng, n_species=n_species, correlation=correlation)
            for _ in range(n_draws)
        ]
        if hermitian:
            matrices = [0.5 * (m + m.T) for m in matrices]
        eigs = np.concatenate([np.linalg.eigvals(m) for m in matrices])
        frac = ellipse_fraction(
            eigs,
            radius=bulk_radius(n_species),
            correlation=correlation if map_correlation is None else map_correlation,
            probe=probe,
        )
        biases.append(frac - probe * probe)
        counts.append(eigs.size)
    return biases, counts


def test_elliptic_law_bias_vanishes_with_system_size() -> None:
    """The headline: the finite-``S`` bias decays, at the probe the direct check cannot touch.

    At ``rho = +0.8`` the bias is 4-5x any circular one, which is precisely what
    makes it unassertable by a bias-negligible rule and cheap by a scaling one. The
    draw counts are priced from the measured bias so every point clears its own SE
    by about 5x, comfortably above the ``resolved`` guard at 3. Measured exponent
    across seeds 0-3: ``-0.838 / -0.820 / -0.887 / -0.805``.
    """
    biases, counts = _elliptic_biases(0, (25, 30, 40, 60))
    report = bias_scaling_report(SCALING_SIZES, biases, counts, predicted=0.25)
    print(report)
    assert report.resolved, str(report)
    assert report.significant, str(report)
    assert report.passed


# ------------------------------------------------------------------------- teeth

TOOTH_DRAWS = (20, 20, 20, 25)


def _assert_tooth_bites(report: BiasScalingReport) -> None:
    """A tooth must bite on the *decay* leg, having produced a real discrepancy.

    Asserting only ``not passed`` would accept a tooth that failed because its bias
    was too noisy to resolve — a check failing for a reason unrelated to the claim.
    Requiring ``resolved`` first makes the tooth prove the stronger thing: there is
    a large, well-measured discrepancy, and it does not shrink with ``S``.
    """
    assert report.resolved, f"tooth failed for the wrong reason (unresolved):\n{report}"
    assert not report.significant, f"tooth did not bite:\n{report}"
    assert not report.passed


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tooth_a_flipped_correlation_sign_does_not_shrink(seed: int) -> None:
    """Swapping the semi-axes. Asserted only where the ellipse is eccentric.

    At ``rho = 0`` this tooth has **no leg at all** — flipping the sign of zero
    swaps nothing and the ellipse is a circle — so it is meaningless there and is
    asserted at ``|rho| = 0.8``, where the semi-axes differ by 9x. Measured
    exponents ``+0.008 / +0.020 / +0.023 / +0.025``: the discrepancy does not merely
    fail to shrink, it grows.
    """
    biases, counts = _elliptic_biases(seed, TOOTH_DRAWS, map_correlation=-0.8)
    _assert_tooth_bites(bias_scaling_report(SCALING_SIZES, biases, counts, predicted=0.25))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tooth_a_hermitian_draw_does_not_shrink(seed: int) -> None:
    """Symmetrising collapses the spectrum onto the real line — semicircle, not ellipse.

    Measured exponents ``-0.025 / -0.016 / -0.023 / -0.009`` on a discrepancy of
    ``0.34``: flat to within 2% across an 8x range in ``S``.
    """
    biases, counts = _elliptic_biases(seed, TOOTH_DRAWS, hermitian=True)
    _assert_tooth_bites(bias_scaling_report(SCALING_SIZES, biases, counts, predicted=0.25))


def test_the_flipped_sign_tooth_is_correctly_toothless_at_zero_correlation() -> None:
    """Stating the tooth's own blind spot, rather than letting it hide.

    This is the mirror of the ``linspace`` lesson: a check that happens to be
    vacuous in one configuration is a hazard only if nobody has written down which
    configuration. At ``rho = 0`` the flipped-sign mapping is bit-identical.
    """
    eigs = ensemble_eigenvalues(np.random.default_rng(7), n_draws=3, n_species=40)
    radius = bulk_radius(40)
    assert ellipse_fraction(eigs, radius=radius, correlation=0.0, probe=0.5) == ellipse_fraction(
        eigs, radius=radius, correlation=-0.0, probe=0.5
    )
