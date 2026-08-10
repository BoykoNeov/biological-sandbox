"""Stochastic gLV: the network, the O(1/Omega) propensity bias, and Kurtz's law.

Written against ``docs/plans/phase3-{plan,context,tasks}.md`` step 3c.

**What is anchored to what.**

* **Independent.** :func:`textbook_rhs` is the same hand-written double loop 3a
  uses — transcribed from ``dx_i/dt = x_i (r_i + sum_j A_ij x_j)``, sharing no
  code with the model — checked against the *network's* ``deterministic_rhs``.
  Since ``deterministic_rhs`` is ``stoich.T @ rates(c)`` over the very ``rates``
  the SSA fires, agreeing with the loop pins the stoichiometry, the ``A``
  convention **and** which species each loss reaction removes. That last one is
  why the reference system is ``glv``'s asymmetric 3-species matrix: with a
  symmetric ``A``, a network that removes ``X_j`` instead of ``X_i`` has a
  bit-identical ODE limit and nothing here would see it.
* **Category B, the headline.** ``D(Omega) ~ Omega^{-1/2}``, via
  :func:`~sandbox.core.convergence.convergence_report`, with two teeth.
* **The phase's open item, closed.** The macroscopic-propensity bias, measured
  with a *split-coupled* estimator — see the section below.

**The convergence config, and why each number is what it is.** ``omegas =
[50, 100, 200, 400, 800]``, ``t_max = 20``, ``R = 8``, ``z = 3``, ~25 s.

* The window was *slid down*, not narrowed. Cost is dominated by the largest
  ``Omega`` while the OLS slope SE goes as ``1/(sqrt(K) sd(log Omega))``, so
  ``50..800`` buys the same 16x lever arm as ``100..1600`` for half the bill.
  Measured across seeds 0-3: ``100..1600`` 49-57 s, worst ``|slope+1/2| / (z SE)``
  = 0.47; ``100..800`` 27-36 s, worst 0.67; **``50..800`` 20-29 s, worst 0.37**,
  with the tightest slopes of the three (``-0.4952 / -0.5134 / -0.4960 /
  -0.5228``, against ``-0.4635 ... -0.5370`` for the widest window).
* ``x_init`` sits *away* from ``x*``. Starting at the equilibrium makes the ODE
  reference exactly flat, and then ``richardson_delta`` is ~0 and ``reference_ok``
  becomes a threshold nothing can fail. Off equilibrium it measures a real
  ``1.18e-8`` against a smallest ``D`` of ``0.0295`` — passing by six orders,
  but *passing on evidence*.
* No ``fit_mask``: ``D*sqrt(Omega)`` is flat across the whole sweep
  (``0.80 / 0.84 / 0.88 / 0.80 / 0.83`` at seed 0), so there is no
  phase-saturation knee to exclude, and ``monotone`` holds at all four seeds.
* ``Omega = 50`` is the smallest size that is *safe*: 10 molecules per species at
  ``t = 0`` and 0/24 replicates losing a species. Extinction is absorbing, so one
  loss would inflate that replicate's ``D`` permanently — and it would do so
  *silently*, because ``convergence_report`` only guards against ``max_steps``,
  not against a run that terminated early at the absorbing state.
  :func:`test_no_replicate_loses_a_species` asserts it directly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sandbox.models  # noqa: F401  (registers models)
from sandbox.core.convergence import convergence_report
from sandbox.core.protocol import Experiment
from sandbox.core.registry import register
from sandbox.core.sweep import run_experiment
from sandbox.models.glv import equilibrium, glv_rhs
from sandbox.models.glv_stochastic import (
    MODEL,
    GLVStochastic,
    GLVStochasticParams,
    observable_keys,
    propensity_bias,
    replicates_until_bias_dominates,
)

# The 3a reference system. Asymmetric on purpose (see the module docstring).
R3 = [1.0, 0.8, 1.2]
A3 = [[-1.0, -0.3, -0.2], [-0.4, -1.0, -0.1], [-0.2, -0.5, -1.0]]
KEYS = observable_keys(3)
BASE = {"r": R3, "A": A3, "x_init": [0.2, 0.2, 0.2]}

_OMEGAS = [50.0, 100.0, 200.0, 400.0, 800.0]
_T_MAX = 20.0
_DT = 1e-3
_REPLICATES = 8
_Z = 3.0
_N_GRID = 200


def factory(d: dict) -> GLVStochasticParams:
    return GLVStochasticParams(**d)


def reference(**overrides) -> GLVStochasticParams:
    return GLVStochasticParams(**{**BASE, "Omega": 100.0, "t_max": _T_MAX, **overrides})


def textbook_rhs(x: np.ndarray, p: GLVStochasticParams) -> np.ndarray:
    """``dx_i/dt = x_i (r_i + sum_j A_ij x_j)``, hand-written as a double loop.

    Deliberately shares no code with the model — no ``@``, no ``outer``, no
    broadcasting — so a transposed ``A``, a loss reaction charged to the wrong
    species, or a missing ``x_i`` prefactor shows up as a disagreement.
    """
    n = len(p.r)
    out = np.empty(n, dtype=float)
    for i in range(n):
        total = p.r[i]
        for j in range(n):
            total += p.A[i][j] * x[j]
        out[i] = x[i] * total
    return out


# --------------------------------------------------------------------------
# The network: independent anchor, and the conventions it pins
# --------------------------------------------------------------------------


def test_deterministic_limit_matches_the_hand_written_loop():
    """``stoich.T @ rates(c)`` must be the gLV RHS, at random points.

    This is the load-bearing structural test. ``deterministic_rhs`` is assembled
    from the *same* ``rates`` the SSA fires, so pinning it to the transcribed
    equation pins the stochastic side too — the "one ``rates(c)`` drives both
    worlds" design is what lets a single check cover both.
    """
    p = reference()
    rhs = MODEL.deterministic_rhs(p)
    rng = np.random.default_rng(11)
    for _ in range(25):
        x = rng.uniform(0.0, 3.0, size=3)
        assert np.allclose(rhs(x), textbook_rhs(x, p), rtol=1e-13, atol=0.0)


def test_the_deterministic_limit_is_the_3a_vector_field():
    """The stochastic model's ODE limit *is* ``glv``'s, not a second copy of it.

    Structural rather than numerical: if the two ever drift apart, the phase's
    "same vector field, two worlds" claim is false and 3a's deterministic
    validation stops underwriting anything here.
    """
    p = reference()
    stochastic = MODEL.deterministic_rhs(p)
    deterministic = glv_rhs(p.deterministic_params())
    rng = np.random.default_rng(3)
    for _ in range(25):
        x = rng.uniform(0.0, 3.0, size=3)
        assert np.allclose(stochastic(x), deterministic(x), rtol=1e-13, atol=0.0)


def test_a_is_indexed_as_the_effect_of_j_on_i():
    # THE transpose detector, and simultaneously the "which species does the loss
    # reaction remove" detector: A[0][1] = -0.3 and A[1][0] = -0.4 differ, so the
    # two-species-present state pins both. A transposed A gives -0.4 in dx[0]; a
    # network that removes X_j instead of X_i swaps the two cross terms.
    p = reference()
    dx = MODEL.deterministic_rhs(p)(np.array([1.0, 1.0, 0.0]))
    assert dx[0] == pytest.approx(1.0 * (1.0 - 1.0 - 0.3), rel=1e-14)
    assert dx[1] == pytest.approx(1.0 * (0.8 - 0.4 - 1.0), rel=1e-14)


def test_a_symmetric_matrix_could_not_have_caught_either():
    """State the blind spot the reference matrix exists to cover (the 3b habit).

    Under ``A = A^T`` a transposed convention and a loss reaction that removes
    ``X_j`` instead of ``X_i`` both leave the ODE limit *bit-identical*. Asserting
    that here keeps the asymmetry of ``A3`` from looking like an arbitrary choice
    someone could later "simplify" away.
    """
    symmetric = [[-1.0, -0.3, -0.2], [-0.3, -1.0, -0.5], [-0.2, -0.5, -1.0]]
    p = reference(A=symmetric)
    transposed = replace(p, A=tuple(tuple(row) for row in np.transpose(symmetric).tolist()))
    x = np.array([0.3, 0.7, 1.1])
    assert np.array_equal(MODEL.deterministic_rhs(p)(x), MODEL.deterministic_rhs(transposed)(x)), (
        "a symmetric A is supposed to be blind to transposition -- the premise failed"
    )
    # ...and the asymmetric reference is not.
    q = reference()
    q_t = replace(q, A=tuple(tuple(row) for row in np.transpose(A3).tolist()))
    assert not np.allclose(MODEL.deterministic_rhs(q)(x), MODEL.deterministic_rhs(q_t)(x))


def test_an_absent_species_cannot_appear():
    # Every reaction is proportional to the count of the species it changes, so a
    # zero count is absorbing for that species. is_terminal's cheap absorbing
    # check (counts.sum() <= 0) is only exact because of this.
    p = reference()
    dx = MODEL.deterministic_rhs(p)(np.array([0.0, 0.5, 0.5]))
    assert dx[0] == 0.0, "an absent species grew from nothing"


def test_observable_keys_match_initial_concentrations_order():
    """``observable_keys`` must line up column-for-column with the ODE vector.

    ``convergence_report`` pairs ``observable_keys[s]`` with column ``s`` of the
    integrated ODE; a disagreement compares species against the *wrong* species
    and yields a wrong-but-plausible ``D`` whose slope could still look fine.
    Distinct per-component initial values make the check real.
    """
    p = reference(x_init=[0.1, 0.3, 0.7], Omega=1000.0)
    observed = MODEL.observables(MODEL.initial_state(p, np.random.default_rng(0)))
    assert tuple(observed)[:3] == KEYS
    got = np.array([observed[k] for k in KEYS])
    assert np.allclose(got, MODEL.initial_concentrations(p), atol=1e-9)


def test_extra_observables_are_not_ode_components():
    # total_biomass and n_survivors are diagnostics. They must NOT be in
    # observable_keys, or convergence_report would compare them against a species
    # column -- and n_survivors is O(S) where concentrations are O(1), so the
    # equally-weighted L1 discrepancy would be swamped (the Phase-2 V-vs-gates
    # lesson).
    p = reference()
    keys = tuple(MODEL.observables(MODEL.initial_state(p, np.random.default_rng(0))))
    assert set(keys) - set(KEYS) == {"total_biomass", "n_survivors"}
    assert len(KEYS) == MODEL.initial_concentrations(p).size


def test_n_survivors_counts_nonzero_species_exactly():
    """The extinction threshold is one molecule — exact, not chosen.

    ``glv.py`` defers ``n_survivors`` to this model precisely because ``1/Omega``
    is the scale that sets the threshold. A species at count zero is absorbing
    (its birth propensity ``r_i n_i`` vanishes and nothing produces it), so
    "survivor" needs no tolerance at all.
    """
    p = reference()
    state = MODEL.initial_state(p, np.random.default_rng(0))
    dead_one = replace(state, counts=np.array([0, 5, 7], dtype=np.int64))
    assert MODEL.observables(dead_one)["n_survivors"] == 2.0
    dead_all = replace(state, counts=np.array([0, 0, 0], dtype=np.int64))
    assert MODEL.observables(dead_all)["n_survivors"] == 0.0
    assert MODEL.is_terminal(dead_all), "the all-extinct state must be terminal"
    assert not MODEL.is_terminal(dead_one), "one extinction is not the absorbing state"


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


def test_a_positive_interaction_is_refused():
    # A_ij > 0 is a benefit of j on i, which mass action writes as a different
    # reaction. Building it from |A_ij| would flip the sign silently.
    with pytest.raises(ValueError, match="purely competitive"):
        reference(A=[[-1.0, +0.3, -0.2], [-0.4, -1.0, -0.1], [-0.2, -0.5, -1.0]])


def test_missing_self_limitation_is_refused():
    with pytest.raises(ValueError, match="self-limitation"):
        reference(A=[[0.0, -0.3, -0.2], [-0.4, -1.0, -0.1], [-0.2, -0.5, -1.0]])


def test_a_non_positive_growth_rate_is_refused():
    with pytest.raises(ValueError, match="growth rate must be positive"):
        reference(r=[1.0, 0.0, 1.2])


def test_the_all_zero_initial_condition_is_refused():
    with pytest.raises(ValueError, match="absorbing state"):
        reference(x_init=[0.0, 0.0, 0.0])


def test_shape_mismatches_are_refused():
    with pytest.raises(ValueError, match="A must be"):
        reference(A=[[-1.0, -0.3], [-0.4, -1.0]])
    with pytest.raises(ValueError, match="x_init must have"):
        reference(x_init=[0.2, 0.2])


def test_no_analytic_predictions():
    """Deliberate: ``<x>`` is not ``x*``, and a check against ``x*`` passes *today*.

    The stationary mean carries an ``O(1/Omega)`` van Kampen correction plus the
    propensity bias below, while ``validate()``'s tolerance shrinks as
    ``1/sqrt(R)``. So the tempting closed form is green at small ``R`` and red at
    large ``R`` — the same bias-limited configuration 3b's ``fraction_report``
    refuses. Asserting the absence keeps someone from "fixing" the omission.
    """
    assert not hasattr(MODEL, "analytic_predictions")


def test_the_tempting_closed_form_would_expire_at_a_measurable_replicate_count():
    """The artifact behind that refusal, rather than prose about it.

    Measured across-replicate SDs of ``x0(t_max)`` at ``R = 24`` were ``0.10502 /
    0.04512 / 0.02886`` at ``Omega = 100 / 400 / 1600``, putting the crossing at
    ``R = 229 / 677 / 4429``. Those are *reachable* replicate counts — 229 is
    below what several checks in this suite already use — so "``<x> = x*`` is a
    fine analytic prediction" is false at this model's own working sizes, not
    merely false in principle.
    """
    crossings = {}
    for omega, sd in ((100.0, 0.105018), (400.0, 0.045120), (1600.0, 0.028856)):
        crossings[omega] = replicates_until_bias_dominates(reference(Omega=omega), sd)
    assert crossings[100.0] == pytest.approx(229.1, rel=1e-3)
    assert crossings[400.0] == pytest.approx(676.7, rel=1e-3)
    assert crossings[1600.0] == pytest.approx(4428.7, rel=1e-3)
    # It grows with Omega (the bias shrinks faster than the noise) but never out
    # of reach: even at Omega = 1600 a 4429-replicate check would go red.
    assert crossings[100.0] < crossings[400.0] < crossings[1600.0]


# --------------------------------------------------------------------------
# The O(1/Omega) macroscopic-propensity bias — the phase's open item
# --------------------------------------------------------------------------
#
# The engine uses a_j = Omega f_j(n/Omega), giving |A_ii| n^2 / Omega where the
# microscopically exact self-limitation propensity is |A_ii| n(n-1) / Omega (a
# molecule cannot react with itself). The claim to check is that the resulting
# bias is O(1/Omega) -- subdominant to the O(Omega^{-1/2}) signal, hence unable
# to floor the slope above -- and that it equals propensity_bias() in magnitude.
#
# With INDEPENDENT arms this is not affordable: the planning slice got one
# informative row out of three, because the SSA's own fluctuations grow faster
# than the bias as Omega falls ("the bias cannot be measured where it is
# largest"), and closing it that way needs ~340 s of SSA.
#
# The estimator below is a SPLIT COUPLING (Anderson 2012's coupled finite
# difference). The macro arm is the exact arm plus S extra loss channels, since
# macro's |A_ii| n^2/Omega exceeds exact's |A_ii| n(n-1)/Omega by exactly
# |A_ii| n/Omega. So the pair is simulated as ONE chain: each reaction splits
# into a shared channel (rate min(aE, aM), fires in both arms), an E-only
# channel and an M-only channel. Two consequences, both measured below: the arms
# stay within a few molecules of each other so the difference is far quieter
# than either arm, and -- because the shared rate is nearly the whole rate -- the
# coupled chain costs about ONE arm's events rather than two.

_S = 3
_ABS_A = np.abs(np.asarray(A3, dtype=float))
_R_ARR = np.asarray(R3, dtype=float)
_N_REACTIONS = _S + _S * _S
# Loss reaction at flat index i*S + j decrements species i.
_LOSS_SPECIES = np.repeat(np.arange(_S), _S)


def _propensities(n: np.ndarray, omega: float, exact: bool, out: np.ndarray) -> float:
    """Propensities of the 3-species reference network, written independently.

    A stand-alone transcription — like :func:`textbook_rhs`, it shares no code
    with the model. ``exact=True`` uses the microscopically exact self-limitation
    propensity ``|A_ii| n(n-1)/Omega``; ``exact=False`` is the engine's
    macroscopic ``|A_ii| n^2/Omega``.
    """
    nf = n.astype(float)
    out[:_S] = _R_ARR * nf
    outer = np.outer(nf, nf)
    if exact:
        np.fill_diagonal(outer, nf * (nf - 1.0))
    out[_S:] = (_ABS_A * outer).ravel() / omega
    return float(out.sum())


def _apply(n: np.ndarray, j: int) -> None:
    if j < _S:
        n[j] += 1
    else:
        n[_LOSS_SPECIES[j - _S]] -= 1


def coupled_bias(
    omega: float, t_max: float, seed: int, burn: float, exact_both: bool = False
) -> np.ndarray:
    """Time-averaged ``x_exact - x_macro`` from one split-coupled trajectory.

    ``exact_both=True`` runs *both* arms under the exact rule, which is the
    estimator's own tooth: every excess rate is then identically zero, the arms
    can never separate, and the answer must be exactly ``0`` — not "small".
    """
    rng = np.random.default_rng(seed)
    # Root-found per call rather than pinned: identical arguments every replicate,
    # so the initial counts are shared across the ensemble only as long as the
    # solver converges identically. It does — but that is why the arms are seeded
    # from ONE x_star here rather than from two independent computations.
    x_star = equilibrium(reference(Omega=omega).deterministic_params())
    n_e = np.rint(omega * x_star).astype(np.int64)
    n_m = n_e.copy()
    a_e = np.empty(_N_REACTIONS)
    a_m = np.empty(_N_REACTIONS)
    channels = np.empty(3 * _N_REACTIONS)
    acc = np.zeros(_S)
    t = 0.0
    while t < t_max:
        _propensities(n_e, omega, True, a_e)
        _propensities(n_m, omega, exact_both, a_m)
        shared = np.minimum(a_e, a_m)
        channels[:_N_REACTIONS] = shared
        channels[_N_REACTIONS : 2 * _N_REACTIONS] = a_e - shared
        channels[2 * _N_REACTIONS :] = a_m - shared
        cum = np.cumsum(channels)
        a0 = float(cum[-1])
        if a0 <= 0.0:
            break
        dt = rng.exponential(1.0 / a0)
        if t >= burn:
            acc += (n_e - n_m) * min(dt, t_max - t)
        t += dt
        k = int(np.searchsorted(cum, rng.random() * a0, side="right"))
        if k < _N_REACTIONS:
            _apply(n_e, k)
            _apply(n_m, k)
        elif k < 2 * _N_REACTIONS:
            _apply(n_e, k - _N_REACTIONS)
        else:
            _apply(n_m, k - 2 * _N_REACTIONS)
    return acc / (t_max - burn) / omega


def test_identical_propensity_rules_give_a_bit_for_bit_zero_difference():
    """The estimator's own tooth: with no excess rate the arms cannot separate.

    Without this, "the arms are coupled" is an assumption. A refactor that gave
    the two arms independent streams — or that mis-split the channels — would
    still produce a plausible bias number, just a much noisier one, and nothing
    else in this file would notice. Run both arms under the *same* rule and the
    shared channel carries the entire propensity, so the difference is ``0.0``.
    """
    for omega in (100.0, 400.0):
        difference = coupled_bias(omega, 60.0, 5, 10.0, exact_both=True)
        assert np.all(difference == 0.0), f"Omega={omega}: the arms are not coupled"


# The shipped configuration. See the assertion's docstring for why each number.
_BIAS_OMEGA = 100.0
_BIAS_T_MAX = 400.0
_BIAS_BURN = 20.0
_BIAS_SEED0 = 100
_BIAS_REPLICATES = 8
_BIAS_Z = 3.0


def measure_bias(
    omega: float = _BIAS_OMEGA,
    t_max: float = _BIAS_T_MAX,
    burn: float = _BIAS_BURN,
    seed0: int = _BIAS_SEED0,
    replicates: int = _BIAS_REPLICATES,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the coupled bias, per species."""
    vals = np.array([coupled_bias(omega, t_max, seed0 + k, burn) for k in range(replicates)])
    return vals.mean(axis=0), vals.std(axis=0, ddof=1) / np.sqrt(replicates)


def wrong_bias_formulas(omega: float) -> dict[str, np.ndarray]:
    """The two nearby formulas the measurement has to *exclude*, not merely agree
    less well with than the right one.

    ``naive`` drops the ``(-A)^{-1}`` solve, which is the entire non-trivial
    content of the claim: the excess loss rate ``|A_ii| n_i / Omega`` is a
    per-species perturbation to ``r``, and what it does to the *equilibrium* is a
    linear-response problem coupling all three species. ``transpose`` is 3a's
    trap one layer up — a consistently transposed system still has a genuine
    fixed point, so it has to be excluded by a number, not by inspection.
    """
    a = np.asarray(A3, dtype=float)
    diag = np.abs(np.diag(a))
    return {
        "naive |A_ii| / Omega (no linear-response solve)": diag / omega,
        "transposed A": np.linalg.solve(-a.T, diag) / omega,
    }


def test_the_propensity_bias_is_the_linear_response_vector_and_excludes_two_wrong_ones():
    """Closes the phase's open item: the bias equals ``(-A)^-1 diag|A_ii| 1 / Omega``.

    **All three components, and that is the point.** Every earlier measurement in
    this phase reported species 0 alone, which is the single worst choice: in
    units of ``1/Omega`` the correct vector is ``(0.6938, 0.6699, 0.5263)`` and
    the transposed-``A`` vector is ``(0.6818, 0.3828, 0.8254)`` — species 0
    differs by **1.7%**, species 1 by 43% and species 2 by 57%. Excluding the
    transpose on species 0 would need ``SE <= 2.4e-5``, unreachable at any
    affordable cost; on species 1 and 2 it needs ``~6e-4``, which this
    configuration reaches. A third near-miss is the plan's own recorded formula
    ``x*/Omega`` (``docs/plans/phase3-tasks.md``), which is not a different
    derivation but *this* one specialized to a system where ``diag|A_ii| 1 = r``
    — true of the plan's symmetric 2-species reference, false here, where it
    lands ``+1.0% / -35% / +60%`` off.

    So the blind spot is specifically for the errors that are *structurally*
    like the truth. Dropping the solve is ``+44% / +49% / +90%`` off and any
    component catches it; transposing ``A`` and the plan's ``x*/Omega`` are
    ``-1.7%`` and ``+1.0%`` on species 0 and only species 1 and 2 see them.

    **Why the wrong formulas are asserted here rather than as separate mutants.**
    A tooth is "the check goes red under the mutant", and the mutant here is the
    predicted vector, not the model — so all three verdicts come from *one*
    measurement. Three test functions would have xdist rebuild it three times.

    **Why this configuration.** ``Omega = 100`` (the bias is largest where it is
    hardest to measure, so the smallest affordable size wins), ``t_max = 400``,
    ``R = 8``, seeds 100-107, ~50-68 s. Halving to ``t_max = 200`` halves the cost
    and is *seed-lucky*: across four independent seed-sets the transpose is
    excluded at only ``4.213 / 3.056 / 3.025 / 5.519`` sigma, so at ``z = 3`` that
    tooth survives by **0.8%** on one of them. At ``t_max = 400`` the same four
    sets give ``4.780 / 5.872 / 5.622 / 12.352``.

    **Why z = 3.** Placed in a measured gap, 3b's rule. Over the four seed-sets:

    ====================  =====  =====  =====  ======
    max ``|z|`` against     100    300    500     700
    ====================  =====  =====  =====  ======
    correct               1.417  0.971  0.753   1.418
    naive                 7.782  9.133  9.344  18.738
    transposed ``A``      4.780  5.872  5.622  12.352
    ====================  =====  =====  =====  ======

    so the usable band is ``(1.418, 4.780)`` and ``z = 3`` sits 2.12x above the
    worst correct value and 1.59x below the best tooth. Note *which* component
    does the work: the transpose is rejected by species 1 and 2 at 4.1-12.4
    sigma, while species 0 never exceeds 1.8 in any set.

    **Burn.** Both arms start identical, so the difference grows from exactly 0
    and too short a burn biases *downward*. Measured, not assumed: ``burn = 20``
    and ``burn = 50`` agree within SE on every component at ``Omega = 100 / 200 /
    400``, so 20 is enough.
    """
    mean, sem = measure_bias()
    predicted = propensity_bias(reference(Omega=_BIAS_OMEGA))

    z_correct = np.abs(mean - predicted) / sem
    assert np.all(z_correct <= _BIAS_Z), (
        f"the measured bias is inconsistent with (-A)^-1 diag|A_ii| 1 / Omega: "
        f"z = {z_correct}, mean = {mean}, predicted = {predicted}, sem = {sem}"
    )

    for name, wrong in wrong_bias_formulas(_BIAS_OMEGA).items():
        z_wrong = np.abs(mean - wrong) / sem
        assert z_wrong.max() > _BIAS_Z, (
            f"{name} is NOT excluded by this measurement (max z = {z_wrong.max():.2f}) "
            f"— the assertion above would pass against a formula known to be wrong"
        )


# --------------------------------------------------------------------------
# Kurtz convergence: D(Omega) ~ Omega^{-1/2}
# --------------------------------------------------------------------------
#
# The teeth override ONLY initial_state, substituting a wrong system size into
# the params embedded in the State. step / observables / is_terminal read that
# embedded copy, so the whole SSA runs at the wrong size, while deterministic_rhs
# and initial_concentrations never touch Omega -- the ODE reference stays
# bit-identical. That isolates exactly one thing: how fluctuations scale.
#
# The repressilator's Omega^2 tooth does NOT transplant here: squaring
# Omega in [50, 800] is ~2e7 events per replicate at the top. The sqrt(Omega)
# tooth is the affordable complement -- D ~ Omega^{-1/4}, still significantly
# negative, so only the "consistent with -1/2" leg can reject it -- and it makes
# runs cheaper rather than more expensive.

_FIXED_OMEGA = 200.0


class _FixedOmegaGLVStochastic(GLVStochastic):
    """Propensities pinned to a constant system size, ignoring the swept ``Omega``.

    The realistic bug of simply failing to thread ``Omega`` through. Fluctuations
    never shrink, so ``D(Omega)`` is flat. Measured over seeds 0-3 at the config
    below: slope ``+0.0217 / +0.0005 / +0.0498 / +0.0029``, ``consistent=False``
    and ``significant=False`` at all four.
    """

    def initial_state(self, params: GLVStochasticParams, rng):
        return super().initial_state(replace(params, Omega=_FIXED_OMEGA), rng)


class _SqrtOmegaGLVStochastic(GLVStochastic):
    """Propensities use ``sqrt(Omega)`` — fluctuations shrink too *slowly*.

    Effective system size ``sqrt(Omega)`` gives ``D ~ Omega^{-1/4}``, i.e. slope
    ~ ``-0.25``: still *significantly negative*, so only the "consistent with
    ``-1/2``" leg can reject it. That is the point — without this tooth a green
    check would only prove "the noise decreases somehow".
    """

    def initial_state(self, params: GLVStochasticParams, rng):
        return super().initial_state(replace(params, Omega=float(np.sqrt(params.Omega))), rng)


register("_test_glv_stochastic_fixed_omega", _FixedOmegaGLVStochastic())
register("_test_glv_stochastic_sqrt_omega", _SqrtOmegaGLVStochastic())

# The sqrt tooth's nominal sweep is chosen so the EFFECTIVE sizes land on
# 25...400. Nominal Omega costs nothing here -- the SSA runs at sqrt(Omega) -- so
# this is free, and it matters: see test_sqrt_omega_... for what the cheap
# version was actually failing through.
_SQRT_OMEGAS = [625.0, 2500.0, 10000.0, 40000.0, 160000.0]


def _report(model_name: str, omegas, replicates: int, seed: int = 0):
    return convergence_report(
        model_name,
        BASE,
        factory,
        omegas=omegas,
        t_max=_T_MAX,
        dt=_DT,
        replicates=replicates,
        n_grid=_N_GRID,
        observable_keys=KEYS,
        seed=seed,
        z=_Z,
        n_bootstrap=300,
        max_steps=8_000_000,
    )


def test_fixed_omega_propensities_fail_the_convergence_check():
    """Noise that never shrinks: ``D(Omega)`` is flat, so the check must reject it.

    Every point runs at the *same* effective system size, so the three ``D``
    values are statistically identical and the fitted slope is noise about 0.
    Following the repressilator's split: ``consistent`` is asserted (it gets more
    robust as replicates rise — slope -> 0 and SE -> 0), ``significant`` is not
    (it trips when ``slope/SE < -z``, and both scale as ``1/sqrt(R)``, so that
    ratio's distribution is replicate-independent). The substance — "the noise
    does not shrink" — is asserted as a scale-free magnitude anchor instead.

    Measured over seeds 0-3: ``D_last/D_first`` = ``1.062 / 1.001 / 1.148 /
    1.008``, against ``0.283`` for the real model over the same 16x range.
    """
    report = _report("_test_glv_stochastic_fixed_omega", [100.0, 400.0, 1600.0], 8)
    assert not report.passed, str(report)
    assert not report.consistent, str(report)
    ratio = float(report.discrepancy[-1] / report.discrepancy[0])
    assert ratio > 0.65, f"D fell like Omega^-1/2 despite fixed propensities: {ratio}"


def test_sqrt_omega_propensities_fail_the_convergence_check():
    """Noise that shrinks too slowly: slope ~ ``-1/4``, rejected by the ``-1/2`` leg.

    ``significant`` stays True here, which is the point: this failure is caught
    *only* by the "consistent with -1/2" test, proving that leg has teeth.

    **The first draft of this tooth was seed-lucky, and the second was failing
    for the wrong reason.** Swept over the real model's own ``Omega`` range the
    effective sizes are ``10...40``, and there it *passed* at seed 1 (gap 0.200
    against a tolerance of 0.227). Raising replicates to 24 fixed the pass but
    not the diagnosis: at effective size 10 the initial count is **2 molecules
    per species**, and **22 of 24 replicates lost a species**. That tooth was
    biting through extinction, not through its exponent — the failure mode this
    project has recorded twice before. Re-sited so the effective sizes are
    ``25...400`` (nominal Omega is free, since the SSA runs at ``sqrt(Omega)``),
    it reads a clean ``-0.2619 / -0.2510 / -0.2598 / -0.2532`` over seeds 0-3
    with ``|slope+1/2|`` clearing the tolerance by ``7.86 / 6.63 / 4.99 / 7.72``,
    and 1 of 24 replicates losing a species at the smallest point.
    """
    report = _report("_test_glv_stochastic_sqrt_omega", _SQRT_OMEGAS, _REPLICATES)
    assert not report.passed, str(report)
    assert not report.consistent, str(report)
    # It must fail through its EXPONENT, not through vanishing: assert the slope
    # is genuinely near -1/4 rather than merely "not -1/2".
    assert report.slope == pytest.approx(-0.25, abs=0.05), str(report)
    assert report.significant, "the point of this tooth is that significance passes"


def test_no_replicate_loses_a_species_at_the_smallest_omega():
    """Extinction is absorbing, and ``convergence_report`` cannot see it.

    Its only anti-truncation guard is ``max_steps``; a run that reached the
    all-extinct state terminates *legitimately*, so a lost species would inflate
    that replicate's ``D`` permanently and silently. ``Omega = 50`` starts each
    species at 10 molecules — measured 0/24 replicates losing one, against 1/24
    at effective size 25 and 22/24 at 10.
    """
    result = run_experiment(
        Experiment(
            model="glv_stochastic",
            params={**BASE, "Omega": min(_OMEGAS), "t_max": _T_MAX},
            replicates=_REPLICATES,
            observables=[*KEYS, "n_survivors"],
            seed=0,
            max_steps=4_000_000,
        ),
        factory,
    )
    survivors = [float(np.min(t.series["n_survivors"])) for t in result.trajectories[0]]
    assert min(survivors) == 3.0, f"a replicate lost a species: {survivors}"


def test_glv_stochastic_discrepancy_scales_as_omega_minus_half():
    """The definition of "done" for this model (it has no closed form to match).

    Seed-verified at 0-3 before being pinned — 3b's lesson that the assertion you
    are defending needs the same seed treatment as the teeth defending it:
    slopes ``-0.4952 / -0.5134 / -0.4960 / -0.5228``, all PASS, with
    ``|slope+1/2| / (z SE)`` = ``0.08 / 0.22 / 0.03 / 0.37``.
    """
    report = _report("glv_stochastic", _OMEGAS, _REPLICATES, seed=0)
    assert report.passed, str(report)
    assert report.consistent and report.significant and report.reference_ok
    assert abs(report.slope + 0.5) <= _Z * report.slope_se

    # Averaged per replicate, never mean-first (the phase-diffusion trap).
    assert np.allclose(report.discrepancy, report.per_replicate.mean(axis=1))

    # No fit_mask is used, so the claim is that EVERY point is in the scaling
    # regime: D*sqrt(Omega) must be flat rather than knee-ing at the small end.
    anchor = report.discrepancy * np.sqrt(report.omegas)
    assert float(anchor.std(ddof=1) / anchor.mean()) < 0.15, (
        f"D*sqrt(Omega) is not flat across the sweep: {anchor}"
    )

    # The Richardson check must pass on evidence, not vacuously: an equilibrium
    # start would make the ODE reference flat and richardson_delta ~ 0.
    assert report.richardson_delta > 0.0
