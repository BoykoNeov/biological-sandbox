"""Gillespie SSA engine — unit checks on the pure Direct-Method core.

Written before ``models/gillespie.py`` exists (workflow rule: confirm the test
can fail first). These pin the conventions the concrete models (birth-death,
isomerization, repressilator) and the convergence pathway will all rely on:

* the ``(R, S)`` stoichiometry layout and its ``stoich.T @ f`` transpose;
* the system-size scaling ``a_j = Omega * f_j(n / Omega)`` (exact for the
  unimolecular reactions we use);
* Direct-Method reaction selection with ``searchsorted(..., side="right")`` so a
  *zero-propensity* reaction is never chosen (which would push a count negative);
* the shared ``rates(c)`` definition driving both the SSA propensities and the
  deterministic ODE RHS, so the two worlds cannot silently drift apart.
"""

from __future__ import annotations

import numpy as np

from sandbox.models.gillespie import (
    ReactionNetwork,
    gillespie_step,
    propensities,
    total_propensity,
)


def death_only(gamma: float = 1.0) -> ReactionNetwork:
    # A -> 0, macroscopic rate gamma * c_A.  R=1, S=1.
    return ReactionNetwork(
        stoichiometry=np.array([[-1]]),
        rates=lambda c: np.array([gamma * c[0]]),
        species=("A",),
    )


def birth_death(k: float = 5.0, gamma: float = 1.0) -> ReactionNetwork:
    # 0 -> A (rate k),  A -> 0 (rate gamma * c_A).  R=2, S=1.
    return ReactionNetwork(
        stoichiometry=np.array([[+1], [-1]]),
        rates=lambda c: np.array([k, gamma * c[0]]),
        species=("A",),
    )


def isomerization(k1: float = 2.0, k2: float = 3.0) -> ReactionNetwork:
    # A -> B (rate k1 * c_A),  B -> A (rate k2 * c_B).  R=2, S=2.
    return ReactionNetwork(
        stoichiometry=np.array([[-1, +1], [+1, -1]]),
        rates=lambda c: np.array([k1 * c[0], k2 * c[1]]),
        species=("A", "B"),
    )


class _StubRng:
    """Minimal Generator stand-in: fixed waiting time, fixed uniform draw.

    Lets a test force the exact boundary the ``side="right"`` fix guards against.
    """

    def __init__(self, uniform: float) -> None:
        self._u = uniform

    def exponential(self, scale: float) -> float:
        return scale  # any positive number; the test doesn't inspect tau

    def random(self) -> float:
        return self._u


def test_single_death_decrements_by_exactly_one_and_advances_time():
    # Pins the (R, S) stoichiometry orientation and that tau > 0.
    net = death_only()
    rng = np.random.default_rng(0)
    counts0 = np.array([10])
    counts1, t1 = gillespie_step(counts0, 0.0, net, Omega=100.0, rng=rng)
    assert counts1.tolist() == [9]
    assert t1 > 0.0


def test_unimolecular_propensity_is_omega_independent():
    # Death f = gamma * c gives a = Omega * gamma * (n / Omega) = gamma * n,
    # independent of Omega. This exactness is what lets validate() trust the
    # birth-death / isomerization stationary closed forms at every system size.
    net = death_only(gamma=1.5)
    n = np.array([40])
    a_small = propensities(net, n, Omega=10.0)
    a_large = propensities(net, n, Omega=1_000.0)
    assert np.isclose(a_small[0], 1.5 * 40)
    assert np.isclose(a_large[0], 1.5 * 40)


def test_zeroth_order_propensity_scales_with_omega():
    # Immigration f = k (constant) gives a = Omega * k — it DOES scale with Omega
    # (only unimolecular rates are Omega-independent). Guards the scaling rule.
    net = birth_death(k=5.0, gamma=1.0)
    n = np.array([0])
    a10 = propensities(net, n, Omega=10.0)
    a20 = propensities(net, n, Omega=20.0)
    assert np.isclose(a10[0], 5.0 * 10.0)  # immigration
    assert np.isclose(a20[0], 5.0 * 20.0)


def test_reaction_selection_frequencies_match_propensity_ratio():
    # At fixed counts, the fraction of steps that fire reaction j must approach
    # a_j / a0. birth (count +1) vs death (count -1) are distinguishable by sign.
    net = birth_death(k=5.0, gamma=1.0)
    n = np.array([3])
    Omega = 1.0
    a = propensities(net, n, Omega)  # [k*Omega, gamma*n] = [5, 3]
    p_birth = a[0] / a.sum()

    rng = np.random.default_rng(12345)
    trials = 20_000
    births = 0
    for _ in range(trials):
        counts1, _ = gillespie_step(n, 0.0, net, Omega, rng)
        if counts1[0] == 4:  # +1 => birth fired
            births += 1
    frac = births / trials
    se = np.sqrt(p_birth * (1 - p_birth) / trials)
    assert abs(frac - p_birth) < 5 * se


def test_deterministic_rhs_equals_stoich_T_times_rates():
    # dc/dt = sum_j nu_j f_j(c) = stoich.T @ rates(c). Hand-checked on
    # isomerization: dA/dt = -k1 A + k2 B,  dB/dt = +k1 A - k2 B.
    net = isomerization(k1=2.0, k2=3.0)
    rhs = net.deterministic_rhs()
    c = np.array([0.4, 0.6])
    expected = np.array([-2.0 * 0.4 + 3.0 * 0.6, +2.0 * 0.4 - 3.0 * 0.6])
    assert np.allclose(rhs(c), expected)


def test_selection_skips_zero_propensity_reaction():
    # The side="right" fix: in the all-B state the A->B reaction has zero
    # propensity. A uniform draw of exactly 0.0 must NOT select it (side="left"
    # would, pushing A to -1). Correct choice is reaction 1 (B->A): A goes 0->1.
    net = isomerization(k1=2.0, k2=3.0)
    counts = np.array([0, 5])  # no A present
    assert propensities(net, counts, Omega=1.0)[0] == 0.0
    new_counts, _ = gillespie_step(counts, 0.0, net, Omega=1.0, rng=_StubRng(0.0))
    assert new_counts.tolist() == [1, 4]
    assert (new_counts >= 0).all()


def test_absorbing_state_returns_unchanged():
    # Death-only started with no molecules: a0 == 0, nothing can fire. The step
    # is a no-op and total_propensity reports the absorbing condition. Non-
    # negativity emerges from the rate vanishing, not from clamping.
    net = death_only()
    counts = np.array([0])
    assert total_propensity(net, counts, Omega=100.0) == 0.0
    new_counts, t = gillespie_step(counts, 7.0, net, Omega=100.0, rng=np.random.default_rng(0))
    assert new_counts.tolist() == [0]
    assert t == 7.0


def test_same_seed_reproduces_step():
    # Non-negotiable #3: a seeded step is deterministic.
    net = birth_death()
    n = np.array([3])
    c_a, t_a = gillespie_step(n, 0.0, net, 1.0, np.random.default_rng(99))
    c_b, t_b = gillespie_step(n, 0.0, net, 1.0, np.random.default_rng(99))
    assert c_a.tolist() == c_b.tolist()
    assert t_a == t_b
