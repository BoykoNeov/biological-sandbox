"""Gillespie SSA engine — the stochastic side of the stochastic-vs-limit thread.

This is a *shared engine*, not a registered model. The concrete registered models
(``birth_death``, ``isomerization``, ``repressilator``) each build a
:class:`ReactionNetwork` internally from their plain-number params and drive it
with :func:`gillespie_step`. Keeping the engine model-agnostic mirrors how
``core/ode.py`` is the shared deterministic integrator.

**Macroscopic definition + one system-size knob.** A network is defined
*macroscopically*: integer stoichiometry vectors ``nu_j`` and reaction rates
``f_j(c)`` as a function of **concentrations** ``c``. A single system size
``Omega`` (reaction volume; molecule counts scale with it) links the stochastic
and deterministic worlds:

* stochastic propensity at size ``Omega``:  ``a_j(n) = Omega * f_j(n / Omega)``;
* initial counts:                           ``n0 = round(Omega * c0)``;
* deterministic limit (mass-action ODE):    ``dc/dt = sum_j nu_j * f_j(c)``.

``Omega`` is exactly the "molecule counts grow" knob the convergence demo turns.

**One ``rates(c)`` drives both worlds — deliberately.** ``rates`` is a single
vector-valued function ``f(c) -> (R,)``. The SSA propensities and the ODE RHS are
*both* derived from it (``deterministic_rhs`` is ``stoich.T @ rates(c)``), so the
stochastic and deterministic sides cannot silently disagree about the same
network — the precise failure that would make a convergence check green yet
meaningless.

**Scope of exactness (Phase 1).** ``a_j = Omega * f_j(n/Omega)`` is *exact* for
zeroth- and first-order (unimolecular) reactions, which is all of birth-death and
isomerization — so their exact stationary closed forms hold and ``validate()``
checks them honestly. Bimolecular reactions would need the combinatorial
correction (``n(n-1)/2`` vs ``Omega*(n/Omega)**2/2``, differing at ``O(1/Omega)``)
and are **not** used in Phase 1.

**Serializability.** A :class:`ReactionNetwork` carries a Python callable and is
therefore *not* JSON-serializable. That is fine: it lives only inside the
in-memory ``State``. Each model reconstructs its network deterministically from
serialized plain-number params, so ``Experiment`` reproducibility (non-negotiable
#3) is preserved without ever serializing the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class ReactionNetwork:
    """A reaction network defined macroscopically (concentrations, not counts).

    Attributes
    ----------
    stoichiometry:
        Integer array of shape ``(R, S)`` — ``R`` reactions, ``S`` species.
        Row ``j`` is the state-change vector ``nu_j`` applied to the counts when
        reaction ``j`` fires.
    rates:
        Vector-valued macroscopic rate function ``f(c) -> array of shape (R,)``,
        taking the concentration vector ``c`` (length ``S``) and returning each
        reaction's macroscopic rate ``f_j(c)``. Drives *both* the SSA propensities
        (:func:`propensities`) and the deterministic RHS (:meth:`deterministic_rhs`).
    species:
        Optional species names (length ``S``), for labelling only.
    """

    stoichiometry: np.ndarray
    rates: Callable[[np.ndarray], np.ndarray]
    species: tuple[str, ...] = ()

    def deterministic_rhs(self) -> Callable[[np.ndarray], np.ndarray]:
        """The mass-action ODE RHS ``dc/dt = sum_j nu_j f_j(c) = stoich.T @ rates(c)``.

        Autonomous ``f(c) -> dc/dt``, matching both the ``DeterministicLimitModel``
        contract and :func:`sandbox.core.ode.integrate_rk4`. Built from the *same*
        ``rates`` the SSA uses, which is what keeps the two worlds consistent.
        """
        stoich_t = self.stoichiometry.T.astype(float)
        rates = self.rates

        def rhs(c: np.ndarray) -> np.ndarray:
            return stoich_t @ np.asarray(rates(c), dtype=float)

        return rhs


def propensities(network: ReactionNetwork, counts: np.ndarray, Omega: float) -> np.ndarray:
    """Stochastic propensities ``a_j(n) = Omega * f_j(n / Omega)`` at system size ``Omega``.

    Public so callers (a model's ``is_terminal``, the convergence pathway) share
    one definition of the propensity math with :func:`gillespie_step`.
    """
    c = np.asarray(counts, dtype=float) / Omega
    return Omega * np.asarray(network.rates(c), dtype=float)


def total_propensity(network: ReactionNetwork, counts: np.ndarray, Omega: float) -> float:
    """Total propensity ``a0 = sum_j a_j``. ``a0 == 0`` marks an absorbing state."""
    return float(propensities(network, counts, Omega).sum())


def gillespie_step(
    counts: np.ndarray,
    t: float,
    network: ReactionNetwork,
    Omega: float,
    rng: Generator,
) -> tuple[np.ndarray, float]:
    """One Direct-Method event: sample a waiting time, fire one reaction.

    The algorithm (Gillespie's Direct Method), performing exactly one event so it
    realizes the protocol's "the model owns its own time increment" — the
    increment is a *sampled* ``tau`` only the model can compute:

    1. propensities ``a_j(n)`` and total ``a0``;
    2. if ``a0 == 0`` no reaction can fire (absorbing state) — return ``(counts, t)``
       unchanged. The recorder loop is ``max_steps``-bounded and a model's
       ``is_terminal`` exits before this is reached, so this is defensive, not a
       spin risk;
    3. waiting time ``tau ~ Exp(a0)``; advance ``t += tau``;
    4. pick reaction ``j`` with probability ``a_j / a0`` and apply ``n += nu_j``.

    Reaction selection uses ``searchsorted(cumsum(a), r, side="right")`` with
    ``r = U * a0``. ``side="right"`` is load-bearing: it never selects a
    zero-propensity reaction (which ``side="left"`` can at an exact boundary,
    e.g. picking ``A -> B`` in the all-``B`` state and pushing ``A`` to ``-1``).

    Non-negativity of counts is *not* enforced here; it emerges from mass-action
    rates vanishing at the boundary (``gamma*c -> 0`` as ``n -> 0``). Clamping
    would hide a malformed network instead.
    """
    a = propensities(network, counts, Omega)
    a0 = a.sum()
    if a0 <= 0.0:
        return counts, t
    tau = rng.exponential(1.0 / a0)
    r = rng.random() * a0
    j = int(np.searchsorted(np.cumsum(a), r, side="right"))
    new_counts = np.asarray(counts) + network.stoichiometry[j]
    return new_counts, t + tau
