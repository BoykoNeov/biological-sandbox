"""Birth-death (immigration/death) — the Gillespie engine's exact-closed-form check.

A single species ``X`` with two reactions, defined *macroscopically* (rates as
functions of concentration ``c = n / Omega``):

* immigration ``0 -> X``   rate ``f_1(c) = k``      (zeroth order),
* death       ``X -> 0``   rate ``f_2(c) = gamma*c`` (first order).

This is the M/M/infinity queue. Its stationary law is ``n ~ Poisson(Omega*k/gamma)``,
so two exact statements hold and pin the SSA engine's correctness:

* **mean concentration** ``<x> = k / gamma`` — the model's ``analytic_predictions``.
  It is *Omega-independent*: because both reactions are zeroth/first order,
  ``a_j = Omega * f_j(n/Omega)`` is exact (no combinatorial correction), so the
  closed form holds at every system size. The ValidationSuite checks it directly.
* **Fano factor** ``Var(n)/<n> = 1`` (Poisson) — a stronger noise check, verified
  in a dedicated test in *counts* (``n = x*Omega``).

This is the "engine correctness headline" of Phase 1: unlike the repressilator
(validated only by the fuzzier Omega^-1/2 convergence law), birth-death rests on
an exact stationary closed form, so ``validate()`` checks it honestly.

The model is also a :class:`~sandbox.core.protocol.DeterministicLimitModel`: its
mass-action limit is ``dc/dt = k - gamma*c`` (fixed point ``k/gamma``), derived
from the *same* ``rates`` the SSA uses (via the shared
:class:`~sandbox.models.gillespie.ReactionNetwork`), so the stochastic and
deterministic worlds cannot silently disagree.

**Serializability (non-negotiable #3).** Params are plain floats; the
non-serializable :class:`ReactionNetwork` (which carries a Python callable) is
reconstructed deterministically from those params inside ``initial_state`` and
lives only in the in-memory ``State`` — never serialized.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.registry import register
from sandbox.models.gillespie import (
    ReactionNetwork,
    gillespie_step,
    total_propensity,
)


@dataclass(frozen=True)
class BirthDeathParams:
    """Birth-death parameters (all plain, JSON-serializable numbers).

    ``k``: macroscopic immigration rate. ``gamma``: per-capita death rate.
    ``Omega``: system size (reaction volume; counts scale with it). ``t_max``:
    time horizon — the run's terminal. ``c0``: initial concentration (default 0,
    so the run relaxes up to ``k/gamma`` and the mean check is a real relaxation
    test rather than a fixed-point no-op).
    """

    k: float
    gamma: float
    Omega: float
    t_max: float
    c0: float = 0.0

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.Omega <= 0:
            raise ValueError(f"Omega must be positive, got {self.Omega}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.c0 < 0:
            raise ValueError(f"c0 must be non-negative, got {self.c0}")


@dataclass(frozen=True)
class BirthDeathState:
    """Molecule count, current time, and the embedded params + reaction network.

    ``counts`` is a length-1 integer array (the SSA engine is vector-valued for
    all networks). Params and the reconstructed network are embedded so
    ``step`` / ``observables`` / ``is_terminal`` are pure functions of the state
    alone (see ``core.protocol`` for the rationale).
    """

    counts: np.ndarray
    t: float
    params: BirthDeathParams
    network: ReactionNetwork


class BirthDeath:
    """Stateless birth-death model. Register one shared instance."""

    @staticmethod
    def _network(params: BirthDeathParams) -> ReactionNetwork:
        k = params.k
        gamma = params.gamma
        # Reaction 0: 0 -> X (+1), rate k.  Reaction 1: X -> 0 (-1), rate gamma*c.
        return ReactionNetwork(
            stoichiometry=np.array([[+1], [-1]]),
            rates=lambda c: np.array([k, gamma * c[0]]),
            species=("X",),
        )

    def initial_state(self, params: BirthDeathParams, rng: Generator) -> BirthDeathState:
        # Deterministic initial count (rng unused); randomness enters via step.
        n0 = int(round(params.Omega * params.c0))
        return BirthDeathState(
            counts=np.array([n0], dtype=np.int64),
            t=0.0,
            params=params,
            network=self._network(params),
        )

    def step(self, state: BirthDeathState, rng: Generator) -> BirthDeathState:
        counts, t = gillespie_step(state.counts, state.t, state.network, state.params.Omega, rng)
        return BirthDeathState(counts=counts, t=t, params=state.params, network=state.network)

    def observables(self, state: BirthDeathState) -> dict[str, float]:
        # Concentration x = n / Omega (the enforced project-wide unit convention).
        return {"x": float(state.counts[0]) / state.params.Omega}

    def is_terminal(self, state: BirthDeathState) -> bool:
        # The intended terminal is the time horizon; the a0 == 0 absorbing guard is
        # defensive (immigration keeps a0 > 0 for birth-death, so it never fires).
        if state.t >= state.params.t_max:
            return True
        return total_propensity(state.network, state.counts, state.params.Omega) <= 0.0

    def analytic_predictions(self, params: BirthDeathParams) -> dict[str, float]:
        # Stationary Poisson mean concentration; Omega-independent (see module doc).
        return {"x": params.k / params.gamma}

    def deterministic_rhs(self, params: BirthDeathParams) -> Callable[[np.ndarray], np.ndarray]:
        # dc/dt = k - gamma*c, built from the same rates the SSA uses.
        return self._network(params).deterministic_rhs()

    def initial_concentrations(self, params: BirthDeathParams) -> np.ndarray:
        return np.array([params.c0], dtype=float)


# The single shared, stateless instance used throughout the sandbox.
MODEL = BirthDeath()
register("birth_death", MODEL, BirthDeathParams)
