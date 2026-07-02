"""Isomerization ``A <-> B`` — the Gillespie engine's second exact-closed-form check.

Two species ``A`` and ``B`` interconverting reversibly, defined *macroscopically*
(rates as functions of concentration ``c = n / Omega``):

* forward  ``A -> B``   rate ``f_1(c) = k1 * c_A``  (first order),
* backward ``B -> A``   rate ``f_2(c) = k2 * c_B``  (first order).

Every reaction moves one molecule between the two species, so the total
``N = n_A + n_B`` is **conserved**. This is what makes isomerization the engine's
second exact check *after* birth-death: it exercises **multi-species stoichiometry
and a conservation law** that the single-species birth-death model does not.

At stationarity detailed balance gives ``k1 <n_A> = k2 <n_B>``; with the
conservation constraint each molecule is independently ``A`` with probability
``p = k2 / (k1 + k2)``, so ``n_A ~ Binomial(N, p)`` and two exact statements hold:

* **mean concentration** ``<x_A> = (k2/(k1+k2)) * c_tot`` — the model's
  ``analytic_predictions``. Like birth-death it is *Omega-independent*: both
  reactions are first order, so ``a_j = Omega * f_j(n/Omega)`` is exact and the
  closed form holds at every system size. The ValidationSuite checks it directly.
* the binomial variance ``Var(n_A) = N p (1-p)`` is exact too, but the dedicated
  noise check lives with birth-death (Poisson Fano factor); here we validate the
  mean and, structurally, the conservation law.

The model is also a :class:`~sandbox.core.protocol.DeterministicLimitModel`: its
mass-action limit is ``dc_A/dt = -k1 c_A + k2 c_B``, ``dc_B/dt = +k1 c_A - k2 c_B``
(a conserved 2-vector with fixed point ``c_A* = p c_tot``), derived from the *same*
``rates`` the SSA uses (via the shared
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
class IsomerizationParams:
    """Isomerization parameters (all plain, JSON-serializable numbers).

    ``k1``: forward ``A -> B`` rate constant. ``k2``: backward ``B -> A`` rate
    constant. ``Omega``: system size (reaction volume; counts scale with it).
    ``t_max``: time horizon — the run's terminal. ``c_tot``: total (conserved)
    concentration. ``cA0``: initial concentration of ``A`` (default 0, i.e. start
    all-``B``, so the run relaxes up to ``p*c_tot`` and the mean check is a real
    relaxation test). ``cA0`` must lie in ``[0, c_tot]``.
    """

    k1: float
    k2: float
    Omega: float
    t_max: float
    c_tot: float
    cA0: float = 0.0

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError(f"k1 must be positive, got {self.k1}")
        if self.k2 <= 0:
            raise ValueError(f"k2 must be positive, got {self.k2}")
        if self.Omega <= 0:
            raise ValueError(f"Omega must be positive, got {self.Omega}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.c_tot <= 0:
            raise ValueError(f"c_tot must be positive, got {self.c_tot}")
        if not (0.0 <= self.cA0 <= self.c_tot):
            raise ValueError(f"cA0 must lie in [0, c_tot={self.c_tot}], got {self.cA0}")


@dataclass(frozen=True)
class IsomerizationState:
    """Species counts ``[n_A, n_B]``, current time, and the embedded params + network.

    ``counts`` is a length-2 integer array. Params and the reconstructed network
    are embedded so ``step`` / ``observables`` / ``is_terminal`` are pure functions
    of the state alone (see ``core.protocol`` for the rationale).
    """

    counts: np.ndarray
    t: float
    params: IsomerizationParams
    network: ReactionNetwork


class Isomerization:
    """Stateless isomerization model. Register one shared instance."""

    @staticmethod
    def _network(params: IsomerizationParams) -> ReactionNetwork:
        k1 = params.k1
        k2 = params.k2
        # Species order (A, B). Reaction 0: A -> B, rate k1*c_A, stoich (-1, +1).
        # Reaction 1: B -> A, rate k2*c_B, stoich (+1, -1). Each conserves n_A+n_B.
        return ReactionNetwork(
            stoichiometry=np.array([[-1, +1], [+1, -1]]),
            rates=lambda c: np.array([k1 * c[0], k2 * c[1]]),
            species=("A", "B"),
        )

    def initial_state(self, params: IsomerizationParams, rng: Generator) -> IsomerizationState:
        # Deterministic initial counts (rng unused); randomness enters via step.
        # Fix the total N = round(Omega*c_tot) first, then split off n_A, so the
        # conserved total is exact regardless of rounding in the split.
        total = int(round(params.Omega * params.c_tot))
        n_a0 = int(round(params.Omega * params.cA0))
        n_a0 = min(n_a0, total)  # cA0 <= c_tot guaranteed, but guard rounding at the edge
        counts = np.array([n_a0, total - n_a0], dtype=np.int64)
        return IsomerizationState(
            counts=counts,
            t=0.0,
            params=params,
            network=self._network(params),
        )

    def step(self, state: IsomerizationState, rng: Generator) -> IsomerizationState:
        counts, t = gillespie_step(state.counts, state.t, state.network, state.params.Omega, rng)
        return IsomerizationState(counts=counts, t=t, params=state.params, network=state.network)

    def observables(self, state: IsomerizationState) -> dict[str, float]:
        # Concentrations x = n / Omega (the enforced project-wide unit convention).
        omega = state.params.Omega
        return {
            "x_A": float(state.counts[0]) / omega,
            "x_B": float(state.counts[1]) / omega,
        }

    def is_terminal(self, state: IsomerizationState) -> bool:
        # The intended terminal is the time horizon; the a0 == 0 absorbing guard is
        # defensive (with N > 0, a0 = k1*n_A + k2*n_B > 0, so it never fires).
        if state.t >= state.params.t_max:
            return True
        return total_propensity(state.network, state.counts, state.params.Omega) <= 0.0

    def analytic_predictions(self, params: IsomerizationParams) -> dict[str, float]:
        # Stationary binomial mean concentration; Omega-independent (see module doc).
        p = params.k2 / (params.k1 + params.k2)
        return {"x_A": p * params.c_tot}

    def deterministic_rhs(self, params: IsomerizationParams) -> Callable[[np.ndarray], np.ndarray]:
        # dc/dt = [-k1 c_A + k2 c_B, +k1 c_A - k2 c_B], from the same rates the SSA uses.
        return self._network(params).deterministic_rhs()

    def initial_concentrations(self, params: IsomerizationParams) -> np.ndarray:
        return np.array([params.cA0, params.c_tot - params.cA0], dtype=float)


# The single shared, stateless instance used throughout the sandbox.
MODEL = Isomerization()
register("isomerization", MODEL)
