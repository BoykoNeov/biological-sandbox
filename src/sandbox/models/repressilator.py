"""Repressilator — three genes repressing each other in a cycle (Elowitz-Leibler).

This is Phase 1's **headline** model and the one that motivates the whole
convergence pathway. Unlike ``birth_death`` and ``isomerization`` it has **no
``analytic_predictions``**: its deterministic limit is a *limit cycle*, not a
fixed point, so no scalar closed form exists to match an ensemble mean against.
It is still a *verifiable* core model — its checkable claim is Kurtz convergence
(``D(Omega) ~ Omega^{-1/2}``, see :mod:`sandbox.core.convergence`) — so it belongs
here and **not** under the quarantined ``models/ecosystem/``.

**The network.** Six species ``(m_1, m_2, m_3, p_1, p_2, p_3)``: an mRNA and a
protein per gene, with ``p_i`` repressing the *next* gene's transcription
cyclically (``p_3 -| m_1``, ``p_1 -| m_2``, ``p_2 -| m_3``). In the standard
dimensionless form the macroscopic rates are

* transcription of ``m_i``:  ``alpha / (1 + p_{i-1}^nH) + alpha0``  (Hill repression + leak)
* mRNA decay:                ``m_i``                                (sets the time unit)
* translation of ``p_i``:    ``beta * m_i``
* protein decay:             ``beta * p_i``

which gives the textbook ODE ``dm_i/dt = alpha/(1 + p_{i-1}^nH) + alpha0 - m_i``,
``dp_i/dt = beta (m_i - p_i)``. Twelve reactions, all zeroth or first order in the
*count* of the species they consume; the Hill term is a state-dependent
production rate, which the Direct Method handles without special-casing.

**Propensity scaling.** ``a_j = Omega * f_j(n/Omega)`` as everywhere else. For the
decay and translation terms this is exact (unimolecular). For the Hill term the
Hill function is *already* a coarse-grained fast-promoter reduction, so
``Omega * f_j(n/Omega)`` **is** the definition of its stochastic version, and
Kurtz convergence to the ODE above holds by construction. No bimolecular
reactions appear, so the Phase-1 exactness caveat in
:mod:`sandbox.models.gillespie` is not engaged.

**Symmetry breaking is required, not cosmetic.** The all-equal state
``m_1=m_2=m_3, p_1=p_2=p_3`` lies on a cyclically symmetric invariant manifold of
the ODE, and the dynamics *within* that manifold are a stable 2-D relaxation to
the symmetric fixed point. Starting there yields **no oscillation at all** in the
deterministic limit — the very thing the convergence check needs. The default
initial protein concentrations are therefore deliberately unequal
(``p1_0 != p2_0 != p3_0``); ``initial_concentrations`` is what
:func:`sandbox.core.convergence.convergence_report` integrates, so a symmetric
default would silently produce a flat reference and a meaningless slope.

**Serializability (non-negotiable #3).** Params are plain floats; the
:class:`~sandbox.models.gillespie.ReactionNetwork` (which carries a Python
callable) is rebuilt deterministically from them inside ``initial_state`` and
lives only in the in-memory ``State``.
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

# Species order, fixed once and shared by the network, the ODE vector and the
# observable keys. Indices 0..2 are the mRNAs, 3..5 the proteins.
SPECIES: tuple[str, ...] = ("m1", "m2", "m3", "p1", "p2", "p3")

# Repressor of each mRNA: p_{i-1} cyclically. m1 <- p3, m2 <- p1, m3 <- p2.
# Values are indices into the concentration vector.
_REPRESSOR_OF = (5, 3, 4)

# Ordered observable keys ("x_" + species), matching the ODE vector column order.
# Pass this to convergence_report as observable_keys rather than relying on the
# insertion order of the dict returned by observables().
OBSERVABLE_KEYS: tuple[str, ...] = tuple(f"x_{s}" for s in SPECIES)


def _stoichiometry() -> np.ndarray:
    """The ``(12, 6)`` integer stoichiometry matrix, in the reaction order of ``rates``.

    Reactions are emitted per species as ``(+1, -1)`` pairs: production then decay
    for ``m_1, m_2, m_3``, then production then decay for ``p_1, p_2, p_3``. Every
    reaction touches exactly one species by one molecule.
    """
    stoich = np.zeros((12, 6), dtype=int)
    for s in range(6):
        stoich[2 * s, s] = +1  # production of species s
        stoich[2 * s + 1, s] = -1  # decay of species s
    return stoich


@dataclass(frozen=True)
class RepressilatorParams:
    """Repressilator parameters (all plain, JSON-serializable numbers).

    ``alpha``: maximum transcription rate (repressor absent). ``alpha0``: leak
    transcription (repressor saturating) — the ``alpha0 << alpha`` ratio is what
    sets how deep the troughs go. ``n_hill``: Hill coefficient of the repression.
    ``beta``: protein/mRNA timescale-and-abundance ratio. Time is measured in mRNA
    lifetimes (the mRNA decay rate is 1 by construction), so ``t_max`` is in those
    units. ``Omega``: system size (counts scale with it).

    ``m0`` is the initial concentration of all three mRNAs; ``p1_0``/``p2_0``/
    ``p3_0`` are the initial protein concentrations and **must not all be equal**
    (see the module docstring: an all-equal start does not oscillate).
    """

    alpha: float
    alpha0: float
    n_hill: float
    beta: float
    Omega: float
    t_max: float
    m0: float = 0.0
    p1_0: float = 0.0
    p2_0: float = 5.0
    p3_0: float = 15.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.alpha0 < 0:
            raise ValueError(f"alpha0 must be non-negative, got {self.alpha0}")
        if self.n_hill <= 0:
            raise ValueError(f"n_hill must be positive, got {self.n_hill}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.Omega <= 0:
            raise ValueError(f"Omega must be positive, got {self.Omega}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.m0 < 0:
            raise ValueError(f"m0 must be non-negative, got {self.m0}")
        for name in ("p1_0", "p2_0", "p3_0"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        # A cyclically symmetric start stays on the symmetric manifold and relaxes
        # to the (within-manifold stable) fixed point: no limit cycle, and the
        # convergence check would compare noise to a flat reference.
        if self.p1_0 == self.p2_0 == self.p3_0:
            raise ValueError(
                "p1_0, p2_0, p3_0 must not all be equal: the cyclically symmetric "
                "state is an invariant manifold on which the ODE does NOT oscillate"
            )


@dataclass(frozen=True)
class RepressilatorState:
    """Counts ``[n_m1, n_m2, n_m3, n_p1, n_p2, n_p3]``, time, embedded params + network."""

    counts: np.ndarray
    t: float
    params: RepressilatorParams
    network: ReactionNetwork


class Repressilator:
    """Stateless repressilator model. Register one shared instance."""

    @staticmethod
    def _network(params: RepressilatorParams) -> ReactionNetwork:
        alpha = params.alpha
        alpha0 = params.alpha0
        n_hill = params.n_hill
        beta = params.beta

        def rates(c: np.ndarray) -> np.ndarray:
            # Order must match _stoichiometry(): (prod, decay) per species, m then p.
            f = np.empty(12, dtype=float)
            for i in range(3):
                repressor = c[_REPRESSOR_OF[i]]
                f[2 * i] = alpha / (1.0 + repressor**n_hill) + alpha0  # transcription
                f[2 * i + 1] = c[i]  # mRNA decay (rate 1 sets the time unit)
                f[2 * (i + 3)] = beta * c[i]  # translation
                f[2 * (i + 3) + 1] = beta * c[i + 3]  # protein decay
            return f

        return ReactionNetwork(
            stoichiometry=_stoichiometry(),
            rates=rates,
            species=SPECIES,
        )

    def initial_state(self, params: RepressilatorParams, rng: Generator) -> RepressilatorState:
        # Deterministic initial counts (rng unused); randomness enters via step.
        counts = np.rint(params.Omega * self.initial_concentrations(params)).astype(np.int64)
        return RepressilatorState(
            counts=counts,
            t=0.0,
            params=params,
            network=self._network(params),
        )

    def step(self, state: RepressilatorState, rng: Generator) -> RepressilatorState:
        counts, t = gillespie_step(state.counts, state.t, state.network, state.params.Omega, rng)
        return RepressilatorState(counts=counts, t=t, params=state.params, network=state.network)

    def observables(self, state: RepressilatorState) -> dict[str, float]:
        # Concentrations x = n / Omega (the enforced project-wide unit convention).
        x = np.asarray(state.counts, dtype=float) / state.params.Omega
        return dict(zip(OBSERVABLE_KEYS, (float(v) for v in x), strict=True))

    def is_terminal(self, state: RepressilatorState) -> bool:
        # The intended terminal is the time horizon. The a0 == 0 guard is defensive:
        # with alpha0 > 0 transcription never vanishes, so a0 > 0 always.
        if state.t >= state.params.t_max:
            return True
        return total_propensity(state.network, state.counts, state.params.Omega) <= 0.0

    # NOTE: deliberately no analytic_predictions. The deterministic limit is a limit
    # cycle; there is no stationary scalar to match. Validation goes through
    # sandbox.core.convergence instead.

    def deterministic_rhs(self, params: RepressilatorParams) -> Callable[[np.ndarray], np.ndarray]:
        # dm_i/dt = alpha/(1+p_{i-1}^nH) + alpha0 - m_i ; dp_i/dt = beta (m_i - p_i),
        # assembled by the shared network from the same rates the SSA uses.
        return self._network(params).deterministic_rhs()

    def initial_concentrations(self, params: RepressilatorParams) -> np.ndarray:
        return np.array(
            [params.m0, params.m0, params.m0, params.p1_0, params.p2_0, params.p3_0],
            dtype=float,
        )


# The single shared, stateless instance used throughout the sandbox.
MODEL = Repressilator()
register("repressilator", MODEL)
