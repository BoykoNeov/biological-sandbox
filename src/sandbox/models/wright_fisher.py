"""Wright-Fisher — genetic drift, and the deterministic change it collapses into.

A haploid population of fixed size ``N`` carries two alleles, A and a. Each
generation is formed by sampling ``N`` alleles *with replacement* from the
current pool — i.e. the next count of A is ``Binomial(N, p')`` where ``p'`` is
the (optionally selection-adjusted) current frequency of A. That binomial
sampling **is** genetic drift.

The validated, headline prediction is the neutral case (``s = 0``): the
probability that allele A eventually fixes equals its initial frequency ``p0``.
This is exact — ``freq`` is a bounded martingale, so it converges to 0 or 1 with
``P(fix) = E[freq] = p0``. The ValidationSuite confirms the fixation fraction
over many replicates equals ``p0`` within statistical tolerance.

The deterministic limit this collapses into: with no drift (``N -> infinity``)
and no selection, frequency simply stays at ``p0`` forever. With selection it
follows the logistic mass-action update ``p' = p(1+s) / (1 + s p)``. Overlaying
stochastic replicates on that deterministic line — and watching them hug it more
tightly as ``N`` grows — is the project's central teaching move.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from sandbox.core.registry import register


@dataclass(frozen=True)
class WFParams:
    """Wright-Fisher parameters.

    ``N``: population size. ``p0``: initial frequency of allele A in [0, 1].
    ``s``: selection coefficient for A (0 = neutral; >0 favours A).
    """

    N: int
    p0: float
    s: float = 0.0

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if not 0.0 <= self.p0 <= 1.0:
            raise ValueError(f"p0 must be in [0, 1], got {self.p0}")
        if self.s <= -1.0:
            raise ValueError(f"s must be > -1, got {self.s}")

    def selection_adjusted_frequency(self, p: float) -> float:
        """Haploid mass-action update: p' = p(1+s) / (1 + s p). Neutral when s = 0."""
        if self.s == 0.0:
            return p
        return p * (1.0 + self.s) / (1.0 + self.s * p)


@dataclass(frozen=True)
class WFState:
    """Count of allele A and the current generation.

    Params are embedded so ``step`` / ``observables`` / ``is_terminal`` are pure
    functions of the state alone (see ``core.protocol`` for why).
    """

    count: int
    t: float  # generation number
    params: WFParams


class WrightFisher:
    """Stateless Wright-Fisher model. Register one shared instance."""

    def initial_state(self, params: WFParams, rng: Generator) -> WFState:
        # Deterministic initial count (rng unused); randomness enters via step.
        count = round(params.p0 * params.N)
        return WFState(count=count, t=0.0, params=params)

    def step(self, state: WFState, rng: Generator) -> WFState:
        p = state.params.selection_adjusted_frequency(state.count / state.params.N)
        new_count = int(rng.binomial(state.params.N, p))
        return WFState(count=new_count, t=state.t + 1.0, params=state.params)

    def observables(self, state: WFState) -> dict[str, float]:
        n = state.params.N
        return {
            "freq": state.count / n,
            "fixed_A": 1.0 if state.count == n else 0.0,
            "lost_A": 1.0 if state.count == 0 else 0.0,
        }

    def is_terminal(self, state: WFState) -> bool:
        # Once an allele is fixed or lost, the population can never change again.
        return state.count == 0 or state.count == state.params.N

    def analytic_predictions(self, params: WFParams) -> dict[str, float]:
        if params.s != 0.0:
            # The clean closed form (neutral fixation = p0) only holds for s = 0.
            # Selection has a Kimura diffusion result, but it must be matched
            # carefully to this exact sampling convention before it can be a
            # trustworthy check — deferred rather than shipped half-validated.
            raise NotImplementedError(
                "analytic_predictions currently covers only the neutral case (s = 0)"
            )
        # Neutral fixation probability of A equals its initial frequency.
        return {"fixed_A": params.p0}


# The single shared, stateless instance used throughout the sandbox.
MODEL = WrightFisher()
register("wright_fisher", MODEL)
