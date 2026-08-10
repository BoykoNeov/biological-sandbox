"""Evolutionary branching — the trait grid, and the divergence of its timing.

The same Gaussian competition model as ``models/adaptive_dynamics.py``, but with
many traits coexisting at once rather than one resident at a time. Bins on a
trait grid follow a generalized Lotka-Volterra system,

    dn_i/dt = r n_i (1 - (A n)_i / K_i),
    A_ij = a(x_i, x_j),   K_i = K(x_i)

with the kernels imported from ``adaptive_dynamics`` so the two halves of the
phase cannot drift apart. When the singular point is a fitness *minimum*
(``sa < sK``, i.e. :func:`~sandbox.models.adaptive_dynamics.splitting_rate`
positive) a population sitting there splits into two diverging morphs —
**evolutionary branching**, this arc's headline qualitative phenomenon.

**There is no mutation/diffusion term, and that is forced rather than chosen.**
A first draft carried nearest-neighbour diffusion and seeded every bin at
``1e-6``. It failed in two ways, and both are evidence:

* the *outer* bins won. Against a resident at the singular point every mutant has
  positive invasion fitness and the far ones have the most, so a uniformly seeded
  grid "branched" at ``t = 7.5`` into five clusters out to ``+-2.6``. That is a
  seed growing everywhere, not branching.
* ``sa = 1.5`` **overflowed**. At ``x = +-4``, ``K = 3.35e-4`` against a
  competition load of ``0.029``, so the local eigenvalue is ``-84`` and
  ``dt = 0.5`` gives ``lambda dt = -42``, far outside RK4's stability region.

The planning slice ran ``dt = 0.5`` on this domain without blowing up, so *its*
outer bins must have been exactly zero and stayed exactly zero — which happens
**iff** there is no diffusion, since ``n_i = 0`` is a per-bin invariant of pure
gLV. Domain, prefactor arithmetic and stability all point at one formulation, and
the tests assert the consequence directly: 158 untouched bins are
**bit-identically** ``0.0`` at the branch time.

**So the two morphs are not merely a grid artifact.** Seeding the centre and its
two immediate neighbours means those three bins are the only ones that ever carry
mass, which is why the morphs sit at exactly one grid spacing either side of the
seed at ``n_grid = 81 / 161 / 321``. Stated plainly: **this is a 3-species gLV in
which ``n_grid`` enters only through the spacing ``h``.** That is a stronger and
more honest statement than "grid-dependent, category C", and it is a different
claim from the Gyllenberg-Meszena positive-definiteness argument, which is about
the *continuum* model and is reported in the demo as category C.

**What is asserted, and what is refused.**

* **Category A.** With two bins seeded, this is an exactly solvable 2-species
  competition system (:func:`pair_equilibrium`), and that is what
  :meth:`TraitBranching.analytic_predictions` returns. The traits are chosen
  **asymmetric** on purpose: the symmetric case collapses to ``n* = K/(1 + a)``
  for both species, which cannot tell a wrong ``K(x)`` from a wrong ``a(x, y)``
  — 3a's lesson about self-consistency masquerading as independence.
* **Category B.** The branch/no-branch **sign change** across ``sa = sK``, and
  the divergence ``t_branch * splitting_rate = const``, asserted as **the
  exponent**. The prefactor is *not* universal — it moves with the detection
  threshold and the seed amplitude — so it is available as a second, independent
  check (:func:`predicted_product`) and never as a bound.
* **Refused.** ``analytic_predictions`` **raises** for the branching initial
  condition. Where the population ends up after it splits is the category-B
  claim, not a closed form, and returning the seeded state's arithmetic there
  would be a number that still looks green.

**The absence half of the sign change is horizon-bounded, and says so.** "Does
not branch" was originally checked to ``t = 20 000`` while the nearest *presence*
claim (``sa = 0.95``) takes ``149 822`` — an absence asserted 7.5x short of the
phenomenon is a statement about the horizon, not a sign change. It is now
measured to ``t = 200 000``, past every branch time this model exhibits, and the
test states the horizon in its own name.

**Detection resolution, and the direction the error runs.** The slice tested for
a gap every 200 steps at ``dt = 0.5``, quantizing ``t_branch`` to 100 time units:
at ``sa = 0.60`` its recorded ``3200.5`` carried ``+-50``, i.e. ``+-1.6%``, while
the *product* was reported constant to ``1.2%`` — a tolerance below its own
measurement's resolution. :func:`find_branch_time` keeps the previous coarse
checkpoint and, on first detection, replays that one interval step by step:
``+-dt/2`` for ~200 extra RK4 steps on one interval of one run, rather than
400 000 extra gap checks on every long run.

Doing so **inverted the expected answer.** Coarse checking made the products look
*more* constant than they are — ``0.381%`` at quantum 100 against ``0.622%``
true — because detection rounds *up*, and that inflation is largest where
``t_branch`` is smallest, which is exactly where the true product is lowest. **A
tolerance read off the coarse measurement would have been too tight and would
have failed a correct model.** What survives refinement is ``O(h^2)``:
``2.7481 / 0.6216 / 0.1517 %`` at ``h = 0.1 / 0.05 / 0.025``, and Richardson to
``h -> 0`` leaves the product constant to **0.006%** across a 16.5x rate range.
Note which instrument that is — refining a *discretization* parameter, the
Phase-2 stencil tool, not the Richardson-in-the-amplitude used for the ``O(sm)``
limit next door.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.ode import rk4_step
from sandbox.core.registry import register
from sandbox.models.adaptive_dynamics import (
    AdaptiveDynamicsParams,
    carrying_capacity,
    competition,
    splitting_rate,
)

INITIAL_CONDITIONS = ("branching", "pair")

#: Keys :meth:`TraitBranching.analytic_predictions` returns, in reporting order.
PREDICTED_KEYS = ("total", "mean_trait")


@dataclass(frozen=True)
class TraitBranchingParams:
    """The trait grid, the seeding, and the gap detector's two thresholds.

    ``initial`` selects the claim, the way ``DaisyworldParams.initial`` does.
    ``"branching"`` seeds the centre bin at its carrying capacity plus its two
    immediate neighbours at ``seed_amplitude`` — the configuration every recorded
    branch time was measured in. ``"pair"`` seeds two bins at
    ``pair_traits`` and is the exactly solvable case.

    ``t_max`` defaults to the long horizon the *absence* claim needs; anything
    that only has to converge should pass its own.
    """

    sigma_a: float = 0.7
    sigma_k: float = 1.0
    r_growth: float = 1.0
    k0: float = 1.0
    n_grid: int = 161
    half_width: float = 4.0
    dt: float = 0.5
    t_max: float = 200_000.0
    seed_amplitude: float = 1e-6
    threshold: float = 1e-3
    check_interval: int = 200
    initial: str = "branching"
    centre_fraction: float = 1.0
    pair_traits: tuple[float, float] = (-0.5, 1.0)
    pair_init: tuple[float, float] = (0.2, 0.2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_traits", tuple(float(v) for v in self.pair_traits))
        object.__setattr__(self, "pair_init", tuple(float(v) for v in self.pair_init))

        for name in ("sigma_a", "sigma_k", "r_growth", "k0", "half_width", "dt", "t_max"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.initial not in INITIAL_CONDITIONS:
            raise ValueError(f"initial must be one of {INITIAL_CONDITIONS}, got {self.initial!r}")
        if self.n_grid < 3 or self.n_grid % 2 == 0:
            raise ValueError(
                f"n_grid must be odd and at least 3 so a centre bin exists, got {self.n_grid}"
            )
        if not 0.0 < self.seed_amplitude < self.threshold:
            raise ValueError(
                f"need 0 < seed_amplitude ({self.seed_amplitude}) < threshold "
                f"({self.threshold}): a seed already above the detection threshold "
                "would register as a cluster at t = 0"
            )
        if self.check_interval < 1:
            raise ValueError(f"check_interval must be at least 1, got {self.check_interval}")
        if len(self.pair_traits) != 2 or len(self.pair_init) != 2:
            raise ValueError("pair_traits and pair_init must each have two entries")
        if any(v <= 0.0 for v in self.pair_init):
            raise ValueError(f"pair_init must be positive, got {self.pair_init}")

    @property
    def spacing(self) -> float:
        """``h``, the trait-grid spacing. The only way ``n_grid`` enters."""
        return 2.0 * self.half_width / (self.n_grid - 1)

    @property
    def trait_params(self) -> AdaptiveDynamicsParams:
        """The same params object the trait-space kernels take.

        Deliberate: ``K(x)`` and ``a(y, x)`` have exactly one definition in the
        phase, in ``adaptive_dynamics``, and this grid is built from it. A
        branching run and an invasion-fitness calculation therefore cannot
        disagree about the model they are simulating.
        """
        return AdaptiveDynamicsParams(
            sigma_a=self.sigma_a,
            sigma_k=self.sigma_k,
            r_growth=self.r_growth,
            k0=self.k0,
        )


def n_branching_steps(params: TraitBranchingParams) -> int:
    """Number of ``dt`` steps in a run — the terminal step index."""
    return int(round(params.t_max / params.dt))


def trait_grid(params: TraitBranchingParams) -> np.ndarray:
    """The trait values, ``n_grid`` points spanning ``[-half_width, half_width]``."""
    return np.linspace(-params.half_width, params.half_width, params.n_grid)


def interaction_matrix(params: TraitBranchingParams) -> np.ndarray:
    """``A_ij = a(x_i, x_j)``. Symmetric, unit diagonal, positive definite."""
    x = trait_grid(params)
    return np.asarray(competition(x[:, None], x[None, :], params.trait_params))


def capacities(params: TraitBranchingParams) -> np.ndarray:
    """``K_i = K(x_i)`` — the per-bin carrying capacity."""
    return np.asarray(carrying_capacity(trait_grid(params), params.trait_params))


def trait_branching_rhs(params: TraitBranchingParams) -> Callable[[np.ndarray], np.ndarray]:
    """The autonomous gLV RHS on the trait grid.

    The single definition of the vector field: :meth:`TraitBranching.step`,
    :func:`find_branch_time` and the demo all integrate *this*, so a branch time
    measured by the detector is a branch time of the model the tests validate.
    """
    a = interaction_matrix(params)
    k = capacities(params)
    r = params.r_growth

    def rhs(n: np.ndarray) -> np.ndarray:
        return r * n * (1.0 - (a @ n) / k)

    return rhs


def centre_index(params: TraitBranchingParams) -> int:
    """Index of the trait-zero bin — the singular point, exact because
    ``n_grid`` is odd and the domain is symmetric."""
    return params.n_grid // 2


def pair_indices(params: TraitBranchingParams) -> tuple[int, int]:
    """The two grid bins nearest ``pair_traits``, as indices.

    Snapped rather than assumed: the closed form is evaluated at the grid's
    *actual* traits, so a domain or ``n_grid`` change moves the prediction with
    the simulation instead of silently invalidating it.
    """
    x = trait_grid(params)
    i, j = (int(np.argmin(np.abs(x - t))) for t in params.pair_traits)
    if i == j:
        raise ValueError(
            f"pair_traits {params.pair_traits} snap to the same bin (index {i}) at "
            f"n_grid = {params.n_grid}; the pair case needs two distinct traits"
        )
    return i, j


def initial_abundances(params: TraitBranchingParams) -> np.ndarray:
    """The seeded grid. Every bin not named here is left at exactly ``0.0``."""
    n = np.zeros(params.n_grid)
    if params.initial == "pair":
        i, j = pair_indices(params)
        n[i], n[j] = params.pair_init
        return n
    mid = centre_index(params)
    n[mid] = params.centre_fraction * capacities(params)[mid]
    n[mid - 1] = n[mid + 1] = params.seed_amplitude
    return n


def pair_equilibrium(params: TraitBranchingParams) -> np.ndarray:
    """Closed-form coexistence abundances of the two seeded bins.

    With only bins ``i`` and ``j`` alive, ``A n = K`` reduces to a 2x2 system
    whose solution, using the kernel's symmetry ``A_ij = A_ji = alpha``, is

        n_i* = (K_i - alpha K_j) / (1 - alpha^2),   and ``i <-> j``.

    Raises when either component is non-positive: the competitor then excludes it
    and the attractor is a single-species boundary state, so predicting the
    interior point would be validating against a state the system does not have
    (``glv``'s stance on an infeasible ``x*``, and ``daisyworld``'s outside the
    regulating band).
    """
    i, j = pair_indices(params)
    k = capacities(params)
    alpha = float(interaction_matrix(params)[i, j])
    denominator = 1.0 - alpha**2
    if denominator <= 0.0:
        raise ValueError(
            f"the two traits are competitively indistinguishable (alpha = {alpha:.6g}); "
            "the 2x2 system is singular and has no isolated interior solution"
        )
    n_i = (k[i] - alpha * k[j]) / denominator
    n_j = (k[j] - alpha * k[i]) / denominator
    if n_i <= 0.0 or n_j <= 0.0:
        raise ValueError(
            f"no coexistence equilibrium at traits {params.pair_traits} with "
            f"sigma_a = {params.sigma_a}: the closed form gives ({n_i:.6g}, {n_j:.6g}), "
            "which is not strictly positive. Competitive exclusion sends the system to a "
            "single-species boundary state, so the interior point is not what a long run "
            "converges to"
        )
    return np.array([n_i, n_j])


# --------------------------------------------------------------------------
# Branch detection
# --------------------------------------------------------------------------


def neighbour_fitness(params: TraitBranchingParams) -> float:
    """Exact invasion fitness of the bin one spacing off centre, ``s_0(h)``.

    ``splitting_rate * h^2 / 2`` is its small-``h`` limit. Keeping the exact form
    alongside is what lets the residual drift in ``t_branch * splitting_rate`` be
    *attributed*: if it tracks the gap between these two it is the next order in
    ``h^2``, a systematic function of ``sa``, and not scatter.
    """
    return params.r_growth * (
        1.0 - math.exp(-(params.spacing**2) * splitting_rate(params.trait_params) / 2.0)
    )


def predicted_product(params: TraitBranchingParams) -> float:
    """``(2/h^2) log(1/(threshold * seed_amplitude)) + const`` — the prefactor law.

    The derived half is the neighbour bin rising from ``seed_amplitude`` to
    ``threshold`` at rate ``s_0(h) ~ rate h^2 / 2``. **The offset is fitted, and
    the mechanism is incomplete — say so rather than reading this as a closed
    form.** The neighbour rise alone predicts ``2 log(thr/seed)/h^2 = 5526``
    against a measured ``16086``, 2.9x short, because the *centre* bin must also
    fall below the threshold and that half is not derived. It shows up as a real
    residual in the limit: ``product * h^2 -> 40.058`` while
    ``2 log(1/(thr*seed)) = 41.447``, leaving ``1.39`` unexplained.

    With the slope **fixed** at ``2/h^2`` by the mechanism and only the offset
    fitted, this predicts all six measured ``(threshold, seed)`` configurations to
    within ``0.21%``. That is why the shipped assertion is on the *exponent*: this
    is a good predictor, not an identity.
    """
    h = params.spacing
    return (2.0 / h**2) * math.log(
        1.0 / (params.threshold * params.seed_amplitude)
    ) + PRODUCT_OFFSET


#: Fitted offset in :func:`predicted_product`, at ``h = 0.05``. Not derived.
PRODUCT_OFFSET = -488.1


def has_gap(n: np.ndarray, threshold: float) -> bool:
    """Two or more clusters of above-threshold bins, separated by dead bins.

    A **gap** criterion, not a peak count. Counting local maxima reported
    "2 peaks = branching" for every ``sa < sK`` in the slice — but the peaks were
    the seed's own immediate grid neighbours, so it reported success at ``t =
    4000`` where nothing had branched. Replacing it changed the answer by more
    than an order of magnitude: ``sa = 0.7`` needs ``t = 15 455``, not ``4000``.
    """
    alive = n > threshold
    if not alive.any():
        return False
    return int(np.count_nonzero(alive[1:] & ~alive[:-1])) + int(alive[0]) >= 2


@dataclass
class BranchResult:
    """Where the gap first opened, coarsely and then to within one ``dt``."""

    t_coarse: float | None
    t_refined: float | None
    n_final: np.ndarray
    steps: int
    refine_steps: int

    @property
    def branched(self) -> bool:
        return self.t_refined is not None


def find_branch_time(params: TraitBranchingParams) -> BranchResult:
    """Integrate until the trait distribution splits, or until ``t_max``.

    Checkpoint-and-refine: gaps are tested every ``check_interval`` steps, and the
    previous checkpoint's state is kept so that on first detection the last coarse
    interval can be replayed one step at a time. Resolution is ``+-dt/2`` for
    ``~check_interval`` extra RK4 steps on a single interval of a single run —
    versus checking every step of a 400 000-step run at every parameter point. See
    the module docstring for why the coarse answer was not merely imprecise but
    biased in a direction that *flattered* the constancy claim.
    """
    rhs = trait_branching_rhs(params)
    n = initial_abundances(params)
    n_steps = n_branching_steps(params)

    previous, previous_t = n.copy(), 0.0
    t_coarse: float | None = None
    steps = 0

    for i in range(1, n_steps + 1):
        n = rk4_step(rhs, n, params.dt)
        steps = i
        if i % params.check_interval == 0:
            if has_gap(n, params.threshold):
                t_coarse = i * params.dt
                break
            previous, previous_t = n.copy(), i * params.dt

    if t_coarse is None:
        return BranchResult(None, None, n, steps, 0)

    m = previous
    t_refined = t_coarse
    refine_steps = 0
    for j in range(1, params.check_interval + 1):
        m = rk4_step(rhs, m, params.dt)
        refine_steps = j
        if has_gap(m, params.threshold):
            t_refined = previous_t + j * params.dt
            break

    return BranchResult(t_coarse, t_refined, n, steps, refine_steps)


# --------------------------------------------------------------------------
# The protocol model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TraitBranchingState:
    """Abundances on the trait grid, the step counter, and the embedded params."""

    n: np.ndarray
    step_index: int
    t: float
    params: TraitBranchingParams
    rhs: Callable[[np.ndarray], np.ndarray]


class TraitBranching:
    """Stateless trait-grid gLV. Register one shared instance."""

    def initial_state(self, params: TraitBranchingParams, rng: Generator) -> TraitBranchingState:
        # Deterministic: rng is unused.
        return TraitBranchingState(
            n=initial_abundances(params),
            step_index=0,
            t=0.0,
            params=params,
            rhs=trait_branching_rhs(params),
        )

    def step(self, state: TraitBranchingState, rng: Generator) -> TraitBranchingState:
        index = state.step_index + 1
        return TraitBranchingState(
            n=rk4_step(state.rhs, state.n, state.params.dt),
            step_index=index,
            t=index * state.params.dt,  # counted, not accumulated
            params=state.params,
            rhs=state.rhs,
        )

    def observables(self, state: TraitBranchingState) -> dict[str, float]:
        """Scalars that summarize a whole trait distribution.

        ``mean_trait`` is the discriminating one for the pair prediction: it moves
        if either component is wrong, and unlike ``total`` it is sensitive to
        *which* bin carries the mass. It is ``nan`` on an empty grid rather than a
        silent ``0.0``, which would read as "the mean trait is the singular
        point".
        """
        n = state.n
        total = float(n.sum())
        x = trait_grid(state.params)
        return {
            "total": total,
            "mean_trait": float((x * n).sum() / total) if total > 0.0 else float("nan"),
            "max_abundance": float(n.max()),
            "centre": float(n[centre_index(state.params)]),
            "n_alive": float(np.count_nonzero(n > state.params.threshold)),
            "n_clusters": float(_cluster_count(n, state.params.threshold)),
        }

    def is_terminal(self, state: TraitBranchingState) -> bool:
        return state.step_index >= n_branching_steps(state.params)

    def fields(self, state: TraitBranchingState) -> dict[str, np.ndarray]:
        """The abundance profile over trait space — for plotting only."""
        return {"abundance": state.n}

    def analytic_predictions(self, params: TraitBranchingParams) -> dict[str, float]:
        """Total abundance and mean trait of the two-species coexistence state.

        **Raises for the branching initial condition**, and that refusal is the
        point: where a split population settles is a category-B claim about a
        sign change and a divergence rate, not a closed form. Returning the
        seeded state's arithmetic here would be a wrong number that still looks
        green — Phase 2's Gray-Scott error.

        Both keys are *aggregates*, so neither is the trivial restatement of an
        input. ``mean_trait`` in particular cannot be right unless the split
        between the two bins is right, which is why ``pair_traits`` is asymmetric:
        at symmetric traits both components equal ``K/(1 + alpha)`` and a wrong
        ``K(x)`` is indistinguishable from a wrong ``a(x, y)``.
        """
        if params.initial != "pair":
            raise ValueError(
                f"no closed-form prediction for initial = {params.initial!r}: the "
                "branching outcome is validated as a sign change and a divergence "
                "exponent (find_branch_time), not as an equilibrium. Use "
                "initial = 'pair' for the exactly solvable two-species case"
            )
        n_star = pair_equilibrium(params)
        i, j = pair_indices(params)
        x = trait_grid(params)
        total = float(n_star.sum())
        return {
            "total": total,
            "mean_trait": float((x[i] * n_star[0] + x[j] * n_star[1]) / total),
        }


def _cluster_count(n: np.ndarray, threshold: float) -> int:
    """Number of maximal runs of above-threshold bins."""
    alive = n > threshold
    if not alive.any():
        return 0
    return int(np.count_nonzero(alive[1:] & ~alive[:-1])) + int(alive[0])


#: The single shared, stateless instance used throughout the sandbox.
MODEL = TraitBranching()
register("trait_branching", MODEL)
