"""Stochastic generalized Lotka-Volterra — demographic noise on 3a's vector field.

Phase 3's instance of the stochastic-vs-limit thread (after ``birth_death``,
``isomerization`` and the repressilator). The *same* gLV vector field validated
deterministically in :mod:`sandbox.models.glv` is here written as a reaction
network and run by the Gillespie engine, so the checkable claim is Kurtz
convergence: ``D(Omega) ~ Omega^{-1/2}`` (see :mod:`sandbox.core.convergence`).

**The network.** For ``dx_i/dt = x_i (r_i + sum_j A_ij x_j)`` with every
``A_ij <= 0`` (a purely competitive web), ``S + S^2`` reactions:

* birth ``X_i -> 2 X_i``, macroscopic rate ``r_i x_i``  — ``S`` reactions;
* loss ``X_i + X_j -> X_j``, macroscopic rate ``|A_ij| x_i x_j`` — ``S^2``
  reactions, one per ordered pair, **removing ``X_i``** because ``A_ij`` is the
  effect *of j on i*. ``j == i`` is self-limitation and needs no special case.

The removed species is worth stating twice because it is the one thing a
symmetric ``A`` cannot detect: with ``A = A^T`` the network that removes ``X_j``
instead has a bit-identical ODE limit. That is why the RHS anchor test uses
``glv``'s deliberately **asymmetric** 3-species matrix, and why this model's
default params are that same system rather than a tidy symmetric one.

**Three refusals**, each because the alternative is a network whose limit is not
the gLV that was asked for:

* ``A_ij > 0`` — a *benefit* of ``j`` on ``i``. Mass action would need
  ``X_i + X_j -> 2 X_i + X_j``, a different reaction. Taking ``|A_ij|`` would
  silently flip the sign of the interaction and still integrate to something;
* ``A_ii >= 0`` — no self-limitation, so the birth reaction is unbounded and the
  process explodes rather than converging to anything;
* ``r_i <= 0`` — the birth propensity would be negative or absent. It is also
  what makes ``is_terminal`` cheap: with every ``r_i > 0``, ``a0 == 0`` holds
  **iff** every count is zero, so ``counts.sum() <= 0`` is the exact absorbing
  condition without a ``rates()`` call on the hot path.

**No ``analytic_predictions``, and the reason is measured, not stylistic.** The
tempting closed form is ``<x> = x*``. It is wrong at ``O(1/Omega)`` twice over —
the van Kampen nonlinearity correction, and the propensity bias below — and it
would *pass* at any affordable replicate count: the bias-to-SE ratio grows as
``sqrt(R)``, so a check that is green today fails as someone adds replicates.
That is precisely the configuration ``random_matrix.fraction_report`` refuses
(3b) and the trap the canonical-equation item flags (3e), so this model declines
the same way the repressilator does — its claim is the scaling law, not a number.
:func:`replicates_until_bias_dominates` computes the ``R`` at which the tempting
check would flip, so the refusal rests on an artifact rather than on prose.

**The macroscopic-propensity bias — an open item of the phase, closed here.**
This is the project's first bimolecular reaction, the item Phase 1 deferred. The
engine's ``a_j = Omega f_j(n/Omega)`` gives ``|A_ii| n_i^2 / Omega`` where the
microscopically exact self-limitation propensity is ``|A_ii| n_i (n_i - 1) /
Omega`` (a molecule cannot react with itself). The excess loss rate is
``|A_ii| n_i / Omega``, i.e. an effective ``r_i' = r_i - |A_ii| / Omega``, so at
the drift level the two arms differ by :func:`propensity_bias` — ``O(1/Omega)``,
**subdominant** to the ``O(Omega^{-1/2})`` signal it would have to corrupt and
therefore incapable of flooring the slope. Phase 3's plan carried this as a
*derivation with one supporting point*; ``tests/test_glv_stochastic.py`` closes
it with a common-random-numbers estimator. See that file for the measured law.

**Serializability (non-negotiable #3).** Params are plain nested lists/tuples of
numbers; the :class:`~sandbox.models.gillespie.ReactionNetwork` (which carries a
Python callable) is rebuilt deterministically inside ``initial_state`` and lives
only in the in-memory ``State``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.registry import register
from sandbox.models.gillespie import ReactionNetwork, gillespie_step
from sandbox.models.glv import GLVParams, species_keys

# The 3a reference system, reused deliberately: r and A are asymmetric, so a
# transposed A or a loss reaction that removes the wrong species is detectable.
_DEFAULT_R: tuple[float, ...] = (1.0, 0.8, 1.2)
_DEFAULT_A: tuple[tuple[float, ...], ...] = (
    (-1.0, -0.3, -0.2),
    (-0.4, -1.0, -0.1),
    (-0.2, -0.5, -1.0),
)


def observable_keys(n_species: int) -> tuple[str, ...]:
    """The ordered species keys — the same convention as the deterministic gLV.

    Pass this to :func:`~sandbox.core.convergence.convergence_report` as
    ``observable_keys``: it must line up column-for-column with
    ``initial_concentrations`` / ``deterministic_rhs``, and ``observables()``
    also emits ``total_biomass`` and ``n_survivors``, which are *not* components
    of the ODE limit and must not be compared against it.
    """
    return species_keys(n_species)


def _stoichiometry(n_species: int) -> np.ndarray:
    """The ``(S + S^2, S)`` integer stoichiometry, in the reaction order of ``rates``.

    Rows ``0 .. S-1`` are the births (``+1`` on species ``i``); row
    ``S + i*S + j`` is the loss reaction for the ordered pair ``(i, j)`` and puts
    ``-1`` on species **i** — the species the interaction acts *on*.
    """
    stoich = np.zeros((n_species + n_species * n_species, n_species), dtype=int)
    for i in range(n_species):
        stoich[i, i] = +1
        for j in range(n_species):
            stoich[n_species + i * n_species + j, i] = -1
    return stoich


@dataclass(frozen=True)
class GLVStochasticParams:
    """Stochastic gLV parameters. ``r`` and ``A`` are nested plain numbers.

    Lists survive a JSON round-trip and tuples do not, so ``Experiment.params``
    carries **lists** while this frozen dataclass normalizes to nested **tuples**
    (a list field would be both unhashable and aliased into "immutable" params) —
    the same arrangement as :class:`~sandbox.models.glv.GLVParams`.

    ``Omega`` is the system size (counts scale with it; ``n0 = round(Omega x0)``)
    and is the knob the convergence check turns. ``x_init`` deliberately defaults
    *away* from the interior equilibrium: starting at ``x*`` makes the ODE
    reference exactly flat, which turns ``convergence_report``'s Richardson check
    into a threshold nothing can fail.
    """

    Omega: float
    t_max: float
    r: tuple[float, ...] = _DEFAULT_R
    A: tuple[tuple[float, ...], ...] = _DEFAULT_A
    x_init: tuple[float, ...] = (0.2, 0.2, 0.2)

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", tuple(float(v) for v in self.r))
        object.__setattr__(self, "A", tuple(tuple(float(v) for v in row) for row in self.A))
        object.__setattr__(self, "x_init", tuple(float(v) for v in self.x_init))

        n = len(self.r)
        if n == 0:
            raise ValueError("r must name at least one species")
        if len(self.A) != n or any(len(row) != n for row in self.A):
            raise ValueError(
                f"A must be {n}x{n} to match r (len {n}), got "
                f"{len(self.A)}x{tuple(len(row) for row in self.A)}"
            )
        if len(self.x_init) != n:
            raise ValueError(f"x_init must have {n} entries to match r, got {len(self.x_init)}")

        bad_r = [i for i, v in enumerate(self.r) if v <= 0.0]
        if bad_r:
            raise ValueError(
                f"r{bad_r} <= 0: every growth rate must be positive. The birth "
                "reaction X_i -> 2 X_i has macroscopic rate r_i x_i, which a "
                "non-positive r_i turns into a negative propensity; it is also what "
                "makes 'a0 == 0 iff every count is 0' true, which is_terminal relies on"
            )
        positive = [(i, j) for i, row in enumerate(self.A) for j, v in enumerate(row) if v > 0.0]
        if positive:
            raise ValueError(
                f"A{positive} > 0: this model only builds purely competitive webs. A "
                "positive A_ij is a benefit of j on i, which mass action writes as "
                "X_i + X_j -> 2 X_i + X_j -- a different reaction, not |A_ij| with the "
                "sign dropped. Building it from |A_ij| would silently flip the "
                "interaction and still produce a trajectory"
            )
        no_self = [i for i in range(n) if self.A[i][i] >= 0.0]
        if no_self:
            raise ValueError(
                f"A[i][i] >= 0 for i in {no_self}: without self-limitation the birth "
                "reaction is unbounded and the process has no equilibrium to converge to"
            )
        if any(v < 0.0 for v in self.x_init):
            raise ValueError(f"x_init must be non-negative, got {self.x_init}")
        if all(v == 0.0 for v in self.x_init):
            raise ValueError(
                "x_init is all zeros, which is the absorbing state: no reaction can "
                "ever fire and the run terminates at t = 0"
            )
        if self.Omega <= 0:
            raise ValueError(f"Omega must be positive, got {self.Omega}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")

    @property
    def n_species(self) -> int:
        return len(self.r)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(r, A)`` as float arrays."""
        return np.asarray(self.r, dtype=float), np.asarray(self.A, dtype=float)

    def deterministic_params(self) -> GLVParams:
        """The 3a :class:`~sandbox.models.glv.GLVParams` carrying the same vector field.

        Lets this model's ODE limit be checked against — and its equilibrium
        computed by — the *deterministic* gLV's own code rather than a second
        copy of the same algebra. ``t_max``/``dt`` are that model's defaults and
        carry no meaning here; only ``r``, ``A`` and ``x_init`` transfer.
        """
        return GLVParams(r=self.r, A=self.A, x_init=self.x_init)


@dataclass(frozen=True)
class GLVStochasticState:
    """Counts ``n_i``, current time, and the embedded params + reaction network."""

    counts: np.ndarray
    t: float
    params: GLVStochasticParams
    network: ReactionNetwork


def propensity_bias(params: GLVStochasticParams) -> np.ndarray:
    """``<x>_exact - <x>_macro`` predicted by the effective-``r`` shift, at ``O(1/Omega)``.

    The engine's macroscopic self-limitation propensity ``|A_ii| n_i^2 / Omega``
    exceeds the microscopically exact ``|A_ii| n_i (n_i - 1) / Omega`` by
    ``|A_ii| n_i / Omega`` — an extra per-capita loss, i.e. the macroscopic arm is
    the exact arm run at ``r' = r - d`` with ``d_i = |A_ii| / Omega``. gLV's
    equilibrium is *linear* in ``r`` (``x* = -A^{-1} r``), so at the drift level

        ``<x>_exact - <x>_macro = -A^{-1} d = (-A^{-1} diag|A_ii| 1) / Omega``.

    For the symmetric 2-species reference with ``r = (1, 1)`` and ``A_ii = -1``
    this collapses to the plan's ``x* / Omega``; the general form is used here so
    the claim is about the model rather than about one hand-picked system.

    This is a **drift-level** derivation: each arm additionally carries its own
    van Kampen nonlinearity correction, and those cancel only to ``O(1/Omega^2)``
    in the difference. Which is why the arms are *differenced* rather than either
    one being compared to ``x*`` alone. Measured in ``tests/test_glv_stochastic.py``.
    """
    _, a = params.arrays()
    return np.linalg.solve(-a, np.abs(np.diag(a))) / params.Omega


def replicates_until_bias_dominates(
    params: GLVStochasticParams, per_replicate_sd: float, species: int = 0
) -> float:
    """Replicate count at which the ``O(1/Omega)`` bias equals one standard error.

    The argument for *not* giving this model an ``analytic_predictions`` of
    ``x*``: such a check compares a replicate mean against a closed form with
    tolerance ``z * sd / sqrt(R)``, while the systematic offset stays fixed. So
    ``bias / SE = bias * sqrt(R) / sd`` grows without bound and the check flips
    from green to red purely by adding replicates. Returns ``(sd / bias)^2``, the
    ``R`` where that ratio reaches 1 — a measured artifact rather than a claim.

    ``per_replicate_sd`` is the measured across-replicate standard deviation of
    the observable at the configuration in question; ``species`` selects which
    component's bias to use.
    """
    bias = float(propensity_bias(params)[species])
    if bias <= 0.0:
        return float("inf")
    return (per_replicate_sd / bias) ** 2


class GLVStochastic:
    """Stateless stochastic gLV model. Register one shared instance."""

    @staticmethod
    def _network(params: GLVStochasticParams) -> ReactionNetwork:
        r, a = params.arrays()
        abs_a = np.abs(a)
        n = params.n_species

        def rates(c: np.ndarray) -> np.ndarray:
            # Order matches _stoichiometry(): S births, then the S^2 losses in
            # row-major (i, j) order. np.outer(c, c)[i, j] = c_i c_j, so the loss
            # block is |A_ij| c_i c_j -- ravel() reads it row-major, i.e. i outer.
            f = np.empty(n + n * n, dtype=float)
            f[:n] = r * c
            f[n:] = (abs_a * np.outer(c, c)).ravel()
            return f

        return ReactionNetwork(
            stoichiometry=_stoichiometry(n),
            rates=rates,
            species=observable_keys(n),
        )

    def initial_state(self, params: GLVStochasticParams, rng: Generator) -> GLVStochasticState:
        # Deterministic initial counts (rng unused); randomness enters via step.
        counts = np.rint(params.Omega * self.initial_concentrations(params)).astype(np.int64)
        return GLVStochasticState(
            counts=counts,
            t=0.0,
            params=params,
            network=self._network(params),
        )

    def step(self, state: GLVStochasticState, rng: Generator) -> GLVStochasticState:
        counts, t = gillespie_step(state.counts, state.t, state.network, state.params.Omega, rng)
        return GLVStochasticState(counts=counts, t=t, params=state.params, network=state.network)

    def observables(self, state: GLVStochasticState) -> dict[str, float]:
        """Per-species concentrations, total biomass, and the survivor count.

        ``n_survivors`` is emitted **here** and not in the deterministic gLV
        (``glv.py`` says so explicitly): extinction needs a threshold, and only
        the stochastic model has a scale to tie one to. That scale is ``1/Omega``
        — one molecule — so the threshold is exact rather than chosen:
        ``n_i == 0`` is absorbing for species ``i`` (its birth propensity
        ``r_i n_i`` vanishes and no reaction produces it), so a survivor is
        exactly a species with a nonzero count.

        It is a *diagnostic*, not a component of the ODE limit — pass
        :func:`observable_keys` to the convergence pathway so it is never
        compared against a species column.
        """
        x = np.asarray(state.counts, dtype=float) / state.params.Omega
        out = dict(zip(observable_keys(x.size), x.tolist(), strict=True))
        out["total_biomass"] = float(x.sum())
        out["n_survivors"] = float(np.count_nonzero(state.counts))
        return out

    def is_terminal(self, state: GLVStochasticState) -> bool:
        # The time horizon is the intended terminal. The second clause is the
        # exact absorbing condition, not an approximation of one: __post_init__
        # enforces r_i > 0, so every propensity vanishes iff every count is 0.
        # Checking counts.sum() is O(S) and, unlike a total_propensity() call,
        # costs no rates() evaluation on the per-event hot path (the profiled
        # ~20% the repressilator declined to pay).
        if state.t >= state.params.t_max:
            return True
        return int(state.counts.sum()) <= 0

    # NOTE: deliberately no analytic_predictions -- see the module docstring. The
    # stationary mean is NOT x*: it carries an O(1/Omega) nonlinearity correction
    # plus the propensity bias, and a check against x* would pass today and fail
    # as replicates grow (replicates_until_bias_dominates computes where).

    def deterministic_rhs(self, params: GLVStochasticParams) -> Callable[[np.ndarray], np.ndarray]:
        # dx_i/dt = x_i (r_i + sum_j A_ij x_j), assembled by the shared network
        # from the same rates the SSA uses -- so the two worlds cannot disagree.
        return self._network(params).deterministic_rhs()

    def initial_concentrations(self, params: GLVStochasticParams) -> np.ndarray:
        return np.asarray(params.x_init, dtype=float)


# The single shared, stateless instance used throughout the sandbox.
MODEL = GLVStochastic()
register("glv_stochastic", MODEL)
