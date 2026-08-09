"""Channel-noise Hodgkin-Huxley — the stochastic side of Phase 2's thread.

A real membrane does not have gating *variables*; it has a finite number ``N`` of
ion channels, each a little Markov machine flipping between discrete
conformations. The deterministic ``m``, ``h``, ``n`` of ``hodgkin_huxley`` are the
``N -> infinity`` limit of the fraction of channels in each conformation. This
model simulates the finite-``N`` truth, and the convergence pathway measures how
fast it collapses onto that limit: ``D(N) ~ N^{-1/2}``.

**State = occupancy counts, not channels.** Na is an 8-state chain — ``(i, j)``
with ``i`` the number of open ``m`` subunits (0..3) and ``j`` the ``h`` gate,
conducting only at ``(3, 1)``; K is a 5-state chain ``n0..n4``, conducting at
``n4``. Only the *count per state* is tracked, so a step costs ``O(#states)`` and
is **independent of ``N``** — measured at 0.203 s per replicate at every ``N``
from 1.6e4 to 4.1e6. That is what makes a 256x lever arm in ``N`` free, and the
whole convergence check affordable.

**The step is exact where it can be, and split where it cannot.** Rates depend on
``V``, which moves continuously, so propensities are time-inhomogeneous and
Gillespie's "constant between events" assumption fails. One ``step`` therefore
advances a fixed ``dt`` (a *param*, per the protocol's third time-advance
discipline) in two exactly-solved halves:

1. **Channels, at frozen ``V``.** With ``V`` fixed the 8-state Na chain
   *factorizes* into three independent ``m`` subunits and one ``h``, so the
   propagator is ``P_Na = P_m (x) P_h`` where each factor is a binomial
   convolution of the closed-form 2-state solution ``p_co = x_inf(1 - e^{-dt/tau})``,
   ``p_oo = x_inf + (1 - x_inf)e^{-dt/tau}`` — the very expression
   ``hh_voltage_clamp`` validates. This is ``expm(Q dt)`` to 7.4e-15, verified in
   the test file against a *separately hand-transcribed* generator.
2. **Voltage, at frozen conductance.** With the open fractions held fixed the
   membrane equation is linear with constant coefficients, so it too has a closed
   form: ``V <- V_inf + (V - V_inf) e^{-dt g_tot / C_m}``. No RK4 is involved.

**Why exactness in step 1 is not a luxury.** A ``rate * dt`` transition scheme
carries an ``O(dt)`` bias that is **``N``-independent** — so it would sit under
``D(N)`` as a floor, flatten the measured slope, and do it in a way that looks
like physics rather than like a bug. The residual splitting error between the two
halves is the only discretization left, and it was measured before this file was
written: **0 exactly at the resting fixed point** (both maps share it — gates at
``x_inf(V*)`` do not move, and ``V_inf`` at the fixed-point conductances *is*
``V*``), and ``1.0e-3 mV`` on a 1.5 mV sub-rheobase transient at ``dt = 0.025``.
Against ``D(N) >= 0.035 mV`` over the swept range that is under 3%, and halving
``dt`` moves ``D(N)`` by at most 1.9 standard errors — no floor.

(Advancing ``V`` at the *post*-transition conductance, rather than a pre/post
midpoint, was also measured: 1.7e-4 mV vs 1.9e-2 mV on the same transient. The
"more accurate" midpoint rule is 100x worse here, because using the newer gate
value partially compensates for having frozen ``V`` at the older one. It is not
built.)

**No ``analytic_predictions``, deliberately.** The obvious candidate — the
stationary mean of ``V`` equals the ODE fixed point — is *false* at finite ``N``:
channel noise in a nonlinear system shifts the mean by ``O(1/N)``. Predicting
``V*`` would be a wrong number that passes at loose tolerance and fails as
replicates grow. Like the repressilator, this model is validated by the
convergence track instead.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from sandbox.core.registry import register
from sandbox.models.hh_rates import alphas_betas, steady_state
from sandbox.models.hodgkin_huxley import hh_rhs

# Na state (i, j) -> index i * 2 + j: i open m subunits (0..3), j = h gate (1 = open).
NA_SUBUNITS = 3
K_SUBUNITS = 4
NA_STATES = 2 * (NA_SUBUNITS + 1)
K_STATES = K_SUBUNITS + 1
NA_CONDUCTING = NA_SUBUNITS * 2 + 1  # (3 open m, h open)
K_CONDUCTING = K_SUBUNITS  # n4

OBSERVABLE_KEYS: tuple[str, str, str] = ("V", "na_open", "k_open")


def subunit_propagator(x_inf: float, rate_sum: float, dt: float) -> tuple[float, float]:
    """``(p_open->open, p_closed->open)`` for one subunit over ``dt`` at frozen ``V``.

    Exact, not a rate expansion: this is ``x(dt)`` of ``dx/dt = (x_inf - x)/tau``
    started from 1 and from 0, with ``tau = 1/rate_sum``. ``rate_sum`` is passed
    as ``alpha + beta`` rather than ``tau`` so the exponent is one multiply and no
    reciprocal round-trip.
    """
    decay = math.exp(-dt * rate_sum)
    return x_inf + (1.0 - x_inf) * decay, x_inf * (1.0 - decay)


def count_propagator(k: int, p_oo: float, p_co: float) -> np.ndarray:
    """``P[i, i']`` for ``k`` independent identical subunits, ``i`` = number open.

    Of the ``i`` currently-open subunits, the number still open is
    ``Bin(i, p_oo)``; of the ``k - i`` closed ones, the number now open is
    ``Bin(k - i, p_co)``; ``i'`` is their sum. So row ``i`` is the coefficient
    list of ``(1 - p_oo + p_oo z)^i (1 - p_co + p_co z)^{k-i}``.

    **Built with plain Python floats on purpose.** The obvious NumPy version —
    broadcast tables, ``np.where`` masks, an accumulation loop — was written first
    and measured at **80.5 us** per step for the three propagators, against 24.2 us
    here and 72 us for the naive ``math.comb`` triple loop it was supposed to
    replace. On a 4x4 there is no arithmetic to amortize, only per-call overhead,
    and "vectorized" bought a 3x slowdown. The two agree to 1.1e-16.
    """
    p_closed, q_closed = 1.0 - p_oo, 1.0 - p_co
    open_pows: list[list[float]] = [[1.0]]
    closed_pows: list[list[float]] = [[1.0]]
    for _ in range(k):
        for pows, stay, go in ((open_pows, p_closed, p_oo), (closed_pows, q_closed, p_co)):
            prev = pows[-1]
            nxt = [0.0] * (len(prev) + 1)
            for idx, c in enumerate(prev):
                nxt[idx] += c * stay
                nxt[idx + 1] += c * go
            pows.append(nxt)

    rows = []
    for i in range(k + 1):
        a, b = open_pows[i], closed_pows[k - i]
        row = [0.0] * (k + 1)
        for ia, ca in enumerate(a):
            for ib, cb in enumerate(b):
                row[ia + ib] += ca * cb
        rows.append(row)
    return np.array(rows, dtype=float)


def _gate_propagator(gate: int, k: int, alpha: np.ndarray, beta: np.ndarray, dt: float):
    a, b = float(alpha[gate]), float(beta[gate])
    rate_sum = a + b
    return count_propagator(k, *subunit_propagator(a / rate_sum, rate_sum, dt))


def na_propagator(v: float, dt: float) -> np.ndarray:
    """The exact 8x8 Na propagator at frozen ``v``.

    ``P[(i,j), (i',j')] = P_m[i,i'] P_h[j,j']`` — the Kronecker product, written
    as a broadcast-and-reshape because ``np.kron`` costs 16.9 us against 3.1 us
    for 64 multiplies. (Measured; it is the same arithmetic and the difference is
    exactly zero.)
    """
    alpha, beta = alphas_betas(v)
    p_m = _gate_propagator(0, NA_SUBUNITS, alpha, beta, dt)
    p_h = _gate_propagator(1, 1, alpha, beta, dt)
    return (p_m[:, None, :, None] * p_h[None, :, None, :]).reshape(NA_STATES, NA_STATES)


def k_propagator(v: float, dt: float) -> np.ndarray:
    """The exact 5x5 K propagator at frozen ``v``."""
    alpha, beta = alphas_betas(v)
    return _gate_propagator(2, K_SUBUNITS, alpha, beta, dt)


def initial_occupancies(v0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrium state distributions at ``v0``: ``(p_na (8,), p_k (5,))``.

    Each channel's subunits are independent and at their steady state, so the Na
    distribution is ``Bin(3, m_inf) x Bernoulli(h_inf)`` and the K distribution is
    ``Bin(4, n_inf)``. Sampling counts from these — rather than rounding ``N p`` —
    is what keeps the initial condition from carrying a deterministic ``O(1/N)``
    artifact into the very ``N^{-1/2}`` law being measured.
    """
    m, h, n = (float(x) for x in steady_state(v0))
    p_m = np.array(
        [math.comb(NA_SUBUNITS, i) * m**i * (1 - m) ** (NA_SUBUNITS - i) for i in range(4)]
    )
    p_h = np.array([1.0 - h, h])
    p_n = np.array(
        [math.comb(K_SUBUNITS, i) * n**i * (1 - n) ** (K_SUBUNITS - i) for i in range(5)]
    )
    return (p_m[:, None] * p_h[None, :]).reshape(NA_STATES), p_n


@dataclass(frozen=True)
class HHStochasticParams:
    """Channel-noise HH parameters — the eight membrane constants plus ``n_channels``.

    ``n_channels`` sizes **both** populations equally, so it is the single
    ``Omega`` knob the convergence pathway sweeps. Equal counts (rather than the
    ~3:1 Na:K density ratio of squid axon) is a modelling choice, not an oversight:
    it removes a second knob that no check depends on, and it simply means the
    single-channel conductances are ``g_na/N`` and ``g_k/N`` respectively. The
    ``N^{-1/2}`` law is indifferent to the ratio.

    It is typed ``float`` because ``convergence_report`` builds params from
    ``{**base, "n_channels": float(omega)}`` and the dict must stay plain and
    JSON-serializable; ``__post_init__`` insists it is a whole number anyway.
    """

    n_channels: float = 10_000.0
    i_ext: float = 0.0
    t_max: float = 40.0
    dt: float = 0.025
    v0: float = -65.0
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    g_l: float = 0.3
    e_na: float = 50.0
    e_k: float = -77.0
    e_l: float = -54.387

    def __post_init__(self) -> None:
        if self.c_m <= 0:
            raise ValueError(f"c_m must be positive, got {self.c_m}")
        for name in ("g_na", "g_k", "g_l"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if self.t_max <= 0:
            raise ValueError(f"t_max must be positive, got {self.t_max}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be at least 1, got {self.n_channels}")
        if self.n_channels != round(self.n_channels):
            raise ValueError(f"n_channels must be a whole number, got {self.n_channels}")
        n = round(self.t_max / self.dt)
        if n < 1:
            raise ValueError(f"dt ({self.dt}) is larger than t_max ({self.t_max})")
        # Same reason as hodgkin_huxley: t is counted as step_index * dt, and the
        # convergence grid is built from the same multiplication, so the horizon
        # must land on a step boundary exactly.
        if abs(n * self.dt - self.t_max) > 1e-9 * max(1.0, abs(self.t_max)):
            raise ValueError(
                f"dt ({self.dt}) must divide t_max ({self.t_max}) exactly; "
                f"{n} steps land at {n * self.dt}"
            )


@dataclass(frozen=True)
class HHStochasticState:
    """Membrane potential, the two occupancy-count vectors, and the embedded params."""

    v: float
    na_counts: np.ndarray  # (8,) int, sums to n_channels
    k_counts: np.ndarray  # (5,) int, sums to n_channels
    step_index: int
    t: float
    params: HHStochasticParams


def n_hh_steps(params: HHStochasticParams) -> int:
    """Number of ``dt`` steps in the run — the terminal step index."""
    return round(params.t_max / params.dt)


class HHStochastic:
    """Stateless channel-noise Hodgkin-Huxley. Register one shared instance."""

    def initial_state(self, params: HHStochasticParams, rng: Generator) -> HHStochasticState:
        p_na, p_k = initial_occupancies(params.v0)
        n = int(params.n_channels)
        return HHStochasticState(
            v=params.v0,
            na_counts=rng.multinomial(n, p_na),
            k_counts=rng.multinomial(n, p_k),
            step_index=0,
            t=0.0,
            params=params,
        )

    def step(self, state: HHStochasticState, rng: Generator) -> HHStochasticState:
        params = state.params
        dt, v = params.dt, state.v
        n = int(params.n_channels)

        # 1. channels, exactly, at frozen V. One broadcast multinomial per
        #    population: rng.multinomial(counts, P) draws row s from P[s] with
        #    n_s trials, so summing down the rows gives the new occupancies.
        #    (Batching the 13 rows into 2 calls was measured at 13.30 us against
        #    13.46 us looped -- a wash; the batched form is kept because it is
        #    shorter, not because it is faster.)
        na_counts = rng.multinomial(state.na_counts, na_propagator(v, dt)).sum(axis=0)
        k_counts = rng.multinomial(state.k_counts, k_propagator(v, dt)).sum(axis=0)

        # 2. voltage, exactly, at the resulting (frozen) conductance.
        g_na = params.g_na * float(na_counts[NA_CONDUCTING]) / n
        g_k = params.g_k * float(k_counts[K_CONDUCTING]) / n
        g_tot = g_na + g_k + params.g_l
        if g_tot > 0.0:
            v_inf = (
                params.i_ext + g_na * params.e_na + g_k * params.e_k + params.g_l * params.e_l
            ) / g_tot
            v_new = v_inf + (v - v_inf) * math.exp(-dt * g_tot / params.c_m)
        else:
            # Every channel shut and no leak: the membrane is a bare capacitor and
            # the closed form degenerates (tau_V = infinity). Reachable only with
            # g_l = 0, which is a legal param, so it gets the exact limit rather
            # than a division by zero.
            v_new = v + dt * params.i_ext / params.c_m

        index = state.step_index + 1
        return HHStochasticState(
            v=v_new,
            na_counts=na_counts,
            k_counts=k_counts,
            step_index=index,
            t=index * dt,  # counted, not accumulated
            params=params,
        )

    def observables(self, state: HHStochasticState) -> dict[str, float]:
        n = float(state.params.n_channels)
        return {
            "V": state.v,
            "na_open": float(state.na_counts[NA_CONDUCTING]) / n,
            "k_open": float(state.k_counts[K_CONDUCTING]) / n,
        }

    def is_terminal(self, state: HHStochasticState) -> bool:
        return state.step_index >= n_hh_steps(state.params)

    def deterministic_rhs(self, params: HHStochasticParams) -> Callable[[np.ndarray], np.ndarray]:
        """The Hodgkin-Huxley vector field — literally ``hh_rhs``, not a copy of it.

        ``hh_rhs`` reads only the eight membrane constants, which these params
        carry under the same names, so it duck-types. Routing through the one
        definition is the structural guarantee that the stochastic model and its
        declared limit cannot drift apart; a translation layer here would be a
        second place for ``g_na`` to be wrong.
        """
        return hh_rhs(params)  # type: ignore[arg-type]

    def initial_concentrations(self, params: HHStochasticParams) -> np.ndarray:
        """``[V, m, h, n]`` at ``t = 0`` — the limit's initial condition.

        The ODE starts at the *mean* the counts are drawn around, which is what
        makes ``D(N)`` at ``t = 0`` a pure ``N^{-1/2}`` sampling fluctuation.
        """
        return np.concatenate([[params.v0], np.asarray(steady_state(params.v0), dtype=float)])


# The single shared, stateless instance used throughout the sandbox.
MODEL = HHStochastic()
register("hh_stochastic", MODEL)
