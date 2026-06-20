# Phase 1 — Gillespie / repressilator (plan)

**Status: planned (not yet implemented).** Supersedes the "Next" stub in
`phase0-wright-fisher-tasks.md`.

## Goal

Establish the project's organizing thread as a concrete, reusable pattern:
**run a stochastic simulation alongside the deterministic limit it collapses
into, and watch them converge as molecule counts grow.** Phase 0 proved the
protocol on a model with an exact scalar closed form. Phase 1 proves it on a
*continuous-time jump process* — which stress-tests the no-`dt` step signature
(the model samples its own waiting time) — and introduces the deterministic
(mass-action ODE) limit as a first-class, overlaid comparison.

The headline deliverable is the **repressilator**: three genes repressing each
other in a cycle, oscillating stochastically, converging to a mass-action limit
cycle as the system size grows.

## The methodological crux (read this first)

The repressilator has **no scalar closed-form prediction**, and its
deterministic limit is a **limit cycle, not a fixed point**. This breaks the
naive validation instinct in a way worth stating loudly, because getting it
wrong would produce a green check that proves nothing:

> **You cannot validate an oscillator by checking "ensemble mean of replicates
> → ODE solution."** Independent stochastic replicates undergo *phase
> diffusion*: they drift out of sync over time, so the ensemble mean **damps
> toward the time-average of the cycle** even though every individual
> trajectory keeps oscillating at full amplitude. The mean of many repressilator
> replicates is a decaying spiral, not the limit cycle. Matching *that* to the
> ODE would be matching two different things.

The correct statement is **Kurtz's law of large numbers for density-dependent
jump processes**: as the system size `Ω` (volume; molecule counts scale with it)
grows, a **single** scaled trajectory `X_Ω(t)/Ω` converges *uniformly on any
finite time horizon* `[0, T]` to the deterministic ODE solution `x(t)`. The
fluctuations around it are `O(Ω^{-1/2})` in concentration (van Kampen system-size
expansion / functional CLT). So the rigorous, checkable claims are:

1. **Per-replicate, finite-horizon tracking.** For a fixed horizon `T`, the
   discrepancy `D(Ω) = mean over replicates of ⟨ |X_Ω(t)/Ω − x(t)| ⟩_t`
   (note: average the *per-replicate* error, **never** the error of the
   ensemble-averaged trajectory) **decreases monotonically** as `Ω` grows.
2. **The `Ω^{-1/2}` scaling rate.** On a log–log plot of `D(Ω)` vs `Ω`, the slope
   is ≈ `−1/2`. This is a sharper, quantitative check than mere monotonicity.

Because that is not a single-scalar match, it does not fit the existing
`validate()` (which compares the replicate-mean of a *final* observable to a
predicted scalar). Phase 1 therefore adds a **second, equally-rigorous
validation pathway** — a convergence/scaling check — rather than distorting
`analytic_predictions` into something it isn't.

To keep non-negotiable #2 honest ("a core model is done only when its
`analytic_predictions` are reproduced by the ValidationSuite"), the **Gillespie
*engine*** is validated against systems that *do* have exact closed forms, so the
engine's correctness never rests on the fuzzier convergence check alone.

## Architecture

### One engine, several concrete models (serializability-preserving)

A `ReactionNetwork` carries Python callables (propensity functions), which are
**not** JSON-serializable — and non-negotiable #3 requires every `Experiment` to
serialize and re-run. So we do **not** register a single generic `gillespie`
model whose params are a network. Instead, mirroring how `wright_fisher` is a
concrete named model with plain-number params:

- A shared, reusable **SSA engine** (`models/gillespie.py`): the `ReactionNetwork`
  dataclass + a pure `gillespie_step(counts, t, network, Ω, rng) -> (counts, t)`
  implementing the Direct Method. Stateless, no I/O.
- **Concrete registered models**, each with plain `float`/`int` params that are
  fully JSON-serializable; each builds its `ReactionNetwork` *internally* in
  `initial_state` and embeds it in the in-memory `State` (the network is
  reconstructed deterministically from the serialized params, never serialized):
  - `birth_death` — immigration/death, exact Poisson stationary law. **Has
    `analytic_predictions`.** Engine-correctness headline.
  - `isomerization` — reversible `A ⇌ B`, conserved total, exact binomial
    stationary law. **Has `analytic_predictions`.** Second exact check (exercises
    multi-species stoichiometry + a conservation law).
  - `repressilator` — 6-species Hill-repression network. **No
    `analytic_predictions`**; validated by the convergence pathway. Still a
    *verifiable* core model (it has a checkable claim — Kurtz convergence), so it
    is **not** quarantined under `ecosystem/`.

### The macroscopic-definition + system-size scaling trick

Every network is defined **macroscopically**: integer stoichiometry vectors `ν_j`
and reaction rates `f_j(c)` as functions of **concentrations** `c`. A single
system-size parameter `Ω` then links the two worlds:

- **Stochastic propensity** at size `Ω`:  `a_j(n) = Ω · f_j(n / Ω)`.
- **Initial counts:**  `n0 = round(Ω · c0)`.
- **Deterministic limit (ODE):**  `dc/dt = Σ_j ν_j · f_j(c)`.

`Ω` is exactly the "molecule counts grow" knob the convergence demo turns.

Why this is rigorous, not a fudge:
- For **zeroth- and first-order (unimolecular)** reactions — which is *all* of
  birth-death and isomerization — `a_j(n) = Ω·f_j(n/Ω)` is **exact** (e.g.
  death `γc` → `Ω·γ·(n/Ω) = γn`, the textbook unimolecular propensity). So the
  exact stationary closed forms hold and `validate()` checks them honestly.
- For the **Hill propensities** of the repressilator, the Hill form is already a
  coarse-grained (fast-promoter) reduction; `a_j(n)=Ω·f_j(n/Ω)` *is* the
  definition of its stochastic version, and Kurtz convergence to the ODE holds by
  construction.
- **Bimolecular** reactions would need the combinatorial correction
  (`n(n−1)/2` vs `Ω·(n/Ω)²/2`) which differs at `O(1/Ω)`. We use **no**
  bimolecular reactions in Phase 1; this limitation is documented in the engine
  and deferred.

### The `step` granularity (stress-testing the protocol)

`step(state, rng)` performs **one reaction event** via the Direct Method:

1. compute propensities `a_j(n)` and total `a0 = Σ a_j`;
2. if `a0 == 0` (no reaction can fire — absorbing state), mark terminal and stop;
3. sample waiting time `τ ~ Exp(a0)` and advance `t += τ`;
4. pick reaction `j` with probability `a_j / a0`, apply `n += ν_j`.

This is the honest realization of "the model owns its own time increment" — the
increment is a *sampled* `τ` only the model can compute. It is the sharpest test
yet of the no-`dt` signature decision from Phase 0.

`is_terminal(state)` returns `t >= t_max` (the run's time horizon) **or** the
absorbing `a0 == 0`. Reaching `t_max` is the *intended* terminal for these models
(not absorption), so `Trajectory.terminated` is legitimately `True` there — the
anti-bias guard in `validate()` will not false-alarm. A small overshoot of
`t_max` (the first event past it) is accepted; stationary-mean checks are
unaffected and trajectory figures sample on a sub-`t_max` time grid.

### Deterministic limit: hand-rolled RK4, no scipy

The mass-action ODE is integrated with a small fixed-step **RK4** in
`core/ode.py`. Rationale: stays NumPy-only (non-negotiable #4 — don't add scipy
until profiling forces it), dependency-light, and RK4 at a modest step is
ample-accuracy for the repressilator's smooth Hill RHS over the validation
horizon. `scipy.integrate.solve_ivp` (adaptive/stiff) is noted as the escape
hatch if a future model's RHS is stiff.

This introduces an optional protocol extension `DeterministicLimitModel` (sibling
to `ValidatableModel` / `TerminableModel`):

```python
@runtime_checkable
class DeterministicLimitModel(Model, Protocol):
    def deterministic_rhs(self, params) -> Callable[[np.ndarray], np.ndarray]: ...
    def initial_concentrations(self, params) -> np.ndarray: ...
```

It is on-thread (the whole project is "stochastic vs its deterministic limit")
and reused in Phase 2 (HH, Gray-Scott both have deterministic limits). Kept
optional and minimal; flagged as a protocol change to review carefully.

### Units: `observables()` returns concentrations, not counts

One convention, enforced everywhere, or the stochastic (counts) and
deterministic (concentration) worlds won't line up. **`observables()` returns
concentrations `x = n/Ω`** (`Ω` lives in the state). Consequences:

- `birth_death.analytic_predictions` returns `{x: k/γ}` (a concentration), not a
  count — and it no longer depends on `Ω`, so the same prediction validates at
  every system size.
- The convergence metric (`n/Ω` vs ODE) and the viz overlay are then directly
  comparable with no rescaling, and the proposed `concentration_observables`
  method is unnecessary — dropped from `DeterministicLimitModel` above.
- **The one place counts are still required is the Fano-factor check** — the
  Poisson identity `Var/⟨n⟩ = 1` holds in *counts*; in concentrations it reads
  `Var(x)/⟨x⟩ = 1/Ω`. So that dedicated test reconstructs `n = x·Ω` (Ω is in the
  state/params) before computing the ratio.

### The convergence pathway

`core/convergence.py` — `ConvergenceReport` + a function that, given a
`DeterministicLimitModel`, a horizon `T`, and a list of system sizes `Ω`:

1. integrates the ODE once on a time grid over `[0, T]`;
2. for each `Ω`, runs seeded SSA replicates, samples each on the same grid
   (step-interpolated — the recorder already tolerates irregular `t`), and
   computes the **per-replicate** time-averaged discrepancy to the ODE, then
   averages those discrepancies over replicates → `D(Ω)`;
3. fits the log–log slope of `D(Ω)` vs `Ω` and asserts it is ≈ `−1/2` *within a
   statistically-derived CI* (bootstrap / SE over replicates), never a hardcoded
   epsilon — consistent with the ValidationSuite's philosophy.

**Where this test misleads if built naively (settle these before writing it):**

- **The −½ slope flattens at BOTH ends, for reasons unrelated to engine
  correctness — so fit the slope only in the clean middle regime.**
  - *Low-Ω knee (phase saturation):* phase-diffusion variance grows as `~T/Ω`.
    If `T` is not `≪ Ω`, replicates fully dephase and `D(Ω)` saturates at
    `O(amplitude)` — flat, not `Ω^{-1/2}`. Keep `T` to **~1–2 periods** and make
    `Ω_min` large enough that `T ≪ Ω_min` across the *whole* sweep.
  - *High-Ω floor (reference error):* the RK4 ODE is the *reference*; if its
    integration error is not `≪ D(Ω)` at the largest `Ω`, the discrepancy floors
    and the slope flattens at the top. **Richardson check:** halve `dt` and
    confirm the ODE moves by `≪` the smallest measured `D(Ω)`.
  - Span **≥ 1 decade in `Ω` with 4–6 points**; fit the slope on the `Ω ≫ T`
    regime only.
- **Do not hang pass/fail on strict monotonicity of `D(Ω)`** — replicate noise
  makes adjacent-`Ω` values cross by chance, so a monotonicity assertion flakes.
  The real check is the **slope CI** (significantly `< 0` and consistent with
  `−1/2`); keep "monotone decrease" only as a soft printed diagnostic.
- **Teeth:** confirm a deliberately-broken propensity (e.g. wrong `Ω` scaling)
  drives the slope away from `−1/2` and FAILS, mirroring the Phase-0
  wrong-prediction test.

## The repressilator, concretely

Standard dimensionless Elowitz–Leibler form, 6 species `(m_i, p_i)` for `i=1,2,3`
with `p_i` repressing `m_{i+1}` cyclically:

```
dm_i/dt = α / (1 + (p_{i-1})^nH) + α0 − m_i
dp_i/dt = β (m_i − p_i)
```

`α` max transcription, `α0` leak, `nH` Hill coefficient, `β`
protein/mRNA timescale ratio. Parameters in the oscillatory regime (large `α`,
`nH ≈ 2`, `β` order 1). The stochastic propensities use the
`a_j = Ω·f_j(n/Ω)` rule above; production terms are state-dependent (Hill),
which the Direct Method handles since propensities may be arbitrary functions.

## Validation summary (what "done" means for Phase 1)

| Track | Model | Check | Pathway |
|---|---|---|---|
| Engine, exact | `birth_death` | stationary `⟨x⟩ = k/γ` (Poisson; concentration) | existing `validate()` |
| Engine, exact | `isomerization` | stationary `⟨x_A⟩ = (k₂/(k₁+k₂))·c_tot` (binomial) | existing `validate()` |
| Noise, exact | `birth_death` | Fano factor `Var/⟨n⟩ = 1` (in **counts**, across-replicate) | dedicated test |
| Thread, headline | `repressilator` | log–log slope of `D(Ω)` ≈ `−1/2` (CI) | new `convergence` |

`t_max` validity caveat: the stationary-mean checks assume the system has
relaxed before the final sample. The anti-bias guard catches too-few
`max_steps`, **not** a too-short `t_max` — so set `t_max ≫` relaxation time
(`1/γ` for birth-death) generously and state it; a green guard does not by itself
mean the mean is unbiased.

The repressilator demo is the payoff figure: stochastic replicates oscillating
around the ODE limit cycle, the cloud tightening as `Ω` grows — the generalized
form of the Wright-Fisher overlay.

## Reusable UX: the overlay (already supports this)

`plot_replicates` **already** accepts `deterministic=(t_seq, y_seq)` of arbitrary
length — Phase 0 just happened to pass a 2-point flat line. Overlaying the ODE
limit cycle is `deterministic=(t_grid, y_grid)` today; **no rewrite needed**,
only confirm units match (both concentrations, per the units decision above).
The one genuinely new viz is:
- `plot_convergence(D, Ω)` — a log–log helper showing the `Ω^{-1/2}` law;
- keep it import-guarded (matplotlib stays an optional extra).

## Deliberate deviations from the Phase-0 "Next" stub

- The stub said **"next-reaction method."** Phase 1 uses the **Direct Method**
  instead. Gibson–Bruck next-reaction is an *optimization* (priority queue +
  reaction-dependency graph, reusing random numbers) that pays off only for large
  networks with many reactions and disparate rates. Our networks are tiny (≤ 12
  reactions). Non-negotiable #4 says profile before optimizing; Direct Method is
  simpler, obviously correct, and fast enough. Next-reaction is **deferred** and
  noted as the first Gillespie optimization to reach for if profiling demands it.

## Deferred (explicitly out of scope for Phase 1)

- Gibson–Bruck next-reaction method; τ-leaping (approximate, faster) — profile
  first.
- numba / Cython acceleration of the SSA inner loop (HANDOFF flags Gillespie as
  NumPy/numba, **not** JAX — data-dependent control flow fights the functional
  style). Introduce only against a real bottleneck at large `Ω`.
- Bimolecular reactions (combinatorial propensity correction).
- scipy stiff integrators.
- Time-grid recording in the core recorder (Phase 1 sub-samples events via
  `record_every` and interpolates for the convergence grid; a first-class
  time-grid recorder can come later if event counts at large `Ω` force it).
- The Kimura-style *distribution* convergence beyond mean/slope.

## Build order within Phase 1

1. `core/ode.py` (RK4) + `DeterministicLimitModel` protocol extension.
2. `models/gillespie.py` — `ReactionNetwork` + `gillespie_step` (Direct Method).
3. `models/birth_death.py` — model + exact `analytic_predictions`. **Write its
   validation test first and confirm it fails before the engine is correct.**
4. `models/isomerization.py` — model + exact `analytic_predictions` + test
   (second exact check: conservation law + 2-species stoichiometry).
5. `core/convergence.py` — convergence/scaling pathway.
6. `models/repressilator.py` — network + deterministic RHS; convergence test.
7. Confirm the `plot_replicates` overlay (units match); add `plot_convergence`.
8. `demos/repressilator.py` — end-to-end (validate engine, run convergence, save
   the overlay + scaling figures).
9. Register all three models in `models/__init__.py`.
