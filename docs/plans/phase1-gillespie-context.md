# Phase 1 — context (key files & decisions for a fresh session)

Read `phase1-gillespie-plan.md` first for the full rationale. This is the quick
orientation: where new code lives, the contracts it must honor, and the traps.

## What Phase 1 adds (planned layout)

| Concern | File | New? |
|---|---|---|
| Deterministic ODE limit (RK4) | `src/sandbox/core/ode.py` | new |
| `DeterministicLimitModel` protocol | `src/sandbox/core/protocol.py` | extend |
| Convergence / scaling pathway | `src/sandbox/core/convergence.py` | new |
| Reaction network + SSA Direct Method | `src/sandbox/models/gillespie.py` | new |
| Birth-death (exact Poisson) | `src/sandbox/models/birth_death.py` | new |
| Isomerization (exact binomial) | `src/sandbox/models/isomerization.py` | new |
| Repressilator (Hill, convergence) | `src/sandbox/models/repressilator.py` | new |
| Overlay + convergence plots | `src/sandbox/viz/backends/matplotlib_backend.py` | extend |
| Repressilator end-to-end | `src/sandbox/demos/repressilator.py` | new |
| Tests (validation + convergence) | `tests/` | new |

## The one idea that drives every design choice here

**An oscillator's ensemble mean is not its deterministic limit.** Phase-diffusion
makes the mean of many repressilator replicates a *damped spiral*, while each
replicate keeps oscillating. So:

- validate the **engine** on exact closed forms (`birth_death`,
  `isomerization`) through the existing `validate()`;
- validate the **repressilator** by **Kurtz convergence**: a *single* scaled
  trajectory tracks the ODE on a finite horizon, discrepancy `~Ω^{-1/2}`.
- **Average the per-replicate discrepancy** — never `|mean(X) − ODE|`. That
  single sign error is the whole trap.

## Contracts the new code must honor

Everything from `phase0-...-context.md` still holds (stateless model, params
embedded in state, set `state.t` in `step`, register in `models/__init__.py`,
`analytic_predictions` keys name observables checked at their *final* value).
Phase-1 additions:

- **Serializability is preserved by design.** Do not register a generic
  `gillespie` model whose params hold a callable-bearing `ReactionNetwork` — that
  cannot round-trip to JSON. Each concrete model's params are plain numbers; the
  network is rebuilt from them inside `initial_state`.
- **Macroscopic definition + `Ω` scaling.** Define networks by stoichiometry
  `ν_j` and concentration-rates `f_j(c)`. Stochastic propensity
  `a_j(n) = Ω·f_j(n/Ω)`; initial counts `round(Ω·c0)`; ODE `dc/dt = Σ ν_j f_j(c)`.
  Exact for the unimolecular validation nets; definitional for Hill.
- **`step` = one reaction event** (Direct Method): propensities → `a0` → if
  `a0==0` terminal → `τ ~ Exp(a0)`, `t += τ` → choose reaction by `a_j/a0` →
  `n += ν_j`. This is the no-`dt` stress test.
- **`is_terminal`**: `t >= t_max` or `a0 == 0`. Reaching `t_max` is the intended
  terminal, so `terminated=True` there is correct and the anti-bias guard stays
  quiet. (The guard does **not** check `t_max ≫ relaxation time` — set `t_max`
  generously, e.g. `≫ 1/γ` for birth-death, or the stationary mean is biased.)
- **`observables()` returns concentrations `x = n/Ω`** (single enforced unit
  convention). So `birth_death` predicts `{x: k/γ}` (Ω-independent), the
  convergence metric and viz overlay line up with no rescaling, and there is no
  separate `concentration_observables` method. **Exception:** the Fano-factor
  test needs *counts* (`Var/⟨n⟩=1` is a counts identity; `=1/Ω` in
  concentrations), so it reconstructs `n = x·Ω`.

## The validation convention extension

- Exact-closed-form models reuse `validate(experiment, params_factory, z=4.0)`
  unchanged — measure the replicate-mean of the *final* observable. Run long
  enough (`t_max ≫ relaxation time`) that the system is in stationarity before
  the final sample.
- The convergence pathway (`core/convergence.py`) is *separate* from `validate()`:
  it checks a trajectory-vs-ODE *scaling law*, not a scalar. Keep its tolerance
  statistical (SE/bootstrap over replicates), never a hardcoded epsilon.
- **Convergence-test traps (the slope flattens at both ends — fit the middle):**
  - *Low-Ω knee:* phase-diffusion variance `~T/Ω`; if `T` is not `≪ Ω` replicates
    dephase and `D(Ω)` saturates flat. Keep `T ~ 1–2 periods`, `Ω_min` large so
    `T ≪ Ω_min` across the sweep; span `≥ 1 decade`, 4–6 points.
  - *High-Ω floor:* the RK4 ODE is the reference; Richardson-check (halve `dt`,
    ODE must move `≪` smallest `D(Ω)`) or its error floors the discrepancy.
  - **Pass/fail = slope CI** (`< 0`, consistent with `−1/2`). Monotonicity of
    `D(Ω)` flakes (replicate noise) — keep it only as a soft diagnostic.

## Exact closed forms to check against (so you don't re-derive them)

Means below are given in **concentration** (`x = n/Ω`, the observable unit); the
Fano check is the one that stays in counts.

- **Birth-death** `∅ →(k) X →(γ) ∅`: stationary `n ~ Poisson(Ω·k/γ)`. Predicted
  observable `⟨x⟩ = k/γ` (Ω-independent); **Fano `Var/⟨n⟩ = 1`** in *counts* (the
  stronger noise-level check).
- **Isomerization** `A ⇌ B` (`k₁` forward, `k₂` back), conserved total:
  stationary `n_A ~ Binomial(N, k₂/(k₁+k₂))`; predicted
  `⟨x_A⟩ = (k₂/(k₁+k₂))·c_tot`. The second exact check — exercises a conservation
  law + 2-species stoichiometry that birth-death doesn't.

## Repressilator parameters (oscillatory regime)

Dimensionless Elowitz–Leibler: `dm_i/dt = α/(1+p_{i-1}^nH) + α0 − m_i`,
`dp_i/dt = β(m_i − p_i)`, indices cyclic mod 3. Oscillates for large `α`
(e.g. `α ≈ 200`), `α0` small (leak), `nH ≈ 2`, `β` order 1. Confirm the chosen
params actually oscillate in the ODE before trusting the convergence check.

## Environment / gotchas (carried from Phase 0, plus new)

- Windows console is cp1252 — **no non-ASCII in `print`ed strings** (a stray `Ω`
  or `τ` in printed output will crash the demo). Source docstrings/comments are
  UTF-8 and may contain Unicode; printed strings must be ASCII (use `Omega`,
  `tau`, `<n>`).
- `uv` venv; numpy 2.x; matplotlib only under `--extra viz`. No scipy yet (RK4 is
  hand-rolled in `core/ode.py`).
- Event counts scale as `~Ω·T·(rates)`. Large `Ω` → many events → slow in pure
  Python. Keep `Ω` modest in the convergence sweep and bound `T`; numba is the
  deferred escape hatch (NOT JAX — Gillespie's data-dependent control flow fights
  the functional style).
- `Experiment.observables` is a tuple; `validate()` rejects a sweep; use
  `spawn_rngs`, never `seed + i`.
