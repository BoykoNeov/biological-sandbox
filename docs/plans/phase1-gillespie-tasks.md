# Phase 1 — tasks

Build order is dependency-ordered; each `[ ]` is a commit-sized unit. Per the
workflow rule, **write each validation/convergence test first and confirm it can
fail** before the implementation is correct.

## Core / protocol

- [x] `core/ode.py` — fixed-step RK4 integrator (NumPy-only). Signature
      `integrate_rk4(rhs, y0, t_max, dt) -> (t, y)`, dense uniform; convergence
      code interpolates onto sample times (decouples step from sample grid so the
      Richardson check is honest). Autonomous RHS `f(y)`. Test written first
      (exact linear closed forms + 4th-order convergence; verified Euler fails it).
- [x] `core/protocol.py` — added optional `DeterministicLimitModel` protocol
      (`deterministic_rhs`, `initial_concentrations`). Unit convention decided and
      documented: `observables()` returns **concentrations** `x = n/Omega`.
- [x] `core/convergence.py` — `ConvergenceReport` + `convergence_report()`:
      integrate ODE once (Omega-independent, concentration space), sweep system
      sizes `Omega` via `run_experiment`, compute per-replicate time-averaged
      discrepancy to the ODE (step-interpolated onto a grid), average over
      replicates -> `D(Omega)`. Pass/fail on the **log-log slope**: consistent with
      `-1/2` (`|slope+1/2| <= z*SE`) *and* significantly negative (`slope+z*SE < 0`),
      with `slope_se = max(bootstrap-over-replicates SE, OLS fit SE)` — never a
      hardcoded epsilon. `fit_mask` selects the `T << Omega` middle regime;
      monotonicity is a soft diagnostic only. Richardson-checks the RK4 reference
      (halve dt) and folds `reference_ok` into pass/fail; anti-bias guard raises if
      any replicate fails to reach `t_max`. **Validated on `birth_death`** (its
      *linear* dynamics make the LNA variance exact, so `-1/2` is clean at modest
      Omega): slope ~ -0.49 +/- 0.011, z=3. Teeth are non-statistical — the pure
      `_per_replicate_discrepancy` helper is unit-tested (per-replicate vs
      mean-first give 1.0 vs 0.0), and a `D(Omega)*sqrt(Omega)` magnitude anchor
      catches the mean-first (phase-diffusion) bug: it collapses ~sqrt(R) ~ 11x,
      92% off vs 3.7% for the correct code. Broken-Omega-scaling slope teeth deferred
      to the repressilator (step 6).

## Engine + exactly-solvable models (the `validate()` track)

- [x] `models/gillespie.py` — `ReactionNetwork` (integer `(R,S)` stoichiometry +
      single vector-valued macroscopic `rates(c) -> (R,)`) and pure
      `gillespie_step` (Direct Method, one event/step, `a_j = Omega*f_j(n/Omega)`,
      `a0==0` returns unchanged). `deterministic_rhs()` = `stoich.T @ rates(c)`
      derives from the *same* `rates` (SSA and ODE can't drift apart); public
      `propensities`/`total_propensity` helpers. Reaction selection uses
      `searchsorted(cumsum, r, side="right")` — never picks a zero-propensity
      reaction. Non-negativity emerges from vanishing rates (no clamping).
      Engine unit tests written first (confirmed red).
- [x] `models/birth_death.py` — model + plain-number params; `analytic_predictions`
      = `{x: k/gamma}` (concentration, Omega-independent). Set `t_max >> 1/gamma`.
      Validation test written first (confirmed red). Also implements
      `DeterministicLimitModel` (`dc/dt = k - gamma*c`, tested). Starts at `c0=0`
      so the mean check is a real relaxation test; `t_max = 10/gamma`.
- [x] `models/isomerization.py` — `A <-> B`, conserved total; `analytic_predictions`
      = `{x_A: (k2/(k1+k2))*c_tot}`. **Validation test first** (confirmed red).
      Second exact check: multi-species stoichiometry + a conservation law. Fixes
      the total `N = round(Omega*c_tot)` then splits off `n_A` so conservation is
      exact under rounding; starts all-`B` (`cA0=0`) so the mean check is a real
      relaxation test; `t_max = 5 >> 1/(k1+k2)`. Registered in `models/__init__.py`.
- [x] Fano-factor test: across-replicate `Var/<n> ~ 1` for birth-death, computed
      in **counts** (`n = x*Omega`) — a stronger noise check than the mean alone.
      Tolerance `4 * sqrt(2/(R-1))` (Poisson SE), not a hardcoded epsilon.

## Repressilator (the convergence track — headline)

- [ ] `models/repressilator.py` — 6-species Hill network, system-size scaled
      propensities, `DeterministicLimitModel` RHS; **no** `analytic_predictions`.
- [ ] Confirm chosen params oscillate in the ODE (sanity) before convergence test.
- [ ] Convergence test: `D(Omega)` decreases and slope ~ -1/2 across an `Omega`
      sweep. **Write it first**; confirm a deliberately-broken propensity
      (e.g. wrong `Omega` scaling) makes it FAIL — proving teeth, mirroring the
      Phase-0 wrong-prediction test.

## Viz + demo

- [ ] Confirm `plot_replicates` overlays the ODE limit cycle via its existing
      `deterministic=(t_grid, y_grid)` arg (no rewrite — just check units match).
- [ ] `plot_convergence(D, Omega)` — log-log helper showing the `Omega^-1/2` law.
- [ ] `demos/repressilator.py` — validate engine (birth-death/isomerization),
      run the convergence sweep, save the overlay (replicates vs ODE limit cycle)
      and the scaling figure. ASCII-only printed output.

## Wiring + green

- [ ] Register `birth_death`, `isomerization`, `repressilator` in
      `models/__init__.py`.
- [ ] `uv run pytest -q` green; `uv run ruff check .` clean; `uv run ruff format .`.
- [ ] Demo runs and produces both figures.
- [ ] Update `CLAUDE.md` Status line, memory, and `phase0-...-tasks.md` "Next"
      stub; commit (Conventional Commits) and push per the batch/session-end ritual.

## Explicitly deferred (do NOT do in Phase 1)

- [ ] ~~Gibson-Bruck next-reaction method~~ — Direct Method suffices; profile first.
- [ ] ~~tau-leaping~~ / ~~numba~~ — only against a real bottleneck at large `Omega`.
- [ ] ~~bimolecular reactions~~ (combinatorial propensity correction).
- [ ] ~~scipy stiff integrator~~ — RK4 is enough for the repressilator RHS.
