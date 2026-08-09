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

- [x] `core/recorder.py` — `run_replicate` evaluated `is_terminal(state)` **twice
      per iteration** (loop condition + `terminal_now`). For a Gillespie model
      `is_terminal` ends in an absorbing-state check that re-evaluates the whole
      propensity vector, so every event re-ran the model's `rates` an extra time
      (profiled: `rates` called 3x per event, `propensities` = 41% of runtime).
      Now evaluated once per state and reused; the initial state is still tested
      *before* the loop because a model can be terminal at `t=0` (Wright-Fisher
      initialized at fixation). No RNG stream moves — the birth-death convergence
      config reproduces **bit-identically** (slope -0.48886136889992604, all five
      `D(Omega)`), at 26.4s -> 18.0s. Found while pricing the repressilator, but a
      win for every model.

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

- [x] `models/repressilator.py` — 6-species Hill network, system-size scaled
      propensities, `DeterministicLimitModel` RHS; **no** `analytic_predictions`.
      Species order `(m1,m2,m3,p1,p2,p3)`, repression `p3-|m1, p1-|m2, p2-|m3`;
      exports `OBSERVABLE_KEYS` to pass explicitly to `convergence_report`. A
      cyclically symmetric IC (`p1_0==p2_0==p3_0`) is **rejected** — that manifold
      is invariant and does not oscillate, which would give a flat reference.
- [x] Confirm chosen params oscillate in the ODE (sanity) before convergence test.
      Done as an ODE-only slice *first* (no SSA, so it also priced the sweep via
      `events ~ Omega * int sum_j f_j(c(t)) dt`). Key finding: **`beta=5` does not
      oscillate** — it damps to the fixed point (m1 amplitude 136 -> 60 over 60
      time units); `(beta+1)^2/beta` is minimized at `beta=1`. Chosen
      `alpha=216, alpha0=0.216, n_hill=2, beta=1`; **period 16.095**, amplitude
      ratio last/mid cycle over 19 periods = 1.00000. Richardson over T=300:
      4.4e-10 at `dt=1e-3` (4.5e-6 at `dt=1e-2`) — the reference floor is a
      non-issue here, contrary to the worry that a limit cycle would accumulate
      phase error.
- [x] Convergence test: slope ~ -1/2 across the `Omega` sweep, with the broken-
      propensity teeth. Config measured, not guessed: `t_max` = 2 periods (a
      single-oscillation time average inherits phase-alignment noise);
      `fit_mask` excludes `Omega <= 1`, which sit in the **phase-saturation knee**
      (`D` capped near the fully-dephased `O(amplitude)` ~45, visible as
      `D*sqrt(Omega)` falling *below* the plateau: 22.3, 24.5 vs ~26) — those
      points are still run and printed with a blank flag so the knee stays visible;
      `Omega` pushed to 16 for **lever arm** (OLS slope SE ~
      `sigma/(sqrt(K)*sd(log Omega))`, so widening the range beat adding
      replicates: SE 0.0975 over `[2,8]` vs 0.0734 out to 16 at the same `R`).
      Result seed 0: **slope -0.4606 +/- 0.0734**, CI[3 SE] = [-0.681, -0.240],
      PASS in ~245 s; seed 1 verified offline (-0.5191 +/- 0.1093, PASS), so the
      pass is not seed-luck. Single seed in the suite (a second would dominate it).
      **Teeth (both FAIL, as required):** fixed-`Omega` propensities (never
      threaded through) -> slope **-0.1669 +/- 0.0722**, fails the
      *significantly-negative* leg; `Omega^2` propensities -> slope
      **-1.1315 +/- 0.1850**, which *is* significantly negative and is rejected
      **only** by the *consistent-with--1/2* leg — proving that leg has teeth on
      its own. Both are test-local subclasses overriding `initial_state` to embed
      a wrong system size; `gillespie.py` is untouched and the ODE reference is
      unchanged, isolating the fluctuation scaling alone.
- [x] Non-statistical teeth alongside the slope: sustained limit cycle over ~10
      periods, symmetric-IC rejection, `OBSERVABLE_KEYS` column-matching
      `initial_concentrations` (a silent transposition would give a
      wrong-but-plausible `D` the slope cannot catch), and `deterministic_rhs`
      against the hand-written textbook equations.

## Viz + demo

- [ ] Confirm `plot_replicates` overlays the ODE limit cycle via its existing
      `deterministic=(t_grid, y_grid)` arg (no rewrite — just check units match).
- [ ] `plot_convergence(D, Omega)` — log-log helper showing the `Omega^-1/2` law.
- [ ] `demos/repressilator.py` — validate engine (birth-death/isomerization),
      run the convergence sweep, save the overlay (replicates vs ODE limit cycle)
      and the scaling figure. ASCII-only printed output.

## Wiring + green

- [x] Register `birth_death`, `isomerization`, `repressilator` in
      `models/__init__.py`.
- [x] `uv run pytest -q` green; `uv run ruff check .` clean; `uv run ruff format .`.
      **79 passed in 362 s** — the repressilator convergence check is ~245 s of
      that, which is the price of the phase's headline validation.
- [ ] Demo runs and produces both figures.
- [ ] Update `CLAUDE.md` Status line, memory, and `phase0-...-tasks.md` "Next"
      stub; commit (Conventional Commits) and push per the batch/session-end ritual.

## Explicitly deferred (do NOT do in Phase 1)

- [ ] ~~Gibson-Bruck next-reaction method~~ — Direct Method suffices; profile first.
- [ ] ~~tau-leaping~~ / ~~numba~~ — only against a real bottleneck at large `Omega`.
- [ ] ~~bimolecular reactions~~ (combinatorial propensity correction).
- [ ] ~~scipy stiff integrator~~ — RK4 is enough for the repressilator RHS.
