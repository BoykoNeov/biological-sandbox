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
      **Teeth (both FAIL, as required), each verified across seeds 0-3** — a tooth
      that only bites on the pinned seed proves nothing, and the first draft of the
      `Omega^2` one was exactly that (it *passed* at 2 of 4 seeds). Both are
      test-local subclasses overriding `initial_state` to embed a wrong system
      size; `gillespie.py` is untouched and the ODE reference is unchanged, so only
      the fluctuation scaling differs.
      - fixed-`Omega` propensities (never threaded through) -> slope
        **-0.1363 +/- 0.0766**. Asserted on the *consistent* leg plus a scale-free
        `D(8)/D(2) > 0.65` anchor (the law demands ~0.5; the real model measures
        0.46). The *significantly-negative* leg is deliberately **not** asserted:
        it trips when `slope/SE < -z`, and since slope and SE **both** scale as
        `1/sqrt(R)` that ratio is replicate-independent — more replicates cannot
        make it safer. Measured `slope/SE` over seeds 0-3: -1.78, -0.25, +2.25,
        +1.53, i.e. ~1.7x wider than the nominal SE implies.
      - `Omega^2` propensities -> slope **-0.9904 +/- 0.0774**, `|slope+1/2|`
        0.4904 vs tolerance 0.2322 (2.11x). *Significantly negative* stays True, so
        this is rejected **only** by the *consistent-with--1/2* leg — proving that
        leg has teeth on its own. Here replicates *do* help (slope sits at a real
        -1 while SE shrinks), so `R=24` and swept `Omega` out to 4 for lever arm;
        margins over seeds 0-3: 2.11x, 1.78x, 1.67x, 2.19x.
- [x] Non-statistical teeth alongside the slope: sustained limit cycle over ~10
      periods, symmetric-IC rejection, `OBSERVABLE_KEYS` column-matching
      `initial_concentrations` (a silent transposition would give a
      wrong-but-plausible `D` the slope cannot catch), and `deterministic_rhs`
      against the hand-written textbook equations.

## Viz + demo

- [x] Confirm `plot_replicates` overlays the ODE limit cycle via its existing
      `deterministic=(t_grid, y_grid)` arg. **No rewrite needed** — `observables()`
      returns concentrations and the ODE is integrated in concentration space, so
      the two align unrescaled. What it needed was a *test*, and the live trap
      turned out not to be units but the **column index**: `deterministic` takes a
      single 1-D series, so a `(n_t, 6)` ODE solution must be sliced, and on this
      cyclically symmetric limit cycle the wrong protein column is the right shape
      at the wrong phase. `test_viz.py` runs one real replicate at `Omega=20`
      (1.5 s), reads both series back **off the Axes**, and asserts it tracks the
      overlay; the neighbouring column (4.9x-45x worse) and counts-vs-concentrations
      (40x-440x) are the teeth. Thresholds measured across seeds 0-3, where the
      discrepancy/amplitude ratio spans 10x on phase-diffusion luck (0.097, 0.010,
      0.023, 0.011) — so the pinned seed is **0, the worst of the four**. The scale
      check uses `std`, not `ptp`: `ptp` is a max-statistic one SSA spike moves,
      leaving 1.2x margin against a counts-scaled mutant where `std` leaves 2.1x.
      Also documented: a mistyped observable draws *nothing* while still drawing the
      deterministic line (a clean-looking figure of no data).
- [x] `plot_convergence(omegas, discrepancy, ...)` — log-log helper showing the
      `Omega^-1/2` law. Takes **plain arrays, not a `ConvergenceReport`**, so its own
      test costs zero SSA time (the suite is already 155 s); the demo unpacks a
      report in one call. Two design points carry `convergence.py`'s stance into the
      figure: masked-out points are still **drawn** (hollow markers — the knee is
      evidence about where the law stops applying, and the figure is where that is
      most visible), and both the fit line and the `-1/2` guide are anchored at the
      **centroid of the fitted points**. Anchoring at the first point instead pins
      the guide to the saturated knee and makes a passing check look like a failing
      one — mutation-checked, that break fails the test. Artists carry `gid`s
      (`FIT_GID`/`GUIDE_GID`/...) so tests locate them without indexing into
      `ax.lines` (errorbar adds several artists) or matching legend prose; errorbar
      puts its label on the *container*, not the line, hence the explicit tag.
      All three viz tests were mutation-checked (ignore the mask; anchor at the
      knee; overlay in counts) — each mutation fails the intended test.
- [x] `demos/repressilator.py` — three acts: engine vs exact closed forms
      (`birth_death`, `isomerization` via `validate`), the overlay figure
      (`Omega=1` vs `Omega=8`, 6 replicates each over 2 periods, `record_every=20`),
      and the convergence sweep + log-log figure. ASCII-only output, about a minute
      on an idle machine (the convergence sweep alone measured 22.4 s).
      The demo's convergence config is **reduced** (`R=6`, 1 period, out to
      `Omega=16`) and says so in its own output — the authoritative check stays
      `tests/test_repressilator.py`. It was seed-checked before being pinned so the
      demo cannot print a red check on an unlucky draw: PASS at seeds 0, 1, 2 with
      slopes -0.399, -0.411, -0.443 (~23 s each). `record_every` cuts both ways and
      the two uses want opposite values — 1 for the slope run (sub-sampling sharpens
      the interpolation *with* `Omega` and biases the slope), 20 for the figure.
      The demo also prints the compensated `D*sqrt(Omega)` column so the knee is
      visible as numbers, not only as hollow markers.

## Wiring + green

- [x] Register `birth_death`, `isomerization`, `repressilator` in
      `models/__init__.py`.
- [x] `uv run pytest -q` green; `uv run ruff check .` clean; `uv run ruff format .`.
      **79 passed in 155 s.** It first landed at 637 s (5x the pre-repressilator
      122 s), which was too slow a development loop, so two things were done rather
      than hiding tests behind a `slow` marker:
      - **SSA optimization, 1.77x** (40.6 -> 22.9 us/event), all **bit-identical**
        — verified by sha256-fingerprinting a replicate's times+series before and
        after, and by re-running the birth-death anchor. Two changes: the
        repressilator's `is_terminal` dropped its `a0 == 0` absorbing check (a full
        propensity evaluation on the hot path, doubling `rates()` calls per event,
        for a branch that *provably* cannot fire since `alpha/(1+p^nH) + alpha0 > 0`
        always), and `observables` builds its dict from one `tolist()` instead of a
        per-element generator. `Trajectory.record` avoiding `setdefault`'s throwaway
        list turned out to be a wash (0.443 vs 0.449 us) — kept for clarity only.
      - **`-n 4` (pytest-xdist) at below-normal process priority** (set in
        `tests/conftest.py`, per-worker, best-effort). Wall clock floors at the
        longest single test (~140 s), which 4 workers already reach, so `-n auto`
        would claim all 16 cores for nothing. 388 s -> 155 s.
- [x] Demo runs and produces both figures (`repressilator_overlay.png`,
      `repressilator_convergence.png`; both gitignored). Verified by running it and
      *looking at* the output, not just at the exit code. The overlay shows one
      feature worth naming rather than glossing over: at small `Omega` the replicate
      peaks sit systematically **above** the ODE rather than symmetrically around it.
      That is a finite-size correction, not a bug — the Hill term is strongly
      nonlinear, so the mean of the stochastic system is not the solution of the
      mean-field equation — and it vanishes in the limit: the mean second-cycle peak
      of `x_p1` measures **+29.2% at Omega=1, +10.9% at 4, +1.0% at 16**. Measured,
      then written into the demo's own printed output.
- [ ] Update `CLAUDE.md` Status line, memory, and `phase0-...-tasks.md` "Next"
      stub; commit (Conventional Commits) and push per the batch/session-end ritual.

## Explicitly deferred (do NOT do in Phase 1)

- [ ] ~~Gibson-Bruck next-reaction method~~ — Direct Method suffices; profile first.
- [ ] ~~tau-leaping~~ / ~~numba~~ — only against a real bottleneck at large `Omega`.
- [ ] ~~bimolecular reactions~~ (combinatorial propensity correction).
- [ ] ~~scipy stiff integrator~~ — RK4 is enough for the repressilator RHS.
