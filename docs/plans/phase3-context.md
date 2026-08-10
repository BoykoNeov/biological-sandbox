# Phase 3 — context (key files & decisions for a fresh session)

Read `phase3-plan.md` first for the full rationale and every measured number. This
is the quick orientation: where new code lives, the contracts it must honor, and
the traps.

## What Phase 3 adds (planned layout)

| Concern | File | New? |
|---|---|---|
| Generalized Lotka-Volterra (deterministic) | `src/sandbox/models/glv.py` | new |
| LV predator-prey exact invariants | `src/sandbox/models/lotka_volterra.py` | new |
| May / Allesina-Tang random community matrix | `src/sandbox/core/random_matrix.py` | new |
| Demographic-noise gLV (Gillespie) | `src/sandbox/models/glv_stochastic.py` | new |
| Daisyworld | `src/sandbox/models/daisyworld.py` | new |
| Adaptive dynamics / evolutionary branching | `src/sandbox/models/adaptive_dynamics.py` | new |
| Demos | `src/sandbox/demos/{glv,daisyworld,adaptive_dynamics}.py` | new |

**Nothing goes in `models/ecosystem/`.** HANDOFF §6 calls all three of these "still
verifiable"; the measurements agree. The quarantine staying empty is the correct
outcome of this phase, not an omission.

## The four ideas that drive every choice here

1. **May's criterion was reframed, not dropped** — the Phase-3 analogue of Phase
   2's Gray-Scott reframe. May's result is about a random matrix *assumed to be* a
   Jacobian, not about a gLV whose equilibrium came from random `A` and `r`. The
   random-gLV ensemble is **empty** at the `S` where the criterion means anything
   (feasibility 0.000 at `S = 40, sigma = 0.25`), and conditioning on feasibility
   **moves the spectrum** (`max Re eig(A) = -0.335` vs
   `max Re eig(diag(x*)A) = -0.117`). So: validate the **matrix law directly**,
   validate **gLV on hand-built systems**, and **do not claim** the composition.
2. **Daisyworld's regulation is closed-form, not qualitative.** `beta(T_w) =
   beta(T_b)` forces `T_w, T_b = T_opt -+ delta` with `delta` from a cubic in `q`
   alone, so `T_w*`, `T_b*` and the bare fraction `x*` are **independent of
   luminosity**. `dT_w/dL = dT_b/dL = 0` **exactly** — measured `+0.000e+00`.
3. **Branching is a sign change at `sa = sK`** — the `lambda(q)` pattern again. The
   *criterion* is the claim; the post-branching **morph positions are measured
   grid-dependent** and are not claimed.
4. **Richardson in the amplitude, for the third time.** gLV relaxation,
   Daisyworld relaxation, and the canonical equation are all *linearizations* or
   *limits*, so the discrepancy is `O(eps)` / `O(sm)` and the honest instrument is
   the small-parameter sweep, never a tolerance at one value.

## Contracts the new code must honor

Everything from Phase 0-2 still holds (stateless model, params embedded in state by
`initial_state`, `step` writes `state.t`, register in `models/__init__.py`,
plain-number JSON-serializable params, `spawn_rngs` never `seed + i`,
`analytic_predictions` keys name observables checked at their *final* value,
category C never enters `analytic_predictions`). Phase-3 additions:

- **`observables()` must stay scalar for `S` species.** Do not emit one key per
  species for large `S`. Use summaries — survivor count, total biomass,
  `norm(x - x*)`, leading eigenvalue — and when the convergence pathway is
  involved pass explicit `observable_keys`, because `_per_replicate_discrepancy`
  weights species **equally** and mixed scales silently reweight the check (the
  Phase-2 `V`-swamps-the-gates lesson).
- **Deterministic models hit `validate()`'s degenerate-SE path.** With no
  stochasticity every replicate returns the identical value, so `sem = 0`. Supply
  a numerical `sem_floor` **and use two replicates** — one replicate gives
  `sem = inf` and the check passes vacuously.
- **A fourth time-advance discipline is NOT introduced.** gLV, Daisyworld and the
  adaptive-dynamics trait grid are all fixed-`dt` RK4 models, exactly like Phase
  2's. The trait-substitution sequence *is* new (an event-driven jump process with
  a sampled waiting time) but that is Gillespie's discipline, not a new one.
- **`analytic_predictions` must raise outside the regulating band.** Daisyworld
  has no interior equilibrium for `L` outside `[0.7387224182, 1.3594723713]`;
  returning a wrong-but-green number there is the failure Phase 2 established the
  raise-instead discipline for.

## Measured constants (so you don't re-derive them)

**Generalized Lotka-Volterra.** Three-species reference case `r = (1.0, 0.8, 1.2)`,
`A = [[-1,-.3,-.2],[-.4,-1,-.1],[-.2,-.5,-1]]`:

- `x* = (0.7009569378, 0.4354066986, 0.8421052632)`, `|rhs(x*)| = 1.87e-16`,
  `eig(diag(x*) A) = {-1.0367556167, -0.5911131304, -0.3506001525}`.
- Non-stiff: `tau` spans only `0.96` to `2.85`. Reuse `core/ode.py`.
- Relaxation rate vs leading eigenvalue: rel err `3.03e-4 / 3.01e-5 / 3.01e-6` at
  `eps = 1e-2 / 1e-3 / 1e-4` — **exactly linear in `eps`**.
- **RK4 order-4 window is bounded on BOTH sides: use `t_max = 5`, `dt in [0.125,
  0.03125]`** (ratios `17.02, 16.54, 16.28`). At `t_max = 20` the error has
  collapsed to roundoff and the ratios read `65.89, 5.76, 12.65, 15.09`.

**LV predator-prey** (`alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1`):
`(x*, y*) = (4.0, 2.75)`, `eig(J) = +-0.6633249581i`, small-oscillation period
`2 pi/sqrt(alpha gamma) = 9.472258250995`. Conserved `V` drifts `3.97e-13` over
`t = 400` at `dt = 0.01`. **Time-average identity** `<x> = x*` holds at any
amplitude (`3.9999934909` at amp 1.2, `3.9998998989` at amp 4.0) — the residual is
**cycle-endpoint detection**, not the integrator. The period *grows* with
amplitude (`9.491 / 9.805 / 11.271` at amp `1.2 / 2.0 / 4.0`), so the closed-form
period is a **limit**, checked by extrapolation.

**Random community matrix** (`M = -d I + B`, off-diagonals nonzero w.p. `C`,
variance `sigma^2`, radius `R = sigma sqrt(SC)`):

- **Circular law** — fraction inside `rho R` is `rho^2`. At `S = 400`, 10 000
  eigenvalues: `z <= 1.38` for `rho = 0.2 ... 0.9`.
- **Elliptic law** (correlated pairs, correlation `rho`) — ellipse semi-axes
  `R(1+rho)`, `R(1-rho)`; stability needs `sigma sqrt(SC)(1+rho) < d`. Validated
  for `rho = -0.8 ... +0.8`, `z <= 2.31`, with `E[max Re]` tracking `R(1+rho)`
  (`4.22 / 11.99 / 19.85 / 27.69 / 35.42` vs `4 / 12 / 20 / 28 / 36`).
- **The finite-`S` bias is `~ 0.6 / S`** (measured slope `-0.9279`; predicts
  `1.2e-2, 6.0e-3, 3.0e-3, 1.5e-3` at `S = 50, 100, 200, 400`, matching every
  row). **Derive `S` from `0.6/S << SE(n)`.** At `S = 400` with 20 000 eigenvalues
  bias and SE are equal — that is the boundary, not a safe point.
- **Rejected as anchors, with the measurement:** the spectral radius
  (`E[max|lam|]/R` stuck at `1.02-1.04`, excess slope only `-0.1446`) and the
  `P(stable)` transition width (grid-sensitive; `S = 400` returned a meaningless
  `0.229`; 45 s per point).

**Stochastic gLV** (`r = (1,1)`, `A = [[-1,-.5],[-.5,-1]]`, `x* = 2/3`):

- Total propensity at `x*` is `2.6667 * Omega`/time -> `53.3 * Omega` events per
  replicate at `t_max = 20`.
- `slope = -0.4984 +/- 0.0488` over `Omega in [100, 1600]`, 8 replicates, **15.6 s**
  at **11.8 us/event**.
- **The `O(1/Omega)` macroscopic-propensity bias is confirmed in magnitude at
  `Omega = 100` only** (`z = -0.65`, ratio 0.756) and its `1/Omega` scaling is
  **not resolved** — at `Omega = 25, 50` the SE exceeds the signal. **The bias
  cannot be measured where it is largest.** Open item; do not write "confirmed".

**Daisyworld** (`S = 917`, `q = 2.06e9`, `gamma = 0.3`, albedos `0.75/0.25/0.5`,
`T_opt = 295.5`, `beta` coefficient `0.003265`):

- `delta = 4.988282516541`, `T_w* = 290.511717483459`, `T_b* = 300.488282516541`,
  `beta* = 0.918757127552`, `x* = 0.326528079079` — **none depends on `L`**.
- `A*(L) = (k - T_w*^4 - q A_w)/(k - q)` with `k = S L / sigma`; then
  `a_w = (A* - (1-x*) A_b - x* A_g)/(A_w - A_b)`. `|rhs(y*)| <= 3.68e-16`.
- **Regulating band `L in [0.7387224182, 1.3594723713]`**, width `0.620750`.
- `dT_e/dL` is **negative** across the band (`-13.57 / -10.77 / -8.75` at
  `L = 0.9 / 1.0 / 1.1`) vs the bare planet's `+81.13 / +74.97 / +69.80`.
- Stable throughout; relaxation rel err `4.33e-4 / 4.37e-5 / 4.37e-6` at
  `eps = 1e-2/1e-3/1e-4`.
- **Trap: an order test at `t_max = 200` reads ratio 1.00 at every `dt`** — the
  trajectory has reached its attractor and the "error" is the distance to the fixed
  point, which is `dt`-independent. Use `t_max = 20` (`15.3, 15.6, 15.8`).
- `beta`'s clip at zero makes the RHS only `C^0`, but on the `L = 1.0` transient it
  **never bites** (clipped and smooth give bit-identical results). It would bite on
  a dieback trajectory.

**Adaptive dynamics** (`K(x) = K0 exp(-x^2/2sK^2)`, `a(x,y) = exp(-(x-y)^2/2sa^2)`,
`s_x(y) = r(1 - a(y,x) K(x)/K(y))`):

- `D(x) = -r x / sK^2`; `d2s/dy2 = r(1/sa^2 - 1/sK^2)`; `d2s/dxdy = -r/sa^2`;
  `dD/dx = -r/sK^2 = -1.000000000000` (convergence-stable **always**).
- **Branching iff `sa < sK`.** Verified against finite differences to `7.6e-8`.
- **`t_branch * (splitting rate)` is constant to 1.2%** across a 16x range
  (`5689.8, 5620.9, 5625.3, 5629.7, 5682.6` at `sa = 0.60 ... 0.95`), log-log slope
  **`-1.0003`**. The **exponent is the claim; the prefactor is
  `log(threshold/seed)` and is not universal.**
- Horizons: `sa = 0.7` branches by `t = 40 000`; **`sa = 0.95` needs
  `t = 200 000`**. Trimming the horizon trims the sign change.
- Canonical equation teeth (1200 reps, `sm = 0.0125`): correct `z = 2.36`, drop the
  `1/2` `z = 110.84`, omit `K(x)` `z = 371.21`, `sm` for `sm^2` `z = 1074.26`.

## The traps this phase already walked into (all five were green or plausible)

1. **An order-4 test on a converged attractor** reads ratio `1.00` at every `dt`
   and looks like a clean result. Only a **transient** can see discretization
   error.
2. **The order-4 window is bounded on both sides** — too-coarse `dt` leaves the
   asymptotic regime (ratios `19.8`), too-fine hits roundoff. Which grids are
   inside the window is part of the claim (the Phase-2 Laplacian lesson).
3. **Closed form vs finite difference agreed in magnitude and disagreed in sign**
   for both adaptive-dynamics second derivatives. A test comparing `|closed|` to
   `|fd|` would have passed. **Compare signed values.**
4. **Counting local maxima on a trait grid** reported "2 peaks = branching" for
   every `sa < sK` — but the peaks were the seed's two immediate grid neighbours,
   and it reported success at `t = 4000` where nothing had branched. The **gap
   criterion** (clusters separated by genuinely dead bins) changed the answer.
5. **A five-point `sm` sweep that was a single-point check in disguise** — holding
   `mu sm^2 t_max` fixed pins the canonical prediction at `1.849492` for *every*
   `sm`, so nothing varied but the noise.

And one that is not a trap but a limit: **the `O(1/Omega)` bias cannot be measured
where it is largest**, because the SSA's fluctuations grow faster than the bias as
`Omega` falls.

## Environment / gotchas (carried forward)

- Windows console is cp1252 — **no non-ASCII in `print`ed strings** (`Omega`,
  `lambda`, `sigma`, `delta`, `<x>`). Docstrings/comments may be UTF-8.
- `uv` venv; numpy 2.x; matplotlib only under `--extra viz`; **no scipy** (RK4 is
  hand-rolled and measured sufficient again here — gLV and Daisyworld are both
  non-stiff).
- Suite baseline: see `phase3-tasks.md` for the clean re-time. **A run taken with a
  slice script on the same cores read `310 passed in 365.51s` against a recorded
  130 s — never time the suite against a busy machine.**
- Teeth tests must be **verified across 3-4 seeds**, and assert only the leg that is
  *structurally* robust for that break.
- Any optimization must be **bit-identical**, fingerprinted (`sha256` of
  times + series) before and after.
- Temp/scratch work goes in `M:\claud_projects\temp\` — the Phase-3 slice lives in
  `M:\claud_projects\temp\phase3-slice\` and is **not** in the repo.
