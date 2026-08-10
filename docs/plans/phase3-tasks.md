# Phase 3 — tasks

Build order is dependency-ordered; each `[ ]` is a commit-sized unit. Per the
workflow rule, **write each validation test first and confirm it can fail** before
the implementation is correct. Numbers recorded here are *measured results*, not
targets — fill them in as each step lands.

## Step 0 — sanity slices (done before any planning)

- [x] **gLV slice.** Three-species reference `r = (1.0, 0.8, 1.2)`,
      `A = [[-1,-.3,-.2],[-.4,-1,-.1],[-.2,-.5,-1]]`:
      `x* = (0.7009569378, 0.4354066986, 0.8421052632)`, `|rhs(x*)| = 1.87e-16`,
      `eig(diag(x*)A) = {-1.0367556167, -0.5911131304, -0.3506001525}`.
      **Non-stiff** (`tau` spans `0.96`-`2.85`) -> reuse `core/ode.py`, no new
      integrator. Relaxation rate vs leading eigenvalue: rel err
      `3.03e-4 / 3.01e-5 / 3.01e-6` at `eps = 1e-2/1e-3/1e-4` — **linear in
      `eps`**, so Richardson in the amplitude. LV predator-prey
      (`1.1, 0.4, 0.4, 0.1`): `(x*,y*) = (4.0, 2.75)`, `eig(J) = +-0.6633249581i`,
      period `2 pi/sqrt(alpha gamma) = 9.472258250995`, conserved `V` drifting
      `3.97e-13`, and the **time-average identity** `<x> = x*` at any amplitude
      (`3.9999934909` at amp 1.2, `3.9998998989` at amp 4.0).
- [x] **May / feasibility slice — the phase's HANDOFF deviation.** Feasibility of
      `x* = -A^{-1} r` collapses with `S`: `0.958 / 0.028 / 0.000` at `S = 20` for
      `sigma = 0.1 / 0.25 / 0.5`, and `0.003 / 0.000 / 0.000` at `S = 80`.
      Conditioning on feasibility **moves the spectrum** — at `S = 20,
      sigma = 0.25` (8183 draws for 200 feasible), `max Re eig(A) = -0.3345`
      (sd 0.1239) vs `max Re eig(diag(x*)A) = -0.1170` (sd 0.0916), `x*` spanning
      67x. **So HANDOFF §6's "May becomes a checkable prediction about your own
      simulated webs" cannot be run as written.**
- [x] **Random-matrix anchor selection.** Spectral radius **rejected**:
      `E[max|lam|]/R = 1.035, 1.044, 1.041, 1.038, 1.031, 1.022` for
      `S = 25...800`, excess log-log slope only `-0.1446`. Circular law
      **accepted**: fraction inside `rho R` equals `rho^2` with `z <= 1.38` for
      `rho = 0.2...0.9` at `S = 400` (10 000 eigenvalues). Elliptic law
      **accepted and preferred**: `z <= 2.31` for `rho = -0.8...+0.8`, and
      `E[max Re]` tracking `R(1+rho)` = `4.22/11.99/19.85/27.69/35.42` vs
      `4/12/20/28/36`. **Finite-`S` bias measured `~ 0.6/S`** (slope `-0.9279`), so
      `S` is *derived* from `0.6/S << SE(n)`, not chosen. `P(stable)` transition
      width measured (`0.322/0.236/0.141/0.070` at `S = 25...200`) and **rejected**
      as an anchor — grid-sensitive and 45 s per point at `S = 400`.
- [x] **Stochastic-gLV pricing.** Priced from the ODE first: `2.6667 * Omega`
      events/time at `x*` -> `53.3 * Omega` per replicate at `t_max = 20`.
      Measured **`slope = -0.4984 +/- 0.0488`** over `Omega in [100,1600]`,
      8 replicates, 1.32 M events in **15.6 s** at **11.8 us/event** — an eighth of
      the repressilator floor, so **zero wall-clock cost at `-n 6`**.
- [x] **Daisyworld slice — the closed form.** Set out to root-find `T*(L)`; found
      the interior equilibrium is **closed-form**. `beta(T_w) = beta(T_b)` forces
      `T_w,T_b = T_opt -+ delta` with `delta` from a depressed cubic in `q` alone:
      `delta = 4.988282516541`, `T_w* = 290.511717483459`,
      `T_b* = 300.488282516541`, `beta* = 0.918757127552`,
      `x* = 0.326528079079` — **none depends on `L`**. `|rhs(y*)| <= 3.68e-16`
      across the band. **`dT_w/dL = dT_b/dL = +0.000e+00` exactly**, and `dT_e/dL`
      is *negative* (`-13.57/-10.77/-8.75` at `L = 0.9/1.0/1.1`) against the bare
      planet's `+81.13/+74.97/+69.80` — overcompensation, not just flattening.
      Regulating band **`L in [0.7387224182, 1.3594723713]`**. Stable throughout.
      Hysteresis/dieback confirmed (down-ramp settles on a white-only `(0.339, 0)`
      at `L = 1.0` where the interior state is `(0.400, 0.273)`) — category C.
- [x] **Adaptive-dynamics slice.** `D(x) = -r x/sK^2`,
      `d2s/dy2 = r(1/sa^2 - 1/sK^2)`, `d2s/dxdy = -r/sa^2`,
      `dD/dx = -1.000000000000` (convergence-stable **always**), all verified
      against finite differences to `7.6e-8`. **Branching iff `sa < sK`**,
      confirmed by simulation at `sa/sK = 0.70, 0.95` (branch) and `1.05, 1.50`
      (ESS). **`t_branch * rate` constant to 1.2%** over a 16x rate range
      (`5689.8, 5620.9, 5625.3, 5629.7, 5682.6`), log-log slope **`-1.0003`**.
      Canonical-equation teeth at 1200 reps: correct `z = 2.36`, drop `1/2`
      `z = 110.84`, omit `K(x)` `z = 371.21`, `sm` for `sm^2` `z = 1074.26`.
- [x] **Five traps found in the slice itself, all green or plausible first:**
      (1) an order-4 test at `t_max = 200` read **ratio 1.00 at every `dt`** —
      the trajectory had reached its attractor, so the "error" was the
      `dt`-independent distance to the fixed point; (2) the order-4 window is
      bounded on **both** sides (ratios `19.8 -> 16.4`, then roundoff); (3) both
      adaptive-dynamics second derivatives had a **sign error** that matched the
      finite difference in magnitude — a `|closed|` vs `|fd|` test would have
      passed; (4) counting local maxima on the trait grid called every `sa < sK`
      case "branched" when the peaks were the seed's grid neighbours, and called
      `t = 4000` a success where nothing had branched; (5) the five-point `sm`
      sweep was a **single-point check in disguise** — holding `mu sm^2 t_max`
      fixed pins the prediction at `1.849492` for every `sm`.
- [x] **`O(1/Omega)` bimolecular bias — measured, and NOT resolved.** Predicted
      `<x>_exact - <x>_macro = x*/Omega` from the effective-`r` shift. Measured
      (`T = 3000`, 10 reps/arm): `Omega = 100` gives `5.043e-3 +/- 2.51e-3` vs
      predicted `6.667e-3` (`z = -0.65`, ratio 0.756) — the only informative row.
      At `Omega = 25, 50` **the SE exceeds the signal** (`2.10e-1` and `3.29e-2`),
      because the SSA's fluctuations grow faster than the bias as `Omega` falls
      (`<x> = 0.427` vs `x* = 0.667` at `Omega = 25`). **The bias cannot be
      measured where it is largest.** Open item — see 3c step 9.
- [x] **Suite baseline re-timed clean — and the recorded 130 s is not
      comparable.** Three runs: `365.51s` **with a slice script on the same cores**
      (invalid — never time the suite against a busy machine), then clean runs of
      `232.18s` and `203.46s` (**~14% run-to-run variance**). `--durations=8` says
      why: the repressilator floor test now costs **162.17s** against the **122 s**
      recorded in Phase 2, so **the machine is ~1.33x slower and nothing
      regressed**. Runners-up: repressilator `Omega^2` tooth 130.45s, HH
      channel-noise slope 39.17s, repressilator fixed-`Omega` tooth 34.78s,
      birth-death slopes 31.63s / 25.89s. **All Phase-3 slice timings are
      same-machine as these, so relative claims hold and absolute seconds do
      not travel.**

## 3a — generalized Lotka-Volterra

- [x] **`models/glv.py`** — gLV RHS on `core/ode.py`; `analytic_predictions` = the
      closed-form interior equilibrium. **The `A` convention was pinned against the
      slice's recorded numbers before the module was written**: row-major
      `A_ij` = effect of `j` on `i` reproduces `x*` and `eig(diag(x*)A)` to all
      recorded digits, the transpose does not. Two hand-built systems, because the
      3-species case is *self-consistency only* — the 2-species symmetric closed
      form `x* = 1/(1+a)`, `eig = x*(-1 -+ a)` (swept at `a = 0, 0.25, 0.5, 0.8`)
      does **not** come from `-A^{-1}r`, and a hand-written double-loop RHS in the
      test file is the independent anchor. Degenerate-SE path handled: two
      replicates, `sem_floor` from Richardson in `dt` — which measures **exactly
      `0.0`** here (`|rhs(x*)| = 1.9e-16` moves `x* ~ 0.7` by `2e-18`, below the
      ULP), so the floor falls back to `1e-13` and that check is *structural*;
      precision lives in the attraction test. **Refuses** on a singular `A`, an
      infeasible `x*`, an unstable `x*`, and — for the relaxation claim only — a
      complex slowest pair.
- [x] **Relaxation rate = slowest eigenvalue of `diag(x*) A`**, Richardson in the
      amplitude. **The slice's constant does not transfer and this was caught by
      re-measuring first:** the slice *fitted* `log|x - x*|` over a window and read
      `3.03e-4` at `eps = 1e-2`; this model's single **endpoint** log-ratio reads
      `4.27e-3` — 14x larger — and drifts with the horizon (`4.27e-3 / 2.28e-3 /
      1.52e-3` at `T = 10/20/30`). The `O(eps)` *scaling* transferred (ratios
      `10.08 / 10.01 / 10.00` per decade), the constant did not, which is exactly
      why the tolerance is derived at runtime. First order, so the factor is
      `2|m(eps) - m(eps/2)|`, not Gray-Scott's `4/3`.
- [x] **`models/lotka_volterra.py`** — conserved `V` through `validate()` at two
      amplitudes with a **per-amplitude** `sem_floor` (drift `9.3e-15` at `amp=0.4`
      vs `1.6e-11` at `amp=4.0`, a factor of 1670 at identical `dt`), the
      time-average identity `<x> = x*` at `amp = 0.4/1.2/4.0`, and the period as an
      extrapolated limit. **The plan's amplitude probes were wrong for this claim**
      — see the next entry.
- [x] RK4 order 4 at the prescribed window `t_max = 5`, `dt in [0.125, 0.03125]`:
      **16.42 / 16.17 / 16.08**, converging on 16 from above.
- [x] **Both models mutation-checked, 13 mutants, all red.** gLV: transpose `A` in
      the RHS only (9 tests), transpose it **everywhere** (6 — and
      `test_validate_reproduces_the_equilibrium` **passes** under it, which is the
      measured proof that the hand-written loop is load-bearing rather than
      decorative), drop the `x_i` prefactor (6), seed the fastest mode instead of
      the slowest (4), drop each of the three guards (1 each). LV: swap
      `beta`/`delta` (10), sign-slip `V` (4), wrong closed-form period (1), swap the
      fixed-point components (8), plus two **test-side** mutants — grid-snapped
      crossings instead of interpolated (4) and an order-4 extrapolant instead of
      order-2 (1) — confirming the interpolation and the extrapolation order are
      themselves load-bearing.

### Two more recorded numbers that did not survive re-measurement

Besides the relaxation constant above, two others. All three were caught by
measuring *before* writing the tolerance, and none is a defect in the slice —
they are places where the slice did not record the convention its numbers
depended on. **Only the scaling travels; the constant does not.**

- **The plan's `t_max = 20` order-4 trap did not reproduce.** The plan warns it
  reads `65.89 / 5.76 / 12.65 / 15.09` — noise from an error collapsed to
  roundoff. From `x_init = (0.1, 0.1, 0.1)` with a max-norm error against a
  `dt = 1e-3` reference it reads **`15.18 / 15.59 / 15.80`**, a perfectly
  measurable approach to 16. The trap depends on the initial condition and the
  error norm, neither of which the slice recorded. `t_max = 5` is used anyway, on
  its own merits: three orders more error (`3.8e-6` vs `2.2e-9`) and ratios
  closer to 16, so it is unambiguously inside the asymptotic regime.
- **The LV period probes `amp = 1.2 / 2.0 / 4.0` are outside the asymptotic
  regime**, and the plan's amplitude scale is not this model's. `amp` is now
  *defined* as the initial displacement in `x` with `y` at `y*`; on that scale the
  slice's `9.4913` occurs at `amp = 0.8` and its `9.8053` at `amp = 4.0`, so the
  two scales are not related by any constant factor. Re-measured, `excess/amp^2`
  runs `0.033360 / 0.033087 / 0.032558 / 0.031555` at `amp = 0.05 / 0.1 / 0.2 /
  0.4` — converging — against `0.0282 / 0.0255 / 0.0208` at the plan's probes,
  which are nowhere near constant. **An extrapolation fitted to the plan's three
  points would not have been measuring a limit.**

## 3b — the random community matrix

- [ ] **Pin the ensemble construction against one recorded row before writing any
      tolerance** — the three-minute check that caught the `A` convention in 3a,
      and the discipline 3a's three non-transferring constants argue for. The
      recorded `z <= 1.38`, `z <= 2.31` and bias slope `-0.9279` depend on the draw
      convention, `C`, and the ellipse-to-disc mapping at `t = 0.5`, none of which
      the slice fully recorded. See [[numbers-travel-with-their-estimator]].
- [ ] `core/random_matrix.py` — May/elliptic ensemble; circular- and elliptic-law
      fraction checks with **binomial** SE and `S` **derived** from
      `0.6/S << SE(n)`. **NOT a `Model`** — no `step`, no `observables`, no
      `state.t`; a plain pytest outside the `ValidationSuite` path, not registered
      in `models/__init__.py`. Do not contrive a `Model` whose `step` draws a
      matrix.
- [ ] Teeth: wrong `R`, wrong `rho` sign, a Hermitian-ized draw. Verify across 3-4
      seeds; assert only the structurally robust leg.
- [ ] Demo reports the `P(stable)` transition and its sharpening — **reported, not
      asserted** — and states plainly that feasibility-conditioned gLV is **not**
      claimed to obey it, with the feasibility table as evidence.

## 3c — stochastic gLV

- [ ] `models/glv_stochastic.py` on the existing Gillespie engine.
- [ ] **Close the `O(1/Omega)` bias measurement**, or label it open in the code
      comment as well as the docs. Needs a variance-reduced estimator (the two arms
      share a common `O(1/Omega)` nonlinearity bias, which is why differencing is
      right and comparing either arm to `x*` alone is not) or ~340 s of SSA.
- [ ] `D(Omega) ~ Omega^{-1/2}` via `core/convergence.py`, `observable_keys`
      explicit, teeth verified across seeds.

## 3d — Daisyworld

- [ ] `models/daisyworld.py` — `delta` by Cardano; `analytic_predictions` =
      `T_w*`, `T_b*`, `x*`, `a_w*(L)` and **`dT_w/dL = 0`**. Must **raise** outside
      `L in [0.7387224182, 1.3594723713]`.
- [ ] Order-4 on a **transient** (`t_max = 20`, never 200); record the `beta`-clip
      `C^0` note (measured not to bite on the `L = 1.0` transient).
- [ ] Category C: hysteresis, dieback, bare-planet comparison — demo only, never in
      `analytic_predictions`.

## 3e — adaptive dynamics

- [ ] `models/adaptive_dynamics.py` — invasion fitness, selection gradient and both
      second derivatives in closed form, checked **signed** against finite
      differences.
- [ ] Branch/no-branch sign change across `sa = sK` with the **gap** criterion and
      the measured horizons (`sa = 0.95` needs `t = 200 000`).
- [ ] `t_branch * rate = const` (slope `-1.0003`) — assert **the exponent, not the
      prefactor** (`log(threshold/seed)`, not universal). **Fix the detection
      resolution first:** the slice checked for a gap every 200 steps at
      `dt = 0.5`, quantizing `t_branch` to 100 time units (every measured value
      ends in `.5`), so at `sa = 0.60` the `3200.5` carries **+-1.6%** while the
      product is reported constant to 1.2%. **Asserting 1.2% would be asserting
      below the measurement's own resolution** — the "threshold nothing can fail"
      discipline pointed the other way. Either shrink the check interval or derive
      the tolerance from it.
- [ ] **Re-measure the `sm` sweep at 1200-reps-equivalent across at least three
      `sm` BEFORE writing the test** (~30 s of compute). The `O(sm)` claim is
      currently *consistent with* the data, not measured: ratios `2.15` and `2.19`
      come from low-replicate points, only `sm = 0.0125` has 1200 reps, and the
      `sm = 0.025` point is non-monotone. Treat it exactly like the `O(1/Omega)`
      bias — **open until measured**.
- [ ] Canonical equation as an **`O(sm)` vanishing discrepancy**, not a single-`sm`
      tolerance — a tolerance at one `sm` was measured to tighten onto a real
      `+0.004` bias and would fail a correct model as replicates grow.
- [ ] Label the post-branching morph structure exploratory, with the
      grid-dependence measurement (`+-0.100 / +-0.050 / +-0.025` at
      `ngrid = 81/161/321`) as the stated reason, and the positive-definiteness
      explanation labelled **category C**.

## All

- [ ] Re-time the suite; update `CLAUDE.md`, memory, docs; commit and push.
      **Use `--durations=8` and settle the open 3a anomaly:** the post-3a run read
      **171.05 s** against a 203-232 s baseline — *faster* despite 47 new tests, on
      a machine that was not idle. If the repressilator floor test also dropped
      from its recorded 162.17 s, it is the machine and nothing else. If the floor
      held near 162 s while the total fell, **xdist packing changed when 47 fast
      tests entered collection** — a real finding, and the same mechanism that made
      the Phase-1 `-n` "fix" measure 75% worse.

## Measurements to record as they land

| Item | Target | Measured |
|---|---|---|
| Suite baseline before Phase 3, clean, `-n 6` | — | 203.46 / 232.18 s (floor test 162.17 s) |
| gLV RK4 order, `t_max = 5` | 4 | 17.02 / 16.54 / 16.28 (slice); **16.42 / 16.17 / 16.08 (built)** |
| gLV RK4 order, `t_max = 20` | noise | 65.89 / 5.76 / 12.65 / 15.09 (slice); **15.18 / 15.59 / 15.80 (built) — did not reproduce** |
| gLV relaxation vs slowest eigenvalue | `O(eps)` | 3.03e-4 / 3.01e-5 / 3.01e-6 (slice, *fitted window*); **4.27e-3 / 4.24e-4 / 4.24e-5 / 4.23e-6 (built, *endpoint ratio*)** |
| gLV relaxation: Richardson-predicted / true error | 1 | **0.9954 / 0.9977 / 0.9991 / 0.9995** at `eps = 1e-2 / 5e-3 / 2e-3 / 1e-3` |
| gLV `validate()` Richardson-in-`dt` bound | — | **exactly 0.0** (`x*` sits below the ULP of its own drift) |
| LV period excess / `amp^2` | const as `amp -> 0` | **0.033360 / 0.033087 / 0.032558 / 0.031555** at `amp = 0.05 / 0.1 / 0.2 / 0.4` |
| LV amplitude-extrapolated period vs closed form | 0 | **5.35e-5 / 7.06e-6 / 9.08e-7** for `a = 0.4 / 0.2 / 0.1` (residual ratio ~7.6) |
| LV `V` drift over `t = 100` at `dt = 0.01` | 0 | **9.3e-15** (`amp = 0.4`), **1.6e-11** (`amp = 4.0`) |
| LV cycle average `\|<x> - x*\|` at `dt = 2.5e-3` | 0 | **3.2e-10 / 3.7e-9 / 4.0e-8** at `amp = 0.4 / 1.2 / 4.0` |
| 3a mutants confirmed red | all | **13 / 13** (11 model-side, 2 test-side) |
| Circular-law fraction, `z` | < 4 | <= 1.38 (slice) |
| Elliptic-law fraction, `z` | < 4 | <= 2.31 (slice) |
| Circular-law finite-`S` bias | — | `~ 0.6/S`, slope -0.9279 (slice) |
| Stochastic gLV `D(Omega)` slope | -1/2 | -0.4984 +/- 0.0488 (slice) |
| Daisyworld `\|rhs(y*)\|` | 0 | <= 3.68e-16 (slice) |
| Daisyworld `dT_w/dL` | 0 | +0.000e+00 (slice) |
| Adaptive-dynamics derivatives vs FD | 0 | <= 7.6e-8 (slice) |
| `t_branch * rate` log-log slope | -1 | -1.0003 (slice) |
| Suite after Phase 3, `-n 6` | < baseline + new, **re-timed same-session** | *(pending)* |
