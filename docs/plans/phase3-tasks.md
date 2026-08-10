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
- [x] **Suite baseline re-timed clean.** A run taken with a slice script on the
      same cores read `310 passed in 365.51s` against a recorded 130 s. **Never
      time the suite against a busy machine.** Clean figure: see below.

## 3a — generalized Lotka-Volterra

- [ ] `models/glv.py` — gLV RHS on `core/ode.py`; `analytic_predictions` = the
      closed-form interior equilibrium of a hand-built system. Test first, red
      first. Remember the **degenerate-SE path**: deterministic, so supply a
      numerical `sem_floor` and use **two** replicates.
- [ ] Relaxation rate = leading eigenvalue of `diag(x*) A`, with **Richardson in
      the amplitude** (error measured linear in `eps`).
- [ ] `models/lotka_volterra.py` — conserved `V`, the **time-average identity**
      `<x> = x*` at any amplitude (interpolate the cycle endpoints; that, not the
      integrator, is the error floor), and the small-oscillation period as an
      **extrapolated limit** (it grows with amplitude: `9.491/9.805/11.271`).
- [ ] RK4 order 4 at the **prescribed window `t_max = 5`, `dt in [0.125,
      0.03125]`** — never `t_max = 20`.

## 3b — the random community matrix

- [ ] `core/random_matrix.py` — May/elliptic ensemble; circular- and elliptic-law
      fraction checks with **binomial** SE and `S` **derived** from
      `0.6/S << SE(n)`.
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
      prefactor** (`log(threshold/seed)`, not universal).
- [ ] Canonical equation as an **`O(sm)` vanishing discrepancy**, not a single-`sm`
      tolerance — a tolerance at one `sm` was measured to tighten onto a real
      `+0.004` bias and would fail a correct model as replicates grow.
- [ ] Label the post-branching morph structure exploratory, with the
      grid-dependence measurement (`+-0.100 / +-0.050 / +-0.025` at
      `ngrid = 81/161/321`) as the stated reason, and the positive-definiteness
      explanation labelled **category C**.

## All

- [ ] Re-time the suite; update `CLAUDE.md`, memory, docs; commit and push.

## Measurements to record as they land

| Item | Target | Measured |
|---|---|---|
| Suite baseline before Phase 3, clean, `-n 6` | — | *(see below)* |
| gLV RK4 order | 4 | 17.02 / 16.54 / 16.28 (slice) |
| gLV relaxation vs leading eigenvalue | `O(eps)` | 3.03e-4 / 3.01e-5 / 3.01e-6 (slice) |
| Circular-law fraction, `z` | < 4 | <= 1.38 (slice) |
| Elliptic-law fraction, `z` | < 4 | <= 2.31 (slice) |
| Circular-law finite-`S` bias | — | `~ 0.6/S`, slope -0.9279 (slice) |
| Stochastic gLV `D(Omega)` slope | -1/2 | -0.4984 +/- 0.0488 (slice) |
| Daisyworld `\|rhs(y*)\|` | 0 | <= 3.68e-16 (slice) |
| Daisyworld `dT_w/dL` | 0 | +0.000e+00 (slice) |
| Adaptive-dynamics derivatives vs FD | 0 | <= 7.6e-8 (slice) |
| `t_branch * rate` log-log slope | -1 | -1.0003 (slice) |
| Suite after Phase 3, `-n 6` | < 130 s + new | *(pending)* |
