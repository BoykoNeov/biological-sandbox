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
      > **Closed in 3c, and the formula above needed generalizing first.**
      > `x*/Omega` is not a separate derivation — it is `(-A)^-1 diag|A_ii| 1 /
      > Omega` specialized to a system where `diag|A_ii| 1 = r`, which the plan's
      > symmetric 2-species reference satisfies and 3a's asymmetric 3-species one
      > does not (`+1.0% / -35% / +60%` off there). The "cannot be measured where
      > it is largest" finding **survives the better estimator**: even split-coupled
      > at `T = 500`, `Omega = 50` is unresolved (`-7.19e-4 +/- 1.13e-2`).
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

- [x] **Pin the ensemble construction against one recorded row before writing any
      tolerance** — done, and it paid. The convention was **recovered from the
      tables rather than guessed**: the elliptic `pred R(1+rho)` column
      `4/12/20/28/36` solves to `R = 20` at `S = 400`, so `sigma sqrt(C) = 1`, i.e.
      `C = 1, sigma = 1`; and the `t = 0.5` mapping is
      `(Re/(R(1+rho)), Im/(R(1-rho)))` inside radius `t`, since only that gives the
      recorded `pred = t^2 = 0.25`. Under that convention both tables reproduce,
      including the `E[max Re]` column — which is the real pin, being sensitive to
      the draw convention in a way the fraction column is not (the fraction is
      **insensitive to `d` and to the overall scale of `sigma`/`C`**, both absorbed
      into `R`, which is exactly what makes a `rho^2` failure diagnostic of the
      *draw*).
- [x] **The recorded bias constant did not survive re-measurement — and this is a
      fourth instance of [[numbers-travel-with-their-estimator]], with a new
      mechanism.** Re-running the plan's own estimator (`rho = 0.5`, eigenvalue
      count held at 40 000) reproduced the bias at `S = 50/100/200` (`z = 6.48,
      3.33, 1.51`) but **not** at `S = 400/800`, where the measured bias sits
      *below its own SE* (`z = 0.69, 0.90`, SE `2.17e-3`). Those two rows are not
      measurements of a bias; the recorded slope `-0.9279` was fitted partly
      through noise. Re-measured at 200 000 eigenvalues over `S = 25/50/100/200`,
      where every row carries `z >> 1` (`27.53, 13.85, 7.58, 3.27`), the law is
      **`bias = 0.70/S`, slope `-1.0086`** — i.e. exactly `1/S`, and a ~17% larger
      constant. The plan's own sentence ("at `S = 400` and 20 000 eigenvalues bias
      and SE are the same size — the boundary, not a safe point") was *understating*
      it: at 40 000 eigenvalues `S = 400` is already past the boundary. Use
      `0.70/S`, and note the irony that **the bias is only measurable at the small
      `S` where the asymptotic law it corrects is least valid.**
- [x] **`0.70/S` is itself a `rho = 0.5` artifact, and the bias-negligible rule
      cannot be met.** Two measurements, both at two seeds:
      - **`c(rho) = bias * S` is not constant in `rho`** — at `S = 100`, 200 000
        eigenvalues, it runs `0.484 / 0.611 / 0.825 / 0.874 / 1.126 / 1.105 / 0.272`
        for `rho = 0.2 / 0.4 / 0.5 / 0.6 / 0.8 / 0.9 / 0.95` (all `z > 4`).
      - **Worse, `c(rho)` is not constant in `S` either, except at `rho = 0.5`.**
        First read from two sizes, which was **not enough** — a two-point slope
        gave `S^-0.67` with a `+-0.3` ratio uncertainty, putting `-1.0` only
        ~1.4 sigma out. *The same error this section catches in the slice.*
        Re-measured over **four** sizes (`S = 25/50/100/200`), 200 000 eigenvalues,
        two seeds, every point at `z > 4`:

        | probe | exponent | `c = bias * S`, `S = 25 -> 200` |
        |---|---|---|
        | circular `rho = 0.2` | `-0.8814 +- 0.0250` | 0.386 / 0.417 / 0.430 / 0.503 |
        | circular `rho = 0.5` | `-1.0274 +- 0.0231` | 0.692 / 0.722 / 0.696 / 0.658 — **flat** |
        | circular `rho = 0.9` | `-0.6189 +- 0.0673` | 0.589 / 0.906 / 1.143 / 1.315 |
        | elliptic `rho = -0.8` | `-0.7733 +- 0.1472` | 0.414 / 0.464 / 0.405 / 0.731 — non-monotone |
        | elliptic `rho = +0.8` | `-0.8404 +- 0.0150` | 2.729 / 3.169 / 3.504 / 3.816 |

        The exponents **do** differ — `rho = 0.9` vs `rho = 0.5` is 5.7 sigma
        apart — and `rho = 0.5` is the exception, the only probe with `c` flat and
        the exponent at `-1`. **But the tempting mechanism is wrong.** A bulk/edge
        split (an `O(1/S)` bulk correction plus a slower one inside a Ginibre edge
        layer of width `~S^-1/2`) predicts `rho = 0.2` deep in the bulk at `-1`;
        it measures `-0.8814`, shallower than `rho = 0.5`. The ordering is not
        monotone in `rho`, so the exponent is recorded as **probe-dependent and
        unexplained**, not as a two-term law.
      - **The elliptic bias is a different animal at `rho = +0.8`**: `bias * S` is
        `3.50` at `S = 100` and `3.90` at `S = 200` (decay `~S^-0.75`), against a
        circular peak of `1.15`. Extrapolating to `S = 400` predicts `1.24e-2` and
        the pin measured `1.21e-2` — so the law is confirmed, and the probe is
        4-5x worse than any circular one. `rho = 0` recovers the circular `0.70`
        exactly, as it must.
      - **Consequence.** The binding quantity is `bias/SE`, not `bias`, so the
        binding probe maximizes `c(rho)/sqrt(p(1-p))` — which is `rho = 0.9`
        (`2.82`), with `rho = 0.2` (`2.47`) *ahead of* `rho = 0.8` (`2.35`),
        because SE collapses near `p -> 1` faster than `c` grows. Requiring
        `bias <= SE/4` at `S = 400` then allows **~2 draws at `rho = 0.2`, ~1.4 at
        `rho = 0.9`, and under one draw at elliptic `rho = +0.8`**. There is no
        affordable `S` that fixes this: see the cost cliff below. **A
        bias-negligible multi-probe assertion is not available**, and deriving the
        elliptic config from the circular constant would have shipped a test that
        fails a correct implementation.
- [x] **`S = 400` is derived by a cost cliff, not by taste.** `eigvals` per draw:
      `0.018 s` at `S = 200`, `0.15 s` at `400` (8x for 8x FLOPs, as expected),
      then `0.28-0.92 s` at `500` and **`2.0 s` at `600` — 13x the `S = 400` cost
      for 3.4x the FLOPs**, and `4.3 s` at `800`. Whatever the cause (cache or
      LAPACK blocking), `S = 400` is the last cheap size and the `S^3` cost model
      does not hold past it.
- [x] **Most of the edge bias is the zero diagonal — and that is May's convention,
      not a bug.** Filling the diagonal Ginibre-style drops `c(0.9)` from `1.06` to
      `0.35` while barely moving `c(0.2)` (`0.41 -> 0.42`) or `c(0.5)`
      (`0.70 -> 0.51`). Zeroing the diagonal pins `trace = 0` exactly, which
      constrains the eigenvalue sum and redistributes mass at the edge. May's `B`
      *has* a zero diagonal — self-interaction is the `-d I` term — and the
      zero-diagonal draw is what reproduces the recorded tables, so this is a
      property of the ensemble under test, not something to correct away.
- [x] `core/random_matrix.py` — May/elliptic ensemble, **not a `Model`** (no `step`,
      no `observables`, no `state.t`; a plain pytest outside the `ValidationSuite`
      path, not registered in `models/__init__.py`). **32 tests, suite 389 passed
      in 240.59 s.** The design ended up with *two tracks*, because the direct
      fraction check and the scaling check want opposite regimes:
      - `fraction_report` — the direct probe, at the **one** bias-negligible
        configuration (`probe = 0.5`, `S = 400`, **9 draws derived** by
        `max_draws_for_negligible_bias`, bias `0.24 SE`). It refuses a
        bias-limited configuration rather than asserting against a tolerance it
        will eventually violate, which is what stops a caller buying spurious
        precision with draws.
      - `bias_scaling_report` — the asymptotic law, asserted on **elliptic
        `rho = +0.8`**, whose 4-5x larger bias makes it *unassertable* by the
        direct check and *cheap* by this one. Measured exponent `-0.838 / -0.820 /
        -0.887 / -0.805` at seeds 0-3.
- [x] **The first version of the scaling assertion was green for the wrong reason,
      and the teeth caught it.** It asserted only that the exponent was
      *significantly negative*. A wrong-`R` draw passed: a constant offset fits a
      near-perfect straight line, so its exponent comes out **`-0.0089 +/- 0.0015`**
      — clearing zero at 6 sigma — while the discrepancy itself sits at `0.33` and
      does not move. **Statistical significance is free when the residuals are
      small; the discriminating quantity is the decay *rate*.** The check now
      requires the exponent below `-min_decay_rate` (default `0.15`), a threshold
      placed in a *measured* gap: ~90 SE above the teeth (`~-0.01`) and ~7 SE below
      the shallowest correct probe (`-0.62 +- 0.067`). Kept as a data-only unit
      test, `test_significance_alone_would_have_passed_a_constant_offset`.
- [x] Teeth verified across seeds 0-3 — and **one of the three does not belong on
      this check**, which is the repressilator `Omega^2` lesson in a new instance.
      - *flipped `rho` sign* (`+0.008 / +0.020 / +0.023 / +0.025` — the discrepancy
        **grows**) and *Hermitian-ized draw* (`-0.025 / -0.016 / -0.023 / -0.009`
        on a discrepancy of `0.34`, flat to 2% across 8x in `S`) bite at every seed.
      - *wrong `R`* does **not** bite the scaling check — it passes at 3 of 4 seeds,
        because a misscaled radius leaves an offset that *itself* decays, measured
        at `S^-0.34 ... S^-0.42`, which no threshold separates from the shallowest
        correct probe. It was moved to the **direct probe**, where it bites at
        `>5 SE`. A tooth must bite the right check.
      - The flipped-sign tooth's own blind spot is asserted rather than left
        implicit: at `rho = 0` it is bit-identical, so it is only meaningful at
        `|rho| >= 0.4`. (The `linspace` lesson.)
- [x] `demos/random_matrix.py` — five acts. The elliptic law with a figure whose
      ellipse is **predicted, not fitted**; the finite-`S` bias with the exponents
      *reported*; `P(stable)` sharpening (`S = 25/50/100`) **reported, not
      asserted**; and the two-act refusal, with the feasibility table re-measured
      live (`0.435 / 0.000 / 0.000` at `S = 40`) and the spectrum shift
      (`max Re eig(A) = -0.391` against `max Re eig(diag(x*) A) = -0.106`).
      Note the slice's `x*`-span figure of 67 did not reproduce (this run: mean
      546) — `max/min` is heavy-tailed, so its **mean is an outlier report, not a
      summary**; the demo prints the median.
- [x] **Two review catches worth recording, both about applying the discipline to
      the artifact rather than only to the model.**
      - *The positive assertion was seed-checked at one seed while its teeth were
        checked at four.* Priced draw counts predicted `z ~ 4` everywhere; measured,
        the cheap config's worst seed sat at `3.83` against a guard of `3`. Predicted
        `z` undershot measured `z` twice in this step. **Seed-verify the assertion
        you are defending, not just the teeth that defend it.**
      - *The demo printed exponents fitted from points its own `resolved` guard
        rejects* (`z = 1.05 ... 2.19` at `S = 200`) beside a reference column, which
        reads as agreement — the `-0.9279` error reproduced inside the artifact
        built to explain it. Fixed by printing `resolved` and the per-point `z` and
        making the under-resolution the act's teaching point: the printed exponents
        visibly disagree with the resolved reference (`-0.96/-0.84/-0.65` against
        `-0.88/-1.03/-0.62`), which demonstrates the trap better than prose.

## 3c — stochastic gLV

- [x] **`models/glv_stochastic.py` on the existing Gillespie engine.** `S` birth
      reactions plus `S^2` loss reactions (flat index `S + i*S + j` decrements
      species `i`), one `rates(c)` driving both the SSA propensities and
      `deterministic_rhs = stoich.T @ rates(c)` — so the hand-written double-loop
      check pins the stoichiometry, the `A` convention **and** which species each
      loss removes at once. The reference system is 3a's **asymmetric** 3-species
      one on purpose: under `A = A^T` a transposed convention and a loss charged
      to `X_j` instead of `X_i` both leave the ODE limit bit-identical, and that
      blind spot is asserted rather than left implicit. Five refusals; no
      `analytic_predictions`, with the artifact rather than prose behind the
      refusal (`<x> = x*` would go red at `R = 229 / 677 / 4429` for
      `Omega = 100 / 400 / 1600`).
- [x] **`O(1/Omega)` bias — CLOSED, and the plan's formula needed generalizing.**
      `bias = (-A)^-1 diag|A_ii| 1 / Omega`; the plan's `x*/Omega` is that same
      formula specialized to `diag|A_ii| 1 = r`. Estimator: **split coupling**
      (Anderson 2012 CFD) — the macro arm *is* the exact arm plus `S` extra loss
      channels, so both run as ONE chain (shared channel `min(aE, aM)`, plus
      E-only and M-only), costing about one arm's events. At `Omega = 100`,
      `T = 200`, `R = 8` and equal cost the SE is **6.208e-3 independent /
      1.357e-3 CRN / 9.247e-4 CFD** (4.6x, 6.7x). Its own tooth: run both arms
      under the exact rule and the difference is **bit-for-bit `0`** (25510 and
      99478 events). Recorded at `T = 500`, `R = 8`, `burn = 20`, 488 s: weighted
      log-log slopes **`-1.0525 +- 0.0624 / -0.9945 +- 0.0870 / -0.8522 +- 0.1378`**
      (all within 1.1 sigma of `-1`, and 8.9 / 5.7 / 2.6 sigma from `-1/2` — the
      claim that matters, since it is what stops the bias flooring 3c's slope).
      Per-component `z` against the prediction never exceeds **1.67**.
      `burn = 20` and `burn = 50` agree within SE everywhere, so the burn is
      measured rather than assumed.
- [x] **`D(Omega) ~ Omega^{-1/2}`** at `omegas = [50, 100, 200, 400, 800]`,
      `t_max = 20`, `R = 8`, `z = 3`, `observable_keys` explicit. Seeds 0-3:
      **`-0.4952 / -0.5134 / -0.4960 / -0.5228`**, all pass, `|slope+1/2| / (z SE)`
      = `0.08 / 0.22 / 0.03 / 0.37`. No `fit_mask` — `D*sqrt(Omega)` is flat
      (`0.80 / 0.84 / 0.88 / 0.80 / 0.83`) so *every* point is claimed to be in the
      regime, and that is asserted. Teeth: fixed-`Omega` (slope `+0.0217 / +0.0005
      / +0.0498 / +0.0029`, 4/4) and sqrt-`Omega` (`-0.2619 / -0.2510 / -0.2598 /
      -0.2532`, clearing tolerance by `7.86 / 6.63 / 4.99 / 7.72`, 4/4).
- [x] **Three traps in 3c itself, all green or plausible first:**
      (1) *The sqrt-`Omega` tooth was seed-lucky*, and its cheap fix bit the wrong
      thing — raising replicates passed, but at effective size 10 the initial count
      is 2 molecules/species and **22 of 24 replicates lost a species**, so the
      tooth was failing through extinction rather than through its exponent.
      Re-sited to effective sizes `25...400` (nominal `Omega` is free when the SSA
      runs at `sqrt(Omega)`).
      (2) *The bias assertion measured species 0 only* — and species 0 is the one
      component on which **two of the three wrong formulas are invisible**
      (transposed `A` is `-1.7%` off there, the plan's `x*/Omega` `+1.0%`, against
      `-43%/+57%` and `-35%/+60%` on species 1 and 2). Excluding the transpose on
      species 0 needs `SE <= 2.4e-5`, unreachable; on species 1 and 2 it needs
      `~6e-4`, which the shipped config reaches. **An assertion has to exclude the
      wrong formulas, not merely be consistent with the right one** — 3b's
      "significance is free when the residuals are small" one layer up.
      (3) *The recorded slope SEs were residual-only.* A 3-point fit with 1 dof
      returns a tiny SE no matter how fat each point's own error bar is: species 1
      read `+-0.0133` where propagating the per-point SEs gives `+-0.0870`, **6.5x
      wider**, and species 2's `-0.8335 +- 0.0372` — a 4.5-sigma "subleading
      correction" — became `-0.8522 +- 0.1378`, **1.1 sigma from `-1`**. Caught
      before it was written down; it would have put a physical effect that does not
      exist into this file. `chi2/dof` = `0.28 / 0.03 / 0.05` is the tell.

## 3d — Daisyworld

- [x] `models/daisyworld.py` — `delta` by Cardano (**`np.cbrt`**, not `x ** (1/3)`:
      one of the two cube-root arguments is negative, `+5.19e6` and `-4.75e6`, and
      the roots `+173.12 / -168.13` cancel to `4.988`, so ~1.5 digits go and the
      agreement with a 200-iteration bisection is `5.7e-15` relative, not the last
      bit). `analytic_predictions` = `a_w`, `a_b`, `x_bare`, `albedo`, `T_e`,
      `T_w`, `T_b`. Raises outside the band, on `A_w <= A_b`, and on
      `beta(T_w*) <= gamma`. The **band is closed-form too** — invert the linear
      albedo balance at `A_lo`/`A_hi` instead of bisecting; reproduces the slice's
      `0.738722418247 / 1.359472371265` exactly. 45 tests, ~7 s.
- [x] **`dT_w/dL = 0` cannot be an `analytic_predictions` key** — `validate()`
      matches each predicted key to an observable's final value, and a derivative
      with respect to a *parameter* is not an observable of any run (a key with no
      observable gives `samples = []` -> `nan` -> fail). **And the obvious
      replacement is a tautology**: "the prediction is the same at every `L`" just
      reads the source, where `T_w* = T_opt - delta` contains no `L`. Split into
      two checks that can actually fail: the local temperature law evaluated at the
      **`L`-dependent cover** (`a_w*` runs `0.024 -> 0.668`) returns the same `T_w`,
      and **simulated** endpoints from one common bare start agree across `L`.
- [x] **Richardson in `dt` is the wrong instrument for the second one, measured.**
      At `L = 1.15` the deviation from `T_w*` is `1.7e-7` while
      `|T_w(dt) - T_w(dt/2)|` is **exactly `0.0`** — the residual is leftover
      transient, not discretization error (the Phase-2 lesson verbatim). The
      two-horizon decay law `|T(t) - T(t/2)| exp(-t / 2 tau)` predicts it at
      **`1.000 / 1.000 / 1.000 / 1.000 / 1.003`**.
- [x] **`t_max = 100`, and longer is WORSE.** At `150` and `200` the true residual
      has hit the floating-point floor (`~8e-13`, ~14 ULP of `T_w`) while the
      predicted bound keeps shrinking, so the ratio runs to `24.8`, `312`, and at
      `t_max = 200` up to `3.2e7`. The horizon is part of the claim, and the tooth
      confirms it: moving it to 200 turns the test red.
- [x] Order-4 on the transient at `t_max = 20`: **`15.26 / 15.56 / 15.76`** at
      `dt = 0.4 -> 0.05`, against the plan's recorded `15.3 / 15.6 / 15.8`. **This
      one travelled** — unlike 3a's `t_max = 20` trap. The `t_max = 200` trap
      reproduces exactly: `1.283e-13 -> 1.233e-13`, **ratio 1.00 at every `dt`**.
- [x] **The `beta`-clip `C^0` note is much stronger than "real in principle".** At
      `L = 1.0` the clip never bites (`beta_raw >= +0.733`) and clipped and smooth
      integrations are **bit-identical**. At `L = 0.8` the same start drives
      `beta_raw` to `-0.025` and the same order measurement reads
      **`58.68 / 0.49 / 2.84`** with non-monotone errors (`9.97e-7`, `1.70e-8`,
      `3.43e-8`, `1.21e-8`). Both are asserted; **which luminosity the order claim
      is made at is part of the claim.**
- [x] Category C: hysteresis, dieback, bare-planet comparison — demo only.

### The invasion window, and the wrong figure claim it fixed

- [x] **The demo's first hysteresis panel was wrong, and only looking at the
      numbers caught it** — the fifth such figure in the project. The up-ramp's
      "dead" readings at low `L` were not a second attractor: the state sat at
      `a_b = 5.4e-144`, still growing exponentially from a denormal, and at
      `t = 5000` instead of `1000` the same point recovers to `a_b = 0.662`.
      Extinction in this ODE is **asymptotic** — `a_i` never reaches zero — so an
      unfloored ramp reports relaxation time as bistability. Fixed with a `1e-6`
      propagule floor, which makes "dead" mean "cannot grow from a seed".
- [x] **`invasion_luminosities` — a second closed form, answering a different
      question from the band.** A rare species grows on a bare planet iff
      `beta(T_i) > gamma`, i.e. `T_i` inside `T_opt +- sqrt((1-gamma)/k)`; a bare
      planet has albedo `A_g` whatever `L`, so `T_i^4` is *linear* in `L` and
      inverts directly. White invades for `L in [0.8332000704, 1.2079168175]`,
      black for `[0.7058188359, 1.0805355830]`.
- [x] **So the bistability is one-sided, and that is now a number rather than an
      impression.** Genuine bistability needs the interior state to exist *and*
      the dead planet to be uninvadable: hot end `L in (1.2079, 1.3595)`, width
      **`+0.151556`**; cold end width **`-0.032904`, i.e. EMPTY** — invadability
      reaches *below* the band there. The demo now labels each ramp row `BIST`
      (interior vs dead), `BIST-bdy` (a single-species boundary state vs dead,
      outside the band) or `(slow)` — and `L = 1.20`, which sits `0.008` inside the
      white window, is correctly marked `(slow)` rather than counted as
      bistability.
- [x] **The basin fact explains the extinctions**: from `(0.01, 0.01)` the planet
      reaches the interior state for `L in [0.8, 1.2]` and dies for `L >= 1.25`. At
      `L = 1.20` it is the **white** daisies that seed (`beta = 0.350`) while black
      cannot (`0.000`). `analytic_predictions` does **not** refuse on reachability —
      that is a basin question, not an existence one.

### Mutants: 23/25 red, and every survivor is a finding

- [x] **The headline mutant behaves exactly as designed.** Shifting `t_opt` by 1 K
      **in the params**, so the closed form and the RHS move *together*, leaves
      `|rhs(y*)| = 1.8e-16` — a perfectly good fixed point of the wrong planet. It
      kills only `test_the_recorded_constants_reproduce`, the band test, the three
      `a_w*(L)` literals and two band-boundary tests. The hand-written RHS, the
      Newton root-find, the equilibrium conditions, **all three `validate()` runs**,
      both invariance tests and the order test stay **green**. The recorded
      literals are the only thing standing between this model and a wrong planet.
- [x] **Two survivors are provably invisible, not gaps.** `bare_fraction` reading
      `beta(T_b)` instead of `beta(T_w)` cannot be caught — the reduction makes them
      equal, and a test asserts that. `analytic_predictions` reading the cover's
      temperatures instead of Cardano's cannot be caught by `validate()` either,
      because `test_the_temperature_law_returns_the_pinned_temperatures_at_every_
      luminosity` has already pinned the two routes together. **The model docstring
      claimed `validate()` was what checked the cross-route agreement; that was
      wrong and is corrected**, and the albedo leg — the one part of the round trip
      nothing checked — was added.
- [x] **A third mutant was a live trap, and it corrupted a run before it was
      understood (it is RED now, after the sweep below).** `scale = S (1 - A_g)` mutated to `S A_g` is a **no-op** with
      the W&L constants, because `A_g = 0.5` and `1 - A_g = 0.5` are the same
      number. A killed run left that mutation on disk; the baseline stayed green
      *because the mutation does nothing*, and the next run read the mutated file
      as pristine. Fixed by sweeping `albedo_ground in (0.5, 0.4)` in the invasion
      test — Phase 2's "**sweep the constant that the probe happens to land on**"
      for the third time, and the second time in this file after the white/black
      albedo swap (which leaves `x*` **bit-identical** and merely exchanges
      `T_w*`/`T_b*`, so only the `a_w*(L)` literals catch it).
- [x] **Two process lessons from that corruption.** A killed process runs no
      `finally`, so a mutation runner must (a) **verify the baseline is green
      before it starts** and (b) restore from **git**, not from an in-memory copy —
      and the files must be *committed first*, because `git status` cannot protect
      an untracked file. Both are now in the runner. (It also has to run in
      **batches of ~5**: this environment kills a subprocess after ~2-3 minutes.)

### The figure, twice — and the second one was caught by looking

- [x] **Panel 1 drew its own headline instead of measuring it.** The first draft
      plotted `[t_w_star] * len(lums)` — one value repeated — under the title *"The
      daisy temperatures are flat"*. Flatness by construction, which is the Phase-2
      Turing panel verbatim (stripes from a hand-seeded mode presented as a
      selected one). It now plots the per-`L` values **computed at each `L`'s own
      cover**. The picture is identical (the variation is `~1e-13` on a 290 K axis);
      the difference is whether the line demonstrates the claim or restates it.
- [x] **Panel 3 said every gap between the two ramps was bistability.** The act-5
      text spends a paragraph refusing exactly that for `L = 1.20`, but a figure
      gets looked at without its text. The panel now **shades the closed-form
      window** `(1.208, 1.359)`, and the down-ramp visibly jumps onto the
      no-daisies curve right at its left edge.
- [x] Panel 3's title was **clipped** at the right edge on the first render.
      Checked by opening the PNG, which is the only way that class of defect
      surfaces.

## 3e — adaptive dynamics

**Both open measurements are closed, ahead of any model code — this section
records what they returned. Nothing is implemented yet.**

### The convention had to be recovered before either could be measured

No slice code survives in the repo, so the 3e numbers exist only as the plan's
tables. The canonical-equation sweep held `mu sm^2 t_max` fixed, which pins one
scalar: `U = (1/2) mu sm^2 K0 t_max`, because under `u = (1/2) mu sm^2 K0 t` the
canonical equation collapses to the parameter-free `dx/du = -x exp(-x^2/2)`.

- [x] **`U = 1/2` exactly.** `x(0.5) = 1.849492240597` reproduces the recorded
      `1.849492` on all six digits, and bisecting for the `U` that hits the
      recorded value gives `0.500000719`, inside the `+-1.5e-6` window 6-digit
      rounding allows. With `r = 1, sK = 1` (pinned by the recorded
      `dD/dx = -r/sK^2 = -1.000000000000`) and `x0 = 2`, the process is a
      function of `(U, sm, sa)` alone: **`K0` and `mu` are not separately
      identifiable and do not need to be** — `K0` cancels in the invasion
      fitness (a ratio of `K`s) and the event count is `mu K0 t_max = 2U/sm^2`.
- [x] **Three recorded numbers the convention was *not* fitted to confirm it.**
      The teeth targets are properties of the deterministic side alone, so they
      are independent of `sa` and of the mutation-step distribution — the two
      things nothing recorded could pin. Reconstructed: `1.662326 / 1.213061 /
      0.000000` against recorded `1.662326 / 1.213061 / 0.000000`, agreeing to
      `3e-7`, which is just the recorded values' own rounding.
- [x] **`sa` and the step distribution are choices, and they are load-bearing.**
      Both enter the `O(sm)` coefficient. `sa = 0.7` (matching the branching
      runs) and a Gaussian step are *stated*, not inherited. **Consequence: the
      recorded `+0.004` offset and the teeth `z` values (`110.84 / 371.21 /
      1074.26`) are estimator-dependent and may not be cited against this
      convention** — the measured offset at `sm = 0.0125` is `+2.39e-3`, not
      `+0.004`. A fifth instance of "a number travels with its estimator", and
      the first where the estimator was only partly recoverable.

### The `O(sm)` law is now measured, not "consistent with"

- [x] **Slope `+0.9977 +- 0.0225` over a 32x range in `sm`** (`0.2 ... 0.00625`),
      i.e. **0.10 sigma from 1 and 44.6 sigma from 2**, `chi2/dof = 1.07`. All six
      discrepancies are **positive** — a sign flip would have refuted the reading
      regardless of the slope. Dropping `sm = 0.2` gives `+1.0174 +- 0.0303`.
      `err/sm` is flat at `0.17723` with `chi2/dof = 0.85` about a constant, so
      **no `O(sm^2)` correction is resolvable** over that range.
- [x] **The replicate counts were derived, not typed.** The pilot gives
      `err ~ 0.17 sm` and `SD ~ 0.5 sqrt(sm)` (the `sqrt(sm)` scaling is itself a
      prediction — accepted steps are `O(sm)` in number and `O(sm)` in size — and
      it is confirmed: `SD/sqrt(sm)` runs `0.439 ... 0.555`). Demanding `z ~ 15`
      gives `R ~ 1945/sm`. The plan's "1200-reps-equivalent" is a number from a
      different estimator; the measured `SD = 0.0621` at `sm = 0.0125` against the
      slice's implied `0.001725*sqrt(1200) = 0.0598` is what says the convention
      matches at all.
- [x] **A first run was wrong, and the free self-check caught it.** Group-to-group
      scatter must agree with the within-group `SD/sqrt(R)`; it read
      `0.75/0.16/0.34/0.94/0.55` — systematically *under*-dispersed — and
      `chi2/dof` was `2.62`. Cause: `spawn_rngs` was called with the same seed
      inside the `sm` loop, so **every sweep point ran on common random numbers**
      and the fit treated correlated points as independent. With an independent
      seed branch per `sm` the self-check reads `1.14/1.60/0.71/0.73/0.82/1.10`
      and `chi2/dof` falls to `1.07`. The excess scatter was *never* physics.
- [x] **The teeth reproduce 3b's trap exactly, and it decides the assertion's
      form.** A wrong canonical equation makes the discrepancy an `O(1)` constant,
      and a constant fits a near-perfect line: the three teeth read
      `+0.0181 +- 0.0007`, `+0.0053 +- 0.0002`, `-0.0030 +- 0.0001` — SEs **30-200x
      smaller** than the correct point's, so every one of them is *tens of sigma
      from zero*. **"Significantly nonzero" passes all three.** Only a two-sided
      band around 1 rejects them.
- [x] **Shippable config, seed-verified — and the saturating point was replaced.**
      `sm = 0.2` puts **4.57%** of offered mutants at `s > 0.5`, where the
      linearization the canonical equation rests on is worst, *and* it is the
      leftmost point, carrying the most leverage on the slope. Three candidates at
      `R = 1945/(4 sm)`, seeds 0-3:

      | config | slopes | spread | max sat | cost |
      |---|---|---|---|---|
      | A `0.2 ... 0.0125` | `0.9514 / 0.9070 / 0.9660 / 0.9985` | 0.0914 | 4.57% | 2.3 s |
      | **B `0.15 ... 0.0125`** | `0.9558 / 0.9802 / 1.0113 / 1.0409` | **0.0850** | **1.41%** | 2.3 s |
      | C `0.1 ... 0.00625` | `0.9900 / 1.1224 / 1.1205 / 0.9071` | 0.2153 | 0.05% | 17.5 s |

      **The saturation is harmless — A and B agree inside their spreads — but B is
      strictly better** at equal cost, and its slopes centre on `1.00` where A's
      centre on `0.955`, consistent with `sm = 0.2` contributing extra positive
      error that flattens the log-log line. **C is rejected**: 7.6x the cost *and*
      2.5x the spread, because dropping the long lever arm at high `sm` costs more
      than the extra low point buys. `4 points, R/4` is also rejected — seed 1
      reads `0.7930`, spread `0.2523`.
- [x] **Band `[0.6, 1.4]` placed against config B's teeth at all four seed-sets**,
      not one. Correct `+0.9558 / +0.9802 / +1.0113 / +1.0409` (**4/4 in band**);
      every tooth **0/4**, worst `+0.0342 +- 0.0013`. The lower edge sits **5.7
      correct-SE** below the worst correct seed and **431 tooth-SE** above the
      worst tooth; the upper edge **5.8 correct-SE** above the best. A tooth scored
      on a single seed is what got a 3c config rejected for clearing by 0.8%.
- [x] **A cost trap in the test's *deterministic* half.** The `sm for sm^2` tooth's
      target is `x` integrated to `u = U/sm`, which at `sm = 0.00625` is `u = 80` —
      eight million pure-Python RK4 steps at `du = 1e-5`, and it dominated the
      verification run's wall clock (36 s for four seeds against 2.3 s/seed of
      actual simulation). `du = 1e-4` is ample (`pin_convention.py` shows `x(0.5)`
      identical to twelve digits at `1e-4 / 1e-5 / 1e-6`), and the shipped test
      needs only the *correct* target at `u = 0.5`. **The expensive part of a
      stochastic test can be the deterministic helper.**
- [ ] Ship it: `models/adaptive_dynamics.py`, the derivative checks (signed
      against finite differences), and this slope as the canonical-equation test.

### The branching formulation was reconstructed from its own failure modes

- [x] **There is no mutation/diffusion term, and that is forced.** A first draft
      carried nearest-neighbour diffusion and seeded every bin at `1e-6`. It
      failed twice: the *outer* bins won (against a resident at the singular point
      every mutant has positive invasion fitness and the far ones have the most,
      so it "branched" at `t = 7.5` into five clusters out to `+-2.6`), and
      `sa = 1.5` **overflowed** — at `x = +-4`, `K = 3.35e-4` against a competition
      load of `0.029`, so the local eigenvalue is `-84` and `dt = 0.5` gives
      `lambda dt = -42`, far outside RK4's stability region. The slice ran
      `dt = 0.5` on `[-4, 4]` without blowing up, so its outer bins must have been
      **exactly zero and stayed exactly zero** — which happens iff there is no
      diffusion, since pure gLV has `n_i = 0` as a per-bin invariant. Domain
      (recovered from the recorded morph positions), prefactor arithmetic and
      stability all agree on one formulation. Asserted directly: 158 untouched
      bins are **bit-identically `0.0`** at `t_branch`.
- [x] **So the morph positions are not merely a grid artifact — those are the only
      bins that ever carried mass.** The seed is the centre plus its two immediate
      neighbours, which is why the morphs sit at exactly `+-1` spacing at
      `ngrid = 81/161/321`. Stronger than the plan's "grid-dependent, category C",
      and it means **this is a 3-species gLV in which `ngrid` enters only through
      `h`** — worth saying plainly rather than describing a 161-species trait grid.
      The Gyllenberg-Meszena positive-definiteness argument is a separate
      category-C claim about the *continuum* model and is not what produces `+-h`.

### The detection resolution is fixed, and the answer inverts the assumption

- [x] **Checkpoint-and-refine**, not per-step checking: keep the previous coarse
      checkpoint's state and, on first detection, replay that one interval step by
      step. Resolution goes from `+-50` to `+-dt/2 = +-0.25` for ~200 extra RK4
      steps on one interval of one run, rather than 400 000 extra gap checks on
      every `sa = 0.95` run at every parameter point.
- [x] **Coarse checking made the products look MORE constant than they are.** True
      spread `0.622%`; at quantum `10` it reads `0.594%`, at quantum `100`
      **`0.381%`**. Detection rounds *up* to the next checkpoint, and that
      inflation is largest where `t_branch` is smallest — large rate, small `sa` —
      which is exactly where the true product is lowest, so quantization partly
      cancels the real drift. **A tolerance derived from the coarse measurement
      would be too tight and would fail a correct model.** The plan was right to
      refuse the 1.2%, but the direction is the opposite of the natural guess:
      the hazard was not inflated scatter, it was flattened scatter.
- [x] **The surviving drift is `O(h^2)`, and in the `h -> 0` limit the product is
      constant to 0.006%.** Spread across `sa` is `2.7481% / 0.6216% / 0.1517%` at
      `h = 0.1 / 0.05 / 0.025` — ratios **4.421** and **4.098**. Richardson
      (4/3 rule) on `product * h^2` gives `40.0578 / 40.0580 / 40.0580 / 40.0575 /
      40.0558` across `sa = 0.60 ... 0.95`, i.e. over a **16.5x range in rate**.
      Note *which* instrument this is: refining `h` is Richardson in a
      **discretization parameter**, the same tool as Phase 2's order-2 stencil
      check — **not** the Richardson-in-the-amplitude used for HH's transient,
      Gray-Scott's linearization, gLV relaxation, the LV period and the `sm` sweep
      above. That one has now generalized five times and the count should not be
      inflated by conflating the two.
- [x] **It is not discretization.** `dt = 0.5 / 0.25 / 0.125` give products
      identical to five significant figures (spread `0.622 / 0.623 / 0.624%`), so
      `dt = 0.5` is already converged and the drift is a property of the grid
      spacing, not the step.
- [x] **Threshold-invariance is verified with a formula rather than asserted.**
      The prefactor is **not** a function of `log(thr/seed)`: at the same ratio
      `2.3026` it reads `19805` (`thr = 1e-5`) and `12399` (`seed = 1e-4`).
      Measured, `d(product)/d log(1/thr) = 807.5` and `d(product)/d log(1/seed) =
      800.6`, both `~ 2/h^2 = 800`, giving
      `t_branch * rate = (2/h^2) log(1/(thr*seed)) + const`. With the slope
      **fixed** at `800` by the mechanism and only the offset fitted, that predicts
      all six `(thr, seed)` configurations to **<= 0.21%**. Across all six the
      exponent stays at `-1.001 ... -1.003`.
- [x] **The `const` is fitted, and the mechanism is incomplete — say so.** The
      derived part is the neighbour bin rising from `seed` to `thr` at
      `s_0(h) ~ rate h^2/2`. That alone predicted `2 log(thr/seed)/h^2 = 5526`
      against a measured `16086` — off by 2.9x — because the **centre bin must
      also fall below `thr`**, and that half is not derived. It shows up as a real
      additive residual in the limit: `product * h^2 -> 40.058` as `h -> 0`, while
      `2 log(1/(thr*seed)) = 41.447`, leaving **`1.39` unexplained**. The formula
      predicts, but it is not a closed form, and must not be read as one.
- [x] **The no-branch side is now measured at the horizon the branch side needs.**
      It had only been run to `t_max = 20 000`, while the nearest *presence* claim
      (`sa = 0.95`) takes `149 822.5` — an absence asserted 7.5x short is a
      statement about the horizon, not a sign change. At `t_max = 200 000` both
      `sa = 1.05` and `sa = 1.5` still do not branch, stay finite, and leave the
      centre bin at `K(0) = 1.000000`. The decaying bins reach **exactly `+0.0`**
      (`n_i = 0` is a gLV invariant, so once a bin underflows it is pinned, with no
      subnormal-arithmetic penalty), and each run costs **17 s**.
- [ ] Ship it: the branch/no-branch sign change (the no-branch side stated as
      **"does not branch by `t = 200 000`"** — it is an absence claim and is
      horizon-bounded), and `t_branch * rate` asserting **the exponent**, with the
      prefactor law available as a second, independent check.
- [ ] Label the post-branching morph structure exploratory, now with the
      *mechanism* (only three bins are ever seeded) as the stated reason rather
      than the bare grid-dependence measurement.

## All

- [x] **The open 3a timing anomaly is closed — as *not decidable at this
      resolution*, which is itself the finding.** The dichotomy asked us to read a
      signal from whether the repressilator floor test "held near 162.17 s". It
      cannot: that test was measured **three times in one session at 123.03 s and
      154.99 s (serial, alone, nothing contending) and 189.81 s (in-suite, `-n 6`)**
      — a spread of +-30%, comfortably larger than the effect the dichotomy was
      trying to detect. Both branches are consistent with the data, so the
      xdist-packing hypothesis is neither confirmed nor refuted. Note also that the
      two serial runs and the in-suite run are **three different estimators**, so
      even the 155-vs-190 gap is not a like-for-like comparison; see
      [[numbers-travel-with-their-estimator]]. **Any future suite-timing claim on
      this machine must be bracketed by +-30% or it is noise.**
- [ ] Re-time the suite at phase end; update `CLAUDE.md`, memory, docs; commit and
      push.

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
| Circular-law fraction, `z` | < 4 | <= 1.38 (slice); **<= 0.76 (pinned, seed 0)** |
| Elliptic-law fraction, `z` | < 4 | <= 2.31 (slice); **<= 2.79 (pinned, seed 1; both peaks at `rho = +0.8`)** |
| Elliptic `E[max Re]` vs `R(1+rho)` | — | **4.193 / 11.974 / 19.622 / 27.332 / 35.500** vs 4/12/20/28/36 (pinned) |
| Circular-law finite-`S` bias | — | `~ 0.6/S`, slope -0.9279 (slice); **`0.70/S`, slope -1.0086 (re-measured)** |
| Bias exponent, circular `probe = 0.2 / 0.5 / 0.9` | — | **-0.8814 +- 0.0250 / -1.0274 +- 0.0231 / -0.6189 +- 0.0673** (`S = 25...200`) |
| Bias exponent, elliptic `rho = -0.8 / +0.8` | — | **-0.7733 +- 0.1472 / -0.8404 +- 0.0150** |
| Same probe (`0.9`) over `S = 12...96` instead | — | **-0.4307 / -0.3396 / -0.4950** — *a different regime; the range is part of the claim* |
| Elliptic scaling check, seeds 0-3 | decays | **-0.838 / -0.820 / -0.887 / -0.805**, all pass (`min z = 4.97/6.29/6.93/6.88`) |
| ...at a 45%-cheaper draw count | decays | passes, but `min z = 3.83/4.66/4.42/5.51` — only 28% over the `resolved` guard, so **not shipped** |
| Teeth exponents (flipped sign / Hermitian) | ~0 | **+0.008...+0.025 / -0.025...-0.009** — bite at 4/4 seeds |
| Tooth exponent, wrong `R` | ~0 | **-0.34...-0.42** — does *not* bite the scaling check; moved to the direct probe |
| `eigvals` cost per draw | `S^3` | 0.018 s (`S=200`), 0.15 s (`400`), **2.0 s (`600`) — 13x for 3.4x FLOPs** |
| Repressilator floor test, one session, 3 estimators | — | **123.03 / 154.99 s (serial alone) / 189.81 s (in-suite)** — +-30% |
| Suite after 3b | — | **389 passed in 240.59 s** at `-n 6` (357 before) |
| Stochastic gLV `D(Omega)` slope | -1/2 | -0.4984 +/- 0.0488 (slice); **-0.4952 / -0.5134 / -0.4960 / -0.5228 (built, seeds 0-3)** |
| gLV-stochastic bias estimator SE, equal cost | smaller | **6.208e-3 independent / 1.357e-3 CRN / 9.247e-4 split-coupled** (4.6x, 6.7x) |
| Coupled arms under identical rules | exactly 0 | **bit-for-bit `0.0`** at `Omega = 100` (25510 events) and `400` (99478) |
| Bias log-log slope, `T = 500` `R = 8`, per species | -1 | **-1.0525 +- 0.0624 / -0.9945 +- 0.0870 / -0.8522 +- 0.1378** (WLS, per-point SEs propagated) |
| ...the same fit with residual-only SEs | — | `+-0.0423 / +-0.0136 / +-0.0364` — **6.5x too tight on species 1**, and invents a 4.5-sigma effect on species 2 |
| Bias `z` vs prediction, `Omega = 100/200/400` | < 3 | **<= 1.67** on every component at both burns |
| Bias at `Omega = 50`, even split-coupled at `T = 500` | — | **-7.19e-4 +- 1.13e-2 — unresolved.** The 3a finding survives the better instrument |
| Ship config gap, `Omega = 100` `T = 400` `R = 8` | — | correct `<= 1.418`, transpose `>= 4.780`, naive `>= 7.782` (4 seed-sets) — **`z = 3` sits in it** |
| ...the same at `T = 200` | — | correct `<= 1.675`, transpose `>= 3.025` — **clears `z = 3` by 0.8%; seed-lucky, not shipped** |
| Wrong-formula error at species 0 / 1 / 2 | — | no-solve `+44/+49/+90%`; transposed `A` **`-1.7%`**`/-43/+57%`; `x*/Omega` **`+1.0%`**`/-35/+60%` |
| Daisyworld `\|rhs(y*)\|` | 0 | <= 3.68e-16 (slice); **<= 3.24e-16 (built, `L = 0.75...1.35`)** |
| Daisyworld `dT_w/dL` | 0 | +0.000e+00 (slice); **exactly `+0.000e+00` at every `L` and every `h` except `L = 1.30`, where it is a 1-ULP wobble (`5.684e-14`) divided by `2h`** |
| Daisyworld `delta`, Cardano vs 200x bisection | 0 | **5.698e-15 relative** (cube roots `+173.119 / -168.131` cancelling to `4.988`) |
| Daisyworld band, closed form vs the slice's bisection | — | **0.738722418247 / 1.359472371265** — reproduces to every recorded digit |
| Daisyworld order-4, `t_max = 20`, `L = 1.0` | 4 | **15.26 / 15.56 / 15.76** (plan recorded 15.3/15.6/15.8 — **this one travelled**) |
| ...at `t_max = 10 / 40 / 200` | — | `15.77/15.84/15.91`; `12.69/14.38/15.11` (roundoff at 7e-15); **`1.00/1.00/1.01` — the attractor trap reproduces** |
| ...at `L = 0.8`, where the `beta` clip bites | — | **58.68 / 0.49 / 2.84**, errors non-monotone. `beta_raw` reaches `-0.025` |
| `beta` clip at `L = 1.0`: clipped vs smooth | identical | **bit-identical**, `beta_raw >= +0.733` |
| Simulated `T_w` spread over `L`, bare start, `t = 100` | 0 | **2.141e-6 K**, against `T_e` spread **2.179068 K** — ratio **1.02e6** |
| Decay-law predicted / true residual, `t_max = 100` | 1 | **1.000 / 1.000 / 1.000 / 1.000 / 1.003** |
| ...at `t_max = 150 / 200` | — | `1.42 / 312 / 24.8 / 1.01 / 1.00`; up to **3.2e7** — the true residual is on the fp floor (`~8e-13`), so **longer is worse** |
| Richardson in `dt` on the same quantity | — | **exactly `0.0` at `L = 1.15`** where the deviation is `1.7e-7` — blind to a transient |
| Daisyworld Jacobian, slow `tau` across the band | — | `51.0` (`L=0.75`), `4.33` (`0.95`), `8.76` (`1.20`), **`151.1` (`1.35`)** — real and negative throughout |
| Invasion window, white / black | — | **`[0.8332000704, 1.2079168175]` / `[0.7058188359, 1.0805355830]`** |
| Genuine bistability, hot end / cold end | — | width **`+0.151556`** / **`-0.032904` (EMPTY)** — the bistability is one-sided |
| 3d mutants confirmed red | all | **23 / 25**; both survivors provably invisible (see above) |
| Cost of `tests/test_daisyworld.py` | — | **45 passed in 5.8-9.8 s** serial (17.7 us / RK4 step) |
| Adaptive-dynamics derivatives vs FD | 0 | <= 7.6e-8 (slice) |
| Canonical `U = (1/2) mu sm^2 K0 t_max` | — | **`1/2` exactly**: `x(0.5) = 1.849492240597` reproduces the recorded `1.849492`; bisected `U = 0.500000719`, inside 6-digit rounding |
| Teeth targets, reconstructed vs recorded | 0 | **`+3.04e-7 / +3.19e-7 / +5.4e-17`** on `1.662326 / 1.213061 / 0.000000` — three numbers the convention was not fitted to |
| Canonical discrepancy, `sm = 0.2 ... 0.00625` | `O(sm)` | **`+3.412e-2 / +1.814e-2 / +9.402e-3 / +4.146e-3 / +2.390e-3 / +1.047e-3`**, all positive, `z = 13.8 ... 16.6` |
| ...log-log slope, WLS with propagated SEs | 1 | **`+0.9977 +- 0.0225`, `chi2/dof = 1.07`** — 0.10 sigma from 1, **44.6 sigma from 2** |
| ...`err/sm` about a constant | const | **`0.17723`, `chi2/dof = 0.85`** — no `O(sm^2)` term resolvable over 32x |
| ...the same fit on common random numbers (bug) | — | `+0.9687 +- 0.0339`, **`chi2/dof = 2.62`**, self-check `0.75/0.16/0.34/0.94/0.55` — one `spawn_rngs` seed reused across the sweep |
| `SD(sm) / sqrt(sm)` | const | **`0.439 / 0.497 / 0.499 / 0.538 / 0.555`**; `SD = 0.0621` at `sm = 0.0125` vs the slice's implied `0.0598` |
| Canonical offset at `sm = 0.0125` | — | **`+2.39e-3`**, against the slice's recorded `+0.004` — `sa` and the step law are not recoverable, so the offset does not travel |
| Canonical teeth, on the **slope** | — | **`+0.0181 +- 0.0007 / +0.0053 +- 0.0002 / -0.0030 +- 0.0001`** — all tens of sigma from zero, so "significantly nonzero" passes all three |
| Ship config B (`0.15 ... 0.0125`, `R/4`, 2.3 s), seeds 0-3 | — | **`0.9558 / 0.9802 / 1.0113 / 1.0409`**, spread `0.0850`, max saturation `1.41%` |
| ...config A (`0.2 ... 0.0125`) | — | `0.9514 / 0.9070 / 0.9660 / 0.9985`, spread `0.0914`, saturation **`4.57%`** — agrees with B, but centres on `0.955` not `1.00` |
| ...config C (`0.1 ... 0.00625`) | — | `0.9900 / 1.1224 / 1.1205 / 0.9071`, spread **`0.2153` at 7.6x the cost — rejected** |
| ...`4 points, R/4` | — | `1.0453 / 0.7930 / 0.9743 / 0.9986`, spread **0.2523 — rejected as seed-fragile** |
| Band `[0.6, 1.4]` on config B, 4 seed-sets | — | correct **4/4 in**, every tooth **0/4**; edges **5.7 / 5.8 correct-SE** out, worst tooth **431 tooth-SE** below |
| No branch at `sa = 1.05 / 1.5`, `t_max = 200 000` | — | **holds**, finite, centre `1.000000`, decayed bins **exactly `+0.0`**, 17 s each (the earlier check ran only to `20 000`, 7.5x short of the `sa = 0.95` branch) |
| Untouched trait bins at `t_branch` | 0 | **bit-identically `0.0`** (158 bins) — the stability guarantee, and it fails loudly if diffusion returns |
| `t_branch` at `sa = 0.60 ... 0.95`, `ngrid = 161` | — | **`9053.0 / 15455.0 / 28600.5 / 68707.5 / 149822.5`** (refined to `+-0.25`) |
| `t_branch * rate` spread across `sa` | const | **0.622%** true; **0.594%** at quantum 10; **0.381%** at quantum 100 — *coarse checking flattens it* |
| ...at `h = 0.1 / 0.05 / 0.025` | `O(h^2)` | **`2.7481% / 0.6216% / 0.1517%`** — ratios **4.421** and **4.098** |
| ...Richardson to `h -> 0`, `product * h^2` | const | **`40.0578 / 40.0580 / 40.0580 / 40.0575 / 40.0558`** — constant to **0.006%** over a 16.5x rate range |
| ...at `dt = 0.5 / 0.25 / 0.125` | — | products identical to 5 s.f. — **the drift is not discretization** |
| `t_branch * rate` log-log slope | -1 | -1.0003 (slice); **`-1.00196 +- 0.00070` (built)**, and `-1.001 ... -1.003` across all six `(thr, seed)` |
| Prefactor law `(2/h^2) log(1/(thr*seed)) + const` | — | slope **fixed** at `2/h^2 = 800`, offset `-488.1` **fitted**: predicts all six configs to **<= 0.21%** |
| ...the part the mechanism does derive | — | neighbour rise alone gives `2 log(thr/seed)/h^2 = 5526` vs measured `16086` — **2.9x short**; the centre-bin fall is not derived, leaving `41.447 - 40.058 = 1.39` in the `h -> 0` limit |
| Suite after 3c, `-n 6` | — | **410 passed in 453.91 s** |
| ...same-session baseline, 3c ignored | — | **389 passed in 544.96 s** — *slower without the new tests*, so **no 3c regression is visible**. The runs were sequential, so this does not separate "3c is cheap" from "the machine drifted between them" |
| Repressilator floor test, this session | — | **287.06 s** (with 3c) / **318.02 s** (without) vs **189.81 s** recorded in-suite — machine ~1.5-1.7x slower |
| Suite after 3d, `-n 6` | — | **455 passed in 180.82 s** (floor test **137.26 s**) |
| ...same-session baseline, 3d ignored | — | **410 passed in 217.12 s** (floor test **158.58 s**) |
| ...what that does and does not say | — | The floor test, **untouched by 3d**, moved **15.5%** between the two runs, so the totals are not attributable to the test set. **No 3d regression is visible**; nothing stronger is claimable. Cross-session it is worse still: the same test read **287.06 s** last session against **137.26 s** now, a **2.1x** machine swing, so the 453.91 s recorded after 3c is uninterpretable here. Runs were sequential, not interleaved, so monotone drift is confounded with the test-set difference |
| Suite after Phase 3, `-n 6` | < baseline + new, **re-timed same-session** | *(pending — 3e)* |
