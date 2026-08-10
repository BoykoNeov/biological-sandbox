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
| Elliptic scaling check, seeds 0-3 | decays | **-0.838 / -0.820 / -0.887 / -0.805**, all pass |
| Teeth exponents (flipped sign / Hermitian) | ~0 | **+0.008...+0.025 / -0.025...-0.009** — bite at 4/4 seeds |
| Tooth exponent, wrong `R` | ~0 | **-0.34...-0.42** — does *not* bite the scaling check; moved to the direct probe |
| `eigvals` cost per draw | `S^3` | 0.018 s (`S=200`), 0.15 s (`400`), **2.0 s (`600`) — 13x for 3.4x FLOPs** |
| Repressilator floor test, one session, 3 estimators | — | **123.03 / 154.99 s (serial alone) / 189.81 s (in-suite)** — +-30% |
| Suite after 3b | — | **389 passed in 240.59 s** at `-n 6` (357 before) |
| Stochastic gLV `D(Omega)` slope | -1/2 | -0.4984 +/- 0.0488 (slice) |
| Daisyworld `\|rhs(y*)\|` | 0 | <= 3.68e-16 (slice) |
| Daisyworld `dT_w/dL` | 0 | +0.000e+00 (slice) |
| Adaptive-dynamics derivatives vs FD | 0 | <= 7.6e-8 (slice) |
| `t_branch * rate` log-log slope | -1 | -1.0003 (slice) |
| Suite after Phase 3, `-n 6` | < baseline + new, **re-timed same-session** | *(pending)* |
