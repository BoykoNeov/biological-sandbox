# Phase 2c — Schnakenberg wavelength selection: the measurement slice

Phase 2 deferred one item explicitly: *"Schnakenberg / Brusselator for near-onset
wavelength selection."* It was deferred because Gray-Scott had already validated
`lambda(q)` more sharply, and adding a third reaction-diffusion model to buy back
one check looked like scope creep. What that reasoning missed is that `lambda(q)`
and *wavelength selection* are different claims: the first is about the growth rate
of a mode you seed by hand, the second is about which mode appears when you seed
**nothing**. HANDOFF §5 asked for the second one, and Phase 2 could not deliver it
because at Pearson's parameters Gray-Scott has no Turing state to select about.

This document is the measurement slice, taken **before** any plan or model was
written, in the order the questions had to be answered. Nothing here is a
prediction about what the build will find; every number below was measured, and
the ones that were wrong are kept with the reason they were wrong.

Scripts: `M:\claud_projects\temp\phase2c-slice\{schnak,s1..s8}.py`. Not part of the
repo — the reproducible versions are the model and its tests.

## 0. The model

    du/dt = Du lap(u) + a - u + u^2 v
    dv/dt = Dv lap(v) + b - u^2 v

on a periodic box, with `a = 0.05`, `b = 1.0`, `Du = 1e-3` throughout, and
`d = Dv/Du` as the control parameter.

## 1. The closed forms are real, and checked rather than recalled  (s1)

Every one of these was written down from memory first, so every one is verified
against an independent route:

| quantity | closed form | check | result |
|---|---|---|---|
| homogeneous state | `u* = a+b`, `v* = b/(a+b)^2` | `|rhs(y*)|` | `0.0` exactly (3 of 4 probes), `5.6e-17` |
| reaction Jacobian | `[[-1+2uv, u^2], [-2uv, -u^2]]` | central differences | `1e-10 ... 3e-11` |
| its determinant | `det J = u*^2` | closed form minus `u*^2` | `0.0` **exactly**, all four probes |
| critical ratio | `d_c = [u*(1+sqrt(1+f_u))/f_u]^2` | bisection on `max_q Re lambda` | `1.9e-11 ... 2.7e-9` |
| critical wavenumber | `q_c^2 = u*/sqrt(Du Dv)` | argmax of `lambda(q)` at onset | `19.496952` vs `19.497001` |

At `(a, b) = (0.05, 1.0)`: `u* = 1.05`, `v* = 0.907029`, `f_u = +0.904762`,
`trace = -0.197738` (so the state is stable without diffusion, which is what a
Turing claim is *about*), `d_c = 7.629775`, `q_c = 19.496952`.

`det J = u*^2` exactly is not a coincidence worth hiding: it is why `d_c` has a
closed form at all. It also means the Turing determinant condition is satisfied
automatically for every positive `a, b`, and the whole onset question reduces to
one inequality in `f_u` — a much cleaner structure than Gray-Scott, whose
non-trivial state stops existing altogether over most of its plane.

## 2. Whether "selection" means anything here at all  (s1, s2)

The first question, because everything else depends on it: **how many integer
modes are unstable?** A periodic box of length `L` admits only `q = 2 pi j / L`, and
if exactly one of those is unstable then "the pattern selects the fastest-growing
mode" is not a measurement — it is linear theory noting that the only mode that
could grow, grew. This project has a name for that: *a threshold nothing can fail
is not a check.*

Measured at `n = 512`, mode count in the unstable band:

| `d/d_c` | `L=1` | `L=2` | `L=4` | `L=8` | `L=16` |
|---|---|---|---|---|---|
| 1.02 | 1 | 1 | 2 | 4 | 8 |
| 1.05 | 1 | 1 | 3 | 7 | 14 |
| 1.20 | 1 | 3 | 6 | 12 | 25 |
| 2.00 | 3 | 5 | 11 | 22 | 46 |

So near onset in a small box the claim **is** vacuous, and the box length is the
lever that buys it content. `L = 8` at `d/d_c = 1.2` gives 13-17 unstable modes
depending on resolution, and that is where the rest of the slice sits.

**Correction, made when the shipped test was written from this table.** The row
above holds `n` fixed, which means a longer box is also a *coarser* one, and §3
shows the coarseness moves the band on its own — so the table conflates two
variables. Re-measured at fixed grid **spacing** (`h = 0.03125`, `n` proportional to
`L`), the `d/d_c = 1.2` counts are **1, 3, 6, 13, 25** for `L = 1, 2, 4, 8, 16`; the
`L = 8` entry is 13 rather than 12. The qualitative conclusion is unchanged and the
shipped test asserts the fixed-spacing numbers, since those isolate the box from the
resolution.

The competing hypotheses were then enumerated in arithmetic, before any simulation:
the **band centre**, the **lowest unstable mode**, and the fastest mode of the
**continuum** operator `-D q^2` instead of the stencil. The lowest unstable mode
separates from the fastest one everywhere. The band centre separates poorly right
at onset (at `d/d_c = 1.05, L = 4` centre and fastest are the same integer) and
well further out. Recorded before spending a single integration step on it.

## 3. The first design was wrong, and the grid is why  (s2)

The plan at this point was to coarsen the grid until the stencil and the continuum
disagreed about the fastest mode, so that selection would discriminate them. That
went straight into a wall, and the wall is worth recording because it is *not* the
Gray-Scott lesson repeating — it is its limit.

At `L = 16, n = 128` the stencil's fastest mode is `j = 64`, which is **Nyquist**:
`pi j / n = pi/2` exactly. The "prediction" there is a grid-scale checkerboard, and
the measured pattern came out at 56-62, so the prediction was simply wrong. One
step coarser (`n = 96`) the instability **vanishes entirely** — the largest
`q_eff^2` the stencil can represent is `4/h^2 = 144`, below the unstable band's
`q^2 ~ 347`, so every representable mode is stable. A grid too coarse to carry the
pattern does not model the pattern badly; it models a different problem.

Two further consequences of over-coarsening, both measured: the pattern's second
harmonic **aliases** (at `j = 60` on `n = 128`, `2j = 120` exceeds Nyquist `64`),
and the spectral peak holds only 11-37% of the power, so the peak and the
power-weighted centroid disagree by 5 or more modes — more than the mode spacing,
which makes the choice of estimator a load-bearing decision rather than a detail.

Resolution is therefore not free to tune: it must resolve the wavelength, and
`h = 0.03125` (about 10.6 cells per wavelength) is the reference grid.

## 4. Selection happens, and it does not track the initial condition  (s3)

Holding `h` fixed and varying the box, from a random perturbation of amplitude
`1e-3`, four seeds, `t_max = 1500`:

| `L` | `n` | band | fastest `j` | centre | lowest | emergent peaks | peak power |
|---|---|---|---|---|---|---|---|
| 2 | 64 | 5-7 | 6 | 6.0 | 5 | 6, 6, 6, 6 | 0.9997 |
| 4 | 128 | 10-15 | 12 | 12.5 | 10 | 12, 12, 12, 12 | 0.9997 |
| 8 | 256 | 19-31 | 24 | 25.0 | 19 | 24, 24, 24, 24 | 0.997-0.999 |
| 16 | 512 | 38-62 | 48 | 50.0 | 38 | 48, 48, 47, 48 | 0.46-0.88 |

**The initial-projection trap is clear, and this is the slice's cleanest positive
result.** The mode most loaded by the random noise at `t = 0` was `112, 32, 24, 17`
across the four seeds of the `L = 8` row; all four runs ended at `24`. The emergent
wavelength is set by the growth rate, not by which mode the noise happened to
favour. Varying the noise amplitude over `1e-4 ... 1e-2` changes nothing either.

At `L = 16` the peak power fraction collapses to 0.46-0.88 and one seed lands off
by one: a long box holds defects for a long time. Bigger is not better here.

## 5. The 4/4 above is seed-luck, and 8 seeds say so  (s5)

The `L = 8, n = 256` row read 4/4 exactly on seeds 0-3. At **8** seeds the same
configuration reads **6/8** — two seeds land at 23 and 25. Per-seed exact equality
is **false**, and had the slice stopped at four seeds it would have been asserted.
This project's own rule (verify at 3-4 seeds) was not enough; the claim is about a
distribution, so it needs the distribution.

The resolution sweep at `L = 8, d/d_c = 1.2`, 8 seeds each, is where the shape of
the real result appears:

| `n` | cells/wl | band | fastest `j` | continuum `j` | emergent peaks | at fastest | at continuum |
|---|---|---|---|---|---|---|---|
| 96 | 3.56 | 20-46 | 27 | 24 | 28,29,27,27,28,27,26,27 | 4/8 | **0/8** |
| 112 | 4.31 | 20-36 | 26 | 24 | 26,26,27,26,27,25,26,27 | 4/8 | **0/8** |
| 128 | 5.12 | 20-34 | 25 | 24 | 25,25,25,25,26,25,27,25 | 6/8 | **0/8** |
| 160 | 6.40 | 19-32 | 25 | 24 | 25,25,25,24,25,25,25,26 | 6/8 | 1/8 |
| 192 | 8.00 | 19-31 | 24 | 24 | 25,24,24,26,25,25,24,24 | 4/8 | (same) |
| 256 | 10.67 | 19-31 | 24 | 24 | 24,24,24,24,24,23,24,25 | 6/8 | (same) |

The **0/8 column** is the finding, and it needs stating exactly, because the loose
version ("never once") contradicts the table's own `n = 160` row. Across the four grids
where the two operators name different integers, the emergent peak equalled the
continuum's answer **once in 32 runs** — one seed at `n = 160`, the grid where the two
answers are adjacent — and **0 times in 24** on the three coarser grids, where they are
2 to 3 modes apart. That
is Phase 2b's central lesson — the simulation obeys the operator being integrated,
not the equation on paper — arriving in the **nonlinear** regime, where it was not
obvious it would survive.

## 6. The bias was in the prediction, not the measurement  (s6, s7)

At 16 seeds the mean emergent mode sat **2.78 SE above** the fastest integer mode
at `n = 112`, 1.78 SE at `n = 160`, 0.81 SE at `n = 256` — a "bias" shrinking with
resolution, which looked like a discretization effect to be characterised.

It was mostly an artefact of comparing against the wrong quantity. An ensemble
**mean** over integer-valued peaks need not be an integer, while
argmax-over-integers is one by construction. The quantity a mean should track is
the **continuous** maximiser of the discrete operator's growth rate, in mode units
`j* = q* L / 2 pi`:

| `n` | fastest integer | continuous maximiser | measured mean (R=16) | vs integer | vs continuous |
|---|---|---|---|---|---|
| 112 | 26 | 26.0978 | 26.4375 ± 0.1573 | 2.78 SE | **2.16 SE** |
| 160 | 25 | 24.7967 | 24.6875 ± 0.1760 | 1.78 SE | **0.62 SE** |
| 256 | 24 | 24.1819 | 24.1250 ± 0.1548 | 0.81 SE | **0.37 SE** |

The continuum operator's continuous maximiser is `23.8285` — one number for every
`n`, since it does not know about the grid — and the measured means sit **16.59,
4.88 and 1.92 SE** away from it. The coarse grid is where the claim has teeth; the
fine grid is where the two operators agree and there is nothing to discriminate.

Same pattern this project keeps meeting from a new side: the number was fine, the
estimator's **target** was wrong.

## 7. And then the ship configuration failed the only test that could catch it  (s8)

`n = 160` looked like the config to ship: 0.62 SE from the prediction at R=16, and
about 1.8 s a run. The residual was checked against replicate count, because a
deviation that is real does not shrink while its standard error does:

| R | mean | SE | deviation from prediction |
|---|---|---|---|
| 8 | 25.0000 | 0.1890 | 1.08 SE |
| 16 | 24.9375 | 0.1434 | 0.98 SE |
| 32 | 24.9688 | 0.0951 | 1.81 SE |
| 48 | 25.0208 | 0.0815 | **2.75 SE** |

The mean does not move; the error bar shrinks under it. Extrapolated, `z = 4` is
reached around `R = 150`. **A test at R = 16 there would have passed because R was
small**, which is 3c's "a bias cannot be measured where it is largest" turned
inside out, and it is exactly what the check was run to find.

Worse for the tidy version of the story, the two candidate targets **disagree about
which is right, in opposite directions**:

* `n = 160`: measured `25.0208`, integer 25 (**0.3 SE**), continuous `24.7967` (2.75 SE).
* `n = 256`: measured `24.2500`, integer 24 (2.3 SE), continuous `24.1819` (**0.62 SE**).

Neither target survives at 4-SE precision on both grids. The horizon explains part
of it and makes it worse rather than better: as the run gets longer the mean drifts
*toward the integer*, because a periodic box admits nothing else. At `n = 160` the
mean runs `24.875 → 24.833 → 24.896 → 24.896 → 25.021` over `lambda* t = 12, 16,
20, 24, 30`, and stops there. At `n = 256` it settles by `lambda* t = 16` and holds
`24.25` from then on.

**So the emergent wavenumber is a quantized quantity whose ensemble mean has no
single scalar prediction accurate to 4 SE, and the honest claim is a
discrimination, not an agreement.** What every configuration supports:

* the prediction from the **stencil** is within about a third of a mode spacing;
* the prediction from the **continuum** is wrong by 1 to 2.6 mode spacings
  (`3.7-16.6 SE`);
* the **band centre** is wrong by `4.4-9.9 SE`, the band **edges** by `33-74 SE`;
* every replicate, at every configuration, lands **inside** the unstable band.

A continuous estimator (parabolic peak interpolation, or a power-weighted centroid)
was considered as a way to make the measured quantity match a continuous
prediction, and rejected: the pattern really is an integer mode on a periodic box,
so a sub-mode estimator would report the integer with tiny scatter and dress up the
quantization rather than remove it.

### 7a. Half of §7 did not survive the shipped step size (s10)

**Everything above §7 was integrated at `0.4 x CFL`, which does not divide `t_max`
exactly and so is not a step size the model will accept.** Re-taken through the
shipped code — `dt = 0.1` at `n = 112`, `0.05` at `n = 160`, both about `0.36 x CFL`
and both dividing `t_max` — two of §7's conclusions change:

| | `n = 112` (R = 8 → 48) | `n = 160` (R = 8 → 48) |
|---|---|---|
| mean | 26.25 → 26.42 | 25.00 → 24.81 |
| gap to continuous maximiser | 0.61 → **2.24 SE** | 0.76 → **0.17 SE** |
| gap to fastest integer | 1.00 → 2.93 SE | 0.00 → 2.03 SE |
| margin vs continuum operator | 9.08 → **15.95 SE** | 3.62 → **10.46 SE** |
| margin vs band centre | 6.39 → **8.89 SE** | 1.11 → **7.26 SE** |

1. **The `n = 160` trap does not reproduce.** At the shipped step size its deviation
   *falls* to 0.17 SE at R = 48 instead of climbing to 2.75. The mechanism §7
   describes is real and was worth finding — a fixed mean under a shrinking error bar
   — but the number belonged to the slice's estimator, not to the model. Sixth
   instance in this project of a number travelling only with its estimator, and the
   first where the estimator was *this document's own*.
2. **The targets no longer disagree.** At the shipped step size the continuous
   maximiser is the better target on **both** grids (0.17 and 2.24 SE, against 2.03
   and 2.93 for the best integer). So "the best target flips between grids" was also
   a step-size artefact. What survives is that no target is exact: `n = 112` carries a
   real residual of about `+0.32` modes that grows to 2.24 SE by R = 48 and would
   cross `z = 4` near R ≈ 190.
3. **What the design got right, and it is now measured rather than assumed.** Margins
   *grow* with replicates at both grids, roughly as `sqrt(R)`, because they are
   differences of gaps divided by a shrinking error bar. So more replicates make a
   discrimination report **stronger**, where they eventually break an equality check.
   That asymmetry is the whole reason for the design and it is why the conclusion of
   §7 stands even though half its numbers did not.

**The shipped configuration, and why it is not R = 16.** `n = 112` at `R = 32`,
`z = 4`, verified at four top-level seed-sets: weakest margin `6.68 SE`, i.e. 1.67x
of headroom. `R = 16` was **rejected for being seed-lucky** — it clears the
band-centre competitor by `4.11 SE` on one seed-set of four, 0.11 above the
threshold. `n = 160` ships as the *contrast* case at `R = 8`, where the report
correctly **fails**: the prediction agrees better (0.76 SE) and yet the band centre
cannot be excluded (1.11 SE), at 4/4 seed-sets with a margin never above 2.81. Better
agreement and weaker discrimination at once, which is the trade-off the coarse grid
resolves.

## 8. Supercriticality — the precondition, and a number that is not the exponent

Near-onset selection is an argument about a **supercritical** bifurcation: the
amplitude has to grow continuously from zero rather than jump. Measured at
`n = 112`, horizon scaled as `60/lambda*` so near-onset points are actually
saturated:

| `d/d_c` | `eps = d - d_c` | `lambda*` | amplitude | `amp/sqrt(eps)` |
|---|---|---|---|---|
| 1.01 | 0.07630 | 0.00418 | 0.12587 | 0.45570 |
| 1.02 | 0.15260 | 0.00851 | 0.17582 | 0.45009 |
| 1.04 | 0.30519 | 0.01693 | 0.24299 | 0.43985 |
| 1.08 | 0.61038 | 0.03288 | 0.33171 | 0.42458 |
| 1.16 | 1.22076 | 0.06201 | 0.45754 | 0.41411 |
| 1.32 | 2.44153 | 0.11186 | 0.62257 | 0.39843 |

Continuous from zero, monotone, no jump: **supercritical**, so the precondition
holds and Gray-Scott's subcritical outcome does not repeat here.

**The exponent, however, must not be quoted.** A log-log fit over that range gives
`0.4606`, which is not `1/2` and is not evidence against `1/2` either — it is the
`O(eps)` correction leaking into a slope fitted outside the asymptotic regime.
Richardson in the bifurcation distance (`c ~ 2 m(eps) - m(2 eps)`, first order)
gives `0.4613, 0.4603, 0.4551, 0.4351, 0.4298` as the pair moves outward: the
extrapolated constant is *itself* still drifting, so the correction is not purely
linear either. Recorded as "consistent with the square-root law with corrections",
with the slope number labelled as what it is. Sixth instance in this project of a
number that only travels with its estimator.

## 9. The category-A anchor, which is sharp and nearly free  (s4)

The seeded-single-mode growth rate, against the **discrete** dispersion relation,
at `L = 8, n = 256, d/d_c = 1.2`:

| `j` | `pi j/n` | `eps` | measured | predicted (stencil) | rel err | continuum is off by |
|---|---|---|---|---|---|---|
| 24 | 0.295 | 1e-4 | 0.075533 | 0.075533 | 2.1e-06 | 0.003% |
| 24 | 0.295 | 1e-5 | 0.075533 | 0.075533 | 2.1e-08 | 0.003% |
| 24 | 0.295 | 1e-6 | 0.075533 | 0.075533 | 2.1e-10 | 0.003% |
| 19 | 0.233 | 1e-5 | 0.012362 | 0.012362 | 8.0e-10 | **39.3%** |
| 40 | 0.491 | 1e-5 | -0.233633 | -0.233633 | 1.0e-08 | **26.5%** |
| 60 | 0.736 | 1e-5 | -1.065482 | -1.066399 | 8.6e-04 | **33.2%** |
| 100 | 1.227 | 1e-5 | -2.792932 | -2.792914 | 6.6e-06 | **89.9%** |
| 128 | 1.571 | 1e-5 | -3.250637 | -3.250638 | 2.0e-07 | **184%** |

Exact to 8-10 digits, **spanning a sign change** (`j = 19` grows, `j = 40` decays),
at 0.01-0.25 s a probe, and the continuum form is wrong by up to 184%. The error
falls linearly in the seeded amplitude, which identifies the residual as the
nonlinear correction rather than anything about the stencil.

One probe **refuses**: at `j = 12` the eigenvalue pair is **complex**, and the
endpoint log-ratio reads `-0.562119` against a common real part of `-0.546667`,
2.8% off, because the amplitude oscillates while it decays. Gray-Scott's refusal
case, reproduced in a second model, with the error it produces when ignored.

## 10. Two dimensions  (s4)

`L = 4, n = 128`, random start, radial power spectrum of the final field:

| `d/d_c` | fastest `j` | band | radial peak (2 seeds) | amplitude | cost |
|---|---|---|---|---|---|
| 1.2 | 12 | 10-15 | 12, 12 | 0.70, 0.69 | 53-58 s |
| 2.0 | 11 | 7-17 | 12, 12 | 1.30, 1.29 | 97-105 s |

Two dimensions selects too, at the mildly-supercritical point. At `d/d_c = 2` it
misses by one, in the same direction as every 1-D measurement. At about a minute a
run it belongs in a demo, not in the suite.

## 11. What the slice establishes, and what it does not

**Establishes.** Three claims, in descending sharpness: the seeded-mode growth rate
against the stencil (exact, spans a sign change, refuses complex pairs); the onset
closed forms (`u*`, `v*`, `det J = u*^2`, `d_c`, `q_c`, each against an independent
route); and wavelength selection as a **discrimination** — the emergent mode is set
by the growth rate rather than by the initial noise, lands inside the unstable band
at every replicate, sits within a third of a mode spacing of the stencil's
prediction, and is many standard errors from the continuum's prediction, the band
centre and the band edges.

**Does not establish.** That the ensemble mean of the emergent mode equals any
single scalar to 4 SE — measured, twice, and it does not. That the amplitude
exponent is `1/2` — consistent with it, but no configuration here measures it
inside its asymptotic regime. That 2-D selection is more than a two-seed
observation. And nothing about `(a, b)` other than `(0.05, 1.0)`: `d_c` was checked
at four parameter points but every dynamical number above is at one of them.

## 12. Consequences for the plan

1. **The selection claim must not go into `analytic_predictions`.** `validate()`
   asserts `|mean - predicted| <= z * SE`, and §7 shows that assertion failing as
   `R` grows on one grid and the target flipping between grids. It would be a test
   that passes because the replicate count is small. `analytic_predictions` returns
   the seeded-mode `growth_rate` and **refuses** the random-start initial condition
   — a refusal on *statistical* grounds, which is a new kind for this project.
2. Selection gets its own report, the way `repressilator` is validated by
   `convergence_report` rather than by a prediction dict: mean, SE, and the
   separation from each competing hypothesis, with the assertion on the
   **discrimination margin**.
3. Ship the coarse grid for the discrimination (`n = 112`, where the operators
   differ by 2.27 modes and the continuum is excluded at 13-16 SE) at `R = 32`, and
   the `n = 160` grid as the **contrast** case at `R = 8`, where the report correctly
   fails on the band centre. See §7a for both, and for why `R = 16` was rejected.
4. Refuse a horizon below `lambda* t = 20` — swept at 12/16/20/24/30, and then
   checked far out (§13): the selected mode is *identical* at 22.7, 45.3 and 113.3
   e-folds on both shipped grids, so past the threshold the answer is
   horizon-independent rather than merely slow-moving.
5. `dominant_mode` is `nan` before the pattern exists, the way Gray-Scott's
   `growth_rate` is `nan` at `t = 0`; and `dt` must divide `t_max` exactly, which
   Gray-Scott already guards and this slice's own instrument got wrong (an
   `s4` snapshot at `t = 300` was silently missed by a float comparison).

## 13. What the build found that the slice had not

Every item here comes from building the thing, and three of them corrected something
written above.

**The horizon is settled, and generously.** The threshold was calibrated on a sweep
of the *mean*; checked instead on the selected mode itself, three seeds pick the
identical mode at `lambda* t = 22.7, 45.3 and 113.3` on both shipped grids. The
threshold is not a knife edge.

**A power-fraction number went into a docstring with one horizon behind it.** The
share of power in the dominant mode was recorded as "0.997-0.999 on a resolved grid",
which is true only at 113 e-folds. At the shipped 22.7 it reads `0.53-0.90` on the
fine grid and `0.45-0.70` on the coarse one, and on the coarse grid it then **freezes**
(`0.39-0.78` at 45 e-folds and unchanged at 113) because the harmonics it carries have
nowhere to go. At `1.2 d_c` the modulation is ~50% of the background: this is not a
weakly nonlinear pattern, and roughly half its power is genuinely in harmonics. The
docstring is corrected, and the distinction it blurred — how *sinusoidal* the pattern
is, versus whether the *mode* has settled — is now stated in both places.

**Three figure defects, found by looking, as in every phase that shipped one.** The
selection panel's field plot was a jagged sawtooth captioned as though it were a clean
pattern; the dispersion panel's "complex pair" curve was listed in the legend and
invisible on the axes, because at small `j` diffusion barely matters and it hides
exactly under the continuum curve; and the continuum's rejected prediction sat flush
against the axis spine, where a refuted claim reads as decoration. All three are fixed
in the figure rather than in its caption. The 2-D panel was then re-gridded from
`n = 96` to `n = 128` for the same reason: at 3.6 cells per wavelength the picture
shows the grid instead of the wavelength — and at `n = 128` both seeds hit the
predicted mode exactly (25 and 25), where `n = 96` had missed by one.

**Mutation: 20 of 22 killed, and the two real gaps were both "a check nothing can
fail".** The spectral estimator subtracted the mean *and* zeroed mode 0, so on a
symmetric field the subtraction cancelled exactly and deleting the zeroing changed
nothing — two mechanisms where one is testable. And a margin mutated from
`(gap_competitor - gap_prediction)/SE` to `gap_competitor/SE` survived every test,
because on the contrast grid both formulas land under `z = 4` and the test only
asserted that the report fails. Both now have their own assertions. The two remaining
survivors are provable no-ops: a tie-break in `argmax` that no real spectrum reaches,
and a population-vs-sample standard deviation, which moves a margin by 1.6% at
`R = 32` against 67% of headroom.

**Two test fixtures reached for parameter points that lacked the property they
needed**, which is worth recording because both looked obviously right. A mode past
the band edge is *not* a mode with an unmeasurably small rate — out there the rate is
large and negative, and perfectly measurable — so the near-zero-rate refusal is
reached by moving the *ratio* to onset instead (`1.001 d_c`, rate `2.15e-4`). And
`a = 0.02, b = 4.0` looks like a wildly unstable state and has `trace = -15.2`;
trace > 0 needs `(a+b)^3 < b-a`, so both parameters must be small (`a = 0.01,
b = 0.5` gives `+0.70`).
