# Phase 2c — plan: Schnakenberg and the wavelength-selection claim

**This is not Phase 5.** `HANDOFF.md` §8 says there is no Phase 5 in that document
and that inventing one is not the next action. This is Phase 2's one explicitly
deferred item — *"Schnakenberg / Brusselator for near-onset wavelength
selection"* — taken now because Phase 2 reframed HANDOFF §5's "validate pattern
wavelength against linear stability analysis" into a `lambda(q)` check and left the
*wavelength* half unclaimed. Phase 2's deferral list is amended in the same commit,
so a later reader does not find "deferred" sitting beside a built model.

Everything below follows `phase2c-schnakenberg-measurement.md`, which was taken
first. Numbers quoted here are measured, not targets.

## What is being claimed

Three claims, in descending sharpness.

**A1 — the seeded-mode growth rate** (category A, exact). Seed one Fourier mode
along the discrete operator's eigenvector; `log(a(T)/a(0))/T` reproduces the
stencil's dispersion relation to 8-10 digits, spanning a sign change, while the
continuum form `-D q^2` is wrong by 39% at a growing mode and 184% at Nyquist. Goes
through `analytic_predictions` and the ValidationSuite, exactly as Gray-Scott's
does.

**A2 — the onset closed forms** (category A, exact). `u* = a+b`,
`v* = b/(a+b)^2`, `det J = u*^2` *exactly*, `d_c = [u*(1+sqrt(1+f_u))/f_u]^2`,
`q_c^2 = u*/sqrt(Du Dv)`. Each asserted against an independent route — central
differences for the Jacobian, a bisection for `d_c`, an argmax for `q_c` — never
against a recorded literal alone.

**B — wavelength selection** (the new claim, statistical, and a *discrimination*).
From a random perturbation, the mode that emerges is set by the growth rate and not
by the initial noise; it lands inside the unstable band at every replicate; the
ensemble mean sits within about a third of a mode spacing of the **stencil's**
fastest wavenumber and 3.7-16.6 SE from the **continuum's**, 4.4-9.9 SE from the
band centre and 33-74 SE from the band edges.

## The design decision the measurement forced

**Claim B must not go through `analytic_predictions`.** `validate()` asserts
`|mean - predicted| <= z * SE`, and §7 of the measurement doc shows that assertion
failing as replicates grow at `n = 160` (1.08 → 0.98 → 1.81 → **2.75 SE** at
R = 8/16/32/48, the mean fixed while the error bar shrinks under it) and the best
target *flipping between grids* — the fastest integer at `n = 160`, the continuous
maximiser at `n = 256`. A prediction dict entry would be a test that passes because
the replicate count is small.

So `analytic_predictions` covers the seeded initial condition and **refuses** the
random one, with the reason stated in the refusal: the emergent wavenumber is
quantized by the box, and no single scalar matches its ensemble mean to 4 SE. This
is the project's fourth kind of refusal and its first on **statistical** grounds
(the others: no real state, a complex pair, a rate too near zero).

Claim B gets `selection_report`, the way `repressilator` is validated by
`convergence_report` rather than by a prediction dict.

## Build order

Each `[ ]` is a commit-sized unit; the validation test comes before the
implementation is correct and must be confirmed able to fail.

1. `[ ]` **`models/schnakenberg.py`, the algebra.** `SchnakenbergParams`
   (`a, b, du, dv, n, length, ndim, mode_j, eps, noise, initial, t_max, dt`),
   `homogeneous_state`, `reaction_jacobian`, `critical_ratio`,
   `critical_wavenumber`, `dispersion` (stencil, returns rate/eigenvector/is_real),
   `unstable_band`, `fastest_mode` (both the best integer **and** the continuous
   maximiser — they are different quantities and the measurement needed both).
   Params guards: CFL, `dt` divides `t_max` exactly, `mode_j <= n/2`, `ndim in
   (1, 2)`, `initial` in the allowed set.
2. `[ ]` **Tests for 1.** Closed forms against central differences, a bisection and
   an argmax; `det J = u*^2` at four `(a, b)`; the two `fastest_mode` routes pinned
   against each other; the band's integer content at the recorded configurations.
3. `[ ]` **The protocol model.** `initial_state` (two initial conditions),
   `step` (RK4), `observables`, `is_terminal`, `fields` (1-D **or** 2-D — Phase 4
   already reports `field_shapes` for a non-2-D field, so the front-end needs no
   change), `analytic_predictions` with its refusals. Register with its params type.
4. `[ ]` **Tests for 3 — claim A1.** `validate()` reproducing the seeded rate at
   `eps = 1e-4/1e-5/1e-6` (the residual must fall linearly in `eps`, which is what
   identifies it as the nonlinear correction); a growing and a decaying mode, so the
   check spans a sign change; the continuum form as a **tooth** (39% and 184% off);
   the complex-pair and near-zero-rate refusals.
5. `[ ]` **`selection_report` + tests — claim B.** Mean, SE, and the separation from
   every competing hypothesis, asserting the **discrimination margin** rather than
   agreement. Ship configs: `n = 112` for the discrimination (the operators differ
   by 2.6 modes there; the continuum is excluded at 16.6 SE) and `n = 256` for the
   agreement (settled by `lambda* t = 16`; 0.37-0.62 SE). Refuse a horizon below
   `lambda* t = 20`. Assert independence from the initial condition using the
   `initial_mode` observable. **`n = 160` is not a config — it is a recorded
   finding**, and a test pins it so the trap cannot be re-entered silently.
6. `[ ]` **Teeth, seed-verified at 4 seeds.** Continuum operator in `dispersion`;
   band centre as the prediction; lowest unstable mode; the transposed reaction
   Jacobian; `det J` mutated off `u*^2`. Each must be red for a *structurally*
   robust reason, and any leg that is not gets moved rather than loosened.
7. `[ ]` **`demos/schnakenberg.py`.** The dispersion curve across its sign change,
   the emergent 1-D pattern with its spectrum, the amplitude-vs-`sqrt(d - d_c)`
   panel **labelled with why its slope is not the exponent**, and a 2-D panel
   labelled qualitative. Every panel's claim checked by looking at the PNG, because
   in this project a figure has been wrong in every phase that shipped one.
8. `[ ]` **Close-out.** Re-time the suite with a same-session baseline *and* the
   xdist worker tags; update `CLAUDE.md`, `HANDOFF.md` §5's Phase 2 line, Phase 2's
   deferral list, memory; commit and push.

## Explicitly deferred (do NOT do here)

- **Brusselator.** Schnakenberg carries the claim; a second model buys nothing.
- **The amplitude exponent as an assertion.** Measured `0.4606` by log-log and a
  Richardson constant still drifting at `0.4613 → 0.4298`; no affordable
  configuration sits inside the asymptotic regime. Recorded, not asserted.
- **2-D selection as an assertion.** Two seeds, and about a minute a run. Demo only.
- **A sub-mode spectral estimator.** Considered and rejected: the pattern really is
  an integer mode on a periodic box, so a parabolic-interpolation peak would dress
  up the quantization rather than remove it.
- **Amplitude equations / weakly nonlinear theory.** The supercriticality
  precondition is checked by measurement (continuous from zero, monotone, no jump);
  deriving the Landau coefficient is a different project.
- **Promoting the spectral helpers to `core/`.** They live in the model until a
  second model wants them; Gray-Scott's genuine Turing sliver is the candidate, and
  that is a later decision with a real caller behind it.
