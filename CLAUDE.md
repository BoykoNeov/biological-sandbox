# CLAUDE.md — Biological Sandbox

Stochastic biological models and the deterministic limits they collapse into,
built **validation-first**. Read [`HANDOFF.md`](HANDOFF.md) for the full vision
and build order; this file is the working-session quick reference.

## Commands

```bash
uv sync --extra viz     # install deps + dev tools + matplotlib
uv run pytest -q        # run tests = run the ValidationSuite (the real check)
uv run pytest -q -n 0   # ...serially, when debugging (-n 6 is the default)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run python -m sandbox.demos.wright_fisher   # end-to-end demo
```

## The non-negotiables

1. **Protocol first.** Every model implements `core.protocol.Model`. Shared
   services (Recorder, sweep, ValidationSuite, viz) consume only the protocol
   surface — they never reach inside a model's concrete state.
2. **A core model is done only when its `analytic_predictions` are reproduced
   by the ValidationSuite within tolerance.** Validation *is* the definition of
   done, and the tolerance is statistical (`z * standard error`), never a
   hardcoded epsilon. Prefer this over inventing new pass/fail logic.
3. **Reproducibility.** Seed every stochastic run. Derive replicate RNGs with
   `core.rng.spawn_rngs` (`SeedSequence.spawn`) — never `default_rng(seed + i)`.
   Every `Experiment` serializes and re-runs to the same `Result`.
4. **Profile before optimizing.** Stay in NumPy until a real bottleneck forces
   numba/JAX/WebGL (expected first at Gray-Scott grids and HH networks). When you
   do optimize the SSA, the trajectory must stay **bit-identical** — every
   recorded slope anchor depends on it. Fingerprint a replicate before and after
   (`sha256` of times + series) rather than trusting that a change "looks safe".
5. **Quarantine the speculative arc** under `models/ecosystem/`. Any model
   without a checkable prediction is exploratory and must be labelled so in code
   and UI. Guard the verifiable/exploratory boundary continuously.

## Protocol conventions (easy to get wrong)

- `step(state, rng)` takes **no `dt`**: the model owns its own time increment and
  writes it into `state.t`. Numerical step sizes are *params*, not call args.
  (Wright-Fisher = one generation; Gillespie samples its own waiting time.)
- Model objects are **stateless**. `initial_state(params, rng)` embeds the
  params into the returned `State`, so `step`/`observables`/`is_terminal` are
  pure functions of the state. `analytic_predictions(params)` is the one method
  taking params directly.
- Optional methods: `is_terminal(state)` (early stop) and `analytic_predictions`
  (required for core models, omitted for speculative ones).

## Layout

`src/sandbox/core/` protocol + services · `models/` the models ·
`viz/backends/` pluggable plotting · `demos/` runnable examples ·
`tests/` wraps the ValidationSuite · `docs/plans/` per-phase dev docs.

## Workflow

- Add a model: write its `*Params`/`*State`, implement the protocol, register it
  in `models/__init__.py`, then write the validation test **first** and confirm
  it can fail before the implementation is correct.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`). Each
  commit should pass `pytest` and `ruff`.
- **Batch / session end:** update memory + `docs/`, then commit and push.

## Status

Phase 0 (Wright-Fisher) complete and validated. **Phase 1 complete**: RK4
integrator + `DeterministicLimitModel` protocol, Gillespie SSA engine,
`birth_death` and `isomerization` (the two exact closed-form checks),
`core/convergence.py` (the second validation track: log-log slope of `D(Omega)`
against `-1/2` with a statistical `max(bootstrap, OLS)` SE), `repressilator` (the
headline — no `analytic_predictions`, validated by `convergence_report` at slope
`-0.4606 +/- 0.0734` with two broken-`Omega` teeth), `viz`, and
`demos/repressilator.py`.

**Phase 2 complete** — Hodgkin-Huxley (2a) and Gray-Scott (2b). See
`docs/plans/phase2-{plan,context,tasks}.md`; the tasks doc carries every measured
number. Suite **310 passed**, in 130 s at `-n 6` when Phase 2 closed — but that
figure **does not travel**: re-timed clean on 2026-08-10 the same suite takes
**203-232 s** with the repressilator floor test at **162 s** against its recorded
122 s, i.e. the machine is ~1.33x slower and nothing regressed. **Always re-time in
the same session before comparing.**

Two decisions shaped the phase. First, **three categories of checkable claim** —
*A* exact analytic (`analytic_predictions`), *B* asymptotic law (log-log slope +
CI), *C* literature-anchored (rheobase, f-I, spike shape) — and **category C never
enters `analytic_predictions` nor an assertion's bound**; it is reported in demos,
and only *structural* facts about it are asserted. Second, HANDOFF's "validate
Gray-Scott's pattern wavelength against LSA" was **reframed after measurement**: at
Pearson's famous `(F, k)` there is no real non-trivial homogeneous steady state, so
those patterns are not Turing patterns and LSA makes no claim about them. Phase 2
validates **`lambda(q)` itself** and labels the emergent pattern qualitative.

**2a — Hodgkin-Huxley.** `hh_rates` (six rate functions, `_linoid` for the removable
`0/0` at `V=-40`/`-55` via `expm1`), `hh_voltage_clamp` (the category-A lead anchor:
clamped gating decouples into an exact `x_inf + (x0-x_inf)e^{-t/tau}`),
`hodgkin_huxley` (deterministic 4-D; `analytic_predictions` is the root-found fixed
point and **raises** past the subcritical Hopf), `hh_stochastic` (8-state Na +
5-state K **occupancy counts**, `O(#states)` per step and independent of `N`, with
an **exact factorized propagator** `P = kron(P_m, P_h)` verified against a
hand-transcribed generator + local `expm` at 7.4e-15), and the headline
**`D(N) ~ N^{-1/2}`** at slope `-0.5092 +/- 0.0106` (seeds 0-3: `-0.5092, -0.4933,
-0.4988, -0.5101`), with two teeth. `demos/hodgkin_huxley.py` reports category C.

**2b — Gray-Scott.** `core/laplacian.py` (periodic 5-point stencil; a Fourier mode is
an **exact** eigenfunction, pinning stencil + wrap + integrator at once; order-2
consistency slope `1.99835`), `gray_scott` (`analytic_predictions` = the growth rate
from the **discrete** operator), `lambda(q)` validated at 8 probes **across a sign
change**, the `FieldModel` viz-only extension, and `demos/gray_scott.py`.

## Lessons worth carrying forward

**On tolerances.** Match the tolerance to the *claim*, and derive it rather than
typing it. Richardson in `dt` sees discretization error and nothing else — it cannot
see leftover transient (HH attraction) and it cannot see a *linearization* error
(Gray-Scott), where the right instrument is **Richardson in the amplitude**:
`(4/3)|m(a) - m(a/2)|` predicted the true error to a ratio of **1.000** at all eight
probes. For a deterministic model `validate()`'s statistical SE degenerates to zero,
so supply a numerical `sem_floor` and use two replicates (one gives `sem = inf` and
passes vacuously).

**On tests that are green for the wrong reason.** This phase produced five, and they
are the most valuable thing in it:

- `assert std > 0.0` passed a *rounded* initial condition, because `np.std` of sixty
  bit-identical values returns `2.7e-20`, not `0.0`. **A threshold nothing can fail
  is not a check.**
- A conservation test ran at exactly `N = 5000`, where `round(N p)` happens to sum to
  `5000` for both populations. At `N = 10000` it sums to `10001`. **Sweep the
  constant that the probe happens to land on.**
- A mutant that **never moved a single channel** (multinomial aggregated along the
  wrong axis) passed everything, because at a fixed point a frozen membrane sits
  still exactly like a correct one. Caught only by a *transient* regime and a **ratio
  between two system sizes** — an `N`-independent error has no tolerance that
  distinguishes "small" from "not shrinking".
- Asserting `linspace` grids misalign **failed**: at stride 8 they align perfectly,
  because dividing by a power of two is exact. 373 other configurations misalign. The
  hazard is not "it is wrong" but "it is right until someone changes `n_grid`".
- The order-2 slope read `1.9737` and the shortfall was *real*, not noise: `k h / 2`
  was `0.785` at the coarsest grid. **An asymptotic order is only measurable inside
  the asymptotic regime**, and which grids are inside it is part of the claim.

**On teeth.** Verify at 3-4 seeds and assert only the leg that is *structurally*
robust for that break. A tooth must also bite the right thing: the repressilator's
`Omega^2` tooth could not be reused for HH, because squaring `N` drives `D` an order
of magnitude below the splitting bias and the broken model would fail through a
*discretization floor* rather than through its scaling.

**On performance, both directions.** Profile before optimizing — and record the
measurement when it says "do nothing". Twice here the *obvious* optimization was
backwards: a "vectorized" NumPy propagator build cost **80.5 us/step against the
72 us naive loop it replaced** (on a 4x4 there is no arithmetic to amortize, only
per-call overhead), and `np.kron` cost 5.5x a broadcast-and-reshape. Batching 13
multinomial draws into 2 was a **wash** (13.30 vs 13.46 us). Equally, the Phase-1
close-out's proposed `-n` fix — dispatch the long tests first — measured **75%
worse** (228 s vs 130 s) and was reverted.

**On figures.** A figure carries a claim and gets looked at, not exit-code checked.
Four claims were wrong until someone looked: an f-I curve that put rheobase at
`[2, 4]` by counting onset spikes; channel noise demonstrated at `I = 20`, where the
drive swamps it; a dispersion curve drawn solid through the region where the
eigenvalue pair is *complex*; and a Turing panel implying its stripes were a
*selected* wavelength when the mode had been seeded by hand.

**On refusing to answer.** `analytic_predictions` raises rather than returning a
wrong number that still looks green — past the subcritical Hopf, where no real
non-trivial homogeneous state exists, on a complex eigenvalue pair, and on a growth
rate too near zero to measure (`j=13` needs `t_max = 18822` and returns `nan`).

**Phase 3 is planned, not built** — `docs/plans/phase3-{plan,context,tasks}.md`.
HANDOFF §6's arc into the speculative, taking the three stops that still have a
checkable prediction: **gLV + the May/Allesina-Tang random community matrix**,
**Daisyworld**, **adaptive dynamics**, plus a **demographic-noise gLV** (priced at
15.6 s, slope `-0.4984 +/- 0.0488`) to carry the stochastic-vs-limit thread. The
`models/ecosystem/` quarantine stays **empty**, and that is the correct outcome.

Two reframes, both measured before planning. First, **HANDOFF's May promise cannot
be run as written**: the random-gLV ensemble is *empty* at the `S` where the
asymptotic criterion means anything (feasibility `0.000` at `S = 40, sigma = 0.25`)
and conditioning on feasibility *moves the spectrum* (`max Re eig(A) = -0.335` vs
`max Re eig(diag(x*)A) = -0.117`). Phase 3 validates the **matrix law directly** —
the circular law, and better the **elliptic law**, where predator-prey correlation
makes a web *more* stable — and refuses to claim the composition. Second,
**Daisyworld's regulation turned out to be closed-form**: `beta(T_w) = beta(T_b)`
pins `T_w, T_b = T_opt -+ delta` with `delta` from a cubic in `q` alone, so
`T_w* = 290.5117`, `T_b* = 300.4883` and the bare fraction `x* = 0.32653` are all
**independent of luminosity** — `dT_w/dL` measured *exactly* `0`, and `dT_e/dL` is
*negative*, i.e. overcompensation rather than mere flattening.

**3a is built** — `glv` and `lotka_volterra`, 47 tests in ~6 s; suite **357 passed**
(310 + 47) in 171 s at `-n 6`. That 171 s is a **regression check, not a timing**:
it is *below* the 203-232 s baseline despite 47 new tests, and the run shared the
machine, so it settles nothing. Step 18 re-times cleanly with `--durations=8` — if
the repressilator floor test also dropped from its recorded 162.17 s it is the
machine, and if the floor held while the total fell then xdist packing changed
when 47 fast tests entered collection, which this suite has been burned by once.
gLV carries the
interior equilibrium (validated on *two* hand-built systems, because the 3-species
case is self-consistency only and the 2-species symmetric closed form
`x* = 1/(1+a)` is not), the community matrix `diag(x*)A` checked signed against
central differences, and the relaxation rate as an `O(eps)` limit. It **refuses**
on a singular `A`, an infeasible `x*`, an unstable `x*`, and — for the relaxation
claim alone — a complex slowest pair. `lotka_volterra` is the project's only
deterministic model validated **on the orbit**: conserved `V` far from any fixed
point, `<x> = x*` across a 10x amplitude range, and the small-oscillation period
as an extrapolated limit.

**Two recorded numbers did not survive re-measurement, and measuring first is
what caught them.** The relaxation constant: the slice *fitted* `log|x - x*|`
over a window (`3.03e-4` at `eps = 1e-2`), this model takes a single **endpoint**
log-ratio (`4.27e-3`, and drifting with the horizon). The `O(eps)` *scaling*
transferred; the constant did not. And the LV period probes `amp = 1.2/2.0/4.0`
are **outside the asymptotic regime** — `excess/amp^2` runs `0.0282/0.0255/0.0208`
there but `0.03336/0.03309/0.03256/0.03156` at `amp = 0.05...0.4`, so an
extrapolation fitted to the plan's points would not have been measuring a limit.
Separately, the plan's `t_max = 20` order-4 trap **did not reproduce** (15.18/
15.59/15.80, not 65.89/5.76/12.65/15.09): it depends on the initial condition and
error norm, which the slice did not record. **A number is only transferable
together with the estimator that produced it.**

Richardson in the amplitude for the fourth and fifth time (HH transient,
Gray-Scott linearization, now gLV relaxation and the LV period). gLV's is *first*
order, so the factor is `2|m(eps) - m(eps/2)|`, not `4/3` — and it predicts the
true error to a ratio of `0.9954 ... 0.9995`.

All 13 3a mutants confirmed red, including two **test-side** ones. The one worth
carrying: transposing `A` **everywhere** leaves `test_validate_reproduces_the_
equilibrium` green, because a consistently transposed system still has a genuine
fixed point. Only the hand-written double-loop RHS catches it.

**3b is built** — `core/random_matrix.py` (**not** a `Model`: no `step`, no
`observables`, no `state.t`; a plain pytest outside the `ValidationSuite` path) plus
`demos/random_matrix.py`. 32 tests; suite **389 passed in 240.59 s** at `-n 6`.

**Pinning the ensemble before writing a tolerance paid for itself immediately.** The
convention was *recovered from the recorded tables rather than guessed*: the elliptic
`pred R(1+rho)` column `4/12/20/28/36` solves to `R = 20` at `S = 400`, hence
`sigma sqrt(C) = 1`; and only the map `(Re/(R(1+rho)), Im/(R(1-rho)))` reproduces the
recorded `pred = 0.25`. Under it both tables reproduce, `E[max Re]` included — and
`E[max Re]` is the real pin, since the *fraction* is insensitive to `d` and to the
overall scale of `sigma`/`C` (both absorbed into `R`), which is exactly what makes a
`rho^2` failure diagnostic of the **draw**.

**Then the plan's bias constant collapsed under re-measurement — a fourth instance of
"a number travels with its estimator", with a new mechanism.** Re-running the plan's
own estimator reproduced the bias at `S = 50/100/200` but not at `S = 400/800`, where
it sits **below its own SE** (`z = 0.69, 0.90`): the recorded slope `-0.9279` was
partly fitted through noise. Re-measured where every point resolves, it is `0.70/S`,
slope `-1.0086`. Worse, `0.70/S` is itself a `probe = 0.5` artifact: `bias * S` spans
`0.48...1.13` across circular probes, and the *exponent* differs too — `-0.8814`,
`-1.0274`, `-0.6189` at `probe = 0.2/0.5/0.9`, with elliptic `rho = +0.8` at `-0.8404`
on a bias 4-5x larger. **The tempting bulk/edge mechanism is wrong**: it predicts
`probe = 0.2`, deep in the bulk, at `-1`; it measures `-0.8814`, shallower than
`probe = 0.5`. Recorded as probe-dependent and unexplained.

That killed the single-track design. The direct fraction check needs `bias << SE` and
the scaling check needs `bias >> SE`, so they reach **opposite** probes: the one
affordable direct configuration is `probe = 0.5, S = 400, 9 draws` (bias `0.24 SE`,
draws *derived*, not chosen), while the asymptotic law is asserted on elliptic
`rho = +0.8` — unassertable directly at any affordable `S`, cheap here. And no larger
`S` rescues it: `eigvals` costs 0.018 s at `S = 200`, 0.15 s at `400`, then **2.0 s at
`600` — 13x for 3.4x the FLOPs.** `S = 400` is the last cheap size.

**The scaling assertion was green for the wrong reason until the teeth caught it** —
the phase's sixth such test, and the sharpest. It asserted only that the exponent was
*significantly negative*, and a wrong-`R` draw passed: a constant offset fits a
near-perfect line, so its exponent reads `-0.0089 +/- 0.0015`, clearing zero at 6
sigma, while the discrepancy sits at `0.33` and does not move. **Statistical
significance is free when the residuals are small.** The check now demands a decay
*rate* below `-0.15`, a threshold placed in a measured gap (~90 SE above the teeth,
~7 SE below the shallowest correct probe). And **one tooth had to be moved**: wrong
`R` does not bite the scaling check at all (its offset itself decays, `S^-0.34...-0.42`,
passing 3 of 4 seeds) and now lives on the direct probe, where it bites at `>5 SE` —
the repressilator `Omega^2` lesson again. Flipped-`rho`-sign and Hermitian-ized bite
at 4/4 seeds, and the flipped-sign tooth's blind spot (bit-identical at `rho = 0`) is
asserted rather than left implicit.

**The open 3a timing anomaly is closed as *not decidable*.** The floor test was
measured three times in one session — `123.03` and `154.99 s` serial-and-alone, and
`189.81 s` in-suite — a `+-30%` spread larger than the effect the dichotomy was meant
to detect, and across three different estimators besides. **Bracket any suite-timing
claim on this machine by `+-30%` or it is noise.**

**3c is built** — `models/glv_stochastic.py` (`S` births + `S^2` losses on the
Phase-1 Gillespie engine, one `rates(c)` driving both the propensities and
`deterministic_rhs`), `demos/glv_stochastic.py`, 21 tests. `D(Omega) ~
Omega^{-1/2}` at `-0.4952 / -0.5134 / -0.4960 / -0.5228` over seeds 0-3, with a
fixed-`Omega` and a sqrt-`Omega` tooth. **The phase's one open item is closed.**

**The `O(1/Omega)` bias, and why the plan's number was not wrong so much as
under-generalized.** `bias = (-A)^-1 diag|A_ii| 1 / Omega`; the plan's `x*/Omega`
is that formula specialized to `diag|A_ii| 1 = r`, true of its symmetric
2-species reference and false of 3a's asymmetric one. Closing it needed a
**split-coupled** estimator (Anderson 2012): the macro arm *is* the exact arm plus
`S` extra loss channels, so both run as ONE chain — shared channel `min(aE, aM)`
plus E-only and M-only — for about one arm's events. At equal cost the SE is
`6.208e-3` independent, `1.357e-3` CRN, `9.247e-4` coupled (**6.7x**), and the
estimator's own tooth is exact: run both arms under the same rule and the
difference is **bit-for-bit `0.0`** over 25510 and 99478 events. Recorded slopes
`-1.0525 +- 0.0624 / -0.9945 +- 0.0870 / -0.8522 +- 0.1378`, all within 1.1 sigma
of `-1` and **2.6-8.9 sigma from `-1/2`** — which is the claim that matters, since
subdominance is what stops the bias flooring the slope above. And the 3a finding
**survives the better instrument**: at `Omega = 50` even `T = 500` coupled reads
`-7.19e-4 +- 1.13e-2`. A 6.7x variance reduction moves that boundary; it does not
remove it.

**The assertion was about to ship green for the wrong reason — the phase's seventh
such test, and the first where the flaw was in *which component* was measured.**
It reported `z` against the prediction and `z` against zero, neither of which any
competitor formula can fail. There are three near-misses — drop the `(-A)^-1`
linear-response solve, transpose `A`, or use `x*/Omega` — and **two of the three
are invisible on species 0, the component every earlier measurement in the phase
had reported** (`-1.7%` and `+1.0%` off there, against `-43%/+57%` and `-35%/+60%`
on species 1 and 2). Excluding the transpose on species 0 alone needs
`SE <= 2.4e-5`, unreachable; on species 1 and 2 it needs `~6e-4`, which is
affordable. The shipped test asserts the **whole vector** and rejects both wrong
formulas from **one** measurement (three test functions would have xdist rebuild
it three times). Its config was seed-verified at 4 sets and its threshold placed in
a measured gap: at `Omega = 100, T = 400, R = 8` the correct vector's worst `|z|`
is `1.418` and the best tooth is `4.780`, so `z = 3` sits 2.12x above one and 1.59x
below the other. **`T = 200` was rejected for being seed-lucky** — it clears the
transpose tooth by 0.8% on one seed-set of four.

**A third instance of "significance is free when the residuals are small", now on
the fit itself.** The recorded slope SEs were **residual-only**: 3 points, 1 dof,
no per-point error propagated. Species 1 read `+-0.0133` where weighted least
squares gives `+-0.0870` (**6.5x**), and species 2's `-0.8335 +- 0.0372` — a
4.5-sigma "subleading correction" — became `-0.8522 +- 0.1378`, **1.1 sigma from
`-1`**. `chi2/dof` of `0.03-0.28` is the tell: residuals far *smaller* than the
points' own error bars. Caught before it was written down, and it would have put a
physical effect that does not exist into two docs.

**The demo caught a fourth wrong figure-claim, in prose this time**: act 4(c)
asserted all three wrong formulas sit within 2% at species 0, and printing the
table showed the no-solve formula is **+44%** off there. Only the two
*structurally* similar errors are invisible. Fixed in the demo, the figure title
and the test docstring.

**The suite timing rule paid off, and the same-session baseline inverted the
reading.** Suite **410 passed in 453.91 s** at `-n 6` against a recorded 240.59 s —
which looks alarming until you take the baseline the rule demands: **389 passed in
544.96 s** with 3c ignored, i.e. the suite is *slower without the new tests*. So
the comparison to 240.59 s is **uninterpretable** — the repressilator floor test,
untouched by 3c, reads `287.06 / 318.02 s` this session against `189.81 s`
recorded in-suite — and **no 3c regression is visible**. Note what this does *not*
establish: the two runs were sequential, not interleaved, so a monotone drift
during the session is confounded with the test-set difference. "3c costs less than
the noise" is the tempting reading and is one inference too far.

**3d is built** — `models/daisyworld.py`, `demos/daisyworld.py`, 45 tests in ~6 s.
The interior equilibrium needs **no root-find**: `beta(T_w) = beta(T_b)` pins
`T_w, T_b = T_opt -+ delta` with `delta` from a cubic in `delta` alone, so
`T_w* = 290.511717483459222`, `T_b* = 300.488282516540778` and `x* = 0.326528079079211`
are all luminosity-independent, and the **regulating band is closed-form too** —
inverting the linear albedo balance reproduces the slice's bisected
`[0.738722418247, 1.359472371265]` exactly. Cardano needs `np.cbrt` (one cube-root
argument is negative) and the roots `+173.12 / -168.13` cancel to `4.988`, so it
agrees with a bisection at `5.7e-15`, not to the last bit.

**Asserting the headline `dT_w/dL = 0` was the hard part, and both obvious routes
are traps.** It cannot be an `analytic_predictions` key — `validate()` matches each
predicted key to an *observable's* final value, and a derivative with respect to a
**parameter** is not an observable of any run. And "the prediction is the same at
every `L`" is a **tautology about the source file**: `T_opt - delta` contains no
`L`. What 3d asserts instead is (a) the local temperature law evaluated at the
**`L`-dependent cover** (`a_w*` runs `0.024 -> 0.668`) returning the same `T_w`,
and (b) **simulated** endpoints from one common bare start agreeing across `L`.
Only (b) can fail for a model whose algebra is `L`-free but whose dynamics drift.

**Richardson in `dt` is blind to (b), measured**: at `L = 1.15` it reads *exactly*
`0.0` while the deviation from `T_w*` is `1.7e-7`, because the residual is leftover
transient. The two-horizon decay law predicts it at
`1.000/1.000/1.000/1.000/1.003` — and `t_max = 100` is chosen because **longer is
worse**: by `150`/`200` the true residual is on the floating-point floor (`~8e-13`)
while the bound keeps shrinking, so the ratio runs to `3.2e7`. Over those same runs
`T_w` moves `2.14e-6 K` while `T_e` moves `2.179068 K` — a factor of `1.02e6`, and
`T_e`'s motion is asserted as a **non-vacuity guard**.

**The `C^0` kink from `beta`'s clip is not academic.** At `L = 1.0` it never bites
and clipped/smooth integrations are **bit-identical**, so RK4 reads
`15.26 / 15.56 / 15.76` — matching the plan's recorded `15.3/15.6/15.8`, so *this*
number travelled, unlike 3a's. At `L = 0.8` the same start drives `beta_raw` to
`-0.025` and the same measurement reads **`58.68 / 0.49 / 2.84`** with non-monotone
errors. Both are asserted; **which luminosity the order claim is made at is part of
the claim.** The `t_max = 200` attractor trap also reproduces exactly: ratio `1.00`
at every `dt`.

**A second closed form was added because the demo's hysteresis figure was wrong —
the project's fifth wrong figure-claim.** Extinction in this ODE is *asymptotic*,
so an unfloored luminosity ramp left the state at `a_b = 5.4e-144`, still growing
exponentially, and printed it as a dead planet; at `t = 5000` instead of `1000` the
same point recovers to `0.662`. Relaxation time wearing the costume of bistability.
The fix is a propagule floor **plus** `invasion_luminosities`: a rare species grows
on a bare planet iff `beta(T_i) > gamma`, and a bare planet's albedo is `A_g`
whatever `L`, so `T_i^4` is linear in `L` and inverts. White invades for
`[0.8332, 1.2079]`, black for `[0.7058, 1.0805]`, against a band of
`[0.7387, 1.3595]` — so **genuine bistability is one-sided**: hot end width
`+0.1516`, cold end `-0.0329`, i.e. empty. Each ramp row is now labelled from the
closed form rather than from the disagreement.

**23/25 mutants red, and all three interesting results are about symmetry or
process.** The headline mutant works exactly as designed: shifting `t_opt` **in the
params**, so closed form and RHS move together, leaves `|rhs(y*)| = 1.8e-16` — a
perfectly good fixed point of the wrong planet — and kills *only* the recorded
literals and two band-boundary tests, while all three `validate()` runs, the
hand-written RHS, the Newton root-find, both invariance tests and the order test
stay **green**. Both survivors are provably invisible: `beta(T_b)` for `beta(T_w)`
(the reduction makes them equal, and a test asserts it), and predicting from the
cover instead of Cardano (a separate test already pins the routes together — the
docstring claiming `validate()` caught this was **wrong and is corrected**).

**And one mutant was a live trap that corrupted a run.** `S (1 - A_g)` mutated to
`S A_g` is a **no-op**, because `A_g = 0.5`. A killed process runs no `finally`, so
it left the mutation on disk; the baseline stayed green *because the mutation does
nothing*; the next run read the mutated file as pristine and scored 25/25 red
against a broken baseline. Fixed by sweeping `albedo_ground in (0.5, 0.4)` — the
"sweep the constant the probe lands on" lesson for the third time, and the second
time in this file after the white/black swap (which leaves `x*` **bit-identical**).
The runner now verifies a green baseline first and restores from **git**, which
required committing the new files first — `git status` cannot protect an untracked
file.

**The figure needed fixing twice, and the second one only surfaced by looking at
the PNG.** The first draft of panel 1 plotted `[T_w*] * len(L)` — one value
repeated — under the title *"The daisy temperatures are flat"*, i.e. flatness **by
construction**. That is the Phase-2 Turing panel verbatim, and it would have been
the fifth wrong figure-claim shipped in the commit documenting the fourth. It now
plots the per-`L` values computed at each `L`'s own cover; the picture is identical
(`~1e-13` on a 290 K axis) but the line now demonstrates the claim instead of
restating it. Then panel 3 implied **every** gap between the two ramps was
bistability, including the `L = 1.20` one the act-5 text spends a paragraph
refusing — a figure gets looked at without its text — so it now shades the
closed-form window `(1.208, 1.359)`, where the down-ramp visibly jumps onto the
no-daisies curve. And its title was clipped on the first render.

**Suite: 455 passed in 180.82 s at `-n 6`, and the same-session baseline is what
makes that readable.** With 3d ignored: **410 passed in 217.12 s**. So the suite is
*faster with the new tests* — but the discriminating quantity is the repressilator
floor test, untouched by 3d, and it reads **137.26 s** in one run and **158.58 s**
in the other, a **15.5%** swing. The totals are therefore not attributable to the
test set; **no 3d regression is visible** and nothing stronger is claimable.
Cross-session is worse: that same test read **287.06 s** last session against
`137.26 s` now, a **2.1x** machine swing, so 3c's recorded 453.91 s is
uninterpretable here. As in 3c, the two runs were sequential rather than
interleaved, so monotone drift is confounded with the test-set difference.

Next: 3e (adaptive dynamics), the last stop. Its `O(sm)` canonical-equation claim
is **consistent with the slice data, not measured**, and must be re-measured at
1200-reps-equivalent across >= 3 `sm` *before* a test is written — treat it exactly
like 3c's bias. `t_branch`'s detection resolution needs fixing first too: the slice
quantized it to 100 time units (+-1.6%) while reporting the product constant to
1.2%. See `docs/plans/phase3-tasks.md`.
