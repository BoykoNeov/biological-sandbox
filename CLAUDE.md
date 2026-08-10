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

Next: implement 3c (`models/glv_stochastic.py`), and close or explicitly label the
`O(1/Omega)` bias. See `docs/plans/phase3-tasks.md`.
