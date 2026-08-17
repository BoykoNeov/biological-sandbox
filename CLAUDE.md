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
uv run python web/serve.py --download          # the browser front-end, on :8765
```

The front-end rebuilds the wheel on every start and the page shows its sha256:
a stale wheel runs old code and passes every check in silence. Pyodide is
**vendored** (`--pyodide-src <dir>` copies a local distribution, `--download`
fetches one), never a CDN, so the bytes are the ones the measurements used and
development works offline. `--bandwidth <Mbit>` paces the server for a cold-load
measurement.

## The non-negotiables

1. **Protocol first.** Every model implements `core.protocol.Model`. Shared
   services (Recorder, sweep, ValidationSuite, viz) consume only the protocol
   surface — they never reach inside a model's concrete state.
2. **A core model is done only when its `analytic_predictions` are reproduced
   by the ValidationSuite within tolerance.** Validation *is* the definition of
   done, and the tolerance is statistical (`z * standard error`), never a
   hardcoded epsilon. Prefer this over inventing new pass/fail logic.
   **Three tracks exist, and a new one needs a measured reason to exist** — not a
   preference. `core/validation.py` is a replicate mean against a predicted scalar;
   `core/convergence.py` is a power law's exponent (the repressilator's limit is a
   *cycle*, so no scalar exists for a mean to match); `core/selection.py` is a
   discrimination between named hypotheses (Schnakenberg's emergent wavenumber is
   *quantized*, so its mean matches no scalar to `z*SE` and the equality check fails
   as replicates grow — measured, `docs/plans/phase2c-schnakenberg-measurement.md`
   §7a). All three take the model-specific arithmetic from their **caller** and read
   only named scalar observables, so non-negotiable #1 holds for them in full.
   When `analytic_predictions` cannot answer honestly it **raises** — no real state,
   a complex eigenvalue pair, a rate too near zero, or a quantized quantity with no
   exact target.
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

`src/sandbox/core/` protocol + services (three validation tracks:
`validation.py`, `convergence.py`, `selection.py`) · `models/` the models ·
`viz/backends/` pluggable plotting · `web/` the browser front-end's Python side ·
`demos/` runnable examples · `tests/` wraps the ValidationSuite ·
`docs/plans/` per-phase dev docs · top-level `web/` the JS, HTML and the server.

**A model never imports a service.** Models import `core.{laplacian,ode,rng,registry}`
and `models.gillespie` — numerics and registration — and nothing from
`core.{sweep,recorder,validation,convergence,selection}`. The tests and demos wire a
model's own arithmetic into a track, which is how `repressilator` meets
`convergence_report` and `schnakenberg` meets `selection_report`.

**Registering a model owes the front-end a preset.** `web/presets.json` needs one
entry per built-in model and `test_the_presets_cover_every_built_in_model` fails
otherwise — deliberately, so adding a model is a decision about the picker rather than
a silent omission from it. Its `max_steps` must come from
`web/serve.py --measure-presets`, never from another model's cost.

**`src/sandbox/web/bridge.py` is a shared service and non-negotiable #1 applies
to it in full** — protocol surface only, never inside a concrete state, exactly
as the Recorder does. **No numerics in JavaScript**: the JS side owns the DOM,
the canvas and the message loop and nothing else, because a JS or shader
reimplementation of a validated model *is* the cost this branch was chosen to
avoid, in a language no test here can reach.

## Workflow

- Add a model: write its `*Params`/`*State`, implement the protocol, register it
  in `models/__init__.py`, then write the validation test **first** and confirm
  it can fail before the implementation is correct.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`). Each
  commit should pass `pytest` and `ruff`.
- **Batch / session end:** update memory + `docs/`, then commit and push.

## Status

**Phase 2c is built: Schnakenberg, and the wavelength claim Phase 2 deferred.** It is
**not** a new phase — `HANDOFF.md` §8 says there is no Phase 5 and that inventing one
is not the next action — it is Phase 2's one explicitly deferred item, taken because
Phase 2's stated reason for deferring it was half wrong. `lambda(q)` is the growth rate
of a mode seeded **by hand**; which wavelength appears when you seed **nothing** is a
different claim, and it is the one HANDOFF §5 actually asked for. Gray-Scott cannot
carry it at all: at Pearson's parameters there is no Turing state to select about.
`models/schnakenberg.py`, `core/selection.py`, `demos/schnakenberg.py`,
`tests/test_schnakenberg.py` (74 tests), and a `web/presets.json` entry. Plan and
record: `docs/plans/phase2c-{plan,schnakenberg-measurement}.md`. Both deferral notes in
`docs/plans/phase2-{plan,tasks}.md` are amended in place, so nobody finds "deferred"
beside a built model.

**Everything about the onset is closed form, and `det J = u*^2` *exactly* is why**:
`u* = a+b`, `v* = b/(a+b)^2`, so `q_c^2 = u*/sqrt(Du Dv)` and
`d_c = [u*(1+sqrt(1+f_u))/f_u]^2`. Each is asserted against an independent route — the
Jacobian against central differences, `d_c` against a bisection (`2e-11`), `q_c`
against an argmax — never against a recorded literal. The seeded-mode anchor reproduces
the **stencil's** rate to 8-10 digits **across a sign change**, while the continuum
`-Dq^2` is 39% off at a growing mode and **184%** at Nyquist; the residual falls as
`eps^2`, which identifies it as the nonlinear correction and not the operator.

**The design finding, and it is the phase's transferable one: a quantized measurement
has no exact scalar target, so the claim is a *discrimination*.** A periodic box admits
only integer modes, so the emergent wavenumber's ensemble mean matches no continuous
prediction exactly — measured, the deviation **grows** with replicate count (0.61 →
2.24 SE at R = 8 → 48) because the mean stands still while the error bar shrinks under
it. So `analytic_predictions` **refuses** the random initial condition (the project's
first refusal on *statistical* rather than algebraic grounds) and `core/selection.py`
asserts that the prediction beats each **named alternative** by a margin exceeding
`z*SE`. Margins *grow* with replicates (continuum 9.1 → 16.0 SE), so more data
strengthens a discrimination where it eventually breaks an equality — that asymmetry is
the whole point of the design and it was measured, not assumed. Two non-vacuity guards
ship with it: every replicate must land inside the unstable band, and the selected mode
must not track the initial noise (initially-loaded modes scatter across the whole
spectrum; every run ends within a mode of the same answer).

**The claim only has content where the hypotheses differ, and that is a *coarse*
grid.** At 4.3 cells per wavelength the stencil and the continuum disagree by 2.27
modes about which mode is fastest, and across the coarse grids where the two answers are
2-3 modes apart the emergent peak matched the continuum's **0 times in 24 runs** (once in
32 including the grid where the answers are adjacent)
— Phase 2b's lesson arriving in the **nonlinear** regime. At 10.6 cells they agree, the
prediction fits *better* (0.32 SE), and the discrimination collapses. Ship config
`n = 112, R = 32, z = 4`, verified at four seed-sets with a weakest margin of 6.68 SE;
**`R = 16` was rejected for seed-luck** — it clears the band-centre competitor by
0.11 SE on one seed-set of four.

**A grid can be too coarse to have the claim at all**, and the first design walked into
it: coarsen and the stencil's fastest mode becomes **Nyquist** (a checkerboard the
pattern does not match), coarsen once more and the instability **vanishes**, because
the largest representable `q_eff^2 = 4/h^2` drops below the unstable band. Also
measured: the box length is the lever that makes selection non-vacuous at all (1, 3, 6,
13, 25 unstable modes for `L = 1..16` at fixed spacing) — with one unstable mode, "the
fastest mode won" says only that the only mode which could grow, grew.

**Half of the measurement document did not survive the shipped step size, and the
estimator that failed was its own.** The slice integrated at `0.4 x CFL`, which the
model refuses (`dt` must divide `t_max`). Re-taken at the shipped step, the recorded
"deviation grows with R" trap at `n = 160` **does not reproduce** (0.17 SE at R = 48),
and the "best target flips between grids" conclusion was the same artefact. The
mechanism was real and worth finding; the numbers belonged to the instrument. Recorded
as a correction in place (§7a), not by editing the originals away.

**Three figure defects found by looking, and a fourth caught in a docstring.** A field
plot that was a jagged sawtooth captioned as a clean pattern; a legend entry for a
curve invisible on the axes (at small `j` the stencil hides under the continuum); a
rejected prediction drawn flush against the spine, where a refutation reads as
decoration. The 2-D panel was re-gridded from `n = 96` to `n = 128` for the same reason
— at 3.6 cells per wavelength the picture shows the grid, not the wavelength — and at
`n = 128` both seeds hit the predicted mode exactly where `n = 96` missed by one. And a
docstring quoted a power fraction of `0.997-0.999` that holds only at 113 e-folds: at
the shipped 22.7 it is `0.53-0.90`, and on the coarse grid it **freezes** because the
harmonics have nowhere to go. The selected *mode*, by contrast, is identical at 22.7,
45.3 and 113.3 e-folds on both grids. **How sinusoidal the pattern is and whether the
mode has settled are different questions.**

**Mutation: 20/22 red, and both real gaps were "a check nothing can fail".** The
spectral estimator subtracted the mean *and* zeroed mode 0, so on a symmetric field the
subtraction cancelled exactly and deleting the zeroing changed nothing — two mechanisms
where only one is testable. And a margin mutated from `(gap_comp - gap_pred)/SE` to
`gap_comp/SE` survived everything, because on the contrast grid both formulas land
under `z = 4`. The remaining two survivors are provable no-ops. **The amplitude
exponent is deliberately not asserted**: the log-log slope reads `0.4606` and a
Richardson extrapolation of `amp/sqrt(eps)` is still drifting, so "supercritical,
consistent with `1/2` with `O(eps)` corrections" is the honest claim and the slope
number is labelled as not being the exponent.

**Phase 4 — the browser front-end — is COMPLETE (4a, 4b, 4c), and its three
recorded deferrals are now built too.** The record with every number is
`docs/plans/phase4-tasks.md` (§9 for the deferrals); the plan is
`docs/plans/phase4-plan.md`. `src/sandbox/web/{bridge,colormap}.py` plus `web/`
(worker, client, two renderers, five pages, a staging-and-serving script) and
`tests/test_web_{bridge,assets}.py` (56 + 68 tests). Serve it with
`uv run python web/serve.py --pyodide-src <dir>` (or `--download`); it rebuilds
the wheel on every start and the page displays its sha256, because **a stale
wheel runs old code and passes every check silently**. Add `--with-figures` for
the on-demand figure export, and `--measure-presets` prices every model-picker
preset to termination.

**The three deferrals, and the reason to read §9: every one of them was broken in
a way only doing it could show, and two were in code that had been read and
called correct.** The `stop` button did nothing at all for the window between
`create` and `run` — the pause a page spends integrating the deterministic limit
— because a cancel arriving then had no loop to set a flag on; the worker replied
`cancelling: false` and the page threw that away, disabled the button, and let the
run finish with nothing able to stop it. A cancel is a fact about the **run**, not
about whichever loop is live. Clicked for real it now stops at `116 000` of a
`338 400`-event budget and says *you* stopped it, which is a different fact from
running out of budget and produces an identical picture.

**The model picker (`web/models.html`) is driven by `describe` and
`default_params`, and driving all fourteen models found three defects three pages
had not.** `web_field` broke on a JS `null`, because **Pyodide 0.28 maps `null` to
a `JsNull` singleton and not to `None`** — invisible because no page autoscaled.
`glv_stochastic` can never have its ODE limit drawn (five observables against a
three-component ODE: two are derived), and the bridge's one-to-one refusal is
right. And **a `FieldModel`'s field need not be 2-D** — `trait_branching`'s is the
`(161,)` trait grid — so `create` now reports `field_shapes`. Its budgets are
**measured** (`--measure-presets`), never carried over from the demo's
repressilator-specific `470 events per unit`, and it reports whether the
replicates are actually different by **comparing what was drawn**, so three
bit-identical curves are named as one curve drawn three times rather than shown as
a spread.

**The figure export reuses `plot_replicates` rather than a second plotting path,
and matches limit columns by observable NAME** (the wrong column is the right
shape at the wrong phase). **An exported PNG leaves the page without the page's
caption, so it carries its own**: a figure of a run that stopped early shades the
region past the last time every replicate reached, and says in its own ink that
the gap is absence rather than agreement — otherwise the export re-opens the
exact wrong figure this project already shipped.

**The deferral's constraint was verified, and the first wording of the result was
wrong.** A worker that only boots pulls `9.006 MB` — the recorded figure — and
"zero matplotlib bytes" went into four documents. The real page also calls
`figure_available` at load, which that instrument never did; measured before and
after, it is `9.006 → 9.032 MB`. The `+26 KB` is a re-read of `pyodide-lock.json`
(the server sends `no-store`) plus a `HEAD` on the wheel costing **300 bytes of
headers with `decodedBodySize = 0`**. Correct claim: **no matplotlib *body*,
300 bytes of headers.** Same class of error as measuring in a hidden tab — an
instrument that did not do what the page does. Costs are three numbers, not one:
`0.52 s` local fetch-and-install (once per worker), `1.24 s` import, `0.58 s`
draw; the browser's import is ~3x the native repeat because Pyodide has no
persistent font cache.

**`tests/test_web_assets.py` (68 tests, 2.2 s, no browser) exists because every
bug above was found by opening one.** It compiles the Python that `worker.js`
embeds in a JS template literal — and earned itself immediately: writing the
comment about the `JsNull` fix, backticks around the word `null` **ended the
template literal and broke the whole worker**, hanging the page on "starting
Python …" with an empty console, because the failure killed the module script
before it could log. All six mutations bite the intended test. One test was wrong
and the model right (a 100-recorded-point floor failed `adaptive_dynamics` at 57,
whose jump process has 57 events in the entire run — a floor on point count is a
claim about the model; a preset can only be held to its *thinning*), and one was
wrong the other way (comparing presets to `available()` passes alone and fails in
the suite, because other modules register teeth models into the same registry).

**The browser's verdict matches native**, computed same-commit and same-machine
rather than pasted from a previous session: Wright-Fisher `[PASS] fixed_A:
measured 0.3375 vs predicted 0.3` identically, and the repressilator **refuses**
for the same reason native `validate()` raises. The refusal *is* the verdict
there — its limit is a cycle, so no scalar exists for an ensemble mean to match,
and a pass mark quoting a tolerance would be theatre. The agreement bar is
**statistical**; that both came out bit-identical is reported as a convenience
under a label saying so.

**Two changes outside `web/` that the front-end forced, and both were overdue.**
The registry now carries a **params type** per model: an `Experiment` names its
model by a string so it can be serialized, yet every service still took a
`params_factory` from its caller, so an experiment was serializable except for an
out-of-band Python callable. And `run_replicate` became a thin wrapper over an
incremental `ReplicateRunner`, so a front-end that must interrupt a run drives
**one** stepping loop rather than a second implementation of subtle semantics —
four trajectories sha256-fingerprinted across the change, bit-identical, at
`0.2238` against `0.2228 s`.

**The phase's transferable finding: a hidden tab is a different machine, not a
noisier one.** The automated browser tab is always hidden (`document.hasFocus()`
can be `true` while `visibilityState` is `"hidden"`), and Chrome deprioritizes a
background renderer, handing its worker back to a throttled scheduler at every
yield. Measured hidden, a chunk-size sweep rose to **3.59x** across a 240x range
of chunk sizes and the worker looked **2.03-2.10x** slower than the main thread.
Measured in a real visible window: **1.00-1.02x** and **0.89-0.99x**. The hidden
curve was monotone in the number of yields, with small spreads, and reproduced —
*exactly what a real per-yield cost looks like*. Three things caught it: a third
arm running the same work as **one unchunked call** (which isolates the yield
from the thread), a two-column decomposition separating pausing from reporting
(they agreed, clearing the reporting), and a visible window that **posts its
results back** to the serving process, since a visible window cannot be driven by
the automation that would otherwise read them. Every page states its own tab
visibility and refuses to quote frame figures when hidden.

**What visible measurement then establishes** — the plan's one open item. The
page holds **144 fps with a worst frame of 7-14 ms** while a worker streams a
400 000-event SSA, and at Gray-Scott 256² where a single frame costs 78.6 ms of
compute. The main-thread arm gets **zero event-loop turns and paints no frames at
all** for ~2 s in a visible tab. The drawing path is **5.4-6.2% of a frame**, not
the slice's 1.6-2.2%: this colormap costs ~3x more and the step here is faster.
Recorded rather than optimized — the step is 94% and non-negotiable #4 says
believe the profile. **The fraction is the measurement**, and the evidence is
that it reads `6.2/5.4/6.0%` and `6.2/5.8/6.0%` across two runs whose step times
differ by 1.6x.

**The cold first load, which the plan named as the one thing that could still
overturn the fork:** **9.01 MB over the wire** (15.78 MB raw), to a running
interpreter in **2.53 s** local, **4.29 s** at 25 Mbit/s and **12.22 s** at
5 Mbit/s; first simulated points about a second later in each case. The decision
stands. **This is throughput only** — a paced local server models no latency, no
slow start, no loss, no CDN and no mobile radio, so a real 5 Mbit/s link is worse
than 12.2 s by an amount this does not establish.

**One trap found while building the demo, and it is the phase's sharpest.**
`discrepancy_to_limit` samples the SSA by step-hold, which is exact only while
the recording is *denser* than the comparison grid. Coarser, and every grid point
reads a stale value — an error that is **`Omega`-independent**, so it does not
cancel between system sizes: it floors `D` and flattens the very scaling the
measurement exists to show. At `Omega = 5` against a 200-point grid, `D` reads
`8.72` with 15 578 recorded points and `8.83 / 10.28 / 13.00 / 21.15` with
`79 / 17 / 9 / 5`. It now **refuses**, like `check_grid_is_exact`, which exists in
the convergence module for this exact failure.

**Three wrong figure-claims, all caught by looking.** A trace whose replicates ran
out of `max_steps` at `t ≈ 8.5` while the limit was drawn to `t = 30`, so the
right of the frame showed the limit alone and read as perfect agreement — both
pages now size the budget from a measured event cost (~470 events per time unit
per unit of `Omega`) **and report whether every replicate reached the horizon**.
A two-panel split justified as *"the proteins reach ~160 and the mRNAs ~10"* when
they reach `135.4` and `153.0` — the split is right for legibility, the stated
reason was invented, and the panels' near-identical shape is a *result*
(`beta = 1` equalizes the timescales). And a field benchmark whose picture is the
seeded square patch barely moved, now captioned as a stopwatch rather than a
demonstration. **A fourth was corrected in the opposite direction**: the demo's
`D` caption claimed the errors overlap and runs would come out misordered, which
the seed check had just disproved — being falsely modest about a measurement is
the same failure as overclaiming it.

**Suite (deferrals built): 846 passed, 11 skipped in 157.34 s at `-n 6`**, against
a same-session baseline of **776 passed in 407.13 s** with the new module ignored
— i.e. *faster with 70 more tests*, so by this project's own rule the totals are
not attributable to the test set and no regression is visible. Four runs in that
one session read **`407 / 342 / 255 / 157 s`** — a monotone `2.6x` drift, far
larger than anything being measured, and the sharpest illustration this project
has of why a suite total needs a same-session baseline. What *is* attributable is
the new module's standalone cost: **2.7 s**.

**Suite (Phase 4 close-out): 776 passed in 346.23 s at `-n 6`**, against a
same-session baseline of **720 passed in 286.12 s** with the bridge tests ignored. The decomposition is
the cleanest this project has managed: the total grew by `+60.11 s`, and the two
repressilator tests — untouched by Phase 4 — grew by `+62.15 s`, **more than the
entire gap**, while everything else fell slightly (`70.14 → 68.10 s`). No cost
from the 56 new tests is visible. **The worker tags say why, and this time it was
observed rather than inferred**: all three long repressilator tests land on
**`gw4`, back to back** — about `313 s` of a `334 s` total, so one worker is
**94% of the wall clock** and the other five finish and idle. The suite total is
not a measure of the suite; it is a measure of `gw4`, and adding tests cannot
move it unless they land there. (Watch the flags: `-q -v` silently drops the
tags, because `-q` wins.) **Both runs were taken with the machine at
100% CPU and 30 Python processes from unrelated projects on it**, so the
absolutes compare to nothing recorded earlier; what survives is the
decomposition, because both arms were contended together.

The remainder of this section is the two measurement slices that preceded the
build. **Their worker and chunking numbers were all taken in a hidden tab** and
are superseded above; the instrument lessons stand.

The phase closes the `HANDOFF.md` §4 browser-vs-local fork,
which that document told the reader to *"decide early"* and which went undecided
through four phases; it was settled by measurement, in two slices
(`docs/plans/phase4-{browser-fork,worker-and-rendering}-measurement.md`), and both
of HANDOFF's load-bearing claims about it were corrected in the same commit —
§4's "costs a reimplementation of the numerics" (it does not; the wheel runs
unmodified in Pyodide at ~2x native) and §8's "implement Phase 0".

**The real cost is the plumbing, and the plumbing is now a measurement rather than
a design consequence.** Running an SSA on the main thread blocks the page for the
*entire* run — 1 750 ms at 60 000 events, with the event loop getting **zero**
turns — while the same run in a Web Worker blocks it for **10-23 ms** at identical
Python speed (`0.5823` vs `0.5826 s`). Crossing JS→Python is free (4 096 calls cost
less than one 115 ms workload's noise). The slice put the whole drawing path at
1.6-2.2% of a frame and the step at 98%; the shipped code measures **5.4-6.2%**
and 94%, for the reason given above. matplotlib renders correctly but stays out of the default
bundle: it multiplies the gzipped download **2.1x** (8.7 → 18.4 MB) to buy
0.07-0.16 s static PNGs.

**Three measuring instruments were wrong before any of them was right, and two
produced plausible numbers rather than errors** — the phase's most transferable
result, and the "green for the wrong reason" pattern arriving in a new domain.
`requestAnimationFrame` is *suspended* in a background tab, so the freeze detector
reported `worst freeze 1840 ms` for the **worker** arm, identical to the
main-thread arm, because neither was being measured; replaced by a MessageChannel
ping-pong, which is a task rather than a timer and so survives the throttling.
`performance.now()` is coarsened to **0.1 ms** outside a cross-origin-isolated
page, so every single-shot sub-millisecond reading was a rounding artifact. And a
`getBuffer()`-vs-`toJs()` comparison measured equal because **`toJs()` on a float64
ndarray returns a `Float64Array`** — one operation spelled twice. And the
figure lesson recurred, in a new medium: the first plot rendered in a browser was a
valid PNG with correct axes and **no data**, because its filter tested
`startswith("m")` against observables named `x_m1 … x_p3`. (No ordinal — 3e retired
the running count as broken; the lesson is what travels.)

**Two things the slice does not establish, stated rather than buried.** Every probe
ran with the tab hidden, so the drawing path's *CPU* cost is measured while
**compositing and vsync are not** — "the animation is smooth" is a different claim,
and the slice carries a probe that waits for a visible tab instead of faking it.
And the machine was contended by unrelated processes at ~25-30%, with **4x swings
inside one session**, so only ratios and fractions are quoted as measurements.

**Phases 0-3 are all complete.** The validated core is 14 models plus
`core/random_matrix.py`; the `models/ecosystem/` quarantine is still **empty**,
and that remains the correct outcome. The suite was **720 passed in 373.81 s** at
`-n 6` before Phase 4 added 56 (the Phase-3 close-out's `718` predates `fadfbe8`,
which added two; see that close-out below for why a wall-clock number needs a
same-session baseline *and* the xdist worker assignment before it means
anything).

**Two things that re-run caught, both about a green run that is not what it looks
like.** The environment had drifted: `matplotlib` was gone, so `uv run pytest`
reported `710 passed, 1 skipped` — an entire absent test module showing up as
**one quiet line**, ten tests missing with nothing failing. Use
`uv sync --extra viz`, and read the skip count, not just the failures. And the
timing decomposes the way the close-out predicts rather than the way it looks:
`390.09 s` for the 710-test run against `373.81 s` for the 720-test one, i.e.
*more* tests in *less* time, so the gap to the recorded `225.80-303.91 s` is
machine drift and **no regression is visible**.

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

**Phase 3 is complete** (3a-3e; the plan as written follows, and each sub-phase's
findings are recorded below it) — `docs/plans/phase3-{plan,context,tasks}.md`.
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

**3e is built, and Phase 3 is complete.** `models/adaptive_dynamics.py` (the
trait-substitution sequence and its canonical-equation limit) and
`models/trait_branching.py` (the trait-grid gLV), plus `demos/adaptive_dynamics.py`
and 263 tests. Everything the measurement phase recorded reproduces **exactly** —
the four config-B slopes and all four branch times bit-identically — which is what
made every disagreement below a real finding rather than a porting artifact. The
measurement notes that follow are kept as written; the build findings come after.

**The convention had to be recovered, because no slice code survives.** Holding
`mu sm^2 t_max` fixed pins `U = (1/2) mu sm^2 K0 t_max`, since the canonical
equation collapses to a parameter-free `dx/du = -x exp(-x^2/2)`. `U = 1/2`
exactly — `x(0.5) = 1.849492240597` reproduces the recorded `1.849492` — and `K0`
and `mu` are **not separately identifiable and need not be**. Three teeth targets
the reconstruction was *not* fitted to confirm it to `3e-7`. But `sa` and the
mutation-step law are **not recoverable and are load-bearing**, so the recorded
`+0.004` offset and teeth `z` values **may not be cited**: the measured offset is
`+2.39e-3`. A fifth "a number travels with its estimator", and the first where
the estimator was only *partly* recoverable — separate what a record pins from
what it merely used.

**The `O(sm)` law is now measured:** slope **`+0.9977 +- 0.0225`** over a 32x
range, `0.10 sigma` from 1 and **44.6 sigma from 2**, `chi2/dof = 1.07`, all six
discrepancies positive, `err/sm` flat at `0.17723`. Replicate counts were
*derived* (`R ~ 1945/sm`), not taken from the plan's "1200 reps". **A first run
was wrong and the group-scatter self-check caught it** — `spawn_rngs` called with
one seed inside the `sm` loop ran every point on common random numbers, giving
under-dispersed groups (`0.16`, `0.34`) and `chi2/dof = 2.62`; per-`sm` seed
branches send both to ~1. The teeth reproduce **3b's trap exactly**: a wrong
canonical equation makes the discrepancy an `O(1)` constant, and a constant fits
a near-perfect line, so all three sit *tens of sigma from zero* with SEs 30-200x
smaller than the correct point's. **"Significantly nonzero" passes all three.**
Ship config `sm = 0.15...0.0125` at 2.3 s (`0.2` dropped — 4.57% of its mutants
saturate at `s > 0.5` and it is the highest-leverage point; harmless, but the
replacement is free), band `[0.6, 1.4]`: correct **4/4**, every tooth **0/4**.

**The branching model has no mutation term, and its own failure modes force it.**
A draft with diffusion let the *outer* bins win (every mutant invades a resident
at the singular point, the far ones most) and **overflowed** at `sa = 1.5`, where
`lambda dt = -42` at the domain edge. The slice ran `dt = 0.5` there without
blowing up, so its outer bins were *exactly* zero — true **iff** pure gLV, where
`n_i = 0` is a per-bin invariant. 158 untouched bins are now asserted
**bit-identically `0.0`**. So the morphs at `+-1` spacing are not merely a grid
artifact: those are the only bins that ever carried mass, and **this is a
3-species gLV in which `ngrid` enters only through `h`**.

**The resolution fix inverts the assumption.** Checkpoint-and-refine gives
`+-dt/2` instead of `+-50`. Coarse checking made the products look **more**
constant than they are — `0.381%` at quantum 100 against `0.622%` true — because
detection rounds *up* and that inflation is largest where `t_branch` is smallest,
which is where the true product is lowest. **A tolerance read off the coarse
measurement would have been too tight.** What survives is `O(h^2)`:
`2.7481/0.6216/0.1517%` at `h = 0.1/0.05/0.025` (ratios `4.421`, `4.098`), and
Richardson to `h -> 0` leaves the product constant to **0.006%** over a 16.5x rate
range — Richardson in a *discretization* parameter, the Phase-2 stencil
instrument, **not** the amplitude one. Exponent `-1.00196 +- 0.00070`, stable
across six `(thr, seed)` configs, and threshold-invariance is a formula —
`(2/h^2) log(1/(thr*seed)) + const`, predicting all six to `<= 0.21%` — **whose
`const` is fitted**, since the centre-bin fall is not derived and leaves `1.39`
unexplained in the limit. The no-branch side holds at `t_max = 200 000` for
`sa = 1.05` and `1.5` (the earlier `20 000` check was 7.5x short of the nearest
branch at `149 822`).

**Two implementations of one jump process, pinned on the generator state.** The
`O(sm)` sweep needs 40 000 replicates at the smallest step, so it runs through a
vectorized `run_cohort` (whole living cohort, one event per iteration, 2.3 s)
while the protocol model's `step` advances one replicate. They are pinned by
comparing `rng.bit_generator.state` after both runs, **not** the final trait — a
final-value match can hide a compensating reordering of draws. The hazard is the
horizon: the cohort advances the clock *before* testing it, so a replicate carried
past `t_max` consumes its waiting time and nothing else, and testing terminality
first — the natural way to write `step` — desynchronizes the stream on the last
event of *every* replicate. Shipped as a mutant, red.

**Both headline bands are two-sided, and both had to be.** The canonical teeth
were known (a wrong equation makes the discrepancy an `O(1)` constant, so it fits
a near-perfect line at 30-200x smaller SE). The branching side turned out the
same way, measured: a rate formula missing its `-1/sK^2` reads **`-2.454`** and a
competition kernel missing the factor of two in its exponent reads **`-0.553`**,
and "the exponent is significantly negative" passes both. Band `[-1.15, -0.85]`
against a correct `-1.00072 +- 0.00050`. Two further teeth — `K` built with
`sigma_a`, and one-neighbour seeding — **do not branch at all**, and the kernel
tooth bites the absence leg at `sa = 1.05` (spurious branch at `19 759`) but not
at `1.5`, which is why both absence points are kept.

**The cheap lever was the horizon, not the grid — and the obvious choice was
backwards.** Dropping `sa = 0.95` (whose branch time exceeds the other four
combined) gives a fit that is 2.2x cheaper *and* closer to `-1`. Coarsening the
trait grid would have been the natural alternative and is wrong: `h = 0.1` is the
**least** converged point. The absence leg keeps `t = 200 000` — the weak half of
a sign change is not where you buy time.

**Two bounds became predictions once they were looked at.** The neighbour bins'
decay is not merely "small": against a resident pinned at `K0` their per-capita
rate *is* the neighbour invasion fitness, so `seed * exp(s_0 t)` predicts the
final value to **0.2%** over decay factors of `1e-10` and `1e-60`. And the
neighbour's departure from its own small-`h` limit is exactly `-rate h^2 / 4`,
matching to three digits — a `rel=1e-3` tolerance was tried first and **failed by
11%**, because the departure is real.

**`min eig > 0` on the Gaussian kernel failed, and the failure is the point.** It
reads `-4.9e-16` at `n_grid = 41` against a top eigenvalue of `8.5`: positive
definite in exact arithmetic, at the roundoff floor on any useful grid. Asserted
as *not meaningfully negative* against a size-scaled floor, plus strict positivity
on a coarse grid. **The near-singularity is not a nuisance to tolerate — it is the
structural degeneracy the Gyllenberg-Meszena argument is about, in the
arithmetic.**

**32/33 mutants red; the first run scored 25/33 and every survivor was
instructive.** Two shared one cause: `r_growth` and `sigma_k` both default to
exactly `1.0`, where `-r x / sK^2` is indistinguishable from `-x / sK`, `-x / sK^2`
and `-r x / sK` — **the fourth time "sweep the constant the probe lands on" has
caught something in this project**. A third was a threshold nothing could fail: a
mutant returning the *unrefined* coarse checkpoint passed the refinement test,
since `t_refined == t_coarse` satisfies both "within one quantum" and "fewer than
`check_interval` steps". The lone survivor is a **provable no-op** — `rng.random()`
is in `[0, 1)`, so `< p` is False for every `p <= 0` — and the generator-state pin
*demonstrates* it by holding bit-identical between a mutated cohort and an
unmutated `step`. Four patterns matched zero times because `ruff format` reflowed
them; the runner reports that as a **runner bug rather than a survivor**, which is
the difference between 32/33 and a silent blind spot.

**A finite-difference check can be green against the wrong derivative.** The
mixed-derivative stencil, written the natural way, walked the *diagonal* and
returned exactly `-1.000000` at every probe — a correct number for `dD/dx`, not
for `d2s/dxdy` (`+1.959` at `x = 2`). It disagreed with the closed form only
because the closed form was right.

**And the deterministic helper was again the expensive part.** `CANONICAL_DU`
is `1e-3`, chosen by measuring where the answer stops moving: identical to twelve
digits, and **1.7 s against 27.7 s** at `1e-4`, versus 2.3 s for the entire
stochastic sweep it checks. It is the module default rather than a caller's
opt-in, because in the measurement phase it was a monkey-patch and a helper that
must be *remembered* to be cheap will drift back.

**The demo caught another wrong figure-claim, and this one stated nothing false.**
Panel (c) said "the two neighbours rise" while drawing one line exactly on top of
the other — the morphs are bit-identically symmetric, so a reader counts two
curves under a title claiming three. Naming the coincidence in the legend turns
the overlap into a result. Panel (b)'s axis said "|mean - canonical prediction|"
when three of its four curves are measured against a *different* equation's
prediction, which is the whole point of the panel.

**Stop citing the running count of wrong figure-claims; it is broken.** Phase 2
records "four claims were wrong until someone looked", then 3c calls its one "a
fourth" (it was the fifth), then 3d calls two different things "the fifth". The
*lesson* is what travels — **a figure carries a claim and gets looked at, not
exit-code checked** — and it has now recurred in every phase that shipped one.
The tally has not been reconstructed, because doing so would mean rewriting three
sub-phases' prose to fix an ordinal that carries no information.

**Suite: 718 passed — and the timing needed three instruments, of which the first
two still gave a wrong answer.** Against a same-session baseline of `455 passed in
164.29 s`, the full suite read `300.06 s`, which looks like an `+82%` regression on
a test set that costs `19.5-21.2 s` standalone. So the **worker assignment** was
read directly with `-v`: both repressilator tests had landed on **`gw0`**, run back
to back, and that is the critical path. The baseline cannot have packed them
together — its total is *below their sum* (`260.71 s`). That much is solid, and it
is the first time in this phase a timing mechanism was **observed rather than
inferred**.

**Then two more runs of the identical command broke the tidy story: `300.06 /
303.91 / 225.80 s`.** The two long tests themselves ran `247 s` in one and `175 s`
in another — the machine's recorded `+-30%`, arriving *within* a single session.
So there are **two effects, not one**, and at fixed machine speed the `+136 s`
decomposes roughly as packing `~86 s`, 3e's own tests `~20-30 s`, drift `~75 s`.
`-n 8` was measured as the documented fix and **rejected**: `266.82 s` against a
same-session `-n 6` of `225.80 s`, sequential so drift is confounded — "no
improvement demonstrated", not "worse". `-n 6` stays, the posture the Phase-1
close-out took when its `-n` fix measured worse.

**The rule, now on its third repetition and finally specific.** A total needs a
same-session baseline; the baseline needs the **worker tags**, because adding
*any* tests reshuffles xdist's collection-order packing; and the worker tags need
the **per-test durations**, because this suite's critical path is two tests whose
own timing varies by 1.5x inside one session. Expect to *decompose*, not to
attribute — and note that the first attribution written from the `-v` run was
clean, confident, and only two-thirds right.
