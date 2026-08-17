# Phase 4 — the browser front-end (tasks and measurements)

The per-phase record, in the shape every earlier phase used: what was built, what
was measured, and what the measurements refuse to support. `phase4-plan.md` is
the plan this executes; the two slice documents it rests on are
`phase4-browser-fork-measurement.md` and `phase4-worker-and-rendering-measurement.md`.

## 0. Read this before any number below

**Every absolute figure here needs its conditions.** Two of them, and the second
is the phase's main result:

1. **The machine swings.** Unrelated processes on this machine move absolute
   timings by up to 4x inside one session, which both slices recorded and this
   phase reproduced (a Pyodide boot read `3.85 s` and `24.6 s` an hour apart, on
   the same page and the same command). Ratios and fractions reproduce; seconds
   do not.
2. **A hidden tab is a different machine.** Chrome deprioritizes a background
   renderer, and the automated browser tab this work was driven from is *always*
   hidden — `document.hasFocus()` can be `true` while `visibilityState` is
   `"hidden"`. Numbers taken there are not merely noisier; two of them came out
   **qualitatively wrong**, monotone, reproducible, and inverted by moving to a
   real window. See §3.

Measurements marked **visible** were taken in a real Chrome window that posts its
results back to the serving process (`web/serve.py` accepts `POST /results/<name>`),
because a visible window cannot be driven by the automation that would otherwise
read them.

## 1. What was built

| | |
|---|---|
| `src/sandbox/web/bridge.py` | protocol-only adapter: create a run from an `Experiment`-shaped dict, advance it, drain observables, report terminality, integrate the deterministic limit, run `validate()`, map a field to RGBA |
| `src/sandbox/web/colormap.py` | field → RGBA in numpy; three sequential maps and one diverging, this project's own |
| `web/worker.js` | boots Pyodide, installs the wheel, dispatches messages, owns the stepping loop |
| `web/client.js`, `web/monitor.js`, `web/render.js` | main-thread client, the event-loop instrument, the two renderers |
| `web/check.html` | 4a — conformance against native, and the responsiveness arms |
| `web/draw.html` | 4b — the drawing path, on a visible tab |
| `web/index.html` | 4c — the demo |
| `web/coldload.html` | the cold first load |
| `web/serve.py` | rebuilds the wheel, stages Pyodide, serves with gzip and an optional bandwidth cap, computes the native expectations |
| `web/protocol.md` | the message schema |
| `tests/test_web_bridge.py` | 48 tests, native |

Two changes outside `web/`:

- **`core/registry.py` gained a params type per model.** An `Experiment` names its
  model by a string so it can be serialized, but every service still took a
  `params_factory` from its *caller*. That works when a human writes the call and
  not at all for a front-end handed JSON and nothing else — an experiment that is
  serializable except for an out-of-band Python callable is only half serializable.
- **`core/recorder.py` grew `ReplicateRunner`, and `run_replicate` became a thin
  wrapper over it.** A front-end cannot run a replicate to completion and then
  draw; it must interrupt. That is one loop rather than two on purpose: the
  stepping semantics are subtle enough that a second implementation would drift.
  Four trajectories were sha256-fingerprinted before and after and are
  bit-identical, at `0.2238 s` against `0.2228 s` on the repressilator hot path
  (non-negotiable #4).

## 2. 4a — the worker runtime

### The verdicts match native

`web/serve.py` computes the native answer to every conformance spec **at
start-up, in the process serving the files**, and writes `web/expected.json`; the
page runs the same specs in the worker and compares. Same commit, same machine,
same NumPy — not a constant pasted in from a previous session.

| | browser | native |
|---|---|---|
| Wright-Fisher | `[PASS] fixed_A: measured 0.3375 vs predicted 0.3 (1.58 SE, tol 0.0947, n=400)` | identical, `\|Δ\| = 0` |
| repressilator | refuses: *no `analytic_predictions`* | `validate()` raises for the same reason |

**The repressilator's refusal is its verdict**, and "at the same tolerance" is
vacuous for it: its limit is a cycle, so there is no scalar for an ensemble mean
to match. A pass mark quoting a tolerance there would be theatre.

**The agreement bar is statistical, not bitwise**, as the plan requires. Both
happened to come out bit-identical — reported on the page as *informational*,
under a label saying so. Recorded fingerprints are native anchors; nothing here
is built on them holding. (They matched despite NumPy `2.2.5` in WebAssembly
against `2.4.6` native, which is a nice illustration of why the boundary is the
toolchain's transcendental functions rather than the version.)

### The page stays alive — and the worker is not the reason it looked slow

Identical work, 60 000 repressilator events, three arms, two interleaved rounds.
"Blocked" is the longest interval in which the main thread's event loop got no
turn.

**Visible window:**

| arm | blocked | on screen | stepping (Python's own clock) |
|---|---|---|---|
| worker, chunked 2 000 | 7 / 12 ms | 144 fps, worst frame 7 ms | 2.06 / 2.00 s |
| worker, one call | 8 / 18 ms | 144 fps | 1.99 / 1.82 s |
| main thread | **≥ 2 082 / 2 036 ms — zero event-loop turns, and no frames at all** | — | 2.08 / 2.03 s |

The main-thread arm does not merely stutter: in a *visible* tab it paints nothing
for two seconds. The worker arm costs **0.89-0.99x** the main thread's stepping
time, i.e. nothing.

### The finding: two conclusions that were artifacts of a hidden tab

The same three arms, run hidden, said the opposite — reproducibly, monotonically,
and with small spreads:

| | hidden | visible |
|---|---|---|
| worker penalty, chunked | **2.03x / 2.10x** | 0.99x / 0.98x |
| worker penalty, one call | 1.16x / 1.15x | 0.96x / 0.89x |

and the chunk-size sweep, over a **240x range** of chunk sizes:

| chunk | hidden (yield only / + drain & post) | visible |
|---|---|---|
| 250 | 3.59x / 3.23x | 1.01x / 1.01x |
| 1 000 | 2.09x / 2.38x | 1.02x / 1.02x |
| 4 000 | 1.31x / 1.66x | 1.00x / 1.00x |
| 16 000 | 1.01x / 1.02x | 0.98x / 1.01x |
| 60 000 (one call) | 1.00x / 0.91x | 1.00x / 1.00x |

Hidden, that is a clean monotone curve in the number of yields — exactly what a
real per-yield cost looks like, and it would have been written up as one. It is
Chrome handing a background renderer's worker back to a throttled scheduler once
per yield.

**So: chunking is free, and the chunk size is a display choice.** The slice's
conclusion stands, but its evidence did not reach the case — it split a
*synchronous* loop 4 096 ways with no yields between the pieces, and a streaming
UI yields.

Three things made this findable, and all three are transferable:

- **A third arm.** Two arms could say the worker was slower; only "the same work
  as one unchunked call" could say *why*.
- **A two-column decomposition** — yield-only against yield-plus-drain-plus-post.
  The two agreed at every chunk size, which ruled out the reporting and left the
  pausing, before the visible-tab run explained the pausing.
- **Interleaving.** The first sequential pair would have blamed whichever arm ran
  second, which is the trap the first slice recorded ("the interleaving was not
  ceremony").

### Instruments

`web/serve.py` sends COOP/COEP, so the page is cross-origin isolated and
`performance.now()` reads **0.005 ms** granularity instead of the slice's
coarsened 0.1 ms. The event-loop monitor is the slice's MessageChannel ping-pong,
and its "zero turns is the strongest reading, not a dead instrument" branch fired
correctly on every main-thread arm.

## 3. 4b — the drawing path

**All figures below: visible window, devicePixelRatio 1.5.** The plan's open item
was that the slice measured the drawing path's CPU cost while compositing and
vsync were never exercised, so *"the drawing path costs 0.3 ms"* was established
and *"the animation is smooth"* was not. It is now.

### Traces

The shipped configuration — 4 replicates of the repressilator at `Ω = 50` over
`t_max = 15`, streamed, **297 378-330 051 events each and every one of them
reaching the horizon**:

| | |
|---|---|
| redraw, both panels from scratch | median **1.37 ms**, p95 **1.70 ms** |
| page during the run | **144 fps**, worst frame 14 ms, event loop blocked 16 ms over 48.1 s |
| share of a 60 fps frame | 8.2% median, 10.2% p95 |
| points held | 6 279, drawn as per-column min/max envelopes |

(An earlier run of the *previous* configuration read 1.08 ms / 2.87 ms / 144 fps.
It is not quoted as the result because that configuration is the one that ran out
of `max_steps` at `t ≈ 8.5` — the figure fault in §6 — and a table describing the
shipped page should come from the shipped page.)

**It started at 43 ms median and a p95 of 1 030 ms**, because a redraw recomputed
the min/max envelope over every point every time — so drawing got more expensive
the longer a run went, which is exactly backwards for a page that streams. The
data is append-only and the x-scale is fixed for a run, so a point's column never
moves once assigned: folding the envelope in at `append` makes a redraw
`O(columns)` instead of `O(points)`.

Decimation is per-column **min/max, never a stride**. A stride draws a tidier
trace than the model produces, and the untidiness is the subject.

### Fields

Gray-Scott, `dt` derived from the grid rather than typed (CFL), 40 frames of 10
steps, median per stage:

| grid | dt | step | colormap | wasm→JS | canvas | drawing total | frame | on screen |
|---|---|---|---|---|---|---|---|---|
| 64² | 0.5 | 11.83 ms | 0.610 | 0.115 | 0.095 | **0.82 ms (6.2%)** | 13.2 ms | 142 fps |
| 128² | 0.2 | 24.24 ms | 1.295 | 0.145 | 0.105 | **1.55 ms (5.8%)** | 26.4 ms | 144 fps |
| 256² | 0.05 | 72.89 ms | 4.380 | 0.195 | 0.125 | **4.70 ms (6.0%)** | 78.6 ms | 144 fps |

The page holds 144 fps at 256², where a single frame costs 78.6 ms of compute —
which is the whole point of the worker.

**The fraction is the measurement, and here is the evidence for saying so.** The
sweep was run twice on the same code, hours apart, on a machine that had moved
under it: the step times differ by **1.5-1.6x** between the runs (`6.03 / 15.18 /
46.15 ms` in the first, `11.83 / 24.24 / 72.89 ms` in the second) while the
drawing fraction reads **6.2 / 5.4 / 6.0%** and **6.2 / 5.8 / 6.0%**. The
absolute seconds are illustration; the fraction reproduces.

**The fraction disagrees with the slice and the disagreement is real, not noise.**
The slice recorded 1.6-2.2%; this reads 5.4-6.2%. Two causes, both measurable:
this colormap costs `0.33 / 0.77 / 2.76 ms` against its `0.13-0.82`, and the step
here is faster. **Recorded rather than optimized** — the profile says the step is
94% and non-negotiable #4 says believe it. The conclusion is unchanged and the
number is not; quoting the slice's 1.6-2.2% for this code would have been wrong.

**A finer grid pays twice**, and simulated time per second is the figure that
shows it: **306 / 69.8 / 6.0** time-units per second, a **51x** drop where the
frame rate suggests 6x, because CFL forces `dt` from 0.5 to 0.05. (The first run
read `545 / 96 / 9.3`, a 59x drop — the absolutes moved with the machine, the
shape did not.)

## 4. The cold first load

The plan named this the one measurement that could still overturn the fork
decision, and told the reader to take it early rather than discover it at 4c.
`web/serve.py --bandwidth` paces its writes; `web/coldload.html` reports every
byte, **including the ones the worker fetched on its own thread**, which the
page's own resource timeline cannot see.

| condition | over the wire | uncompressed | running interpreter | first simulated points |
|---|---|---|---|---|
| local, unthrottled | 9.01 MB | 15.78 MB | 2.53 s | 3.72 s |
| 25 Mbit/s throughput | 9.01 MB | 15.78 MB | 4.29 s | 5.28 s |
| 5 Mbit/s throughput | 9.01 MB | 15.78 MB | 12.22 s | 13.20 s |

Biggest items over the wire: numpy 3.11 MB, `pyodide.asm.wasm` 2.83 MB (from
8.65 MB raw), `python_stdlib.zip` 2.42 MB, the project's own wheel 0.19 MB.

**The decision stands.** A common home connection puts a reader in front of
running simulated data in about five seconds. Five megabits is slow but usable.

**What this is not: a real network.** A paced local server models the
*throughput* term and nothing else — no latency, no TCP slow start, no loss, no
CDN, no mobile radio. A real 5 Mbit/s link is worse than 12.2 s, and by how much
is **not established here**. The measurement also downloads the wheel twice
(once to hash it, once to install it) because the server sends `no-store` so a
cold load stays cold; that is 2% of the total and an artifact of the rig, not of
the design.

## 5. 4c — the demo

Three acts: watch it run with the system size as a control; **D**, the distance
from the limit; and the ValidationSuite live, passing for Wright-Fisher and
refusing for the repressilator.

`D` reuses `convergence.py`'s own `_per_replicate_discrepancy` rather than a
lookalike, because that function averages over time and species **per replicate
before** the replicate mean, which is the order that avoids the phase-diffusion
trap the module was built around.

**The D panel was seed-verified before it shipped, and the first config would
have contradicted its own caption.** At `t_max = 6` with 3 replicates it read
`2.60 / 3.50 / 1.51` — out of order. Measured properly at 10 replicates the law
is clean:

| Ω | 5 | 15 | 45 | 135 |
|---|---|---|---|---|
| D | 6.9664 ± 0.9454 | 4.1887 ± 0.4845 | 2.7620 ± 0.4374 | 1.4056 ± 0.2030 |
| D·√Ω | 15.577 | 16.223 | 18.528 | 16.331 |

The shipping config — `t_max = 8`, 6 replicates, `Ω ∈ {5, 15, 45}`, about a
minute in the browser — is monotone at **4 seeds of 4**, with the ends separated
by `2.8 / 3.0 / 4.9 / 2.8` standard errors and nothing truncated. In the browser
it read `7.29 / 3.68 / 3.03`, ends at 4.3 SE.

**Then the caption had to be corrected in the other direction.** It said the
standard errors overlap and runs would come out in the wrong order — which the
seed check had just disproved. Being falsely modest about a measurement is the
same failure as overclaiming it. It now says the direction is solid, the
*exponent* is not established by three noisy points, and it prints its own
measured factor (2.41) against the predicted one (3.0) rather than rounding
quietly toward the theory.

## 6. Figures, again

The project's most-repeated lesson recurred three times in this phase, and all
three were caught by looking at the picture rather than by any check.

- **A run that stopped early read as perfect agreement.** The trace panel asked
  for 400 000 events at `Ω = 100` over `t_max = 30`; that needs ~1.4 million, so
  the replicates stopped at `t ≈ 8.5` while the limit was drawn all the way to
  30. The right-hand side of the frame showed the limit alone, which looks like
  the replicates settling onto it exactly. Both pages now size the budget from a
  measured event cost (~470 events per time unit per unit of `Ω`) **and report
  whether every replicate reached the horizon.**
- **A justification invented rather than measured.** The two-panel split was
  explained as "the proteins reach ~160 and the mRNAs ~10". They reach 135.4 and
  153.0. The split is right — six oscillators in one frame cannot be followed —
  but the stated reason was false, and the panels' near-identical shape is a
  *result* (`β = 1` equalizes the timescales), not a duplication.
- **A stopwatch mistaken for a demonstration.** The field benchmark runs 400
  steps per grid, i.e. `t = 200 / 80 / 20`, so its picture is the seeded square
  patch barely moved. Gray-Scott does not make squares. Captioned.

And one caught before it could appear: the scale bar under the field is rendered
from a strip the **bridge** produced through the same `to_rgba` as the image, so
it cannot disagree with the picture it labels.

## 7. Green for the wrong reason, and the teeth

- **The spawn chain.** `run_experiment` spawns one `SeedSequence` per sweep point
  and *then* one generator per replicate. A bridge spawning directly from the
  seed produces a different — equally valid, equally reproducible, completely
  incomparable — run, and the disagreement would look like a WebAssembly
  floating-point problem. Mutating the bridge to a one-level spawn turns exactly
  four tests red. A non-vacuity test records that the two chains do differ, so
  the pin cannot be passing for free.
- **Chunk sizes that are not multiples of `record_every`.** Recording is gated on
  the *cumulative* step index. A per-chunk counter is shipped as a tooth and is
  red — and it is checked at chunk 7 against stride 5, because a grid of
  multiples could not see it. Its generator states are asserted **equal**, so the
  tooth is biting the recording and not some other bug.
- **`json.dumps` emits a bare `NaN` by default**, which is not JSON and which
  `JSON.parse` rejects — a payload that round-trips in Python and dies at the
  boundary. Non-finite values cross as `null`, asserted with `allow_nan=False`.
- **Colormap luminance.** Sequential maps are asserted monotone in linearized
  Rec. 709 luminance; the diverging map is asserted **not** monotone, so the
  first check cannot be passing for free.
- **A worker outlives its page.** `WorkerClient` never terminated its worker, so
  every reload during a run left an orphan grinding through the rest of it with a
  whole Pyodide heap. A few reloads while iterating are enough to make the next
  boot look mysteriously slow, which is how it was found.

## 8. Suite timing

The rule, on its fourth repetition: a total needs a **same-session baseline**, the
baseline needs the **worker tags**, and the tags need the **per-test durations**.
Both runs below are one session, back to back, same command apart from the ignore.

| | tests | total | repressilator pair | rest |
|---|---|---|---|---|
| full | 776 | **346.23 s** | 152.69 + 125.44 = **278.13 s** | 68.10 s |
| baseline (`--ignore=tests/test_web_bridge.py`) | 720 | **286.12 s** | 119.43 + 96.55 = **215.98 s** | 70.14 s |

**The decomposition is unusually clean, and it settles the question without
needing the packing.** The total grew by **+60.11 s** for 56 added tests — and the
two repressilator tests, which this phase does not touch, grew by **+62.15 s**,
*more than the entire gap*. Everything else fell slightly (70.14 → 68.10 s). So
the difference is drift in the critical path, **no cost from the new tests is
visible**, and the 56 bridge tests cost 9-12 s standalone anyway.

**And the worker tags say why, definitively.** A third run with `-v` (the first
attempt passed `-q -v` together, and `-q` wins — the tags were simply absent):

```
[gw4] [ 82%] PASSED test_repressilator.py::test_fixed_omega_propensities_fail_the_convergence_check
[gw4] [ 99%] PASSED test_repressilator.py::test_squared_omega_propensities_fail_the_convergence_check
[gw4] [ 99%] PASSED test_repressilator.py::test_repressilator_discrepancy_scales_as_omega_minus_half
776 passed in 334.00 s
```

**All three long repressilator tests land on `gw4`, back to back** — roughly
`34.9 + 125.4 + 152.7 = 313 s` of a `334 s` total. One worker is **94% of the
wall clock**, and the other five finish and idle. So the suite's total is not a
measure of the suite: it is a measure of `gw4`, and adding tests cannot move it
unless they land there. That is the mechanism behind the decomposition above,
*observed* rather than inferred — and it is why the `+60.11 s` and the
`+62.15 s` are the same number wearing two hats.

(The identical command read `346.23 s` and `334.00 s` on two runs, a 3.5% swing,
which is this machine on a good day.)

**Conditions, because they are unusually bad and it would be dishonest to omit
them.** The machine was at **100% CPU across 16 cores** during both runs, with
**30 Python processes belonging to unrelated projects** on it (a space-station
build and a particle-accelerator one) alongside this suite's 14. The absolute
totals are therefore not comparable to any earlier recorded figure — the Phase-3
close-out's `225.80-303.91 s` and the `373.81 s` re-run are from different machine
states. What survives contention is the *decomposition*, because both arms were
contended together.

## 9. The three deferrals, now built — and what building them found

All three items §9 recorded as deferred are done. Each is worth reading for the
same reason: **every one of them was broken in a way that only doing it could
show**, and two of the three defects were in code that had been read and
pronounced correct.

### 9a. The stop button — it was broken, and reading it again is what found it

The deferral said *"the path reads correctly … but it was never actually
clicked."* Reading it a second time found a window in which it did nothing at
all. A page holds a `run_id` **before** anything is stepping it — the demo
creates the run, integrates the deterministic limit (a visible pause), and only
then sends `run` — and a `cancel` arriving in that window had no loop to set a
flag on. The worker replied `cancelling: false`; the page ignored that field,
disabled its stop button, and let the run go to completion **with nothing able to
stop it**.

A cancel is a fact about the *run*, not about whichever loop happens to be live,
so one arriving early is now remembered and applied when the loop starts, which
returns immediately with `stepped: 0` and `terminal` still false. The reply
carries `when` — `"running"` or `"before-first-step"` — so a page says which
happened instead of inferring it.

`check.html` §5 checks three cases and is confirmed able to fail: reverting the
worker to the old semantics leaves (a) and (c) green and fails (b) with the run
taking **the full 60 000 events despite having been cancelled**. Then the button
was actually clicked, which is what the deferral was about: at Ω = 30 with four
replicates it stopped at **116 000 of a 338 400-event budget**, the replicate
lines end at `t = 11.0 / 9.0 / 8.9 / 9.3` while the limit runs to 15, and the
readout attributes that to *you* rather than to a budget shortfall. Those are
different facts producing an identical picture, and the page now separates them.

**Correctness only, and not latency.** A cancel lands at the stepping loop's
`await setTimeout(0)`, which is exactly where a hidden tab hands the worker back
to a throttled scheduler, so a cancel-latency figure taken by automation would be
a number about tab visibility wearing the costume of a number about the protocol.

A second, smaller defect fell out of clicking: **the run button was clickable
before boot finished.** Its listener is attached after `await boot`, so an early
click landed on a button with no listener and did nothing — an unresponsive page
rather than a slow one, with no way to tell which. It now starts disabled, like
the two buttons under it.

### 9b. The model picker — `web/models.html`, and it found three real defects

Driven entirely by `describe` and `default_params`, so the list, the capability
flags and the parameter fields are asked for at run time and adding a model to
the project adds it to the page. **13 of the 14 were driven through it in the
browser** (the 14th is the repressilator, which is `index.html`'s own model), and
all 14 are covered natively by `tests/test_web_assets.py`. Driving all of them
found three things three pages had not:

- **The field path was broken for its only untaken branch.** `web_field` received
  a JS `null` for an absent `vmin`/`vmax`, and Pyodide 0.28 maps `null` to a
  **`JsNull` singleton, not to `None`** — so `vmin is None` was false and
  `float(vmin)` raised. Nothing caught it because no page autoscaled: `draw.html`
  deliberately fixes its range. Now JSON in, like every other entry point, where
  there is no JS value left to mistranslate.
- **A model can declare an ODE limit it can never have drawn.** `glv_stochastic`
  reports five observables — three species plus `total_biomass` and
  `n_survivors` — against a three-component ODE, and the bridge overlays the
  limit column by column and refuses anything not corresponding one-to-one. The
  refusal is right; declaring the block was the error. Any model reporting
  *derived* observables alongside its state has the same problem.
- **A `FieldModel`'s field need not be 2-D.** `trait_branching` declares
  `abundance` with shape `(161,)`, the trait grid, and the image path needs two
  dimensions. `create` now reports `field_shapes` so a page can say which it is
  *before* drawing, rather than putting an exception where the picture was going
  to be. Filtering it out of `fields` was the alternative and is worse — the
  model does declare it, and silently omitting a model's own output is the more
  expensive lie.

Two hazards the page was built to avoid, both from this project's own record.
**The budget is measured, not guessed:** the demo's `470 events per time unit per
unit of Omega` is a *repressilator* number, and reusing it would be the recorded
mistake of carrying a constant away from its estimator, so
`serve.py --measure-presets` prices every preset to termination and the numbers
live in `web/presets.json` beside the params that produced them. And **the page
says whether the replicates are actually different**, measured from what was
drawn rather than looked up in a table of which models are stochastic — because
three bit-identical curves drawn on top of each other look exactly like one, and
a legend claiming three is then a figure making a false claim. Checked on
`hodgkin_huxley` at three replicates: *"no — all 3 are bit-identical … you are
looking at one curve drawn 3 times, not at a spread."*

The presets are **data**, not page code, for the reason `conformance.json` is:
every params block is copied from a demo or a test that already runs it, with
`source` naming which, so no configuration on that page was invented.

### 9c. The figure export — and the cold load held at 9.01 MB

`bridge.figure_png` renders a finished run through
`viz/backends/matplotlib_backend.plot_replicates` — the project's own teaching
figure, already exercised by the suite — rather than a second plotting path free
to drift from the one every recorded figure was made with. The limit's columns
are matched to the trace **by observable name**, never by position: on a
symmetric limit cycle the wrong column is the right shape at the wrong phase,
which looks entirely plausible.

`serve.py --with-figures` stages matplotlib and its **eleven** dependencies —
12 wheels, 13.09 MB on disk — with the closure resolved transitively from
`pyodide-lock.json` rather than hand-typed, in both the server and the worker. It
gets its **own** presence check, because `stage_runtime` returns as soon as the
wasm, the lock and a numpy wheel are present and folding matplotlib into that
test would silently skip it on an already-staged directory.

**The constraint that justified the deferral was then verified rather than
assumed — and the first wording of the result was wrong.** A worker that only
boots pulls **9.006 MB**, the recorded figure unchanged, and the phrase written
into four places was *"zero matplotlib bytes"*. But the real page also calls
`figure_available` at load, which the throwaway worker never did. Measuring the
same worker **before and after** that call gives **9.006 → 9.032 MB**: `+26 KB`,
being a re-read of `pyodide-lock.json` (the server sends `Cache-Control:
no-store`, so nothing is cached) plus a `HEAD` on the matplotlib wheel costing
**300 bytes of response headers with `decodedBodySize = 0`**.

So the correct claim is **no matplotlib *body*, 300 bytes of headers**, and the
cold load is **9.03 MB** rather than 9.01. The 2.1x bundle multiplication the
deferral exists to avoid is still entirely absent. The correction is worth more
than the 0.3% it changes: the original number was measured on an instrument that
did not do what the page does, which is the same class of error as measuring in a
hidden tab.

A second untaken branch was then taken deliberately: moving the staged wheel aside
and reloading, the page **still boots and still runs** — `run`, the D measurement
and the validation all work — and only the export is disabled, reading
*"matplotlib is in the lock but its wheel is not staged. Restart the server
with: python web/serve.py --with-figures …"*. The top-level `figure_available`
call is `.catch`-guarded for the same reason: a rejection there would kill the
module script before any listener attached, which is the inert-page-with-empty-
console signature this phase had already debugged once.

Three costs are reported separately, because they differ by an order of magnitude
and one number would misprice all three. Measured in the browser: **0.52 s** to
fetch and install (once per worker), **1.24 s** to import, **0.58 s** to draw, for
a six-panel two-replicate figure at 179 KB. The 0.52 s is a **local** fetch of
~10 MB and is not a network measurement. Natively the import is 0.42 s on a warm
font cache and the *first ever* call took 3.88 s building it — which is why the
browser's import is 3x the native repeat: Pyodide has no persistent disk, so
every session pays it.

And the figure was **looked at**, not exit-code checked, twice — once natively and
once rendered in the browser. The test that guards it counts ink rather than
structure, because the failure this project actually shipped was a valid PNG with
correct axes and no data at all: every structural property held and only the ink
was missing. The thresholds sit in a measured gap — rendering one panel four ways
gives blue/red pixel counts of `2412/3271` (both drawn), `0/3457` (a run that
recorded nothing), `3845/0` (no limit) and `0/0` (neither). **The empty cases are
not merely small; they are zero.**

### 9d. What the browser found that a test could have

`tests/test_web_assets.py` — 68 tests, **2.2 s**, no browser — exists because
every bug above was found by opening one, which is the expensive way to find any
of them. It compiles the Python that `worker.js` embeds in a JS template literal,
checks every `pyCall` has a matching `def`, and runs each preset briefly.

That first check earned itself immediately. Writing the comment explaining the
`JsNull` fix, the word `null` was wrapped in **backticks** — inside a JS template
literal, which ended the string and broke the entire worker with a syntax error
pointing at a line nowhere near the cause. The page hung on *"starting Python in a
background thread …"* with nothing in the console, because the failure killed the
module script before it could log anything. A test that `compile()`s the embedded
block finds it in 0.02 s.

All six mutations bite the intended test and nothing else, and one test was
**wrong and the model was right**: a flat floor of "at least 100 recorded points"
failed `adaptive_dynamics` at 57, because its trait-substitution sequence is a
jump process with 57 events in the whole run and at `record_every = 1` it is
already keeping everything that exists. A floor on the point count is a claim
about the model; what a preset can be held to is a claim about the *thinning*,
which is the only part it chooses. A second was wrong in the other direction:
comparing the presets to `available()` passed alone and failed in the suite,
because other test modules register deliberately-broken teeth models into the
same global registry.

### 9e. Suite

**846 passed, 11 skipped in 157.34 s** at `-n 6`, against a same-session baseline
of **776 passed in 407.13 s** with the new module ignored. The suite is *faster*
with 70 more tests in it, so — by this project's own rule, now on its fourth
repetition — **the totals are not attributable to the test set** and no
regression is visible.

Four runs of the same command in one session read **`407 / 342 / 255 / 157 s`**:
a **2.6x monotone drift**, far larger than anything being measured, and the
sharpest illustration this project has yet produced of why a suite total is
meaningless without a same-session baseline. (Earlier phases recorded `±30%`
within a session; this is `2.6x`. The machine was shedding load from the
concurrent browser automation and repeated pytest runs across the session.) What
*is* attributable is the new module's standalone cost: **2.7 s**.

And one thing the plan raises that correctly needed **no** action: the WebGL
trap. The default answer stayed no. Nothing here was tempted, because the drawing
path is 6% of a frame and a shader would be a second implementation of a
validated model's numerics in a language no test here can reach.

## 10. What this phase does not establish

- **A real network.** §4 measures throughput at a known byte rate. Latency, slow
  start, loss, CDN behaviour and mobile radios are not modelled.
- **Any device but this one.** One machine, one Chrome, dpr 1.5, 144 Hz display.
  Every fps figure is capped by that display.
- **That the browser and native agree bit-for-bit in general.** They did for both
  conformance specs, and that is reported as a convenience under a label saying
  so. The pass mark is statistical.
- **Anything about the models.** Nothing on any page is asserted anywhere. What
  is asserted is checked natively, and the tolerances are derived.
