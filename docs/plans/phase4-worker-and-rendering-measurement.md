# The worker and the drawing path, measured

`phase4-browser-fork-measurement.md` priced the *numerics* in a browser (~2x
native, bit-identity for six models, ~8.3 MB over the wire) and deliberately
skipped two things: matplotlib under Pyodide, and any front-end at all. It then
concluded that an interactive page "needs a Web Worker regardless of how fast
the numerics turn out to be" — and labelled that honestly as **"a design
consequence, not a benchmark result"**.

This document turns that consequence into a benchmark result, and prices the
part the first measurement left out: **what it costs to get a running simulation
onto a screen.** Like its predecessor it **commits to nothing**; the plan built
on it is `phase4-plan.md`.

Reproducible from `M:/claud_projects/temp/browser-slice` (`index.html`,
`worker.js`, `py_setup.py`, served over `http.server`; the Pyodide runtime and
the project's own wheel are copied from the earlier slice).

## 0. Read this before any number below

**The machine was contended throughout.** Unrelated user processes held
~25-30% of 16 cores, and absolute timings swung by **4x inside one session** —
the Gray-Scott compute rate at `n = 128` read `384 / 429 / 110` steps/s across
three consecutive repeats of the same loop. This is the project's recorded
"bracket any suite-timing claim by ±30%" being generous again, exactly as the
first browser measurement found.

So this document quotes **two runs**: a quiet one (single-shot, `?v=3`) and a
contended one (3 repeats per point, `?v=5`). **Ratios and fractions are quoted
as measurements** — they reproduce across both runs. **Absolute seconds are
quoted as ranges spanning both**, and are illustration, not measurement.

## 1. Three instruments were wrong before any of them was right

This is the most transferable part of the slice, and two of the three failures
produced *plausible numbers* rather than errors.

**The freeze detector could not fail.** The obvious way to measure "is the page
frozen" is to watch `requestAnimationFrame` and look for gaps. But rAF is
**suspended entirely in a background tab**, and the tab driving this slice is
backgrounded (the automation can screenshot a window that Chrome considers
hidden — `document.hasFocus()` was `true` while `document.visibilityState` was
`"hidden"`). The monitor therefore reported:

```
main_thread_busy_60000: worst freeze 1641 ms      <- correct, by accident
worker_busy_60000:      worst freeze 1840 ms      <- the worker arm. Nothing was blocked.
```

Both arms read the same, because neither was being measured. **A check that
reports "blocked" for the arm designed not to block is not a check.** Replaced
with a **MessageChannel ping-pong**: a message is a *task*, not a timer, so it
is throttled neither by background-tab timer clamping nor by rAF suspension, and
event-loop latency is precisely what "the page is frozen" means. The same probes
then separated by three orders of magnitude (§2).

One further correction inside that instrument: a window with **zero** ping-pong
turns is not a dead monitor, it is the **strongest reading available** — the
thread never reached the task queue, so it was blocked for the entire window.
The first version printed that case as `[event-loop monitor produced nothing]`,
which reads like an instrument failure and is in fact the headline result.

**`performance.now()` is coarsened to 0.1 ms here**, on any page that is not
cross-origin-isolated. Every single-shot sub-millisecond reading in the first
run was a rounding artifact — the conversion column came out as
`0.200 / 0.000 / 0.100 / 0.500 / 0.800 ms`, i.e. exact multiples of the clock
tick, including two zeroes. Everything short is now timed as a **batch of 200-300
and divided**. The measured granularity is printed by the page itself.

**And one comparison was vacuous.** The probe compared pulling a numpy array out
via `getBuffer()` against the "obvious, slower" `toJs()`, and measured them as
equal (`0.5-1.0x`). They are equal because **`toJs()` on a float64 ndarray
returns a `Float64Array`** — verified by reading `.constructor.name` rather than
by trusting the ratio. The two arms were one operation spelled twice. The
finding is not "`getBuffer` wins"; it is **there is no slow path here to avoid**.

**A fourth, in the oldest category this project has.** The first repressilator
figure rendered in the browser came back as a valid 20 KB PNG with correct axes
and **no data**: the series filter tested `k.startswith("m")` while the
observables are named `x_m1 … x_p3`. Exit code clean, byte count plausible,
figure empty. Caught by looking at it. *A figure carries a claim and gets looked
at, not exit-code checked* — now also true of figures drawn in a browser.

## 2. The Worker claim, now a measurement

Identical work — one repressilator SSA run — on the main thread and in a worker.
"Blocked" is the longest interval in which the main thread's event loop got no
turn.

| work | main thread | in a worker |
|---|---|---|
| 5 000 events | **blocked ≥ 200 ms** (the whole run; 0 turns) | blocked 1-13 ms |
| 20 000 events | **blocked ≥ 590 ms** (0 turns) | blocked 8-45 ms |
| 60 000 events | **blocked ≥ 1 750 ms** (0 turns) | blocked 10-23 ms |
| Pyodide boot | **blocked 1 081 ms** of a 2.04 s boot | blocked 11-16 ms of a 1.83-1.89 s boot |

The main-thread arm never yields **once**: not a long frame, a total stall for
the duration. The worker arm leaves the page live throughout (35 000-440 000
event-loop turns per window).

**The worker costs nothing in throughput.** Python-side timing of the same 20 000-event
run: `0.5826 s` on the main thread against `0.5823 s` in the worker. At 60 000
events the two differ by ~8%, inside this machine's scatter.

So HANDOFF's design consequence holds, and is now quantitative: **the plumbing is
not optional, and it is not expensive.**

## 3. Crossing the boundary is free; copying at scale is not

**JS → Python calls: no cost resolved.** Identical total work split 1, 4, 16, 64,
256, 1 024 and 4 096 ways, three repeats each. The point estimate comes out
*negative*; the honest statement is a bound from the run-to-run scatter:

- quiet run: scatter `11 ms` on a ~115 ms workload → **< 3 µs per call**
- contended run: scatter `93 ms` → **< 23 µs per call**

Either way, **4 096 extra boundary crossings cost less than the noise on a single
115 ms workload.** A stepping loop may be chunked as finely as the UI wants; the
chunking itself is not what costs.

**numpy → JS**, per array (batched, 200-300 reps):

| floats | out of the WASM heap |
|---|---|
| 100 | 0.014-0.017 ms |
| 1 000 | 0.015-0.019 ms |
| 10 000 | 0.050-0.055 ms |
| 100 000 | 0.16-0.25 ms |
| 1 000 000 | 1.8-3.9 ms |

**worker → main round trip**, transferable versus structured-clone copy:

| floats | transferred | copied |
|---|---|---|
| 0 | 0.09-0.24 ms | 0.04-0.10 ms |
| 1 000 | 0.04-0.07 ms | 0.05-0.21 ms |
| 100 000 | 0.27-0.40 ms | 0.70 ms |
| 1 000 000 (8 MB) | **0.41-0.78 ms** | **9.7 ms quiet, 2 812 ms contended** |

Below ~1 000 floats the two are indistinguishable and transferring has a small
fixed overhead. At 8 MB the copy is 24x worse when the machine is quiet and
**collapses by three orders of magnitude when it is not** — the copy path is the
one that degrades catastrophically under memory pressure, which is exactly the
condition a heavy page creates for itself. Use transferables for fields; do not
bother for scalar observables.

## 4. The drawing path is ~2% of the frame. The step is everything.

Gray-Scott stepped in the worker, colour-mapped in numpy, shipped as RGBA bytes,
drawn with `putImageData` + `drawImage`. Ten steps per frame, 40 frames, three
repeats.

| grid | dt | step | colormap | wasm→JS | canvas | **drawing total** | frame |
|---|---|---|---|---|---|---|---|
| 64² | 0.2 | 9.2-10.6 ms | 0.13-0.18 | 0.04-0.06 | 0.04-0.06 | **0.23-0.28 ms (2.2%)** | 13.0 ms |
| 128² | 0.2 | 19.4-24.2 ms | 0.27-0.29 | 0.07 | 0.06 | **0.40-0.43 ms (1.6%)** | 26.0 ms |
| 256² | 0.1 | 57.9-58.4 ms | 0.79-0.82 | 0.10 | 0.10-0.12 | **0.99-1.02 ms (1.7%)** | 60.3 ms |

**The fraction is the measurement**: 1.6-2.2%, stable across every grid and both
runs, while the absolute rates swing 4x. Getting a field onto a canvas is not a
cost worth designing around. **The simulation step is 98% of the frame**, and it
is the only thing worth optimizing if anything ever needs to be.

**A finer grid pays twice, and quoting frames per second hides it.** The CFL
limit forces `dt` down as the grid refines, so simulated time per second is
`steps/s × dt`:

| grid | steps/s | dt | **simulated time-units/s** |
|---|---|---|---|
| 64² | 771-976 | 0.2 | **154-195** |
| 128² | 384-485 | 0.2 | **77-97** |
| 256² | 166-167 | 0.1 | **16.6** |

From 64² to 256² that is a factor of **~9-12**, not the ~5-6 the frame rate
suggests. `dt` is derived from the grid in the probe rather than typed, because a
`dt` that is stable at 64² is unstable at 256².

**Caveat this document cannot discharge: none of this was on screen.**
`putImageData` and `drawImage` execute whether or not the tab is visible, so the
**CPU cost** of the drawing path above is genuinely measured. **Compositing and
vsync were never exercised**, so *"the drawing path costs 0.3 ms"* is supported
while *"the animation is smooth at 128²"* is a different claim and is **not**
established here. The slice carries a probe that measures on-screen frame rate
and refuses to fake it, waiting for a visible tab.

## 5. matplotlib works, and more than doubles the download

It renders correctly in the browser — real axes, fonts, legend, six series,
verified by looking at the PNG (§1).

| | raw | gzip |
|---|---|---|
| core (Pyodide + numpy + stdlib + our wheel) | 15.42 MB | **8.73 MB** |
| matplotlib + its 10 dependencies | 9.98 MB | **9.69 MB** |
| both | 25.40 MB | **18.42 MB** |

**Gzip buys nothing on the matplotlib half** — wheels are already-compressed
archives — so adding it multiplies the transfer by **2.1x**, from 8.7 MB to
18.4 MB. Against that: first import 1.5-6.3 s, then **0.07-0.16 s per figure**,
~60 KB per PNG. Static figures are cheap once loaded; the download is the cost.

## 6. The demos need more stepping than the first measurement assumed

The earlier document concluded a browser can host "a few seconds of stepping".
Two observations from actually running the demos say that is short:

- Gray-Scott after 400 steps shows **only the seeded blob**. Pattern formation at
  Pearson's parameters is `t ~ 10³`, i.e. thousands of steps — **tens of seconds**
  of continuous stepping at 128².
- The repressilator at `max_steps = 40 000` reached `t ≈ 3.2` of a `t_max` of 30,
  so the figure showed the first rise and **no oscillation**. A full trace is
  ~400 000 events, **~15 s** at the measured ~27 000 events/s.

Neither is a problem for the fork decision — both are still seconds, not the
minutes the convergence sweeps take, and those stay native. It is a constraint on
**what a demo looks like**: it must display progressively while it runs rather
than compute-then-draw, which the worker architecture supports natively and a
main-thread page could not do at all.

## 7. What was deliberately not measured

- **On-screen frame rate** — pending a visible tab; see §4.
- **matplotlib's interactive backend.** Only `Agg` to a PNG was measured, which is
  the mode a worker can use. Interactive figures need the main thread and would
  reintroduce exactly the blocking §2 measures.
- **Mobile, and cold-cache network time.** Neither transfers from a warm desktop.
- **The ValidationSuite in a worker.** It ran in-page last session; nothing here
  suggests a worker changes it, and it was not re-run.
