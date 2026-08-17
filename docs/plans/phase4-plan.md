# Phase 4 — The browser front-end (plan)

**Status: planning.** Phases 0-3 are complete. This phase closes the one item
HANDOFF marks *"decide early"* and that was never decided: the browser-vs-local
fork (§4).

**The decision is: build the browser front-end.** It was taken after the fork was
measured rather than argued, in two passes that between them replace both of
HANDOFF's stated cost drivers with numbers:

- `phase4-browser-fork-measurement.md` — the numerics. ~2x native, six models
  bit-for-bit identical in WebAssembly, ~8.3 MB over the wire.
- `phase4-worker-and-rendering-measurement.md` — the plumbing and the drawing.
  The worker requirement as a measurement, and the cost of getting a running
  simulation onto a screen.

Everything quantitative below comes from those two documents. **Read §0 of the
second one before quoting any absolute number from either**: the machine was
contended, and only ratios reproduce.

## Two things in HANDOFF do not survive the measurements

**§4's cost for this branch is wrong.**

> "**Browser** (Pyodide, or a JS/WASM core) … Costs a reimplementation of the
> numerics."

It does not, for this codebase. The dependency surface is `numpy` plus the
standard library; Pyodide ships NumPy; the project's own wheel installs
unmodified and the ValidationSuite passes in a browser tab. There is no second
implementation to write and — the cost that would actually have mattered — **no
second implementation to keep correct**.

**§8 is stale.** "Immediate next action: implement Phase 0" has been true of no
session since Phase 0 closed. It should record the fork decision instead; that
edit belongs in this phase's first commit.

**What the browser genuinely costs** is therefore neither of the things HANDOFF
names. It is: a **Web Worker and a message protocol** (§2 of the second
measurement — the main thread blocks for the *entire* duration of any run
otherwise, 1 750 ms for a 60 000-event SSA, never yielding once), and a
**download** (8.7 MB gzipped, and 18.4 MB if matplotlib ships with it).

## Goal

Make the project's central teaching move — *stochastic replicates converging onto
their deterministic limit, with a validation check the reader watches pass* —
visible **without a Python install**. Nothing else.

Explicitly **not** a goal: moving the project into the browser. The convergence
sweeps are minutes of CPU, 2x worse in WASM, and stay native. The test suite
stays native (`pytest-xdist` cannot spawn processes in WASM). This phase adds a
**front-end for the demo-shaped workload** and changes nothing about how the
project defines "done".

## The validation posture (unchanged, and stated before any code)

A browser run is validated **statistically**, through `analytic_predictions` and
the ValidationSuite — which is how every model in this project is defined as
correct anyway. Non-negotiable #4's bit-identity requirement governs **SSA
optimizations within one implementation**, so recorded slope anchors stay valid;
it says nothing about a second *platform*, and a WASM build differing from x86 at
`2.2e-15` does not violate it.

**Recorded fingerprints remain native anchors.** That six models happen to
reproduce bit-for-bit in WebAssembly is a convenience, and no part of this phase
may be built on it holding.

## Architecture

Three layers, and the boundary between them is the existing protocol.

```
main thread          worker                     Pyodide
-----------          ------                     -------
DOM, canvas   <--->  message protocol   <--->   sandbox.web.bridge
                     (JSON + transferables)       |
                                                  +-- Model.step / observables / fields
                                                  +-- Recorder, validate()
```

**`sandbox/web/bridge.py` is another shared service, and non-negotiable #1 applies
to it in full.** It consumes only the protocol surface — `initial_state`, `step`,
`observables`, `is_terminal`, `analytic_predictions`, and `fields` for the models
that have it — and never reaches inside a concrete state. It is the same
relationship the Recorder already has. `FieldModel.fields` exists for exactly this
purpose and remains **viz-only**; nothing checked may depend on it.

**No numerics in JavaScript.** The JS side owns the DOM, the canvas and the
message loop, and nothing else. This is not stylistic: a JS or shader
reimplementation of a model *is* the "reimplementation of the numerics" that
HANDOFF wrongly attributed to Pyodide and that the measurement showed this branch
avoids. See "The WebGL trap" below.

### What the measurements settle about the design

- **Chunk the stepping loop as finely as the UI wants.** 4 096 JS→Python calls
  cost less than the run-to-run noise on a single 115 ms workload; the per-call
  cost could not be resolved (bound: < 3 µs quiet, < 23 µs contended).
- **Draw fields as RGBA bytes mapped in numpy, shipped as transferables.** The
  whole drawing path is **1.6-2.2% of a frame** at every grid size tested. It is
  not worth designing around.
- **Use transferables for fields, don't bother for scalars.** Below ~1 000 floats
  a transfer and a copy are indistinguishable; at 8 MB the copy is 24x worse when
  the machine is quiet and **300x worse when it is not**.
- **The simulation step is 98% of the frame.** If anything is ever optimized, it
  is that, and only after profiling says so (non-negotiable #4).

## Sub-phases

### 4a — The worker runtime and the message protocol

Deliverable: any registered model runs in a worker and streams its observables to
the main thread, and `validate()` runs and reports there.

- `src/sandbox/web/bridge.py` — protocol-only adapter: create a run from an
  `Experiment`-shaped dict, advance it by `n` steps, return observables since the
  last call, report terminality, run `validate()`.
- `web/worker.js` — boots Pyodide, installs the wheel, dispatches messages.
- `web/protocol.md` — the message schema, written down rather than implied.
- Deployment: `loadPackage(wheelUrl)` directly. **`micropip` is not shipped in the
  npm distribution** and `loadPackage("micropip")` returns `[]` while reporting
  success, surfacing later as an unrelated-looking `ModuleNotFoundError`.

**Done when:** the ValidationSuite's verdict in the browser matches native for
Wright-Fisher and the repressilator, at the same tolerance; and the main thread
stays live, measured with the MessageChannel instrument from the slice — *not*
with `requestAnimationFrame`, which cannot fail in a background tab.

**The responsiveness criterion needs care, because the obvious number is one
nothing can fail.** The slice's worker arm already measured **1-23 ms** blocked,
and **45 ms** in its worst contended window — so a "never blocked more than 50 ms"
bar is *inside the noise* and would ship green before any 4a code existed, which
is the `assert std > 0.0` failure this project has caught five times. Treat the
slice's band as a **regression guard**, not a pass mark: 4a's blocking must stay
within the measured worker band, established against a **same-session** re-run of
the slice, and any window past ~50 ms is a signal to look rather than evidence of
success. The real pass mark is 4b's on-screen measurement, which is the thing no
CPU timing can stand in for.

### 4b — The drawing path

Deliverable: two renderers on the main thread, both consuming protocol data only.

- **Traces** — observables against time, stochastic replicates overlaid on the
  deterministic limit. Canvas 2D; this is the project's organizing picture.
- **Fields** — `fields()` → RGBA → `putImageData`. Already measured end to end.

**Done when:** a demo runs at an on-screen frame rate that has actually been
measured on a **visible** tab. The slice measured the drawing path's CPU cost but
never exercised compositing, so *"the drawing path costs 0.3 ms"* is established
and *"the animation is smooth"* is not. That probe is written and pending.

**matplotlib is not in the default bundle.** It renders correctly under Pyodide
and costs 0.07-0.16 s per figure once loaded, but it multiplies the gzipped
download by **2.1x** (8.7 → 18.4 MB) because wheels are already-compressed
archives that gzip cannot shrink. It belongs behind an explicit "export a figure"
action that loads it on demand — not on the path a reader takes to see anything.

### 4c — One demo, end to end

The thinnest complete path, in the spirit of Phase 0: **one** model through
worker → protocol → canvas → validation, before a second one is added.

The repressilator is the candidate, because it carries the organizing thread and
already has a native demo to check against. Wright-Fisher is the fallback if the
horizon (below) proves awkward.

**Demos must display progressively, not compute-then-draw.** From the
measurements: Gray-Scott after 400 steps shows only the seeded blob — patterns
need thousands of steps, tens of seconds at 128² — and the repressilator needs
~400 000 events (~15 s) before it oscillates rather than merely rising. Both are
seconds rather than the minutes a convergence sweep costs, so both are viable;
neither is the "few seconds of stepping" the first measurement assumed. The
worker architecture supports streaming natively. A main-thread page could not do
it at all.

## The WebGL trap (decide now, not when it is tempting)

HANDOFF §4 lists shaders as an escape hatch and calls Gray-Scott a natural fit.
It is — and the step is 98% of the frame, so a shader is the *only* thing that
would speed a field demo up meaningfully.

It is also **a second implementation of a validated model's numerics, in a
language with different floating-point behaviour, that no test in this project
can reach.** That is precisely the cost the measurements showed this branch
avoids. If it is ever built it must be **display-only and labelled**, in code and
in UI, exactly as the speculative arc is quarantined under non-negotiable #5 — and
it must never be the thing a validation check runs on. The default answer is
**no**, and this paragraph exists so that answer does not get re-litigated by
whoever next sees a slow field render.

## Risks, and what would falsify the decision

- **The download.** 8.7 MB gzipped and ~2 s to a running interpreter on a warm
  local connection. **Cold-cache and mobile were never measured**, and neither
  number transfers. If a cold first load is unacceptable on a real connection, the
  fork decision is worth revisiting — that is the one measurement that could still
  overturn it, and it should be taken early in 4a rather than discovered in 4c.
- **Machine-dependent timing.** Every absolute figure in the measurements swung up
  to 4x within one session under unrelated load. Any performance claim this phase
  makes needs a same-session baseline, as the suite-timing rule already requires.
- **Instrument failure over model failure.** Three of the four things that went
  wrong in the slice were *measurement* bugs producing plausible numbers, not code
  bugs producing errors. Expect that ratio to hold.

## Out of scope

- Any UI framework, styling system, or build step beyond serving static files.
- Interactive matplotlib backends (they need the main thread, reintroducing
  exactly the blocking 4a exists to remove).
- Running the test suite in a browser.
- numba / JAX. Nothing has been profiled that calls for them, and the step being
  98% of a *browser* frame is not evidence about the native path.
- Anything in `models/ecosystem/`. The quarantine stays empty; that is still the
  correct outcome.
