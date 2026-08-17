# The worker message protocol

Written down rather than implied. Everything numerical happens in Python inside
the worker; the main thread owns the DOM, the canvas and this message loop, and
nothing else.

## Envelope

Every message in either direction is a plain object with an `id` that the reply
echoes back. The main thread keeps a map of `id -> {resolve, reject}` and the
worker never invents an `id` of its own **except** for `progress` events, which
carry the `id` of the request that is producing them.

```
main -> worker   { id, cmd, arg }
worker -> main   { id, payload }                       // success
worker -> main   { id, error, traceback }              // failure
worker -> main   { id, event: "progress", payload }    // an unsolicited update
```

Field pixels do not travel inside `payload`. They ride alongside it as a
`Uint8ClampedArray` in a `rgba` key, listed in the `postMessage` transfer list:

```
worker -> main   { id, payload: {width, height, vmin, vmax, ...}, rgba }
```

That split is a measurement, not a preference. Below ~1 000 floats a transfer
and a structured-clone copy are indistinguishable, so scalars go through JSON
where they are readable; at 8 MB the copy is 24x worse on a quiet machine and
**three orders of magnitude worse on a busy one**, so pixels are transferred.

## Commands

| `cmd` | `arg` | `payload` |
|---|---|---|
| `boot` | `{}` | build report: pyodide/python/numpy versions, wheel sha256 + size, per-stage seconds |
| `describe` | `{}` | `{models: [{name, validatable, terminable, deterministic_limit, fields, params_from_json}], colormaps}` |
| `defaults` | `{model}` | the params dataclass's defaults; `null` marks a field with no default |
| `create` | spec (below) | `{run_id, model, replicates, observable_keys, fields, max_steps, record_every, ...}` |
| `advance` | `{run_id, n_steps}` | `{steps_taken, steps, t, terminal, finished, pending}` |
| `drain` | `{run_id}` | `{replicates: [{from, t, series}]}` — everything recorded since the last drain, once |
| `run` | `{run_id, n_steps, chunk, every_ms, quiet}` | drives `advance`+`drain` in a loop inside the worker, emitting `progress` events; resolves with the final status |
| `cancel` | `{run_id}` | asks a live `run` loop to stop at its next chunk boundary |
| `status` | `{run_id}` | as `create`, plus `recorded` |
| `field` | `{run_id, field, cmap, vmin, vmax, replicate}` | `{width, height, vmin, vmax, t, ...}` + transferred `rgba` |
| `colormap_strip` | `{cmap, width, height}` | a colour ramp as `rgba`, for a scale bar drawn by the same code as the image |
| `limit` | spec with a `limit` block | `{available, t, series}` or `{available: false, reason}` |
| `validate` | spec | `{validatable, passed, reason, checks: [...]}` |
| `fingerprint` | spec | `{digests, recorded, steps}` — sha256 per replicate, **informational** |
| `close` | `{run_id}` | `{closed: true, live: [...]}` |

## The spec

The spec **is** an `Experiment` — the project's existing unit of reproducibility
— as plain JSON:

```json
{
  "model": "repressilator",
  "params": {"alpha": 216.0, "alpha0": 0.216, "n_hill": 2.0,
             "beta": 1.0, "Omega": 50.0, "t_max": 30.0},
  "replicates": 4,
  "seed": 7,
  "max_steps": 500000,
  "record_every": 200,
  "limit": {"t_max": 30.0, "dt": 0.005}
}
```

Two restrictions. A `sweep` is refused: a sweep is a batch job with nothing to
watch, and those stay native. `limit` is not an `Experiment` field — it is the
horizon and step for integrating the deterministic limit, stated explicitly
rather than dug out of the params, because models spell their horizon
differently and a front-end that guessed a param name would work for the models
it was written against and fail silently on the next one.

**The same spec is the same run natively.** The replicate generators are spawned
through the same two-level chain `run_experiment` uses, and the run is driven by
the same `ReplicateRunner`, so a browser run of a given spec is byte-identical to
the native one — not merely similar. `tests/test_web_bridge.py` pins that on the
trajectory *and* on the generator state.

## Why `run` exists when `advance` and `drain` already do

`advance` is the primitive and the main thread could drive it directly, one
message per chunk. `run` does that loop *inside* the worker and posts a
`progress` event every `every_ms` milliseconds instead. The difference is not
throughput — 4 096 boundary crossings measured cheaper than the run-to-run noise
on a single 115 ms workload — it is that a loop driven from the main thread ties
the simulation's pace to the page's event loop, so a busy page slows the
simulation down and a background tab (where timers are clamped to once a second)
nearly stops it. The worker's loop is bounded by its own work.

`chunk` is the number of steps per `advance` call and is a **display** choice:
smaller means smoother progress and more messages. It cannot change the answer,
and — measured — it does not change the speed either: across a 240x range of
chunk sizes the stepping time varies by at most 2%, inside the run-to-run
spread. `quiet` drops the drain and the progress post while keeping the yield;
it exists so that measurement can separate the cost of pausing from the cost of
reporting, and it found both to be nil.

**That is true in a visible tab and false in a hidden one**, which is worth
stating because it cost this phase an afternoon and a wrong conclusion. Run the
same sweep in a background tab and it rises past 3x, monotonically in the number
of yields, looking exactly like a real per-yield cost. It is Chrome
deprioritizing a background renderer: every yield hands the worker back to a
throttled scheduler. The same conditions make the worker look 2-2.8x slower than
the main thread, which it is not (0.89-0.99x when visible).

## Errors

The worker replies with `{id, error, traceback}` and stays alive. A Python
exception is a legitimate answer here — `analytic_predictions` raises rather than
returning a number it does not believe, and a params dataclass raises on an
invalid combination — so the page renders the message rather than treating it as
a crash.

A refusal is not an error. `validate` on a model with no `analytic_predictions`
resolves successfully with `{validatable: false, reason: "..."}`, and `limit` on
a model with no ODE limit resolves with `{available: false, reason: "..."}`.
The distinction matters: the repressilator has no closed-form scalar to check
because its deterministic limit is a limit cycle, and that is a fact about the
model worth displaying, not a failure to handle.

## Deployment notes

- Pyodide is **vendored**, not loaded from a CDN, so the bytes are the ones the
  measurements were taken against and development works offline.
- The project's wheel is installed with `loadPackage(wheelUrl)` directly.
  `micropip` is **not shipped** in the npm distribution and
  `loadPackage("micropip")` returns `[]` while reporting success — the failure
  surfaces later as an unrelated-looking `ModuleNotFoundError`.
- `boot` returns the wheel's sha256 and size, and the page displays them. A
  stale wheel runs old code and passes validation with no symptom at all; the
  serve script rebuilds on every start, and the hash is how you can tell.
