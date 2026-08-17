// The worker: Pyodide, the project's wheel, and a message loop. No numerics.
//
// Everything that can block runs in here. That is not a style preference -- it
// is the measured difference between a page that responds and one that does not:
// the same 60 000-event SSA blocks the main thread for 1 750 ms with the event
// loop getting ZERO turns, and blocks it for 10-23 ms from a worker, at
// identical Python speed (0.5823 s vs 0.5826 s). See
// docs/plans/phase4-worker-and-rendering-measurement.md.
//
// The message schema lives in web/protocol.md.

importScripts("./vendor/pyodide/pyodide.js");

let py = null;
let api = null; // the sandbox.web.bridge module, as a PyProxy
let build = null;

const now = () => performance.now();

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function boot(arg) {
  const wheelUrl = (arg && arg.wheel) || "./dist/biological_sandbox-0.1.0-py3-none-any.whl";
  const timings = {};
  let t0 = now();

  py = await loadPyodide({ indexURL: "./vendor/pyodide/" });
  timings.runtime_s = (now() - t0) / 1000;

  t0 = now();
  await py.loadPackage("numpy");
  timings.numpy_s = (now() - t0) / 1000;

  // A stale wheel is the quietest failure available here: loadPackage will
  // happily install a build from three commits ago, every check will pass, and
  // nothing anywhere will say so. The hash is reported so the page can show it.
  const wheelBytes = await (await fetch(wheelUrl, { cache: "no-store" })).arrayBuffer();
  const wheelSha = await sha256Hex(wheelBytes);

  t0 = now();
  await py.loadPackage(wheelUrl);
  timings.wheel_s = (now() - t0) / 1000;

  t0 = now();
  // `micropip` is deliberately absent from this path: it is not shipped in the
  // npm distribution, and loadPackage("micropip") returns [] while reporting
  // success -- surfacing much later as an unrelated-looking ModuleNotFoundError.
  api = py.pyimport("sandbox.web.bridge");
  timings.import_s = (now() - t0) / 1000;

  const versions = JSON.parse(
    py.runPython(`
import json, sys, numpy, sandbox
json.dumps({
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "sandbox": sandbox.__version__,
})
`)
  );

  timings.total_s = Object.values(timings).reduce((a, b) => a + b, 0);
  build = {
    ...versions,
    pyodide: py.version,
    wheel_url: wheelUrl,
    wheel_sha256: wheelSha,
    wheel_bytes: wheelBytes.byteLength,
    timings,
  };
  return build;
}

// --------------------------------------------------------------------------
// Calling Python
//
// Every bridge entry point returns a plain dict; it crosses as a JSON string so
// that no PyProxy escapes into the JS side to be leaked or destroyed twice. The
// crossing itself is free -- 4 096 calls measured below the run-to-run noise of
// a single 115 ms workload -- so the readable option costs nothing.
// --------------------------------------------------------------------------

function callJson(name, ...args) {
  if (!api) throw new Error("worker not booted: send {cmd: 'boot'} first");
  const fn = api.get(name);
  try {
    const result = fn(...args);
    if (typeof result === "string") return JSON.parse(result);
    const value = result.toJs({ dict_converter: Object.fromEntries });
    if (result.destroy) result.destroy();
    return value;
  } finally {
    if (fn && fn.destroy) fn.destroy();
  }
}

// The bridge speaks dicts; JSON is the wire format. A tiny Python shim keeps the
// json.dumps on the Python side, where allow_nan=False can reject a NaN before
// it becomes an unparseable token on the wire.
function setupShim() {
  py.runPython(`
import json
from sandbox.web import bridge

def _dumps(payload):
    # allow_nan=False on purpose: Python emits a bare NaN token by default,
    # which is not JSON and which JSON.parse rejects. The bridge already maps
    # non-finite values to None; this is the guard that says so out loud.
    return json.dumps(payload, allow_nan=False)

def web_describe():
    return _dumps(bridge.describe_models())

def web_defaults(model):
    return _dumps(bridge.default_params(model))

def web_create(spec_json):
    return _dumps(bridge.SESSION.create(json.loads(spec_json)))

def web_advance(run_id, n_steps):
    return _dumps(bridge.SESSION.get(run_id).advance(int(n_steps)))

def web_drain(run_id):
    return _dumps(bridge.SESSION.get(run_id).drain())

def web_status(run_id):
    return _dumps(bridge.SESSION.get(run_id).status())

def web_close(run_id):
    return _dumps(bridge.SESSION.close(run_id))

def web_limit(spec_json):
    return _dumps(bridge.deterministic_limit(json.loads(spec_json)))

def web_validate(spec_json):
    return _dumps(bridge.validate_spec(json.loads(spec_json)))

def web_fingerprint(spec_json):
    return _dumps(bridge.fingerprint_spec(json.loads(spec_json)))

def web_field(run_id, field, cmap, vmin, vmax, replicate):
    run = bridge.SESSION.get(run_id)
    rgba, meta = run.field_rgba(
        field,
        replicate=int(replicate),
        cmap=cmap,
        vmin=None if vmin is None else float(vmin),
        vmax=None if vmax is None else float(vmax),
    )
    # The pixels are returned separately from their metadata: the JS side pulls
    # them straight out of the WASM heap and transfers the buffer.
    globals()["_field_rgba"] = rgba
    return _dumps(meta)

def web_colormap_strip(name, width, height):
    rgba, meta = bridge.colormap_strip(name, int(width), int(height))
    globals()["_field_rgba"] = rgba
    return _dumps(meta)

def web_field_bytes():
    return globals()["_field_rgba"]
`);
}

function pyCall(name, ...args) {
  const fn = py.globals.get(name);
  if (!fn) throw new Error("missing python entry point: " + name);
  try {
    return JSON.parse(fn(...args));
  } finally {
    fn.destroy();
  }
}

// Pull the last field's RGBA out of the WASM heap as an owned, transferable
// buffer. `toJs()` on a uint8 ndarray returns a typed array too -- the slice
// measured the two as equal because they are one operation spelled twice -- so
// getBuffer is used for the explicit release, not for speed.
function grabFieldBytes() {
  const fn = py.globals.get("web_field_bytes");
  const proxy = fn();
  const view = proxy.getBuffer("u8clamped");
  const owned = view.data.slice();
  view.release();
  proxy.destroy();
  fn.destroy();
  return owned;
}

// --------------------------------------------------------------------------
// The stepping loop, driven from in here
//
// The main thread could drive `advance` one message at a time, and the crossing
// would be free. What would not be free is the pacing: a loop driven from the
// main thread runs at the page's event-loop rate, so a busy page slows the
// simulation and a background tab (timers clamped to ~1 Hz) nearly stops it.
// --------------------------------------------------------------------------

const running = new Map(); // run_id -> {cancelled}

async function runLoop(id, arg) {
  const { run_id, chunk = 2000, every_ms = 60, quiet = false } = arg;
  const budget = arg.n_steps == null ? Infinity : Number(arg.n_steps);
  const token = { cancelled: false };
  running.set(run_id, token);

  let done = 0;
  let lastPost = 0;
  try {
    for (;;) {
      const step = pyCall("web_advance", run_id, Math.min(chunk, budget - done));
      done += Math.max(...step.steps_taken, 0);
      const finished = step.finished || done >= budget || step.steps_taken.every((n) => n === 0);

      // `quiet` keeps the yield but drops the drain and the progress post. It
      // exists so a measurement can tell the cost of PAUSING from the cost of
      // REPORTING -- the two are bundled in every realistic loop and only one of
      // them is avoidable.
      if (!quiet && (finished || now() - lastPost >= every_ms)) {
        lastPost = now();
        self.postMessage({
          id,
          event: "progress",
          payload: { status: step, data: pyCall("web_drain", run_id), done, finished },
        });
      }
      if (finished || token.cancelled) break;
      // Yield to this worker's own event loop so a `cancel` message can land.
      // Without this the loop is a synchronous block and cancel never arrives --
      // the same failure the whole worker exists to avoid, one level down.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  } finally {
    running.delete(run_id);
  }
  return { ...pyCall("web_status", run_id), cancelled: token.cancelled, stepped: done };
}

// --------------------------------------------------------------------------
// Dispatch
// --------------------------------------------------------------------------

self.onmessage = async (event) => {
  const { id, cmd, arg = {} } = event.data;
  try {
    let payload;
    let transfer = [];

    switch (cmd) {
      case "boot":
        payload = await boot(arg);
        setupShim();
        break;
      case "build":
        payload = build;
        break;
      case "describe":
        payload = pyCall("web_describe");
        break;
      case "defaults":
        payload = pyCall("web_defaults", arg.model);
        break;
      case "create":
        payload = pyCall("web_create", JSON.stringify(arg));
        break;
      case "advance":
        payload = pyCall("web_advance", arg.run_id, arg.n_steps);
        break;
      case "drain":
        payload = pyCall("web_drain", arg.run_id);
        break;
      case "status":
        payload = pyCall("web_status", arg.run_id);
        break;
      case "close":
        payload = pyCall("web_close", arg.run_id);
        break;
      case "limit":
        payload = pyCall("web_limit", JSON.stringify(arg));
        break;
      case "validate":
        payload = pyCall("web_validate", JSON.stringify(arg));
        break;
      case "fingerprint":
        payload = pyCall("web_fingerprint", JSON.stringify(arg));
        break;
      case "field": {
        payload = pyCall(
          "web_field",
          arg.run_id,
          arg.field,
          arg.cmap || "ember",
          arg.vmin == null ? null : arg.vmin,
          arg.vmax == null ? null : arg.vmax,
          arg.replicate || 0
        );
        const rgba = grabFieldBytes();
        self.postMessage({ id, payload, rgba }, [rgba.buffer]);
        return;
      }
      case "colormap_strip": {
        payload = pyCall("web_colormap_strip", arg.cmap || "ember", arg.width || 256, arg.height || 1);
        const strip = grabFieldBytes();
        self.postMessage({ id, payload, rgba: strip }, [strip.buffer]);
        return;
      }
      case "run":
        payload = await runLoop(id, arg);
        break;
      case "cancel": {
        const token = running.get(arg.run_id);
        if (token) token.cancelled = true;
        payload = { run_id: arg.run_id, cancelling: Boolean(token) };
        break;
      }
      default:
        throw new Error("unknown cmd: " + cmd);
    }

    self.postMessage({ id, payload }, transfer);
  } catch (error) {
    // A Python exception is a legitimate answer here -- analytic_predictions
    // raises rather than returning a number it does not believe -- so the worker
    // reports it and stays alive rather than dying.
    self.postMessage({
      id,
      error: String((error && error.message) || error),
      traceback: String((error && error.stack) || ""),
    });
  }
};
