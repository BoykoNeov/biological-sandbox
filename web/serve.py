"""Stage and serve the browser front-end.

Three jobs, in order:

1. **Rebuild the wheel.** Every start, unconditionally. ``loadPackage`` will
   happily install a build from three commits ago, every check will pass, and
   nothing will say so — a stale wheel is the quietest failure available in this
   phase. The page displays the wheel's sha256 so you can tell.
2. **Stage the Pyodide runtime** into ``web/vendor/pyodide`` if it is not there.
   Vendored rather than loaded from a CDN: the bytes are then the ones the
   measurements were taken against, and development works offline.
3. **Serve the directory**, with gzip (the download is the one risk that could
   still overturn the browser decision, and measuring it uncompressed would
   measure the wrong number) and an optional bandwidth cap.

The bandwidth cap is a real measurement of *this client at a known byte rate*.
It is **not** a real network: there is no latency, no TCP slow start, no packet
loss, no CDN. Time-to-first-frame under ``--bandwidth 5`` is what a 5 Mbit/s
link's throughput term costs; the rest of a real connection is not modelled and
must not be claimed. See ``docs/plans/phase4-cold-load-measurement.md``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
VENDOR = WEB / "vendor" / "pyodide"
DIST = WEB / "dist"
# Measurements posted back by a page. Outside the repo: they are regenerable
# working files, and the numbers that matter get written into docs/plans/.
RESULTS = Path("M:/claud_projects/temp/phase4/results")

PYODIDE_VERSION = "0.28.3"
CDN = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full"

# The minimum set that boots an interpreter and imports numpy. Anything else
# (matplotlib and its ten dependencies) is deliberately absent: it multiplies the
# gzipped download by 2.1x, from 8.7 MB to 18.4 MB, to buy 0.07-0.16 s static
# PNGs — so it belongs behind an explicit action, not on the path a reader takes
# to see anything.
RUNTIME_FILES = (
    "pyodide.js",
    "pyodide.asm.js",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
)

# Content types that are already compressed; gzipping them costs CPU and buys
# nothing (a wheel is a zip archive, and the measurement recorded gzip saving
# ~0 on 5.4 MB of the total).
INCOMPRESSIBLE = (".whl", ".wasm.gz", ".zip", ".png", ".gz", ".woff2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_wheel() -> Path:
    """``uv build`` into ``web/dist``, replacing whatever was there."""
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    print("building the wheel ...", flush=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(DIST)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    wheels = sorted(DIST.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {DIST}, found {[w.name for w in wheels]}")
    wheel = wheels[0]
    print(f"  {wheel.name}  {wheel.stat().st_size / 1024:.0f} KB  sha256 {sha256(wheel)[:16]}...")
    return wheel


def _numpy_wheel_name(lock: Path) -> str:
    return _wheel_closure(lock, ["numpy"])["numpy"]


def _wheel_closure(lock: Path, roots: list[str]) -> dict[str, str]:
    """``{package: wheel file name}`` for ``roots`` and everything they depend on.

    Resolved transitively from the lock rather than hand-typed. matplotlib pulls
    in eleven other packages here, and a hand-written list is a list that goes
    stale against the next Pyodide release without anything saying so — the wheel
    simply fails to import at the moment somebody presses the button.
    """
    data = json.loads(lock.read_text(encoding="utf-8"))
    packages = data.get("packages", data)
    resolved: dict[str, str] = {}
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in resolved:
            continue
        entry = packages.get(name)
        if entry is None:
            raise SystemExit(f"no {name!r} entry in {lock}")
        resolved[name] = entry["file_name"]
        stack.extend(entry.get("depends", []))
    return resolved


def stage_figure_packages(source: Path | None, download: bool) -> None:
    """Stage matplotlib and its dependencies, for the on-demand figure export.

    **Its own presence check, deliberately.** ``stage_runtime`` returns as soon as
    the wasm, the lock and a numpy wheel are there; folding matplotlib into that
    test would mean an already-staged vendor directory silently skips it, and the
    failure then surfaces as a confusing error at the moment a reader clicks
    "export".

    These bytes are staged but **not fetched by the page unless asked for**. That
    is the whole point of the deferral: matplotlib multiplies the gzipped download
    2.1x, from 8.7 MB to 18.4 MB, to buy static PNGs — so it belongs behind an
    explicit action rather than on the path a reader takes to see anything.
    """
    lock = VENDOR / "pyodide-lock.json"
    if not lock.exists():
        raise SystemExit(f"stage the runtime before the figure packages: {lock} is missing")
    wanted = _wheel_closure(lock, ["matplotlib"])
    missing = {n: f for n, f in wanted.items() if not (VENDOR / f).exists()}
    if not missing:
        print(f"figure packages already staged ({len(wanted)} wheels)")
        return

    print(f"staging matplotlib and its dependencies ({len(missing)} wheels) ...")
    for name, file_name in sorted(missing.items()):
        if source is not None:
            src = source / file_name
            if not src.exists():
                raise SystemExit(f"{src} is missing; is that a full Pyodide distribution?")
            shutil.copy2(src, VENDOR / file_name)
            print(f"  {name}: {file_name}")
        elif download:
            _fetch(f"{CDN}/{file_name}", VENDOR / file_name)
        else:
            raise SystemExit(
                f"{file_name} is not in {VENDOR}. Pass --pyodide-src <dir> or --download."
            )
    staged = sum((VENDOR / f).stat().st_size for f in wanted.values())
    print(f"  {len(wanted)} wheels, {staged / 1e6:.2f} MB on disk (not fetched unless asked for)")


def stage_runtime(source: Path | None, download: bool) -> None:
    """Put the Pyodide runtime under ``web/vendor/pyodide`` if it is not already."""
    if (VENDOR / "pyodide.asm.wasm").exists() and (VENDOR / "pyodide-lock.json").exists():
        wheels = list(VENDOR.glob("numpy-*.whl"))
        if wheels:
            return
    VENDOR.mkdir(parents=True, exist_ok=True)

    if source is not None:
        print(f"staging the Pyodide runtime from {source}")
        for name in RUNTIME_FILES:
            src = source / name
            if not src.exists():
                raise SystemExit(f"{src} is missing; is that a Pyodide distribution?")
            shutil.copy2(src, VENDOR / name)
        numpy_name = _numpy_wheel_name(VENDOR / "pyodide-lock.json")
        shutil.copy2(source / numpy_name, VENDOR / numpy_name)
    elif download:
        print(f"downloading the Pyodide runtime from {CDN}")
        for name in RUNTIME_FILES:
            _fetch(f"{CDN}/{name}", VENDOR / name)
        numpy_name = _numpy_wheel_name(VENDOR / "pyodide-lock.json")
        _fetch(f"{CDN}/{numpy_name}", VENDOR / numpy_name)
    else:
        raise SystemExit(
            f"no Pyodide runtime in {VENDOR}. Pass --pyodide-src <dir> to copy one "
            "from a local pyodide distribution, or --download to fetch it."
        )

    total = sum(f.stat().st_size for f in VENDOR.iterdir() if f.is_file())
    print(f"  staged {total / 1e6:.1f} MB into {VENDOR}")


def _fetch(url: str, target: Path) -> None:
    print(f"  {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f" {target.stat().st_size / 1e6:.2f} MB")


def write_expectations() -> None:
    """Compute the native answer to every conformance spec, for the page to check against.

    Computed here rather than pasted into the page because a recorded number is
    only comparable when it comes from the same commit on the same machine — the
    project has been burned three times comparing timings across sessions, and a
    validation mean is no more portable than a timing. This runs the *native*
    interpreter that is serving the files, so a drift between the two sides is a
    real disagreement rather than a stale constant.
    """
    from sandbox.web import bridge  # imported late: staging must work without it

    conformance = json.loads((WEB / "conformance.json").read_text(encoding="utf-8"))
    expected: dict[str, object] = {}
    print("computing the native expectations ...", flush=True)
    for entry in conformance["specs"]:
        spec = entry["spec"]
        started = time.perf_counter()
        expected[entry["id"]] = {
            "validate": bridge.validate_spec(spec),
            "fingerprint": bridge.fingerprint_spec(spec),
        }
        print(f"  {entry['id']}: {time.perf_counter() - started:.2f} s")

    import numpy

    import sandbox

    (WEB / "expected.json").write_text(
        json.dumps(
            {
                "generated_by": "web/serve.py",
                "python": sys.version.split()[0],
                "numpy": numpy.__version__,
                "sandbox": sandbox.__version__,
                "expected": expected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def measure_presets() -> None:
    """Run every preset in ``web/presets.json`` to termination and price it.

    The model picker lets a reader run any of the fourteen registered models, and
    each needs a step budget. There is no shared constant to derive one from: the
    demo page's ``470 events per time unit per unit of Omega`` is a *repressilator*
    measurement, and reusing it elsewhere would be this project's recorded mistake
    of carrying a number away from the estimator that produced it.

    So the budget is measured, once, here — and the failure it prevents is the
    specific one the demo page was caught by: a run that stops early leaves the
    deterministic limit alone on the right of the frame, which reads as perfect
    agreement rather than as absence.

    This writes nothing. It prints, and the numbers are pasted into
    ``presets.json`` with the date, because a measurement that silently rewrites
    its own input has no record of having changed.
    """
    from sandbox.web import bridge

    presets = json.loads((WEB / "presets.json").read_text(encoding="utf-8"))["presets"]
    cap = 4_000_000
    print(f"pricing {len(presets)} presets (cap {cap:,} steps each) ...\n", flush=True)
    print(f"  {'model':22s} {'steps':>10s} {'recorded':>9s} {'seconds':>8s}  terminated")
    for name, preset in presets.items():
        spec = {
            "model": name,
            "params": preset["params"],
            "replicates": 1,
            "seed": preset.get("seed", 0),
            "max_steps": cap,
            "record_every": 1,
        }
        started = time.perf_counter()
        try:
            run = bridge.Run("price", bridge.experiment_from_spec(spec))
            while not run.finished:
                run.advance(20000)
            status = run.status()
            steps = status["steps"][0]
            elapsed = time.perf_counter() - started
            print(
                f"  {name:22s} {steps:>10,} {status['recorded'][0]:>9,} "
                f"{elapsed:>8.2f}  {status['terminal'][0]}"
            )
        except Exception as exc:  # a preset that cannot run is the point of running this
            print(f"  {name:22s} {'FAILED':>10s} {'':>9s} {'':>8s}  {type(exc).__name__}: {exc}")


class Handler(SimpleHTTPRequestHandler):
    """Static files, gzipped where it helps, optionally rate limited."""

    bandwidth_bytes_per_s: float | None = None
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        sys.stderr.write(f"  {self.path.split('?')[0]}\n")

    def end_headers(self) -> None:
        # No caching anywhere: a cold load is the measurement, and a warm cache
        # would silently turn it into a different one.
        self.send_header("Cache-Control", "no-store")
        # Cross-origin isolation, which buys a finer performance.now(). Without
        # it the clock is coarsened to 0.1 ms and every single-shot
        # sub-millisecond reading is a rounding artifact rather than a
        # measurement -- the slice recorded a whole column of exact multiples of
        # the tick, including two zeroes.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_POST(self) -> None:
        """Accept a page's measurements at ``/results/<name>``.

        A measurement page that can only be read through the devtools protocol
        cannot be run in a *visible* window, and the plan's one open item — an
        on-screen frame rate — is exactly the measurement a hidden tab cannot
        take (``requestAnimationFrame`` is suspended, not slow). So a real
        browser window is opened on the page and the page posts its results
        back here. Local, unauthenticated, and bound to 127.0.0.1: this is a
        measurement rig, not a service.
        """
        if not self.path.startswith("/results/"):
            self.send_error(404)
            return
        name = Path(self.path[len("/results/") :]).name or "results"
        if not name.endswith(".json"):
            name += ".json"
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        RESULTS.mkdir(parents=True, exist_ok=True)
        target = RESULTS / name
        target.write_bytes(body)
        print(f"  <- results/{name}  ({len(body)} bytes)", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()

        raw = path.read_bytes()
        encoding = None
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "")
        if accepts_gzip and len(raw) > 1024 and not path.name.endswith(INCOMPRESSIBLE):
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as gz:
                gz.write(raw)
            compressed = buffer.getvalue()
            if len(compressed) < len(raw):
                raw, encoding = compressed, "gzip"

        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(len(raw)))
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.end_headers()
        return io.BytesIO(raw)

    def copyfile(self, source, outputfile) -> None:
        limit = type(self).bandwidth_bytes_per_s
        if not limit:
            shutil.copyfileobj(source, outputfile)
            return
        # Paced in 16 KB slices. This models the *throughput* term of a slow link
        # and nothing else -- no latency, no slow start, no loss.
        slice_bytes = 16 * 1024
        started = time.perf_counter()
        sent = 0
        while chunk := source.read(slice_bytes):
            outputfile.write(chunk)
            sent += len(chunk)
            behind = sent / limit - (time.perf_counter() - started)
            if behind > 0:
                time.sleep(behind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        metavar="MBIT",
        help="cap the download at this many Mbit/s (throughput only; not a real link)",
    )
    parser.add_argument(
        "--pyodide-src",
        type=Path,
        default=None,
        help="directory holding a Pyodide distribution to copy from",
    )
    parser.add_argument("--download", action="store_true", help="fetch the runtime from the CDN")
    parser.add_argument(
        "--with-figures",
        action="store_true",
        help=(
            "also stage matplotlib for the on-demand figure export. Staged, not "
            "bundled: nothing fetches these until a reader presses the button"
        ),
    )
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding the wheel")
    parser.add_argument(
        "--measure-presets",
        action="store_true",
        help="price every model-picker preset to termination and exit (does not serve)",
    )
    args = parser.parse_args()

    if args.measure_presets:
        measure_presets()
        return

    stage_runtime(args.pyodide_src, args.download)
    if args.with_figures:
        stage_figure_packages(args.pyodide_src, args.download)
    if args.no_build:
        wheels = sorted(DIST.glob("*.whl"))
        print(f"reusing {wheels[0].name if wheels else '(no wheel!)'}")
    else:
        build_wheel()
    write_expectations()

    if args.bandwidth:
        Handler.bandwidth_bytes_per_s = args.bandwidth * 1e6 / 8.0
        print(f"bandwidth capped at {args.bandwidth} Mbit/s (throughput term only)")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nserving {WEB} at http://{args.host}:{args.port}/  (ctrl-c to stop)\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
