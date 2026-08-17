# The browser-vs-local fork, measured

`HANDOFF.md` §4 says *"decide early"* about the browser fork and the decision was
never made; Phases 0-3 closed without it. This document is the measurement that
has to come before the plan — the same slice-before-plan order every phase in
this project has used. **It commits to nothing.** It replaces one assumption in
HANDOFF with numbers, records where a second platform stops agreeing with the
first, and names the two things a browser build would still have to solve.

Everything here is reproducible from `M:/claud_projects/temp/pyodide-slice`
(`bench.py`, `ufunc_probe.py`, `run_pyodide.mjs`, `interleave.sh`, `index.html`).

## 1. HANDOFF's stated cost for this branch is wrong

> "**Browser** (Pyodide, or a JS/WASM core) — what makes it genuinely *usable by
> students*. Costs a reimplementation of the numerics." — HANDOFF §4

It does not, for **this** codebase. The package's entire dependency surface is
`numpy>=2.0` plus the standard library, with `matplotlib` as an optional extra —
no SciPy, no compiled extension, no C of our own. Pyodide ships NumPy. So the
validated core is already a browser-compatible artifact; there is no second
implementation to write and, more to the point, **no second implementation to
keep correct**, which is the cost that would actually have mattered here.

Demonstrated end to end, in Chrome, with no CDN and no source edits. **The
timings in this block are a single run and are illustrative only** — the
repeated-measurement discipline lives in §2, and the same page read
`1.64 / 0.35 / 0.4629 s` on the run before this one, a 1.5x spread on
`validate()` from two consecutive loads. Take §2's table for speed; take this
block for *what happened*:

```
pyodide runtime booted in 2.00 s
numpy loaded in 0.46 s: ["numpy"]
wheel installed in 0.03 s: ["biological_sandbox"]     <- `uv build` output, 177 KB

Validation of 'wright_fisher': PASS
  [PASS] fixed_A: measured 0.2775 vs predicted 0.3
         (|diff| = 0.0225, 1.00 SE, tol 0.0897, n=400)

repressilator SSA: 20001 recorded events in 0.5872 s
repressilator trajectory fingerprint: d429cbea1e694177
  (native x86 recorded:                d429cbea1e694177)
```

The ValidationSuite — the thing this project defines "done" by — runs in a
browser tab, and the SSA trajectory it runs alongside is **bit-identical to the
native x86 one**. That was not the expected result; see §3. (The `PASS` line and
the fingerprint *are* reproducible run to run; only the seconds move.)

**A dead end worth recording so it is not rediscovered:** `micropip` is *not*
shipped in the `pyodide` npm distribution, and `loadPackage("micropip")` returns
`[]` while reporting success — the failure surfaces later as a
`ModuleNotFoundError` from a line that looks unrelated. `loadPackage(wheelUrl)`
installs a pure-Python wheel directly and removes micropip from the deployment
path altogether. That is a simplification, not a workaround.

## 2. Speed: ~2x native, except BLAS

Medians of **8 interleaved rounds** per configuration, `x` = in-configuration
spread (max/min) over those rounds:

| probe | native 3.11 | native 3.13 | Pyodide (wasm32) | ratio (med / min) |
|---|---|---|---|---|
| pure Python loop | 0.1407 s (x2.01) | 0.1208 s (x1.99) | 0.2545 s (x1.74) | 1.81 / 2.64 |
| 100k scalar NumPy calls | 0.0407 s (x2.30) | 0.0508 s (x2.24) | 0.0817 s (x2.10) | 2.01 / 2.48 |
| 400x400 matmul | 0.0015 s (x1.36) | 0.0016 s (x1.42) | 0.0475 s (x1.30) | **31.95 / 34.72** |
| `eigvals`, S = 200 | 0.0200 s (x1.75) | 0.0191 s (x1.62) | 0.0357 s (x1.71) | 1.79 / 2.00 |
| Wright-Fisher, 20 reps | 0.0141 s (x1.98) | 0.0152 s (x1.87) | 0.0288 s (x1.60) | 2.04 / 2.32 |
| repressilator, 20k events | 0.3697 s (x1.98) | 0.4100 s (x1.89) | 0.7386 s (x1.72) | 2.00 / 2.42 |
| Gray-Scott, 200 steps | 0.1190 s (x1.40) | 0.1186 s (x1.36) | 0.2009 s (x1.47) | 1.69 / 1.89 |
| `validate()`, 400 reps | 0.2836 s (x1.44) | 0.2883 s (x1.35) | 0.5303 s (x1.66) | 1.87 / 2.05 |

Pyodide runs this project's actual workloads at **1.7-2.5x** native — not the
3-10x the folklore quotes. The one catastrophe is **dense BLAS at ~32x**, where
native is multithreaded and vectorized and Pyodide is neither; this project
barely uses `matmul`, and `eigvals` — the LAPACK call it *does* use — is only
2x, because that routine is not the one native BLAS accelerates hardest.

Two estimators are quoted because they disagree: the median is what a user
experiences on a shared machine, the minimum is the machine's capability, since
interference can only slow a run down. **A ratio quoted with only one of them is
not a measurement.**

**The interleaving was not ceremony.** The first sequential pair read native
Python 3.13 as *slower than WebAssembly* on three probes. It is not — over eight
interleaved rounds 3.11 and 3.13 are within a few percent of each other on
everything. And a single native reading of the repressilator probe came in at
`0.2101 s` against the eventual 8-round median of `0.3697 s` — a **1.76x** swing
on one probe from one machine in one session. This project's recorded "bracket
any suite-timing claim by ±30%" is, on this evidence, **too generous**.

## 3. Bit-identity survives much further than expected — and the boundary is measurable

A 21-rung ladder, ordered by how much floating-point machinery each rung touches,
run on native 3.11, native 3.13 and Pyodide. Every rung was **stable within each
configuration across repeated rounds**, so the fingerprints are fingerprints.

**Bit-identical across platforms** — every RNG stream (`PCG64` raw, uniform,
binomial, multinomial, exponential, normal), sum reductions, matrix draws, and
the trajectories of **Wright-Fisher, the repressilator SSA (20 001 events),
Gray-Scott, stochastic gLV, adaptive dynamics, and trait branching**.

**Differs across platforms** — the HH rate functions and hence every
Hodgkin-Huxley trajectory, Daisyworld, `eigvals`, and any log-log slope fit
(`np.log` on an array). The disagreement is roundoff, not a wrong answer:
`max Re eig` reads `14.385169360417766` native against `14.385169360417734` in
WASM, a relative difference of **2.2e-15**.

Native 3.11 and native 3.13 agree on **all 21 rungs**, so this is a wasm-vs-x86
split and not a NumPy-version artifact (2.4.6 vs Pyodide's pinned 2.2.5).

### The rule is not "does it call `np.exp`"

Adaptive dynamics and trait branching both build Gaussian competition kernels and
reproduce bit-for-bit; the HH rate functions do not. So the boundary was measured
directly rather than inferred — per ufunc, per array size (`.` = identical,
`X` = differs):

```
ufunc              1      3      7      8     15     16     31     32     64    161   1501  10000  25921
exp                .      .      .      .      .      .      .      .      .      X      X      X      X
log                .      .      .      .      .      .      .      .      .      .      X      X      X
log1p              X      X      X      X      X      X      X      X      X      X      X      X      X
expm1              X      X      X      X      X      X      X      X      X      X      X      X      X
sqrt               .      .      .      .      .      .      .      .      .      .      .      .      .
cbrt               X      X      X      X      X      X      X      X      X      X      X      X      X
power_0.25         .      .      .      .      .      .      .      .      .      .      X      X      X
power_2.0          .      .      .      .      .      .      .      .      .      .      .      .      .
sin                .      .      X      X      X      X      X      X      X      X      X      X      X
tanh               X      X      X      X      X      X      X      X      X      X      X      X      X
reciprocal         .      .      .      .      .      .      .      .      .      .      .      .      .
divide             .      .      .      .      .      .      .      .      .      .      .      .      .
```

Two mechanisms, cleanly separated:

- **IEEE-754 requires correctly-rounded `+ - * /` and `sqrt`.** Those, and
  integer powers, are identical at every size and always will be. This is why an
  SSA over mass-action propensities and a 5-point Laplacian survive intact — the
  hot loops of this project are mostly IEEE-required arithmetic.
- **Transcendentals are not required to be correctly rounded**, so they differ by
  toolchain. `log1p`, `expm1`, `cbrt` and `tanh` differ at **size 1**: MSVC's
  scalar libm is simply not Emscripten's. `exp`, `log`, `sin` and fractional
  powers agree at small sizes and diverge above a threshold, where NumPy switches
  from scalar libm to its own SIMD kernel — and the x86 and wasm SIMD kernels are
  different code. The threshold is a property of NumPy's *dispatch*, not of the
  source line, which is exactly why "it calls `np.exp`" predicts nothing.

### What a divergence licenses, and what it does not

CLAUDE.md non-negotiable #4 requires bit-identity **for SSA optimizations within
one implementation**, so that recorded slope anchors stay valid. It says nothing
about a second *platform*. A wasm build differing at 2.2e-15 does **not** fail
that rule. The correct posture, stated before the numbers were in and unchanged
by them:

> Recorded fingerprints are **native anchors**. A browser run is validated
> **statistically**, through `analytic_predictions` and the ValidationSuite —
> which is how every model in this project is defined as correct anyway.

That the fingerprints happen to survive for six of the models is a convenience,
not a load-bearing claim, and no future work should be built on it holding.

## 4. Page weight and the one architectural constraint

| file | raw | gzip |
|---|---|---|
| `pyodide.asm.wasm` | 8.25 MB | 2.70 MB |
| `numpy-*.whl` | 2.97 MB | 2.93 MB |
| `python_stdlib.zip` | 2.30 MB | 2.27 MB |
| `pyodide.asm.js` | 1.02 MB | 0.21 MB |
| `biological_sandbox-0.1.0.whl` | 0.17 MB | 0.17 MB |
| **total** | **14.7 MB** | **~8.3 MB** |

~8.3 MB over the wire and ~2 s to a running interpreter on a warm local
connection. The wheels and the stdlib zip are already-compressed archives, so
gzip buys nothing on 5.4 MB of that total.

**The constraint Node could not show.** In a browser, Pyodide runs on the main
thread. Any stepping loop of more than a fraction of a second freezes the tab —
no repaint, no input, and a browser "page unresponsive" prompt past a few
seconds. At the speeds in §2, `validate()` at 400 replicates already takes
0.5-0.7 s and the repressilator probe takes 0.6 s, so **an interactive front-end
needs a Web Worker regardless of how fast the numerics turn out to be.** That is
a design consequence, not a benchmark result, and it is the real work in a
browser phase: not the numerics, the *plumbing* — worker, message protocol, and
a rendering path that consumes `observables()` incrementally.

## 5. What was deliberately not measured

- **matplotlib under Pyodide.** Every figure in this project is matplotlib, and
  none of them were rendered in a browser. A browser phase would likely want a
  different rendering path anyway (see §4), so measuring the old one would price
  a road not taken.
- **The test suite in a browser.** Skipped on purpose: `pytest-xdist` cannot
  spawn processes in WASM, and the multi-minute convergence tests are precisely
  the workloads a browser tab cannot host. The suite stays a native artifact.
- **Cold-cache network time and mobile devices.** The 2 s boot is a warm local
  server on a desktop; neither number transfers.
- **Any front-end.** No UI was designed, and none of the above should be read as
  a plan for one.

## 6. What this settles

The fork is no longer a choice between "reimplement the numerics" and "stay
local". It is a choice between **shipping the existing core to a browser at ~2x
slowdown behind a Web Worker** and **not shipping it**. The heavy convergence
sweeps stay native either way — they are minutes of CPU, and 2x makes that worse,
not viable. What a browser can host is exactly the demo-shaped workload: a few
seconds of stepping, an overlay of stochastic replicates on a deterministic
limit, and a validation check the reader watches pass.

That is the project's central teaching move, and it is the part that currently
requires a Python install to see.
