# Biological Sandbox

[![CI](https://github.com/BoykoNeov/biological-sandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/BoykoNeov/biological-sandbox/actions/workflows/ci.yml)

> **Stochastic dynamics and the deterministic limits they collapse into.**

A sandbox for simulating biological systems *and checking the simulation against
what the math says should happen.* Every model in the validated core has a known
analytic result — a fixation probability, a pattern wavelength, an f–I curve —
and the project's central move is to show the same phenomenon **stochastically
and deterministically, then watch them converge as the system grows large.**

Validation is not a testing afterthought here. It is the pedagogy, and it is
what makes the simulations trustworthy: each core model is wrong until its
`analytic_predictions` are reproduced by the `ValidationSuite` within a
statistically-derived tolerance.

## Two kinds of software (kept apart on purpose)

- **Verifiable models — the core.** Wright-Fisher, Gillespie/repressilator,
  Hodgkin-Huxley, Gray-Scott, Schnakenberg, generalized Lotka-Volterra,
  Daisyworld, adaptive dynamics, and seven more. Each has a checkable claim.
  "Correct" is testable.
- **Generative systems — the speculative arc.** Ecosystem / biosphere
  simulation. No ground truth. Legitimate *exploratory artificial life* (lineage:
  Tierra, Avida, Polyworld) — but **labelled as exploration, never prediction.**
  This code lives quarantined under `models/ecosystem/`.

See [`HANDOFF.md`](HANDOFF.md) for the full vision, the build order, and the
honest treatment of the speculative arc.

## Status

**Every phase is complete — 0 through 4, plus Phase 2c.** The validated core is
**15 registered models** plus `core/random_matrix.py`, and the suite is
**929 tests passing, 12 skipped** (run 2026-08-17). No wall-clock is quoted here
on purpose: this project has repeatedly found that a suite total measures the
machine and the worker packing rather than the suite, so a bare number in a
README would invite exactly the comparison it cannot support.

| phase | what landed |
|---|---|
| 0 | Wright-Fisher, and the vertical slice through the whole architecture |
| 1 | RK4 + the deterministic-limit protocol, the Gillespie engine, `birth_death` and `isomerization` (exact closed forms), the log-log convergence track, and the repressilator |
| 2 | Hodgkin-Huxley (deterministic, voltage-clamp, and channel-noise) and Gray-Scott |
| 3 | generalized Lotka-Volterra, the May / Allesina-Tang random community matrix, stochastic gLV, Daisyworld, adaptive dynamics, and trait branching |
| 4 | the browser front-end: a worker runtime and message protocol, two renderers, a model picker, an on-demand figure export, and one demo end to end |
| 2c | Schnakenberg, and the one claim Phase 2 explicitly deferred — *which* wavelength emerges when nothing is seeded. Built last, after Phase 4, because Phase 2's stated reason for deferring it was half wrong |

`models/ecosystem/` — the speculative quarantine — is still **empty**, and that
is the intended outcome, not an omission: nothing so far required giving up a
checkable prediction.

The last thing built was **Phase 2c**, and it changed what a validation can be.
Seed no pattern at all and the wavenumber that emerges is *quantized* by the box,
so its ensemble mean matches no continuous prediction exactly — and the deviation
**grows** with replicate count, because the mean stands still while the error bar
shrinks under it. So `analytic_predictions` **refuses** the question (this
project's first refusal on statistical rather than algebraic grounds) and a third
validation track, `core/selection.py`, asserts that the prediction beats each
**named alternative** by a margin. Margins widen with more data where an equality
check would eventually break — that asymmetry is the point. See
[`docs/plans/phase2c-plan.md`](docs/plans/phase2c-plan.md).

The `HANDOFF.md` §4 browser-vs-local fork was the project's long-open decision.
It was **measured** rather than argued, and then built. The core runs unmodified
under Pyodide at about **2x** native, so there is no second implementation of the
numerics to keep correct; the real cost was a Web Worker, without which a run
blocks the page for its entire duration with the event loop getting *zero* turns.

Run it:

```bash
uv run python web/serve.py --download     # stage Pyodide, build the wheel, serve
# then open http://127.0.0.1:8765/
```

- `index.html` — the demo: stochastic replicates over their deterministic limit,
  with the system size as a control, and the ValidationSuite running live.
- `models.html` — the model picker: every registered model, driven entirely by
  `describe` and `default_params`, on budgets measured by `--measure-presets`.
- `check.html` — does the browser's verdict match native, and does the page stay
  alive while it runs.
- `draw.html`, `coldload.html` — what drawing costs, and what the first load does.

**The demo is a demonstration, not a laboratory.** The convergence sweeps, the
test suite and every figure in the repository stay native; nothing shown in a
browser is asserted anywhere. See
[`docs/plans/phase4-tasks.md`](docs/plans/phase4-tasks.md) for the measurements
— including the two that came out reproducibly *wrong* until they were re-taken
in a browser window that was actually on screen.

## Quickstart

```bash
uv sync --extra viz     # create .venv and install deps + dev tools + matplotlib
uv run pytest           # run the validation suite (this is the real check)
uv run ruff check .     # lint
```

Use `--extra viz`, and **read the skip count, not just the failures.** Without
matplotlib an entire test module skips as *one quiet line* — ten tests missing,
nothing failing, and a green run that is a version of itself you did not intend.

Run the Wright-Fisher fixation demo:

```bash
uv run python -m sandbox.demos.wright_fisher
```

## The Model protocol (the spine)

Every model implements the same small interface; shared services are built *on
top of* it and never reach inside a model's state:

```python
class Model(Protocol):
    def initial_state(self, params, rng) -> State: ...   # embeds params into State
    def step(self, state, rng) -> State: ...             # model owns its own Δt; writes state.t
    def observables(self, state) -> dict[str, float]: ...
    # optional — the validation contract, required for any core model:
    def analytic_predictions(self, params) -> dict[str, float]: ...
```

`step` takes no `dt`: a Wright-Fisher step is one generation, a Gillespie step
advances by a *sampled* waiting time only the model knows, a PDE step advances
by a numerical increment that is a *parameter*. The model owns the increment and
writes the new time into `state.t`; consumers read it and cope with irregular
sampling.

## Layout

```
src/sandbox/
  core/        protocol, rng, recorder, sweep, registry, ode, laplacian,
               random_matrix, and the three validation tracks —
               validation (a mean against a scalar), convergence (a power
               law's exponent), selection (a discrimination between named
               hypotheses)
  models/      the 15 validated models  (+ ecosystem/ — speculative, quarantined)
  viz/         visualization backends (matplotlib is currently the only one)
  web/         the browser front-end's Python side (bridge, colormap)
  demos/       runnable end-to-end examples
tests/         wraps the ValidationSuite into CI
docs/plans/    per-phase dev docs (plan / context / tasks)
```

## License

[MIT](LICENSE).
