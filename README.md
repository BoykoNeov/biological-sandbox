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
  Hodgkin-Huxley, Gray-Scott, generalized Lotka-Volterra, Daisyworld, adaptive
  dynamics, and eight more. Each has a checkable claim. "Correct" is testable.
- **Generative systems — the speculative arc.** Ecosystem / biosphere
  simulation. No ground truth. Legitimate *exploratory artificial life* (lineage:
  Tierra, Avida, Polyworld) — but **labelled as exploration, never prediction.**
  This code lives quarantined under `models/ecosystem/`.

See [`HANDOFF.md`](HANDOFF.md) for the full vision, the build order, and the
honest treatment of the speculative arc.

## Status

**Phases 0-3 are complete.** The validated core is **14 registered models** plus
`core/random_matrix.py`, and the suite is **720 tests, all passing**
(`373.81 s` at `-n 6`, re-run 2026-08-17):

| phase | what landed |
|---|---|
| 0 | Wright-Fisher, and the vertical slice through the whole architecture |
| 1 | RK4 + the deterministic-limit protocol, the Gillespie engine, `birth_death` and `isomerization` (exact closed forms), the log-log convergence track, and the repressilator |
| 2 | Hodgkin-Huxley (deterministic, voltage-clamp, and channel-noise) and Gray-Scott |
| 3 | generalized Lotka-Volterra, the May / Allesina-Tang random community matrix, stochastic gLV, Daisyworld, adaptive dynamics, and trait branching |
| 4 | the browser front-end: a worker runtime and message protocol, two renderers, and one demo end to end |

`models/ecosystem/` — the speculative quarantine — is still **empty**, and that
is the intended outcome, not an omission: nothing so far required giving up a
checkable prediction.

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
uv sync                 # create .venv and install deps + dev tools
uv run pytest           # run the validation suite (this is the real check)
uv run ruff check .     # lint
```

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
  core/        protocol, rng, recorder, sweep, validation, registry,
               ode, convergence, laplacian, random_matrix
  models/      the 14 validated models  (+ ecosystem/ — speculative, quarantined)
  viz/         visualization backends (matplotlib is currently the only one)
  demos/       runnable end-to-end examples
tests/         wraps the ValidationSuite into CI
docs/plans/    per-phase dev docs (plan / context / tasks)
```

## License

[MIT](LICENSE).
