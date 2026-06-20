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
  Hodgkin-Huxley, Gray-Scott. Each has a checkable claim. "Correct" is testable.
- **Generative systems — the speculative arc.** Ecosystem / biosphere
  simulation. No ground truth. Legitimate *exploratory artificial life* (lineage:
  Tierra, Avida, Polyworld) — but **labelled as exploration, never prediction.**
  This code lives quarantined under `models/ecosystem/`.

See [`HANDOFF.md`](HANDOFF.md) for the full vision, the build order, and the
honest treatment of the speculative arc.

## Status

**Phase 0 — Wright-Fisher vertical slice.** The thinnest complete path through
the whole architecture (`protocol → model → recorder → sweep → validation`),
with the neutral fixation-probability check passing over many seeded replicates.

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
  core/        protocol, rng, recorder, sweep, validation, registry
  models/      wright_fisher  (+ ecosystem/ — speculative, quarantined)
  viz/         pluggable visualization backends (matplotlib for now)
  demos/       runnable end-to-end examples
tests/         wraps the ValidationSuite into CI
docs/plans/    per-phase dev docs (plan / context / tasks)
```

## License

[MIT](LICENSE).
