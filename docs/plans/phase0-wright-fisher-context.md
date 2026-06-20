# Phase 0 — context (key files & decisions for a fresh session)

## Where things live

| Concern | File |
|---|---|
| The protocol (spine) | `src/sandbox/core/protocol.py` |
| RNG / reproducibility | `src/sandbox/core/rng.py` |
| Trajectory capture + single run | `src/sandbox/core/recorder.py` |
| Sweep / replicate runner | `src/sandbox/core/sweep.py` |
| Validation (credibility) | `src/sandbox/core/validation.py` |
| Name → model registry | `src/sandbox/core/registry.py` |
| Wright-Fisher model | `src/sandbox/models/wright_fisher.py` |
| Plotting | `src/sandbox/viz/backends/matplotlib_backend.py` |
| End-to-end demo | `src/sandbox/demos/wright_fisher.py` |
| Tests | `tests/` |

## Contracts a new model must honor

- Implement `initial_state(params, rng) -> State`, `step(state, rng) -> State`,
  `observables(state) -> dict[str, float]`.
- Embed params in the returned `State`; keep the model object stateless.
- Set `state.t` to the model's own notion of time inside `step`.
- For a core model: implement `analytic_predictions(params) -> dict[str, float]`,
  where each key names an observable whose replicate-mean *final* value the
  ValidationSuite checks against the predicted value.
- Register the model in `src/sandbox/models/__init__.py`.

## The validation convention (so you don't reinvent it)

`validate(experiment, params_factory, z=4.0)`:
- runs the experiment (single param point — no sweep),
- for each predicted key, measures the mean over replicates of that observable's
  final value, and the standard error of that mean,
- passes when `|measured - predicted| <= z * SE`.

`params_factory` (e.g. `lambda d: WFParams(**d)`) is passed in by the caller so
the core never imports a concrete params type.

## Environment notes

- `uv` manages the venv; it resolved CPython 3.11.x, numpy 2.x, matplotlib 3.x.
- Windows console is cp1252 — **avoid non-ASCII in printed output** (a `Δ` in a
  printed string crashed the demo; use `diff`/ASCII). Source files are UTF-8 and
  may contain Unicode in comments/docstrings, just not in `print`ed strings.
- numba/JAX wheels may lag new CPython (e.g. 3.14); irrelevant while NumPy-only,
  relevant at Phase 2.

## Gotchas

- `Experiment.observables` is a tuple (so equality / JSON round-trip is stable).
- `validate()` rejects a sweep — predictions are defined per single param point.
- Reproducibility tests assume `spawn_rngs`; don't switch to `seed + i`.
