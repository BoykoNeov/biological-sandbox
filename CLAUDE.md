# CLAUDE.md — Biological Sandbox

Stochastic biological models and the deterministic limits they collapse into,
built **validation-first**. Read [`HANDOFF.md`](HANDOFF.md) for the full vision
and build order; this file is the working-session quick reference.

## Commands

```bash
uv sync --extra viz     # install deps + dev tools + matplotlib
uv run pytest -q        # run tests = run the ValidationSuite (the real check)
uv run pytest -q -n 0   # ...serially, when debugging (-n 6 is the default)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run python -m sandbox.demos.wright_fisher   # end-to-end demo
```

## The non-negotiables

1. **Protocol first.** Every model implements `core.protocol.Model`. Shared
   services (Recorder, sweep, ValidationSuite, viz) consume only the protocol
   surface — they never reach inside a model's concrete state.
2. **A core model is done only when its `analytic_predictions` are reproduced
   by the ValidationSuite within tolerance.** Validation *is* the definition of
   done, and the tolerance is statistical (`z * standard error`), never a
   hardcoded epsilon. Prefer this over inventing new pass/fail logic.
3. **Reproducibility.** Seed every stochastic run. Derive replicate RNGs with
   `core.rng.spawn_rngs` (`SeedSequence.spawn`) — never `default_rng(seed + i)`.
   Every `Experiment` serializes and re-runs to the same `Result`.
4. **Profile before optimizing.** Stay in NumPy until a real bottleneck forces
   numba/JAX/WebGL (expected first at Gray-Scott grids and HH networks). When you
   do optimize the SSA, the trajectory must stay **bit-identical** — every
   recorded slope anchor depends on it. Fingerprint a replicate before and after
   (`sha256` of times + series) rather than trusting that a change "looks safe".
5. **Quarantine the speculative arc** under `models/ecosystem/`. Any model
   without a checkable prediction is exploratory and must be labelled so in code
   and UI. Guard the verifiable/exploratory boundary continuously.

## Protocol conventions (easy to get wrong)

- `step(state, rng)` takes **no `dt`**: the model owns its own time increment and
  writes it into `state.t`. Numerical step sizes are *params*, not call args.
  (Wright-Fisher = one generation; Gillespie samples its own waiting time.)
- Model objects are **stateless**. `initial_state(params, rng)` embeds the
  params into the returned `State`, so `step`/`observables`/`is_terminal` are
  pure functions of the state. `analytic_predictions(params)` is the one method
  taking params directly.
- Optional methods: `is_terminal(state)` (early stop) and `analytic_predictions`
  (required for core models, omitted for speculative ones).

## Layout

`src/sandbox/core/` protocol + services · `models/` the models ·
`viz/backends/` pluggable plotting · `demos/` runnable examples ·
`tests/` wraps the ValidationSuite · `docs/plans/` per-phase dev docs.

## Workflow

- Add a model: write its `*Params`/`*State`, implement the protocol, register it
  in `models/__init__.py`, then write the validation test **first** and confirm
  it can fail before the implementation is correct.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`). Each
  commit should pass `pytest` and `ruff`.
- **Batch / session end:** update memory + `docs/`, then commit and push.

## Status

Phase 0 (Wright-Fisher) complete and validated. **Phase 1 complete**: RK4
integrator + `DeterministicLimitModel` protocol (step 1), Gillespie SSA engine
(step 2), `birth_death` — the engine's exact-closed-form check (step 3:
stationary mean `k/gamma` via `validate()` + Fano-factor `Var/<n>=1` in counts),
and `isomerization` — the second exact check (step 4: `A<->B`, conserved total,
stationary mean `(k2/(k1+k2))*c_tot` via `validate()`), and `core/convergence.py`
— the second validation track (step 5: `convergence_report()` checks the log-log
slope of `D(Omega)` is consistent with `-1/2` and significantly negative, with a
statistical `max(bootstrap, OLS)` slope SE; validated on `birth_death`, teeth via a
pure per-replicate-vs-mean-first unit test + a `D*sqrt(Omega)` magnitude anchor) —
and `repressilator` — the headline (step 6: 6-species Elowitz-Leibler Hill network,
**no** `analytic_predictions`, validated by `convergence_report` at slope
`-0.4606 +/- 0.0734`, with broken-Omega-scaling teeth failing at `-0.1363` and
`-0.9904`, each verified across seeds 0-3), `viz` (step 7: `plot_convergence`
log-log helper — plain arrays, drawn-but-excluded knee, fit and guide both anchored
at the fitted centroid — plus a *tested* `plot_replicates` ODE overlay, which needed
no rewrite since both sides are concentrations), and `demos/repressilator.py`
(step 8: engine-vs-closed-forms, the overlay figure, the scaling figure; about a
minute on a deliberately reduced, seed-checked config that says so in its own
output).

Lessons worth carrying forward: assert a broken-model tooth only on the leg that is
*structurally* robust for that break (for a flat-slope break, `slope/SE` is
replicate-independent, so more replicates never de-flake it); `beta=1`, **not** the
textbook `beta=5`, is what actually oscillates (beta=5 damps to the fixed point);
`fit_mask` must exclude the low-`Omega` **phase-saturation knee** (`Omega <= 1`
here), spotted as `D*sqrt(Omega)` falling below the plateau; a **figure carries a
claim and gets a test like anything else** — the overlay's live trap is the ODE
*column index* (a permuted protein is the right shape at the wrong phase), and both
viz helpers were mutation-checked; and pin the *worst* seed of the ones you checked,
not a representative one. Next: **Phase 2** (Hodgkin-Huxley, Gray-Scott) — the
performance ceilings, where numba/JAX/WebGL get introduced against real profiling.
See `docs/plans/` and HANDOFF.md §5.
