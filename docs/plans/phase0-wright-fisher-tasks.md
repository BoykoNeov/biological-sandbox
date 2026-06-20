# Phase 0 — tasks

- [x] `pyproject.toml` (uv), `.gitignore`, `LICENSE` (MIT), `README.md`
- [x] `core/protocol.py` — Model/ValidatableModel/TerminableModel, Experiment, Result
- [x] `core/rng.py` — `make_rng`, `spawn_rngs`
- [x] `core/recorder.py` — `Trajectory`, `run_replicate`
- [x] `core/registry.py` — name → model
- [x] `core/sweep.py` — `run_experiment` (sweep × replicates)
- [x] `core/validation.py` — `validate` with statistical tolerance
- [x] `models/wright_fisher.py` — model + params/state
- [x] `viz/backends/matplotlib_backend.py` — `plot_replicates`
- [x] `demos/wright_fisher.py` — end-to-end
- [x] tests: protocol, rng, model unit, **validation (headline)**, reproducibility
- [x] `uv run pytest` green (22 passed); `uv run ruff check .` clean
- [x] demo produces the stochastic-vs-deterministic figure
- [x] `CLAUDE.md`, dev docs, public repo
- [x] GitHub Actions CI (`.github/workflows/ci.yml`): lint + ValidationSuite on
      Python 3.11/3.12/3.13
- [x] anti-bias guard: `validate()` fails loudly if a terminable model's
      replicates hit `max_steps` without absorbing (`Trajectory.terminated`)

## Next (Phase 1 — Gillespie / repressilator)

- [ ] `models/gillespie.py` — next-reaction method; **stress-tests the no-`dt`
      step signature** (model samples its own waiting time)
- [ ] Mass-action ODE limit of the same reaction set (deterministic comparison)
- [ ] Validation: stochastic mean converges to the ODE as molecule counts grow
- [ ] Reusable "overlay stochastic replicates on the deterministic limit" UX,
      generalized from the Wright-Fisher figure
- [ ] Recorder already tolerates irregular `t` — confirm against Gillespie output
