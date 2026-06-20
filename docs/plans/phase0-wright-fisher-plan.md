# Phase 0 — Wright-Fisher vertical slice (plan)

**Status: complete and validated.**

## Goal

Build the *thinnest complete path* through the entire architecture before adding
a second model — `protocol → model → recorder → sweep → validation` — and get
the neutral fixation-probability check passing over many seeded replicates. When
this slice is green, the protocol is proven and every later model slots into the
same scaffolding.

## What was built

1. `core/protocol.py` — `Model` / `ValidatableModel` / `TerminableModel`
   protocols, `State` marker, serializable `Experiment` and `Result`.
2. `core/rng.py` — `make_rng`, `spawn_rngs` (`SeedSequence.spawn`).
3. `core/recorder.py` — `Trajectory` + `run_replicate` (tolerates irregular `t`).
4. `core/registry.py` — name → model resolution (keeps `Experiment` serializable).
5. `core/sweep.py` — `run_experiment` (Cartesian sweep × seeded replicates).
6. `core/validation.py` — `validate` with statistically-derived tolerance.
7. `models/wright_fisher.py` — the model + `WFParams`/`WFState`.
8. `viz/backends/matplotlib_backend.py` — `plot_replicates` (stochastic cloud +
   deterministic limit), import-guarded.
9. `demos/wright_fisher.py` — full end-to-end run.
10. `tests/` — protocol, rng, model unit, validation (headline), reproducibility.

## Key design decisions (and why)

- **`step(state, rng)` takes no `dt`.** Wright-Fisher is the one model where `dt`
  is trivial, so a `dt`-taking signature would "prove" the protocol against the
  easy case while being unable to express Gillespie (Phase 1), whose step is a
  *sampled* waiting time. The model owns its increment and writes `state.t`.
- **Models are stateless; `initial_state` embeds params into the state.** This is
  what lets `step` be a pure function of `(state, rng)` despite taking no params.
- **Statistical tolerance, not a hardcoded epsilon.** Fixation fraction over `R`
  replicates has SE `≈ sqrt(p0(1-p0)/R)`; we assert within `z=4` SE. A correct
  model passes reliably; a wrong one fails harder as `R` grows.

## The validation that defines success

Run many neutral replicates; the fixation fraction of allele A must equal its
initial frequency `p0` within statistical tolerance. Verified at `p0 ∈
{0.3, 0.5, 0.7}` (see `tests/test_validation.py`), plus a deliberately-wrong
prediction is confirmed to FAIL (the check has teeth).

## Deferred

- Kimura diffusion *distribution* convergence as `N` grows (handoff Phase-0 step
  5, stretch) — fixation-probability check is the headline and is done.
- Selection (`s != 0`) analytic prediction — `step` already supports selection,
  but `analytic_predictions` raises `NotImplementedError` for `s != 0` rather
  than ship a Kimura formula not yet matched to this sampling convention.
