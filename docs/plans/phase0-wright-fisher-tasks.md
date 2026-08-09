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

## Phase 1 (Gillespie / repressilator) — done

Superseded by `phase1-gillespie-plan.md` / `-tasks.md`, which are the record of
what was actually built. Kept here because two items in this stub were **wrong**,
and the corrections are the interesting part:

- [x] `models/gillespie.py` — the **Direct Method**, not the next-reaction method
      this stub called for. Gibson-Bruck was explicitly deferred: profile first,
      and the Direct Method never became the bottleneck.
- [x] Mass-action ODE limit of the same reaction set — and *derived from the same
      `rates`* (`stoich.T @ rates(c)`), so the SSA and the ODE cannot drift apart.
- [x] ~~"stochastic **mean** converges to the ODE as molecule counts grow"~~ —
      this is the phase-diffusion trap, and following it would have produced a
      green check that proves nothing: replicates of an oscillator dephase, so
      their ensemble mean damps toward the cycle's time-average while every
      replicate keeps oscillating. The correct claim is Kurtz's: a **single**
      scaled trajectory converges, with `O(Omega^-1/2)` fluctuations. See
      `core/convergence.py`.
- [x] Reusable "overlay stochastic replicates on the deterministic limit" UX —
      `plot_replicates` generalized straight from the Wright-Fisher figure with no
      rewrite, plus `plot_convergence` for the scaling law.
- [x] Recorder tolerates irregular `t` — confirmed against Gillespie output.

## Next (Phase 2 — the expensive ones)

Per HANDOFF.md §5. No plan doc exists yet; writing one is the first move.

- [ ] **Hodgkin-Huxley** — single cell, then small networks. Checkable claims:
      spike shape, refractory period, f-I curve.
- [ ] **Gray-Scott** — pattern wavelength against linear stability analysis; the
      natural first target for a shader/browser backend.
- [ ] These are the performance ceilings: numba/JAX/WebGL get introduced here,
      **guided by profiling rather than guesswork** (non-negotiable #4), and any
      SSA-side optimization must stay bit-identical.
