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
integrator + `DeterministicLimitModel` protocol, Gillespie SSA engine,
`birth_death` and `isomerization` (the two exact closed-form checks),
`core/convergence.py` (the second validation track: log-log slope of `D(Omega)`
against `-1/2` with a statistical `max(bootstrap, OLS)` SE), `repressilator` (the
headline — no `analytic_predictions`, validated by `convergence_report` at slope
`-0.4606 +/- 0.0734` with two broken-`Omega` teeth), `viz`, and
`demos/repressilator.py`.

**Phase 2 complete** — Hodgkin-Huxley (2a) and Gray-Scott (2b). See
`docs/plans/phase2-{plan,context,tasks}.md`; the tasks doc carries every measured
number. Suite **310 passed in 130 s at `-n 6`**.

Two decisions shaped the phase. First, **three categories of checkable claim** —
*A* exact analytic (`analytic_predictions`), *B* asymptotic law (log-log slope +
CI), *C* literature-anchored (rheobase, f-I, spike shape) — and **category C never
enters `analytic_predictions` nor an assertion's bound**; it is reported in demos,
and only *structural* facts about it are asserted. Second, HANDOFF's "validate
Gray-Scott's pattern wavelength against LSA" was **reframed after measurement**: at
Pearson's famous `(F, k)` there is no real non-trivial homogeneous steady state, so
those patterns are not Turing patterns and LSA makes no claim about them. Phase 2
validates **`lambda(q)` itself** and labels the emergent pattern qualitative.

**2a — Hodgkin-Huxley.** `hh_rates` (six rate functions, `_linoid` for the removable
`0/0` at `V=-40`/`-55` via `expm1`), `hh_voltage_clamp` (the category-A lead anchor:
clamped gating decouples into an exact `x_inf + (x0-x_inf)e^{-t/tau}`),
`hodgkin_huxley` (deterministic 4-D; `analytic_predictions` is the root-found fixed
point and **raises** past the subcritical Hopf), `hh_stochastic` (8-state Na +
5-state K **occupancy counts**, `O(#states)` per step and independent of `N`, with
an **exact factorized propagator** `P = kron(P_m, P_h)` verified against a
hand-transcribed generator + local `expm` at 7.4e-15), and the headline
**`D(N) ~ N^{-1/2}`** at slope `-0.5092 +/- 0.0106` (seeds 0-3: `-0.5092, -0.4933,
-0.4988, -0.5101`), with two teeth. `demos/hodgkin_huxley.py` reports category C.

**2b — Gray-Scott.** `core/laplacian.py` (periodic 5-point stencil; a Fourier mode is
an **exact** eigenfunction, pinning stencil + wrap + integrator at once; order-2
consistency slope `1.99835`), `gray_scott` (`analytic_predictions` = the growth rate
from the **discrete** operator), `lambda(q)` validated at 8 probes **across a sign
change**, the `FieldModel` viz-only extension, and `demos/gray_scott.py`.

## Lessons worth carrying forward

**On tolerances.** Match the tolerance to the *claim*, and derive it rather than
typing it. Richardson in `dt` sees discretization error and nothing else — it cannot
see leftover transient (HH attraction) and it cannot see a *linearization* error
(Gray-Scott), where the right instrument is **Richardson in the amplitude**:
`(4/3)|m(a) - m(a/2)|` predicted the true error to a ratio of **1.000** at all eight
probes. For a deterministic model `validate()`'s statistical SE degenerates to zero,
so supply a numerical `sem_floor` and use two replicates (one gives `sem = inf` and
passes vacuously).

**On tests that are green for the wrong reason.** This phase produced five, and they
are the most valuable thing in it:

- `assert std > 0.0` passed a *rounded* initial condition, because `np.std` of sixty
  bit-identical values returns `2.7e-20`, not `0.0`. **A threshold nothing can fail
  is not a check.**
- A conservation test ran at exactly `N = 5000`, where `round(N p)` happens to sum to
  `5000` for both populations. At `N = 10000` it sums to `10001`. **Sweep the
  constant that the probe happens to land on.**
- A mutant that **never moved a single channel** (multinomial aggregated along the
  wrong axis) passed everything, because at a fixed point a frozen membrane sits
  still exactly like a correct one. Caught only by a *transient* regime and a **ratio
  between two system sizes** — an `N`-independent error has no tolerance that
  distinguishes "small" from "not shrinking".
- Asserting `linspace` grids misalign **failed**: at stride 8 they align perfectly,
  because dividing by a power of two is exact. 373 other configurations misalign. The
  hazard is not "it is wrong" but "it is right until someone changes `n_grid`".
- The order-2 slope read `1.9737` and the shortfall was *real*, not noise: `k h / 2`
  was `0.785` at the coarsest grid. **An asymptotic order is only measurable inside
  the asymptotic regime**, and which grids are inside it is part of the claim.

**On teeth.** Verify at 3-4 seeds and assert only the leg that is *structurally*
robust for that break. A tooth must also bite the right thing: the repressilator's
`Omega^2` tooth could not be reused for HH, because squaring `N` drives `D` an order
of magnitude below the splitting bias and the broken model would fail through a
*discretization floor* rather than through its scaling.

**On performance, both directions.** Profile before optimizing — and record the
measurement when it says "do nothing". Twice here the *obvious* optimization was
backwards: a "vectorized" NumPy propagator build cost **80.5 us/step against the
72 us naive loop it replaced** (on a 4x4 there is no arithmetic to amortize, only
per-call overhead), and `np.kron` cost 5.5x a broadcast-and-reshape. Batching 13
multinomial draws into 2 was a **wash** (13.30 vs 13.46 us). Equally, the Phase-1
close-out's proposed `-n` fix — dispatch the long tests first — measured **75%
worse** (228 s vs 130 s) and was reverted.

**On figures.** A figure carries a claim and gets looked at, not exit-code checked.
Four claims were wrong until someone looked: an f-I curve that put rheobase at
`[2, 4]` by counting onset spikes; channel noise demonstrated at `I = 20`, where the
drive swamps it; a dispersion curve drawn solid through the region where the
eigenvalue pair is *complex*; and a Turing panel implying its stripes were a
*selected* wavelength when the mode had been seeded by hand.

**On refusing to answer.** `analytic_predictions` raises rather than returning a
wrong number that still looks green — past the subcritical Hopf, where no real
non-trivial homogeneous state exists, on a complex eigenvalue pair, and on a growth
rate too near zero to measure (`j=13` needs `t_max = 18822` and returns `nan`).

Next: Phase 3. See HANDOFF.md.
