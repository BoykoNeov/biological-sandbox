# Biology Simulation & Experimentation Sandbox — Handoff

## Purpose of this document

This is the orienting document for continued development in Claude Code. It defines the project's organizing principle, the architecture, the build order, and the long-range directions. Read it before writing code. The single most important instruction is the build order: **start simple and validated, and only cross into the speculative once the validated core is real.** Everything else follows from that.

---

## 1. The organizing principle

This project is not a grab-bag of biology demos. It has one intellectual spine:

> **Stochastic dynamics and the deterministic limits they collapse into.**

Every model in the validated core is an instance of this idea:

- **Wright-Fisher** — genetic drift (stochastic) vs. deterministic allele-frequency change; converges to the Kimura diffusion as population size grows.
- **Gillespie / repressilator** — molecular noise (stochastic) vs. mass-action ODEs (mean-field); converges as molecule counts grow.
- **Hodgkin-Huxley** — stochastic ion channels vs. the deterministic HH equations; converges as channel count grows.
- **Gray-Scott / Turing** — the deterministic PDE whose pattern wavelength is *predicted* by linear stability analysis.

The central teaching move — and the central piece of UX — should be: **show the same phenomenon stochastically and deterministically, and watch them converge as the system gets large.** If a feature doesn't serve that thread, question whether it belongs in the core.

This is also why the project is *trustworthy*. Each core model has a known analytic result you can check the code against. Validation is not a testing afterthought — it is the pedagogy.

---

## 2. Two categorically different kinds of software

Be explicit about this boundary at all times, because it determines what "done" and "correct" even mean.

**Verifiable models (the core).** Wright-Fisher, Gray-Scott, Gillespie, Hodgkin-Huxley. Each has an analytic limit, a closed-form prediction, or a measurable emergent quantity. "Correct" is a checkable claim.

**Generative systems (the speculative arc).** The ecosystem / biosphere simulator. There is no ground truth, no analytic limit, no f–I curve to confirm against. This does not make it worthless — it makes it a different category. It is **exploratory artificial life, not predictive science.** The lineage is Tierra, Avida, Polyworld, SimEarth. Those projects are legitimate precisely because they never claimed to predict real ecosystems.

The failure mode of the generative arc is specific and seductive: producing something that looks gorgeously alive but contains no biology — beautiful eye-candy presented as a claim about real biospheres. The rule that prevents this: **never add a generative feature without being able to answer "what can I check?"** If the honest answer is "nothing," it is art, and it must be labelled as exploratory in the UI and the docs.

---

## 3. Architecture: one interface, not one engine

The models are genuinely different numerically and must not share a solver:

| Model | Mathematical type | Numerics |
|---|---|---|
| Wright-Fisher | discrete-time Markov chain | multinomial sampling |
| Gray-Scott | reaction-diffusion PDE | finite differences, CFL-constrained |
| Gillespie | continuous-time jump process | variable Δt (next-reaction) |
| Hodgkin-Huxley | stiff coupled ODEs | adaptive/implicit integrator |

No single engine covers these. **Loose coupling is achieved through a common protocol, not a common implementation.** Every model implements the same small interface; shared services are built *on top of* the protocol and never reach inside a model.

### 3.1 The Model protocol (the spine)

```python
class Model(Protocol):
    def initial_state(self, params: Params) -> State: ...
    def step(self, state: State, dt: float, rng) -> State: ...        # one increment
    def observables(self, state: State) -> dict[str, float]: ...      # scalar summaries
    # optional, declared per-model — this is the validation contract:
    def analytic_predictions(self, params: Params) -> dict[str, float]: ...
```

`step` is the only thing a model must get right. `analytic_predictions` is what makes the model trustworthy — it returns the closed-form values the simulation should reproduce (e.g. neutral fixation probability, Turing wavelength). A model without `analytic_predictions` is allowed only in the speculative arc, and must be flagged as such.

### 3.2 Shared services (built on the protocol)

Keep these strictly decoupled from model internals. Each consumes only the protocol surface.

- **Recorder** — captures trajectories / observables over time. Pure consumer of `observables`.
- **Visualizer** — renders state and trajectories. Pluggable backends (see §4 on browser vs. local).
- **ParameterSweep** — runs a model across a grid of params and/or many stochastic replicates.
- **ValidationSuite** — runs replicates, compares measured observables against `analytic_predictions`, reports pass/fail with tolerances. This is the heart of the project's credibility.

### 3.3 The Experiment as a first-class object

Make the experiment a declarative, serializable spec — not a script. This single abstraction delivers reproducibility, the "experimentation" in the project's title, and the test corpus, all at once.

```python
@dataclass
class Experiment:
    model: str
    params: Params
    sweep: dict | None          # parameter grid
    replicates: int             # stochastic repeats
    observables: list[str]
    seed: int                   # reproducibility
```

An `Experiment` runs to a reproducible `Result`. Validation cases are just `Experiment`s whose expected output is known. Saved experiments are how a user shares or revisits a finding.

### 3.4 Suggested layout

```
sandbox/
  core/
    protocol.py          # Model, State, Params, Experiment, Result
    recorder.py
    sweep.py
    validation.py
    rng.py               # seeded RNG plumbing (reproducibility lives here)
  models/
    wright_fisher.py
    gillespie.py         # repressilator as a reaction set
    hodgkin_huxley.py
    gray_scott.py
    ecosystem/           # speculative arc — quarantined here
  viz/
    backends/            # matplotlib / web / shader
  experiments/           # saved Experiment specs (incl. validation cases)
  tests/                 # wraps ValidationSuite into CI
```

The `ecosystem/` package being a sibling of the validated models — not entangled with them — is the structural expression of §2. The boundary is visible in the directory tree.

---

## 4. Compute: where the walls are

Start in plain **NumPy** for clarity and correctness everywhere. Optimize only the proven bottlenecks. Reaction-diffusion grids and HH networks are where Python dies first.

Escape hatches, in rough order of effort:

1. **numba** — JIT the hot loops (PDE stencils, network integration). Smallest change.
2. **JAX** — `vmap` over parameters and replicates (run ten thousand Wright-Fisher populations at once), `scan` for time-stepping, free GPU. Best fit for the embarrassingly-parallel stochastic sweeps that the core's central thread depends on. **Weakness: Gillespie.** Data-dependent control flow and variable Δt fight the functional style — keep Gillespie in NumPy/numba.
3. **WebGL / shaders** — Gray-Scott runs live and beautifully in the browser. Natural fit for the education side.

### The browser-vs-local fork (decide early)

This shapes everything downstream, so make the call deliberately:

- **Browser** (Pyodide, or a JS/WASM core) — what makes it genuinely *usable by students*. Costs a reimplementation of the numerics.
- **Python / Jupyter** — faster to build, but stays a tool for people who already code.

A reasonable hybrid: build the validated core in Python first (fastest path to a correct, checkable engine), keep the `Visualizer` backend pluggable, and add a browser/shader front-end for the demos that benefit most (Gray-Scott above all) once the core is proven.

---

## 5. Build order (the spine of the whole effort)

### Phase 0 — Vertical slice: Wright-Fisher, end to end

Build the *thinnest complete path* through the entire architecture before adding a second model:

1. `protocol.py` with `Model`, `State`, `Params`, `Experiment`, `Result`.
2. `wright_fisher.py` — `step` via multinomial sampling; `analytic_predictions` returns neutral fixation probability = initial allele frequency.
3. Recorder + a minimal Visualizer (matplotlib is fine).
4. ParameterSweep + ValidationSuite.
5. **The validation that defines success:** run many replicates of a neutral population; confirm the measured fixation fraction matches `p0` within statistical tolerance. Then confirm the trajectory distribution approaches the Kimura diffusion as N grows.

When this slice passes, the protocol is *proven* and every later model slots into the same scaffolding. Do not skip ahead until it does.

### Phase 1 — Establish the stochastic↔deterministic thread

Add **Gillespie / repressilator** next, because it's where the thread becomes visible and interactive: run the stochastic simulation alongside the mass-action ODE limit and show convergence as molecule counts rise. This locks in the project's organizing idea as a concrete, reusable UX pattern (overlay stochastic replicates on the deterministic limit).

### Phase 2 — The expensive ones

**Hodgkin-Huxley** (single cell → small networks; validate spike shape, refractory period, f–I curve) and **Gray-Scott** (validate pattern wavelength against linear stability analysis; ideal first target for a shader/browser backend). These are the performance ceilings — this is where numba/JAX/WebGL get introduced, guided by real profiling rather than guesswork.

### Phase 3+ — Cross gradually into the speculative

See §6. Each step must keep a checkable prediction for as long as one exists, and be explicitly labelled exploratory once one no longer does.

---

## 6. The speculative arc, done honestly

Do not build "generate entire biospheres" top-down. Build it bottom-up from components that still have validation, and add structure only while you can still answer "what can I check?" The arc crosses gradually from verifiable to exploratory:

- **Food webs — still verifiable.** Start with generalized Lotka-Volterra: known equilibria and stability theory. Then May's result (random community matrices: stability *decreases* with complexity) becomes a real, checkable prediction about your own simulated webs. You are still doing verifiable science here.
- **Climate coupling — bridge.** Daisyworld is the perfect entry: tiny model, one interpretable result (albedo feedback regulating planetary temperature), bridges biology to planetary dynamics without hand-waving. Still checkable.
- **Speciation — the crossing point.** This is where most artificial-life projects cheat: draw a line, declare two clusters "species." The rigorous version is **adaptive dynamics**, where evolutionary branching is a *predicted* outcome of frequency-dependent selection — speciation gets a mechanism instead of a label. This is roughly the last point where a clean prediction exists.
- **Full ecosystem / biosphere — exploratory artificial life.** Beyond this point you are in Tierra/Avida/Polyworld territory. Legitimate and worthwhile *if and only if* it is framed and labelled as exploration, never as prediction.

### The hard limitation to name up front

The deepest obstacle across the whole ambitious arc is **multi-scale coupling**: molecular → cellular → population → ecosystem → planet, each with its own formalism. Linking them faithfully is open research, not engineering. This is also exactly where scope creep will eat the project alive — the ambitious list is enormous. Treat each scale as a separately-validated module and be honest that the *couplings between scales* are the speculative part, even when the individual scales are sound.

---

## 7. Standing rules for the Claude Code session

1. **Protocol first, implementations second.** Never let a shared service reach inside a model's internals.
2. **A core model is not done until its `analytic_predictions` are reproduced by the ValidationSuite within tolerance.** Validation cases are the definition of "done," not an extra.
3. **Reproducibility is non-negotiable:** every stochastic run is seeded; every `Experiment` is serializable and re-runnable to the same `Result`.
4. **Optimize only what you've profiled.** NumPy until a real bottleneck forces numba/JAX/WebGL.
5. **Keep the speculative arc quarantined** in its own package, and label any model without a checkable prediction as exploratory in both code and UI.
6. **Guard the boundary in §2** continuously. The moment generative eye-candy starts being presented as a claim about real biology, the project has failed on its own terms.

---

## 8. Immediate next action

Implement Phase 0: the Wright-Fisher vertical slice through `protocol → model → recorder → visualizer → sweep → validation`, with the neutral-fixation-probability check passing over many replicates. Get that green, then proceed to Gillespie.
