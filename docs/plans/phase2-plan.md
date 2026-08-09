# Phase 2 — Hodgkin-Huxley & Gray-Scott (plan)

**Status: planning.** Phase 1 is complete (`phase1-gillespie-plan.md`). This is the
phase HANDOFF §5 calls "the expensive ones" — the two performance ceilings, and
the first phase where a promised validation had to be **reframed to stay honest**
(see "The HANDOFF deviation" below; it is the most important section here).

## Goal

Carry the organizing thread — *stochastic dynamics and the deterministic limits
they collapse into* — onto two models that break the Phase-1 mould:

- **Hodgkin-Huxley** — a **stiff, threshold** ODE system. Its stochastic
  counterpart is *channel noise*: a finite number `N` of ion channels, converging
  to the deterministic HH equations as `N -> infinity`. The `Omega` of Phase 1
  becomes `N_channels`, so the existing convergence pathway is the *reused*
  machinery rather than new machinery.
- **Gray-Scott** — a **reaction-diffusion PDE**. Its checkable claim is not a
  large-system limit at all but **linear stability analysis**: the growth rate
  `lambda(q)` of a small Fourier perturbation about a homogeneous steady state.

Both are also the profiling targets. Neither is allowed to acquire a dependency
(numba/JAX/WebGL) that profiling has not demanded — see "Performance" below.

---

## The three categories of "checkable" (name them, or they blur)

Phase 0 and 1 only ever used one category. Phase 2 needs all three, and the
project's credibility depends on never letting the third impersonate the first.

| Category | Meaning | Phase-2 instances |
|---|---|---|
| **A. Exact analytic** | A closed form the simulation must reproduce to statistical/numerical tolerance. The `analytic_predictions` contract. | HH voltage-clamp gating; HH resting fixed point; discrete-Laplacian Fourier eigenvalue; Gray-Scott `lambda(q)` |
| **B. Asymptotic law** | A *scaling exponent*, checked by a log-log slope with a statistical CI — the Phase-1 convergence pathway. | HH `D(N) ~ N^{-1/2}`; RK4 order 4; Laplacian consistency order 2 |
| **C. Literature-anchored** | A number from the published record, not derivable in closed form. Reproducing it is evidence, **not proof**, and it must be *labelled* as such. | HH rheobase, f-I curve, spike shape, refractory period, subcritical-Hopf bistability |

**Rule:** category C never goes in `analytic_predictions`. It gets its own
reporting path and says "literature-anchored" in its own output. A model is
"done" on category A + B alone.

---

## The HANDOFF deviation (read this before writing any Gray-Scott code)

HANDOFF §5 promises: *"Gray-Scott — validate pattern wavelength against linear
stability analysis."* **As literally written, that check is not available at the
parameters that make Gray-Scott famous.** This was not an argument from theory;
it was measured before any code was planned (`temp/phase2-slice/`):

At Pearson's classic pattern points — `(F, k) = (0.037, 0.060)`, `(0.025, 0.055)`,
`(0.014, 0.054)`, `(0.040, 0.060)`, `(0.035, 0.065)` — the non-trivial homogeneous
steady state **does not exist as a real solution** (`F < 4(F+k)^2`). The only
homogeneous state is the trivial `(u, v) = (1, 0)`. The famous self-replicating
spots and labyrinths there are **not Turing patterns**: they are excitable /
subcritical structures whose wavelength linear theory does not set. "Validate the
emergent wavelength against LSA" at those parameters would be comparing a
measurement to a prediction that makes no claim about it.

Where a real non-trivial state *does* exist, most of the plane fails for a second
reason: the maximum of `lambda(q)` sits at `q = 0` (a bulk instability, not a
diffusion-driven one). A **genuine** Turing point needs all three of: a real
non-trivial state, stability *without* diffusion, and an interior maximum
`lambda(q*) > 0` at `q* > 0`. Scanning `F in [0.010, 0.120] x k in [0.030, 0.080]`
(11k points, `Du = 2e-5`, `Dv = 1e-5`) found genuine Turing points in a **razor
sliver**: `F in [0.049, 0.117]`, `k in [0.054, 0.062]`, and the window in `k` at
fixed `F` is only ~0.5% wide (`F = 0.074`: `k in [0.06166, 0.06200]`).

**So Phase 2 splits the promise in two, and labels both:**

1. **Category A — validate `lambda(q)` itself.** Seed the homogeneous steady
   state plus a small single-mode perturbation `eps * cos(q x)`, measure the
   mode's exponential growth/decay rate, and compare it to the eigenvalue of
   `J - q^2 D`. This *is* validated linear stability analysis, it is exact, it is
   cheap, and it works at any `(F, k)` and any `q` — including **decaying** modes,
   so the test spans a **sign change** and a wrong sign cannot hide. At the chosen
   point `(F, k) = (0.074, 0.062)` the measured band is `q in [16.05, 76.37]` with
   `lambda(q*) = +0.014986` at `q* = 43.828`; probes at `q = 10` and `q = 80, 120`
   decay. This is the honest, sharp version of HANDOFF's promise.
2. **Category C / exploratory — the emergent far-from-onset pattern.** The
   Pearson-regime spots and worms are kept as a *demo and a figure*, explicitly
   labelled as qualitative. Their wavelength is **not** asserted against LSA.

A near-onset wavelength-selection prediction *is* cleanly available from a
Schnakenberg or Brusselator model. It is **noted and deferred** — adding a third
model to buy back one check is scope creep, and category A above already
validates the same physics more sharply.

---

## Hodgkin-Huxley

### Measured before planning (the sanity slice)

Standard HH-1952 in the modern convention (`V` mV, `t` ms, `C_m = 1 uF/cm^2`,
`g_Na, g_K, g_L = 120, 36, 0.3 mS/cm^2`, `E_Na, E_K, E_L = 50, -77, -54.387 mV`):

- **Resting fixed point** (`I = 0`): `V_rest = -64.996379331 mV`,
  `(m, h, n) = (0.05295509, 0.59599412, 0.31773240)`; `|rhs| = 1.8e-15`;
  `eig(J) = {-4.675, -0.2026 +/- 0.3832i, -0.1207}` — **stable**, slowest mode
  `tau = 8.29 ms`, and the complex pair is the subthreshold resonance.
- **Stiffness.** The fastest time constant anywhere in `[-90, 60] mV` is
  `tau_m = 0.0622 ms`. `dt = 0.01 ms` is 6.2x smaller — enough for **explicit
  RK4**, so `core/ode.py` is reused and **no scipy stiff integrator is needed**.
  Measured RK4 error vs a `dt = 5e-4` reference over 20 ms at `I = 10`:
  `6.45e-7, 4.43e-8, 2.89e-9, 1.84e-10` at `dt = 0.02, 0.01, 0.005, 0.0025` —
  ratios `14.6, 15.3, 15.7` converging on `16 = 2^4`. **Fourth order confirmed on
  the real RHS**, which is category B and a test in its own right.
- **Rheobase and bistability** (category C). Repetitive firing starts between
  `I = 6.2` and `I = 6.5 uA/cm^2` — the textbook value. The fixed point remains
  **stable** at `I = 6.5, 7, 8` while the model fires, and loses stability between
  `I = 8` and `I = 10`: the classic HH **subcritical Hopf with a bistable band**.
  Worth reproducing *as* category C, because it is exactly the kind of result that
  looks like a prediction and is not one.

### The implementation trap already found

`alpha_m` and `alpha_n` are `x / (1 - exp(-x/k))` forms with removable `0/0`
singularities at **`V = -40`** and **`V = -55`** — the round numbers a caller is
most likely to probe. The limit is `k`; a `_linoid(x, k)` helper handles it with
the series `k + x/2` near zero. This gets a dedicated test (evaluate exactly at
the singular voltages, and check continuity across them), not a comment.

### Validation, by category

- **A. Voltage clamp — the `birth_death` of Phase 2.** Hold `V` fixed and the
  gating equations *decouple* into `dx/dt = (x_inf(V) - x)/tau(V)`, exactly solved
  by `x(t) = x_inf + (x0 - x_inf) e^{-t/tau}`. This validates **every rate
  function individually**, to numerical precision, in milliseconds of compute. It
  is the lead anchor and the `analytic_predictions` payload of a dedicated
  `hh_voltage_clamp` model.
- **A. Resting fixed point.** `rhs(y_rest) = 0` and the simulation does not drift.
- **B. RK4 order 4** on the real RHS (numbers above).
- **B. Channel-noise convergence `D(N) ~ N^{-1/2}`** — see below.
- **C. Rheobase / f-I / spike shape / refractory / bistability** — reported and
  labelled, never in `analytic_predictions`.

A linearized-Jacobian matrix-exponential check is **deferred**: comparing the
nonlinear simulation to the linearized solution is only valid to `O(eps^2)`, so an
honest version is a *slope-2-in-eps* check, not a tolerance at one `eps`. The
voltage-clamp anchor already validates the same rate functions more sharply.

### Stochastic HH: channel noise, and where its slope check must live

The stochastic model tracks **state-occupancy counts** of a finite channel
population: 8-state Na (`m0..m3` x `h0/h1`, conducting at `m3h1`) and 5-state K
(`n0..n4`, conducting at `n4`). Because only the *counts per state* are tracked,
a step is **O(#states), independent of `N`** — the `N`-sweep is therefore cheap,
which is what makes this affordable at all. (Fox-Lu independent-*subunit* noise is
the cheaper-looking alternative but is a known *approximation*: subunit and
channel-state models disagree at finite `N` (Goldwyn & Shea-Brown). Both converge
to the same HH ODE, so a `-1/2` claim would be safe either way — but the
channel-state model is the microscopically correct one, and costs the same.)

**`step` is a fixed-`dt` hybrid, not an SSA event.** Channel rates depend on `V`,
which changes continuously, so propensities are *time-inhomogeneous* and the
Direct Method's "constant between events" assumption fails. One `step` therefore
advances a fixed `dt`: multinomial channel-state transitions, then an RK4 update
of `V` at the resulting conductance. This is a *third* distinct time-advance
discipline under the same protocol (Wright-Fisher = one generation, Gillespie = a
sampled `tau`, HH = a fixed numerical `dt` owned by the model as a **param**) —
which is precisely what the no-`dt`-argument signature was designed to allow.

**The trap — and it is NOT phase diffusion.** HH is a *threshold* system. In the
spiking regime, low channel counts produce **spurious and missing spikes**: a
different spike *count*, not a phase shift, and a `100 mV` discrepancy from a
tiny time shift. That obeys no `-1/2` law. So the slope check runs in a
**sub-rheobase, subthreshold, short-horizon** regime, where `V` fluctuates about
a stable fixed point and the linear-noise approximation is exact. Measured
candidates: `I = 0` (`V* = -64.996`, slowest `tau = 8.29 ms`), `I = 2`
(`V* = -63.482`, `8.08 ms`), `I = 4` (`V* = -62.263`, `8.42 ms`) — all stable.
Open fractions at rest are `Na m^3h = 8.85e-5` and `K n^4 = 0.0102`, so **K
carries the noise** and Na is nearly shut; the `N` range must be chosen against
the K number, not the Na one. Spiking-regime channel noise is a **demo**, not the
slope test.

### Three fixes the convergence pathway needs for HH

Reusing `core/convergence.py` is right (`omega_key="n_channels"` already works —
it is a parameter), but three things in it are Gillespie-shaped:

1. **`_sample_on_grid` step-holds** (`searchsorted`), correct for a
   piecewise-constant SSA trajectory and **wrong** for a fixed-`dt` continuous
   one. Its error `~ |dV/dt| * dt_record` is **`N`-independent**, so it would
   floor `D(N)` and flatten the slope — the module's signature failure mode, and
   the existing Richardson check does **not** cover it (that checks only the ODE
   reference). Fix: **align the sample grid with the recording grid exactly** and
   assert the alignment, so the interpolation is a no-op.
2. **Add the symmetric stochastic-side floor check.** Halve the stochastic `dt`
   and confirm `D(N)` does not move, exactly as Richardson does for the ODE side.
   Fold it into `reference_ok`.
3. **`_per_replicate_discrepancy` weights species equally.** `V` (~100 mV) would
   swamp gating variables (~1). Pass `observable_keys=("V",)`, and say why — a
   mixed-scale L1 blend across mV and dimensionless gates is unprincipled.

---

## Gray-Scott

```
du/dt = Du lap(u) - u v^2 + F (1 - u)
dv/dt = Dv lap(v) + u v^2 - (F + k) v
```

### Validation, by category

- **A. The discrete Laplacian's Fourier eigenvalue — the exact engine anchor.**
  A single mode is an *exact eigenfunction* of the periodic 5-point stencil:
  `lap_h e^{i k.x} = -(4/h^2)(sin^2(k_x h/2) + sin^2(k_y h/2)) e^{i k.x}`. So
  pure diffusion of one seeded mode decays as `exp(lambda_h t)` with `lambda_h`
  known in closed form. This validates **stencil + periodic BCs + integrator at
  once**, and it is the Phase-2 analogue of `birth_death`.
- **B. Consistency order 2.** `lambda_h -> -D|k|^2` with `O(h^2)` error, so the
  log-log slope of `|lambda_h - lambda_continuum|` vs `h` is `-2`. A second
  slope check that reuses the pathway's shape at negligible cost.
- **A. The dispersion relation `lambda(q)` of the full nonlinear system.** As
  argued above: perturb the homogeneous steady state by `eps * cos(q x)`, measure
  the growth rate, compare to `max Re eig(J - q^2 D)`. Validated across a **sign
  change** (`q = 10` decays, `q = 20..60` grow, `q = 80, 120` decay at the chosen
  point). Like the HH-Jacobian check this is a *linearization*, so the honest
  form is that the discrepancy vanishes as `eps -> 0` — the amplitude sweep is
  part of the test, not an afterthought.
- **C / exploratory. The emergent Pearson-regime pattern** — figure and demo,
  labelled qualitative, no wavelength assertion.

### Protocol impact: smaller than it looks

The dispersion observable is the **amplitude of one Fourier mode**, a scalar:
`{"a_q": |u_hat(q)|}`. That fits the existing `Recorder` and `observables()`
contract with **zero protocol change** — the validation track needs nothing new.
A `FieldModel.field(state) -> np.ndarray` extension is needed **for viz only**
(so the shared plotting service can render a 2-D field without reaching inside a
model's state, per non-negotiable #1). Justifying it on those narrow grounds is
what keeps it defensible; it is not load-bearing for any check.

### CFL and cost

Explicit diffusion needs `dt < h^2 / (4 D_max)`. At `L = 1`, `N = 128`,
`Du = 2e-5`: `dt < 0.76`. The reaction timescale is `~1/F ~ 14`, and the fastest
growth rate at the chosen Turing point is `0.0150` (e-fold `66.7` time units), so
a growth-rate measurement over ~3 e-folds at `dt = 0.2` is ~1000 steps of `128^2`
array work — **cheap in pure NumPy**.

---

## Performance: the honesty clause

HANDOFF says Phase 2 is "where numba/JAX/WebGL get introduced, guided by real
profiling rather than guesswork." Read that as a *permission*, not an
*instruction*. Two things are already suspected and must be **measured, then
written down either way**:

- Gray-Scott at `128^2`-`256^2` is vectorized NumPy array work; the Python-level
  loop is one iteration per time step, not per cell. It may simply be fast enough.
- Stochastic HH is `O(#states)` per step by construction, so `N` is free and only
  the *step count* costs.

**If profiling says NumPy suffices, the tasks doc records that measurement and no
dependency is added.** A plan that promised numba is not a reason to add numba.
Non-negotiable #4 already says this; Phase 2 is where it will be tempting to
forget. Whatever is optimized must stay bit-identical and be fingerprinted before
and after, per the Phase-1 lesson.

---

## Suite budget

The suite is currently **129 s at `-n 6`**, floored by the 122 s repressilator
slope check. Every category-A check above is milliseconds-to-seconds. The one
plausible budget item is the **stochastic-HH `N`-sweep**, which is priced before
it is written (as the repressilator sweep was). If it lands under ~120 s it costs
nothing in wall clock; if it exceeds that, `-n` gets **re-timed**, because the
right `-n` is a function of the *runner-up* durations, not of core count.

---

## Build order

Ordered so that every step is validated before the next depends on it, and so the
two models stay independent (either could be dropped without stranding the other).

**2a — Hodgkin-Huxley**

1. `models/hh_rates.py` — the six rate functions + `_linoid`; test the removable
   singularities at `V = -40` / `V = -55` **first**.
2. `models/hh_voltage_clamp.py` — clamped gating, `analytic_predictions` from the
   exact exponential. Test first, confirm red.
3. `models/hodgkin_huxley.py` — the deterministic 4-D model on `core/ode.py`;
   resting-fixed-point and RK4-order tests.
4. Category-C reporting: rheobase, f-I curve, spike shape, refractory period,
   bistability — labelled, and out of `analytic_predictions`.
5. `models/hh_stochastic.py` — channel-state Markov counts, fixed-`dt` hybrid
   step, `DeterministicLimitModel`. **Price the sweep before writing the test.**
6. `core/convergence.py` — grid alignment, stochastic-side floor check,
   single-observable weighting. Then the `D(N) ~ N^{-1/2}` test, with teeth
   (broken `N` scaling) verified across seeds.
7. Viz + `demos/hodgkin_huxley.py`.

**2b — Gray-Scott**

8. `core/laplacian.py` — periodic 5-point stencil; exact Fourier-eigenvalue test
   first, then the order-2 consistency slope.
9. `models/gray_scott.py` — the PDE model, spectral scalar observables,
   `analytic_predictions` for the dispersion relation.
10. `lambda(q)` validation across the sign change, with the `eps -> 0` amplitude
    sweep.
11. `FieldModel` protocol extension + a field-rendering viz backend helper.
12. `demos/gray_scott.py` — the Turing point (validated) *and* the Pearson
    pattern (labelled exploratory).

**Both**

13. Profile; optimize only what profiling names; record the measurement whether
    or not it leads to a dependency. Re-time `-n`. Update `CLAUDE.md`, memory,
    docs; commit and push.

## Explicitly deferred

- Schnakenberg / Brusselator for near-onset wavelength selection.
- The HH linearized matrix-exponential check (the voltage clamp is sharper).
- scipy stiff integrators — measured unnecessary (`tau_min = 0.0622 ms`).
- WebGL / browser Gray-Scott — the HANDOFF fork stays deferred until the Python
  core is proven.
- HH *networks* (synaptic coupling) — single cell first; networks are the
  vectorization target if and when profiling asks.
- Bimolecular SSA propensity corrections (carried over from Phase 1).
