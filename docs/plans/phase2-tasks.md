# Phase 2 — tasks

Build order is dependency-ordered; each `[ ]` is a commit-sized unit. Per the
workflow rule, **write each validation test first and confirm it can fail** before
the implementation is correct. Numbers recorded here are *measured results*, not
targets — fill them in as each step lands.

## Step 0 — sanity slices (done before any planning)

- [x] **Gray-Scott dispersion slice.** Scanned `F in [0.010,0.120] x
      k in [0.030,0.080]` (11k points) for a *genuine* Turing point: real
      non-trivial homogeneous state, **stable without diffusion**, interior
      `lambda(q*) > 0` at `q* > 0`. **Key finding: Pearson's famous pattern points
      have no real non-trivial state at all** (`F < 4(F+k)^2`) — their patterns are
      not Turing patterns, so HANDOFF's "validate wavelength against LSA" is
      unavailable there. Genuine Turing points occupy a razor sliver,
      `F in [0.049,0.117]`, `k in [0.054,0.062]`, `k`-window ~0.5% wide.
      Chosen `(F,k) = (0.074, 0.062)`: `u*=0.49264785, v*=0.27605926`,
      `eig(J) = -0.00710436 +/- 0.01580864i` (stable), band `q in [16.048, 76.366]`,
      max `lambda = +0.0149862` at `q* = 43.828`. Probes span a **sign change**
      (`q=10` decays, `20..60` grow, `80,120` decay) — that is the teeth.
- [x] **Hodgkin-Huxley slice.** Rest at `I=0`: `V = -64.996379331 mV`,
      `(m,h,n) = (0.05295509, 0.59599412, 0.31773240)`, `|rhs| = 1.8e-15`,
      `eig(J) = {-4.675, -0.2026 +/- 0.3832i, -0.1207}` (stable, slowest
      `tau = 8.29 ms`). `tau_m_min = 0.0622 ms` -> **explicit RK4 suffices, no
      scipy**. RK4 order confirmed on the real RHS: `6.45e-7 / 4.43e-8 / 2.89e-9 /
      1.84e-10` at `dt = 0.02/0.01/0.005/0.0025` (ratios 14.6, 15.3, 15.7 -> 16).
      Rheobase between `I = 6.2` and `6.5`; **bistable band** — fixed point still
      stable at `I = 6.5, 7, 8` while firing, unstable by `I = 10`. Open fractions
      at rest `Na m^3h = 8.85e-5`, `K n^4 = 0.0102` -> **K carries the noise**.
      Found the `0/0` trap in `alpha_m`/`alpha_n` at `V = -40` / `V = -55`.

## 2a — Hodgkin-Huxley

- [ ] `models/hh_rates.py` — six rate functions + `_linoid(x, k)` for the
      removable `0/0`. **Test first:** evaluate exactly at `V = -40` and `V = -55`
      (limit `= k`), check continuity across them, and check `x_inf`/`tau` against
      the slice table. A naive implementation returns `nan` there — confirm the
      test is red against it.
- [ ] `models/hh_voltage_clamp.py` — clamped gating, `dx/dt = (x_inf - x)/tau`,
      `analytic_predictions` from the exact `x(t) = x_inf + (x0 - x_inf)e^{-t/tau}`.
      **The category-A lead anchor** — validates every rate function individually.
      Test first, confirm red.
- [ ] `models/hodgkin_huxley.py` — deterministic 4-D model on `core/ode.py`
      (`dt` is a **param**, not a `step` arg). Tests: resting fixed point
      (`rhs = 0`, no drift), RK4 order 4 on the real RHS.
- [ ] Category-C report: rheobase, f-I curve, spike shape, refractory period, the
      subcritical-Hopf bistable band. **Labelled literature-anchored in its own
      output; not in `analytic_predictions`.**
- [ ] `models/hh_stochastic.py` — 8-state Na + 5-state K **occupancy counts**,
      fixed-`dt` hybrid step (multinomial transitions, then RK4 on `V`),
      `DeterministicLimitModel`. `O(#states)` per step, independent of `N`.
      **Price the sweep before writing the test** (as the repressilator was).
- [ ] `core/convergence.py` fixes — (1) align sample grid to the recording grid
      exactly and assert it (step-hold interpolation is `N`-independent and would
      floor `D(N)`); (2) stochastic-side floor check (halve stochastic `dt`, `D(N)`
      must not move) folded into `reference_ok`; (3) use
      `observable_keys=("V",)` so mV does not swamp the dimensionless gates.
- [ ] `D(N) ~ N^{-1/2}` convergence test in the **sub-rheobase subthreshold**
      regime, with teeth (broken `N` scaling) **verified across seeds 0-3**,
      asserting only the structurally-robust leg for each break.
- [ ] Viz + `demos/hodgkin_huxley.py` — deterministic spike train, the f-I curve
      (labelled C), channel-noise replicates vs the ODE, the `N`-scaling figure.
      Spiking-regime channel noise belongs **here**, not in the slope test.

## 2b — Gray-Scott

- [ ] `core/laplacian.py` — periodic 5-point stencil. **Test first with the exact
      Fourier eigenvalue** `lambda_h = -(4D/h^2)(sin^2(k_x h/2) + sin^2(k_y h/2))`:
      a single mode is an exact eigenfunction, so this validates stencil + periodic
      BCs + integrator at once (the `birth_death` of 2b).
- [ ] Order-2 consistency slope: `|lambda_h - (-D|k|^2)|` vs `h` has log-log slope
      `-2`. Category B, negligible cost.
- [ ] `models/gray_scott.py` — PDE model, **scalar** observables (mode amplitude
      `|u_hat(q)|`, means) so the existing Recorder needs no change. CFL
      `dt < h^2/(4 D_max)` validated in `__post_init__`.
- [ ] `lambda(q)` validation at `(F,k) = (0.074, 0.062)` across the **sign change**
      (`q = 10, 20, 30, 43.828, 60, 80, 120`), plus the `eps -> 0` amplitude sweep
      that makes the linearization claim honest.
- [ ] `FieldModel.field(state)` protocol extension — **viz only**, justified on
      exactly those grounds; no validation depends on it.
- [ ] `demos/gray_scott.py` — the validated Turing point **and** the Pearson
      pattern, the latter explicitly labelled qualitative/exploratory with the
      reason (no real non-trivial homogeneous state there).

## Both — close-out

- [ ] Profile. Optimize only what profiling names; **record the measurement
      whether or not it leads to a dependency** — if NumPy suffices for Gray-Scott
      at `128^2`-`256^2`, write that and add nothing. Any optimization must be
      **bit-identical**, sha256-fingerprinted before and after.
- [ ] Re-time `-n` (currently 6, floored by the 122 s repressilator check). The
      right `-n` is a function of the *runner-up* durations, not core count.
- [ ] `uv run pytest -q` green; `uv run ruff check .` clean; `ruff format .`.
- [ ] Demos run and the figures were **looked at**, not just exit-code checked.
- [ ] Update `CLAUDE.md` Status, memory, and this doc; commit and push.

## Explicitly deferred (do NOT do in Phase 2)

- [ ] ~~Schnakenberg / Brusselator~~ — a near-onset wavelength-selection
      prediction is cleanly available there, but category A above validates the
      same physics more sharply; adding a third model to buy one check is scope
      creep.
- [ ] ~~HH linearized matrix-exponential check~~ — only valid to `O(eps^2)`, so an
      honest version is a slope-2-in-`eps` check; the voltage clamp is sharper and
      cheaper.
- [ ] ~~scipy stiff integrator~~ — measured unnecessary (`tau_m_min = 0.0622 ms`).
- [ ] ~~WebGL / browser Gray-Scott~~ — the HANDOFF browser fork stays deferred
      until the Python core is proven.
- [ ] ~~HH networks (synaptic coupling)~~ — single cell first.
- [ ] ~~numba / JAX by default~~ — permission, not instruction. Profile first.
