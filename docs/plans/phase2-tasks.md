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

- [x] `models/hh_rates.py` — six rate functions + `_linoid(x, k)` for the
      removable `0/0` at `V = -40` / `V = -55`. Test written first and the naive
      transcription **confirmed red (11 failed / 3 passed)**. 15 tests, all
      mutation-checked:
      - *no guard at all* (literal `x/(1-exp(-x/k))`) -> 11 failures: `nan` at both
        singularities, and it poisons the dense sweep, the vectorized-vs-scalar
        check and every `tau > 0` assertion.
      - *guard but no `expm1`* -> 3 failures. This one needed the test to be
        **rebuilt to have teeth**: the first draft probed via `alpha_m(-40 + delta)`
        and passed the mutant, because recovering `x` as `(-40 + delta) + 40`
        carries ~7e-15 of rounding (one ulp at 40) — enough to flip which side of
        the `1e-7 k` guard a nominal `1e-6` lands on, so every probe silently took
        the series branch and the exp branch went untested. Measured naive relative
        error at `k=10`: **4.9e-10 at x=1e-6, 1.6e-11 at x=1e-5** — so the band
        *just above* the guard is the only place the choice shows, and the test now
        calls `_linoid` directly across `x in [1e-6, 0.1]` against an independent
        Bernoulli series `k(1 + u/2 + u^2/12 - u^4/720)` at `rel=1e-14`.
      - *`alpha_h`/`beta_h` swapped* -> 3 failures (published-fit values,
        activation-vs-inactivation direction, asymptotic limits).
      Also recorded: `n_inf` reaches only **0.9982 at +200 mV** (`beta_n` has the
      slowest length constant of the six, 80 mV), so the asymptotic test asserts
      per-gate bounds rather than one shared tolerance — a uniform `abs=1e-3` was
      simply wrong there, and it was the test that was wrong, not the code.
- [x] `core/ode.py` — extracted `rk4_step(rhs, y, h)` from `integrate_rk4` so
      protocol models can take a *single* RK4 increment inside `step` (HH and
      Gray-Scott both need one; a dense trajectory is the wrong shape there).
      `integrate_rk4` now calls it and is **bit-identical** — sha256 of `t` + `y`
      across three RHS shapes matched before and after
      (`eb4224500d0b4923dd80d2a031bed71cfd3943a21a2b0fd6b1a10bf9950a5b31`).
- [x] `models/hh_voltage_clamp.py` — clamped gating, `dx/dt = (x_inf - x)/tau`,
      `analytic_predictions` from the exact `x(t) = x_inf + (x0 - x_inf)e^{-t/tau}`.
      **The category-A lead anchor.** Test written first, confirmed red. 17 tests;
      four mutants killed, 10-11 failures each: *tau doubled*, *RHS sign flipped*,
      *start at `v_clamp` instead of `v_hold`* (removes the transient entirely),
      *`x_inf` gates m/h transposed*.
      - **Scope stated, not glossed:** the closed form is built from the same
        `x_inf`/`tau` that `step` integrates, so an error *inside* a rate function
        cancels on both sides. This file validates the integrator + the decoupled
        gating structure; the rate functions are pinned independently by
        `test_hh_rates.py`. Neither closes the loop alone.
      - **Tolerance is measured, not typed.** The model is deterministic, so
        `validate()`'s statistical SE is exactly zero and its tolerance degenerates;
        the honest tolerance is *numerical*, so Richardson (`dt` vs `dt/2`) supplies
        `sem_floor` on each run. Two replicates, not one — `validate()` returns
        `sem = inf` for a single sample and the check would pass vacuously. The
        replicates having distinct RNG streams and still agreeing exactly is itself
        the determinism check.
      - **Two of my own tests were wrong first, in the same way.** Both measured the
        value at `t_max` — but by 10 ms every gate has fully relaxed onto `x_inf`,
        so both the exact and numerical solutions agree there to ~1e-14 *whatever
        `dt` was*. The endpoint carries no information about integration error: all
        of it lives in the transient. Measured max-over-trajectory error at
        `dt=0.01`: **1.1e-8 at v=-80, 8.9e-9 at v=0, 4.1e-8 at v=+20** (worst where
        `tau_m` is fastest), versus **~1e-13 at the endpoint**. So the trajectory
        test now uses the max over time against a Richardson bound, and the
        order-4 check does too — on the endpoint it was a ratio of two
        machine-noise numbers that passed for no reason. Ratios now measured at
        16.6/16.3 (v=-80), 16.3/16.1 (v=0), 16.4/16.2 (v=+20).
      - `state.t` is `step_index * dt`, never accumulated, and `dt` must divide
        `t_max` exactly — 1000 additions of 0.01 drift enough to make `t >= t_max`
        miss and overshoot a whole step, sampling the closed form at the wrong time.
      - Clamping to `V = -40` / `V = -55` is a test in its own right: a clamp
        protocol steps to round numbers, so those are the realistic way step 1's
        `0/0` trap gets hit.
- [ ] `models/hodgkin_huxley.py` — deterministic 4-D model on `core/ode.py`
      (`dt` is a **param**, not a `step` arg). Tests: resting fixed point
      (`rhs = 0`, no drift), RK4 order 4 on the real RHS.
- [ ] Category-C report: rheobase, f-I curve, spike shape, refractory period, the
      subcritical-Hopf bistable band. **Labelled literature-anchored in its own
      output; not in `analytic_predictions` — and never in an assertion whose bound
      is a literature number.** "Rheobase in [6.2, 6.5]" would be asserting the
      0.5-resolution of the planning scan (200 ms window, 100 ms transient dropped),
      brittle to any change of horizon or spike-detection threshold. These belong in
      the **demo's printed output**. If anything is asserted in the suite it is a
      *structural* fact only: spike count is monotone non-decreasing in `I` over the
      swept range, some `I` gives zero spikes and some gives more than zero.
- [ ] `models/hh_stochastic.py` — 8-state Na + 5-state K **occupancy counts**,
      fixed-`dt` hybrid step (multinomial transitions, then RK4 on `V`),
      `DeterministicLimitModel`. `O(#states)` per step, independent of `N`.
      **Price the sweep before writing the test** (as the repressilator was).
- [ ] Convergence-pathway fixes for HH — **(1) revised, and it needs no
      `convergence.py` change at all.** The original plan was to add an
      interpolation mode, because `_sample_on_grid` step-holds (right for SSA,
      `N`-independent error for a fixed-`dt` trajectory, so it would floor `D(N)`
      and flatten the slope). The cleaner route reuses what
      `hh_voltage_clamp` already established: with `state.t = step_index * dt`
      (**counted, never accumulated**) the recorded times are exact multiples of
      `dt` by construction, so choosing `n_grid = n_steps/j + 1` for integer `j`
      makes the comparison grid a *subset* of the recorded times and
      `searchsorted` returns the exact recorded point — the interpolation becomes a
      no-op. **Assert in the test that every grid time is a recorded time**, or the
      day someone changes `n_grid` this silently degrades back to interpolation and
      the symptom is a flattened slope, not an error.
      (2) stochastic-side floor check (halve stochastic `dt`, `D(N)` must not move)
      folded into `reference_ok`; (3) use `observable_keys=("V",)` so mV does not
      swamp the dimensionless gates.
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
      **Measured, after a scare that turned out to be nothing.** The step-1 run
      reported 100 passed in **203.8 s** against the 129 s recorded at the end of
      Phase 1, and the first instinct was to write down "xdist packed the two long
      tests onto one worker" as the cause. That would have been a guess: the
      `--durations=12` run settles it instead.
      Measured at step 2 — **117 passed in 147.0 s**, with the long tests at
      **117.1 s** (repressilator slope) and **94.3 s** (squared-Omega tooth), then
      26.6 s, 21.4 s, 21.3 s. Those are their *nominal* durations, so the 203.8 s was
      **not** a per-test regression and not contention inflating individual tests;
      it was a one-off scheduling draw. The suite floors at 117 s and 147 s sits
      30 s above it, so **`-n 6` stays** — no change on a single noisy measurement.
      Re-check whenever a new multi-minute test lands; if packing does recur, the
      durable fix is to make the two long tests **dispatch first** (they are late in
      alphabetical collection order), not to raise the worker count.
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
