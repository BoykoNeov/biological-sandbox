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
- [x] `models/hodgkin_huxley.py` — deterministic 4-D model on `core/ode.py`
      (`dt` is a **param**, not a `step` arg). Test written first, confirmed red.
      22 tests; five mutants killed: `m^3 -> m^2` (12 failures), `n^4 -> n^3` (13),
      *ionic current sign flipped* (11), *gate closing term sign flipped* (13),
      *stability guard disabled* (1 — its dedicated test, as intended).
      - **The anchoring boundary, stated rather than glossed.**
        `analytic_predictions` returns the resting fixed point found by
        **root-finding** the algebraic steady state, which the simulation reaches by
        **time-integration** — two genuinely different code paths, so agreement
        catches drift between them. It **cannot** catch a wrong `g_na` or `m^2` for
        `m^3`: both paths move *consistently* and the root stays a root. What does
        catch those is `textbook_rhs` in the test file — a **separate
        hand-transcription of the published equations**, written from the paper, not
        from the implementation. That is the real teeth (mutants 1, 2, 3, 5).
        `V_rest ~ -65 mV` is the only other independent anchor and it is category C,
        so it is asserted as a coarse bound (`-66 < V < -64`), nothing tighter.
      - **`analytic_predictions` refuses an unstable fixed point.** Past the
        subcritical Hopf (stable at `I=8`, unstable by `I=10`) the attractor is a
        limit cycle; returning the fixed point would be a wrong number that still
        looks green, so it raises — the same stance as `require_termination`.
      - **Two separate claims, two separate tolerances.** *Stationarity* (start AT
        the fixed point, nothing moves) is checked through `validate()` with a
        Richardson `sem_floor`, and is razor-sharp. *Attraction* (start 5 mV away,
        arrive) is a different claim whose error is dominated by the leftover
        transient, which Richardson cannot see — both `dt` and `dt/2` carry the same
        one — so its tolerance is the **measured** `|y(t_max) - y(t_max/2)|`.
        Measured residual from a 5 mV offset (`tau = 8.29 ms`): **2.9e-8 at 100 ms,
        6.9e-11 at 150 ms, 1.8e-13 at 200 ms**.
      - Category C asserted **structurally only**: spike count monotone
        non-decreasing in `I`, zero spikes at `I=0`, some at `I=20`, and a spike
        overshoots 0 mV then after-hyperpolarises. No literature number is a bound.
      - `resting_state` scans for sign changes and **raises unless there is exactly
        one root**, listing the brackets it found. But the scan is not what
        *justifies* uniqueness — a grid cannot see a tangency or two roots inside one
        cell. **Measured and asserted instead: `I_ss(V)` is monotonically increasing**
        across the whole window at the 1952 conductances, so `i_ext - I_ss(V)` crosses
        zero exactly once for *any* `i_ext` and the guard is unreachable at the
        defaults. It is kept because the conductances are params: raise `g_na` and
        `I_ss` can acquire the N-shape that gives three equilibria, and then the honest
        report is "there are three and I will not pick one", not "your params are odd".
      - The attraction test was **too slack by 44,780x** on first writing
        (`residual <= still_decaying`). Sharpened with the decay law itself — for
        `r(t) ~ A e^{-t/tau}`, `still_decaying` is dominated by `r(t_max/2)`, so
        `r(t_max) ~ still_decaying * exp(-t_max/(2 tau))`, with `tau` from the
        Jacobian rather than typed in. Measured: residual 5.8e-10, still_decaying
        2.6e-5, predicted 3.1e-9 — the law is right to 5.3x. **840x tighter, 53x
        margin.** Stated in the test: even sharpened it is a *convergence-happened*
        check, and a deliberate 1e-8 nudge of the root still slips under it — the
        precision is carried by the root-residual and no-drift tests (1e-11) and by
        `validate()`, all three of which do catch that nudge.
      - **Profiling signal, recorded not acted on: 70.8 us per RK4 step** (~17.7 us
        per RHS evaluation), dominated by NumPy scalar overhead in the six rate
        functions. Irrelevant for the deterministic checks, but it sizes the
        stochastic sweep: at `t_max = 40 ms` (subthreshold `tau ~ 8 ms`) a replicate
        is 4000 steps ~ 0.28 s, so `R=20 x K=5` is ~28 s — affordable. A 200 ms
        horizon would have been ~140 s and a suite floor on its own.
- [x] `models/hh_stochastic.py` — 8-state Na + 5-state K occupancy counts, exact
      factorized propagator, fixed-`dt` hybrid step, `DeterministicLimitModel`.
      **Priced before writing** (`temp/phase2-work/`), and four of the plan's
      expectations turned out wrong:
      - **The splitting bias — the one thing that could floor `D(N)` — is exactly
        `0` at the resting fixed point.** Both maps share it: gates at `x_inf(V*)`
        do not move, and `V_inf` at the fixed-point conductances *is* `V*`. On a
        1.5 mV sub-rheobase transient it is `1.0e-3 mV` at `dt = 0.025`.
      - Advancing `V` at the **post**-transition conductance beats a pre/post
        midpoint by 100x (1.7e-4 vs 1.9e-2 mV). The "more accurate" midpoint rule
        is *worse* here, because using the newer gate value compensates for having
        frozen `V` at the older one. Not built.
      - Batching the 13 multinomial draws into 2 broadcast calls: **13.30 us vs
        13.46 us — a wash.** The assumption was wrong; the measurement was cheap.
      - The propagator build dominated instead, and the "vectorized" NumPy version
        cost **80.5 us/step against the 72 us naive loop it was meant to replace**:
        on a 4x4 there is no arithmetic to amortize, only per-call overhead. Plain
        float construction is 24.2 us; `np.kron` is 16.9 us against 3.1 us for
        broadcast-and-reshape.
      Nine mutants killed, three only after tests that were green for the wrong
      reason were rebuilt — see the lessons below.
- [x] Category-C report: rheobase, f-I curve, spike shape, refractory period, the
      subcritical-Hopf bistable band. **In `demos/hodgkin_huxley.py`'s printed
      output, asserted nowhere.** Measured: rheobase brackets to `[6, 6.5]`, f-I
      reaches 100 Hz at `I = 30`, spike peak `+41.30 mV` and after-hyperpolarisation
      `-74.04 mV` (published squid axon ~+40 / ~-75), half-width `0.72 ms`, ISI
      `12.06 ms`, and the fixed point is still **stable** at `I = 6.5, 7, 8` while
      the cell fires, unstable by `I = 10`.
      - **The first version put rheobase at `[2, 4]`** — it counted spikes over the
        whole window, and a sub-rheobase step still fires an *onset* spike before
        falling silent. Counted over `t in [50, 100] ms` it lands on the textbook
        value. The "incl. onset" column stays in the table so the artifact is
        visible rather than merely corrected.
- [x] Convergence-pathway fixes — **two were needed, not one.** `compare_keys` is
      not optional: the existing one-to-one length guard *rejects*
      `observable_keys=("V",)` against a 4-component ODE outright, so the plan's
      "just pass it" could not work as written. Grid alignment landed as an explicit
      `grid` argument plus `require_exact_grid`, verified per replicate.
      - **The test for it was wrong first, instructively.** It asserted that a
        `linspace` grid misaligns, and failed: at `n_steps=1600, stride=8` it aligns
        perfectly, because `linspace` computes `i * (t_max/(n_grid-1))`, which equals
        `(i*stride)*dt` exactly when the division is by a **power of two**. Stride 3
        puts 25 of 41 grid times off, and 373 of the swept configurations misalign.
        The hazard is not that `linspace` is wrong but that it is right until someone
        changes `n_grid` from 201 to 121.
      - The stochastic-side floor check went in as `stochastic_dt_key`, passing on
        either a small relative shift **or** statistical indistinguishability at `z`
        combined SEs — the two runs use independent streams, so demanding a small
        *absolute* shift would turn replicate noise into a failure.
- [x] `D(N) ~ N^{-1/2}` in the sub-rheobase subthreshold regime. Slope across seeds
      0-3 at `R=16, z=3`: **-0.5092, -0.4933, -0.4988, -0.5101** (SE 0.0106-0.0140).
      Seed 0 is pinned because it is the *worst* of the four.
      - **The threshold trap is real and was measured, not reasoned about:** at
        `N = 1000` and `4000` the membrane fires spontaneously (max `|V-V*|` of 108
        and 103 mV) and `D*sqrt(N)` leaps from its ~71 plateau to 266 and 109,
        dragging an all-points slope to `-0.734`. The sweep starts at 16000.
      - The high end is **free**: a step is `O(#states)` and `N`-independent (0.203 s
        per replicate at 1.6e4 and at 4.1e6 alike), so 256x of lever arm costs
        nothing. That is where the slope precision comes from.
      - Teeth, each asserted only on its structurally-robust leg: `N` pinned gives
        slope `+0.011/-0.019/+0.037/+0.021`, `significant=False`; `N -> 200 sqrt(N)`
        gives `-0.2544/-0.2476/-0.2637/-0.2364`, `consistent=False` but
        `significant=True`. **The repressilator's `Omega^2` tooth could not be
        reused**: squaring `N` drives `D` to ~1.7e-5 mV, an order of magnitude below
        the 1.0e-3 mV splitting bias, so the broken model would fail through a
        discretization floor rather than through its scaling.
- [x] Viz + `demos/hodgkin_huxley.py` — four acts kept apart by category, four
      figures written **and looked at**. Channel noise is shown at `I = 5`
      (sub-rheobase), not `I = 20`: deep in the firing regime even `N = 2000` gives
      `[5,5,4,5]` against the limit's 5, because the drive swamps the noise. At
      threshold the limit fires 1 onset spike while `N = 1000` fires `[4,3,5]`.

## 2b — Gray-Scott

- [x] `core/laplacian.py` — periodic 5-point stencil, dimension-agnostic via
      `np.roll` so periodicity is structural rather than an edge case. Exact
      Fourier-eigenvalue test first; nine mutants killed.
      - **Two of its tests asserted a limit outside its regime.** The order-2 slope
        read **1.9737** over `n = 16..256`, and the shortfall was real: the expansion
        is in `k h / 2`, which is `0.785` at `n=16, j=4`. On `n = 64..1024` it is
        **1.99835**, coarsest point within 0.5% of `D k^4 h^2/12`.
      - "A constant field has exactly zero Laplacian" is **false**: four additions
        then a divide by `h^2` leave `1.08 eps |c| / h^2`, and that ratio is identical
        at `n=32` and `n=128` — which is what identifies it as cancellation rather
        than a stencil error.
      - The decay test ran for 9 e-folds at first, so "the error is small" would have
        described a field that had already vanished. It is 3 now.
- [x] Order-2 consistency slope: **1.99835** (category B, negligible cost).
- [x] `models/gray_scott.py` — PDE model, scalar observables, CFL validated in
      `__post_init__` against `cfl_limit` (the **forward-Euler** bound, though the
      model steps with RK4: a margin borrowed from the integrator's stability polygon
      vanishes the day the integrator changes).
- [x] `lambda(q)` validation across the **sign change** at `(F,k) = (0.074, 0.062)`:
      `j in {3,5,7,10,12}` grow, `{14,16,20}` decay. 12 mutants killed.
      - **The reference is the DISCRETE eigenvalue, and that is not a refinement.**
        At `n=64` the stencil and the continuum differ by **408% at `j=12`** and
        **disagree about the sign at `j=13`**. No tolerance covers an opposite sign.
        Asserted directly against a continuum formula written out in the test.
      - **The tolerance is Richardson in the AMPLITUDE, not in `dt`.** The error is
        `O(a^2)` — a linearization artifact, not a discretization one — so `dt` and
        `dt/2` carry the same term and Richardson in `dt` would report a reassuringly
        tiny number about the wrong thing. Measured, `(4/3)|m(a)-m(a/2)|` predicts the
        true error to a ratio of **1.000** at all eight probes, and the extrapolant
        lands on `lambda` to 1e-11..1e-14.
      - **Every probe ends at the same amplitude rather than starting there.** The
        error is set by the *final* amplitude, so seeding all probes at one `eps` gave
        `j=3` a relative error of 7.5e-3 against `j=7`'s 1.5e-5 — a 500x spread no
        single tolerance could describe honestly. Equalizing the endpoint brings the
        worst case to 1.4e-4.
      - The `eps -> 0` slope measured **1.9978** against the predicted 2 —
        corroborating, now that amplitude-Richardson carries the tolerance.
      - Three refusals, each because the alternative is a wrong number that looks
        green: no real non-trivial state, a complex pair (`j=1,2`), and a rate too
        near zero to measure. The last is not fastidiousness: `j=13` sits at
        `lambda = 1.06e-4`, needs `t_max = 18822` for two e-folds, and returns `nan`.
- [x] `FieldModel` protocol extension — **viz only**, and nothing validated depends
      on it: the dispersion check runs through the scalar `a_q`. Landed as
      `fields(state) -> dict[str, ndarray]`, mirroring `observables`.
- [x] `demos/gray_scott.py` — the validated Turing point **and** the Pearson pattern,
      the latter printing the `homogeneous_state` raise verbatim so the reframing
      appears as an error message rather than as prose.
      - **Two figures were wrong until they were looked at.** The dispersion plot drew
        the low-`q` region as part of the solid prediction curve, where the pair is
        *complex* and the plotted value is `tr/2` — a decay envelope, not a measurable
        rate, and the visible kink read as physics. And the Turing panel was titled "a
        Turing pattern", implying its 7 stripes were a *selected* wavelength; they are
        the **seeded** mode saturating. It now says so, and reports that `j=7` is
        separately computed to be the fastest-growing mode.

## Both — close-out

- [x] Profile. **No dependency added, and the measurement is why.** The HH slope
      test costs ~25 s against a 111 s suite floor, so nothing about the stochastic
      step needs optimizing; Gray-Scott at `64^2`-`128^2` is vectorized NumPy whose
      python-level loop runs once per time step, not per cell. Where measurement
      *did* change code it went the opposite way to the plan's expectation: the
      hand-written scalar propagator build beat the "vectorized" one 3.3x, and
      broadcast-and-reshape beat `np.kron` 5.5x. numba and JAX stay out.
- [x] Re-time `-n`. **Measured — and the Phase-1 close-out's proposed fix was tried
      and rejected on measurement.** Suite: **310 passed in 129.7 / 130.0 / 130.8 s**
      at `-n 6` — reproducible, floored by the single 110.9 s repressilator test.
      (The 174.5 s reading seen mid-phase was a scheduling draw, not a regression;
      `--durations` showed every test at its normal cost.)
      The Phase-1 doc proposed making the two long tests **dispatch first**. It was
      implemented as a `pytest_collection_modifyitems` hook and measured at
      **228.7 s and 227.1 s — 75% worse**, consistently, and very close to
      `110.9 + 91.2 = 202 s` of long tests sharing one worker: sorting them to the
      front changes which items fall into xdist's initial batch and evidently
      co-schedules the two largest. Reverted. **`-n 6` stays and collection order is
      left alone**; raising the worker count cannot help either, because the floor is
      one indivisible test.
      Runner-up durations: 110.9, 91.2, 25.1, 24.5, 19.5, 19.3 s.
- [x] `uv run pytest -q` green (310 passed); `uv run ruff check .` clean;
      `ruff format .` applied.
- [x] Demos run and the figures were **looked at**, not exit-code checked — which is
      how the f-I onset transient, the wrong-regime channel noise, the complex-pair
      kink and the seeded-vs-selected wavelength were all caught.
- [x] Update `CLAUDE.md` Status, memory, and this doc; commit and push.

## Explicitly deferred (do NOT do in Phase 2)

- [x] ~~Schnakenberg / Brusselator~~ — **TAKEN in Phase 2c** (Schnakenberg;
      Brusselator stays deferred). The reasoning here was half wrong: category A
      validates `lambda(q)`, the growth rate of a mode seeded *by hand*, and
      wavelength **selection** is a different claim about the mode that appears when
      nothing is seeded. It is not the same physics more sharply, and Gray-Scott
      cannot carry it at all — at Pearson's parameters there is no Turing state to
      select about. See `docs/plans/phase2c-{plan,schnakenberg-measurement}.md`.
- [ ] ~~HH linearized matrix-exponential check~~ — only valid to `O(eps^2)`, so an
      honest version is a slope-2-in-`eps` check; the voltage clamp is sharper and
      cheaper.
- [ ] ~~scipy stiff integrator~~ — measured unnecessary (`tau_m_min = 0.0622 ms`).
- [ ] ~~WebGL / browser Gray-Scott~~ — the HANDOFF browser fork stays deferred
      until the Python core is proven.
- [ ] ~~HH networks (synaptic coupling)~~ — single cell first.
- [ ] ~~numba / JAX by default~~ — permission, not instruction. Profile first.
