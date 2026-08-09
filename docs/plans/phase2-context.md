# Phase 2 — context (key files & decisions for a fresh session)

Read `phase2-plan.md` first for the full rationale and the measured numbers. This
is the quick orientation: where new code lives, the contracts it must honor, and
the traps.

## What Phase 2 adds (planned layout)

| Concern | File | New? |
|---|---|---|
| HH rate functions + `_linoid` singularity fix | `src/sandbox/models/hh_rates.py` | new |
| Voltage-clamp gating (exact anchor) | `src/sandbox/models/hh_voltage_clamp.py` | new |
| Deterministic Hodgkin-Huxley | `src/sandbox/models/hodgkin_huxley.py` | new |
| Channel-noise HH (fixed-`dt` hybrid) | `src/sandbox/models/hh_stochastic.py` | new |
| Grid alignment + stochastic-side floor check | `src/sandbox/core/convergence.py` | extend |
| Periodic 5-point Laplacian | `src/sandbox/core/laplacian.py` | new |
| Gray-Scott PDE + dispersion predictions | `src/sandbox/models/gray_scott.py` | new |
| `FieldModel` protocol (**viz only**) | `src/sandbox/core/protocol.py` | extend |
| Field rendering | `src/sandbox/viz/backends/matplotlib_backend.py` | extend |
| Demos | `src/sandbox/demos/{hodgkin_huxley,gray_scott}.py` | new |

## The three ideas that drive every choice here

1. **Three categories of "checkable", never blurred.** **A** exact analytic
   (`analytic_predictions`), **B** asymptotic law (log-log slope + statistical
   CI), **C** literature-anchored (rheobase, f-I, spike shape — *evidence, not
   proof*). **Category C never enters `analytic_predictions`** and must label
   itself in its own output. "Done" is A + B.
2. **HANDOFF's Gray-Scott wavelength promise was reframed, not dropped.** At
   Pearson's famous `(F, k)` there is *no real non-trivial homogeneous state*, so
   those patterns are not Turing patterns and LSA makes no claim about their
   wavelength. Phase 2 validates **`lambda(q)` itself** (sharper, works
   everywhere, spans a sign change) and labels the emergent pattern qualitative.
3. **HH's convergence trap is threshold crossings, not phase diffusion.** Low
   channel counts produce *spurious/missing spikes* — a changed spike **count**,
   which obeys no `-1/2` law. The slope check lives in a **sub-rheobase,
   subthreshold, short-horizon** regime. Spiking channel noise is a demo.

## Contracts the new code must honor

Everything from Phase 0/1 still holds (stateless model, params embedded in state
by `initial_state`, `step` writes `state.t`, register in `models/__init__.py`,
plain-number JSON-serializable params, `spawn_rngs` never `seed + i`,
`analytic_predictions` keys name observables checked at their *final* value).
Phase-2 additions:

- **A third time-advance discipline.** Wright-Fisher = one generation, Gillespie =
  a sampled `tau`, **HH/Gray-Scott = a fixed numerical `dt` that is a *param* of
  the model**, not a `step` argument. This is the case the no-`dt` signature was
  designed for; do not add a `dt` argument to `step`.
- **HH stochastic state = channel-state occupancy counts**, 8-state Na + 5-state
  K. A step is `O(#states)`, **independent of `N`** — that is what makes the
  `N`-sweep affordable. Do not write a per-channel loop.
- **Gray-Scott `observables()` returns scalars** (mode amplitude `|u_hat(q)|`,
  means). The 2-D field is exposed **only** through the viz-only `FieldModel`
  extension; no shared service may reach into the state for it.
- **Units.** HH: `V` mV, `t` ms, `C_m` uF/cm^2, `g` mS/cm^2, `I` uA/cm^2.
  Gray-Scott: dimensionless `u, v` on `[0, L]^2` with periodic BCs.

## Convergence-pathway fixes required before the HH slope check

1. **`_sample_on_grid` step-holds** — right for SSA, **wrong** for a fixed-`dt`
   continuous trajectory. Its error is **`N`-independent**, so it floors `D(N)`
   and flattens the slope, and the existing Richardson check does **not** catch it
   (that checks only the ODE reference). **Align the sample grid to the recording
   grid exactly and assert it.**
2. **Add a stochastic-side floor check** — halve the stochastic `dt`, confirm
   `D(N)` does not move; fold into `reference_ok`.
3. **Pass `observable_keys=("V",)`** — `_per_replicate_discrepancy` averages
   `|stoch - ode|` over species with equal weight, and `V` (~100 mV) would swamp
   the dimensionless gates (~1).

## Measured constants (so you don't re-derive them)

**Hodgkin-Huxley** (`C_m=1`, `g_Na,g_K,g_L = 120, 36, 0.3`,
`E_Na,E_K,E_L = 50, -77, -54.387`):

- Rest at `I=0`: `V = -64.996379331 mV`, `(m,h,n) = (0.05295509, 0.59599412,
  0.31773240)`, `|rhs| = 1.8e-15`. `eig(J) = {-4.675, -0.2026 +/- 0.3832i,
  -0.1207}` — stable, slowest `tau = 8.29 ms`.
- `tau_m_min = 0.0622 ms` over `[-90, 60] mV` -> **explicit RK4 at `dt = 0.01 ms`
  is fine; no scipy stiff solver.** RK4 error vs a `dt=5e-4` reference over 20 ms
  at `I=10`: `6.45e-7 / 4.43e-8 / 2.89e-9 / 1.84e-10` for
  `dt = 0.02 / 0.01 / 0.005 / 0.0025` (ratios 14.6, 15.3, 15.7 -> `2^4`).
- Rheobase between `I = 6.2` and `6.5 uA/cm^2`. **Bistable band:** the fixed point
  is still *stable* at `I = 6.5, 7, 8` while the model fires, losing stability
  between `8` and `10` — the classic subcritical Hopf. Category C.
- Subthreshold fixed points: `I=0 -> -64.996`, `I=2 -> -63.482`, `I=4 -> -62.263`;
  all stable, slowest `tau ~ 8 ms`. Open fractions at rest: `Na m^3h = 8.85e-5`,
  `K n^4 = 0.0102` — **K carries the channel noise**, size the `N` range on it.
- **Trap:** `alpha_m`, `alpha_n` are `x/(1-exp(-x/k))` with removable `0/0` at
  **`V = -40`** and **`V = -55`** — the exact round numbers a test will probe.
  Limit is `k`; use a `_linoid` helper (series `k + x/2` near zero) and test it.

**Gray-Scott** (`Du = 2e-5`, `Dv = 1e-5`, `du/dt = Du lap u - u v^2 + F(1-u)`,
`dv/dt = Dv lap v + u v^2 - (F+k) v`):

- Non-trivial homogeneous states exist iff `F >= 4 (F+k)^2`:
  `v = [F +/- sqrt(F^2 - 4F(F+k)^2)] / (2(F+k))`, `u = (F+k)/v`.
- **Genuine Turing points** (real state + stable without diffusion + interior
  `lambda(q*) > 0` at `q* > 0`) occupy only `F in [0.049, 0.117]`,
  `k in [0.054, 0.062]`; the `k`-window at fixed `F` is ~**0.5% wide**
  (`F = 0.074`: `k in [0.06166, 0.06200]`). Do not nudge these params casually.
- **Chosen validation point `(F, k) = (0.074, 0.062)`**, upper state
  `u = 0.49264785`, `v = 0.27605926`, `eig(J) = -0.00710436 +/- 0.01580864i`
  (**stable without diffusion**). `lambda(0) = -0.0071044`; max `+0.0149862` at
  `q* = 43.828` (wavelength `0.143360`); unstable band `q in [16.048, 76.366]`.
  Probes: `q=10 -> -0.0086044`, `20 -> +0.0049651`, `30 -> +0.0119268`,
  `43.828 -> +0.0149862`, `60 -> +0.0111856`, `80 -> -0.0034411`,
  `120 -> -0.0632885`. **The sign change is the teeth.**
- Explicit CFL `dt < h^2/(4 D_max)`: at `L=1, N=128, Du=2e-5` that is `dt < 0.76`;
  `dt = 0.2` over ~3 e-folds (`1/0.015 = 66.7` t.u. each) is ~1000 steps of
  `128^2` array work.
- Pearson's famous points (`(0.037,0.060)`, `(0.025,0.055)`, `(0.014,0.054)`,
  `(0.040,0.060)`, `(0.035,0.065)`) have **only** the trivial `(1, 0)` state.

## Environment / gotchas (carried forward)

- Windows console is cp1252 — **no non-ASCII in `print`ed strings** (`Omega`,
  `lambda`, `tau`, `<n>`). Docstrings/comments may be UTF-8.
- `uv` venv; numpy 2.x; matplotlib only under `--extra viz`; **no scipy** (RK4 is
  hand-rolled and measured sufficient for HH).
- Suite is **129 s at `-n 6`**, floored by the 122 s repressilator slope check.
  Re-time `-n` whenever a new multi-minute test lands — the right `-n` is a
  function of the *runner-up* durations, not of core count.
- Teeth tests must be **verified across 3-4 seeds**, and assert only the leg that
  is *structurally* robust for that break (for a flat-slope break, `slope/SE` is
  replicate-independent — more replicates never de-flake it).
- Any optimization must be **bit-identical**, fingerprinted (`sha256` of
  times + series) before and after.
